import torch
import torch.nn as nn
from .config import BrainConfig
from .llm.qwen_wrapper import QwenWrapper
from .core.brain_module import BrainModule
from .core.stdp import STDPLearning
from .core.injection import InjectionGate, BrainHookManager, HippocampalPrefixInjector
from .memory.hippocampus import HippocampalMemory
from .utils.device import get_device


class BrainHybridModel(nn.Module):
    """
    Modèle principal — assemble Qwen + modules CfC+SNN + hippocampe + injection.

    Flux de traitement :
      Phase 1 — Observation + Apprentissage :
        1. Input → QwenWrapper → hidden states couches 9, 18, 27, 36
        2. Chaque BrainModule prédit la couche suivante
        3. Erreur de prédiction → dopamine → STDP
        4. Si saillant → stockage hippocampe

      Phase 2 — Génération influencée :
        5. Récupération souvenirs hippocampe
        6. Forward hooks CfC injectent des résidus dans les hidden states
        7. Qwen génère la réponse (modifiée par le cerveau)

    Ce qui apprend : modules CfC+SNN (STDP) + gates d'injection (feedback)
    Ce qui reste gelé : le Qwen entier
    """

    def __init__(self, config: BrainConfig = None):
        super().__init__()
        self.config = config or BrainConfig()

        # Noyau LLM — gelé
        self.llm = QwenWrapper(
            self.config.model_name,
            use_quantization=self.config.use_quantization,
        )
        self.device = self.llm.device

        # 4 modules CfC+SNN aux points d'intercept
        self.brain_modules = nn.ModuleList([
            BrainModule(
                input_dim=self.config.llm_hidden_dim,
                hidden_dim=self.config.module_hidden_dim,
                device=self.device
            )
            for _ in range(self.config.n_modules)
        ])

        # STDP indépendant pour chaque module
        self.stdp_learners = [
            STDPLearning(
                lr_plus=self.config.stdp_lr_plus,
                lr_minus=self.config.stdp_lr_minus
            )
            for _ in range(self.config.n_modules)
        ]

        # Gates d'injection CfC → Qwen
        self.injection_gates = nn.ModuleList([
            InjectionGate(
                hidden_dim=self.config.llm_hidden_dim,
                rank=self.config.injection_gate_rank,
                init_alpha=self.config.injection_init_alpha,
            ).to(self.device)
            for _ in range(self.config.n_modules)
        ])

        # Hook manager
        self.hook_manager = BrainHookManager()

        # Hippocampe — mémoire épisodique persistante
        self.hippocampus = HippocampalMemory(
            address_dim=self.config.memory_address_dim,
            content_dim=self.config.llm_hidden_dim,
            n_locations=self.config.memory_n_locations
        )

        # Prefix injector hippocampe
        self.hippocampal_injector = HippocampalPrefixInjector(
            content_dim=self.config.llm_hidden_dim,
            n_prefix_tokens=self.config.injection_n_prefix_tokens,
        ).to(self.device)

        # État global persistant
        self.global_state = torch.zeros(
            self.config.state_dim,
            device=self.device
        )

        # Historique des métriques
        self.error_history = []
        self.step_count = 0

    def forward(self, prompt: str, image=None, learn: bool = True) -> dict:
        """
        Traitement complet : observation + apprentissage + génération influencée.
        """
        # === Phase 1 : Observation + Apprentissage ===
        layer_reps = self.llm.get_layer_representations(
            prompt,
            layers=self.config.intercept_layers,
            image=image,
        )

        errors = []

        for i, (module, stdp) in enumerate(
            zip(self.brain_modules, self.stdp_learners)
        ):
            prediction, spikes = module(layer_reps[i])

            if i < len(layer_reps) - 1:
                error = module.compute_prediction_error(
                    prediction, layer_reps[i + 1]
                )
                error_val = error.abs().mean().item()
                errors.append(error_val)

                stdp.update_dopamine(error_val, self.config.dopamine_threshold)

                if learn and module.pre_trace is not None:
                    for param in module.cfc.parameters():
                        if param.requires_grad and len(param.shape) == 2:
                            stdp.apply(
                                param,
                                module.pre_trace,
                                module.post_trace
                            )

        # Mettre à jour l'état global
        last_rep_mean = layer_reps[-1].float().mean(dim=1).squeeze().detach()
        if last_rep_mean.shape[0] >= self.config.state_dim:
            last_rep_mean = last_rep_mean[:self.config.state_dim]
        self.global_state = (
            0.95 * self.global_state +
            0.05 * last_rep_mean.to(self.device)
        )

        # Stocker dans l'hippocampe si saillant
        mean_error = sum(errors) / len(errors) if errors else 0.0
        self.error_history.append(mean_error)
        self.step_count += 1

        if learn and mean_error > self.config.salience_threshold:
            self.hippocampus.write(
                embedding=layer_reps[0].mean(dim=1).squeeze(),
                content=layer_reps[-1].mean(dim=1).squeeze(),
                metadata={
                    'step': self.step_count,
                    'prompt': prompt[:100],
                    'error': mean_error
                }
            )

        # === Phase 2 : Génération influencée ===
        if self.config.injection_enabled:
            response = self.llm.generate_with_brain(
                prompt,
                brain_modules=self.brain_modules,
                injection_gates=self.injection_gates,
                hook_manager=self.hook_manager,
                image=image,
            )
        else:
            response = self.llm.generate(prompt, image=image)

        # Feedback adaptatif sur les gates
        if learn:
            self._adjust_gates()

        return {
            'response': response,
            'prediction_errors': errors,
            'mean_error': mean_error,
            'dopamine': [s.dopamine_signal for s in self.stdp_learners],
            'memories_stored': len(self.hippocampus.metadata),
            'global_state_norm': self.global_state.norm().item(),
            'gate_alphas': [g.alpha.item() for g in self.injection_gates],
            'step': self.step_count
        }

    def _adjust_gates(self):
        """
        Ajuste les gates selon la tendance de l'erreur.
        Erreur baisse → ouvrir les gates (plus d'influence CfC)
        Erreur monte → fermer les gates (protéger la qualité)
        """
        if len(self.error_history) < 20:
            return

        recent = self.error_history[-10:]
        older = self.error_history[-20:-10]
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)

        ratio = recent_mean / older_mean if older_mean > 0 else 1.0

        for gate in self.injection_gates:
            with torch.no_grad():
                if ratio < 0.95:
                    gate.alpha.data *= 1.01
                    gate.alpha.data.clamp_(max=self.config.injection_max_alpha)
                elif ratio > 1.05:
                    gate.alpha.data *= 0.95
                    gate.alpha.data.clamp_(min=1e-5)

    def remember(self, query: str) -> torch.Tensor:
        """Interroge l'hippocampe depuis une requête textuelle."""
        query_reps = self.llm.get_layer_representations(
            query, layers=[self.config.intercept_layers[0]]
        )
        query_emb = query_reps[0].mean(dim=1).squeeze()
        return self.hippocampus.read(query_emb)

    def save_state(self, path: str):
        """Sauvegarde complète — modules + gates + hippocampe + état global."""
        torch.save({
            'brain_modules': self.brain_modules.state_dict(),
            'injection_gates': self.injection_gates.state_dict(),
            'hippocampal_injector': self.hippocampal_injector.state_dict(),
            'hippocampus_contents': self.hippocampus.contents,
            'hippocampus_counts': self.hippocampus.access_counts,
            'hippocampus_metadata': self.hippocampus.metadata,
            'global_state': self.global_state,
            'error_history': self.error_history,
            'step_count': self.step_count,
            'stdp_dopamine': [s.dopamine_signal for s in self.stdp_learners],
            'config': self.config,
        }, path)
        print(f"Checkpoint sauvegardé : {path}")

    def load_state(self, path: str):
        """Reprend depuis un checkpoint — aucune perte de mémoire."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.brain_modules.load_state_dict(ckpt['brain_modules'])
        if 'injection_gates' in ckpt:
            self.injection_gates.load_state_dict(ckpt['injection_gates'])
        if 'hippocampal_injector' in ckpt:
            self.hippocampal_injector.load_state_dict(ckpt['hippocampal_injector'])
        self.hippocampus.contents = ckpt['hippocampus_contents'].cpu()
        self.hippocampus.access_counts = ckpt['hippocampus_counts'].cpu()
        self.hippocampus.metadata = ckpt['hippocampus_metadata']
        self.global_state = ckpt['global_state'].to(self.device)
        self.error_history = ckpt['error_history']
        self.step_count = ckpt['step_count']
        for i, d in enumerate(ckpt['stdp_dopamine']):
            self.stdp_learners[i].dopamine_signal = d
        print(f"Reprise depuis step {self.step_count}")
