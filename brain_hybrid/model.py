import torch
import torch.nn as nn
from .config import BrainConfig
from .llm.qwen_wrapper import QwenWrapper
from .core.brain_module import BrainModule
from .core.stdp import STDPLearning
from .memory.hippocampus import HippocampalMemory
from .utils.device import get_device


class BrainHybridModel(nn.Module):
    """
    Modèle principal — assemble Qwen + modules CfC+SNN + hippocampe.

    Flux de traitement :
      1. Input → QwenWrapper → hidden states couches 8, 16, 24, 36
      2. Chaque BrainModule reçoit une couche et prédit la suivante
      3. Erreur de prédiction → signal dopamine → STDP
      4. Si saillant → stockage dans l'hippocampe
      5. Qwen génère la réponse finale

    Ce qui apprend : uniquement les modules CfC+SNN (via STDP)
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

        # Hippocampe — mémoire épisodique persistante
        self.hippocampus = HippocampalMemory(
            address_dim=self.config.memory_address_dim,
            content_dim=self.config.llm_hidden_dim,
            n_locations=self.config.memory_n_locations
        )

        # État global persistant — jamais resetté
        self.global_state = torch.zeros(
            self.config.state_dim,
            device=self.device
        )

        # Historique des métriques
        self.error_history = []
        self.step_count = 0

    def forward(self, prompt: str, image=None, learn: bool = True) -> dict:
        """
        Traitement complet d'un input avec apprentissage STDP optionnel.

        prompt : texte
        image  : PIL.Image, path str, ou None (texte seul)
        """
        # Extraire les représentations internes du Qwen
        layer_reps = self.llm.get_layer_representations(
            prompt,
            layers=self.config.intercept_layers,
            image=image,
        )

        errors = []

        # Passer dans les 4 modules
        for i, (module, stdp) in enumerate(
            zip(self.brain_modules, self.stdp_learners)
        ):
            prediction, spikes = module(layer_reps[i])

            # Erreur de prédiction vs couche suivante
            if i < len(layer_reps) - 1:
                error = module.compute_prediction_error(
                    prediction, layer_reps[i + 1]
                )
                error_val = error.abs().mean().item()
                errors.append(error_val)

                # Signal dopamine depuis l'erreur
                stdp.update_dopamine(error_val, self.config.dopamine_threshold)

                # STDP si learning activé
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

        # Générer la réponse avec le Qwen
        response = self.llm.generate(prompt, image=image)

        return {
            'response': response,
            'prediction_errors': errors,
            'mean_error': mean_error,
            'dopamine': [s.dopamine_signal for s in self.stdp_learners],
            'memories_stored': len(self.hippocampus.metadata),
            'global_state_norm': self.global_state.norm().item(),
            'step': self.step_count
        }

    def remember(self, query: str) -> torch.Tensor:
        """
        Interroge l'hippocampe depuis une requête textuelle partielle.
        """
        query_reps = self.llm.get_layer_representations(query, layers=[8])
        query_emb = query_reps[0].mean(dim=1).squeeze()
        return self.hippocampus.read(query_emb)

    def save_state(self, path: str):
        """Sauvegarde complète — modules + hippocampe + état global."""
        torch.save({
            'brain_modules': self.brain_modules.state_dict(),
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
        self.hippocampus.contents = ckpt['hippocampus_contents'].cpu()
        self.hippocampus.access_counts = ckpt['hippocampus_counts'].cpu()
        self.hippocampus.metadata = ckpt['hippocampus_metadata']
        self.global_state = ckpt['global_state'].to(self.device)
        self.error_history = ckpt['error_history']
        self.step_count = ckpt['step_count']
        for i, d in enumerate(ckpt['stdp_dopamine']):
            self.stdp_learners[i].dopamine_signal = d
        print(f"Reprise depuis step {self.step_count}")
