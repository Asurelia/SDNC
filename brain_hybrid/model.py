"""
BrainHybridModel — Cerveau artificiel complet.

Assemble Qwen (gelé) + CfC+SNN + STDP + Hippocampe + Injection +
ACC (détection conflit) + StepScheduler (rythmes) + Predictive Coding.
"""

import torch
import torch.nn as nn
from .config import BrainConfig
from .llm.qwen_wrapper import QwenWrapper
from .core.brain_module import BrainModule
from .core.stdp import STDPLearning
from .core.injection import InjectionGate, BrainHookManager, HippocampalPrefixInjector
from .core.acc import ACCModule
from .core.predictive_coding import HierarchicalPC
from .memory.hippocampus import HippocampalMemory
from .utils.device import get_device, HW_CONFIG
from .utils.step_scheduler import StepScheduler


class BrainHybridModel(nn.Module):
    """
    Modèle principal — cerveau artificiel complet.

    Flux :
      1. scheduler.step() — rythmes cérébraux
      2. Extraire layer_reps depuis Qwen
      3. Codage prédictif hiérarchique (PC) → erreurs + précisions
      4. BrainModules CfC+SNN → prédictions + spikes
      5. ACC → détection conflit → enrichissement prompt si nécessaire
      6. Génération influencée via hooks CfC
      7. Hippocampe conditionné par scheduler + saillance
      8. Sleep mode : replay épisodique si scheduler le déclenche
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

        # 4 modules CfC+SNN
        self.brain_modules = nn.ModuleList([
            BrainModule(
                input_dim=self.config.llm_hidden_dim,
                hidden_dim=self.config.module_hidden_dim,
                device=self.device
            )
            for _ in range(self.config.n_modules)
        ])

        # STDP
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

        # Hippocampe
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

        # ACC — détection de conflit
        self.acc = ACCModule(
            n_modules=self.config.n_modules,
            conflict_threshold=self.config.acc_conflict_threshold,
            exploration_threshold=self.config.acc_exploration_threshold,
            lr=self.config.acc_lr,
            device=self.device,
        )

        # Scheduler — rythmes cérébraux
        self.scheduler = StepScheduler(
            snn_freq=self.config.snn_freq,
            cfc_freq=self.config.cfc_freq,
            hippocampus_freq=self.config.hippocampus_freq,
            qwen_freq=self.config.qwen_freq,
            sleep_freq=self.config.sleep_freq,
            arousal_boost_steps=self.config.arousal_boost_steps,
        )

        # Codage prédictif hiérarchique
        self.pc = HierarchicalPC(
            brain_modules=self.brain_modules,
            pc_lr=self.config.pc_lr,
        )

        # État global
        self.global_state = torch.zeros(
            self.config.state_dim,
            device=self.device
        )

        self.error_history = []
        self.step_count = 0

    def forward(self, prompt: str, image=None, learn: bool = True) -> dict:
        """Traitement complet avec rythmes, PC, ACC, injection, sleep."""

        # 1. Scheduler step
        self.scheduler.step()
        step = self.scheduler.global_step

        # 2. Extraire hidden states Qwen
        layer_reps = self.llm.get_layer_representations(
            prompt,
            layers=self.config.intercept_layers,
            image=image,
        )

        # 3. Codage prédictif hiérarchique
        pc_errors = self.pc.forward(layer_reps)

        # 4. BrainModules + STDP (conditionné par scheduler)
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

                # STDP conditionné par scheduler
                if learn and module.pre_trace is not None:
                    if self.scheduler.should_update("cfc", step):
                        for param in module.cfc.parameters():
                            if param.requires_grad and len(param.shape) == 2:
                                stdp.apply(param, module.pre_trace, module.post_trace)

        # PC update (si scheduler autorise)
        if learn and self.scheduler.should_update("cfc", step):
            dopamine_vals = [s.dopamine_signal for s in self.stdp_learners]
            self.pc.update_all(pc_errors, dopamine_vals, self.stdp_learners)

        # 5. ACC — détection de conflit
        dopamine_list = [s.dopamine_signal for s in self.stdp_learners]
        acc_output = self.acc.forward(errors, dopamine_list)

        # Si conflit → arousal boost
        if acc_output.is_conflict:
            self.scheduler.arousal_boost()

        # Mettre à jour l'état global
        mean_error = sum(errors) / len(errors) if errors else 0.0
        self.error_history.append(mean_error)
        self.step_count += 1

        last_rep_mean = layer_reps[-1].float().mean(dim=1).squeeze().detach()
        if last_rep_mean.shape[0] >= self.config.state_dim:
            last_rep_mean = last_rep_mean[:self.config.state_dim]
        self.global_state = (
            0.95 * self.global_state +
            0.05 * last_rep_mean.to(self.device)
        )

        # 6. Génération — enrichir le prompt si conflit + scheduler autorise Qwen
        gen_prompt = prompt
        if acc_output.is_conflict and self.scheduler.should_update("qwen", step):
            gen_prompt = acc_output.conflict_prompt_fragment + prompt

        if self.config.injection_enabled:
            response = self.llm.generate_with_brain(
                gen_prompt,
                brain_modules=self.brain_modules,
                injection_gates=self.injection_gates,
                hook_manager=self.hook_manager,
                image=image,
            )
        else:
            response = self.llm.generate(gen_prompt, image=image)

        # 7. Hippocampe — conditionné par scheduler + saillance ACC
        if learn and self.scheduler.should_update("hippocampus", step):
            if acc_output.conflict_score > self.config.salience_threshold:
                self.hippocampus.write(
                    embedding=layer_reps[0].mean(dim=1).squeeze(),
                    content=layer_reps[-1].mean(dim=1).squeeze(),
                    metadata={
                        'step': self.step_count,
                        'prompt': prompt[:100],
                        'error': mean_error,
                        'conflict': acc_output.conflict_score,
                    }
                )

        # 8. Sleep — consolidation offline
        sleep_triggered = False
        if learn and self.scheduler.should_update("sleep", step):
            sleep_triggered = self._sleep_consolidation()

        # Feedback adaptatif sur les gates
        if learn:
            self._adjust_gates()

        # Flush VRAM si nécessaire (AMD)
        if HW_CONFIG.get("empty_cache_after_qwen") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'response': response,
            'prediction_errors': errors,
            'mean_error': mean_error,
            'dopamine': dopamine_list,
            'memories_stored': len(self.hippocampus.metadata),
            'global_state_norm': self.global_state.norm().item(),
            'gate_alphas': [g.alpha.item() for g in self.injection_gates],
            'conflict_score': acc_output.conflict_score,
            'is_conflict': acc_output.is_conflict,
            'acc_trend': self.acc.trend(),
            'pc_errors': self.pc.get_errors(),
            'pc_precision': self.pc.get_precisions(),
            'scheduler_phase': self.scheduler.get_phase(),
            'sleep_triggered': sleep_triggered,
            'step': self.step_count,
        }

    def _sleep_consolidation(self) -> bool:
        """
        Mode sommeil — replay les épisodes les plus saillants depuis l'hippocampe.
        Consolide les poids CfC via codage prédictif.
        """
        metadata = self.hippocampus.metadata
        if len(metadata) < 2:
            return False

        # Trier par erreur (les plus saillants d'abord)
        sorted_episodes = sorted(metadata, key=lambda m: m.get('error', 0), reverse=True)
        replay_count = min(self.config.sleep_replay_count, len(sorted_episodes))

        for episode in sorted_episodes[:replay_count]:
            prompt = episode.get('prompt', '')
            if not prompt:
                continue
            # Recall depuis l'hippocampe
            recalled = self.hippocampus.read(
                self.llm.get_layer_representations(
                    prompt, layers=[self.config.intercept_layers[0]]
                )[0].mean(dim=1).squeeze()
            )
            # Le recalled est un vecteur mémoire — on l'utilise pour moduler
            # les poids CfC via une règle locale simplifiée
            if recalled.norm().item() > 1e-6:
                dopamine_vals = [s.dopamine_signal for s in self.stdp_learners]
                self.pc.update_all([], dopamine_vals, self.stdp_learners)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def _adjust_gates(self):
        """Ajuste les gates selon la tendance de l'erreur."""
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
        """Interroge l'hippocampe."""
        query_reps = self.llm.get_layer_representations(
            query, layers=[self.config.intercept_layers[0]]
        )
        query_emb = query_reps[0].mean(dim=1).squeeze()
        return self.hippocampus.read(query_emb)

    def save_state(self, path: str):
        """Sauvegarde complète."""
        torch.save({
            'brain_modules': self.brain_modules.state_dict(),
            'injection_gates': self.injection_gates.state_dict(),
            'hippocampal_injector': self.hippocampal_injector.state_dict(),
            'acc': self.acc.save_state(),
            'scheduler': self.scheduler.save_state(),
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
        """Reprend depuis un checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.brain_modules.load_state_dict(ckpt['brain_modules'])
        if 'injection_gates' in ckpt:
            self.injection_gates.load_state_dict(ckpt['injection_gates'])
        if 'hippocampal_injector' in ckpt:
            self.hippocampal_injector.load_state_dict(ckpt['hippocampal_injector'])
        if 'acc' in ckpt:
            self.acc.load_state(ckpt['acc'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state(ckpt['scheduler'])
        self.hippocampus.contents = ckpt['hippocampus_contents'].cpu()
        self.hippocampus.access_counts = ckpt['hippocampus_counts'].cpu()
        self.hippocampus.metadata = ckpt['hippocampus_metadata']
        self.global_state = ckpt['global_state'].to(self.device)
        self.error_history = ckpt['error_history']
        self.step_count = ckpt['step_count']
        for i, d in enumerate(ckpt['stdp_dopamine']):
            self.stdp_learners[i].dopamine_signal = d
        print(f"Reprise depuis step {self.step_count}")
