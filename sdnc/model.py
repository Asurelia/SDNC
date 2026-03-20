"""SDNC Model — Predictive Coding Architecture.

The brain doesn't passively receive — it PREDICTS, then corrects.

New forward flow:
    1. Read global state (contextual prior)
    2. Enrich input with temporal stream context
    3. Generate predictions on expected input
    4. Receive actual input → compute prediction error
    5. Update circuits via Hebbian on prediction error
    6. Write new knowledge to global state
    7. Return emergent understanding

    Encoder (frozen) → temporal context → prior-biased input
                                              ↓
                    Router → Predictive Circuits → prediction + error
                                              ↓
                              circuit_repr + global state update
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sdnc.config import SDNCConfig
from sdnc.core.predictive_circuits import PredictiveCircuitBank
from sdnc.core.global_state import GlobalState
from sdnc.core.temporal_stream import TemporalStream
from sdnc.core.hebbian_update import (
    apply_hebbian_to_cfc,
    oja_update,
    synaptic_pruning,
    strengthen_co_active_circuits,
)
from sdnc.core.integrator import TemporalIntegrator
from sdnc.core.sparse_router import SparseRouter, ExpertChoiceRouter
from sdnc.encoders.vision_encoder import VisionEncoder
from sdnc.encoders.text_encoder import TextEncoder
from sdnc.encoders.audio_encoder import AudioEncoder
from sdnc.encoders.signal_encoder import SignalEncoder
from sdnc.memory.consolidation import SalienceDetector
from sdnc.memory.episodic_memory import EpisodicMemory


class SDNCModel(nn.Module):
    """Sparse Dynamic Neural Circuits — Predictive Coding Architecture.

    Key changes from v1:
      - Circuits PREDICT next input before receiving it
      - Prediction error is the primary learning signal
      - Global state persists as contextual prior across ALL inputs
      - Temporal stream provides history — no isolated processing

        Encoder → Stream Context → Prior Bias → Router → Predictive Circuits
                                                              ↓
                                              prediction error → Hebbian update
                                              circuit_repr → classification
                                              global state ← write back
    """

    def __init__(self, config: SDNCConfig | None = None):
        super().__init__()
        self.config = config or SDNCConfig()

        # Encoders — frozen feature extractors
        self.vision_encoder = VisionEncoder(self.config)
        self.text_encoder = TextEncoder(self.config)
        self.audio_encoder = AudioEncoder(self.config)
        self.signal_encoder = SignalEncoder(self.config)

        # Router
        if self.config.use_expert_choice:
            self.router = ExpertChoiceRouter(
                input_dim=self.config.input_dim,
                n_circuits=self.config.n_circuits,
                capacity_factor=self.config.expert_capacity_factor,
            )
        else:
            self.router = SparseRouter(
                input_dim=self.config.input_dim,
                n_circuits=self.config.n_circuits,
                k=self.config.sparsity_k,
                noise_std=self.config.router_noise_std,
            )

        # NEW: Predictive circuit bank (replaces CircuitBank)
        self.circuit_bank = PredictiveCircuitBank(self.config)

        # NEW: Global state — contextual prior
        self.global_state = GlobalState(self.config)

        # NEW: Temporal stream — input history
        self.temporal_stream = TemporalStream(self.config)

        # Temporal integrator (kept — synchronizes modalities)
        self.integrator = TemporalIntegrator(self.config)

        # Projection: circuit output → circuit representation space
        self.circuit_proj = nn.Sequential(
            nn.Linear(self.circuit_bank.output_size, self.config.circuit_repr_dim),
            nn.GELU(),
            nn.Linear(self.config.circuit_repr_dim, self.config.circuit_repr_dim),
        )

        # Memory
        self.episodic_memory = EpisodicMemory(
            embedding_dim=self.config.circuit_repr_dim,
            max_memories=self.config.memory_capacity,
        )
        self.salience_detector = SalienceDetector(
            threshold=self.config.salience_threshold,
        )

        # Optimizer for trainable components
        self._optimizer = None
        self.encoder = self.vision_encoder  # backward compat

    def _get_optimizer(self):
        if self._optimizer is None:
            params = (
                list(self.router.parameters())
                + list(self.circuit_bank.parameters())
                + list(self.circuit_proj.parameters())
                + list(self.integrator.parameters())
                + list(self.global_state.parameters())
                + list(self.temporal_stream.parameters())
                + list(self.audio_encoder.proj.parameters())
                + list(self.signal_encoder.parameters())
            )
            self._optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        return self._optimizer

    def encode(self, **kwargs) -> torch.Tensor:
        """Encode any modality to encoder feature space (512-dim)."""
        if kwargs.get("images") is not None:
            return self.vision_encoder(kwargs["images"])
        elif kwargs.get("text_tokens") is not None:
            return self.text_encoder(kwargs["text_tokens"])
        elif kwargs.get("audio") is not None:
            return self.audio_encoder(kwargs["audio"])
        elif kwargs.get("signals") is not None:
            return self.signal_encoder(kwargs["signals"])
        else:
            raise ValueError("Must provide at least one modality")

    def forward(self, **kwargs) -> dict:
        """Predictive forward: predict → receive → error → update → understand.

        Flow:
            1. Encode raw input (frozen)
            2. Read global state as contextual prior
            3. Enrich with temporal stream history
            4. Integrate temporal context (CfC)
            5. Route to sparse predictive circuits
            6. Circuits predict + process → prediction error
            7. Push to temporal stream
            8. Write to global state
            9. Project to circuit representation
        """
        # 1. Encode (frozen)
        encoder_features = self.encode(**kwargs)

        # 2. Read global state — prior context biases interpretation
        prior_context = self.global_state.read(encoder_features)
        prior_biased = encoder_features + prior_context  # residual fusion

        # 3. Enrich with temporal stream — see present in context of past
        stream_context = self.temporal_stream.get_context(prior_biased)

        # 4. Integrate temporal context (CfC integrator)
        integrated = self.integrator(stream_context)

        # 5. Route to sparse circuits
        if self.config.use_expert_choice:
            routing_info, top_weights, token_counts = self.router(integrated)
            circuit_result = self.circuit_bank.forward_expert_choice(
                integrated, routing_info, top_weights
            )
            # Reconstruct indices for compatibility
            assignments, weights_r, circuit_map, counts = routing_info
            active_indices = circuit_map.unsqueeze(0).expand(encoder_features.shape[0], -1)
            route_weights = top_weights[:encoder_features.shape[0]] if top_weights.shape[0] >= encoder_features.shape[0] else top_weights
        else:
            indices, weights = self.router(integrated)
            circuit_result = self.circuit_bank.forward_token_choice(
                integrated, indices, weights
            )
            active_indices = indices
            route_weights = weights

        # 6. Push input to temporal stream (for next call's history)
        self.temporal_stream.push(encoder_features)

        # 7. Write prediction error to global state (new knowledge)
        # NOTE: write is detached inside — no grad flows through global state
        with torch.no_grad():
            self.global_state.write(circuit_result["prediction_error"])
            # 8. Apply gradual decay to global state
            self.global_state.apply_decay()

        # 9. Project to circuit representation space
        circuit_repr = self.circuit_proj(circuit_result["circuit_output"])
        circuit_repr = circuit_repr / (circuit_repr.norm(dim=-1, keepdim=True) + 1e-8)

        return {
            "encoder_features": encoder_features,
            "circuit_repr": circuit_repr,
            "active_indices": active_indices,
            "weights": route_weights,
            "circuit_outputs_raw": circuit_result["circuit_output"],
            "hidden_states": circuit_result["hidden_states"],
            # NEW: predictive coding signals
            "prediction": circuit_result["prediction"],
            "prediction_error": circuit_result["prediction_error"],
            "prediction_error_norm": circuit_result["prediction_error"].norm(dim=-1).mean(),
            "global_state_summary": self.global_state.get_state_summary(),
            "stream_stats": self.temporal_stream.get_stream_stats(),
        }

    def learn(self, labels: torch.Tensor | None = None, **kwargs):
        """Learn via prediction error + prototypical loss in circuit space.

        Two learning signals:
            1. Prediction error → Hebbian update (local, no backprop)
            2. Prototypical loss → gradient update (router + projections)
        """
        result = self.forward(**kwargs)

        # === SIGNAL 1: Prototypical loss in circuit space (backprop) ===
        if labels is not None and len(labels.unique()) > 1:
            opt = self._get_optimizer()
            opt.zero_grad()

            repr_ = result["circuit_repr"]
            n_way = labels.max().item() + 1
            prototypes = []
            for c in range(n_way):
                mask = labels == c
                if mask.sum() > 0:
                    prototypes.append(repr_[mask].mean(dim=0))

            if len(prototypes) == n_way:
                prototypes_t = torch.stack(prototypes)
                dists = torch.cdist(repr_, prototypes_t)
                log_p = F.log_softmax(-dists, dim=-1)
                loss = F.nll_loss(log_p, labels)

                # Add prediction error as auxiliary loss
                pred_error_loss = result["prediction_error"].norm(dim=-1).mean()
                loss = loss + self.config.prediction_error_weight * pred_error_loss

                # Entropy regularization on router
                if hasattr(self.router, 'entropy_loss'):
                    loss = loss + 0.1 * self.router.entropy_loss(
                        result["encoder_features"].detach()
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

        # === SIGNAL 2: Hebbian update driven by prediction error (local) ===
        with torch.no_grad():
            if result["hidden_states"] is not None:
                # Scale Hebbian LR by prediction error magnitude
                pred_error_scale = result["prediction_error_norm"].item()
                scaled_lr = self.config.prediction_hebbian_lr * min(pred_error_scale, 5.0)

                apply_hebbian_to_cfc(
                    self.circuit_bank.circuit.processing_cfc,
                    result["encoder_features"].detach(),
                    result["hidden_states"].detach(),
                    lr=scaled_lr,
                )

        # === Inter-circuit wiring ===
        with torch.no_grad():
            self._update_inter_circuit_wiring(result)

        if not self.config.use_expert_choice and hasattr(self.router, 'update_bias'):
            self.router.update_bias(gamma=self.config.router_bias_gamma)

        return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in result.items()}

    def _update_inter_circuit_wiring(self, result):
        active = result["active_indices"]
        device = active.device
        co_act = torch.zeros(
            self.circuit_bank.n_circuits, self.circuit_bank.n_circuits, device=device,
        )
        if not self.config.use_expert_choice:
            batch_size = active.shape[0]
            k = active.shape[1]
            for b in range(batch_size):
                indices = active[b]
                for i in range(k):
                    for j in range(k):
                        if i != j:
                            co_act[indices[i], indices[j]] += 1.0

        self.circuit_bank.inter_circuit_weights = strengthen_co_active_circuits(
            co_act,
            self.circuit_bank.inter_circuit_weights,
            lr=self.config.inter_circuit_hebbian_lr,
            decay=self.config.inter_circuit_decay,
        )

    def recognize(
        self,
        support_labels: torch.Tensor,
        support_kwargs: dict | None = None,
        query_kwargs: dict | None = None,
        # Legacy
        support_images: torch.Tensor | None = None,
        query_images: torch.Tensor | None = None,
        support_text: torch.Tensor | None = None,
        query_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Few-shot recognition in CIRCUIT SPACE."""
        if support_kwargs is None:
            support_kwargs = {}
            if support_images is not None:
                support_kwargs["images"] = support_images
            elif support_text is not None:
                support_kwargs["text_tokens"] = support_text
        if query_kwargs is None:
            query_kwargs = {}
            if query_images is not None:
                query_kwargs["images"] = query_images
            elif query_text is not None:
                query_kwargs["text_tokens"] = query_text

        with torch.no_grad():
            support_result = self.forward(**support_kwargs)
            query_result = self.forward(**query_kwargs)

            support_repr = support_result["circuit_repr"]
            query_repr = query_result["circuit_repr"]

            n_way = support_labels.max().item() + 1
            prototypes = []
            for c in range(n_way):
                mask = support_labels == c
                prototypes.append(support_repr[mask].mean(dim=0))
            prototypes = torch.stack(prototypes)

            dists = torch.cdist(query_repr, prototypes)
            return -dists

    def get_activation_pattern(self, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            result = self.forward(**kwargs)
            return result["active_indices"]

    def get_prediction_error(self, **kwargs) -> dict:
        """Get prediction error without learning — for evaluation."""
        with torch.no_grad():
            result = self.forward(**kwargs)
            return {
                "prediction_error_norm": result["prediction_error_norm"].item(),
                "prediction_error": result["prediction_error"],
                "prediction": result["prediction"],
            }

    def prune(self):
        synaptic_pruning(
            self.circuit_bank.circuit.processing_cfc,
            threshold=self.config.pruning_threshold,
        )
        with torch.no_grad():
            mask = (
                self.circuit_bank.inter_circuit_weights.abs()
                > self.config.inter_circuit_prune_threshold
            )
            self.circuit_bank.inter_circuit_weights *= mask.float()

    def reset(self):
        """Full reset — circuits, integrator, global state, stream."""
        self.circuit_bank.reset_states()
        self.integrator.reset_state()
        self.global_state.reset()
        self.temporal_stream.reset()

    def soft_reset(self):
        """Soft reset — keep global state and stream, reset circuits only."""
        self.circuit_bank.reset_states()
        self.integrator.reset_state()

    def get_circuit_wiring_stats(self) -> dict:
        w = self.circuit_bank.inter_circuit_weights
        nonzero = (w.abs() > 1e-8).sum().item()
        total = w.numel() - w.shape[0]
        return {
            "nonzero_connections": nonzero,
            "total_possible": total,
            "density": nonzero / total if total > 0 else 0,
            "mean_weight": w[w.abs() > 1e-8].mean().item() if nonzero > 0 else 0,
            "max_weight": w.max().item(),
        }

    def get_predictive_stats(self) -> dict:
        """Diagnostic: prediction error history across circuits."""
        err_hist = self.circuit_bank.prediction_error_history
        active_mask = self.circuit_bank.activation_history > 0
        return {
            "mean_prediction_error": err_hist[active_mask].mean().item() if active_mask.any() else 0,
            "min_prediction_error": err_hist[active_mask].min().item() if active_mask.any() else 0,
            "max_prediction_error": err_hist[active_mask].max().item() if active_mask.any() else 0,
            "n_active_circuits": active_mask.sum().item(),
            "global_state": self.global_state.get_state_summary(),
            "temporal_stream": self.temporal_stream.get_stream_stats(),
        }
