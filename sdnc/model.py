"""SDNC Model — Circuit-centric architecture.

The circuits BUILD the representation. Encoders are just translators.

    Encoder (frozen) → features → Router → CfC Circuits → circuit_repr
                                                              ↓
                                                    prototypical classifier
                                                    operates HERE, not in CLIP space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sdnc.config import SDNCConfig
from sdnc.core.circuit_bank import CircuitBank
from sdnc.core.hebbian_update import (
    apply_hebbian_to_circuit,
    apply_hebbian_to_cfc,
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
    """Sparse Dynamic Neural Circuits — Circuit-centric architecture.

    Key change: circuits produce the FINAL representation.
    Prototypical comparison happens in circuit space, not CLIP space.

        Encoder → 512-dim → Router → CfC Circuits → circuit_repr (128-dim)
                                                          ↓
                                                  classify HERE
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

        # Circuit bank (the core)
        self.circuit_bank = CircuitBank(self.config)
        self.integrator = TemporalIntegrator(self.config)

        # Projection: circuit output → circuit representation space
        # This is where the circuits build their own representation
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
                + list(self.circuit_bank.parameters())     # CfC weights!
                + list(self.circuit_proj.parameters())
                + list(self.integrator.parameters())       # temporal integrator!
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
        """Full forward: encoder → router → circuits → circuit_repr.

        The output lives in CIRCUIT space, not CLIP space.
        """
        # 1. Encode (frozen) — just feature extraction
        encoder_features = self.encode(**kwargs)

        # 2. Integrate temporal context
        integrated = self.integrator(encoder_features)

        # 3. Route to sparse circuits
        indices, weights = self.router(integrated)

        # 4. Process through CfC circuits
        circuit_out, hidden_states = self.circuit_bank.forward_token_choice(
            integrated, indices, weights
        )

        # 5. Project to circuit representation space
        circuit_repr = self.circuit_proj(circuit_out)
        circuit_repr = circuit_repr / (circuit_repr.norm(dim=-1, keepdim=True) + 1e-8)

        return {
            "encoder_features": encoder_features,
            "circuit_repr": circuit_repr,      # THIS is the representation now
            "active_indices": indices,
            "weights": weights,
            "circuit_outputs_raw": circuit_out,
            "hidden_states": hidden_states,
        }

    def learn(self, labels: torch.Tensor | None = None, **kwargs):
        """Learn via prototypical loss in circuit space + Hebbian updates."""
        result = self.forward(**kwargs)

        # Prototypical loss in CIRCUIT space (not CLIP space)
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

                # Entropy regularization on router
                if hasattr(self.router, 'entropy_loss'):
                    loss = loss + 0.1 * self.router.entropy_loss(result["encoder_features"].detach())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

        # Hebbian on CfC internals
        with torch.no_grad():
            if result["hidden_states"] is not None:
                apply_hebbian_to_cfc(
                    self.circuit_bank.circuit.cfc,
                    result["encoder_features"].detach(),
                    result["hidden_states"].detach(),
                    lr=self.config.hebbian_lr,
                )

        # Inter-circuit wiring
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

            # Prototypes in CIRCUIT space
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

    def prune(self):
        synaptic_pruning(self.circuit_bank.circuit.cfc, threshold=self.config.pruning_threshold)
        with torch.no_grad():
            mask = self.circuit_bank.inter_circuit_weights.abs() > self.config.inter_circuit_prune_threshold
            self.circuit_bank.inter_circuit_weights *= mask.float()

    def reset(self):
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
