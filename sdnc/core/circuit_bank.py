"""Circuit bank: the heart of SDNC.

A collection of N micro-circuits sharing weights, differentiated by state.
Supports both TokenChoice and ExpertChoice routing modes.

Key design: ONE shared CfC model, N independent hidden states.
Parameters are O(1), state is O(N).
"""

import torch
import torch.nn as nn

from sdnc.config import SDNCConfig
from sdnc.core.micro_circuit import MicroCircuit


class CircuitBank(nn.Module):
    """Bank of N micro-circuits with shared weights and independent states."""

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config
        self.n_circuits = config.n_circuits

        self.circuit = MicroCircuit(config)

        state_size = self.circuit.state_size
        self.register_buffer("circuit_states", torch.zeros(config.n_circuits, state_size))
        self.register_buffer("activation_history", torch.zeros(config.n_circuits))

        # Inter-circuit connection weights (Objective 3: NCP wiring)
        self.register_buffer(
            "inter_circuit_weights",
            torch.zeros(config.n_circuits, config.n_circuits),
        )

    @property
    def state_size(self) -> int:
        return self.circuit.state_size

    @property
    def output_size(self) -> int:
        return self.circuit.output_size

    def forward_token_choice(
        self, x: torch.Tensor, active_indices: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token-choice routing: each token picks k circuits."""
        batch_size = x.shape[0]
        k = active_indices.shape[1]
        output_dim = self.output_size

        all_outputs = torch.zeros(batch_size, k, output_dim, device=x.device)
        last_hidden = None

        for i in range(k):
            slot_indices = active_indices[:, i]
            selected_states = self.circuit_states[slot_indices]
            out, new_states = self.circuit(x, hx=selected_states)
            self.circuit_states[slot_indices] = new_states.detach()
            all_outputs[:, i, :] = out
            last_hidden = new_states

            with torch.no_grad():
                for idx in slot_indices:
                    self.activation_history[idx] += 1

        weighted = all_outputs * weights.unsqueeze(-1)
        return weighted.sum(dim=1), last_hidden

    def forward_expert_choice(
        self, x: torch.Tensor, routing_info: tuple, top_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert-choice routing with efficient grouped execution.

        Args:
            x: (batch, input_dim)
            routing_info: (assignments, weights, circuit_map, counts_per_circuit)
            top_weights: unused (weights are inside routing_info)

        Returns:
            output: (batch, output_dim)
            last_hidden: (last_batch, state_size)
        """
        assignments, weights, circuit_map, counts_per_circuit = routing_info
        batch_size = x.shape[0]
        output_dim = self.output_size
        device = x.device

        token_outputs = torch.zeros(batch_size, output_dim, device=device)
        token_weight_sums = torch.zeros(batch_size, 1, device=device)
        last_hidden = None

        n_active = circuit_map.shape[0]

        for i in range(n_active):
            cid = circuit_map[i].item()
            n_tokens = counts_per_circuit[i].item()
            if n_tokens == 0:
                continue

            token_ids = assignments[i, :n_tokens]
            w = weights[i, :n_tokens]

            circuit_inputs = x[token_ids]
            circuit_state = self.circuit_states[cid].unsqueeze(0).expand(n_tokens, -1)

            out, new_states = self.circuit(circuit_inputs, hx=circuit_state)

            self.circuit_states[cid] = new_states.detach().mean(dim=0)
            last_hidden = new_states

            weighted_out = out * w.unsqueeze(-1)
            for j in range(n_tokens):
                tid = token_ids[j]
                token_outputs[tid] += weighted_out[j]
                token_weight_sums[tid] += w[j]

            with torch.no_grad():
                self.activation_history[cid] += n_tokens

        # Normalize
        safe_sums = token_weight_sums.clamp(min=1e-8)
        token_outputs = token_outputs / safe_sums

        return token_outputs, last_hidden

    # Keep backward-compatible forward
    def forward(
        self, x: torch.Tensor, active_indices: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_token_choice(x, active_indices, weights)

    def get_states(self, indices: torch.Tensor) -> torch.Tensor:
        return self.circuit_states[indices]

    def reset_states(self):
        self.circuit_states.zero_()

    def decay_states(self, rate: float = 0.99):
        self.circuit_states *= rate
