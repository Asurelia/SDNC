"""Global State — persistent contextual prior shared across all circuits.

The "what I know so far" tensor that persists between ALL inputs.
Every circuit reads from it and writes to it simultaneously.

This is the contextual prior from predictive coding:
  - Before seeing input: global state biases interpretation
  - After processing: circuits update global state with new knowledge

Uses lightweight attention for read/write — not a full transformer,
just enough to let circuits influence and be influenced by the whole.
"""

import torch
import torch.nn as nn

from sdnc.config import SDNCConfig


class GlobalState(nn.Module):
    """Persistent global state — the contextual prior.

    A fixed-size tensor that accumulates knowledge across ALL inputs.
    Circuits read it to bias their processing, then write back updates.

    Properties:
        - Persists across forward calls (never reset during training)
        - Read via lightweight attention (query from input)
        - Write via gated update (what to remember, what to forget)
        - Dimensionality matches input_dim for seamless integration
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config
        dim = config.input_dim
        n_slots = config.global_state_slots

        # The global state: (n_slots, dim) — multiple "memory slots"
        # Registered as buffer so it persists but doesn't need grad
        self.register_buffer(
            "state", torch.zeros(n_slots, dim)
        )

        # READ: lightweight attention — project input to query
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

        # WRITE: gated update — what to add, what to forget
        self.write_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.write_value = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
        )

        # Decay — gradual forgetting of old information
        self.decay_rate = config.global_state_decay

    def read(self, x: torch.Tensor) -> torch.Tensor:
        """Read from global state — attention-weighted context.

        Args:
            x: (batch, dim) — current input/representation

        Returns:
            context: (batch, dim) — weighted sum of global state slots
        """
        # x queries a snapshot of global state (detached to avoid inplace conflicts)
        state_snapshot = self.state.detach().clone()
        q = self.query_proj(x)  # (batch, dim)
        k = self.key_proj(state_snapshot)  # (n_slots, dim)

        # Attention: (batch, n_slots)
        attn = torch.matmul(q, k.t()) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # Weighted read: (batch, dim)
        context = torch.matmul(attn, state_snapshot)
        return context

    def write(self, x: torch.Tensor) -> None:
        """Write to global state — gated update from processed input.

        Args:
            x: (batch, dim) — circuit output or prediction error to integrate
        """
        # Mean across batch — global state is shared, not per-sample
        x_mean = x.detach().mean(dim=0)  # (dim,)

        for i in range(self.state.shape[0]):
            slot = self.state[i]  # (dim,)
            combined = torch.cat([slot, x_mean])  # (2*dim,)

            gate = self.write_gate(combined)  # (dim,) — how much to update
            value = self.write_value(combined)  # (dim,) — what to write

            # Gated update: keep old * (1-gate) + new * gate
            self.state[i] = (1 - gate) * slot + gate * value

    def apply_decay(self) -> None:
        """Gradual forgetting — prevents state from saturating."""
        self.state.data *= self.decay_rate

    def get_state_summary(self) -> dict:
        """Diagnostic: summarize global state for logging."""
        return {
            "state_norm": self.state.norm().item(),
            "state_mean": self.state.mean().item(),
            "state_std": self.state.std().item(),
            "slot_norms": self.state.norm(dim=-1).tolist(),
        }

    def reset(self):
        """Hard reset — only for completely new contexts."""
        self.state.zero_()
