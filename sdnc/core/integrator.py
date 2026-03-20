"""Temporal integrator: synchronizes multi-modal inputs.

Maintains a persistent state that encodes "what is happening now".
In Phase 1 (vision only), acts as a state compressor between
the encoder output and the circuit bank.
"""

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

from sdnc.config import SDNCConfig


class TemporalIntegrator(nn.Module):
    """Central CfC integrator with persistent state.

    Compresses and integrates input signals over time.
    State never resets between calls (persistent memory).
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        # AutoNCP requires output_size < units - 2
        # Use a small CfC core + linear projection back to input_dim
        cfc_output = max(2, config.integrator_state_dim // 4)
        wiring = AutoNCP(
            units=config.integrator_state_dim,
            output_size=cfc_output,
            sparsity_level=0.5,
        )
        self.cfc = CfC(
            input_size=config.input_dim,
            units=wiring,
            batch_first=True,
            return_sequences=False,
        )
        self.proj = nn.Linear(cfc_output, config.input_dim)
        self._state_size = wiring.units

        # Persistent state — survives across calls
        self.register_buffer("state", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate input into persistent state.

        Args:
            x: (batch, input_dim)

        Returns:
            Integrated representation (batch, input_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        batch_size = x.shape[0]

        # Initialize state on first call or batch size change
        if self.state is None or self.state.shape[0] != batch_size:
            self.state = torch.zeros(
                batch_size, self._state_size, device=x.device
            )

        cfc_out, self.state = self.cfc(x, hx=self.state)
        self.state = self.state.detach()  # prevent gradient accumulation

        return self.proj(cfc_out)

    def reset_state(self):
        """Reset integrator state."""
        self.state = None
