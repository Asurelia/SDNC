"""Micro-circuit: a minimal CfC liquid neural network unit.

Each micro-circuit responds to ONE concept via sparse NCP wiring.
Its internal state evolves DURING inference (liquid property).
"""

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import NCP

from sdnc.config import SDNCConfig


def build_wiring(config: SDNCConfig) -> NCP:
    """Build NCP wiring from config."""
    return NCP(
        inter_neurons=config.ncp_inter_neurons,
        command_neurons=config.ncp_command_neurons,
        motor_neurons=config.circuit_output_dim,
        sensory_fanout=config.ncp_sensory_fanout,
        inter_fanout=config.ncp_inter_fanout,
        recurrent_command_synapses=config.ncp_recurrent_command_synapses,
        motor_fanin=config.ncp_motor_fanin,
    )


class MicroCircuit(nn.Module):
    """A single CfC micro-circuit with sparse NCP wiring.

    Intentionally ultra-small. Strength comes from the collective
    of many circuits, not individual capacity.

    State is NOT managed internally — the CircuitBank manages
    all circuit states via a batched tensor for GPU efficiency.
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        wiring = build_wiring(config)
        self.cfc = CfC(
            input_size=config.input_dim,
            units=wiring,
            batch_first=True,
            return_sequences=False,
            mode="pure",
        )
        self._state_size = wiring.units

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def output_size(self) -> int:
        return self.cfc.output_size

    def forward(
        self, x: torch.Tensor, hx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim).
            hx: Hidden state (batch, state_size). None = zeros.

        Returns:
            (output, new_hidden_state)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        output, hx_new = self.cfc(x, hx=hx)
        return output, hx_new
