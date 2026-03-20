"""Working memory: wraps the circuit bank's hidden states.

Working memory IS the current state of active LNN circuits.
It decays over time if not refreshed — like biological short-term memory.
"""

import torch


class WorkingMemory:
    """Interface to circuit bank hidden states as working memory.

    Not an nn.Module — it operates on the CircuitBank's state buffer directly.
    """

    def __init__(self, circuit_states: torch.Tensor, decay_rate: float = 0.99):
        self._states = circuit_states
        self.decay_rate = decay_rate

    @property
    def states(self) -> torch.Tensor:
        return self._states

    def get(self, indices: torch.Tensor) -> torch.Tensor:
        """Read working memory for specific circuits."""
        return self._states[indices]

    def update(self, indices: torch.Tensor, new_states: torch.Tensor):
        """Write new states to working memory."""
        self._states[indices] = new_states

    def decay(self):
        """Apply exponential decay — inactive memories fade."""
        self._states *= self.decay_rate

    def clear(self):
        """Clear all working memory."""
        self._states.zero_()

    def active_summary(self, indices: torch.Tensor) -> torch.Tensor:
        """Get mean state of active circuits as a summary vector."""
        return self._states[indices].mean(dim=0)
