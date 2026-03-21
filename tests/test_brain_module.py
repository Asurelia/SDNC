import torch
import pytest
from brain_hybrid.core.brain_module import BrainModule


def test_forward_shape():
    """Le module retourne la bonne forme."""
    m = BrainModule(input_dim=2560, hidden_dim=64)
    x = torch.randn(1, 10, 2560)
    pred, spk = m(x)
    assert pred.shape == (1, 10, 2560)


def test_state_persists():
    """L'état CfC doit persister entre appels."""
    m = BrainModule(input_dim=2560, hidden_dim=64)
    x = torch.randn(1, 5, 2560)
    m(x)
    state_after_1 = m.cfc_state.clone() if m.cfc_state is not None else None
    m(x)
    state_after_2 = m.cfc_state
    if state_after_1 is not None:
        assert not torch.allclose(state_after_1, state_after_2), \
            "L'état CfC ne change pas entre les appels"
