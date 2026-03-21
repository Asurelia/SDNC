import torch
import pytest
from brain_hybrid.core.brain_module import BrainModule


def test_forward_shape_3584():
    """Le module retourne la bonne forme avec hidden_dim=3584 (Qwen2.5-VL)."""
    m = BrainModule(input_dim=3584, hidden_dim=64)
    x = torch.randn(1, 10, 3584)
    pred, spk = m(x)
    assert pred.shape == (1, 10, 3584)


def test_forward_shape_variable_seqlen():
    """Le module gère des seq_len différents entre appels."""
    m = BrainModule(input_dim=3584, hidden_dim=64)
    x1 = torch.randn(1, 5, 3584)
    x2 = torch.randn(1, 12, 3584)
    pred1, _ = m(x1)
    pred2, _ = m(x2)
    assert pred1.shape == (1, 5, 3584)
    assert pred2.shape == (1, 12, 3584)


def test_state_persists():
    """L'état CfC doit persister entre appels."""
    m = BrainModule(input_dim=3584, hidden_dim=64)
    x = torch.randn(1, 5, 3584)
    m(x)
    state_after_1 = m.cfc_state.clone() if m.cfc_state is not None else None
    m(x)
    state_after_2 = m.cfc_state
    if state_after_1 is not None:
        assert not torch.allclose(state_after_1, state_after_2), \
            "L'état CfC ne change pas entre les appels"
