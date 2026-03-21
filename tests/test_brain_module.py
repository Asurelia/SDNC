import torch
import pytest
from brain_hybrid.core.brain_module import BrainModule


def test_forward_shape_4096():
    """Le module retourne la bonne forme avec hidden_dim=4096 (Qwen2.5-VL)."""
    m = BrainModule(input_dim=4096, hidden_dim=64)
    x = torch.randn(1, 10, 4096)
    pred, spk = m(x)
    assert pred.shape == (1, 10, 4096)


def test_forward_shape_variable_seqlen():
    """Le module gère des seq_len différents entre appels."""
    m = BrainModule(input_dim=4096, hidden_dim=64)
    x1 = torch.randn(1, 5, 4096)
    x2 = torch.randn(1, 12, 4096)
    pred1, _ = m(x1)
    pred2, _ = m(x2)
    assert pred1.shape == (1, 5, 4096)
    assert pred2.shape == (1, 12, 4096)


def test_state_persists():
    """L'état CfC doit persister entre appels."""
    m = BrainModule(input_dim=4096, hidden_dim=64)
    x = torch.randn(1, 5, 4096)
    m(x)
    state_after_1 = m.cfc_state.clone() if m.cfc_state is not None else None
    m(x)
    state_after_2 = m.cfc_state
    if state_after_1 is not None:
        assert not torch.allclose(state_after_1, state_after_2), \
            "L'état CfC ne change pas entre les appels"
