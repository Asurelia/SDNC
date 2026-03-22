import torch
from brain_hybrid.core.injection import InjectionGate, HippocampalPrefixInjector


def test_gate_init_near_zero():
    """Le delta doit être très petit à l'initialisation."""
    gate = InjectionGate(hidden_dim=4096, rank=16, init_alpha=0.001)
    x = torch.randn(1, 10, 4096)
    delta = gate(x)
    assert delta.shape == (1, 10, 4096)
    assert delta.abs().mean().item() < 0.01, "Delta trop grand à l'init"


def test_gate_alpha_clamp():
    """Alpha doit rester positif."""
    gate = InjectionGate(hidden_dim=4096, rank=16, init_alpha=0.001)
    assert gate.alpha.item() > 0


def test_hippocampal_prefix_shape():
    """Le prefix doit avoir la bonne forme."""
    injector = HippocampalPrefixInjector(content_dim=4096, n_prefix_tokens=4)
    memory = torch.randn(4096)
    prefix = injector(memory)
    assert prefix is not None
    assert prefix.shape == (1, 4, 4096)


def test_hippocampal_empty_memory():
    """Pas de prefix si la mémoire est vide."""
    injector = HippocampalPrefixInjector(content_dim=4096, n_prefix_tokens=4)
    memory = torch.zeros(4096)
    prefix = injector(memory)
    assert prefix is None


def test_gate_gradient_flows():
    """Les gradients doivent passer à travers le gate."""
    gate = InjectionGate(hidden_dim=256, rank=8, init_alpha=0.01)
    x = torch.randn(1, 5, 256, requires_grad=True)
    delta = gate(x)
    loss = delta.sum()
    loss.backward()
    assert gate.alpha.grad is not None
