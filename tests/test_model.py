"""Tests for the circuit-centric SDNC model."""

import torch
import pytest
from sdnc.config import SDNCConfig
from sdnc.model import SDNCModel


class MockEncoder(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.proj = torch.nn.Linear(3 * 32 * 32, output_dim)
        self.preprocess = None

    def forward(self, images):
        flat = images.reshape(images.shape[0], -1)
        if flat.shape[1] != self.proj.in_features:
            flat = flat[:, :self.proj.in_features] if flat.shape[1] > self.proj.in_features else \
                torch.cat([flat, torch.zeros(flat.shape[0], self.proj.in_features - flat.shape[1], device=flat.device)], dim=1)
        out = self.proj(flat)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


@pytest.fixture
def config():
    return SDNCConfig(
        n_circuits=20, input_dim=32, circuit_dim=8, circuit_output_dim=2,
        circuit_repr_dim=64, sparsity_k=1, n_way=2, k_shot=1,
        query_per_class=5, use_expert_choice=False,
    )


@pytest.fixture
def model(config):
    m = SDNCModel(config)
    mock = MockEncoder(config.input_dim)
    m.vision_encoder = mock
    m.encoder = mock
    return m


def test_forward_circuit_repr(model, config):
    images = torch.randn(4, 3, 32, 32)
    result = model(images=images)
    assert result["circuit_repr"].shape == (4, config.circuit_repr_dim)
    assert result["encoder_features"].shape == (4, config.input_dim)
    # circuit_repr is normalized
    norms = result["circuit_repr"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_learn(model):
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    result = model.learn(images=images, labels=labels)
    assert "circuit_repr" in result


def test_recognize_in_circuit_space(model):
    support = torch.randn(2, 3, 32, 32)
    queries = torch.randn(10, 3, 32, 32)
    support_labels = torch.tensor([0, 1])
    scores = model.recognize(
        support_labels=support_labels,
        support_kwargs={"images": support},
        query_kwargs={"images": queries},
    )
    assert scores.shape == (10, 2)


def test_reset(model):
    model(images=torch.randn(2, 3, 32, 32))
    assert model.circuit_bank.circuit_states.abs().sum() > 0
    model.reset()
    assert model.circuit_bank.circuit_states.abs().sum() == 0


def test_prune(model):
    model.learn(images=torch.randn(4, 3, 32, 32), labels=torch.tensor([0, 1, 0, 1]))
    model.prune()


def test_inter_circuit_wiring(model):
    config = model.config
    config.sparsity_k = 2
    config.inter_circuit_hebbian_lr = 0.1
    m = SDNCModel(config)
    mock = MockEncoder(config.input_dim)
    m.vision_encoder = mock
    m.encoder = mock
    m.learn(images=torch.randn(4, 3, 32, 32), labels=torch.tensor([0, 1, 0, 1]))
    stats = m.get_circuit_wiring_stats()
    assert stats["nonzero_connections"] > 0
