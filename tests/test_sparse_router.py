"""Tests for SparseRouter."""

import torch
import pytest
from sdnc.core.sparse_router import SparseRouter


@pytest.fixture
def router():
    return SparseRouter(input_dim=32, n_circuits=100, k=5, noise_std=0.1)


def test_output_shape(router):
    x = torch.randn(4, 32)
    indices, weights = router(x)
    assert indices.shape == (4, 5)
    assert weights.shape == (4, 5)


def test_sparsity_guarantee(router):
    x = torch.randn(8, 32)
    indices, weights = router(x)
    # Max 5 circuits active per sample
    assert indices.shape[1] == 5
    # All indices valid
    assert (indices >= 0).all()
    assert (indices < 100).all()


def test_weights_sum_to_one(router):
    x = torch.randn(4, 32)
    _, weights = router(x)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_top1_router():
    router = SparseRouter(input_dim=32, n_circuits=20, k=1)
    x = torch.randn(4, 32)
    indices, weights = router(x)
    assert indices.shape == (4, 1)
    assert router.sparsity_ratio == 1 / 20


def test_bias_update():
    router = SparseRouter(input_dim=32, n_circuits=10, k=1, noise_std=0.0)
    router.train()

    # Simulate biased activations
    x = torch.randn(100, 32)
    router(x)

    initial_bias = router.expert_bias.clone()
    router.update_bias(gamma=0.01)

    # Bias should have changed
    assert not torch.allclose(initial_bias, router.expert_bias)


def test_no_noise_eval():
    router = SparseRouter(input_dim=32, n_circuits=10, k=1, noise_std=1.0)
    router.eval()

    x = torch.randn(4, 32)
    # Should be deterministic in eval mode
    idx1, _ = router(x)
    idx2, _ = router(x)
    assert torch.equal(idx1, idx2)
