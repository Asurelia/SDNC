"""Tests for Hebbian learning rules."""

import torch
import pytest
from sdnc.core.hebbian_update import (
    oja_update,
    apply_hebbian_to_circuit,
    synaptic_pruning,
    strengthen_co_active_circuits,
)


def test_oja_update_shape():
    weight = torch.randn(4, 8)
    pre = torch.randn(10, 8)
    post = torch.randn(10, 4)
    updated = oja_update(weight, pre, post, lr=1e-3)
    assert updated.shape == weight.shape


def test_oja_update_changes_weights():
    weight = torch.randn(4, 8)
    pre = torch.randn(10, 8)
    post = torch.randn(10, 4)
    updated = oja_update(weight, pre, post, lr=1e-3)
    assert not torch.allclose(weight, updated)


def test_oja_stability():
    """Oja's rule should not cause weight explosion over many steps."""
    weight = torch.randn(4, 8) * 0.1
    for _ in range(1000):
        pre = torch.randn(32, 8)
        post = pre @ weight.T  # Correlated
        weight = oja_update(weight, pre, post, lr=1e-3)

    # Weights should remain bounded
    assert weight.abs().max() < 100, f"Weights exploded: max={weight.abs().max()}"


def test_synaptic_pruning():
    model = torch.nn.Linear(8, 4)
    # Set some weights very small
    with torch.no_grad():
        model.weight[0, :] = 1e-6

    synaptic_pruning(model, threshold=1e-5)

    # Small weights should be zeroed
    assert (model.weight[0, :] == 0).all()
    # Other weights should survive
    assert model.weight[1:].abs().sum() > 0


def test_co_active_strengthening():
    n = 10
    co_activation = torch.zeros(n, n)
    co_activation[0, 1] = 5.0
    co_activation[1, 0] = 5.0

    conn_weights = torch.zeros(n, n)
    updated = strengthen_co_active_circuits(co_activation, conn_weights, lr=0.1)

    assert updated[0, 1] > 0
    assert updated[1, 0] > 0
    assert updated[2, 3] == 0  # Not co-active


def test_apply_hebbian_to_module():
    # Use a module where input/output dims match the Linear layer dims
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4),
    )
    inp = torch.randn(10, 8)
    out = torch.randn(10, 4)

    weights_before = model[0].weight.data.clone()
    apply_hebbian_to_circuit(model, inp, out, lr=1e-2)

    # Weights should have changed (dimensions match: 8 in, 4 out)
    assert not torch.allclose(weights_before, model[0].weight.data)
