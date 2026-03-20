"""Tests for CircuitBank."""

import torch
import pytest
from sdnc.config import SDNCConfig
from sdnc.core.circuit_bank import CircuitBank


@pytest.fixture
def config():
    return SDNCConfig(n_circuits=20, input_dim=32, circuit_dim=8, circuit_output_dim=2)


@pytest.fixture
def bank(config):
    return CircuitBank(config)


def test_forward_shape(bank, config):
    x = torch.randn(4, config.input_dim)
    indices = torch.tensor([[0], [5], [10], [15]])
    weights = torch.ones(4, 1)
    output, hidden = bank(x, indices, weights)
    assert output.shape == (4, config.circuit_output_dim)
    assert hidden.shape == (4, bank.state_size)


def test_state_isolation(bank, config):
    x = torch.randn(1, config.input_dim)
    indices_a = torch.tensor([[0]])
    indices_b = torch.tensor([[1]])
    weights = torch.ones(1, 1)

    bank(x, indices_a, weights)
    state_0 = bank.circuit_states[0].clone()
    state_1 = bank.circuit_states[1].clone()

    # Circuit 0 was activated, circuit 1 was not
    assert state_0.abs().sum() > 0
    assert state_1.abs().sum() == 0


def test_multi_k(config):
    config.sparsity_k = 2
    bank = CircuitBank(config)
    x = torch.randn(3, config.input_dim)
    indices = torch.tensor([[0, 1], [2, 3], [4, 5]])
    weights = torch.softmax(torch.randn(3, 2), dim=-1)
    output, hidden = bank(x, indices, weights)
    assert output.shape == (3, config.circuit_output_dim)


def test_reset_states(bank, config):
    x = torch.randn(1, config.input_dim)
    indices = torch.tensor([[0]])
    weights = torch.ones(1, 1)
    bank(x, indices, weights)

    assert bank.circuit_states[0].abs().sum() > 0
    bank.reset_states()
    assert bank.circuit_states.abs().sum() == 0


def test_activation_history(bank, config):
    x = torch.randn(2, config.input_dim)
    indices = torch.tensor([[0], [0]])
    weights = torch.ones(2, 1)
    bank(x, indices, weights)
    assert bank.activation_history[0] == 2
    assert bank.activation_history[1] == 0
