"""Tests for MicroCircuit."""

import torch
import pytest
from sdnc.config import SDNCConfig
from sdnc.core.micro_circuit import MicroCircuit


@pytest.fixture
def config():
    return SDNCConfig(input_dim=32, circuit_dim=8, circuit_output_dim=2)


@pytest.fixture
def circuit(config):
    return MicroCircuit(config)


def test_forward_shape(circuit, config):
    x = torch.randn(4, 1, config.input_dim)
    output, hx = circuit(x)
    assert output.shape == (4, config.circuit_output_dim)
    assert hx.shape[0] == 4
    assert hx.shape[1] == circuit.state_size


def test_forward_2d_input(circuit, config):
    x = torch.randn(4, config.input_dim)
    output, hx = circuit(x)
    assert output.shape == (4, config.circuit_output_dim)


def test_state_persistence(circuit, config):
    x = torch.randn(2, config.input_dim)
    out1, hx1 = circuit(x)
    out2, hx2 = circuit(x, hx=hx1)
    # State should differ after processing
    assert not torch.allclose(hx1, hx2, atol=1e-6)


def test_state_size(circuit, config):
    # state_size = inter + command + motor neurons from the actual NCP wiring
    expected = config.ncp_inter_neurons + config.ncp_command_neurons + config.circuit_output_dim
    assert circuit.state_size == expected


def test_output_size(circuit, config):
    assert circuit.output_size == config.circuit_output_dim
