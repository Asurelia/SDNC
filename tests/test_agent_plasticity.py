import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.plasticity import LocalCircuitLearner


def test_local_circuit_learner_respects_sparsity():
    config = AutonomousConfig(input_dim=32, n_circuits=100, max_active_ratio=0.05)
    learner = LocalCircuitLearner(config)
    vector = np.ones(32, dtype=np.float32)
    vector /= np.linalg.norm(vector)

    activation = learner.activate(vector)

    assert activation.active_count == 5
    assert activation.active_count / config.n_circuits <= 0.05


def test_learning_touches_active_circuits_only():
    config = AutonomousConfig(input_dim=16, n_circuits=30, max_active_ratio=0.05)
    learner = LocalCircuitLearner(config)
    vector = np.arange(16, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    activation = learner.activate(vector)

    before = learner.circuit_keys.copy()
    learner.learn(vector, activation, salience=1.0, feedback_score=1.0)
    changed = np.linalg.norm(learner.circuit_keys - before, axis=1) > 1e-7

    assert set(np.where(changed)[0].tolist()).issubset(set(activation.indices))
    assert changed.sum() > 0


def test_grow_circuit_expands_capacity():
    config = AutonomousConfig(input_dim=8, n_circuits=4, max_circuits=6)
    learner = LocalCircuitLearner(config)
    prototype = np.ones(8, dtype=np.float32)
    prototype /= np.linalg.norm(prototype)

    idx, mode = learner.grow_circuit(prototype)

    assert mode == "added"
    assert idx == 4
    assert config.n_circuits == 5
    assert learner.circuit_keys.shape == (5, 8)
    assert learner.connection_weights.shape == (5, 5)
