"""Local sparse circuit learning for interaction-driven SDNC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.types import CircuitActivation


@dataclass
class PlasticitySnapshot:
    """Serializable state of the local learner."""

    circuit_keys: np.ndarray
    circuit_state: np.ndarray
    usage_counts: np.ndarray
    connection_weights: np.ndarray


class LocalCircuitLearner:
    """Sparse local learner using Oja-style prototype updates.

    Each circuit owns a key vector. Observation vectors activate only the
    top-k matching circuits. Learning updates those active keys and their
    pairwise connections; inactive circuits are untouched.
    """

    def __init__(self, config: AutonomousConfig, seed: int = 13):
        self.config = config
        rng = np.random.default_rng(seed)
        keys = rng.normal(0.0, 1.0, size=(config.n_circuits, config.input_dim)).astype(np.float32)
        self.circuit_keys = self._normalize_rows(keys)
        self.circuit_state = np.zeros(config.n_circuits, dtype=np.float32)
        self.usage_counts = np.zeros(config.n_circuits, dtype=np.float32)
        self.connection_weights = np.zeros(
            (config.n_circuits, config.n_circuits), dtype=np.float32
        )

    def activate(self, vector: np.ndarray) -> CircuitActivation:
        """Activate at most 5 percent of circuits for one observation."""
        vector = self._unit(vector)
        base_scores = self.circuit_keys @ vector
        liquid_scores = self.config.state_influence * self.circuit_state
        scores = base_scores + liquid_scores

        k = self.config.max_active_circuits
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        confidence = float(np.clip((scores[top[0]] + 1.0) * 0.5, 0.0, 1.0))
        novelty = 1.0 - confidence

        if confidence < self.config.activation_threshold:
            recruit = int(np.argmin(self.usage_counts + np.abs(self.circuit_state)))
            top[-1] = recruit
            self.circuit_keys[recruit] = vector
            scores[recruit] = 1.0
            confidence = max(confidence, 0.5)
            novelty = 1.0

        weights = self._softmax(scores[top])
        self.circuit_state *= self.config.state_decay
        self.circuit_state[top] += weights.astype(np.float32)
        self.circuit_state = np.clip(self.circuit_state, 0.0, 4.0)

        return CircuitActivation(
            indices=[int(i) for i in top.tolist()],
            weights=[float(w) for w in weights.tolist()],
            scores=[float(s) for s in scores[top].tolist()],
            confidence=confidence,
            novelty=float(novelty),
        )

    def learn(
        self,
        vector: np.ndarray,
        activation: CircuitActivation,
        salience: float,
        feedback_score: float = 0.0,
    ) -> None:
        """Update only active circuits and their local connections."""
        if not activation.indices:
            return

        vector = self._unit(vector)
        feedback_scale = 1.0 + max(-0.5, min(1.0, feedback_score))
        lr = self.config.circuit_lr * max(0.0, salience) * feedback_scale
        if lr <= 0.0:
            return

        for idx, weight in zip(activation.indices, activation.weights):
            key = self.circuit_keys[idx]
            dot = float(np.dot(key, vector))
            direction = vector - dot * key
            if feedback_score < 0:
                direction = -direction
            self.circuit_keys[idx] = self._unit(key + lr * weight * direction)
            self.usage_counts[idx] += max(0.1, salience)

        self.connection_weights *= self.config.connection_decay
        active = np.array(activation.indices, dtype=np.int64)
        weights = np.array(activation.weights, dtype=np.float32)
        co_activation = np.outer(weights, weights) * self.config.connection_lr * max(0.0, salience)
        for i, src in enumerate(active):
            for j, dst in enumerate(active):
                if src != dst:
                    self.connection_weights[src, dst] += co_activation[i, j]

    def grow_circuit(self, prototype: np.ndarray, source: str = "consensus") -> tuple[int, str]:
        """Add or recycle one circuit from a consensus-approved prototype.

        Growth is bounded by config.max_circuits. If capacity is full, the least
        used quiet circuit is recycled instead of expanding without limit.
        """
        prototype = self._unit(prototype)
        max_circuits = int(self.config.max_circuits or self.config.n_circuits)

        if self.config.n_circuits < max_circuits:
            idx = self.config.n_circuits
            self.circuit_keys = np.vstack([self.circuit_keys, prototype[None, :]]).astype(np.float32)
            self.circuit_state = np.concatenate([self.circuit_state, np.zeros(1, dtype=np.float32)])
            self.usage_counts = np.concatenate([self.usage_counts, np.zeros(1, dtype=np.float32)])

            old = self.connection_weights
            expanded = np.zeros((idx + 1, idx + 1), dtype=np.float32)
            expanded[:idx, :idx] = old
            self.connection_weights = expanded
            self.config.n_circuits = idx + 1
            return idx, "added"

        idx = self.recyclable_circuit_index()
        self.circuit_keys[idx] = prototype
        self.circuit_state[idx] = 0.0
        self.usage_counts[idx] = 0.0
        self.connection_weights[idx, :] = 0.0
        self.connection_weights[:, idx] = 0.0
        return idx, f"recycled:{source}"

    def recyclable_circuit_index(self) -> int:
        """Pick a low-use, low-state circuit for controlled recycling."""
        score = self.usage_counts + np.abs(self.circuit_state) * 10.0
        return int(np.argmin(score))

    def prune_connections(self, threshold: float) -> int:
        """Remove weak inter-circuit links and return the count pruned."""
        if threshold <= 0:
            return 0
        mask = np.abs(self.connection_weights) < threshold
        mask &= self.connection_weights != 0
        count = int(mask.sum())
        self.connection_weights[mask] = 0.0
        return count

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            circuit_keys=self.circuit_keys,
            circuit_state=self.circuit_state,
            usage_counts=self.usage_counts,
            connection_weights=self.connection_weights,
        )

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = np.load(path)
        snapshot = PlasticitySnapshot(
            circuit_keys=data["circuit_keys"].astype(np.float32),
            circuit_state=data["circuit_state"].astype(np.float32),
            usage_counts=data["usage_counts"].astype(np.float32),
            connection_weights=data["connection_weights"].astype(np.float32),
        )
        if snapshot.circuit_keys.shape[1] != self.config.input_dim:
            return False
        if snapshot.circuit_keys.shape[0] > int(self.config.max_circuits or snapshot.circuit_keys.shape[0]):
            return False
        if snapshot.connection_weights.shape != (
            snapshot.circuit_keys.shape[0],
            snapshot.circuit_keys.shape[0],
        ):
            return False
        if snapshot.circuit_state.shape != (snapshot.circuit_keys.shape[0],):
            return False
        if snapshot.usage_counts.shape != (snapshot.circuit_keys.shape[0],):
            return False
        self.circuit_keys = self._normalize_rows(snapshot.circuit_keys)
        self.circuit_state = snapshot.circuit_state
        self.usage_counts = snapshot.usage_counts
        self.connection_weights = snapshot.connection_weights
        self.config.n_circuits = snapshot.circuit_keys.shape[0]
        return True

    def reset_state(self) -> None:
        self.circuit_state.fill(0.0)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-8)

    def _unit(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / max(float(exp.sum()), 1e-8)
