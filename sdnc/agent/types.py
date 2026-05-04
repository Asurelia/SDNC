"""Shared dataclasses for the autonomous SDNC loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CircuitActivation:
    """Sparse activation state for one observation."""

    indices: list[int]
    weights: list[float]
    scores: list[float]
    confidence: float
    novelty: float

    @property
    def active_count(self) -> int:
        return len(self.indices)


@dataclass
class MemoryRecord:
    """A retrieved or stored episodic memory."""

    id: str
    timestamp: float
    text: str
    context: dict[str, Any]
    embedding: np.ndarray
    active_circuits: list[int]
    salience: float
    outcome: str | None = None
    feedback_score: float | None = None
    similarity: float = 0.0


@dataclass(frozen=True)
class ToolResult:
    """Result returned by a tool."""

    tool_name: str
    success: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Feedback:
    """External feedback for one interaction."""

    score: float
    text: str = ""

    def clipped_score(self) -> float:
        return max(-1.0, min(1.0, float(self.score)))


@dataclass
class InteractionResult:
    """Full result of one observe-act-learn cycle."""

    episode_id: str
    timestamp: float
    input_text: str
    response: str
    activation: CircuitActivation
    memories: list[MemoryRecord]
    tool_results: list[ToolResult]
    learned: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, text: str, response: str) -> "InteractionResult":
        return cls(
            episode_id="",
            timestamp=time(),
            input_text=text,
            response=response,
            activation=CircuitActivation([], [], [], 0.0, 1.0),
            memories=[],
            tool_results=[],
            learned=False,
        )
