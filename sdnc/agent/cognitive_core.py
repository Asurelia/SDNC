"""Sparse cognitive workspace for SDNC.

The core is deliberately not an all-knowing model. It is a bounded coordinator:
perception, circuits, memory, experts, tools, and feedback publish compact
signals; the workspace keeps only the few most salient signals needed to choose
what happens next.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, replace
from time import time
from typing import Any

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.types import CircuitActivation, MemoryRecord, ToolResult


@dataclass(frozen=True)
class WorkspaceSlot:
    """One bounded item in the current cognitive workspace."""

    kind: str
    key: str
    salience: float
    confidence: float
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "salience": round(float(self.salience), 4),
            "confidence": round(float(self.confidence), 4),
            "summary": self.summary[:240],
            "payload": self.payload,
        }


@dataclass(frozen=True)
class CognitiveWorkspace:
    """A bounded belief packet for one SDNC cycle."""

    id: str
    timestamp: float
    mode: str
    input_summary: str
    slots: tuple[WorkspaceSlot, ...]
    attention_focus: tuple[str, ...]
    prediction: dict[str, Any]
    uncertainty: float
    surprise: float = 0.0
    observation: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "input_summary": self.input_summary[:500],
            "slots": [slot.to_payload() for slot in self.slots],
            "slot_count": len(self.slots),
            "attention_focus": list(self.attention_focus),
            "prediction": self.prediction,
            "uncertainty": round(float(self.uncertainty), 4),
            "surprise": round(float(self.surprise), 4),
            "observation": self.observation,
        }


class CognitiveCore:
    """Sparse global workspace and prediction trace builder."""

    def __init__(self, config: AutonomousConfig):
        self.config = config

    def start(
        self,
        *,
        input_text: str,
        mode: str,
        activation: CircuitActivation,
        memories: list[MemoryRecord],
        tool_names: list[str],
        experts: list[Any],
        context_summary: str = "",
    ) -> CognitiveWorkspace:
        """Create a bounded workspace before tools/actions are observed."""
        slots: list[WorkspaceSlot] = [
            WorkspaceSlot(
                kind="input",
                key="current",
                salience=0.90,
                confidence=activation.confidence,
                summary=_compact(input_text),
                payload={"context_summary": _compact(context_summary, 320)},
            )
        ]
        slots.extend(self._circuit_slots(activation))
        slots.extend(self._memory_slots(memories))
        slots.extend(self._expert_slots(experts))
        slots.extend(self._tool_slots(tool_names, input_text, activation.confidence))

        best_memory = max((memory.similarity for memory in memories), default=0.0)
        uncertainty = _clip01(
            0.55 * (1.0 - activation.confidence)
            + 0.30 * activation.novelty
            + 0.15 * (1.0 - max(0.0, best_memory))
        )
        expected_tool_success = self._expected_tool_success(tool_names, input_text, activation.confidence)
        predicted_action = self._predicted_action(uncertainty, tool_names)
        prediction = {
            "predicted_confidence": round(float(activation.confidence), 4),
            "predicted_action": predicted_action,
            "predicted_useful_tools": list(tool_names),
            "predicted_tool_success": round(float(expected_tool_success), 4),
            "expected_surprise": round(float(uncertainty), 4),
            "memory_support": round(float(best_memory), 4),
        }

        bounded = self._bound_slots(slots)
        return CognitiveWorkspace(
            id=str(uuid.uuid4()),
            timestamp=time(),
            mode=mode,
            input_summary=_compact(input_text, 500),
            slots=tuple(bounded),
            attention_focus=tuple(
                slot.summary
                for slot in bounded[: min(self.config.cognitive_attention_focus, len(bounded))]
            ),
            prediction=prediction,
            uncertainty=uncertainty,
        )

    def complete(
        self,
        workspace: CognitiveWorkspace,
        *,
        tool_results: list[ToolResult],
        salience: float,
        feedback_score: float | None = None,
    ) -> CognitiveWorkspace:
        """Close the prediction loop after tools/feedback/salience are known."""
        observed_success = (
            sum(1 for result in tool_results if result.success) / len(tool_results)
            if tool_results
            else 0.0
        )
        expected_success = float(workspace.prediction.get("predicted_tool_success", 0.0))
        tool_error = abs(expected_success - observed_success) if tool_results else 0.0
        negative_feedback = max(0.0, -(feedback_score or 0.0))
        positive_feedback = max(0.0, feedback_score or 0.0)
        surprise = _clip01(
            0.35 * float(workspace.prediction.get("expected_surprise", workspace.uncertainty))
            + 0.25 * tool_error
            + 0.20 * negative_feedback
            + 0.15 * salience
            - 0.10 * positive_feedback
        )
        observation = {
            "tool_count": len(tool_results),
            "successful_tools": [result.tool_name for result in tool_results if result.success],
            "failed_tools": [result.tool_name for result in tool_results if not result.success],
            "observed_tool_success": round(float(observed_success), 4),
            "tool_prediction_error": round(float(tool_error), 4),
            "salience": round(float(salience), 4),
            "feedback_score": feedback_score,
        }
        return replace(workspace, surprise=surprise, observation=observation)

    def _bound_slots(self, slots: list[WorkspaceSlot]) -> list[WorkspaceSlot]:
        limit = max(1, int(self.config.cognitive_workspace_slots))
        slots.sort(key=lambda slot: (slot.salience, slot.confidence), reverse=True)
        return slots[:limit]

    def _circuit_slots(self, activation: CircuitActivation) -> list[WorkspaceSlot]:
        slots: list[WorkspaceSlot] = []
        for index, weight, score in zip(activation.indices, activation.weights, activation.scores):
            slots.append(
                WorkspaceSlot(
                    kind="circuit",
                    key=str(index),
                    salience=_clip01(0.55 * abs(float(weight)) + 0.45 * max(0.0, float(score))),
                    confidence=activation.confidence,
                    summary=f"circuit {index} score={float(score):.3f}",
                    payload={"weight": float(weight), "score": float(score)},
                )
            )
        return slots

    def _memory_slots(self, memories: list[MemoryRecord]) -> list[WorkspaceSlot]:
        slots: list[WorkspaceSlot] = []
        for memory in memories:
            slots.append(
                WorkspaceSlot(
                    kind="memory",
                    key=memory.id,
                    salience=_clip01(0.65 * max(0.0, memory.similarity) + 0.35 * memory.salience),
                    confidence=max(0.0, memory.similarity),
                    summary=f"memory sim={memory.similarity:.3f}: {_compact(memory.text, 180)}",
                    payload={"salience": memory.salience, "feedback_score": memory.feedback_score},
                )
            )
        return slots

    def _expert_slots(self, experts: list[Any]) -> list[WorkspaceSlot]:
        slots: list[WorkspaceSlot] = []
        for expert in experts:
            utility = float(getattr(expert, "utility", 0.0))
            similarity = float(getattr(expert, "similarity", 0.0))
            name = str(getattr(expert, "name", getattr(expert, "id", "expert")))
            slots.append(
                WorkspaceSlot(
                    kind="expert",
                    key=str(getattr(expert, "id", name)),
                    salience=_clip01(0.55 * utility + 0.45 * max(0.0, similarity)),
                    confidence=max(utility, similarity),
                    summary=f"expert {name} utility={utility:.3f}",
                    payload={
                        "kind": getattr(expert, "kind", ""),
                        "status": getattr(expert, "status", ""),
                        "hot": bool(getattr(expert, "hot", False)),
                    },
                )
            )
        return slots

    def _tool_slots(self, tool_names: list[str], input_text: str, confidence: float) -> list[WorkspaceSlot]:
        return [
            WorkspaceSlot(
                kind="tool",
                key=name,
                salience=self._tool_prior(name, input_text, confidence),
                confidence=self._tool_prior(name, input_text, confidence),
                summary=f"tool proposal: {name}",
                payload={"reason": self._tool_reason(name, input_text, confidence)},
            )
            for name in tool_names
        ]

    def _expected_tool_success(self, tool_names: list[str], input_text: str, confidence: float) -> float:
        if not tool_names:
            return 0.0
        priors = [self._tool_prior(name, input_text, confidence) for name in tool_names]
        return float(sum(priors) / max(len(priors), 1))

    def _tool_prior(self, name: str, input_text: str, confidence: float) -> float:
        lowered = input_text.lower()
        if name == "memory_recall":
            return 0.78
        if name == "calculator":
            return 0.92 if re.search(r"\d+\s*[-+*/%]\s*\d+", input_text) else 0.55
        if name == "file_search":
            return 0.74 if any(marker in lowered for marker in ["fichier", "code", "repo", "projet"]) else 0.50
        if name == "file_read":
            return 0.72 if re.search(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+", input_text) else 0.45
        if name == "web_search":
            question_like = "?" in input_text or any(
                marker in lowered for marker in ["recherche", "internet", "source", "actuel", "latest"]
            )
            return 0.70 if question_like else max(0.35, 1.0 - confidence)
        return max(0.35, min(0.75, confidence))

    def _tool_reason(self, name: str, input_text: str, confidence: float) -> str:
        if name == "memory_recall":
            return "keep continuity with prior episodes"
        if name == "calculator":
            return "arithmetic pattern detected"
        if name.startswith("file"):
            return "workspace or file-like request"
        if name == "web_search":
            return "question/current/source signal or low confidence"
        return f"learned procedure or registry candidate at confidence {confidence:.3f}"

    def _predicted_action(self, uncertainty: float, tool_names: list[str]) -> str:
        non_memory_tools = [name for name in tool_names if name != "memory_recall"]
        if non_memory_tools:
            return "use_tools"
        if uncertainty >= 0.68:
            return "ask_feedback"
        if uncertainty >= 0.48:
            return "answer_with_low_confidence"
        return "answer"


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _compact(text: str, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit]
