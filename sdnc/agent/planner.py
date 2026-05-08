"""Active-inference-style action planning for SDNC.

The planner is a sparse coordinator helper, not a controller that knows the
answer. It ranks a few bounded action hypotheses and lets the existing runtime
execute the selected tools/memory path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from sdnc.agent.cognitive_core import CognitiveWorkspace
from sdnc.agent.config import AutonomousConfig


@dataclass(frozen=True)
class ActionCandidate:
    """One possible next action scored by practical expected free energy."""

    action: str
    tool_names: tuple[str, ...]
    uncertainty_reduction: float
    task_utility: float
    user_relevance: float
    cost: float
    risk: float
    expected_surprise: float
    reason: str

    @property
    def score(self) -> float:
        return float(
            self.uncertainty_reduction
            + self.task_utility
            + self.user_relevance
            - self.cost
            - self.risk
        )

    @property
    def expected_free_energy(self) -> float:
        return float(
            self.expected_surprise
            + self.cost
            + self.risk
            - self.uncertainty_reduction
            - self.task_utility
            - self.user_relevance
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "tool_names": list(self.tool_names),
            "uncertainty_reduction": round(self.uncertainty_reduction, 4),
            "task_utility": round(self.task_utility, 4),
            "user_relevance": round(self.user_relevance, 4),
            "cost": round(self.cost, 4),
            "risk": round(self.risk, 4),
            "expected_surprise": round(self.expected_surprise, 4),
            "score": round(self.score, 4),
            "expected_free_energy": round(self.expected_free_energy, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ActionPlan:
    """Ranked action plan for one cognitive workspace."""

    selected: ActionCandidate
    candidates: tuple[ActionCandidate, ...]
    policy: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "selected_action": self.selected.action,
            "selected_tools": list(self.selected.tool_names),
            "selected_score": round(self.selected.score, 4),
            "selected_expected_free_energy": round(self.selected.expected_free_energy, 4),
            "reason": self.selected.reason,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
        }


class ActionPlanner:
    """Rank answer/tool/feedback/exploration candidates with bounded heuristics."""

    def __init__(self, config: AutonomousConfig):
        self.config = config

    def plan(
        self,
        workspace: CognitiveWorkspace,
        *,
        available_tools: list[str],
        memory_count: int,
        hot_expert_count: int,
    ) -> ActionPlan:
        uncertainty = float(workspace.uncertainty)
        memory_support = float(workspace.prediction.get("memory_support", 0.0))
        tool_success = float(workspace.prediction.get("predicted_tool_success", 0.0))
        tool_names = tuple(name for name in available_tools if name != "memory_recall")
        candidates = [
            self._answer_candidate(workspace, memory_support),
            self._memory_candidate(workspace, memory_count, memory_support),
            self._feedback_candidate(workspace),
        ]
        if tool_names:
            candidates.append(self._tool_candidate(workspace, tool_names, tool_success))
        if uncertainty >= 0.48:
            candidates.append(self._explore_candidate(workspace, hot_expert_count))

        ranked = tuple(
            sorted(
                candidates,
                key=lambda candidate: (candidate.score, -candidate.expected_free_energy),
                reverse=True,
            )
        )
        selected = ranked[0]
        if selected.score < 0.10:
            selected = self._feedback_candidate(workspace)
            ranked = tuple(sorted((selected, *ranked), key=lambda candidate: candidate.score, reverse=True))
        return ActionPlan(selected=selected, candidates=ranked, policy="practical_expected_free_energy")

    def _answer_candidate(self, workspace: CognitiveWorkspace, memory_support: float) -> ActionCandidate:
        uncertainty = float(workspace.uncertainty)
        reduction = _clip01(0.08 + 0.28 * memory_support + 0.18 * (1.0 - uncertainty))
        utility = _clip01(0.60 - 0.45 * uncertainty + 0.15 * memory_support)
        relevance = _clip01(0.45 + 0.25 * memory_support)
        cost = 0.02
        risk = _clip01(0.22 * uncertainty)
        return ActionCandidate(
            action="answer",
            tool_names=(),
            uncertainty_reduction=reduction,
            task_utility=utility,
            user_relevance=relevance,
            cost=cost,
            risk=risk,
            expected_surprise=_clip01(uncertainty - 0.35 * reduction),
            reason="low-cost response from current workspace and memory support",
        )

    def _memory_candidate(
        self,
        workspace: CognitiveWorkspace,
        memory_count: int,
        memory_support: float,
    ) -> ActionCandidate:
        uncertainty = float(workspace.uncertainty)
        has_memory = 1.0 if memory_count > 0 else 0.0
        reduction = _clip01(0.18 + 0.35 * memory_support + 0.12 * has_memory)
        utility = _clip01(0.25 + 0.25 * has_memory + 0.20 * memory_support)
        relevance = _clip01(0.45 + 0.25 * memory_support)
        cost = 0.05
        risk = _clip01(0.08 * (1.0 - memory_support))
        return ActionCandidate(
            action="recall_memory",
            tool_names=("memory_recall",),
            uncertainty_reduction=reduction,
            task_utility=utility,
            user_relevance=relevance,
            cost=cost,
            risk=risk,
            expected_surprise=_clip01(uncertainty - 0.45 * reduction),
            reason="reuse prior episodes before spending on external actions",
        )

    def _tool_candidate(
        self,
        workspace: CognitiveWorkspace,
        tool_names: tuple[str, ...],
        tool_success: float,
    ) -> ActionCandidate:
        uncertainty = float(workspace.uncertainty)
        tool_bonus = max((_tool_relevance(name, workspace.input_summary) for name in tool_names), default=0.0)
        web_penalty = 0.12 if "web_search" in tool_names else 0.0
        cost = _clip01(0.08 + 0.04 * len(tool_names) + web_penalty)
        risk = _clip01(0.05 + 0.12 * ("web_search" in tool_names) + 0.03 * ("file_read" in tool_names))
        reduction = _clip01(0.22 + 0.45 * tool_success + 0.20 * tool_bonus)
        utility = _clip01(0.35 + 0.35 * tool_bonus + 0.15 * min(1.0, len(tool_names) / 3.0))
        relevance = _clip01(0.42 + 0.38 * tool_bonus)
        return ActionCandidate(
            action="use_tools",
            tool_names=tool_names,
            uncertainty_reduction=reduction,
            task_utility=utility,
            user_relevance=relevance,
            cost=cost,
            risk=risk,
            expected_surprise=_clip01(uncertainty - 0.50 * reduction),
            reason="selected tools are expected to reduce uncertainty enough for their cost",
        )

    def _feedback_candidate(self, workspace: CognitiveWorkspace) -> ActionCandidate:
        uncertainty = float(workspace.uncertainty)
        high = 1.0 if uncertainty >= 0.58 else 0.0
        reduction = _clip01(0.12 + 0.38 * high + 0.20 * uncertainty)
        utility = _clip01(0.10 + 0.50 * high)
        relevance = _clip01(0.32 + 0.30 * high)
        return ActionCandidate(
            action="ask_feedback",
            tool_names=(),
            uncertainty_reduction=reduction,
            task_utility=utility,
            user_relevance=relevance,
            cost=0.18,
            risk=0.04,
            expected_surprise=_clip01(uncertainty * 0.55),
            reason="uncertainty is high enough that user feedback may be cheaper than guessing",
        )

    def _explore_candidate(self, workspace: CognitiveWorkspace, hot_expert_count: int) -> ActionCandidate:
        uncertainty = float(workspace.uncertainty)
        expert_support = min(1.0, hot_expert_count / max(1, self.config.max_hot_experts))
        reduction = _clip01(0.22 + 0.35 * uncertainty + 0.10 * expert_support)
        utility = _clip01(0.28 + 0.30 * uncertainty)
        relevance = _clip01(0.36 + 0.20 * _question_signal(workspace.input_summary))
        cost = 0.26
        risk = 0.12
        return ActionCandidate(
            action="investigate_gap",
            tool_names=(),
            uncertainty_reduction=reduction,
            task_utility=utility,
            user_relevance=relevance,
            cost=cost,
            risk=risk,
            expected_surprise=_clip01(uncertainty - 0.40 * reduction),
            reason="record a lacune or sandbox hypothesis instead of trusting one weak answer",
        )


def _tool_relevance(tool_name: str, text: str) -> float:
    lowered = text.lower()
    if tool_name == "calculator":
        return 1.0 if _math_signal(text) else 0.25
    if tool_name == "file_search":
        return 0.82 if any(marker in lowered for marker in ["fichier", "code", "repo", "projet", ".md", ".py"]) else 0.30
    if tool_name == "file_read":
        return 0.85 if re.search(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+", text) else 0.25
    if tool_name == "web_search":
        return 0.75 if any(marker in lowered for marker in ["source", "internet", "web", "latest", "actuel"]) else 0.35
    return 0.40


def _question_signal(text: str) -> float:
    lowered = text.lower()
    return 1.0 if "?" in text or any(word in lowered for word in ["pourquoi", "comment", "cherche"]) else 0.0


def _math_signal(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\d+\s*[-+*/%]\s*\d+", text):
        return True
    has_two_numbers = len(re.findall(r"\d+(?:[,.]\d+)?", text)) >= 2
    math_markers = [
        "calcule",
        "résous",
        "resous",
        "combien",
        "reste",
        "donne",
        "mange",
        "perd",
        "retire",
        "ajoute",
        "gagne",
        "fois",
        "divise",
        "somme",
        "soustra",
    ]
    return has_two_numbers and any(marker in lowered for marker in math_markers)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
