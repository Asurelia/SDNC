"""Cognitive and resource budgets for SDNC.

This module makes the anti-dense-transformer constraint explicit: SDNC should
spend more computation only when the situation needs it, and keep most capacity
cold in RAM/NVMe instead of resident in VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sdnc.agent.config import AutonomousConfig

CognitiveMode = Literal["fast", "think", "max"]


@dataclass(frozen=True)
class CognitiveBudget:
    """Runtime limits for one interaction."""

    mode: CognitiveMode
    memory_top_k: int
    max_tool_calls: int
    max_context_segments: int
    max_context_chars: int
    allow_gap_learning: bool
    allow_external_advisors: bool
    max_hot_experts: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentBudget:
    """Estimated memory footprint for one active component."""

    name: str
    kind: str
    estimated_vram_gb: float = 0.0
    estimated_ram_gb: float = 0.0
    hot: bool = True


@dataclass(frozen=True)
class ResourceSnapshot:
    """Budget view of hot/cold SDNC capacity."""

    total_vram_gb: float
    usable_vram_gb: float
    hot_vram_gb: float
    cold_ram_gb: float
    cold_disk_gb: float
    within_budget: bool
    components: tuple[ComponentBudget, ...]


class BudgetManager:
    """Choose cognitive mode and estimate hot/cold resource pressure."""

    def __init__(self, config: AutonomousConfig):
        self.config = config

    def choose(
        self,
        text: str,
        confidence: float | None = None,
        explicit_mode: str | None = None,
    ) -> CognitiveBudget:
        mode = self._normalize_mode(explicit_mode) or self._infer_mode(text, confidence)
        if mode == "fast":
            return CognitiveBudget(
                mode="fast",
                memory_top_k=max(1, min(self.config.memory_top_k, 3)),
                max_tool_calls=max(1, min(self.config.max_tool_calls, 2)),
                max_context_segments=self.config.fast_context_segments,
                max_context_chars=self.config.fast_context_chars,
                allow_gap_learning=False,
                allow_external_advisors=False,
                max_hot_experts=self.config.fast_hot_experts,
                notes=("low latency", "minimal recall"),
            )
        if mode == "think":
            return CognitiveBudget(
                mode="think",
                memory_top_k=self.config.memory_top_k,
                max_tool_calls=self.config.max_tool_calls,
                max_context_segments=self.config.think_context_segments,
                max_context_chars=self.config.think_context_chars,
                allow_gap_learning=True,
                allow_external_advisors=False,
                max_hot_experts=self.config.think_hot_experts,
                notes=("balanced recall", "local verification"),
            )
        return CognitiveBudget(
            mode="max",
            memory_top_k=max(self.config.memory_top_k, self.config.max_memory_top_k),
            max_tool_calls=max(self.config.max_tool_calls, self.config.max_tool_calls_max_mode),
            max_context_segments=self.config.max_context_segments,
            max_context_chars=self.config.max_context_chars,
            allow_gap_learning=True,
            allow_external_advisors=self.config.allow_external_advisors,
            max_hot_experts=self.config.max_hot_experts,
            notes=("deep recall", "multi-source learning allowed"),
        )

    def snapshot(self, budget: CognitiveBudget | None = None) -> ResourceSnapshot:
        budget = budget or self.choose("")
        components = [
            ComponentBudget(
                name="sdnc_sparse_core",
                kind="core",
                estimated_vram_gb=self.config.core_hot_vram_gb,
                estimated_ram_gb=self.config.core_hot_ram_gb,
            ),
            ComponentBudget(
                name="working_state",
                kind="state",
                estimated_vram_gb=self.config.state_hot_vram_gb,
                estimated_ram_gb=self.config.state_hot_ram_gb,
            ),
            ComponentBudget(
                name="active_expert_cache",
                kind="experts",
                estimated_vram_gb=budget.max_hot_experts * self.config.avg_hot_expert_vram_gb,
                estimated_ram_gb=budget.max_hot_experts * self.config.avg_hot_expert_ram_gb,
            ),
            ComponentBudget(
                name="context_lod_cache",
                kind="context",
                estimated_vram_gb=0.0,
                estimated_ram_gb=budget.max_context_segments * self.config.context_segment_ram_mb / 1024.0,
            ),
            ComponentBudget(
                name="cold_expert_library",
                kind="cold_storage",
                estimated_vram_gb=0.0,
                estimated_ram_gb=self.config.cold_library_ram_gb,
                hot=False,
            ),
        ]
        hot_vram = sum(item.estimated_vram_gb for item in components if item.hot)
        cold_ram = sum(item.estimated_ram_gb for item in components if not item.hot)
        usable = max(0.0, self.config.total_vram_gb - self.config.reserved_vram_gb)
        return ResourceSnapshot(
            total_vram_gb=self.config.total_vram_gb,
            usable_vram_gb=usable,
            hot_vram_gb=hot_vram,
            cold_ram_gb=cold_ram,
            cold_disk_gb=self.config.cold_library_disk_gb,
            within_budget=hot_vram <= usable,
            components=tuple(components),
        )

    def _infer_mode(self, text: str, confidence: float | None) -> CognitiveMode:
        lowered = text.lower()
        if any(marker in lowered for marker in ["max", "profond", "recherche complète", "benchmark", "architecture"]):
            return "max"
        if any(marker in lowered for marker in ["pourquoi", "comment", "analyse", "plan", "debug", "erreur", "test"]):
            return "think"
        if confidence is not None and confidence < self.config.confidence_threshold:
            return "think"
        return self.config.default_cognitive_mode

    def _normalize_mode(self, mode: str | None) -> CognitiveMode | None:
        if not mode:
            return None
        lowered = mode.lower().strip()
        if lowered in {"fast", "think", "max"}:
            return lowered  # type: ignore[return-value]
        return None
