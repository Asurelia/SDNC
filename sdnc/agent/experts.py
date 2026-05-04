"""Lifecycle manager for SDNC self-created experts.

Experts are not static model parts. They are assets with a lifecycle:
candidate -> probation -> active -> hot/cold residency -> retired/recycled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from sdnc.agent.budget import CognitiveBudget
from sdnc.agent.config import AutonomousConfig
from sdnc.agent.memory import ExpertRecord, PersistentMemory


@dataclass(frozen=True)
class ExpertLifecycleReport:
    """Summary of one expert maintenance pass."""

    selected_hot: list[ExpertRecord]
    cooled: list[str]
    retired: list[str]
    total_hot_vram_gb: float
    total_hot_ram_gb: float


class ExpertManager:
    """Create, select, promote, cool, and retire experts under budget."""

    def __init__(self, config: AutonomousConfig, memory: PersistentMemory):
        self.config = config
        self.memory = memory

    def create_from_learning(
        self,
        name: str,
        kind: str,
        description: str,
        trigger_embedding: np.ndarray,
        payload: dict[str, Any] | None = None,
        success: bool = True,
    ) -> str:
        status = "probation" if success else "candidate"
        return self.memory.upsert_expert(
            name=self._safe_name(name),
            kind=kind,
            description=description[:500],
            trigger_embedding=trigger_embedding,
            status=status,
            success=success,
            payload=payload or {},
            estimated_vram_gb=self.config.avg_hot_expert_vram_gb,
            estimated_ram_gb=self.config.avg_hot_expert_ram_gb,
            hot=False,
        )

    def select_for_interaction(
        self,
        embedding: np.ndarray,
        budget: CognitiveBudget,
    ) -> ExpertLifecycleReport:
        candidates = self.memory.retrieve_experts(
            embedding,
            top_k=max(budget.max_hot_experts * 2, budget.max_hot_experts),
        )
        usable_vram = max(0.0, self.config.total_vram_gb - self.config.reserved_vram_gb)
        selected: list[ExpertRecord] = []
        hot_vram = 0.0
        hot_ram = 0.0
        for expert in candidates:
            if expert.status == "retired":
                continue
            if len(selected) >= budget.max_hot_experts:
                break
            projected = hot_vram + expert.estimated_vram_gb
            if projected > usable_vram:
                continue
            selected.append(expert)
            hot_vram = projected
            hot_ram += expert.estimated_ram_gb

        selected_ids = {expert.id for expert in selected}
        cooled: list[str] = []
        retired: list[str] = []
        for expert in self.memory.list_experts():
            if expert.id in selected_ids:
                new_status = "active" if expert.utility >= self.config.expert_active_utility else expert.status
                self.memory.set_expert_residency(expert.id, hot=True, status=new_status)
                continue
            if expert.hot:
                self.memory.set_expert_residency(expert.id, hot=False, status="cold")
                cooled.append(expert.name)
            if self._should_retire(expert):
                self.memory.set_expert_residency(expert.id, hot=False, status="retired")
                retired.append(expert.name)

        refreshed = [expert for expert in self.memory.list_experts() if expert.id in selected_ids]
        return ExpertLifecycleReport(
            selected_hot=refreshed,
            cooled=cooled,
            retired=retired,
            total_hot_vram_gb=round(hot_vram, 4),
            total_hot_ram_gb=round(hot_ram, 4),
        )

    def record_outcome(self, experts: list[ExpertRecord], success: bool) -> None:
        for expert in experts:
            self.memory.record_expert_result(expert.id, success)

    def summary(self) -> dict[str, Any]:
        experts = self.memory.list_experts()
        by_status: dict[str, int] = {}
        hot = 0
        hot_vram = 0.0
        for expert in experts:
            by_status[expert.status] = by_status.get(expert.status, 0) + 1
            if expert.hot:
                hot += 1
                hot_vram += expert.estimated_vram_gb
        return {
            "total": len(experts),
            "hot": hot,
            "hot_vram_gb": round(hot_vram, 4),
            "by_status": by_status,
            "top": [
                {
                    "name": expert.name,
                    "kind": expert.kind,
                    "status": expert.status,
                    "utility": expert.utility,
                    "success_rate": expert.success_rate,
                    "hot": expert.hot,
                }
                for expert in experts[:8]
            ],
        }

    def _should_retire(self, expert: ExpertRecord) -> bool:
        total = expert.success_count + expert.failure_count
        return (
            total >= self.config.expert_retire_after_trials
            and expert.failure_count >= self.config.expert_retire_failures
            and expert.utility < self.config.expert_min_utility
        )

    def _safe_name(self, name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", name.strip().lower())
        return cleaned[:160] or "expert:unnamed"
