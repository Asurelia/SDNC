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
from sdnc.agent.expert_atlas import (
    DecodedExpert,
    build_procedure_payload,
    decode_payload,
    payload_from_dict,
)
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
        prepared_payload = dict(payload or {})
        expert_payload = prepared_payload.get("expert_payload")
        if expert_payload is None:
            source_payload = {key: value for key, value in prepared_payload.items() if key != "expert_payload"}
            if kind == "procedure":
                expert_payload = build_procedure_payload(
                    name=name,
                    description=description,
                    tool_name=str(source_payload.get("tool_name", "")),
                    trigger_embedding=trigger_embedding,
                    payload=source_payload,
                    decode_level="L1",
                )
            else:
                expert_payload = build_procedure_payload(
                    name=name,
                    description=description,
                    trigger_embedding=trigger_embedding,
                    payload=source_payload,
                    decode_level="L0",
                )
            prepared_payload["expert_payload"] = expert_payload.to_dict()
        else:
            expert_payload = payload_from_dict(expert_payload)
            prepared_payload["expert_payload"] = expert_payload.to_dict()

        return self.memory.upsert_expert(
            name=self._safe_name(name),
            kind=kind,
            description=description[:500],
            trigger_embedding=trigger_embedding,
            status=status,
            success=success,
            payload=prepared_payload,
            estimated_vram_gb=expert_payload.hot_vram_gb,
            estimated_ram_gb=expert_payload.hot_ram_gb,
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
            if self._has_invalid_payload(expert):
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

    def decode_expert(self, expert: ExpertRecord, level: str = "L2") -> DecodedExpert | None:
        raw_payload = expert.payload.get("expert_payload")
        if not isinstance(raw_payload, dict):
            return None
        return decode_payload(raw_payload, level=level)

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

    def _has_invalid_payload(self, expert: ExpertRecord) -> bool:
        raw_payload = expert.payload.get("expert_payload")
        if raw_payload is None:
            return False
        try:
            decoded = decode_payload(raw_payload, level="L0")
        except (KeyError, TypeError, ValueError):
            return True
        return not decoded.integrity_ok

    def _safe_name(self, name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", name.strip().lower())
        return cleaned[:160] or "expert:unnamed"
