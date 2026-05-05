"""Bounded replay and sleep consolidation for SDNC."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.memory import PersistentMemory, ProcedureRecord
from sdnc.agent.plasticity import LocalCircuitLearner
from sdnc.agent.types import MemoryRecord


@dataclass(frozen=True)
class ReplayAction:
    """One replay/sleep decision."""

    kind: str
    accepted: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SleepReport:
    """Inspectable outcome of one replay cycle."""

    timestamp: float
    preview: bool
    actions: list[ReplayAction] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return sum(1 for action in self.actions if action.accepted)

    @property
    def replayed_count(self) -> int:
        return sum(1 for action in self.actions if action.kind == "episode_replay" and action.accepted)

    @property
    def strengthened_count(self) -> int:
        return sum(
            1
            for action in self.actions
            if action.kind == "procedure_strengthening" and action.accepted
        )

    @property
    def rejected_count(self) -> int:
        return sum(1 for action in self.actions if not action.accepted)

    def add(self, kind: str, accepted: bool, reason: str, **details: Any) -> None:
        self.actions.append(ReplayAction(kind=kind, accepted=accepted, reason=reason, details=details))

    def summary(self) -> str:
        mode = "preview" if self.preview else "run"
        if not self.actions:
            return f"Sleep {mode}: no replay candidates."
        return (
            f"Sleep {mode}: {self.replayed_count} replayed, "
            f"{self.strengthened_count} procedures strengthened, "
            f"{self.rejected_count} rejected."
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "preview": self.preview,
            "accepted_count": self.accepted_count,
            "replayed_count": self.replayed_count,
            "strengthened_count": self.strengthened_count,
            "rejected_count": self.rejected_count,
            "summary": self.summary(),
            "actions": [
                {
                    "kind": action.kind,
                    "accepted": action.accepted,
                    "reason": action.reason,
                    "details": action.details,
                }
                for action in self.actions
            ],
        }


class SleepConsolidationCycle:
    """Replay high-salience memories with bounded local plasticity."""

    def __init__(
        self,
        config: AutonomousConfig,
        memory: PersistentMemory,
        learner: LocalCircuitLearner,
    ):
        self.config = config
        self.memory = memory
        self.learner = learner

    def run(
        self,
        preview: bool = False,
        batch_size: int | None = None,
        recent_limit: int | None = None,
    ) -> SleepReport:
        report = SleepReport(timestamp=time(), preview=preview)
        candidates = self._replay_queue(
            limit=recent_limit or self.config.replay_recent_limit,
            batch_size=batch_size or self.config.replay_batch_size,
        )
        if not candidates:
            report.add(
                "replay_queue",
                False,
                "no salient episodes available",
                min_salience=self.config.replay_min_salience,
            )
            return report

        protected = self._protected_procedures()
        report.add(
            "anti_forgetting_check",
            True,
            f"protected {len(protected)} useful procedures",
            protected_count=len(protected),
            protected=[procedure.name for procedure in protected[:8]],
        )

        replayed: list[MemoryRecord] = []
        for record in candidates:
            if record.feedback_score is not None and record.feedback_score < 0:
                if not preview:
                    self.memory.record_experiment(
                        kind="sleep_replay",
                        candidate=record.id,
                        accepted=False,
                        metrics={"salience": record.salience, "feedback_score": record.feedback_score},
                        notes="negative feedback episode held out",
                    )
                report.add(
                    "episode_replay",
                    False,
                    "negative feedback episode held out",
                    episode_id=record.id,
                    salience=record.salience,
                )
                continue
            if preview:
                report.add(
                    "episode_preview",
                    True,
                    "would replay salient episode",
                    episode_id=record.id,
                    salience=record.salience,
                    text=record.text[:120],
                )
                replayed.append(record)
                continue
            accepted, reason, details = self._replay_one(record)
            report.add("episode_replay", accepted, reason, episode_id=record.id, **details)
            if accepted:
                replayed.append(record)

        self._strengthen_repeated_procedures(replayed, candidates, report, preview=preview)
        return report

    def _replay_queue(self, limit: int, batch_size: int) -> list[MemoryRecord]:
        records = self.memory.salient_episodes(
            limit=limit,
            min_salience=self.config.replay_min_salience,
        )
        records.sort(key=self._replay_priority, reverse=True)
        return records[: max(1, int(batch_size))]

    def _replay_priority(self, record: MemoryRecord) -> float:
        feedback = 0.0 if record.feedback_score is None else float(record.feedback_score)
        tool_bonus = 0.1 if record.context.get("__sdnc", {}).get("successful_tools") else 0.0
        return record.salience + 0.15 * max(0.0, feedback) + tool_bonus

    def _replay_one(self, record: MemoryRecord) -> tuple[bool, str, dict[str, Any]]:
        before = self.learner.snapshot()
        activation = self.learner.activate(record.embedding)
        feedback = 0.0 if record.feedback_score is None else float(np.clip(record.feedback_score, -1.0, 1.0))
        effective_salience = float(np.clip(record.salience * self.config.replay_strength, 0.0, 1.0))
        self.learner.learn(record.embedding, activation, effective_salience, feedback)
        drift = self._mean_active_drift(before.circuit_keys, self.learner.circuit_keys, activation.indices)
        if drift > self.config.replay_max_drift:
            self.learner.restore(before)
            self.memory.record_experiment(
                kind="sleep_replay",
                candidate=record.id,
                accepted=False,
                metrics={"drift": drift, "effective_salience": effective_salience},
                notes="replay rejected by drift guard",
            )
            return (
                False,
                f"drift guard rejected replay: {drift:.4f}",
                {
                    "drift": round(drift, 6),
                    "active_circuits": activation.indices,
                    "effective_salience": effective_salience,
                },
            )
        self.memory.record_experiment(
            kind="sleep_replay",
            candidate=record.id,
            accepted=True,
            metrics={"drift": drift, "effective_salience": effective_salience},
            notes="bounded replay accepted",
        )
        return (
            True,
            "bounded replay accepted",
            {
                "drift": round(drift, 6),
                "active_circuits": activation.indices,
                "effective_salience": effective_salience,
            },
        )

    def _strengthen_repeated_procedures(
        self,
        replayed: list[MemoryRecord],
        reference_records: list[MemoryRecord],
        report: SleepReport,
        preview: bool,
    ) -> None:
        buckets: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
        for record in replayed:
            internal = record.context.get("__sdnc", {})
            for tool_name in internal.get("successful_tools", []):
                buckets[(str(tool_name), self._topic_key(record.text))].append(record)

        for (tool_name, topic), bucket in sorted(buckets.items()):
            if len(bucket) < self.config.procedure_min_successes:
                if not preview:
                    self.memory.record_experiment(
                        kind="sleep_procedure_consolidation",
                        candidate=f"{tool_name}:{topic}",
                        accepted=False,
                        metrics={"evidence": len(bucket)},
                        notes="not enough repeated successful replay evidence",
                    )
                report.add(
                    "procedure_candidate",
                    False,
                    "not enough repeated successful replay evidence",
                    tool_name=tool_name,
                    topic=topic,
                    evidence=len(bucket),
                )
                continue
            failures = self._matching_failures(reference_records, tool_name, topic)
            if failures > len(bucket):
                if not preview:
                    self.memory.record_experiment(
                        kind="sleep_procedure_consolidation",
                        candidate=f"{tool_name}:{topic}",
                        accepted=False,
                        metrics={"evidence": len(bucket), "failures_checked": failures},
                        notes="recent failures outweigh replay evidence",
                    )
                report.add(
                    "procedure_candidate",
                    False,
                    "recent failures outweigh replay evidence",
                    tool_name=tool_name,
                    topic=topic,
                    evidence=len(bucket),
                    failures=failures,
                )
                continue
            if preview:
                report.add(
                    "procedure_preview",
                    True,
                    f"would strengthen {tool_name} for {topic}",
                    tool_name=tool_name,
                    topic=topic,
                    evidence=len(bucket),
                    failures=failures,
                )
                continue
            prototype = self._mean_embedding(bucket)
            self.memory.upsert_procedure(
                name=f"sleep:{tool_name}:{topic}",
                description=f"Sleep-consolidated use of {tool_name} for {topic}",
                tool_name=tool_name,
                trigger_embedding=prototype,
                success=True,
                payload={
                    "evidence": len(bucket),
                    "failures_checked": failures,
                    "created_by": "sleep_consolidation",
                    "episode_ids": [record.id for record in bucket],
                },
            )
            self.memory.record_experiment(
                kind="sleep_procedure_consolidation",
                candidate=f"{tool_name}:{topic}",
                accepted=True,
                metrics={"evidence": len(bucket), "failures_checked": failures},
                notes="repeated replay success strengthened procedure",
            )
            report.add(
                "procedure_strengthening",
                True,
                f"strengthened {tool_name} for {topic}",
                tool_name=tool_name,
                topic=topic,
                evidence=len(bucket),
                failures=failures,
            )

    def _matching_failures(
        self,
        records: list[MemoryRecord],
        tool_name: str,
        topic: str,
    ) -> int:
        failures = 0
        for record in records:
            if self._topic_key(record.text) != topic:
                continue
            internal = record.context.get("__sdnc", {})
            proposed = set(internal.get("tool_names", [])) | set(internal.get("proposed_tool_names", []))
            successful = set(internal.get("successful_tools", []))
            if tool_name in proposed and tool_name not in successful:
                failures += 1
            if record.feedback_score is not None and record.feedback_score < 0:
                failures += 1
        return failures

    def _protected_procedures(self) -> list[ProcedureRecord]:
        protected: list[ProcedureRecord] = []
        for procedure in self.memory.list_procedures():
            if (
                procedure.success_count >= self.config.procedure_min_successes
                and procedure.success_rate >= self.config.procedure_min_success_rate
            ):
                protected.append(procedure)
        return protected

    def _mean_embedding(self, records: list[MemoryRecord]) -> np.ndarray:
        matrix = np.stack([self._unit(record.embedding) for record in records])
        return self._unit(matrix.mean(axis=0))

    def _mean_active_drift(
        self,
        before: np.ndarray,
        after: np.ndarray,
        indices: list[int],
    ) -> float:
        if not indices:
            return 0.0
        drift = []
        for index in indices:
            old = self._unit(before[index])
            new = self._unit(after[index])
            drift.append(1.0 - float(np.dot(old, new)))
        return float(np.mean(drift)) if drift else 0.0

    def _unit(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def _topic_key(self, text: str) -> str:
        words = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        stop = {"dans", "avec", "pour", "cherche", "recherche", "fichier", "projet", "docs"}
        useful = [word for word in words if word not in stop]
        return "-".join(useful[:4]) or "general"
