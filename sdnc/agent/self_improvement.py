"""Controlled self-improvement for SDNC.

This module is the opposite of uncontrolled self-modification. It turns raw
experience into proposals, then only accepts changes that pass a small
multi-source consensus gate.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.memory import PersistentMemory, ProcedureRecord
from sdnc.agent.plasticity import LocalCircuitLearner
from sdnc.agent.types import MemoryRecord


@dataclass
class ImprovementAction:
    """One accepted or rejected self-improvement decision."""

    kind: str
    accepted: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementReport:
    """Summary of one maintenance/self-improvement cycle."""

    timestamp: float
    actions: list[ImprovementAction] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return sum(1 for action in self.actions if action.accepted)

    def add(self, kind: str, accepted: bool, reason: str, **details: Any) -> None:
        self.actions.append(
            ImprovementAction(kind=kind, accepted=accepted, reason=reason, details=details)
        )

    def summary(self) -> str:
        if not self.actions:
            return "Self-improvement cycle: no candidates."
        lines = [
            f"Self-improvement cycle: {self.accepted_count}/{len(self.actions)} accepted."
        ]
        for action in self.actions:
            status = "accepted" if action.accepted else "rejected"
            lines.append(f"- {action.kind}:{status}: {action.reason}")
        return "\n".join(lines)


class ConsensusGate:
    """Validates whether experience is strong enough to alter architecture."""

    def __init__(self, config: AutonomousConfig):
        self.config = config

    def evaluate_cluster(self, cluster: list[MemoryRecord]) -> tuple[bool, str, set[str]]:
        if len(cluster) < self.config.min_growth_evidence:
            return False, "not enough repeated evidence", set()

        mean_salience = sum(record.salience for record in cluster) / len(cluster)
        if mean_salience < self.config.min_growth_salience:
            return False, f"mean salience too low: {mean_salience:.3f}", set()

        sources = self._sources(cluster)
        if len(sources) < self.config.min_consensus_sources:
            return False, f"not enough source diversity: {sorted(sources)}", sources

        negative = [
            record for record in cluster
            if record.feedback_score is not None and record.feedback_score < 0
        ]
        if len(negative) > len(cluster) / 3:
            return False, "negative feedback ratio too high", sources

        return True, "cluster passed consensus", sources

    def _sources(self, cluster: list[MemoryRecord]) -> set[str]:
        sources: set[str] = set()
        for record in cluster:
            internal = record.context.get("__sdnc", {})
            if record.salience >= self.config.min_growth_salience:
                sources.add("salience")
            if internal.get("novelty", 0.0) >= self.config.min_growth_novelty:
                sources.add("novelty")
            if record.feedback_score is not None and record.feedback_score > 0:
                sources.add("feedback")
            if internal.get("successful_tools"):
                sources.add("tool_success")
            if record.similarity >= self.config.growth_similarity_threshold:
                sources.add("memory_match")
        return sources


class SelfImprovementCycle:
    """Runs bounded growth, consolidation, and pruning cycles."""

    def __init__(
        self,
        config: AutonomousConfig,
        memory: PersistentMemory,
        learner: LocalCircuitLearner,
    ):
        self.config = config
        self.memory = memory
        self.learner = learner
        self.gate = ConsensusGate(config)

    def run(self, recent_limit: int = 200) -> ImprovementReport:
        report = ImprovementReport(timestamp=time())
        records = self.memory.recent(limit=recent_limit)
        clusters = self._cluster_records(records)

        growths = 0
        for cluster in clusters:
            accepted, reason, sources = self.gate.evaluate_cluster(cluster)
            if not accepted:
                if len(cluster) >= self.config.min_growth_evidence:
                    report.add("growth_candidate", False, reason, size=len(cluster))
                continue
            if growths >= self.config.max_growth_per_cycle:
                report.add("growth_candidate", False, "cycle growth limit reached", size=len(cluster))
                continue
            experiment_ok, experiment_reason, metrics = self._experiment_growth_candidate(
                cluster,
                records,
            )
            experiment_id = self.memory.record_experiment(
                kind="circuit_growth",
                candidate=self._topic_key(cluster[0].text),
                accepted=experiment_ok,
                metrics=metrics,
                notes=experiment_reason,
            )
            report.add(
                "sandbox_experiment",
                experiment_ok,
                experiment_reason,
                experiment_id=experiment_id,
                **metrics,
            )
            if not experiment_ok:
                continue
            prototype = self._mean_embedding(cluster)
            idx, mode = self.learner.grow_circuit(prototype, source="consensus")
            growths += 1
            report.add(
                "circuit_growth",
                True,
                f"{mode} circuit {idx}",
                circuit_id=idx,
                evidence=len(cluster),
                sources=sorted(sources),
            )

        self._promote_repeated_tool_patterns(records, report)
        self._prune_weak_procedures(report)
        pruned = self.learner.prune_connections(self.config.connection_prune_threshold)
        if pruned:
            report.add("connection_pruning", True, f"pruned {pruned} weak links", count=pruned)
        return report

    def _cluster_records(self, records: list[MemoryRecord]) -> list[list[MemoryRecord]]:
        candidates = [
            record for record in records
            if record.salience >= self.config.min_growth_salience
            and record.feedback_score != -1.0
        ]
        clusters: list[list[MemoryRecord]] = []
        for record in candidates:
            placed = False
            for cluster in clusters:
                centroid = self._mean_embedding(cluster)
                sim = float(np.dot(self._unit(record.embedding), centroid))
                if sim >= self.config.growth_similarity_threshold:
                    cluster.append(record)
                    placed = True
                    break
            if not placed:
                clusters.append([record])
        clusters.sort(key=len, reverse=True)
        return clusters

    def _experiment_growth_candidate(
        self,
        cluster: list[MemoryRecord],
        records: list[MemoryRecord],
    ) -> tuple[bool, str, dict[str, float]]:
        """Sandbox-test whether a new circuit adds useful separation."""
        if not self.config.exploration_enabled:
            return True, "exploration disabled by config", {}

        prototype = self._mean_embedding(cluster)
        positives = [float(np.dot(self._unit(record.embedding), prototype)) for record in cluster]
        cluster_ids = {record.id for record in cluster}
        negatives = [
            float(np.dot(self._unit(record.embedding), prototype))
            for record in records
            if record.id not in cluster_ids
        ]

        positive_fit = float(np.mean(positives)) if positives else 0.0
        negative_fit = float(np.mean(negatives)) if negatives else 0.0
        contrast_gain = positive_fit - negative_fit

        existing_coverage = 0.0
        if self.learner.circuit_keys.size:
            existing_coverage = float(
                np.mean([
                    np.max(self.learner.circuit_keys @ self._unit(record.embedding))
                    for record in cluster
                ])
            )

        metrics = {
            "positive_fit": positive_fit,
            "negative_fit": negative_fit,
            "contrast_gain": contrast_gain,
            "existing_coverage": existing_coverage,
        }

        if existing_coverage >= self.config.existing_coverage_threshold:
            return False, "existing circuits already cover candidate", metrics
        if contrast_gain < self.config.min_experiment_gain:
            return False, f"experiment gain too low: {contrast_gain:.3f}", metrics
        return True, "sandbox experiment shows useful separation", metrics

    def _promote_repeated_tool_patterns(
        self,
        records: list[MemoryRecord],
        report: ImprovementReport,
    ) -> None:
        buckets: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
        for record in records:
            internal = record.context.get("__sdnc", {})
            for tool_name in internal.get("successful_tools", []):
                key = (tool_name, self._topic_key(record.text))
                buckets[key].append(record)

        for (tool_name, topic), bucket in buckets.items():
            if len(bucket) < self.config.procedure_min_successes:
                continue
            feedbacks = [
                record.feedback_score for record in bucket
                if record.feedback_score is not None
            ]
            if feedbacks and sum(1 for score in feedbacks if score > 0) / len(feedbacks) < self.config.procedure_min_success_rate:
                continue
            prototype = self._mean_embedding(bucket)
            self.memory.upsert_procedure(
                name=f"auto:{tool_name}:{topic}",
                description=f"Auto-learned use of {tool_name} for {topic}",
                tool_name=tool_name,
                trigger_embedding=prototype,
                success=True,
                payload={"evidence": len(bucket), "created_by": "self_improvement"},
            )
            self.memory.record_experiment(
                kind="skill_promotion",
                candidate=f"{tool_name}:{topic}",
                accepted=True,
                metrics={"evidence": len(bucket)},
                notes="repeated successful tool pattern promoted",
            )
            report.add(
                "skill_promotion",
                True,
                f"promoted {tool_name} for {topic}",
                evidence=len(bucket),
            )

    def _prune_weak_procedures(self, report: ImprovementReport) -> None:
        for procedure in self.memory.list_procedures():
            total = procedure.success_count + procedure.failure_count
            if (
                procedure.failure_count >= self.config.procedure_prune_failures
                and total > 0
                and procedure.success_rate < self.config.procedure_min_success_rate
            ):
                self.memory.delete_procedure(procedure.id)
                report.add(
                    "skill_pruning",
                    True,
                    f"removed weak procedure {procedure.name}",
                    success_rate=procedure.success_rate,
                )
        self._dedupe_procedures(report)

    def _dedupe_procedures(self, report: ImprovementReport) -> None:
        by_tool: dict[str, list[ProcedureRecord]] = defaultdict(list)
        for procedure in self.memory.list_procedures():
            by_tool[procedure.tool_name].append(procedure)
        for procedures in by_tool.values():
            procedures.sort(key=lambda item: (item.success_rate, item.success_count), reverse=True)
            keep: list[ProcedureRecord] = []
            for procedure in procedures:
                duplicate_of = None
                for kept in keep:
                    sim = float(np.dot(self._unit(procedure.trigger_embedding), self._unit(kept.trigger_embedding)))
                    if sim > 0.95:
                        duplicate_of = kept
                        break
                if duplicate_of is None:
                    keep.append(procedure)
                else:
                    self.memory.delete_procedure(procedure.id)
                    report.add(
                        "skill_dedupe",
                        True,
                        f"merged duplicate {procedure.name} into {duplicate_of.name}",
                    )

    def _mean_embedding(self, records: list[MemoryRecord]) -> np.ndarray:
        matrix = np.stack([self._unit(record.embedding) for record in records])
        return self._unit(matrix.mean(axis=0))

    def _unit(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def _topic_key(self, text: str) -> str:
        words = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        stop = {"dans", "avec", "pour", "cherche", "recherche", "fichier", "projet", "docs"}
        useful = [word for word in words if word not in stop]
        return "-".join(useful[:4]) or "general"
