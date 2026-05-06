"""Provenance-preserving memory compaction for SDNC."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.memory import PersistentMemory, RuleLinkRecord, RuleRecord
from sdnc.agent.types import MemoryRecord


@dataclass(frozen=True)
class MemoryCompactionAction:
    """One memory compaction decision."""

    kind: str
    accepted: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCompactionReport:
    """Inspectable result of one compaction cycle."""

    timestamp: float
    preview: bool
    actions: list[MemoryCompactionAction] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return sum(1 for action in self.actions if action.accepted)

    @property
    def compacted_count(self) -> int:
        return sum(1 for action in self.actions if action.kind == "memory_compaction" and action.accepted)

    @property
    def protected_count(self) -> int:
        return sum(
            int(action.details.get("protected_count", 0))
            for action in self.actions
            if action.kind == "memory_compaction"
        )

    @property
    def rejected_count(self) -> int:
        return sum(1 for action in self.actions if not action.accepted)

    def add(self, kind: str, accepted: bool, reason: str, **details: Any) -> None:
        self.actions.append(MemoryCompactionAction(kind=kind, accepted=accepted, reason=reason, details=details))

    def summary(self) -> str:
        mode = "preview" if self.preview else "run"
        if not self.actions:
            return f"Memory compaction {mode}: no candidates."
        return (
            f"Memory compaction {mode}: {self.compacted_count} compacted, "
            f"{self.protected_count} protected evidence ids, "
            f"{self.rejected_count} rejected."
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "preview": self.preview,
            "accepted_count": self.accepted_count,
            "compacted_count": self.compacted_count,
            "protected_count": self.protected_count,
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


class MemoryCompactionCycle:
    """Build compact episode prototypes while preserving rule evidence."""

    def __init__(self, config: AutonomousConfig, memory: PersistentMemory):
        self.config = config
        self.memory = memory

    def run(
        self,
        preview: bool = False,
        batch_size: int | None = None,
        recent_limit: int | None = None,
    ) -> MemoryCompactionReport:
        report = MemoryCompactionReport(timestamp=time(), preview=preview)
        records = self.memory.recent(limit=recent_limit or self.config.memory_compaction_recent_limit)
        if not records:
            report.add("memory_scan", False, "no episodes available")
            return report

        rules = self.memory.list_rules()
        links = self.memory.list_rule_links(limit=1000)
        evidence_ids = self._evidence_ids(rules, links)
        buckets = self._buckets(records)
        compacted = 0
        batch_limit = max(1, int(batch_size or self.config.memory_compaction_batch_size))

        for topic, bucket in sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True):
            compactable = [
                record
                for record in bucket
                if record.id not in evidence_ids and not _negative_feedback(record)
            ]
            protected = [
                record.id
                for record in bucket
                if record.id in evidence_ids or _negative_feedback(record)
            ]
            if len(compactable) < self.config.memory_compaction_min_group_size:
                report.add(
                    "memory_candidate",
                    False,
                    "not enough non-protected repeated episodes",
                    topic=topic,
                    candidate_count=len(compactable),
                    protected_count=len(protected),
                )
                continue
            selected = compactable[:batch_limit]
            related_ids = {record.id for record in selected} | set(protected)
            rule_ids = self._related_rule_ids(rules, related_ids)
            link_ids = self._related_link_ids(links, related_ids)
            summary = self._summary(topic, selected, protected)
            if preview:
                report.add(
                    "memory_compaction",
                    True,
                    f"would compact {topic}",
                    topic=topic,
                    episode_count=len(selected),
                    protected_count=len(protected),
                    rule_count=len(rule_ids),
                    rule_link_count=len(link_ids),
                    summary=summary,
                )
            else:
                compaction_id = self.memory.upsert_memory_compaction(
                    key=f"topic:{topic}",
                    summary=summary,
                    embedding=self._mean_embedding(selected),
                    episode_ids=[record.id for record in selected],
                    protected_episode_ids=protected,
                    rule_ids=rule_ids,
                    rule_link_ids=link_ids,
                    payload={
                        "topic": topic,
                        "created_by": "memory_compaction",
                        "raw_episodes_preserved": True,
                        "source_episode_count": len(selected),
                    },
                )
                self.memory.record_experiment(
                    kind="memory_compaction",
                    candidate=topic,
                    accepted=True,
                    metrics={
                        "episode_count": len(selected),
                        "protected_count": len(protected),
                        "rule_count": len(rule_ids),
                        "rule_link_count": len(link_ids),
                    },
                    notes="compacted repeated episodes without deleting source evidence",
                )
                report.add(
                    "memory_compaction",
                    True,
                    f"compacted {topic}",
                    compaction_id=compaction_id,
                    topic=topic,
                    episode_count=len(selected),
                    protected_count=len(protected),
                    rule_count=len(rule_ids),
                    rule_link_count=len(link_ids),
                    summary=summary,
                )
            compacted += 1
            if compacted >= batch_limit:
                break

        if compacted == 0 and not any(action.accepted for action in report.actions):
            report.add("memory_scan", False, "no compactable repeated episode groups")
        return report

    def _buckets(self, records: list[MemoryRecord]) -> dict[str, list[MemoryRecord]]:
        buckets: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in records:
            buckets[self._topic_key(record.text)].append(record)
        return buckets

    def _evidence_ids(self, rules: list[RuleRecord], links: list[RuleLinkRecord]) -> set[str]:
        evidence: set[str] = set()
        for rule in rules:
            evidence.update(rule.provenance)
            evidence.update(rule.counterexamples)
        for link in links:
            evidence.update(link.provenance)
        return {item for item in evidence if item}

    def _related_rule_ids(self, rules: list[RuleRecord], ids: set[str]) -> list[str]:
        return [
            rule.id
            for rule in rules
            if ids.intersection(set(rule.provenance) | set(rule.counterexamples))
        ]

    def _related_link_ids(self, links: list[RuleLinkRecord], ids: set[str]) -> list[str]:
        return [link.id for link in links if ids.intersection(set(link.provenance))]

    def _summary(self, topic: str, records: list[MemoryRecord], protected_ids: list[str]) -> str:
        limit = max(160, int(self.config.memory_compaction_max_summary_chars))
        snippets = []
        for record in records[:6]:
            snippets.append(record.text.replace("\n", " ")[:140])
        text = f"topic={topic}; compacted={len(records)}; protected={len(protected_ids)}; " + " | ".join(snippets)
        return text[:limit]

    def _mean_embedding(self, records: list[MemoryRecord]) -> np.ndarray:
        matrix = np.stack([self._unit(record.embedding) for record in records])
        return self._unit(matrix.mean(axis=0))

    def _unit(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def _topic_key(self, text: str) -> str:
        words = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        stop = {
            "avec",
            "dans",
            "depuis",
            "fichier",
            "pour",
            "projet",
            "recherche",
            "signal",
        }
        useful = [word for word in words if word not in stop]
        return "-".join(useful[:4]) or "general"


def _negative_feedback(record: MemoryRecord) -> bool:
    return record.feedback_score is not None and record.feedback_score < 0
