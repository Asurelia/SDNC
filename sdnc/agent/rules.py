"""Neuro-symbolic rules extracted from verified SDNC experience."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from time import time
from typing import Any

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.memory import PersistentMemory, RuleRecord
from sdnc.agent.types import MemoryRecord


@dataclass(frozen=True)
class RuleMatch:
    """One rule that can influence routing or explanation."""

    rule: RuleRecord
    score: float

    @property
    def tool_name(self) -> str:
        return self.rule.action_tool

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.rule.id,
            "name": self.rule.name,
            "trigger_pattern": self.rule.trigger_pattern,
            "action_tool": self.rule.action_tool,
            "expected_outcome": self.rule.expected_outcome,
            "confidence": self.rule.confidence,
            "similarity": self.rule.similarity,
            "score": self.score,
            "provenance": list(self.rule.provenance),
            "counterexamples": list(self.rule.counterexamples),
        }


@dataclass(frozen=True)
class RuleAction:
    """One rule consolidation decision."""

    kind: str
    accepted: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleConsolidationReport:
    """Result of one neuro-symbolic consolidation pass."""

    timestamp: float
    actions: list[RuleAction] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return sum(1 for action in self.actions if action.accepted)

    @property
    def promoted_count(self) -> int:
        return sum(1 for action in self.actions if action.kind == "rule_promotion" and action.accepted)

    @property
    def weakened_count(self) -> int:
        return sum(1 for action in self.actions if action.kind == "rule_contradiction" and action.accepted)

    @property
    def forked_count(self) -> int:
        return sum(1 for action in self.actions if action.kind == "rule_conflict_fork" and action.accepted)

    def add(self, kind: str, accepted: bool, reason: str, **details: Any) -> None:
        self.actions.append(RuleAction(kind=kind, accepted=accepted, reason=reason, details=details))

    def summary(self) -> str:
        if not self.actions:
            return "Rule consolidation: no candidates."
        return (
            f"Rule consolidation: {self.promoted_count} promoted, "
            f"{self.weakened_count} weakened, "
            f"{self.forked_count} forked, "
            f"{len(self.actions) - self.accepted_count} rejected."
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "accepted_count": self.accepted_count,
            "promoted_count": self.promoted_count,
            "weakened_count": self.weakened_count,
            "forked_count": self.forked_count,
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


class RuleEngine:
    """Promote repeated verified traces into local, provenance-backed rules."""

    def __init__(self, config: AutonomousConfig, memory: PersistentMemory):
        self.config = config
        self.memory = memory

    def match(self, embedding: np.ndarray, top_k: int = 5) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        for rule in self.memory.retrieve_rules(embedding, top_k=top_k):
            score = rule.similarity * 0.55 + rule.confidence * 0.45
            if (
                rule.confidence >= self.config.rule_min_confidence
                and rule.similarity >= self.config.rule_match_similarity
            ):
                matches.append(RuleMatch(rule=rule, score=round(float(score), 6)))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches

    def attach_matches(
        self,
        matches: list[RuleMatch],
        target_kind: str,
        target_id: str,
        relation: str,
        provenance: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Attach matched rules to one SDNC asset without making them truth."""
        if not target_id:
            return 0
        if target_kind not in {"expert", "sensory_prototype"}:
            raise ValueError(f"unsupported rule link target: {target_kind}")
        attached = 0
        for match in matches:
            if (
                match.rule.confidence < self.config.rule_min_confidence
                or match.rule.similarity < self.config.rule_match_similarity
            ):
                continue
            link_payload = {
                **dict(payload or {}),
                "rule_name": match.rule.name,
                "action_tool": match.rule.action_tool,
                "rule_similarity": match.rule.similarity,
                "rule_score": match.score,
            }
            self.memory.upsert_rule_link(
                rule_id=match.rule.id,
                target_kind=target_kind,
                target_id=target_id,
                relation=relation,
                confidence=match.score,
                provenance=provenance,
                payload=link_payload,
            )
            attached += 1
        return attached

    def consolidate_recent(self, limit: int | None = None) -> RuleConsolidationReport:
        report = RuleConsolidationReport(timestamp=time())
        records = self.memory.recent(limit or self.config.rule_recent_limit)
        buckets = self._success_buckets(records)
        self._scan_existing_contradictions(records, report)
        if not buckets:
            self._fork_incompatible_rules(report)
            if not report.actions:
                report.add("rule_scan", False, "no repeated successful tool traces")
            return report

        for (tool_name, topic), positives in sorted(buckets.items()):
            failures = self._counterexamples(records, tool_name, topic)
            evidence_count = len(positives)
            confidence = self._confidence(evidence_count, len(failures))
            name = f"rule:{tool_name}:{topic}"
            if evidence_count < self.config.rule_min_evidence:
                report.add(
                    "rule_candidate",
                    False,
                    "not enough verified traces",
                    tool_name=tool_name,
                    topic=topic,
                    evidence=evidence_count,
                )
                continue
            if failures and len(failures) >= evidence_count:
                self._weaken_existing(name, failures, report)
                continue
            if confidence < self.config.rule_min_confidence:
                report.add(
                    "rule_candidate",
                    False,
                    "confidence below rule threshold",
                    tool_name=tool_name,
                    topic=topic,
                    evidence=evidence_count,
                    counterexamples=len(failures),
                    confidence=confidence,
                )
                continue
            prototype = self._mean_embedding(positives)
            rule_id = self.memory.upsert_rule(
                name=name,
                trigger_pattern=topic,
                preconditions={
                    "topic": topic,
                    "successful_tool_required": tool_name,
                    "min_evidence": self.config.rule_min_evidence,
                },
                action_tool=tool_name,
                expected_outcome=f"{tool_name} should help for {topic}",
                trigger_embedding=prototype,
                confidence=confidence,
                status="enabled",
                provenance=[record.id for record in positives],
                counterexamples=[record.id for record in failures],
                payload={
                    "created_by": "rule_consolidation",
                    "evidence": evidence_count,
                    "counterexamples": len(failures),
                    "source": "episodes",
                },
            )
            self.memory.record_experiment(
                kind="rule_promotion",
                candidate=name,
                accepted=True,
                metrics={
                    "evidence": evidence_count,
                    "counterexamples": len(failures),
                    "confidence": confidence,
                },
                notes="verified traces promoted into neuro-symbolic rule",
            )
            report.add(
                "rule_promotion",
                True,
                f"promoted {name}",
                rule_id=rule_id,
                tool_name=tool_name,
                topic=topic,
                evidence=evidence_count,
                counterexamples=len(failures),
                confidence=confidence,
            )
        self._fork_incompatible_rules(report)
        return report

    def _scan_existing_contradictions(
        self,
        records: list[MemoryRecord],
        report: RuleConsolidationReport,
    ) -> None:
        for rule in self.memory.list_rules(status="enabled"):
            failures = self._counterexamples(records, rule.action_tool, rule.trigger_pattern)
            if len(failures) >= self.config.rule_min_evidence:
                self._weaken_existing(rule.name, failures, report)

    def _fork_incompatible_rules(self, report: RuleConsolidationReport) -> None:
        """Keep useful conflicting rules as explicit forks for later arbitration."""
        enabled = [
            rule
            for rule in self.memory.list_rules(status="enabled")
            if (
                rule.confidence >= self.config.rule_min_confidence
                and len(rule.provenance) >= self.config.rule_min_evidence
            )
        ]
        by_topic: dict[str, list[RuleRecord]] = defaultdict(list)
        for rule in enabled:
            by_topic[rule.trigger_pattern].append(rule)

        for topic, rules in sorted(by_topic.items()):
            if len(rules) < 2:
                continue
            for left, right in combinations(sorted(rules, key=lambda item: item.id), 2):
                if left.action_tool == right.action_tool:
                    continue
                evidence = {
                    "topic": topic,
                    "min_evidence": self.config.rule_min_evidence,
                    "rules": {
                        left.id: self._conflict_rule_evidence(left),
                        right.id: self._conflict_rule_evidence(right),
                    },
                }
                conflict_id, created = self.memory.upsert_rule_conflict(
                    topic=topic,
                    left_rule_id=left.id,
                    right_rule_id=right.id,
                    reason="same trigger pattern proposes incompatible tools",
                    evidence=evidence,
                    payload={
                        "created_by": "rule_consolidation",
                        "left_rule_name": left.name,
                        "right_rule_name": right.name,
                        "left_action_tool": left.action_tool,
                        "right_action_tool": right.action_tool,
                    },
                    status="forked",
                )
                if not created:
                    continue
                self.memory.record_experiment(
                    kind="rule_conflict_fork",
                    candidate=f"{topic}:{left.action_tool}|{right.action_tool}",
                    accepted=True,
                    metrics={
                        "left_confidence": left.confidence,
                        "right_confidence": right.confidence,
                        "left_evidence": len(left.provenance),
                        "right_evidence": len(right.provenance),
                    },
                    notes="incompatible useful rules kept as competing forks",
                )
                report.add(
                    "rule_conflict_fork",
                    True,
                    f"forked incompatible rules for {topic}",
                    conflict_id=conflict_id,
                    topic=topic,
                    left_rule_id=left.id,
                    right_rule_id=right.id,
                    left_tool=left.action_tool,
                    right_tool=right.action_tool,
                )

    def record_rule_outcomes(
        self,
        matches: list[RuleMatch],
        successful_tools: set[str],
        episode_id: str,
    ) -> None:
        for match in matches:
            success = match.tool_name in successful_tools
            self.memory.record_rule_evidence(
                match.rule.id,
                success=success,
                episode_id=episode_id,
                confidence_delta=(
                    self.config.rule_success_boost
                    if success
                    else -self.config.rule_counterexample_penalty
                ),
                counterexample=not success,
            )

    def _success_buckets(self, records: list[MemoryRecord]) -> dict[tuple[str, str], list[MemoryRecord]]:
        buckets: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
        for record in records:
            if record.feedback_score is not None and record.feedback_score < 0:
                continue
            internal = record.context.get("__sdnc", {})
            for tool_name in internal.get("successful_tools", []):
                buckets[(str(tool_name), self._topic_key(record.text))].append(record)
        return buckets

    def _counterexamples(
        self,
        records: list[MemoryRecord],
        tool_name: str,
        topic: str,
    ) -> list[MemoryRecord]:
        failures: list[MemoryRecord] = []
        for record in records:
            if self._topic_key(record.text) != topic:
                continue
            internal = record.context.get("__sdnc", {})
            proposed = set(internal.get("tool_names", [])) | set(internal.get("proposed_tool_names", []))
            successful = set(internal.get("successful_tools", []))
            if tool_name in proposed and tool_name not in successful:
                failures.append(record)
            elif record.feedback_score is not None and record.feedback_score < 0:
                failures.append(record)
        return failures

    def _weaken_existing(
        self,
        name: str,
        failures: list[MemoryRecord],
        report: RuleConsolidationReport,
    ) -> None:
        existing = next((rule for rule in self.memory.list_rules() if rule.name == name), None)
        if existing is None:
            report.add(
                "rule_contradiction",
                False,
                "contradictory evidence found before rule promotion",
                rule_name=name,
                counterexamples=len(failures),
            )
            return
        for failure in failures:
            self.memory.record_rule_evidence(
                existing.id,
                success=False,
                episode_id=failure.id,
                confidence_delta=-self.config.rule_counterexample_penalty,
                counterexample=True,
            )
        self.memory.record_experiment(
            kind="rule_contradiction",
            candidate=name,
            accepted=True,
            metrics={"counterexamples": len(failures)},
            notes="contradictory evidence weakened rule",
        )
        report.add(
            "rule_contradiction",
            True,
            f"weakened {name}",
            rule_id=existing.id,
            counterexamples=len(failures),
        )

    def _confidence(self, evidence_count: int, counterexample_count: int) -> float:
        raw = 0.48 + evidence_count * self.config.rule_success_boost
        raw -= counterexample_count * self.config.rule_counterexample_penalty
        return round(float(np.clip(raw, 0.0, 1.0)), 6)

    def _conflict_rule_evidence(self, rule: RuleRecord) -> dict[str, Any]:
        return {
            "name": rule.name,
            "action_tool": rule.action_tool,
            "confidence": rule.confidence,
            "provenance_count": len(rule.provenance),
            "counterexample_count": len(rule.counterexamples),
            "status": rule.status,
        }

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
