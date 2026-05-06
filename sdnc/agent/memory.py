"""Persistent memory for the autonomous SDNC loop."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from sdnc.agent.types import MemoryRecord


@dataclass
class ProcedureRecord:
    """A learned tool-use pattern."""

    id: str
    name: str
    description: str
    tool_name: str
    trigger_embedding: np.ndarray
    success_count: int
    failure_count: int
    payload: dict[str, Any]
    similarity: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total else 0.0


@dataclass
class SyncEvent:
    """Local realtime/sync event."""

    id: int
    timestamp: float
    event_type: str
    source: str
    payload: dict[str, Any]


@dataclass
class LearningGapRecord:
    """A detected lack that SDNC can investigate later."""

    id: str
    timestamp: float
    kind: str
    description: str
    severity: float
    status: str
    payload: dict[str, Any]


@dataclass
class SourceStat:
    """Reliability trace for one learning source."""

    source_name: str
    proposals: int
    accepted: int
    rejected: int
    trust: float


@dataclass
class SensoryBindingRecord:
    """A persisted cross-modal sensory event."""

    id: str
    timestamp: float
    episode_id: str
    source: str
    modalities: list[str]
    sample_ids: list[str]
    summary: str
    embedding: np.ndarray
    binding_score: float
    salience: float
    features: dict[str, Any]
    context: dict[str, Any]
    similarity: float = 0.0


@dataclass
class SensoryPrototypeRecord:
    """A compact prototype for repeated sensory events."""

    id: str
    key: str
    timestamp: float
    updated_at: float
    modalities: list[str]
    sources: list[str]
    sample_ids: list[str]
    summary: str
    centroid_embedding: np.ndarray
    observation_count: int
    confidence: float
    features: dict[str, Any]
    payload: dict[str, Any]
    similarity: float = 0.0


@dataclass
class TrainingFileRecord:
    """A local file queued for SDNC experience learning."""

    id: str
    timestamp: float
    updated_at: float
    name: str
    path: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    modality: str
    preview: str
    processed_episode_id: str
    error: str
    payload: dict[str, Any]


@dataclass
class CognitiveTraceRecord:
    """A persisted sparse cognitive workspace trace."""

    id: str
    timestamp: float
    episode_id: str
    input_text: str
    mode: str
    prediction: dict[str, Any]
    observation: dict[str, Any]
    attention: list[str]
    surprise: float
    uncertainty: float
    payload: dict[str, Any]


@dataclass
class RuleRecord:
    """A provenance-backed neuro-symbolic routing rule."""

    id: str
    name: str
    trigger_pattern: str
    preconditions: dict[str, Any]
    action_tool: str
    expected_outcome: str
    confidence: float
    status: str
    trigger_embedding: np.ndarray
    provenance: list[str]
    counterexamples: list[str]
    payload: dict[str, Any]
    similarity: float = 0.0


@dataclass
class RuleLinkRecord:
    """A learned relation between one rule and a living SDNC asset."""

    id: str
    timestamp: float
    updated_at: float
    rule_id: str
    target_kind: str
    target_id: str
    relation: str
    confidence: float
    provenance: list[str]
    payload: dict[str, Any]


@dataclass
class ExpertRecord:
    """A self-managed SDNC expert asset."""

    id: str
    name: str
    kind: str
    description: str
    status: str
    trigger_embedding: np.ndarray
    success_count: int
    failure_count: int
    utility: float
    estimated_vram_gb: float
    estimated_ram_gb: float
    hot: bool
    payload: dict[str, Any]
    similarity: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total else 0.0


class PersistentMemory:
    """SQLite-backed episodic and procedural memory."""

    def __init__(self, path: Path, embedding_dim: int):
        self.path = Path(path)
        self.embedding_dim = embedding_dim
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def store_episode(
        self,
        text: str,
        context: dict[str, Any],
        embedding: np.ndarray,
        active_circuits: list[int],
        salience: float,
        outcome: str | None = None,
        feedback_score: float | None = None,
    ) -> str:
        episode_id = str(uuid.uuid4())
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO episodes
                    (id, timestamp, text, context_json, embedding, active_circuits_json,
                     salience, outcome, feedback_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    time(),
                    text,
                    json.dumps(context, sort_keys=True, default=str),
                    self._pack_vector(embedding),
                    json.dumps(active_circuits),
                    float(salience),
                    outcome,
                    feedback_score,
                ),
            )
            self.conn.commit()
        return episode_id

    def update_episode_feedback(self, episode_id: str, feedback_score: float, outcome: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE episodes SET feedback_score = ?, outcome = ? WHERE id = ?",
                (float(feedback_score), outcome, episode_id),
            )
            self.conn.commit()

    def retrieve_similar(self, embedding: np.ndarray, top_k: int = 5) -> list[MemoryRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 2000"
            ).fetchall()
        query = self._unit(embedding)
        scored: list[MemoryRecord] = []
        for row in rows:
            emb = self._unpack_vector(row["embedding"])
            similarity = float(np.dot(query, self._unit(emb)))
            scored.append(self._row_to_memory(row, emb, similarity))
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:top_k]

    def recent(self, limit: int = 10) -> list[MemoryRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_memory(row, self._unpack_vector(row["embedding"]), 0.0) for row in rows]

    def salient_episodes(
        self,
        limit: int = 20,
        min_salience: float = 0.0,
    ) -> list[MemoryRecord]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM episodes
                WHERE salience >= ?
                ORDER BY salience DESC, timestamp DESC
                LIMIT ?
                """,
                (float(min_salience), int(limit)),
            ).fetchall()
        return [self._row_to_memory(row, self._unpack_vector(row["embedding"]), 0.0) for row in rows]

    def upsert_procedure(
        self,
        name: str,
        description: str,
        tool_name: str,
        trigger_embedding: np.ndarray,
        success: bool,
        payload: dict[str, Any] | None = None,
    ) -> None:
        payload = payload or {}
        with self._lock:
            existing = self.conn.execute(
                "SELECT * FROM procedures WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                self.conn.execute(
                    """
                    UPDATE procedures
                    SET description = ?, tool_name = ?, trigger_embedding = ?,
                        success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        last_used = ?, payload_json = ?
                    WHERE name = ?
                    """,
                    (
                        description,
                        tool_name,
                        self._pack_vector(trigger_embedding),
                        1 if success else 0,
                        0 if success else 1,
                        time(),
                        json.dumps(payload, sort_keys=True, default=str),
                        name,
                    ),
                )
            else:
                self.conn.execute(
                    """
                    INSERT INTO procedures
                        (id, name, description, tool_name, trigger_embedding,
                         success_count, failure_count, last_used, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        name,
                        description,
                        tool_name,
                        self._pack_vector(trigger_embedding),
                        1 if success else 0,
                        0 if success else 1,
                        time(),
                        json.dumps(payload, sort_keys=True, default=str),
                    ),
                )
            self.conn.commit()

    def retrieve_procedures(self, embedding: np.ndarray, top_k: int = 5) -> list[ProcedureRecord]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM procedures").fetchall()
        query = self._unit(embedding)
        procedures: list[ProcedureRecord] = []
        for row in rows:
            trigger = self._unpack_vector(row["trigger_embedding"])
            proc = ProcedureRecord(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                tool_name=row["tool_name"],
                trigger_embedding=trigger,
                success_count=int(row["success_count"]),
                failure_count=int(row["failure_count"]),
                payload=json.loads(row["payload_json"] or "{}"),
                similarity=float(np.dot(query, self._unit(trigger))),
            )
            procedures.append(proc)
        procedures.sort(key=lambda item: (item.similarity, item.success_rate), reverse=True)
        return procedures[:top_k]

    def list_procedures(self) -> list[ProcedureRecord]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM procedures").fetchall()
        procedures: list[ProcedureRecord] = []
        for row in rows:
            procedures.append(
                ProcedureRecord(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    tool_name=row["tool_name"],
                    trigger_embedding=self._unpack_vector(row["trigger_embedding"]),
                    success_count=int(row["success_count"]),
                    failure_count=int(row["failure_count"]),
                    payload=json.loads(row["payload_json"] or "{}"),
                )
            )
        return procedures

    def delete_procedure(self, procedure_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM procedures WHERE id = ?", (procedure_id,))
            self.conn.commit()

    def upsert_rule(
        self,
        name: str,
        trigger_pattern: str,
        preconditions: dict[str, Any],
        action_tool: str,
        expected_outcome: str,
        trigger_embedding: np.ndarray,
        confidence: float,
        status: str = "enabled",
        provenance: list[str] | None = None,
        counterexamples: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        now = time()
        provenance = provenance or []
        counterexamples = counterexamples or []
        payload = payload or {}
        with self._lock:
            existing = self.conn.execute("SELECT * FROM rules WHERE name = ?", (name,)).fetchone()
            if existing:
                rule_id = existing["id"]
                old_provenance = json.loads(existing["provenance_json"] or "[]")
                old_counterexamples = json.loads(existing["counterexamples_json"] or "[]")
                merged_provenance = _dedupe_strings([*old_provenance, *provenance])
                merged_counterexamples = _dedupe_strings([*old_counterexamples, *counterexamples])
                self.conn.execute(
                    """
                    UPDATE rules
                    SET trigger_pattern = ?, preconditions_json = ?, action_tool = ?,
                        expected_outcome = ?, confidence = ?, status = ?,
                        trigger_embedding = ?, updated_at = ?, provenance_json = ?,
                        counterexamples_json = ?, payload_json = ?
                    WHERE name = ?
                    """,
                    (
                        trigger_pattern,
                        json.dumps(preconditions, sort_keys=True, default=str),
                        action_tool,
                        expected_outcome,
                        float(confidence),
                        status,
                        self._pack_vector(trigger_embedding),
                        now,
                        json.dumps(merged_provenance, sort_keys=True, default=str),
                        json.dumps(merged_counterexamples, sort_keys=True, default=str),
                        json.dumps(payload, sort_keys=True, default=str),
                        name,
                    ),
                )
            else:
                rule_id = str(uuid.uuid4())
                self.conn.execute(
                    """
                    INSERT INTO rules
                        (id, name, trigger_pattern, preconditions_json, action_tool,
                         expected_outcome, confidence, status, trigger_embedding,
                         created_at, updated_at, provenance_json, counterexamples_json,
                         payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule_id,
                        name,
                        trigger_pattern,
                        json.dumps(preconditions, sort_keys=True, default=str),
                        action_tool,
                        expected_outcome,
                        float(confidence),
                        status,
                        self._pack_vector(trigger_embedding),
                        now,
                        now,
                        json.dumps(_dedupe_strings(provenance), sort_keys=True, default=str),
                        json.dumps(_dedupe_strings(counterexamples), sort_keys=True, default=str),
                        json.dumps(payload, sort_keys=True, default=str),
                    ),
                )
            self.conn.commit()
        return rule_id

    def record_rule_evidence(
        self,
        rule_id: str,
        success: bool,
        episode_id: str,
        confidence_delta: float,
        counterexample: bool = False,
    ) -> None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()
            if row is None:
                return
            provenance = json.loads(row["provenance_json"] or "[]")
            counterexamples = json.loads(row["counterexamples_json"] or "[]")
            if success:
                provenance = _dedupe_strings([*provenance, episode_id])
            if counterexample:
                counterexamples = _dedupe_strings([*counterexamples, episode_id])
            confidence = max(0.0, min(1.0, float(row["confidence"]) + float(confidence_delta)))
            status = row["status"]
            if confidence < 0.25 and counterexamples:
                status = "rejected"
            self.conn.execute(
                """
                UPDATE rules
                SET confidence = ?, status = ?, updated_at = ?,
                    provenance_json = ?, counterexamples_json = ?
                WHERE id = ?
                """,
                (
                    confidence,
                    status,
                    time(),
                    json.dumps(provenance, sort_keys=True, default=str),
                    json.dumps(counterexamples, sort_keys=True, default=str),
                    rule_id,
                ),
            )
            self.conn.commit()

    def retrieve_rules(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        statuses: tuple[str, ...] = ("enabled",),
    ) -> list[RuleRecord]:
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM rules WHERE status IN ({placeholders}) ORDER BY confidence DESC",
                statuses,
            ).fetchall()
        query = self._unit(embedding)
        rules = []
        for row in rows:
            trigger = self._unpack_vector(row["trigger_embedding"])
            rules.append(self._row_to_rule(row, trigger, float(np.dot(query, self._unit(trigger)))))
        rules.sort(key=lambda item: (item.similarity * 0.55 + item.confidence * 0.45), reverse=True)
        return rules[:top_k]

    def list_rules(self, status: str | None = None) -> list[RuleRecord]:
        with self._lock:
            if status is None:
                rows = self.conn.execute("SELECT * FROM rules ORDER BY confidence DESC").fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM rules WHERE status = ? ORDER BY confidence DESC",
                    (status,),
                ).fetchall()
        return [self._row_to_rule(row, self._unpack_vector(row["trigger_embedding"]), 0.0) for row in rows]

    def upsert_rule_link(
        self,
        rule_id: str,
        target_kind: str,
        target_id: str,
        relation: str,
        confidence: float,
        provenance: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Persist a provenance-backed attachment between a rule and an asset."""
        now = time()
        provenance = provenance or []
        payload = payload or {}
        confidence = float(max(0.0, min(1.0, confidence)))
        with self._lock:
            existing = self.conn.execute(
                """
                SELECT * FROM rule_links
                WHERE rule_id = ? AND target_kind = ? AND target_id = ? AND relation = ?
                """,
                (rule_id, target_kind, target_id, relation),
            ).fetchone()
            if existing:
                link_id = existing["id"]
                old_provenance = json.loads(existing["provenance_json"] or "[]")
                old_payload = json.loads(existing["payload_json"] or "{}")
                self.conn.execute(
                    """
                    UPDATE rule_links
                    SET updated_at = ?, confidence = ?, provenance_json = ?, payload_json = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        max(float(existing["confidence"]), confidence),
                        json.dumps(
                            _dedupe_strings([*old_provenance, *provenance]),
                            sort_keys=True,
                            default=str,
                        ),
                        json.dumps({**old_payload, **payload}, sort_keys=True, default=str),
                        link_id,
                    ),
                )
            else:
                link_id = str(uuid.uuid4())
                self.conn.execute(
                    """
                    INSERT INTO rule_links
                        (id, timestamp, updated_at, rule_id, target_kind, target_id,
                         relation, confidence, provenance_json, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        link_id,
                        now,
                        now,
                        rule_id,
                        target_kind,
                        target_id,
                        relation,
                        confidence,
                        json.dumps(_dedupe_strings(provenance), sort_keys=True, default=str),
                        json.dumps(payload, sort_keys=True, default=str),
                    ),
                )
            self.conn.commit()
        return link_id

    def list_rule_links(
        self,
        rule_id: str | None = None,
        target_kind: str | None = None,
        target_id: str | None = None,
        limit: int = 100,
    ) -> list[RuleLinkRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if rule_id:
            clauses.append("rule_id = ?")
            params.append(rule_id)
        if target_kind:
            clauses.append("target_kind = ?")
            params.append(target_kind)
        if target_id:
            clauses.append("target_id = ?")
            params.append(target_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM rule_links {where} ORDER BY updated_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._row_to_rule_link(row) for row in rows]

    def record_tool_result(self, tool_name: str, success: bool) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO tool_stats (tool_name, uses, successes, failures, preference)
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(tool_name) DO UPDATE SET
                    uses = uses + 1,
                    successes = successes + ?,
                    failures = failures + ?,
                    preference = preference + ?
                """,
                (
                    tool_name,
                    1 if success else 0,
                    0 if success else 1,
                    0.05 if success else -0.05,
                    1 if success else 0,
                    0 if success else 1,
                    0.05 if success else -0.05,
                ),
            )
            self.conn.commit()

    def record_experiment(
        self,
        kind: str,
        candidate: str,
        accepted: bool,
        metrics: dict[str, Any],
        notes: str,
    ) -> str:
        experiment_id = str(uuid.uuid4())
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO experiments
                    (id, timestamp, kind, candidate, accepted, metrics_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    time(),
                    kind,
                    candidate,
                    1 if accepted else 0,
                    json.dumps(metrics, sort_keys=True, default=str),
                    notes,
                ),
            )
            self.conn.commit()
        return experiment_id

    def record_learning_gap(
        self,
        kind: str,
        description: str,
        severity: float,
        payload: dict[str, Any] | None = None,
        status: str = "open",
    ) -> str:
        gap_id = str(uuid.uuid4())
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO learning_gaps
                    (id, timestamp, kind, description, severity, status, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gap_id,
                    time(),
                    kind,
                    description,
                    float(severity),
                    status,
                    json.dumps(payload or {}, sort_keys=True, default=str),
                ),
            )
            self.conn.commit()
        return gap_id

    def update_learning_gap_status(self, gap_id: str, status: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE learning_gaps SET status = ? WHERE id = ?",
                (status, gap_id),
            )
            self.conn.commit()

    def recent_learning_gaps(self, limit: int = 20) -> list[LearningGapRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM learning_gaps ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            LearningGapRecord(
                id=row["id"],
                timestamp=float(row["timestamp"]),
                kind=row["kind"],
                description=row["description"],
                severity=float(row["severity"]),
                status=row["status"],
                payload=json.loads(row["payload_json"] or "{}"),
            )
            for row in rows
        ]

    def record_source_feedback(self, source_name: str, accepted: bool) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO source_stats (source_name, proposals, accepted, rejected, trust)
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(source_name) DO UPDATE SET
                    proposals = proposals + 1,
                    accepted = accepted + ?,
                    rejected = rejected + ?,
                    trust = max(0.0, min(1.0, trust + ?))
                """,
                (
                    source_name,
                    1 if accepted else 0,
                    0 if accepted else 1,
                    0.55 + (0.05 if accepted else -0.05),
                    1 if accepted else 0,
                    0 if accepted else 1,
                    0.04 if accepted else -0.04,
                ),
            )
            self.conn.commit()

    def list_source_stats(self) -> list[SourceStat]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM source_stats ORDER BY trust DESC").fetchall()
        return [
            SourceStat(
                source_name=row["source_name"],
                proposals=int(row["proposals"]),
                accepted=int(row["accepted"]),
                rejected=int(row["rejected"]),
                trust=float(row["trust"]),
            )
            for row in rows
        ]

    def store_sensory_binding(
        self,
        event_id: str,
        episode_id: str,
        source: str,
        modalities: list[str],
        sample_ids: list[str],
        summary: str,
        embedding: np.ndarray,
        binding_score: float,
        salience: float,
        features: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO sensory_bindings
                    (id, timestamp, episode_id, source, modalities_json, sample_ids_json,
                     summary, embedding, binding_score, salience, features_json, context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    summary = excluded.summary,
                    embedding = excluded.embedding,
                    binding_score = excluded.binding_score,
                    salience = excluded.salience,
                    features_json = excluded.features_json,
                    context_json = excluded.context_json
                """,
                (
                    event_id,
                    time(),
                    episode_id,
                    source,
                    json.dumps(modalities, sort_keys=True, default=str),
                    json.dumps(sample_ids, sort_keys=True, default=str),
                    summary,
                    self._pack_vector(embedding),
                    float(binding_score),
                    float(salience),
                    json.dumps(features, sort_keys=True, default=str),
                    json.dumps(context, sort_keys=True, default=str),
                ),
            )
            self.conn.commit()
        return event_id

    def recent_sensory_bindings(self, limit: int = 20) -> list[SensoryBindingRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM sensory_bindings ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            self._row_to_sensory_binding(row, self._unpack_vector(row["embedding"]), 0.0)
            for row in rows
        ]

    def retrieve_sensory_bindings(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SensoryBindingRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM sensory_bindings ORDER BY timestamp DESC LIMIT 1000"
            ).fetchall()
        query = self._unit(embedding)
        bindings: list[SensoryBindingRecord] = []
        for row in rows:
            emb = self._unpack_vector(row["embedding"])
            bindings.append(
                self._row_to_sensory_binding(row, emb, float(np.dot(query, self._unit(emb))))
            )
        bindings.sort(key=lambda item: (item.similarity, item.salience), reverse=True)
        return bindings[:top_k]

    def upsert_sensory_prototype(
        self,
        key: str,
        modalities: list[str],
        source: str,
        sample_ids: list[str],
        summary: str,
        embedding: np.ndarray,
        confidence: float,
        features: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> str:
        payload = payload or {}
        now = time()
        with self._lock:
            existing = self.conn.execute(
                "SELECT * FROM sensory_prototypes WHERE key = ?",
                (key,),
            ).fetchone()
            if existing:
                prototype_id = existing["id"]
                old_count = int(existing["observation_count"])
                old_centroid = self._unpack_vector(existing["centroid_embedding"])
                new_count = old_count + 1
                centroid = self._unit((old_centroid * old_count + self._unit(embedding)) / new_count)
                sources = _dedupe_strings([*json.loads(existing["sources_json"] or "[]"), source])[:32]
                merged_sample_ids = _dedupe_strings(
                    [*json.loads(existing["sample_ids_json"] or "[]"), *sample_ids]
                )[:64]
                merged_modalities = _dedupe_strings(
                    [*json.loads(existing["modalities_json"] or "[]"), *modalities]
                )
                old_payload = json.loads(existing["payload_json"] or "{}")
                old_features = json.loads(existing["features_json"] or "{}")
                self.conn.execute(
                    """
                    UPDATE sensory_prototypes
                    SET updated_at = ?, modalities_json = ?, sources_json = ?,
                        sample_ids_json = ?, summary = ?, centroid_embedding = ?,
                        observation_count = ?, confidence = ?, features_json = ?,
                        payload_json = ?
                    WHERE key = ?
                    """,
                    (
                        now,
                        json.dumps(merged_modalities, sort_keys=True, default=str),
                        json.dumps(sources, sort_keys=True, default=str),
                        json.dumps(merged_sample_ids, sort_keys=True, default=str),
                        summary,
                        self._pack_vector(centroid),
                        new_count,
                        float(max(float(existing["confidence"]), confidence)),
                        json.dumps({**old_features, **features}, sort_keys=True, default=str),
                        json.dumps({**old_payload, **payload}, sort_keys=True, default=str),
                        key,
                    ),
                )
            else:
                prototype_id = str(uuid.uuid4())
                self.conn.execute(
                    """
                    INSERT INTO sensory_prototypes
                        (id, key, timestamp, updated_at, modalities_json, sources_json,
                         sample_ids_json, summary, centroid_embedding, observation_count,
                         confidence, features_json, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prototype_id,
                        key,
                        now,
                        now,
                        json.dumps(_dedupe_strings(modalities), sort_keys=True, default=str),
                        json.dumps(_dedupe_strings([source]), sort_keys=True, default=str),
                        json.dumps(_dedupe_strings(sample_ids), sort_keys=True, default=str),
                        summary,
                        self._pack_vector(self._unit(embedding)),
                        1,
                        float(confidence),
                        json.dumps(features, sort_keys=True, default=str),
                        json.dumps(payload, sort_keys=True, default=str),
                    ),
                )
            self.conn.commit()
        return prototype_id

    def retrieve_sensory_prototypes(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SensoryPrototypeRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM sensory_prototypes ORDER BY confidence DESC, observation_count DESC"
            ).fetchall()
        query = self._unit(embedding)
        prototypes: list[SensoryPrototypeRecord] = []
        for row in rows:
            centroid = self._unpack_vector(row["centroid_embedding"])
            prototypes.append(self._row_to_sensory_prototype(row, centroid, float(np.dot(query, self._unit(centroid)))))
        prototypes.sort(
            key=lambda item: (item.similarity * 0.6 + item.confidence * 0.25 + min(item.observation_count, 10) * 0.015),
            reverse=True,
        )
        return prototypes[:top_k]

    def recent_sensory_prototypes(self, limit: int = 20) -> list[SensoryPrototypeRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM sensory_prototypes ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            self._row_to_sensory_prototype(row, self._unpack_vector(row["centroid_embedding"]), 0.0)
            for row in rows
        ]

    def store_cognitive_trace(
        self,
        trace_id: str,
        episode_id: str,
        input_text: str,
        mode: str,
        prediction: dict[str, Any],
        observation: dict[str, Any],
        attention: list[str],
        surprise: float,
        uncertainty: float,
        payload: dict[str, Any],
    ) -> str:
        """Persist one bounded cognitive workspace trace."""
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO cognitive_traces
                    (id, timestamp, episode_id, input_text, mode, prediction_json,
                     observation_json, attention_json, surprise, uncertainty, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    prediction_json = excluded.prediction_json,
                    observation_json = excluded.observation_json,
                    attention_json = excluded.attention_json,
                    surprise = excluded.surprise,
                    uncertainty = excluded.uncertainty,
                    payload_json = excluded.payload_json
                """,
                (
                    trace_id,
                    time(),
                    episode_id,
                    input_text,
                    mode,
                    json.dumps(prediction, sort_keys=True, default=str),
                    json.dumps(observation, sort_keys=True, default=str),
                    json.dumps(attention, sort_keys=True, default=str),
                    float(surprise),
                    float(uncertainty),
                    json.dumps(payload, sort_keys=True, default=str),
                ),
            )
            self.conn.commit()
        return trace_id

    def recent_cognitive_traces(self, limit: int = 20) -> list[CognitiveTraceRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM cognitive_traces ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_cognitive_trace(row) for row in rows]

    def add_training_file(
        self,
        name: str,
        path: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        status: str = "queued",
        modality: str = "text",
        preview: str = "",
        payload: dict[str, Any] | None = None,
        file_id: str | None = None,
    ) -> str:
        file_id = file_id or str(uuid.uuid4())
        now = time()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO training_files
                    (id, timestamp, updated_at, name, path, content_type, size_bytes,
                     sha256, status, modality, preview, processed_episode_id, error, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?)
                """,
                (
                    file_id,
                    now,
                    now,
                    name,
                    path,
                    content_type,
                    int(size_bytes),
                    sha256,
                    status,
                    modality,
                    preview,
                    json.dumps(payload or {}, sort_keys=True, default=str),
                ),
            )
            self.conn.commit()
        return file_id

    def get_training_file(self, file_id: str) -> TrainingFileRecord | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM training_files WHERE id = ?",
                (file_id,),
            ).fetchone()
        return self._row_to_training_file(row) if row else None

    def list_training_files(
        self,
        status: str | None = None,
        limit: int = 200,
    ) -> list[TrainingFileRecord]:
        with self._lock:
            if status is None:
                rows = self.conn.execute(
                    "SELECT * FROM training_files ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM training_files WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [self._row_to_training_file(row) for row in rows]

    def update_training_file(
        self,
        file_id: str,
        status: str | None = None,
        processed_episode_id: str | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        current = self.get_training_file(file_id)
        if current is None:
            raise KeyError(file_id)
        new_payload = dict(current.payload)
        if payload:
            new_payload.update(payload)
        with self._lock:
            self.conn.execute(
                """
                UPDATE training_files
                SET updated_at = ?,
                    status = ?,
                    processed_episode_id = ?,
                    error = ?,
                    payload_json = ?
                WHERE id = ?
                """,
                (
                    time(),
                    status if status is not None else current.status,
                    processed_episode_id if processed_episode_id is not None else current.processed_episode_id,
                    error if error is not None else current.error,
                    json.dumps(new_payload, sort_keys=True, default=str),
                    file_id,
                ),
            )
            self.conn.commit()

    def training_file_summary(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT status, COUNT(*) AS count FROM training_files GROUP BY status"
            ).fetchall()
        summary = {"queued": 0, "running": 0, "done": 0, "failed": 0, "held": 0}
        for row in rows:
            summary[row["status"]] = int(row["count"])
        summary["total"] = sum(summary.values())
        return summary

    def upsert_expert(
        self,
        name: str,
        kind: str,
        description: str,
        trigger_embedding: np.ndarray,
        status: str = "probation",
        success: bool = True,
        payload: dict[str, Any] | None = None,
        estimated_vram_gb: float = 0.08,
        estimated_ram_gb: float = 0.02,
        hot: bool = False,
    ) -> str:
        payload = payload or {}
        now = time()
        with self._lock:
            existing = self.conn.execute("SELECT id FROM experts WHERE name = ?", (name,)).fetchone()
            if existing:
                expert_id = existing["id"]
                self.conn.execute(
                    """
                    UPDATE experts
                    SET kind = ?, description = ?, status = ?, trigger_embedding = ?,
                        updated_at = ?, success_count = success_count + ?,
                        failure_count = failure_count + ?, utility = max(0.0, min(1.0, utility + ?)),
                        estimated_vram_gb = ?, estimated_ram_gb = ?, hot = ?, payload_json = ?
                    WHERE name = ?
                    """,
                    (
                        kind,
                        description,
                        status,
                        self._pack_vector(trigger_embedding),
                        now,
                        1 if success else 0,
                        0 if success else 1,
                        0.08 if success else -0.08,
                        float(estimated_vram_gb),
                        float(estimated_ram_gb),
                        1 if hot else 0,
                        json.dumps(payload, sort_keys=True, default=str),
                        name,
                    ),
                )
            else:
                expert_id = str(uuid.uuid4())
                self.conn.execute(
                    """
                    INSERT INTO experts
                        (id, name, kind, description, status, trigger_embedding,
                         created_at, updated_at, success_count, failure_count, utility,
                         estimated_vram_gb, estimated_ram_gb, hot, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        expert_id,
                        name,
                        kind,
                        description,
                        status,
                        self._pack_vector(trigger_embedding),
                        now,
                        now,
                        1 if success else 0,
                        0 if success else 1,
                        0.58 + (0.08 if success else -0.08),
                        float(estimated_vram_gb),
                        float(estimated_ram_gb),
                        1 if hot else 0,
                        json.dumps(payload, sort_keys=True, default=str),
                    ),
                )
            self.conn.commit()
        return expert_id

    def record_expert_result(self, expert_id: str, success: bool) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE experts
                SET success_count = success_count + ?,
                    failure_count = failure_count + ?,
                    utility = max(0.0, min(1.0, utility + ?)),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    1 if success else 0,
                    0 if success else 1,
                    0.04 if success else -0.06,
                    time(),
                    expert_id,
                ),
            )
            self.conn.commit()

    def set_expert_residency(self, expert_id: str, hot: bool, status: str | None = None) -> None:
        with self._lock:
            if status is None:
                self.conn.execute(
                    "UPDATE experts SET hot = ?, updated_at = ? WHERE id = ?",
                    (1 if hot else 0, time(), expert_id),
                )
            else:
                self.conn.execute(
                    "UPDATE experts SET hot = ?, status = ?, updated_at = ? WHERE id = ?",
                    (1 if hot else 0, status, time(), expert_id),
                )
            self.conn.commit()

    def list_experts(self, status: str | None = None) -> list[ExpertRecord]:
        with self._lock:
            if status is None:
                rows = self.conn.execute("SELECT * FROM experts ORDER BY utility DESC").fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM experts WHERE status = ? ORDER BY utility DESC",
                    (status,),
                ).fetchall()
        return [self._row_to_expert(row, self._unpack_vector(row["trigger_embedding"]), 0.0) for row in rows]

    def retrieve_experts(
        self,
        embedding: np.ndarray,
        top_k: int = 8,
        statuses: tuple[str, ...] = ("probation", "active", "cold"),
    ) -> list[ExpertRecord]:
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM experts WHERE status IN ({placeholders}) ORDER BY utility DESC",
                statuses,
            ).fetchall()
        query = self._unit(embedding)
        experts = []
        for row in rows:
            emb = self._unpack_vector(row["trigger_embedding"])
            experts.append(self._row_to_expert(row, emb, float(np.dot(query, self._unit(emb)))))
        experts.sort(key=lambda item: (item.similarity * 0.65 + item.utility * 0.35), reverse=True)
        return experts[:top_k]

    def append_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str = "local",
    ) -> SyncEvent:
        now = time()
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO sync_events (timestamp, event_type, source, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    now,
                    event_type,
                    source,
                    json.dumps(payload, sort_keys=True, default=str),
                ),
            )
            self.conn.commit()
            event_id = int(cursor.lastrowid)
        return SyncEvent(
            id=event_id,
            timestamp=now,
            event_type=event_type,
            source=source,
            payload=payload,
        )

    def recent_events(self, limit: int = 50, after_id: int | None = None) -> list[SyncEvent]:
        with self._lock:
            if after_id is None:
                rows = self.conn.execute(
                    "SELECT * FROM sync_events ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = self.conn.execute(
                    "SELECT * FROM sync_events WHERE id > ? ORDER BY id ASC LIMIT ?",
                    (after_id, limit),
                ).fetchall()
        return [
            SyncEvent(
                id=int(row["id"]),
                timestamp=float(row["timestamp"]),
                event_type=row["event_type"],
                source=row["source"],
                payload=json.loads(row["payload_json"] or "{}"),
            )
            for row in rows
        ]

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                text TEXT NOT NULL,
                context_json TEXT NOT NULL,
                embedding BLOB NOT NULL,
                active_circuits_json TEXT NOT NULL,
                salience REAL NOT NULL,
                outcome TEXT,
                feedback_score REAL
            );

            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                trigger_embedding BLOB NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_used REAL NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_stats (
                tool_name TEXT PRIMARY KEY,
                uses INTEGER NOT NULL DEFAULT 0,
                successes INTEGER NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0,
                preference REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                kind TEXT NOT NULL,
                candidate TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                notes TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS learning_gaps (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                kind TEXT NOT NULL,
                description TEXT NOT NULL,
                severity REAL NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_stats (
                source_name TEXT PRIMARY KEY,
                proposals INTEGER NOT NULL DEFAULT 0,
                accepted INTEGER NOT NULL DEFAULT 0,
                rejected INTEGER NOT NULL DEFAULT 0,
                trust REAL NOT NULL DEFAULT 0.5
            );

            CREATE TABLE IF NOT EXISTS sensory_bindings (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                episode_id TEXT NOT NULL,
                source TEXT NOT NULL,
                modalities_json TEXT NOT NULL,
                sample_ids_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                embedding BLOB NOT NULL,
                binding_score REAL NOT NULL,
                salience REAL NOT NULL,
                features_json TEXT NOT NULL,
                context_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sensory_prototypes (
                id TEXT PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,
                timestamp REAL NOT NULL,
                updated_at REAL NOT NULL,
                modalities_json TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                sample_ids_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                centroid_embedding BLOB NOT NULL,
                observation_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                features_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cognitive_traces (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                episode_id TEXT NOT NULL,
                input_text TEXT NOT NULL,
                mode TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                observation_json TEXT NOT NULL,
                attention_json TEXT NOT NULL,
                surprise REAL NOT NULL,
                uncertainty REAL NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_files (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                updated_at REAL NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                modality TEXT NOT NULL,
                preview TEXT NOT NULL,
                processed_episode_id TEXT NOT NULL,
                error TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experts (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                trigger_embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                utility REAL NOT NULL DEFAULT 0.5,
                estimated_vram_gb REAL NOT NULL DEFAULT 0.0,
                estimated_ram_gb REAL NOT NULL DEFAULT 0.0,
                hot INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rules (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                trigger_pattern TEXT NOT NULL,
                preconditions_json TEXT NOT NULL,
                action_tool TEXT NOT NULL,
                expected_outcome TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                trigger_embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                provenance_json TEXT NOT NULL,
                counterexamples_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rule_links (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                updated_at REAL NOT NULL,
                rule_id TEXT NOT NULL,
                target_kind TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                confidence REAL NOT NULL,
                provenance_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(rule_id, target_kind, target_id, relation)
            );

            CREATE TABLE IF NOT EXISTS sync_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
                ON episodes(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_procedures_name
                ON procedures(name);
            CREATE INDEX IF NOT EXISTS idx_sync_events_timestamp
                ON sync_events(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_sync_events_type
                ON sync_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_learning_gaps_timestamp
                ON learning_gaps(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_learning_gaps_status
                ON learning_gaps(status);
            CREATE INDEX IF NOT EXISTS idx_sensory_bindings_timestamp
                ON sensory_bindings(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_sensory_bindings_source
                ON sensory_bindings(source);
            CREATE INDEX IF NOT EXISTS idx_sensory_prototypes_updated_at
                ON sensory_prototypes(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_sensory_prototypes_confidence
                ON sensory_prototypes(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_cognitive_traces_timestamp
                ON cognitive_traces(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_cognitive_traces_episode
                ON cognitive_traces(episode_id);
            CREATE INDEX IF NOT EXISTS idx_training_files_status
                ON training_files(status);
            CREATE INDEX IF NOT EXISTS idx_training_files_updated_at
                ON training_files(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_training_files_sha256
                ON training_files(sha256);
            CREATE INDEX IF NOT EXISTS idx_experts_status
                ON experts(status);
            CREATE INDEX IF NOT EXISTS idx_experts_hot
                ON experts(hot);
            CREATE INDEX IF NOT EXISTS idx_experts_utility
                ON experts(utility DESC);
            CREATE INDEX IF NOT EXISTS idx_rules_status
                ON rules(status);
            CREATE INDEX IF NOT EXISTS idx_rules_confidence
                ON rules(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_rule_links_rule
                ON rule_links(rule_id);
            CREATE INDEX IF NOT EXISTS idx_rule_links_target
                ON rule_links(target_kind, target_id);
            CREATE INDEX IF NOT EXISTS idx_rule_links_updated_at
                ON rule_links(updated_at DESC);
            """
            )
            self.conn.commit()

    def _configure_connection(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA temp_store = MEMORY")
            self.conn.execute("PRAGMA foreign_keys = ON")

    def _pack_vector(self, vector: np.ndarray) -> bytes:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.embedding_dim,):
            raise ValueError(f"expected vector shape {(self.embedding_dim,)}, got {vector.shape}")
        return vector.tobytes()

    def _unpack_vector(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).copy()

    def _row_to_memory(self, row: sqlite3.Row, embedding: np.ndarray, similarity: float) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            timestamp=float(row["timestamp"]),
            text=row["text"],
            context=json.loads(row["context_json"] or "{}"),
            embedding=embedding,
            active_circuits=json.loads(row["active_circuits_json"] or "[]"),
            salience=float(row["salience"]),
            outcome=row["outcome"],
            feedback_score=row["feedback_score"],
            similarity=similarity,
        )

    def _row_to_sensory_binding(
        self,
        row: sqlite3.Row,
        embedding: np.ndarray,
        similarity: float,
    ) -> SensoryBindingRecord:
        return SensoryBindingRecord(
            id=row["id"],
            timestamp=float(row["timestamp"]),
            episode_id=row["episode_id"],
            source=row["source"],
            modalities=json.loads(row["modalities_json"] or "[]"),
            sample_ids=json.loads(row["sample_ids_json"] or "[]"),
            summary=row["summary"],
            embedding=embedding,
            binding_score=float(row["binding_score"]),
            salience=float(row["salience"]),
            features=json.loads(row["features_json"] or "{}"),
            context=json.loads(row["context_json"] or "{}"),
            similarity=similarity,
        )

    def _row_to_sensory_prototype(
        self,
        row: sqlite3.Row,
        embedding: np.ndarray,
        similarity: float,
    ) -> SensoryPrototypeRecord:
        return SensoryPrototypeRecord(
            id=row["id"],
            key=row["key"],
            timestamp=float(row["timestamp"]),
            updated_at=float(row["updated_at"]),
            modalities=json.loads(row["modalities_json"] or "[]"),
            sources=json.loads(row["sources_json"] or "[]"),
            sample_ids=json.loads(row["sample_ids_json"] or "[]"),
            summary=row["summary"],
            centroid_embedding=embedding,
            observation_count=int(row["observation_count"]),
            confidence=float(row["confidence"]),
            features=json.loads(row["features_json"] or "{}"),
            payload=json.loads(row["payload_json"] or "{}"),
            similarity=similarity,
        )

    def _row_to_training_file(self, row: sqlite3.Row) -> TrainingFileRecord:
        return TrainingFileRecord(
            id=row["id"],
            timestamp=float(row["timestamp"]),
            updated_at=float(row["updated_at"]),
            name=row["name"],
            path=row["path"],
            content_type=row["content_type"],
            size_bytes=int(row["size_bytes"]),
            sha256=row["sha256"],
            status=row["status"],
            modality=row["modality"],
            preview=row["preview"],
            processed_episode_id=row["processed_episode_id"],
            error=row["error"],
            payload=json.loads(row["payload_json"] or "{}"),
        )

    def _row_to_cognitive_trace(self, row: sqlite3.Row) -> CognitiveTraceRecord:
        return CognitiveTraceRecord(
            id=row["id"],
            timestamp=float(row["timestamp"]),
            episode_id=row["episode_id"],
            input_text=row["input_text"],
            mode=row["mode"],
            prediction=json.loads(row["prediction_json"] or "{}"),
            observation=json.loads(row["observation_json"] or "{}"),
            attention=json.loads(row["attention_json"] or "[]"),
            surprise=float(row["surprise"]),
            uncertainty=float(row["uncertainty"]),
            payload=json.loads(row["payload_json"] or "{}"),
        )

    def _row_to_rule(self, row: sqlite3.Row, embedding: np.ndarray, similarity: float) -> RuleRecord:
        return RuleRecord(
            id=row["id"],
            name=row["name"],
            trigger_pattern=row["trigger_pattern"],
            preconditions=json.loads(row["preconditions_json"] or "{}"),
            action_tool=row["action_tool"],
            expected_outcome=row["expected_outcome"],
            confidence=float(row["confidence"]),
            status=row["status"],
            trigger_embedding=embedding,
            provenance=json.loads(row["provenance_json"] or "[]"),
            counterexamples=json.loads(row["counterexamples_json"] or "[]"),
            payload=json.loads(row["payload_json"] or "{}"),
            similarity=similarity,
        )

    def _row_to_rule_link(self, row: sqlite3.Row) -> RuleLinkRecord:
        return RuleLinkRecord(
            id=row["id"],
            timestamp=float(row["timestamp"]),
            updated_at=float(row["updated_at"]),
            rule_id=row["rule_id"],
            target_kind=row["target_kind"],
            target_id=row["target_id"],
            relation=row["relation"],
            confidence=float(row["confidence"]),
            provenance=json.loads(row["provenance_json"] or "[]"),
            payload=json.loads(row["payload_json"] or "{}"),
        )

    def _row_to_expert(self, row: sqlite3.Row, embedding: np.ndarray, similarity: float) -> ExpertRecord:
        return ExpertRecord(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            description=row["description"],
            status=row["status"],
            trigger_embedding=embedding,
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            utility=float(row["utility"]),
            estimated_vram_gb=float(row["estimated_vram_gb"]),
            estimated_ram_gb=float(row["estimated_ram_gb"]),
            hot=bool(row["hot"]),
            payload=json.loads(row["payload_json"] or "{}"),
            similarity=similarity,
        )

    def _unit(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
