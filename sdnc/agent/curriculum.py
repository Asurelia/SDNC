"""Guided local training curriculum for SDNC."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np

from sdnc.agent.multimodal import ModalitySample
from sdnc.agent.types import InteractionResult

if TYPE_CHECKING:
    from sdnc.agent.system import InteractionLearningSystem


@dataclass(frozen=True)
class CurriculumStepSpec:
    """One beginner-safe training exercise."""

    id: str
    title: str
    objective: str
    kind: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "objective": self.objective,
            "kind": self.kind,
        }


@dataclass
class CurriculumStepResult:
    """Result for one guided exercise."""

    id: str
    title: str
    kind: str
    passed: bool
    score: float
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "passed": self.passed,
            "score": round(float(self.score), 4),
            "notes": list(self.notes),
            "metrics": self.metrics,
            "interactions": self.interactions,
            "reports": self.reports,
        }


@dataclass
class CurriculumReport:
    """A complete guided training run."""

    timestamp: float
    mode: str
    sleep_preview: bool
    steps: list[CurriculumStepResult]
    before: dict[str, Any]
    after: dict[str, Any]

    @property
    def passed_count(self) -> int:
        return sum(1 for step in self.steps if step.passed)

    @property
    def score(self) -> float:
        if not self.steps:
            return 0.0
        return sum(step.score for step in self.steps) / len(self.steps)

    def summary(self) -> str:
        return (
            f"Guided training: {self.passed_count}/{len(self.steps)} passed, "
            f"score {self.score:.2f}."
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "sleep_preview": self.sleep_preview,
            "summary": self.summary(),
            "passed_count": self.passed_count,
            "step_count": len(self.steps),
            "score": round(float(self.score), 4),
            "before": self.before,
            "after": self.after,
            "steps": [step.to_payload() for step in self.steps],
        }


GUIDED_CURRICULUM: tuple[CurriculumStepSpec, ...] = (
    CurriculumStepSpec(
        id="calculator",
        title="Calcul simple",
        objective="Verify arithmetic routing and positive feedback.",
        kind="tool",
    ),
    CurriculumStepSpec(
        id="memory",
        title="Memoire courte",
        objective="Store a fact, recall it, and reinforce the recall.",
        kind="memory",
    ),
    CurriculumStepSpec(
        id="file_lookup",
        title="Fichier local",
        objective="Create a small local fixture, search it, and read it.",
        kind="tool",
    ),
    CurriculumStepSpec(
        id="sensory",
        title="Signal multimodal",
        objective="Bind repeated text+image signals into a sensory prototype.",
        kind="sensory",
    ),
    CurriculumStepSpec(
        id="rules",
        title="Regles verifiees",
        objective="Repeat useful traces and consolidate local rules.",
        kind="rules",
    ),
    CurriculumStepSpec(
        id="replay",
        title="Replay borne",
        objective="Preview or run bounded sleep replay over salient traces.",
        kind="replay",
    ),
)


def curriculum_manifest() -> dict[str, Any]:
    return {"steps": [step.to_payload() for step in GUIDED_CURRICULUM]}


def run_guided_curriculum(
    system: InteractionLearningSystem,
    step_id: str | None = None,
    mode: str = "think",
    sleep_preview: bool = True,
    batch_size: int = 6,
) -> CurriculumReport:
    selected = _selected_steps(step_id)
    before = _system_snapshot(system)
    results: list[CurriculumStepResult] = []
    for step in selected:
        results.append(_run_step(system, step, mode=mode, sleep_preview=sleep_preview, batch_size=batch_size))
    return CurriculumReport(
        timestamp=time(),
        mode=mode,
        sleep_preview=sleep_preview,
        steps=results,
        before=before,
        after=_system_snapshot(system),
    )


def _selected_steps(step_id: str | None) -> list[CurriculumStepSpec]:
    if not step_id or step_id == "all":
        return list(GUIDED_CURRICULUM)
    for step in GUIDED_CURRICULUM:
        if step.id == step_id:
            return [step]
    raise ValueError(f"unknown curriculum step: {step_id}")


def _run_step(
    system: InteractionLearningSystem,
    step: CurriculumStepSpec,
    mode: str,
    sleep_preview: bool,
    batch_size: int,
) -> CurriculumStepResult:
    if step.id == "calculator":
        return _run_calculator_step(system, step, mode)
    if step.id == "memory":
        return _run_memory_step(system, step, mode)
    if step.id == "file_lookup":
        return _run_file_lookup_step(system, step, mode)
    if step.id == "sensory":
        return _run_sensory_step(system, step, mode)
    if step.id == "rules":
        return _run_rules_step(system, step, mode)
    if step.id == "replay":
        return _run_replay_step(system, step, sleep_preview=sleep_preview, batch_size=batch_size)
    raise ValueError(f"unknown curriculum step: {step.id}")


def _run_calculator_step(
    system: InteractionLearningSystem,
    step: CurriculumStepSpec,
    mode: str,
) -> CurriculumStepResult:
    result = system.interact("SDNCGUIDED calcule 37 + 58", context={"mode": mode, "curriculum": step.id})
    tools = _tool_names(result)
    text = _evidence_text(result)
    notes: list[str] = []
    if "calculator" not in tools:
        notes.append("calculator tool was not selected")
    if "95" not in text:
        notes.append("expected arithmetic result 95 was not found")
    score = _score([not notes, "calculator" in tools, "95" in text])
    _feedback(system, score >= 0.66, step.id)
    return _step_result(
        step,
        passed=score >= 0.66,
        score=score,
        notes=notes,
        interactions=[result],
        metrics={"tools": tools, "expected": "95"},
    )


def _run_memory_step(
    system: InteractionLearningSystem,
    step: CurriculumStepSpec,
    mode: str,
) -> CurriculumStepResult:
    seed = system.interact(
        "SDNCGUIDED note preference stable couleur_preferee bleu_electrique numero_secret 713",
        context={"mode": mode, "curriculum": step.id, "phase": "seed"},
    )
    _feedback(system, True, f"{step.id}:seed")
    recall = system.interact(
        "SDNCGUIDED rappelle preference stable couleur_preferee et numero_secret",
        context={"mode": mode, "curriculum": step.id, "phase": "recall"},
    )
    text = _evidence_text(recall)
    best_memory = max((memory.similarity for memory in recall.memories), default=0.0)
    notes: list[str] = []
    if "bleu_electrique" not in text:
        notes.append("preferred color was not recalled")
    if "713" not in text:
        notes.append("secret number was not recalled")
    if best_memory < 0.55:
        notes.append("memory similarity stayed below 0.55")
    score = _score([not notes, "bleu_electrique" in text, "713" in text, best_memory >= 0.55])
    _feedback(system, score >= 0.66, step.id)
    return _step_result(
        step,
        passed=score >= 0.66,
        score=score,
        notes=notes,
        interactions=[seed, recall],
        metrics={"best_memory_similarity": round(float(best_memory), 4)},
    )


def _run_file_lookup_step(
    system: InteractionLearningSystem,
    step: CurriculumStepSpec,
    mode: str,
) -> CurriculumStepResult:
    fixture = _ensure_fixture(system)
    result = system.interact(
        f"SDNCGUIDED cherche et lis {fixture.as_posix()} MARQUEUR_GUIDED_ALPHA",
        context={"mode": mode, "curriculum": step.id},
    )
    text = _evidence_text(result)
    tools = _tool_names(result)
    notes: list[str] = []
    if "file_search" not in tools:
        notes.append("file_search tool was not selected")
    if "file_read" not in tools:
        notes.append("file_read tool was not selected")
    if "MARQUEUR_GUIDED_ALPHA" not in text:
        notes.append("fixture marker was not read")
    score = _score([not notes, "file_search" in tools, "file_read" in tools, "MARQUEUR_GUIDED_ALPHA" in text])
    _feedback(system, score >= 0.66, step.id)
    return _step_result(
        step,
        passed=score >= 0.66,
        score=score,
        notes=notes,
        interactions=[result],
        metrics={"fixture": str(fixture), "tools": tools},
    )


def _run_sensory_step(
    system: InteractionLearningSystem,
    step: CurriculumStepSpec,
    mode: str,
) -> CurriculumStepResult:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    first = system.observe(
        [
            ModalitySample("text", text="SDNCGUIDED voyant bleu stable", label="guided-blue", source="curriculum", sample_id="guided-a"),
            ModalitySample("image", content=image, text="image bleue stable", label="guided-blue", source="curriculum", sample_id="guided-a"),
        ],
        context={"mode": mode, "curriculum": step.id, "phase": "first"},
        learn=True,
        use_tools=False,
    )
    second = system.observe(
        [
            ModalitySample("text", text="SDNCGUIDED voyant bleu stable", label="guided-blue", source="curriculum", sample_id="guided-b"),
            ModalitySample("image", content=image, text="image bleue stable", label="guided-blue", source="curriculum", sample_id="guided-b"),
        ],
        context={"mode": mode, "curriculum": step.id, "phase": "repeat"},
        learn=True,
        use_tools=False,
    )
    sensory = second.metadata.get("sensory_prototypes", {})
    matches = sensory.get("matches") or []
    learned = sensory.get("learned") or {}
    observation_count = int(learned.get("observation_count") or 0)
    best_score = max((float(match.get("score", 0.0)) for match in matches), default=0.0)
    notes: list[str] = []
    if not matches:
        notes.append("second signal did not match a sensory prototype")
    if observation_count < 2:
        notes.append("prototype observation_count stayed below 2")
    score = _score([not notes, bool(matches), observation_count >= 2, best_score >= 0.75])
    return _step_result(
        step,
        passed=score >= 0.66,
        score=score,
        notes=notes,
        interactions=[first, second],
        metrics={"match_count": len(matches), "observation_count": observation_count, "best_score": round(best_score, 4)},
    )


def _run_rules_step(
    system: InteractionLearningSystem,
    step: CurriculumStepSpec,
    mode: str,
) -> CurriculumStepResult:
    fixture = _ensure_fixture(system)
    interactions: list[InteractionResult] = []
    for suffix in ("alpha", "beta"):
        result = system.interact(
            f"SDNCGUIDED fichierpreuve cherche {fixture.as_posix()} MARQUEUR_GUIDED_ALPHA variante {suffix}",
            context={"mode": mode, "curriculum": step.id, "variant": suffix},
        )
        interactions.append(result)
        _feedback(system, _tool_success(result), f"{step.id}:{suffix}")
    report = system.run_rule_consolidation()
    summary = system.rule_summary()
    notes: list[str] = []
    if report.promoted_count < 1 and summary.get("total", 0) < 1:
        notes.append("no rule was promoted or available after consolidation")
    score = _score([not notes, report.promoted_count >= 1, summary.get("total", 0) >= 1])
    return _step_result(
        step,
        passed=score >= 0.5,
        score=score,
        notes=notes,
        interactions=interactions,
        metrics={"rule_total": summary.get("total", 0), "promoted_count": report.promoted_count, "forked_count": report.forked_count},
        reports={"rule_consolidation": report.to_payload()},
    )


def _run_replay_step(
    system: InteractionLearningSystem,
    step: CurriculumStepSpec,
    sleep_preview: bool,
    batch_size: int,
) -> CurriculumStepResult:
    report = system.run_sleep_cycle(preview=sleep_preview, batch_size=batch_size)
    notes: list[str] = []
    if report.accepted_count < 1:
        notes.append("sleep cycle did not accept any replay action")
    if not sleep_preview and report.replayed_count < 1:
        notes.append("actual replay did not replay any episode")
    score = _score([not notes, report.accepted_count >= 1, sleep_preview or report.replayed_count >= 1])
    return _step_result(
        step,
        passed=score >= 0.5,
        score=score,
        notes=notes,
        metrics={
            "preview": report.preview,
            "accepted_count": report.accepted_count,
            "replayed_count": report.replayed_count,
            "strengthened_count": report.strengthened_count,
            "rejected_count": report.rejected_count,
        },
        reports={"sleep": report.to_payload()},
    )


def _ensure_fixture(system: InteractionLearningSystem) -> Path:
    root = Path(system.config.workspace_root)
    path = root / "data" / "curriculum_sources" / "guided_curriculum_fixture.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Guided SDNC fixture\n"
            "MARQUEUR_GUIDED_ALPHA: SDNC relie fichier, memoire, outils et regles locales.\n",
            encoding="utf-8",
        )
    return path


def _step_result(
    step: CurriculumStepSpec,
    passed: bool,
    score: float,
    notes: list[str],
    interactions: list[InteractionResult] | None = None,
    metrics: dict[str, Any] | None = None,
    reports: dict[str, Any] | None = None,
) -> CurriculumStepResult:
    return CurriculumStepResult(
        id=step.id,
        title=step.title,
        kind=step.kind,
        passed=passed,
        score=max(0.0, min(1.0, float(score))),
        notes=notes,
        metrics=metrics or {},
        interactions=[_interaction_payload(result) for result in interactions or []],
        reports=reports or {},
    )


def _interaction_payload(result: InteractionResult) -> dict[str, Any]:
    metadata = result.metadata or {}
    return {
        "episode_id": result.episode_id,
        "input_text": result.input_text,
        "response": result.response[:900],
        "confidence": round(float(result.activation.confidence), 4),
        "novelty": round(float(result.activation.novelty), 4),
        "active_count": result.activation.active_count,
        "tools": _tool_names(result),
        "successful_tools": [tool.tool_name for tool in result.tool_results if tool.success],
        "best_memory_similarity": round(float(max((memory.similarity for memory in result.memories), default=0.0)), 4),
        "planner_action": (metadata.get("action_plan") or {}).get("selected_action", ""),
        "selected_tools": (metadata.get("action_plan") or {}).get("selected_tools", []),
    }


def _system_snapshot(system: InteractionLearningSystem) -> dict[str, Any]:
    return {
        "memory_count": len(system.recent_memories(limit=200)),
        "rule_summary": system.rule_summary(),
        "rule_conflict_summary": system.rule_conflict_summary(),
        "sensory_prototype_summary": system.sensory_prototype_summary(),
        "expert_summary": system.expert_manager.summary(),
    }


def _tool_names(result: InteractionResult) -> list[str]:
    return [tool.tool_name for tool in result.tool_results]


def _tool_success(result: InteractionResult) -> bool:
    return any(tool.success for tool in result.tool_results)


def _evidence_text(result: InteractionResult) -> str:
    parts = [result.response]
    parts.extend(tool.content for tool in result.tool_results)
    parts.extend(memory.text for memory in result.memories[:5])
    return "\n".join(parts)


def _score(checks: list[bool]) -> float:
    if not checks:
        return 0.0
    return sum(1 for check in checks if check) / len(checks)


def _feedback(system: InteractionLearningSystem, positive: bool, note: str) -> None:
    system.give_feedback(1.0 if positive else -1.0, f"guided curriculum {note}")
