"""Evaluation harness for the SDNC interaction learner.

The goal is not to benchmark against LLM leaderboards. This module measures the
properties SDNC claims to care about: local learning over repeated interaction,
tool routing, memory reuse, bounded sparsity, surprise, and resource estimates.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.multimodal import ModalitySample
from sdnc.agent.system import InteractionLearningSystem
from sdnc.agent.types import InteractionResult


@dataclass(frozen=True)
class EvaluationCase:
    """One repeated SDNC interaction probe."""

    name: str
    prompt: str
    repeat_prompt: str | None = None
    expected_tools: tuple[str, ...] = ()
    expected_substrings: tuple[str, ...] = ()
    require_memory_hit: bool = False
    feedback_score: float | None = None
    feedback_text: str = ""


@dataclass
class CaseRun:
    """Metrics for one case pass."""

    case_name: str
    pass_name: str
    prompt: str
    success: bool
    latency_ms: float
    confidence: float
    novelty: float
    surprise: float
    uncertainty: float
    memory_hits: int
    tools: list[str]
    planner_action: str
    planner_score: float
    active_count: int
    max_active_ratio: float
    hot_vram_gb: float
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "pass_name": self.pass_name,
            "prompt": self.prompt,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 3),
            "confidence": round(self.confidence, 4),
            "novelty": round(self.novelty, 4),
            "surprise": round(self.surprise, 4),
            "uncertainty": round(self.uncertainty, 4),
            "memory_hits": self.memory_hits,
            "tools": self.tools,
            "planner_action": self.planner_action,
            "planner_score": round(self.planner_score, 4),
            "active_count": self.active_count,
            "max_active_ratio": round(self.max_active_ratio, 4),
            "hot_vram_gb": round(self.hot_vram_gb, 4),
            "notes": list(self.notes),
        }


@dataclass
class FeatureProbeRun:
    """Metrics for a component-level learning probe."""

    probe_name: str
    success: bool
    metrics: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "probe_name": self.probe_name,
            "success": self.success,
            "metrics": self.metrics,
            "notes": list(self.notes),
        }


@dataclass
class EvaluationReport:
    """Full benchmark report."""

    timestamp: float
    case_runs: list[CaseRun]
    summary: dict[str, Any]
    feature_probes: list[FeatureProbeRun] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": self.summary,
            "case_runs": [run.to_payload() for run in self.case_runs],
            "feature_probes": [probe.to_payload() for probe in self.feature_probes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2, sort_keys=True, ensure_ascii=False)


DEFAULT_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        name="arithmetic_tool_routing",
        prompt="calcule 12 + 30",
        expected_tools=("calculator",),
        expected_substrings=("42",),
    ),
    EvaluationCase(
        name="file_search_routing",
        prompt="dans le projet cherche docs/ARCHITECTURE.md",
        expected_tools=("file_search",),
        expected_substrings=("ARCHITECTURE.md",),
    ),
    EvaluationCase(
        name="repeated_memory",
        prompt="note pour eval sdnc: preference utilisateur = bleu electrique",
        repeat_prompt="note pour eval sdnc: preference utilisateur = bleu electrique",
        expected_tools=("memory_recall",),
        expected_substrings=("bleu electrique",),
        require_memory_hit=True,
    ),
)


def run_default_evaluation(
    config: AutonomousConfig | None = None,
    cases: tuple[EvaluationCase, ...] = DEFAULT_CASES,
    reset: bool = False,
) -> EvaluationReport:
    """Run the default deterministic SDNC learning evaluation."""
    if config is None:
        with tempfile.TemporaryDirectory(prefix="sdnc_eval_") as tmp:
            root = Path(tmp)
            eval_config = AutonomousConfig(
                memory_path=root / "eval_memory.sqlite3",
                state_path=root / "eval_circuits.npz",
                workspace_root=root,
                allow_web=False,
                auto_improve_enabled=False,
            )
            _prepare_workspace(eval_config.workspace_root)
            return run_evaluation(eval_config, cases=cases, reset=False)
    _prepare_workspace(config.workspace_root)
    return run_evaluation(config, cases=cases, reset=reset)


def run_evaluation(
    config: AutonomousConfig,
    cases: tuple[EvaluationCase, ...] = DEFAULT_CASES,
    reset: bool = False,
) -> EvaluationReport:
    """Run train/repeat passes and summarize learning-sensitive metrics."""
    if reset:
        _remove_runtime_state(config.memory_path)
        _remove_runtime_state(config.state_path)
    system = InteractionLearningSystem(config)
    runs: list[CaseRun] = []
    try:
        for case in cases:
            first = _run_case(system, case, pass_name="initial", prompt=case.prompt)
            runs.append(first)
            if case.feedback_score is not None:
                system.give_feedback(case.feedback_score, case.feedback_text)
            repeat_prompt = case.repeat_prompt or case.prompt
            second = _run_case(system, case, pass_name="repeat", prompt=repeat_prompt)
            runs.append(second)
        feature_probes = _run_feature_probes(system)
    finally:
        system.close()
    return EvaluationReport(
        timestamp=time.time(),
        case_runs=runs,
        feature_probes=feature_probes,
        summary=_summary(runs, feature_probes),
    )


def _run_case(
    system: InteractionLearningSystem,
    case: EvaluationCase,
    pass_name: str,
    prompt: str,
) -> CaseRun:
    started = time.perf_counter()
    result = system.interact(prompt, context={"mode": "think", "eval_case": case.name})
    latency_ms = (time.perf_counter() - started) * 1000.0
    success, notes = _score_case(result, case)
    cognitive = result.metadata.get("cognitive_core", {})
    action_plan = result.metadata.get("action_plan", {})
    resource = result.metadata.get("resource_budget", {})
    max_active_ratio = result.activation.active_count / max(system.config.n_circuits, 1)
    return CaseRun(
        case_name=case.name,
        pass_name=pass_name,
        prompt=prompt,
        success=success,
        latency_ms=latency_ms,
        confidence=result.activation.confidence,
        novelty=result.activation.novelty,
        surprise=float(cognitive.get("surprise", 0.0)),
        uncertainty=float(cognitive.get("uncertainty", 0.0)),
        memory_hits=sum(1 for memory in result.memories if memory.similarity >= 0.55),
        tools=[tool.tool_name for tool in result.tool_results],
        planner_action=str(action_plan.get("selected_action", "")),
        planner_score=float(action_plan.get("selected_score", 0.0)),
        active_count=result.activation.active_count,
        max_active_ratio=max_active_ratio,
        hot_vram_gb=float(resource.get("hot_vram_gb", 0.0)),
        notes=notes,
    )


def _score_case(result: InteractionResult, case: EvaluationCase) -> tuple[bool, list[str]]:
    notes: list[str] = []
    tools = {tool.tool_name for tool in result.tool_results}
    action_plan = result.metadata.get("action_plan", {})
    planner_action = str(action_plan.get("selected_action", ""))
    content = "\n".join([result.response, *(tool.content for tool in result.tool_results)]).lower()

    for expected in case.expected_tools:
        if expected not in tools:
            memory_substitute = (
                planner_action == "recall_memory"
                and any(memory.similarity >= 0.55 for memory in result.memories)
            )
            if not memory_substitute:
                notes.append(f"missing tool {expected}")
    for substring in case.expected_substrings:
        if substring.lower() not in content:
            notes.append(f"missing text {substring}")
    if case.require_memory_hit and not any(memory.similarity >= 0.55 for memory in result.memories):
        notes.append("missing memory hit")
    n_circuits = max(int(result.metadata.get("n_circuits", 1)), 1)
    sparse_ratio = result.activation.active_count / n_circuits
    if sparse_ratio > 0.05:
        notes.append(f"sparse activation exceeded 5 percent: {sparse_ratio:.4f}")
    return not notes, notes


def _run_feature_probes(system: InteractionLearningSystem) -> list[FeatureProbeRun]:
    return [
        _probe_sensory_prototype_recall(system),
        _probe_rule_consolidation(system),
        _probe_sleep_replay(system),
    ]


def _probe_sensory_prototype_recall(system: InteractionLearningSystem) -> FeatureProbeRun:
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    first = system.observe(
        [
            ModalitySample("text", text="eval voyant bleu stable", label="eval-ui", source="eval", sample_id="proto-a"),
            ModalitySample("image", content=image, text="voyant bleu", label="eval-ui", source="eval", sample_id="proto-a"),
        ],
        context={"mode": "think", "eval_probe": "sensory_prototype"},
        learn=True,
    )
    second = system.observe(
        [
            ModalitySample("text", text="eval voyant bleu stable", label="eval-ui", source="eval", sample_id="proto-b"),
            ModalitySample("image", content=image, text="voyant bleu", label="eval-ui", source="eval", sample_id="proto-b"),
        ],
        context={"mode": "think", "eval_probe": "sensory_prototype"},
        learn=True,
    )
    prototypes = system.recent_sensory_prototypes(limit=10)
    matches = second.metadata.get("sensory_prototypes", {}).get("matches", [])
    learned = second.metadata.get("sensory_prototypes", {}).get("learned") or {}
    notes: list[str] = []
    if not matches:
        notes.append("second observation did not match a sensory prototype")
    if learned.get("observation_count", 0) < 2:
        notes.append("prototype observation count did not consolidate")
    return FeatureProbeRun(
        probe_name="sensory_prototype_recall",
        success=not notes,
        metrics={
            "prototype_count": len(prototypes),
            "first_episode_id": first.episode_id,
            "second_episode_id": second.episode_id,
            "match_count": len(matches),
            "learned_key": learned.get("key", ""),
            "learned_observation_count": learned.get("observation_count", 0),
        },
        notes=notes,
    )


def _probe_rule_consolidation(system: InteractionLearningSystem) -> FeatureProbeRun:
    embedding = system.encoder.encode("eval rule file search stable")
    for index in range(2):
        system.memory.store_episode(
            text="eval rule file search stable",
            context={
                "__sdnc": {
                    "tool_names": ["memory_recall", "file_search"],
                    "proposed_tool_names": ["memory_recall", "file_search"],
                    "successful_tools": ["file_search"],
                }
            },
            embedding=embedding,
            active_circuits=[0],
            salience=0.82,
            feedback_score=1.0,
        )
    report = system.run_rule_consolidation()
    matches = system.rule_engine.match(embedding)
    notes: list[str] = []
    if report.promoted_count < 1:
        notes.append("no rule was promoted")
    if not any(match.tool_name == "file_search" for match in matches):
        notes.append("promoted rule did not match its trigger")
    return FeatureProbeRun(
        probe_name="rule_consolidation",
        success=not notes,
        metrics={
            "promoted_count": report.promoted_count,
            "weakened_count": report.weakened_count,
            "match_count": len(matches),
            "matched_tools": [match.tool_name for match in matches],
            "rule_total": system.rule_summary()["total"],
        },
        notes=notes,
    )


def _probe_sleep_replay(system: InteractionLearningSystem) -> FeatureProbeRun:
    embedding = system.encoder.encode("eval sleep calculator stable")
    for index in range(3):
        system.memory.store_episode(
            text="eval sleep calculator stable",
            context={
                "__sdnc": {
                    "tool_names": ["memory_recall", "calculator"],
                    "proposed_tool_names": ["memory_recall", "calculator"],
                    "successful_tools": ["calculator"],
                }
            },
            embedding=embedding,
            active_circuits=[0],
            salience=0.78 + index * 0.01,
            feedback_score=1.0,
        )
    report = system.run_sleep_cycle(batch_size=8)
    notes: list[str] = []
    if report.replayed_count < 1:
        notes.append("sleep replay did not replay any episode")
    if report.strengthened_count < 1:
        notes.append("sleep replay did not strengthen a procedure")
    return FeatureProbeRun(
        probe_name="sleep_replay_consolidation",
        success=not notes,
        metrics={
            "replayed_count": report.replayed_count,
            "strengthened_count": report.strengthened_count,
            "rejected_count": report.rejected_count,
        },
        notes=notes,
    )


def _summary(runs: list[CaseRun], feature_probes: list[FeatureProbeRun]) -> dict[str, Any]:
    initial = [run for run in runs if run.pass_name == "initial"]
    repeat = [run for run in runs if run.pass_name == "repeat"]
    all_success = sum(1 for run in runs if run.success)
    initial_success = _rate(run.success for run in initial)
    repeat_success = _rate(run.success for run in repeat)
    initial_surprise = _mean(run.surprise for run in initial)
    repeat_surprise = _mean(run.surprise for run in repeat)
    initial_memory_hits = _mean(run.memory_hits for run in initial)
    repeat_memory_hits = _mean(run.memory_hits for run in repeat)
    max_sparse_ratio = max((run.max_active_ratio for run in runs), default=0.0)
    planner_actions = {}
    for run in runs:
        planner_actions[run.planner_action] = planner_actions.get(run.planner_action, 0) + 1
    return {
        "total_runs": len(runs),
        "success_rate": round(all_success / max(len(runs), 1), 4),
        "initial_success_rate": round(initial_success, 4),
        "repeat_success_rate": round(repeat_success, 4),
        "avg_initial_surprise": round(initial_surprise, 4),
        "avg_repeat_surprise": round(repeat_surprise, 4),
        "surprise_delta": round(repeat_surprise - initial_surprise, 4),
        "avg_initial_memory_hits": round(initial_memory_hits, 4),
        "avg_repeat_memory_hits": round(repeat_memory_hits, 4),
        "memory_hit_delta": round(repeat_memory_hits - initial_memory_hits, 4),
        "max_active_ratio": round(max_sparse_ratio, 4),
        "within_sparse_limit": max_sparse_ratio <= 0.05,
        "avg_latency_ms": round(_mean(run.latency_ms for run in runs), 3),
        "avg_hot_vram_gb": round(_mean(run.hot_vram_gb for run in runs), 4),
        "planner_actions": planner_actions,
        "feature_probe_count": len(feature_probes),
        "feature_success_rate": round(_rate(probe.success for probe in feature_probes), 4),
        "feature_probes": {
            probe.probe_name: {
                "success": probe.success,
                "metrics": probe.metrics,
                "notes": probe.notes,
            }
            for probe in feature_probes
        },
    }


def _prepare_workspace(workspace_root: Path) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    note = workspace_root / "docs" / "ARCHITECTURE.md"
    if not note.exists():
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            "ARCHITECTURE.md: SDNC evaluation file for local file_search routing.\n",
            encoding="utf-8",
        )


def _remove_runtime_state(path: Path) -> None:
    path = Path(path)
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists() and candidate.is_file():
            candidate.unlink()


def _rate(values) -> float:
    values = list(values)
    return sum(1 for value in values if value) / max(len(values), 1)


def _mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic SDNC learning evaluation.")
    parser.add_argument("--memory", type=Path, default=Path("data/eval_memory.sqlite3"))
    parser.add_argument("--state", type=Path, default=Path("data/eval_circuits.npz"))
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--reset", action="store_true", help="Delete only the selected eval memory/state files first.")
    parser.add_argument("--allow-web", action="store_true", help="Allow web tool during eval. Disabled by default.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AutonomousConfig(
        memory_path=args.memory,
        state_path=args.state,
        workspace_root=args.workspace,
        allow_web=args.allow_web,
        auto_improve_enabled=False,
    )
    report = run_default_evaluation(config=config, reset=args.reset)
    rendered = report.to_json()
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report.summary.get("within_sparse_limit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
