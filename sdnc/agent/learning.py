"""Self-directed learning loop for SDNC.

External tools and models are treated as advisors: they can propose
hypotheses, but local evidence decides what is consolidated.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from time import time
from typing import Protocol

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.encoding import HashingExperienceEncoder
from sdnc.agent.experts import ExpertManager
from sdnc.agent.memory import PersistentMemory
from sdnc.agent.types import InteractionResult, ToolResult


@dataclass(frozen=True)
class LackSignal:
    """A concrete uncertainty or failure SDNC should investigate."""

    kind: str
    description: str
    severity: float
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceProposal:
    """One hypothesis from one source."""

    source: str
    claim: str
    recommendation: str
    confidence: float
    evidence: str = ""
    tool_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsensusCandidate:
    """A grouped proposal with multi-source support."""

    claim: str
    recommendation: str
    proposals: list[SourceProposal]
    score: float
    sources: list[str]
    tool_name: str | None = None


@dataclass(frozen=True)
class ExperimentVerdict:
    """Local sandbox result for one consensus candidate."""

    candidate: ConsensusCandidate
    accepted: bool
    score: float
    reason: str
    metrics: dict[str, object] = field(default_factory=dict)


@dataclass
class LearningCycleReport:
    """Summary of one lack-driven learning cycle."""

    timestamp: float
    gaps: list[LackSignal]
    proposals: list[SourceProposal]
    candidates: list[ConsensusCandidate]
    verdicts: list[ExperimentVerdict]
    consolidated: int

    def summary(self) -> str:
        if not self.gaps:
            return "Learning cycle: no significant gap detected."
        accepted = sum(1 for verdict in self.verdicts if verdict.accepted)
        return (
            f"Learning cycle: {len(self.gaps)} gap(s), "
            f"{len(self.proposals)} proposal(s), {accepted} accepted."
        )


class SourceAdvisor(Protocol):
    """Provider of hypotheses for a detected gap."""

    name: str

    def propose(
        self,
        gap: LackSignal,
        result: InteractionResult,
        embedding: np.ndarray,
    ) -> list[SourceProposal]:
        ...


class LackDetector:
    """Detect when SDNC should stop and learn instead of pretending."""

    def __init__(self, config: AutonomousConfig):
        self.config = config

    def detect(self, result: InteractionResult) -> list[LackSignal]:
        signals: list[LackSignal] = []
        confidence_gap = max(0.0, self.config.confidence_threshold - result.activation.confidence)
        if confidence_gap > 0.0:
            signals.append(
                LackSignal(
                    kind="low_confidence",
                    description="Activation confidence is below the local threshold.",
                    severity=min(1.0, confidence_gap / max(self.config.confidence_threshold, 1e-6)),
                    evidence={"confidence": result.activation.confidence},
                )
            )

        if result.activation.novelty >= self.config.lack_novelty_threshold:
            signals.append(
                LackSignal(
                    kind="novelty",
                    description="The observation is far from known circuit state.",
                    severity=float(result.activation.novelty),
                    evidence={"novelty": result.activation.novelty},
                )
            )

        failed_tools = [tool for tool in result.tool_results if not tool.success]
        if failed_tools:
            signals.append(
                LackSignal(
                    kind="tool_failure",
                    description="One or more selected tools failed.",
                    severity=min(1.0, 0.35 + 0.15 * len(failed_tools)),
                    evidence={"tools": [tool.tool_name for tool in failed_tools]},
                )
            )

        feedback = float(result.metadata.get("feedback_score", 0.0) or 0.0)
        if feedback < 0.0:
            signals.append(
                LackSignal(
                    kind="negative_feedback",
                    description="User feedback says the previous behavior was not useful.",
                    severity=abs(feedback),
                    evidence={"feedback": feedback},
                )
            )

        return [signal for signal in signals if signal.severity >= self.config.lack_min_severity]


class MemoryAdvisor:
    """Use past verified episodes as a source of hypotheses."""

    name = "memory"

    def __init__(self, memory: PersistentMemory, top_k: int = 5):
        self.memory = memory
        self.top_k = top_k

    def propose(
        self,
        gap: LackSignal,
        result: InteractionResult,
        embedding: np.ndarray,
    ) -> list[SourceProposal]:
        proposals: list[SourceProposal] = []
        for record in self.memory.retrieve_similar(embedding, top_k=self.top_k):
            successful_tools = record.context.get("__sdnc", {}).get("successful_tools", [])
            tool_name = successful_tools[0] if successful_tools else None
            confidence = max(0.0, min(1.0, 0.45 + record.similarity * 0.45))
            if record.feedback_score is not None and record.feedback_score > 0:
                confidence = min(1.0, confidence + 0.15)
            proposals.append(
                SourceProposal(
                    source=self.name,
                    claim=f"Similar memory may apply: {record.text[:160]}",
                    recommendation="Reuse the tool/procedure pattern that worked for this similar episode.",
                    confidence=confidence,
                    evidence=f"similarity={record.similarity:.3f} salience={record.salience:.3f}",
                    tool_name=tool_name,
                    metadata={
                        "memory_id": record.id,
                        "similarity": record.similarity,
                        "feedback_score": record.feedback_score,
                    },
                )
            )
        return proposals


class ToolTraceAdvisor:
    """Learn from the immediate tool trace."""

    name = "tool_trace"

    def propose(
        self,
        gap: LackSignal,
        result: InteractionResult,
        embedding: np.ndarray,
    ) -> list[SourceProposal]:
        proposals: list[SourceProposal] = []
        for tool in result.tool_results:
            if tool.success:
                proposals.append(
                    SourceProposal(
                        source=self.name,
                        claim=f"Tool {tool.tool_name} produced usable evidence.",
                        recommendation=f"Prefer {tool.tool_name} for similar gaps.",
                        confidence=0.72,
                        evidence=tool.content[:240],
                        tool_name=tool.tool_name,
                        metadata={"tool_metadata": tool.metadata},
                    )
                )
            else:
                proposals.append(
                    SourceProposal(
                        source=self.name,
                        claim=f"Tool {tool.tool_name} failed in this context.",
                        recommendation=f"Do not consolidate {tool.tool_name} for this pattern yet.",
                        confidence=0.65,
                        evidence=tool.content[:240],
                        tool_name=tool.tool_name,
                        metadata={"failed": True},
                    )
                )
        return proposals


class ExternalCommandAdvisor:
    """Optional adapter for local CLI advisors such as gemini, claude, or codex.

    Commands are exact argv tuples and are never run through a shell. This
    adapter stays disabled unless `allow_external_advisors` is set explicitly.
    """

    name = "external_cli"

    def __init__(self, config: AutonomousConfig):
        self.config = config

    def propose(
        self,
        gap: LackSignal,
        result: InteractionResult,
        embedding: np.ndarray,
    ) -> list[SourceProposal]:
        if not self.config.allow_external_advisors:
            return []
        proposals: list[SourceProposal] = []
        prompt = _advisor_prompt(gap, result)
        for command in self.config.external_advisor_commands:
            if not command:
                continue
            source_name = f"external:{command[0]}"
            try:
                completed = subprocess.run(
                    list(command),
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.config.external_advisor_timeout_s,
                    shell=False,
                    check=False,
                )
                output = (completed.stdout or completed.stderr or "").strip()
                if output:
                    proposals.append(
                        SourceProposal(
                            source=source_name,
                            claim=output[:500],
                            recommendation="Treat this advisor output as a hypothesis to test locally.",
                            confidence=0.45 if completed.returncode else 0.58,
                            evidence=f"returncode={completed.returncode}",
                            metadata={"command": list(command)},
                        )
                    )
            except Exception as exc:
                proposals.append(
                    SourceProposal(
                        source=source_name,
                        claim=f"Advisor command failed: {type(exc).__name__}",
                        recommendation="Do not consolidate this source for the current gap.",
                        confidence=0.1,
                        evidence=str(exc),
                        metadata={"command": list(command), "failed": True},
                    )
                )
        return proposals


class ConsensusEngine:
    """Group proposals and require multi-source support."""

    def __init__(self, config: AutonomousConfig):
        self.config = config

    def build(self, proposals: list[SourceProposal]) -> list[ConsensusCandidate]:
        groups: dict[str, list[SourceProposal]] = {}
        for proposal in proposals:
            if proposal.metadata.get("failed"):
                continue
            key = self._key(proposal)
            groups.setdefault(key, []).append(proposal)

        candidates: list[ConsensusCandidate] = []
        for group in groups.values():
            sources = sorted({proposal.source for proposal in group})
            score = min(1.0, sum(proposal.confidence for proposal in group) / max(1, len(group)))
            if len(sources) < self.config.learning_min_sources and score < 0.82:
                continue
            if score < self.config.learning_min_consensus_score:
                continue
            best = max(group, key=lambda item: item.confidence)
            candidates.append(
                ConsensusCandidate(
                    claim=best.claim,
                    recommendation=best.recommendation,
                    proposals=group,
                    score=score,
                    sources=sources,
                    tool_name=best.tool_name,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _key(self, proposal: SourceProposal) -> str:
        if proposal.tool_name:
            return f"tool:{proposal.tool_name}"
        tokens = re.findall(r"[a-z0-9_]{4,}", proposal.claim.lower())
        return "text:" + "-".join(sorted(set(tokens))[:8])


class ExperimentSandbox:
    """Verify candidates with local, non-destructive evidence."""

    def __init__(self, config: AutonomousConfig, available_tools: list[str]):
        self.config = config
        self.available_tools = set(available_tools)

    def test(self, candidate: ConsensusCandidate, result: InteractionResult) -> ExperimentVerdict:
        metrics: dict[str, object] = {
            "consensus_score": candidate.score,
            "sources": candidate.sources,
        }
        score = candidate.score
        reasons = []

        if candidate.tool_name:
            tool_available = candidate.tool_name in self.available_tools
            tool_succeeded = any(
                tool.tool_name == candidate.tool_name and tool.success
                for tool in result.tool_results
            )
            metrics["tool_available"] = tool_available
            metrics["tool_succeeded_currently"] = tool_succeeded
            if tool_available:
                score += 0.12
                reasons.append("tool exists locally")
            if tool_succeeded:
                score += 0.18
                reasons.append("tool already succeeded in this trace")

        if len(candidate.sources) >= self.config.learning_min_sources:
            score += 0.08
            reasons.append("multi-source support")

        score = min(1.0, score)
        accepted = score >= self.config.learning_min_experiment_score
        return ExperimentVerdict(
            candidate=candidate,
            accepted=accepted,
            score=score,
            reason=", ".join(reasons) if reasons else "insufficient local evidence",
            metrics=metrics,
        )


class Consolidator:
    """Turn accepted local evidence into compact procedural memory."""

    def __init__(
        self,
        config: AutonomousConfig,
        memory: PersistentMemory,
        encoder: HashingExperienceEncoder,
        expert_manager: ExpertManager | None = None,
    ):
        self.config = config
        self.memory = memory
        self.encoder = encoder
        self.expert_manager = expert_manager or ExpertManager(config, memory)

    def consolidate(
        self,
        gap: LackSignal,
        result: InteractionResult,
        embedding: np.ndarray,
        verdict: ExperimentVerdict,
        gap_id: str,
    ) -> None:
        candidate = verdict.candidate
        payload = {
            "gap_id": gap_id,
            "gap_kind": gap.kind,
            "sources": candidate.sources,
            "claim": candidate.claim,
            "recommendation": candidate.recommendation,
            "experiment_score": verdict.score,
        }
        if candidate.tool_name and verdict.accepted:
            self.memory.upsert_procedure(
                name=f"learned:{candidate.tool_name}:{_procedure_key(result.input_text)}",
                description=candidate.recommendation[:240],
                tool_name=candidate.tool_name,
                trigger_embedding=embedding,
                success=True,
                payload=payload,
            )
            self.expert_manager.create_from_learning(
                name=f"expert:{candidate.tool_name}:{_procedure_key(result.input_text)}",
                kind="procedure",
                description=candidate.recommendation,
                trigger_embedding=embedding,
                payload={**payload, "tool_name": candidate.tool_name},
                success=True,
            )
        self.memory.record_experiment(
            kind="self_directed_learning",
            candidate=candidate.claim,
            accepted=verdict.accepted,
            metrics={**verdict.metrics, "score": verdict.score},
            notes=verdict.reason,
        )
        for proposal in candidate.proposals:
            self.memory.record_source_feedback(proposal.source, verdict.accepted)
        self.memory.update_learning_gap_status(gap_id, "consolidated" if verdict.accepted else "tested")


class SelfDirectedLearner:
    """Run the full gap -> sources -> consensus -> sandbox -> memory loop."""

    def __init__(
        self,
        config: AutonomousConfig,
        memory: PersistentMemory,
        encoder: HashingExperienceEncoder,
        available_tools: list[str],
        expert_manager: ExpertManager | None = None,
        advisors: list[SourceAdvisor] | None = None,
    ):
        self.config = config
        self.memory = memory
        self.encoder = encoder
        self.detector = LackDetector(config)
        self.advisors = advisors or [
            MemoryAdvisor(memory, top_k=config.memory_top_k),
            ToolTraceAdvisor(),
            ExternalCommandAdvisor(config),
        ]
        self.consensus = ConsensusEngine(config)
        self.sandbox = ExperimentSandbox(config, available_tools)
        self.consolidator = Consolidator(config, memory, encoder, expert_manager=expert_manager)

    def run(self, result: InteractionResult) -> LearningCycleReport:
        embedding = self.encoder.encode(
            result.input_text,
            {
                "learning_cycle": True,
                "tools": result.metadata.get("tool_names", []),
                "episode_id": result.episode_id,
            },
        )
        gaps = self.detector.detect(result)
        proposals: list[SourceProposal] = []
        verdicts: list[ExperimentVerdict] = []
        candidates: list[ConsensusCandidate] = []
        consolidated = 0

        for gap in gaps:
            gap_id = self.memory.record_learning_gap(
                kind=gap.kind,
                description=gap.description,
                severity=gap.severity,
                payload={"evidence": gap.evidence, "episode_id": result.episode_id},
            )
            gap_proposals: list[SourceProposal] = []
            for advisor in self.advisors:
                gap_proposals.extend(advisor.propose(gap, result, embedding))
            proposals.extend(gap_proposals)
            gap_candidates = self.consensus.build(gap_proposals)
            candidates.extend(gap_candidates)
            for candidate in gap_candidates:
                verdict = self.sandbox.test(candidate, result)
                verdicts.append(verdict)
                self.consolidator.consolidate(gap, result, embedding, verdict, gap_id)
                if verdict.accepted:
                    consolidated += 1

        return LearningCycleReport(
            timestamp=time(),
            gaps=gaps,
            proposals=proposals,
            candidates=candidates,
            verdicts=verdicts,
            consolidated=consolidated,
        )


def _advisor_prompt(gap: LackSignal, result: InteractionResult) -> str:
    return "\n".join(
        [
            "SDNC needs a hypothesis, not a final truth.",
            f"Gap: {gap.kind} severity={gap.severity:.3f}",
            f"Description: {gap.description}",
            f"Input: {result.input_text}",
            "Return one concise proposal and how to test it locally.",
        ]
    )


def _procedure_key(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
    return "-".join(words[:6]) or "general"
