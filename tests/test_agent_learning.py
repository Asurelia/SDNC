import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.learning import (
    LackDetector,
    SelfDirectedLearner,
    SourceProposal,
)
from sdnc.agent.system import InteractionLearningSystem
from sdnc.agent.types import CircuitActivation, InteractionResult, ToolResult


class FixedAdvisor:
    name = "fixed_advisor"

    def propose(self, gap, result, embedding):
        return [
            SourceProposal(
                source=self.name,
                claim="calculator is useful for arithmetic",
                recommendation="Prefer calculator for arithmetic gaps.",
                confidence=0.9,
                evidence="fixture",
                tool_name="calculator",
            )
        ]


def test_lack_detector_flags_low_confidence_and_novelty(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=20,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        allow_web=False,
    )
    result = InteractionResult(
        episode_id="ep",
        timestamp=0.0,
        input_text="question nouvelle",
        response="",
        activation=CircuitActivation([1], [1.0], [0.1], confidence=0.2, novelty=0.9),
        memories=[],
        tool_results=[],
        learned=True,
    )

    signals = LackDetector(config).detect(result)

    assert {signal.kind for signal in signals} >= {"low_confidence", "novelty"}


def test_self_directed_learning_consolidates_verified_tool_pattern(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=20,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        learning_min_sources=1,
    )
    system = InteractionLearningSystem(config)
    try:
        result = InteractionResult(
            episode_id="ep",
            timestamp=0.0,
            input_text="calcule 9 + 1",
            response="",
            activation=CircuitActivation([1], [1.0], [0.1], confidence=0.2, novelty=0.9),
            memories=[],
            tool_results=[ToolResult("calculator", True, "9 + 1 = 10")],
            learned=True,
            metadata={"tool_names": ["calculator"]},
        )
        learner = SelfDirectedLearner(
            config,
            system.memory,
            system.encoder,
            available_tools=system.registry.names(),
            advisors=[FixedAdvisor()],
        )
        report = learner.run(result)
        procedures = system.memory.retrieve_procedures(
            system.encoder.encode("calcule 3 + 4"),
            top_k=5,
        )

        assert report.consolidated >= 1
        assert any(proc.tool_name == "calculator" for proc in procedures)
        assert system.memory.list_source_stats()[0].source_name == "fixed_advisor"
    finally:
        system.close()


def test_system_learn_from_last_gap_records_learning_event(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=20,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        learning_min_sources=1,
    )
    system = InteractionLearningSystem(config)
    try:
        system.interact("calcule 5 + 6")
        report = system.learn_from_last_gap()
        events = system.recent_events(limit=10)

        assert report.summary().startswith("Learning cycle:")
        assert any(event.event_type == "learning" for event in events)
    finally:
        system.close()
