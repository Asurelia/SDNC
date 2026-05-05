from sdnc.agent.cognitive_core import CognitiveCore
from sdnc.agent.config import AutonomousConfig
from sdnc.agent.planner import ActionPlanner
from sdnc.agent.types import CircuitActivation


def test_planner_prefers_relevant_tool_when_it_reduces_uncertainty():
    config = AutonomousConfig(input_dim=16, n_circuits=100, allow_web=False)
    workspace = CognitiveCore(config).start(
        input_text="calcule 12 + 30",
        mode="think",
        activation=CircuitActivation([1, 2], [0.8, 0.7], [0.8, 0.7], confidence=0.45, novelty=0.65),
        memories=[],
        tool_names=["memory_recall", "calculator"],
        experts=[],
    )

    plan = ActionPlanner(config).plan(
        workspace,
        available_tools=["memory_recall", "calculator"],
        memory_count=0,
        hot_expert_count=0,
    )

    assert plan.selected.action == "use_tools"
    assert "calculator" in plan.selected.tool_names
    assert plan.selected.expected_free_energy < plan.candidates[-1].expected_free_energy


def test_planner_asks_for_feedback_when_tools_are_not_worth_the_cost():
    config = AutonomousConfig(input_dim=16, n_circuits=100, allow_web=False)
    workspace = CognitiveCore(config).start(
        input_text="explique ce point ambigu sans source disponible",
        mode="think",
        activation=CircuitActivation([1], [0.3], [0.3], confidence=0.18, novelty=0.9),
        memories=[],
        tool_names=["memory_recall"],
        experts=[],
    )

    plan = ActionPlanner(config).plan(
        workspace,
        available_tools=["memory_recall"],
        memory_count=0,
        hot_expert_count=0,
    )

    assert plan.selected.action in {"ask_feedback", "investigate_gap"}
    assert plan.selected.score > 0
