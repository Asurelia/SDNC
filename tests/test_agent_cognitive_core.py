from sdnc.agent.cognitive_core import CognitiveCore
from sdnc.agent.config import AutonomousConfig
from sdnc.agent.types import CircuitActivation


def test_cognitive_core_keeps_bounded_workspace():
    config = AutonomousConfig(
        input_dim=16,
        n_circuits=100,
        cognitive_workspace_slots=4,
        cognitive_attention_focus=2,
        allow_web=False,
    )
    core = CognitiveCore(config)
    activation = CircuitActivation(
        indices=[1, 2, 3, 4, 5],
        weights=[0.9, 0.8, 0.7, 0.6, 0.5],
        scores=[0.95, 0.8, 0.7, 0.6, 0.5],
        confidence=0.42,
        novelty=0.7,
    )

    workspace = core.start(
        input_text="calcule 2 + 2 dans ce test",
        mode="think",
        activation=activation,
        memories=[],
        tool_names=["memory_recall", "calculator"],
        experts=[],
        context_summary="test",
    )
    completed = core.complete(workspace, tool_results=[], salience=0.6)

    assert len(workspace.slots) == 4
    assert len(workspace.attention_focus) == 2
    assert workspace.prediction["predicted_action"] == "use_tools"
    assert 0.0 <= completed.surprise <= 1.0
    assert completed.to_payload()["slot_count"] == 4
