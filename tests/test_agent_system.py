from sdnc.agent.config import AutonomousConfig
from sdnc.agent.system import InteractionLearningSystem


def test_interaction_system_runs_without_external_model(tmp_path):
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=40,
        max_active_ratio=0.05,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        result = system.interact("calcule 2 + 2 et memorise le contexte")

        assert result.activation.active_count <= 2
        assert "calculator" in result.metadata["tool_names"]
        assert result.metadata["cognitive_core"]["prediction"]["predicted_action"] == "use_tools"
        assert result.metadata["cognitive_core"]["slot_count"] <= config.cognitive_workspace_slots
        assert result.learned
        assert (tmp_path / "state.npz").exists()
        assert system.recent_cognitive_traces(limit=1)[0].id == result.metadata["cognitive_core"]["id"]
    finally:
        system.close()


def test_feedback_creates_procedural_tool_memory(tmp_path):
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        system.interact("dans le projet cherche le fichier architecture")
        system.give_feedback(1.0, "file_search etait utile")
        procedures = system.memory.retrieve_procedures(
            system.encoder.encode("cherche fichier architecture"),
            top_k=3,
        )

        assert any(proc.tool_name == "file_search" for proc in procedures)
    finally:
        system.close()
