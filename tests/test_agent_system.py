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
        assert result.metadata["action_plan"]["selected_action"] == "use_tools"
        assert result.metadata["action_plan"]["selected_tools"] == ["calculator"]
        assert result.metadata["cognitive_core"]["slot_count"] <= config.cognitive_workspace_slots
        assert result.learned
        assert (tmp_path / "state.npz").exists()
        trace = system.recent_cognitive_traces(limit=1)[0]
        assert trace.id == result.metadata["cognitive_core"]["id"]
        assert trace.payload["action_plan"]["selected_action"] == "use_tools"
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


def test_direct_file_read_keeps_priority_under_tool_budget(tmp_path):
    fixture = tmp_path / "guided_fixture.md"
    fixture.write_text("MARQUEUR_LOCAL_TOOL_BUDGET: contenu lu par SDNC.", encoding="utf-8")
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=True,
        max_tool_calls=3,
    )
    system = InteractionLearningSystem(config)
    try:
        result = system.interact(
            "cherche et lis guided_fixture.md MARQUEUR_LOCAL_TOOL_BUDGET",
            context={"mode": "think"},
        )
        tools = [tool.tool_name for tool in result.tool_results]
        text = "\n".join(tool.content for tool in result.tool_results)

        assert "file_read" in tools
        assert "file_search" in tools
        assert "web_search" not in tools
        assert "MARQUEUR_LOCAL_TOOL_BUDGET" in text
    finally:
        system.close()


def test_open_mode_observes_all_local_tool_proposals(tmp_path):
    fixture = tmp_path / "guided_fixture.md"
    fixture.write_text("MARQUEUR_OPEN_MODE: lecture ouverte.", encoding="utf-8")
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        max_tool_calls=1,
    )
    system = InteractionLearningSystem(config)
    try:
        result = system.interact(
            "cherche et lis guided_fixture.md MARQUEUR_OPEN_MODE puis calcule 1 + 2",
            context={"mode": "open"},
        )
        tools = [tool.tool_name for tool in result.tool_results]
        introspection = result.metadata["introspection"]

        assert result.metadata["cognitive_budget"]["mode"] == "open"
        assert result.metadata["cognitive_budget"]["max_tool_calls"] is None
        assert {"memory_recall", "file_read", "file_search", "calculator"}.issubset(set(tools))
        assert introspection["open_laboratory"] is True
        assert introspection["skipped_tools"] == []
    finally:
        system.close()
