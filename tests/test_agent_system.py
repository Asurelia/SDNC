import pytest

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


def test_interaction_uses_learned_conversation_example(tmp_path):
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
        system.memory.upsert_conversation_example(
            source="fixture",
            prompt="bonjour",
            response="salut, je t'écoute",
            embedding=system.encoder.encode("bonjour"),
            confidence=0.9,
        )

        result = system.interact("bonjour", learn=False)

        assert result.response == "salut, je t'écoute"
        assert result.metadata["conversation_examples"][0]["response"] == "salut, je t'écoute"
    finally:
        system.close()


def test_teach_response_learns_direct_conversation_example(tmp_path):
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
        taught = system.teach_response(
            "mot de passe atelier",
            "réponse apprise localement",
            source="test_teach",
        )

        result = system.interact("mot de passe atelier", learn=False)

        assert taught.metadata["conversation_teach"]["source"] == "test_teach"
        assert result.response == "réponse apprise localement"
        assert result.metadata["conversation_decision"]["accepted"] is True
        assert result.metadata["conversation_decision"]["selected_id"] == taught.metadata["conversation_teach"]["example_id"]
    finally:
        system.close()


def test_feedback_updates_selected_conversation_example(tmp_path):
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
        taught = system.teach_response(
            "salutation spéciale",
            "réponse initiale",
            source="test_teach",
        )
        result = system.interact("salutation spéciale", learn=False)
        feedback = system.give_feedback(-1.0, "mauvaise réponse")
        matches = system.memory.retrieve_conversation_examples(
            system.encoder.encode("salutation spéciale"),
            top_k=1,
        )

        assert result.metadata["conversation_decision"]["selected_id"] == taught.metadata["conversation_teach"]["example_id"]
        assert feedback.metadata["conversation_feedback"]["updated"] is True
        assert matches[0].failure_count >= 1
    finally:
        system.close()


def test_interaction_rejects_incompatible_conversation_example(tmp_path):
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
        system.memory.upsert_conversation_example(
            source="fixture",
            prompt='Convertis la phrase "Bonjour, comment vas-tu ?" en espagnol.',
            response="Hola, como estas?",
            embedding=system.encoder.encode('Convertis la phrase "Bonjour, comment vas-tu ?" en espagnol.'),
            confidence=0.9,
        )

        result = system.interact("Bonjour, comment vas-tu ?", learn=False)

        assert result.response != "Hola, como estas?"
        assert result.response.startswith("Salut")
    finally:
        system.close()


def test_interaction_prefers_calculator_over_nearby_math_example(tmp_path):
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
        system.memory.upsert_conversation_example(
            source="fixture",
            prompt="Si j'ai 3 pommes et j'en mange 2, combien m'en reste-t-il ?",
            response="Il te reste 1 pomme.",
            embedding=system.encoder.encode("Si j'ai 3 pommes et j'en mange 2, combien m'en reste-t-il ?"),
            confidence=0.9,
        )

        result = system.interact("Si j'ai 3 pommes et que j'en donne 1, combien il m'en reste ?", learn=False)

        assert "calculator" in result.metadata["tool_names"]
        assert result.response == "3 - 1 = 2"
    finally:
        system.close()


def test_interaction_forces_calculator_when_memory_is_confident(tmp_path):
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    prompt = "Si j'ai 3 pommes et que j'en donne 1, combien il m'en reste ?"
    try:
        system.memory.store_episode(
            text="ancienne réponse proche mais non fiable",
            context={"kind": "near_math_memory"},
            embedding=system.encoder.encode(prompt),
            active_circuits=[1],
            salience=0.9,
        )

        result = system.interact(prompt, learn=False)

        assert "calculator" in result.metadata["tool_names"]
        assert result.response == "3 - 1 = 2"
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


def test_parquet_training_file_ingests_dataset_rows(tmp_path):
    pl = pytest.importorskip("polars")
    parquet_path = tmp_path / "pairs.parquet"
    pl.DataFrame(
        {
            "source": ["hello", "good night", "blue house", "open door"],
            "target": ["bonjour", "bonne nuit", "maison bleue", "porte ouverte"],
        }
    ).write_parquet(parquet_path)
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        training_dataset_row_limit=3,
    )
    system = InteractionLearningSystem(config)
    try:
        record = system.queue_training_file(
            "pairs.parquet",
            parquet_path.read_bytes(),
            content_type="application/octet-stream",
        )
        assert record.modality == "dataset"

        result = system.process_training_file(record.id, mode="open", learn=True)
        updated = system.memory.get_training_file(record.id)
        dataset = result.metadata["dataset_ingestion"]

        assert "Dataset ingestion complete" in result.response
        assert dataset["row_count"] == 4
        assert dataset["records_seen"] == 3
        assert dataset["episodes_stored"] >= 1
        assert updated is not None
        assert updated.payload["dataset"]["records_seen"] == 3
    finally:
        system.close()
