from sdnc.agent.config import AutonomousConfig
from sdnc.agent.system import InteractionLearningSystem


def test_sleep_cycle_replays_salient_episodes_and_strengthens_procedure(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        replay_min_salience=0.3,
        procedure_min_successes=2,
    )
    system = InteractionLearningSystem(config)
    try:
        embedding = system.learner.circuit_keys[0].copy()
        context = {
            "__sdnc": {
                "tool_names": ["memory_recall", "calculator"],
                "proposed_tool_names": ["memory_recall", "calculator"],
                "successful_tools": ["calculator"],
            }
        }
        for index in range(3):
            system.memory.store_episode(
                text="calcule addition stable",
                context=context,
                embedding=embedding,
                active_circuits=[0],
                salience=0.82 - index * 0.02,
                feedback_score=1.0,
            )

        preview = system.run_sleep_cycle(preview=True)
        report = system.run_sleep_cycle()
        procedures = system.memory.retrieve_procedures(embedding, top_k=5)
        events = system.recent_events(limit=5)

        assert preview.preview
        assert any(action.kind == "episode_preview" for action in preview.actions)
        assert report.replayed_count >= 3
        assert report.strengthened_count >= 1
        assert any(proc.tool_name == "calculator" for proc in procedures)
        assert any(event.event_type == "sleep" for event in events)
        assert (tmp_path / "state.npz").exists()
    finally:
        system.close()


def test_sleep_cycle_holds_negative_feedback_out_of_replay(tmp_path):
    config = AutonomousConfig(
        input_dim=16,
        n_circuits=20,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        replay_min_salience=0.2,
    )
    system = InteractionLearningSystem(config)
    try:
        embedding = system.learner.circuit_keys[0].copy()
        system.memory.store_episode(
            text="mauvaise procedure a ne pas rejouer",
            context={"__sdnc": {"successful_tools": ["calculator"]}},
            embedding=embedding,
            active_circuits=[0],
            salience=0.9,
            feedback_score=-1.0,
        )

        report = system.run_sleep_cycle()

        assert report.replayed_count == 0
        assert any(
            action.kind == "episode_replay" and not action.accepted
            and "negative feedback" in action.reason
            for action in report.actions
        )
    finally:
        system.close()
