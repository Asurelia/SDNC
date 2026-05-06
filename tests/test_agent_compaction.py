import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.system import InteractionLearningSystem


def test_memory_compaction_preserves_rule_evidence(tmp_path):
    config = AutonomousConfig(
        input_dim=16,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        memory_compaction_min_group_size=3,
        memory_compaction_batch_size=4,
    )
    system = InteractionLearningSystem(config)
    try:
        embedding = np.ones(16, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        episode_ids = []
        for index in range(5):
            episode_ids.append(
                system.memory.store_episode(
                    text=f"alpha motif repeated useful sample {index}",
                    context={"kind": "compaction"},
                    embedding=embedding,
                    active_circuits=[0],
                    salience=0.6,
                    feedback_score=1.0,
                )
            )
        rule_id = system.memory.upsert_rule(
            name="rule:memory_recall:alpha-motif",
            trigger_pattern="alpha-motif",
            preconditions={"topic": "alpha-motif"},
            action_tool="memory_recall",
            expected_outcome="memory recall should help",
            trigger_embedding=embedding,
            confidence=0.75,
            provenance=[episode_ids[0]],
            counterexamples=[episode_ids[1]],
        )
        link_id = system.memory.upsert_rule_link(
            rule_id=rule_id,
            target_kind="sensory_prototype",
            target_id="prototype-a",
            relation="matches_sensory_prototype",
            confidence=0.8,
            provenance=[episode_ids[1]],
            payload={"prototype_key": "text:alpha"},
        )

        preview = system.run_memory_compaction(preview=True)
        assert preview.compacted_count == 1
        assert system.memory.recent_memory_compactions(limit=5) == []

        report = system.run_memory_compaction()
        compactions = system.memory.recent_memory_compactions(limit=5)

        assert report.compacted_count == 1
        assert compactions[0].episode_count == 3
        assert set(compactions[0].episode_ids).isdisjoint({episode_ids[0], episode_ids[1]})
        assert {episode_ids[0], episode_ids[1]}.issubset(set(compactions[0].protected_episode_ids))
        assert compactions[0].rule_ids == [rule_id]
        assert compactions[0].rule_link_ids == [link_id]
        assert compactions[0].payload["raw_episodes_preserved"] is True
        assert any(event.event_type == "compaction" for event in system.recent_events(limit=5))
    finally:
        system.close()
