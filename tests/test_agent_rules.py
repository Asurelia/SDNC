import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.memory import PersistentMemory
from sdnc.agent.multimodal import ModalitySample
from sdnc.agent.rules import RuleEngine
from sdnc.agent.system import InteractionLearningSystem


def test_rule_consolidation_promotes_repeated_successes(tmp_path):
    config = AutonomousConfig(
        input_dim=24,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        rule_min_evidence=2,
    )
    system = InteractionLearningSystem(config)
    try:
        embedding = system.encoder.encode("architecture stable")
        for _ in range(2):
            system.memory.store_episode(
                text="architecture stable",
                context={
                    "__sdnc": {
                        "tool_names": ["memory_recall", "file_search"],
                        "proposed_tool_names": ["memory_recall", "file_search"],
                        "successful_tools": ["file_search"],
                    }
                },
                embedding=embedding,
                active_circuits=[0],
                salience=0.8,
                feedback_score=1.0,
            )

        report = system.run_rule_consolidation()
        rules = system.memory.retrieve_rules(embedding, top_k=3)
        matches = system.rule_engine.match(embedding)
        chosen = system._choose_tools(
            "architecture stable",
            embedding,
            activation_confidence=0.9,
            rule_matches=matches,
        )

        assert report.promoted_count == 1
        assert rules[0].action_tool == "file_search"
        assert rules[0].confidence >= config.rule_min_confidence
        assert len(rules[0].provenance) == 2
        assert "file_search" in chosen
        assert any(event.event_type == "rules" for event in system.recent_events(limit=3))
    finally:
        system.close()


def test_rule_contradiction_weakens_existing_rule(tmp_path):
    config = AutonomousConfig(input_dim=8, n_circuits=20, rule_min_evidence=2)
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=8)
    engine = RuleEngine(config, memory)
    embedding = np.ones(8, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    rule_id = memory.upsert_rule(
        name="rule:calculator:addition-stable",
        trigger_pattern="addition-stable",
        preconditions={"topic": "addition-stable"},
        action_tool="calculator",
        expected_outcome="calculator should help",
        trigger_embedding=embedding,
        confidence=0.7,
        provenance=["p1", "p2"],
    )
    for index in range(3):
        memory.store_episode(
            text="addition stable",
            context={
                "__sdnc": {
                    "tool_names": ["calculator"],
                    "proposed_tool_names": ["calculator"],
                    "successful_tools": [],
                }
            },
            embedding=embedding,
            active_circuits=[0],
            salience=0.7,
            feedback_score=-1.0,
        )

    report = engine.consolidate_recent()
    rule = next(item for item in memory.list_rules() if item.id == rule_id)

    assert report.weakened_count == 1
    assert rule.confidence < 0.7
    assert len(rule.counterexamples) >= 3
    memory.close()


def test_rule_engine_links_rules_to_experts_with_provenance(tmp_path):
    config = AutonomousConfig(
        input_dim=24,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        embedding = system.encoder.encode("analyse UI stable")
        expert_id = system.expert_manager.create_from_learning(
            name="expert:ui-analysis",
            kind="procedure",
            description="Analyse les signaux UI recurrent.",
            trigger_embedding=embedding,
            payload={"tool_name": "memory_recall"},
        )
        rule_id = system.memory.upsert_rule(
            name="rule:memory_recall:ui-analysis",
            trigger_pattern="ui-analysis",
            preconditions={"topic": "ui-analysis"},
            action_tool="memory_recall",
            expected_outcome="memory_recall should help for ui-analysis",
            trigger_embedding=embedding,
            confidence=0.8,
            provenance=["episode-a"],
        )

        matches = system.rule_engine.match(embedding)
        first_count = system.rule_engine.attach_matches(
            matches,
            target_kind="expert",
            target_id=expert_id,
            relation="influences_hot_expert",
            provenance=["trace-a"],
            payload={"target_name": "expert:ui-analysis"},
        )
        second_count = system.rule_engine.attach_matches(
            matches,
            target_kind="expert",
            target_id=expert_id,
            relation="influences_hot_expert",
            provenance=["trace-b"],
        )
        links = system.memory.list_rule_links(target_kind="expert", target_id=expert_id)

        assert first_count == 1
        assert second_count == 1
        assert len(links) == 1
        assert links[0].rule_id == rule_id
        assert links[0].relation == "influences_hot_expert"
        assert {"trace-a", "trace-b"}.issubset(set(links[0].provenance))
    finally:
        system.close()


def test_system_observe_links_matching_rules_to_sensory_prototypes(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=40,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        samples = [
            ModalitySample(
                "text",
                text="bouton vert confirme une action",
                label="ui-confirm",
                source="fixture",
                sample_id="s1",
            )
        ]
        event = system.perception.observe(samples)
        rule_id = system.memory.upsert_rule(
            name="rule:memory_recall:ui-confirm",
            trigger_pattern="ui-confirm",
            preconditions={"topic": "ui-confirm"},
            action_tool="memory_recall",
            expected_outcome="memory_recall should help for ui-confirm",
            trigger_embedding=event.embedding,
            confidence=0.82,
            provenance=["seed-event"],
        )

        result = system.observe(samples, learn=True, use_tools=False)
        learned = result.metadata["sensory_prototypes"]["learned"]
        links = system.memory.list_rule_links(
            target_kind="sensory_prototype",
            target_id=learned["id"],
        )

        assert result.metadata["rule_attachments"]["sensory_prototype_links"] == 1
        assert len(links) == 1
        assert links[0].rule_id == rule_id
        assert links[0].payload["prototype_key"] == learned["key"]
        assert result.metadata["neuro_symbolic_rules"][0]["name"] == "rule:memory_recall:ui-confirm"
    finally:
        system.close()
