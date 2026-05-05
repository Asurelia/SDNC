import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.memory import PersistentMemory
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
