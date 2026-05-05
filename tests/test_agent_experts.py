import numpy as np

from sdnc.agent.budget import BudgetManager
from sdnc.agent.config import AutonomousConfig
from sdnc.agent.experts import ExpertManager
from sdnc.agent.learning import SelfDirectedLearner, SourceProposal
from sdnc.agent.memory import PersistentMemory
from sdnc.agent.system import InteractionLearningSystem
from sdnc.agent.types import CircuitActivation, InteractionResult, ToolResult


class ExpertAdvisor:
    name = "expert_advisor"

    def propose(self, gap, result, embedding):
        return [
            SourceProposal(
                source=self.name,
                claim="calculator expert should handle arithmetic gaps",
                recommendation="Create or reinforce a calculator procedure expert.",
                confidence=0.9,
                evidence="fixture",
                tool_name="calculator",
            )
        ]


def test_expert_memory_roundtrip_and_retrieval(tmp_path):
    memory = PersistentMemory(tmp_path / "memory.sqlite3", embedding_dim=8)
    emb = np.ones(8, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    expert_id = memory.upsert_expert(
        name="expert:calculator:addition",
        kind="procedure",
        description="addition expert",
        trigger_embedding=emb,
        status="probation",
        success=True,
    )
    memory.record_expert_result(expert_id, success=True)
    found = memory.retrieve_experts(emb, top_k=1)

    assert found[0].id == expert_id
    assert found[0].success_count == 2
    assert found[0].similarity > 0.99
    memory.close()


def test_expert_manager_selects_hot_under_budget(tmp_path):
    config = AutonomousConfig(
        input_dim=8,
        n_circuits=20,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        max_hot_experts=2,
    )
    memory = PersistentMemory(config.memory_path, embedding_dim=8)
    manager = ExpertManager(config, memory)
    emb = np.ones(8, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    for index in range(4):
        manager.create_from_learning(
            name=f"expert:test:{index}",
            kind="procedure",
            description="test expert",
            trigger_embedding=emb,
        )

    budget = BudgetManager(config).choose("", explicit_mode="max")
    report = manager.select_for_interaction(emb, budget)

    assert len(report.selected_hot) <= budget.max_hot_experts
    assert all(expert.hot for expert in memory.list_experts() if expert.id in {item.id for item in report.selected_hot})
    assert report.selected_hot[0].payload["expert_payload"]["kind"] == "procedure"
    decoded = manager.decode_expert(report.selected_hot[0], level="L1")
    assert decoded is not None
    assert decoded.integrity_ok
    assert decoded.data["trigger_sketch"]["dim"] == 8
    memory.close()


def test_learning_cycle_creates_probation_expert(tmp_path):
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
    try:
        learner = SelfDirectedLearner(
            config,
            system.memory,
            system.encoder,
            available_tools=system.registry.names(),
            expert_manager=system.expert_manager,
            advisors=[ExpertAdvisor()],
        )
        report = learner.run(result)
        experts = system.memory.list_experts()

        assert report.consolidated >= 1
        assert any(expert.kind == "procedure" and "calculator" in expert.name for expert in experts)
        created = next(expert for expert in experts if expert.kind == "procedure" and "calculator" in expert.name)
        assert created.payload["expert_payload"]["decode_level"] == "L1"
        decoded = system.expert_manager.decode_expert(created, level="L2")
        assert decoded is not None
        assert decoded.integrity_ok
        assert decoded.data["evidence"]["tool_name"] == "calculator"
    finally:
        system.close()
