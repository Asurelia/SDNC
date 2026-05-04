import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.self_improvement import SelfImprovementCycle
from sdnc.agent.system import InteractionLearningSystem
from sdnc.agent.types import Feedback


def test_circuit_growth_is_bounded_by_consensus(tmp_path):
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=20,
        max_circuits=24,
        min_growth_evidence=3,
        improvement_interval=100,
        min_experiment_gain=0.0,
        existing_coverage_threshold=1.1,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        for _ in range(3):
            system.interact(
                "calcule 12 + 30 pour verifier cette procedure",
                feedback=Feedback(1.0, "utile"),
            )

        before = system.config.n_circuits
        report = system.run_self_improvement()

        assert system.config.n_circuits == before + 1
        assert system.config.n_circuits <= system.config.max_circuits
        assert any(action.kind == "circuit_growth" and action.accepted for action in report.actions)
    finally:
        system.close()


def test_single_experience_does_not_grow_architecture(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=10,
        max_circuits=12,
        min_growth_evidence=3,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        system.interact("calcule 2 + 8", feedback=Feedback(1.0, "ok"))
        report = system.run_self_improvement()

        assert system.config.n_circuits == 10
        assert not any(action.kind == "circuit_growth" and action.accepted for action in report.actions)
    finally:
        system.close()


def test_discovery_must_pass_sandbox_experiment(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=10,
        max_circuits=12,
        min_growth_evidence=3,
        min_experiment_gain=2.0,
        existing_coverage_threshold=1.1,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        for _ in range(3):
            system.interact("calcule 5 + 5 pour explorer cette hypothese", feedback=Feedback(1.0, "utile"))

        report = system.run_self_improvement()

        assert system.config.n_circuits == 10
        assert any(
            action.kind == "sandbox_experiment" and not action.accepted
            for action in report.actions
        )
    finally:
        system.close()


def test_learner_recycles_when_growth_capacity_is_full(tmp_path):
    system = InteractionLearningSystem(
        AutonomousConfig(
            input_dim=8,
            n_circuits=3,
            max_circuits=3,
            memory_path=tmp_path / "memory.sqlite3",
            state_path=tmp_path / "state.npz",
            workspace_root=tmp_path,
            allow_web=False,
            auto_improve_enabled=False,
        )
    )
    try:
        prototype = np.ones(8, dtype=np.float32)
        idx, mode = system.learner.grow_circuit(prototype, source="test")

        assert system.config.n_circuits == 3
        assert 0 <= idx < 3
        assert mode.startswith("recycled")
    finally:
        system.close()


def test_weak_procedure_is_pruned(tmp_path):
    config = AutonomousConfig(
        input_dim=8,
        n_circuits=8,
        max_circuits=8,
        procedure_prune_failures=3,
        min_experiment_gain=0.0,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    system = InteractionLearningSystem(config)
    try:
        embedding = np.ones(8, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        for _ in range(3):
            system.memory.upsert_procedure(
                name="bad:skill",
                description="bad skill",
                tool_name="calculator",
                trigger_embedding=embedding,
                success=False,
            )

        report = SelfImprovementCycle(config, system.memory, system.learner).run()

        assert not system.memory.list_procedures()
        assert any(action.kind == "skill_pruning" for action in report.actions)
    finally:
        system.close()
