from sdnc.agent.config import AutonomousConfig
from sdnc.agent.curriculum import curriculum_manifest, run_guided_curriculum
from sdnc.agent.system import InteractionLearningSystem


def test_guided_curriculum_runs_core_training_steps(tmp_path):
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=64,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        auto_improve_enabled=False,
        rule_min_evidence=2,
    )
    system = InteractionLearningSystem(config)
    try:
        manifest = curriculum_manifest()
        assert [step["id"] for step in manifest["steps"]] == [
            "calculator",
            "memory",
            "file_lookup",
            "sensory",
            "rules",
            "replay",
        ]

        memory_report = run_guided_curriculum(system, step_id="memory", mode="think")
        assert len(memory_report.steps) == 1
        assert memory_report.steps[0].passed
        assert memory_report.steps[0].metrics["best_memory_similarity"] >= 0.55

        full_report = run_guided_curriculum(system, mode="think", sleep_preview=True, batch_size=5)
        payload = full_report.to_payload()
        assert payload["step_count"] == 6
        assert payload["score"] >= 0.5
        assert any(step["id"] == "sensory" and step["passed"] for step in payload["steps"])
        assert full_report.after["sensory_prototype_summary"]["total"] >= 1
        assert all(
            interaction["active_count"] / config.n_circuits <= 0.05
            for step in payload["steps"]
            for interaction in step["interactions"]
        )
    finally:
        system.close()
