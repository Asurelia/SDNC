import json

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.eval import run_default_evaluation


def test_default_evaluation_reports_learning_and_sparse_budget(tmp_path):
    config = AutonomousConfig(
        input_dim=64,
        n_circuits=40,
        max_active_ratio=0.05,
        memory_path=tmp_path / "eval_memory.sqlite3",
        state_path=tmp_path / "eval_state.npz",
        workspace_root=tmp_path,
        allow_web=False,
        auto_improve_enabled=False,
    )

    report = run_default_evaluation(config=config, reset=True)
    payload = report.to_payload()
    rendered = json.loads(report.to_json())

    assert payload["summary"]["total_runs"] == 6
    assert payload["summary"]["within_sparse_limit"]
    assert payload["summary"]["max_active_ratio"] <= 0.05
    assert payload["summary"]["repeat_success_rate"] >= payload["summary"]["initial_success_rate"]
    assert payload["summary"]["memory_hit_delta"] >= 0.0
    assert payload["summary"]["planner_actions"]["use_tools"] >= 2
    assert len(payload["case_runs"]) == 6
    assert any(run["case_name"] == "arithmetic_tool_routing" and run["success"] for run in payload["case_runs"])
    assert any(run["case_name"] == "file_search_routing" and run["success"] for run in payload["case_runs"])
    assert any(
        run["case_name"] == "repeated_memory"
        and run["pass_name"] == "repeat"
        and run["memory_hits"] >= 1
        and run["planner_action"]
        for run in payload["case_runs"]
    )
    assert rendered["summary"]["total_runs"] == payload["summary"]["total_runs"]
    assert (tmp_path / "docs" / "ARCHITECTURE.md").exists()
