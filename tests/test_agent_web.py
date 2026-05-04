import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from base64 import b64encode

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.web import SDNCWebApp, build_handler


def test_web_api_status_and_interact(tmp_path):
    config = AutonomousConfig(
        input_dim=32,
        n_circuits=20,
        memory_path=tmp_path / "memory.sqlite3",
        state_path=tmp_path / "state.npz",
        workspace_root=tmp_path,
        allow_web=False,
    )
    app = SDNCWebApp(config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status = _get_json(f"{base}/api/status")
        assert status["n_circuits"] == 20
        assert "calculator" in status["tools"]
        assert status["expert_summary"]["total"] == 0

        result = _post_json(f"{base}/api/interact", {"text": "calcule 6 + 7", "mode": "think"})
        assert result["result"]["activation"]["active_count"] <= 1
        assert any(tool["tool_name"] == "calculator" for tool in result["result"]["tool_results"])
        assert result["result"]["metadata"]["cognitive_budget"]["mode"] == "think"
        assert result["result"]["metadata"]["cognitive_core"]["prediction"]["predicted_action"] == "use_tools"
        assert "resource_budget" in result["status"]
        assert result["status"]["cognitive_core"]["workspace_slots"] == config.cognitive_workspace_slots
        assert result["status"]["modalities"] == ["audio", "image", "text", "video"]

        observed = _post_json(
            f"{base}/api/observe",
            {
                "samples": [
                    {"modality": "text", "text": "signal bleu", "source": "test-ui", "sample_id": "m1"},
                    {
                        "modality": "image",
                        "content": [[[0, 0, 255], [0, 0, 255]]],
                        "text": "pixel bleu",
                        "source": "test-ui",
                        "sample_id": "m1",
                    },
                ]
            },
        )
        assert observed["result"]["metadata"]["sensory_event"]["modalities"] == ["text", "image"]
        assert observed["result"]["metadata"]["sensory_event"]["binding_score"] > 0

        upload = _post_json(
            f"{base}/api/files/upload",
            {
                "name": "note.txt",
                "content_type": "text/plain",
                "data_base64": b64encode(b"sdnc apprend depuis fichier").decode("ascii"),
            },
        )
        file_id = upload["file"]["id"]
        assert upload["summary"]["queued"] == 1

        processed = _post_json(
            f"{base}/api/files/process",
            {"file_id": file_id, "mode": "fast", "learn": True},
        )
        assert processed["summary"]["done"] == 1
        assert processed["result"]["metadata"]["sensory_event"]["modalities"] == ["text"]

        learning = _post_json(f"{base}/api/learn-gap", {})
        assert learning["report"]["summary"].startswith("Learning cycle:")

        events = _get_json(f"{base}/api/events")
        assert any(event["event_type"] == "interaction" for event in events["events"])
        assert any(event["event_type"] == "observation" for event in events["events"])
        assert any(event["event_type"] == "file" for event in events["events"])
        assert any(event["event_type"] == "learning" for event in events["events"])

        recent = _get_json(f"{base}/api/recent")
        assert any(binding["modalities"] == ["text", "image"] for binding in recent["sensory_bindings"])
        assert recent["cognitive_traces"]
        assert "attention" in recent["cognitive_traces"][0]
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def _get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))
