"""Local web interface for the SDNC interaction learner."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import time
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.learning import LearningCycleReport
from sdnc.agent.multimodal import SUPPORTED_MODALITIES, ModalitySample
from sdnc.agent.self_improvement import ImprovementReport
from sdnc.agent.system import InteractionLearningSystem
from sdnc.agent.types import InteractionResult

STATIC_DIR = Path(__file__).with_name("static")


class SDNCWebApp:
    """Holds one long-lived SDNC system for the web server."""

    def __init__(self, config: AutonomousConfig):
        self.system = InteractionLearningSystem(config)

    def close(self) -> None:
        self.system.close()


def build_handler(app: SDNCWebApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SDNCWeb/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_static("index.html")
            elif parsed.path in {"/app.css", "/app.js"}:
                self._send_static(parsed.path.lstrip("/"))
            elif parsed.path == "/api/status":
                self._send_json(self._status_payload())
            elif parsed.path == "/api/recent":
                memories = [
                    _memory_payload(memory)
                    for memory in app.system.recent_memories(limit=12)
                ]
                sensory = [
                    _sensory_binding_payload(binding)
                    for binding in app.system.recent_sensory_bindings(limit=12)
                ]
                cognitive = [
                    _cognitive_trace_payload(trace)
                    for trace in app.system.recent_cognitive_traces(limit=12)
                ]
                self._send_json(
                    {
                        "memories": memories,
                        "sensory_bindings": sensory,
                        "cognitive_traces": cognitive,
                    }
                )
            elif parsed.path == "/api/events":
                query = parse_qs(parsed.query)
                after = int(query["after"][0]) if "after" in query else None
                events = [
                    _event_payload(event)
                    for event in app.system.recent_events(limit=50, after_id=after)
                ]
                self._send_json({"events": events})
            elif parsed.path == "/api/files":
                files = [_training_file_payload(record) for record in app.system.list_training_files()]
                self._send_json({"files": files, "summary": app.system.training_file_summary()})
            elif parsed.path == "/api/stream":
                query = parse_qs(parsed.query)
                after = int(query["after"][0]) if "after" in query else None
                self._send_event_stream(after)
            else:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            payload = self._read_json()
            if parsed.path == "/api/interact":
                text = str(payload.get("text", "")).strip()
                if not text:
                    self._send_json({"error": "text is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                interaction_context = {"ui": "web"}
                if payload.get("mode"):
                    interaction_context["mode"] = str(payload["mode"])
                result = app.system.interact(text, context=interaction_context)
                self._send_json({"result": _interaction_payload(result), "status": self._status_payload()})
            elif parsed.path == "/api/observe":
                try:
                    samples = _samples_from_payload(payload)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                observation_context = {"ui": "web"}
                if payload.get("mode"):
                    observation_context["mode"] = str(payload["mode"])
                result = app.system.observe(
                    samples,
                    context=observation_context,
                    learn=bool(payload.get("learn", True)),
                    use_tools=bool(payload.get("use_tools", False)),
                )
                self._send_json({"result": _interaction_payload(result), "status": self._status_payload()})
            elif parsed.path == "/api/files/upload":
                try:
                    name, data, content_type, upload_payload = _decode_upload_payload(payload)
                    record = app.system.queue_training_file(
                        name=name,
                        data=data,
                        content_type=content_type,
                        status=str(payload.get("status") or "queued"),
                        payload=upload_payload,
                    )
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(
                    {
                        "file": _training_file_payload(record),
                        "files": [_training_file_payload(item) for item in app.system.list_training_files()],
                        "summary": app.system.training_file_summary(),
                        "status": self._status_payload(),
                    }
                )
            elif parsed.path == "/api/files/status":
                file_id = str(payload.get("file_id") or "")
                new_status = str(payload.get("status") or "")
                if not file_id or not new_status:
                    self._send_json({"error": "file_id and status are required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    record = app.system.set_training_file_status(file_id, new_status)
                except (KeyError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(
                    {
                        "file": _training_file_payload(record),
                        "files": [_training_file_payload(item) for item in app.system.list_training_files()],
                        "summary": app.system.training_file_summary(),
                        "status": self._status_payload(),
                    }
                )
            elif parsed.path == "/api/files/process":
                file_id = str(payload.get("file_id") or "")
                if not file_id:
                    self._send_json({"error": "file_id is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = app.system.process_training_file(
                        file_id,
                        mode=str(payload.get("mode") or app.system.config.default_cognitive_mode),
                        learn=bool(payload.get("learn", True)),
                        use_tools=bool(payload.get("use_tools", False)),
                    )
                except (KeyError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(
                    {
                        "result": _interaction_payload(result),
                        "files": [_training_file_payload(item) for item in app.system.list_training_files()],
                        "summary": app.system.training_file_summary(),
                        "status": self._status_payload(),
                    }
                )
            elif parsed.path == "/api/files/process-next":
                processed = []
                count = max(1, min(20, int(payload.get("count") or 1)))
                try:
                    for _ in range(count):
                        result = app.system.process_next_training_file(
                            mode=str(payload.get("mode") or app.system.config.default_cognitive_mode),
                            learn=bool(payload.get("learn", True)),
                            use_tools=bool(payload.get("use_tools", False)),
                        )
                        if result is None:
                            break
                        processed.append(_interaction_payload(result))
                except (KeyError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(
                    {
                        "processed": processed,
                        "files": [_training_file_payload(item) for item in app.system.list_training_files()],
                        "summary": app.system.training_file_summary(),
                        "status": self._status_payload(),
                    }
                )
            elif parsed.path == "/api/feedback":
                score = float(payload.get("score", 0.0))
                note = str(payload.get("text", ""))
                result = app.system.give_feedback(score, note)
                self._send_json({"result": _interaction_payload(result), "status": self._status_payload()})
            elif parsed.path == "/api/improve":
                report = app.system.run_self_improvement()
                self._send_json({"report": _report_payload(report), "status": self._status_payload()})
            elif parsed.path == "/api/sleep":
                report = app.system.run_sleep_cycle(
                    preview=bool(payload.get("preview", False)),
                    batch_size=int(payload["batch_size"]) if payload.get("batch_size") else None,
                )
                self._send_json({"report": report.to_payload(), "status": self._status_payload()})
            elif parsed.path == "/api/rules/consolidate":
                report = app.system.run_rule_consolidation()
                self._send_json({"report": report.to_payload(), "status": self._status_payload()})
            elif parsed.path == "/api/learn-gap":
                report = app.system.learn_from_last_gap()
                self._send_json({"report": _learning_report_payload(report), "status": self._status_payload()})
            else:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_static(self, filename: str) -> None:
            path = (STATIC_DIR / filename).resolve()
            try:
                path.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self._send_json({"error": "invalid static path"}, status=HTTPStatus.BAD_REQUEST)
                return
            if not path.exists() or not path.is_file():
                self._send_json({"error": "static file not found"}, status=HTTPStatus.NOT_FOUND)
                return
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
            }.get(path.suffix, "application/octet-stream")
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            body = self.rfile.read(length)
            return json.loads(body.decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _status_payload(self) -> dict[str, Any]:
            config = app.system.config
            return {
                "n_circuits": config.n_circuits,
                "max_circuits": config.max_circuits,
                "max_active_circuits": config.max_active_circuits,
                "sparsity_ratio": config.sparsity_ratio,
                "default_cognitive_mode": config.default_cognitive_mode,
                "resource_budget": _resource_snapshot_payload(
                    app.system.budget_manager.snapshot(
                        app.system.budget_manager.choose("", explicit_mode=config.default_cognitive_mode)
                    )
                ),
                "expert_summary": app.system.expert_manager.summary(),
                "rule_summary": app.system.rule_summary(),
                "file_queue": app.system.training_file_summary(),
                "cognitive_core": {
                    "workspace_slots": config.cognitive_workspace_slots,
                    "attention_focus": config.cognitive_attention_focus,
                },
                "modalities": sorted(SUPPORTED_MODALITIES),
                "tools": app.system.registry.names(),
                "memory_path": str(config.memory_path),
                "state_path": str(config.state_path),
                "auto_improve_enabled": config.auto_improve_enabled,
                "improvement_interval": config.improvement_interval,
                "sync_to_convex": config.sync_to_convex,
                "convex_url": config.convex_url,
                "convex_last_error": getattr(app.system.event_mirror, "last_error", None),
                "source_stats": [
                    {
                        "source_name": stat.source_name,
                        "proposals": stat.proposals,
                        "accepted": stat.accepted,
                        "rejected": stat.rejected,
                        "trust": stat.trust,
                    }
                    for stat in app.system.memory.list_source_stats()
                ],
            }

        def _send_event_stream(self, after_id: int | None) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            last_id = after_id
            end_at = time.time() + 60.0
            while time.time() < end_at:
                events = app.system.recent_events(limit=50, after_id=last_id)
                for event in events:
                    last_id = event.id
                    body = json.dumps(_event_payload(event), ensure_ascii=False, default=_json_default)
                    message = f"id: {event.id}\nevent: {event.event_type}\ndata: {body}\n\n"
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()
                time.sleep(1.0)

    return Handler


def run_server(config: AutonomousConfig, host: str, port: int) -> None:
    app = SDNCWebApp(config)
    server = ThreadingHTTPServer((host, port), build_handler(app))
    print(f"SDNC web UI listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the SDNC local web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--memory", type=Path, default=Path("data/autonomous_memory.sqlite3"))
    parser.add_argument("--state", type=Path, default=Path("data/autonomous_circuits.npz"))
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--no-web", action="store_true", help="Disable external web search tool.")
    parser.add_argument("--sync-to-convex", action="store_true", help="Mirror local events to Convex.")
    parser.add_argument("--convex-url", default=None)
    parser.add_argument("--convex-mutation", default="sdnc:ingestEvent")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AutonomousConfig(
        memory_path=args.memory,
        state_path=args.state,
        workspace_root=args.workspace,
        allow_web=not args.no_web,
        sync_to_convex=args.sync_to_convex,
        convex_url=args.convex_url,
        convex_event_mutation=args.convex_mutation,
    )
    run_server(config, args.host, args.port)
    return 0


def _interaction_payload(result: InteractionResult) -> dict[str, Any]:
    return {
        "episode_id": result.episode_id,
        "timestamp": result.timestamp,
        "input_text": result.input_text,
        "response": result.response,
        "learned": result.learned,
        "activation": {
            "indices": result.activation.indices,
            "weights": result.activation.weights,
            "scores": result.activation.scores,
            "confidence": result.activation.confidence,
            "novelty": result.activation.novelty,
            "active_count": result.activation.active_count,
        },
        "memories": [_memory_payload(memory) for memory in result.memories],
        "tool_results": [
            {
                "tool_name": tool.tool_name,
                "success": tool.success,
                "content": tool.content,
                "metadata": tool.metadata,
            }
            for tool in result.tool_results
        ],
        "metadata": _metadata_payload(result.metadata),
    }


def _memory_payload(memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "timestamp": memory.timestamp,
        "text": memory.text,
        "active_circuits": memory.active_circuits,
        "salience": memory.salience,
        "outcome": memory.outcome,
        "feedback_score": memory.feedback_score,
        "similarity": memory.similarity,
    }


def _sensory_binding_payload(binding) -> dict[str, Any]:
    return {
        "id": binding.id,
        "timestamp": binding.timestamp,
        "episode_id": binding.episode_id,
        "source": binding.source,
        "modalities": binding.modalities,
        "sample_ids": binding.sample_ids,
        "summary": binding.summary,
        "binding_score": binding.binding_score,
        "salience": binding.salience,
        "similarity": binding.similarity,
    }


def _cognitive_trace_payload(trace) -> dict[str, Any]:
    return {
        "id": trace.id,
        "timestamp": trace.timestamp,
        "episode_id": trace.episode_id,
        "input_text": trace.input_text,
        "mode": trace.mode,
        "prediction": trace.prediction,
        "observation": trace.observation,
        "attention": trace.attention,
        "surprise": trace.surprise,
        "uncertainty": trace.uncertainty,
        "payload": trace.payload,
    }


def _training_file_payload(record) -> dict[str, Any]:
    return {
        "id": record.id,
        "timestamp": record.timestamp,
        "updated_at": record.updated_at,
        "name": record.name,
        "path": record.path,
        "content_type": record.content_type,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
        "status": record.status,
        "modality": record.modality,
        "preview": record.preview,
        "processed_episode_id": record.processed_episode_id,
        "error": record.error,
        "payload": record.payload,
    }


def _report_payload(report: ImprovementReport) -> dict[str, Any]:
    return {
        "timestamp": report.timestamp,
        "accepted_count": report.accepted_count,
        "summary": report.summary(),
        "actions": [
            {
                "kind": action.kind,
                "accepted": action.accepted,
                "reason": action.reason,
                "details": action.details,
            }
            for action in report.actions
        ],
    }


def _learning_report_payload(report: LearningCycleReport) -> dict[str, Any]:
    return {
        "timestamp": report.timestamp,
        "summary": report.summary(),
        "consolidated": report.consolidated,
        "gaps": [
            {
                "kind": gap.kind,
                "description": gap.description,
                "severity": gap.severity,
                "evidence": gap.evidence,
            }
            for gap in report.gaps
        ],
        "proposals": [
            {
                "source": proposal.source,
                "claim": proposal.claim,
                "recommendation": proposal.recommendation,
                "confidence": proposal.confidence,
                "evidence": proposal.evidence,
                "tool_name": proposal.tool_name,
            }
            for proposal in report.proposals
        ],
        "candidates": [
            {
                "claim": candidate.claim,
                "recommendation": candidate.recommendation,
                "score": candidate.score,
                "sources": candidate.sources,
                "tool_name": candidate.tool_name,
            }
            for candidate in report.candidates
        ],
        "verdicts": [
            {
                "accepted": verdict.accepted,
                "score": verdict.score,
                "reason": verdict.reason,
                "candidate": verdict.candidate.claim,
                "metrics": verdict.metrics,
            }
            for verdict in report.verdicts
        ],
    }


def _event_payload(event) -> dict[str, Any]:
    return {
        "id": event.id,
        "timestamp": event.timestamp,
        "event_type": event.event_type,
        "source": event.source,
        "payload": event.payload,
    }


def _metadata_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = dict(metadata)
    report = payload.get("improvement_report")
    if isinstance(report, ImprovementReport):
        payload["improvement_report"] = _report_payload(report)
    return payload


def _resource_snapshot_payload(snapshot) -> dict[str, Any]:
    return {
        "total_vram_gb": snapshot.total_vram_gb,
        "usable_vram_gb": snapshot.usable_vram_gb,
        "hot_vram_gb": round(snapshot.hot_vram_gb, 4),
        "cold_ram_gb": round(snapshot.cold_ram_gb, 4),
        "cold_disk_gb": round(snapshot.cold_disk_gb, 4),
        "within_budget": snapshot.within_budget,
    }


def _samples_from_payload(payload: dict[str, Any]) -> list[ModalitySample]:
    raw_samples = payload.get("samples")
    if raw_samples is not None:
        if not isinstance(raw_samples, list):
            raise ValueError("samples must be a list")
        samples = [_sample_from_payload(item, default_source=str(payload.get("source") or "web")) for item in raw_samples]
    else:
        samples = [_sample_from_payload(payload, default_source="web")]
    if not samples:
        raise ValueError("at least one sample is required")
    return samples


def _decode_upload_payload(payload: dict[str, Any]) -> tuple[str, bytes, str, dict[str, Any]]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    encoded = str(payload.get("data_base64") or "")
    if not encoded:
        raise ValueError("data_base64 is required")
    if "," in encoded and encoded.split(",", 1)[0].startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("data_base64 is invalid") from exc
    content_type = str(payload.get("content_type") or "application/octet-stream")
    upload_payload = {
        "label": payload.get("label"),
        "origin": str(payload.get("origin") or "web-ui"),
        "tags": payload.get("tags") if isinstance(payload.get("tags"), list) else [],
    }
    return name, data, content_type, upload_payload


def _sample_from_payload(payload: dict[str, Any], default_source: str) -> ModalitySample:
    if not isinstance(payload, dict):
        raise ValueError("each sample must be an object")
    modality = str(payload.get("modality") or "text").strip().lower()
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"unsupported modality: {modality}")
    text = str(payload.get("text") or "")
    metadata = dict(payload.get("metadata") or {})
    content = payload.get("content", text if modality == "text" else None)
    return ModalitySample(
        modality=modality,
        content=_coerce_content(modality, content),
        text=text,
        label=str(payload["label"]) if payload.get("label") is not None else None,
        source=str(payload.get("source") or default_source),
        sample_id=str(payload["sample_id"]) if payload.get("sample_id") is not None else None,
        metadata=metadata,
    )


def _coerce_content(modality: str, content: Any) -> Any:
    if modality == "image" and isinstance(content, list):
        return np.asarray(content)
    if modality == "audio":
        if isinstance(content, list):
            return np.asarray(content, dtype=np.float32)
        if isinstance(content, dict) and isinstance(content.get("array"), list):
            coerced = dict(content)
            coerced["array"] = np.asarray(content["array"], dtype=np.float32)
            return coerced
    if modality == "video":
        if isinstance(content, dict):
            coerced = dict(content)
            for key in ("frames", "images"):
                if isinstance(coerced.get(key), list):
                    coerced[key] = [_coerce_frame(frame) for frame in coerced[key]]
            return coerced
        if isinstance(content, list):
            return [_coerce_frame(frame) for frame in content]
    return content


def _coerce_frame(frame: Any) -> Any:
    return np.asarray(frame) if isinstance(frame, list) else frame


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
