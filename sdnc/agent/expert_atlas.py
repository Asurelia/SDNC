"""Compressed expert payloads for the SDNC expert atlas.

The atlas keeps experts as cold, verifiable assets. Decoding is deliberately
deterministic first: L0 gives the symbolic handle, L1 gives a compact sketch,
and L2 materializes the full local payload when the budget allows it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DECODE_LEVELS = ("L0", "L1", "L2")
PAYLOAD_VERSION = 1


@dataclass(frozen=True)
class ExpertPayload:
    """Cold expert asset stored in SQLite or an external atlas pack."""

    version: int
    kind: str
    decode_level: str
    data: dict[str, Any]
    checksum: str
    byte_size: int
    decode_cost_ms: float
    hot_ram_gb: float
    hot_vram_gb: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "decode_level": self.decode_level,
            "data": self.data,
            "checksum": self.checksum,
            "byte_size": self.byte_size,
            "decode_cost_ms": self.decode_cost_ms,
            "hot_ram_gb": self.hot_ram_gb,
            "hot_vram_gb": self.hot_vram_gb,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DecodedExpert:
    """A decoded view of an expert for the current interaction."""

    kind: str
    decode_level: str
    data: dict[str, Any]
    checksum: str
    integrity_ok: bool
    byte_size: int
    decode_cost_ms: float
    estimated_ram_gb: float
    estimated_vram_gb: float
    available_levels: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def build_procedure_payload(
    name: str,
    description: str,
    tool_name: str = "",
    trigger_embedding: np.ndarray | None = None,
    payload: dict[str, Any] | None = None,
    decode_level: str = "L1",
) -> ExpertPayload:
    """Build a compressed symbolic procedure expert."""

    source_payload = _jsonable(payload or {})
    levels = {
        "L0": {
            "name": str(name),
            "description": str(description)[:500],
            "tool_name": str(tool_name or source_payload.get("tool_name", "")),
            "kind": "procedure",
        },
        "L1": {
            "trigger_sketch": _sketch_vector(trigger_embedding, top_k=8),
            "action_hints": _action_hints(source_payload),
        },
        "L2": {
            "evidence": source_payload,
        },
    }
    return _make_payload(
        kind="procedure",
        decode_level=decode_level,
        levels=levels,
        hot_vram_gb=0.0,
        metadata={"storage": "sqlite-json", "materializer": "deterministic"},
    )


def build_prototype_payload(
    key: str,
    centroid: np.ndarray | list[float],
    threshold: float = 0.82,
    metadata: dict[str, Any] | None = None,
    decode_level: str = "L1",
) -> ExpertPayload:
    """Build a vector-prototype expert with LOD materialization."""

    centroid_array = _as_vector(centroid)
    levels = {
        "L0": {
            "key": str(key),
            "kind": "prototype",
            "dim": int(centroid_array.size),
            "threshold": round(float(threshold), 6),
        },
        "L1": {
            "centroid_sketch": _sketch_vector(centroid_array, top_k=12),
        },
        "L2": {
            "centroid": _round_array(centroid_array),
        },
    }
    return _make_payload(
        kind="prototype",
        decode_level=decode_level,
        levels=levels,
        hot_vram_gb=0.0,
        metadata=_jsonable(metadata or {}),
    )


def build_low_rank_payload(
    name: str,
    left: np.ndarray | list[list[float]],
    right: np.ndarray | list[list[float]],
    bias: np.ndarray | list[float] | None = None,
    metadata: dict[str, Any] | None = None,
    decode_level: str = "L2",
) -> ExpertPayload:
    """Build a deterministic low-rank expert payload."""

    left_array = _as_matrix(left)
    right_array = _as_matrix(right)
    bias_array = _as_vector(bias) if bias is not None else np.zeros(0, dtype=np.float32)
    rank = min(
        int(left_array.shape[1]) if left_array.ndim == 2 else 0,
        int(right_array.shape[0]) if right_array.ndim == 2 else 0,
    )
    levels = {
        "L0": {
            "name": str(name),
            "kind": "low_rank",
            "left_shape": list(left_array.shape),
            "right_shape": list(right_array.shape),
            "bias_shape": list(bias_array.shape),
            "rank": rank,
        },
        "L1": {
            "left_norm": round(float(np.linalg.norm(left_array)), 6),
            "right_norm": round(float(np.linalg.norm(right_array)), 6),
            "bias_norm": round(float(np.linalg.norm(bias_array)), 6),
        },
        "L2": {
            "left": _round_array(left_array),
            "right": _round_array(right_array),
            "bias": _round_array(bias_array),
        },
    }
    return _make_payload(
        kind="low_rank",
        decode_level=decode_level,
        levels=levels,
        hot_vram_gb=0.0,
        metadata=_jsonable(metadata or {}),
    )


def build_sparse_delta_payload(
    name: str,
    shape: tuple[int, ...] | list[int],
    indices: np.ndarray | list[list[int]] | list[int],
    values: np.ndarray | list[float],
    metadata: dict[str, Any] | None = None,
    decode_level: str = "L2",
) -> ExpertPayload:
    """Build a sparse residual update payload."""

    index_array = np.asarray(indices, dtype=np.int64)
    value_array = _as_vector(values)
    flat_indices = index_array.reshape(-1).astype(int).tolist()
    levels = {
        "L0": {
            "name": str(name),
            "kind": "sparse_delta",
            "shape": [int(item) for item in shape],
            "nnz": int(value_array.size),
        },
        "L1": {
            "index_preview": flat_indices[:24],
            "value_norm": round(float(np.linalg.norm(value_array)), 6),
        },
        "L2": {
            "indices": _jsonable(index_array),
            "values": _round_array(value_array),
        },
    }
    return _make_payload(
        kind="sparse_delta",
        decode_level=decode_level,
        levels=levels,
        hot_vram_gb=0.0,
        metadata=_jsonable(metadata or {}),
    )


def build_codebook_payload(
    name: str,
    codebook: np.ndarray | list[list[float]],
    codes: np.ndarray | list[int],
    output_shape: tuple[int, ...] | list[int],
    metadata: dict[str, Any] | None = None,
    decode_level: str = "L2",
) -> ExpertPayload:
    """Build a quantized codebook-reference payload."""

    codebook_array = _as_matrix(codebook)
    code_array = np.asarray(codes, dtype=np.int64).reshape(-1)
    histogram = np.bincount(
        np.maximum(code_array, 0),
        minlength=max(int(codebook_array.shape[0]) if codebook_array.ndim == 2 else 0, 1),
    )
    levels = {
        "L0": {
            "name": str(name),
            "kind": "codebook",
            "output_shape": [int(item) for item in output_shape],
            "codebook_shape": list(codebook_array.shape),
            "code_count": int(code_array.size),
        },
        "L1": {
            "code_histogram": histogram.astype(int).tolist()[:64],
        },
        "L2": {
            "codebook": _round_array(codebook_array),
            "codes": code_array.astype(int).tolist(),
        },
    }
    return _make_payload(
        kind="codebook",
        decode_level=decode_level,
        levels=levels,
        hot_vram_gb=0.0,
        metadata=_jsonable(metadata or {}),
    )


def payload_from_dict(payload: ExpertPayload | dict[str, Any]) -> ExpertPayload:
    """Normalize a stored atlas payload."""

    if isinstance(payload, ExpertPayload):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("expert payload must be a dictionary")
    return ExpertPayload(
        version=int(payload["version"]),
        kind=str(payload["kind"]),
        decode_level=_validate_level(str(payload.get("decode_level", "L0"))),
        data=dict(payload.get("data") or {}),
        checksum=str(payload["checksum"]),
        byte_size=int(payload.get("byte_size", 0)),
        decode_cost_ms=float(payload.get("decode_cost_ms", 0.0)),
        hot_ram_gb=float(payload.get("hot_ram_gb", 0.0)),
        hot_vram_gb=float(payload.get("hot_vram_gb", 0.0)),
        metadata=dict(payload.get("metadata") or {}),
    )


def decode_payload(payload: ExpertPayload | dict[str, Any], level: str | None = None) -> DecodedExpert:
    """Decode an expert payload up to the requested level."""

    normalized = payload_from_dict(payload)
    requested = _validate_level(level or normalized.decode_level)
    body = _checksum_body(normalized.kind, normalized.decode_level, normalized.data, normalized.metadata)
    integrity_ok = _checksum(body) == normalized.checksum
    data = _decode_level_data(normalized.data, requested)
    return DecodedExpert(
        kind=normalized.kind,
        decode_level=requested,
        data=data,
        checksum=normalized.checksum,
        integrity_ok=integrity_ok,
        byte_size=normalized.byte_size,
        decode_cost_ms=normalized.decode_cost_ms,
        estimated_ram_gb=normalized.hot_ram_gb,
        estimated_vram_gb=normalized.hot_vram_gb,
        available_levels=_available_levels(normalized.data),
        metadata=normalized.metadata,
    )


def payload_summary(payload: ExpertPayload | dict[str, Any]) -> dict[str, Any]:
    """Return a small report suitable for UI metadata."""

    normalized = payload_from_dict(payload)
    decoded = decode_payload(normalized, level="L0")
    return {
        "kind": normalized.kind,
        "decode_level": normalized.decode_level,
        "byte_size": normalized.byte_size,
        "decode_cost_ms": normalized.decode_cost_ms,
        "hot_ram_gb": normalized.hot_ram_gb,
        "hot_vram_gb": normalized.hot_vram_gb,
        "checksum": normalized.checksum,
        "checksum_prefix": normalized.checksum[:12],
        "integrity_ok": decoded.integrity_ok,
        "available_levels": decoded.available_levels,
    }


def _make_payload(
    kind: str,
    decode_level: str,
    levels: dict[str, dict[str, Any]],
    hot_vram_gb: float,
    metadata: dict[str, Any],
) -> ExpertPayload:
    validated_level = _validate_level(decode_level)
    data = {"format": "sdnc.expert_payload", "levels": _jsonable(levels)}
    metadata = _jsonable(metadata)
    body = _checksum_body(kind, validated_level, data, metadata)
    checksum = _checksum(body)
    byte_size = len(_canonical({**body, "checksum": checksum}))
    hot_ram_gb = max(round(byte_size / (1024**3), 9), 0.000001)
    return ExpertPayload(
        version=PAYLOAD_VERSION,
        kind=kind,
        decode_level=validated_level,
        data=data,
        checksum=checksum,
        byte_size=byte_size,
        decode_cost_ms=_estimate_decode_cost_ms(byte_size, validated_level),
        hot_ram_gb=hot_ram_gb,
        hot_vram_gb=round(float(hot_vram_gb), 9),
        metadata=metadata,
    )


def _checksum_body(
    kind: str,
    decode_level: str,
    data: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": PAYLOAD_VERSION,
        "kind": str(kind),
        "decode_level": _validate_level(decode_level),
        "data": _jsonable(data),
        "metadata": _jsonable(metadata),
    }


def _decode_level_data(data: dict[str, Any], requested: str) -> dict[str, Any]:
    levels = data.get("levels") if isinstance(data, dict) else None
    if not isinstance(levels, dict):
        return _jsonable(data)
    merged: dict[str, Any] = {}
    for level in DECODE_LEVELS[: DECODE_LEVELS.index(requested) + 1]:
        value = levels.get(level)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _available_levels(data: dict[str, Any]) -> list[str]:
    levels = data.get("levels") if isinstance(data, dict) else None
    if not isinstance(levels, dict):
        return []
    return [level for level in DECODE_LEVELS if level in levels]


def _canonical(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _checksum(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(data)).hexdigest()


def _validate_level(level: str) -> str:
    normalized = str(level).upper()
    if normalized not in DECODE_LEVELS:
        raise ValueError(f"unknown expert decode level: {level}")
    return normalized


def _estimate_decode_cost_ms(byte_size: int, level: str) -> float:
    level_multiplier = {"L0": 0.5, "L1": 1.0, "L2": 1.8}[_validate_level(level)]
    return round(0.02 + level_multiplier * byte_size / 750_000.0, 4)


def _sketch_vector(vector: np.ndarray | list[float] | None, top_k: int) -> dict[str, Any]:
    if vector is None:
        return {"dim": 0, "norm": 0.0, "density": 0.0, "top": []}
    array = _as_vector(vector)
    if array.size == 0:
        return {"dim": 0, "norm": 0.0, "density": 0.0, "top": []}
    abs_values = np.abs(array)
    count = min(int(top_k), int(array.size))
    top_indices = np.argpartition(abs_values, -count)[-count:]
    top_indices = sorted(top_indices.tolist(), key=lambda index: (-float(abs_values[index]), int(index)))
    nonzero = int(np.count_nonzero(array))
    return {
        "dim": int(array.size),
        "norm": round(float(np.linalg.norm(array)), 6),
        "density": round(nonzero / float(array.size), 6),
        "top": [
            {"index": int(index), "value": round(float(array[index]), 6)}
            for index in top_indices
        ],
    }


def _action_hints(payload: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for key in ("tool_name", "source", "claim", "recommendation"):
        if key in payload:
            hints[key] = payload[key]
    if "verdict" in payload:
        hints["verdict"] = payload["verdict"]
    return hints


def _as_vector(value: np.ndarray | list[float] | None) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _as_matrix(value: np.ndarray | list[list[float]]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def _round_array(value: np.ndarray) -> Any:
    rounded = np.round(np.asarray(value, dtype=np.float32), 6)
    return rounded.tolist()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _round_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
