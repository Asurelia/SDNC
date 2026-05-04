"""Local multimodal observation encoding for SDNC.

The goal is not to compete with CLIP or Whisper. This module gives SDNC a
small, deterministic sensory bridge that works before heavyweight frozen
encoders are installed. Stronger encoders can later replace individual
extractors without changing the sparse learning loop.
"""

from __future__ import annotations

import hashlib
import io
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sdnc.agent.encoding import HashingExperienceEncoder

SUPPORTED_MODALITIES = {"text", "image", "audio", "video"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class ModalitySample:
    """One learnable observation from any modality."""

    modality: str
    content: Any = None
    text: str = ""
    label: str | None = None
    source: str = "manual"
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"unsupported modality: {self.modality}")


@dataclass(frozen=True)
class EncodedObservation:
    """A modality sample transformed into the SDNC vector space."""

    sample: ModalitySample
    embedding: np.ndarray
    summary: str
    context: dict[str, Any]
    features: dict[str, Any]


class LocalMultimodalEncoder:
    """Encode text, image, audio, and video observations locally."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.text_encoder = HashingExperienceEncoder(dim)

    def encode_sample(self, sample: ModalitySample) -> EncodedObservation:
        features = self.extract_features(sample)
        summary = self.summarize(sample, features)
        context = {
            "modality": sample.modality,
            "source": sample.source,
            "sample_id": sample.sample_id or "",
            "label": sample.label or "",
            "features": _compact_features(features),
            **{f"meta_{key}": value for key, value in sample.metadata.items()},
        }
        embedding = self.text_encoder.encode(summary, context)
        embedding = self._blend_numeric_signature(embedding, sample.modality, features)
        return EncodedObservation(
            sample=sample,
            embedding=embedding,
            summary=summary,
            context=context,
            features=features,
        )

    def extract_features(self, sample: ModalitySample) -> dict[str, Any]:
        if sample.modality == "text":
            return self._text_features(sample)
        if sample.modality == "image":
            return self._image_features(sample.content)
        if sample.modality == "audio":
            return self._audio_features(sample.content, sample.metadata)
        if sample.modality == "video":
            return self._video_features(sample.content, sample.metadata)
        raise ValueError(f"unsupported modality: {sample.modality}")

    def summarize(self, sample: ModalitySample, features: dict[str, Any]) -> str:
        pieces = [f"modality={sample.modality}"]
        if sample.label:
            pieces.append(f"label={sample.label}")
        if sample.text:
            pieces.append(f"text={sample.text[:500]}")
        for key, value in sorted(_compact_features(features).items()):
            pieces.append(f"{key}={value}")
        return " | ".join(pieces)

    def _text_features(self, sample: ModalitySample) -> dict[str, Any]:
        text = sample.text or str(sample.content or "")
        tokens = self.text_encoder._tokens(text)
        return {
            "char_len": len(text),
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens)),
        }

    def _image_features(self, content: Any) -> dict[str, Any]:
        array = _image_to_array(content)
        if array is None:
            return {"available": False, "kind": type(content).__name__}
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.shape[-1] > 3:
            array = array[..., :3]
        array = np.asarray(array, dtype=np.float32)
        height, width = int(array.shape[0]), int(array.shape[1])
        flat = array.reshape(-1, array.shape[-1])
        scale = 255.0 if float(np.nanmax(flat)) > 1.5 else 1.0
        flat = np.clip(flat / scale, 0.0, 1.0)
        hist = []
        for channel in range(min(3, flat.shape[1])):
            bins, _ = np.histogram(flat[:, channel], bins=4, range=(0.0, 1.0))
            hist.extend((bins / max(1, flat.shape[0])).round(3).tolist())
        return {
            "available": True,
            "width": width,
            "height": height,
            "aspect": round(width / max(height, 1), 3),
            "mean": np.mean(flat, axis=0).round(3).tolist(),
            "std": np.std(flat, axis=0).round(3).tolist(),
            "hist4": hist,
        }

    def _audio_features(self, content: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        array, sampling_rate = _audio_to_array(content, metadata)
        if array is None:
            return {"available": False, "kind": type(content).__name__}
        array = np.asarray(array, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return {"available": False, "kind": "empty"}
        if np.nanmax(np.abs(array)) > 1.5:
            array = array / max(float(np.nanmax(np.abs(array))), 1.0)
        rms = float(np.sqrt(np.mean(np.square(array))))
        zero_crossings = float(np.mean(np.diff(np.signbit(array)) != 0)) if array.size > 1 else 0.0
        duration = float(array.size / sampling_rate) if sampling_rate else 0.0
        return {
            "available": True,
            "samples": int(array.size),
            "sampling_rate": int(sampling_rate or 0),
            "duration_s": round(duration, 3),
            "rms": round(rms, 4),
            "zero_crossing": round(zero_crossings, 4),
            "mean": round(float(np.mean(array)), 4),
            "std": round(float(np.std(array)), 4),
        }

    def _video_features(self, content: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        frames = _video_frames(content, metadata)
        if frames:
            image_features = [self._image_features(frame) for frame in frames[:4]]
            means = [
                feature.get("mean", [0.0, 0.0, 0.0])
                for feature in image_features
                if feature.get("available")
            ]
            motion = _frame_motion(frames[:4])
            return {
                "available": True,
                "frames_sampled": len(frames[:4]),
                "frame_mean": np.mean(np.asarray(means, dtype=np.float32), axis=0).round(3).tolist()
                if means
                else [0.0, 0.0, 0.0],
                "motion": round(motion, 4),
                "duration_s": round(float(metadata.get("duration_s", 0.0) or 0.0), 3),
                "fps": round(float(metadata.get("fps", 0.0) or 0.0), 3),
            }
        path = _path_from_content(content)
        return {
            "available": bool(path),
            "path_suffix": path.suffix.lower() if path else "",
            "duration_s": round(float(metadata.get("duration_s", 0.0) or 0.0), 3),
            "fps": round(float(metadata.get("fps", 0.0) or 0.0), 3),
            "frames": int(metadata.get("frames", 0) or 0),
        }

    def _blend_numeric_signature(
        self,
        embedding: np.ndarray,
        modality: str,
        features: dict[str, Any],
    ) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).copy()
        for key, value in _flatten_numeric_features(features).items():
            digest = hashlib.blake2b(f"{modality}:{key}".encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little", signed=False)
            index = raw % self.dim
            sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
            vec[index] += sign * float(value) * 0.12
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-8)


def samples_from_record(
    record: dict[str, Any],
    source: str,
    sample_id: str | None = None,
    text_columns: Iterable[str] | None = None,
    label_columns: Iterable[str] | None = None,
) -> list[ModalitySample]:
    """Infer one or more modality samples from a dataset row."""

    text_columns = list(text_columns or [])
    label_columns = list(label_columns or [])
    text = _joined_text(record, text_columns)
    label = _first_label(record, label_columns)
    samples: list[ModalitySample] = []

    for key, value in record.items():
        modality = infer_modality(key, value)
        if modality and modality != "text":
            samples.append(
                ModalitySample(
                    modality=modality,
                    content=value,
                    text=text,
                    label=label,
                    source=source,
                    sample_id=sample_id,
                    metadata={"column": key},
                )
            )

    if text or not samples:
        samples.append(
            ModalitySample(
                modality="text",
                content=text,
                text=text,
                label=label,
                source=source,
                sample_id=sample_id,
                metadata={"columns": text_columns or _likely_text_keys(record)},
            )
        )
    return samples


def infer_modality(key: str, value: Any) -> str | None:
    lowered = key.lower()
    path = _path_from_content(value)
    suffix = path.suffix.lower() if path else ""
    if "audio" in lowered or suffix in AUDIO_EXTENSIONS or _looks_like_audio_dict(value):
        return "audio"
    if "video" in lowered or suffix in VIDEO_EXTENSIONS:
        return "video"
    if "image" in lowered or "img" in lowered or suffix in IMAGE_EXTENSIONS or _looks_like_image(value):
        return "image"
    if isinstance(value, str) and lowered in {"text", "caption", "prompt", "question", "answer", "title"}:
        return "text"
    return None


def _image_to_array(content: Any) -> np.ndarray | None:
    if isinstance(content, np.ndarray):
        return content
    if isinstance(content, dict):
        if "array" in content and isinstance(content["array"], np.ndarray):
            return content["array"]
        if content.get("bytes"):
            return _pil_bytes_to_array(content["bytes"])
        if content.get("path"):
            return _pil_path_to_array(Path(content["path"]))
    path = _path_from_content(content)
    if path:
        return _pil_path_to_array(path)
    if hasattr(content, "convert") and hasattr(content, "size"):
        try:
            return np.asarray(content.convert("RGB"))
        except Exception:
            return None
    return None


def _audio_to_array(content: Any, metadata: dict[str, Any]) -> tuple[np.ndarray | None, int | None]:
    sampling_rate = int(metadata.get("sampling_rate") or metadata.get("sample_rate") or 0) or None
    if isinstance(content, dict):
        if "sampling_rate" in content:
            sampling_rate = int(content["sampling_rate"])
        if "array" in content:
            return np.asarray(content["array"], dtype=np.float32), sampling_rate
        if content.get("path"):
            return _wav_path_to_array(Path(content["path"]))
    if isinstance(content, np.ndarray):
        return content, sampling_rate
    path = _path_from_content(content)
    if path and path.suffix.lower() == ".wav":
        return _wav_path_to_array(path)
    return None, sampling_rate


def _video_frames(content: Any, metadata: dict[str, Any]) -> list[Any]:
    if isinstance(content, dict):
        for key in ("frames", "images"):
            value = content.get(key)
            if isinstance(value, list):
                return value
    if isinstance(content, list):
        return content
    if "frames" in metadata and isinstance(metadata["frames"], list):
        return metadata["frames"]
    return []


def _frame_motion(frames: list[Any]) -> float:
    arrays = [_image_to_array(frame) for frame in frames]
    arrays = [np.asarray(array, dtype=np.float32) for array in arrays if array is not None]
    if len(arrays) < 2:
        return 0.0
    scores = []
    for left, right in zip(arrays, arrays[1:]):
        if left.shape != right.shape:
            continue
        scores.append(float(np.mean(np.abs(left - right))) / 255.0)
    return float(np.mean(scores)) if scores else 0.0


def _wav_path_to_array(path: Path) -> tuple[np.ndarray | None, int | None]:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if width == 1:
            dtype = np.uint8
            audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            return None, rate
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, rate
    except Exception:
        return None, None


def _pil_path_to_array(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception:
        return None


def _pil_bytes_to_array(data: bytes) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))
    except Exception:
        return None


def _path_from_content(content: Any) -> Path | None:
    if isinstance(content, Path):
        return content
    if isinstance(content, str):
        path = Path(content)
        if path.suffix:
            return path
    return None


def _looks_like_audio_dict(value: Any) -> bool:
    return isinstance(value, dict) and "array" in value and "sampling_rate" in value


def _looks_like_image(value: Any) -> bool:
    if isinstance(value, np.ndarray) and value.ndim in {2, 3}:
        return True
    return hasattr(value, "convert") and hasattr(value, "size")


def _joined_text(record: dict[str, Any], text_columns: list[str]) -> str:
    keys = text_columns or _likely_text_keys(record)
    chunks = []
    for key in keys:
        value = record.get(key)
        if value is not None:
            chunks.append(str(value))
    return " ".join(chunks).strip()


def _likely_text_keys(record: dict[str, Any]) -> list[str]:
    preferred = ["text", "caption", "prompt", "question", "answer", "title", "description"]
    return [key for key in preferred if isinstance(record.get(key), str)]


def _first_label(record: dict[str, Any], label_columns: list[str]) -> str | None:
    keys = label_columns or ["label", "labels", "class", "category", "target"]
    for key in keys:
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def _compact_features(features: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, float):
            compact[key] = round(value, 4)
        elif isinstance(value, list):
            compact[key] = ",".join(str(round(float(item), 3)) for item in value[:12])
        else:
            compact[key] = value
    return compact


def _flatten_numeric_features(features: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in features.items():
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = _squash(float(value))
        elif isinstance(value, list):
            for index, item in enumerate(value[:24]):
                try:
                    out[f"{key}_{index}"] = _squash(float(item))
                except (TypeError, ValueError):
                    continue
    return out


def _squash(value: float) -> float:
    return math.tanh(value / 10.0)
