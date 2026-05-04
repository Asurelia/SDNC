"""Sensory binding layer for SDNC.

This module keeps multimodal perception in SDNC's shape: small sensory
signals are encoded, bound into an event, stored in memory, and routed through
the sparse learner. It is not a transformer-style "eat everything" context.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from time import time
from typing import Any, Iterable

import numpy as np

from sdnc.agent.encoding import HashingExperienceEncoder
from sdnc.agent.multimodal import EncodedObservation, LocalMultimodalEncoder, ModalitySample


@dataclass(frozen=True)
class SensorySignal:
    """One encoded sensory stream inside a bound event."""

    modality: str
    summary: str
    embedding: np.ndarray
    features: dict[str, Any]
    reliability: float
    source: str
    sample_id: str
    label: str


@dataclass(frozen=True)
class SensoryEvent:
    """A cross-modal event routed into sparse SDNC circuits."""

    id: str
    timestamp: float
    source: str
    summary: str
    modalities: list[str]
    sample_ids: list[str]
    signals: list[SensorySignal]
    embedding: np.ndarray
    binding_score: float
    reliability: float
    context: dict[str, Any]


class PerceptionBus:
    """Bind text/image/audio/video samples into compact SDNC events."""

    def __init__(
        self,
        multimodal_encoder: LocalMultimodalEncoder,
        event_encoder: HashingExperienceEncoder,
    ):
        self.multimodal_encoder = multimodal_encoder
        self.event_encoder = event_encoder

    def observe(
        self,
        samples: Iterable[ModalitySample],
        context: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> SensoryEvent:
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("at least one sensory sample is required")

        encoded = [self.multimodal_encoder.encode_sample(sample) for sample in sample_list]
        signals = [self._signal(item) for item in encoded]
        modalities = _stable_unique(signal.modality for signal in signals)
        sample_ids = _stable_unique(signal.sample_id for signal in signals if signal.sample_id)
        source = _source_name(signals)
        event_context = {
            **dict(context or {}),
            "modalities": "+".join(modalities),
            "source": source,
            "sample_ids": ",".join(sample_ids),
            "signal_count": len(signals),
        }
        summary = self._summary(signals)
        embedding = self._fuse(signals, summary, event_context)
        binding_score = self._binding_score(signals)
        reliability = self._event_reliability(signals, binding_score)

        return SensoryEvent(
            id=event_id or str(uuid.uuid4()),
            timestamp=time(),
            source=source,
            summary=summary,
            modalities=modalities,
            sample_ids=sample_ids,
            signals=signals,
            embedding=embedding,
            binding_score=binding_score,
            reliability=reliability,
            context=event_context,
        )

    def _signal(self, encoded: EncodedObservation) -> SensorySignal:
        sample = encoded.sample
        return SensorySignal(
            modality=sample.modality,
            summary=encoded.summary,
            embedding=encoded.embedding,
            features=encoded.features,
            reliability=_sample_reliability(sample, encoded.features),
            source=sample.source,
            sample_id=sample.sample_id or "",
            label=sample.label or "",
        )

    def _summary(self, signals: list[SensorySignal]) -> str:
        head = "sensory_event modalities=" + "+".join(_stable_unique(signal.modality for signal in signals))
        labels = _stable_unique(signal.label for signal in signals if signal.label)
        if labels:
            head += " labels=" + ",".join(labels[:4])
        pieces = [head]
        for signal in signals[:8]:
            compact = signal.summary.replace("\n", " ")[:260]
            pieces.append(f"{signal.modality}[r={signal.reliability:.2f}] {compact}")
        return " | ".join(pieces)

    def _fuse(
        self,
        signals: list[SensorySignal],
        summary: str,
        context: dict[str, Any],
    ) -> np.ndarray:
        weights = np.asarray([signal.reliability for signal in signals], dtype=np.float32)
        weights = weights / max(float(np.sum(weights)), 1e-8)
        stacked = np.stack([signal.embedding for signal in signals], axis=0).astype(np.float32)
        sensory_vector = np.sum(stacked * weights[:, None], axis=0)
        event_vector = self.event_encoder.encode(summary, context).astype(np.float32)
        fused = 0.82 * sensory_vector + 0.18 * event_vector
        norm = float(np.linalg.norm(fused))
        return fused / max(norm, 1e-8)

    def _binding_score(self, signals: list[SensorySignal]) -> float:
        if len(signals) == 1:
            return signals[0].reliability
        scores = []
        for left_index, left in enumerate(signals):
            for right in signals[left_index + 1 :]:
                similarity = float(np.dot(_unit(left.embedding), _unit(right.embedding)))
                agreement = (similarity + 1.0) * 0.5
                weight = min(left.reliability, right.reliability)
                scores.append(agreement * weight)
        return float(np.clip(np.mean(scores) if scores else 0.0, 0.0, 1.0))

    def _event_reliability(self, signals: list[SensorySignal], binding_score: float) -> float:
        mean_signal = float(np.mean([signal.reliability for signal in signals]))
        diversity_bonus = min(0.12, 0.04 * max(0, len(_stable_unique(signal.modality for signal in signals)) - 1))
        return float(np.clip(0.72 * mean_signal + 0.28 * binding_score + diversity_bonus, 0.0, 1.0))


def _sample_reliability(sample: ModalitySample, features: dict[str, Any]) -> float:
    base = 0.35
    if sample.text:
        base += 0.12
    if sample.label:
        base += 0.08
    if sample.modality == "text":
        token_count = int(features.get("token_count", 0) or 0)
        base += min(0.35, token_count / 120.0)
    elif bool(features.get("available", False)):
        base += 0.36
        if sample.modality == "image" and features.get("width") and features.get("height"):
            base += 0.08
        if sample.modality == "audio" and features.get("sampling_rate"):
            base += 0.08
        if sample.modality == "video" and (features.get("frames_sampled") or features.get("frames")):
            base += 0.08
    else:
        base += 0.06
    return float(np.clip(base, 0.05, 1.0))


def _source_name(signals: list[SensorySignal]) -> str:
    sources = _stable_unique(signal.source for signal in signals if signal.source)
    return "+".join(sources[:4]) if sources else "manual"


def _stable_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)
