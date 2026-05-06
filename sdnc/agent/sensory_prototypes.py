"""Prototype memory for repeated multimodal sensory events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sdnc.agent.config import AutonomousConfig
from sdnc.agent.memory import PersistentMemory, SensoryPrototypeRecord
from sdnc.agent.perception import SensoryEvent


@dataclass(frozen=True)
class SensoryPrototypeMatch:
    """A prototype matched against the current sensory event."""

    prototype: SensoryPrototypeRecord
    score: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.prototype.id,
            "key": self.prototype.key,
            "modalities": self.prototype.modalities,
            "summary": self.prototype.summary,
            "observation_count": self.prototype.observation_count,
            "confidence": self.prototype.confidence,
            "similarity": self.prototype.similarity,
            "score": self.score,
            "sources": self.prototype.sources,
            "sample_ids": self.prototype.sample_ids,
        }


class SensoryPrototypeLearner:
    """Maintain compact identity-like anchors for repeated sensory events."""

    def __init__(self, config: AutonomousConfig, memory: PersistentMemory):
        self.config = config
        self.memory = memory

    def match(self, event: SensoryEvent) -> list[SensoryPrototypeMatch]:
        matches: list[SensoryPrototypeMatch] = []
        for prototype in self.memory.retrieve_sensory_prototypes(
            event.embedding,
            top_k=self.config.sensory_prototype_top_k,
        ):
            score = prototype.similarity * 0.7 + prototype.confidence * 0.2
            score += min(prototype.observation_count, 10) * 0.01
            if prototype.similarity >= self.config.sensory_prototype_similarity:
                matches.append(SensoryPrototypeMatch(prototype=prototype, score=round(float(score), 6)))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches

    def learn(
        self,
        event: SensoryEvent,
        episode_id: str,
        matches: list[SensoryPrototypeMatch] | None = None,
    ) -> SensoryPrototypeRecord | None:
        if event.reliability < self.config.sensory_prototype_min_reliability:
            return None
        key = matches[0].prototype.key if matches else self._prototype_key(event)
        confidence = min(1.0, 0.45 + 0.4 * event.reliability + 0.15 * event.binding_score)
        prototype_id = self.memory.upsert_sensory_prototype(
            key=key,
            modalities=event.modalities,
            source=event.source,
            sample_ids=event.sample_ids,
            summary=event.summary[:700],
            embedding=event.embedding,
            confidence=confidence,
            features=self._prototype_features(event),
            payload={
                "last_event_id": event.id,
                "last_episode_id": episode_id,
                "binding_score": event.binding_score,
                "reliability": event.reliability,
            },
        )
        return next(
            (prototype for prototype in self.memory.recent_sensory_prototypes(limit=20) if prototype.id == prototype_id),
            None,
        )

    def _prototype_key(self, event: SensoryEvent) -> str:
        labels = sorted({signal.label.lower() for signal in event.signals if signal.label})
        if labels:
            topic = "-".join(labels[:4])
        else:
            words = re.findall(r"[A-Za-z0-9_]{3,}", event.summary.lower())
            stop = {
                "audio",
                "available",
                "event",
                "features",
                "image",
                "label",
                "modality",
                "modalities",
                "signal",
                "sensory",
                "source",
                "text",
                "video",
            }
            useful = [word for word in words if word not in stop and not word.isdigit()]
            topic = "-".join(useful[:6]) or "general"
        return "+".join(event.modalities) + ":" + topic

    def _prototype_features(self, event: SensoryEvent) -> dict[str, Any]:
        return {
            "modalities": event.modalities,
            "signal_count": len(event.signals),
            "binding_score": round(event.binding_score, 6),
            "reliability": round(event.reliability, 6),
            "labels": sorted({signal.label for signal in event.signals if signal.label}),
            "per_signal": [
                {
                    "modality": signal.modality,
                    "reliability": round(signal.reliability, 6),
                    "sample_id": signal.sample_id,
                }
                for signal in event.signals
            ],
        }
