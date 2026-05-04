"""Compressed experience packs for sparse recall.

Packs are lightweight prototypes built from many observations. They are closer
to game-engine assets than model weights: compact, indexed, and expanded only
when a similar situation needs them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sdnc.agent.multimodal import EncodedObservation


@dataclass
class ExperiencePrototype:
    """Centroid for a small cluster of similar observations."""

    key: str
    modality: str
    source: str
    count: int
    centroid: np.ndarray
    labels: dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def similarity(self, embedding: np.ndarray) -> float:
        query = _unit(embedding)
        return float(np.dot(_unit(self.centroid), query))


@dataclass
class ExperiencePack:
    """A compressed set of prototypes."""

    name: str
    prototypes: list[ExperiencePrototype]
    metadata: dict[str, Any] = field(default_factory=dict)

    def top(self, embedding: np.ndarray, k: int = 5) -> list[tuple[ExperiencePrototype, float]]:
        scored = [(prototype, prototype.similarity(embedding)) for prototype in self.prototypes]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"centroid_{index}": prototype.centroid for index, prototype in enumerate(self.prototypes)}
        manifest = {
            "name": self.name,
            "metadata": self.metadata,
            "prototypes": [
                {
                    "key": prototype.key,
                    "modality": prototype.modality,
                    "source": prototype.source,
                    "count": prototype.count,
                    "labels": prototype.labels,
                    "summary": prototype.summary,
                    "array": f"centroid_{index}",
                }
                for index, prototype in enumerate(self.prototypes)
            ],
        }
        arrays["manifest"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "ExperiencePack":
        data = np.load(path, allow_pickle=False)
        manifest = json.loads(bytes(data["manifest"].tolist()).decode("utf-8"))
        prototypes = []
        for item in manifest["prototypes"]:
            prototypes.append(
                ExperiencePrototype(
                    key=item["key"],
                    modality=item["modality"],
                    source=item["source"],
                    count=int(item["count"]),
                    centroid=np.asarray(data[item["array"]], dtype=np.float32),
                    labels=dict(item.get("labels", {})),
                    summary=item.get("summary", ""),
                )
            )
        return cls(name=manifest["name"], prototypes=prototypes, metadata=manifest.get("metadata", {}))


class ExperiencePackBuilder:
    """Accumulate observations into compact prototypes."""

    def __init__(self, name: str, similarity_threshold: float = 0.86):
        self.name = name
        self.similarity_threshold = similarity_threshold
        self.prototypes: list[ExperiencePrototype] = []

    def add(self, observation: EncodedObservation) -> ExperiencePrototype:
        label = observation.sample.label or "unlabeled"
        key = self._key(observation)
        compatible = [
            prototype
            for prototype in self.prototypes
            if prototype.modality == observation.sample.modality and prototype.source == observation.sample.source
        ]
        best = max(
            compatible,
            key=lambda prototype: prototype.similarity(observation.embedding),
            default=None,
        )
        if best is None or best.similarity(observation.embedding) < self.similarity_threshold:
            prototype = ExperiencePrototype(
                key=key,
                modality=observation.sample.modality,
                source=observation.sample.source,
                count=1,
                centroid=observation.embedding.copy(),
                labels={label: 1},
                summary=observation.summary[:240],
            )
            self.prototypes.append(prototype)
            return prototype

        total = best.count + 1
        best.centroid = _unit((best.centroid * best.count + observation.embedding) / total)
        best.count = total
        best.labels[label] = best.labels.get(label, 0) + 1
        return best

    def build(self, metadata: dict[str, Any] | None = None) -> ExperiencePack:
        return ExperiencePack(self.name, list(self.prototypes), metadata or {})

    def _key(self, observation: EncodedObservation) -> str:
        sample = observation.sample
        label = sample.label or "unlabeled"
        return f"{sample.source}:{sample.modality}:{label}:{len(self.prototypes)}"


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)
