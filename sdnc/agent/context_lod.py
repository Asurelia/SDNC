"""Context compression with level-of-detail summaries.

Transformers usually keep long context as an ever-growing KV cache. SDNC keeps a
small routing summary plus sparse prototypes, and only reopens details when the
task needs them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sdnc.agent.budget import CognitiveBudget
from sdnc.agent.encoding import HashingExperienceEncoder


@dataclass(frozen=True)
class ContextSegment:
    """One chunk of original context."""

    index: int
    text: str
    summary: str
    embedding: np.ndarray
    weight: float


@dataclass(frozen=True)
class ContextPrototype:
    """Compressed centroid over related context segments."""

    indices: tuple[int, ...]
    centroid: np.ndarray
    summary: str
    weight: float


@dataclass(frozen=True)
class ContextPacket:
    """Level-of-detail context representation for one interaction."""

    original_chars: int
    routing_text: str
    global_summary: str
    segments: tuple[ContextSegment, ...]
    prototypes: tuple[ContextPrototype, ...]
    compression_ratio: float
    estimated_tokens_saved: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextLODCompressor:
    """Build compact routing context before SDNC activates circuits."""

    def __init__(self, encoder: HashingExperienceEncoder, similarity_threshold: float = 0.82):
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold

    def compress(
        self,
        text: str,
        context: dict[str, Any] | None,
        budget: CognitiveBudget,
    ) -> ContextPacket:
        text = text or ""
        context = dict(context or {})
        segment_texts = self._split(text, max_chars=max(200, budget.max_context_chars // max(1, budget.max_context_segments)))
        segment_texts = segment_texts[: budget.max_context_segments]
        segments = []
        for index, segment in enumerate(segment_texts):
            summary = self._summarize_segment(segment)
            embedding = self.encoder.encode(summary, {"segment": index, "mode": budget.mode, **context})
            weight = self._segment_weight(segment, index)
            segments.append(ContextSegment(index, segment, summary, embedding, weight))

        prototypes = self._build_prototypes(segments)
        global_summary = self._global_summary(text, prototypes, budget)
        routing_text = self._routing_text(text, global_summary, prototypes, budget)
        compressed_chars = len(routing_text) + sum(len(proto.summary) for proto in prototypes)
        ratio = compressed_chars / max(len(text), 1)
        return ContextPacket(
            original_chars=len(text),
            routing_text=routing_text,
            global_summary=global_summary,
            segments=tuple(segments),
            prototypes=tuple(prototypes),
            compression_ratio=round(float(ratio), 4),
            estimated_tokens_saved=max(0, int((len(text) - compressed_chars) / 4)),
            metadata={
                "mode": budget.mode,
                "segment_count": len(segments),
                "prototype_count": len(prototypes),
                "max_context_chars": budget.max_context_chars,
            },
        )

    def _split(self, text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs or [text]:
            if len(current) + len(paragraph) + 2 <= max_chars:
                current = f"{current}\n\n{paragraph}".strip()
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= max_chars:
                current = paragraph
            else:
                chunks.extend(paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars))
                current = ""
        if current:
            chunks.append(current)
        return chunks or [text[:max_chars]]

    def _summarize_segment(self, text: str) -> str:
        clean = " ".join(text.split())
        if len(clean) <= 240:
            return clean
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        head = sentences[0] if sentences else clean[:180]
        keywords = self._keywords(clean, limit=8)
        return f"{head[:180]} | keywords: {', '.join(keywords)}"

    def _global_summary(
        self,
        text: str,
        prototypes: list[ContextPrototype],
        budget: CognitiveBudget,
    ) -> str:
        clean = " ".join(text.split())
        if len(clean) <= min(480, budget.max_context_chars):
            return clean
        keywords = self._keywords(clean, limit=12)
        proto_bits = "; ".join(proto.summary[:120] for proto in prototypes[:4])
        return f"{clean[:240]} ... | keywords: {', '.join(keywords)} | prototypes: {proto_bits}"

    def _routing_text(
        self,
        text: str,
        global_summary: str,
        prototypes: list[ContextPrototype],
        budget: CognitiveBudget,
    ) -> str:
        if len(text) <= budget.max_context_chars:
            return text
        proto_text = " ".join(proto.summary for proto in prototypes[: budget.max_context_segments])
        return f"{global_summary}\n\n[context_lod:{budget.mode}] {proto_text}"[: budget.max_context_chars]

    def _build_prototypes(self, segments: list[ContextSegment]) -> list[ContextPrototype]:
        prototypes: list[ContextPrototype] = []
        for segment in segments:
            best_index = -1
            best_score = -1.0
            for index, prototype in enumerate(prototypes):
                score = float(np.dot(self._unit(segment.embedding), self._unit(prototype.centroid)))
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index >= 0 and best_score >= self.similarity_threshold:
                existing = prototypes[best_index]
                count = len(existing.indices)
                centroid = self._unit((existing.centroid * count + segment.embedding) / (count + 1))
                summary = self._merge_summary(existing.summary, segment.summary)
                prototypes[best_index] = ContextPrototype(
                    indices=(*existing.indices, segment.index),
                    centroid=centroid,
                    summary=summary,
                    weight=max(existing.weight, segment.weight),
                )
            else:
                prototypes.append(
                    ContextPrototype(
                        indices=(segment.index,),
                        centroid=segment.embedding.copy(),
                        summary=segment.summary,
                        weight=segment.weight,
                    )
                )
        prototypes.sort(key=lambda item: item.weight, reverse=True)
        return prototypes

    def _segment_weight(self, text: str, index: int) -> float:
        lowered = text.lower()
        score = 1.0 / (1 + index * 0.15)
        if any(marker in lowered for marker in ["erreur", "bug", "test", "important", "objectif", "question"]):
            score += 0.4
        return float(score)

    def _keywords(self, text: str, limit: int) -> list[str]:
        words = re.findall(r"[A-Za-zÀ-ÿ0-9_]{4,}", text.lower())
        counts: dict[str, int] = {}
        for word in words:
            if word in {"avec", "pour", "dans", "comme", "plus", "mais", "donc", "cette", "cela"}:
                continue
            counts[word] = counts.get(word, 0) + 1
        return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]

    def _merge_summary(self, left: str, right: str) -> str:
        if right in left:
            return left
        return f"{left[:160]} / {right[:160]}"

    def _unit(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)
