"""Lightweight local encoders for interaction data.

The autonomous loop needs to work before CLIP, Whisper, or a language model
is downloaded. This encoder uses stable signed feature hashing over words,
character n-grams, and small context fields. It is deterministic, cheap, and
learns downstream through local circuit plasticity.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


class HashingExperienceEncoder:
    """Deterministic sparse text/context encoder."""

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode(self, text: str, context: dict[str, Any] | None = None) -> np.ndarray:
        """Encode text plus optional structured context into a unit vector."""
        vec = np.zeros(self.dim, dtype=np.float32)
        text = text or ""

        for token in self._tokens(text):
            self._add_feature(vec, f"tok:{token}", 1.0)
            if len(token) >= 4:
                for gram in self._char_ngrams(token):
                    self._add_feature(vec, f"chr:{gram}", 0.35)

        if context:
            flattened = self._flatten_context(context)
            for key, value in flattened.items():
                self._add_feature(vec, f"ctx:{key}", 0.6)
                self._add_feature(vec, f"ctx:{key}={value}", 0.8)

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            self._add_feature(vec, "empty", 1.0)
            norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-8)

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left, right))

    def _tokens(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]

    def _char_ngrams(self, token: str, n: int = 3) -> list[str]:
        padded = f"^{token}$"
        return [padded[i : i + n] for i in range(max(0, len(padded) - n + 1))]

    def _add_feature(self, vec: np.ndarray, feature: str, value: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little", signed=False)
        index = raw % self.dim
        sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
        vec[index] += sign * value

    def _flatten_context(self, context: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in sorted(context.items()):
            safe_key = str(key).lower().strip()
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[safe_key] = str(value).lower()
            else:
                out[safe_key] = json.dumps(value, sort_keys=True, default=str)[:200]
        return out
