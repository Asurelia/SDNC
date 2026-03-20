"""Consolidation: decides what experiences are worth memorizing.

Not everything should be stored — only novel or important events.
Inspired by hippocampal replay and memory consolidation during sleep.
"""

import torch


class SalienceDetector:
    """Evaluates whether an experience is salient enough to store.

    Criteria:
    - High prediction error (novel / surprising)
    - High activation magnitude (strong response)
    - Dissimilarity to recent memories
    """

    def __init__(self, threshold: float = 0.7, history_size: int = 100):
        self.threshold = threshold
        self.history_size = history_size
        self.recent_scores: list[float] = []

    def evaluate(
        self,
        activation_magnitude: float,
        prediction_error: float | None = None,
        novelty_score: float | None = None,
    ) -> bool:
        """Decide if an experience should be stored.

        Args:
            activation_magnitude: How strongly circuits responded (0-1).
            prediction_error: Difference from expected output (optional).
            novelty_score: How different from recent memories (optional).

        Returns:
            True if the experience is worth storing.
        """
        score = activation_magnitude

        if prediction_error is not None:
            score = 0.5 * score + 0.5 * min(prediction_error, 1.0)

        if novelty_score is not None:
            score = 0.7 * score + 0.3 * novelty_score

        self.recent_scores.append(score)
        if len(self.recent_scores) > self.history_size:
            self.recent_scores.pop(0)

        # Adaptive threshold: relative to recent history
        adaptive_threshold = self.threshold
        if len(self.recent_scores) > 10:
            mean_score = sum(self.recent_scores) / len(self.recent_scores)
            adaptive_threshold = max(self.threshold, mean_score)

        return score > adaptive_threshold

    def compute_novelty(
        self, embedding: torch.Tensor, memory_embeddings: torch.Tensor | None
    ) -> float:
        """Compute novelty as inverse max similarity to existing memories."""
        if memory_embeddings is None or memory_embeddings.shape[0] == 0:
            return 1.0

        query_norm = embedding / (embedding.norm() + 1e-8)
        mem_norm = memory_embeddings / (memory_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        similarities = mem_norm @ query_norm
        max_sim = similarities.max().item()

        return 1.0 - max_sim
