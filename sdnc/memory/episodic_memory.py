"""Episodic memory: stores and retrieves past experiences.

Phase 1: in-memory cosine similarity search.
Phase 2+: PostgreSQL + pgvector for persistence.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class Episode:
    """A stored memory episode."""
    embedding: torch.Tensor
    label: int | None = None
    activation_pattern: torch.Tensor | None = None
    strength: float = 1.0


class EpisodicMemory:
    """Vector-based episodic memory with similarity retrieval.

    Stores embeddings of past experiences and retrieves
    the most similar ones via cosine similarity.
    """

    def __init__(self, embedding_dim: int, max_memories: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.episodes: list[Episode] = []

        # Cached tensor for efficient batch retrieval
        self._embeddings_cache: torch.Tensor | None = None
        self._cache_dirty = True

    def store(
        self,
        embedding: torch.Tensor,
        label: int | None = None,
        activation_pattern: torch.Tensor | None = None,
    ):
        """Store a new episodic memory.

        Args:
            embedding: (embedding_dim,) vector.
            label: Optional class label.
            activation_pattern: Which circuits were active.
        """
        episode = Episode(
            embedding=embedding.detach().cpu(),
            label=label,
            activation_pattern=activation_pattern.detach().cpu() if activation_pattern is not None else None,
        )
        self.episodes.append(episode)
        self._cache_dirty = True

        # Evict oldest if over capacity
        if len(self.episodes) > self.max_memories:
            self.episodes.pop(0)

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> list[Episode]:
        """Retrieve most similar episodes via cosine similarity.

        Args:
            query: (embedding_dim,) query vector.
            top_k: Number of results.

        Returns:
            List of most similar episodes.
        """
        if not self.episodes:
            return []

        self._rebuild_cache_if_needed(query.device)

        query_norm = query.detach() / (query.detach().norm() + 1e-8)
        similarities = self._embeddings_cache.to(query.device) @ query_norm

        k = min(top_k, len(self.episodes))
        _, indices = similarities.topk(k)

        return [self.episodes[i] for i in indices.tolist()]

    def consolidate(self, similarity_threshold: float = 0.95):
        """Merge very similar memories to save capacity.

        Episodes with cosine similarity > threshold are merged
        by averaging their embeddings and keeping the stronger one.
        """
        if len(self.episodes) < 2:
            return

        self._rebuild_cache_if_needed(self._embeddings_cache.device if self._embeddings_cache is not None else torch.device("cpu"))

        # Pairwise similarities
        sims = self._embeddings_cache @ self._embeddings_cache.T
        merged = set()
        new_episodes = []

        for i in range(len(self.episodes)):
            if i in merged:
                continue
            cluster = [i]
            for j in range(i + 1, len(self.episodes)):
                if j not in merged and sims[i, j] > similarity_threshold:
                    cluster.append(j)
                    merged.add(j)

            # Average the cluster
            avg_emb = torch.stack([self.episodes[idx].embedding for idx in cluster]).mean(0)
            ep = self.episodes[i]
            ep.embedding = avg_emb
            ep.strength = max(self.episodes[idx].strength for idx in cluster)
            new_episodes.append(ep)

        self.episodes = new_episodes
        self._cache_dirty = True

    def _rebuild_cache_if_needed(self, device: torch.device):
        if self._cache_dirty and self.episodes:
            embs = torch.stack([ep.embedding for ep in self.episodes])
            norms = embs.norm(dim=-1, keepdim=True) + 1e-8
            self._embeddings_cache = (embs / norms).to(device)
            self._cache_dirty = False

    def __len__(self) -> int:
        return len(self.episodes)
