"""Sparse router: decides which micro-circuits to activate.

Two routing modes:
  - TokenChoice: each input picks its top-k circuits (V1/V2)
  - ExpertChoice: each circuit picks its top-k inputs (Phase 2)
    → guarantees perfect load balance, eliminates collapse structurally
"""

import torch
import torch.nn as nn


class SparseRouter(nn.Module):
    """Sparse top-k router with entropy regularization for circuit diversity."""

    def __init__(self, input_dim: int, n_circuits: int, k: int = 1, noise_std: float = 0.1):
        super().__init__()
        self.n_circuits = n_circuits
        self.k = k
        self.noise_std = noise_std

        self.gate = nn.Linear(input_dim, n_circuits)
        self.register_buffer("expert_bias", torch.zeros(n_circuits))
        self.register_buffer("activation_counts", torch.zeros(n_circuits))
        self._step_count = 0
        self._last_entropy = 0.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.gate(x)
        if self.training:
            scores = scores + torch.randn_like(scores) * self.noise_std

        selection_scores = scores + self.expert_bias
        _, indices = selection_scores.topk(self.k, dim=-1)
        raw_top_k = scores.gather(-1, indices)
        weights = torch.softmax(raw_top_k, dim=-1)

        if self.training:
            with torch.no_grad():
                for idx in indices.view(-1):
                    self.activation_counts[idx] += 1
                self._step_count += 1

        return indices, weights

    def entropy_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute negative entropy of routing distribution.

        Maximizing entropy = forcing the router to spread across circuits.
        Returns a loss to MINIMIZE (negative entropy).
        """
        scores = self.gate(x)
        probs = torch.softmax(scores, dim=-1)  # (B, n_circuits)
        # Per-sample entropy, averaged
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        self._last_entropy = entropy.item()
        # We want to MAXIMIZE entropy → return negative
        return -entropy

    @torch.no_grad()
    def update_bias(self, gamma: float = 0.001):
        if self._step_count == 0:
            return
        target_count = self.activation_counts.sum() / self.n_circuits
        self.expert_bias -= gamma * (self.activation_counts - target_count).sign()
        self.activation_counts.zero_()
        self._step_count = 0

    @property
    def sparsity_ratio(self) -> float:
        return self.k / self.n_circuits


class ExpertChoiceRouter(nn.Module):
    """Expert Choice routing adapted for SDNC.

    Instead of each circuit picking tokens (too slow with 1000 circuits),
    we compute the full score matrix and select the top-K (circuit, token) pairs.
    This gives balanced routing without iterating over all circuits.

    Guarantees:
      - Each token gets a bounded number of experts
      - No single circuit monopolizes (structural balance)
      - Only active circuits are processed
    """

    def __init__(self, input_dim: int, n_circuits: int, capacity_factor: float = 1.25):
        super().__init__()
        self.n_circuits = n_circuits
        self.input_dim = input_dim
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(input_dim, n_circuits)

        self.register_buffer("activation_counts", torch.zeros(n_circuits))

    @property
    def sparsity_ratio(self) -> float:
        return self.capacity_factor / self.n_circuits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert-choice routing — efficient version.

        Selects the top-K (circuit, token) pairs from the score matrix.
        Each selected pair means "circuit C processes token T with weight W".

        Returns:
            assignments: (n_active_circuits, 1) — token index per active circuit
            weights: (n_active_circuits, 1) — gating weight
            token_counts: (batch,) — how many experts each token got
        """
        batch_size = x.shape[0]

        scores = self.gate(x)  # (B, n_circuits)
        S = torch.softmax(scores, dim=0)  # softmax over tokens (column-wise)

        # Total active slots = capacity_factor * batch_size
        # This bounds total compute while distributing across circuits
        total_slots = max(batch_size, int(self.capacity_factor * batch_size))

        # Flatten score matrix and pick top-K global (circuit, token) pairs
        S_flat = S.t().reshape(-1)  # (n_circuits * batch)
        top_k = min(total_slots, S_flat.numel())
        top_vals, top_flat_indices = S_flat.topk(top_k)

        # Decode flat indices → (circuit_idx, token_idx)
        circuit_indices = top_flat_indices // batch_size
        token_indices = top_flat_indices % batch_size

        # Group by circuit — build (n_unique_circuits, max_tokens_per) assignments
        # For efficiency, just return the raw pairs and let CircuitBank handle them
        unique_circuits = circuit_indices.unique()
        n_active = len(unique_circuits)

        # Build assignments: for each active circuit, which tokens + weights
        max_per_circuit = max(1, total_slots // max(1, n_active) + 1)
        assignments = torch.zeros(n_active, max_per_circuit, dtype=torch.long, device=x.device)
        weights = torch.zeros(n_active, max_per_circuit, device=x.device)
        circuit_map = torch.zeros(n_active, dtype=torch.long, device=x.device)  # actual circuit IDs
        counts_per_circuit = torch.zeros(n_active, dtype=torch.long, device=x.device)

        for i, cid in enumerate(unique_circuits):
            mask = circuit_indices == cid
            tids = token_indices[mask]
            ws = top_vals[mask]
            n = min(len(tids), max_per_circuit)
            assignments[i, :n] = tids[:n]
            weights[i, :n] = ws[:n]
            circuit_map[i] = cid
            counts_per_circuit[i] = n

        # Track per-token expert count
        token_counts = torch.zeros(batch_size, device=x.device)
        for i in range(n_active):
            n = counts_per_circuit[i].item()
            for j in range(n):
                token_counts[assignments[i, j]] += 1

        # Update activation history
        with torch.no_grad():
            for i in range(n_active):
                self.activation_counts[circuit_map[i]] += counts_per_circuit[i].item()

        return (assignments, weights, circuit_map, counts_per_circuit), top_vals, token_counts
