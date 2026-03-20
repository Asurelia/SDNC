"""Temporal Stream — continuous input flow with history.

No more isolated image-by-image processing.
Every input joins a STREAM of recent context.

Circuits see the present AND the recent past simultaneously.
CfC states NEVER reset between examples — the stream is continuous.

Buffer circulaire des N derniers inputs — les circuits voient
le présent ET l'historique récent.
"""

import torch
import torch.nn as nn

from sdnc.config import SDNCConfig


class TemporalStream(nn.Module):
    """Circular buffer of recent inputs with temporal encoding.

    Maintains the last N inputs in a ring buffer.
    Provides a fused view (current + history) for circuit processing.

    The key insight: circuits don't just see one image —
    they see the current input IN CONTEXT of what came before.
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config
        dim = config.input_dim
        history_len = config.temporal_history_len

        # Ring buffer: (history_len, dim)
        self.register_buffer("buffer", torch.zeros(history_len, dim))
        self.register_buffer("write_idx", torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

        # Temporal position encoding — learnable
        self.position_encoding = nn.Embedding(history_len, dim)

        # Fusion: compress (current + history_summary) → single representation
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Temporal attention weights — learn which history steps matter
        self.temporal_attn = nn.Linear(dim, 1)

    def push(self, x: torch.Tensor) -> None:
        """Push new input into the stream buffer.

        Args:
            x: (batch, dim) or (dim,) — new input to record
        """
        # Mean across batch — buffer stores representative inputs
        if x.dim() > 1:
            x_repr = x.detach().mean(dim=0)
        else:
            x_repr = x.detach()

        idx = self.write_idx.item()
        self.buffer[idx] = x_repr
        self.write_idx = (self.write_idx + 1) % self.buffer.shape[0]
        self.filled = torch.clamp(self.filled + 1, max=self.buffer.shape[0])

    def get_history(self) -> torch.Tensor:
        """Get ordered history from oldest to newest.

        Returns:
            history: (filled_len, dim)
        """
        n = self.filled.item()
        if n == 0:
            return torch.zeros(1, self.config.input_dim, device=self.buffer.device)

        idx = self.write_idx.item()
        if n < self.buffer.shape[0]:
            # Not yet wrapped — just take [0:n]
            return self.buffer[:n]
        else:
            # Wrapped — reorder: [write_idx:] + [:write_idx]
            return torch.cat([self.buffer[idx:], self.buffer[:idx]], dim=0)

    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse current input with temporal history.

        Args:
            x: (batch, dim) — current input

        Returns:
            context: (batch, dim) — input enriched with temporal context
        """
        history = self.get_history()  # (T, dim)
        n = history.shape[0]

        # Add position encoding
        positions = torch.arange(n, device=x.device)
        # Clamp to embedding size
        positions = positions.clamp(max=self.position_encoding.num_embeddings - 1)
        pos_encoded = history + self.position_encoding(positions)

        # Temporal attention: which history steps matter now?
        attn_scores = self.temporal_attn(pos_encoded).squeeze(-1)  # (T,)
        attn_weights = torch.softmax(attn_scores, dim=0)  # (T,)

        # Weighted history summary
        history_summary = (attn_weights.unsqueeze(-1) * pos_encoded).sum(dim=0)  # (dim,)

        # Expand to batch and fuse with current input
        history_batch = history_summary.unsqueeze(0).expand(x.shape[0], -1)
        fused = self.fusion(torch.cat([x, history_batch], dim=-1))

        return fused

    def get_stream_stats(self) -> dict:
        """Diagnostic: stream state for logging."""
        n = self.filled.item()
        if n == 0:
            return {"filled": 0, "buffer_norm": 0.0}
        history = self.get_history()
        return {
            "filled": n,
            "buffer_norm": history.norm().item(),
            "buffer_mean": history.mean().item(),
            "buffer_std": history.std().item() if n > 1 else 0.0,
        }

    def reset(self):
        """Hard reset — only for completely new contexts."""
        self.buffer.zero_()
        self.write_idx.fill_(0)
        self.filled.fill_(0)
