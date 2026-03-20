"""Signal encoder: MLP custom for continuous signals.

Encodes arbitrary continuous signals (mouse position, OS state,
timestamps, sensor data) into the unified SDNC embedding space.

Unlike frozen encoders (CLIP, Whisper), this encoder IS trainable —
it learns to project raw signals into meaningful representations
alongside the Hebbian circuit learning.
"""

import torch
import torch.nn as nn

from sdnc.config import SDNCConfig


class SignalEncoder(nn.Module):
    """Trainable MLP encoder for continuous signal inputs.

    Converts raw signal vectors (variable dim) → input_dim embeddings.
    Includes temporal encoding via sinusoidal position features.
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config

        self.mlp = nn.Sequential(
            nn.Linear(config.signal_input_dim, config.signal_hidden_dim),
            nn.GELU(),
            nn.Linear(config.signal_hidden_dim, config.signal_hidden_dim),
            nn.GELU(),
            nn.Linear(config.signal_hidden_dim, config.input_dim),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Encode signal vectors to normalized embeddings.

        Args:
            signals: (batch, signal_input_dim) raw signal values.

        Returns:
            Normalized embeddings (batch, input_dim).
        """
        features = self.mlp(signals)
        return features / (features.norm(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def make_context_signal(
        mouse_x: float = 0.0,
        mouse_y: float = 0.0,
        timestamp: float = 0.0,
        app_id: int = 0,
        extra: list[float] | None = None,
    ) -> torch.Tensor:
        """Build a signal vector from contextual inputs.

        Args:
            mouse_x, mouse_y: normalized mouse position [0, 1]
            timestamp: unix timestamp (will be normalized)
            app_id: application identifier (encoded as float)
            extra: additional signal values

        Returns:
            (1, signal_dim) tensor
        """
        vals = [mouse_x, mouse_y, timestamp % 86400 / 86400, float(app_id) / 100]
        if extra:
            vals.extend(extra)
        # Pad to signal_input_dim
        while len(vals) < 8:
            vals.append(0.0)
        return torch.tensor([vals[:8]], dtype=torch.float32)
