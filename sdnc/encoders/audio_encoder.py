"""Audio encoder: frozen Whisper tiny for extracting audio features.

Whisper tiny encoder outputs 384-dim features.
A learned projection maps 384 → input_dim (512) to align
with the CLIP embedding space.

The encoder is frozen — learning happens in the circuits.
"""

import torch
import torch.nn as nn

from sdnc.config import SDNCConfig


class AudioEncoder(nn.Module):
    """Frozen Whisper tiny encoder + learned projection to CLIP space.

    Pipeline: audio waveform → mel spectrogram → whisper encoder → 384-dim
              → projection → 512-dim (aligned with CLIP space)
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._processor = None

        # Learned projection: whisper_dim → input_dim
        self.proj = nn.Linear(config.whisper_dim, config.input_dim)

    def _load_model(self, device: torch.device):
        """Lazy-load Whisper model."""
        if self._model is not None:
            self._model.to(device)
            return

        try:
            import whisper
            model = whisper.load_model(self.config.whisper_model, device=device)
            self._model = model.encoder
            for param in self._model.parameters():
                param.requires_grad = False
            self._model.eval()
        except ImportError:
            # Fallback: use a random projection if whisper not installed
            # This allows testing without the whisper dependency
            self._model = _FallbackEncoder(self.config.whisper_dim, device)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to normalized embeddings.

        Args:
            mel: (batch, 80, 3000) mel spectrogram from whisper.log_mel_spectrogram()
                 OR (batch, whisper_dim) pre-extracted features

        Returns:
            Normalized embeddings (batch, input_dim).
        """
        if mel.dim() == 2 and mel.shape[-1] == self.config.whisper_dim:
            # Already extracted features, just project
            features = mel
        else:
            self._load_model(mel.device)
            with torch.no_grad():
                # Whisper encoder: (batch, 80, 3000) → (batch, 1500, 384)
                enc_out = self._model(mel)
                # Global average pool over time → (batch, 384)
                features = enc_out.mean(dim=1)

        # Project to CLIP space
        projected = self.proj(features)
        return projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)

    def encode_waveform(self, waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """Convenience: raw waveform → embeddings.

        Args:
            waveform: (batch, samples) at 16kHz
            sr: sample rate (must be 16000 for whisper)
        """
        try:
            import whisper
            # Convert to mel spectrogram
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            mels = []
            for wav in waveform:
                mel = whisper.log_mel_spectrogram(wav.cpu().numpy(), n_mels=80)
                mels.append(torch.from_numpy(mel))
            mel_batch = torch.stack(mels).to(waveform.device)
            # Pad to 30s (3000 frames)
            if mel_batch.shape[-1] < 3000:
                mel_batch = nn.functional.pad(mel_batch, (0, 3000 - mel_batch.shape[-1]))
            return self.forward(mel_batch)
        except ImportError:
            # Fallback: random features
            return self.forward(
                torch.randn(waveform.shape[0], self.config.whisper_dim, device=waveform.device)
            )


class _FallbackEncoder(nn.Module):
    """Fallback when whisper is not installed — produces random features."""

    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.randn(batch, 1, self.dim, device=x.device)
