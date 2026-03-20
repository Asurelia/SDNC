"""Vision encoder: frozen CLIP model for extracting visual features.

The encoder is NOT trained — it serves as a translator from
pixel space to the SDNC embedding space. All learning happens
in the LNN circuits, not here.
"""

import torch
import torch.nn as nn

from sdnc.config import SDNCConfig


class VisionEncoder(nn.Module):
    """Frozen CLIP visual encoder.

    Extracts normalized embeddings from images.
    All parameters are frozen — no gradient flows through this module.
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self, device: torch.device):
        """Lazy-load CLIP model on first use, move to device if needed."""
        if self._model is None:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model_name,
                pretrained=self.config.clip_pretrained,
            )
            self._model = model.visual
            self._preprocess = preprocess

            # Freeze all parameters
            for param in self._model.parameters():
                param.requires_grad = False

            self._model.eval()

        # Always ensure model is on the right device
        self._model.to(device)

    @property
    def preprocess(self):
        """Get the preprocessing transform for images."""
        if self._preprocess is None:
            self._load_model(torch.device("cpu"))
        return self._preprocess

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to normalized embeddings.

        Args:
            images: Preprocessed image tensor (batch, C, H, W).

        Returns:
            Normalized embeddings (batch, input_dim).
        """
        self._load_model(images.device)

        with torch.no_grad():
            features = self._model(images)

        # L2 normalize
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features.float()
