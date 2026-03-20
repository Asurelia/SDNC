"""Text encoder: frozen CLIP text model for encoding text descriptions.

Uses the same CLIP model as VisionEncoder — both project into the
SAME 512-dim embedding space. This is the key to cross-modal transfer:
"a photo of a cat" and an image of a cat land near each other.
"""

import torch
import torch.nn as nn

from sdnc.config import SDNCConfig


class TextEncoder(nn.Module):
    """Frozen CLIP text encoder.

    Extracts normalized embeddings from text descriptions.
    Shares the embedding space with VisionEncoder — same CLIP model.
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load_model(self, device: torch.device):
        """Lazy-load CLIP model on first use."""
        if self._model is None:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                self.config.clip_model_name,
                pretrained=self.config.clip_pretrained,
            )
            self._model = model
            self._tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)

            for param in self._model.parameters():
                param.requires_grad = False

            self._model.eval()

        self._model.to(device)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model(torch.device("cpu"))
        return self._tokenizer

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """Tokenize text strings."""
        return self.tokenizer(texts)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokenized text to normalized embeddings.

        Args:
            text_tokens: (batch, token_len) from tokenize().

        Returns:
            Normalized embeddings (batch, input_dim) — same space as images.
        """
        self._load_model(text_tokens.device)

        with torch.no_grad():
            features = self._model.encode_text(text_tokens)

        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features.float()

    def encode_descriptions(self, descriptions: list[str], device: torch.device) -> torch.Tensor:
        """Convenience: text strings → normalized embeddings."""
        tokens = self.tokenize(descriptions).to(device)
        return self.forward(tokens)
