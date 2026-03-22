import torch
import torch.nn as nn
from typing import Optional


class InjectionGate(nn.Module):
    """
    Gate apprennable pour l'injection CfC → hidden states Qwen.

    Implémente : delta = alpha * up_proj(tanh(down_proj(norm(prediction))))

    - Low-rank projection (4096→rank→4096) pour adapter l'espace CfC à Qwen
    - alpha scalaire initialisé près de zéro (démarrage safe)
    - tanh borne la sortie pour la stabilité
    """

    def __init__(self, hidden_dim: int = 4096, rank: int = 16, init_alpha: float = 0.001):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cfc_prediction: torch.Tensor) -> torch.Tensor:
        """Retourne le delta à ajouter aux hidden states."""
        x = self.norm(cfc_prediction.float())
        x = self.up_proj(torch.tanh(self.down_proj(x)))
        return self.alpha * x


class BrainHookManager:
    """
    Gère le cycle de vie des forward hooks sur les couches Qwen.

    Enregistre les hooks avant generate(), les supprime après.
    Chaque hook injecte la prédiction CfC comme résidu.
    """

    def __init__(self):
        self.hook_handles = []

    def register_hooks(self, qwen_model, brain_modules, injection_gates, layer_indices):
        """
        Installe les hooks sur les couches Qwen.

        qwen_model    : le modèle Qwen (Qwen3VLForConditionalGeneration)
        brain_modules : liste de BrainModule
        injection_gates : liste de InjectionGate
        layer_indices : indices des couches (0-indexed), ex: [8, 17, 26, 35]
        """
        self.remove_hooks()

        # Trouver le chemin vers les decoder layers
        layers = self._get_decoder_layers(qwen_model)

        for i, layer_idx in enumerate(layer_indices):
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            module = brain_modules[i]
            gate = injection_gates[i]

            def make_hook(mod, gt):
                def hook_fn(layer_module, layer_input, layer_output):
                    # layer_output peut être un tuple ou un tensor
                    if isinstance(layer_output, tuple):
                        hidden_states = layer_output[0]
                        rest = layer_output[1:]
                    else:
                        hidden_states = layer_output
                        rest = None

                    with torch.no_grad():
                        prediction, _ = mod(hidden_states)
                    delta = gt(prediction)
                    modified = hidden_states + delta.to(hidden_states.dtype)

                    if rest is not None:
                        return (modified,) + rest
                    return modified
                return hook_fn

            handle = layer.register_forward_hook(make_hook(module, gate))
            self.hook_handles.append(handle)

    def remove_hooks(self):
        """Supprime tous les hooks enregistrés."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def _get_decoder_layers(self, qwen_model):
        """Trouve les decoder layers dans le modèle Qwen (compatible Qwen2.5-VL et Qwen3-VL)."""
        # Qwen3-VL : model.model.language_model.layers
        if hasattr(qwen_model, 'model'):
            inner = qwen_model.model
            if hasattr(inner, 'language_model') and hasattr(inner.language_model, 'layers'):
                return inner.language_model.layers
            # Qwen2.5-VL : model.model.layers
            if hasattr(inner, 'layers'):
                return inner.layers
        # Fallback : chercher récursivement
        for name, module in qwen_model.named_modules():
            if name.endswith('.layers') and isinstance(module, nn.ModuleList):
                return module
        raise RuntimeError("Impossible de trouver les decoder layers dans le modèle Qwen")


class HippocampalPrefixInjector(nn.Module):
    """
    Convertit un souvenir hippocampique en prefix embeddings.

    Récupère un vecteur mémoire depuis l'hippocampe et le projette
    en N tokens virtuels dans l'espace d'embedding du Qwen.
    Ces tokens sont préfixés aux embeddings d'entrée.
    """

    def __init__(self, content_dim: int = 4096, n_prefix_tokens: int = 4):
        super().__init__()
        self.content_dim = content_dim
        self.n_prefix_tokens = n_prefix_tokens
        self.memory_to_prefix = nn.Linear(content_dim, n_prefix_tokens * content_dim)
        self.prefix_gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, recalled_memory: torch.Tensor) -> Optional[torch.Tensor]:
        """
        recalled_memory : (content_dim,) — vecteur récupéré de l'hippocampe
        Retourne : (1, n_prefix_tokens, content_dim) ou None si mémoire vide
        """
        if recalled_memory.norm().item() < 1e-6:
            return None
        prefix = self.memory_to_prefix(recalled_memory.float())
        prefix = prefix.view(1, self.n_prefix_tokens, self.content_dim)
        return self.prefix_gate * prefix
