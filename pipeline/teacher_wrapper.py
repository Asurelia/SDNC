"""
Wrapper pour le modèle teacher Qwen3.5-35B-A3B.

Texte uniquement (causal LM). Gelé définitivement.
Fournit les représentations internes et les soft targets pour la distillation.
"""

import os
os.environ.setdefault("PYTORCH_ATTENTION_BACKEND", "math")

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def _can_use_bnb() -> bool:
    """Vérifie si bitsandbytes est disponible."""
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


class TeacherWrapper:
    """
    Wrapper pour le modèle teacher (Qwen3.5-35B-A3B).

    Charge le modèle en BF16 sur A100. Si la VRAM dépasse le seuil,
    bascule automatiquement en 8-bit via bitsandbytes.

    Gelé — aucun poids ne sera jamais modifié.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-35B-A3B",
        use_8bit: bool = False,
        max_seq_pool: int = 512,
    ):
        """
        Charge le modèle teacher.

        Args:
            model_name: Nom HuggingFace du modèle teacher.
            use_8bit: Forcer le chargement en 8-bit.
            max_seq_pool: Si seq_len > max_seq_pool, pool vers cette taille.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_seq_pool = max_seq_pool

        # Dtype
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            use_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            use_dtype = torch.float32

        # Kwargs de chargement
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        if use_8bit and _can_use_bnb():
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["lm_head"],
            )
            load_kwargs["torch_dtype"] = use_dtype
            print(f"[Teacher] Chargement {model_name} en Q8 ({use_dtype})...")
        else:
            load_kwargs["torch_dtype"] = use_dtype
            if use_8bit:
                print(f"[Teacher] bitsandbytes indisponible — chargement en {use_dtype}...")
            else:
                print(f"[Teacher] Chargement {model_name} en {use_dtype}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        self.model.eval()

        # Geler TOUS les poids — non négociable
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = next(self.model.parameters()).device
        self._is_8bit = use_8bit

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"[Teacher] Chargé sur {self.device} | VRAM : {vram_used:.1f} / {vram:.1f} GB")
        else:
            print(f"[Teacher] Chargé sur {self.device} (CPU)")

    def _tokenize(self, prompt: str) -> dict:
        """Tokenise un prompt pour le modèle causal."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _pool_seq(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pool la dimension séquence si elle dépasse max_seq_pool.

        (batch, seq_len, hidden) → (batch, max_seq_pool, hidden)
        """
        if tensor.shape[1] <= self.max_seq_pool:
            return tensor
        return F.adaptive_avg_pool1d(
            tensor.transpose(1, 2),
            self.max_seq_pool
        ).transpose(1, 2)

    def get_teacher_representations(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Retourne les hidden states aux couches d'intercept.

        Args:
            prompt: Texte d'entrée.
            layers: Indices des couches (1-indexed). Par défaut [16, 32, 48, 64].

        Returns:
            Liste de tenseurs (1, seq_len, teacher_hidden).
            Poolés si seq_len > max_seq_pool.
        """
        if layers is None:
            layers = [16, 32, 48, 64]

        inputs = self._tokenize(prompt)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states
        reps = []
        for i in layers:
            idx = min(i, len(hidden) - 1)
            rep = self._pool_seq(hidden[idx])
            reps.append(rep)
        return reps

    def get_soft_targets(
        self,
        prompt: str,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """
        Retourne les soft targets (softmax(logits/T)) pour la distillation KL.

        Args:
            prompt: Texte d'entrée.
            temperature: Température de lissage.

        Returns:
            Tensor (1, seq_len, vocab_size) de probabilités lissées.
        """
        inputs = self._tokenize(prompt)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # (1, seq_len, vocab_size)
        return F.softmax(logits / temperature, dim=-1)

    def get_representations_and_targets(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
        temperature: float = 2.0,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Retourne representations ET soft targets en un seul forward pass.

        Économise la VRAM en évitant deux forwards sur un modèle 35B.

        Args:
            prompt: Texte d'entrée.
            layers: Indices des couches d'intercept.
            temperature: Température pour les soft targets.

        Returns:
            (representations, soft_targets)
        """
        if layers is None:
            layers = [16, 32, 48, 64]

        inputs = self._tokenize(prompt)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # Hidden states
        hidden = outputs.hidden_states
        reps = []
        for i in layers:
            idx = min(i, len(hidden) - 1)
            rep = self._pool_seq(hidden[idx])
            reps.append(rep)

        # Soft targets
        logits = outputs.logits
        soft_targets = F.softmax(logits / temperature, dim=-1)

        return reps, soft_targets

    def unload(self):
        """Décharge le modèle de la VRAM."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Teacher] Modèle déchargé de la VRAM.")

    @staticmethod
    def check_vram_and_reload(
        model_name: str = "Qwen/Qwen3.5-35B-A3B",
        threshold_gb: float = 75.0,
    ) -> "TeacherWrapper":
        """
        Charge le teacher en BF16, puis bascule en 8-bit si VRAM > seuil.

        Args:
            model_name: Nom du modèle.
            threshold_gb: Seuil VRAM en GB.

        Returns:
            TeacherWrapper (éventuellement en 8-bit).
        """
        # Premier essai en BF16
        teacher = TeacherWrapper(model_name, use_8bit=False)

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            if vram_used > threshold_gb:
                print(f"[Teacher] VRAM {vram_used:.1f} GB > {threshold_gb} GB — rechargement en 8-bit...")
                teacher.unload()
                teacher = TeacherWrapper(model_name, use_8bit=True)

        return teacher
