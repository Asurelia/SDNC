"""
Chargeur du modèle teacher Qwen3.5-35B-A3B en RAM CPU.

Stratégie A100 haute-RAM (83.5 GB) :
  - Teacher 35B en BF16 natif → RAM CPU (~70 GB)
  - Forward sur CPU → hidden_states en RAM
  - Transfert asynchrone RAM → VRAM via PCIe
  - Student 4B reste en VRAM (~8 GB)

Aucun compromis : BF16 natif partout, pas de quantification.
"""

import gc
import os
import time

os.environ.setdefault("PYTORCH_ATTENTION_BACKEND", "math")

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _ram_used_gb() -> float:
    """Retourne la RAM utilisée par ce processus en GB."""
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1e9
    return 0.0


def _ram_available_gb() -> float:
    """Retourne la RAM système disponible en GB."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().available / 1e9
    return 0.0


class TeacherLoader:
    """
    Chargeur du modèle teacher en RAM CPU BF16 natif.

    Le teacher est chargé explicitement sur CPU (device_map="cpu").
    Les hidden states sont transférés vers VRAM de manière asynchrone
    via .to(target_device, non_blocking=True).

    Tous les poids sont gelés — jamais modifiés.
    """

    def load(
        self,
        model_name: str = "Qwen/Qwen3.5-35B-A3B",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple:
        """
        Charge le modèle teacher en RAM CPU.

        Args:
            model_name: Nom HuggingFace du modèle teacher.
            dtype: Type de données (bfloat16 par défaut).

        Returns:
            (model, tokenizer) — le modèle est sur CPU, gelé.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ram_before = _ram_used_gb()
        ram_avail = _ram_available_gb()

        print(f"[TeacherLoader] Chargement {model_name} en {dtype} sur CPU RAM...")
        print(f"  RAM disponible : {ram_avail:.1f} GB")
        print(f"  Taille estimée : ~70 GB — chargement ~3-5 minutes...")

        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()

        # Geler TOUS les poids — non négociable
        for p in model.parameters():
            p.requires_grad = False

        elapsed = time.time() - t0
        ram_after = _ram_used_gb()
        delta = ram_after - ram_before

        print(f"[TeacherLoader] Teacher chargé en {elapsed:.0f}s")
        print(f"  RAM avant  : {ram_before:.1f} GB")
        print(f"  RAM après  : {ram_after:.1f} GB")
        print(f"  Δ RAM      : {delta:.1f} GB")

        return model, tokenizer

    def get_representations(
        self,
        model,
        tokenizer,
        prompt: str,
        layers: Optional[List[int]] = None,
        target_device: str = "cuda",
        max_seq_pool: int = 512,
    ) -> List[torch.Tensor]:
        """
        Extrait les hidden_states du teacher (en RAM CPU)
        et les transfère en VRAM pour le student.

        Flux :
          Tokenize → forward CPU (teacher) → hidden_states CPU
          → .to("cuda", non_blocking=True) → hidden_states GPU

        Transfert PCIe estimé : ~5-10 MB pour 4 couches × seq_len × 5120
        Temps estimé : ~0.5ms (négligeable vs forward teacher ~500ms)

        Args:
            model: Modèle teacher (sur CPU).
            tokenizer: Tokenizer du teacher.
            prompt: Texte d'entrée.
            layers: Indices des couches d'intercept.
            target_device: Device cible pour les tenseurs de sortie.
            max_seq_pool: Pool seq_len si > cette valeur.

        Returns:
            Liste de tenseurs sur target_device.
        """
        if layers is None:
            layers = [16, 32, 48, 64]

        # Tokenize — inputs restent sur CPU
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        # Pas de .to(device) — le teacher est sur CPU

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states  # tuple de tenseurs en CPU
        pooled = []

        for i in layers:
            idx = min(i, len(hidden) - 1)
            h = hidden[idx]

            # Pool seq_len → max_seq_pool si nécessaire
            if h.shape[1] > max_seq_pool:
                h = F.adaptive_avg_pool1d(
                    h.transpose(1, 2),
                    max_seq_pool
                ).transpose(1, 2)

            # Transfert CPU RAM → VRAM non-bloquant
            pooled.append(h.to(target_device, non_blocking=True))

        # Attendre la fin de tous les transferts
        if target_device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

        return pooled

    def get_soft_targets(
        self,
        model,
        tokenizer,
        prompt: str,
        temperature: float = 2.0,
        target_device: str = "cuda",
    ) -> torch.Tensor:
        """
        Retourne les logits softmax (distribution de probabilités)
        du teacher pour la distillation KL.

        Température élevée = distribution plus douce = meilleure distillation.

        Args:
            model: Modèle teacher (sur CPU).
            tokenizer: Tokenizer du teacher.
            prompt: Texte d'entrée.
            temperature: Température de lissage.
            target_device: Device cible.

        Returns:
            Tensor (1, vocab_size) de probabilités lissées sur target_device.
        """
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            out = model(**inputs)

        logits = out.logits[:, -1, :]  # (1, vocab_size) dernier token
        soft = F.softmax(logits / temperature, dim=-1)

        result = soft.to(target_device, non_blocking=True)
        if target_device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

        return result

    def get_representations_and_targets(
        self,
        model,
        tokenizer,
        prompt: str,
        layers: Optional[List[int]] = None,
        temperature: float = 2.0,
        target_device: str = "cuda",
        max_seq_pool: int = 512,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Retourne representations ET soft targets en un seul forward pass.

        Économise le temps CPU en évitant deux forwards sur un modèle 35B.

        Args:
            model: Modèle teacher (sur CPU).
            tokenizer: Tokenizer du teacher.
            prompt: Texte d'entrée.
            layers: Indices des couches d'intercept.
            temperature: Température pour les soft targets.
            target_device: Device cible.
            max_seq_pool: Pool seq_len si > cette valeur.

        Returns:
            (representations, soft_targets)
        """
        if layers is None:
            layers = [16, 32, 48, 64]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Hidden states
        hidden = outputs.hidden_states
        pooled = []
        for i in layers:
            idx = min(i, len(hidden) - 1)
            h = hidden[idx]
            if h.shape[1] > max_seq_pool:
                h = F.adaptive_avg_pool1d(
                    h.transpose(1, 2), max_seq_pool
                ).transpose(1, 2)
            pooled.append(h.to(target_device, non_blocking=True))

        # Soft targets (dernier token)
        logits = outputs.logits[:, -1, :]
        soft = F.softmax(logits / temperature, dim=-1)
        soft_gpu = soft.to(target_device, non_blocking=True)

        if target_device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

        return pooled, soft_gpu

    @staticmethod
    def unload(model) -> None:
        """
        Décharge le modèle teacher de la RAM.

        Args:
            model: Modèle teacher à libérer.
        """
        ram_before = _ram_used_gb()
        del model
        gc.collect()
        ram_after = _ram_used_gb()
        print(f"[TeacherLoader] Modèle déchargé — RAM libérée : {ram_before - ram_after:.1f} GB")
