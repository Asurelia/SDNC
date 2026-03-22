"""Configuration du pipeline dual-model teacher-student."""

from dataclasses import dataclass, field
from typing import List

from brain_hybrid.config import BrainConfig


@dataclass
class DualModelConfig:
    """
    Configuration complète du pipeline de distillation teacher → student.

    Teacher : Qwen3-VL-32B (dense, VL, hidden=5120, 64 layers)
    Student : Qwen3.5-4B (dense, VL, hidden=2560, 32 layers)

    Le teacher est gelé. Le student apprend via STDP/PC/hippocampe.
    Seul le ProjectionBridge (teacher → student) utilise du backprop.
    """

    # ── Teacher ──────────────────────────────────────────────────
    teacher_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    teacher_hidden: int = 5120          # confirmé par audit
    teacher_layers: List[int] = field(
        default_factory=lambda: [16, 32, 48, 64]
    )

    # ── Student ──────────────────────────────────────────────────
    student_model: str = "Qwen/Qwen3.5-4B"
    student_hidden: int = 2560
    student_layers: List[int] = field(
        default_factory=lambda: [8, 16, 24, 32]
    )

    # ── ProjectionBridge ─────────────────────────────────────────
    distill_projection_dim: int = 1024  # espace latent commun (5120 → 1024 → 2560)
    distill_temperature: float = 2.0
    distill_alpha: float = 0.7          # poids MSE (1-alpha = poids KL)

    # ── Sleep distillation ───────────────────────────────────────
    sleep_replay_count: int = 50
    sleep_distill_lr: float = 5e-5      # AdamW lr pour ProjectionBridge

    # ── Hardware placement ──────────────────────────────────────
    teacher_device: str = "cpu"         # teacher 35B en RAM système
    student_device: str = "cuda"        # student 4B en VRAM
    teacher_dtype: str = "bfloat16"     # BF16 natif, aucun compromis

    # ── Monitoring ──────────────────────────────────────────────
    ram_safety_margin_gb: float = 3.0   # arrêt propre si marge RAM < 3 GB
    log_interval: int = 10              # log RAM/VRAM/losses toutes les N steps

    # ── Cloud sync ───────────────────────────────────────────────
    gcs_bucket: str = "sdnc-models"
    gcs_model_prefix: str = "checkpoints"
    github_repo: str = ""               # rempli par l'utilisateur
    github_branch: str = "main"
    auto_push_every_n_distillations: int = 1

    @property
    def student_brain_config(self) -> BrainConfig:
        """Génère un BrainConfig compatible pour le student."""
        return BrainConfig(
            model_name=self.student_model,
            llm_hidden_dim=self.student_hidden,
            intercept_layers=self.student_layers,
            n_modules=len(self.student_layers),
            use_quantization=False,
        )
