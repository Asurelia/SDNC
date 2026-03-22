from dataclasses import dataclass, field
from typing import List


@dataclass
class BrainConfig:
    # LLM
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    llm_hidden_dim: int = 4096           # Qwen3-VL-8B hidden size
    intercept_layers: List[int] = field(
        default_factory=lambda: [9, 18, 27, 36]
    )

    # Modules CfC+SNN
    module_hidden_dim: int = 64          # dimension interne des modules
    n_modules: int = 4                   # 1 par point d'intercept

    # STDP
    stdp_lr_plus: float = 0.01
    stdp_lr_minus: float = 0.01
    dopamine_threshold: float = 1.5      # seuil de saillance (adapté aux erreurs ~1.4)

    # Hippocampe
    memory_address_dim: int = 256
    memory_n_locations: int = 10000
    salience_threshold: float = 0.3      # erreur min pour mémoriser

    # État global
    state_dim: int = 512

    # Quantization — bitsandbytes Q8 (Colab/Linux only, ignored on Windows/ROCm)
    use_quantization: bool = True

    # Injection CfC → Qwen
    injection_enabled: bool = True
    injection_gate_rank: int = 16          # dimension low-rank projection
    injection_init_alpha: float = 0.001    # force initiale du gate
    injection_max_alpha: float = 0.1       # force maximale
    injection_n_prefix_tokens: int = 4     # tokens virtuels hippocampe

    # ACC — Cortex Cingulaire Antérieur
    acc_conflict_threshold: float = 0.7
    acc_exploration_threshold: float = 0.4
    acc_lr: float = 0.001

    # StepScheduler — rythmes cérébraux
    snn_freq: int = 1           # SNN : chaque step
    cfc_freq: int = 10          # CfC : tous les 10 steps
    hippocampus_freq: int = 6   # SDM : phase thêta simulée
    qwen_freq: int = 100        # Qwen enrichi : rare, coûteux
    sleep_freq: int = 5000      # consolidation offline
    arousal_boost_steps: int = 50

    # Predictive Coding
    pc_lr: float = 0.001

    # Sleep
    sleep_replay_count: int = 10
