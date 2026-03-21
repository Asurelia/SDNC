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
