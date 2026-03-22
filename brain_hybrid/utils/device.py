"""
Détection hardware et configuration adaptative.

Détecte A100 / AMD ROCm / CPU au démarrage et expose HW_CONFIG global.
Toute la logique hardware est centralisée ici — aucun import conditionnel ailleurs.
"""

import torch

# Configuration hardware globale — peuplée par get_device()
HW_CONFIG = {
    "device": torch.device("cpu"),
    "hw_type": "cpu",
    "batch_size": 1,
    "max_length": 256,
    "empty_cache_after_qwen": False,
}


def get_device() -> torch.device:
    """
    Détection automatique du GPU et configuration adaptative.

    - A100/H100 : tf32, batch_size=4, max_length=1024
    - AMD ROCm  : batch_size=1, max_length=512, empty_cache après Qwen
    - CPU       : batch_size=1, max_length=256

    Retourne le device et peuple HW_CONFIG global.
    """
    global HW_CONFIG

    if not torch.cuda.is_available():
        print("Pas de GPU détecté — fallback CPU (très lent)")
        HW_CONFIG.update({
            "device": torch.device("cpu"),
            "hw_type": "cpu",
            "batch_size": 1,
            "max_length": 256,
            "empty_cache_after_qwen": False,
        })
        return HW_CONFIG["device"]

    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)

    # A100/H100 : compute capability 8.0+ et VRAM >= 30 GB
    if cap[0] >= 8 and vram_gb >= 30:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        HW_CONFIG.update({
            "device": device,
            "hw_type": "a100",
            "batch_size": 4,
            "max_length": 1024,
            "empty_cache_after_qwen": False,
        })
        print(f"GPU : {name} ({vram_gb:.1f} GB) — mode A100 (tf32, batch=4)")

    # AMD ROCm : compute capability >= 9 (ROCm report 11.0) ou nom contient AMD/Radeon
    elif cap[0] >= 9 or "amd" in name.lower() or "radeon" in name.lower():
        HW_CONFIG.update({
            "device": device,
            "hw_type": "amd",
            "batch_size": 1,
            "max_length": 512,
            "empty_cache_after_qwen": True,
        })
        print(f"GPU : {name} ({vram_gb:.1f} GB) — mode AMD (batch=1, cache flush)")

    # Autre GPU NVIDIA (T4, V100, etc.)
    else:
        HW_CONFIG.update({
            "device": device,
            "hw_type": "a100" if vram_gb >= 30 else "amd",
            "batch_size": 2 if vram_gb >= 15 else 1,
            "max_length": 512,
            "empty_cache_after_qwen": vram_gb < 20,
        })
        print(f"GPU : {name} ({vram_gb:.1f} GB) — mode standard (batch={HW_CONFIG['batch_size']})")

    return HW_CONFIG["device"]
