"""
Détection hardware et configuration adaptative.

Détecte A100 haute-RAM / A100 standard / AMD ROCm / CPU au démarrage
et expose HW_CONFIG global.

Profils :
  a100_highram — A100 + 83.5 GB RAM : teacher 35B en RAM CPU BF16
  a100         — A100 + RAM limitée : teacher en float8 ou 8-bit
  amd          — RX 7800 XT / ROCm : student seul, cache flush
  cpu          — fallback sans GPU

Toute la logique hardware est centralisée ici — aucun import conditionnel ailleurs.
"""

import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Configuration hardware globale — peuplée par get_device()
HW_CONFIG = {
    "device": torch.device("cpu"),
    "hw_type": "cpu",
    "batch_size": 1,
    "max_length": 256,
    "empty_cache_after_qwen": False,
}


def _get_ram_gb() -> float:
    """Retourne la RAM système totale en GB."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().total / 1e9
    return 0.0


def get_device() -> torch.device:
    """
    Détection automatique du GPU et configuration adaptative.

    Profils détectés :
      - A100 haute-RAM (VRAM >= 38 GB, RAM >= 80 GB) : teacher CPU BF16
      - A100 standard  (VRAM >= 38 GB, RAM < 80 GB)  : teacher CPU float8
      - A100/H100      (VRAM >= 30 GB)                : tf32, batch=4
      - AMD ROCm       : batch=1, cache flush
      - CPU            : batch=1, max_length=256

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
    ram_gb = _get_ram_gb()

    # A100 haute mémoire : VRAM >= 38 GB + RAM >= 80 GB
    # Teacher 35B en BF16 natif dans la RAM CPU, student en VRAM
    if cap[0] >= 8 and vram_gb >= 38 and ram_gb >= 80:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        HW_CONFIG.update({
            "device": device,
            "hw_type": "a100_highram",
            "batch_size": 4,
            "max_length": 1024,
            "empty_cache_after_qwen": False,
            # Pipeline teacher-student
            "n_circuits": 50000,
            "circuit_dim": 256,
            "k_active": 200,
            "vram_cache_max": 400,
            "dtype": torch.bfloat16,
            "teacher_device": "cpu",
            "student_device": "cuda",
            "teacher_dtype": torch.bfloat16,
            "use_teacher": True,
            "teacher_model": "Qwen/Qwen3.5-35B-A3B",
            "student_model": "Qwen/Qwen3.5-4B",
        })
        print(f"GPU : {name} ({vram_gb:.1f} GB VRAM, {ram_gb:.1f} GB RAM) — mode A100 haute-RAM")

    # A100 RAM normale : VRAM >= 38 GB mais RAM < 80 GB
    elif cap[0] >= 8 and vram_gb >= 38:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        HW_CONFIG.update({
            "device": device,
            "hw_type": "a100",
            "batch_size": 4,
            "max_length": 1024,
            "empty_cache_after_qwen": False,
            # Pipeline teacher-student (RAM limitée)
            "n_circuits": 50000,
            "circuit_dim": 256,
            "k_active": 30,
            "vram_cache_max": 100,
            "dtype": torch.bfloat16,
            "teacher_device": "cpu",
            "student_device": "cuda",
            "teacher_dtype": torch.float8_e4m3fn,
            "use_teacher": True,
            "teacher_model": "Qwen/Qwen3.5-35B-A3B",
            "student_model": "Qwen/Qwen3.5-4B",
        })
        print(f"GPU : {name} ({vram_gb:.1f} GB VRAM, {ram_gb:.1f} GB RAM) — mode A100")

    # A100/H100 : compute capability 8.0+ et VRAM >= 30 GB (mais < 38)
    elif cap[0] >= 8 and vram_gb >= 30:
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
