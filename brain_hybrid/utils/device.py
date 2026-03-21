import torch


def get_device() -> torch.device:
    """
    Détection automatique du GPU.
    ROCm (AMD) se présente comme CUDA dans PyTorch.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU détecté : {name} ({vram_gb:.1f} GB)")

        if vram_gb < 15.0:
            print("⚠️  VRAM < 15 GB — Qwen3-4B risque OOM")
            print("   → Fallback recommandé : Qwen/Qwen3-4B")

        return device

    print("⚠️  Pas de GPU détecté — fallback CPU (très lent)")
    return torch.device("cpu")
