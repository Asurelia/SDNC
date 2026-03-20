"""Device detection for AMD GPU / CPU fallback."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Detect the best available device.

    Priority: CUDA (ROCm) > DirectML > CPU.
    """
    if preference != "auto":
        return torch.device(preference)

    # ROCm presents as CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # DirectML fallback
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass

    return torch.device("cpu")
