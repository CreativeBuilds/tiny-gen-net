"""Best available PyTorch device: CUDA > MPS (Apple Silicon) > CPU."""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def device_str(device: torch.device | str | None = None) -> str:
    if device is None: return str(get_device())
    return str(device) if isinstance(device, torch.device) else device