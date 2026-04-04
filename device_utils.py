"""
device_utils.py — drop this in your project root.
Import get_device() instead of writing device logic inline.
"""
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_autocast_ctx(device):
    """Returns a no-op context on MPS (AMP not supported yet)."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu", enabled=False)

def get_scaler(device):
    """GradScaler only works on CUDA."""
    return torch.amp.GradScaler(enabled=(device.type == "cuda"))
