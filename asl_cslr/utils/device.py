"""
Device management and mixed precision utilities for Apple Silicon (§1, §11).
"""

import os
import logging
from contextlib import contextmanager, nullcontext

import torch

logger = logging.getLogger(__name__)


def setup_mps_fallback() -> bool:
    """Ensure PyTorch can fall back to CPU for unsupported MPS ops.

    Returns:
        True if the environment flag was set by this call, else False.
    """
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        logger.debug("PYTORCH_ENABLE_MPS_FALLBACK is already enabled")
        return False

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logger.info("Enabled PyTorch MPS fallback for unsupported ops")
    return True


def get_device() -> torch.device:
    """Get the best available device: MPS > CUDA > CPU.

    Also sets up MPS fallback if MPS is available.

    Returns:
        torch.device
    """
    if torch.backends.mps.is_available():
        setup_mps_fallback()
        logger.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def get_autocast_context(device: torch.device, enabled: bool = True):
    """Get the appropriate autocast context for mixed precision.

    Uses torch.amp.autocast with the correct device_type string.
    Returns a nullcontext if disabled.

    Args:
        device: The torch device.
        enabled: Whether to enable mixed precision.

    Returns:
        Context manager for autocast.
    """
    if not enabled:
        return nullcontext()

    device_type = device.type  # 'mps', 'cuda', or 'cpu'

    if device_type == "mps":
        return torch.amp.autocast(device_type="mps", dtype=torch.float16)
    elif device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        return nullcontext()


def get_device_info() -> dict:
    """Get information about the current compute device.

    Returns:
        Dict with device info (type, name, memory, etc.).
    """
    info = {"device_type": "cpu", "device_name": "CPU"}

    if torch.backends.mps.is_available():
        info["device_type"] = "mps"
        info["device_name"] = "Apple Silicon GPU (MPS)"
        info["mps_available"] = True
    elif torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["gpu_memory_gb"] = (
            torch.cuda.get_device_properties(0).total_mem / 1024**3
        )

    info["pytorch_version"] = torch.__version__
    return info
