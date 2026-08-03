"""Explicit CPU/CUDA runtime selection shared by portable experiment routes."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda"})


def resolve_device(
    requested: str | torch.device,
    *,
    require_accelerator: bool = False,
) -> torch.device:
    """Resolve an explicit device and fail closed when CUDA is required.

    ``auto`` is convenient for bounded development smokes.  Hardware
    acceptance must use ``require_accelerator=True`` (or request ``cuda``)
    so an unavailable GPU can never turn into a misleading CPU success.
    """

    requested_text = str(requested)
    if requested_text == "auto":
        requested_text = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_text)
    if device.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(f"unsupported experiment device: {device}")
    if require_accelerator and device.type != "cuda":
        raise RuntimeError("CUDA hardware acceptance requires an explicit CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA replay requested but torch.cuda.is_available() is false"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        device = torch.device("cuda", index)
    return device


def synchronize(device: torch.device) -> None:
    """Synchronize only at an explicit timing/evidence boundary."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_determinism(
    *,
    device: torch.device,
    seed: int,
    torch_threads: int,
    interop_threads: int = 1,
) -> None:
    """Configure deterministic execution without changing the default device."""

    if torch_threads <= 0 or interop_threads <= 0:
        raise ValueError("thread counts must be positive")
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(int(interop_threads))
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work begins.  A
        # reused test process may already have fixed the same process setting;
        # a conflicting setting is a reproducibility error, not a warning.
        if torch.get_num_interop_threads() != int(interop_threads):
            raise
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(seed))


def runtime_receipt(device: torch.device) -> dict[str, Any]:
    """Return the small device/determinism identity common to run receipts."""

    result: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda_runtime": str(torch.version.cuda),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32": False,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    return result
