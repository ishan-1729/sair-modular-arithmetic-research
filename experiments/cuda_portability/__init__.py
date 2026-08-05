"""Shared, explicit CUDA portability boundaries for experiment replays.

The package deliberately does not mutate PyTorch's default device.  Frozen
experiment runners remain immutable; current replay/adaptor entry points opt in
to these helpers explicitly.
"""

from .runtime import configure_determinism, resolve_device, synchronize

__all__ = ["configure_determinism", "resolve_device", "synchronize"]
