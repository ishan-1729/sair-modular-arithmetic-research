"""Training-only CE-mask helpers over unchanged Candidate005 inference."""

from __future__ import annotations

import hashlib
from typing import Literal, Sequence

import torch
from torch import Tensor

from experiments.exp_676_lla_moving_window.model import (
    reinjection_receipt,
    state_dict_identity,
    trace_forward,
)


MaskKind = Literal["target_only", "mixed", "endpoint_weight", "shuffled"]


def mask_weights(kind: MaskKind, *, positions: int, step: int, seed: int) -> tuple[float, ...]:
    """Return frozen coefficients for equal-tensor-work continuation losses.

    Every arm computes ``positions`` loss tensors and divides their weighted sum
    by ``positions``. Target-only and endpoint-weight therefore give the final
    target the same unit coefficient in the reduced loss.
    """
    if positions < 1:
        raise ValueError("positions must be positive")
    if step < 1:
        raise ValueError("step must be positive")
    if positions == 1:
        return (1.0,)
    if kind == "mixed":
        return (1.0,) * positions
    if kind == "target_only":
        return (0.0,) * (positions - 1) + (float(positions),)
    if kind == "endpoint_weight":
        return (1.0,) * (positions - 1) + (float(positions),)
    if kind == "shuffled":
        identity = f"exp678:{seed}:{step}:{positions}".encode()
        local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
        index = int(torch.randint(positions, (1,), generator=torch.Generator().manual_seed(local_seed)))
        return tuple(float(positions) if position == index else 0.0 for position in range(positions))
    raise ValueError(f"unknown mask kind: {kind}")


def masked_mean(losses: Sequence[Tensor], weights: Sequence[float]) -> tuple[Tensor, dict[str, object]]:
    """Reduce precomputed losses while recording label and gradient work."""
    if not losses or len(losses) != len(weights):
        raise ValueError("losses and weights must have the same nonzero length")
    if any(loss.ndim != 0 or not torch.isfinite(loss) for loss in losses):
        raise ValueError("losses must be finite scalars")
    tensor_weights = losses[0].new_tensor(weights)
    reduced = (torch.stack(tuple(losses)) * tensor_weights).sum() / len(losses)
    active = [index for index, weight in enumerate(weights) if weight != 0.0]
    return reduced, {
        "ce_tensor_calls": len(losses),
        "effective_supervised_positions": len(active),
        "active_positions": active,
        "weights": [float(weight) for weight in weights],
        "total_weight": float(sum(weights)),
        "final_target_reduced_coefficient": float(weights[-1]) / len(weights),
        "irreducible_gradient_weight_sum": float(sum(abs(weight) for weight in weights)),
    }


__all__ = [
    "mask_weights",
    "masked_mean",
    "reinjection_receipt",
    "state_dict_identity",
    "trace_forward",
]
