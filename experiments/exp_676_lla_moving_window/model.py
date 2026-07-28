"""Training-only physical-visit traces for unchanged Candidate005 inference."""

from __future__ import annotations

import hashlib
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


WindowKind = Literal["moving", "endpoint", "fixed", "equal_work", "shuffled"]


def readout(model: nn.Module, hidden: Tensor) -> tuple[Tensor, Tensor]:
    normalized = F.silu(model.output_norm(hidden))
    logits = model.local_head(normalized).squeeze(1)
    pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
    return logits, model.quotient_head(pooled)


def trace_forward(model: nn.Module, features: Tensor) -> list[tuple[Tensor, Tensor]]:
    """Expose every existing physical visit without adding model parameters."""
    if features.ndim != 3 or features.shape[1] != 6 or not torch.isfinite(features).all():
        raise ValueError("features must be finite [batch, 6, width]")
    hidden = model.stem(features)
    outputs: list[tuple[Tensor, Tensor]] = []
    for dilation in model.dilation_schedule(features.shape[-1]):
        hidden = model.recurrent_update(hidden, features, int(dilation))
        outputs.append(readout(model, hidden))
    return outputs


def selected_depths(
    kind: WindowKind,
    *,
    visits: int,
    step: int,
    total_steps: int,
    window_size: int,
    seed: int,
) -> tuple[int, ...]:
    """Return zero-based visit indices for one matched four-loss update."""
    if visits < window_size or window_size < 1:
        raise ValueError("window must fit the physical visit count")
    if step < 1 or step > total_steps or total_steps < 1:
        raise ValueError("step must lie inside the frozen update schedule")
    if kind == "endpoint":
        return (visits - 1,) * window_size
    if kind == "fixed":
        return tuple(range(visits - window_size, visits))
    if kind == "equal_work":
        if window_size == 1:
            return (visits - 1,)
        return tuple(round(index * (visits - 1) / (window_size - 1)) for index in range(window_size))
    if kind == "moving":
        progress = 0.0 if total_steps == 1 else (step - 1) / (total_steps - 1)
        endpoint = window_size - 1 + round(progress * (visits - window_size))
        return tuple(range(endpoint - window_size + 1, endpoint + 1))
    if kind == "shuffled":
        identity = f"{seed}:{step}:{visits}:{window_size}".encode()
        local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
        generator = torch.Generator().manual_seed(local_seed)
        return tuple(sorted(torch.randperm(visits, generator=generator)[:window_size].tolist()))
    raise ValueError(f"unknown window kind: {kind}")


def reinjection_receipt(model: nn.Module, width: int) -> dict[str, int | bool]:
    """Prove all physical updates consume the immutable six-channel input."""
    channels = int(model.hidden) + 6
    schedule = model.dilation_schedule(width)
    return {
        "width": width,
        "physical_visits": len(schedule),
        "gate_input_channels": int(model.gate_weight.shape[1]),
        "candidate_input_channels": int(model.candidate_weight.shape[1]),
        "legal_feature_channels": 6,
        "gate_reinjects_every_visit": int(model.gate_weight.shape[1]) == channels,
        "candidate_reinjects_every_visit": int(model.candidate_weight.shape[1]) == channels,
    }


def state_dict_identity(first: nn.Module, second: nn.Module) -> bool:
    a, b = first.state_dict(), second.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)
