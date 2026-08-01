"""Exactly embedded wider Candidate005 core with deterministic reduced visits."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def evenly_spaced_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 1 or count < 1 or count > length:
        raise ValueError("count must lie in [1, length]")
    if count == 1:
        return (length - 1,)
    result = tuple(round(index * (length - 1) / (count - 1)) for index in range(count))
    if len(set(result)) != count:
        raise ValueError("rounded route indices are not unique")
    return result


def scaled_call_count(full_calls: int, calls_at_w128: int, reference_calls: int = 14) -> int:
    return max(2, min(full_calls, round(full_calls * calls_at_w128 / reference_calls)))


class CapacityRecycledConvGRU(nn.Module):
    """Source core plus a separately normalized learned channel bank.

    Separate normalization makes the old 92-channel subspace mathematically
    identical to Candidate005. Zero extra readouts make the full-call reference
    route bitwise identical at initialization while retaining trainable new state.
    """

    def __init__(self, source: nn.Module, *, total_hidden: int, calls_at_w128: int | None) -> None:
        super().__init__()
        if total_hidden < int(source.hidden):
            raise ValueError("total_hidden cannot shrink the source subspace")
        self.source = source
        self.width = int(source.width)
        self.hidden = total_hidden
        self.source_hidden = int(source.hidden)
        self.extra_hidden = total_hidden - self.source_hidden
        self.cycles = int(source.cycles)
        self.calls_at_w128 = calls_at_w128
        if self.extra_hidden:
            channels = self.extra_hidden + 6
            self.extra_stem = nn.Conv1d(6, self.extra_hidden, 1)
            self.extra_norm = nn.GroupNorm(1, self.extra_hidden)
            self.extra_gate_weight = nn.Parameter(torch.empty(2 * self.extra_hidden, channels, 3))
            self.extra_gate_bias = nn.Parameter(torch.zeros(2 * self.extra_hidden))
            self.extra_candidate_weight = nn.Parameter(torch.empty(self.extra_hidden, channels, 3))
            self.extra_candidate_bias = nn.Parameter(torch.zeros(self.extra_hidden))
            self.extra_output_norm = nn.GroupNorm(1, self.extra_hidden)
            self.extra_local_head = nn.Conv1d(self.extra_hidden, 1, 1)
            self.extra_quotient_head = nn.Sequential(
                nn.Linear(2 * self.extra_hidden, self.extra_hidden), nn.SiLU(), nn.Linear(self.extra_hidden, 3)
            )
            nn.init.kaiming_uniform_(self.extra_gate_weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.extra_candidate_weight, a=math.sqrt(5))
            nn.init.zeros_(self.extra_local_head.weight); nn.init.zeros_(self.extra_local_head.bias)
            nn.init.zeros_(self.extra_quotient_head[-1].weight); nn.init.zeros_(self.extra_quotient_head[-1].bias)

    def full_schedule(self, width: int) -> list[int]:
        return list(self.source.dilation_schedule(width))

    def route_indices(self, width: int) -> tuple[int, ...]:
        full = self.full_schedule(width)
        if self.calls_at_w128 is None:
            return tuple(range(len(full)))
        return evenly_spaced_indices(len(full), scaled_call_count(len(full), self.calls_at_w128))

    def dilation_schedule(self, width: int) -> list[int]:
        full = self.full_schedule(width)
        return [full[index] for index in self.route_indices(width)]

    def extra_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if name.startswith("extra_"):
                yield parameter

    def _extra_update(self, state: Tensor, inputs: Tensor, dilation: int) -> Tensor:
        normalized = self.extra_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gates = torch.sigmoid(F.conv1d(combined, self.extra_gate_weight, self.extra_gate_bias, padding=dilation, dilation=dilation))
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        candidate = torch.tanh(F.conv1d(candidate_inputs, self.extra_candidate_weight, self.extra_candidate_bias, padding=dilation, dilation=dilation))
        return (1.0 - update) * state + update * candidate

    def forward(self, features: Tensor, *, ablate_extra: bool = False) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1] != 6 or not torch.isfinite(features).all():
            raise ValueError("features must be finite [batch, 6, width]")
        old = self.source.stem(features)
        extra = self.extra_stem(features) if self.extra_hidden else None
        for dilation in self.dilation_schedule(features.shape[-1]):
            old = self.source.recurrent_update(old, features, int(dilation))
            if extra is not None:
                extra = self._extra_update(extra, features, int(dilation))
        normalized = F.silu(self.source.output_norm(old))
        logits = self.source.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        quotient = self.source.quotient_head(pooled)
        if extra is not None and not ablate_extra:
            extra_normalized = F.silu(self.extra_output_norm(extra))
            logits = logits + self.extra_local_head(extra_normalized).squeeze(1)
            extra_pool = torch.cat((extra_normalized.mean(dim=-1), extra_normalized.amax(dim=-1)), dim=1)
            quotient = quotient + self.extra_quotient_head(extra_pool)
        return logits, quotient

    def accounting(self, spatial_width: int) -> dict[str, int]:
        calls = len(self.dilation_schedule(spatial_width))
        parameters = sum(parameter.numel() for parameter in self.parameters())
        recurrent_parameters = sum(parameter.numel() for name, parameter in self.named_parameters() if "gate_" in name or "candidate_" in name)
        recurrent_macs = calls * spatial_width * recurrent_parameters
        return {
            "stored_parameters": parameters,
            "recurrent_parameters": recurrent_parameters,
            "recurrent_calls": calls,
            "recurrent_mac_proxy": recurrent_macs,
            "source_hidden": self.source_hidden,
            "extra_hidden": self.extra_hidden,
            "total_hidden": self.hidden,
        }


def exact_embedding_receipt(source: nn.Module, wider: CapacityRecycledConvGRU, features: Tensor) -> dict[str, object]:
    if wider.calls_at_w128 is not None:
        raise ValueError("exact reference receipt requires the full-call route")
    source.eval(); wider.eval()
    with torch.no_grad():
        expected = source(features); observed = wider(features)
    return {
        "old_channel_shapes": [list(expected[0].shape), list(expected[1].shape)],
        "bit_logits_equal": torch.equal(expected[0], observed[0]),
        "quotient_logits_equal": torch.equal(expected[1], observed[1]),
        "maximum_absolute_delta": max(float((expected[0] - observed[0]).abs().max()), float((expected[1] - observed[1]).abs().max())),
    }


def select_resource_route(rows: list[dict[str, object]], *, frozen: dict[str, object], rss_headroom: float, activation_headroom: float) -> dict[str, object] | None:
    """Select without accepting or inspecting labels, loss, validity, or accuracy."""
    forbidden = {"label", "answer", "accuracy", "exact", "valid", "loss"}
    for row in [frozen, *rows]:
        if any(any(token in key.lower() for token in forbidden) for key in row):
            raise ValueError("resource selection received forbidden scientific fields")
    eligible = [
        row for row in rows
        if bool(row["embedding_passed"])
        and float(row["guarded_candidate_seconds"]) <= float(frozen["guarded_candidate_seconds"])
        and int(row["peak_rss_bytes"]) <= int(float(frozen["peak_rss_bytes"]) * rss_headroom)
        and int(row["activation_bytes"]) <= int(float(frozen["activation_bytes"]) * activation_headroom)
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (-int(row["total_hidden"]), float(row["guarded_candidate_seconds"]), int(row["peak_rss_bytes"]), int(row["calls_at_w128"])))[0]


__all__ = ["CapacityRecycledConvGRU", "evenly_spaced_indices", "exact_embedding_receipt", "scaled_call_count", "select_resource_route"]
