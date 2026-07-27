"""Exact-initialized adapters for the raw recurrent ConvGRU kernels.

Only learned PyTorch tensor operations occur here. Integer truth generation and
evaluation remain outside the inference module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


AdapterKind = Literal["visit_dilation", "global"]


class VisitDilationAdapterCell(nn.Module):
    """Freeze a cellular base and add low-rank raw-kernel deltas per visit.

    Slots are addressed by ``cycle * max_dilation_levels + log2(dilation)``.
    Consequently the same semantic cycle/dilation pair has the same slot at
    W64, W128, and W256 even though the schedules contain 12, 14, and 16 calls.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        kind: AdapterKind,
        rank: int,
        alpha: float,
        max_dilation_levels: int = 8,
    ) -> None:
        super().__init__()
        if kind not in ("visit_dilation", "global"):
            raise ValueError(f"unknown adapter kind: {kind}")
        if rank < 1 or max_dilation_levels < 1:
            raise ValueError("rank and max_dilation_levels must be positive")
        if int(base.cycles) != 2:
            raise ValueError("EXP-651 freezes exactly two recurrent cycles")
        self.base = base
        self.kind = kind
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.max_dilation_levels = int(max_dilation_levels)
        self.slot_count = 2 * self.max_dilation_levels if kind == "visit_dilation" else 1
        self.width = int(base.width)

        gate_out, gate_in, gate_kernel = base.gate_weight.shape
        candidate_out, candidate_in, candidate_kernel = base.candidate_weight.shape
        if (gate_in, gate_kernel) != (candidate_in, candidate_kernel):
            raise ValueError("gate and candidate kernels must share flattened inputs")
        flattened = gate_in * gate_kernel
        self.gate_a = nn.Parameter(torch.empty(self.slot_count, self.rank, flattened))
        self.gate_b = nn.Parameter(torch.zeros(self.slot_count, gate_out, self.rank))
        self.candidate_a = nn.Parameter(torch.empty(self.slot_count, self.rank, flattened))
        self.candidate_b = nn.Parameter(torch.zeros(self.slot_count, candidate_out, self.rank))
        nn.init.kaiming_uniform_(self.gate_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def slot_for(self, cycle: int, dilation: int) -> int:
        if cycle not in (0, 1):
            raise ValueError("cycle must be zero or one")
        if dilation < 1 or dilation & (dilation - 1):
            raise ValueError("dilation must be a positive power of two")
        level = dilation.bit_length() - 1
        if level >= self.max_dilation_levels:
            raise ValueError("dilation exceeds the frozen adapter bank")
        return 0 if self.kind == "global" else cycle * self.max_dilation_levels + level

    def _slot(self, slot: int, permutation: Sequence[int] | None) -> int:
        if permutation is None:
            return slot
        if len(permutation) != self.slot_count or sorted(permutation) != list(range(self.slot_count)):
            raise ValueError("slot permutation must be a complete bijection")
        return int(permutation[slot])

    def delta_gate(self, slot: int) -> Tensor:
        return (self.gate_b[slot] @ self.gate_a[slot]).view_as(self.base.gate_weight) * self.scaling

    def delta_candidate(self, slot: int) -> Tensor:
        return (self.candidate_b[slot] @ self.candidate_a[slot]).view_as(self.base.candidate_weight) * self.scaling

    def recurrent_update(
        self, state: Tensor, inputs: Tensor, dilation: int, gate_weight: Tensor, candidate_weight: Tensor
    ) -> Tensor:
        normalized = self.base.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gates = torch.sigmoid(F.conv1d(combined, gate_weight, self.base.gate_bias, padding=dilation, dilation=dilation))
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        candidate = torch.tanh(F.conv1d(candidate_inputs, candidate_weight, self.base.candidate_bias, padding=dilation, dilation=dilation))
        return (1.0 - update) * state + update * candidate

    def forward(
        self,
        features: Tensor,
        *,
        adapter_enabled: bool = True,
        slot_permutation: Sequence[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1] != 6:
            raise ValueError("features must have shape [batch, 6, width]")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        self.base.width = self.width
        schedule = self.base.dilation_schedule(features.shape[-1])
        levels = len(schedule) // 2
        hidden = self.base.stem(features)
        for visit, dilation in enumerate(schedule):
            slot = self._slot(self.slot_for(visit // levels, int(dilation)), slot_permutation)
            gate = self.base.gate_weight
            candidate = self.base.candidate_weight
            if adapter_enabled:
                gate = gate + self.delta_gate(slot)
                candidate = candidate + self.delta_candidate(slot)
            hidden = self.recurrent_update(hidden, features, int(dilation), gate, candidate)
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [self.gate_a, self.gate_b, self.candidate_a, self.candidate_b]

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def active_slot_count(self, width: int) -> int:
        schedule = self.base.dilation_schedule(width)
        levels = len(schedule) // 2
        return len({self.slot_for(index // levels, dilation) for index, dilation in enumerate(schedule)})
