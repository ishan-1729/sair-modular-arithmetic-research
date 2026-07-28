"""Visit-untied teacher and SVD-initialized physical-visit Conv1d LoRA.

This is the SAIR ConvGRU analogue of RRT equations 3--5.  Inference contains
only learned tensor operations; exact integer arithmetic remains in offline
training/evaluation helpers.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


AdapterKind = Literal["visit", "global"]


def _layout(base: nn.Module) -> tuple[int, int, int, int]:
    go, gi, gk = base.gate_weight.shape
    co, ci, ck = base.candidate_weight.shape
    if (gi, gk) != (ci, ck):
        raise ValueError("gate and candidate kernels must have the same flattened input")
    return go, co, gi * gk, 2 * int(base.cycles) * 0 + int(base.cycles)


class _VisitMixin:
    max_dilation_levels: int
    slot_count: int

    def slot_for(self, cycle: int, dilation: int) -> int:
        if cycle < 0 or cycle >= int(self.base.cycles):
            raise ValueError("cycle outside frozen topology")
        if dilation < 1 or dilation & (dilation - 1):
            raise ValueError("dilation must be a positive power of two")
        level = dilation.bit_length() - 1
        if level >= self.max_dilation_levels:
            raise ValueError("dilation exceeds frozen physical-visit bank")
        return cycle * self.max_dilation_levels + level

    def active_slots(self, width: int) -> list[int]:
        schedule = self.base.dilation_schedule(width)
        levels = len(schedule) // int(self.base.cycles)
        return [self.slot_for(index // levels, int(dilation)) for index, dilation in enumerate(schedule)]

    def _recurrent_update(self, state: Tensor, inputs: Tensor, dilation: int, gate: Tensor, candidate: Tensor) -> Tensor:
        normalized = self.base.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gates = torch.sigmoid(F.conv1d(combined, gate, self.base.gate_bias, padding=dilation, dilation=dilation))
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        proposed = torch.tanh(F.conv1d(candidate_inputs, candidate, self.base.candidate_bias, padding=dilation, dilation=dilation))
        return (1.0 - update) * state + update * proposed

    def _finish(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)


class UntiedVisitTeacher(nn.Module, _VisitMixin):
    """Full physical-visit teacher initialized as exact recurrent-kernel copies."""

    def __init__(self, base: nn.Module, *, max_dilation_levels: int = 8) -> None:
        super().__init__()
        if int(base.cycles) != 2:
            raise ValueError("EXP675 freezes two CYCLE traversals")
        self.base = copy.deepcopy(base)
        self.width = int(base.width)
        self.max_dilation_levels = int(max_dilation_levels)
        self.slot_count = int(base.cycles) * self.max_dilation_levels
        self.gate_weights = nn.Parameter(base.gate_weight.detach().repeat(self.slot_count, 1, 1, 1))
        self.candidate_weights = nn.Parameter(base.candidate_weight.detach().repeat(self.slot_count, 1, 1, 1))
        # Kernels are untied; all remaining transformations stay copied and shared.
        self.base.gate_weight.requires_grad_(False)
        self.base.candidate_weight.requires_grad_(False)

    def exact_copy_identity(self) -> bool:
        return all(torch.equal(item, self.base.gate_weight) for item in self.gate_weights) and all(
            torch.equal(item, self.base.candidate_weight) for item in self.candidate_weights
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1] != 6 or not torch.isfinite(features).all():
            raise ValueError("features must be finite [batch, 6, width]")
        schedule = self.base.dilation_schedule(features.shape[-1])
        slots = self.active_slots(features.shape[-1])
        hidden = self.base.stem(features)
        for dilation, slot in zip(schedule, slots, strict=True):
            hidden = self._recurrent_update(hidden, features, int(dilation), self.gate_weights[slot], self.candidate_weights[slot])
        return self._finish(hidden)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


class RRTVisitLoRA(nn.Module, _VisitMixin):
    """Shared base plus fixed physical-visit low-rank recurrent-kernel deltas."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        max_dilation_levels: int = 8,
        kind: AdapterKind = "visit",
    ) -> None:
        super().__init__()
        if rank < 1 or kind not in ("visit", "global"):
            raise ValueError("positive rank and visit/global kind required")
        self.base = copy.deepcopy(base)
        self.width = int(base.width)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.kind = kind
        self.max_dilation_levels = int(max_dilation_levels)
        self.physical_slot_count = int(base.cycles) * self.max_dilation_levels
        self.slot_count = self.physical_slot_count if kind == "visit" else 1
        gate_out, candidate_out, flattened, _ = _layout(base)
        self.gate_a = nn.Parameter(torch.empty(self.slot_count, self.rank, flattened))
        self.gate_b = nn.Parameter(torch.empty(self.slot_count, gate_out, self.rank))
        self.candidate_a = nn.Parameter(torch.empty(self.slot_count, self.rank, flattened))
        self.candidate_b = nn.Parameter(torch.empty(self.slot_count, candidate_out, self.rank))
        self.zero_adapters()

    def zero_adapters(self) -> None:
        with torch.no_grad():
            self.gate_a.zero_(); self.gate_b.zero_(); self.candidate_a.zero_(); self.candidate_b.zero_()

    @staticmethod
    def truncated_svd_factors(residual: Tensor, rank: int, scaling: float) -> tuple[Tensor, Tensor, dict[str, float | int]]:
        matrix = residual.reshape(residual.shape[0], -1).to(dtype=torch.float64)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        retained = min(int(rank), int(s.numel()))
        # B@A*scaling reconstructs U_r Sigma_r V_r^T.
        b = (u[:, :retained] * s[:retained]) / scaling
        a = vh[:retained]
        approximation = (b @ a) * scaling
        denominator = float(matrix.square().sum().sqrt())
        error = float((matrix - approximation).square().sum().sqrt())
        meta = {
            "requested_rank": int(rank),
            "effective_rank": retained,
            "residual_frobenius": denominator,
            "reconstruction_frobenius": float(approximation.square().sum().sqrt()),
            "error_frobenius": error,
            "relative_error": 0.0 if denominator == 0.0 else error / denominator,
            "retained_energy": 1.0 if denominator == 0.0 else float(s[:retained].square().sum() / s.square().sum()),
        }
        return a.to(residual.dtype), b.to(residual.dtype), meta

    def initialize_from_teacher(self, teacher: UntiedVisitTeacher) -> dict[str, list[dict[str, float | int]]]:
        if self.kind != "visit" or teacher.slot_count != self.physical_slot_count:
            raise ValueError("SVD teacher initialization requires aligned visit slots")
        report: dict[str, list[dict[str, float | int]]] = {"gate": [], "candidate": []}
        with torch.no_grad():
            for slot in range(self.slot_count):
                for name, source, shared, target_a, target_b in (
                    ("gate", teacher.gate_weights[slot], self.base.gate_weight, self.gate_a, self.gate_b),
                    ("candidate", teacher.candidate_weights[slot], self.base.candidate_weight, self.candidate_a, self.candidate_b),
                ):
                    a, b, meta = self.truncated_svd_factors(source - shared, self.rank, self.scaling)
                    target_a[slot].zero_(); target_b[slot].zero_()
                    target_a[slot, : a.shape[0]].copy_(a)
                    target_b[slot, :, : b.shape[1]].copy_(b)
                    report[name].append({"slot": slot, **meta})
        return report

    def initialize_equal_norm_random(self, teacher: UntiedVisitTeacher, *, seed: int, shuffle_slots: bool = False) -> None:
        """Equal-Frobenius random subspaces, optionally assigned to shuffled slots."""
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(self.slot_count, generator=generator).tolist() if shuffle_slots else list(range(self.slot_count))
        with torch.no_grad():
            for destination, source_slot in enumerate(order):
                for source, shared, target_a, target_b in (
                    (teacher.gate_weights[source_slot], self.base.gate_weight, self.gate_a, self.gate_b),
                    (teacher.candidate_weights[source_slot], self.base.candidate_weight, self.candidate_a, self.candidate_b),
                ):
                    residual = (source - shared).reshape(source.shape[0], -1)
                    left = torch.randn(residual.shape[0], self.rank, generator=generator, dtype=residual.dtype)
                    right = torch.randn(self.rank, residual.shape[1], generator=generator, dtype=residual.dtype)
                    delta = left @ right
                    norm = residual.norm()
                    delta.mul_(norm / delta.norm().clamp_min(torch.finfo(delta.dtype).tiny))
                    target_b[destination].copy_(left)
                    target_a[destination].copy_(right * (norm / (left @ right).norm().clamp_min(torch.finfo(delta.dtype).tiny)) / self.scaling)

    def _adapter_slot(self, physical_slot: int, permutation: Sequence[int] | None) -> int:
        if self.kind == "global":
            return 0
        if permutation is None:
            return physical_slot
        if len(permutation) != self.slot_count or sorted(permutation) != list(range(self.slot_count)):
            raise ValueError("slot permutation must be a complete bijection")
        return int(permutation[physical_slot])

    def delta_gate(self, slot: int) -> Tensor:
        return ((self.gate_b[slot] @ self.gate_a[slot]) * self.scaling).view_as(self.base.gate_weight)

    def delta_candidate(self, slot: int) -> Tensor:
        return ((self.candidate_b[slot] @ self.candidate_a[slot]) * self.scaling).view_as(self.base.candidate_weight)

    def forward(
        self,
        features: Tensor,
        *,
        adapter_enabled: bool = True,
        slot_permutation: Sequence[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1] != 6 or not torch.isfinite(features).all():
            raise ValueError("features must be finite [batch, 6, width]")
        schedule = self.base.dilation_schedule(features.shape[-1])
        slots = self.active_slots(features.shape[-1])
        hidden = self.base.stem(features)
        for dilation, physical in zip(schedule, slots, strict=True):
            gate, candidate = self.base.gate_weight, self.base.candidate_weight
            if adapter_enabled:
                slot = self._adapter_slot(physical, slot_permutation)
                gate = gate + self.delta_gate(slot)
                candidate = candidate + self.delta_candidate(slot)
            hidden = self._recurrent_update(hidden, features, int(dilation), gate, candidate)
        return self._finish(hidden)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [self.gate_a, self.gate_b, self.candidate_a, self.candidate_b]

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.base.parameters())

    def joint_parameters(self) -> list[nn.Parameter]:
        return self.shared_parameters() + self.adapter_parameters()

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def reconstruction_error(self, teacher: UntiedVisitTeacher) -> dict[str, list[float]]:
        result = {"gate": [], "candidate": []}
        with torch.no_grad():
            for slot in range(self.physical_slot_count):
                for name, source, shared, delta in (
                    ("gate", teacher.gate_weights[slot], self.base.gate_weight, self.delta_gate(slot)),
                    ("candidate", teacher.candidate_weights[slot], self.base.candidate_weight, self.delta_candidate(slot)),
                ):
                    residual = source - shared
                    result[name].append(float((residual - delta).norm() / residual.norm().clamp_min(torch.finfo(residual.dtype).tiny)))
        return result


def clone_state(module: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
