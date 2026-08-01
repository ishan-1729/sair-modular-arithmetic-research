"""Training-only slot teacher and deployed rank-4 factored ConvGRU student."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _svd(residual: Tensor, rank: int) -> tuple[Tensor, Tensor, dict[str, float | int]]:
    """Native deterministic CPU truncated SVD in raw Conv1d matrix layout."""
    matrix = residual.reshape(residual.shape[0], -1).to(torch.float64)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    kept = min(rank, s.numel())
    b, a = u[:, :kept] * s[:kept], vh[:kept]
    approx = b @ a
    norm = matrix.norm()
    return a.to(residual.dtype), b.to(residual.dtype), {
        "requested_rank": rank,
        "effective_rank": kept,
        "relative_error": float((matrix - approx).norm() / norm.clamp_min(torch.finfo(matrix.dtype).tiny)),
        "retained_energy": float(s[:kept].square().sum() / s.square().sum().clamp_min(torch.finfo(s.dtype).tiny)),
        "residual_frobenius": float(norm),
    }


class UntiedPhysicalSlotTeacher(nn.Module):
    """Offline-only h56 teacher with one raw-kernel copy per outward visit."""

    def __init__(self, base: nn.Module, dilations: Sequence[int]) -> None:
        super().__init__()
        if int(base.cycles) != 1:
            raise ValueError("C4 teacher requires the frozen one-cycle core")
        self.base = copy.deepcopy(base)
        self.dilations = tuple(map(int, dilations))
        self.gate_weights = nn.Parameter(base.gate_weight.detach().repeat(len(self.dilations), 1, 1, 1))
        self.candidate_weights = nn.Parameter(base.candidate_weight.detach().repeat(len(self.dilations), 1, 1, 1))
        self.base.gate_weight.requires_grad_(False)
        self.base.candidate_weight.requires_grad_(False)

    def exact_copy_identity(self) -> bool:
        return all(torch.equal(item, self.base.gate_weight) for item in self.gate_weights) and all(
            torch.equal(item, self.base.candidate_weight) for item in self.candidate_weights)

    def _update(self, hidden: Tensor, inputs: Tensor, dilation: int, gate: Tensor, candidate: Tensor) -> Tensor:
        normalized = self.base.state_norm(hidden)
        gates = torch.sigmoid(F.conv1d(torch.cat((normalized, inputs), 1), gate, self.base.gate_bias,
                                       padding=dilation, dilation=dilation))
        reset, update = gates.chunk(2, 1)
        proposed = torch.tanh(F.conv1d(torch.cat((reset * normalized, inputs), 1), candidate,
                                      self.base.candidate_bias, padding=dilation, dilation=dilation))
        return (1 - update) * hidden + update * proposed

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        schedule = self.base.dilation_schedule(features.shape[-1])
        hidden = self.base.stem(features)
        for dilation in schedule:
            slot = self.dilations.index(int(dilation))
            hidden = self._update(hidden, features, int(dilation), self.gate_weights[slot], self.candidate_weights[slot])
        normalized = F.silu(self.base.output_norm(hidden))
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), 1)
        return self.base.local_head(normalized).squeeze(1), self.base.quotient_head(pooled)


class DilationDirectionFactoredStudent(nn.Module):
    """One-cycle shared core plus rank-4 raw-kernel direction/dilation factors."""

    INTERVENTIONS = {"normal", "zero", "permute_dilations", "reverse_directions", "random_subspace", "randomize_adapters"}

    def __init__(self, base: nn.Module, rank: int, high_dilations: Sequence[int]) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        self.width = int(base.width)
        self.rank = int(rank)
        self.high_dilations = tuple(map(int, high_dilations))
        go, gi, gk = self.base.gate_weight.shape
        co, ci, ck = self.base.candidate_weight.shape
        if (gi, gk) != (ci, ck) or gk != 3:
            raise ValueError("C4 requires matching raw Conv1d width-3 matrices")
        flat = gi * gk
        # Direction factors address left/right raw-kernel columns; dilation factors
        # address residual marginals for visits whose dilation is at least eight.
        self.gate_direction_a = nn.Parameter(torch.zeros(2, rank, flat))
        self.gate_direction_b = nn.Parameter(torch.zeros(2, go, rank))
        self.candidate_direction_a = nn.Parameter(torch.zeros(2, rank, flat))
        self.candidate_direction_b = nn.Parameter(torch.zeros(2, co, rank))
        self.gate_dilation_a = nn.Parameter(torch.zeros(len(self.high_dilations), rank, flat))
        self.gate_dilation_b = nn.Parameter(torch.zeros(len(self.high_dilations), go, rank))
        self.candidate_dilation_a = nn.Parameter(torch.zeros(len(self.high_dilations), rank, flat))
        self.candidate_dilation_b = nn.Parameter(torch.zeros(len(self.high_dilations), co, rank))
        self.register_buffer("random_gate_direction", torch.zeros(2, go, gi, gk), persistent=False)
        self.register_buffer("random_candidate_direction", torch.zeros(2, co, ci, ck), persistent=False)
        self.register_buffer("random_gate_dilation", torch.zeros(len(self.high_dilations), go, gi, gk), persistent=False)
        self.register_buffer("random_candidate_dilation", torch.zeros(len(self.high_dilations), co, ci, ck), persistent=False)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [value for name, value in self.named_parameters() if "direction_" in name or "dilation_" in name]

    @staticmethod
    def _delta(a: Tensor, b: Tensor, shape: torch.Size) -> Tensor:
        return (b @ a).view(shape)

    def _direction_delta(self, kind: str, reverse: bool = False) -> Tensor:
        weight = getattr(self.base, f"{kind}_weight")
        a = getattr(self, f"{kind}_direction_a")
        b = getattr(self, f"{kind}_direction_b")
        order = (1, 0) if reverse else (0, 1)
        pieces = [self._delta(a[index], b[index], weight.shape) for index in order]
        masks = [torch.zeros_like(weight), torch.zeros_like(weight)]
        masks[0][..., 0] = 1; masks[1][..., 2] = 1
        return pieces[0] * masks[0] + pieces[1] * masks[1]

    def _dilation_delta(self, kind: str, dilation: int, permutation: Sequence[int] | None = None) -> Tensor:
        weight = getattr(self.base, f"{kind}_weight")
        if dilation not in self.high_dilations:
            return torch.zeros_like(weight)
        index = self.high_dilations.index(dilation)
        if permutation is not None:
            if sorted(permutation) != list(range(len(self.high_dilations))):
                raise ValueError("dilation permutation must be a bijection")
            index = int(permutation[index])
        return self._delta(getattr(self, f"{kind}_dilation_a")[index],
                           getattr(self, f"{kind}_dilation_b")[index], weight.shape)

    def effective_delta(self, kind: str, dilation: int, intervention: str = "normal",
                        generator: torch.Generator | None = None) -> Tensor:
        if intervention not in self.INTERVENTIONS:
            raise ValueError(f"unsupported adapter intervention: {intervention}")
        weight = getattr(self.base, f"{kind}_weight")
        if intervention == "zero":
            return torch.zeros_like(weight)
        if intervention == "random_subspace":
            direction = getattr(self, f"random_{kind}_direction").sum(0)
            dilation_delta = torch.zeros_like(weight) if dilation not in self.high_dilations else getattr(
                self, f"random_{kind}_dilation")[self.high_dilations.index(dilation)]
            return direction + dilation_delta
        permutation = tuple(reversed(range(len(self.high_dilations)))) if intervention == "permute_dilations" else None
        delta = self._direction_delta(kind, intervention == "reverse_directions") + self._dilation_delta(kind, dilation, permutation)
        if intervention == "randomize_adapters":
            if generator is None:
                raise ValueError("randomization requires a generator")
            noise = torch.randn(delta.shape, generator=generator, device=delta.device, dtype=delta.dtype)
            delta = noise * (delta.norm() / noise.norm().clamp_min(torch.finfo(noise.dtype).tiny))
        return delta

    def step(self, features: Tensor, adapter_access: bool, intervention: str = "normal",
             generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, dict[str, Any]]:
        self.base.width = features.shape[-1]
        hidden = self.base.stem(features); norms = []
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            gate_delta = self.effective_delta("gate", int(dilation), intervention, generator)
            candidate_delta = self.effective_delta("candidate", int(dilation), intervention, generator)
            gate = self.base.gate_weight + gate_delta * float(adapter_access)
            candidate = self.base.candidate_weight + candidate_delta * float(adapter_access)
            normalized = self.base.state_norm(hidden)
            gates = torch.sigmoid(F.conv1d(torch.cat((normalized, features), 1), gate, self.base.gate_bias,
                                           padding=int(dilation), dilation=int(dilation)))
            reset, update = gates.chunk(2, 1)
            proposed = torch.tanh(F.conv1d(torch.cat((reset * normalized, features), 1), candidate,
                                          self.base.candidate_bias, padding=int(dilation), dilation=int(dilation)))
            hidden = (1 - update) * hidden + update * proposed
            norms.append(float((gate_delta.norm() + candidate_delta.norm()).detach()))
        normalized = F.silu(self.base.output_norm(hidden)); pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), 1)
        return self.base.local_head(normalized).squeeze(1), self.base.quotient_head(pooled), {
            "adapter_delta_norm": float(sum(norms) / max(1, len(norms))), "adapter_calls": len(norms)}

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        logits, quotient, _ = self.step(features, True)
        return logits, quotient

    def initialize_from_teacher(self, teacher: UntiedPhysicalSlotTeacher, seed: int) -> dict[str, Any]:
        report: dict[str, Any] = {"gate": [], "candidate": [], "library": "torch.linalg.svd", "rank": self.rank}
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for kind in ("gate", "candidate"):
                source = getattr(teacher, f"{kind}_weights")
                centroid = source.mean(0)
                getattr(self.base, f"{kind}_weight").copy_(centroid)
                residual = source - centroid
                # Raw-kernel left/right marginals are the direction components.
                direction_sum = torch.zeros_like(centroid)
                for direction, kernel_index in enumerate((0, 2)):
                    marginal = torch.zeros_like(centroid); marginal[..., kernel_index] = residual[..., kernel_index].mean(0)
                    a, b, meta = _svd(marginal, self.rank)
                    getattr(self, f"{kind}_direction_a")[direction].copy_(a)
                    getattr(self, f"{kind}_direction_b")[direction].copy_(b)
                    direction_sum += (b @ a).view_as(centroid)
                    report[kind].append({"component": "direction", "index": direction, **meta})
                for index, dilation in enumerate(self.high_dilations):
                    slot = teacher.dilations.index(dilation)
                    marginal = residual[slot] - direction_sum
                    a, b, meta = _svd(marginal, self.rank)
                    getattr(self, f"{kind}_dilation_a")[index].copy_(a)
                    getattr(self, f"{kind}_dilation_b")[index].copy_(b)
                    report[kind].append({"component": "dilation", "dilation": dilation, **meta})
                for bank_name, bank in (("direction", getattr(self, f"random_{kind}_direction")),
                                        ("dilation", getattr(self, f"random_{kind}_dilation"))):
                    for index in range(bank.shape[0]):
                        target = self._direction_delta(kind)[...] if bank_name == "direction" else self._dilation_delta(kind, self.high_dilations[index])
                        noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
                        bank[index].copy_(noise * (target.norm() / noise.norm().clamp_min(torch.finfo(noise.dtype).tiny)))
        report["reconstruction"] = self.reconstruction_report(teacher)
        return report

    def reconstruction_loss(self, teacher: UntiedPhysicalSlotTeacher) -> Tensor:
        losses = []
        for slot, dilation in enumerate(teacher.dilations):
            for kind in ("gate", "candidate"):
                target = getattr(teacher, f"{kind}_weights")[slot]
                effective = getattr(self.base, f"{kind}_weight") + self.effective_delta(kind, dilation)
                losses.append(F.mse_loss(effective, target) / target.square().mean().clamp_min(1e-8))
        return torch.stack(losses).mean()

    def reconstruction_report(self, teacher: UntiedPhysicalSlotTeacher) -> dict[str, float]:
        errors, random_errors = [], []
        with torch.no_grad():
            for slot, dilation in enumerate(teacher.dilations):
                for kind in ("gate", "candidate"):
                    target = getattr(teacher, f"{kind}_weights")[slot]
                    residual = target - getattr(self.base, f"{kind}_weight")
                    errors.append(float((residual - self.effective_delta(kind, dilation)).norm() / residual.norm().clamp_min(1e-12)))
                    random_errors.append(float((residual - self.effective_delta(kind, dilation, "random_subspace")).norm() / residual.norm().clamp_min(1e-12)))
        return {"mean_rank4_relative_error": sum(errors) / len(errors),
                "mean_equal_norm_random_relative_error": sum(random_errors) / len(random_errors)}
