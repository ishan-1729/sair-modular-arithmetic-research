"""Mergeable low-rank adaptation of the cellular recurrent core."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RecurrentCoreLoRACell(nn.Module):
    """Adapt gate and candidate kernels while freezing the trained base."""

    def __init__(self, base: nn.Module, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.width = int(base.width)
        gate_out, gate_in, kernel = base.gate_weight.shape
        candidate_out, candidate_in, candidate_kernel = base.candidate_weight.shape
        if (gate_in, kernel) != (candidate_in, candidate_kernel):
            raise ValueError("gate and candidate kernels must share flattened inputs")
        flattened = gate_in * kernel
        self.gate_a = nn.Parameter(torch.empty(self.rank, flattened))
        self.gate_b = nn.Parameter(torch.zeros(gate_out, self.rank))
        self.candidate_a = nn.Parameter(torch.empty(self.rank, flattened))
        self.candidate_b = nn.Parameter(torch.zeros(candidate_out, self.rank))
        nn.init.kaiming_uniform_(self.gate_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def delta_gate(self) -> Tensor:
        return (self.gate_b @ self.gate_a).view_as(self.base.gate_weight) * self.scaling

    def delta_candidate(self) -> Tensor:
        return (
            (self.candidate_b @ self.candidate_a).view_as(self.base.candidate_weight)
            * self.scaling
        )

    def recurrent_update(
        self,
        state: Tensor,
        inputs: Tensor,
        dilation: int,
        gate_weight: Tensor,
        candidate_weight: Tensor,
    ) -> Tensor:
        normalized = self.base.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gates = torch.sigmoid(
            F.conv1d(
                combined,
                gate_weight,
                self.base.gate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        candidate = torch.tanh(
            F.conv1d(
                candidate_inputs,
                candidate_weight,
                self.base.candidate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        return (1.0 - update) * state + update * candidate

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        self.base.width = self.width
        gate_weight = self.base.gate_weight + self.delta_gate()
        candidate_weight = self.base.candidate_weight + self.delta_candidate()
        hidden = self.base.stem(features)
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            hidden = self.recurrent_update(
                hidden, features, dilation, gate_weight, candidate_weight
            )
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)

    def merged_base(self) -> nn.Module:
        """Return an ordinary base cell with the trained deltas folded in."""
        merged = copy.deepcopy(self.base)
        with torch.no_grad():
            merged.gate_weight.add_(self.delta_gate())
            merged.candidate_weight.add_(self.delta_candidate())
        for parameter in merged.parameters():
            parameter.requires_grad_(True)
        return merged

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
