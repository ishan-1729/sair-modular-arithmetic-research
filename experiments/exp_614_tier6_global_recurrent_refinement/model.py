"""Exactly gated global hidden-state refinement for the cellular transition."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GlobalRefinementBlock(nn.Module):
    """A separately parameterized bidirectional dilation pass over hidden state."""

    def __init__(self, hidden: int = 92, passes: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.passes = passes
        channels = hidden + 6
        self.state_norm = nn.GroupNorm(1, hidden)
        self.gate_weight = nn.Parameter(torch.empty(2 * hidden, channels, 3))
        self.gate_bias = nn.Parameter(torch.zeros(2 * hidden))
        self.candidate_weight = nn.Parameter(torch.empty(hidden, channels, 3))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_weight, a=math.sqrt(5))

    def dilation_schedule(self, width: int) -> list[int]:
        maximum = max(1, 1 << max(0, math.ceil(math.log2(width)) - 1))
        dilations = []
        value = 1
        while value <= maximum:
            dilations.append(value)
            value *= 2
        schedule = []
        for index in range(self.passes):
            schedule.extend(dilations if index % 2 == 0 else reversed(dilations))
        return schedule

    def recurrent_update(self, state: Tensor, inputs: Tensor, dilation: int) -> Tensor:
        normalized = self.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gates = torch.sigmoid(
            F.conv1d(
                combined,
                self.gate_weight,
                self.gate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        candidate = torch.tanh(
            F.conv1d(
                candidate_inputs,
                self.candidate_weight,
                self.candidate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        return (1.0 - update) * state + update * candidate

    def forward(self, hidden: Tensor, features: Tensor) -> Tensor:
        for dilation in self.dilation_schedule(features.shape[-1]):
            hidden = self.recurrent_update(hidden, features, dilation)
        return hidden


class GloballyRefinedCell(nn.Module):
    """Preserve a frozen base exactly and learn global recurrent refinement."""

    def __init__(
        self,
        base: nn.Module,
        refinement: GlobalRefinementBlock,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if int(base.hidden) != refinement.hidden:
            raise ValueError("base and refinement hidden widths must match")
        self.base = base
        self.refinement = refinement
        self.width = int(base.width)
        self.blend = nn.Parameter(torch.zeros(1, int(base.hidden), 1))
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        self.base.width = self.width
        hidden = self.base.stem(features)
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            hidden = self.base.recurrent_update(hidden, features, dilation)
        refined = self.refinement(hidden, features)
        hidden = hidden + torch.tanh(self.blend) * (refined - hidden)
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
