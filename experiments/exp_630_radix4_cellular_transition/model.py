"""Width-shared radix-4 cellular transition model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Radix4SharedDilatedConvGRU(nn.Module):
    """Consume two multiplier bits per learned recurrent transition."""

    input_channels = 7
    quotient_classes = 7

    def __init__(self, width: int, hidden: int = 92, cycles: int = 2) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden = int(hidden)
        self.cycles = int(cycles)
        channels = self.hidden + self.input_channels
        self.stem = nn.Conv1d(self.input_channels, self.hidden, 1)
        self.state_norm = nn.GroupNorm(1, self.hidden)
        self.gate_weight = nn.Parameter(torch.empty(2 * self.hidden, channels, 3))
        self.gate_bias = nn.Parameter(torch.zeros(2 * self.hidden))
        self.candidate_weight = nn.Parameter(torch.empty(self.hidden, channels, 3))
        self.candidate_bias = nn.Parameter(torch.zeros(self.hidden))
        self.output_norm = nn.GroupNorm(1, self.hidden)
        self.local_head = nn.Conv1d(self.hidden, 1, 1)
        self.quotient_head = nn.Sequential(
            nn.Linear(2 * self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.quotient_classes),
        )
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
        for cycle in range(self.cycles):
            schedule.extend(dilations if cycle % 2 == 0 else reversed(dilations))
        return schedule

    def recurrent_update(
        self, state: torch.Tensor, inputs: torch.Tensor, dilation: int
    ) -> torch.Tensor:
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

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} feature planes")
        hidden = self.stem(features)
        for dilation in self.dilation_schedule(features.shape[-1]):
            hidden = self.recurrent_update(hidden, features, dilation)
        normalized = F.silu(self.output_norm(hidden))
        logits = self.local_head(normalized).squeeze(1)
        pooled = torch.cat(
            (normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1
        )
        return logits, self.quotient_head(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

