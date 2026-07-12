"""Tensor-native learned correction path conditioned on a frozen base model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConditionedCorrector(nn.Module):
    """Width-shared ConvGRU corrector over inputs and base confidence."""

    def __init__(self, width: int, hidden: int = 48, cycles: int = 2) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.cycles = cycles
        self.input_channels = 8
        channels = hidden + self.input_channels
        self.stem = nn.Conv1d(self.input_channels, hidden, 1)
        self.state_norm = nn.GroupNorm(1, hidden)
        self.gate_weight = nn.Parameter(torch.empty(2 * hidden, channels, 3))
        self.gate_bias = nn.Parameter(torch.zeros(2 * hidden))
        self.candidate_weight = nn.Parameter(torch.empty(hidden, channels, 3))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden))
        self.output_norm = nn.GroupNorm(1, hidden)
        self.local_head = nn.Conv1d(hidden, 1, 1)
        self.quotient_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3)
        )
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_weight, a=math.sqrt(5))
        self.zero_outputs()

    def zero_outputs(self) -> None:
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        nn.init.zeros_(self.quotient_head[-1].weight)
        nn.init.zeros_(self.quotient_head[-1].bias)

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

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.stem(features)
        for dilation in self.dilation_schedule(features.shape[-1]):
            hidden = self.recurrent_update(hidden, features, dilation)
        normalized = F.silu(self.output_norm(hidden))
        logits = self.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.quotient_head(pooled)


class ErrorCorrectedCell(nn.Module):
    """Add learned corrections to a base model without exact arithmetic."""

    def __init__(self, base: nn.Module, corrector: ConditionedCorrector) -> None:
        super().__init__()
        self.base = base
        self.corrector = corrector
        self.width = int(base.width)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def correction_features(self, features: Tensor, base_logits: Tensor) -> Tensor:
        signed_confidence = torch.tanh(base_logits).unsqueeze(1)
        confidence_magnitude = torch.tanh(base_logits.abs()).unsqueeze(1)
        return torch.cat((features, signed_confidence, confidence_magnitude), dim=1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        self.base.width = self.width
        self.corrector.width = self.width
        with torch.no_grad():
            base_logits, base_quotient = self.base(features)
        correction_logits, correction_quotient = self.corrector(
            self.correction_features(features, base_logits)
        )
        return base_logits + correction_logits, base_quotient + correction_quotient

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
