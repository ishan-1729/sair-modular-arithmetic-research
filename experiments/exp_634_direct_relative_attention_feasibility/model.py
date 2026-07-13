"""Width-shared direct modular-product attention and convolution models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def ints_to_bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def make_features(
    left: list[int], right: list[int], modulus: list[int], width: int
) -> torch.Tensor:
    """Return identical width-shared token features for both model arms."""

    if not (len(left) == len(right) == len(modulus)):
        raise ValueError("input batches must have equal length")
    batch = len(left)
    lsb = torch.zeros(batch, width)
    msb = torch.zeros(batch, width)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    position = torch.linspace(0.0, 1.0, width)
    sine = torch.sin(math.pi * position)[None].expand(batch, width)
    cosine = torch.cos(math.pi * position)[None].expand(batch, width)
    binary = torch.stack(
        (ints_to_bits(left, width), ints_to_bits(right, width), ints_to_bits(modulus, width), lsb, msb),
        dim=-1,
    ).mul(2.0).sub(1.0)
    return torch.cat((binary, sine[..., None], cosine[..., None]), dim=-1)


class RecurrentAttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, feedforward),
            nn.GELU(),
            nn.Linear(feedforward, hidden),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden = hidden + attended
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class DirectRelativeAttention(nn.Module):
    """Apply one globally connected block recurrently across learned depth."""

    input_features = 7

    def __init__(
        self,
        width: int,
        hidden: int = 80,
        heads: int = 4,
        feedforward: int = 160,
        depth_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by attention heads")
        if width < 2 or depth_multiplier < 1:
            raise ValueError("width and depth multiplier must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        self.depth_multiplier = int(depth_multiplier)
        self.input_projection = nn.Linear(self.input_features, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.block = RecurrentAttentionBlock(hidden, heads, feedforward, dropout)
        self.reinjection_logit = nn.Parameter(torch.tensor(-2.0))
        self.output_norm = nn.LayerNorm(hidden)
        self.local_head = nn.Linear(hidden, 1)

    def iterations(self, width: int) -> int:
        return self.depth_multiplier * math.ceil(math.log2(width))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_features:
            raise ValueError("features must have shape [batch, width, 7]")
        self.width = features.shape[1]
        embedded = self.input_norm(self.input_projection(features))
        hidden = embedded
        reinjection = torch.sigmoid(self.reinjection_logit)
        for _ in range(self.iterations(self.width)):
            hidden = self.block(hidden + reinjection * embedded)
        return self.local_head(self.output_norm(hidden)).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DirectDilatedConvGRU(nn.Module):
    """Parameter-matched width-shared direct convolution control."""

    input_features = 7

    def __init__(self, width: int, hidden: int = 72, cycles: int = 2) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden = int(hidden)
        self.cycles = int(cycles)
        channels = hidden + self.input_features
        self.stem = nn.Conv1d(self.input_features, hidden, 1)
        self.state_norm = nn.GroupNorm(1, hidden)
        self.gate_weight = nn.Parameter(torch.empty(2 * hidden, channels, 3))
        self.gate_bias = nn.Parameter(torch.zeros(2 * hidden))
        self.candidate_weight = nn.Parameter(torch.empty(hidden, channels, 3))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden))
        self.output_norm = nn.GroupNorm(1, hidden)
        self.local_head = nn.Conv1d(hidden, 1, 1)
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_weight, a=math.sqrt(5))

    def dilation_schedule(self, width: int) -> list[int]:
        maximum = max(1, 1 << max(0, math.ceil(math.log2(width)) - 1))
        dilations: list[int] = []
        value = 1
        while value <= maximum:
            dilations.append(value)
            value *= 2
        schedule: list[int] = []
        for cycle in range(self.cycles):
            schedule.extend(dilations if cycle % 2 == 0 else reversed(dilations))
        return schedule

    def recurrent_update(
        self, state: torch.Tensor, inputs: torch.Tensor, dilation: int
    ) -> torch.Tensor:
        normalized = self.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        reset, update = torch.sigmoid(
            F.conv1d(
                combined,
                self.gate_weight,
                self.gate_bias,
                padding=dilation,
                dilation=dilation,
            )
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            F.conv1d(
                torch.cat((reset * normalized, inputs), dim=1),
                self.candidate_weight,
                self.candidate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        return (1.0 - update) * state + update * candidate

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_features:
            raise ValueError("features must have shape [batch, width, 7]")
        self.width = features.shape[1]
        inputs = features.transpose(1, 2)
        hidden = self.stem(inputs)
        for dilation in self.dilation_schedule(self.width):
            hidden = self.recurrent_update(hidden, inputs, dilation)
        return self.local_head(F.silu(self.output_norm(hidden))).squeeze(1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
