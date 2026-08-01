"""Width-independent shared dilated ConvGRU for modular state transitions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ints_to_bits(values: Tensor, width: int) -> Tensor:
    shifts = torch.arange(width, dtype=torch.int64, device=values.device)
    return ((values[:, None] >> shifts[None, :]) & 1).float()


def make_features(state: Tensor, multiplicand: Tensor, modulus: Tensor, bit: Tensor, width: int) -> Tensor:
    batch = state.shape[0]
    bit_plane = bit.float()[:, None].expand(batch, width)
    lsb = torch.zeros((batch, width), device=state.device)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack((ints_to_bits(state, width), ints_to_bits(multiplicand, width), ints_to_bits(modulus, width), bit_plane, lsb, msb), dim=1).mul(2.0).sub(1.0)


class SharedDilatedConvGRU(nn.Module):
    """One input-injected gated cell shared across widths and dilation levels."""

    def __init__(self, width: int, hidden: int = 92, cycles: int = 2) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.cycles = cycles
        channels = hidden + 6
        self.stem = nn.Conv1d(6, hidden, 1)
        self.state_norm = nn.GroupNorm(1, hidden)
        self.gate_weight = nn.Parameter(torch.empty(2 * hidden, channels, 3))
        self.gate_bias = nn.Parameter(torch.zeros(2 * hidden))
        self.candidate_weight = nn.Parameter(torch.empty(hidden, channels, 3))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden))
        self.output_norm = nn.GroupNorm(1, hidden)
        self.local_head = nn.Conv1d(hidden, 1, 1)
        self.quotient_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
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

    def recurrent_update(self, state: Tensor, inputs: Tensor, dilation: int) -> Tensor:
        normalized = self.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gates = torch.sigmoid(F.conv1d(combined, self.gate_weight, self.gate_bias, padding=dilation, dilation=dilation))
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        candidate = torch.tanh(F.conv1d(candidate_inputs, self.candidate_weight, self.candidate_bias, padding=dilation, dilation=dilation))
        return (1.0 - update) * state + update * candidate

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.stem(features)
        for dilation in self.dilation_schedule(features.shape[-1]):
            hidden = self.recurrent_update(hidden, features, dilation)
        normalized = F.silu(self.output_norm(hidden))
        logits = self.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.quotient_head(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.inference_mode()
def predict_state(model: SharedDilatedConvGRU, state: Tensor, multiplicand: Tensor, modulus: Tensor, bit: Tensor) -> tuple[Tensor, Tensor]:
    logits, _ = model(make_features(state, multiplicand, modulus, bit, model.width))
    weights = 1 << torch.arange(model.width, dtype=torch.int64, device=state.device)
    return ((logits >= 0).to(torch.int64) * weights[None, :]).sum(1), logits
