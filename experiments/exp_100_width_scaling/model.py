"""Shared bit-conditioned learned Horner transition used by EXP-100."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ints_to_bits(values: Tensor, width: int) -> Tensor:
    shifts = torch.arange(width, dtype=torch.int64, device=values.device)
    return ((values[:, None] >> shifts[None, :]) & 1).float()


def make_features(
    state: Tensor,
    multiplicand: Tensor,
    modulus: Tensor,
    bit: Tensor,
    width: int,
) -> Tensor:
    batch = state.shape[0]
    bit_plane = bit.float()[:, None].expand(batch, width)
    lsb = torch.zeros((batch, width), device=state.device)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (
            ints_to_bits(state, width),
            ints_to_bits(multiplicand, width),
            ints_to_bits(modulus, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul(2.0).sub(1.0)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class SharedContextHornerTCN(nn.Module):
    """Fixed-width TCN with learned context derived from modulus bits."""

    def __init__(self, width: int, hidden: int = 48, blocks: int = 7) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.blocks_count = blocks
        self.stem = nn.Conv1d(6, hidden, 1)
        self.position = nn.Parameter(torch.zeros(1, hidden, width))
        self.modulus_context = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        cycle = max(1, math.ceil(math.log2(width)))
        self.dilations = [1 << (index % cycle) for index in range(blocks)]
        self.body = nn.ModuleList(
            ResidualBlock(hidden, dilation) for dilation in self.dilations
        )
        self.head_norm = nn.GroupNorm(1, hidden)
        self.local_head = nn.Conv1d(hidden, 1, 1)
        self.global_head = nn.Linear(hidden * width, width)
        self.quotient_head = nn.Linear(hidden * width, 3)
        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        modulus_bits = features[:, 2, :]
        context = self.modulus_context(modulus_bits)[:, :, None]
        hidden = self.stem(features) + self.position + context
        for block in self.body:
            hidden = block(hidden)
        hidden = F.silu(self.head_norm(hidden))
        flat = hidden.flatten(1)
        logits = self.local_head(hidden).squeeze(1) + self.global_head(flat)
        return logits, self.quotient_head(flat)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.inference_mode()
def predict_state(
    model: SharedContextHornerTCN,
    state: Tensor,
    multiplicand: Tensor,
    modulus: Tensor,
    bit: Tensor,
) -> tuple[Tensor, Tensor]:
    logits, _ = model(
        make_features(state, multiplicand, modulus, bit, model.width)
    )
    predicted_bits = (logits >= 0).to(torch.int64)
    weights = 1 << torch.arange(
        model.width, dtype=torch.int64, device=state.device
    )
    return (predicted_bits * weights[None, :]).sum(1), logits
