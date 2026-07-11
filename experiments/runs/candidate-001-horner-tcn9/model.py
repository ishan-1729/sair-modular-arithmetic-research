"""Learned nine-bit Horner transition candidate.

The transition is intentionally predicted end-to-end by trained weights.  The
module does not contain an addition, multiplication, division, or reduction
routine in its forward path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


WIDTH = 9


def ints_to_bits(values: Tensor, width: int = WIDTH) -> Tensor:
    shifts = torch.arange(width, dtype=torch.int64, device=values.device)
    return ((values[:, None] >> shifts[None, :]) & 1).float()


def make_features(state: Tensor, multiplicand: Tensor, modulus: Tensor, bit: Tensor) -> Tensor:
    """Return raw LSB-first factors and position markers only."""
    batch = state.shape[0]
    bit_plane = bit.float()[:, None].expand(batch, WIDTH)
    lsb = torch.zeros((batch, WIDTH), device=state.device)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    features = torch.stack(
        (
            ints_to_bits(state),
            ints_to_bits(multiplicand),
            ints_to_bits(modulus),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    )
    return features.mul(2.0).sub(1.0)


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


class HornerTCN9(nn.Module):
    """Full-canvas TCN with a small learned modulus-conditioning embedding."""

    def __init__(self, hidden: int = 48, blocks: int = 7, modulus_embedding: bool = True) -> None:
        super().__init__()
        self.width = WIDTH
        self.hidden = hidden
        self.blocks_count = blocks
        self.modulus_embedding_enabled = modulus_embedding
        self.stem = nn.Conv1d(6, hidden, 1)
        self.position = nn.Parameter(torch.zeros(1, hidden, WIDTH))
        self.modulus_embedding = nn.Embedding(258, hidden) if modulus_embedding else None
        cycle = math.ceil(math.log2(WIDTH))
        self.dilations = [1 << (index % cycle) for index in range(blocks)]
        self.body = nn.ModuleList(ResidualBlock(hidden, dilation) for dilation in self.dilations)
        self.head_norm = nn.GroupNorm(1, hidden)
        self.local_head = nn.Conv1d(hidden, 1, 1)
        self.global_head = nn.Linear(hidden * WIDTH, WIDTH)
        self.quotient_head = nn.Linear(hidden * WIDTH, 3)
        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)

    def forward(self, features: Tensor, modulus: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.stem(features) + self.position
        if self.modulus_embedding is not None:
            hidden = hidden + self.modulus_embedding(modulus.clamp(0, 257))[:, :, None]
        for block in self.body:
            hidden = block(hidden)
        hidden = F.silu(self.head_norm(hidden))
        flat = hidden.flatten(1)
        logits = self.local_head(hidden).squeeze(1) + self.global_head(flat)
        quotient_logits = self.quotient_head(flat)
        return logits, quotient_logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.inference_mode()
def predict_state(
    model: HornerTCN9,
    state: Tensor,
    multiplicand: Tensor,
    modulus: Tensor,
    bit: Tensor,
) -> tuple[Tensor, Tensor]:
    logits, _ = model(make_features(state, multiplicand, modulus, bit), modulus)
    predicted_bits = (logits >= 0).to(torch.int64)
    weights = 1 << torch.arange(WIDTH, dtype=torch.int64, device=state.device)
    return (predicted_bits * weights[None, :]).sum(1), logits
