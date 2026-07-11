"""Inference-only learned transition architecture.

This module contains no target modular arithmetic. The neural cell receives
raw binary factor channels and predicts the next fixed-width state bits.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import POCConfig


def ints_to_bits(values: Tensor, width: int) -> Tensor:
    """Convert non-negative int64 values to float LSB-first bit rows."""
    shifts = torch.arange(width, device=values.device, dtype=torch.int64)
    return ((values[:, None] >> shifts[None, :]) & 1).float()


def make_features(state: Tensor, multiplicand: Tensor, modulus: Tensor, bit: Tensor, width: int) -> Tensor:
    """Build raw factor channels without a precomputed addend or reduction."""
    batch = state.shape[0]
    bit_plane = bit.float()[:, None].expand(batch, width)
    lsb_marker = torch.zeros((batch, width), device=state.device)
    msb_marker = torch.zeros_like(lsb_marker)
    lsb_marker[:, 0] = 1.0
    msb_marker[:, -1] = 1.0
    features = torch.stack(
        (
            ints_to_bits(state, width),
            ints_to_bits(multiplicand, width),
            ints_to_bits(modulus, width),
            bit_plane,
            lsb_marker,
            msb_marker,
        ),
        dim=1,
    )
    return features.mul(2.0).sub(1.0)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm(inputs)))
        hidden = self.conv2(F.silu(hidden))
        return inputs + hidden


class HornerTCN(nn.Module):
    """Non-causal dilated TCN transition cell with full-canvas bit logits."""

    def __init__(self, config: POCConfig):
        super().__init__()
        self.width = config.width
        self.stem = nn.Conv1d(6, config.hidden, 1)
        cycle = max(1, math.ceil(math.log2(config.width)))
        self.dilations = [1 << (index % cycle) for index in range(config.blocks)]
        self.body = nn.ModuleList(
            ResidualDilatedBlock(config.hidden, dilation) for dilation in self.dilations
        )
        self.head_norm = nn.GroupNorm(1, config.hidden)
        self.head = nn.Conv1d(config.hidden, 1, 1)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.stem(features)
        for block in self.body:
            hidden = block(hidden)
        return self.head(F.silu(self.head_norm(hidden))).squeeze(1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.inference_mode()
def predict_state(
    model: HornerTCN,
    state: Tensor,
    multiplicand: Tensor,
    modulus: Tensor,
    bit: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run the learned cell and hard-threshold its predicted recurrent state."""
    logits = model(make_features(state, multiplicand, modulus, bit, model.width))
    predicted_bits = (logits >= 0).long()
    weights = 1 << torch.arange(model.width, device=state.device, dtype=torch.int64)
    predicted_state = (predicted_bits * weights[None, :]).sum(dim=1)
    return predicted_state, logits
