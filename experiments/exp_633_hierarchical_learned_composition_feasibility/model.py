"""Width-shared learned binary composition tree for modular products."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def ints_to_bits(values: list[int], width: int) -> torch.Tensor:
    """Encode Python integers as LSB-first float bit rows."""

    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def make_static_features(
    multiplicand: list[int], modulus: list[int], width: int
) -> torch.Tensor:
    """Create width-independent spatial inputs shared by all tree nodes."""

    if len(multiplicand) != len(modulus):
        raise ValueError("multiplicand and modulus batches must match")
    batch = len(multiplicand)
    lsb = torch.zeros(batch, width)
    msb = torch.zeros(batch, width)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (ints_to_bits(multiplicand, width), ints_to_bits(modulus, width), lsb, msb),
        dim=1,
    ).mul(2.0).sub(1.0)


class HierarchicalCompositionTree(nn.Module):
    """Reduce multiplier bits with one learned latent composition operator."""

    static_channels = 4

    def __init__(self, width: int, hidden: int = 56, cycles: int = 1) -> None:
        super().__init__()
        if width < 2 or not math.log2(width).is_integer():
            raise ValueError("width must be a power of two")
        if hidden < 1 or cycles < 1:
            raise ValueError("hidden and cycles must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        self.cycles = int(cycles)
        self.leaf_stem = nn.Conv1d(self.static_channels + 1, hidden, 1)
        composition_channels = 2 * hidden + self.static_channels
        self.merge = nn.Conv1d(composition_channels, hidden, 1)
        recurrent_channels = hidden + composition_channels
        self.state_norm = nn.GroupNorm(1, hidden)
        self.gate_weight = nn.Parameter(
            torch.empty(2 * hidden, recurrent_channels, 3)
        )
        self.gate_bias = nn.Parameter(torch.zeros(2 * hidden))
        self.candidate_weight = nn.Parameter(
            torch.empty(hidden, recurrent_channels, 3)
        )
        self.candidate_bias = nn.Parameter(torch.zeros(hidden))
        self.output_norm = nn.GroupNorm(1, hidden)
        self.local_head = nn.Conv1d(hidden, 1, 1)
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_weight, a=math.sqrt(5))

    def dilation_schedule(self, spatial_width: int) -> list[int]:
        maximum = max(1, 1 << max(0, math.ceil(math.log2(spatial_width)) - 1))
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
        self, state: torch.Tensor, composition_inputs: torch.Tensor, dilation: int
    ) -> torch.Tensor:
        normalized = self.state_norm(state)
        combined = torch.cat((normalized, composition_inputs), dim=1)
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
                torch.cat((reset * normalized, composition_inputs), dim=1),
                self.candidate_weight,
                self.candidate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        return (1.0 - update) * state + update * candidate

    def decode(self, states: torch.Tensor) -> torch.Tensor:
        return self.local_head(F.silu(self.output_norm(states))).squeeze(1)

    def forward(
        self,
        multiplier_bits: torch.Tensor,
        static_features: torch.Tensor,
        return_all_levels: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if multiplier_bits.ndim != 2:
            raise ValueError("multiplier_bits must have shape [batch, leaves]")
        batch, leaves = multiplier_bits.shape
        spatial_width = static_features.shape[-1]
        if leaves != spatial_width or not math.log2(leaves).is_integer():
            raise ValueError("leaf count must equal a power-of-two spatial width")
        if static_features.shape != (batch, self.static_channels, spatial_width):
            raise ValueError("static feature shape does not match multiplier batch")
        self.width = spatial_width
        static_nodes = static_features[:, None].expand(
            batch, leaves, self.static_channels, spatial_width
        )
        bit_planes = multiplier_bits[:, :, None, None].expand(
            batch, leaves, 1, spatial_width
        ).mul(2.0).sub(1.0)
        states = self.leaf_stem(
            torch.cat((bit_planes, static_nodes), dim=2).reshape(
                batch * leaves, self.static_channels + 1, spatial_width
            )
        ).reshape(batch, leaves, self.hidden, spatial_width)
        level_logits: list[torch.Tensor] = []
        nodes = leaves
        while nodes > 1:
            left = states[:, 0::2]
            right = states[:, 1::2]
            parents = nodes // 2
            shared = static_features[:, None].expand(
                batch, parents, self.static_channels, spatial_width
            )
            composition_inputs = torch.cat((left, right, shared), dim=2).reshape(
                batch * parents, 2 * self.hidden + self.static_channels, spatial_width
            )
            merged = torch.tanh(self.merge(composition_inputs))
            for dilation in self.dilation_schedule(spatial_width):
                merged = self.recurrent_update(merged, composition_inputs, dilation)
            logits = self.decode(merged).reshape(batch, parents, spatial_width)
            level_logits.append(logits)
            states = merged.reshape(batch, parents, self.hidden, spatial_width)
            nodes = parents
        root_logits = level_logits[-1][:, 0]
        if return_all_levels:
            return root_logits, level_logits
        return root_logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
