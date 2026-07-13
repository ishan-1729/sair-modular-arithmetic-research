"""Width-shared persistent latent-state Neural GPU for modular multiplication."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PersistentLatentNeuralGPU(nn.Module):
    """Carry a spatial ConvGRU state across multiplier bits without decoding."""

    def __init__(self, width: int, hidden: int = 64, microsteps: int = 2) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden = int(hidden)
        self.microsteps = int(microsteps)
        if self.microsteps < 1:
            raise ValueError("microsteps must be positive")
        self.static_stem = nn.Conv1d(4, hidden, 1)
        channels = hidden + 5
        self.state_norm = nn.GroupNorm(1, hidden)
        self.gates = nn.Conv1d(channels, 2 * hidden, 3, padding=1)
        self.candidate = nn.Conv1d(channels, hidden, 3, padding=1)
        self.output_norm = nn.GroupNorm(1, hidden)
        self.local_head = nn.Conv1d(hidden, 1, 1)

    @staticmethod
    def bits(values: list[int], width: int) -> Tensor:
        return torch.tensor(
            [[(value >> position) & 1 for position in range(width)] for value in values],
            dtype=torch.float32,
        )

    def static_features(
        self, multiplicand: list[int], modulus: list[int], width: int
    ) -> Tensor:
        batch = len(multiplicand)
        lsb = torch.zeros(batch, width)
        msb = torch.zeros(batch, width)
        lsb[:, 0] = 1.0
        msb[:, -1] = 1.0
        return torch.stack(
            (self.bits(multiplicand, width), self.bits(modulus, width), lsb, msb),
            dim=1,
        ).mul(2.0).sub(1.0)

    def initialize(self, static_features: Tensor) -> Tensor:
        return self.static_stem(static_features)

    def recurrent_update(
        self, state: Tensor, static_features: Tensor, input_bit: Tensor
    ) -> Tensor:
        bit_plane = input_bit.float()[:, None, None].expand(
            len(input_bit), 1, static_features.shape[-1]
        ).mul(2.0).sub(1.0)
        step_inputs = torch.cat((static_features, bit_plane), dim=1)
        normalized = self.state_norm(state)
        reset, update = torch.sigmoid(
            self.gates(torch.cat((normalized, step_inputs), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((reset * normalized, step_inputs), dim=1))
        )
        return (1.0 - update) * state + update * candidate

    def decode(self, state: Tensor) -> Tensor:
        return self.local_head(F.silu(self.output_norm(state))).squeeze(1)

    def forward(
        self,
        multiplier_bits: Tensor,
        static_features: Tensor,
        return_all: bool = False,
    ) -> Tensor:
        if multiplier_bits.ndim != 2:
            raise ValueError("multiplier_bits must have shape [batch, steps]")
        if static_features.shape[1] != 4:
            raise ValueError("static features require four channels")
        self.width = static_features.shape[-1]
        state = self.initialize(static_features)
        outputs = []
        for position in range(multiplier_bits.shape[1]):
            for _ in range(self.microsteps):
                state = self.recurrent_update(
                    state, static_features, multiplier_bits[:, position]
                )
            if return_all:
                outputs.append(self.decode(state))
        if return_all:
            return torch.stack(outputs, dim=1)
        return self.decode(state)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
