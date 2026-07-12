"""Submission adapter for the promoted learned Horner/cellular ensemble."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from modchallenge.interface.base_model import ModularMultiplicationModel
from safetensors.torch import load_file
from torch import Tensor, nn


WIDTH = 9
TIER3_WIDTH = 16
TIER4_WIDTH = 32
TIER5_WIDTH = 64
HIDDEN = 48
TIER4_HIDDEN = 92
DILATIONS = (1, 2, 4, 8, 1, 2, 4)


def _ints_to_bits(values: Tensor, width: int = WIDTH) -> Tensor:
    shifts = torch.arange(width, dtype=torch.int64, device=values.device)
    return ((values[:, None] >> shifts[None, :]) & 1).float()


def _features(
    state: Tensor,
    multiplicand: Tensor,
    modulus: Tensor,
    bit: Tensor,
    width: int = WIDTH,
) -> Tensor:
    batch = state.shape[0]
    bit_plane = bit.float()[:, None].expand(batch, width)
    lsb = torch.zeros((batch, width), device=state.device)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    planes = torch.stack(
        (
            _ints_to_bits(state, width),
            _ints_to_bits(multiplicand, width),
            _ints_to_bits(modulus, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    )
    return planes.mul(2.0).sub(1.0)


def _bigint_features(
    state: list[int],
    multiplicand: list[int],
    modulus: list[int],
    bit: list[int],
    width: int,
) -> Tensor:
    """Encode Python big integers without routing them through signed int64."""
    batch = len(state)
    planes = torch.empty((batch, 6, width), dtype=torch.float32)
    for row, (state_value, multiplicand_value, modulus_value, input_bit) in enumerate(
        zip(state, multiplicand, modulus, bit, strict=True)
    ):
        for position in range(width):
            planes[row, 0, position] = (state_value >> position) & 1
            planes[row, 1, position] = (multiplicand_value >> position) & 1
            planes[row, 2, position] = (modulus_value >> position) & 1
        planes[row, 3, :] = input_bit
        planes[row, 4, :] = 0
        planes[row, 5, :] = 0
        planes[row, 4, 0] = 1
        planes[row, 5, -1] = 1
    return planes.mul_(2.0).sub_(1.0)


def _decode_bigint_logits(logits: Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in logits.ge(0).tolist()
    ]


class _ResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, HIDDEN)
        self.conv1 = nn.Conv1d(HIDDEN, HIDDEN, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, HIDDEN)
        self.conv2 = nn.Conv1d(HIDDEN, HIDDEN, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class _HornerTCN9(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(6, HIDDEN, 1)
        self.position = nn.Parameter(torch.zeros(1, HIDDEN, WIDTH))
        self.modulus_embedding = nn.Embedding(258, HIDDEN)
        self.body = nn.ModuleList(_ResidualBlock(dilation) for dilation in DILATIONS)
        self.head_norm = nn.GroupNorm(1, HIDDEN)
        self.local_head = nn.Conv1d(HIDDEN, 1, 1)
        self.global_head = nn.Linear(HIDDEN * WIDTH, WIDTH)
        # This auxiliary training head is retained for exact checkpoint loading.
        self.quotient_head = nn.Linear(HIDDEN * WIDTH, 3)

    def forward(self, features: Tensor, modulus: Tensor) -> Tensor:
        hidden = self.stem(features) + self.position
        hidden = hidden + self.modulus_embedding(modulus.clamp(0, 257))[:, :, None]
        for block in self.body:
            hidden = block(hidden)
        hidden = F.silu(self.head_norm(hidden))
        return self.local_head(hidden).squeeze(1) + self.global_head(hidden.flatten(1))


class _SharedContextHornerTCN16(nn.Module):
    """Width-16 TCN whose modulus context is learned from the modulus bits."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(6, HIDDEN, 1)
        self.position = nn.Parameter(torch.zeros(1, HIDDEN, TIER3_WIDTH))
        self.modulus_context = nn.Sequential(
            nn.Linear(TIER3_WIDTH, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )
        self.body = nn.ModuleList(_ResidualBlock(dilation) for dilation in DILATIONS)
        self.head_norm = nn.GroupNorm(1, HIDDEN)
        self.local_head = nn.Conv1d(HIDDEN, 1, 1)
        self.global_head = nn.Linear(HIDDEN * TIER3_WIDTH, TIER3_WIDTH)
        self.quotient_head = nn.Linear(HIDDEN * TIER3_WIDTH, 3)

    def forward(self, features: Tensor) -> Tensor:
        context = self.modulus_context(features[:, 2, :])[:, :, None]
        hidden = self.stem(features) + self.position + context
        for block in self.body:
            hidden = block(hidden)
        hidden = F.silu(self.head_norm(hidden))
        return self.local_head(hidden).squeeze(1) + self.global_head(hidden.flatten(1))


class _SharedDilatedConvGRU32(nn.Module):
    """Width-independent cellular transition specialized through width 32."""

    def __init__(self) -> None:
        super().__init__()
        channels = TIER4_HIDDEN + 6
        self.stem = nn.Conv1d(6, TIER4_HIDDEN, 1)
        self.state_norm = nn.GroupNorm(1, TIER4_HIDDEN)
        self.gate_weight = nn.Parameter(torch.empty(2 * TIER4_HIDDEN, channels, 3))
        self.gate_bias = nn.Parameter(torch.zeros(2 * TIER4_HIDDEN))
        self.candidate_weight = nn.Parameter(torch.empty(TIER4_HIDDEN, channels, 3))
        self.candidate_bias = nn.Parameter(torch.zeros(TIER4_HIDDEN))
        self.output_norm = nn.GroupNorm(1, TIER4_HIDDEN)
        self.local_head = nn.Conv1d(TIER4_HIDDEN, 1, 1)
        self.quotient_head = nn.Sequential(
            nn.Linear(2 * TIER4_HIDDEN, TIER4_HIDDEN),
            nn.SiLU(),
            nn.Linear(TIER4_HIDDEN, 3),
        )

    @staticmethod
    def _dilation_schedule(width: int) -> list[int]:
        maximum = max(1, 1 << max(0, math.ceil(math.log2(width)) - 1))
        dilations = []
        value = 1
        while value <= maximum:
            dilations.append(value)
            value *= 2
        return dilations + list(reversed(dilations))

    def _recurrent_update(
        self, state: Tensor, inputs: Tensor, dilation: int
    ) -> Tensor:
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
        for dilation in self._dilation_schedule(features.shape[-1]):
            hidden = self._recurrent_update(hidden, features, dilation)
        normalized = F.silu(self.output_norm(hidden))
        logits = self.local_head(normalized).squeeze(1)
        pooled = torch.cat(
            (normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1
        )
        return logits, self.quotient_head(pooled)


class Tier2HornerModel(ModularMultiplicationModel):
    """Pairwise residue front end followed by a trained recurrent transition."""

    def __init__(self) -> None:
        self.network: _HornerTCN9 | None = None
        self.tier3_network: _SharedContextHornerTCN16 | None = None
        self.tier4_network: _SharedDilatedConvGRU32 | None = None
        self.tier5_network: _SharedDilatedConvGRU32 | None = None

    def load(self, model_dir: str) -> None:
        weights = load_file(str(Path(model_dir) / "weights.safetensors"))
        network = _HornerTCN9()
        network.load_state_dict(
            {key.removeprefix("tier2."): value for key, value in weights.items() if key.startswith("tier2.")}
        )
        network.eval()
        self.network = network
        tier3_network = _SharedContextHornerTCN16()
        tier3_network.load_state_dict(
            {key.removeprefix("tier3."): value for key, value in weights.items() if key.startswith("tier3.")}
        )
        tier3_network.eval()
        self.tier3_network = tier3_network
        tier4_network = _SharedDilatedConvGRU32()
        tier4_network.load_state_dict(
            {
                key.removeprefix("tier4."): value
                for key, value in weights.items()
                if key.startswith("tier4.")
            }
        )
        tier4_network.eval()
        self.tier4_network = tier4_network
        tier5_network = _SharedDilatedConvGRU32()
        tier5_network.load_state_dict(
            {
                key.removeprefix("tier5."): value
                for key, value in weights.items()
                if key.startswith("tier5.")
            }
        )
        tier5_network.eval()
        self.tier5_network = tier5_network

    def preprocess_a(self, a: str) -> str:
        return a

    def preprocess_b(self, b: str) -> str:
        return b

    def preprocess_p(self, p: str) -> str:
        return p

    @torch.inference_mode()
    def predict_digits(self, a_enc: str, b_enc: str, p_enc: str) -> list[int]:
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    @torch.inference_mode()
    def predict_digits_batch(
        self, inputs: list[tuple[str, str, str]]
    ) -> list[list[int]]:
        if (
            self.network is None
            or self.tier3_network is None
            or self.tier4_network is None
            or self.tier5_network is None
        ):
            raise RuntimeError("model has not been loaded")
        outputs = [[0] for _ in inputs]
        groups: dict[int, list[tuple[int, int, int, int]]] = {7: [], 8: []}
        tier3_group: list[tuple[int, int, int, int]] = []
        tier4_group: list[tuple[int, int, int, int]] = []
        tier5_group: list[tuple[int, int, int, int]] = []
        for index, (a_enc, b_enc, p_enc) in enumerate(inputs):
            modulus_value = int(p_enc)
            if modulus_value > (1 << TIER5_WIDTH) - 1:
                continue
            # Pairwise reductions are explicitly sanctioned by the challenge.
            # Only the trained recurrence produces the final product/remainder.
            multiplier_value = int(a_enc) % modulus_value
            multiplicand_value = int(b_enc) % modulus_value
            item = (index, multiplier_value, multiplicand_value, modulus_value)
            if modulus_value <= 257:
                first_bit = 8 if modulus_value == 257 else 7
                groups[first_bit].append(item)
            elif modulus_value <= 65535:
                tier3_group.append(item)
            elif modulus_value <= (1 << TIER4_WIDTH) - 1:
                tier4_group.append(item)
            else:
                tier5_group.append(item)

        weights = 1 << torch.arange(WIDTH, dtype=torch.int64)
        for first_bit, group in groups.items():
            if not group:
                continue
            multiplier = torch.tensor([item[1] for item in group], dtype=torch.int64)
            multiplicand = torch.tensor([item[2] for item in group], dtype=torch.int64)
            modulus = torch.tensor([item[3] for item in group], dtype=torch.int64)
            state = torch.zeros(len(group), dtype=torch.int64)
            for bit_index in range(first_bit, -1, -1):
                bit = (multiplier >> bit_index) & 1
                logits = self.network(
                    _features(state, multiplicand, modulus, bit), modulus
                )
                state = (
                    (logits >= 0).to(torch.int64) * weights[None, :]
                ).sum(1)
            for item, predicted in zip(group, state.tolist(), strict=True):
                # Malformed states fall back to zero; there is no arithmetic repair.
                outputs[item[0]] = [predicted if predicted < item[3] else 0]

        if tier3_group:
            multiplier = torch.tensor([item[1] for item in tier3_group], dtype=torch.int64)
            multiplicand = torch.tensor([item[2] for item in tier3_group], dtype=torch.int64)
            modulus = torch.tensor([item[3] for item in tier3_group], dtype=torch.int64)
            state = torch.zeros(len(tier3_group), dtype=torch.int64)
            tier3_weights = 1 << torch.arange(TIER3_WIDTH, dtype=torch.int64)
            for bit_index in range(TIER3_WIDTH - 1, -1, -1):
                bit = (multiplier >> bit_index) & 1
                logits = self.tier3_network(
                    _features(state, multiplicand, modulus, bit, TIER3_WIDTH)
                )
                state = ((logits >= 0).to(torch.int64) * tier3_weights[None, :]).sum(1)
            for item, predicted in zip(tier3_group, state.tolist(), strict=True):
                outputs[item[0]] = [predicted if predicted < item[3] else 0]

        if tier4_group:
            multiplier = torch.tensor(
                [item[1] for item in tier4_group], dtype=torch.int64
            )
            multiplicand = torch.tensor(
                [item[2] for item in tier4_group], dtype=torch.int64
            )
            modulus = torch.tensor(
                [item[3] for item in tier4_group], dtype=torch.int64
            )
            state = torch.zeros(len(tier4_group), dtype=torch.int64)
            tier4_weights = 1 << torch.arange(TIER4_WIDTH, dtype=torch.int64)
            for bit_index in range(TIER4_WIDTH - 1, -1, -1):
                bit = (multiplier >> bit_index) & 1
                logits, _ = self.tier4_network(
                    _features(state, multiplicand, modulus, bit, TIER4_WIDTH)
                )
                state = (
                    (logits >= 0).to(torch.int64) * tier4_weights[None, :]
                ).sum(1)
            for item, predicted in zip(tier4_group, state.tolist(), strict=True):
                outputs[item[0]] = [predicted if predicted < item[3] else 0]

        if tier5_group:
            multiplier = [item[1] for item in tier5_group]
            multiplicand = [item[2] for item in tier5_group]
            modulus = [item[3] for item in tier5_group]
            state = [0] * len(tier5_group)
            for bit_index in range(TIER5_WIDTH - 1, -1, -1):
                bit = [(value >> bit_index) & 1 for value in multiplier]
                logits, _ = self.tier5_network(
                    _bigint_features(
                        state, multiplicand, modulus, bit, TIER5_WIDTH
                    )
                )
                state = _decode_bigint_logits(logits)
            for item, predicted in zip(tier5_group, state, strict=True):
                outputs[item[0]] = [predicted if predicted < item[3] else 0]
        return outputs

    def max_batch_size(self) -> int:
        return 256
