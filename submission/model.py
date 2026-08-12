"""Submission adapter with Candidate005 through Tier 6 and EXP701 at Tier 7."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
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
TIER6_WIDTH = 128
TIER7_WIDTH = 256
TIER6_ROUTE_FRACTION = 0.40
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


def _bigint_bits(values: list[int], width: int) -> Tensor:
    return torch.tensor(
        [
            [(value >> position) & 1 for position in range(width)]
            for value in values
        ],
        dtype=torch.float32,
    )


def _bigint_features(
    state: list[int],
    multiplicand: list[int],
    modulus: list[int],
    bit: list[int],
    width: int,
) -> Tensor:
    """Encode Python big integers without signed-int64 conversion."""
    batch = len(state)
    bit_plane = torch.tensor(bit, dtype=torch.float32)[:, None].expand(
        batch, width
    )
    lsb = torch.zeros((batch, width), dtype=torch.float32)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (
            _bigint_bits(state, width),
            _bigint_bits(multiplicand, width),
            _bigint_bits(modulus, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul_(2.0).sub_(1.0)


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
    """Width-independent cellular transition used by the wider specialists."""

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
        """Evaluate only the trained remainder head in a CPU-friendly layout."""
        inputs = features.unsqueeze(2).contiguous(
            memory_format=torch.channels_last
        )
        hidden = F.conv2d(
            inputs,
            self.stem.weight.unsqueeze(2),
            self.stem.bias,
        )
        gate_weight = self.gate_weight.unsqueeze(2)
        candidate_weight = self.candidate_weight.unsqueeze(2)
        for dilation in self._dilation_schedule(features.shape[-1]):
            normalized = self.state_norm(hidden)
            combined = torch.cat((normalized, inputs), dim=1)
            gates = torch.sigmoid(
                F.conv2d(
                    combined,
                    gate_weight,
                    self.gate_bias,
                    padding=(0, dilation),
                    dilation=(1, dilation),
                )
            )
            reset, update = gates.chunk(2, dim=1)
            candidate_inputs = torch.cat(
                (reset * normalized, inputs), dim=1
            )
            candidate = torch.tanh(
                F.conv2d(
                    candidate_inputs,
                    candidate_weight,
                    self.candidate_bias,
                    padding=(0, dilation),
                    dilation=(1, dilation),
                )
            )
            hidden = (1.0 - update) * hidden + update * candidate
        normalized = F.silu(self.output_norm(hidden))
        logits = F.conv2d(
            normalized,
            self.local_head.weight.unsqueeze(2),
            self.local_head.bias,
        ).squeeze(2).squeeze(1)
        # The quotient head was training-only and every inference caller discards it.
        return logits, logits.new_empty((logits.shape[0], 0))


class Tier2HornerModel(ModularMultiplicationModel):
    """Pairwise residue front end followed by a trained recurrent transition."""

    def __init__(self) -> None:
        self.network: _HornerTCN9 | None = None
        self.tier3_network: _SharedContextHornerTCN16 | None = None
        self.tier4_network: _SharedDilatedConvGRU32 | None = None
        self.tier5_network: _SharedDilatedConvGRU32 | None = None
        self.tier6_network: _SharedDilatedConvGRU32 | None = None
        self.tier7_network: _SharedDilatedConvGRU32 | None = None

    def load(self, model_dir: str) -> None:
        # Retain four-way intra-op execution for the narrow TCN routes. Wider
        # cellular routes switch to independent single-threaded shards locally.
        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
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
        tier6_network = _SharedDilatedConvGRU32()
        tier6_network.load_state_dict(
            {
                key.removeprefix("tier6."): value
                for key, value in weights.items()
                if key.startswith("tier6.")
            }
        )
        tier6_network.eval()
        self.tier6_network = tier6_network
        tier7_network = _SharedDilatedConvGRU32()
        tier7_network.load_state_dict(
            {
                key.removeprefix("tier7."): value
                for key, value in weights.items()
                if key.startswith("tier7.")
            }
        )
        tier7_network.eval()
        self.tier7_network = tier7_network

    def preprocess_a(self, a: str) -> str:
        return a

    def preprocess_b(self, b: str) -> str:
        return b

    def preprocess_p(self, p: str) -> str:
        return p

    @torch.inference_mode()
    def predict_digits(self, a_enc: str, b_enc: str, p_enc: str) -> list[int]:
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    def _tier2_rollout_shard(
        self,
        items: list[tuple[int, int, int, int]],
        first_bit: int,
    ) -> list[int]:
        if self.network is None:
            raise RuntimeError("model has not been loaded")
        with torch.inference_mode():
            multiplier = torch.tensor([item[1] for item in items], dtype=torch.int64)
            multiplicand = torch.tensor([item[2] for item in items], dtype=torch.int64)
            modulus = torch.tensor([item[3] for item in items], dtype=torch.int64)
            state = torch.zeros(len(items), dtype=torch.int64)
            weights = 1 << torch.arange(WIDTH, dtype=torch.int64)
            for bit_index in range(first_bit, -1, -1):
                bit = (multiplier >> bit_index) & 1
                logits = self.network(
                    _features(state, multiplicand, modulus, bit), modulus
                )
                state = (
                    (logits >= 0).to(torch.int64) * weights[None, :]
                ).sum(1)
            return state.tolist()

    def _tier3_rollout_shard(
        self,
        items: list[tuple[int, int, int, int]],
    ) -> list[int]:
        if self.tier3_network is None:
            raise RuntimeError("model has not been loaded")
        with torch.inference_mode():
            multiplier = torch.tensor([item[1] for item in items], dtype=torch.int64)
            multiplicand = torch.tensor([item[2] for item in items], dtype=torch.int64)
            modulus = torch.tensor([item[3] for item in items], dtype=torch.int64)
            state = torch.zeros(len(items), dtype=torch.int64)
            weights = 1 << torch.arange(TIER3_WIDTH, dtype=torch.int64)
            for bit_index in range(TIER3_WIDTH - 1, -1, -1):
                bit = (multiplier >> bit_index) & 1
                logits = self.tier3_network(
                    _features(state, multiplicand, modulus, bit, TIER3_WIDTH)
                )
                state = (
                    (logits >= 0).to(torch.int64) * weights[None, :]
                ).sum(1)
            return state.tolist()

    @staticmethod
    def _parallel_items(
        items: list[tuple[int, int, int, int]],
        worker,
    ) -> list[int]:
        workers = min(4, len(items))
        if workers <= 1:
            return worker(items)
        bounds = [len(items) * index // workers for index in range(workers + 1)]
        shards = [items[bounds[index] : bounds[index + 1]] for index in range(workers)]
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(worker, shards))
            return [value for shard in results for value in shard]
        finally:
            torch.set_num_threads(previous_threads)

    def _tier6_rollout(
        self,
        multiplier: list[int],
        multiplicand: list[int],
        modulus: list[int],
    ) -> tuple[list[int], list[float]]:
        if self.tier6_network is None:
            raise RuntimeError("model has not been loaded")
        return self._parallel_wide_rollout(
            self.tier6_network,
            multiplier,
            multiplicand,
            modulus,
            TIER6_WIDTH,
            track_margin=True,
        )

    @staticmethod
    def _wide_rollout_shard(
        network: _SharedDilatedConvGRU32,
        multiplier: list[int],
        multiplicand: list[int],
        modulus: list[int],
        width: int,
        track_margin: bool,
    ) -> tuple[list[int], list[float]]:
        with torch.inference_mode():
            batch = len(multiplier)
            multiplier_bits = _bigint_bits(multiplier, width)
            features = torch.empty((batch, 6, width), dtype=torch.float32)
            features[:, 1, :] = _bigint_bits(multiplicand, width).mul(2).sub(1)
            features[:, 2, :] = _bigint_bits(modulus, width).mul(2).sub(1)
            features[:, 4, :].fill_(-1)
            features[:, 4, 0] = 1
            features[:, 5, :].fill_(-1)
            features[:, 5, -1] = 1
            state_bits = torch.zeros((batch, width), dtype=torch.bool)
            minimum_margin = torch.full((batch,), math.inf)
            for position in range(width):
                features[:, 0, :] = state_bits.float().mul(2).sub(1)
                features[:, 3, :] = multiplier_bits[
                    :, width - 1 - position, None
                ].mul(2).sub(1)
                logits, _ = network(features)
                if track_margin:
                    minimum_margin = torch.minimum(
                        minimum_margin, logits.abs().amin(dim=1)
                    )
                state_bits = logits.ge(0)
            state = [
                sum(int(bit) << position for position, bit in enumerate(row))
                for row in state_bits.tolist()
            ]
            return state, minimum_margin.tolist()

    def _parallel_wide_rollout(
        self,
        network: _SharedDilatedConvGRU32,
        multiplier: list[int],
        multiplicand: list[int],
        modulus: list[int],
        width: int,
        track_margin: bool = False,
    ) -> tuple[list[int], list[float]]:
        workers = min(4, len(multiplier))
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            if workers <= 1:
                return self._wide_rollout_shard(
                    network, multiplier, multiplicand, modulus, width, track_margin
                )
            bounds = [
                len(multiplier) * index // workers
                for index in range(workers + 1)
            ]
            arguments = [
                (
                    network,
                    multiplier[bounds[index] : bounds[index + 1]],
                    multiplicand[bounds[index] : bounds[index + 1]],
                    modulus[bounds[index] : bounds[index + 1]],
                    width,
                    track_margin,
                )
                for index in range(workers)
            ]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                shards = list(
                    executor.map(
                        lambda values: self._wide_rollout_shard(*values),
                        arguments,
                    )
                )
            state = [value for shard, _ in shards for value in shard]
            margins = [value for _, shard in shards for value in shard]
            return state, margins
        finally:
            torch.set_num_threads(previous_threads)

    def _tier7_rollout(
        self,
        multiplier: list[int],
        multiplicand: list[int],
        modulus: list[int],
    ) -> list[int]:
        if self.tier7_network is None:
            raise RuntimeError("model has not been loaded")
        state, _ = self._parallel_wide_rollout(
            self.tier7_network,
            multiplier,
            multiplicand,
            modulus,
            TIER7_WIDTH,
        )
        return state

    @torch.inference_mode()
    def predict_digits_batch(
        self, inputs: list[tuple[str, str, str]]
    ) -> list[list[int]]:
        if (
            self.network is None
            or self.tier3_network is None
            or self.tier4_network is None
            or self.tier5_network is None
            or self.tier6_network is None
            or self.tier7_network is None
        ):
            raise RuntimeError("model has not been loaded")
        outputs = [[0] for _ in inputs]
        groups: dict[int, list[tuple[int, int, int, int]]] = {7: [], 8: []}
        tier3_group: list[tuple[int, int, int, int]] = []
        tier4_group: list[tuple[int, int, int, int]] = []
        tier5_group: list[tuple[int, int, int, int]] = []
        tier6_group: list[tuple[int, int, int, int]] = []
        tier7_group: list[tuple[int, int, int, int]] = []
        for index, (a_enc, b_enc, p_enc) in enumerate(inputs):
            modulus_value = int(p_enc)
            if modulus_value > (1 << TIER7_WIDTH) - 1:
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
            elif modulus_value <= (1 << TIER5_WIDTH) - 1:
                tier5_group.append(item)
            elif modulus_value <= (1 << TIER6_WIDTH) - 1:
                tier6_group.append(item)
            else:
                tier7_group.append(item)

        weights = 1 << torch.arange(WIDTH, dtype=torch.int64)
        for first_bit, group in groups.items():
            if not group:
                continue
            state = self._parallel_items(
                group,
                lambda items: self._tier2_rollout_shard(items, first_bit),
            )
            for item, predicted in zip(group, state, strict=True):
                # Malformed states fall back to zero; there is no arithmetic repair.
                outputs[item[0]] = [predicted if predicted < item[3] else 0]

        if tier3_group:
            state = self._parallel_items(tier3_group, self._tier3_rollout_shard)
            for item, predicted in zip(tier3_group, state, strict=True):
                outputs[item[0]] = [predicted if predicted < item[3] else 0]

        if tier4_group:
            state, _ = self._parallel_wide_rollout(
                self.tier4_network,
                [item[1] for item in tier4_group],
                [item[2] for item in tier4_group],
                [item[3] for item in tier4_group],
                TIER4_WIDTH,
            )
            for item, predicted in zip(tier4_group, state, strict=True):
                outputs[item[0]] = [predicted if predicted < item[3] else 0]

        if tier5_group:
            multiplier = [item[1] for item in tier5_group]
            multiplicand = [item[2] for item in tier5_group]
            modulus = [item[3] for item in tier5_group]
            state, _ = self._parallel_wide_rollout(
                self.tier5_network,
                multiplier,
                multiplicand,
                modulus,
                TIER5_WIDTH,
            )
            for item, predicted in zip(tier5_group, state, strict=True):
                outputs[item[0]] = [predicted if predicted < item[3] else 0]

        if tier6_group:
            multiplier = [item[1] for item in tier6_group]
            multiplicand = [item[2] for item in tier6_group]
            modulus = [item[3] for item in tier6_group]
            selected, original_margin = self._tier6_rollout(
                multiplier, multiplicand, modulus
            )
            route_count = round(len(tier6_group) * TIER6_ROUTE_FRACTION)
            routed_indices = sorted(
                range(len(tier6_group)),
                key=lambda index: (original_margin[index], index),
            )[:route_count]
            if routed_indices:
                swapped, swapped_margin = self._tier6_rollout(
                    [multiplicand[index] for index in routed_indices],
                    [multiplier[index] for index in routed_indices],
                    [modulus[index] for index in routed_indices],
                )
                for local, index in enumerate(routed_indices):
                    if swapped_margin[local] > original_margin[index]:
                        selected[index] = swapped[local]
            for item, predicted in zip(tier6_group, selected, strict=True):
                outputs[item[0]] = [predicted if predicted < item[3] else 0]

        if tier7_group:
            modulus = [item[3] for item in tier7_group]
            multiplier = [item[1] for item in tier7_group]
            multiplicand = [item[2] for item in tier7_group]
            predicted = self._tier7_rollout(multiplier, multiplicand, modulus)
            for item, value in zip(tier7_group, predicted, strict=True):
                outputs[item[0]] = [value if value < item[3] else 0]
        return outputs

    def max_batch_size(self) -> int:
        return 256
