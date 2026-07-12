"""Submission adapter for the promoted Tier-2/Tier-3 learned Horner ensemble."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from modchallenge.interface.base_model import ModularMultiplicationModel
from safetensors.torch import load_file
from torch import Tensor, nn


WIDTH = 9
TIER3_WIDTH = 16
HIDDEN = 48
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


class Tier2HornerModel(ModularMultiplicationModel):
    """Pairwise residue front end followed by a trained recurrent transition."""

    def __init__(self) -> None:
        self.network: _HornerTCN9 | None = None
        self.tier3_network: _SharedContextHornerTCN16 | None = None

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
        if self.network is None or self.tier3_network is None:
            raise RuntimeError("model has not been loaded")
        outputs = [[0] for _ in inputs]
        groups: dict[int, list[tuple[int, int, int, int]]] = {7: [], 8: []}
        tier3_group: list[tuple[int, int, int, int]] = []
        for index, (a_enc, b_enc, p_enc) in enumerate(inputs):
            modulus_value = int(p_enc)
            if modulus_value > 65535:
                continue
            # Pairwise reductions are explicitly sanctioned by the challenge.
            # Only the trained recurrence produces the final product/remainder.
            multiplier_value = int(a_enc) % modulus_value
            multiplicand_value = int(b_enc) % modulus_value
            item = (index, multiplier_value, multiplicand_value, modulus_value)
            if modulus_value <= 257:
                first_bit = 8 if modulus_value == 257 else 7
                groups[first_bit].append(item)
            else:
                tier3_group.append(item)

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
        return outputs

    def max_batch_size(self) -> int:
        return 256
