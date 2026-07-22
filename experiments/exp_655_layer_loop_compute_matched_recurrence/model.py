"""Dense, width-flexible Transformer recurrence for EXP-655.

This inference module contains only learned PyTorch operations. Exact integer
arithmetic is deliberately confined to offline data generation and evaluation
in :mod:`run`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

import torch
from torch import Tensor, nn


Arm = Literal[
    "layer_loop_r2",
    "model_loop_r2",
    "r1",
    "untied_equal_compute",
]

PRIMARY_ARMS = ("layer_loop_r2", "model_loop_r2")
ARM_ORDER = (*PRIMARY_ARMS, "r1", "untied_equal_compute")


class DenseBlock(nn.Module):
    """One complete pre-norm self-attention plus MLP block."""

    def __init__(self, hidden: int, heads: int, expansion: int) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.mlp_norm = nn.RMSNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, expansion * hidden),
            nn.SiLU(),
            nn.Linear(expansion * hidden, hidden),
        )

    def forward(self, state: Tensor) -> Tensor:
        normalized = self.attention_norm(state)
        mixed, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        state = state + mixed
        return state + self.mlp(self.mlp_norm(state))


class LayerLoopReasoner(nn.Module):
    """Transformer transition predictor whose only primary difference is order."""

    INPUT_ROLES = 6
    POSITION_BASIS = 7

    def __init__(
        self,
        arm: Arm,
        *,
        hidden: int = 32,
        heads: int = 4,
        stored_blocks: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if arm not in ARM_ORDER:
            raise ValueError(f"unknown arm: {arm}")
        if stored_blocks < 2:
            raise ValueError("EXP-655 requires at least two stored blocks")
        if hidden % heads:
            raise ValueError("hidden size must be divisible by attention heads")
        self.arm = arm
        self.hidden = int(hidden)
        self.stored_blocks = int(stored_blocks)
        self.input_projection = nn.Linear(
            self.INPUT_ROLES + self.POSITION_BASIS, hidden
        )
        block_count = 2 * stored_blocks if arm == "untied_equal_compute" else stored_blocks
        self.blocks = nn.ModuleList(
            DenseBlock(hidden, heads, expansion) for _ in range(block_count)
        )
        self.output_norm = nn.RMSNorm(hidden)
        self.output = nn.Linear(hidden, 1)
        self.quotient_head = nn.Linear(hidden, 3)

    @staticmethod
    def position_basis(
        width: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if width < 2:
            raise ValueError("width must be at least two")
        position = torch.arange(width, device=device, dtype=dtype)
        fraction = position / float(width - 1)
        return torch.stack(
            (
                fraction,
                1.0 - fraction,
                torch.sin(math.pi * fraction),
                torch.cos(math.pi * fraction),
                torch.sin(2.0 * math.pi * fraction),
                position.eq(0).to(dtype),
                position.eq(width - 1).to(dtype),
            ),
            dim=-1,
        )

    def schedule(self, override: str | None = None) -> tuple[int, ...]:
        schedule = self.arm if override is None else override
        if schedule == "layer_loop_r2":
            return tuple(index for index in range(self.stored_blocks) for _ in range(2))
        if schedule == "model_loop_r2":
            return tuple(range(self.stored_blocks)) * 2
        if schedule == "r1":
            return tuple(range(self.stored_blocks))
        if schedule == "untied_equal_compute":
            return tuple(range(2 * self.stored_blocks))
        raise ValueError(f"unknown schedule: {schedule}")

    def encode(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, role, position]")
        if features.shape[1] != self.INPUT_ROLES:
            raise ValueError(f"expected {self.INPUT_ROLES} feature roles")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        width = features.shape[-1]
        raw = features.permute(0, 2, 1)
        basis = self.position_basis(
            width, device=features.device, dtype=features.dtype
        )
        return self.input_projection(
            torch.cat((raw, basis[None].expand(raw.shape[0], -1, -1)), dim=-1)
        )

    def apply_schedule(
        self,
        state: Tensor,
        *,
        schedule: str | None = None,
        block_order: Iterable[int] | None = None,
        visits: int | None = None,
        reset_before_visit: int | None = None,
    ) -> Tensor:
        order = tuple(block_order) if block_order is not None else self.schedule(schedule)
        if visits is not None:
            if visits < 0:
                raise ValueError("visits must be nonnegative")
            order = order[:visits]
        initial = state
        for visit, block_index in enumerate(order):
            if not 0 <= block_index < len(self.blocks):
                raise ValueError("block order references an unavailable block")
            if reset_before_visit is not None and visit == reset_before_visit:
                state = torch.zeros_like(initial)
            state = self.blocks[block_index](state)
        return state

    def forward(
        self,
        features: Tensor,
        *,
        schedule: str | None = None,
        block_order: Iterable[int] | None = None,
        visits: int | None = None,
        reset_before_visit: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        state = self.apply_schedule(
            self.encode(features),
            schedule=schedule,
            block_order=block_order,
            visits=visits,
            reset_before_visit=reset_before_visit,
        )
        state = self.output_norm(state)
        return self.output(state).squeeze(-1), self.quotient_head(state.mean(dim=1))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def complete_block_calls(self, schedule: str | None = None) -> int:
        return len(self.schedule(schedule))


def state_bytes(model: nn.Module) -> bytes:
    """Canonical byte stream used only for exact initialization identity tests."""
    chunks = []
    for name, value in sorted(model.state_dict().items()):
        chunks.extend((name.encode("utf-8"), b"\0", value.detach().cpu().contiguous().numpy().tobytes(), b"\0"))
    return b"".join(chunks)

