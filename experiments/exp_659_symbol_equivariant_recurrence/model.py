"""Bounded-faithful symbol-axis recurrent reasoning models for EXP-659.

The recurrent core follows SE-RRM equations 9--11: position attention,
symbol attention, then a token-wise MLP, each with residual RMSNorm.  The
binary arithmetic interface is deliberately anchored because swapping zero
and one is not a symmetry of the legal modular-arithmetic domain.  Exact
arithmetic never appears in this inference module.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


Arm = Literal["position_only", "symbol_equivariant", "broken_tying"]


class EquivariantBlock(nn.Module):
    """Equations 9--11 with attention over position then symbol axes."""

    def __init__(self, hidden: int, heads: int, expansion: int = 2) -> None:
        super().__init__()
        self.position_attention = nn.MultiheadAttention(
            hidden, heads, batch_first=True
        )
        self.symbol_attention = nn.MultiheadAttention(
            hidden, heads, batch_first=True
        )
        self.position_norm = nn.RMSNorm(hidden)
        self.symbol_norm = nn.RMSNorm(hidden)
        self.mlp_norm = nn.RMSNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, expansion * hidden),
            nn.SiLU(),
            nn.Linear(expansion * hidden, hidden),
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 4:
            raise ValueError("symbol-axis state must have shape [batch, position, symbol, feature]")
        batch, positions, symbols, hidden = state.shape
        by_symbol = state.permute(0, 2, 1, 3).reshape(batch * symbols, positions, hidden)
        mixed, _ = self.position_attention(
            by_symbol, by_symbol, by_symbol, need_weights=False
        )
        state = self.position_norm(by_symbol + mixed).reshape(
            batch, symbols, positions, hidden
        ).permute(0, 2, 1, 3)
        by_position = state.reshape(batch * positions, symbols, hidden)
        mixed, _ = self.symbol_attention(
            by_position, by_position, by_position, need_weights=False
        )
        state = self.symbol_norm(by_position + mixed).reshape(
            batch, positions, symbols, hidden
        )
        return self.mlp_norm(state + self.mlp(state))


class PositionBlock(nn.Module):
    """Matched recurrent position-mixing control without a symbol axis."""

    def __init__(self, hidden: int, heads: int, expansion: int = 2) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.attention_norm = nn.RMSNorm(hidden)
        self.mlp_norm = nn.RMSNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, expansion * hidden),
            nn.SiLU(),
            nn.Linear(expansion * hidden, hidden),
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 3:
            raise ValueError("position state must have shape [batch, position, feature]")
        mixed, _ = self.attention(state, state, state, need_weights=False)
        state = self.attention_norm(state + mixed)
        return self.mlp_norm(state + self.mlp(state))


class SymbolRecurrentReasoner(nn.Module):
    """Width-flexible recurrent transition predictor with matched arms.

    ``symbol_equivariant`` has an exactly tied symbol-axis core.
    ``broken_tying`` adds symbol-indexed affine parameters inside that core.
    ``position_only`` removes the symbol axis and symbol-attention operation.
    """

    SYMBOLS = 2
    INPUT_ROLES = 6
    POSITION_BASIS = 7

    def __init__(
        self,
        arm: Arm,
        *,
        hidden: int = 32,
        position_hidden: int = 37,
        heads: int = 4,
        recurrent_steps: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if arm not in {"position_only", "symbol_equivariant", "broken_tying"}:
            raise ValueError(f"unknown arm: {arm}")
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be positive")
        chosen_hidden = position_hidden if arm == "position_only" else hidden
        if chosen_hidden // heads * heads != chosen_hidden:
            raise ValueError("hidden size must be divisible by attention heads")
        self.arm = arm
        self.hidden = int(chosen_hidden)
        self.recurrent_steps = int(recurrent_steps)

        if arm == "position_only":
            self.position_input = nn.Linear(
                self.INPUT_ROLES + self.POSITION_BASIS, self.hidden
            )
            self.position_block = PositionBlock(self.hidden, heads, expansion)
            self.position_output = nn.Linear(self.hidden, 1)
            self.quotient_head = nn.Linear(self.hidden, 3)
        else:
            self.role_embedding = nn.Parameter(
                torch.empty(self.INPUT_ROLES, self.hidden)
            )
            self.semantic_anchor = nn.Parameter(
                torch.empty(self.SYMBOLS, self.hidden)
            )
            self.position_projection = nn.Linear(self.POSITION_BASIS, self.hidden)
            self.equivariant_block = EquivariantBlock(self.hidden, heads, expansion)
            self.symbol_output = nn.Linear(self.hidden, 1)
            self.quotient_head = nn.Linear(self.SYMBOLS * self.hidden, 3)
            nn.init.normal_(self.role_embedding, std=0.05)
            nn.init.normal_(self.semantic_anchor, std=0.05)
            if arm == "broken_tying":
                self.core_symbol_scale = nn.Parameter(
                    torch.tensor([[0.85], [1.15]], dtype=torch.float32).expand(
                        self.SYMBOLS, self.hidden
                    ).clone()
                )
                self.core_symbol_bias = nn.Parameter(
                    torch.linspace(-0.08, 0.08, self.SYMBOLS)[:, None].expand(
                        self.SYMBOLS, self.hidden
                    ).clone()
                )

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

    def _symbol_embedding(self, features: Tensor) -> Tensor:
        batch, roles, width = features.shape
        if roles != self.INPUT_ROLES:
            raise ValueError(f"expected {self.INPUT_ROLES} feature roles")
        positive = features.gt(0).to(features.dtype)
        one_hot = torch.stack((1.0 - positive, positive), dim=-1)
        embedded = torch.einsum("brwk,rd->bwkd", one_hot, self.role_embedding)
        positions = self.position_projection(
            self.position_basis(width, device=features.device, dtype=features.dtype)
        )
        return (
            embedded
            + positions[None, :, None, :]
            + self.semantic_anchor[None, None, :, :]
        )

    def core_operator(self, state: Tensor) -> Tensor:
        """Apply the recurrent core once, excluding arithmetic boundaries."""
        if self.arm == "position_only":
            return self.position_block(state)
        if self.arm == "broken_tying":
            state = (
                state * self.core_symbol_scale[None, None, :, :]
                + self.core_symbol_bias[None, None, :, :]
            )
        return self.equivariant_block(state)

    def forward(
        self, features: Tensor, *, return_symbol_logits: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, role, position]")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        width = features.shape[-1]
        if self.arm == "position_only":
            basis = self.position_basis(
                width, device=features.device, dtype=features.dtype
            )
            raw = features.permute(0, 2, 1)
            embedding = self.position_input(
                torch.cat((raw, basis[None].expand(raw.shape[0], -1, -1)), dim=-1)
            )
            state = torch.zeros_like(embedding)
            for _ in range(self.recurrent_steps):
                state = self.core_operator(state + embedding)
            logits = self.position_output(state).squeeze(-1)
            quotient = self.quotient_head(state.mean(dim=1))
            return logits, quotient

        embedding = self._symbol_embedding(features)
        state = torch.zeros_like(embedding)
        for _ in range(self.recurrent_steps):
            state = self.core_operator(state + embedding)
        symbol_logits = self.symbol_output(state).squeeze(-1)
        logits = symbol_logits[..., 1] - symbol_logits[..., 0]
        quotient = self.quotient_head(state.mean(dim=1).flatten(1))
        if return_symbol_logits:
            return logits, quotient, symbol_logits
        return logits, quotient

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
