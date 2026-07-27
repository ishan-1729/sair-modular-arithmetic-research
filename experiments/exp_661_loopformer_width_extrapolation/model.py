"""Small bounded-faithful LoopFormer and parameter-matched controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoidal_scalar(value: Tensor, dimensions: int) -> Tensor:
    """Encode a normalized scalar without using an example or task identity."""
    if dimensions % 2:
        raise ValueError("conditioning dimensions must be even")
    frequencies = torch.exp(
        torch.arange(0, dimensions, 2, device=value.device, dtype=value.dtype)
        * (-math.log(10_000.0) / dimensions)
    )
    angles = value.unsqueeze(-1) * frequencies.unsqueeze(0) * 10_000.0
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ConditionedLoopBlock(nn.Module):
    """Pre-norm Transformer block with LoopFormer scale and shortcut gates."""

    def __init__(self, hidden: int, heads: int, ff_hidden: int, condition: int) -> None:
        super().__init__()
        self.norm_attention = nn.RMSNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm_ffn = nn.RMSNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff_hidden), nn.SiLU(), nn.Linear(ff_hidden, hidden)
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(condition, 4 * hidden)
        )

    def forward(
        self,
        hidden: Tensor,
        condition: Tensor,
        *,
        use_time_step: bool,
        use_shortcut: bool,
    ) -> Tensor:
        gamma_attention, alpha_attention, gamma_ffn, alpha_ffn = self.modulation(
            condition
        ).chunk(4, dim=-1)
        if not use_time_step:
            gamma_attention = torch.zeros_like(gamma_attention)
            gamma_ffn = torch.zeros_like(gamma_ffn)
        if not use_shortcut:
            alpha_attention = torch.zeros_like(alpha_attention)
            alpha_ffn = torch.zeros_like(alpha_ffn)
        attention_input = self.norm_attention(hidden) * (1.0 + gamma_attention[:, None])
        update, _ = self.attention(
            attention_input, attention_input, attention_input, need_weights=False
        )
        gate_attention = torch.sigmoid(alpha_attention[:, None]) if use_shortcut else 1.0
        hidden = hidden + gate_attention * update
        ffn_input = self.norm_ffn(hidden) * (1.0 + gamma_ffn[:, None])
        gate_ffn = torch.sigmoid(alpha_ffn[:, None]) if use_shortcut else 1.0
        return hidden + gate_ffn * self.ffn(ffn_input)


@dataclass
class LoopOutput:
    hidden: Tensor
    logits: Tensor
    hidden_curve: list[Tensor]
    logit_curve: list[Tensor]
    call_count: int


class LoopFormer(nn.Module):
    """One shared cyclic Transformer stack with fixed user loop budgets."""

    architecture = "loopformer"

    def __init__(
        self,
        hidden: int = 24,
        heads: int = 4,
        ff_hidden: int = 48,
        condition: int = 16,
        max_width: int = 32,
        default_budget: int = 8,
        *,
        use_time_step: bool = True,
        use_shortcut: bool = True,
        decoded_feedback: bool = False,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if default_budget < 1 or max_width < 1:
            raise ValueError("budget and width must be positive")
        self.hidden = int(hidden)
        self.condition = int(condition)
        self.max_width = int(max_width)
        self.default_budget = int(default_budget)
        self.use_time_step = bool(use_time_step)
        self.use_shortcut = bool(use_shortcut)
        self.decoded_feedback = bool(decoded_feedback)
        self.input_projection = nn.Linear(5, hidden)
        self.position = nn.Parameter(torch.empty(1, max_width, hidden))
        self.time_projection = nn.Sequential(
            nn.Linear(condition, condition), nn.SiLU(), nn.Linear(condition, condition)
        )
        self.step_projection = nn.Sequential(
            nn.Linear(condition, condition), nn.SiLU(), nn.Linear(condition, condition)
        )
        self.block = ConditionedLoopBlock(hidden, heads, ff_hidden, condition)
        self.output_norm = nn.RMSNorm(hidden)
        self.output_head = nn.Linear(hidden, 1)
        # Retained in every arm so decoded feedback changes semantics, not parameter count.
        self.bit_feedback = nn.Linear(1, hidden)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position, std=0.02)
        for module in self.modules():
            if module is self:
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

    def trajectory_condition(self, time: float, step: float, batch: int, device: torch.device) -> Tensor:
        dtype = self.position.dtype
        time_value = torch.full((batch,), float(time), dtype=dtype, device=device)
        step_value = torch.full((batch,), float(step), dtype=dtype, device=device)
        return self.time_projection(sinusoidal_scalar(time_value, self.condition)) + self.step_projection(
            sinusoidal_scalar(step_value, self.condition)
        )

    @staticmethod
    def uniform_schedule(budget: int) -> list[float]:
        if budget < 1:
            raise ValueError("budget must be positive")
        return [1.0 / budget] * budget

    def loop(
        self,
        features: Tensor,
        *,
        schedule: list[float] | None = None,
        use_time_step: bool | None = None,
        use_shortcut: bool | None = None,
    ) -> LoopOutput:
        if features.ndim != 3 or features.shape[1] != 5:
            raise ValueError("features must have shape [batch, 5, width]")
        width = features.shape[-1]
        if width > self.max_width:
            raise ValueError("width exceeds generic positional capacity")
        schedule = self.uniform_schedule(self.default_budget) if schedule is None else schedule
        if not schedule or any(step <= 0.0 for step in schedule):
            raise ValueError("trajectory steps must be positive")
        if not math.isclose(sum(schedule), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("trajectory steps must sum to one")
        time_step = self.use_time_step if use_time_step is None else bool(use_time_step)
        shortcut = self.use_shortcut if use_shortcut is None else bool(use_shortcut)
        hidden = self.input_projection(features.transpose(1, 2)) + self.position[:, :width]
        hidden_curve: list[Tensor] = []
        logit_curve: list[Tensor] = []
        elapsed = 0.0
        for step in schedule:
            condition = self.trajectory_condition(elapsed, float(step), features.shape[0], features.device)
            hidden = self.block(
                hidden,
                condition,
                use_time_step=time_step,
                use_shortcut=shortcut,
            )
            logits = self.output_head(self.output_norm(hidden)).squeeze(-1)
            hidden_curve.append(hidden)
            logit_curve.append(logits)
            if self.decoded_feedback:
                decoded = logits.ge(0).to(hidden.dtype).unsqueeze(-1)
                hidden = hidden + self.bit_feedback(decoded)
            elapsed += float(step)
        return LoopOutput(hidden, logit_curve[-1], hidden_curve, logit_curve, len(schedule))

    def forward(self, features: Tensor, budget: int | None = None) -> Tensor:
        chosen = self.default_budget if budget is None else int(budget)
        return self.loop(features, schedule=self.uniform_schedule(chosen)).logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class PlainLoopedTransformer(LoopFormer):
    architecture = "plain_looped_transformer"

    def __init__(self, *args, **kwargs) -> None:
        kwargs.update(use_time_step=False, use_shortcut=False, decoded_feedback=False)
        super().__init__(*args, **kwargs)


class DecodedCellularControl(LoopFormer):
    architecture = "decoded_cellular_control"

    def __init__(self, *args, **kwargs) -> None:
        kwargs.update(use_time_step=False, use_shortcut=False, decoded_feedback=True)
        super().__init__(*args, **kwargs)
