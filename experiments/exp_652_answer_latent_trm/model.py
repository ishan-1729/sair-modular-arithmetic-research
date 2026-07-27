"""Bounded-faithful Tiny Recursion Model and matched controls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLULayer(nn.Module):
    """One width-shared local recurrent layer."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden)
        self.gate_value = nn.Conv1d(hidden, 2 * hidden, 3, padding=1)
        self.output = nn.Conv1d(hidden, hidden, 1)

    def forward(self, value: Tensor) -> Tensor:
        gate, update = self.gate_value(self.norm(value)).chunk(2, dim=1)
        return value + self.output(F.silu(gate) * update)


class TinySharedNetwork(nn.Module):
    """The single two-layer network shared by z and y updates."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SwiGLULayer(hidden), SwiGLULayer(hidden)])

    def forward(self, value: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


@dataclass
class RecursionOutput:
    y: Tensor
    z: Tensor
    logits: Tensor
    q_logit: Tensor
    answer_curve: list[Tensor]
    q_curve: list[Tensor]


class AnswerLatentTRM(nn.Module):
    """Persistent answer y plus latent z with bounded-faithful TRM ordering."""

    architecture = "answer_latent_trm"

    def __init__(
        self,
        hidden: int = 24,
        latent_updates: int = 6,
        outer_recursions: int = 3,
    ) -> None:
        super().__init__()
        if latent_updates < 1 or outer_recursions < 1:
            raise ValueError("recursion counts must be positive")
        self.hidden = int(hidden)
        self.latent_updates = int(latent_updates)
        self.outer_recursions = int(outer_recursions)
        self.input_projection = nn.Conv1d(5, hidden, 1)
        self.bit_feedback = nn.Conv1d(1, hidden, 1)
        self.shared_network = TinySharedNetwork(hidden)
        self.answer_initial = nn.Parameter(torch.zeros(1, hidden, 1))
        self.latent_initial = nn.Parameter(torch.zeros(1, hidden, 1))
        self.answer_head = nn.Conv1d(hidden, 1, 1)
        self.q_head = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        nn.init.normal_(self.answer_initial, std=0.02)
        nn.init.normal_(self.latent_initial, std=0.02)

    def initial_states(
        self,
        embedded_input: Tensor,
        intervention: str = "none",
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, _, width = embedded_input.shape
        y = self.answer_initial.expand(batch, -1, width).clone()
        z = self.latent_initial.expand(batch, -1, width).clone()
        if intervention == "swap_y_z":
            y, z = z, y
        elif intervention == "randomize_y":
            y = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)
        elif intervention == "randomize_z":
            z = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        elif intervention not in {"none", "reset_y", "reset_z"}:
            raise ValueError(f"unknown state intervention: {intervention}")
        return y, z

    def full_recursion(
        self,
        embedded_input: Tensor,
        y: Tensor,
        z: Tensor,
        latent_updates: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        updates = self.latent_updates if latent_updates is None else int(latent_updates)
        for _ in range(updates):
            z = self.shared_network(embedded_input + y + z)
        y = self.shared_network(y + z)
        return y, z

    def _q(self, y: Tensor) -> Tensor:
        return self.q_head(y.mean(dim=-1)).squeeze(-1)

    def deep_recursion(
        self,
        features: Tensor,
        supervision_steps: int = 1,
        outer_recursions: int | None = None,
        latent_updates: int | None = None,
        state_intervention: str = "none",
        learned_halt: bool = False,
        halt_threshold: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> RecursionOutput:
        if features.ndim != 3 or features.shape[1] != 5:
            raise ValueError("features must have shape [batch, 5, width]")
        steps = int(supervision_steps)
        if steps < 1:
            raise ValueError("supervision_steps must be positive")
        outer = self.outer_recursions if outer_recursions is None else int(outer_recursions)
        if outer < 1:
            raise ValueError("outer_recursions must be positive")
        x = self.input_projection(features)
        y, z = self.initial_states(x, state_intervention, generator)
        answer_curve: list[Tensor] = []
        q_curve: list[Tensor] = []
        halted = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        for _ in range(steps):
            if state_intervention == "reset_y":
                y = torch.zeros_like(y)
            elif state_intervention == "reset_z":
                z = torch.zeros_like(z)
            for _ in range(outer - 1):
                with torch.no_grad():
                    y, z = self.full_recursion(x, y, z, latent_updates)
                y, z = y.detach(), z.detach()
            y, z = self.full_recursion(x, y, z, latent_updates)
            logits = self.answer_head(y).squeeze(1)
            q_logit = self._q(y)
            answer_curve.append(logits)
            q_curve.append(q_logit)
            if learned_halt:
                newly_halted = q_logit.sigmoid().ge(halt_threshold)
                halted = halted | newly_halted
                if bool(halted.all()):
                    break
            y, z = y.detach(), z.detach()
        return RecursionOutput(y, z, answer_curve[-1], q_curve[-1], answer_curve, q_curve)

    def forward(self, features: Tensor) -> Tensor:
        return self.deep_recursion(features).logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class NoLatentControl(AnswerLatentTRM):
    """Same parameters/work calls, but no persistent independent latent state."""

    architecture = "no_latent_control"

    def full_recursion(
        self,
        embedded_input: Tensor,
        y: Tensor,
        z: Tensor,
        latent_updates: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        updates = self.latent_updates if latent_updates is None else int(latent_updates)
        for _ in range(updates + 1):
            y = self.shared_network(embedded_input + y)
        return y, torch.zeros_like(z)


class DecodedCellularControl(AnswerLatentTRM):
    """Repeatedly decode and feed back the current answer, without latent z."""

    architecture = "decoded_cellular_control"

    def full_recursion(
        self,
        embedded_input: Tensor,
        y: Tensor,
        z: Tensor,
        latent_updates: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        updates = self.latent_updates if latent_updates is None else int(latent_updates)
        for _ in range(updates + 1):
            decoded = self.answer_head(y).ge(0).to(y.dtype)
            y = self.shared_network(embedded_input + y + self.bit_feedback(decoded))
        return y, torch.zeros_like(z)


class ExponentialMovingAverage:
    """Serializable EMA shadow with exact update accounting."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)
        self.updates += 1

    def model_copy(self, model: nn.Module) -> nn.Module:
        copied = deepcopy(model)
        copied.load_state_dict(self.shadow, strict=True)
        return copied
