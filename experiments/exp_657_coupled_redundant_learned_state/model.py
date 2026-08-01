"""Learned coupled redundant state for EXP-657.

The module consumes only tensor feature planes.  It deliberately contains no
integer decoding, label construction, or arithmetic solver.  Both experimental
arms instantiate this exact graph; only the learned-state feedback rule differs
between ``single`` and ``redundant`` persistence modes.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StepOutput(NamedTuple):
    """Outputs of one learned transition and its persistent belief planes."""

    fused_logits: Tensor
    channel_a_logits: Tensor
    channel_b_logits: Tensor
    syndrome_error_logits: Tensor
    next_state_a: Tensor
    next_state_b: Tensor
    channel_a_latent: Tensor
    channel_b_latent: Tensor
    syndrome_latent: Tensor
    fusion_gate: Tensor


class LearnedBeliefBranch(nn.Module):
    """One independently parameterized width-shared belief transition."""

    def __init__(self, hidden: int, cycles: int, context_channels: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.cycles = int(cycles)
        self.context_channels = int(context_channels)
        input_channels = 1 + self.context_channels
        recurrent_channels = self.hidden + input_channels
        self.stem = nn.Conv1d(input_channels, self.hidden, 1)
        self.state_norm = nn.GroupNorm(1, self.hidden)
        self.gate_weight = nn.Parameter(
            torch.empty(2 * self.hidden, recurrent_channels, 3)
        )
        self.gate_bias = nn.Parameter(torch.zeros(2 * self.hidden))
        self.candidate_weight = nn.Parameter(
            torch.empty(self.hidden, recurrent_channels, 3)
        )
        self.candidate_bias = nn.Parameter(torch.zeros(self.hidden))
        self.output_norm = nn.GroupNorm(1, self.hidden)
        self.belief_head = nn.Conv1d(self.hidden, 1, 1)
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.candidate_weight, a=math.sqrt(5.0))

    def dilation_schedule(self, width: int) -> list[int]:
        maximum = max(1, int(2 ** max(0, math.ceil(math.log2(width)) - 1)))
        ascending: list[int] = []
        value = 1
        while value <= maximum:
            ascending.append(value)
            value *= 2
        schedule: list[int] = []
        forward = True
        for _ in range(self.cycles):
            schedule.extend(ascending if forward else reversed(ascending))
            forward = not forward
        return schedule

    def recurrent_update(self, hidden: Tensor, inputs: Tensor, dilation: int) -> Tensor:
        normalized = self.state_norm(hidden)
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
        return (1.0 - update) * hidden + update * candidate

    def forward(self, state_belief: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat((state_belief, context), dim=1)
        hidden = self.stem(inputs)
        for dilation in self.dilation_schedule(inputs.shape[-1]):
            hidden = self.recurrent_update(hidden, inputs, dilation)
        latent = F.silu(self.output_norm(hidden))
        return self.belief_head(latent), latent


class CoupledRedundantLearnedState(nn.Module):
    """Two learned belief planes with learned syndrome, correction, and fusion."""

    VALID_PERSISTENCE = frozenset(("single", "redundant"))
    VALID_ABLATIONS = frozenset(("none", "drop_a", "drop_b"))

    def __init__(
        self,
        *,
        hidden: int = 24,
        syndrome_hidden: int = 16,
        cycles: int = 2,
        context_channels: int = 5,
        persistence: str = "redundant",
        correction_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if persistence not in self.VALID_PERSISTENCE:
            raise ValueError(f"unsupported persistence mode: {persistence}")
        self.hidden = int(hidden)
        self.syndrome_hidden = int(syndrome_hidden)
        self.cycles = int(cycles)
        self.context_channels = int(context_channels)
        self.persistence = persistence
        self.correction_scale = float(correction_scale)

        # These objects never share parameters.  A static and runtime identity
        # check in L0 guards against accidentally turning redundancy into copying.
        self.channel_a = LearnedBeliefBranch(hidden, cycles, context_channels)
        self.channel_b = LearnedBeliefBranch(hidden, cycles, context_channels)

        joint_channels = 2 * self.hidden + 2
        self.syndrome_encoder = nn.Sequential(
            nn.Conv1d(joint_channels, self.syndrome_hidden, 1),
            nn.SiLU(),
            nn.Conv1d(self.syndrome_hidden, self.syndrome_hidden, 3, padding=1),
            nn.Tanh(),
        )
        self.syndrome_error_head = nn.Conv1d(self.syndrome_hidden, 2, 1)
        correction_inputs = self.syndrome_hidden + 2
        self.learned_corrections = nn.Sequential(
            nn.Conv1d(correction_inputs, self.syndrome_hidden, 1),
            nn.SiLU(),
            nn.Conv1d(self.syndrome_hidden, 2, 1),
        )
        fusion_inputs = joint_channels + self.syndrome_hidden
        self.learned_fusion_gate = nn.Sequential(
            nn.Conv1d(fusion_inputs, self.syndrome_hidden, 1),
            nn.SiLU(),
            nn.Conv1d(self.syndrome_hidden, 1, 1),
        )
        self.learned_decoder = nn.Sequential(
            nn.Conv1d(fusion_inputs, self.syndrome_hidden, 1),
            nn.SiLU(),
            nn.Conv1d(self.syndrome_hidden, 1, 1),
        )

    @staticmethod
    def initial_state(
        batch: int,
        width: int,
        *,
        device: Union[torch.device, str] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        state = torch.zeros((batch, 1, width), device=device, dtype=dtype)
        return state, state.clone()

    def forward_step(
        self,
        state_a: Tensor,
        state_b: Tensor,
        context: Tensor,
        *,
        ablation: str = "none",
    ) -> StepOutput:
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"unsupported ablation: {ablation}")
        if state_a.shape != state_b.shape:
            raise ValueError("persistent belief planes must have identical shapes")
        if state_a.ndim != 3 or state_a.shape[1] != 1:
            raise ValueError("persistent belief planes must be [batch, 1, width]")
        if context.ndim != 3 or context.shape[1] != self.context_channels:
            raise ValueError("context has the wrong channel count")
        if context.shape[0] != state_a.shape[0] or context.shape[2] != state_a.shape[2]:
            raise ValueError("context and state shapes do not align")

        raw_a, latent_a = self.channel_a(state_a, context)
        raw_b, latent_b = self.channel_b(state_b, context)
        if ablation == "drop_a":
            raw_a = torch.zeros_like(raw_a)
            latent_a = torch.zeros_like(latent_a)
        elif ablation == "drop_b":
            raw_b = torch.zeros_like(raw_b)
            latent_b = torch.zeros_like(latent_b)

        joint = torch.cat((latent_a, latent_b, raw_a, raw_b), dim=1)
        syndrome = self.syndrome_encoder(joint)
        syndrome_error_logits = self.syndrome_error_head(syndrome)
        correction_inputs = torch.cat((syndrome, raw_a, raw_b), dim=1)
        corrections = self.learned_corrections(correction_inputs)
        corrected_a = raw_a + self.correction_scale * corrections[:, 0:1]
        corrected_b = raw_b + self.correction_scale * corrections[:, 1:2]

        fusion_inputs = torch.cat((joint, syndrome), dim=1)
        gate = torch.sigmoid(self.learned_fusion_gate(fusion_inputs))
        fused = (
            gate * corrected_a
            + (1.0 - gate) * corrected_b
            + self.learned_decoder(fusion_inputs)
        )
        next_a = torch.tanh(corrected_a)
        next_b = torch.tanh(corrected_b)
        if self.persistence == "single":
            shared = torch.tanh(fused)
            next_a = shared
            next_b = shared.clone()
        return StepOutput(
            fused_logits=fused.squeeze(1),
            channel_a_logits=corrected_a.squeeze(1),
            channel_b_logits=corrected_b.squeeze(1),
            syndrome_error_logits=syndrome_error_logits,
            next_state_a=next_a,
            next_state_b=next_b,
            channel_a_latent=latent_a,
            channel_b_latent=latent_b,
            syndrome_latent=syndrome,
            fusion_gate=gate,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_signature(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((name, tuple(parameter.shape)) for name, parameter in self.named_parameters())

    def branch_parameters_are_distinct(self) -> bool:
        left = tuple(self.channel_a.parameters())
        right = tuple(self.channel_b.parameters())
        return len(left) == len(right) and all(
            a.data_ptr() != b.data_ptr() for a, b in zip(left, right)
        )
