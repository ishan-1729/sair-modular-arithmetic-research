"""Two-phase teacher-distilled macro-transition model.

Inference is tensor-native.  The first learned phase emits a continuous
intermediate bit plane; the second learned phase emits the only state that is
hard-decoded between radix-4 macro steps.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class OneCycleCell(nn.Module):
    """One directional dilation sweep with the radix-2 cell parameterization."""

    input_channels = 6

    def __init__(self, hidden: int = 92, reverse_dilations: bool = False) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.reverse_dilations = bool(reverse_dilations)
        channels = self.hidden + self.input_channels
        self.stem = nn.Conv1d(self.input_channels, self.hidden, 1)
        self.state_norm = nn.GroupNorm(1, self.hidden)
        self.gate_weight = nn.Parameter(torch.empty(2 * self.hidden, channels, 3))
        self.gate_bias = nn.Parameter(torch.zeros(2 * self.hidden))
        self.candidate_weight = nn.Parameter(torch.empty(self.hidden, channels, 3))
        self.candidate_bias = nn.Parameter(torch.zeros(self.hidden))
        self.output_norm = nn.GroupNorm(1, self.hidden)
        self.local_head = nn.Conv1d(self.hidden, 1, 1)
        self.quotient_head = nn.Sequential(
            nn.Linear(2 * self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 3),
        )
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_weight, a=math.sqrt(5))

    def dilation_schedule(self, width: int) -> list[int]:
        maximum = max(1, 1 << max(0, math.ceil(math.log2(width)) - 1))
        values: list[int] = []
        dilation = 1
        while dilation <= maximum:
            values.append(dilation)
            dilation *= 2
        return list(reversed(values)) if self.reverse_dilations else values

    def recurrent_update(
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
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError("expected [batch, 6, width] features")
        hidden = self.stem(features)
        for dilation in self.dilation_schedule(features.shape[-1]):
            hidden = self.recurrent_update(hidden, features, dilation)
        normalized = F.silu(self.output_norm(hidden))
        logits = self.local_head(normalized).squeeze(1)
        pooled = torch.cat(
            (normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1
        )
        return logits, self.quotient_head(pooled)


class TeacherDistilledMacroTransition(nn.Module):
    """Two learned one-cycle phases with a differentiable middle state."""

    def __init__(
        self,
        hidden: int = 92,
        phase1_reverse_dilations: bool = False,
        phase2_reverse_dilations: bool = True,
        intermediate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if intermediate_temperature <= 0:
            raise ValueError("intermediate temperature must be positive")
        self.hidden = int(hidden)
        self.intermediate_temperature = float(intermediate_temperature)
        self.phase1 = OneCycleCell(hidden, phase1_reverse_dilations)
        self.phase2 = OneCycleCell(hidden, phase2_reverse_dilations)

    @staticmethod
    def replace_state_plane(features: Tensor, state_plane: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != 6:
            raise ValueError("expected [batch, 6, width] phase-2 context")
        if state_plane.shape != features[:, 0].shape:
            raise ValueError("state plane shape mismatch")
        return torch.cat(
            (state_plane[:, None, :], features[:, 1:, :]), dim=1
        )

    def continuous_intermediate(self, logits: Tensor) -> Tensor:
        # 2*sigmoid(x/T)-1 == tanh(x/(2T)); legal bit planes use [-1, 1].
        return torch.tanh(logits / (2.0 * self.intermediate_temperature))

    def forward_phase2(self, phase2_features: Tensor) -> tuple[Tensor, Tensor]:
        return self.phase2(phase2_features)

    def forward(
        self, phase1_features: Tensor, phase2_context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        intermediate_logits, phase1_quotient_logits = self.phase1(phase1_features)
        continuous = self.continuous_intermediate(intermediate_logits)
        final_logits, phase2_quotient_logits = self.phase2(
            self.replace_state_plane(phase2_context, continuous)
        )
        return (
            final_logits,
            intermediate_logits,
            phase1_quotient_logits,
            phase2_quotient_logits,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def transfer_teacher_state(
    student: TeacherDistilledMacroTransition,
    teacher_state: dict[str, Tensor],
) -> dict[str, object]:
    """Initialize each one-cycle phase from the frozen radix-2 teacher."""
    student.phase1.load_state_dict(teacher_state, strict=True)
    student.phase2.load_state_dict(teacher_state, strict=True)
    return {
        "phase1": "exact radix-2 teacher state",
        "phase2": "exact radix-2 teacher state",
        "phase1_dilations": "increasing",
        "phase2_dilations": "decreasing",
        "parameters_untied_after_initialization": True,
    }
