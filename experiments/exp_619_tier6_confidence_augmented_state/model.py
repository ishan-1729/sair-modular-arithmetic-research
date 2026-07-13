"""Function-preserving recurrent cell with persistent confidence features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConfidenceAugmentedCell(nn.Module):
    """Inject two confidence planes through zero-initialized recurrent kernels."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.width = int(base.width)
        hidden = int(base.hidden)
        self.confidence_stem = nn.Conv1d(2, hidden, 1, bias=False)
        self.confidence_gate_weight = nn.Parameter(torch.zeros(2 * hidden, 2, 3))
        self.confidence_candidate_weight = nn.Parameter(torch.zeros(hidden, 2, 3))
        nn.init.zeros_(self.confidence_stem.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def confidence_from_logits(logits: Tensor) -> Tensor:
        return torch.stack((torch.tanh(logits), torch.tanh(logits.abs())), dim=1)

    def recurrent_update(
        self,
        state: Tensor,
        inputs: Tensor,
        confidence: Tensor,
        dilation: int,
    ) -> Tensor:
        normalized = self.base.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gate_pre = F.conv1d(
            combined,
            self.base.gate_weight,
            self.base.gate_bias,
            padding=dilation,
            dilation=dilation,
        ) + F.conv1d(
            confidence,
            self.confidence_gate_weight,
            padding=dilation,
            dilation=dilation,
        )
        reset, update = torch.sigmoid(gate_pre).chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        candidate_pre = F.conv1d(
            candidate_inputs,
            self.base.candidate_weight,
            self.base.candidate_bias,
            padding=dilation,
            dilation=dilation,
        ) + F.conv1d(
            confidence,
            self.confidence_candidate_weight,
            padding=dilation,
            dilation=dilation,
        )
        candidate = torch.tanh(candidate_pre)
        return (1.0 - update) * state + update * candidate

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.shape[1] != 8:
            raise ValueError("confidence-augmented features require eight channels")
        self.base.width = self.width
        inputs = features[:, :6]
        confidence = features[:, 6:]
        hidden = self.base.stem(inputs) + self.confidence_stem(confidence)
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            hidden = self.recurrent_update(hidden, inputs, confidence, dilation)
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
