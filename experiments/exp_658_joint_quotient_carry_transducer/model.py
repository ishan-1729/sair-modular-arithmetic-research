"""Split-convolution learned quotient/carry transducer for EXP-658.

The inference graph consumes floating feature tensors only.  Quotient and
signed-carry messages are posterior-weighted learned embeddings; they are never
decoded, selected, or applied through an arithmetic routine.  The incumbent
98-channel recurrent convolutions remain separate from the four learned message
channels so loading EXP-609 is exactly function preserving.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn


class TransducerOutput(NamedTuple):
    """Learned outputs available to future residue and auxiliary losses."""

    residue_logits: Tensor
    quotient_logits: Tensor
    signed_carry_logits: Tensor


class JointQuotientCarryTransducer(nn.Module):
    """EXP-609 cellular transition plus soft learned message deltas."""

    VALID_MESSAGE_ABLATIONS = frozenset(
        ("none", "zero-quotient-message", "zero-signed-carry-message")
    )

    def __init__(self, *, hidden: int = 92, cycles: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.cycles = cycles
        recurrent_channels = hidden + 6

        # This is the unchanged EXP-609/SharedDilatedConvGRU parameter graph.
        self.stem = nn.Conv1d(6, hidden, 1)
        self.state_norm = nn.GroupNorm(1, hidden)
        self.gate_weight = nn.Parameter(
            torch.empty(2 * hidden, recurrent_channels, 3)
        )
        self.gate_bias = nn.Parameter(torch.zeros(2 * hidden))
        self.candidate_weight = nn.Parameter(
            torch.empty(hidden, recurrent_channels, 3)
        )
        self.candidate_bias = nn.Parameter(torch.zeros(hidden))
        self.output_norm = nn.GroupNorm(1, hidden)
        self.local_head = nn.Conv1d(hidden, 1, 1)
        self.quotient_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3)
        )
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_weight, a=math.sqrt(5))

        # Four learned message channels stay outside both incumbent recurrent
        # convolutions.  Exact-zero deltas make source initialization exact.
        self.signed_carry_head = nn.Conv1d(hidden, 4, 1)
        self.quotient_message_embedding = nn.Parameter(torch.empty(3, 2))
        self.signed_carry_message_embedding = nn.Parameter(torch.empty(4, 2))
        self.message_gate_weight = nn.Parameter(torch.zeros(2 * hidden, 4, 3))
        self.message_candidate_weight = nn.Parameter(torch.zeros(hidden, 4, 3))
        nn.init.xavier_uniform_(self.quotient_message_embedding)
        nn.init.xavier_uniform_(self.signed_carry_message_embedding)

    def dilation_schedule(self, width: int) -> list[int]:
        """Return the unchanged width-dependent EXP-609 dilation schedule."""

        maximum = max(1, 2 ** max(0, math.ceil(math.log2(width)) - 1))
        dilations = []
        value = 1
        while value <= maximum:
            dilations.append(value)
            value *= 2
        schedule = []
        ascending = True
        for _ in range(self.cycles):
            schedule.extend(dilations if ascending else reversed(dilations))
            ascending = not ascending
        return schedule

    def _message_posteriors(self, state: Tensor) -> tuple[Tensor, Tensor]:
        message_hidden = F.silu(self.output_norm(state))
        pooled = torch.cat(
            (message_hidden.mean(dim=-1), message_hidden.amax(dim=-1)), dim=1
        )
        quotient_posterior = F.softmax(self.quotient_head(pooled), dim=1)
        signed_carry_posterior = F.softmax(
            self.signed_carry_head(message_hidden), dim=1
        )
        return quotient_posterior, signed_carry_posterior

    def _soft_messages(
        self, state: Tensor, message_ablation: str
    ) -> tuple[Tensor, Tensor]:
        if message_ablation not in self.VALID_MESSAGE_ABLATIONS:
            raise ValueError(f"unsupported message ablation: {message_ablation}")
        quotient_posterior, signed_carry_posterior = self._message_posteriors(state)
        quotient_message = quotient_posterior @ self.quotient_message_embedding
        quotient_message = quotient_message.unsqueeze(-1).expand(
            -1, -1, state.shape[-1]
        )
        signed_carry_message = torch.einsum(
            "bcw,cd->bdw",
            signed_carry_posterior,
            self.signed_carry_message_embedding,
        )
        if message_ablation == "zero-quotient-message":
            quotient_message = torch.zeros_like(quotient_message)
        elif message_ablation == "zero-signed-carry-message":
            signed_carry_message = torch.zeros_like(signed_carry_message)
        return quotient_message, signed_carry_message

    def recurrent_update(
        self,
        state: Tensor,
        inputs: Tensor,
        dilation: int,
        *,
        message_ablation: str = "none",
    ) -> Tensor:
        """Apply the source recurrence plus separate learned message deltas."""

        normalized = self.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        source_gate_preactivation = F.conv1d(
            combined,
            self.gate_weight,
            self.gate_bias,
            padding=dilation,
            dilation=dilation,
        )
        quotient_message, signed_carry_message = self._soft_messages(
            state, message_ablation
        )
        messages = torch.cat((quotient_message, signed_carry_message), dim=1)
        message_gate_delta = F.conv1d(
            messages,
            self.message_gate_weight,
            None,
            padding=dilation,
            dilation=dilation,
        )
        gates = torch.sigmoid(source_gate_preactivation + message_gate_delta)
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        source_candidate_preactivation = F.conv1d(
            candidate_inputs,
            self.candidate_weight,
            self.candidate_bias,
            padding=dilation,
            dilation=dilation,
        )
        message_candidate_delta = F.conv1d(
            messages,
            self.message_candidate_weight,
            None,
            padding=dilation,
            dilation=dilation,
        )
        candidate = torch.tanh(
            source_candidate_preactivation + message_candidate_delta
        )
        return (1.0 - update) * state + update * candidate

    def forward(
        self, features: Tensor, *, message_ablation: str = "none"
    ) -> TransducerOutput:
        hidden = self.stem(features)
        for dilation in self.dilation_schedule(features.shape[-1]):
            hidden = self.recurrent_update(
                hidden,
                features,
                dilation,
                message_ablation=message_ablation,
            )
        normalized = F.silu(self.output_norm(hidden))
        residue_logits = self.local_head(normalized).squeeze(1)
        pooled = torch.cat(
            (normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1
        )
        quotient_logits = self.quotient_head(pooled)
        signed_carry_logits = self.signed_carry_head(normalized)
        return TransducerOutput(
            residue_logits=residue_logits,
            quotient_logits=quotient_logits,
            signed_carry_logits=signed_carry_logits,
        )

    def load_exp609_source(self, checkpoint: Path) -> None:
        """Load exactly the frozen source tensors, leaving extensions intact."""

        source = load_file(str(checkpoint))
        result = self.load_state_dict(source, strict=False)
        expected_missing = {
            "signed_carry_head.weight",
            "signed_carry_head.bias",
            "quotient_message_embedding",
            "signed_carry_message_embedding",
            "message_gate_weight",
            "message_candidate_weight",
        }
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError("EXP-609 checkpoint geometry does not match")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
