"""Candidate005 head/body/tail decomposition with AbbIE-C and AbbIE-D recurrence."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CandidateBody(nn.Module):
    """Only Candidate005's learned shared recurrent transformation."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.state_norm = copy.deepcopy(source.state_norm)
        self.gate_weight = nn.Parameter(source.gate_weight.detach().clone())
        self.gate_bias = nn.Parameter(source.gate_bias.detach().clone())
        self.candidate_weight = nn.Parameter(source.candidate_weight.detach().clone())
        self.candidate_bias = nn.Parameter(source.candidate_bias.detach().clone())

    def update(self, state: Tensor, inputs: Tensor, dilation: int) -> Tensor:
        normalized = self.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gates = torch.sigmoid(F.conv1d(combined, self.gate_weight, self.gate_bias, padding=dilation, dilation=dilation))
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        candidate = torch.tanh(F.conv1d(candidate_inputs, self.candidate_weight, self.candidate_bias, padding=dilation, dilation=dilation))
        return (1.0 - update) * state + update * candidate


class AbbieConvGRU(nn.Module):
    """Head -> recurrent body -> tail, with no input-chunk recurrence."""

    def __init__(self, source: nn.Module, *, form: str, default_recurrence: int = 2, untied: bool = False) -> None:
        super().__init__()
        if form not in {"C", "D"} or default_recurrence not in {1, 2, 4, 8}:
            raise ValueError("invalid AbbIE form or recurrence")
        self.form = form; self.default_recurrence = default_recurrence; self.untied = untied
        self.width = int(source.width); self.hidden = int(source.hidden); self.cycles = int(source.cycles)
        self.head = copy.deepcopy(source.stem)
        self.bodies = nn.ModuleList([CandidateBody(source) for _ in range(2 if untied else 1)])
        self.tail_norm = copy.deepcopy(source.output_norm)
        self.tail_local = copy.deepcopy(source.local_head)
        self.tail_quotient = copy.deepcopy(source.quotient_head)
    def dilation_schedule(self, width: int) -> list[int]:
        maximum = max(1, 1 << max(0, math.ceil(math.log2(width)) - 1))
        ascending: list[int] = []
        dilation = 1
        while dilation <= maximum:
            ascending.append(dilation)
            dilation *= 2
        schedule: list[int] = []
        for cycle in range(self.cycles):
            schedule.extend(ascending if cycle % 2 == 0 else reversed(ascending))
        return schedule

    def body_parameters(self):
        yield from self.bodies.parameters()

    def forward(
        self,
        features: Tensor,
        *,
        recurrence: int | None = None,
        reset_state: bool = False,
        shuffle_state: bool = False,
        ablate_head: bool = False,
        ablate_body: bool = False,
        ablate_tail: bool = False,
        return_trace: bool = False,
    ):
        if features.ndim != 3 or features.shape[1] != 6 or not torch.isfinite(features).all():
            raise ValueError("features must be finite [batch, 6, width]")
        repeats = self.default_recurrence if recurrence is None else int(recurrence)
        if repeats not in {1, 2, 4, 8}:
            raise ValueError("recurrence must be one of 1, 2, 4, 8")
        initial = self.head(features)
        if ablate_head: initial = torch.zeros_like(initial)
        state = initial; states = []
        if not ablate_body:
            for iteration in range(repeats):
                if reset_state and iteration: state = initial
                if shuffle_state and iteration: state = state.roll(1, dims=0)
                prior = state; body = self.bodies[iteration % len(self.bodies)]
                for dilation in self.dilation_schedule(features.shape[-1]):
                    state = body.update(state, features, int(dilation))
                if self.form == "D": state = state + prior
                states.append(state)
        normalized = F.silu(self.tail_norm(state))
        logits = self.tail_local(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        quotient = self.tail_quotient(pooled)
        if ablate_tail:
            logits = torch.zeros_like(logits); quotient = torch.zeros_like(quotient)
        trace = {"form": self.form, "recurrence": repeats, "body_calls": repeats * len(self.dilation_schedule(features.shape[-1])), "state_norms": [float(item.detach().norm()) for item in states], "final_state": state.detach()}
        return (logits, quotient, trace) if return_trace else (logits, quotient)

    def accounting(self, width: int, recurrence: int) -> dict[str, int]:
        return {"stored_parameters": sum(p.numel() for p in self.parameters()), "body_parameters": sum(p.numel() for p in self.body_parameters()), "body_calls": recurrence * len(self.dilation_schedule(width)), "stored_bodies": len(self.bodies)}


class RecurrenceView(nn.Module):
    def __init__(self, model: AbbieConvGRU, recurrence: int, **intervention: Any) -> None:
        super().__init__(); self.model = model; self.recurrence = recurrence; self.intervention = intervention
        self.width = model.width; self.hidden = model.hidden; self.cycles = model.cycles

    def forward(self, features: Tensor):
        return self.model(features, recurrence=self.recurrence, **self.intervention)


def source_identity_receipt(source: nn.Module, abbie: AbbieConvGRU, features: Tensor) -> dict[str, Any]:
    source.eval(); abbie.eval()
    with torch.no_grad(): expected = source(features); observed = abbie(features, recurrence=1)
    return {"bit_logits_equal": torch.equal(expected[0], observed[0]), "quotient_logits_equal": torch.equal(expected[1], observed[1]), "maximum_absolute_delta": max(float((expected[0]-observed[0]).abs().max()), float((expected[1]-observed[1]).abs().max()))}


def randomize_body(model: AbbieConvGRU, seed: int) -> None:
    torch.manual_seed(seed)
    for parameter in model.body_parameters():
        if parameter.ndim >= 2: nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        else: nn.init.zeros_(parameter)


__all__ = ["AbbieConvGRU", "CandidateBody", "RecurrenceView", "randomize_body", "source_identity_receipt"]
