"""Bounded-faithful MoR depth routing around Candidate005."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _pooled(state: Tensor) -> Tensor:
    return torch.cat((state.mean(dim=-1), state.amax(dim=-1)), dim=1)


def depth_histogram(depths: Tensor, depth_set: tuple[int, int, int]) -> dict[str, int]:
    counts = Counter(int(value) for value in depths.detach().cpu().tolist())
    return {str(depth): counts.get(depth, 0) for depth in depth_set}


def expert_topk(scores: Tensor, capacity: int, eligible: Tensor | None = None) -> Tensor:
    """Native deterministic CPU expert-choice selection over eligible samples."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    indices = torch.arange(scores.numel(), device=scores.device) if eligible is None else eligible
    if indices.ndim != 1 or capacity < 0 or capacity > indices.numel():
        raise ValueError("invalid expert-choice capacity")
    if capacity == 0:
        return indices[:0]
    chosen = torch.topk(scores.index_select(0, indices), k=capacity, largest=True, sorted=True).indices
    return indices.index_select(0, chosen)


def capacity_counts(batch: int) -> tuple[int, int]:
    """MoR N_r=3 survival capacities: ceil(2B/3), ceil(B/3)."""
    if batch < 1:
        raise ValueError("batch must be positive")
    return math.ceil(2 * batch / 3), math.ceil(batch / 3)


class RoutedDepthConvGRU(nn.Module):
    """Candidate005 core with sample-level expert/token-choice depth routing.

    A sample is the smallest legal sparse unit because spatial positions are
    coupled by Candidate005's dilated convolutions. Router inputs contain only
    current learned hidden state. No labels, correctness, validity, or answers
    enter route selection.
    """

    def __init__(self, source: nn.Module, *, route_kind: str, depth_sets: dict[int, tuple[int, int, int]]) -> None:
        super().__init__()
        if route_kind not in {"expert_choice", "token_choice", "fixed_full", "equal_fixed", "equal_two_point"}:
            raise ValueError(f"unsupported route kind: {route_kind}")
        self.source = source
        self.route_kind = route_kind
        self.depth_sets = {int(width): tuple(int(v) for v in values) for width, values in depth_sets.items()}
        self.width = int(source.width)
        self.hidden = int(source.hidden)
        self.cycles = int(source.cycles)
        self.expert_routers = nn.ModuleList([nn.Linear(2 * self.hidden, 1), nn.Linear(2 * self.hidden, 1)])
        self.token_router = nn.Linear(2 * self.hidden, 3)
        self.register_buffer("expert_thresholds", torch.tensor([-1.0e9, -1.0e9]))
        self.register_buffer("threshold_updates", torch.zeros((), dtype=torch.int64))
        self.last_trace: dict[str, Any] = {}
        for layer in [*self.expert_routers, self.token_router]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def router_parameters(self):
        yield from self.expert_routers.parameters()
        yield from self.token_router.parameters()

    def core_parameters(self):
        yield from self.source.parameters()

    def depth_set(self, width: int) -> tuple[int, int, int]:
        if width not in self.depth_sets:
            raise ValueError(f"no frozen depth set for width {width}")
        values = self.depth_sets[width]
        calls = len(self.source.dilation_schedule(width))
        if not (0 < values[0] < values[1] < values[2] == calls):
            raise ValueError("depth set must be strictly increasing and end at the full Candidate005 schedule")
        return values

    def _update_indices(self, state: Tensor, features: Tensor, dilation: int, indices: Tensor, scale: Tensor | None = None) -> Tensor:
        if indices.numel() == 0:
            return state
        old = state.index_select(0, indices)
        candidate = self.source.recurrent_update(old, features.index_select(0, indices), int(dilation))
        if scale is not None:
            selected_scale = scale.index_select(0, indices).view(-1, 1, 1)
            candidate = old + selected_scale * (candidate - old)
        return state.index_copy(0, indices, candidate)

    def _threshold_update(self, slot: int, scores: Tensor, chosen: Tensor) -> None:
        if not self.training or chosen.numel() == 0:
            return
        threshold = scores.index_select(0, chosen).min().detach()
        with torch.no_grad():
            momentum = 0.90 if int(self.threshold_updates) else 0.0
            self.expert_thresholds[slot].mul_(momentum).add_(threshold * (1.0 - momentum))
            self.threshold_updates.add_(1)

    def _fixed_depths(self, features: Tensor, depth_set: tuple[int, int, int]) -> Tensor:
        batch = features.shape[0]
        if self.route_kind == "fixed_full":
            return torch.full((batch,), depth_set[2], dtype=torch.long, device=features.device)
        if self.route_kind == "equal_fixed":
            return torch.full((batch,), depth_set[1], dtype=torch.long, device=features.device)
        # Equal-average two-point control: deterministic alternating samples.
        indices = torch.arange(batch, device=features.device)
        return torch.where(indices.remainder(2).eq(0), depth_set[0], depth_set[2])

    def forward(self, features: Tensor, *, force_depths: Tensor | None = None, return_trace: bool = False):
        if features.ndim != 3 or features.shape[1] != 6 or not torch.isfinite(features).all():
            raise ValueError("features must be finite [batch, 6, width]")
        width, batch = int(features.shape[-1]), int(features.shape[0])
        depth_set = self.depth_set(width)
        schedule = self.source.dilation_schedule(width)
        state = self.source.stem(features)
        load_loss = state.new_zeros(())
        gate_scores: list[Tensor] = []

        if force_depths is not None:
            depths = force_depths.to(device=features.device, dtype=torch.long)
            if depths.shape != (batch,) or any(int(value) not in depth_set for value in depths.detach().cpu()):
                raise ValueError("forced depths must contain one frozen depth per sample")
        elif self.route_kind in {"fixed_full", "equal_fixed", "equal_two_point"}:
            depths = self._fixed_depths(features, depth_set)
        elif self.route_kind == "token_choice":
            logits = self.token_router(_pooled(state))
            probabilities = logits.softmax(dim=-1)
            choices = logits.argmax(dim=-1)
            values = torch.tensor(depth_set, device=features.device)
            depths = values.index_select(0, choices)
            load_loss = ((probabilities.mean(dim=0) - probabilities.new_full((3,), 1.0 / 3.0)) ** 2).sum()
            gate_scores = [logits]
        else:
            depths = torch.full((batch,), depth_set[0], dtype=torch.long, device=features.device)

        all_indices = torch.arange(batch, device=features.device)
        for call_index, dilation in enumerate(schedule, start=1):
            if force_depths is not None or self.route_kind in {"fixed_full", "equal_fixed", "equal_two_point", "token_choice"}:
                active = all_indices[depths >= call_index]
                scale = None
                if self.route_kind == "token_choice" and force_depths is None:
                    probabilities = gate_scores[0].softmax(dim=-1)
                    chosen = gate_scores[0].argmax(dim=-1)
                    scale = probabilities.gather(1, chosen[:, None]).squeeze(1)
                state = self._update_indices(state, features, dilation, active, scale)
                continue

            # Expert-choice: all samples receive the first segment. At each
            # boundary, only the prior survivors can be re-evaluated.
            active = all_indices[depths >= call_index]
            scale = None
            if call_index > depth_set[0]:
                slot = 0 if call_index <= depth_set[1] else 1
                scale = torch.sigmoid(gate_scores[slot])
            state = self._update_indices(state, features, dilation, active, scale)
            if call_index == depth_set[0]:
                scores = self.expert_routers[0](_pooled(state)).squeeze(1)
                gate_scores.append(scores)
                if self.training:
                    first_capacity, _ = capacity_counts(batch)
                    selected = expert_topk(scores, first_capacity)
                    self._threshold_update(0, scores, selected)
                else:
                    selected = all_indices[scores >= self.expert_thresholds[0]]
                depths[selected] = depth_set[1]
                target = scores.new_tensor(2.0 / 3.0)
                load_loss = load_loss + (torch.sigmoid(scores).mean() - target).square()
            elif call_index == depth_set[1]:
                eligible = all_indices[depths >= depth_set[1]]
                scores = self.expert_routers[1](_pooled(state)).squeeze(1)
                gate_scores.append(scores)
                if self.training:
                    _, final_capacity = capacity_counts(batch)
                    selected = expert_topk(scores, min(final_capacity, int(eligible.numel())), eligible)
                    self._threshold_update(1, scores, selected)
                else:
                    selected = eligible[scores.index_select(0, eligible) >= self.expert_thresholds[1]]
                depths[selected] = depth_set[2]
                survivor_probability = torch.sigmoid(scores.index_select(0, eligible)).mean() if eligible.numel() else scores.new_zeros(())
                load_loss = load_loss + (survivor_probability - scores.new_tensor(0.5)).square()

        normalized = F.silu(self.source.output_norm(state))
        logits = self.source.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        quotient = self.source.quotient_head(pooled)
        histogram = depth_histogram(depths, depth_set)
        trace = {
            "route_kind": self.route_kind,
            "depth_set": list(depth_set),
            "depths": depths.detach(),
            "histogram": histogram,
            "average_calls": float(depths.float().mean().detach()),
            "maximum_calls": int(depths.max().detach()),
            "saved_calls": int(depth_set[2] * batch - depths.sum().detach()),
            "load_loss": load_loss,
            "thresholds": self.expert_thresholds.detach().clone(),
        }
        self.last_trace = trace
        return (logits, quotient, trace) if return_trace else (logits, quotient)

    def accounting(self, width: int) -> dict[str, int]:
        return {
            "stored_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "source_parameters": sum(parameter.numel() for parameter in self.source.parameters()),
            "router_parameters": sum(parameter.numel() for parameter in self.router_parameters()),
            "full_calls": len(self.source.dilation_schedule(width)),
        }


def fixed_full_identity(source: nn.Module, routed: RoutedDepthConvGRU, features: Tensor) -> dict[str, Any]:
    depth = routed.depth_set(int(features.shape[-1]))[-1]
    forced = torch.full((features.shape[0],), depth, dtype=torch.long, device=features.device)
    source.eval(); routed.eval()
    with torch.no_grad():
        expected = source(features)
        observed = routed(features, force_depths=forced)
    return {
        "bit_logits_equal": torch.equal(expected[0], observed[0]),
        "quotient_logits_equal": torch.equal(expected[1], observed[1]),
        "maximum_absolute_delta": max(float((expected[0] - observed[0]).abs().max()), float((expected[1] - observed[1]).abs().max())),
    }


__all__ = ["RoutedDepthConvGRU", "capacity_counts", "depth_histogram", "expert_topk", "fixed_full_identity"]
