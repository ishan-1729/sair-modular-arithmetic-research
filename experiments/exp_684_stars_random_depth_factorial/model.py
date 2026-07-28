"""Training-only STARS and random cycle exposure on the unchanged Candidate005 core."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch
import torch.nn.functional as F
from torch import Tensor

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU


ROUTES = (
    "fixed_no_jvp",
    "fixed_stars",
    "random_no_jvp",
    "random_stars",
    "shuffled_random_stars",
    "random_stop_gradient",
)
TRAINED_ROUTES = ROUTES
JVP_ROUTES = ("fixed_stars", "random_stars", "shuffled_random_stars", "random_stop_gradient")


@dataclass
class TrainingOutput:
    logits: Tensor
    quotient_logits: Tensor
    boundary: Tensor
    final_state: Tensor
    final_dilation: int
    recurrent_visits: int


@dataclass
class ProxyRecord:
    proxy: Tensor
    first_jvp_norm: Tensor
    vector_norm: Tensor
    state_norm: Tensor
    jvp_calls: int
    parameter_gradient_enabled: bool


def compact_hash(values: list[int]) -> str:
    raw = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def frozen_count_schedules(config: dict, updates: int) -> dict[str, list[int]]:
    """Return replayable equal-work schedules; balancing is confined within width strata."""
    spec = config["count_schedule"]
    total = int(spec["frozen_updates"])
    if updates < 1 or updates > total:
        raise ValueError("updates outside frozen schedule")
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    counts = torch.poisson(
        torch.full((total,), float(spec["poisson_mean"])), generator=generator
    ).clamp(int(spec["minimum_cycles"]), int(spec["maximum_cycles"])).long()
    balance = torch.Generator().manual_seed(int(spec["balance_seed"]))
    target = total * int(spec["fixed_cycles"]) // 4
    for stratum in range(4):
        indices = torch.arange(stratum, total, 4)
        order = indices[torch.randperm(len(indices), generator=balance)]
        delta = target - int(counts[indices].sum())
        if delta > 0:
            for index in order.tolist():
                amount = min(delta, int(spec["maximum_cycles"]) - int(counts[index]))
                counts[index] += amount
                delta -= amount
                if delta == 0:
                    break
        elif delta < 0:
            delta = -delta
            for index in order.tolist():
                amount = min(delta, int(counts[index]) - int(spec["minimum_cycles"]))
                counts[index] -= amount
                delta -= amount
                if delta == 0:
                    break
        if delta:
            raise RuntimeError("unable to balance frozen count stratum")
    shuffled = counts.clone()
    shuffle = torch.Generator().manual_seed(int(spec["shuffle_seed"]))
    for stratum in range(4):
        indices = torch.arange(stratum, total, 4)
        shuffled[indices] = counts[indices[torch.randperm(len(indices), generator=shuffle)]]
    if compact_hash(counts.tolist()) != spec["random_sha256"]:
        raise RuntimeError("random schedule hash drift")
    if compact_hash(shuffled.tolist()) != spec["shuffled_sha256"]:
        raise RuntimeError("shuffled schedule hash drift")
    fixed = [int(spec["fixed_cycles"])] * total
    random_values = counts.tolist()
    shuffled_values = shuffled.tolist()
    complete = {
        "fixed_no_jvp": fixed,
        "fixed_stars": fixed,
        "random_no_jvp": random_values,
        "random_stars": random_values,
        "shuffled_random_stars": shuffled_values,
        "random_stop_gradient": random_values,
    }
    return {route: values[:updates] for route, values in complete.items()}


class FactorialCell(SharedDilatedConvGRU):
    """The source inference class plus helpers that are reachable only during training."""

    def load_source(self, state: dict[str, Tensor]) -> None:
        self.load_state_dict(state, strict=True)

    def cycle_schedule(self, width: int, cycles: int) -> list[int]:
        if not 1 <= int(cycles) <= 4:
            raise ValueError("cycle count outside frozen bounds")
        base = self.dilation_schedule(width)[: len(self.dilation_schedule(width)) // self.cycles]
        schedule: list[int] = []
        for cycle in range(int(cycles)):
            schedule.extend(base if cycle % 2 == 0 else reversed(base))
        return schedule

    def forward_train(self, features: Tensor, *, cycles: int) -> TrainingOutput:
        self.width = features.shape[-1]
        hidden = self.stem(features)
        schedule = self.cycle_schedule(features.shape[-1], cycles)
        boundary = None
        for index, dilation in enumerate(schedule):
            if index + 1 == len(schedule):
                boundary = hidden
            hidden = self.recurrent_update(hidden, features, dilation)
        if boundary is None:
            raise RuntimeError("empty recurrent schedule")
        normalized = F.silu(self.output_norm(hidden))
        logits = self.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), 1)
        return TrainingOutput(logits, self.quotient_head(pooled), boundary, hidden,
                              int(schedule[-1]), len(schedule))

    def jvp_proxy(self, state: Tensor, features: Tensor, dilation: int, *, seed: int,
                  stop_gradient: bool = False) -> ProxyRecord:
        generator = torch.Generator(device=state.device).manual_seed(int(seed))
        vector = torch.randn(state.shape, generator=generator, device=state.device,
                             dtype=state.dtype)
        vector = vector / vector.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
        transition = lambda hidden: self.recurrent_update(hidden, features, dilation)
        _, first = torch.func.jvp(transition, (state,), (vector,))
        direction = (first / first.flatten(1).norm(dim=1).view(-1, 1, 1)
                     .clamp_min(1e-12)).detach()
        _, product = torch.func.jvp(transition, (state,), (direction,))
        proxy = product.flatten(1).square().sum(1).mean()
        if stop_gradient:
            proxy = proxy.detach()
        return ProxyRecord(proxy, first.flatten(1).norm(dim=1).mean(),
                           direction.flatten(1).norm(dim=1).mean(),
                           state.flatten(1).norm(dim=1).mean(), 2, not stop_gradient)

    def randomized_intervention(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                             dtype=parameter.dtype) * 0.02)
