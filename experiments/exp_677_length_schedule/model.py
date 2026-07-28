"""Training-only known-step scheduling over unchanged Candidate005 recurrence."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Literal

import torch
from torch import Tensor, nn

from experiments.exp_676_lla_moving_window.model import readout


ScheduleKind = Literal["scheduled", "fixed", "equal_work_shuffled"]


def _expanded_schedule(schedule: dict[str, Any]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    expected_start = 1
    for raw in schedule["tuples"]:
        item = {key: int(raw[key]) for key in ("start", "end", "width", "physical_visits", "supervised_endpoint_length")}
        if item["start"] != expected_start or item["end"] < item["start"]:
            raise ValueError("schedule ranges must be contiguous and ordered")
        if item["width"] != item["supervised_endpoint_length"]:
            raise ValueError("SAIR endpoint length must equal the legal operand width")
        result.extend([item] * (item["end"] - item["start"] + 1))
        expected_start = item["end"] + 1
    if len(result) != int(schedule["steps"]):
        raise ValueError("schedule must cover every frozen update exactly once")
    return result


def schedule_plan(kind: ScheduleKind, *, step: int, total_steps: int, schedule: dict[str, Any], seed: int) -> dict[str, int | str]:
    """Return the preregistered width/known-step/final-endpoint tuple."""
    if total_steps < 1 or step < 1 or step > total_steps:
        raise ValueError("step must lie inside the executed update schedule")
    expanded = _expanded_schedule(schedule)
    frozen_steps = len(expanded)
    frozen_index = 0 if total_steps == 1 else round((step - 1) * (frozen_steps - 1) / (total_steps - 1))
    if kind == "fixed":
        fixed = schedule["fixed"]
        selected = {key: int(fixed[key]) for key in ("width", "physical_visits", "supervised_endpoint_length")}
    elif kind == "scheduled":
        selected = expanded[frozen_index]
    elif kind == "equal_work_shuffled":
        digest = hashlib.sha256(f"exp677:{seed}:{frozen_steps}".encode()).digest()
        order = list(range(frozen_steps)); random.Random(int.from_bytes(digest[:8], "little")).shuffle(order)
        selected = expanded[order[frozen_index]]
    else:
        raise ValueError(f"unknown schedule kind: {kind}")
    return {
        "kind": kind,
        "frozen_schedule_step": frozen_index + 1,
        "width": int(selected["width"]),
        "physical_visits": int(selected["physical_visits"]),
        "supervised_endpoint_length": int(selected["supervised_endpoint_length"]),
    }


def trace_forward_budget(model: nn.Module, features: Tensor, physical_visits: int) -> list[tuple[Tensor, Tensor]]:
    """Trace an explicitly bounded training visit budget without changing inference."""
    if features.ndim != 3 or features.shape[1] != 6 or not torch.isfinite(features).all():
        raise ValueError("features must be finite [batch, 6, width]")
    schedule = model.dilation_schedule(features.shape[-1])
    if physical_visits < 1 or physical_visits > len(schedule):
        raise ValueError("physical visit budget must fit the existing dilation schedule")
    hidden = model.stem(features); outputs: list[tuple[Tensor, Tensor]] = []
    for dilation in schedule[:physical_visits]:
        hidden = model.recurrent_update(hidden, features, int(dilation)); outputs.append(readout(model, hidden))
    return outputs


def schedule_coverage(schedule: dict[str, Any], seed: int) -> dict[str, Any]:
    total = int(schedule["steps"]); receipts: dict[str, Any] = {}
    for kind in ("scheduled", "fixed", "equal_work_shuffled"):
        plans = [schedule_plan(kind, step=step, total_steps=total, schedule=schedule, seed=seed) for step in range(1, total + 1)]
        counts: dict[str, int] = {}
        for plan in plans:
            key = f"{plan['width']}:{plan['physical_visits']}:{plan['supervised_endpoint_length']}"
            counts[key] = counts.get(key, 0) + 1
        receipts[kind] = {"counts": counts, "first": plans[0], "last": plans[-1]}
    receipts["equal_work_multiset_matches"] = receipts["scheduled"]["counts"] == receipts["equal_work_shuffled"]["counts"]
    return receipts
