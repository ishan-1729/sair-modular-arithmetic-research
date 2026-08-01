"""Small dependency-free paired sequential statistics used by council entrants."""

from __future__ import annotations

import math


def paired_counts(primary: list[bool], control: list[bool]) -> dict[str, int]:
    if len(primary) != len(control):
        raise ValueError("paired observations must have equal length")
    wins = sum(a and not b for a, b in zip(primary, control, strict=True))
    losses = sum(b and not a for a, b in zip(primary, control, strict=True))
    return {"cases": len(primary), "wins": wins, "losses": losses, "ties": len(primary) - wins - losses}


def anytime_hoeffding_interval(values: list[float], alpha: float) -> dict[str, float]:
    """Conservative time-uniform confidence sequence via a summable alpha schedule."""
    if not values or not 0.0 < alpha < 1.0:
        raise ValueError("nonempty values and 0<alpha<1 required")
    n = len(values)
    spent = alpha * 6.0 / (math.pi * math.pi * n * n)
    radius = math.sqrt(math.log(2.0 / spent) / (2.0 * n))
    mean = sum(values) / n
    return {"mean": mean, "lower": max(-1.0, mean - radius), "upper": min(1.0, mean + radius), "alpha_at_n": spent}
