"""Offline replay-record helpers for EXP-629."""

from __future__ import annotations


def exact_transition_at(
    left: int,
    right: int,
    modulus: int,
    position: int,
    width: int,
) -> dict[str, int]:
    """Return the exact teacher transition at one Horner position."""
    if not 0 <= position < width:
        raise ValueError("position must be within the rollout width")
    state = 0
    for current in range(position + 1):
        bit = (left >> (width - 1 - current)) & 1
        raw = 2 * state + bit * right
        target = raw % modulus
        quotient = raw // modulus
        if current == position:
            return {
                "state": state,
                "right": right,
                "modulus": modulus,
                "bit": bit,
                "target": target,
                "quotient": quotient,
                "width": width,
                "position": position,
            }
        state = target
    raise AssertionError("unreachable")

