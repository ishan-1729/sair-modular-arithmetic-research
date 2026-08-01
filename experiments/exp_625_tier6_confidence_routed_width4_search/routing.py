"""Target-free ranking helpers for conditional learned search diagnostics."""

from __future__ import annotations


def ranked_indices(scores: list[float], count: int) -> list[int]:
    """Return the highest-score rows with deterministic index tie-breaking."""
    if not 0 <= count <= len(scores):
        raise ValueError("count must be between zero and the number of scores")
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:count]


def routed_coverage(
    greedy_correct: list[bool], beam_contains: list[bool], routed: list[int]
) -> dict:
    """Score evaluation-only coverage when routed rows may use any beam state."""
    if len(greedy_correct) != len(beam_contains):
        raise ValueError("paired outcomes must have equal length")
    selected = set(routed)
    outcomes = [
        beam_contains[index] if index in selected else greedy_correct[index]
        for index in range(len(greedy_correct))
    ]
    return {
        "correct": sum(outcomes),
        "coverage": sum(outcomes) / len(outcomes),
        "routed": len(selected),
        "routed_greedy_errors": sum(
            not greedy_correct[index] for index in selected
        ),
        "routed_beam_recoveries": sum(
            (not greedy_correct[index]) and beam_contains[index]
            for index in selected
        ),
        "routed_greedy_losses": sum(
            greedy_correct[index] and not beam_contains[index] for index in selected
        ),
    }


def oracle_allocation(
    greedy_correct: list[bool], beam_contains: list[bool], count: int
) -> list[int]:
    """Return the evaluation-only allocation that maximizes paired coverage."""
    gains = [int(beam) - int(greedy) for greedy, beam in zip(greedy_correct, beam_contains)]
    return ranked_indices([float(gain) for gain in gains], count)
