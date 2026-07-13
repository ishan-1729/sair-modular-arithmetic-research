"""Legal candidate features for trained beam reranking."""

from __future__ import annotations

import torch


CONFIDENCE_KEYS = (
    "mean_bit_nll",
    "negative_trajectory_min_margin",
    "negative_mean_step_min_margin",
    "negative_last32_mean_min_margin",
    "negative_final_min_margin",
    "mean_quotient_entropy",
)

FEATURE_NAMES = (
    "beam_rank",
    "learned_score_per_step",
    "score_gap_per_step",
    "equals_greedy",
    "hamming_to_greedy",
    "hamming_to_top",
    "candidate_popcount",
    "greedy_popcount",
    "multiplier_popcount",
    "multiplicand_popcount",
    "modulus_popcount",
    "candidate_xor_modulus_popcount",
    "candidate_xor_multiplicand_popcount",
    "candidate_bit_length",
    *CONFIDENCE_KEYS,
)


def build_candidate_features(
    multiplier: list[int],
    multiplicand: list[int],
    modulus: list[int],
    greedy_predictions: list[int],
    confidence: dict[str, list[float]],
    beam_states: list[list[int]],
    beam_scores: list[list[float]],
    width: int,
) -> torch.Tensor:
    """Build `[case, candidate, feature]` tensors without answer access."""
    cases = len(multiplier)
    if not (
        len(multiplicand)
        == len(modulus)
        == len(greedy_predictions)
        == len(beam_states)
        == len(beam_scores)
        == cases
    ):
        raise ValueError("paired inputs must have equal case counts")
    if any(len(confidence[key]) != cases for key in CONFIDENCE_KEYS):
        raise ValueError("confidence inputs must cover every case")
    rows = []
    for case in range(cases):
        if len(beam_states[case]) != len(beam_scores[case]):
            raise ValueError("beam states and scores must align")
        top_state = beam_states[case][0]
        top_score = beam_scores[case][0]
        shared_confidence = [confidence[key][case] for key in CONFIDENCE_KEYS]
        candidates = []
        for rank, (state, score) in enumerate(
            zip(beam_states[case], beam_scores[case])
        ):
            candidates.append(
                [
                    rank / max(1, len(beam_states[case]) - 1),
                    score / width,
                    (top_score - score) / width,
                    float(state == greedy_predictions[case]),
                    (state ^ greedy_predictions[case]).bit_count() / width,
                    (state ^ top_state).bit_count() / width,
                    state.bit_count() / width,
                    greedy_predictions[case].bit_count() / width,
                    multiplier[case].bit_count() / width,
                    multiplicand[case].bit_count() / width,
                    modulus[case].bit_count() / width,
                    (state ^ modulus[case]).bit_count() / width,
                    (state ^ multiplicand[case]).bit_count() / width,
                    state.bit_length() / width,
                    *shared_confidence,
                ]
            )
        rows.append(candidates)
    return torch.tensor(rows, dtype=torch.float32)
