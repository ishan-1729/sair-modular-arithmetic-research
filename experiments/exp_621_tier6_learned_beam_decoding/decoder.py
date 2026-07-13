"""Learned-likelihood beam decoding for cellular residue transitions."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def decode_rows(bits: Tensor) -> list[int]:
    """Decode little-endian Boolean rows without performing arithmetic repair."""
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in bits.tolist()
    ]


def expand_least_confident_bit(logits: Tensor) -> tuple[list[int], list[int], Tensor, Tensor]:
    """Return hard and one-bit alternative states with learned log scores."""
    hard_bits = logits.ge(0)
    hard_values = decode_rows(hard_bits)
    least_confident = logits.abs().argmin(dim=1)
    alternate_bits = hard_bits.clone()
    alternate_bits[
        torch.arange(len(logits), device=logits.device), least_confident
    ] ^= True
    alternate_values = decode_rows(alternate_bits)
    hard_log_probability = F.logsigmoid(logits.abs()).sum(dim=1)
    alternate_log_probability = hard_log_probability - logits.abs().gather(
        1, least_confident[:, None]
    ).squeeze(1)
    return (
        hard_values,
        alternate_values,
        hard_log_probability,
        alternate_log_probability,
    )


class LearnedLikelihoodBeamDecoder:
    """Retain top unique learned state hypotheses across Horner steps."""

    def __init__(
        self,
        model: nn.Module,
        width: int,
        beam_width: int = 2,
        *,
        initial_beam_width: int | None = None,
        widen_at_position: int = 0,
    ) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if initial_beam_width is None:
            initial_beam_width = beam_width
        if not 1 <= initial_beam_width <= beam_width:
            raise ValueError("initial_beam_width must be between one and beam_width")
        if not 0 <= widen_at_position <= width:
            raise ValueError("widen_at_position must be between zero and width")
        self.model = model
        self.width = int(width)
        self.beam_width = int(beam_width)
        self.initial_beam_width = int(initial_beam_width)
        self.widen_at_position = int(widen_at_position)

    @torch.inference_mode()
    def rollout(
        self,
        multiplier: list[int],
        multiplicand: list[int],
        modulus: list[int],
        feature_builder: Callable[
            [list[int], list[int], list[int], list[int], int], Tensor
        ],
    ) -> dict:
        batch = len(multiplier)
        if not (len(multiplicand) == len(modulus) == batch):
            raise ValueError("operand batches must have equal length")
        self.model.width = self.width
        beams: list[list[tuple[int, float]]] = [[(0, 0.0)] for _ in range(batch)]
        for position in range(self.width):
            states = []
            rights = []
            moduli = []
            bits = []
            owners = []
            parent_scores = []
            for row, hypotheses in enumerate(beams):
                input_bit = (multiplier[row] >> (self.width - 1 - position)) & 1
                for state, score in hypotheses:
                    states.append(state)
                    rights.append(multiplicand[row])
                    moduli.append(modulus[row])
                    bits.append(input_bit)
                    owners.append(row)
                    parent_scores.append(score)
            logits, _ = self.model(
                feature_builder(states, rights, moduli, bits, self.width)
            )
            hard, alternate, hard_score, alternate_score = expand_least_confident_bit(
                logits
            )
            candidates: list[list[tuple[int, float]]] = [[] for _ in range(batch)]
            for index, owner in enumerate(owners):
                parent = parent_scores[index]
                candidates[owner].append((hard[index], parent + float(hard_score[index])))
                candidates[owner].append(
                    (alternate[index], parent + float(alternate_score[index]))
                )
            retained_width = (
                self.initial_beam_width
                if position < self.widen_at_position
                else self.beam_width
            )
            next_beams = []
            for row_candidates in candidates:
                best_by_state: dict[int, float] = {}
                for state, score in row_candidates:
                    best_by_state[state] = max(score, best_by_state.get(state, -torch.inf))
                ranked = sorted(
                    best_by_state.items(), key=lambda item: item[1], reverse=True
                )[:retained_width]
                next_beams.append([(state, score) for state, score in ranked])
            beams = next_beams
        predictions = [hypotheses[0][0] for hypotheses in beams]
        return {
            "predictions": predictions,
            "beam_states": [[state for state, _ in hypotheses] for hypotheses in beams],
            "beam_scores": [[score for _, score in hypotheses] for hypotheses in beams],
        }
