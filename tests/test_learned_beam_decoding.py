from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expansion_flips_only_least_confident_bit_and_scores_it() -> None:
    beam = load_module(
        "learned_beam_test_decoder",
        ROOT / "experiments/exp_621_tier6_learned_beam_decoding/decoder.py",
    )
    logits = torch.tensor([[2.0, -0.1, 3.0], [-4.0, 1.5, 0.2]])
    hard, alternate, hard_score, alternate_score = beam.expand_least_confident_bit(
        logits
    )
    assert hard == [5, 6]
    assert alternate == [7, 2]
    assert torch.allclose(hard_score - alternate_score, torch.tensor([0.1, 0.2]))


def test_decoder_deduplicates_states_and_uses_only_model_scores() -> None:
    beam = load_module(
        "learned_beam_rollout_test_decoder",
        ROOT / "experiments/exp_621_tier6_learned_beam_decoding/decoder.py",
    )

    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.width = 2

        def forward(self, features: torch.Tensor):
            batch = len(features)
            logits = torch.tensor([[3.0, -0.2]]).expand(batch, 2)
            return logits, torch.zeros(batch, 3)

    def features(states, rights, moduli, bits, width):
        del rights, moduli, bits
        return torch.tensor(states, dtype=torch.float32)[:, None, None].expand(
            len(states), 1, width
        )

    decoder = beam.LearnedLikelihoodBeamDecoder(FixedModel(), width=2, beam_width=2)
    result = decoder.rollout([0], [0], [3], features)
    assert len(result["beam_states"][0]) == 2
    assert len(set(result["beam_states"][0])) == 2
    assert result["predictions"][0] == result["beam_states"][0][0]
    assert result["beam_scores"][0][0] >= result["beam_scores"][0][1]


def test_decoder_contains_no_modulus_filter_or_exact_answer_feedback() -> None:
    source = (
        ROOT / "experiments/exp_621_tier6_learned_beam_decoding/decoder.py"
    ).read_text()
    rollout_body = source.split("def rollout(", 1)[1]
    assert "%" not in rollout_body
    assert "expected" not in rollout_body
    assert "target" not in rollout_body


def test_staged_beam_widens_after_declared_position() -> None:
    beam = load_module(
        "learned_beam_staged_test_decoder",
        ROOT / "experiments/exp_621_tier6_learned_beam_decoding/decoder.py",
    )

    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.width = 2

        def forward(self, features: torch.Tensor):
            first = torch.where(features[:, 0, 0] > 1, 2.0, -2.0)
            logits = torch.stack((first, torch.full_like(first, 0.5)), dim=1)
            batch = len(logits)
            return logits, torch.zeros((batch, 3))

    decoder = beam.LearnedLikelihoodBeamDecoder(
        FixedModel(),
        width=2,
        beam_width=4,
        initial_beam_width=2,
        widen_at_position=1,
    )
    result = decoder.rollout(
        [0],
        [0],
        [3],
        lambda states, rights, moduli, bits, width: torch.tensor(
            states, dtype=torch.float32
        )[:, None, None].expand(len(states), 6, width),
    )

    assert len(result["beam_states"][0]) == 4
    assert len(set(result["beam_states"][0])) == 4
