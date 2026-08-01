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


def test_candidate_features_use_paired_public_and_learned_values() -> None:
    features = load_module(
        "reranker_features_test",
        ROOT / "experiments/exp_627_tier6_trajectory_feature_reranker/features.py",
    )
    confidence = {key: [0.1] for key in features.CONFIDENCE_KEYS}
    result = features.build_candidate_features(
        multiplier=[5],
        multiplicand=[7],
        modulus=[11],
        greedy_predictions=[3],
        confidence=confidence,
        beam_states=[[3, 2, 1, 0]],
        beam_scores=[[-1.0, -2.0, -3.0, -4.0]],
        width=4,
    )
    assert result.shape == (1, 4, len(features.FEATURE_NAMES))
    assert result[0, 0, 3].item() == 1.0
    assert result[0, 1, 3].item() == 0.0
    assert result[0, 1, 4].item() == 0.25


def test_reranker_initializes_to_greedy_candidate_but_is_trainable() -> None:
    model_module = load_module(
        "reranker_model_test",
        ROOT / "experiments/exp_627_tier6_trajectory_feature_reranker/model.py",
    )
    model = model_module.TrajectoryFeatureReranker(feature_count=20, hidden=8)
    values = torch.zeros((2, 4, 20))
    values[:, 2, 3] = 1.0
    scores = model(values)
    assert scores.argmax(dim=1).tolist() == [2, 2]
    loss = torch.nn.functional.cross_entropy(scores, torch.tensor([1, 1]))
    loss.backward()
    assert model.greedy_prior.grad is not None
    assert model.network[-1].weight.grad is not None
