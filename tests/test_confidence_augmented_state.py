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


def test_confidence_cell_is_exact_and_confidence_trainable() -> None:
    cellular = load_module(
        "cellular_for_confidence_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    confidence = load_module(
        "confidence_augmented_test_model",
        ROOT / "experiments/exp_619_tier6_confidence_augmented_state/model.py",
    )
    torch.manual_seed(619)
    base = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference.load_state_dict(base.state_dict())
    model = confidence.ConfidenceAugmentedCell(base)
    base_features = torch.randn(7, 6, 16)
    confidence_features = torch.randn(7, 2, 16)
    features = torch.cat((base_features, confidence_features), dim=1)

    reference_logits, reference_quotient = reference(base_features)
    adapted_logits, adapted_quotient = model(features)
    assert torch.equal(adapted_logits, reference_logits)
    assert torch.equal(adapted_quotient, reference_quotient)
    assert not any(parameter.requires_grad for parameter in model.base.parameters())

    loss = adapted_logits.square().mean() + adapted_quotient.square().mean()
    loss.backward()
    assert model.confidence_stem.weight.grad is not None
    assert model.confidence_stem.weight.grad.abs().sum() > 0
    assert model.confidence_gate_weight.grad is not None
    assert model.confidence_gate_weight.grad.abs().sum() > 0
    assert model.confidence_candidate_weight.grad is not None
    assert model.confidence_candidate_weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base.parameters())


def test_confidence_features_are_bounded() -> None:
    confidence = load_module(
        "confidence_feature_test_model",
        ROOT / "experiments/exp_619_tier6_confidence_augmented_state/model.py",
    )
    logits = torch.tensor([[-100.0, -0.2, 0.0, 0.3, 100.0]])
    features = confidence.ConfidenceAugmentedCell.confidence_from_logits(logits)
    assert features.shape == (1, 2, 5)
    assert features.abs().max() <= 1.0
    assert torch.all(features[:, 1] >= 0)
