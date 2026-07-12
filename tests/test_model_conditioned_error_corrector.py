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


def test_conditioned_corrector_is_exact_frozen_and_trainable() -> None:
    cellular = load_module(
        "cellular_for_corrector_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    correction = load_module(
        "conditioned_corrector_test_model",
        ROOT / "experiments/exp_613_tier6_model_conditioned_error_corrector/model.py",
    )
    torch.manual_seed(613)
    base = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference.load_state_dict(base.state_dict())
    model = correction.ErrorCorrectedCell(
        base, correction.ConditionedCorrector(width=16, hidden=7, cycles=2)
    )
    features = torch.randn(6, 6, 16)

    reference_logits, reference_quotient = reference(features)
    corrected_logits, corrected_quotient = model(features)
    assert torch.equal(corrected_logits, reference_logits)
    assert torch.equal(corrected_quotient, reference_quotient)
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    assert model.trainable_parameter_count() < model.parameter_count()

    loss = corrected_logits.square().mean() + corrected_quotient.square().mean()
    loss.backward()
    assert model.corrector.local_head.weight.grad is not None
    assert model.corrector.local_head.weight.grad.abs().sum() > 0
    assert model.corrector.quotient_head[-1].weight.grad is not None
    assert model.corrector.quotient_head[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base.parameters())
