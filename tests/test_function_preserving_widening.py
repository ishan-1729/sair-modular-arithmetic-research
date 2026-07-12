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


def test_residual_widening_is_exact_and_trainable() -> None:
    cellular = load_module(
        "cellular_for_widening_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    widening = load_module(
        "widening_test_model",
        ROOT / "experiments/exp_611_tier6_function_preserving_widening/model.py",
    )
    torch.manual_seed(611)
    base = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference.load_state_dict(base.state_dict())
    adapter = cellular.SharedDilatedConvGRU(width=16, hidden=5, cycles=2)
    model = widening.ResidualWidenedCell(base, adapter)
    features = torch.randn(7, 6, 16)

    reference_logits, reference_quotient = reference(features)
    widened_logits, widened_quotient = model(features)
    assert torch.equal(widened_logits, reference_logits)
    assert torch.equal(widened_quotient, reference_quotient)
    assert model.parameter_count() == (
        model.base_parameter_count() + model.adapter_parameter_count()
    )

    loss = widened_logits.square().mean() + widened_quotient.square().mean()
    loss.backward()
    assert model.adapter.local_head.weight.grad is not None
    assert model.adapter.local_head.weight.grad.abs().sum() > 0
    assert model.adapter.quotient_head[-1].weight.grad is not None
    assert model.adapter.quotient_head[-1].weight.grad.abs().sum() > 0
