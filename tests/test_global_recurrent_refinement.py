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


def test_global_refinement_is_exact_and_becomes_trainable() -> None:
    cellular = load_module(
        "cellular_for_refinement_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    refinement = load_module(
        "global_refinement_test_model",
        ROOT / "experiments/exp_614_tier6_global_recurrent_refinement/model.py",
    )
    torch.manual_seed(614)
    base = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference.load_state_dict(base.state_dict())
    model = refinement.GloballyRefinedCell(
        base, refinement.GlobalRefinementBlock(hidden=12, passes=2)
    )
    features = torch.randn(8, 6, 16)

    reference_logits, reference_quotient = reference(features)
    refined_logits, refined_quotient = model(features)
    assert torch.equal(refined_logits, reference_logits)
    assert torch.equal(refined_quotient, reference_quotient)
    assert not any(parameter.requires_grad for parameter in model.base.parameters())

    optimizer = torch.optim.SGD(
        [model.blend, *model.refinement.parameters()], lr=0.1
    )
    loss = refined_logits.square().mean() + refined_quotient.square().mean()
    loss.backward()
    assert model.blend.grad is not None and model.blend.grad.abs().sum() > 0
    assert model.refinement.gate_weight.grad is not None
    assert model.refinement.gate_weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_logits, second_quotient = model(features)
    second_loss = second_logits.square().mean() + second_quotient.square().mean()
    second_loss.backward()
    assert model.refinement.gate_weight.grad is not None
    assert model.refinement.gate_weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base.parameters())
