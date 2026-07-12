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


def test_recurrent_lora_is_exact_trainable_and_mergeable() -> None:
    cellular = load_module(
        "cellular_for_lora_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    lora = load_module(
        "recurrent_lora_test_model",
        ROOT / "experiments/exp_617_tier6_recurrent_core_lora/model.py",
    )
    torch.manual_seed(617)
    base = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference = cellular.SharedDilatedConvGRU(width=16, hidden=12, cycles=2)
    reference.load_state_dict(base.state_dict())
    model = lora.RecurrentCoreLoRACell(base, rank=3, alpha=3.0)
    features = torch.randn(6, 6, 16)

    reference_logits, reference_quotient = reference(features)
    adapted_logits, adapted_quotient = model(features)
    assert torch.equal(adapted_logits, reference_logits)
    assert torch.equal(adapted_quotient, reference_quotient)
    assert not any(parameter.requires_grad for parameter in model.base.parameters())

    optimizer = torch.optim.SGD(
        [model.gate_a, model.gate_b, model.candidate_a, model.candidate_b], lr=0.1
    )
    loss = adapted_logits.square().mean() + adapted_quotient.square().mean()
    loss.backward()
    assert model.gate_b.grad is not None and model.gate_b.grad.abs().sum() > 0
    assert model.candidate_b.grad is not None
    assert model.candidate_b.grad.abs().sum() > 0
    assert model.gate_a.grad is not None and model.gate_a.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    changed_logits, changed_quotient = model(features)
    second_loss = changed_logits.square().mean() + changed_quotient.square().mean()
    second_loss.backward()
    assert model.gate_a.grad is not None and model.gate_a.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base.parameters())

    merged = model.merged_base().eval()
    model.eval()
    merged_logits, merged_quotient = merged(features)
    model_logits, model_quotient = model(features)
    assert torch.allclose(merged_logits, model_logits, rtol=1e-6, atol=1e-7)
    assert torch.allclose(merged_quotient, model_quotient, rtol=1e-6, atol=1e-7)
