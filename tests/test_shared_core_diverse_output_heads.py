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


def modules():
    base = load_module(
        "diverse_heads_base_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    diverse = load_module(
        "diverse_heads_model_test",
        ROOT / "experiments/exp_632_tier6_shared_core_diverse_output_heads/model.py",
    )
    return base, diverse


def test_zero_initialized_heads_exactly_reproduce_base_predictions() -> None:
    base_module, diverse_module = modules()
    torch.manual_seed(3)
    base = base_module.SharedDilatedConvGRU(width=8, hidden=9, cycles=1)
    model = diverse_module.SharedCoreDiverseHeads(base, mask_seed=11)
    features = torch.randn(4, 6, 8)
    base_logits, base_quotient = base(features)
    head_logits, quotient = model.forward_heads(features)
    voted, voted_quotient = model(features)
    assert torch.equal(head_logits[:, 0], base_logits)
    assert torch.equal(head_logits[:, 1], base_logits)
    assert torch.equal(head_logits[:, 2], base_logits)
    assert torch.equal(voted.ge(0), base_logits.ge(0))
    assert torch.equal(quotient, base_quotient)
    assert torch.equal(voted_quotient, base_quotient)


def test_complementary_masks_use_each_channel_in_two_heads() -> None:
    base_module, diverse_module = modules()
    base = base_module.SharedDilatedConvGRU(width=8, hidden=12, cycles=1)
    model = diverse_module.SharedCoreDiverseHeads(base, mask_seed=5)
    assert model.channel_masks.shape == (3, 12, 1)
    assert torch.equal(model.channel_masks.sum(dim=0), torch.full((12, 1), 2.0))
    assert model.channel_masks.sum(dim=(1, 2)).tolist() == [8.0, 8.0, 8.0]


def test_only_residual_heads_are_trainable() -> None:
    base_module, diverse_module = modules()
    base = base_module.SharedDilatedConvGRU(width=8, hidden=9, cycles=1)
    model = diverse_module.SharedCoreDiverseHeads(base)
    assert model.trainable_parameter_count() == 3 * (9 + 1)
    assert all(not parameter.requires_grad for parameter in model.base.parameters())

