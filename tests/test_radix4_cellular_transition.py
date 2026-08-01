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


def test_radix4_exact_transition_uses_seven_quotient_classes() -> None:
    data = load_module(
        "radix4_data_test",
        ROOT / "experiments/exp_630_radix4_cellular_transition/data.py",
    )
    target, quotient = data.exact_transition(16, 16, 17, 3)
    assert (target, quotient) == (10, 6)


def test_radix4_teacher_rollout_equals_product_mod_prime() -> None:
    data = load_module(
        "radix4_rollout_test",
        ROOT / "experiments/exp_630_radix4_cellular_transition/data.py",
    )
    for left in range(16):
        for right in range(16):
            assert data.teacher_rollout(left, right, 17, 4) == (left * right) % 17


def test_radix4_features_expose_low_and_high_digit_planes() -> None:
    data = load_module(
        "radix4_features_test",
        ROOT / "experiments/exp_630_radix4_cellular_transition/data.py",
    )
    batch = data.features([0, 0, 0, 0], [1] * 4, [17] * 4, [0, 1, 2, 3], 5)
    assert batch.shape == (4, 7, 5)
    assert batch[:, 3, 0].tolist() == [-1.0, 1.0, -1.0, 1.0]
    assert batch[:, 4, 0].tolist() == [-1.0, -1.0, 1.0, 1.0]


def test_radix2_transfer_expands_feature_and_quotient_shapes() -> None:
    base = load_module(
        "radix2_transfer_source_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    model = load_module(
        "radix4_model_test",
        ROOT / "experiments/exp_630_radix4_cellular_transition/model.py",
    )
    transfer = load_module(
        "radix4_transfer_test",
        ROOT / "experiments/exp_630_radix4_cellular_transition/transfer.py",
    )
    torch.manual_seed(7)
    source = base.SharedDilatedConvGRU(width=8, hidden=8, cycles=1)
    target = model.Radix4SharedDilatedConvGRU(width=8, hidden=8, cycles=1)
    report = transfer.transfer_radix2_state(target, source.state_dict())
    inputs = torch.randn(3, 7, 8)
    logits, quotient = target(inputs)
    assert logits.shape == (3, 8)
    assert quotient.shape == (3, 7)
    assert torch.equal(
        target.state_dict()["quotient_head.2.weight"][:3],
        source.state_dict()["quotient_head.2.weight"],
    )
    assert report["new_quotient_classes"] == [3, 4, 5, 6]

