from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    previous_path = sys.path.copy()
    previous_model = sys.modules.get("model")
    had_model = "model" in sys.modules
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
        if had_model:
            sys.modules["model"] = previous_model
        else:
            sys.modules.pop("model", None)
    return module


def test_straight_through_bits_are_hard_forward_with_soft_gradient() -> None:
    runner = load_module(
        "exp639_straight_through_test",
        ROOT / "experiments/exp_639_full_depth_soft_boundary_adaptation/run.py",
    )
    logits = torch.tensor([[-2.0, 0.0, 3.0]], requires_grad=True)
    state = runner.straight_through_bits(logits)
    assert state.tolist() == [[0.0, 1.0, 1.0]]
    state.sum().backward()
    assert logits.grad is not None
    assert torch.all(logits.grad > 0)


def test_macro_forward_reuses_one_model_for_both_full_depth_phases() -> None:
    runner = load_module(
        "exp639_macro_forward_test",
        ROOT / "experiments/exp_639_full_depth_soft_boundary_adaptation/run.py",
    )
    exp300 = load_module(
        "exp300_for_exp639_test",
        ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
    )

    class CountingModel:
        def __init__(self) -> None:
            self.calls = 0
            self.width = 0

        def __call__(self, features):
            self.calls += 1
            return features[:, 0] + self.calls * 0.1, torch.zeros(features.shape[0], 3)

    holder = types.SimpleNamespace(temperature=1.0, exp300=exp300)
    holder.soft_features = types.MethodType(runner.Experiment.soft_features, holder)
    model = CountingModel()
    final, intermediate, q1, q2 = runner.Experiment.macro_forward(
        holder,
        model,
        torch.zeros(2, 8),
        [3, 5],
        [17, 19],
        [0, 3],
        8,
    )
    assert model.calls == 2
    assert final.shape == intermediate.shape == (2, 8)
    assert q1.shape == q2.shape == (2, 3)


def test_four_macro_steps_equal_eight_radix2_steps() -> None:
    data = load_module(
        "exp639_macro_data_test",
        ROOT / "experiments/exp_637_teacher_distilled_macro_transition/data.py",
    )
    width = 8
    for left in range(16):
        state = 0
        for position in range(width // 2):
            digit = (left >> (width - 2 * (position + 1))) & 3
            _, _, state, _ = data.exact_macro_transition(state, 13, 17, digit)
        assert state == (left * 13) % 17


def test_config_freezes_historical_mode_counts_and_calibrated_budget() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/exp_639_full_depth_soft_boundary_adaptation.yaml").read_text()
    )
    assert config["training"]["steps"] == 400
    assert config["training"]["mode_schedule"] == [
        "transition64",
        "recovery64",
        "recovery128",
        "recovery128",
    ]
    assert config["training"]["recovery_macro_steps"] == 4
    assert config["resource_budget"]["calibration_status"] == "complete"
    assert config["resource_budget"]["expected_total_wall_seconds"] == 3832
    assert config["resource_budget"]["expected_total_aggregate_cpu_minutes"] == 234
