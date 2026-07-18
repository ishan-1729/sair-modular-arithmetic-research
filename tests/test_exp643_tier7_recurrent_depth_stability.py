from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "experiments/exp_643_tier7_recurrent_depth_stability_diagnostic/run.py"
    spec = importlib.util.spec_from_file_location("exp643_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preregistration_freezes_distinct_calibration_and_formal_streams() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/exp_643_tier7_recurrent_depth_stability_diagnostic.yaml").read_text()
    )
    fresh = config["fresh_evidence"]
    assert fresh["widths"] == [128, 256]
    assert len(set(fresh["formal_seeds"].values())) == 2
    assert fresh["calibration_seed"] not in fresh["formal_seeds"].values()
    assert config["evaluation"]["rollout_cases_per_width_seed"] == 100
    assert config["resource_budget"]["calibration_status"] in {"pending", "complete"}
    assert config["resource_budget"]["fidelity_reduced"] is False


def test_instrumented_forward_matches_ordinary_forward() -> None:
    runner = load_runner()
    cellular = runner.load_module(
        "cellular_for_exp643_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    torch.manual_seed(643)
    model = cellular.SharedDilatedConvGRU(width=8, hidden=8, cycles=2).eval()
    features = torch.randn(3, 6, 8)
    teacher = torch.randint(0, 2, (3, 8), dtype=torch.bool)
    with torch.inference_mode():
        ordinary, _ = model(features)
        instrumented, visits = runner.instrumented_forward(model, features, teacher)
    assert torch.equal(ordinary, instrumented)
    assert torch.equal(ordinary.ge(0), instrumented.ge(0))
    assert len(visits) == len(model.dilation_schedule(8))
    assert all(visit["exact_to_teacher"].shape == (3,) for visit in visits)


def test_random_direction_jacobian_proxy_is_finite() -> None:
    runner = load_runner()
    cellular = runner.load_module(
        "cellular_for_exp643_spectral_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    torch.manual_seed(644)
    model = cellular.SharedDilatedConvGRU(width=8, hidden=8, cycles=1).eval()
    features = torch.randn(1, 6, 8)
    result = runner.jacobian_gain_proxy(
        model, features, visit_fractions=[0.0, 1.0], directions=1, seed=645
    )
    assert len(result) == 2
    assert all(item["maximum_random_direction_gain"] >= 0.0 for item in result)
    assert all(torch.isfinite(torch.tensor(item["maximum_random_direction_gain"])) for item in result)


def test_zero_denominator_ratio_remains_canonical_json_safe() -> None:
    runner = load_runner()
    assert runner._safe_ratio(1.0, 0.0) is None
    assert runner._safe_ratio(0.0, 0.0) is None
    assert runner._safe_ratio(3.0, 2.0) == 1.5
