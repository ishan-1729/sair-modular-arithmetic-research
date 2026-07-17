from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def test_higher_margin_selector_uses_original_for_exact_ties() -> None:
    runner = load_module(
        "exp640_selector_test",
        ROOT / "experiments/exp_640_commutative_operand_swap_diversity/run.py",
    )
    result = runner.select_by_margin(
        [10, 20, 30], [11, 21, 31], [0.2, 0.4, 0.6], [0.3, 0.4, 0.5]
    )
    assert result["predictions"] == [11, 20, 30]
    assert result["use_swapped"] == [True, False, False]


def test_paired_counts_partition_every_case() -> None:
    runner = load_module(
        "exp640_paired_test",
        ROOT / "experiments/exp_640_commutative_operand_swap_diversity/run.py",
    )
    counts = runner.paired_counts([1, 2, 0, 0], [1, 0, 3, 0], [1, 2, 3, 4])
    assert counts == {
        "intersection": 1,
        "original_only": 1,
        "swapped_only": 1,
        "neither": 1,
        "prediction_disagreement": 2,
    }


def test_config_freezes_consumed_surface_and_calibrated_budget() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/exp_640_commutative_operand_swap_diversity.yaml").read_text()
    )
    consumed = config["consumed_evidence"]
    assert consumed["source_run_id"] == "exp-639-full-depth-soft-boundary-adaptation"
    assert consumed["cases"] == 300
    assert consumed["expected_original_correct"] == 248
    assert config["selector"]["tie_break"] == "original"
    budget = config["resource_budget"]
    assert budget["calibration_status"] == "complete"
    assert budget["expected_total_wall_seconds"] == 1125
    assert budget["expected_total_aggregate_cpu_minutes"] == 69
    assert budget["soft_budget_wall_seconds"] < budget[
        "hard_runaway_ceiling_wall_seconds"
    ]
