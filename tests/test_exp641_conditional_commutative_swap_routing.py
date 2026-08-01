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


def test_lowest_margin_route_uses_case_index_for_ties() -> None:
    runner = load_module(
        "exp641_route_test",
        ROOT / "experiments/exp_641_conditional_commutative_swap_routing/run.py",
    )
    assert runner.lowest_margin_indices([0.4, 0.2, 0.2, 0.8, 0.1], 3) == [4, 1, 2]


def test_conditional_selection_changes_only_routed_rows() -> None:
    runner = load_module(
        "exp641_select_test",
        ROOT / "experiments/exp_641_conditional_commutative_swap_routing/run.py",
    )
    selected = runner.conditional_select(
        [10, 20, 30, 40],
        [11, 21],
        [0.1, 0.4, 0.3, 0.2],
        [0.2, 0.4],
        [0, 1],
    )
    assert selected["predictions"] == [11, 20, 30, 40]
    assert selected["use_swapped"] == [True, False, False, False]


def test_config_freezes_prior_supported_fraction_and_reused_budget() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/exp_641_conditional_commutative_swap_routing.yaml").read_text()
    )
    assert config["routing"]["fraction"] == 0.40
    assert config["consumed_evidence"]["cases"] == 300
    assert config["consumed_evidence"]["paired_cache_sha256"] == (
        "4f295bbbea499d06796d48c0ac1959d2626bb3e44a3413aa3ab0dfe67065a5dd"
    )
    budget = config["resource_budget"]
    assert budget["calibration_status"] == "complete-reused-exp640"
    assert budget["expected_total_wall_seconds"] == 795
    assert budget["expected_total_aggregate_cpu_minutes"] == 48
