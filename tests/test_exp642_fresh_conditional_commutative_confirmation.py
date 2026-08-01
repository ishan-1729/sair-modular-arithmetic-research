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


def test_config_freezes_fresh_stream_and_exact_exp641_policy() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/exp_642_fresh_conditional_commutative_confirmation.yaml").read_text()
    )
    assert config["seed"] == 20260842
    assert config["fresh_evidence"]["split_seed_offset"] == 64200
    assert config["evaluation"]["transition_cases"] == 3000
    assert config["evaluation"]["rollout_cases"] == 500
    assert config["routing"]["fraction"] == 0.40
    assert config["routing"]["orientation_tie_break"] == "original"


def test_config_uses_distinct_reused_wall_and_cpu_budget() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/exp_642_fresh_conditional_commutative_confirmation.yaml").read_text()
    )
    budget = config["resource_budget"]
    assert budget["calibration_status"] == "complete-reused-exp639-exp641"
    assert budget["expected_total_wall_seconds"] == 1050
    assert budget["expected_total_aggregate_cpu_minutes"] == 63
    assert budget["expected_total_wall_seconds"] < budget["soft_budget_wall_seconds"]
    assert budget["soft_budget_wall_seconds"] < budget[
        "hard_runaway_ceiling_wall_seconds"
    ]


def test_fresh_split_is_disjoint_and_has_frozen_identity_counts() -> None:
    runner = load_module(
        "exp642_split_test",
        ROOT / "experiments/exp_642_fresh_conditional_commutative_confirmation/run.py",
    )
    config_path = ROOT / "configs/exp_642_fresh_conditional_commutative_confirmation.yaml"
    config = yaml.safe_load(config_path.read_text())
    split = runner.make_split(config, runner.load_exp300(), 128)
    assert len(split["train"]) == 512
    assert len(split["test"]) == 128
    assert set(split["train"]).isdisjoint(split["test"])
