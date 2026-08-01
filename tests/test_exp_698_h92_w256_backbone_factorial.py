from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from experiments.exp_698_h92_w256_backbone_factorial import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_698_h92_w256_backbone_factorial.yaml").read_text())


def test_frozen_factorial_and_full_denominators() -> None:
    assert tuple(CONFIG["arms"]) == run.ARM_ORDER
    assert {v["state"] for v in CONFIG["arms"].values()} == {"teacher_forced", "hard_straight_through"}
    assert {v["schedule"] for v in CONFIG["arms"].values()} == {"single_cosine", "warmup_restarts"}
    assert CONFIG["training"]["updates"] == 4800
    assert CONFIG["training"]["batch_size"] == 16 and CONFIG["training"]["horizon"] == 8
    assert CONFIG["evaluation"] == {"transition_cases_per_arm_width": 3000, "w128_rollout_cases_per_arm": 300, "w256_rollout_cases_per_arm": 600, "lower_tier_parity_cases": 500, "confidence_level": 0.99, "multiplicity_family": 8}


def test_exact_roots_and_checkpoint_schedule() -> None:
    assert CONFIG["roots"] == {"calibration": 20261386, "formal": 20261387, "confirmations": [20261388, 20261389], "independent": [20261390, 20261391, 20261392, 20261393, 20261394]}
    assert CONFIG["training"]["checkpoint_updates"] == [400, 800, 1200, 2400, 3600, 4800]
    assert CONFIG["training"]["curve_every_updates"] == 100


def test_width_order_is_w128_then_balanced() -> None:
    assert all(run.balanced_width(i, 1200) == 128 for i in range(1, 1201))
    tail = [run.balanced_width(i, 1200) for i in range(1201, 4801)]
    assert tail.count(128) == tail.count(256) == 1800
    assert tail[:4] == [128, 256, 128, 256]
    assert "profile_width" in inspect.getsource(run.train_arm)


def test_native_schedulers_and_factors_are_isolated() -> None:
    source = inspect.getsource(run.make_scheduler)
    assert "CosineAnnealingLR" in source and "LinearLR" in source and "CosineAnnealingWarmRestarts" in source and "SequentialLR" in source
    training = inspect.getsource(run.train_arm)
    assert 'arm.startswith("teacher_")' in training
    assert '"learner_straight_through"' in training
    assert "balanced_width" in training


def test_source_checkpoint_is_exact_h92_two_cycle() -> None:
    checkpoint = Path(CONFIG["source_checkpoint"])
    assert run.sha256(checkpoint) == CONFIG["source_checkpoint_sha256"]
    model = run.new_model(CONFIG, checkpoint)
    expected = load_file(checkpoint)
    assert model.parameter_count() == 99824
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())


def test_source_health_is_bound_before_roots(tmp_path: Path) -> None:
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "smoke", None)
    assert experiment.identity()["source_checkpoint_sha256"] == CONFIG["source_checkpoint_sha256"]


@pytest.mark.parametrize("mode,root", [("calibration", 20261387), ("formal", 20261386), ("confirmations", 20261390), ("independent", 20261389)])
def test_root_allowlists_fail_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_runtime_is_interpretation_only_and_science_is_full() -> None:
    calibration = inspect.getsource(run.Experiment.calibration)
    science = inspect.getsource(run.Experiment.science)
    assert '"research-only" if route["resource"]["wall_seconds"] > 300' in calibration
    assert "for arm in ARM_ORDER" in science
    assert "3000, 300" in science and "3000, 600" in science
    assert "candidate_runtime" not in science


def test_inference_route_is_learned_hard_residue_only() -> None:
    source = inspect.getsource(run.rollout).lower()
    forbidden = ("barrett", "montgomery", "crt", "repair", "expected", "quotient", "carry", "token", "adapter", "routing")
    assert not any(token in source for token in forbidden)


def test_confidence_prediction_first_error_reload_controls_persist() -> None:
    evaluation = inspect.getsource(run.evaluate)
    training = inspect.getsource(run.train_arm)
    assert "confidence_99_time_uniform_multiplicity_controlled" in evaluation
    for token in ("prediction_sha256", "target_sha256", "first_error_sha256", "any_invalid_residue_rate"):
        assert token in evaluation
    assert "reload_exact" in training and "checkpoints" in training and "curves" in training


def test_resource_policy_is_dynamic_and_never_fidelity_reduced() -> None:
    policy = CONFIG["resource_policy"]
    assert policy["soft_guard_multiplier"] == 1.25 and policy["hard_guard_multiplier"] == 1.75
    assert policy["inference_over_300_is_research_only"] is True
    assert policy["fidelity_reduced"] is False
