from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from experiments.exp_698_h92_w256_backbone_factorial import run as exp698
from experiments.exp_699_h92_w256_backbone_factorial_complete import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_699_h92_w256_backbone_factorial_complete.yaml").read_text())
OLD = yaml.safe_load((ROOT / "configs/exp_698_h92_w256_backbone_factorial.yaml").read_text())


def test_fresh_identity_roots_and_exact_source() -> None:
    assert CONFIG["run_id"] == "exp-699-h92-w256-backbone-factorial-complete"
    assert CONFIG["roots"] == {"calibration": 20261395, "formal": 20261396, "confirmations": [20261397, 20261398], "independent": [20261399, 20261400, 20261401, 20261402, 20261403]}
    assert CONFIG["source_checkpoint_sha256"] == "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
    assert run.sha256(Path(CONFIG["source_checkpoint"])) == CONFIG["source_checkpoint_sha256"]


def test_exp698_primary_contract_is_unchanged_except_identity_and_diagnostics() -> None:
    for key in ("model", "arms", "data", "training", "gates"):
        assert CONFIG[key] == OLD[key]
    for name in ("new_model", "balanced_width", "make_scheduler", "rollout", "interval", "evaluate"):
        assert inspect.getsource(getattr(run, name)) == inspect.getsource(getattr(exp698, name))
    assert run.base.train_update is exp698.base.train_update
    assert run.base.tensor_native_rollout is exp698.base.tensor_native_rollout


def test_factorial_work_and_denominators_are_complete() -> None:
    assert tuple(CONFIG["arms"]) == run.ARM_ORDER
    assert {value["state"] for value in CONFIG["arms"].values()} == {"teacher_forced", "hard_straight_through"}
    assert {value["schedule"] for value in CONFIG["arms"].values()} == {"single_cosine", "warmup_restarts"}
    assert CONFIG["training"]["updates"] == 4800
    assert CONFIG["training"]["batch_size"] == 16 and CONFIG["training"]["horizon"] == 8
    assert CONFIG["training"]["checkpoint_updates"] == [400, 800, 1200, 2400, 3600, 4800]
    assert CONFIG["training"]["curve_every_updates"] == 100
    evaluation = CONFIG["evaluation"]
    assert (evaluation["transition_cases_per_arm_width"], evaluation["w128_rollout_cases_per_arm"], evaluation["w256_rollout_cases_per_arm"]) == (3000, 300, 600)
    assert (evaluation["source_transition_cases_per_width"], evaluation["source_w128_rollout_cases"], evaluation["source_w256_rollout_cases"]) == (3000, 300, 600)
    assert (evaluation["randomized_transition_cases_per_width"], evaluation["randomized_w128_rollout_cases"], evaluation["randomized_w256_rollout_cases"]) == (300, 100, 200)
    assert evaluation["lower_tier_parity_cases"] == 500


def test_width_order_and_identical_balanced_tail() -> None:
    assert all(run.balanced_width(update, 1200) == 128 for update in range(1, 1201))
    tail = [run.balanced_width(update, 1200) for update in range(1201, 4801)]
    assert tail[:4] == [128, 256, 128, 256]
    assert tail.count(128) == tail.count(256) == 1800
    assert "data_order_sha256" in inspect.getsource(run.train_arm)


def test_source_checkpoint_loads_exact_h92_two_cycle() -> None:
    checkpoint = Path(CONFIG["source_checkpoint"])
    model = run.new_model(CONFIG, checkpoint)
    expected = load_file(checkpoint)
    assert model.parameter_count() == 99824
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())


def test_all_required_terminal_outputs_are_enumerated() -> None:
    assert run.TERMINAL_OUTPUT_CONTRACT == (
        "schema_version", "run_id", "status", "mode", "root", "decision", "source",
        "records", "evaluations", "winner", "health", "confirmation_trigger",
        "diagnostics.frozen_source", "diagnostics.reset_to_source",
        "diagnostics.randomized_weights", "diagnostics.lower_tier_parity",
        "diagnostics.trained_weight_collapse", "work_equivalence", "dynamic_budget",
        "resource", "mlflow", "limitations",
    )
    science = inspect.getsource(run.Experiment.science)
    assert "assert_terminal_contract(result)" in science
    smoke = inspect.getsource(run.Experiment.smoke)
    for token in ("calibration", "formal", "confirmations", "independent"):
        assert token in smoke


def test_diagnostic_branches_denominators_cache_and_interpretation_are_wired() -> None:
    source = inspect.getsource(run.diagnostics)
    for token in ("frozen_source", "reset_to_source", "randomized_weights", "lower_tier_parity", "trained_weight_collapse"):
        assert token in source
    cached = inspect.getsource(run.cached_evaluate)
    for token in ("checkpoint_sha256", "evaluator_sha256", "case_sha256", "config_sha256"):
        assert token in cached
    assert "byte-identical to frozen source" in source
    assert "randomized_transition_cases_per_width" in source


def test_paired_hash_confidence_invalid_reload_and_resource_outputs() -> None:
    evidence = inspect.getsource(run.paired_evidence)
    evaluation = inspect.getsource(run.evaluate_with_evidence)
    training = inspect.getsource(run.train_arm)
    for token in ("paired_vectors", "paired_vectors_sha256", "prediction_sha256", "case_sha256", "first_error_sha256"):
        assert token in evidence or token in evaluation
    assert "confidence_99_time_uniform_multiplicity_controlled" in evaluation
    assert "reload_exact" in training and "resource" in training


@pytest.mark.parametrize("mode,root", [("calibration", 20261396), ("formal", 20261395), ("confirmations", 20261399), ("independent", 20261398)])
def test_root_allowlists_fail_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_predecessor_gates_and_runtime_non_truncation() -> None:
    predecessor = inspect.getsource(run.Experiment.require_predecessor_gate)
    assert "both passing confirmations" in predecessor
    calibration = inspect.getsource(run.Experiment.calibration)
    science = inspect.getsource(run.Experiment.science)
    assert '"research-only" if route["resource"]["wall_seconds"] > 300' in calibration
    assert "candidate_runtime" not in science
    assert "for arm in ARM_ORDER" in science
    assert "3000, 300" in science and "3000, 600" in science


def test_root_audit_covers_worktree_outer_repository_scratch_and_mlflow() -> None:
    source = inspect.getsource(run.Experiment.audit_fresh_roots)
    assert 'ROOT.parent.parent.name == ".scratch"' in source
    assert 'workspace / ".scratch"' in source
    assert 'workspace / ".scratch/mlflow/mlflow.db"' in source
    assert 'workspace / "experiments/runs"' in source and 'workspace / "evals/results"' in source


def test_dynamic_budget_reuses_only_equivalent_exp698_route() -> None:
    route = inspect.getsource(run.Experiment.route_equivalence)
    calibration = inspect.getsource(run.Experiment.calibration)
    assert "exact_function_hashes" in route and "exact_component_hashes" in route
    assert "EXP698 root20261386 unchanged-route throughput" in calibration
    assert '"soft_wall_seconds": expected * 1.25' in calibration
    assert '"hard_wall_seconds": expected * 1.75' in calibration


def test_deployed_inference_remains_hard_learned_residue_only() -> None:
    source = inspect.getsource(run.rollout).lower()
    forbidden = ("barrett", "montgomery", "crt", "repair", "expected", "quotient", "carry", "token", "adapter", "routing")
    assert not any(token in source for token in forbidden)
    assert "lower_tier_parity" not in source and "diagnostic" not in source


def test_resource_policy_never_reduces_science_fidelity() -> None:
    policy = CONFIG["resource_policy"]
    assert policy["soft_guard_multiplier"] == 1.25 and policy["hard_guard_multiplier"] == 1.75
    assert policy["inference_over_300_is_research_only"] is True
    assert policy["fidelity_reduced"] is False
