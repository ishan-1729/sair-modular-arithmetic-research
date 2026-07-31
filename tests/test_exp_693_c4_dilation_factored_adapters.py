from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.exp_693_c4_dilation_factored_adapters import run
from experiments.exp_693_c4_dilation_factored_adapters.model import DilationDirectionFactoredStudent, UntiedPhysicalSlotTeacher


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_693_c4_dilation_factored_adapters.yaml").read_text())


def components(seed: int = 693900):
    state, manifest = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    teacher = UntiedPhysicalSlotTeacher(run.base_model(CONFIG, state), CONFIG["model"]["outward_dilations"])
    student, report = run.new_student(CONFIG, state, teacher, seed)
    return state, manifest, teacher, student.eval(), report


def test_frozen_protocol_roots_and_full_work() -> None:
    assert CONFIG["roots"] == {"calibration": 20261341, "formal": 20261342,
                               "confirmations": [20261343, 20261344],
                               "independent_seeds": [20261345, 20261346, 20261347, 20261348, 20261349]}
    assert CONFIG["model"]["hidden"] == 56 and CONFIG["model"]["cycles"] == 1 and CONFIG["model"]["rank"] == 4
    assert CONFIG["teacher"]["updates"] == 400 and CONFIG["training"]["updates"] == 2400
    assert CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"]["w256_sequential_blocks"] == [300, 450, 600]


def test_source_top56_teacher_copy_and_one_cycle_topology() -> None:
    state, manifest, teacher, student, _ = components()
    assert len(manifest["selected_source_channels_ranked"]) == 56
    assert teacher.exact_copy_identity() and len(teacher.dilations) == 8
    assert student.base.dilation_schedule(256) == CONFIG["model"]["outward_dilations"]
    assert all(torch.equal(run.base_model(CONFIG, state).state_dict()[name], value) for name, value in state.items())


def test_native_rank4_svd_centroid_and_high_dilation_banks() -> None:
    _, _, teacher, student, report = components()
    assert report["library"] == "torch.linalg.svd" and report["rank"] == 4
    assert student.high_dilations == (8, 16, 32, 64, 128)
    assert student.gate_dilation_a.shape[0] == student.candidate_dilation_a.shape[0] == 5
    assert all(row["effective_rank"] == 4 for kind in ("gate", "candidate") for row in report[kind])
    assert torch.equal(student.base.gate_weight, teacher.gate_weights.mean(0))
    assert torch.equal(student.base.candidate_weight, teacher.candidate_weights.mean(0))
    assert "torch.linalg.svd" in inspect.getsource(__import__("experiments.exp_693_c4_dilation_factored_adapters.model", fromlist=["_svd"])._svd)


def test_reconstruction_and_equal_norm_random_report() -> None:
    _, _, teacher, _, report = components()
    reconstruction = report["reconstruction"]
    assert set(reconstruction) == {"mean_rank4_relative_error", "mean_equal_norm_random_relative_error"}
    assert all(torch.isfinite(torch.tensor(value)) for value in reconstruction.values())
    assert reconstruction["mean_rank4_relative_error"] <= reconstruction["mean_equal_norm_random_relative_error"] + 1e-6


@pytest.mark.parametrize("width", [64, 128, 256])
def test_primary_control_match_when_adapter_delta_is_zero(width: int) -> None:
    _, _, _, student, _ = components(); inputs = torch.randn(2, 6, width)
    for parameter in student.adapter_parameters(): parameter.data.zero_()
    primary = student.step(inputs, True); control = student.step(inputs, False); base = student.base(inputs)
    assert torch.equal(primary[0], control[0]) and torch.equal(primary[1], control[1])
    assert torch.equal(primary[0], base[0]) and torch.equal(primary[1], base[1])


def test_control_computes_adapter_then_masks_only_access() -> None:
    _, _, _, student, _ = components(); student.gate_direction_b.data.normal_(); student.candidate_direction_b.data.normal_()
    inputs = torch.randn(2, 6, 64); primary = student.step(inputs, True); control = student.step(inputs, False)
    assert not torch.equal(primary[0], control[0])
    assert primary[2]["adapter_calls"] == control[2]["adapter_calls"] == len(student.base.dilation_schedule(64))
    source = inspect.getsource(DilationDirectionFactoredStudent.step)
    assert "gate_delta * float(adapter_access)" in source and "candidate_delta * float(adapter_access)" in source


def test_reconstruction_supplies_adapter_gradients_to_both_arms() -> None:
    _, _, teacher, primary, _ = components(1); _, _, _, control, _ = components(1)
    inputs = torch.randn(2, 6, 64)
    for model, access in ((primary.train(), True), (control.train(), False)):
        logits, _, _ = model.step(inputs, access)
        (logits.square().mean() + model.reconstruction_loss(teacher)).backward()
        assert sum(float(p.grad.abs().sum()) for p in model.adapter_parameters() if p.grad is not None) > 0


@pytest.mark.parametrize("intervention", ["zero", "permute_dilations", "reverse_directions", "random_subspace", "randomize_adapters"])
def test_declared_interventions_execute(intervention: str) -> None:
    _, _, _, student, _ = components()
    result = run.rollout(student, [17, 19], [23, 29], [61, 59], 64, True, intervention)
    assert result["cases"] == 2 and result["adapter_diagnostics"]["adapter_calls"] > 0


def test_rollout_deterministic_and_cross_call_isolated() -> None:
    _, _, _, student, _ = components(); args = ([17, 19], [23, 29], [61, 59], 64)
    first = run.rollout(student, *args); _ = run.rollout(student, [7], [11], [53], 64); second = run.rollout(student, *args)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert first["adapter_diagnostics"] == second["adapter_diagnostics"]


def test_deployed_route_excludes_teacher_svd_labels_and_preserves_lower_tiers() -> None:
    source = inspect.getsource(run.deployed_route)
    assert "incumbent.predict_digits_batch(lower)" in source
    assert '"operand_swap_route": False' in source and '"labels_available_to_timed_route": False' in source
    assert "train_teacher" not in source and "torch.linalg.svd" not in source and "teacher_prefix" not in source


def test_learned_inference_has_no_programmed_arithmetic() -> None:
    source = inspect.getsource(DilationDirectionFactoredStudent.step).lower()
    forbidden = ("barrett", "montgomery", "long_division", "crt", "divmod", "modulus", "target", "expected", "answer")
    assert not any(token in source for token in forbidden)


def test_exact_resource_admission_rule_and_dynamic_budgets() -> None:
    source = inspect.getsource(run.Experiment.calibration)
    assert "2.0 * maximum" in source and "complete_1100_case_route" in source
    assert "soft_guard_multiplier" in source and "hard_guard_multiplier" in source
    assert CONFIG["resource_policy"]["soft_guard_multiplier"] == 1.25
    assert CONFIG["resource_policy"]["hard_guard_multiplier"] == 1.75
    assert CONFIG["resource_policy"]["maximum_peak_rss_bytes"] == int(6.4 * 1024**3)


@pytest.mark.parametrize("mode,root", [("calibration", 20261342), ("formal", 20261341),
                                        ("confirmation", 20261342), ("independent", 20261344)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_science_requires_full_denominators_matching_and_causality() -> None:
    source = inspect.getsource(run.Experiment.science)
    for token in ('"complete_updates"', '"healthy_control"', '"paired_gain"', '"causal_erasure"',
                  '"adapter_noncollapse"', '"teacher_complete"', '"confirmation_threshold"', '"w128_retention"'):
        assert token in source
    for intervention in ("zero", "permute_dilations", "reverse_directions", "random_subspace", "randomize_adapters"):
        assert intervention in source
    assert "randomized_full_weights" in source and "anytime_hoeffding_interval" in source


def test_canonical_records_absent_before_or_schema_valid_after_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-693-c4-dilation-factored-adapters/run.json"
    metrics_path = ROOT / "evals/results/exp-693-c4-dilation-factored-adapters/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if run_path.exists():
        record = json.loads(run_path.read_text()); metrics = json.loads(metrics_path.read_text())
        assert record["completion_marker"] == metrics["completion_marker"] == "terminal-persisted"


def test_tournament_contract_keeps_later_entrants_required() -> None:
    limitations = " ".join(run.Experiment.limitations(False))
    assert "C4 is independently required" in limitations and "cannot cancel or weaken C5-C7" in limitations
