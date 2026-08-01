from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import jsonschema
import pytest
import torch
import yaml

from experiments.exp_695_cycle_factorized_distillation import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_695_cycle_factorized_distillation.yaml").read_text())


def components(seed: int = 696000):
    state, manifest = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    base = run.base_model(CONFIG, state)
    teacher = run.UntiedCycleTeacher(base, run.visit_schedule(CONFIG))
    student = run.new_student(CONFIG, state, seed).eval()
    return state, manifest, teacher, student


def test_frozen_protocol_roots_work_and_denominators() -> None:
    assert CONFIG["roots"] == {"calibration": 20261377, "formal": 20261378, "confirmations": [20261379, 20261380],
                               "independent_seeds": [20261381, 20261382, 20261383, 20261384, 20261385]}
    assert CONFIG["teacher"]["updates"] == CONFIG["teacher"]["calibration_updates"] == 400
    assert CONFIG["training"]["updates"] == 2400 and CONFIG["training"]["horizon"] == 8
    assert CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"]["transition_cases_per_width"] == 3000
    assert CONFIG["evaluation"]["w128_rollout_cases"] == 300
    assert CONFIG["evaluation"]["w256_sequential_blocks"] == [200, 300, 450, 600]


def test_exp694_exact_two_field_normalization_and_semantic_invariance() -> None:
    metrics_path = ROOT / "evals/results/exp-694-routed-antithetic-quorum/metrics.json"
    run_path = ROOT / "experiments/runs/exp-694-routed-antithetic-quorum/run.json"
    metrics, record = json.loads(metrics_path.read_text()), json.loads(run_path.read_text())
    assert metrics["evaluation"] == {"not_performed": True,
        "reason": "The consumed calibration root failed at the first transition evaluator call before any accuracy or cold-route denominator.",
        "labels_or_oracle_present": False}
    assert record["artifacts"]["checkpoint"] == ".scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"
    assert record["decision"] == metrics["decision"] == "park-exp694-after-consumed-calibration-root-lifecycle-defect"
    assert record["roots"]["20261350"].startswith("opened-consumed") and record["scientific_metrics"] is None
    assert metrics["exact_answer_metrics"] is None and metrics["comparison_interpretable"] is False
    jsonschema.validate(metrics, json.loads((ROOT / "evals/schemas/eval-result.schema.json").read_text()))
    jsonschema.validate(record, json.loads((ROOT / "experiments/schemas/run.schema.json").read_text()))


def test_source_checkpoint_and_top48_mapping_are_authenticated() -> None:
    checkpoint = ROOT / CONFIG["source_checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CONFIG["source_checkpoint_sha256"]
    state, manifest, teacher, student = components()
    assert len(manifest["selected_source_channels_ranked"]) == 48
    assert teacher.exact_copy_identity() and sum(p.numel() for p in student.parameters()) > 0
    assert run.state_hash(state) == manifest["pruned_state_sha256"]


def test_exact_sixteen_outward_inward_visits_and_untied_teacher() -> None:
    schedule = run.visit_schedule(CONFIG)
    assert [row[2] for row in schedule] == [1, 2, 4, 8, 16, 32, 64, 128, 128, 64, 32, 16, 8, 4, 2, 1]
    _, _, teacher, student = components()
    assert isinstance(teacher.slots, torch.nn.ModuleList) and len(teacher.slots) == 16
    assert len({id(slot.gate_weight) for slot in teacher.slots}) == 16
    assert student.schedule == teacher.schedule == schedule


def test_factorization_is_direction_by_dilation_by_hidden_not_kernel_adapter() -> None:
    _, _, _, student = components()
    coefficients = student.coefficients()
    assert coefficients.shape == (2, 8, 48)
    assert student.direction_factors.shape == (2, 4)
    assert student.dilation_factors.shape == (8, 4)
    assert student.hidden_basis.shape == (4, 48)
    source = inspect.getsource(run.CycleFactorizedStudent.step)
    assert "coefficient * torch.tanh(hidden)" in source
    assert "gate_weight +" not in source and "candidate_weight +" not in source


@pytest.mark.parametrize("width", [64, 128, 256])
def test_zero_coefficient_control_masks_only_hidden_access(width: int) -> None:
    _, _, _, student = components(); features = torch.randn(2, 6, width)
    for parameter in student.coefficient_parameters(): parameter.data.normal_()
    control = student.step(features, False); zero = student.step(features, True, "zero")
    assert torch.equal(control[0], zero[0]) and torch.equal(control[1], zero[1])
    assert control[3]["coefficient_calls"] == zero[3]["coefficient_calls"] == 16
    assert len(control[2]) == len(zero[2]) == 16


def test_matched_arms_have_exact_initial_hash_parameters_calls_and_losses() -> None:
    state, _, teacher, primary = components(7); control = run.new_student(CONFIG, state, 7)
    assert run.state_hash(primary.state_dict()) == run.state_hash(control.state_dict())
    assert sum(p.numel() for p in primary.parameters()) == sum(p.numel() for p in control.parameters())
    source = inspect.getsource(run.train_update)
    assert source.count("teacher(inputs, capture=True)") == 1
    assert source.count("student.step(inputs, coefficient_access)") == 1
    for token in ("residue_distillation_weight", "quotient_distillation_weight", "trajectory_distillation_weight", "slot_terms"):
        assert token in source
    assert all(p.requires_grad is False for p in teacher.parameters()) is False


@pytest.mark.parametrize("intervention", ["zero", "direction_swap", "dilation_permutation", "mean_coefficients"])
def test_all_design_specific_interventions_execute(intervention: str) -> None:
    _, _, _, student = components()
    result = run.rollout(student, [17, 19], [23, 29], [61, 59], 64, True, intervention)
    assert result["cases"] == 2 and result["coefficient_calls"] == 64 * 16
    assert set(result["coefficient_diagnostics"]) >= {"l1", "l2", "entropy", "shape"}


def test_student_inference_is_input_only_teacher_free_and_deterministic() -> None:
    _, _, _, student = components(); args = ([17, 19], [23, 29], [61, 59], 64)
    first = run.rollout(student, *args); second = run.rollout(student, *args)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    source = inspect.getsource(run.deployed_route).lower()
    assert "untiedcycleteacher" not in source and "train_teacher" not in source and "teacher_prefix" not in source
    assert '"labels_available_to_timed_route": false' in source
    forbidden = ("barrett", "montgomery", "long_division", "crt", "answer_selection", "output_repair")
    assert not any(token in inspect.getsource(run.CycleFactorizedStudent).lower() for token in forbidden)


def test_resource_admission_is_executed_cold_route_rule() -> None:
    source = inspect.getsource(run.Experiment.calibration)
    assert "2.0 * maximum" in source and "complete_1100_case_route" in source
    assert "variability_trigger" in source and "extra_cold_repetitions" in source
    assert "process_cpu_seconds" in source and "cgroup_cpu_seconds" in source and "peak_rss_bytes" in source
    assert CONFIG["resource_policy"]["maximum_peak_rss_bytes"] == int(6.4 * 1024**3)


@pytest.mark.parametrize("mode,root", [("calibration", 20261378), ("formal", 20261377), ("confirmation", 20261378), ("independent", 20261380)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"): run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_science_requires_full_health_causality_and_noncomparative_rule() -> None:
    source = inspect.getsource(run.Experiment.science)
    for token in ('"healthy_control"', '"complete_updates"', '"causal_erasure"', '"coefficient_noncollapse"',
                  '"teacher_complete"', '"confirmation_threshold"', '"comparison_interpretable"', '"non-comparative-control-health-failure"'):
        assert token in source
    for intervention in ("zero", "direction_swap", "dilation_permutation", "mean_coefficients"): assert intervention in source


def test_canonical_records_absent_before_or_valid_after_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-695-cycle-factorized-distillation/run.json"
    metrics_path = ROOT / "evals/results/exp-695-cycle-factorized-distillation/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if not run_path.exists(): return
    record, metrics = json.loads(run_path.read_text()), json.loads(metrics_path.read_text())
    jsonschema.validate(record, json.loads((ROOT / "experiments/schemas/run.schema.json").read_text()))
    jsonschema.validate(metrics, json.loads((ROOT / "evals/schemas/eval-result.schema.json").read_text()))
    assert record["completion_marker"] == metrics["completion_marker"] == "terminal-persisted"


def test_c5_c7_and_chair_remain_mandatory() -> None:
    limitations = " ".join(run.Experiment.limitations(False))
    assert "C5 remains scientifically unresolved" in limitations
    assert "C7" in limitations and "chair remain mandatory" in limitations
