from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from experiments.exp_687_compact_distilled_w256_specialist import run as exp687
from experiments.exp_688_compact_hidden_trajectory_distillation import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_688_compact_hidden_trajectory_distillation.yaml").read_text())


def test_frozen_protocol_is_matched_hidden_trajectory_package() -> None:
    assert CONFIG["roots"] == {"calibration": 20261306, "formal": 20261307, "confirmations": [20261308, 20261309]}
    assert CONFIG["model"] == {"source_hidden": 92, "hidden": 56, "width": 256, "cycles": 2, "parameters": 38588}
    assert CONFIG["training"]["updates"] == 2400
    assert CONFIG["training"]["horizon"] == 8
    assert CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"] == {"transition_cases_per_width": 3000, "rollout_cases_per_width": 300, "widths": [128, 256]}
    assert CONFIG["routes"]["hidden_aligned"]["hidden_distillation_weight"] == 0.05
    assert CONFIG["routes"]["logit_only"]["hidden_distillation_weight"] == 0.0
    for key in ("residue_distillation_weight", "quotient_distillation_weight"):
        assert CONFIG["routes"]["hidden_aligned"][key] == CONFIG["routes"]["logit_only"][key]


def test_canonical_records_are_absent_before_or_schema_valid_after_terminal_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-688-compact-hidden-trajectory-distillation/run.json"
    metrics_path = ROOT / "evals/results/exp-688-compact-hidden-trajectory-distillation/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if not run_path.exists():
        return
    run_record, metrics = json.loads(run_path.read_text()), json.loads(metrics_path.read_text())
    allowed_decisions = {
        "park-exp688-at-package-verification", "park-exp688-at-root20261306-resource-admission",
        "reject-exp688-compact-hidden-trajectory-distillation", "reject-exp688-confirmation",
        "advance-exp688-hidden-trajectory-distillation",
    }
    expected_roots = {"20261306", "20261307", "20261308", "20261309"}
    assert run_record["schema_version"] == metrics["schema_version"] == "exp688-terminal-v1"
    assert run_record["run_id"] == metrics["run_id"] == "exp-688-compact-hidden-trajectory-distillation"
    assert run_record["decision"] == metrics["decision"] in allowed_decisions
    assert run_record["completion_marker"] == metrics["completion_marker"] == "terminal-persisted"
    assert set(run_record["roots"]) == set(metrics["roots"]) == expected_roots


def test_exp687_map_and_pruned_initialization_are_byte_exact() -> None:
    checkpoint = ROOT / CONFIG["source_checkpoint"]
    ours, ours_manifest = run.prepare_pruned(CONFIG, checkpoint)
    prior_config = yaml.safe_load((ROOT / "configs/exp_687_compact_distilled_w256_specialist.yaml").read_text())
    prior, prior_manifest = exp687.prepare_pruned(prior_config, checkpoint)
    assert ours_manifest["selected_source_channels_ranked"] == prior_manifest["selected_source_channels_ranked"]
    assert ours_manifest["pruned_state_sha256"] == prior_manifest["pruned_state_sha256"]
    assert run.state_hash(ours) == exp687.state_hash(prior)
    assert all(torch.equal(ours[name], prior[name]) for name in ours)


@pytest.mark.parametrize("width", [64, 128, 256])
def test_hidden_capture_is_source_exact_and_has_selected_shape(width: int) -> None:
    checkpoint = ROOT / CONFIG["source_checkpoint"]
    teacher = run.new_teacher(CONFIG, checkpoint); teacher.width = width
    inputs = torch.randn(2, 6, width)
    expected_logits, expected_quotient = teacher(inputs)
    logits, quotient, hidden = run.forward_with_final_hidden(teacher, inputs)
    assert torch.equal(logits, expected_logits)
    assert torch.equal(quotient, expected_quotient)
    selected = run.channel_importance(load_file(checkpoint))[0][:56]
    assert hidden.index_select(1, selected).shape == (2, 56, width)


def test_normalized_hidden_alignment_has_nonzero_parameter_gradient() -> None:
    pruned, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    student = run.new_student(CONFIG, pruned); inputs = torch.randn(2, 6, 64)
    _, _, student_hidden = run.forward_with_final_hidden(student, inputs)
    teacher_selected = torch.randn_like(student_hidden)
    loss = torch.nn.functional.mse_loss(run.normalized_hidden(student_hidden), run.normalized_hidden(teacher_selected))
    gradients = torch.autograd.grad(loss, tuple(student.parameters()), allow_unused=True)
    assert sum(float(value.abs().sum()) for value in gradients if value is not None) > 0


def test_both_arms_compute_identical_teacher_hidden_and_logit_work() -> None:
    source = inspect.getsource(run.train_update)
    assert source.count("forward_with_final_hidden(teacher_model, inputs.detach())") == 1
    assert source.count("forward_with_final_hidden(student, inputs)") == 1
    assert source.count("hidden_alignment =") == 1
    assert source.count("exp300.transition_loss(") == 1
    assert "hidden_weight * hidden_alignment" in source
    assert "residue_distillation_weight" in source and "quotient_distillation_weight" in source
    assert run.TRAINED_ARMS == ("hidden_aligned", "logit_only")


def test_student_only_inference_has_no_teacher_capture_or_answer_selection() -> None:
    source = inspect.getsource(run.rollout).lower()
    forbidden = ("teacher_model", "forward_with_final_hidden", "barrett", "montgomery", "crt", "long_division", "repair", "expected")
    assert not any(token in source for token in forbidden)
    candidate = inspect.getsource(exp687.candidate_runtime_once)
    assert "rollout(model" in candidate
    assert "teacher_model" not in candidate


def test_two_trained_arms_start_from_identical_pruned_bytes() -> None:
    pruned, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    primary = run.new_student(CONFIG, pruned); control = run.new_student(CONFIG, pruned)
    assert run.state_hash(primary.state_dict()) == run.state_hash(control.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(primary.state_dict().values(), control.state_dict().values(), strict=True))


def test_randomized_control_differs_and_reset_is_exact_pruned() -> None:
    pruned, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    frozen = run.new_student(CONFIG, pruned); reset = run.new_student(CONFIG, pruned); randomized = run.new_student(CONFIG, None, 688900)
    assert run.state_hash(frozen.state_dict()) == run.state_hash(reset.state_dict())
    assert run.state_hash(randomized.state_dict()) != run.state_hash(reset.state_dict())


@pytest.mark.parametrize("mode,root", [("calibration", 20261307), ("formal", 20261306), ("confirmation", 20261307), ("confirmation", 20261310)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_candidate_runtime_is_input_only_complete_route_and_cold_reload() -> None:
    source = inspect.getsource(exp687.candidate_runtime_once)
    assert "candidate_inputs(root)" in source
    assert "new_student(config,load_file(checkpoint))" in source
    assert '"labels_available_to_timed_route":False' in source
    assert '"complete_1100_case_route":len(rows)==1100' in source
    assert not any(token in source for token in ("expected_answer", "target", "if guess"))


def test_admission_binds_repeats_resources_fidelity_and_headroom() -> None:
    source = inspect.getsource(run.Experiment.admission)
    for token in ("candidate_three_complete_cold", "candidate_runtime", "expected_value", "memory", "no_fidelity_reduction"):
        assert token in source
    assert CONFIG["resource_policy"]["candidate_headroom_floor"] == 1.25
    assert CONFIG["resource_policy"]["package_oracle_hard_wall_seconds"] == 1333.702125
    assert CONFIG["resource_policy"]["maximum_expected_formal_wall_seconds"] == 21600.0


def test_science_requires_mechanism_and_85_percent_before_confirmations() -> None:
    source = inspect.getsource(run.Experiment.science)
    assert '"gain_over_logit_only"' in source
    assert '"nonzero_hidden_gradient"' in source
    assert '"confirmation_threshold"' in source
    assert '"advance-to-confirmations" if confirmation' in source
    assert CONFIG["gates"]["confirmation_admission_rollout"] == 0.85
