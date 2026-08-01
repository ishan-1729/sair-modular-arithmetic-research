from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from experiments.exp_687_compact_distilled_w256_specialist import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_687_compact_distilled_w256_specialist.yaml").read_text())


def test_frozen_protocol_is_full_compact_matched_package() -> None:
    assert CONFIG["roots"] == {"calibration": 20261302, "formal": 20261303, "confirmations": [20261304, 20261305]}
    assert CONFIG["model"] == {"source_hidden": 92, "hidden": 56, "width": 256, "cycles": 2, "parameters": 38588}
    assert CONFIG["training"]["updates"] == 2400
    assert CONFIG["training"]["horizon"] == 8
    assert CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"] == {"transition_cases_per_width": 3000, "rollout_cases_per_width": 300, "widths": [128, 256]}
    assert CONFIG["training"]["residue_distillation_weight"] == 0.10
    assert CONFIG["training"]["quotient_distillation_weight"] == 0.05


def test_exact_source_and_prior_hashes_authenticate(tmp_path: Path) -> None:
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "smoke", None)
    identity = experiment.identity()
    assert identity["challenge_commit"] == run.CHALLENGE_COMMIT
    assert identity["source_checkpoint_sha256"] == CONFIG["source_checkpoint_sha256"]
    assert len(experiment.manifest["selected_source_channels_ranked"]) == 56


def test_canonical_records_are_absent_before_or_exact_after_terminal_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-687-compact-distilled-w256-specialist/run.json"
    metrics_path = ROOT / "evals/results/exp-687-compact-distilled-w256-specialist/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if not run_path.exists():
        return

    run_record = json.loads(run_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    expected_roots = {
        "20261302": "unopened-package-oracle-blocked",
        "20261303": "unopened-package-oracle-blocked",
        "20261304": "unopened-package-oracle-blocked",
        "20261305": "unopened-package-oracle-blocked",
    }
    assert run_record["schema_version"] == "exp687-terminal-v1"
    assert run_record["run_id"] == "exp-687-compact-distilled-w256-specialist"
    assert run_record["status"] == "failed"
    assert run_record["decision"] == "park-exp687-at-package-verification"
    assert run_record["completion_marker"] == "terminal-persisted"
    assert run_record["roots"] == expected_roots
    assert metrics["schema_version"] == "exp687-terminal-v1"
    assert metrics["run_id"] == run_record["run_id"]
    assert metrics["status"] == "parked-package-verification"
    assert metrics["decision"] == run_record["decision"]
    assert metrics["roots"] == {root: "unopened" for root in expected_roots}


def test_channel_map_is_answer_independent_deterministic_and_tie_broken() -> None:
    source = load_file(ROOT / CONFIG["source_checkpoint"])
    first, terms = run.channel_importance(source)
    second, terms_again = run.channel_importance(copy.deepcopy(source))
    assert torch.equal(first, second)
    assert terms == terms_again
    assert sorted(first.tolist()) == list(range(92))
    assert set(terms) == {
        "stem", "state_norm", "gate_rows", "gate_columns", "candidate_rows",
        "candidate_columns", "output_norm", "local_head", "quotient_hidden_rows",
        "quotient_pooled_columns", "quotient_output_columns", "total",
    }
    source_text = inspect.getsource(run.channel_importance).lower()
    assert not any(token in source_text for token in ("answer", "target", "label", "benchmark", "correct"))


def test_every_source_tensor_is_coherently_sliced_and_loadable() -> None:
    source = load_file(ROOT / CONFIG["source_checkpoint"])
    selected = run.channel_importance(source)[0][:56]
    state, manifest = run.slice_state(source, selected)
    assert set(state) == set(source) == set(manifest)
    model = run.new_student(CONFIG, state)
    assert model.parameter_count() == 38588
    assert torch.equal(state["stem.weight"], source["stem.weight"].index_select(0, selected))
    assert torch.equal(state["local_head.weight"], source["local_head.weight"].index_select(1, selected))
    assert state["gate_weight"].shape == (112, 62, 3)
    assert state["quotient_head.0.weight"].shape == (56, 112)


def test_two_trained_arms_start_from_identical_pruned_bytes() -> None:
    pruned, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    distilled = run.new_student(CONFIG, pruned)
    exact_only = run.new_student(CONFIG, pruned)
    assert run.state_hash(distilled.state_dict()) == run.state_hash(exact_only.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(distilled.state_dict().values(), exact_only.state_dict().values(), strict=True))


def test_distillation_is_only_trained_arm_variable_and_teacher_work_is_unconditional() -> None:
    source = inspect.getsource(run.train_update)
    assert "with torch.no_grad(): teacher_logits" in source
    assert source.count("teacher_model(inputs.detach())") == 1
    assert source.count("exp300.transition_loss(") == 1
    assert "if distill else 0.0" in source
    assert "student_calls\": horizon" in source
    assert "teacher_calls\": horizon" in source
    assert run.TRAINED_ARMS == ("distilled", "exact_only")


def test_distillation_terms_have_nonzero_student_gradient() -> None:
    torch.manual_seed(7)
    student = torch.randn(2, 16, requires_grad=True)
    teacher = torch.randn(2, 16)
    residue = torch.nn.functional.mse_loss(student, teacher) / teacher.square().mean().clamp_min(1e-8)
    residue.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
    student_q = torch.randn(2, 3, requires_grad=True)
    teacher_q = torch.randn(2, 3)
    quotient = torch.nn.functional.kl_div(torch.log_softmax(student_q, 1), torch.softmax(teacher_q, 1), reduction="batchmean")
    quotient.backward()
    assert student_q.grad is not None and float(student_q.grad.abs().sum()) > 0


def test_hard_straight_through_is_hard_forward_with_sigmoid_gradient() -> None:
    logits = torch.tensor([[-2.0, 3.0]], requires_grad=True)
    state = run.hard_straight_through(logits)
    assert torch.equal(state.detach(), torch.tensor([[0.0, 1.0]]))
    state.sum().backward()
    expected = torch.sigmoid(logits.detach()) * (1.0 - torch.sigmoid(logits.detach()))
    assert torch.equal(logits.grad, expected)


def test_randomized_control_differs_and_reset_is_exact_pruned() -> None:
    pruned, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    frozen = run.new_student(CONFIG, pruned)
    reset = run.new_student(CONFIG, pruned)
    randomized = run.new_student(CONFIG, None, 687900)
    assert run.state_hash(frozen.state_dict()) == run.state_hash(reset.state_dict())
    assert run.state_hash(randomized.state_dict()) != run.state_hash(reset.state_dict())


@pytest.mark.parametrize("mode,root", [("calibration", 20261303), ("formal", 20261302), ("confirmation", 20261303), ("confirmation", 20261306)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_candidate_runtime_is_input_only_complete_route_and_cold_reload() -> None:
    source = inspect.getsource(run.candidate_runtime_once)
    assert "candidate_inputs(root)" in source
    assert "new_student(config,load_file(checkpoint))" in source
    assert '"labels_available_to_timed_route":False' in source
    assert '"complete_1100_case_route":len(rows)==1100' in source
    assert not any(token in source for token in ("expected_answer", "target", "if guess"))


def test_admission_binds_three_repeats_wall_cpu_cgroup_rss_and_full_fidelity() -> None:
    source = inspect.getsource(run.Experiment.admission)
    for token in ("candidate_three_complete_cold", "candidate_runtime", "expected_value", "memory", "no_fidelity_reduction"):
        assert token in source
    timer_keys = run.Timer().finish()
    assert {"wall_seconds", "process_cpu_seconds", "cgroup_cpu_seconds", "peak_rss_bytes"} <= set(timer_keys)
    assert CONFIG["resource_policy"]["package_oracle_hard_wall_seconds"] == 1333.702125
    assert CONFIG["resource_policy"]["maximum_expected_formal_wall_seconds"] == 21600.0


def test_science_requires_full_85_percent_gate_before_confirmations() -> None:
    assert CONFIG["gates"]["confirmation_admission_rollout"] == 0.85
    source = inspect.getsource(run.Experiment.science)
    assert '"advance-to-confirmations" if confirmation' in source
    assert '"confirmation_threshold"' in source
    assert '"gain_over_exact_only"' in source
    assert '"w128_retention"' in source


def test_formal_inference_has_no_teacher_solver_or_answer_selection() -> None:
    source = inspect.getsource(run.rollout).lower()
    forbidden = ("teacher_model", "barrett", "montgomery", "crt", "long_division", "repair", "expected")
    assert not any(token in source for token in forbidden)
    candidate = inspect.getsource(run.candidate_runtime_once)
    assert "rollout(model" in candidate
    assert "teacher_model" not in candidate
