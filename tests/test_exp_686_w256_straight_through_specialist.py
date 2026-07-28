from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from experiments.exp_686_w256_straight_through_specialist import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_686_w256_straight_through_specialist.yaml").read_text())


def test_frozen_protocol_is_exact_w256_matched_eight_call_package() -> None:
    assert CONFIG["roots"] == {"calibration": 20261294, "formal": 20261295, "confirmations": [20261296, 20261297]}
    assert CONFIG["training"]["updates"] == 800
    assert CONFIG["training"]["horizon"] == 8
    assert CONFIG["model"] == {"width": 256, "hidden": 92, "cycles": 2, "parameters": 99824}
    assert CONFIG["evaluation"]["transition_cases_per_arm"] == 3000
    assert CONFIG["evaluation"]["rollout_cases_per_arm"] == 300
    assert tuple(CONFIG["routes"]) == run.ARM_ORDER


def test_exact_source_challenge_checkpoint_and_prior_hashes_authenticate(tmp_path: Path) -> None:
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "smoke", None)
    identity = experiment.identity()
    assert identity["challenge_commit"] == run.CHALLENGE_COMMIT
    assert identity["source_checkpoint_sha256"] == CONFIG["source_checkpoint_sha256"]
    assert identity["source_contract"] == {name: value["sha256"] for name, value in CONFIG["source_contract"].items()}


def test_canonical_records_are_absent_before_or_exact_after_terminal_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-686-w256-straight-through-specialist/run.json"
    metrics_path = ROOT / "evals/results/exp-686-w256-straight-through-specialist/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if not run_path.exists():
        return

    run_record = json.loads(run_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    expected_roots = {
        "20261294": "completed-calibration-resource-reject",
        "20261295": "unopened-admission-blocked",
        "20261296": "unopened-admission-blocked",
        "20261297": "unopened-admission-blocked",
    }
    assert run_record["schema_version"] == "exp686-terminal-v1"
    assert run_record["run_id"] == "exp-686-w256-straight-through-specialist"
    assert run_record["status"] == "completed"
    assert run_record["decision"] == "park-exp686-at-root20261294-resource-admission"
    assert run_record["completion_marker"] == "terminal-persisted"
    assert run_record["roots"] == expected_roots
    assert metrics["schema_version"] == "exp686-terminal-v1"
    assert metrics["run_id"] == run_record["run_id"]
    assert metrics["status"] == "parked-calibration-candidate-runtime-headroom"
    assert metrics["decision"] == run_record["decision"]
    assert metrics["completion_marker"] == "terminal-persisted"
    assert metrics["roots"] == expected_roots


def test_hard_straight_through_is_hard_forward_with_sigmoid_gradient() -> None:
    logits = torch.tensor([[-2.0, 3.0]], requires_grad=True)
    state = run.hard_straight_through(logits)
    assert torch.equal(state.detach(), torch.tensor([[0.0, 1.0]]))
    state.sum().backward()
    expected = torch.sigmoid(logits.detach()) * (1.0 - torch.sigmoid(logits.detach()))
    assert torch.equal(logits.grad, expected)


def test_teacher_prefix_and_state_targets_are_exact_offline_only() -> None:
    left, right, modulus, starts = [13, 9], [7, 11], [17, 19], [3, 5]
    observed = run.teacher_prefix(left, right, modulus, starts)
    expected = []
    for a, b, p, stop in zip(left, right, modulus, starts, strict=True):
        state = 0
        for position in range(stop):
            state = (2 * state + ((a >> (run.WIDTH - 1 - position)) & 1) * b) % p
        expected.append(state)
    assert observed == expected


def test_two_trained_arms_have_only_state_exposure_difference() -> None:
    source = inspect.getsource(run.train_update)
    assert 'arm == "teacher_forced"' in source
    assert "hard_straight_through(logits)" in source
    assert source.count("exp300.transition_loss(") == 1
    assert source.count("optimizer.step()") == 1
    assert set(run.TRAINED_ARMS) == {"learner_straight_through", "teacher_forced"}


def test_source_model_parameter_and_reload_identity() -> None:
    checkpoint = ROOT / CONFIG["source_checkpoint"]
    model = run.new_model(CONFIG, checkpoint)
    assert model.parameter_count() == 99824
    expected = load_file(checkpoint)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())


def test_tensor_native_route_matches_legacy_outputs_and_margins() -> None:
    checkpoint = ROOT / CONFIG["source_checkpoint"]
    model = run.new_model(CONFIG, checkpoint).eval()
    modulus = [(1 << 255) + 765]
    left, right = [123456789], [987654321]
    parity = run.parity_probe(model, (left, right, modulus))
    assert parity["exact_outputs"] is True
    assert parity["exact_minimum_margins"] is True


def test_randomized_control_differs_and_reset_is_exact_source() -> None:
    checkpoint = ROOT / CONFIG["source_checkpoint"]
    source = run.new_model(CONFIG, checkpoint)
    reset = run.new_model(CONFIG, checkpoint)
    randomized = run.new_model(CONFIG, checkpoint, randomized_seed=686900)
    assert all(torch.equal(a, b) for a, b in zip(source.state_dict().values(), reset.state_dict().values(), strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(source.state_dict().values(), randomized.state_dict().values(), strict=True))


@pytest.mark.parametrize("mode,root", [("calibration", 20261295), ("formal", 20261294), ("confirmation", 20261295), ("confirmation", 20261298)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_admission_binds_runtime_expected_value_rss_parity_and_fidelity() -> None:
    source = inspect.getsource(run.Experiment.admission)
    for token in ("candidate_runtime", "expected_value", "memory", "tensor_native_output_parity", "tensor_native_minimum_margin_parity", "no_fidelity_reduction"):
        assert token in source
    assert CONFIG["resource_policy"]["maximum_expected_formal_wall_seconds"] == 21600.0
    assert CONFIG["resource_policy"]["maximum_peak_rss_bytes"] == 6 * 1024**3
    assert CONFIG["gates"]["maximum_candidate_wall_seconds"] == 285.0


def test_candidate_runtime_is_input_only_and_uses_complete_1100_route_contract() -> None:
    source = inspect.getsource(run.candidate_runtime)
    assert "candidate_inputs(root)" in source
    assert "expected" not in source
    assert '"labels_available_to_timed_route": False' in source
    assert '"complete_1100_case_route": len(rows) == 1100' in source


def test_science_requires_complete_85_percent_gate_before_confirmations() -> None:
    assert CONFIG["gates"]["confirmation_admission_rollout"] == 0.85
    source = inspect.getsource(run.Experiment.science)
    assert '"advance-to-confirmations" if passed_confirmation' in source
    assert "confirmation_threshold" in source
    assert "candidate_runtime" in source


def test_static_safety_no_solver_or_answer_selection_in_inference_route() -> None:
    source = inspect.getsource(run.tensor_native_rollout)
    forbidden = ("barrett", "montgomery", "crt", "long_division", "repair", "expected")
    assert not any(token in source.lower() for token in forbidden)
    candidate = inspect.getsource(run.candidate_runtime)
    assert "wide_outputs = tensor_native_rollout" in candidate
    assert "if guess" not in candidate and "target" not in candidate
