from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_676_lla_moving_window.model import state_dict_identity
from experiments.exp_676_lla_moving_window.run import Timer, aggregate, sha256
from experiments.exp_677_length_schedule.model import schedule_coverage, schedule_plan, trace_forward_budget


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/exp_677_length_schedule.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def model() -> SharedDilatedConvGRU:
    torch.manual_seed(677)
    return SharedDilatedConvGRU(width=128, hidden=12, cycles=2)


def test_frozen_arms_roots_updates_denominators_and_prohibition() -> None:
    payload = config()
    assert payload["training"]["steps"] == payload["schedule"]["steps"] == 400
    assert set(payload["arms"]) == {"scheduled", "fixed_incumbent_endpoint", "equal_work_shuffled", "frozen_source", "randomized_control"}
    assert payload["data"]["calibration"]["root"] == 20261258
    assert payload["data"]["formal"]["root"] == 20261259
    assert [item["root"] for item in payload["data"]["confirmations"]] == [20261260, 20261261]
    assert payload["evaluation"]["transition_cases"] == {64: 3000, 128: 3000, 256: 3000}
    assert payload["evaluation"]["rollout_cases"] == {64: 300, 128: 300, 256: 300}
    assert payload["source_contract"]["checked_independent_official_private_candidate_submission_promotion"] == "prohibited"


def test_known_step_schedule_is_ordered_and_endpoint_matched() -> None:
    schedule = config()["schedule"]
    assert schedule_plan("scheduled", step=1, total_steps=400, schedule=schedule, seed=1)["width"] == 64
    assert schedule_plan("scheduled", step=100, total_steps=400, schedule=schedule, seed=1)["physical_visits"] == 12
    assert schedule_plan("scheduled", step=101, total_steps=400, schedule=schedule, seed=1)["width"] == 128
    assert schedule_plan("scheduled", step=251, total_steps=400, schedule=schedule, seed=1)["width"] == 256
    assert schedule_plan("scheduled", step=400, total_steps=400, schedule=schedule, seed=1) == {
        "kind": "scheduled", "frozen_schedule_step": 400, "width": 256, "physical_visits": 16, "supervised_endpoint_length": 256,
    }


def test_fixed_and_equal_work_controls_match_declared_work() -> None:
    schedule = config()["schedule"]; coverage = schedule_coverage(schedule, 67773001)
    assert coverage["equal_work_multiset_matches"] is True
    assert coverage["scheduled"]["counts"] == {"64:12:64": 100, "128:14:128": 150, "256:16:256": 150}
    assert coverage["fixed"]["counts"] == {"128:14:128": 400}
    first = [schedule_plan("equal_work_shuffled", step=step, total_steps=400, schedule=schedule, seed=7) for step in range(1, 401)]
    second = [schedule_plan("equal_work_shuffled", step=step, total_steps=400, schedule=schedule, seed=7) for step in range(1, 401)]
    assert first == second
    assert [item["width"] for item in first] != [schedule_plan("scheduled", step=step, total_steps=400, schedule=schedule, seed=7)["width"] for step in range(1, 401)]
    with pytest.raises(ValueError, match="executed"): schedule_plan("scheduled", step=0, total_steps=400, schedule=schedule, seed=1)


def test_training_visit_budget_and_full_formal_output_identity() -> None:
    candidate = model().eval()
    for width, visits in ((64, 12), (128, 14), (256, 16)):
        features = torch.randn(3, 6, width)
        trace = trace_forward_budget(candidate, features, visits)
        ordinary = candidate(features)
        assert len(trace) == visits
        assert torch.equal(trace[-1][0], ordinary[0])
        assert torch.equal(trace[-1][1], ordinary[1])
    with pytest.raises(ValueError, match="visit budget"): trace_forward_budget(candidate, torch.randn(2, 6, 64), 13)


def test_schedule_changes_width_visit_gradient_causally() -> None:
    source = model(); schedule = config()["schedule"]; gradients = {}
    for kind in ("scheduled", "fixed", "equal_work_shuffled"):
        plan = schedule_plan(kind, step=400, total_steps=400, schedule=schedule, seed=5)
        candidate = copy.deepcopy(source); features = torch.randn(2, 6, int(plan["width"]), generator=torch.Generator().manual_seed(55))
        logits, quotient = trace_forward_budget(candidate, features, int(plan["physical_visits"]))[-1]
        loss = logits.square().mean() + quotient.square().mean(); loss.backward(); gradient = candidate.gate_weight.grad
        assert gradient is not None and gradient.abs().sum() > 0
        gradients[kind] = gradient.detach().clone()
    assert not torch.equal(gradients["scheduled"], gradients["fixed"])
    assert not torch.equal(gradients["scheduled"], gradients["equal_work_shuffled"])


def test_checkpoint_schema_reload_separation_and_determinism(tmp_path: Path) -> None:
    torch.use_deterministic_algorithms(True); first = model().eval(); path = tmp_path / "weights.safetensors"
    save_file({key: value.contiguous() for key, value in first.state_dict().items()}, path)
    second = model(); second.load_state_dict(load_file(path)); second.eval(); features = torch.randn(2, 6, 128, generator=torch.Generator().manual_seed(7))
    assert state_dict_identity(first, second)
    assert type(first) is type(second)
    assert sum(p.numel() for p in first.parameters()) == sum(p.numel() for p in second.parameters())
    assert torch.equal(first(features)[0], second(features)[0])
    with torch.no_grad(): second.gate_weight[0, 0, 0].add_(1)
    assert not state_dict_identity(first, second)


def test_fixture_source_manifest_and_library_integrity() -> None:
    payload = config(); contract = payload["source_contract"]
    assert sha256(ROOT / contract["paper_path"]) == contract["paper_sha256"]
    assert sha256(ROOT / "research/manifest.json") == contract["manifest_sha256"]
    assert sha256(ROOT / "research/verification.json") == contract["verification_sha256"]
    assert sha256(ROOT / "experiments/exp_220_width_independent_cellular/model.py") == contract["base_model_sha256"]
    fixture = Path("/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors")
    assert fixture.is_file() and sha256(fixture) == contract["checkpoint_sha256"]
    assert payload["library_decision"]["selected"] == "native-pytorch"


def test_malformed_leakage_and_prohibited_inference_fail_closed() -> None:
    candidate = model()
    with pytest.raises(ValueError, match="finite"): trace_forward_budget(candidate, torch.zeros(2, 5, 64), 12)
    malformed = torch.zeros(2, 6, 64); malformed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"): trace_forward_budget(candidate, malformed, 12)
    source = inspect.getsource(trace_forward_budget).lower() + inspect.getsource(schedule_plan).lower()
    for forbidden in ("expected_answer", "barrett", "montgomery", "crt_recomb", "torch.remainder", "fmod(", "data/checked", "data/private", "submission/", "external call", "confidence"):
        assert forbidden not in source


def test_resource_scopes_are_separate() -> None:
    combined = aggregate([
        {"wall_seconds": 2.0, "process_tree_cpu_seconds": 7.0, "cgroup_cpu_seconds": 8.0, "peak_process_tree_rss_bytes": 100},
        {"wall_seconds": 3.0, "process_tree_cpu_seconds": 9.0, "cgroup_cpu_seconds": 10.0, "peak_process_tree_rss_bytes": 120},
    ])
    assert combined == {"wall_seconds": 5.0, "process_tree_cpu_seconds": 16.0, "cgroup_cpu_seconds": 18.0, "peak_process_tree_rss_bytes": 120}
    assert set(Timer().result()) == {"wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds", "peak_process_tree_rss_bytes"}
