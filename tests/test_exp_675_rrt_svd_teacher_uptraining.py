from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_675_rrt_svd_teacher_uptraining.model import RRTVisitLoRA, UntiedVisitTeacher, clone_state
from experiments.exp_675_rrt_svd_teacher_uptraining.run import Timer, aggregate, sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/exp_675_rrt_svd_teacher_uptraining.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def base() -> SharedDilatedConvGRU:
    torch.manual_seed(675)
    return SharedDilatedConvGRU(width=128, hidden=12, cycles=2)


def test_frozen_roots_phases_arms_denominators_and_surfaces() -> None:
    payload = config()
    assert payload["data"]["calibration"]["root"] == 20261250
    assert payload["data"]["formal"]["root"] == 20261251
    assert [item["root"] for item in payload["data"]["confirmations"]] == [20261252, 20261253]
    assert payload["training"]["teacher_steps"] == 100
    assert payload["training"]["joint_steps"] == 200
    assert payload["evaluation"]["rollout_cases"] == {64: 300, 128: 300, 256: 300}
    assert set(payload["arms"]) == {"full_rrt", "exact_zero_visit", "global_lora", "untied_teacher", "shared_continuation", "equal_norm_random_svd", "subspace_shuffled"}
    assert payload["source_contract"]["checked_independent_official_private_candidate_submission_promotion"] == "prohibited"
    assert payload["resource_budget"]["package_expected_seconds"] == 1246
    assert payload["resource_budget"]["package_hard_seconds"] == 2492


def test_exact_copied_teacher_identity_and_forward() -> None:
    source = base().eval(); teacher = UntiedVisitTeacher(source, max_dilation_levels=8).eval()
    features = torch.randn(3, 6, 64)
    assert teacher.exact_copy_identity()
    expected, observed = source(features), teacher(features)
    assert torch.equal(expected[0], observed[0])
    assert torch.equal(expected[1], observed[1])
    assert teacher.slot_count == 16
    assert teacher.active_slots(64) == [0, 1, 2, 3, 4, 5, 13, 12, 11, 10, 9, 8]


def test_svd_factorization_rank_placement_and_optimal_reconstruction() -> None:
    teacher = UntiedVisitTeacher(base(), max_dilation_levels=8)
    with torch.no_grad():
        teacher.gate_weights.add_(torch.randn_like(teacher.gate_weights) * 0.01)
        teacher.candidate_weights.add_(torch.randn_like(teacher.candidate_weights) * 0.01)
    shared = base()
    with torch.no_grad():
        shared.gate_weight.copy_(teacher.gate_weights.mean(0))
        shared.candidate_weight.copy_(teacher.candidate_weights.mean(0))
    rrt = RRTVisitLoRA(shared, rank=8, alpha=8, max_dilation_levels=8)
    report = rrt.initialize_from_teacher(teacher)
    assert rrt.gate_a.shape == (16, 8, (12 + 6) * 3)
    assert rrt.gate_b.shape == (16, 24, 8)
    assert all(item["requested_rank"] == 8 and item["effective_rank"] == 8 for item in report["gate"] + report["candidate"])
    residual = teacher.gate_weights[0] - shared.gate_weight
    approximation = rrt.delta_gate(0)
    _, s, _ = torch.linalg.svd(residual.flatten(1).double(), full_matrices=False)
    expected_error = float(s[8:].square().sum().sqrt())
    assert float((residual - approximation).norm()) == pytest.approx(expected_error, rel=2e-5, abs=1e-7)
    random_control = RRTVisitLoRA(shared, rank=8, alpha=8, max_dilation_levels=8)
    random_control.initialize_equal_norm_random(teacher, seed=1)
    assert (residual - approximation).norm() <= (residual - random_control.delta_gate(0)).norm()


def test_joint_shared_base_and_adapter_gradients() -> None:
    teacher = UntiedVisitTeacher(base())
    with torch.no_grad(): teacher.gate_weights[0].add_(0.01); teacher.candidate_weights[0].sub_(0.01)
    rrt = RRTVisitLoRA(base(), rank=4, alpha=4); rrt.initialize_from_teacher(teacher)
    before = clone_state(rrt)
    output = rrt(torch.randn(2, 6, 64)); loss = output[0].square().mean() + output[1].square().mean(); loss.backward()
    assert rrt.base.gate_weight.grad is not None and rrt.base.gate_weight.grad.abs().sum() > 0
    assert rrt.gate_a.grad is not None and rrt.gate_a.grad.abs().sum() > 0
    assert rrt.gate_b.grad is not None and rrt.gate_b.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(rrt.joint_parameters(), lr=0.01); optimizer.step()
    assert not torch.equal(before["base.gate_weight"], rrt.state_dict()["base.gate_weight"])
    assert not torch.equal(before["gate_a"], rrt.state_dict()["gate_a"])


def test_zero_global_slot_interventions_and_equal_parameter_matching() -> None:
    visit = RRTVisitLoRA(base(), rank=8, alpha=8); visit.zero_adapters()
    global_control = RRTVisitLoRA(base(), rank=128, alpha=128, kind="global"); global_control.zero_adapters()
    reference = base(); reference.load_state_dict(visit.base.state_dict())
    assert visit.adapter_parameter_count() == global_control.adapter_parameter_count()
    features = torch.randn(2, 6, 64)
    assert torch.equal(reference(features)[0], visit(features)[0])
    with torch.no_grad(): visit.gate_a.normal_(); visit.gate_b.normal_(std=0.001); visit.candidate_a.normal_(); visit.candidate_b.normal_(std=0.001)
    ordinary = visit(features)[0]
    assert not torch.equal(ordinary, visit(features, adapter_enabled=False)[0])
    assert not torch.equal(ordinary, visit(features, slot_permutation=tuple(reversed(range(16))))[0])
    with pytest.raises(ValueError, match="bijection"): visit(features, slot_permutation=(0,) * 16)


def test_deterministic_checkpoint_reload_and_separation(tmp_path: Path) -> None:
    torch.use_deterministic_algorithms(True)
    teacher = UntiedVisitTeacher(base()); first = RRTVisitLoRA(base(), rank=8, alpha=8); first.initialize_from_teacher(teacher)
    path = tmp_path / "rrt.safetensors"; save_file({key: value.contiguous() for key, value in first.state_dict().items()}, path)
    second = RRTVisitLoRA(base(), rank=8, alpha=8); second.load_state_dict(load_file(path))
    features = torch.randn(2, 6, 64, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first(features)[0], second(features)[0])
    assert sha256(path) == sha256(path)
    with torch.no_grad(): second.gate_a[0, 0, 0].add_(1)
    assert not torch.equal(first.state_dict()["gate_a"], second.state_dict()["gate_a"])


def test_fixture_hash_library_and_source_map_integrity() -> None:
    payload = config(); paper = ROOT / payload["source_contract"]["paper_path"]
    assert sha256(paper) == payload["source_contract"]["paper_sha256"]
    assert payload["source_contract"]["checkpoint_sha256"] == "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
    assert payload["library_decision"]["selected"] == "torch.linalg.svd"
    assert payload["library_decision"]["peft_installed"] is False


def test_malformed_leakage_and_prohibited_arithmetic_fail_closed() -> None:
    model = RRTVisitLoRA(base(), rank=4, alpha=4)
    with pytest.raises(ValueError, match="finite"): model(torch.zeros(2, 5, 64))
    malformed = torch.zeros(2, 6, 64); malformed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"): model(malformed)
    source = inspect.getsource(RRTVisitLoRA).lower() + inspect.getsource(UntiedVisitTeacher).lower()
    for forbidden in ("expected_answer", "barrett", "montgomery", "crt_recomb", "torch.remainder", "fmod(", "data/checked", "data/private", "submission/", "external call"):
        assert forbidden not in source


def test_resource_scopes_remain_separate() -> None:
    combined = aggregate([
        {"wall_seconds": 2.0, "process_tree_cpu_seconds": 7.0, "cgroup_cpu_seconds": 8.0, "peak_process_tree_rss_bytes": 100},
        {"wall_seconds": 3.0, "process_tree_cpu_seconds": 9.0, "cgroup_cpu_seconds": 10.0, "peak_process_tree_rss_bytes": 120},
    ])
    assert combined == {"wall_seconds": 5.0, "process_tree_cpu_seconds": 16.0, "cgroup_cpu_seconds": 18.0, "peak_process_tree_rss_bytes": 120}
    measured = Timer().result()
    assert set(measured) == {"wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds", "peak_process_tree_rss_bytes"}
