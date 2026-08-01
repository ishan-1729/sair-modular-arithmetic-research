from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_676_lla_moving_window.model import reinjection_receipt, selected_depths, state_dict_identity, trace_forward
from experiments.exp_676_lla_moving_window.run import Timer, aggregate, sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/exp_676_lla_moving_window.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def model() -> SharedDilatedConvGRU:
    torch.manual_seed(676)
    return SharedDilatedConvGRU(width=128, hidden=12, cycles=2)


def test_frozen_window_arms_roots_updates_and_denominators() -> None:
    payload = config()
    assert payload["window"]["size"] == 4
    assert payload["training"]["steps"] == 400
    assert set(payload["arms"]) == {"moving_window", "endpoint_only", "fixed_window", "equal_work_multi_endpoint", "shuffled_window", "frozen_source", "randomized_control"}
    assert payload["data"]["calibration"]["root"] == 20261254
    assert payload["data"]["formal"]["root"] == 20261255
    assert [item["root"] for item in payload["data"]["confirmations"]] == [20261256, 20261257]
    assert payload["evaluation"]["transition_cases"] == {64: 3000, 128: 3000, 256: 3000}
    assert payload["evaluation"]["rollout_cases"] == {64: 300, 128: 300, 256: 300}
    assert payload["source_contract"]["checked_independent_official_private_candidate_submission_promotion"] == "prohibited"


def test_source_faithful_moving_and_matched_control_sampling() -> None:
    assert selected_depths("moving", visits=14, step=1, total_steps=400, window_size=4, seed=1) == (0, 1, 2, 3)
    assert selected_depths("moving", visits=14, step=400, total_steps=400, window_size=4, seed=1) == (10, 11, 12, 13)
    assert selected_depths("endpoint", visits=14, step=10, total_steps=400, window_size=4, seed=1) == (13, 13, 13, 13)
    assert selected_depths("fixed", visits=14, step=10, total_steps=400, window_size=4, seed=1) == (10, 11, 12, 13)
    assert selected_depths("equal_work", visits=14, step=10, total_steps=400, window_size=4, seed=1) == (0, 4, 9, 13)
    first = selected_depths("shuffled", visits=14, step=10, total_steps=400, window_size=4, seed=1)
    assert first == selected_depths("shuffled", visits=14, step=10, total_steps=400, window_size=4, seed=1)
    assert len(first) == len(set(first)) == 4
    with pytest.raises(ValueError, match="window"): selected_depths("moving", visits=3, step=1, total_steps=4, window_size=4, seed=1)


def test_candidate_already_reinjects_legal_input_every_visit() -> None:
    candidate = model()
    for width, calls in ((64, 12), (128, 14), (256, 16)):
        receipt = reinjection_receipt(candidate, width)
        assert receipt["physical_visits"] == calls
        assert receipt["legal_feature_channels"] == 6
        assert receipt["gate_reinjects_every_visit"] is True
        assert receipt["candidate_reinjects_every_visit"] is True


def test_trace_final_is_bit_identical_to_formal_inference() -> None:
    candidate = model().eval()
    for width in (64, 128, 256):
        features = torch.randn(3, 6, width)
        ordinary = candidate(features)
        traced = trace_forward(candidate, features)
        assert len(traced) == len(candidate.dilation_schedule(width))
        assert torch.equal(ordinary[0], traced[-1][0])
        assert torch.equal(ordinary[1], traced[-1][1])
    assert list(candidate.state_dict()) == list(model().state_dict())


def test_window_losses_have_matched_calls_and_distinct_nonzero_gradients() -> None:
    source = model(); features = torch.randn(2, 6, 64)
    gradients = {}
    for kind in ("moving", "endpoint", "fixed", "equal_work", "shuffled"):
        candidate = copy.deepcopy(source); outputs = trace_forward(candidate, features); indices = selected_depths(kind, visits=len(outputs), step=200, total_steps=400, window_size=4, seed=5)
        losses = [outputs[index][0].square().mean() + outputs[index][1].square().mean() for index in indices]
        assert len(losses) == 4
        torch.stack(losses).mean().backward()
        gradient = candidate.gate_weight.grad
        assert gradient is not None and gradient.abs().sum() > 0
        gradients[kind] = gradient.detach().clone()
    assert not torch.equal(gradients["moving"], gradients["endpoint"])
    assert not torch.equal(gradients["moving"], gradients["equal_work"])


def test_checkpoint_schema_reload_separation_and_determinism(tmp_path: Path) -> None:
    torch.use_deterministic_algorithms(True)
    first = model().eval(); path = tmp_path / "weights.safetensors"; save_file({key: value.contiguous() for key, value in first.state_dict().items()}, path)
    second = model(); second.load_state_dict(load_file(path)); second.eval()
    features = torch.randn(2, 6, 64, generator=torch.Generator().manual_seed(7))
    assert state_dict_identity(first, second)
    assert type(first) is type(second)
    assert sum(p.numel() for p in first.parameters()) == sum(p.numel() for p in second.parameters())
    assert torch.equal(first(features)[0], second(features)[0])
    with torch.no_grad(): second.gate_weight[0, 0, 0].add_(1)
    assert not state_dict_identity(first, second)


def test_fixture_source_and_library_integrity() -> None:
    payload = config()
    assert sha256(ROOT / payload["source_contract"]["paper_path"]) == payload["source_contract"]["paper_sha256"]
    assert sha256(ROOT / "experiments/exp_220_width_independent_cellular/model.py") == payload["source_contract"]["base_model_sha256"]
    fixture = ROOT / ".scratch/exp-675-rrt-svd-teacher-uptraining/fixtures/exp609-weights.safetensors"
    assert fixture.is_file() and sha256(fixture) == payload["source_contract"]["checkpoint_sha256"]
    assert payload["library_decision"]["selected"] == "native-pytorch"


def test_malformed_leakage_and_prohibited_inference_fail_closed() -> None:
    candidate = model()
    with pytest.raises(ValueError, match="finite"): trace_forward(candidate, torch.zeros(2, 5, 64))
    malformed = torch.zeros(2, 6, 64); malformed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"): trace_forward(candidate, malformed)
    source = inspect.getsource(trace_forward).lower() + inspect.getsource(selected_depths).lower()
    for forbidden in ("expected_answer", "barrett", "montgomery", "crt_recomb", "torch.remainder", "fmod(", "data/checked", "data/private", "submission/", "external call"):
        assert forbidden not in source


def test_resource_scopes_are_separate() -> None:
    combined = aggregate([
        {"wall_seconds": 2.0, "process_tree_cpu_seconds": 7.0, "cgroup_cpu_seconds": 8.0, "peak_process_tree_rss_bytes": 100},
        {"wall_seconds": 3.0, "process_tree_cpu_seconds": 9.0, "cgroup_cpu_seconds": 10.0, "peak_process_tree_rss_bytes": 120},
    ])
    assert combined == {"wall_seconds": 5.0, "process_tree_cpu_seconds": 16.0, "cgroup_cpu_seconds": 18.0, "peak_process_tree_rss_bytes": 120}
    assert set(Timer().result()) == {"wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds", "peak_process_tree_rss_bytes"}
