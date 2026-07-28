from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_676_lla_moving_window.run import Timer, aggregate
from experiments.exp_678_target_only_continuation.model import (
    mask_weights,
    masked_mean,
    state_dict_identity,
    trace_forward,
)
from experiments.exp_678_target_only_continuation.run import sha256, state_dict_digest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/exp_678_target_only_continuation.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def small_model(seed: int = 678) -> SharedDilatedConvGRU:
    torch.manual_seed(seed)
    return SharedDilatedConvGRU(width=128, hidden=12, cycles=2)


def test_frozen_arms_roots_updates_denominators_and_exclusions() -> None:
    payload = config()
    assert payload["training"]["steps"] == 400
    assert payload["training"]["calibration_steps"] == 4
    assert payload["training"]["mode_schedule"] == ["transition64", "recovery64", "recovery128", "recovery128"]
    assert set(payload["arms"]) == {"target_only", "mixed_trajectory", "endpoint_weight", "shuffled_mask", "frozen_source", "randomized_control"}
    assert payload["data"]["calibration"]["root"] == 20261262
    assert payload["data"]["formal"]["root"] == 20261263
    assert [item["root"] for item in payload["data"]["confirmations"]] == [20261264, 20261265]
    assert payload["evaluation"]["rollout_cases"] == {64: 300, 128: 300, 256: 300}
    assert payload["source_contract"]["checked_independent_official_private_candidate_submission_promotion"] == "prohibited"
    assert "two-trillion-token-scale" in payload["source_mapping"]["exclusions"]


def test_masks_have_exact_coverage_and_equal_final_target_influence() -> None:
    target = mask_weights("target_only", positions=8, step=1, seed=7)
    mixed = mask_weights("mixed", positions=8, step=1, seed=7)
    endpoint = mask_weights("endpoint_weight", positions=8, step=1, seed=7)
    assert target == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0)
    assert mixed == (1.0,) * 8
    assert endpoint == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.0)
    losses = [torch.tensor(float(index + 1)) for index in range(8)]
    _, target_receipt = masked_mean(losses, target)
    _, endpoint_receipt = masked_mean(losses, endpoint)
    assert target_receipt["ce_tensor_calls"] == endpoint_receipt["ce_tensor_calls"] == 8
    assert target_receipt["effective_supervised_positions"] == 1
    assert endpoint_receipt["effective_supervised_positions"] == 8
    assert target_receipt["final_target_reduced_coefficient"] == endpoint_receipt["final_target_reduced_coefficient"] == 1.0


def test_shuffled_mask_is_deterministic_and_position_dependent() -> None:
    first = mask_weights("shuffled", positions=8, step=41, seed=678)
    assert first == mask_weights("shuffled", positions=8, step=41, seed=678)
    assert sum(weight != 0 for weight in first) == 1
    assert sum(first) == 8.0
    assert len({mask_weights("shuffled", positions=8, step=step, seed=678) for step in range(1, 40)}) > 1
    with pytest.raises(ValueError, match="positions"):
        mask_weights("target_only", positions=0, step=1, seed=1)


def test_masked_losses_have_matched_tensor_work_and_distinct_gradients() -> None:
    base = torch.tensor([0.3, -0.7], requires_grad=True)
    gradients = {}
    for kind in ("target_only", "mixed", "endpoint_weight", "shuffled"):
        value = base.detach().clone().requires_grad_(True)
        losses = [((index + 1) * value).square().mean() for index in range(8)]
        reduced, receipt = masked_mean(losses, mask_weights(kind, positions=8, step=9, seed=4))
        reduced.backward()
        assert receipt["ce_tensor_calls"] == 8
        assert value.grad is not None and value.grad.abs().sum() > 0
        gradients[kind] = value.grad.detach().clone()
    assert not torch.equal(gradients["target_only"], gradients["mixed"])
    assert not torch.equal(gradients["target_only"], gradients["endpoint_weight"])


def test_formal_inference_graph_and_outputs_are_exactly_unchanged() -> None:
    candidate = small_model().eval()
    schema = list(candidate.state_dict())
    for width in (64, 128, 256):
        features = torch.randn(3, 6, width, generator=torch.Generator().manual_seed(width))
        ordinary = candidate(features)
        traced = trace_forward(candidate, features)
        assert len(traced) == len(candidate.dilation_schedule(width))
        assert torch.equal(ordinary[0], traced[-1][0])
        assert torch.equal(ordinary[1], traced[-1][1])
    assert list(candidate.state_dict()) == schema
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 2464


def test_checkpoint_reload_separation_determinism_and_weight_dependence(tmp_path: Path) -> None:
    torch.use_deterministic_algorithms(True)
    source = small_model().eval()
    path = tmp_path / "weights.safetensors"
    save_file({key: value.contiguous() for key, value in source.state_dict().items()}, path)
    reloaded = small_model(1)
    reloaded.load_state_dict(load_file(path))
    reloaded.eval()
    features = torch.randn(2, 6, 64, generator=torch.Generator().manual_seed(7))
    assert state_dict_identity(source, reloaded)
    assert state_dict_digest(source.state_dict()) == state_dict_digest(reloaded.state_dict())
    assert torch.equal(source(features)[0], reloaded(features)[0])
    randomized = small_model(679).eval()
    assert not state_dict_identity(source, randomized)
    assert not torch.equal(source(features)[0], randomized(features)[0])


def test_authenticated_fixture_paper_manifest_model_and_challenge() -> None:
    payload = config()["source_contract"]
    checkpoint = ROOT / ".scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"
    assert checkpoint.is_file() and sha256(checkpoint) == payload["checkpoint_sha256"]
    assert sha256(ROOT / payload["paper_path"]) == payload["paper_sha256"]
    assert sha256(ROOT / "research/manifest.json") == payload["research_manifest_sha256"]
    assert sha256(ROOT / "experiments/exp_220_width_independent_cellular/model.py") == payload["base_model_sha256"]
    git_pointer = (ROOT / "sources/github/modular-arithmetic-challenge/.git").read_text(encoding="utf-8")
    assert "gitdir:" in git_pointer


def test_malformed_inputs_leakage_and_prohibited_formal_inference_fail_closed() -> None:
    candidate = small_model()
    with pytest.raises(ValueError, match="finite"):
        trace_forward(candidate, torch.zeros(2, 5, 64))
    malformed = torch.zeros(2, 6, 64)
    malformed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        trace_forward(candidate, malformed)
    source = inspect.getsource(trace_forward).lower() + inspect.getsource(mask_weights).lower() + inspect.getsource(masked_mean).lower()
    for forbidden in ("expected_answer", "barrett", "montgomery", "crt_recomb", "torch.remainder", "fmod(", "data/checked", "data/private", "submission/", "external call"):
        assert forbidden not in source


def test_resource_scopes_remain_distinct() -> None:
    combined = aggregate([
        {"wall_seconds": 2.0, "process_tree_cpu_seconds": 7.0, "cgroup_cpu_seconds": 8.0, "peak_process_tree_rss_bytes": 100},
        {"wall_seconds": 3.0, "process_tree_cpu_seconds": 9.0, "cgroup_cpu_seconds": 10.0, "peak_process_tree_rss_bytes": 120},
    ])
    assert combined == {"wall_seconds": 5.0, "process_tree_cpu_seconds": 16.0, "cgroup_cpu_seconds": 18.0, "peak_process_tree_rss_bytes": 120}
    assert set(Timer().result()) == {"wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds", "peak_process_tree_rss_bytes"}
