from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_676_lla_moving_window.run import sha256
from experiments.exp_679_capacity_recycling.model import (
    CapacityRecycledConvGRU,
    evenly_spaced_indices,
    exact_embedding_receipt,
    scaled_call_count,
    select_resource_route,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/exp_679_capacity_recycling.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def source(seed: int = 679) -> SharedDilatedConvGRU:
    torch.manual_seed(seed)
    return SharedDilatedConvGRU(width=128, hidden=92, cycles=2)


def test_frozen_grid_arms_roots_updates_gates_and_exclusions() -> None:
    cfg = config()
    assert cfg["grid"] == {
        "reference_w128_calls": 14,
        "reduced_calls_w128": [8, 10],
        "total_hidden_widths": [104, 112],
        "route_rule": "evenly-spaced-source-order-indices-including-first-and-last",
        "width_rule": "total hidden width includes the exact 92-channel source subspace",
        "frozen_before_profiling": True,
    }
    assert set(cfg["arms"]) == {"capacity_recycled", "reduced_only", "wider_full", "standard_full", "frozen_source", "randomized_wider"}
    assert cfg["training"]["steps"] == 400 and cfg["training"]["calibration_steps"] == 4
    assert cfg["data"]["calibration"]["root"] == 20261266 and cfg["data"]["formal"]["root"] == 20261267
    assert [item["root"] for item in cfg["data"]["confirmations"]] == [20261268, 20261269]
    assert cfg["evaluation"]["rollout_cases"] == {64: 300, 128: 300, 256: 300}
    assert cfg["source_mapping"]["exclusions"] == ["M043.a-fixed-R2-schedule", "M043.c-MoE", "M009.b-learned-routing", "answer-dependent-depth", "microbatch-scaling-claim"]
    assert cfg["selection"]["label_free"] is True


def test_route_is_deterministic_target_independent_and_preserves_endpoints() -> None:
    assert evenly_spaced_indices(14, 8) == (0, 2, 4, 6, 7, 9, 11, 13)
    assert evenly_spaced_indices(14, 10) == (0, 1, 3, 4, 6, 7, 9, 10, 12, 13)
    assert scaled_call_count(12, 8) == 7 and scaled_call_count(16, 8) == 9
    with pytest.raises(ValueError): evenly_spaced_indices(4, 5)


def test_full_call_widening_is_bitwise_exact_embedded() -> None:
    base = source(); wider = CapacityRecycledConvGRU(base, total_hidden=112, calls_at_w128=None)
    features = torch.randn(5, 6, 128, generator=torch.Generator().manual_seed(1))
    receipt = exact_embedding_receipt(base, wider, features)
    assert receipt["bit_logits_equal"] and receipt["quotient_logits_equal"]
    assert receipt["maximum_absolute_delta"] == 0.0
    assert wider.accounting(128)["extra_hidden"] == 20
    assert wider.accounting(128)["recurrent_calls"] == 14


def test_reduced_wider_old_channel_route_is_exact_and_reinjects_inputs() -> None:
    original = source(); reference = CapacityRecycledConvGRU(original, total_hidden=92, calls_at_w128=8)
    wider = CapacityRecycledConvGRU(source(), total_hidden=112, calls_at_w128=8)
    wider.source.load_state_dict(original.state_dict())
    features = torch.randn(3, 6, 128, generator=torch.Generator().manual_seed(2))
    with torch.no_grad(): expected = reference(features); observed = wider(features)
    assert torch.equal(expected[0], observed[0]) and torch.equal(expected[1], observed[1])
    assert len(wider.dilation_schedule(128)) == 8
    assert wider.extra_gate_weight.shape[1] == wider.extra_hidden + 6
    assert wider.extra_candidate_weight.shape[1] == wider.extra_hidden + 6


def test_extra_channels_receive_gradient_and_causally_affect_output_after_update() -> None:
    model = CapacityRecycledConvGRU(source(), total_hidden=104, calls_at_w128=8).train()
    features = torch.randn(4, 6, 64, generator=torch.Generator().manual_seed(3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad(); logits, quotient = model(features); loss = logits.square().mean() + quotient.square().mean(); loss.backward(); optimizer.step()
    optimizer.zero_grad(); model(features)[0].square().mean().backward()
    assert sum(float(parameter.grad.norm()) for parameter in model.extra_parameters() if parameter.grad is not None) > 0
    with torch.no_grad():
        ordinary = model(features)[0]; ablated = model(features, ablate_extra=True)[0]
    assert float((ordinary - ablated).abs().max()) > 0


def test_resource_selection_is_deterministic_largest_safe_and_rejects_science() -> None:
    frozen = {"guarded_candidate_seconds": 10.0, "peak_rss_bytes": 1000, "activation_bytes": 1000}
    rows = [
        {"embedding_passed": True, "guarded_candidate_seconds": 8.0, "peak_rss_bytes": 900, "activation_bytes": 900, "total_hidden": 104, "calls_at_w128": 8},
        {"embedding_passed": True, "guarded_candidate_seconds": 9.0, "peak_rss_bytes": 1100, "activation_bytes": 1100, "total_hidden": 112, "calls_at_w128": 10},
    ]
    assert select_resource_route(rows, frozen=frozen, rss_headroom=1.3, activation_headroom=1.3) == rows[1]
    with pytest.raises(ValueError, match="forbidden"):
        select_resource_route([{**rows[0], "exact_accuracy": 1.0}], frozen=frozen, rss_headroom=1.3, activation_headroom=1.3)


def test_checkpoint_reload_and_complete_accounting(tmp_path: Path) -> None:
    model = CapacityRecycledConvGRU(source(), total_hidden=112, calls_at_w128=8).eval()
    path = tmp_path / "weights.safetensors"; save_file({key: value.contiguous() for key, value in model.state_dict().items()}, path)
    restored = CapacityRecycledConvGRU(source(1), total_hidden=112, calls_at_w128=8); restored.load_state_dict(load_file(path)); restored.eval()
    features = torch.randn(2, 6, 128, generator=torch.Generator().manual_seed(4))
    with torch.no_grad(): first = model(features); second = restored(features)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    accounting = model.accounting(128)
    assert accounting["stored_parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert accounting["recurrent_mac_proxy"] == accounting["recurrent_calls"] * 128 * accounting["recurrent_parameters"]


def test_authenticated_fixture_and_inference_safety() -> None:
    cfg = config(); checkpoint = ROOT / ".scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"
    assert checkpoint.is_file() and sha256(checkpoint) == cfg["source_contract"]["checkpoint_sha256"]
    assert sha256(ROOT / cfg["source_contract"]["paper_path"]) == cfg["source_contract"]["paper_sha256"]
    source_text = inspect.getsource(CapacityRecycledConvGRU.forward).lower() + inspect.getsource(select_resource_route).lower()
    for forbidden in ("expected_answer", "barrett", "montgomery", "crt_recomb", "torch.remainder", "fmod(", "data/checked", "data/private", "submission/", "external call"):
        assert forbidden not in source_text
    malformed = torch.zeros(2, 5, 64)
    with pytest.raises(ValueError, match="finite"):
        CapacityRecycledConvGRU(source(), total_hidden=104, calls_at_w128=8)(malformed)
