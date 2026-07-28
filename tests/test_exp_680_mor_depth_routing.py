from __future__ import annotations

import inspect

import torch
from safetensors.torch import load_file, save_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_680_mor_depth_routing.model import RoutedDepthConvGRU, capacity_counts, expert_topk, fixed_full_identity


DEPTHS = {64: (7, 9, 12), 128: (8, 11, 14), 256: (9, 13, 16)}


def model(kind: str = "expert_choice") -> RoutedDepthConvGRU:
    torch.manual_seed(680)
    return RoutedDepthConvGRU(SharedDilatedConvGRU(width=128, hidden=12, cycles=2), route_kind=kind, depth_sets=DEPTHS)


def test_native_expert_choice_capacity_and_hierarchy() -> None:
    scores = torch.tensor([0.1, 0.9, 0.4, 0.2, 0.8, 0.3])
    first, final = capacity_counts(6)
    selected = expert_topk(scores, first)
    nested = expert_topk(scores, final, selected)
    assert first == 4 and final == 2
    assert set(nested.tolist()) <= set(selected.tolist())
    assert nested.tolist() == [1, 4]


def test_expert_choice_uses_all_depths_with_exact_capacity() -> None:
    routed = model().train()
    _, _, trace = routed(torch.randn(6, 6, 128), return_trace=True)
    assert trace["histogram"] == {"8": 2, "11": 2, "14": 2}
    assert trace["average_calls"] == 11.0
    assert trace["saved_calls"] == 18


def test_token_choice_is_one_shot_and_load_loss_is_finite() -> None:
    routed = model("token_choice").train()
    logits, quotient, trace = routed(torch.randn(9, 6, 64), return_trace=True)
    assert logits.shape == (9, 64) and quotient.shape == (9, 3)
    assert sum(trace["histogram"].values()) == 9
    assert torch.isfinite(trace["load_loss"])


def test_fixed_full_initialization_is_bit_exact() -> None:
    routed = model("expert_choice")
    receipt = fixed_full_identity(routed.source, routed, torch.randn(4, 6, 128))
    assert receipt == {"bit_logits_equal": True, "quotient_logits_equal": True, "maximum_absolute_delta": 0.0}


def test_equal_average_controls_match_calls() -> None:
    features = torch.randn(8, 6, 128)
    fixed = model("equal_fixed").eval()(features, return_trace=True)[2]
    two_point = model("equal_two_point").eval()(features, return_trace=True)[2]
    assert fixed["average_calls"] == two_point["average_calls"] == 11.0
    assert fixed["histogram"] == {"8": 0, "11": 8, "14": 0}
    assert two_point["histogram"] == {"8": 4, "11": 0, "14": 4}


def test_joint_router_and_core_gradients() -> None:
    routed = model().train()
    logits, quotient, trace = routed(torch.randn(6, 6, 64), return_trace=True)
    (logits.square().mean() + quotient.square().mean() + trace["load_loss"]).backward()
    assert sum(float(p.grad.norm()) for p in routed.router_parameters() if p.grad is not None) > 0
    assert sum(float(p.grad.norm()) for p in routed.core_parameters() if p.grad is not None) > 0


def test_route_intervention_changes_logits_and_preserves_histogram() -> None:
    routed = model().train()
    features = torch.randn(6, 6, 128)
    ordinary = routed(features, return_trace=True)
    depths = ordinary[2]["depths"]
    shuffled = routed(features, force_depths=depths.roll(1))
    assert sorted(depths.tolist()) == sorted(depths.roll(1).tolist())
    assert not torch.equal(ordinary[0], shuffled[0])


def test_reload_and_deterministic_cpu_route(tmp_path) -> None:
    routed = model().train(); features = torch.randn(6, 6, 128)
    routed(features); routed.eval(); expected = routed(features, return_trace=True)
    path = tmp_path / "router.safetensors"
    save_file({key: value.detach().contiguous() for key, value in routed.state_dict().items()}, path)
    restored = model().eval(); restored.load_state_dict(load_file(path)); observed = restored(features, return_trace=True)
    assert torch.equal(expected[0], observed[0])
    assert torch.equal(expected[1], observed[1])
    assert torch.equal(expected[2]["depths"], observed[2]["depths"])


def test_router_surface_has_no_label_answer_or_programmed_arithmetic_inputs() -> None:
    signature = inspect.signature(RoutedDepthConvGRU.forward)
    assert set(signature.parameters) == {"self", "features", "force_depths", "return_trace"}
    source = inspect.getsource(RoutedDepthConvGRU.forward)
    for forbidden in ("correctness", "oracle_depth", "barrett", "montgomery", "crt", "long_division"):
        assert forbidden not in source.lower()


def test_malformed_features_and_forced_depths_fail_closed() -> None:
    routed = model().eval()
    for bad in (torch.randn(2, 5, 128), torch.full((2, 6, 128), float("nan"))):
        try: routed(bad)
        except ValueError: pass
        else: raise AssertionError("malformed features must fail")
    try: routed(torch.randn(2, 6, 128), force_depths=torch.tensor([8, 12]))
    except ValueError: pass
    else: raise AssertionError("non-frozen depth must fail")
