"""Focused L0 and durable-receipt tests for EXP-657."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/exp_657_coupled_redundant_learned_state"
CONFIG_PATH = ROOT / "configs/exp_657_coupled_redundant_learned_state.yaml"
RUN_DIR = ROOT / "experiments/runs/exp-657-coupled-redundant-learned-state"
FAILED_CALIBRATION_DIR = RUN_DIR / "calibration-root-20261170"
CALIBRATION_DIR = RUN_DIR / "calibration-root-20261173"
FAILED_ROOT_HASHES = {
    "attempt-state.json": "0f00042606d29e9554f40eaf61abc28ebe35e39e5e5c4000ff342b4259a748dc",
    "completion.json": "a2d3c5e598b19b4e1f366dc25fd206a25bf87e1ae955cfa44e475e6af9f8311f",
    "worker-stdout.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "worker-stderr.txt": "72f5af7d49b1397fd98fd96e594a2feaa93c692480a14323975360a001203ca6",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model_module = load_module("exp657_model_for_tests", EXPERIMENT / "model.py")
_path_before_run_import = list(sys.path)
_had_model_before_run_import = "model" in sys.modules
_model_before_run_import = sys.modules.get("model")
try:
    sys.path.insert(0, str(EXPERIMENT))
    sys.modules["model"] = model_module
    run_module = load_module("exp657_run_for_tests", EXPERIMENT / "run.py")
finally:
    sys.path[:] = _path_before_run_import
    if _had_model_before_run_import:
        sys.modules["model"] = _model_before_run_import
    else:
        sys.modules.pop("model", None)

assert sys.path == _path_before_run_import
assert ("model" in sys.modules) is _had_model_before_run_import
if _had_model_before_run_import:
    assert sys.modules["model"] is _model_before_run_import


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def tiny_context(batch: int = 3, width: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(657)
    return torch.randn((batch, 5, width), generator=generator)


def tiny_model(persistence: str):
    torch.manual_seed(657)
    return model_module.CoupledRedundantLearnedState(
        hidden=8,
        syndrome_hidden=6,
        cycles=1,
        context_channels=5,
        persistence=persistence,
        correction_scale=0.5,
    )


def walk_keys(value, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from walk_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_keys(child, f"{prefix}[{index}]")


def test_config_preregisters_exact_stages_roots_denominators_and_gates() -> None:
    payload = config()
    assert payload["roots"] == {
        "consumed_failed_calibration": 20261170,
        "superseded_never_authorized": [20261171, 20261172],
        "calibration": 20261173,
        "first_formal": 20261174,
        "second_formal": 20261175,
        "formal_status": "reserved-unopened-authorization-required",
        "confirmation_status": "prohibited-until-formal-promotion-and-separate-authorization",
    }
    assert list(payload["stages"]) == ["L0", "L1", "L2", "L3"]
    evaluation = payload["formal_protocol"]["evaluation"]
    assert evaluation["held_out_natural_error_denominator"] == 256
    assert evaluation["transition_cases_per_width_arm"] == 4096
    assert evaluation["rollout_cases_per_width_arm"] == 512
    assert evaluation["causal_ablation_rollout_cases_per_mode"] == 512
    gates = payload["promotion_gates"]
    assert gates["held_out_natural_one_step_recovery"]["minimum_fraction"] == 0.50
    assert gates["causal_channel_a"]["minimum_drop_fraction"] == 0.02
    assert gates["causal_channel_b"]["minimum_drop_fraction"] == 0.02
    assert gates["width32_rollout_gain"]["minimum_fraction"] == 0.03
    assert gates["width16_retention"]["minimum_fraction"] == -0.01
    assert gates["clean_width32_transition"]["minimum_fraction"] == -0.0005
    assert gates["trained_width_viability"]["minimum_challenger_exact_rollout_numerator"] == 1
    collapse = gates["channel_collapse"]
    assert collapse["collapse_if_all"] == {
        "maximum_mean_absolute_probability_difference": 0.01,
        "maximum_binary_disagreement_fraction": 0.001,
        "minimum_centered_cosine_similarity": 0.999,
    }
    assert collapse["maximum_case_error_jaccard"] == 0.90
    assert collapse["minimum_error_union_cases"] == 32


def test_model_inference_ast_has_no_exact_arithmetic_or_decoding_path() -> None:
    source = (EXPERIMENT / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "remainder",
        "fmod",
        "div",
        "floor_divide",
        "bitwise_xor",
        "bitwise_and",
        "decode",
        "decode_bits",
        "bits_to_int",
        "bits_to_integer",
        "to_integer",
        "tolist",
        "item",
    }
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert not calls.intersection(forbidden_calls)
    forbidden_binary = (ast.Mod, ast.FloorDiv, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor)
    assert not any(isinstance(node, ast.BinOp) and isinstance(node.op, forbidden_binary) for node in ast.walk(tree))
    assert "offline" not in source


def test_channels_are_separately_learned_and_matched_arms_are_exact_geometry() -> None:
    control = tiny_model("single")
    challenger = tiny_model("redundant")
    assert control.branch_parameters_are_distinct()
    assert challenger.branch_parameters_are_distinct()
    assert control.parameter_count() == challenger.parameter_count()
    assert control.parameter_signature() == challenger.parameter_signature()
    for (control_name, control_value), (challenger_name, challenger_value) in zip(
        control.state_dict().items(), challenger.state_dict().items()
    ):
        assert control_name == challenger_name
        assert torch.equal(control_value, challenger_value)


def test_step_shapes_determinism_persistence_and_causal_ablations() -> None:
    width = 16
    context = tiny_context(width=width)
    challenger = tiny_model("redundant").eval()
    state_a, state_b = challenger.initial_state(3, width)
    first = challenger.forward_step(state_a, state_b, context)
    repeated = challenger.forward_step(state_a, state_b, context)
    for left, right in zip(first, repeated):
        assert torch.equal(left, right)
    assert first.fused_logits.shape == (3, width)
    assert first.channel_a_logits.shape == (3, width)
    assert first.channel_b_logits.shape == (3, width)
    assert first.syndrome_error_logits.shape == (3, 2, width)
    assert first.next_state_a.shape == (3, 1, width)
    assert not torch.equal(first.next_state_a, first.next_state_b)
    assert not torch.equal(
        first.fused_logits,
        challenger.forward_step(state_a, state_b, context, ablation="drop_a").fused_logits,
    )
    assert not torch.equal(
        first.fused_logits,
        challenger.forward_step(state_a, state_b, context, ablation="drop_b").fused_logits,
    )

    control = tiny_model("single").eval()
    control_first = control.forward_step(state_a, state_b, context)
    assert torch.equal(control_first.next_state_a, control_first.next_state_b)
    assert torch.equal(control_first.fused_logits, first.fused_logits)
    second_control = control.forward_step(
        control_first.next_state_a, control_first.next_state_b, context
    )
    second_challenger = challenger.forward_step(
        first.next_state_a, first.next_state_b, context
    )
    assert not torch.equal(second_control.fused_logits, second_challenger.fused_logits)


def test_all_learned_mechanism_groups_receive_gradient() -> None:
    model = tiny_model("redundant").train()
    state_a, state_b = model.initial_state(4, 16)
    output = model.forward_step(state_a, state_b, tiny_context(batch=4))
    targets = torch.tensor(
        [[(row + column) % 2 for column in range(16)] for row in range(4)],
        dtype=torch.float32,
    )
    loss, _ = run_module.supervised_loss(output, targets, config())
    loss.backward()
    groups = {
        "channel_a": model.channel_a,
        "channel_b": model.channel_b,
        "syndrome": model.syndrome_encoder,
        "syndrome_error": model.syndrome_error_head,
        "corrections": model.learned_corrections,
        "fusion": model.learned_fusion_gate,
        "decoder": model.learned_decoder,
    }
    for name, group in groups.items():
        gradients = [parameter.grad for parameter in group.parameters()]
        assert gradients and all(value is not None for value in gradients), name
        assert sum(float(value.abs().sum()) for value in gradients if value is not None) > 0.0, name


def test_natural_error_mining_uses_observed_first_errors_without_corruption() -> None:
    payload = config()
    split = run_module.make_prime_split(payload, 16, 20261173)
    sampler = run_module.OfflineSampler(split["train"], 65716)
    model = tiny_model("redundant")
    mined = run_module.mine_natural_error_batch(
        model,
        sampler,
        width=16,
        needed=4,
        maximum_candidate_pairs=32,
    )
    assert mined is not None
    assert mined.state_a.shape == (4, 1, 16)
    assert mined.state_b.shape == (4, 1, 16)
    assert mined.context.shape == (4, 5, 16)
    assert mined.target_bits.shape == (4, 16)
    assert mined.candidate_pairs <= 32
    mined_tensors = (mined.state_a, mined.state_b, mined.context, mined.target_bits)
    assert all(tensor.is_inference() for tensor in mined_tensors)

    replay = run_module.materialize_natural_error_batch(mined)
    replay_tensors = (replay.state_a, replay.state_b, replay.context, replay.target_bits)
    assert all(not tensor.is_inference() for tensor in replay_tensors)
    model.train()
    output = model.forward_step(replay.state_a, replay.state_b, replay.context)
    loss, _ = run_module.supervised_loss(output, replay.target_bits, payload)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sum(float(parameter.grad.abs().sum()) for parameter in model.parameters()) > 0.0
    source = (EXPERIMENT / "run.py").read_text(encoding="utf-8")
    assert "state_corruption" not in source
    assert "torch.rand_like" not in source
    assert "torch.randn_like" not in source


def test_resource_components_release_only_work_shape_and_digests() -> None:
    payload = config()
    split = run_module.make_prime_split(payload, 16, 20261173)
    model = tiny_model("redundant").eval()
    transition = run_module.resource_transition_component(
        model, run_module.OfflineSampler(split["test"], 1), 16, 8
    )
    rollout = run_module.resource_rollout_component(
        model, run_module.OfflineSampler(split["test"], 2), 16, 4, 2
    )
    assert set(transition) == {"digest", "cases"}
    assert set(rollout) == {"digest", "case_bit_steps"}
    for payload_part in (transition, rollout):
        keys = {path.lower() for path, _ in walk_keys(payload_part)}
        assert not any("accuracy" in key or key.endswith("correct") for key in keys)


def test_dynamic_budget_uses_maximum_rates_separate_clocks_and_headroom() -> None:
    payload = config()
    phases = []
    components = (
        ("training_update", "updates", 1),
        ("transition_evaluation", "transition_cases", 10),
        ("rollout_evaluation", "case_bit_steps", 20),
        ("natural_recovery_evaluation", "candidate_bit_steps", 30),
        ("causal_ablation_evaluation", "case_bit_steps", 20),
        ("collapse_diagnostic", "transition_cases", 10),
        ("width256_runtime_shape", "case_bit_steps", 512),
        ("checkpoint_serialization", "bytes", 1000),
    )
    for index, (component, unit_kind, units) in enumerate(components, start=1):
        phases.append(
            {
                "component": component,
                "unit_kind": unit_kind,
                "units": units,
                "monotonic_wall_seconds": float(index),
                "process_user_plus_system_aggregate_cpu_seconds": float(index * 3),
                "peak_rss_bytes": 1000,
            }
        )
    # A slower second update must control the projected update rate.
    phases.append(
        {
            "component": "training_update",
            "unit_kind": "updates",
            "units": 1,
            "monotonic_wall_seconds": 20.0,
            "process_user_plus_system_aggregate_cpu_seconds": 70.0,
            "peak_rss_bytes": 1000,
        }
    )
    outer = {
        "monotonic_wall_seconds": sum(p["monotonic_wall_seconds"] for p in phases) + 2.0,
        "process_user_plus_system_aggregate_cpu_seconds": sum(
            p["process_user_plus_system_aggregate_cpu_seconds"] for p in phases
        )
        + 4.0,
        "peak_rss_bytes": 2000,
    }
    budget = run_module.resolve_dynamic_budget(payload, phases, outer, 1000)
    assert budget["maximum_measured_rates"]["monotonic_wall_seconds"]["training_update"] == 20.0
    assert budget["maximum_measured_rates"]["process_user_plus_system_aggregate_cpu_seconds"]["training_update"] == 70.0
    assert budget["explicit_setup_headroom"]["monotonic_wall_seconds"] >= 15.0
    assert budget["explicit_setup_headroom"]["process_user_plus_system_aggregate_cpu_seconds"] >= 5.0
    assert budget["hard_runaway_ceiling"]["monotonic_wall_seconds"] > budget["soft_budget"]["monotonic_wall_seconds"] > budget["projected"]["monotonic_wall_seconds"]
    assert budget["hard_runaway_ceiling"]["process_user_plus_system_aggregate_cpu_seconds"] > budget["soft_budget"]["process_user_plus_system_aggregate_cpu_seconds"] > budget["projected"]["process_user_plus_system_aggregate_cpu_seconds"]
    assert budget["fixed_short_wall_cap_used"] is False
    assert budget["fidelity_reduced"] is False


def test_decision_safe_attainable_bounds_are_exact() -> None:
    assert run_module.best_attainable_rate(40, 80, 100) == pytest.approx(0.60)
    assert run_module.best_attainable_rate(39, 80, 100) < 0.60
    assert run_module.best_attainable_paired_difference(20, 10, 80, 100) == pytest.approx(0.30)
    with pytest.raises(ValueError):
        run_module.best_attainable_rate(2, 1, 4)
    with pytest.raises(ValueError):
        run_module.best_attainable_paired_difference(1, 3, 2, 4)


def test_formal_root_fails_closed_without_authorization() -> None:
    authorization = RUN_DIR / "formal-authorization.json"
    assert not authorization.exists()
    with pytest.raises(RuntimeError, match="authorization file is absent"):
        run_module.validate_formal_authorization(CONFIG_PATH, 20261174)
    with pytest.raises(RuntimeError, match="at least 20261174"):
        run_module.validate_formal_authorization(CONFIG_PATH, 20261173)


def test_consumed_failed_root_20261170_is_byte_immutable() -> None:
    assert FAILED_CALIBRATION_DIR.is_dir()
    for name, expected in FAILED_ROOT_HASHES.items():
        path = FAILED_CALIBRATION_DIR / name
        assert path.is_file()
        assert run_module.sha256(path) == expected
    completion = json.loads(
        (FAILED_CALIBRATION_DIR / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "failed-child-exited"
    assert completion["root"] == 20261170
    assert completion["scientific_accuracy_or_selection_reported"] is False
    assert completion["formal_root_opened"] is False


def test_calibration_receipt_if_present_is_resource_only_complete_and_sealed() -> None:
    receipt_path = CALIBRATION_DIR / "calibration.json"
    if not receipt_path.exists():
        pytest.skip("L1 has not run yet")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    completion = json.loads((CALIBRATION_DIR / "completion.json").read_text(encoding="utf-8"))
    attempt = json.loads((CALIBRATION_DIR / "attempt-state.json").read_text(encoding="utf-8"))
    assert receipt["root"] == 20261173
    assert receipt["stage"] == "L1"
    assert receipt["status"] == "consumed-complete-non-scientific"
    assert receipt["boundary"] == {
        "scientific_accuracy_or_selection_reported": False,
        "model_update_for_scientific_claim": False,
        "formal_root_opened": False,
        "checked_independent_or_private_answers_read": False,
        "candidate_or_submission_changed": False,
    }
    assert completion["status"] == "completed-child-exited-zero"
    assert completion["outer_resource_boundary"]["child_exit_code"] == 0
    assert completion["completion_observed_after_child_exit"] is True
    assert attempt["status"] == "consumed-complete-non-scientific"
    assert receipt["resolved_resource_budget"]["fidelity_reduced"] is False
    assert receipt["resolved_resource_budget"]["fixed_short_wall_cap_used"] is False
    assert receipt["workload"]["every_arm_trained"] is True
    assert receipt["workload"]["all_planned_evaluation_components_exercised"] is True
    assert receipt["workload"]["natural_first_error_training_available_for_every_arm"] is True
    assert receipt["workload"]["held_out_natural_control_error_threshold_met"] is True
    assert receipt["workload"]["natural_error_counts_released"] is False
    assert receipt["workload"]["generic_or_random_synthetic_corruption_used"] is False
    assert receipt["next_authorization_boundary"]["minimum_root"] == 20261174
    assert receipt["next_authorization_boundary"]["authorization_exists"] is False
    assert not (RUN_DIR / "formal-authorization.json").exists()
    assert not any(path.name.startswith("formal-root-") for path in RUN_DIR.glob("formal-root-*"))

    permitted_metric_paths = {
        "boundary.scientific_accuracy_or_selection_reported",
    }
    for path, value in walk_keys(receipt):
        lowered = path.lower()
        if "accuracy" in lowered or "selection" in lowered:
            assert path in permitted_metric_paths or value in (False, "resource-only-no-accuracy-or-selection")


def test_calibration_artifacts_bind_current_sources_if_present() -> None:
    receipt_path = CALIBRATION_DIR / "calibration.json"
    if not receipt_path.exists():
        pytest.skip("L1 has not run yet")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source"]["config_sha256"] == run_module.sha256(CONFIG_PATH)
    assert receipt["source"]["runner_sha256"] == run_module.sha256(EXPERIMENT / "run.py")
    assert receipt["source"]["model_sha256"] == run_module.sha256(EXPERIMENT / "model.py")
    for checkpoint in receipt["checkpoints"].values():
        path = ROOT / checkpoint["path"]
        assert path.is_file()
        assert run_module.sha256(path) == checkpoint["sha256"]
        assert checkpoint["scientific_checkpoint"] is False
