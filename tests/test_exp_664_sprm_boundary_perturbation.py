from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import sys
import textwrap
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_664_sprm_boundary_perturbation.model import (
    ROUTE_MODES,
    SPRMCell,
    linear_beta,
    state_scaled_perturbation,
)
from experiments.exp_664_sprm_boundary_perturbation.run import (
    ROUTES,
    TRAINED_ROUTES,
    Experiment,
    boundary_diagnostic,
    build_model,
    deterministic_makespan,
    features,
    inference_parity_and_runtime,
    load_module,
    make_split,
    train_route,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_664_sprm_boundary_perturbation.yaml").read_text())
SOURCE = ROOT / CONFIG["source_checkpoint"]


def sample(width: int = 16, batch: int = 3) -> torch.Tensor:
    return features(list(range(batch)), [7] * batch, [65521] * batch, [0, 1, 0][:batch], width)


def exp300():
    previous_path, previous_model = sys.path.copy(), sys.modules.get("model"); had_model = "model" in sys.modules
    try: return load_module("exp300_for_exp664_test", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    finally:
        sys.path[:] = previous_path
        if had_model: sys.modules["model"] = previous_model
        else: sys.modules.pop("model", None)


def test_source_identity_and_noise_free_inference_are_bit_exact() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == CONFIG["source_checkpoint_sha256"]
    model = build_model(CONFIG, SOURCE).eval(); value = sample(); before = torch.random.get_rng_state().clone(); wrapped = model(value); after = torch.random.get_rng_state().clone(); direct = model.base(value)
    assert torch.equal(wrapped[0], direct[0]) and torch.equal(wrapped[1], direct[1])
    assert torch.equal(before, after) and model.parameter_count() == 99824


def test_linear_beta_has_exact_frozen_endpoints_and_is_monotone() -> None:
    tau = torch.tensor([1, 2, 500, 1000]); beta = linear_beta(tau, start=1e-4, end=0.02, steps=1000)
    assert beta[0].item() == torch.tensor(1e-4).item()
    assert beta[-1].item() == torch.tensor(0.02).item()
    assert torch.all(beta[1:] > beta[:-1])


def test_equation9_state_scaled_update_matches_seeded_formula() -> None:
    state = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    beta = torch.tensor([0.001, 0.02])[:, None, None]
    actual, stats = state_scaled_perturbation(state, beta, generator=torch.Generator().manual_seed(9), state_scaled=True)
    mean = state.mean((1, 2), keepdim=True); std = state.var((1, 2), unbiased=False, keepdim=True).sqrt(); noise = torch.randn(state.shape, generator=torch.Generator().manual_seed(9)); expected = (1 - beta).sqrt() * state + beta.sqrt() * (mean + std * noise)
    assert torch.equal(actual, expected)
    assert stats["noise_rms"] > 0 and stats["pre_variance"] > 0


def test_noise_and_detach_occur_only_at_frozen_training_boundaries() -> None:
    model = build_model(CONFIG, SOURCE); value = sample(batch=2)
    outputs = {route: model.training_forward(value, route=route, generator=torch.Generator().manual_seed(11)) for route in TRAINED_ROUTES}
    assert [event["pass"] for event in outputs["sprm_boundary"].boundary_events] == [4]
    assert [event["pass"] for event in outputs["clean_tbptt"].boundary_events] == [4]
    assert [event["pass"] for event in outputs["wrong_boundary"].boundary_events] == [4, 5]
    assert outputs["sprm_boundary"].boundary_events[0]["noise_rms"] > 0
    assert outputs["clean_tbptt"].boundary_events[0]["noise_rms"] == 0
    assert outputs["unscaled_boundary"].block_passes == outputs["sprm_boundary"].block_passes == len(model.base.dilation_schedule(16))


def test_tbptt_final_window_is_detached_but_deep_supervision_reaches_stem() -> None:
    model = build_model(CONFIG, SOURCE); output = model.training_forward(sample(batch=2), route="clean_tbptt", generator=torch.Generator().manual_seed(3)); output.logits.sum().backward(); assert model.base.stem.weight.grad is None
    model.zero_grad(set_to_none=True); output = model.training_forward(sample(batch=2), route="clean_tbptt", generator=torch.Generator().manual_seed(3)); sum(item.sum() for item in output.window_logits).backward(); assert model.base.stem.weight.grad is not None and torch.count_nonzero(model.base.stem.weight.grad)


def test_all_trained_routes_match_parameters_windows_calls_and_flops() -> None:
    model = build_model(CONFIG, SOURCE); results = {route: model.training_forward(sample(batch=2), route=route, generator=torch.Generator().manual_seed(21)) for route in TRAINED_ROUTES}
    assert set(ROUTE_MODES) == set(TRAINED_ROUTES)
    assert len({(item.block_passes, item.analytical_flops, len(item.window_logits)) for item in results.values()}) == 1
    assert model.parameter_count() == model.base.parameter_count()


def test_inference_api_excludes_noise_route_target_and_exact_arithmetic() -> None:
    signature = inspect.signature(SPRMCell.forward); assert list(signature.parameters) == ["self", "features"]
    source = textwrap.dedent(inspect.getsource(SPRMCell.forward)); tree = ast.parse(source)
    assert not [node for node in ast.walk(tree) if isinstance(node, (ast.Mod, ast.FloorDiv, ast.LShift, ast.RShift))]
    lowered = source.lower(); assert all(term not in lowered for term in ("target", "answer", "repair", "barrett", "montgomery", "crt"))
    assert "torch.randn" not in source and "training_forward" not in source


def test_fresh_prime_splits_are_disjoint_and_frozen() -> None:
    module = exp300(); split = make_split(CONFIG, module, 64, 20261207)
    assert {key: len(value) for key, value in split.items()} == {"train": 512, "validation": 128, "test": 128}
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])


def test_checkpoint_roundtrip_and_public_outputs_are_exact(tmp_path: Path) -> None:
    model = build_model(CONFIG, SOURCE).eval(); path = tmp_path / "model.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path); reloaded = build_model(CONFIG, SOURCE); reloaded.load_state_dict(load_file(path)); value = sample()
    assert all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    assert torch.equal(model(value)[0], reloaded(value)[0])


def test_boundary_diagnostic_is_seeded_and_training_only() -> None:
    model = build_model(CONFIG, SOURCE); cases = {"state": [0, 1], "multiplicand": [7, 7], "modulus": [65521, 65521], "input_bit": [0, 1], "target": [0, 9], "quotient": [0, 0], "width": 16}
    first = boundary_diagnostic(model, cases, "sprm_boundary", 20261206); second = boundary_diagnostic(model, cases, "sprm_boundary", 20261206)
    assert first == second and first["event_passes"] == [4] and first["mean_noise_rms"] > 0


def test_inference_parity_receipt_preserves_rng_and_predictions() -> None:
    model = build_model(CONFIG, SOURCE); cases = {"state": list(range(8)), "multiplicand": [7] * 8, "modulus": [65521] * 8, "input_bit": [0, 1] * 4, "target": list(range(8)), "quotient": [0] * 8, "width": 16}
    receipt = inference_parity_and_runtime(model, cases, 2)
    assert receipt["logits_equal"] and receipt["quotient_equal"] and receipt["rng_unchanged"]
    assert receipt["candidate_projected_wall_seconds"] >= 0


def test_config_freezes_roots_routes_work_and_conjunctive_gates() -> None:
    assert CONFIG["roots"] == {"calibration": 20261206, "viability": 20261207, "confirmations": [20261208, 20261209]}
    assert tuple(CONFIG["routes"]) == ROUTES and tuple(CONFIG["trained_routes"]) == TRAINED_ROUTES
    assert CONFIG["training"]["steps_per_trained_route"] == 400 and CONFIG["model"]["boundary_passes"] == 4
    assert CONFIG["evaluation"]["transition_cases_per_width"] == 3000 and CONFIG["evaluation"]["rollout_cases_per_width"] == 300
    assert CONFIG["gates"]["minimum_w128_sprm_rollout"] == 0.85 and CONFIG["gates"]["minimum_w256_sprm_rollout"] == 0.10


def test_runner_freezes_complete_route_projection_and_decision_safe_order() -> None:
    source = inspect.getsource(Experiment.run)
    assert "sprm-w128-primary-futility" in source and source.index("sprm-w128-primary-futility") < source.index("sprm-w256-primary-futility")
    assert "projected_route_durations_wall_seconds" in source and "validation_transition_cases_per_width" in source
    assert "ProcessPoolExecutor" in source and "randomized_weights" in source
    assert deterministic_makespan([10.0, 100.0, 20.0, 30.0], 3) == 100.0


def test_one_update_vertical_training_slice_persists_matched_checkpoint() -> None:
    config = copy.deepcopy(CONFIG); config["data"]["training_widths"] = [16]; config["training"]["schedule"] = [16]; config["training"]["batch_size"] = 2; config["training"]["log_every_steps"] = 1
    splits = {16: {"train": [65521], "validation": [65497], "test": [65479]}}
    model, record = train_route(config, exp300(), SOURCE, splits, 20261206, "sprm_boundary", 1, ROOT / ".scratch/exp-664-sprm-boundary-perturbation/focused", "focused")
    assert record["training"]["updates"] == 1 and record["training"]["examples"] == 2
    assert record["checkpoint_reload_exact"] and record["parameters"] == 99824
    assert Path(ROOT / record["checkpoint"]["path"]).is_file() and model.parameter_count() == 99824


def test_source_claim_exclusions_are_enforced_by_model_text() -> None:
    model_source = inspect.getsource(SPRMCell).lower()
    assert "mask_token" not in model_source and "canvas" not in model_source and "confidence" not in model_source
    assert "q-head" not in model_source and "augmentation" not in model_source
