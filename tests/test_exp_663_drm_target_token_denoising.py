from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import textwrap
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_663_drm_target_token_denoising.model import (
    DRMCell,
    MASK_TOKEN,
    corrupt_target,
    cosine_mask_ratio,
)
from experiments.exp_663_drm_target_token_denoising.run import (
    ROUTES,
    TRAINED_ROUTES,
    build_model,
    calibration_equivalence,
    deterministic_makespan,
    evaluate_transitions,
    features,
    load_module,
    make_split,
    predict,
    route_window,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_663_drm_target_token_denoising.yaml").read_text())
SOURCE = ROOT / CONFIG["source_checkpoint"]


def sample(width: int = 16, batch: int = 3) -> torch.Tensor:
    return features(list(range(batch)), [7] * batch, [65521] * batch, [0, 1, 0][:batch], width)


def exp300():
    previous_path, previous_model = sys.path.copy(), sys.modules.get("model"); had_model = "model" in sys.modules
    try: return load_module("exp300_for_exp663_test", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    finally:
        sys.path[:] = previous_path
        if had_model: sys.modules["model"] = previous_model
        else: sys.modules.pop("model", None)


def test_source_checkpoint_identity_and_function_preserving_canvas_boundary() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == CONFIG["source_checkpoint_sha256"]
    model = build_model(CONFIG, SOURCE); value = sample(); canvas = torch.randint(0, 3, (3, 16))
    source_logits, source_quotient = model.source_forward(value)
    extended = model.denoise_window(value, canvas, warmup_steps=0, gradient_steps=1)
    assert torch.equal(source_logits, extended.logits)
    assert torch.equal(source_quotient, extended.quotient_logits)
    assert torch.count_nonzero(model.canvas_projection.weight) == 0


def test_cosine_schedule_and_uniform_corruption_have_exact_counts() -> None:
    tau = torch.tensor([0.0, 0.5, 1.0]); ratio = cosine_mask_ratio(tau)
    assert torch.allclose(ratio, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)
    target = torch.tensor([[0, 1] * 8] * 3, dtype=torch.float32)
    canvas, mask = corrupt_target(target, tau, schedule="cosine", generator=torch.Generator().manual_seed(7))
    assert mask.sum(1).tolist() == [0, 8, 16]
    assert torch.equal(canvas.eq(MASK_TOKEN), mask)
    assert torch.equal(canvas[~mask], target.long()[~mask])


def test_shared_k_step_window_backpropagates_clean_final_loss() -> None:
    model = build_model(CONFIG, SOURCE); value = sample(); canvas = torch.full((3, 16), MASK_TOKEN)
    output = model.denoise_window(value, canvas, warmup_steps=2, gradient_steps=4)
    assert output.block_passes == 6 and output.logits.shape == (3, 16)
    torch.nn.functional.binary_cross_entropy_with_logits(output.logits, torch.zeros_like(output.logits)).backward()
    assert model.base.gate_weight.grad is not None and torch.count_nonzero(model.base.gate_weight.grad)
    assert model.canvas_projection.weight.grad is not None and torch.count_nonzero(model.canvas_projection.weight.grad)


def test_generate_and_remask_is_fully_masked_deterministic_and_complete() -> None:
    model = build_model(CONFIG, SOURCE); value = sample(batch=2)
    first = model.generate_and_remask(value, warmup_steps=0, gradient_steps=1, inference_steps=3, seed=91)
    second = model.generate_and_remask(value, warmup_steps=0, gradient_steps=1, inference_steps=3, seed=91)
    assert torch.equal(first.canvases[0], torch.full((2, 16), MASK_TOKEN))
    assert not first.canvases[-1].eq(MASK_TOKEN).any()
    assert all(torch.equal(left, right) for left, right in zip(first.canvases, second.canvases, strict=True))
    assert torch.equal(first.logits, second.logits) and first.block_passes == 3


def test_parameter_superset_matches_and_route_compute_is_disclosed() -> None:
    counts = {route: build_model(CONFIG, SOURCE).parameter_count() for route in ROUTES}
    assert len(set(counts.values())) == 1 and counts["drm_cosine"] > 99824
    assert route_window(CONFIG, "drm_cosine") == route_window(CONFIG, "clean_recurrent") == (2, 4)
    assert route_window(CONFIG, "masked_nonrecursive") == (0, 1)
    outputs = {route: predict(build_model(CONFIG, SOURCE), sample(batch=2), route, CONFIG, seed=2) for route in ROUTES}
    assert outputs["frozen_source"][2] == 1
    assert outputs["clean_recurrent"][2] == 6
    assert outputs["masked_nonrecursive"][2] == 1
    assert outputs["drm_cosine"][2] == 96
    assert all(value[3] >= 0 for value in outputs.values())


def test_fresh_splits_are_disjoint_and_generator_only_uses_legal_labels() -> None:
    module = exp300(); split = make_split(CONFIG, module, 64, 20261203)
    assert {key: len(value) for key, value in split.items()} == {"train": 512, "validation": 128, "test": 128}
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    sampler = module.BigIntSampler(split["train"], 4); state, other, modulus, bit, target, _ = sampler.transitions(3)
    assert target == [(2 * a + b * c) % p for a, c, p, b in zip(state, other, modulus, bit, strict=True)]


def test_inference_path_has_no_exact_arithmetic_or_answer_argument() -> None:
    source = inspect.getsource(DRMCell.generate_and_remask)
    tree = ast.parse(textwrap.dedent(source))
    prohibited = (ast.Mod, ast.FloorDiv, ast.LShift, ast.RShift)
    assert not [node for node in ast.walk(tree) if isinstance(node, prohibited)]
    assert "target" not in inspect.signature(DRMCell.generate_and_remask).parameters
    lowered = source.lower()
    assert all(term not in lowered for term in ("montgomery", "barrett", "remainder", "crt", "repair"))


def test_checkpoint_reload_is_exact(tmp_path: Path) -> None:
    model = build_model(CONFIG, SOURCE); path = tmp_path / "model.safetensors"
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
    reloaded = build_model(CONFIG, SOURCE); reloaded.load_state_dict(load_file(path))
    assert all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    assert torch.equal(model.source_forward(sample())[0], reloaded.source_forward(sample())[0])


def test_interventions_independently_switch_recursion_schedule_warmup_and_remask() -> None:
    model = build_model(CONFIG, SOURCE); value = sample(batch=2)
    base = predict(model, value, "drm_cosine", CONFIG, seed=31)
    single = predict(model, value, "drm_cosine", CONFIG, seed=31, intervention="single_step")
    warmup = predict(model, value, "drm_cosine", CONFIG, seed=31, intervention="no_warmup")
    linear = predict(model, value, "drm_cosine", CONFIG, seed=31, intervention="linear_schedule")
    no_remask = predict(model, value, "drm_cosine", CONFIG, seed=31, intervention="no_remask")
    assert base[2] == 96 and single[2] == 48 and warmup[2] == 64
    assert len(base[4]) == len(linear[4]) == len(no_remask[4]) == 17
    assert any(not torch.equal(left, right) for left, right in zip(base[4], linear[4], strict=True))
    assert any(not torch.equal(left, right) for left, right in zip(base[4], no_remask[4], strict=True))


def test_transition_evaluator_records_each_adjacent_canvas_change() -> None:
    model = build_model(CONFIG, SOURCE)
    cases = {"state": [0, 1], "multiplicand": [7, 7], "modulus": [65521, 65521], "input_bit": [0, 1], "target": [0, 9], "width": 16}
    result = evaluate_transitions(model, cases, "drm_cosine", CONFIG, root=20261202)
    assert result["cases"] == 2 and result["mean_canvas_change"] is not None
    assert result["block_passes"] == 2 * 16 * 6


def test_config_freezes_full_allowance_roots_routes_cases_and_conjuncts() -> None:
    assert CONFIG["roots"] == {"calibration": 20261202, "viability": 20261203, "confirmations": [20261204, 20261205]}
    assert tuple(CONFIG["routes"]) == ROUTES and tuple(TRAINED_ROUTES) == ROUTES[1:]
    assert CONFIG["training"]["steps_per_trained_route"] == 400
    assert CONFIG["evaluation"]["transition_cases_per_width"] == 3000
    assert CONFIG["evaluation"]["rollout_cases_per_width"] == 300
    assert CONFIG["data"]["widths"] == [64, 128, 256]
    assert set(CONFIG["evaluation"]["interventions"]) == {"all_mask", "linear_schedule", "single_step", "no_warmup", "no_remask"}
    assert CONFIG["gates"]["minimum_w128_drm_rollout"] == 0.80
    assert CONFIG["gates"]["minimum_w256_drm_rollout"] == 0.10


def test_repair_topology_is_bounded_and_projection_preserves_submission_order() -> None:
    policy = CONFIG["resource_policy"]
    assert policy["evaluator_processes"] == 3 and policy["evaluator_torch_threads_each"] == 4
    assert deterministic_makespan([10.0, 100.0, 20.0, 30.0], 3) == 100.0
    source = inspect.getsource(__import__("experiments.exp_663_drm_target_token_denoising.run", fromlist=["Experiment"]).Experiment.run)
    assert "ProcessPoolExecutor" in source and "max_workers=process_count" in source
    assert "models: dict" not in source


def test_frozen_calibration_equivalence_comparator_covers_scientific_outputs() -> None:
    reference_path = ROOT / CONFIG["resource_policy"]["repair_reference_calibration"]
    prereg_path = ROOT / CONFIG["resource_policy"]["repair_reference_preregistration"]
    reference, prereg = json.loads(reference_path.read_text()), json.loads(prereg_path.read_text())
    receipt = calibration_equivalence(reference, prereg, reference["routes"], reference["diagnostics"], prereg)
    assert receipt["passed"] and all(receipt["checks"].values())
    mutated = json.loads(json.dumps(reference["routes"]))
    mutated["drm_cosine"]["test"]["128"]["rollout"]["prediction_sha256"] = "changed"
    assert not calibration_equivalence(reference, prereg, mutated, reference["diagnostics"], prereg)["passed"]


def test_sprm_and_arc_specific_selection_are_excluded() -> None:
    model_source = inspect.getsource(DRMCell).lower()
    runner_source = inspect.getsource(predict).lower()
    assert "gaussian" not in model_source and "latent noise" not in model_source
    assert "q-head" not in runner_source and "augmentation" not in runner_source
    assert "confidence" not in inspect.getsource(DRMCell.generate_and_remask).lower()
