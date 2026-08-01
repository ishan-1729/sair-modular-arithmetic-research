from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_662_ut_memory_token_transition.model import (
    DerfNorm,
    QKRoPEAttention,
    SharedUTCell,
    UTMemoryTransition,
    apply_rope,
)
from experiments.exp_662_ut_memory_token_transition.run import (
    ROUTES,
    SWEEP_ROUTES,
    bits,
    build_model,
    features,
    gate_summary,
    make_split,
    route_spec,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_662_ut_memory_token_transition.yaml").read_text())


def load_exp300():
    path = ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"
    previous_path, previous_model = sys.path.copy(), sys.modules.get("model")
    had_model = "model" in sys.modules
    try:
        spec = importlib.util.spec_from_file_location("exp300_for_exp662_test", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path
        if had_model:
            sys.modules["model"] = previous_model
        else:
            sys.modules.pop("model", None)


def sample(width: int = 8, batch: int = 3) -> torch.Tensor:
    return features(list(range(batch)), [3] * batch, [251] * batch, [0, 1, 0][:batch], width)


def test_learned_memory_tokens_are_sample_independent_and_trainable() -> None:
    torch.manual_seed(1); model = build_model(CONFIG, "memory_8")
    before, sequence = model._initial_tokens(sample(), 8, "none", None)
    assert before.shape == (3, 8, CONFIG["model"]["hidden"])
    assert torch.equal(before[0], before[1]) and sequence.shape == (3, 8, 16)
    output = model(sample(), memory_count=8); output.logits.sum().backward()
    assert model.memory_tokens.grad is not None and torch.count_nonzero(model.memory_tokens.grad[:8])


def test_source_faithful_block_contains_derfnorm_rope_qk_norm_and_prenorm() -> None:
    norm = DerfNorm(8); value = torch.randn(2, 5, 8)
    assert norm(value).shape == value.shape and torch.isfinite(norm(value)).all()
    rotary = apply_rope(torch.randn(2, 2, 5, 4)); assert rotary.shape == (2, 2, 5, 4)
    attention = QKRoPEAttention(16, 4); assert attention(torch.randn(2, 7, 16)).shape == (2, 7, 16)
    cell = SharedUTCell(16, 4, 32)
    assert isinstance(cell.attention_norm, DerfNorm) and isinstance(cell.ffn_norm, DerfNorm)


def test_act_biases_and_fixed_depth_outputs() -> None:
    deep, default = build_model(CONFIG, "memory_16"), build_model(CONFIG, "default_bias_memory_16")
    assert torch.allclose(torch.sigmoid(deep.router.bias), torch.tensor([0.04742587]), atol=1e-6)
    assert torch.allclose(torch.sigmoid(default.router.bias), torch.tensor([0.5]))
    act = deep(sample(), memory_count=16, use_act=True)
    fixed = deep(sample(), memory_count=16, use_act=False, max_steps=4)
    assert act.halt_probabilities.shape == (3, 6, 24)
    assert fixed.block_calls == 4 and fixed.halt_steps.shape == (3, 8)
    assert torch.isfinite(act.ponder_cost)


def test_memory_sweep_and_interventions_are_bounded_and_deterministic() -> None:
    model = build_model(CONFIG, "memory_16"); value = sample()
    for count in CONFIG["memory_counts"]:
        assert model(value, memory_count=count).logits.shape == (3, 8)
    first = model(value, memory_count=16, ablation="randomize", generator=torch.Generator().manual_seed(9)).logits
    second = model(value, memory_count=16, ablation="randomize", generator=torch.Generator().manual_seed(9)).logits
    assert torch.equal(first, second)
    assert not torch.equal(model(value, memory_count=16).logits, model(value, memory_count=16, ablation="zero").logits)


def test_all_routes_match_parameter_count_and_disclose_calls_and_flops() -> None:
    counts = {route: build_model(CONFIG, route).parameter_count() for route in ROUTES}
    assert len(set(counts.values())) == 1
    outputs = {route: build_model(CONFIG, route)(sample(), memory_count=route_spec(route)["memory_count"], use_act=route_spec(route)["use_act"]) for route in ROUTES}
    assert {output.block_calls for output in outputs.values()} == {6}
    assert all(output.analytical_flops > 0 for output in outputs.values())


def test_offline_generator_split_and_feature_contract() -> None:
    exp300 = load_exp300(); split = make_split(CONFIG, exp300, 8, 20261198)
    assert {key: len(value) for key, value in split.items()} == {"train": 24, "validation": 8, "test": 8}
    assert not (set(split["train"]) & set(split["validation"]) | set(split["train"]) & set(split["test"]) | set(split["validation"]) & set(split["test"]))
    sampler = exp300.BigIntSampler(split["train"], 4); state, other, modulus, bit, target, _ = sampler.transitions(5)
    assert torch.equal(bits(target, 8).bool(), bits([(2 * a + b * c) % p for a, c, p, b in zip(state, other, modulus, bit)], 8).bool())
    assert features(state, other, modulus, bit, 8).shape == (5, 6, 8)


def test_model_inference_path_contains_no_exact_arithmetic_implementation() -> None:
    source = inspect.getsource(UTMemoryTransition)
    tree = ast.parse(source)
    # Tensor boolean masks and PEP-604 annotations use ``&``/``|`` legally;
    # exact integer reduction/division/bit-shift operators remain prohibited.
    prohibited = (ast.Mod, ast.FloorDiv, ast.LShift, ast.RShift)
    assert not [node for node in ast.walk(tree) if isinstance(node, prohibited)]
    lowered = source.lower()
    assert "montgomery" not in lowered and "barrett" not in lowered and "crt" not in lowered


def test_safetensors_reload_is_bit_exact(tmp_path: Path) -> None:
    model = build_model(CONFIG, "memory_8"); path = tmp_path / "model.safetensors"
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
    reloaded = build_model(CONFIG, "memory_8"); reloaded.load_state_dict(load_file(path))
    assert torch.equal(model(sample(), memory_count=8).logits, reloaded(sample(), memory_count=8).logits)


def test_preregistered_roots_routes_counts_and_uncapped_calibration() -> None:
    assert CONFIG["roots"] == {"calibration": 20261198, "viability": 20261199, "confirmations": [20261200, 20261201]}
    assert tuple(CONFIG["calibration"]["route_classes"]) == ROUTES
    assert tuple(f"memory_{count}" for count in CONFIG["memory_counts"]) == SWEEP_ROUTES
    assert CONFIG["training"]["steps_per_arm"] == 300 and CONFIG["calibration"]["representative_updates_per_route"] == 2
    assert "cap" not in CONFIG["resource_policy"] and CONFIG["resource_policy"]["require_complete_route_classes"]


def test_selection_is_validation_only_and_gate_is_strictly_conjunctive() -> None:
    assert SWEEP_ROUTES == ROUTES[:6]
    base_rollout = {"exact_accuracy": 1.0, "any_invalid_residue_rate": 0.0}
    routes = {route: {"parameter_count": 10, "checkpoint_reload_exact": True, "training": {"examples": 100, "updates": 10, "block_calls": 60}, "test": {width: {"rollout": dict(base_rollout)} for width in ("16", "24", "32")}} for route in ROUTES}
    diagnostics = {"randomized_weights": {"exact_accuracy": 0.0}, "memory_interventions": {name: {"exact_accuracy": 0.0} for name in CONFIG["evaluation"]["memory_interventions"]}, "representation_interventions": {name: {"exact_accuracy": 0.0} for name in CONFIG["evaluation"]["representation_interventions"]}}
    # Equal control accuracy deliberately fails every required transfer-gain conjunct.
    gate = gate_summary(CONFIG, "memory_16", routes, diagnostics, True)
    assert not gate["passed"] and not gate["results"]["transfer_gain_each_control"]
