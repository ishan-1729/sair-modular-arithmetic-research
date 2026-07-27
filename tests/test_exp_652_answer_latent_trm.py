from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_652_answer_latent_trm.model import (
    AnswerLatentTRM,
    DecodedCellularControl,
    ExponentialMovingAverage,
    NoLatentControl,
    TinySharedNetwork,
)
from experiments.exp_652_answer_latent_trm.run import (
    ARM_CLASSES,
    calibration_admission,
    endpoint_targets,
    feature_tensor,
    scientific_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def small_features(batch: int = 3, width: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(652)
    return torch.randint(0, 2, (batch, 5, width), generator=generator).float().mul(2).sub(1)


def test_primary_has_fixed_answer_latent_and_one_two_layer_shared_network() -> None:
    model = AnswerLatentTRM(hidden=12, latent_updates=2, outer_recursions=3)
    assert isinstance(model.shared_network, TinySharedNetwork)
    assert len(model.shared_network.layers) == 2
    assert model.answer_initial.shape == model.latent_initial.shape == (1, 12, 1)
    output = model.deep_recursion(small_features(), supervision_steps=2)
    assert output.y.shape == output.z.shape == (3, 12, 8)
    assert output.logits.shape == (3, 8)
    assert output.q_logit.shape == (3,)
    assert len(output.answer_curve) == len(output.q_curve) == 2


def test_first_outer_blocks_are_forward_only_and_final_block_fully_backpropagates() -> None:
    model = AnswerLatentTRM(hidden=8, latent_updates=2, outer_recursions=3)
    gradient_modes: list[bool] = []
    hook = model.shared_network.register_forward_pre_hook(
        lambda _module, _inputs: gradient_modes.append(torch.is_grad_enabled())
    )
    output = model.deep_recursion(small_features(batch=2), supervision_steps=2)
    hook.remove()
    output.logits.square().mean().backward()
    assert gradient_modes == ([False] * 6 + [True] * 3) * 2
    assert all(parameter.grad is not None for parameter in model.shared_network.parameters())


def test_q_correctness_loss_and_ema_are_live_and_serializable(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = AnswerLatentTRM(hidden=8, latent_updates=1, outer_recursions=2)
    features = small_features(batch=4)
    targets = torch.zeros(4, 8)
    output = model.deep_recursion(features, supervision_steps=2)
    correctness = output.logits.detach().ge(0).eq(targets.bool()).all(dim=-1).float()
    loss = F.binary_cross_entropy_with_logits(output.logits, targets)
    loss += F.binary_cross_entropy_with_logits(output.q_logit, correctness)
    loss.backward()
    assert model.q_head.weight.grad is not None
    ema = ExponentialMovingAverage(model, 0.9)
    with torch.no_grad():
        next(model.parameters()).add_(0.25)
    ema.update(model)
    assert ema.updates == 1
    copied = ema.model_copy(model)
    path = tmp_path / "ema.safetensors"
    save_file({key: value.contiguous() for key, value in copied.state_dict().items()}, path)
    reloaded = AnswerLatentTRM(hidden=8, latent_updates=1, outer_recursions=2)
    reloaded.load_state_dict(load_file(path), strict=True)
    copied.eval(), reloaded.eval()
    with torch.inference_mode():
        assert torch.equal(copied(features), reloaded(features))


def test_controls_match_parameters_and_shared_call_work() -> None:
    models = [
        AnswerLatentTRM(hidden=8, latent_updates=2, outer_recursions=3),
        DecodedCellularControl(hidden=8, latent_updates=2, outer_recursions=3),
        NoLatentControl(hidden=8, latent_updates=2, outer_recursions=3),
    ]
    assert len({model.parameter_count() for model in models}) == 1
    calls = []
    for model in models:
        count = 0

        def increment(_module, _inputs):
            nonlocal count
            count += 1

        hook = model.shared_network.register_forward_pre_hook(increment)
        model.deep_recursion(small_features(batch=2), supervision_steps=1)
        hook.remove()
        calls.append(count)
    assert calls == [9, 9, 9]


def test_state_interventions_are_deterministic_and_causally_distinct() -> None:
    torch.manual_seed(3)
    model = AnswerLatentTRM(hidden=8, latent_updates=1, outer_recursions=2).eval()
    features = small_features(batch=2)
    with torch.inference_mode():
        baseline = model.deep_recursion(features).logits
        first = model.deep_recursion(
            features,
            state_intervention="randomize_z",
            generator=torch.Generator().manual_seed(99),
        ).logits
        second = model.deep_recursion(
            features,
            state_intervention="randomize_z",
            generator=torch.Generator().manual_seed(99),
        ).logits
        reset = model.deep_recursion(features, state_intervention="reset_z").logits
    assert torch.equal(first, second)
    assert not torch.equal(first, baseline)
    assert not torch.equal(reset, baseline)


def test_legal_inputs_and_endpoint_labels_have_no_identity_or_answer_channel() -> None:
    cases = [
        {"left": 11, "right": 7, "modulus": 13},
        {"left": 5, "right": 9, "modulus": 11},
    ]
    features = feature_tensor(cases, 4)
    targets = endpoint_targets(cases, 4)
    assert features.shape == (2, 5, 4)
    assert targets.shape == (2, 4)
    assert set(features.unique().tolist()) <= {-1.0, 1.0}
    decoded = [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in targets.tolist()
    ]
    assert decoded == [(11 * 7) % 13, (5 * 9) % 11]


def test_learned_forward_paths_contain_no_prohibited_exact_arithmetic() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            AnswerLatentTRM.full_recursion,
            AnswerLatentTRM.deep_recursion,
            AnswerLatentTRM.forward,
            NoLatentControl.full_recursion,
            DecodedCellularControl.full_recursion,
        )
    )
    tree = ast.parse(inspect.cleandoc(source))
    forbidden = {ast.Mod, ast.FloorDiv, ast.LShift, ast.RShift, ast.BitAnd, ast.BitXor}
    observed = {
        type(node.op)
        for node in ast.walk(tree)
        if isinstance(node, (ast.BinOp, ast.AugAssign))
    }
    assert forbidden.isdisjoint(observed)


def test_dynamic_calibration_profiles_all_routes_and_ignores_no_static_wall_cap() -> None:
    config = yaml.safe_load((ROOT / "configs/exp_652_answer_latent_trm.yaml").read_text())
    arms = {}
    for name in config["calibration"]["route_classes"]:
        arms[name] = {
            "resources": {
                "training_wall_seconds": 2.0,
                "training_process_cpu_seconds": 4.0,
                "training_cgroup_cpu_seconds": 5.0,
                "update_wall_seconds": [0.9, 1.1],
                "peak_rss_bytes": 1024,
            },
            "reload_exact": {"raw": True, "ema": True},
        }
    admission = calibration_admission(arms, 20.0, 40.0, 50.0, config)
    assert admission["admitted"]
    assert admission["checks"]["all_route_classes"]
    assert admission["projected_confirmation_route"]["hard_wall_seconds"] > admission[
        "projected_confirmation_route"
    ]["soft_wall_seconds"]
    assert "admission_maximum_expected_wall_seconds" not in config["resource_policy"]


def test_frozen_roots_gates_and_arm_classes_are_complete() -> None:
    config = yaml.safe_load((ROOT / "configs/exp_652_answer_latent_trm.yaml").read_text())
    assert config["roots"] == {
        "calibration": 20261190,
        "viability": 20261191,
        "confirmations": [20261192, 20261193],
    }
    assert config["gates"]["minimum_primary_w16_exact_accuracy"] >= 0.80
    assert config["gates"]["minimum_primary_w24_exact_accuracy"] >= 0.05
    assert config["gates"]["minimum_primary_w32_exact_accuracy"] >= 0.05
    assert set(config["calibration"]["route_classes"]) == set(ARM_CLASSES)


def test_scientific_gate_is_conjunctive() -> None:
    config = yaml.safe_load((ROOT / "configs/exp_652_answer_latent_trm.yaml").read_text())

    def arm(value16: float, value24: float, value32: float):
        metrics = {
            "16": {"exact_accuracy": value16, "invalid_rate": 0.0},
            "24": {"exact_accuracy": value24, "invalid_rate": 0.0},
            "32": {"exact_accuracy": value32, "invalid_rate": 0.0},
        }
        return {
            "parameters": 100,
            "evaluation": {"raw": metrics, "ema": metrics},
            "reload_exact": {"raw": True, "ema": True},
            "training": {"optimizer_updates": 10, "trajectory_examples": 320},
        }

    arms = {
        "answer_latent_trm": arm(0.90, 0.20, 0.20),
        "decoded_cellular_control": arm(0.60, 0.10, 0.10),
        "no_latent_control": arm(0.60, 0.10, 0.10),
    }
    diagnostics = {
        "state_interventions": {
            "reset_z": {"exact_accuracy": 0.60},
            "randomize_z": {"exact_accuracy": 0.60},
            "swap_y_z": {"exact_accuracy": 0.60},
        },
        "randomized_weights": {"exact_accuracy": 0.0},
    }
    assert scientific_gate(arms, diagnostics, config)["passed"]
    arms["answer_latent_trm"]["evaluation"]["ema"]["32"]["exact_accuracy"] = 0.0
    assert not scientific_gate(arms, diagnostics, config)["passed"]
