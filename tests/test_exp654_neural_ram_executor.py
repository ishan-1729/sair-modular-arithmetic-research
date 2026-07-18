from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neural_ram_shapes_gradients_diagnostics_and_width_sharing() -> None:
    module = load_module(
        "exp654_neural_ram_test",
        ROOT / "experiments/exp_654_neural_ram_executor/model.py",
    )
    torch.manual_seed(654)
    model = module.NeuralRAMExecutor(
        width=8,
        canvas_hidden=12,
        controller_hidden=12,
        memory_dim=8,
        register_slots=4,
        attention_heads=2,
        condition_dim=4,
    )
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    left, right, modulus = [3, 17, 29], [5, 11, 7], [31, 29, 23]
    logits, diagnostics = model(
        model.bits(left, 8).flip(1),
        model.static_features(right, modulus, 8),
        return_all=True,
        return_diagnostics=True,
    )
    assert logits.shape == (3, 8, 8)
    assert all(value.shape == (3, 8) for value in diagnostics.values())
    logits.square().mean().backward()
    assert model.register_initial.grad is not None
    assert model.register_initial.grad.abs().sum() > 0
    assert model.read_attention.in_proj_weight.grad is not None
    assert model.read_attention.in_proj_weight.grad.abs().sum() > 0

    width16 = module.NeuralRAMExecutor(
        width=16,
        canvas_hidden=12,
        controller_hidden=12,
        memory_dim=8,
        register_slots=4,
        attention_heads=2,
        condition_dim=4,
    )
    width16.load_state_dict(initial)
    final = width16(
        width16.bits(left, 16).flip(1),
        width16.static_features(right, modulus, 16),
    )
    assert final.shape == (3, 16)
    assert width16.parameter_count() == model.parameter_count()


def test_memory_ablations_are_active_and_shape_preserving() -> None:
    module = load_module(
        "exp654_neural_ram_ablation_test",
        ROOT / "experiments/exp_654_neural_ram_executor/model.py",
    )
    torch.manual_seed(655)
    model = module.NeuralRAMExecutor(
        width=8,
        canvas_hidden=12,
        controller_hidden=12,
        memory_dim=8,
        register_slots=4,
        attention_heads=2,
        condition_dim=4,
    ).eval()
    bits = model.bits([3, 7], 8).flip(1)
    static = model.static_features([5, 11], [13, 17], 8)
    outputs = {
        ablation: model(bits, static, return_all=True, memory_ablation=ablation)
        for ablation in model.ABLATIONS
    }
    assert all(output.shape == (2, 8, 8) for output in outputs.values())
    assert any(
        not torch.equal(outputs["none"], outputs[ablation])
        for ablation in ("reset", "permute", "uniform")
    )


def test_parameter_matching_and_frozen_config() -> None:
    neural = load_module(
        "exp654_neural_ram_parameter_test",
        ROOT / "experiments/exp_654_neural_ram_executor/model.py",
    )
    cellular = load_module(
        "exp654_cellular_parameter_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    persistent = load_module(
        "exp654_persistent_parameter_test",
        ROOT / "experiments/exp_620_persistent_latent_neural_gpu/model.py",
    )
    ram_count = neural.NeuralRAMExecutor(width=16).parameter_count()
    wider_count = cellular.SharedDilatedConvGRU(width=16, hidden=80, cycles=2).parameter_count()
    workspace_count = persistent.PersistentLatentNeuralGPU(
        width=16, hidden=88, microsteps=2
    ).parameter_count()
    assert ram_count == 74865
    assert 0.97 <= ram_count / wider_count <= 1.03
    assert 0.99 <= ram_count / workspace_count <= 1.01

    config = yaml.safe_load(
        (ROOT / "configs/exp_654_neural_ram_executor_viability.yaml").read_text()
    )
    assert config["training"]["total_steps"] == 4000
    assert config["data"]["train_prime_identities"] == 32
    assert config["data"]["test_prime_identities"] == 16
    assert config["data"]["calibration_seed"] != config["data"]["formal_seed"]
    assert config["resource_budget"]["calibration_status"] in {"pending", "complete"}
    assert config["resource_budget"]["fidelity_reduced"] is False


def test_infeasible_prime_identity_request_fails_explicitly() -> None:
    runner = load_module(
        "exp654_runner_capacity_test",
        ROOT / "experiments/exp_654_neural_ram_executor/run.py",
    )
    config = yaml.safe_load(
        (ROOT / "configs/exp_654_neural_ram_executor_viability.yaml").read_text()
    )
    config["data"]["train_prime_identities"] = 64
    config["data"]["test_prime_identities"] = 32
    try:
        runner.make_split(config, runner.load_exp300(), 8, 654)
    except ValueError as error:
        assert "band contains 48" in str(error)
    else:
        raise AssertionError("infeasible unique-prime request must fail")
