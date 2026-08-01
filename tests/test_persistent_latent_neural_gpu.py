from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_persistent_neural_gpu_shapes_gradients_and_width_sharing() -> None:
    persistent = load_module(
        "persistent_neural_gpu_test_model",
        ROOT / "experiments/exp_620_persistent_latent_neural_gpu/model.py",
    )
    torch.manual_seed(620)
    model = persistent.PersistentLatentNeuralGPU(width=8, hidden=12, microsteps=2)
    state_dict = {key: value.clone() for key, value in model.state_dict().items()}
    left = [3, 17, 29]
    right = [5, 11, 7]
    modulus = [31, 29, 23]
    bits8 = persistent.PersistentLatentNeuralGPU.bits(left, 8).flip(1)
    static8 = model.static_features(right, modulus, 8)
    all_logits = model(bits8, static8, return_all=True)
    assert all_logits.shape == (3, 8, 8)
    loss = all_logits.square().mean()
    loss.backward()
    assert model.gates.weight.grad is not None
    assert model.gates.weight.grad.abs().sum() > 0
    assert model.local_head.weight.grad is not None
    assert model.local_head.weight.grad.abs().sum() > 0

    width16 = persistent.PersistentLatentNeuralGPU(width=16, hidden=12, microsteps=2)
    width16.load_state_dict(state_dict)
    bits16 = persistent.PersistentLatentNeuralGPU.bits(left, 16).flip(1)
    static16 = width16.static_features(right, modulus, 16)
    final16 = width16(bits16, static16)
    assert final16.shape == (3, 16)
    assert model.parameter_count() == width16.parameter_count()


def test_persistent_state_is_not_decoded_and_reinjected() -> None:
    source = (
        ROOT / "experiments/exp_620_persistent_latent_neural_gpu/model.py"
    ).read_text()
    forward_body = source.split("def forward(", 1)[1]
    assert ".ge(0)" not in forward_body
    assert "decode(state)" in forward_body
