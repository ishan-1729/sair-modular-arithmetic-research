from __future__ import annotations

import math
import os

import pytest
import torch
from safetensors.torch import save_file

from experiments.cloud_benchmarks.exp699_cuda_replay import run as replay
from experiments.cuda_portability.graph import Exp510TypedMaxAdapter
from experiments.cuda_portability.recurrent import Exp654NeuralRAMAdapter
from experiments.cuda_portability.runtime import (
    load_safetensors_on_device,
    module_device,
    resolve_device,
    runtime_receipt,
    synchronize,
)
from experiments.exp_100_width_scaling.model import SharedContextHornerTCN
from experiments.exp_401_pyg_graph_message_passing import model as graph_frozen
from experiments.exp_654_neural_ram_executor import model as ram_frozen
from experiments.exp_699_h92_w256_backbone_factorial_complete import run as exp699


pytestmark = pytest.mark.skipif(
    os.environ.get("SAIR_CUDA_ACCEPTANCE") != "1",
    reason="run only through the fail-closed CUDA acceptance command",
)


class _Deadline:
    def check(self, stage: str) -> None:
        assert stage


def _device() -> torch.device:
    return resolve_device(
        os.environ.get("SAIR_CUDA_DEVICE", "cuda"),
        require_accelerator=True,
    )


def test_cuda_runtime_and_exp699_update() -> None:
    device = _device()
    receipt = runtime_receipt(device)
    assert receipt["device"].startswith("cuda")
    assert receipt["gpu"]["total_memory_bytes"] > 0

    parent = exp699.base.load_exp300()
    adapter = replay.DeviceExp300(parent, device, _Deadline())
    model = SharedContextHornerTCN(width=16, hidden=8, blocks=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = {
        "training": {
            "horizon": 2,
            "worst_bit_weight": 0.45,
            "quotient_weight": 0.10,
            "gradient_clip": 2.0,
        }
    }
    stats = replay.train_update(
        config,
        adapter,
        model,
        optimizer,
        left=[11, 173],
        right=[5, 91],
        modulus=[251, 257],
        starts=[0, 3],
        arm="teacher_forced",
        width=16,
    )
    assert math.isfinite(stats["loss"])
    assert module_device(model) == device
    predictions, margins, first_errors, invalid = replay.rollout(
        adapter,
        model.eval(),
        left=[11, 173],
        right=[5, 91],
        modulus=[251, 257],
        width=16,
    )
    assert len(predictions) == len(first_errors) == len(invalid) == 2
    assert margins.shape == (2,)
    synchronize(device)


def test_cuda_exp510_typed_max_graph() -> None:
    device = _device()
    model = graph_frozen.PyGRecurrentTypedMaxMPNN(
        width=9, hidden=16, cycles=1
    ).to(device)
    portable = Exp510TypedMaxAdapter(model, graph_frozen).train()
    features = torch.randn(3, 6, 9, device=device, requires_grad=True)
    logits, quotient = portable(features)
    (logits.square().mean() + quotient.square().mean()).backward()
    assert logits.device == quotient.device == device
    assert features.grad is not None and features.grad.device == device
    assert portable.cache._edges
    assert all(
        edge_index.device == edge_attr.device == device
        for edge_index, edge_attr in portable.cache._edges.values()
    )
    synchronize(device)


def test_cuda_exp654_neural_ram() -> None:
    device = _device()
    model = ram_frozen.NeuralRAMExecutor(
        width=8,
        canvas_hidden=12,
        controller_hidden=12,
        memory_dim=8,
        register_slots=4,
        attention_heads=2,
        condition_dim=4,
    ).to(device)
    portable = Exp654NeuralRAMAdapter(model).train()
    prepared = portable.prepare([3, 17], [5, 11], [31, 29], 8)
    logits, diagnostics = portable(
        prepared,
        return_all=True,
        return_diagnostics=True,
        memory_ablation="none",
    )
    logits.square().mean().backward()
    assert logits.device == device
    assert diagnostics and all(value.device == device for value in diagnostics.values())
    assert any(parameter.grad is not None for parameter in model.parameters())
    synchronize(device)


def test_cuda_checkpoint_round_trip(tmp_path) -> None:
    device = _device()
    torch.manual_seed(20260803)
    source = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.SiLU(), torch.nn.Linear(8, 3))
    checkpoint = tmp_path / "runtime.safetensors"
    save_file(
        {name: value.detach().contiguous() for name, value in source.state_dict().items()},
        checkpoint,
    )
    target = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.SiLU(), torch.nn.Linear(8, 3))
    load_safetensors_on_device(target, checkpoint, device)
    assert module_device(target) == device
    actual = target(torch.randn(4, 6, device=device))
    assert actual.device == device and torch.isfinite(actual).all()
    synchronize(device)
