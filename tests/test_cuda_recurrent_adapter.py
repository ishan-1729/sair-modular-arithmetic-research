from __future__ import annotations

import ast
import copy
import inspect
import textwrap

import torch

from experiments.cuda_portability import inventory, recurrent
from experiments.exp_654_neural_ram_executor import model as frozen


def make_model() -> frozen.NeuralRAMExecutor:
    return frozen.NeuralRAMExecutor(
        width=8,
        canvas_hidden=12,
        controller_hidden=12,
        memory_dim=8,
        register_slots=4,
        attention_heads=2,
        condition_dim=4,
    ).eval()


def test_exp654_adapter_matches_frozen_cpu_semantics() -> None:
    torch.manual_seed(6540)
    reference = make_model()
    portable = recurrent.Exp654NeuralRAMAdapter(copy.deepcopy(reference)).eval()
    left = [3, 17, 29]
    right = [5, 11, 7]
    modulus = [31, 29, 23]
    prepared = portable.prepare(left, right, modulus, 8)
    for ablation in sorted(reference.ABLATIONS):
        expected = reference(
            reference.bits(left, 8).flip(1),
            reference.static_features(right, modulus, 8),
            return_all=True,
            return_diagnostics=True,
            memory_ablation=ablation,
        )
        actual = portable(
            prepared,
            return_all=True,
            return_diagnostics=True,
            memory_ablation=ablation,
        )
        assert torch.equal(actual[0], expected[0])
        assert actual[1].keys() == expected[1].keys()
        assert all(
            torch.equal(actual[1][key], expected[1][key]) for key in actual[1]
        )


def test_exp654_preparation_derives_device_from_model() -> None:
    portable = recurrent.Exp654NeuralRAMAdapter(make_model())
    prepared = portable.prepare([3, 7], [5, 11], [13, 17], 8)
    device = next(portable.model.parameters()).device
    assert prepared.multiplier_bits.device == device
    assert prepared.static_features.device == device
    assert torch.equal(
        prepared.multiplier_bits,
        portable.model.bits([3, 7], 8).flip(1),
    )
    assert torch.equal(
        prepared.static_features,
        portable.model.static_features([5, 11], [13, 17], 8),
    )
    expected_positions = torch.stack(
        [
            portable.model.position_features(
                position,
                8,
                device=device,
                dtype=prepared.static_features.dtype,
            )
            for position in range(8)
        ]
    )
    assert torch.equal(prepared.position_vectors, expected_positions)


def test_exp654_portable_loop_has_no_transfer_or_python_tensor_factory() -> None:
    forbidden = {
        "cpu",
        "cuda",
        "item",
        "numpy",
        "to",
        "tolist",
        "tensor",
        "as_tensor",
        "from_numpy",
    }
    for function in (
        recurrent.neural_ram_forward_prepared,
        recurrent.Exp654NeuralRAMAdapter.forward,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for loop in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.For, ast.While))
        ):
            calls = {
                node.func.attr
                for node in ast.walk(loop)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            assert forbidden.isdisjoint(calls)


def test_inventory_detects_frozen_exp654_nested_position_factory() -> None:
    violations = inventory.audit_hot_loop(
        inventory.ROOT / "experiments/exp_654_neural_ram_executor/model.py",
        {"forward"},
        {"cpu", "cuda", "item", "numpy", "to", "tolist"},
        {"tensor", "as_tensor", "from_numpy"},
    )
    assert any(
        row["kind"] == "python-backed-tensor-factory"
        and row["call"] == "torch.tensor"
        and row["via"] == "self.position_features"
        for row in violations
    )
