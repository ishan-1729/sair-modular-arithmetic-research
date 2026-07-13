from __future__ import annotations

import ast
import inspect

import pytest
import torch

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_633_hierarchical_learned_composition_feasibility import (
    model as tree_module,
)
from experiments.exp_633_hierarchical_learned_composition_feasibility.model import (
    HierarchicalCompositionTree,
    ints_to_bits,
    make_static_features,
)
from experiments.exp_633_hierarchical_learned_composition_feasibility.run import (
    trajectory_batch,
)


def test_tree_shapes_all_levels_and_parameter_gate() -> None:
    model = HierarchicalCompositionTree(width=16, hidden=56, cycles=1)
    multiplier_bits = torch.randint(0, 2, (3, 16), dtype=torch.float32)
    static = make_static_features([3, 5, 7], [11, 13, 17], 16)
    root, levels = model(multiplier_bits, static, return_all_levels=True)

    assert root.shape == (3, 16)
    assert [tuple(item.shape) for item in levels] == [
        (3, 8, 16),
        (3, 4, 16),
        (3, 2, 16),
        (3, 1, 16),
    ]
    control = SharedDilatedConvGRU(width=16, hidden=92, cycles=2)
    assert model.parameter_count() == 94_025
    assert model.parameter_count() / control.parameter_count() < 1.15


def test_tree_is_deterministic_and_all_parameters_receive_gradients() -> None:
    torch.manual_seed(7)
    model = HierarchicalCompositionTree(width=8, hidden=12, cycles=1)
    multiplier_bits = torch.randint(0, 2, (2, 8), dtype=torch.float32)
    static = make_static_features([5, 6], [11, 13], 8)

    first = model(multiplier_bits, static)
    second = model(multiplier_bits, static)
    assert torch.equal(first, second)
    first.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_bit_features_are_lsb_first_and_bounded() -> None:
    encoded = ints_to_bits([0, 5, 7], 4)
    assert encoded.tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
    ]
    static = make_static_features([5], [7], 4)
    assert static.shape == (1, 4, 4)
    assert set(static.unique().tolist()) == {-1.0, 1.0}


def test_offline_tree_targets_match_segment_products() -> None:
    class BitLabels:
        @staticmethod
        def bits(values: list[int], width: int) -> torch.Tensor:
            return ints_to_bits(values, width)

    class FixedSampler:
        @staticmethod
        def pairs(size: int):
            assert size == 1
            return [3], [5], [7]

    batch = trajectory_batch(
        BitLabels(), tree_module, FixedSampler(), width=4, batch_size=1
    )
    assert batch["control_targets"] == [0, 0, 5, 1]
    assert batch["tree_level_targets"][0].shape == (1, 2, 4)
    assert batch["tree_level_targets"][0][0].tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    assert batch["tree_level_targets"][1][0, 0].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize("width", [1, 3, 6, 12])
def test_tree_rejects_non_power_of_two_width(width: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        HierarchicalCompositionTree(width=width)


def test_inference_model_contains_no_exact_modular_operator() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            HierarchicalCompositionTree.recurrent_update,
            HierarchicalCompositionTree.decode,
            HierarchicalCompositionTree.forward,
        )
    )
    tree = ast.parse(inspect.cleandoc(source))
    forbidden = {ast.LShift, ast.RShift, ast.BitAnd, ast.BitXor}
    observed = {
        type(node.op)
        for node in ast.walk(tree)
        if isinstance(node, (ast.BinOp, ast.AugAssign))
    }
    assert forbidden.isdisjoint(observed)
    modulo_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
    ]
    assert modulo_nodes == []
