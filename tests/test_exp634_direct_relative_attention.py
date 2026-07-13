from __future__ import annotations

import ast
import inspect

import pytest
import torch

from experiments.exp_634_direct_relative_attention_feasibility.model import (
    DirectDilatedConvGRU,
    DirectRelativeAttention,
    RecurrentAttentionBlock,
    ints_to_bits,
    make_features,
)


def test_matched_models_have_expected_shapes_and_parameter_ratio() -> None:
    attention = DirectRelativeAttention(width=12)
    control = DirectDilatedConvGRU(width=12)
    features = make_features([3, 5, 7], [2, 4, 6], [11, 13, 17], 12)

    assert attention(features).shape == (3, 12)
    assert control(features).shape == (3, 12)
    assert attention.parameter_count() == 53_122
    assert control.parameter_count() == 52_345
    assert attention.parameter_count() / control.parameter_count() < 1.15


def test_attention_depth_scales_and_parameters_are_width_shared() -> None:
    model = DirectRelativeAttention(width=12)
    assert model.iterations(8) == 6
    assert model.iterations(12) == 8
    assert model.iterations(24) == 10
    parameters = model.parameter_count()
    output = model(make_features([17], [19], [23], 24))
    assert output.shape == (1, 24)
    assert model.parameter_count() == parameters


@pytest.mark.parametrize("model_type", [DirectRelativeAttention, DirectDilatedConvGRU])
def test_models_are_deterministic_and_all_parameters_receive_gradients(model_type) -> None:
    torch.manual_seed(11)
    model = model_type(width=8)
    features = make_features([3, 5], [6, 7], [11, 13], 8)
    first = model(features)
    second = model(features)
    assert torch.equal(first, second)
    first.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_fixed_position_features_are_width_relative_and_bounded() -> None:
    features = make_features([5], [6], [7], 5)
    assert features.shape == (1, 5, 7)
    assert ints_to_bits([5], 5).tolist() == [[1.0, 0.0, 1.0, 0.0, 0.0]]
    assert features[0, 0, 5].item() == pytest.approx(0.0, abs=1e-7)
    assert features[0, -1, 5].item() == pytest.approx(0.0, abs=1e-6)
    assert features[0, 0, 6].item() == pytest.approx(1.0)
    assert features[0, -1, 6].item() == pytest.approx(-1.0)
    assert float(features.abs().max()) <= 1.0


def test_attention_uses_pytorch_library_operator() -> None:
    model = DirectRelativeAttention(width=8)
    assert isinstance(model.block, RecurrentAttentionBlock)
    assert isinstance(model.block.attention, torch.nn.MultiheadAttention)


def test_learned_forward_paths_contain_no_exact_arithmetic_operator() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            DirectRelativeAttention.forward,
            DirectDilatedConvGRU.recurrent_update,
            DirectDilatedConvGRU.forward,
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
