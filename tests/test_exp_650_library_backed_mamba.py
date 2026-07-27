from __future__ import annotations

import ast
import inspect
import io

import pytest
import torch
from transformers import Mamba2Config, Mamba2Model, __version__ as transformers_version
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

from experiments.exp_650_library_backed_mamba.model import (
    LibraryMamba2Hybrid,
    MatchedConvGRU,
    MatchedRecurrentAttention,
    PostNormResidual,
    TransposedSequenceMLP,
)
from experiments.exp_650_library_backed_mamba.run import (
    evaluate_cases,
    load_exp300,
    trajectory_batch,
)


def test_pinned_transformers_cpu_forward_backward_and_reload() -> None:
    assert transformers_version == "5.14.1"
    torch.manual_seed(650)
    config = Mamba2Config(
        hidden_size=16,
        state_size=8,
        num_hidden_layers=1,
        expand=2,
        head_dim=8,
        num_heads=4,
        n_groups=1,
        conv_kernel=4,
        vocab_size=32,
        use_cache=False,
        chunk_size=8,
    )
    model = Mamba2Model(config).cpu().train()
    inputs = torch.randn(2, 8, 16, requires_grad=True)
    output = model(inputs_embeds=inputs, use_cache=False).last_hidden_state
    output.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    reloaded = Mamba2Model(config).cpu().eval()
    reloaded.load_state_dict(
        torch.load(io.BytesIO(buffer.getvalue()), weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        first = model(inputs_embeds=inputs.detach(), use_cache=False).last_hidden_state
        second = reloaded(inputs_embeds=inputs.detach(), use_cache=False).last_hidden_state
    assert torch.equal(first, second)


def test_faithful_hybrid_has_two_library_mixers_and_postnorm_pipeline() -> None:
    model = LibraryMamba2Hybrid(width=16)
    assert len(model.mamba_blocks) == 2
    assert all(isinstance(block, Mamba2Mixer) for block in model.mamba_blocks)
    assert len(model.mamba_postnorm) == 2
    assert all(isinstance(norm, PostNormResidual) for norm in model.mamba_postnorm)
    assert isinstance(model.cross, torch.nn.MultiheadAttention)
    assert isinstance(model.cross_postnorm, PostNormResidual)
    assert isinstance(model.mlp_postnorm, PostNormResidual)


def test_hybrid_directions_and_mlpt_variant_are_deterministic() -> None:
    torch.manual_seed(11)
    features = torch.randn(2, 16, 6)
    model = LibraryMamba2Hybrid(width=16).eval()
    for direction in ("forward", "reverse", "bidirectional"):
        first = model(features, direction=direction)
        second = model(features, direction=direction)
        assert first.shape == (2, 16)
        assert torch.equal(first, second)
    mlpt = LibraryMamba2Hybrid(width=16, cross_mixer="mlp_t").eval()
    assert isinstance(mlpt.cross, TransposedSequenceMLP)
    assert mlpt(features).shape == (2, 16)
    with pytest.raises(ValueError, match="cannot transfer"):
        mlpt(torch.randn(2, 24, 6))


def test_controls_are_parameter_matched_and_all_models_backpropagate() -> None:
    torch.manual_seed(12)
    models = (
        LibraryMamba2Hybrid(width=16),
        MatchedConvGRU(width=16, hidden=48, cycles=2),
        MatchedRecurrentAttention(
            width=16, hidden=52, heads=4, feedforward=104, iterations=2
        ),
    )
    target = models[0].parameter_count()
    for model in models:
        ratio = model.parameter_count() / target
        assert 1 / 1.10 <= ratio <= 1.10
        output = model(torch.randn(2, 16, 6))
        output.square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_learned_forward_paths_contain_no_exact_arithmetic_operators() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            LibraryMamba2Hybrid._mamba_flow,
            LibraryMamba2Hybrid.forward,
            MatchedConvGRU.recurrent_update,
            MatchedConvGRU.forward,
            MatchedRecurrentAttention.forward,
        )
    )
    tree = ast.parse(inspect.cleandoc(source))
    forbidden = {
        ast.Mod,
        ast.FloorDiv,
        ast.LShift,
        ast.RShift,
        ast.BitAnd,
        ast.BitXor,
    }
    observed = {
        type(node.op)
        for node in ast.walk(tree)
        if isinstance(node, (ast.BinOp, ast.AugAssign))
    }
    assert forbidden.isdisjoint(observed)


def test_existing_exact_horner_trajectory_contract_is_reused() -> None:
    class FixedSampler:
        def pairs(self, size: int):
            assert size == 1
            return [11], [7], [13]

    exp300 = load_exp300()
    batch = trajectory_batch(FixedSampler(), 4, 1, exp300)
    assert batch["features"].shape == (4, 4, 6)
    assert batch["targets"].shape == (4, 4)
    decoded = [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in batch["targets"].tolist()
    ]
    assert decoded == [7, 1, 9, 12]


def test_exact_rollout_evaluator_reports_valid_zero_trajectory() -> None:
    class AlwaysZero(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full(features.shape[:2], -1.0)

    metrics = evaluate_cases(
        AlwaysZero(),
        load_exp300(),
        {"left": [0, 0], "right": [3, 5], "modulus": [11, 13]},
        width=4,
        batch_size=2,
    )
    assert metrics["exact_accuracy"] == 1.0
    assert metrics["transition_exact_accuracy"] == 1.0
    assert metrics["all_prefix_exact_accuracy"] == 1.0
    assert metrics["any_invalid_residue_rate"] == 0.0
    assert metrics["first_error"]["error_cases"] == 0
