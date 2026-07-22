from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_659_symbol_equivariant_recurrence.model import (
    SymbolRecurrentReasoner,
)
from experiments.exp_659_symbol_equivariant_recurrence.run import (
    Experiment,
    bits,
    features,
)


def make_model(arm: str) -> SymbolRecurrentReasoner:
    torch.manual_seed(20261177)
    return SymbolRecurrentReasoner(
        arm,
        hidden=16,
        position_hidden=20,
        heads=4,
        recurrent_steps=2,
        expansion=2,
    ).eval()


@pytest.mark.parametrize("arm", ["position_only", "symbol_equivariant", "broken_tying"])
@pytest.mark.parametrize("width", [8, 12])
def test_forward_shapes_and_width_flexibility(arm: str, width: int) -> None:
    model = make_model(arm)
    tensor = features([0, 1], [3, 5], [251, 241], [0, 1], width)
    logits, quotient = model(tensor)
    assert logits.shape == (2, width)
    assert quotient.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_faithful_core_is_equivariant_for_both_binary_permutations() -> None:
    model = make_model("symbol_equivariant")
    generator = torch.Generator().manual_seed(4)
    state = torch.randn(3, 9, 2, model.hidden, generator=generator)
    base = model.core_operator(state)
    for permutation in (torch.tensor([0, 1]), torch.tensor([1, 0])):
        observed = model.core_operator(state[:, :, permutation, :])
        expected = base[:, :, permutation, :]
        assert torch.allclose(observed, expected, atol=1e-6, rtol=0.0)


def test_broken_tying_is_a_causal_counterexample() -> None:
    model = make_model("broken_tying")
    generator = torch.Generator().manual_seed(4)
    state = torch.randn(3, 9, 2, model.hidden, generator=generator)
    permutation = torch.tensor([1, 0])
    base = model.core_operator(state)
    observed = model.core_operator(state[:, :, permutation, :])
    expected = base[:, :, permutation, :]
    assert float((observed - expected).abs().max()) >= 1e-4


def test_arithmetic_task_bit_swap_is_truthfully_non_applicable() -> None:
    experiment = object.__new__(Experiment)
    result = experiment.task_symmetry_applicability()
    assert result["legal_cases_exhausted"] > 0
    assert result["swap_transformed_legal_cases"] == 0
    assert result["verdict"] == "non-applicable-domain-not-closed"


def test_malformed_inputs_fail_closed() -> None:
    model = make_model("symbol_equivariant")
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 8))
    with pytest.raises(ValueError, match="6 feature roles"):
        model(torch.zeros(2, 5, 8))
    malformed = torch.zeros(2, 6, 8)
    malformed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(malformed)


def test_tiny_train_and_eval_fixture_is_deterministic() -> None:
    torch.use_deterministic_algorithms(True)
    input_tensor = features([0, 1, 2, 3], [3, 3, 5, 5], [251, 251, 241, 241], [0, 1, 0, 1], 8)
    targets = bits([0, 4, 4, 11], 8)

    snapshots = []
    predictions = []
    for _ in range(2):
        torch.manual_seed(20261177)
        model = SymbolRecurrentReasoner(
            "symbol_equivariant",
            hidden=16,
            position_hidden=20,
            heads=4,
            recurrent_steps=1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _step in range(2):
            logits, _ = model(input_tensor)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        predictions.append(model(input_tensor)[0].detach())

    assert all(
        torch.equal(snapshots[0][key], snapshots[1][key]) for key in snapshots[0]
    )
    assert torch.equal(predictions[0], predictions[1])


def test_inference_module_has_no_encoded_arithmetic_or_answer_access() -> None:
    path = Path("experiments/exp_659_symbol_equivariant_recurrence/model.py")
    source = path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "expected_answer",
        "barrett",
        "montgomery",
        "crt_recomb",
        "torch.remainder",
        "fmod(",
        "data/checked",
        "data/private",
        "submission/",
    ):
        assert forbidden not in source


def test_parameter_matching_is_within_frozen_range() -> None:
    values = {
        arm: SymbolRecurrentReasoner(
            arm,
            hidden=32,
            position_hidden=36,
            heads=4,
            recurrent_steps=2,
        ).parameter_count()
        for arm in ("position_only", "symbol_equivariant", "broken_tying")
    }
    faithful = values["symbol_equivariant"]
    for arm in ("position_only", "broken_tying"):
        assert 0.75 <= values[arm] / faithful <= 1.25
