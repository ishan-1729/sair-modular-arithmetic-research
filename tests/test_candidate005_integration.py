from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from tools.build_candidate005_artifact import candidate_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def candidate_module():
    incumbent = (ROOT / "submission/model.py").read_text(encoding="utf-8")
    source = candidate_model(incumbent)
    module = types.ModuleType("candidate005_test_model")
    exec(compile(source, "candidate005_test_model.py", "exec"), module.__dict__)
    return module


def test_candidate005_bigint_encoder_is_bit_exact(candidate_module) -> None:
    state = [0, (1 << 96) + 7]
    multiplicand = [(1 << 127) - 3, (1 << 65) + 11]
    modulus = [(1 << 127) + 51, (1 << 126) + 27]
    input_bit = [0, 1]
    actual = candidate_module._bigint_features(
        state, multiplicand, modulus, input_bit, 128
    )
    expected = torch.empty((2, 6, 128), dtype=torch.float32)
    for row in range(2):
        for position in range(128):
            expected[row, 0, position] = (state[row] >> position) & 1
            expected[row, 1, position] = (multiplicand[row] >> position) & 1
            expected[row, 2, position] = (modulus[row] >> position) & 1
        expected[row, 3, :] = input_bit[row]
        expected[row, 4, :] = 0
        expected[row, 5, :] = 0
        expected[row, 4, 0] = 1
        expected[row, 5, -1] = 1
    assert torch.equal(actual, expected.mul(2.0).sub(1.0))


def test_candidate005_routes_lowest_40_percent_and_uses_strict_margin(
    candidate_module,
) -> None:
    model = candidate_module.Tier2HornerModel()
    model.network = object()
    model.tier3_network = object()
    model.tier4_network = object()
    model.tier5_network = object()
    model.tier6_network = object()
    calls = 0

    def fake_rollout(self, multiplier, multiplicand, modulus):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [10, 11, 12, 13, 14], [0.5, 0.1, 0.2, 0.4, 0.3]
        assert len(multiplier) == 2
        return [90, 91], [0.15, 0.2]

    model._tier6_rollout = types.MethodType(fake_rollout, model)
    modulus = (1 << 80) + 7
    rows = [(str(100 + i), str(200 + i), str(modulus)) for i in range(5)]
    outputs = model.predict_digits_batch(rows)
    assert calls == 2
    # Rows 1 and 2 have the lowest original margins. Row 1 switches because
    # 0.15 > 0.1; row 2 retains original on the strict 0.2 == 0.2 tie.
    assert outputs == [[10], [90], [12], [13], [14]]
