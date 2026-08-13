from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from tools import build_candidate007_current_rules_artifact as builder


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_preserves_frozen_weights_and_removes_prereduction(tmp_path: Path) -> None:
    output = tmp_path / "candidate007"
    receipt = builder.build(output)
    source = (output / "model.py").read_text()
    assert receipt["training_executed"] is False
    assert receipt["weights_unchanged"] is True
    assert "% modulus" not in source
    assert "% modulus_value" not in source
    assert "predicted if predicted" not in source
    assert "ACTIVE_MODULUS_BITS = 64" in source
    ast.parse(source)


def test_learned_raw_reducer_solves_nontrivial_small_cases(tmp_path: Path) -> None:
    output = tmp_path / "candidate007"
    builder.build(output)
    module = load_module("candidate007_test", output / "model.py")
    model = module.Tier2HornerModel()
    model.load(str(output))
    rows = [
        (str((1 << 31) + 123), str((1 << 30) + 77), "7"),
        (str((1 << 47) + 901), str((1 << 46) + 313), "251"),
    ]
    predicted = [digits[0] for digits in model.predict_digits_batch(rows)]
    expected = [(int(a) * int(b)) % int(p) for a, b, p in rows]
    assert predicted == expected


def test_unsupported_width_uses_constant_fallback(tmp_path: Path) -> None:
    output = tmp_path / "candidate007"
    builder.build(output)
    module = load_module("candidate007_fallback", output / "model.py")
    model = module.Tier2HornerModel()
    model.load(str(output))
    assert model.predict_digits_batch([("3", "4", str((1 << 64) + 13))]) == [[0]]
    assert model.predict_digits_batch([("3", "4", str((1 << 129) + 25))]) == [[0]]
