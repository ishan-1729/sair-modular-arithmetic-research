from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_transition_at_matches_horner_teacher() -> None:
    replay = load_module(
        "balanced_first_error_replay_test",
        ROOT / "experiments/exp_629_tier6_balanced_first_error_replay/replay.py",
    )
    left, right, modulus, width = 13, 11, 17, 4
    records = [
        replay.exact_transition_at(left, right, modulus, position, width)
        for position in range(width)
    ]
    assert [record["bit"] for record in records] == [1, 1, 0, 1]
    assert [record["state"] for record in records] == [0, 11, 16, 15]
    assert [record["target"] for record in records] == [11, 16, 15, 7]
    assert all(
        record["target"]
        == (2 * record["state"] + record["bit"] * right) % modulus
        for record in records
    )


def test_exact_transition_rejects_out_of_range_position() -> None:
    replay = load_module(
        "balanced_first_error_replay_range_test",
        ROOT / "experiments/exp_629_tier6_balanced_first_error_replay/replay.py",
    )
    try:
        replay.exact_transition_at(1, 1, 3, 4, 4)
    except ValueError as error:
        assert "position" in str(error)
    else:
        raise AssertionError("expected ValueError")
