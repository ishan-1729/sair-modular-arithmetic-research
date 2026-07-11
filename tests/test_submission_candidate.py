from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from modchallenge.evaluation.loader import load_model, validate_manifest
from modchallenge.security.static_check import check_submission
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
WEIGHTS_SHA256 = "22c78b5d1db0136a8cb93d6c010cce025c251d70b62baef79bebc50e016af585"


@pytest.fixture(scope="module")
def candidate():
    manifest = validate_manifest(SUBMISSION)
    return manifest, load_model(SUBMISSION, manifest)


def test_submission_is_the_clean_three_file_artifact() -> None:
    assert {path.name for path in SUBMISSION.iterdir() if path.is_file()} == {
        "manifest.json",
        "model.py",
        "weights.safetensors",
    }
    weights = SUBMISSION / "weights.safetensors"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == WEIGHTS_SHA256
    assert sum(tensor.numel() for tensor in load_file(str(weights)).values()) == 85_021


def test_submission_passes_official_manifest_and_static_checks(candidate) -> None:
    manifest, _ = candidate
    assert manifest.entry_class == "model.Tier2HornerModel"
    assert manifest.output_base == "p"
    assert check_submission(SUBMISSION) == []


def test_tier1_reduced_domain_is_exact(candidate) -> None:
    _, model = candidate
    rows = [
        (str(a), str(b), str(p))
        for p in (2, 3, 5, 7)
        for a in range(p)
        for b in range(p)
    ]
    expected = [
        (int(a) * int(b)) % int(p)
        for a, b, p in rows
    ]
    assert [digits[0] for digits in model.predict_digits_batch(rows)] == expected


def test_checked_public_tier2_release_regression(candidate) -> None:
    _, model = candidate
    records = [
        json.loads(line)
        for line in (
            ROOT / "data" / "external" / "public_benchmark" / "tier_2.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    rows = [(row["a"], row["b"], row["p"]) for row in records]
    outputs = model.predict_digits_batch(rows)
    correct = sum(
        digits == [int(row["expected"])]
        for digits, row in zip(outputs, records, strict=True)
    )
    assert correct == 99


def test_mixed_batch_contract_and_determinism(candidate) -> None:
    _, model = candidate
    rows = [
        ("0", "123456", "2"),
        ("123", "456", "97"),
        ("987654321", "123456789", "257"),
        ("999", "888", "9923"),
    ]
    first = model.predict_digits_batch(rows)
    second = model.predict_digits_batch(rows)
    assert first == second
    assert len(first) == len(rows)
    for digits, (_, _, modulus) in zip(first, rows, strict=True):
        assert isinstance(digits, list) and len(digits) == 1
        assert type(digits[0]) is int
        assert 0 <= digits[0] < int(modulus)
    assert first[-1] == [0]
