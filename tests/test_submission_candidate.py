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
MODEL_SHA256 = "bf4bdbdda90ef148ebc9e0b4db3c5c73d7b73e2156c868ce0c05b5c1471d8833"
MANIFEST_SHA256 = "95707215a920ab2cab09fd16ad1347d5b8055626776999af71e54cc2867364d8"
WEIGHTS_SHA256 = "a7d25ee4eed49c5f625bc8369447be9f51ff1be54766631d8b96ab725860553e"


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
    assert hashlib.sha256((SUBMISSION / "model.py").read_bytes()).hexdigest() == MODEL_SHA256
    assert hashlib.sha256((SUBMISSION / "manifest.json").read_bytes()).hexdigest() == MANIFEST_SHA256
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == WEIGHTS_SHA256
    assert sum(tensor.numel() for tensor in load_file(str(weights)).values()) == 569_873


def test_staged_artifact_is_current_rules_compliant() -> None:
    receipt = json.loads(
        (
            ROOT
            / "experiments/runs/exp-708-candidate007-current-rules/current-rules-provenance.json"
        ).read_text()
    )
    assert receipt["status"] == "verified"
    assert receipt["manual_rule_audit"]["learned_reduction"] is True
    assert receipt["manual_rule_audit"]["deterministic_full_width_prereduction"] is False


def test_submission_passes_official_manifest_and_static_checks(candidate) -> None:
    manifest, _ = candidate
    assert manifest.entry_class == "model.Tier2HornerModel"
    assert manifest.output_base == "p"
    assert check_submission(SUBMISSION) == []


def test_tier1_raw_operand_canary_is_exact(candidate) -> None:
    _, model = candidate
    rows = [(str((1 << 31) + 123), str((1 << 30) + 77), "7")]
    expected = [(int(rows[0][0]) * int(rows[0][1])) % 7]
    assert [digits[0] for digits in model.predict_digits_batch(rows)] == expected


def test_checked_public_tier2_release_regression(candidate) -> None:
    _, model = candidate
    records = [
        json.loads(line)
        for line in (
            ROOT / "sources" / "github" / "modular-arithmetic-challenge" / "public_benchmark" / "tier_2.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    rows = [(row["a"], row["b"], row["p"]) for row in records]
    outputs = model.predict_digits_batch(rows)
    correct = sum(
        digits == [int(row["expected"])]
        for digits, row in zip(outputs, records, strict=True)
    )
    assert correct == 99


def test_checked_public_tier3_release_regression(candidate) -> None:
    _, model = candidate
    records = [
        json.loads(line)
        for line in (
            ROOT / "sources" / "github" / "modular-arithmetic-challenge" / "public_benchmark" / "tier_3.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    rows = [(row["a"], row["b"], row["p"]) for row in records]
    outputs = model.predict_digits_batch(rows)
    correct = sum(
        digits == [int(row["expected"])]
        for digits, row in zip(outputs, records, strict=True)
    )
    assert correct == 96


def test_checked_public_tier4_release_regression(candidate) -> None:
    _, model = candidate
    records = [
        json.loads(line)
        for line in (
            ROOT / "sources" / "github" / "modular-arithmetic-challenge" / "public_benchmark" / "tier_4.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    rows = [(row["a"], row["b"], row["p"]) for row in records]
    outputs = model.predict_digits_batch(rows)
    correct = sum(
        digits == [int(row["expected"])]
        for digits, row in zip(outputs, records, strict=True)
    )
    assert correct == 86


def test_checked_public_tier5_release_regression(candidate) -> None:
    _, model = candidate
    records = [
        json.loads(line)
        for line in (
            ROOT / "sources" / "github" / "modular-arithmetic-challenge" / "public_benchmark" / "tier_5.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    rows = [(row["a"], row["b"], row["p"]) for row in records]
    outputs = model.predict_digits_batch(rows)
    correct = sum(
        digits == [int(row["expected"])]
        for digits, row in zip(outputs, records, strict=True)
    )
    assert correct == 91


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
