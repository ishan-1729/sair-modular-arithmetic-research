"""Evaluate a local submission against the exact checked public JSONL files.

The official CLI regenerates its test set from a seed. At the pinned upstream
commit, that regeneration differs from the checked Tier-0 diagnostic modulus
values. This tool retains the official manifest, static analysis, loader,
preprocessing isolation, determinism, inference, decoder, and scorer while
using the checked 1,100 records as the test-set authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from modchallenge.evaluation.loader import (
    check_artifact_size,
    load_model,
    validate_manifest,
)
from modchallenge.evaluation.pipeline import (
    check_determinism,
    check_preprocess_isolation,
    run_inference,
)
from modchallenge.evaluation.scorer import score_full_in_memory
from modchallenge.security.static_check import check_submission
from modchallenge.testgen.generator import FullTestSet, TestCase, TierTestSet


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"


def display_path(path: Path) -> str:
    """Render workspace paths portably while retaining external absolute paths."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checked_test_set(benchmark_dir: Path) -> tuple[FullTestSet, dict[str, str]]:
    tiers: list[TierTestSet] = []
    hashes: dict[str, str] = {}
    seen: set[tuple[str, str, str]] = set()
    for tier_id in range(11):
        path = benchmark_dir / f"tier_{tier_id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[path.name] = file_sha256(path)
        cases: list[TestCase] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            record = json.loads(line)
            if record.get("tier_id") != tier_id:
                raise ValueError(f"{path}:{line_number}: tier_id mismatch")
            key = (record["a"], record["b"], record["p"])
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate input triple")
            seen.add(key)
            cases.append(TestCase(**record))
        if len(cases) != 100:
            raise ValueError(f"{path}: expected 100 cases, found {len(cases)}")
        tiers.append(TierTestSet(tier_id=tier_id, cases=cases))
    return FullTestSet(tiers=tiers), hashes


def evaluate_checked(
    submission_dir: Path,
    benchmark_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    manifest = validate_manifest(submission_dir)
    artifact_bytes = check_artifact_size(submission_dir, 20 * 1024**3)
    findings = check_submission(submission_dir)
    if findings:
        rendered = "\n".join(finding.format() for finding in findings)
        raise RuntimeError(f"submission failed static analysis:\n{rendered}")
    test_set, hashes = load_checked_test_set(benchmark_dir)

    load_started = time.perf_counter()
    model = load_model(submission_dir, manifest)
    load_seconds = time.perf_counter() - load_started

    isolation_started = time.perf_counter()
    preprocess_isolated = check_preprocess_isolation(model, test_set)
    isolation_seconds = time.perf_counter() - isolation_started

    determinism_started = time.perf_counter()
    deterministic = check_determinism(
        model,
        test_set,
        output_base=manifest.output_base,
    )
    determinism_seconds = time.perf_counter() - determinism_started

    inference_started = time.perf_counter()
    predictions = run_inference(
        model,
        test_set,
        output_base=manifest.output_base,
        timeout_seconds=timeout_seconds,
    )
    inference_seconds = time.perf_counter() - inference_started
    result = score_full_in_memory(test_set, predictions)
    result.deterministic = deterministic
    return {
        "status": "completed",
        "benchmark": {
            "kind": "checked-public-jsonl",
            "directory": display_path(benchmark_dir),
            "cases": test_set.total_cases,
            "sha256": hashes,
        },
        "evaluator": {
            "implementation": "pinned-official-pipeline-with-checked-jsonl-adapter",
            "challenge_commit": CHALLENGE_COMMIT,
        },
        "submission": {
            "directory": display_path(submission_dir),
            "entry_class": manifest.entry_class,
            "output_base": manifest.output_base,
            "artifact_bytes": artifact_bytes,
        },
        "checks": {
            "static_analysis": "clean",
            "preprocess_isolated": preprocess_isolated,
            "deterministic": deterministic,
        },
        "timing_seconds": {
            "load": load_seconds,
            "preprocess_isolation": isolation_seconds,
            "determinism": determinism_seconds,
            "inference": inference_seconds,
        },
        "result": result.summary(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=ROOT / "data" / "external" / "public_benchmark",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = evaluate_checked(
        args.submission_dir.resolve(),
        args.benchmark_dir.resolve(),
        args.timeout,
    )
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
