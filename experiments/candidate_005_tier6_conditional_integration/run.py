"""Calibrate and evaluate the frozen Candidate 005 integration artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import psutil
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "sources" / "github" / "modular-arithmetic-challenge"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT / "src"))

from modchallenge.config import EvalConfig, TIERS  # noqa: E402
from modchallenge.evaluation.decoder import MalformedOutput, decode_answer  # noqa: E402
from modchallenge.evaluation.loader import (  # noqa: E402
    check_artifact_size,
    load_model,
    validate_manifest,
)
from modchallenge.security.static_check import check_submission  # noqa: E402
from modchallenge.testgen.generator import (  # noqa: E402
    TestCase,
    _generate_tier_cases,
    generate_private_test_set,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
REQUIRED_FILES = {"manifest.json", "model.py", "weights.safetensors"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def artifact_evidence(path: Path) -> dict[str, Any]:
    actual = {child.name for child in path.iterdir() if child.is_file()}
    if actual != REQUIRED_FILES:
        raise RuntimeError(f"candidate must be the exact three-file artifact: {actual}")
    unexpected_dirs = [child.name for child in path.iterdir() if child.is_dir()]
    if unexpected_dirs:
        raise RuntimeError(f"candidate contains unexpected directories: {unexpected_dirs}")
    return {
        "directory": str(path.relative_to(ROOT)),
        "model_sha256": sha256(path / "model.py"),
        "manifest_sha256": sha256(path / "manifest.json"),
        "weights_sha256": sha256(path / "weights.safetensors"),
        "bytes": sum(child.stat().st_size for child in path.iterdir()),
    }


def tier6_cases(seed_hex: str, cases: int) -> list[TestCase]:
    return _generate_tier_cases(
        tier=TIERS[6],
        num_cases=cases,
        num_primes=5,
        edge_cases=min(4, cases),
        seed=bytes.fromhex(seed_hex),
    )


def decoded_predictions(model: Any, cases: list[TestCase]) -> list[str]:
    raw = [(case.a, case.b, case.p) for case in cases]
    encoded = [
        (
            model.preprocess_a(a),
            model.preprocess_b(b),
            model.preprocess_p(p),
        )
        for a, b, p in raw
    ]
    digits = model.predict_digits_batch(encoded)
    if len(digits) != len(cases):
        raise RuntimeError("candidate returned the wrong number of predictions")
    predictions: list[str] = []
    for value, case in zip(digits, cases, strict=True):
        try:
            decoded = decode_answer(
                value,
                base="p",
                prime=int(case.p),
                is_tier_zero=False,
            )
            predictions.append(str(decoded))
        except MalformedOutput:
            predictions.append("")
    return predictions


def score_model(model: Any, cases: list[TestCase]) -> dict[str, Any]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    predictions = decoded_predictions(model, cases)
    wall_seconds = time.perf_counter() - wall_started
    aggregate_cpu_seconds = time.process_time() - cpu_started
    correct = sum(
        prediction == case.expected
        for prediction, case in zip(predictions, cases, strict=True)
    )
    valid = sum(
        prediction.isdigit() and int(prediction) < int(case.p)
        for prediction, case in zip(predictions, cases, strict=True)
    )
    return {
        "cases": len(cases),
        "correct": correct,
        "exact_accuracy": correct / len(cases),
        "valid_outputs": valid,
        "valid_output_rate": valid / len(cases),
        "predictions": predictions,
        "wall_seconds": wall_seconds,
        "aggregate_cpu_seconds": aggregate_cpu_seconds,
    }


class CandidateIntegration:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.submission = (ROOT / self.config["artifact"]["directory"]).resolve()
        self.incumbent_submission = (
            ROOT / self.config["incumbent"]["directory"]
        ).resolve()
        self.randomized_submission = (
            ROOT / self.config["randomized_control"]["artifact_directory"]
        ).resolve()
        self.process = psutil.Process()
        torch.set_num_threads(int(self.config["evaluation"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(self.config["seed"]))

    def verify_challenge(self) -> None:
        commit = git_output("rev-parse", "HEAD", cwd=OFFICIAL_ROOT)
        if commit != CHALLENGE_COMMIT:
            raise RuntimeError(
                f"official generator commit is {commit}, expected {CHALLENGE_COMMIT}"
            )

    def verify_artifact(self, path: Path, expected: dict[str, Any]) -> dict[str, Any]:
        evidence = artifact_evidence(path)
        for name in ("model_sha256", "manifest_sha256", "weights_sha256"):
            if evidence[name] != expected[name]:
                raise RuntimeError(f"{path.name} {name} mismatch")
        manifest = validate_manifest(path)
        findings = check_submission(path)
        if findings:
            raise RuntimeError(
                "static analysis failed: " + "; ".join(item.format() for item in findings)
            )
        check_artifact_size(path, int(self.config["evaluation"]["max_artifact_bytes"]))
        evidence.update(
            {
                "entry_class": manifest.entry_class,
                "output_base": manifest.output_base,
                "static_analysis": "clean",
            }
        )
        return evidence

    def load(self, path: Path):
        manifest = validate_manifest(path)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        model = load_model(path, manifest)
        return model, {
            "wall_seconds": time.perf_counter() - wall_started,
            "aggregate_cpu_seconds": time.process_time() - cpu_started,
        }

    def source_evidence(self) -> dict[str, Any]:
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
            "challenge_commit": CHALLENGE_COMMIT,
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "builder_sha256": sha256(ROOT / "tools/build_candidate005_artifact.py"),
        }

    def calibrate(self) -> dict[str, Any]:
        self.verify_challenge()
        source = self.source_evidence()
        artifact = self.verify_artifact(self.submission, self.config["artifact"])
        model, load_runtime = self.load(self.submission)
        budget = self.config["resource_budget"]
        warmup_cases = int(budget["calibration_warmup_cases"])
        profile_cases = int(budget["calibration_profile_cases"])
        cases = tier6_cases(
            self.config["seeds"]["calibration"], warmup_cases + profile_cases
        )
        decoded_predictions(model, cases[:warmup_cases])
        profile = score_model(model, cases[warmup_cases:])
        orientation_equivalents = profile_cases * (
            1.0 + float(self.config["routing"]["fraction"])
        )
        calibration = {
            "run_id": self.config["run_id"],
            "phase": "exact-artifact-resource-calibration",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "artifact": artifact,
            "load_runtime": load_runtime,
            "profile": {
                key: value for key, value in profile.items() if key != "predictions"
            },
            "calibration_seed_hex": self.config["seeds"]["calibration"],
            "warmup_cases": warmup_cases,
            "orientation_equivalent_cases": orientation_equivalents,
            "wall_seconds_per_orientation_equivalent": profile["wall_seconds"]
            / orientation_equivalents,
            "aggregate_cpu_seconds_per_orientation_equivalent": profile[
                "aggregate_cpu_seconds"
            ]
            / orientation_equivalents,
            "fidelity_reduced": False,
            "limitations": [
                "This calibration measures throughput, not independent selection evidence.",
                "Formal budgets must add setup, generator, randomized-control, determinism, and container headroom.",
            ],
        }
        path = self.output / "calibration.json"
        path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        calibration["calibration_sha256"] = sha256(path)
        print(json.dumps(calibration, indent=2))
        return calibration

    def check_formal_budget(self, started: float, partial: dict[str, Any]) -> None:
        elapsed = time.perf_counter() - started
        partial["elapsed_wall_seconds"] = elapsed
        partial["elapsed_aggregate_cpu_seconds"] = time.process_time() - partial[
            "cpu_started"
        ]
        partial_path = self.output / "formal-partial.json"
        serializable = {key: value for key, value in partial.items() if key != "cpu_started"}
        partial_path.write_text(
            json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
        )
        hard = float(self.config["resource_budget"]["hard_runaway_ceiling_wall_seconds"])
        if elapsed > hard:
            raise TimeoutError(
                f"formal local gate reached calibrated {hard:.1f}s wall ceiling; "
                "completed evidence is preserved"
            )

    def formal(self) -> dict[str, Any]:
        self.verify_challenge()
        source = self.source_evidence()
        if source["dirty"]:
            raise RuntimeError("formal execution requires a clean tracked worktree")
        calibration_path = (ROOT / self.config["resource_budget"]["calibration_path"]).resolve()
        if sha256(calibration_path) != self.config["resource_budget"][
            "calibration_sha256"
        ]:
            raise RuntimeError("exact-artifact calibration hash mismatch")
        artifact = self.verify_artifact(self.submission, self.config["artifact"])
        incumbent_artifact = self.verify_artifact(
            self.incumbent_submission, self.config["incumbent"]
        )
        randomized_artifact = self.verify_artifact(
            self.randomized_submission, self.config["randomized_control"]
        )
        candidate, load_runtime = self.load(self.submission)
        started = time.perf_counter()
        cpu_started = time.process_time()
        partial: dict[str, Any] = {
            "run_id": self.config["run_id"],
            "source": source,
            "artifact": artifact,
            "local_seed_results": [],
            "cpu_started": cpu_started,
        }
        cached_cases: dict[str, list[TestCase]] = {}
        raw_seed_results: list[dict[str, Any]] = []
        for seed_id in self.config["seeds"]["local_tier6_ids"]:
            seed_hex = self.config["seeds"]["independent"][seed_id]
            cases = tier6_cases(seed_hex, int(self.config["evaluation"]["tier6_cases"]))
            cached_cases[seed_id] = cases
            result = score_model(candidate, cases)
            durable = {
                "seed_id": seed_id,
                "seed_hex": seed_hex,
                **{key: value for key, value in result.items() if key != "predictions"},
            }
            partial["local_seed_results"].append(durable)
            raw_seed_results.append(
                {
                    **durable,
                    "records": [
                        {
                            "a": case.a,
                            "b": case.b,
                            "p": case.p,
                            "expected": case.expected,
                            "prediction": prediction,
                        }
                        for case, prediction in zip(
                            cases, result["predictions"], strict=True
                        )
                    ],
                }
            )
            self.check_formal_budget(started, partial)

        full_seed_id = self.config["seeds"]["full_sandbox_id"]
        full_seed_hex = self.config["seeds"]["independent"][full_seed_id]
        generator_wall = time.perf_counter()
        generator_cpu = time.process_time()
        full_set = generate_private_test_set(
            master_seed=bytes.fromhex(full_seed_hex), config=EvalConfig()
        )
        generator_runtime = {
            "wall_seconds": time.perf_counter() - generator_wall,
            "aggregate_cpu_seconds": time.process_time() - generator_cpu,
        }
        lower_tiers = [tier for tier in full_set.tiers if 1 <= tier.tier_id <= 5]
        candidate_lower: dict[int, list[str]] = {}
        lower_candidate_wall = time.perf_counter()
        lower_candidate_cpu = time.process_time()
        for tier in lower_tiers:
            candidate_lower[tier.tier_id] = decoded_predictions(candidate, tier.cases)
        candidate_lower_runtime = {
            "wall_seconds": time.perf_counter() - lower_candidate_wall,
            "aggregate_cpu_seconds": time.process_time() - lower_candidate_cpu,
        }
        incumbent, incumbent_load_runtime = self.load(self.incumbent_submission)
        incumbent_lower: dict[int, list[str]] = {}
        lower_incumbent_wall = time.perf_counter()
        lower_incumbent_cpu = time.process_time()
        for tier in lower_tiers:
            incumbent_lower[tier.tier_id] = decoded_predictions(incumbent, tier.cases)
        incumbent_lower_runtime = {
            "wall_seconds": time.perf_counter() - lower_incumbent_wall,
            "aggregate_cpu_seconds": time.process_time() - lower_incumbent_cpu,
        }
        lower_tier_equivalence = {
            "seed_id": full_seed_id,
            "seed_hex": full_seed_hex,
            "tiers": {
                str(tier_id): {
                    "cases": len(candidate_lower[tier_id]),
                    "predictions_equal": candidate_lower[tier_id]
                    == incumbent_lower[tier_id],
                }
                for tier_id in candidate_lower
            },
            "generator_runtime": generator_runtime,
            "candidate_runtime": candidate_lower_runtime,
            "incumbent_runtime": incumbent_lower_runtime,
            "incumbent_load_runtime": incumbent_load_runtime,
        }
        lower_tier_equivalence["passed"] = all(
            item["predictions_equal"]
            for item in lower_tier_equivalence["tiers"].values()
        )
        partial["lower_tier_equivalence"] = lower_tier_equivalence
        self.check_formal_budget(started, partial)

        determinism_seed = self.config["evaluation"]["determinism_seed_id"]
        det_cases = cached_cases[determinism_seed][
            : int(self.config["evaluation"]["determinism_cases"])
        ]
        det_first = decoded_predictions(candidate, det_cases)
        det_second = decoded_predictions(candidate, det_cases)
        determinism = {
            "seed_id": determinism_seed,
            "cases": len(det_cases),
            "predictions_equal": det_first == det_second,
        }
        partial["determinism"] = determinism
        self.check_formal_budget(started, partial)

        randomized, randomized_load_runtime = self.load(self.randomized_submission)
        randomized_seed = self.config["randomized_control"]["seed_id"]
        randomized_result = score_model(randomized, cached_cases[randomized_seed])
        randomized_metrics = {
            "seed_id": randomized_seed,
            **{
                key: value
                for key, value in randomized_result.items()
                if key != "predictions"
            },
        }
        partial["randomized_control"] = randomized_metrics
        self.check_formal_budget(started, partial)

        raw_path = self.output / "local-tier6-raw.json"
        raw_path.write_text(
            json.dumps(
                {
                    "candidate": raw_seed_results,
                    "determinism": {
                        **determinism,
                        "first": det_first,
                        "second": det_second,
                    },
                    "randomized": {
                        **randomized_metrics,
                        "predictions": randomized_result["predictions"],
                    },
                    "lower_tier_equivalence": lower_tier_equivalence,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        local_results = partial["local_seed_results"]
        gates = self.config["gates"]
        gate_checks = {
            "every_local_tier6_seed": all(
                result["exact_accuracy"] >= float(gates["minimum_tier6_accuracy"])
                for result in local_results
            ),
            "every_local_output_valid": all(
                result["valid_output_rate"] == 1.0 for result in local_results
            ),
            "determinism": determinism["predictions_equal"],
            "randomized_tier6_collapse": randomized_metrics["exact_accuracy"]
            <= float(gates["maximum_randomized_tier6_accuracy"]),
            "lower_tier_prediction_equivalence": lower_tier_equivalence["passed"],
            "static_analysis": artifact["static_analysis"] == "clean",
            "randomized_static_analysis": randomized_artifact["static_analysis"]
            == "clean",
        }
        gate_checks["passed"] = all(gate_checks.values())
        runtime = {
            "wall_seconds": time.perf_counter() - started,
            "aggregate_cpu_seconds": time.process_time() - cpu_started,
            "aggregate_cpu_minutes": (time.process_time() - cpu_started) / 60.0,
            "torch_threads": int(self.config["evaluation"]["torch_threads"]),
            "peak_rss_bytes": self.process.memory_info().rss,
            "load_runtime": load_runtime,
            "randomized_load_runtime": randomized_load_runtime,
        }
        decision = (
            "advance-to-isolated-full-seed1-gate"
            if gate_checks["passed"]
            else "reject-candidate005-local-integration"
        )
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=self.config["run_id"]) as run:
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "completed",
                    "decision": decision,
                    "role": "candidate005-local-integration-gate",
                    "repository_commit": source["repository_commit"],
                    "repository_dirty": "false",
                }
            )
            mlflow.log_params(
                {
                    "candidate_weights_sha256": artifact["weights_sha256"],
                    "candidate_model_sha256": artifact["model_sha256"],
                    "route_fraction": self.config["routing"]["fraction"],
                    "tier6_cases_per_seed": self.config["evaluation"]["tier6_cases"],
                    "local_seed_count": len(local_results),
                    "config_sha256": source["config_sha256"],
                    "runner_sha256": source["runner_sha256"],
                }
            )
            metrics = {
                f"tier6_{result['seed_id']}_accuracy": result["exact_accuracy"]
                for result in local_results
            }
            metrics.update(
                {
                    "minimum_local_tier6_accuracy": min(
                        result["exact_accuracy"] for result in local_results
                    ),
                    "randomized_tier6_accuracy": randomized_metrics[
                        "exact_accuracy"
                    ],
                    "wall_seconds": runtime["wall_seconds"],
                    "aggregate_cpu_seconds": runtime["aggregate_cpu_seconds"],
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(self.config_path), artifact_path="preregistration")
            mlflow.log_artifact(
                str(
                    ROOT
                    / "knowledge/autoresearch/ablations/"
                    "candidate-005-tier6-conditional-integration.md"
                ),
                artifact_path="preregistration",
            )
            mlflow.log_artifact(str(raw_path), artifact_path="raw-evidence")
            mlflow_run_id = run.info.run_id

        summary = {
            "run_id": self.config["run_id"],
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "local-four-seed-integration-gate",
            "hypothesis": self.config["hypothesis"],
            "source": source,
            "artifact": artifact,
            "incumbent_artifact": incumbent_artifact,
            "randomized_artifact": randomized_artifact,
            "local_seed_results": local_results,
            "lower_tier_equivalence": lower_tier_equivalence,
            "determinism": determinism,
            "randomized_control": randomized_metrics,
            "gates": {**gates, "checks": gate_checks, "passed": gate_checks["passed"]},
            "resource_budget": self.config["resource_budget"],
            "runtime": runtime,
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": self.config["tracking"]["experiment_name"],
                "experiment_id": experiment_id,
                "run_id": mlflow_run_id,
            },
            "artifacts": {
                "raw_local_evidence": str(raw_path.relative_to(ROOT)),
                "raw_local_evidence_sha256": sha256(raw_path),
            },
            "decision": decision,
            "limitations": [
                "These are fresh official-generator Tier-6 results; the seed-1 full independent container gate remains separate.",
                "Checked and public benchmark answers were not read.",
                "The legitimate official private benchmark is unavailable.",
                "Candidate 005 remains far below the binding 85% overall completion threshold even if Tier 6 passes.",
            ],
        }
        summary_path = self.output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/candidate_005_tier6_conditional_integration.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/candidate-005-tier6-conditional-commutative-integration",
    )
    parser.add_argument("--phase", choices=("calibrate", "formal"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = CandidateIntegration(args.config, args.output)
    if args.phase == "calibrate":
        experiment.calibrate()
    else:
        experiment.formal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
