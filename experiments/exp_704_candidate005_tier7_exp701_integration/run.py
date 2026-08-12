"""Build and evaluate Candidate006's frozen Tier-7-only EXP701 integration."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import psutil
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "sources/github/modular-arithmetic-challenge"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT / "src"))

from modchallenge.config import EvalConfig, TIERS  # noqa: E402
from modchallenge.evaluation.decoder import MalformedOutput, decode_answer  # noqa: E402
from modchallenge.evaluation.loader import check_artifact_size, validate_manifest  # noqa: E402
from modchallenge.security.static_check import check_submission  # noqa: E402
from modchallenge.testgen.generator import TestCase, _generate_tier_cases, generate_private_test_set  # noqa: E402
from tools import build_candidate006_tier7_artifact as builder  # noqa: E402


RUN_ID = "exp-704-candidate005-tier7-exp701-integration"
CONFIG_PATH = ROOT / "configs/exp_704_candidate005_tier7_exp701_integration.yaml"
REQUIRED_FILES = frozenset(("manifest.json", "model.py", "weights.safetensors"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def heartbeat() -> None:
    value = os.environ.get("SAIR_CLOUD_HEARTBEAT_PATH")
    if value:
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_candidate(path: Path, name: str):
    module = load_module(name, path / "model.py")
    model = module.Tier2HornerModel()
    model.load(str(path))
    return model


def artifact_evidence(path: Path, max_bytes: int) -> dict[str, Any]:
    actual = {child.name for child in path.iterdir() if child.is_file()}
    directories = [child.name for child in path.iterdir() if child.is_dir()]
    if actual != REQUIRED_FILES or directories:
        raise RuntimeError(f"artifact is not the exact three-file shape: {actual}, dirs={directories}")
    manifest = validate_manifest(path)
    findings = check_submission(path)
    if findings:
        raise RuntimeError("static analysis failed: " + "; ".join(item.format() for item in findings))
    check_artifact_size(path, max_bytes)
    weights = path / "weights.safetensors"
    return {
        "directory": str(path),
        "model_sha256": sha256(path / "model.py"),
        "manifest_sha256": sha256(path / "manifest.json"),
        "weights_sha256": sha256(weights),
        "bytes": sum(child.stat().st_size for child in path.iterdir()),
        "entry_class": manifest.entry_class,
        "output_base": manifest.output_base,
        "static_analysis": "clean",
    }


def decoded_predictions(model: Any, cases: list[TestCase], tier_id: int) -> list[str]:
    encoded = [
        (model.preprocess_a(case.a), model.preprocess_b(case.b), model.preprocess_p(case.p))
        for case in cases
    ]
    heartbeat()
    digits = model.predict_digits_batch(encoded)
    heartbeat()
    if len(digits) != len(cases):
        raise RuntimeError("candidate returned the wrong number of predictions")
    predictions: list[str] = []
    for value, case in zip(digits, cases, strict=True):
        try:
            decoded = decode_answer(
                value,
                base="p",
                prime=int(case.p),
                is_tier_zero=tier_id == 0,
            )
            predictions.append(str(decoded))
        except MalformedOutput:
            predictions.append("")
    return predictions


def score(predictions: list[str], cases: list[TestCase], tier_id: int) -> dict[str, Any]:
    correct = sum(
        prediction == case.expected
        for prediction, case in zip(predictions, cases, strict=True)
    )
    valid = sum(
        bool(prediction)
        and prediction.lstrip("-").isdigit()
        and (tier_id == 0 or 0 <= int(prediction) < int(case.p))
        for prediction, case in zip(predictions, cases, strict=True)
    )
    return {
        "tier_id": tier_id,
        "cases": len(cases),
        "correct": correct,
        "exact_accuracy": correct / len(cases),
        "valid_outputs": valid,
        "valid_output_rate": valid / len(cases),
        "prediction_sha256": canonical_hash(predictions),
    }


class Integration:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if self.config["run_id"] != RUN_ID:
            raise RuntimeError("unexpected run_id")
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(int(self.config["evaluation"]["torch_threads"]))
        try:
            torch.set_num_interop_threads(int(self.config["evaluation"]["interop_threads"]))
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(self.config["seed"]))

    @property
    def candidate_path(self) -> Path:
        return self.output / "candidate006/submission"

    @property
    def randomized_path(self) -> Path:
        return self.output / "randomized/submission"

    def verify_sources(self) -> dict[str, Any]:
        incumbent = self.config["incumbent"]
        incumbent_path = ROOT / incumbent["directory"]
        for filename, key in (
            ("model.py", "model_sha256"),
            ("manifest.json", "manifest_sha256"),
            ("weights.safetensors", "weights_sha256"),
        ):
            if sha256(incumbent_path / filename) != incumbent[key]:
                raise RuntimeError(f"incumbent {filename} hash mismatch")
        specialist = ROOT / self.config["specialist"]["path"]
        if sha256(specialist) != self.config["specialist"]["sha256"]:
            raise RuntimeError("specialist checkpoint hash mismatch")
        official: dict[str, str] = {}
        for relative, expected in self.config["source_contract"]["official_files"].items():
            actual = sha256(ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"official source hash mismatch: {relative}")
            official[relative] = actual
        return {
            "incumbent": {key: incumbent[key] for key in ("model_sha256", "manifest_sha256", "weights_sha256")},
            "specialist_sha256": sha256(specialist),
            "official_files": official,
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "builder_sha256": sha256(ROOT / "tools/build_candidate006_tier7_artifact.py"),
            "cloud_manifest_sha256": os.environ.get("SAIR_CLOUD_MANIFEST_SHA256"),
        }

    def build(self) -> dict[str, Any]:
        source = self.verify_sources()
        trained = builder.build(self.candidate_path)
        randomized = builder.build(
            self.randomized_path,
            int(self.config["randomized_control"]["seed"]),
        )
        maximum = int(self.config["evaluation"]["max_artifact_bytes"])
        trained_artifact = artifact_evidence(self.candidate_path, maximum)
        randomized_artifact = artifact_evidence(self.randomized_path, maximum)
        load_candidate(self.candidate_path, "exp704_candidate_build_probe")
        result = {
            "schema_version": "exp704-build-v1",
            "run_id": RUN_ID,
            "status": "completed",
            "created_at": utc_now(),
            "training_executed": False,
            "source": source,
            "candidate": {**trained, **trained_artifact},
            "randomized_control": {**randomized, **randomized_artifact},
        }
        write_json(self.output / "build.json", result)
        return result

    def canary(self) -> dict[str, Any]:
        source = self.verify_sources()
        candidate_dir = self.output / "candidate006/submission"
        builder.build(candidate_dir)
        model = load_candidate(candidate_dir, "exp704_canary_candidate")
        cases = _generate_tier_cases(
            tier=TIERS[7],
            num_cases=2,
            num_primes=2,
            edge_cases=1,
            seed=bytes.fromhex(self.config["seeds"]["seed1"]),
        )
        predictions = decoded_predictions(model, cases, 7)
        result = {
            "schema_version": "exp704-canary-v1",
            "run_id": RUN_ID,
            "status": "passed",
            "source": source,
            "tier7_cases": len(cases),
            "prediction_sha256": canonical_hash(predictions),
            "output_count": len(predictions),
        }
        write_json(self.output / "canary.json", result)
        return result

    def _seed_result(self, seed_id: str, seed_hex: str, candidate: Any, incumbent: Any) -> tuple[dict[str, Any], dict[int, list[TestCase]]]:
        generated = generate_private_test_set(bytes.fromhex(seed_hex), EvalConfig())
        tiers = {int(tier.tier_id): list(tier.cases) for tier in generated.tiers}
        if sum(len(cases) for cases in tiers.values()) != int(self.config["evaluation"]["full_total_cases"]):
            raise RuntimeError("official generator returned an unexpected full-set denominator")
        candidate_rows: dict[str, Any] = {}
        incumbent_rows: dict[str, Any] = {}
        parity: dict[str, Any] = {}
        candidate_started = time.perf_counter()
        candidate_cpu = time.process_time()
        for tier_id in sorted(tiers):
            cases = tiers[tier_id]
            predictions = decoded_predictions(candidate, cases, tier_id)
            candidate_rows[str(tier_id)] = score(predictions, cases, tier_id)
            if tier_id in set(map(int, self.config["evaluation"]["parity_tiers"])):
                incumbent_predictions = decoded_predictions(incumbent, cases, tier_id)
                incumbent_rows[str(tier_id)] = score(incumbent_predictions, cases, tier_id)
                parity[str(tier_id)] = {
                    "cases": len(cases),
                    "predictions_equal": predictions == incumbent_predictions,
                    "candidate_prediction_sha256": canonical_hash(predictions),
                    "incumbent_prediction_sha256": canonical_hash(incumbent_predictions),
                }
        # Candidate005's >128-bit route is an immediate zero fallback, but execute it
        # here as well so the matched improvement is measured rather than inferred.
        for tier_id in sorted(tiers):
            if str(tier_id) in incumbent_rows:
                continue
            predictions = decoded_predictions(incumbent, tiers[tier_id], tier_id)
            incumbent_rows[str(tier_id)] = score(predictions, tiers[tier_id], tier_id)
        scored = set(map(int, self.config["evaluation"]["scored_tiers"]))
        candidate_correct = sum(row["correct"] for key, row in candidate_rows.items() if int(key) in scored)
        incumbent_correct = sum(row["correct"] for key, row in incumbent_rows.items() if int(key) in scored)
        denominator = sum(row["cases"] for key, row in candidate_rows.items() if int(key) in scored)
        result = {
            "seed_id": seed_id,
            "seed_hex": seed_hex,
            "candidate": {
                "overall_correct": candidate_correct,
                "overall_cases": denominator,
                "overall_accuracy": candidate_correct / denominator,
                "tiers": candidate_rows,
            },
            "incumbent": {
                "overall_correct": incumbent_correct,
                "overall_cases": denominator,
                "overall_accuracy": incumbent_correct / denominator,
                "tiers": incumbent_rows,
            },
            "matched_overall_gain": (candidate_correct - incumbent_correct) / denominator,
            "lower_tier_equivalence": {
                "tiers": parity,
                "passed": all(row["predictions_equal"] for row in parity.values()),
            },
            "all_candidate_outputs_valid": all(row["valid_output_rate"] == 1.0 for row in candidate_rows.values()),
            "runtime": {
                "wall_seconds": time.perf_counter() - candidate_started,
                "process_cpu_seconds": time.process_time() - candidate_cpu,
            },
        }
        return result, tiers

    def evaluate(self) -> dict[str, Any]:
        build_record = json.loads((self.output / "build.json").read_text(encoding="utf-8"))
        maximum = int(self.config["evaluation"]["max_artifact_bytes"])
        candidate_artifact = artifact_evidence(self.candidate_path, maximum)
        randomized_artifact = artifact_evidence(self.randomized_path, maximum)
        for key in ("model_sha256", "manifest_sha256", "weights_sha256"):
            if candidate_artifact[key] != build_record["candidate"][key]:
                raise RuntimeError(f"candidate artifact changed after build: {key}")
            if randomized_artifact[key] != build_record["randomized_control"][key]:
                raise RuntimeError(f"randomized artifact changed after build: {key}")
        candidate = load_candidate(self.candidate_path, "exp704_candidate")
        incumbent = load_candidate(ROOT / self.config["incumbent"]["directory"], "exp704_incumbent")
        randomized = load_candidate(self.randomized_path, "exp704_randomized")
        process = psutil.Process()
        started = time.perf_counter()
        cpu_started = time.process_time()
        reuse = self.config["evidence_reuse"]
        reuse_path = ROOT / reuse["path"]
        if sha256(reuse_path) != reuse["sha256"]:
            raise RuntimeError("seed-1 evidence-reuse hash mismatch")
        prior = json.loads(reuse_path.read_text(encoding="utf-8"))
        if prior.get("run_id") != RUN_ID or len(prior.get("seed_results", [])) != 1:
            raise RuntimeError("seed-1 evidence-reuse schema mismatch")
        reused = copy.deepcopy(prior["seed_results"][0])
        if reused["seed_id"] != reuse["seed_id"] or reused["seed_hex"] != self.config["seeds"][reuse["seed_id"]]:
            raise RuntimeError("seed-1 evidence-reuse identity mismatch")
        for key in ("model_sha256", "manifest_sha256", "weights_sha256"):
            if prior["candidate_artifact"][key] != candidate_artifact[key]:
                raise RuntimeError(f"seed-1 candidate evidence mismatch: {key}")
            if prior["randomized_artifact"][key] != randomized_artifact[key]:
                raise RuntimeError(f"seed-1 randomized evidence mismatch: {key}")
        retained = set(map(int, self.config["evaluation"]["parity_tiers"]))
        parity = {
            key: value
            for key, value in reused["lower_tier_equivalence"]["tiers"].items()
            if int(key) in retained
        }
        reused["lower_tier_equivalence"] = {
            "tiers": parity,
            "passed": all(value["predictions_equal"] for value in parity.values()),
        }
        reused["reused_evidence"] = {
            "path": reuse["path"],
            "sha256": reuse["sha256"],
            "prior_manifest_sha256": reuse["prior_manifest_sha256"],
            "prior_package_sha256": reuse["prior_package_sha256"],
            "correction": reuse["correction"],
        }
        results: list[dict[str, Any]] = [reused]
        seed1_set = generate_private_test_set(
            bytes.fromhex(self.config["seeds"][reuse["seed_id"]]), EvalConfig()
        )
        first_tiers = {int(tier.tier_id): list(tier.cases) for tier in seed1_set.tiers}
        gates = self.config["gates"]
        write_json(self.output / "progress.json", {"status": "running", "seeds": results})
        for seed_id, seed_hex in self.config["seeds"].items():
            if seed_id == reuse["seed_id"]:
                continue
            result, tiers = self._seed_result(seed_id, seed_hex, candidate, incumbent)
            results.append(result)
            write_json(self.output / "progress.json", {"status": "running", "seeds": results})
        randomized_metrics = None
        determinism = None
        if len(results) == len(self.config["seeds"]):
            tier7_cases = first_tiers[7]
            randomized_predictions = decoded_predictions(randomized, tier7_cases, 7)
            randomized_metrics = score(randomized_predictions, tier7_cases, 7)
            det_cases = tier7_cases[: int(self.config["evaluation"]["determinism_cases"])]
            first = decoded_predictions(candidate, det_cases, 7)
            second = decoded_predictions(candidate, det_cases, 7)
            determinism = {
                "cases": len(det_cases),
                "predictions_equal": first == second,
                "prediction_sha256": canonical_hash(first),
            }
        candidate_tier7_correct = sum(row["candidate"]["tiers"]["7"]["correct"] for row in results)
        candidate_tier7_cases = sum(row["candidate"]["tiers"]["7"]["cases"] for row in results)
        aggregate_tier7 = candidate_tier7_correct / candidate_tier7_cases
        checks = {
            "all_five_seeds_completed": len(results) == len(self.config["seeds"]),
            "every_seed_matched_improvement": all(row["matched_overall_gain"] > 0.0 for row in results),
            "every_full_over_reference": all(row["candidate"]["overall_accuracy"] > float(gates["minimum_every_full_overall_accuracy"]) for row in results),
            "lower_tier_prediction_equivalence": all(row["lower_tier_equivalence"]["passed"] for row in results),
            "all_outputs_valid": all(row["all_candidate_outputs_valid"] for row in results),
            "aggregate_tier7_accuracy": aggregate_tier7 >= float(gates["minimum_aggregate_tier7_accuracy"]),
            "randomized_weight_dependence": randomized_metrics is not None and aggregate_tier7 - randomized_metrics["exact_accuracy"] >= float(gates["minimum_tier7_gain_over_randomized"]),
            "determinism": determinism is not None and determinism["predictions_equal"],
            "static_analysis": candidate_artifact["static_analysis"] == "clean" and randomized_artifact["static_analysis"] == "clean",
            "training_executed_false": True,
        }
        passed = all(checks.values())
        result = {
            "schema_version": "exp704-formal-v1",
            "run_id": RUN_ID,
            "status": "completed",
            "created_at": utc_now(),
            "training_executed": False,
            "source": self.verify_sources(),
            "candidate_artifact": candidate_artifact,
            "randomized_artifact": randomized_artifact,
            "seed_results": results,
            "aggregate": {
                "seeds_completed": len(results),
                "tier7_correct": candidate_tier7_correct,
                "tier7_cases": candidate_tier7_cases,
                "tier7_exact_accuracy": aggregate_tier7,
                "minimum_overall_accuracy": min(row["candidate"]["overall_accuracy"] for row in results),
                "mean_overall_accuracy": sum(row["candidate"]["overall_accuracy"] for row in results) / len(results),
                "minimum_matched_gain": min(row["matched_overall_gain"] for row in results),
            },
            "randomized_control": randomized_metrics,
            "determinism": determinism,
            "gates": {"checks": checks, "passed": passed},
            "decision": "advance-to-official-sandbox" if passed else "complete-negative-integration",
            "resource": {
                "wall_seconds": time.perf_counter() - started,
                "process_cpu_seconds": time.process_time() - cpu_started,
                "peak_rss_bytes": process.memory_info().rss,
                "torch_threads": torch.get_num_threads(),
            },
            "limitations": [
                "Fresh official-generator evidence only; checked and private answers were not read.",
                "The isolated official sandbox gate remains separate from this direct evaluation.",
                "No result above Tier 7, private-seed result, or 85% completion claim follows.",
            ],
        }
        write_json(self.output / "formal.json", result)
        heartbeat()
        return result

    def summarize(self) -> dict[str, Any]:
        formal = json.loads((self.output / "formal.json").read_text(encoding="utf-8"))
        result = {
            "schema_version": "exp704-terminal-v1",
            "run_id": RUN_ID,
            "status": "completed",
            "created_at": utc_now(),
            "decision": formal["decision"],
            "formal_sha256": sha256(self.output / "formal.json"),
            "candidate_artifact": formal["candidate_artifact"],
            "aggregate": formal["aggregate"],
            "gates": formal["gates"],
            "official_sandbox_required": formal["decision"] == "advance-to-official-sandbox",
            "training_executed": False,
        }
        write_json(self.output / "terminal-summary.json", result)
        write_json(
            self.output / "environment.json",
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cpu": platform.processor(),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "torch_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("canary", "build", "evaluate", "summarize"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = Integration(args.config, args.output)
    result = getattr(experiment, args.mode)()
    print(json.dumps({"run_id": RUN_ID, "mode": args.mode, "status": result["status"], "decision": result.get("decision")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
