"""Calibrate and evaluate frozen operand-swapped learned trajectories for EXP-640."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import psutil
import torch
import yaml
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def decode(logits: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in logits.ge(0).tolist()
    ]


def select_by_margin(
    original_predictions: list[int],
    swapped_predictions: list[int],
    original_margins: list[float],
    swapped_margins: list[float],
) -> dict[str, Any]:
    if not (
        len(original_predictions)
        == len(swapped_predictions)
        == len(original_margins)
        == len(swapped_margins)
    ):
        raise ValueError("paired selector inputs must have equal lengths")
    use_swapped = [
        swapped > original
        for original, swapped in zip(original_margins, swapped_margins, strict=True)
    ]
    predictions = [
        swapped if choose else original
        for original, swapped, choose in zip(
            original_predictions, swapped_predictions, use_swapped, strict=True
        )
    ]
    return {"predictions": predictions, "use_swapped": use_swapped}


def paired_counts(
    original: list[int], swapped: list[int], targets: list[int]
) -> dict[str, int]:
    original_correct = [a == truth for a, truth in zip(original, targets, strict=True)]
    swapped_correct = [a == truth for a, truth in zip(swapped, targets, strict=True)]
    return {
        "intersection": sum(a and b for a, b in zip(original_correct, swapped_correct)),
        "original_only": sum(a and not b for a, b in zip(original_correct, swapped_correct)),
        "swapped_only": sum(b and not a for a, b in zip(original_correct, swapped_correct)),
        "neither": sum(not a and not b for a, b in zip(original_correct, swapped_correct)),
        "prediction_disagreement": sum(a != b for a, b in zip(original, swapped)),
    }


class Experiment:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process()
        self.exp300 = load_module(
            "exp300_for_exp640",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
        self.cellular = load_module(
            "cellular_for_exp640",
            ROOT / "experiments/exp_220_width_independent_cellular/model.py",
        )
        self.source_path = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source_path) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")
        torch.set_num_threads(int(self.config["evaluation"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(self.config["seed"]))
        self.width = int(self.config["consumed_evidence"]["width"])
        self.split = self.make_consumed_split()
        self.split_hash = sha256_text(json.dumps(self.split, sort_keys=True))
        self.model = self.cellular.SharedDilatedConvGRU(
            width=self.width,
            hidden=int(self.config["model"]["hidden"]),
            cycles=int(self.config["model"]["cycles"]),
        )
        self.model.load_state_dict(load_file(self.source_path))
        self.model.eval()

    def make_consumed_split(self) -> dict[str, list[int]]:
        evidence = self.config["consumed_evidence"]
        local = {
            "data": {
                "prime_bits_min": self.width - int(evidence["prime_bit_span"]) + 1,
                "prime_bits_max": self.width,
                "train_prime_identities": int(evidence["train_prime_identities"]),
                "test_prime_identities": int(evidence["test_prime_identities"]),
            }
        }
        split_seed = int(evidence["source_seed"]) + int(
            evidence["split_seed_offset"]
        ) + self.width
        return self.exp300.bigint_prime_split(local, split_seed)

    @torch.inference_mode()
    def rollout(
        self, left: list[int], right: list[int], moduli: list[int]
    ) -> dict[str, Any]:
        batch = len(left)
        states = [0] * batch
        minimum_margin = torch.full((batch,), math.inf)
        invalid = [False] * batch
        wall = time.perf_counter()
        cpu = time.process_time()
        self.model.width = self.width
        for position in range(self.width):
            input_bits = [
                (value >> (self.width - 1 - position)) & 1 for value in left
            ]
            logits, _ = self.model(
                self.exp300.features(states, right, moduli, input_bits, self.width)
            )
            minimum_margin = torch.minimum(
                minimum_margin, logits.abs().amin(dim=1)
            )
            states = decode(logits)
            invalid = [
                prior or prediction >= modulus
                for prior, prediction, modulus in zip(
                    invalid, states, moduli, strict=True
                )
            ]
        return {
            "predictions": states,
            "minimum_margins": minimum_margin.tolist(),
            "invalid": invalid,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
        }

    @staticmethod
    def strip(result: dict[str, Any], targets: list[int]) -> dict[str, Any]:
        correct = sum(
            prediction == target
            for prediction, target in zip(result["predictions"], targets, strict=True)
        )
        return {
            "cases": len(targets),
            "correct": correct,
            "exact_accuracy": correct / len(targets),
            "any_invalid_residue_rate": sum(result["invalid"]) / len(targets),
            "wall_seconds": result["wall_seconds"],
            "aggregate_cpu_seconds": result["aggregate_cpu_seconds"],
        }

    def pair_stream(self, train: bool, cases: int, seed: int):
        identities = self.split["train" if train else "test"]
        return self.exp300.BigIntSampler(identities, seed).pairs(cases)

    def source_hashes(self) -> dict[str, Any]:
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
            "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "source_checkpoint_sha256": sha256(self.source_path),
            "split_sha256": self.split_hash,
        }

    def calibrate(self) -> dict[str, Any]:
        budget = self.config["resource_budget"]
        cases = int(budget["calibration_cases"])
        left, right, moduli = self.pair_stream(
            True, cases, int(budget["calibration_seed"])
        )
        self.rollout(left[:2], right[:2], moduli[:2])
        started = time.perf_counter()
        cpu_started = time.process_time()
        original = self.rollout(left, right, moduli)
        swapped = self.rollout(right, left, moduli)
        profile_wall = time.perf_counter() - started
        profile_cpu = time.process_time() - cpu_started
        formal_cases = int(self.config["consumed_evidence"]["cases"])
        determinism_cases = int(self.config["evaluation"]["determinism_cases"])
        orientation_equivalents = 2 * formal_cases + 4 * determinism_cases
        projected_eval_wall = profile_wall / (2 * cases) * orientation_equivalents
        projected_eval_cpu = profile_cpu / (2 * cases) * orientation_equivalents
        setup = 60.0
        expected_wall = math.ceil(projected_eval_wall + setup)
        expected_cpu_minutes = math.ceil((projected_eval_cpu + setup * 2) / 60)
        result = {
            "run_id": self.config["run_id"],
            "status": "calibration-completed",
            "evidence_scope": "training-prime-identities-only",
            "source": self.source_hashes(),
            "profile": {
                "cases_per_orientation": cases,
                "original_wall_seconds": original["wall_seconds"],
                "original_aggregate_cpu_seconds": original[
                    "aggregate_cpu_seconds"
                ],
                "swapped_wall_seconds": swapped["wall_seconds"],
                "swapped_aggregate_cpu_seconds": swapped[
                    "aggregate_cpu_seconds"
                ],
                "combined_wall_seconds": profile_wall,
                "combined_aggregate_cpu_seconds": profile_cpu,
            },
            "projection": {
                "orientation_equivalent_cases": orientation_equivalents,
                "evaluation_wall_seconds": projected_eval_wall,
                "evaluation_aggregate_cpu_seconds": projected_eval_cpu,
                "setup_artifact_wall_seconds": setup,
                "expected_total_wall_seconds": expected_wall,
                "expected_total_aggregate_cpu_minutes": expected_cpu_minutes,
            },
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "aggregate_cpu_seconds": time.process_time() - cpu_started,
                "peak_rss_bytes": self.process.memory_info().rss,
            },
            "limitations": [
                "Training identities only; no consumed test answer was read.",
                "Calibration measures the exact margin-collecting orientation path.",
            ],
        }
        path = self.output / "calibration.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return result

    def check_ceiling(self, started: float) -> None:
        ceiling = float(
            self.config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]
        )
        if time.perf_counter() - started > ceiling:
            raise TimeoutError("EXP-640 reached its calibrated wall ceiling")

    def formal(self) -> dict[str, Any]:
        budget = self.config["resource_budget"]
        if budget["calibration_status"] != "complete":
            raise RuntimeError("formal execution requires completed calibration")
        source = self.source_hashes()
        if source["dirty"]:
            raise RuntimeError("formal execution requires a clean tracked worktree")
        cases = int(self.config["consumed_evidence"]["cases"])
        pair_seed = int(self.config["consumed_evidence"]["source_seed"]) + int(
            self.config["consumed_evidence"]["pair_seed_offset"]
        ) + self.width
        left, right, moduli = self.pair_stream(False, cases, pair_seed)
        targets = [a * b % p for a, b, p in zip(left, right, moduli, strict=True)]
        cache_key = sha256_text(
            json.dumps(
                {
                    "source": source,
                    "selector": self.config["selector"],
                    "consumed_evidence": self.config["consumed_evidence"],
                    "evaluation": self.config["evaluation"],
                    "pair_seed": pair_seed,
                },
                sort_keys=True,
            )
        )
        cache_path = self.output / f"paired-{cache_key}.json"
        started = time.perf_counter()
        cpu_started = time.process_time()
        peak_rss = self.process.memory_info().rss
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            if raw.get("cache_key") != cache_key:
                raise RuntimeError("paired cache key mismatch")
            original = raw["original"]
            swapped = raw["swapped"]
        else:
            original = self.rollout(left, right, moduli)
            self.check_ceiling(started)
            swapped = self.rollout(right, left, moduli)
            self.check_ceiling(started)
            raw = {
                "cache_key": cache_key,
                "left": left,
                "right": right,
                "moduli": moduli,
                "targets": targets,
                "original": original,
                "swapped": swapped,
            }
            cache_path.write_text(json.dumps(raw) + "\n")
        peak_rss = max(peak_rss, self.process.memory_info().rss)
        selected = select_by_margin(
            original["predictions"],
            swapped["predictions"],
            original["minimum_margins"],
            swapped["minimum_margins"],
        )
        selected_invalid = [
            swapped["invalid"][index]
            if choose
            else original["invalid"][index]
            for index, choose in enumerate(selected["use_swapped"])
        ]
        selected_result = {
            "predictions": selected["predictions"],
            "invalid": selected_invalid,
            "wall_seconds": original["wall_seconds"] + swapped["wall_seconds"],
            "aggregate_cpu_seconds": original["aggregate_cpu_seconds"]
            + swapped["aggregate_cpu_seconds"],
        }
        original_metrics = self.strip(original, targets)
        swapped_metrics = self.strip(swapped, targets)
        selected_metrics = self.strip(selected_result, targets)
        paired = paired_counts(
            original["predictions"], swapped["predictions"], targets
        )
        oracle_correct = paired["intersection"] + paired["original_only"] + paired[
            "swapped_only"
        ]
        original_correct = [
            prediction == target
            for prediction, target in zip(
                original["predictions"], targets, strict=True
            )
        ]
        swapped_correct = [
            prediction == target
            for prediction, target in zip(swapped["predictions"], targets, strict=True)
        ]
        selector = {
            "selected_swapped": sum(selected["use_swapped"]),
            "recoveries": sum(
                choose and (not first) and second
                for choose, first, second in zip(
                    selected["use_swapped"],
                    original_correct,
                    swapped_correct,
                    strict=True,
                )
            ),
            "losses": sum(
                choose and first and (not second)
                for choose, first, second in zip(
                    selected["use_swapped"],
                    original_correct,
                    swapped_correct,
                    strict=True,
                )
            ),
        }
        det_cases = int(self.config["evaluation"]["determinism_cases"])
        repeats = []
        for _ in range(2):
            first = self.rollout(left[:det_cases], right[:det_cases], moduli[:det_cases])
            second = self.rollout(right[:det_cases], left[:det_cases], moduli[:det_cases])
            choice = select_by_margin(
                first["predictions"],
                second["predictions"],
                first["minimum_margins"],
                second["minimum_margins"],
            )
            repeats.append(
                {
                    "original_predictions": first["predictions"],
                    "swapped_predictions": second["predictions"],
                    "original_margins": first["minimum_margins"],
                    "swapped_margins": second["minimum_margins"],
                    "original_invalid": first["invalid"],
                    "swapped_invalid": second["invalid"],
                    "use_swapped": choice["use_swapped"],
                }
            )
            self.check_ceiling(started)
        deterministic = repeats[0] == repeats[1]
        combined_wall = original_metrics["wall_seconds"] + swapped_metrics[
            "wall_seconds"
        ]
        projection = float(
            self.config["runtime_projection"]["incumbent_non_tier6_seconds"]
        ) + combined_wall / cases * int(
            self.config["runtime_projection"]["official_tier6_cases"]
        )
        expected_original = int(
            self.config["consumed_evidence"]["expected_original_correct"]
        )
        gates = self.config["gates"]
        results = {
            "oracle_union": oracle_correct / cases
            >= float(gates["minimum_oracle_union_accuracy"]),
            "selected_accuracy": selected_metrics["exact_accuracy"]
            >= float(gates["minimum_selected_accuracy"]),
            "validity": selected_metrics["any_invalid_residue_rate"]
            <= float(gates["maximum_selected_invalid_rate"]),
            "original_reproduction": original_metrics["correct"] == expected_original,
            "runtime": projection
            <= float(gates["maximum_integrated_candidate_wall_seconds"]),
            "determinism": deterministic,
        }
        passed = all(results.values())
        decision = (
            "authorize-fresh-commutative-swap-confirmation"
            if passed
            else "reject-consumed-commutative-swap-selector"
        )
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=self.config["run_id"]) as run:
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "completed",
                    "decision": decision,
                    "role": "consumed-evidence-commutative-swap-diagnostic",
                    "repository_commit": source["repository_commit"],
                    "repository_dirty": "false",
                }
            )
            mlflow.log_params(
                {
                    "source_checkpoint_sha256": source[
                        "source_checkpoint_sha256"
                    ],
                    "config_sha256": source["config_sha256"],
                    "runner_sha256": source["runner_sha256"],
                    "split_sha256": source["split_sha256"],
                    "cases": cases,
                    "selector": self.config["selector"]["name"],
                }
            )
            summary = {
                "run_id": self.config["run_id"],
                "status": "completed",
                "evidence_scope": "consumed-exp639-width128-cases",
                "source": source,
                "data": {
                    "pair_seed": pair_seed,
                    "cases": cases,
                    "cache": str(cache_path.relative_to(ROOT)),
                    "cache_sha256": sha256(cache_path),
                },
                "evaluation": {
                    "original": original_metrics,
                    "swapped": swapped_metrics,
                    "selected": selected_metrics,
                    "paired": paired,
                    "oracle_union": {
                        "correct": oracle_correct,
                        "exact_accuracy": oracle_correct / cases,
                    },
                    "selector": selector,
                    "determinism": {"cases": det_cases, "passed": deterministic},
                },
                "runtime_projection": {
                    "method": "candidate004-complete-plus-two-orientations-scaled-to-100-tier6-cases",
                    "integrated_candidate_wall_seconds": projection,
                },
                "gates": {"thresholds": gates, "results": results, "passed": passed},
                "decision": decision,
                "resource_budget": budget,
                "runtime": {
                    "wall_seconds": time.perf_counter() - started,
                    "aggregate_cpu_seconds": time.process_time() - cpu_started,
                    "peak_rss_bytes": max(peak_rss, self.process.memory_info().rss),
                },
                "mlflow": {
                    "tracking_uri": tracking_uri(),
                    "experiment_name": self.config["tracking"]["experiment_name"],
                    "experiment_id": experiment_id,
                    "run_id": run.info.run_id,
                },
                "limitations": [
                    "Consumed EXP-639 cases only; passing cannot establish fresh generalization.",
                    "The oracle union uses exact answers offline and is not an inference selector.",
                    "The integrated runtime projection is conservative and still requires official isolation before promotion.",
                ],
            }
            summary_path = self.output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            mlflow.log_metrics(
                {
                    "original_accuracy": original_metrics["exact_accuracy"],
                    "swapped_accuracy": swapped_metrics["exact_accuracy"],
                    "selected_accuracy": selected_metrics["exact_accuracy"],
                    "oracle_union_accuracy": oracle_correct / cases,
                    "integrated_candidate_wall_seconds": projection,
                    "wall_seconds": summary["runtime"]["wall_seconds"],
                }
            )
            mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
            mlflow.set_tags({"status": "completed", "decision": decision})
        print(
            json.dumps(
                {
                    "decision": decision,
                    "evaluation": summary["evaluation"],
                    "runtime_projection": summary["runtime_projection"],
                    "gates": summary["gates"],
                    "runtime": summary["runtime"],
                    "mlflow": summary["mlflow"],
                },
                indent=2,
            )
        )
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_640_commutative_operand_swap_diversity.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-640-commutative-operand-swap-diversity",
    )
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output)
    if args.calibrate:
        experiment.calibrate()
    else:
        experiment.formal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
