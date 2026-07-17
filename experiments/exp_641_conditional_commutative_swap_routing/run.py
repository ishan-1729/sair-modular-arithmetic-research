"""Execute the fixed conditional commutative-swap policy for EXP-641."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import psutil
import yaml


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
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def lowest_margin_indices(margins: list[float], count: int) -> list[int]:
    if not 0 <= count <= len(margins):
        raise ValueError("route count must be within the number of cases")
    return sorted(range(len(margins)), key=lambda index: (margins[index], index))[
        :count
    ]


def conditional_select(
    original_predictions: list[int],
    routed_swapped_predictions: list[int],
    original_margins: list[float],
    routed_swapped_margins: list[float],
    routed_indices: list[int],
) -> dict[str, Any]:
    if len(routed_indices) != len(routed_swapped_predictions) or len(
        routed_indices
    ) != len(routed_swapped_margins):
        raise ValueError("routed values must match routed indices")
    predictions = list(original_predictions)
    use_swapped = [False] * len(original_predictions)
    for local, index in enumerate(routed_indices):
        if routed_swapped_margins[local] > original_margins[index]:
            predictions[index] = routed_swapped_predictions[local]
            use_swapped[index] = True
    return {"predictions": predictions, "use_swapped": use_swapped}


class Experiment:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process()
        self.exp640 = load_module(
            "exp640_for_exp641",
            ROOT / "experiments/exp_640_commutative_operand_swap_diversity/run.py",
        )
        self.base = self.exp640.Experiment(self.config_path, self.output)
        self.cache_path = (ROOT / self.config["consumed_evidence"]["paired_cache"]).resolve()
        if sha256(self.cache_path) != self.config["consumed_evidence"][
            "paired_cache_sha256"
        ]:
            raise RuntimeError("EXP-640 paired cache hash mismatch")
        self.cached = json.loads(self.cache_path.read_text())

    def source_hashes(self) -> dict[str, Any]:
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
            "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "exp640_runner_sha256": sha256(
                ROOT / "experiments/exp_640_commutative_operand_swap_diversity/run.py"
            ),
            "source_checkpoint_sha256": sha256(self.base.source_path),
            "split_sha256": self.base.split_hash,
            "paired_cache_sha256": sha256(self.cache_path),
        }

    def check_ceiling(self, started: float) -> None:
        if time.perf_counter() - started > float(
            self.config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]
        ):
            raise TimeoutError("EXP-641 reached its calibrated wall ceiling")

    @staticmethod
    def exact_metrics(
        predictions: list[int], invalid: list[bool], targets: list[int]
    ) -> dict[str, Any]:
        correct = sum(a == b for a, b in zip(predictions, targets, strict=True))
        return {
            "cases": len(targets),
            "correct": correct,
            "exact_accuracy": correct / len(targets),
            "any_invalid_residue_rate": sum(invalid) / len(targets),
        }

    def execute_policy(
        self, left: list[int], right: list[int], moduli: list[int]
    ) -> dict[str, Any]:
        original = self.base.rollout(left, right, moduli)
        route_count = round(len(left) * float(self.config["routing"]["fraction"]))
        routed_indices = lowest_margin_indices(
            original["minimum_margins"], route_count
        )
        swapped = self.base.rollout(
            [right[index] for index in routed_indices],
            [left[index] for index in routed_indices],
            [moduli[index] for index in routed_indices],
        )
        selected = conditional_select(
            original["predictions"],
            swapped["predictions"],
            original["minimum_margins"],
            swapped["minimum_margins"],
            routed_indices,
        )
        selected_invalid = list(original["invalid"])
        for local, index in enumerate(routed_indices):
            if selected["use_swapped"][index]:
                selected_invalid[index] = swapped["invalid"][local]
        return {
            "original": original,
            "swapped": swapped,
            "routed_indices": routed_indices,
            "selected_predictions": selected["predictions"],
            "use_swapped": selected["use_swapped"],
            "selected_invalid": selected_invalid,
            "wall_seconds": original["wall_seconds"] + swapped["wall_seconds"],
            "aggregate_cpu_seconds": original["aggregate_cpu_seconds"]
            + swapped["aggregate_cpu_seconds"],
        }

    def reproduction(self, result: dict[str, Any]) -> dict[str, Any]:
        cached_original = self.cached["original"]
        cached_swapped = self.cached["swapped"]
        indices = result["routed_indices"]
        checks = {
            "original_predictions": result["original"]["predictions"]
            == cached_original["predictions"],
            "original_margins": result["original"]["minimum_margins"]
            == cached_original["minimum_margins"],
            "original_invalid": result["original"]["invalid"]
            == cached_original["invalid"],
            "routed_swapped_predictions": result["swapped"]["predictions"]
            == [cached_swapped["predictions"][index] for index in indices],
            "routed_swapped_margins": result["swapped"]["minimum_margins"]
            == [cached_swapped["minimum_margins"][index] for index in indices],
            "routed_swapped_invalid": result["swapped"]["invalid"]
            == [cached_swapped["invalid"][index] for index in indices],
        }
        checks["passed"] = all(checks.values())
        return checks

    def formal(self) -> dict[str, Any]:
        source = self.source_hashes()
        if source["dirty"]:
            raise RuntimeError("formal execution requires a clean tracked worktree")
        left = self.cached["left"]
        right = self.cached["right"]
        moduli = self.cached["moduli"]
        targets = self.cached["targets"]
        cases = int(self.config["consumed_evidence"]["cases"])
        if not len(left) == len(right) == len(moduli) == len(targets) == cases:
            raise RuntimeError("consumed cache case count mismatch")
        started = time.perf_counter()
        cpu_started = time.process_time()
        result = self.execute_policy(left, right, moduli)
        self.check_ceiling(started)
        reproduction = self.reproduction(result)
        selected_metrics = self.exact_metrics(
            result["selected_predictions"], result["selected_invalid"], targets
        )
        original_correct = [
            prediction == target
            for prediction, target in zip(
                result["original"]["predictions"], targets, strict=True
            )
        ]
        cached_swapped = self.cached["swapped"]
        routed = set(result["routed_indices"])
        routed_oracle = [
            original_correct[index]
            or (
                index in routed
                and cached_swapped["predictions"][index] == targets[index]
            )
            for index in range(cases)
        ]
        recoveries = sum(
            (not original_correct[index])
            and result["selected_predictions"][index] == targets[index]
            for index in range(cases)
        )
        losses = sum(
            original_correct[index]
            and result["selected_predictions"][index] != targets[index]
            for index in range(cases)
        )
        det_cases = int(self.config["evaluation"]["determinism_cases"])
        repeats = []
        for _ in range(2):
            repeat = self.execute_policy(
                left[:det_cases], right[:det_cases], moduli[:det_cases]
            )
            repeats.append(
                {
                    "original_predictions": repeat["original"]["predictions"],
                    "original_margins": repeat["original"]["minimum_margins"],
                    "original_invalid": repeat["original"]["invalid"],
                    "routed_indices": repeat["routed_indices"],
                    "swapped_predictions": repeat["swapped"]["predictions"],
                    "swapped_margins": repeat["swapped"]["minimum_margins"],
                    "swapped_invalid": repeat["swapped"]["invalid"],
                    "selected_predictions": repeat["selected_predictions"],
                    "use_swapped": repeat["use_swapped"],
                    "selected_invalid": repeat["selected_invalid"],
                }
            )
            self.check_ceiling(started)
        deterministic = repeats[0] == repeats[1]
        integrated_projection = float(
            self.config["runtime_projection"]["incumbent_non_tier6_seconds"]
        ) + result["wall_seconds"] / cases * int(
            self.config["runtime_projection"]["official_tier6_cases"]
        )
        thresholds = self.config["gates"]
        gate_results = {
            "routed_oracle": sum(routed_oracle) / cases
            >= float(thresholds["minimum_routed_oracle_accuracy"]),
            "selected_accuracy": selected_metrics["exact_accuracy"]
            >= float(thresholds["minimum_selected_accuracy"]),
            "validity": selected_metrics["any_invalid_residue_rate"]
            <= float(thresholds["maximum_selected_invalid_rate"]),
            "cached_output_reproduction": reproduction["passed"],
            "runtime": integrated_projection
            <= float(thresholds["maximum_integrated_candidate_wall_seconds"]),
            "determinism": deterministic,
        }
        passed = all(gate_results.values())
        decision = (
            "authorize-fresh-conditional-swap-confirmation"
            if passed
            else "reject-fixed-conditional-swap-routing"
        )
        cache_key = sha256_text(
            json.dumps(
                {
                    "source": source,
                    "routing": self.config["routing"],
                    "evidence": self.config["consumed_evidence"],
                },
                sort_keys=True,
            )
        )
        result_cache = self.output / f"conditional-{cache_key}.json"
        result_cache.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "routed_indices": result["routed_indices"],
                    "selected_predictions": result["selected_predictions"],
                    "use_swapped": result["use_swapped"],
                    "selected_invalid": result["selected_invalid"],
                    "reproduction": reproduction,
                }
            )
            + "\n"
        )
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=self.config["run_id"]) as run:
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "completed",
                    "decision": decision,
                    "role": "consumed-evidence-conditional-commutative-routing",
                    "repository_commit": source["repository_commit"],
                    "repository_dirty": "false",
                }
            )
            mlflow.log_params(
                {
                    "route_fraction": self.config["routing"]["fraction"],
                    "source_checkpoint_sha256": source[
                        "source_checkpoint_sha256"
                    ],
                    "paired_cache_sha256": source["paired_cache_sha256"],
                    "config_sha256": source["config_sha256"],
                    "runner_sha256": source["runner_sha256"],
                }
            )
            summary = {
                "run_id": self.config["run_id"],
                "status": "completed",
                "evidence_scope": "consumed-exp640-paired-cache",
                "source": source,
                "evaluation": {
                    "original": self.exact_metrics(
                        result["original"]["predictions"],
                        result["original"]["invalid"],
                        targets,
                    ),
                    "selected": selected_metrics,
                    "routed": {
                        "cases": len(result["routed_indices"]),
                        "fraction": len(result["routed_indices"]) / cases,
                        "selected_swapped": sum(result["use_swapped"]),
                        "recoveries": recoveries,
                        "losses": losses,
                    },
                    "routed_oracle": {
                        "correct": sum(routed_oracle),
                        "exact_accuracy": sum(routed_oracle) / cases,
                    },
                    "reproduction": reproduction,
                    "determinism": {"cases": det_cases, "passed": deterministic},
                },
                "runtime_projection": {
                    "method": "candidate004-complete-plus-measured-original-all-and-swapped-routed-scaled-to-100-tier6-cases",
                    "original_wall_seconds": result["original"]["wall_seconds"],
                    "routed_swapped_wall_seconds": result["swapped"]["wall_seconds"],
                    "conditional_wall_seconds": result["wall_seconds"],
                    "integrated_candidate_wall_seconds": integrated_projection,
                },
                "gates": {
                    "thresholds": thresholds,
                    "results": gate_results,
                    "passed": passed,
                },
                "decision": decision,
                "resource_budget": self.config["resource_budget"],
                "artifacts": {
                    "conditional_cache": str(result_cache.relative_to(ROOT)),
                    "conditional_cache_sha256": sha256(result_cache),
                },
                "runtime": {
                    "wall_seconds": time.perf_counter() - started,
                    "aggregate_cpu_seconds": time.process_time() - cpu_started,
                    "peak_rss_bytes": self.process.memory_info().rss,
                },
                "mlflow": {
                    "tracking_uri": tracking_uri(),
                    "experiment_name": self.config["tracking"]["experiment_name"],
                    "experiment_id": experiment_id,
                    "run_id": run.info.run_id,
                },
                "limitations": [
                    "Consumed EXP-640 evidence only; passing authorizes fresh confirmation, not promotion.",
                    "The routed oracle is offline coverage and is not used by inference.",
                    "Official isolated runtime is still required before candidate integration.",
                ],
            }
            summary_path = self.output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            mlflow.log_metrics(
                {
                    "selected_accuracy": selected_metrics["exact_accuracy"],
                    "routed_oracle_accuracy": sum(routed_oracle) / cases,
                    "recoveries": recoveries,
                    "losses": losses,
                    "integrated_candidate_wall_seconds": integrated_projection,
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
        default=ROOT / "configs/exp_641_conditional_commutative_swap_routing.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-641-conditional-commutative-swap-routing",
    )
    args = parser.parse_args()
    Experiment(args.config, args.output).formal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
