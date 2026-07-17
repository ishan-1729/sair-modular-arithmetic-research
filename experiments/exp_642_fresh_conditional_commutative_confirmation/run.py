"""Run the fresh fixed-policy confirmation for EXP-642."""

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


def load_exp300():
    previous_path = sys.path.copy()
    previous_model = sys.modules.get("model")
    had_model = "model" in sys.modules
    try:
        return load_module(
            "exp300_for_exp642",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
    finally:
        sys.path[:] = previous_path
        if had_model:
            sys.modules["model"] = previous_model
        else:
            sys.modules.pop("model", None)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def make_split(config: dict[str, Any], exp300, width: int) -> dict[str, list[int]]:
    fresh = config["fresh_evidence"]
    local = {
        "data": {
            "prime_bits_min": width - int(fresh["prime_bit_span"]) + 1,
            "prime_bits_max": width,
            "train_prime_identities": int(fresh["train_prime_identities"]),
            "test_prime_identities": int(fresh["test_prime_identities"]),
        }
    }
    return exp300.bigint_prime_split(
        local, int(config["seed"]) + int(fresh["split_seed_offset"]) + width
    )


class Experiment:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process()
        self.width = int(self.config["fresh_evidence"]["width"])
        self.exp300 = load_exp300()
        self.cellular = load_module(
            "cellular_for_exp642",
            ROOT / "experiments/exp_220_width_independent_cellular/model.py",
        )
        self.exp640 = load_module(
            "exp640_for_exp642",
            ROOT / "experiments/exp_640_commutative_operand_swap_diversity/run.py",
        )
        self.exp641 = load_module(
            "exp641_for_exp642",
            ROOT / "experiments/exp_641_conditional_commutative_swap_routing/run.py",
        )
        self.source_path = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source_path) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")
        torch.set_num_threads(int(self.config["evaluation"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(self.config["seed"]))
        self.split = make_split(self.config, self.exp300, self.width)
        self.split_hash = sha256_text(json.dumps(self.split, sort_keys=True))
        self.model = self.cellular.SharedDilatedConvGRU(
            width=self.width,
            hidden=int(self.config["model"]["hidden"]),
            cycles=int(self.config["model"]["cycles"]),
        )
        self.model.load_state_dict(load_file(self.source_path))
        self.model.eval()

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
            "exp641_runner_sha256": sha256(
                ROOT / "experiments/exp_641_conditional_commutative_swap_routing/run.py"
            ),
            "source_checkpoint_sha256": sha256(self.source_path),
            "split_sha256": self.split_hash,
        }

    def check_ceiling(self, started: float) -> None:
        if time.perf_counter() - started > float(
            self.config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]
        ):
            raise TimeoutError("EXP-642 reached its calibrated wall ceiling")

    def rollout(self, left: list[int], right: list[int], moduli: list[int]):
        return self.exp640.Experiment.rollout(self, left, right, moduli)

    @staticmethod
    def exact_metrics(predictions, invalid, targets):
        correct = sum(a == b for a, b in zip(predictions, targets, strict=True))
        return {
            "cases": len(targets),
            "correct": correct,
            "exact_accuracy": correct / len(targets),
            "any_invalid_residue_rate": sum(invalid) / len(targets),
        }

    def execute_policy(self, left, right, moduli):
        original = self.rollout(left, right, moduli)
        count = round(len(left) * float(self.config["routing"]["fraction"]))
        routed_indices = self.exp641.lowest_margin_indices(
            original["minimum_margins"], count
        )
        swapped = self.rollout(
            [right[index] for index in routed_indices],
            [left[index] for index in routed_indices],
            [moduli[index] for index in routed_indices],
        )
        selected = self.exp641.conditional_select(
            original["predictions"],
            swapped["predictions"],
            original["minimum_margins"],
            swapped["minimum_margins"],
            routed_indices,
        )
        invalid = list(original["invalid"])
        for local, index in enumerate(routed_indices):
            if selected["use_swapped"][index]:
                invalid[index] = swapped["invalid"][local]
        return {
            "original": original,
            "swapped": swapped,
            "routed_indices": routed_indices,
            "selected_predictions": selected["predictions"],
            "use_swapped": selected["use_swapped"],
            "selected_invalid": invalid,
            "wall_seconds": original["wall_seconds"] + swapped["wall_seconds"],
            "aggregate_cpu_seconds": original["aggregate_cpu_seconds"]
            + swapped["aggregate_cpu_seconds"],
        }

    def formal(self):
        source = self.source_hashes()
        if source["dirty"]:
            raise RuntimeError("formal execution requires a clean tracked worktree")
        seed = int(self.config["seed"])
        fresh = self.config["fresh_evidence"]
        started = time.perf_counter()
        cpu_started = time.process_time()
        transition_wall = time.perf_counter()
        transition_cpu = time.process_time()
        transition = self.exp300.evaluate_transitions(
            self.model,
            self.exp300.BigIntSampler(
                self.split["test"], seed + int(fresh["transition_seed_offset"]) + self.width
            ),
            int(self.config["evaluation"]["transition_cases"]),
        )
        transition["wall_seconds"] = time.perf_counter() - transition_wall
        transition["aggregate_cpu_seconds"] = time.process_time() - transition_cpu
        self.check_ceiling(started)
        left, right, moduli = self.exp300.BigIntSampler(
            self.split["test"], seed + int(fresh["pair_seed_offset"]) + self.width
        ).pairs(int(self.config["evaluation"]["rollout_cases"]))
        targets = [a * b % p for a, b, p in zip(left, right, moduli, strict=True)]
        result = self.execute_policy(left, right, moduli)
        self.check_ceiling(started)
        original_metrics = self.exact_metrics(
            result["original"]["predictions"], result["original"]["invalid"], targets
        )
        selected_metrics = self.exact_metrics(
            result["selected_predictions"], result["selected_invalid"], targets
        )
        routed = set(result["routed_indices"])
        swapped_by_index = {
            index: result["swapped"]["predictions"][local]
            for local, index in enumerate(result["routed_indices"])
        }
        original_correct = [
            prediction == target
            for prediction, target in zip(
                result["original"]["predictions"], targets, strict=True
            )
        ]
        routed_oracle = [
            original_correct[index]
            or (index in routed and swapped_by_index[index] == targets[index])
            for index in range(len(targets))
        ]
        recoveries = sum(
            (not original_correct[index])
            and result["selected_predictions"][index] == targets[index]
            for index in range(len(targets))
        )
        losses = sum(
            original_correct[index]
            and result["selected_predictions"][index] != targets[index]
            for index in range(len(targets))
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
        integrated = float(
            self.config["runtime_projection"]["incumbent_non_tier6_seconds"]
        ) + result["wall_seconds"] / len(targets) * int(
            self.config["runtime_projection"]["official_tier6_cases"]
        )
        gain = selected_metrics["exact_accuracy"] - original_metrics["exact_accuracy"]
        thresholds = self.config["gates"]
        gate_results = {
            "transition": transition["exact_accuracy"]
            >= float(thresholds["minimum_width128_transition_accuracy"]),
            "routed_oracle": sum(routed_oracle) / len(targets)
            >= float(thresholds["minimum_routed_oracle_accuracy"]),
            "selected_accuracy": selected_metrics["exact_accuracy"]
            >= float(thresholds["minimum_selected_accuracy"]),
            "gain": gain >= float(thresholds["minimum_selected_gain_over_original"]),
            "validity": selected_metrics["any_invalid_residue_rate"]
            <= float(thresholds["maximum_selected_invalid_rate"]),
            "runtime": integrated
            <= float(thresholds["maximum_integrated_candidate_wall_seconds"]),
            "determinism": deterministic,
        }
        passed = all(gate_results.values())
        decision = (
            "authorize-tier6-candidate-integration-and-wider-feasibility"
            if passed
            else "reject-fresh-conditional-commutative-confirmation"
        )
        cache_key = sha256_text(
            json.dumps(
                {
                    "source": source,
                    "fresh_evidence": fresh,
                    "routing": self.config["routing"],
                    "evaluation": self.config["evaluation"],
                },
                sort_keys=True,
            )
        )
        cache_path = self.output / f"fresh-{cache_key}.json"
        cache_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "left": left,
                    "right": right,
                    "moduli": moduli,
                    "targets": targets,
                    "original": result["original"],
                    "routed_indices": result["routed_indices"],
                    "routed_swapped": result["swapped"],
                    "selected_predictions": result["selected_predictions"],
                    "use_swapped": result["use_swapped"],
                    "selected_invalid": result["selected_invalid"],
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
                    "role": "fresh-tier6-conditional-commutative-confirmation",
                    "repository_commit": source["repository_commit"],
                    "repository_dirty": "false",
                }
            )
            mlflow.log_params(
                {
                    "seed": seed,
                    "route_fraction": self.config["routing"]["fraction"],
                    "source_checkpoint_sha256": source[
                        "source_checkpoint_sha256"
                    ],
                    "split_sha256": source["split_sha256"],
                    "config_sha256": source["config_sha256"],
                    "runner_sha256": source["runner_sha256"],
                }
            )
            summary = {
                "run_id": self.config["run_id"],
                "status": "completed",
                "evidence_scope": "fresh-generated-width128-prime-disjoint",
                "source": source,
                "data": {
                    "seed": seed,
                    "train_prime_identities": len(self.split["train"]),
                    "test_prime_identities": len(self.split["test"]),
                    "split_sha256": self.split_hash,
                },
                "evaluation": {
                    "transition": transition,
                    "original": original_metrics,
                    "selected": selected_metrics,
                    "gain": gain,
                    "routed": {
                        "cases": len(result["routed_indices"]),
                        "fraction": len(result["routed_indices"]) / len(targets),
                        "selected_swapped": sum(result["use_swapped"]),
                        "recoveries": recoveries,
                        "losses": losses,
                    },
                    "routed_oracle": {
                        "correct": sum(routed_oracle),
                        "exact_accuracy": sum(routed_oracle) / len(targets),
                    },
                    "determinism": {"cases": det_cases, "passed": deterministic},
                },
                "runtime_projection": {
                    "method": "candidate004-complete-plus-measured-fresh-conditional-work-scaled-to-100-tier6-cases",
                    "original_wall_seconds": result["original"]["wall_seconds"],
                    "routed_swapped_wall_seconds": result["swapped"]["wall_seconds"],
                    "conditional_wall_seconds": result["wall_seconds"],
                    "integrated_candidate_wall_seconds": integrated,
                },
                "gates": {
                    "thresholds": thresholds,
                    "results": gate_results,
                    "passed": passed,
                },
                "decision": decision,
                "resource_budget": self.config["resource_budget"],
                "artifacts": {
                    "fresh_cache": str(cache_path.relative_to(ROOT)),
                    "fresh_cache_sha256": sha256(cache_path),
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
                    "One fresh generated seed, not the required five independent full seeds.",
                    "The routed oracle is offline coverage and does not control inference.",
                    "Official isolated runtime and full candidate integration remain pending.",
                ],
            }
            summary_path = self.output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            mlflow.log_metrics(
                {
                    "transition_accuracy": transition["exact_accuracy"],
                    "original_accuracy": original_metrics["exact_accuracy"],
                    "selected_accuracy": selected_metrics["exact_accuracy"],
                    "selected_gain": gain,
                    "routed_oracle_accuracy": sum(routed_oracle) / len(targets),
                    "integrated_candidate_wall_seconds": integrated,
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
        default=ROOT / "configs/exp_642_fresh_conditional_commutative_confirmation.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-642-fresh-conditional-commutative-confirmation",
    )
    args = parser.parse_args()
    Experiment(args.config, args.output).formal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
