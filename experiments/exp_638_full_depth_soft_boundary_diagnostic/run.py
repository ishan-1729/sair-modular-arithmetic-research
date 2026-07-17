"""Run the frozen EXP-638 full-depth soft-boundary diagnostic."""

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
import torch
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
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def continuous_state_plane(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("intermediate temperature must be positive")
    return torch.tanh(logits / (2.0 * temperature))


def replace_state_plane(features: torch.Tensor, state_plane: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or features.shape[1] != 6:
        raise ValueError("expected [batch, 6, width] features")
    if features[:, 0].shape != state_plane.shape:
        raise ValueError("state plane shape mismatch")
    return torch.cat((state_plane[:, None, :], features[:, 1:, :]), dim=1)


class Diagnostic:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = int(self.config["seed"])
        self.exp637_module = load_module(
            "exp637_for_exp638",
            ROOT / "experiments/exp_637_teacher_distilled_macro_transition/run.py",
        )
        source_config = (ROOT / self.config["source_config"]).resolve()
        self.base = self.exp637_module.Experiment(source_config, self.output / "source")
        if sha256(self.base.teacher_path) != self.config["teacher_checkpoint_sha256"]:
            raise RuntimeError("teacher checkpoint hash mismatch")
        self.teacher = self.base.teacher
        self.data = self.base.data
        self.temperature = float(self.config["model"]["intermediate_temperature"])

    def continuous_transition(self, batch: tuple, width: int) -> dict[str, Any]:
        states, rights, moduli, digits, intermediates, finals, _q1, _q2 = batch
        predictions: list[int] = []
        intermediate_predictions: list[int] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        self.teacher.width = width
        with torch.inference_mode():
            for start in range(0, len(states), 512):
                stop = min(start + 512, len(states))
                high = [digit >> 1 for digit in digits[start:stop]]
                low = [digit & 1 for digit in digits[start:stop]]
                first_logits, _ = self.teacher(
                    self.data.features(
                        states[start:stop],
                        rights[start:stop],
                        moduli[start:stop],
                        high,
                        width,
                    )
                )
                intermediate_predictions.extend(self.exp637_module.decode(first_logits))
                context = self.data.features(
                    [0] * (stop - start),
                    rights[start:stop],
                    moduli[start:stop],
                    low,
                    width,
                )
                second_logits, _ = self.teacher(
                    replace_state_plane(
                        context,
                        continuous_state_plane(first_logits, self.temperature),
                    )
                )
                predictions.extend(self.exp637_module.decode(second_logits))
        cases = len(states)
        return {
            "cases": cases,
            "correct": sum(a == b for a, b in zip(predictions, finals)),
            "exact_accuracy": sum(a == b for a, b in zip(predictions, finals))
            / cases,
            "intermediate_correct": sum(
                a == b for a, b in zip(intermediate_predictions, intermediates)
            ),
            "intermediate_exact_accuracy": sum(
                a == b for a, b in zip(intermediate_predictions, intermediates)
            )
            / cases,
            "valid_residue_rate": sum(
                prediction < modulus
                for prediction, modulus in zip(predictions, moduli)
            )
            / cases,
            "predictions": predictions,
            "intermediate_predictions": intermediate_predictions,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
        }

    def exact_intermediate_transition(self, batch: tuple, width: int) -> dict[str, Any]:
        _states, rights, moduli, digits, intermediates, finals, _q1, _q2 = batch
        predictions: list[int] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        self.teacher.width = width
        with torch.inference_mode():
            for start in range(0, len(intermediates), 512):
                stop = min(start + 512, len(intermediates))
                low = [digit & 1 for digit in digits[start:stop]]
                logits, _ = self.teacher(
                    self.data.features(
                        intermediates[start:stop],
                        rights[start:stop],
                        moduli[start:stop],
                        low,
                        width,
                    )
                )
                predictions.extend(self.exp637_module.decode(logits))
        cases = len(intermediates)
        correct = sum(a == b for a, b in zip(predictions, finals))
        return {
            "cases": cases,
            "correct": correct,
            "exact_accuracy": correct / cases,
            "valid_residue_rate": sum(
                prediction < modulus
                for prediction, modulus in zip(predictions, moduli)
            )
            / cases,
            "predictions": predictions,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "diagnostic_only": True,
        }

    def continuous_rollout(
        self,
        left: list[int],
        right: list[int],
        moduli: list[int],
        width: int,
        keep_logits: bool = False,
    ) -> dict[str, Any]:
        states = [0] * len(left)
        exact_states = [0] * len(left)
        every_macro = [True] * len(left)
        every_intermediate = [True] * len(left)
        invalid = [False] * len(left)
        logit_records: list[list[list[float]]] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        self.teacher.width = width
        with torch.inference_mode():
            for position in range(width // 2):
                digits = [
                    (value >> (width - 2 * (position + 1))) & 3 for value in left
                ]
                high = [digit >> 1 for digit in digits]
                low = [digit & 1 for digit in digits]
                first_logits, _ = self.teacher(
                    self.data.features(states, right, moduli, high, width)
                )
                intermediate_predictions = self.exp637_module.decode(first_logits)
                context = self.data.features(
                    [0] * len(states), right, moduli, low, width
                )
                final_logits, _ = self.teacher(
                    replace_state_plane(
                        context,
                        continuous_state_plane(first_logits, self.temperature),
                    )
                )
                states = self.exp637_module.decode(final_logits)
                exact_intermediates: list[int] = []
                next_exact: list[int] = []
                for state, other, modulus, digit in zip(
                    exact_states, right, moduli, digits
                ):
                    intermediate, _q1, final, _q2 = self.data.exact_macro_transition(
                        state, other, modulus, digit
                    )
                    exact_intermediates.append(intermediate)
                    next_exact.append(final)
                every_intermediate = [
                    prior and prediction == target
                    for prior, prediction, target in zip(
                        every_intermediate,
                        intermediate_predictions,
                        exact_intermediates,
                    )
                ]
                every_macro = [
                    prior and prediction == target
                    for prior, prediction, target in zip(
                        every_macro, states, next_exact
                    )
                ]
                invalid = [
                    prior or prediction >= modulus
                    for prior, prediction, modulus in zip(invalid, states, moduli)
                ]
                exact_states = next_exact
                if keep_logits:
                    logit_records.append(
                        [first_logits.tolist(), final_logits.tolist()]
                    )
        expected = [(a * b) % p for a, b, p in zip(left, right, moduli)]
        correct = sum(a == b for a, b in zip(states, expected))
        result: dict[str, Any] = {
            "cases": len(left),
            "correct": correct,
            "exact_accuracy": correct / len(left),
            "all_macro_steps_exact_accuracy": sum(every_macro) / len(left),
            "all_intermediate_steps_exact_accuracy": sum(every_intermediate)
            / len(left),
            "any_invalid_residue_rate": sum(invalid) / len(left),
            "predictions": states,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "macro_steps": width // 2,
            "full_depth_teacher_calls_per_macro": 2,
        }
        if keep_logits:
            result["logits"] = logit_records
        return result

    @staticmethod
    def strip(result: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in result.items()
            if key not in {"predictions", "intermediate_predictions", "logits"}
        }

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        cpu_started = time.process_time()
        repository_commit = git_output("rev-parse", "HEAD")
        repository_dirty = bool(git_output("status", "--porcelain"))
        runner_path = Path(__file__)
        source_runner = ROOT / "experiments/exp_637_teacher_distilled_macro_transition/run.py"
        source_model = source_runner.with_name("model.py")
        source_data = source_runner.with_name("data.py")
        identity = {
            "config": sha256(self.config_path),
            "runner": sha256(runner_path),
            "source_runner": sha256(source_runner),
            "source_model": sha256(source_model),
            "source_data": sha256(source_data),
            "teacher": sha256(self.base.teacher_path),
            "splits": self.base.split_hash,
        }
        cache_key = sha256_text(json.dumps(identity, sort_keys=True))
        cache_path = self.output / f"diagnostic-{cache_key}.partial.json"
        cached: dict[str, Any] = {
            "cache_key": cache_key,
            "transition": {},
            "active_wall_seconds": 0.0,
            "arm_wall_seconds": 0.0,
            "arm_aggregate_cpu_seconds": 0.0,
        }
        if cache_path.exists():
            loaded = json.loads(cache_path.read_text())
            if loaded.get("cache_key") != cache_key:
                raise RuntimeError("diagnostic cache key mismatch")
            cached = loaded
        prior_active_wall = float(cached["active_wall_seconds"])
        segment_started = time.perf_counter()

        def save_cache() -> None:
            cached["active_wall_seconds"] = prior_active_wall + (
                time.perf_counter() - segment_started
            )
            cache_path.write_text(json.dumps(cached, indent=2) + "\n")
            if cached["active_wall_seconds"] > float(
                self.config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]
            ):
                raise TimeoutError("EXP-638 reached its measured wall ceiling")

        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=self.config["run_id"]) as mlflow_run:
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "running",
                    "role": "full-depth-soft-boundary-diagnostic",
                    "repository_commit": repository_commit,
                    "repository_dirty": str(repository_dirty).lower(),
                    "training_identities_only": "true",
                }
            )
            mlflow.log_params(
                {
                    "seed": self.seed,
                    "transition_cases": self.config["diagnostic"][
                        "transition_cases"
                    ],
                    "rollout_cases": self.config["diagnostic"]["rollout_cases"],
                    "intermediate_temperature": self.temperature,
                    "teacher_checkpoint_sha256": identity["teacher"],
                    "config_sha256": identity["config"],
                    "runner_sha256": identity["runner"],
                }
            )
            mlflow.log_artifact(str(self.config_path), artifact_path="preregistration")
            mlflow.log_artifact(
                str(ROOT / self.config["preregistration"]),
                artifact_path="preregistration",
            )
            for width in self.config["diagnostic"]["transition_widths"]:
                width = int(width)
                key = str(width)
                if key in cached["transition"]:
                    continue
                batch = self.data.MacroSampler(
                    self.base.splits[width]["train"], self.seed + 638200 + width
                ).transitions(int(self.config["diagnostic"]["transition_cases"]))
                hard = self.base.transition_teacher(batch, width)
                continuous = self.continuous_transition(batch, width)
                exact = self.exact_intermediate_transition(batch, width)
                cached["transition"][key] = {
                    "hard_composition": self.strip(hard),
                    "continuous_macro": self.strip(continuous),
                    "exact_intermediate": self.strip(exact),
                    "continuous_vs_hard": self.exp637_module.paired_counts(
                        continuous["predictions"], hard["predictions"], batch[5]
                    ),
                }
                for result in (hard, continuous, exact):
                    cached["arm_wall_seconds"] += result["wall_seconds"]
                    cached["arm_aggregate_cpu_seconds"] += result[
                        "aggregate_cpu_seconds"
                    ]
                save_cache()
            if "rollout" not in cached:
                width = int(self.config["diagnostic"]["rollout_width"])
                pairs = self.data.MacroSampler(
                    self.base.splits[width]["train"], self.seed + 638300 + width
                ).pairs(int(self.config["diagnostic"]["rollout_cases"]))
                warm = tuple(values[:2] for values in pairs)
                self.base.rollout_teacher(*warm, width)
                self.continuous_rollout(*warm, width)
                hard_rollout = self.base.rollout_teacher(*pairs, width)
                continuous_rollout = self.continuous_rollout(*pairs, width)
                targets = [(a * b) % p for a, b, p in zip(*pairs)]
                cached["rollout"] = {
                    "width": width,
                    "hard_composition": self.strip(hard_rollout),
                    "continuous_macro": self.strip(continuous_rollout),
                    "continuous_vs_hard": self.exp637_module.paired_counts(
                        continuous_rollout["predictions"],
                        hard_rollout["predictions"],
                        targets,
                    ),
                    "continuous_to_hard_wall_ratio": continuous_rollout[
                        "wall_seconds"
                    ]
                    / hard_rollout["wall_seconds"],
                }
                for result in (hard_rollout, continuous_rollout):
                    cached["arm_wall_seconds"] += result["wall_seconds"]
                    cached["arm_aggregate_cpu_seconds"] += result[
                        "aggregate_cpu_seconds"
                    ]
                save_cache()
            if "determinism" not in cached:
                width = int(self.config["diagnostic"]["rollout_width"])
                pairs = self.data.MacroSampler(
                    self.base.splits[width]["train"], self.seed + 638400 + width
                ).pairs(int(self.config["diagnostic"]["determinism_cases"]))
                first = self.continuous_rollout(*pairs, width, keep_logits=True)
                second = self.continuous_rollout(*pairs, width, keep_logits=True)
                determinism = {
                    "cases": len(pairs[0]),
                    "predictions_equal": first["predictions"] == second["predictions"],
                    "logits_equal": first["logits"] == second["logits"],
                    "correct_equal": first["correct"] == second["correct"],
                    "validity_equal": first["any_invalid_residue_rate"]
                    == second["any_invalid_residue_rate"],
                }
                determinism["passed"] = all(
                    determinism[key]
                    for key in (
                        "predictions_equal",
                        "logits_equal",
                        "correct_equal",
                        "validity_equal",
                    )
                )
                cached["determinism"] = determinism
                for result in (first, second):
                    cached["arm_wall_seconds"] += result["wall_seconds"]
                    cached["arm_aggregate_cpu_seconds"] += result[
                        "aggregate_cpu_seconds"
                    ]
                save_cache()
            e64 = cached["transition"]["64"]
            rollout = cached["rollout"]
            gates = self.config["gates"]
            transition_deficit = (
                e64["hard_composition"]["exact_accuracy"]
                - e64["continuous_macro"]["exact_accuracy"]
            )
            rollout_deficit = (
                rollout["hard_composition"]["exact_accuracy"]
                - rollout["continuous_macro"]["exact_accuracy"]
            )
            gate_results = {
                "continuous_transition": e64["continuous_macro"]["exact_accuracy"]
                >= float(gates["minimum_width64_continuous_transition"]),
                "continuous_rollout": rollout["continuous_macro"]["exact_accuracy"]
                >= float(gates["minimum_width64_continuous_rollout"]),
                "transition_deficit": transition_deficit
                <= float(gates["maximum_width64_transition_deficit_to_hard"]),
                "rollout_deficit": rollout_deficit
                <= float(gates["maximum_width64_rollout_deficit_to_hard"]),
                "runtime": rollout["continuous_to_hard_wall_ratio"]
                <= float(gates["maximum_continuous_to_hard_wall_ratio"]),
                "determinism": cached["determinism"]["passed"]
                or not bool(gates["require_determinism"]),
            }
            passed = all(gate_results.values())
            decision = (
                "authorize-full-depth-soft-boundary-training-rung"
                if passed
                else "reject-full-depth-soft-boundary-initialization"
            )
            summary = {
                "run_id": self.config["run_id"],
                "status": "completed",
                "hypothesis": self.config["hypothesis"],
                "mlflow": {
                    "tracking_uri": tracking_uri(),
                    "experiment_name": self.config["tracking"]["experiment_name"],
                    "experiment_id": experiment_id,
                    "run_id": mlflow_run.info.run_id,
                },
                "source": {
                    "repository_commit": repository_commit,
                    "dirty": repository_dirty,
                    "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                    **{f"{key}_sha256": value for key, value in identity.items() if key != "splits"},
                    "split_sha256": identity["splits"],
                },
                "evidence_scope": "training-prime-identities-only",
                "diagnostic": cached,
                "comparison": {
                    "width64_transition_deficit": transition_deficit,
                    "width64_rollout_deficit": rollout_deficit,
                },
                "gates": {"thresholds": gates, "results": gate_results, "passed": passed},
                "decision": decision,
                "resource_budget": self.config["resource_budget"],
                "runtime": {
                    "wall_seconds": time.perf_counter() - started,
                    "aggregate_cpu_seconds": time.process_time() - cpu_started,
                    "active_cached_wall_seconds": cached["active_wall_seconds"],
                    "arm_wall_seconds": cached["arm_wall_seconds"],
                    "arm_aggregate_cpu_seconds": cached[
                        "arm_aggregate_cpu_seconds"
                    ],
                    "torch_threads": self.config["resource_budget"]["torch_threads"],
                },
                "limitations": [
                    "Training identities only; this diagnostic is not held-out evidence.",
                    "The exact-intermediate arm is an offline diagnostic and is prohibited from candidate inference.",
                    "No weights were trained or selected.",
                ],
            }
            summary_path = self.output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            mlflow.log_metrics(
                {
                    "continuous_width64_transition": e64["continuous_macro"][
                        "exact_accuracy"
                    ],
                    "continuous_width64_rollout": rollout["continuous_macro"][
                        "exact_accuracy"
                    ],
                    "hard_width64_rollout": rollout["hard_composition"][
                        "exact_accuracy"
                    ],
                    "continuous_to_hard_wall_ratio": rollout[
                        "continuous_to_hard_wall_ratio"
                    ],
                    "wall_seconds": summary["runtime"]["wall_seconds"],
                    "aggregate_cpu_seconds": summary["runtime"][
                        "aggregate_cpu_seconds"
                    ],
                }
            )
            mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
            mlflow.set_tags({"status": "completed", "decision": decision})
        print(
            json.dumps(
                {
                    "decision": decision,
                    "transition": cached["transition"],
                    "rollout": cached["rollout"],
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
        default=ROOT / "configs/exp_638_full_depth_soft_boundary_diagnostic.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-638-full-depth-soft-boundary-diagnostic",
    )
    args = parser.parse_args()
    Diagnostic(args.config, args.output).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
