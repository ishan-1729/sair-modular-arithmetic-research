"""Run the calibrated matched learned neural-RAM viability experiment EXP-654."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from sympy import primepi
from torch import nn


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
            "exp300_for_exp654",
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


def process_cpu_seconds() -> float:
    times = psutil.Process().cpu_times()
    return float(times.user + times.system)


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def decode_bits(rows: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in rows.tolist()
    ]


def stage_sequence(config: dict[str, Any], mode: str) -> list[tuple[int, int]]:
    if mode == "calibration":
        return [
            (int(width), int(steps))
            for width, steps in config["calibration"]["stage_steps"].items()
        ]
    return [
        (int(stage["width"]), int(stage["steps"]))
        for stage in config["training"]["stages"]
    ]


def make_split(
    config: dict[str, Any], exp300, width: int, seed: int
) -> dict[str, list[int]]:
    data = config["data"]
    minimum_bits = max(3, width - int(data["prime_bit_span"]) + 1)
    low = 1 << (minimum_bits - 1)
    high = 1 << width
    requested = int(data["train_prime_identities"]) + int(data["test_prime_identities"])
    available = int(primepi(high - 1) - primepi(low - 1))
    if requested > available:
        raise ValueError(
            f"requested {requested} unique width-{width} primes but band contains {available}"
        )
    local = {
        "data": {
            "prime_bits_min": minimum_bits,
            "prime_bits_max": width,
            "train_prime_identities": int(data["train_prime_identities"]),
            "test_prime_identities": int(data["test_prime_identities"]),
        }
    }
    return exp300.bigint_prime_split(
        local, seed + int(data["split_seed_offset"]) + width
    )


def trajectory_batch(sampler, width: int, batch_size: int, exp300) -> dict[str, Any]:
    left, right, modulus = sampler.pairs(batch_size)
    states: list[int] = []
    repeated_right: list[int] = []
    repeated_modulus: list[int] = []
    input_bits: list[int] = []
    targets: list[int] = []
    target_rows: list[list[list[int]]] = []
    for multiplier, other, prime in zip(left, right, modulus, strict=True):
        state = 0
        rows = []
        for position in range(width):
            bit = (multiplier >> (width - 1 - position)) & 1
            target = (2 * state + bit * other) % prime
            states.append(state)
            repeated_right.append(other)
            repeated_modulus.append(prime)
            input_bits.append(bit)
            targets.append(target)
            rows.append([(target >> index) & 1 for index in range(width)])
            state = target
        target_rows.append(rows)
    return {
        "left": left,
        "right": right,
        "modulus": modulus,
        "multiplier_bits": exp300.bits(left, width).flip(1),
        "target_bits": torch.tensor(target_rows, dtype=torch.float32),
        "states": states,
        "repeated_right": repeated_right,
        "repeated_modulus": repeated_modulus,
        "input_bits": input_bits,
        "flat_targets": targets,
    }


def prefix_loss(
    logits: torch.Tensor, targets: torch.Tensor, training: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, float]]:
    per_bit = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mean = per_bit.mean()
    worst = per_bit.flatten(0, -2).mean(0).max()
    final = per_bit[:, -1].mean()
    loss = (
        mean
        + float(training["worst_bit_weight"]) * worst
        + float(training["final_prefix_weight"]) * final
    )
    exact = (logits.ge(0) == targets.bool()).all(-1)
    return loss, {
        "loss": float(loss.detach()),
        "mean_bce": float(mean.detach()),
        "worst_bit_bce": float(worst.detach()),
        "final_bce": float(final.detach()),
        "prefix_exact": float(exact.float().mean()),
        "final_exact": float(exact[:, -1].float().mean()),
    }


def summarize_predictions(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    moduli: list[int],
) -> tuple[dict[str, Any], list[int]]:
    prefix_exact = (predicted == truth).all(-1)
    final_exact = prefix_exact[:, -1]
    all_prefix = prefix_exact.all(-1)
    first_errors: list[int | None] = []
    invalid = []
    for row, prime in zip(predicted, moduli, strict=True):
        values = decode_bits(row)
        invalid.append(any(value >= prime for value in values))
    for row in prefix_exact.tolist():
        first_errors.append(next((index for index, exact in enumerate(row) if not exact), None))
    positions = [position for position in first_errors if position is not None]
    return (
        {
            "cases": int(predicted.shape[0]),
            "correct": int(final_exact.sum()),
            "exact_accuracy": float(final_exact.float().mean()),
            "all_prefix_exact_accuracy": float(all_prefix.float().mean()),
            "prefix_transition_exact_accuracy": float(prefix_exact.float().mean()),
            "any_invalid_residue_rate": sum(invalid) / len(invalid),
            "first_error": {
                "error_cases": len(positions),
                "mean_position": sum(positions) / len(positions) if positions else None,
                "minimum_position": min(positions) if positions else None,
                "maximum_position": max(positions) if positions else None,
                "histogram": {
                    str(position): positions.count(position)
                    for position in sorted(set(positions))
                },
            },
        },
        [position if position is not None else -1 for position in first_errors],
    )


class Experiment:
    ARM_ORDER = (
        "decoded_control",
        "decoded_wider",
        "nonaddressable_workspace",
        "neural_ram",
    )

    def __init__(self, config_path: Path, output: Path, mode: str) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.exp300 = load_exp300()
        self.cellular = load_module(
            "cellular_for_exp654",
            ROOT / "experiments/exp_220_width_independent_cellular/model.py",
        )
        self.persistent = load_module(
            "persistent_for_exp654",
            ROOT / "experiments/exp_620_persistent_latent_neural_gpu/model.py",
        )
        self.neural = load_module(
            "neural_ram_for_exp654",
            ROOT / "experiments/exp_654_neural_ram_executor/model.py",
        )
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.seed = int(
            self.config["data"][
                "calibration_seed" if mode == "calibration" else "formal_seed"
            ]
        )
        widths = set(self.config["evaluation"]["widths"])
        widths.update(width for width, _ in stage_sequence(self.config, mode))
        self.splits = {
            int(width): make_split(self.config, self.exp300, int(width), self.seed)
            for width in sorted(widths)
        }
        self.dataset_sha256 = sha256_text(json.dumps(self.splits, sort_keys=True))
        self.started = time.perf_counter()
        self.cpu_started = process_cpu_seconds()

    def source(self) -> dict[str, Any]:
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
            "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "model_sha256": sha256(Path(__file__).with_name("model.py")),
            "cell_model_sha256": sha256(
                ROOT / "experiments/exp_220_width_independent_cellular/model.py"
            ),
            "workspace_model_sha256": sha256(
                ROOT / "experiments/exp_620_persistent_latent_neural_gpu/model.py"
            ),
            "generator_runner_sha256": sha256(
                ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"
            ),
            "preregistration_sha256": sha256(ROOT / self.config["preregistration"]),
            "dataset_sha256": self.dataset_sha256,
        }

    def check_budget(self) -> None:
        budget = self.config["resource_budget"]
        prefix = "calibration_" if self.mode == "calibration" else ""
        wall = budget[f"{prefix}hard_runaway_ceiling_wall_seconds"]
        cpu = budget[f"{prefix}hard_runaway_ceiling_aggregate_cpu_seconds"]
        if wall is None or cpu is None:
            raise RuntimeError(f"{self.mode} resource budget is unresolved")
        if time.perf_counter() - self.started > float(wall):
            raise TimeoutError(f"EXP-654 {self.mode} wall runaway ceiling reached")
        if process_cpu_seconds() - self.cpu_started > float(cpu):
            raise TimeoutError(f"EXP-654 {self.mode} aggregate CPU runaway ceiling reached")

    def make_model(self, arm: str, width: int):
        model_config = self.config["models"]
        if arm in {"decoded_control", "decoded_wider"}:
            values = model_config[arm]
            return self.cellular.SharedDilatedConvGRU(
                width=width, hidden=int(values["hidden"]), cycles=int(values["cycles"])
            )
        if arm == "nonaddressable_workspace":
            values = model_config[arm]
            return self.persistent.PersistentLatentNeuralGPU(
                width=width,
                hidden=int(values["hidden"]),
                microsteps=int(values["microsteps"]),
            )
        if arm == "neural_ram":
            return self.neural.NeuralRAMExecutor(width=width, **model_config[arm])
        raise ValueError(arm)

    @staticmethod
    def is_decoded(arm: str) -> bool:
        return arm in {"decoded_control", "decoded_wider"}

    def train_arm(self, arm: str) -> dict[str, Any]:
        source = self.source()
        cache_key = sha256_text(
            json.dumps(
                {
                    "source": source,
                    "mode": self.mode,
                    "arm": arm,
                    "seed": self.seed,
                    "training": self.config[
                        "calibration" if self.mode == "calibration" else "training"
                    ],
                },
                sort_keys=True,
            )
        )
        arm_dir = self.output / self.mode / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        result_path = arm_dir / f"result-{cache_key}.json"
        if (
            self.mode == "formal"
            and self.config["resource_budget"]["cache_completed_arms"]
            and result_path.exists()
        ):
            cached = json.loads(result_path.read_text())
            result = cached.get("result", {})
            checkpoint = ROOT / result.get("checkpoint", {}).get("path", "missing")
            initial = ROOT / result.get("initial_checkpoint", {}).get("path", "missing")
            if (
                cached.get("cache_key") == cache_key
                and checkpoint.exists()
                and initial.exists()
                and sha256(checkpoint) == result["checkpoint"]["sha256"]
                and sha256(initial) == result["initial_checkpoint"]["sha256"]
            ):
                return {**result, "cached": True}

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        stages = stage_sequence(self.config, self.mode)
        model = self.make_model(arm, max(width for width, _ in stages))
        initial_path = arm_dir / "initial.safetensors"
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            initial_path,
        )
        training = self.config["training"]
        total_steps = sum(steps for _, steps in stages)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(training["minimum_learning_rate"]),
        )
        samplers = {
            width: self.exp300.BigIntSampler(
                self.splits[width]["train"],
                self.seed + int(self.config["data"]["training_pair_seed_offset"]) + width,
            )
            for width, _ in stages
        }
        batch_size = int(
            self.config["calibration"]["batch_size"]
            if self.mode == "calibration"
            else training["batch_size"]
        )
        history = []
        training_started = time.perf_counter()
        training_cpu_started = process_cpu_seconds()
        step = 0
        model.train()
        for width, stage_steps in stages:
            for _ in range(stage_steps):
                step += 1
                batch = trajectory_batch(samplers[width], width, batch_size, self.exp300)
                model.width = width
                if self.is_decoded(arm):
                    flat_logits, _ = model(
                        self.exp300.features(
                            batch["states"],
                            batch["repeated_right"],
                            batch["repeated_modulus"],
                            batch["input_bits"],
                            width,
                        )
                    )
                    logits = flat_logits.view(batch_size, width, width)
                else:
                    logits = model(
                        batch["multiplier_bits"],
                        model.static_features(batch["right"], batch["modulus"], width),
                        return_all=True,
                    )
                loss, stats = prefix_loss(logits, batch["target_bits"], training)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = float(
                    nn.utils.clip_grad_norm_(
                        model.parameters(), float(training["gradient_clip"])
                    )
                )
                optimizer.step()
                scheduler.step()
                if step == 1 or step % 100 == 0 or step == total_steps:
                    item = {
                        "step": step,
                        "width": width,
                        "elapsed_wall_seconds": time.perf_counter() - training_started,
                        "gradient_norm": gradient,
                        "learning_rate": scheduler.get_last_lr()[0],
                        **stats,
                    }
                    history.append(item)
                    if mlflow.active_run() is not None:
                        mlflow.log_metrics(
                            {
                                "train_loss": stats["loss"],
                                "train_prefix_exact": stats["prefix_exact"],
                                "train_final_exact": stats["final_exact"],
                                "gradient_norm": gradient,
                            },
                            step=step,
                        )
                    print(json.dumps({"arm": arm, **item}), flush=True)
                self.check_budget()
        training_timing = {
            "wall_seconds": time.perf_counter() - training_started,
            "aggregate_cpu_seconds": process_cpu_seconds() - training_cpu_started,
        }
        model.eval()
        checkpoint_path = arm_dir / "weights.safetensors"
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            checkpoint_path,
        )
        evaluation = {}
        evaluation_wall = evaluation_cpu = 0.0
        cases_by_width = (
            self.config["calibration"]["rollout_cases"]
            if self.mode == "calibration"
            else self.config["evaluation"]["rollout_cases"]
        )
        for width in self.config["evaluation"]["widths"]:
            width = int(width)
            result = self.evaluate(
                model,
                arm,
                width,
                int(cases_by_width[width]),
                seed_offset=0,
                memory_ablation="none",
                diagnostics=arm == "neural_ram",
            )
            evaluation[str(width)] = result
            evaluation_wall += result["timing"]["wall_seconds"]
            evaluation_cpu += result["timing"]["aggregate_cpu_seconds"]
        result = {
            "arm": arm,
            "cached": False,
            "parameters": model.parameter_count(),
            "initial_checkpoint": {
                "path": str(initial_path.relative_to(ROOT)),
                "sha256": sha256(initial_path),
                "bytes": initial_path.stat().st_size,
                "initialization": "ordinary-pytorch-default-plus-declared-xavier-registers",
                "seed": self.seed,
            },
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
            "training": {
                "stages": [{"width": width, "steps": steps} for width, steps in stages],
                "steps": total_steps,
                "trajectory_examples": total_steps * batch_size,
                "history": history,
                "timing": training_timing,
            },
            "evaluation": evaluation,
            "evaluation_timing": {
                "wall_seconds": evaluation_wall,
                "aggregate_cpu_seconds": evaluation_cpu,
            },
            "peak_rss_bytes_after_arm": peak_rss_bytes(),
        }
        result_path.write_text(
            json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n"
        )
        return result

    @torch.inference_mode()
    def evaluate(
        self,
        model,
        arm: str,
        width: int,
        cases: int,
        *,
        seed_offset: int,
        memory_ablation: str,
        diagnostics: bool,
    ) -> dict[str, Any]:
        model.width = width
        sampler = self.exp300.BigIntSampler(
            self.splits[width]["test"],
            self.seed
            + int(self.config["data"]["evaluation_pair_seed_offset"])
            + width
            + seed_offset,
        )
        started = time.perf_counter()
        cpu_started = process_cpu_seconds()
        seen = 0
        summaries = []
        diagnostic_sums: dict[str, float] = {}
        diagnostic_count = 0
        while seen < cases:
            size = min(64, cases - seen)
            batch = trajectory_batch(sampler, width, size, self.exp300)
            truth = batch["target_bits"].bool()
            if self.is_decoded(arm):
                state = [0] * size
                rows = []
                for position in range(width):
                    bits = [
                        (value >> (width - 1 - position)) & 1 for value in batch["left"]
                    ]
                    logits, _ = model(
                        self.exp300.features(
                            state, batch["right"], batch["modulus"], bits, width
                        )
                    )
                    predicted = logits.ge(0)
                    rows.append(predicted)
                    state = decode_bits(predicted)
                predictions = torch.stack(rows, dim=1)
                diagnostic_values = None
            elif arm == "neural_ram":
                output = model(
                    batch["multiplier_bits"],
                    model.static_features(batch["right"], batch["modulus"], width),
                    return_all=True,
                    return_diagnostics=diagnostics,
                    memory_ablation=memory_ablation,
                )
                if diagnostics:
                    logits, diagnostic_values = output
                else:
                    logits, diagnostic_values = output, None
                predictions = logits.ge(0)
            else:
                logits = model(
                    batch["multiplier_bits"],
                    model.static_features(batch["right"], batch["modulus"], width),
                    return_all=True,
                )
                predictions = logits.ge(0)
                diagnostic_values = None
            summary, _ = summarize_predictions(predictions, truth, batch["modulus"])
            summaries.append(summary)
            if diagnostic_values is not None:
                count = size * width
                diagnostic_count += count
                for key, values in diagnostic_values.items():
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(
                        values.sum()
                    )
            seen += size
            self.check_budget()
        positions = [
            int(position)
            for summary in summaries
            for position, count in summary["first_error"]["histogram"].items()
            for _ in range(count)
        ]
        correct = sum(summary["correct"] for summary in summaries)
        all_prefix = sum(
            round(summary["all_prefix_exact_accuracy"] * summary["cases"])
            for summary in summaries
        )
        invalid = sum(
            summary["any_invalid_residue_rate"] * summary["cases"]
            for summary in summaries
        )
        result = {
            "cases": cases,
            "correct": correct,
            "exact_accuracy": correct / cases,
            "all_prefix_exact_accuracy": all_prefix / cases,
            "prefix_transition_exact_accuracy": sum(
                summary["prefix_transition_exact_accuracy"] * summary["cases"]
                for summary in summaries
            )
            / cases,
            "any_invalid_residue_rate": invalid / cases,
            "first_error": {
                "error_cases": len(positions),
                "mean_position": sum(positions) / len(positions) if positions else None,
                "minimum_position": min(positions) if positions else None,
                "maximum_position": max(positions) if positions else None,
                "histogram": {
                    str(position): positions.count(position)
                    for position in sorted(set(positions))
                },
            },
            "timing": {
                "wall_seconds": time.perf_counter() - started,
                "aggregate_cpu_seconds": process_cpu_seconds() - cpu_started,
            },
        }
        if diagnostic_count:
            result["memory_diagnostics"] = {
                key: value / diagnostic_count for key, value in diagnostic_sums.items()
            }
        return result

    def load_trained_model(self, arm_result: dict[str, Any], width: int):
        model = self.make_model(arm_result["arm"], width)
        model.load_state_dict(load_file(ROOT / arm_result["checkpoint"]["path"]))
        return model.eval()

    def neural_ram_diagnostics(self, arm_result: dict[str, Any]) -> dict[str, Any]:
        model = self.load_trained_model(arm_result, 32)
        cases_by_width = (
            self.config["calibration"]["ablation_cases"]
            if self.mode == "calibration"
            else self.config["evaluation"]["ablation_cases"]
        )
        ablations = {}
        for width in (16, 32):
            ablations[str(width)] = {}
            for ablation in ("none", "reset", "permute", "uniform"):
                ablations[str(width)][ablation] = self.evaluate(
                    model,
                    "neural_ram",
                    width,
                    int(cases_by_width[width]),
                    seed_offset=65400,
                    memory_ablation=ablation,
                    diagnostics=True,
                )

        random.seed(self.seed + 9999)
        np.random.seed(self.seed + 9999)
        torch.manual_seed(self.seed + 9999)
        randomized = self.make_model("neural_ram", 16).eval()
        dependence_cases = int(
            self.config["calibration"]["randomized_dependence_cases"]
            if self.mode == "calibration"
            else self.config["evaluation"]["randomized_dependence_cases"]
        )
        trained_dependence = self.evaluate(
            model,
            "neural_ram",
            16,
            dependence_cases,
            seed_offset=65500,
            memory_ablation="none",
            diagnostics=False,
        )
        randomized_dependence = self.evaluate(
            randomized,
            "neural_ram",
            16,
            dependence_cases,
            seed_offset=65500,
            memory_ablation="none",
            diagnostics=False,
        )
        return {
            "ablations": ablations,
            "dependence": {
                "trained": trained_dependence,
                "randomized": randomized_dependence,
                "exact_drop": trained_dependence["exact_accuracy"]
                - randomized_dependence["exact_accuracy"],
                "randomized_initialization_seed": self.seed + 9999,
            },
        }

    @torch.inference_mode()
    def runtime_probe(self, arm_result: dict[str, Any]) -> dict[str, Any]:
        width = int(self.config["evaluation"]["runtime_probe_width"])
        cases = int(
            self.config["calibration"]["runtime_probe_cases"]
            if self.mode == "calibration"
            else self.config["evaluation"]["runtime_probe_cases"]
        )
        split = make_split(self.config, self.exp300, width, self.seed + 70000)
        sampler = self.exp300.BigIntSampler(split["test"], self.seed + 70100)
        left, right, modulus = sampler.pairs(cases)
        model = self.load_trained_model(arm_result, width)
        bits = self.exp300.bits(left, width).flip(1)
        static = model.static_features(right, modulus, width)
        model(bits[:1], static[:1])
        started = time.perf_counter()
        cpu_started = process_cpu_seconds()
        logits = model(bits, static)
        wall = time.perf_counter() - started
        cpu = process_cpu_seconds() - cpu_started
        projected_cases = int(self.config["evaluation"]["official_tier7_cases"])
        return {
            "width": width,
            "cases": cases,
            "output_shape": list(logits.shape),
            "wall_seconds": wall,
            "aggregate_cpu_seconds": cpu,
            "projected_cases": projected_cases,
            "projected_wall_seconds": wall / cases * projected_cases,
            "projected_aggregate_cpu_seconds": cpu / cases * projected_cases,
            "projection_method": "direct-width256-measured-per-case-throughput",
        }

    def run_all(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        results = {}
        partial_path = self.output / self.mode / "partial.json"
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        for arm in self.ARM_ORDER:
            results[arm] = self.train_arm(arm)
            partial_path.write_text(
                json.dumps({"status": "partial", "arms": results}, indent=2) + "\n"
            )
        diagnostics = self.neural_ram_diagnostics(results["neural_ram"])
        partial_path.write_text(
            json.dumps(
                {"status": "partial", "arms": results, "diagnostics": diagnostics},
                indent=2,
            )
            + "\n"
        )
        runtime = self.runtime_probe(results["neural_ram"])
        return results, diagnostics, runtime

    def calibration(self) -> dict[str, Any]:
        results, diagnostics, runtime = self.run_all()
        formal_steps = int(self.config["training"]["total_steps"])
        calibration_steps = sum(steps for _, steps in stage_sequence(self.config, "calibration"))
        formal_rollouts = self.config["evaluation"]["rollout_cases"]
        calibration_rollouts = self.config["calibration"]["rollout_cases"]
        formal_ablations = self.config["evaluation"]["ablation_cases"]
        calibration_ablations = self.config["calibration"]["ablation_cases"]
        projected_wall = projected_cpu = 0.0
        projection: dict[str, Any] = {"arms": {}}
        for arm, result in results.items():
            train_scale = formal_steps / calibration_steps
            train_wall = result["training"]["timing"]["wall_seconds"] * train_scale
            train_cpu = result["training"]["timing"]["aggregate_cpu_seconds"] * train_scale
            eval_wall = eval_cpu = 0.0
            for width in self.config["evaluation"]["widths"]:
                scale = float(formal_rollouts[width]) / float(calibration_rollouts[width])
                eval_wall += result["evaluation"][str(width)]["timing"]["wall_seconds"] * scale
                eval_cpu += result["evaluation"][str(width)]["timing"][
                    "aggregate_cpu_seconds"
                ] * scale
            projection["arms"][arm] = {
                "training_scale": train_scale,
                "projected_training_wall_seconds": train_wall,
                "projected_training_aggregate_cpu_seconds": train_cpu,
                "projected_evaluation_wall_seconds": eval_wall,
                "projected_evaluation_aggregate_cpu_seconds": eval_cpu,
            }
            projected_wall += train_wall + eval_wall
            projected_cpu += train_cpu + eval_cpu
        diagnostic_wall = diagnostic_cpu = 0.0
        for width in (16, 32):
            scale = float(formal_ablations[width]) / float(calibration_ablations[width])
            for result in diagnostics["ablations"][str(width)].values():
                diagnostic_wall += result["timing"]["wall_seconds"] * scale
                diagnostic_cpu += result["timing"]["aggregate_cpu_seconds"] * scale
        dependence_scale = float(self.config["evaluation"]["randomized_dependence_cases"]) / float(
            self.config["calibration"]["randomized_dependence_cases"]
        )
        for result in diagnostics["dependence"].values():
            if isinstance(result, dict) and "timing" in result:
                diagnostic_wall += result["timing"]["wall_seconds"] * dependence_scale
                diagnostic_cpu += result["timing"]["aggregate_cpu_seconds"] * dependence_scale
        runtime_scale = float(self.config["evaluation"]["runtime_probe_cases"]) / float(
            self.config["calibration"]["runtime_probe_cases"]
        )
        runtime_wall = runtime["wall_seconds"] * runtime_scale
        runtime_cpu = runtime["aggregate_cpu_seconds"] * runtime_scale
        projected_wall += diagnostic_wall + runtime_wall
        projected_cpu += diagnostic_cpu + runtime_cpu
        projection["diagnostics"] = {
            "projected_wall_seconds": diagnostic_wall,
            "projected_aggregate_cpu_seconds": diagnostic_cpu,
        }
        projection["runtime_probe"] = {
            "projected_wall_seconds": runtime_wall,
            "projected_aggregate_cpu_seconds": runtime_cpu,
        }
        multiplier = float(self.config["resource_budget"]["throughput_safety_multiplier"])
        setup_wall = max(90.0, projected_wall * 0.05)
        setup_cpu = max(60.0, projected_cpu * 0.02)
        soft_wall = math.ceil(projected_wall * multiplier + setup_wall)
        soft_cpu = math.ceil(projected_cpu * multiplier + setup_cpu)
        hard_wall = math.ceil(max(soft_wall * 1.5, soft_wall + 900.0))
        hard_cpu = math.ceil(max(soft_cpu * 1.5, soft_cpu + 2700.0))
        result = {
            "run_id": self.config["run_id"],
            "mode": "calibration",
            "status": "completed",
            "seed": self.seed,
            "source": self.source(),
            "arms": results,
            "diagnostics": diagnostics,
            "runtime_probe": runtime,
            "component_projection": projection,
            "measured_total": {
                "wall_seconds": time.perf_counter() - self.started,
                "aggregate_cpu_seconds": process_cpu_seconds() - self.cpu_started,
                "peak_rss_bytes": peak_rss_bytes(),
            },
            "projected_formal_without_headroom": {
                "wall_seconds": projected_wall,
                "aggregate_cpu_seconds": projected_cpu,
            },
            "recommended_resource_budget": {
                "projected_formal_wall_seconds": math.ceil(projected_wall),
                "projected_formal_aggregate_cpu_seconds": math.ceil(projected_cpu),
                "setup_and_serialisation_headroom_wall_seconds": math.ceil(setup_wall),
                "setup_and_serialisation_headroom_aggregate_cpu_seconds": math.ceil(
                    setup_cpu
                ),
                "throughput_safety_multiplier": multiplier,
                "soft_budget_wall_seconds": soft_wall,
                "soft_budget_aggregate_cpu_seconds": soft_cpu,
                "hard_runaway_ceiling_wall_seconds": hard_wall,
                "hard_runaway_ceiling_aggregate_cpu_seconds": hard_cpu,
            },
        }
        path = self.output / "calibration.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result

    def formal(self) -> dict[str, Any]:
        if self.config["resource_budget"]["calibration_status"] != "complete":
            raise RuntimeError("formal mode requires a committed complete calibration budget")
        source = self.source()
        if source["dirty"]:
            raise RuntimeError("formal mode requires a clean tracked worktree")
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        run_ids = {}
        with mlflow.start_run(run_name=self.config["run_id"]) as parent:
            parent_id = parent.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "running",
                    "role": "matched-addressable-neural-ram-viability",
                    "repository_commit": source["repository_commit"],
                    "repository_dirty": "false",
                }
            )
            mlflow.log_params(
                {
                    "seed": self.seed,
                    "steps": self.config["training"]["total_steps"],
                    "config_sha256": source["config_sha256"],
                    "runner_sha256": source["runner_sha256"],
                    "model_sha256": source["model_sha256"],
                    "dataset_sha256": source["dataset_sha256"],
                }
            )
            results = {}
            partial_path = self.output / "formal" / "partial.json"
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            for arm in self.ARM_ORDER:
                with mlflow.start_run(run_name=arm, nested=True) as nested:
                    run_ids[arm] = nested.info.run_id
                    results[arm] = self.train_arm(arm)
                    for width, evaluation in results[arm]["evaluation"].items():
                        mlflow.log_metric(f"width{width}_rollout_exact", evaluation["exact_accuracy"])
                partial_path.write_text(
                    json.dumps({"status": "partial", "arms": results}, indent=2) + "\n"
                )
            diagnostics = self.neural_ram_diagnostics(results["neural_ram"])
            runtime = self.runtime_probe(results["neural_ram"])
            neural = results["neural_ram"]
            width16 = neural["evaluation"]["16"]
            width32 = neural["evaluation"]["32"]
            best_control32 = max(
                results[arm]["evaluation"]["32"]["exact_accuracy"]
                for arm in self.ARM_ORDER[:-1]
            )
            ablation_base = diagnostics["ablations"]["16"]["none"]["exact_accuracy"]
            ablation_drops = {
                name: ablation_base
                - diagnostics["ablations"]["16"][name]["exact_accuracy"]
                for name in ("reset", "permute", "uniform")
            }
            memory = width16["memory_diagnostics"]
            gates = self.config["gates"]
            parameter_ratio = neural["parameters"] / results["decoded_wider"]["parameters"]
            gate_results = {
                "trained_width16": width16["exact_accuracy"]
                >= float(gates["minimum_neural_ram_width16_rollout"]),
                "width32_nonzero": width32["exact_accuracy"]
                >= float(gates["minimum_neural_ram_width32_rollout"]),
                "width32_gain": width32["exact_accuracy"] - best_control32
                >= float(gates["minimum_width32_gain_over_best_control"]),
                "causal_memory": max(ablation_drops.values())
                >= float(gates["minimum_width16_memory_ablation_drop"]),
                "register_activity": memory["register_delta_rms"]
                >= float(gates["minimum_register_delta_rms"]),
                "nonuniform_write": memory["maximum_write_probability"]
                >= float(gates["minimum_mean_maximum_write_probability"]),
                "validity": max(
                    result["any_invalid_residue_rate"]
                    for result in neural["evaluation"].values()
                )
                <= float(gates["maximum_invalid_rate"]),
                "randomized_dependence": diagnostics["dependence"]["exact_drop"]
                >= float(gates["minimum_randomized_dependence_drop"]),
                "parameter_match": float(gates["minimum_parameter_ratio_to_wider_control"])
                <= parameter_ratio
                <= float(gates["maximum_parameter_ratio_to_wider_control"]),
                "runtime": runtime["projected_wall_seconds"]
                <= float(gates["maximum_incremental_tier7_wall_seconds"]),
            }
            passed = all(gate_results.values())
            decision = (
                "advance-neural-ram-independent-training-seed-confirmation"
                if passed
                else "reject-current-neural-ram-formulation-before-scale-up"
            )
            summary = {
                "run_id": self.config["run_id"],
                "status": "completed",
                "hypothesis": self.config["hypothesis"],
                "source": source,
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "mlflow": mlflow.__version__,
                    "torch_threads": torch.get_num_threads(),
                },
                "mlflow": {
                    "tracking_uri": tracking_uri(),
                    "experiment_name": self.config["tracking"]["experiment_name"],
                    "experiment_id": experiment_id,
                    "parent_run_id": parent_id,
                    "arm_run_ids": run_ids,
                },
                "arms": results,
                "diagnostics": diagnostics,
                "runtime_probe": runtime,
                "comparison": {
                    "best_control_width32": best_control32,
                    "neural_ram_width32": width32["exact_accuracy"],
                    "width32_gain": width32["exact_accuracy"] - best_control32,
                    "width16_ablation_drops": ablation_drops,
                    "parameter_ratio_to_wider_control": parameter_ratio,
                },
                "gates": {**gates, "results": gate_results, "passed": passed},
                "decision": decision,
                "resource": {
                    "wall_seconds": time.perf_counter() - self.started,
                    "aggregate_cpu_seconds": process_cpu_seconds() - self.cpu_started,
                    "peak_rss_bytes": peak_rss_bytes(),
                    "resolved_budget": self.config["resource_budget"],
                },
                "limitations": [
                    "Single formal training seed viability rung; a pass requires independent training-seed confirmation.",
                    "Fresh generated widths 16/24/32 only; checked and private answers were not read.",
                    "Exact Python arithmetic is confined to offline prefix labels and evaluation.",
                    "The width-256 runtime probe measures throughput but not exact accuracy.",
                ],
            }
            summary_path = self.output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            mlflow.log_metrics(
                {
                    "neural_ram_width16": width16["exact_accuracy"],
                    "neural_ram_width32": width32["exact_accuracy"],
                    "best_control_width32": best_control32,
                    "width32_gain": summary["comparison"]["width32_gain"],
                    "runtime_projection": runtime["projected_wall_seconds"],
                    "wall_seconds": summary["resource"]["wall_seconds"],
                }
            )
            mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
            mlflow.log_artifact(str(self.config_path), artifact_path="preregistration")
            mlflow.set_tags({"status": "completed", "decision": decision})
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_654_neural_ram_executor_viability.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-654-neural-ram-executor-viability",
    )
    parser.add_argument("--mode", choices=("calibration", "formal"), required=True)
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode)
    result = experiment.calibration() if args.mode == "calibration" else experiment.formal()
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "mode": args.mode,
                "status": result["status"],
                "decision": result.get("decision"),
                "resource": result.get("measured_total", result.get("resource")),
                "recommended_resource_budget": result.get("recommended_resource_budget"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
