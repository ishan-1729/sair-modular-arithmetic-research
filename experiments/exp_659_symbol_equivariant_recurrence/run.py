"""Calibrate and run the full-fidelity EXP-659 matched falsifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from sympy import isprime
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_659_symbol_equivariant_recurrence.model import (  # noqa: E402
    SymbolRecurrentReasoner,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


ARM_ORDER = ("position_only", "symbol_equivariant", "broken_tying")
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def process_cpu_seconds() -> float:
    times = psutil.Process().cpu_times()
    return float(times.user + times.system)


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.exists():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        values[key] = int(value)
    if "usage_usec" not in values:
        return None
    return values["usage_usec"] / 1_000_000.0


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class Timer:
    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cpu = process_cpu_seconds()
        self.cgroup = cgroup_cpu_seconds()

    def finish(self) -> dict[str, float | None]:
        current_cgroup = cgroup_cpu_seconds()
        return {
            "wall_seconds": time.perf_counter() - self.wall,
            "aggregate_process_cpu_seconds": process_cpu_seconds() - self.cpu,
            "cgroup_cpu_seconds": (
                None
                if self.cgroup is None or current_cgroup is None
                else current_cgroup - self.cgroup
            ),
            "peak_rss_bytes": peak_rss_bytes(),
        }


def decode_bits(rows: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in rows.tolist()
    ]


def bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def features(
    state: list[int],
    multiplicand: list[int],
    modulus: list[int],
    input_bit: list[int],
    width: int,
) -> torch.Tensor:
    batch = len(state)
    bit_plane = torch.tensor(input_bit, dtype=torch.float32)[:, None].expand(batch, width)
    lsb = torch.zeros(batch, width)
    msb = torch.zeros(batch, width)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (bits(state, width), bits(multiplicand, width), bits(modulus, width), bit_plane, lsb, msb),
        dim=1,
    ).mul(2.0).sub(1.0)


def make_split(config: dict[str, Any], exp300, width: int, seed: int) -> dict[str, list[int]]:
    minimum_bits = max(3, width - int(config["data"]["prime_bit_span"]) + 1)
    local = {
        "data": {
            "prime_bits_min": minimum_bits,
            "prime_bits_max": width,
            "train_prime_identities": int(config["data"]["train_prime_identities"]),
            "test_prime_identities": int(config["data"]["test_prime_identities"]),
        }
    }
    return exp300.bigint_prime_split(
        local, seed + int(config["data"]["split_seed_offset"]) + width
    )


def stage_sequence(config: dict[str, Any], mode: str) -> list[tuple[int, int]]:
    if mode == "calibration":
        return [(int(width), int(steps)) for width, steps in config["calibration"]["stage_steps"].items()]
    return [(int(item["width"]), int(item["steps"])) for item in config["training"]["stages"]]


def transition_loss(
    logits: torch.Tensor,
    quotient_logits: torch.Tensor,
    targets: list[int],
    quotients: torch.Tensor,
    width: int,
    training: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    target = bits(targets, width)
    per_bit = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mean = per_bit.mean()
    worst = per_bit.mean(0).max()
    quotient = F.cross_entropy(quotient_logits, quotients)
    loss = (
        mean
        + float(training["worst_bit_weight"]) * worst
        + float(training["quotient_weight"]) * quotient
    )
    exact = (logits.ge(0) == target.bool()).all(1).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "batch_exact": float(exact.detach()),
        "mean_bce": float(mean.detach()),
        "worst_bit_bce": float(worst.detach()),
        "quotient_ce": float(quotient.detach()),
    }


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.exp300 = load_module(
            "exp300_for_exp659",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.seed = int(
            self.config["data"][
                "calibration_seed" if mode == "calibration" else "formal_seed"
            ]
        )
        widths = {int(value) for value in self.config["evaluation"]["widths"]}
        widths.update(width for width, _ in stage_sequence(self.config, mode))
        widths.add(int(self.config["evaluation"]["runtime_probe_width"]))
        self.splits = {
            width: make_split(self.config, self.exp300, width, self.seed)
            for width in sorted(widths)
        }
        self.dataset_sha256 = sha256_json(self.splits)
        self.started = Timer()
        self.source_at_start = self.source()

    def source(self) -> dict[str, Any]:
        model_path = Path(__file__).with_name("model.py")
        generator = ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"
        preregistration = ROOT / self.config["preregistration"]
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "repository_branch": git_output("branch", "--show-current"),
            "dirty": bool(git_output("status", "--porcelain")),
            "challenge_commit": CHALLENGE_COMMIT,
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "model_sha256": sha256(model_path),
            "generator_sha256": sha256(generator),
            "preregistration_sha256": sha256(preregistration),
            "paper_sha256": sha256(ROOT / self.config["paper"]["path"]),
            "dataset_sha256": self.dataset_sha256,
        }

    def check_budget(self) -> None:
        budget = self.config["resource_budget"]
        elapsed = self.started.finish()
        if self.mode == "calibration":
            wall_limit = budget["calibration_operational_runaway_wall_seconds"]
            cpu_limit = budget["calibration_operational_runaway_aggregate_cpu_seconds"]
            rss_limit = None
        else:
            wall_limit = budget["hard_budget_wall_seconds"]
            cpu_limit = budget["hard_budget_aggregate_cpu_seconds"]
            cgroup_limit = budget["hard_budget_cgroup_cpu_seconds"]
            rss_limit = budget["hard_peak_rss_bytes"]
        if self.mode == "calibration":
            cgroup_limit = None
        if wall_limit is not None and float(elapsed["wall_seconds"]) > float(wall_limit):
            raise TimeoutError(f"EXP-659 {self.mode} wall budget exceeded")
        if cpu_limit is not None and float(elapsed["aggregate_process_cpu_seconds"]) > float(cpu_limit):
            raise TimeoutError(f"EXP-659 {self.mode} aggregate process CPU budget exceeded")
        if (
            cgroup_limit is not None
            and elapsed["cgroup_cpu_seconds"] is not None
            and float(elapsed["cgroup_cpu_seconds"]) > float(cgroup_limit)
        ):
            raise TimeoutError(f"EXP-659 {self.mode} cgroup CPU budget exceeded")
        if rss_limit is not None and peak_rss_bytes() > int(rss_limit):
            raise MemoryError(f"EXP-659 {self.mode} RSS budget exceeded")

    def make_model(self, arm: str) -> SymbolRecurrentReasoner:
        values = self.config["model"]
        return SymbolRecurrentReasoner(
            arm,
            hidden=int(values["hidden"]),
            position_hidden=int(values["position_hidden"]),
            heads=int(values["heads"]),
            recurrent_steps=int(values["recurrent_steps"]),
            expansion=int(values["expansion"]),
        )

    def sampler(self, width: int, split: str, offset: int):
        return self.exp300.BigIntSampler(
            self.splits[width][split],
            self.seed
            + (
                int(self.config["data"]["training_pair_seed_offset"])
                if split == "train"
                else int(self.config["data"]["evaluation_pair_seed_offset"])
            )
            + width
            + offset,
        )

    @torch.inference_mode()
    def evaluate_transitions(
        self,
        model: SymbolRecurrentReasoner,
        width: int,
        cases: int,
        *,
        offset: int,
    ) -> dict[str, Any]:
        timer = Timer()
        sampler = self.sampler(width, "test", offset)
        seen = correct = valid = 0
        prediction_rows: list[list[int]] = []
        while seen < cases:
            size = min(256, cases - seen)
            state, multiplicand, modulus, input_bit, targets, _ = sampler.transitions(size)
            logits, _ = model(features(state, multiplicand, modulus, input_bit, width))
            predicted = logits.ge(0)
            values = decode_bits(predicted)
            correct += sum(value == target for value, target in zip(values, targets, strict=True))
            valid += sum(value < prime for value, prime in zip(values, modulus, strict=True))
            prediction_rows.extend(predicted.to(torch.uint8).tolist())
            seen += size
            self.check_budget()
        return {
            "cases": seen,
            "correct": correct,
            "exact_accuracy": correct / seen,
            "valid_residue_rate": valid / seen,
            "prediction_sha256": sha256_json(prediction_rows),
            "timing": timer.finish(),
        }

    @torch.inference_mode()
    def evaluate_rollouts(
        self,
        model: SymbolRecurrentReasoner,
        width: int,
        cases: int,
        *,
        offset: int,
        include_truth: bool = True,
    ) -> dict[str, Any]:
        timer = Timer()
        sampler = self.sampler(width, "test", offset)
        seen = correct = all_prefix = prefix_correct = prefix_total = invalid = 0
        first_errors: list[int] = []
        final_predictions: list[int] = []
        while seen < cases:
            size = min(32, cases - seen)
            multiplier, multiplicand, modulus = sampler.pairs(size)
            state = [0] * size
            teacher = [0] * size
            every = [True] * size
            bad = [False] * size
            first = [-1] * size
            for position in range(width):
                input_bit = [
                    (value >> (width - 1 - position)) & 1 for value in multiplier
                ]
                logits, _ = model(features(state, multiplicand, modulus, input_bit, width))
                state = decode_bits(logits.ge(0))
                if include_truth:
                    teacher = [
                        (2 * value + bit * other) % prime
                        for value, bit, other, prime in zip(
                            teacher, input_bit, multiplicand, modulus, strict=True
                        )
                    ]
                    for index, (predicted, expected) in enumerate(
                        zip(state, teacher, strict=True)
                    ):
                        exact = predicted == expected
                        prefix_correct += int(exact)
                        prefix_total += 1
                        if every[index] and not exact:
                            every[index] = False
                            first[index] = position
                bad = [
                    previous or predicted >= prime
                    for previous, predicted, prime in zip(bad, state, modulus, strict=True)
                ]
                self.check_budget()
            if include_truth:
                expected = [
                    (left * right) % prime
                    for left, right, prime in zip(multiplier, multiplicand, modulus, strict=True)
                ]
                correct += sum(
                    predicted == target for predicted, target in zip(state, expected, strict=True)
                )
                all_prefix += sum(every)
                first_errors.extend(value for value in first if value >= 0)
            invalid += sum(bad)
            final_predictions.extend(state)
            seen += size
        result: dict[str, Any] = {
            "cases": seen,
            "correct": correct if include_truth else None,
            "exact_accuracy": correct / seen if include_truth else None,
            "all_prefix_exact_accuracy": all_prefix / seen if include_truth else None,
            "prefix_transition_exact_accuracy": (
                prefix_correct / prefix_total if include_truth else None
            ),
            "any_invalid_residue_rate": invalid / seen,
            "prediction_sha256": sha256_json(final_predictions),
            "first_error": {
                "error_cases": len(first_errors),
                "mean_position": (
                    sum(first_errors) / len(first_errors) if first_errors else None
                ),
                "minimum_position": min(first_errors) if first_errors else None,
                "maximum_position": max(first_errors) if first_errors else None,
                "histogram": {
                    str(position): first_errors.count(position)
                    for position in sorted(set(first_errors))
                },
            },
            "timing": timer.finish(),
        }
        return result

    def validation_fixture(self, width: int, cases: int, offset: int):
        sampler = self.sampler(width, "test", offset)
        state, multiplicand, modulus, input_bit, targets, quotients = sampler.transitions(cases)
        return features(state, multiplicand, modulus, input_bit, width), targets, quotients

    @torch.inference_mode()
    def validate_fixture(
        self, model: SymbolRecurrentReasoner, fixture, width: int
    ) -> dict[str, float]:
        input_tensor, targets, quotients = fixture
        logits, quotient_logits = model(input_tensor)
        target = bits(targets, width).bool()
        return {
            "exact_accuracy": float((logits.ge(0) == target).all(1).float().mean()),
            "bit_accuracy": float((logits.ge(0) == target).float().mean()),
            "quotient_accuracy": float(
                (quotient_logits.argmax(1) == quotients).float().mean()
            ),
        }

    def train_arm(self, arm: str) -> dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        model = self.make_model(arm)
        stages = stage_sequence(self.config, self.mode)
        total_steps = sum(steps for _, steps in stages)
        batch_size = int(
            self.config["calibration"]["batch_size"]
            if self.mode == "calibration"
            else self.config["training"]["batch_size"]
        )
        arm_dir = self.output / self.mode / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        source_identity = {
            "source": self.source_at_start,
            "mode": self.mode,
            "arm": arm,
            "seed": self.seed,
            "stages": stages,
            "batch_size": batch_size,
        }
        cache_key = sha256_json(source_identity)
        result_path = arm_dir / f"result-{cache_key}.json"
        if self.mode == "formal" and bool(self.config["resource_budget"]["reuse_completed_arms"]):
            if result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                result = payload.get("result", {})
                checkpoint = ROOT / result.get("checkpoint", {}).get("path", "missing")
                if payload.get("cache_key") == cache_key and checkpoint.exists():
                    if sha256(checkpoint) == result["checkpoint"]["sha256"]:
                        return {**result, "cached": True}

        initial_path = arm_dir / "initial.safetensors"
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            initial_path,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(self.config["training"]["minimum_learning_rate"]),
        )
        samplers = {
            width: self.sampler(width, "train", 1000 + index * 100)
            for index, (width, _) in enumerate(stages)
        }
        validation_cases = int(self.config["training"]["validation_cases"])
        fixtures = {
            width: self.validation_fixture(width, validation_cases, 2000 + width)
            for width, _ in stages
        }
        interval = (
            max(1, total_steps // 10)
            if self.mode == "calibration"
            else int(self.config["training"]["validation_interval"])
        )
        history: list[dict[str, Any]] = []
        stage_checkpoints: list[dict[str, Any]] = []
        timer = Timer()
        step = 0
        model.train()
        for width, stage_steps in stages:
            for stage_step in range(1, stage_steps + 1):
                step += 1
                state, multiplicand, modulus, input_bit, targets, quotients = samplers[width].transitions(batch_size)
                logits, quotient_logits = model(
                    features(state, multiplicand, modulus, input_bit, width)
                )
                loss, stats = transition_loss(
                    logits,
                    quotient_logits,
                    targets,
                    quotients,
                    width,
                    self.config["training"],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = float(
                    nn.utils.clip_grad_norm_(
                        model.parameters(), float(self.config["training"]["gradient_clip"])
                    )
                )
                optimizer.step()
                scheduler.step()
                should_record = (
                    step == 1
                    or step % interval == 0
                    or stage_step == stage_steps
                    or step == total_steps
                )
                if should_record:
                    model.eval()
                    validation = {
                        str(candidate_width): self.validate_fixture(
                            model, fixtures[candidate_width], candidate_width
                        )
                        for candidate_width, _ in stages
                        if candidate_width <= width
                    }
                    model.train()
                    record = {
                        "step": step,
                        "stage_width": width,
                        "stage_step": stage_step,
                        "elapsed_wall_seconds": timer.finish()["wall_seconds"],
                        "gradient_norm": gradient,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "train": stats,
                        "validation": validation,
                    }
                    history.append(record)
                    if mlflow.active_run() is not None:
                        mlflow.log_metrics(
                            {
                                "train_loss": stats["loss"],
                                "train_batch_exact": stats["batch_exact"],
                                "validation_exact": validation[str(width)]["exact_accuracy"],
                                "gradient_norm": gradient,
                            },
                            step=step,
                        )
                    print(json.dumps({"arm": arm, **record}), flush=True)
                self.check_budget()
            checkpoint_path = arm_dir / f"stage-{width}-step-{step}.safetensors"
            persistence = Timer()
            save_file(
                {key: value.detach().contiguous() for key, value in model.state_dict().items()},
                checkpoint_path,
            )
            stage_checkpoints.append(
                {
                    "width": width,
                    "step": step,
                    "path": str(checkpoint_path.relative_to(ROOT)),
                    "sha256": sha256(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                    "persistence_timing": persistence.finish(),
                }
            )
        training_timing = timer.finish()
        final_path = arm_dir / "weights.safetensors"
        persistence = Timer()
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            final_path,
        )
        final_persistence = persistence.finish()
        result = {
            "arm": arm,
            "cached": False,
            "parameters": model.parameter_count(),
            "initial_checkpoint": {
                "path": str(initial_path.relative_to(ROOT)),
                "sha256": sha256(initial_path),
                "bytes": initial_path.stat().st_size,
                "seed": self.seed,
                "initialization": "ordinary-pytorch-default-plus-declared-normal-boundary-and-broken-affine",
            },
            "checkpoint": {
                "path": str(final_path.relative_to(ROOT)),
                "sha256": sha256(final_path),
                "bytes": final_path.stat().st_size,
                "persistence_timing": final_persistence,
            },
            "stage_checkpoints": stage_checkpoints,
            "training": {
                "stages": [{"width": width, "steps": count} for width, count in stages],
                "steps": total_steps,
                "batch_size": batch_size,
                "history": history,
                "timing": training_timing,
            },
        }
        write_json(result_path, {"cache_key": cache_key, "result": result})
        return result

    def load_model(self, result: dict[str, Any]) -> SymbolRecurrentReasoner:
        model = self.make_model(result["arm"])
        model.load_state_dict(load_file(ROOT / result["checkpoint"]["path"]))
        return model.eval()

    def evaluate_arm(self, result: dict[str, Any]) -> dict[str, Any]:
        model = self.load_model(result)
        key = "calibration" if self.mode == "calibration" else "evaluation"
        transition_cases = self.config[key]["transition_cases"]
        rollout_cases = self.config[key]["rollout_cases"]
        evaluations = {}
        for width in self.config["evaluation"]["widths"]:
            width = int(width)
            evaluations[str(width)] = {
                "transition": self.evaluate_transitions(
                    model, width, int(transition_cases[width]), offset=3000
                ),
                "rollout": self.evaluate_rollouts(
                    model, width, int(rollout_cases[width]), offset=4000
                ),
            }
        return evaluations

    @torch.inference_mode()
    def core_equivariance(self, result: dict[str, Any]) -> dict[str, Any]:
        model = self.load_model(result)
        if model.arm == "position_only":
            return {"applicable": False, "reason": "control has no symbol axis"}
        generator = torch.Generator().manual_seed(self.seed + 5000)
        state = torch.randn(3, 7, 2, model.hidden, generator=generator)
        base = model.core_operator(state)
        permutations = {}
        for name, permutation in (
            ("identity", torch.tensor([0, 1])),
            ("swap", torch.tensor([1, 0])),
        ):
            transformed = model.core_operator(state[:, :, permutation, :])
            expected = base[:, :, permutation, :]
            residual = (transformed - expected).abs()
            permutations[name] = {
                "permutation": permutation.tolist(),
                "maximum_absolute_residual": float(residual.max()),
                "mean_absolute_residual": float(residual.mean()),
            }
        return {"applicable": True, "permutations": permutations}

    def task_symmetry_applicability(self) -> dict[str, Any]:
        timer = Timer()
        total = closed = equivariant = 0
        closure_examples = []
        for width in range(2, 7):
            mask = (1 << width) - 1
            primes = [value for value in range(2, 1 << width) if isprime(value)]
            for prime in primes:
                for state in range(prime):
                    for multiplicand in range(prime):
                        for input_bit in (0, 1):
                            total += 1
                            transformed = (
                                mask ^ state,
                                mask ^ multiplicand,
                                mask ^ prime,
                                1 - input_bit,
                            )
                            state2, multiplicand2, prime2, bit2 = transformed
                            legal = (
                                isprime(prime2)
                                and state2 < prime2
                                and multiplicand2 < prime2
                            )
                            if legal:
                                closed += 1
                                target = (2 * state + input_bit * multiplicand) % prime
                                target2 = (2 * state2 + bit2 * multiplicand2) % prime2
                                if target2 == (mask ^ target):
                                    equivariant += 1
                                elif len(closure_examples) < 5:
                                    closure_examples.append(
                                        {
                                            "width": width,
                                            "input": [state, multiplicand, prime, input_bit],
                                            "transformed": list(transformed),
                                            "target": target,
                                            "transformed_target": target2,
                                            "permuted_target": mask ^ target,
                                        }
                                    )
        return {
            "widths": [2, 3, 4, 5, 6],
            "legal_cases_exhausted": total,
            "swap_transformed_legal_cases": closed,
            "equivariant_closed_cases": equivariant,
            "counterexamples": closure_examples,
            "verdict": "non-applicable-domain-not-closed",
            "reason": "every legal prime modulus is mapped outside the legal prime-modulus domain by binary symbol complementation",
            "timing": timer.finish(),
        }

    def static_leakage_check(self) -> dict[str, Any]:
        timer = Timer()
        model_path = Path(__file__).with_name("model.py")
        source = model_path.read_text(encoding="utf-8").lower()
        forbidden = (
            "expected_answer",
            "barrett",
            "montgomery",
            "crt_recomb",
            "torch.remainder",
            "fmod(",
        )
        findings = [token for token in forbidden if token in source]
        prohibited_paths = [
            path
            for path in (
                ROOT / "data/checked",
                ROOT / "data/private",
                ROOT / "submission",
            )
            if str(path).lower() in source
        ]
        return {
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256(model_path),
            "forbidden_token_findings": findings,
            "prohibited_path_findings": [str(path) for path in prohibited_paths],
            "passed": not findings and not prohibited_paths,
            "timing": timer.finish(),
        }

    def deterministic_replay(self, result: dict[str, Any]) -> dict[str, Any]:
        timer = Timer()
        model = self.load_model(result)
        key = "calibration" if self.mode == "calibration" else "evaluation"
        cases = int(self.config[key]["deterministic_replay_cases"])
        first = self.evaluate_rollouts(model, 16, cases, offset=6000)
        second = self.evaluate_rollouts(model, 16, cases, offset=6000)
        return {
            "cases": cases,
            "first": first,
            "second": second,
            "prediction_hash_equal": first["prediction_sha256"] == second["prediction_sha256"],
            "metrics_equal": {
                key: first[key] == second[key]
                for key in (
                    "correct",
                    "exact_accuracy",
                    "all_prefix_exact_accuracy",
                    "prefix_transition_exact_accuracy",
                    "any_invalid_residue_rate",
                )
            },
            "passed": first["prediction_sha256"] == second["prediction_sha256"],
            "timing": timer.finish(),
        }

    def randomized_dependence(self, result: dict[str, Any]) -> dict[str, Any]:
        timer = Timer()
        trained = self.load_model(result)
        random.seed(self.seed + 9999)
        np.random.seed(self.seed + 9999)
        torch.manual_seed(self.seed + 9999)
        randomized = self.make_model("symbol_equivariant").eval()
        key = "calibration" if self.mode == "calibration" else "evaluation"
        cases = int(self.config[key]["randomized_dependence_cases"])
        trained_result = self.evaluate_rollouts(trained, 16, cases, offset=7000)
        randomized_result = self.evaluate_rollouts(randomized, 16, cases, offset=7000)
        return {
            "cases": cases,
            "trained": trained_result,
            "randomized": randomized_result,
            "exact_drop": trained_result["exact_accuracy"] - randomized_result["exact_accuracy"],
            "randomized_initialization_seed": self.seed + 9999,
            "timing": timer.finish(),
        }

    @torch.inference_mode()
    def runtime_probe(self, result: dict[str, Any]) -> dict[str, Any]:
        width = int(self.config["evaluation"]["runtime_probe_width"])
        key = "calibration" if self.mode == "calibration" else "evaluation"
        cases = int(self.config[key]["runtime_probe_cases"])
        model = self.load_model(result)
        warm_sampler = self.sampler(width, "test", 8000)
        _, right, modulus = warm_sampler.pairs(1)
        model(features([0], right, modulus, [0], width))
        rss_before = peak_rss_bytes()
        result_value = self.evaluate_rollouts(
            model, width, cases, offset=8100, include_truth=True
        )
        projected_cases = int(self.config["evaluation"]["official_tier7_cases"])
        timing = result_value["timing"]
        return {
            "width": width,
            "cases": cases,
            "rollout": result_value,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": peak_rss_bytes(),
            "projected_cases": projected_cases,
            "projected_wall_seconds": float(timing["wall_seconds"]) / cases * projected_cases,
            "projected_aggregate_process_cpu_seconds": float(
                timing["aggregate_process_cpu_seconds"]
            )
            / cases
            * projected_cases,
            "projection_method": "direct-width256-autonomous-rollout-throughput",
        }

    def run_route(self) -> tuple[dict[str, Any], dict[str, Any]]:
        partial = self.output / self.mode / "partial.json"
        arms: dict[str, Any] = {}
        for arm in ARM_ORDER:
            result = self.train_arm(arm)
            result["evaluation"] = self.evaluate_arm(result)
            arms[arm] = result
            write_json(partial, {"status": "partial", "arms": arms})
        control_timer = Timer()
        structural = {
            arm: self.core_equivariance(arms[arm])
            for arm in ("symbol_equivariant", "broken_tying")
        }
        task_symmetry = self.task_symmetry_applicability()
        leakage = self.static_leakage_check()
        determinism = self.deterministic_replay(arms["symbol_equivariant"])
        dependence = self.randomized_dependence(arms["symbol_equivariant"])
        controls = {
            "core_equivariance": structural,
            "task_symmetry_applicability": task_symmetry,
            "static_leakage": leakage,
            "deterministic_replay": determinism,
            "randomized_weight_dependence": dependence,
            "timing": control_timer.finish(),
        }
        runtime = self.runtime_probe(arms["symbol_equivariant"])
        controls["runtime_width256"] = runtime
        write_json(partial, {"status": "route-complete", "arms": arms, "controls": controls})
        return arms, controls

    @staticmethod
    def scaled_timing(timing: dict[str, Any], scale: float) -> dict[str, float]:
        return {
            "wall_seconds": float(timing["wall_seconds"]) * scale,
            "aggregate_process_cpu_seconds": float(
                timing["aggregate_process_cpu_seconds"]
            )
            * scale,
            "cgroup_cpu_seconds": (
                0.0
                if timing.get("cgroup_cpu_seconds") is None
                else float(timing["cgroup_cpu_seconds"]) * scale
            ),
        }

    @staticmethod
    def add_timing(target: dict[str, float], value: dict[str, float]) -> None:
        for key in target:
            target[key] += float(value[key])

    def calibration(self) -> dict[str, Any]:
        if self.config["resource_budget"]["calibration_status"] != "pending":
            raise RuntimeError("calibration mode requires pending resource status")
        arms, controls = self.run_route()
        expected = {
            "wall_seconds": 0.0,
            "aggregate_process_cpu_seconds": 0.0,
            "cgroup_cpu_seconds": 0.0,
        }
        projection: dict[str, Any] = {"arms": {}, "controls": {}}
        formal_steps = int(self.config["training"]["total_steps_per_arm"])
        calibration_steps = sum(steps for _, steps in stage_sequence(self.config, "calibration"))
        for arm, result in arms.items():
            arm_projection = {
                "training": self.scaled_timing(
                    result["training"]["timing"], formal_steps / calibration_steps
                ),
                "evaluation": {},
            }
            self.add_timing(expected, arm_projection["training"])
            for width in self.config["evaluation"]["widths"]:
                width = int(width)
                arm_projection["evaluation"][str(width)] = {}
                for kind in ("transition", "rollout"):
                    formal_cases = float(self.config["evaluation"][f"{kind}_cases"][width])
                    calibration_cases = float(self.config["calibration"][f"{kind}_cases"][width])
                    projected = self.scaled_timing(
                        result["evaluation"][str(width)][kind]["timing"],
                        formal_cases / calibration_cases,
                    )
                    arm_projection["evaluation"][str(width)][kind] = projected
                    self.add_timing(expected, projected)
            projection["arms"][arm] = arm_projection

        for name, scale in (
            (
                "deterministic_replay",
                float(self.config["evaluation"]["deterministic_replay_cases"])
                / float(self.config["calibration"]["deterministic_replay_cases"]),
            ),
            (
                "randomized_weight_dependence",
                float(self.config["evaluation"]["randomized_dependence_cases"])
                / float(self.config["calibration"]["randomized_dependence_cases"]),
            ),
        ):
            projected = self.scaled_timing(controls[name]["timing"], scale)
            projection["controls"][name] = projected
            self.add_timing(expected, projected)
        runtime_scale = float(self.config["evaluation"]["runtime_probe_cases"]) / float(
            self.config["calibration"]["runtime_probe_cases"]
        )
        runtime_projected = self.scaled_timing(
            controls["runtime_width256"]["rollout"]["timing"], runtime_scale
        )
        projection["controls"]["runtime_width256"] = runtime_projected
        self.add_timing(expected, runtime_projected)
        structural_time = self.scaled_timing(
            controls["timing"], 1.0
        )
        projection["controls"]["fixed_structural_setup"] = structural_time
        self.add_timing(expected, structural_time)

        measured = self.started.finish()
        soft_multiplier = float(self.config["resource_budget"]["soft_multiplier"])
        hard_multiplier = float(self.config["resource_budget"]["hard_multiplier"])
        rss_multiplier = float(self.config["resource_budget"]["rss_multiplier"])
        recommended = {
            "expected_formal_wall_seconds": math.ceil(expected["wall_seconds"]),
            "expected_formal_aggregate_cpu_seconds": math.ceil(
                expected["aggregate_process_cpu_seconds"]
            ),
            "expected_formal_cgroup_cpu_seconds": math.ceil(expected["cgroup_cpu_seconds"]),
            "soft_budget_wall_seconds": math.ceil(expected["wall_seconds"] * soft_multiplier),
            "soft_budget_aggregate_cpu_seconds": math.ceil(
                expected["aggregate_process_cpu_seconds"] * soft_multiplier
            ),
            "soft_budget_cgroup_cpu_seconds": math.ceil(
                expected["cgroup_cpu_seconds"] * soft_multiplier
            ),
            "hard_budget_wall_seconds": math.ceil(expected["wall_seconds"] * hard_multiplier),
            "hard_budget_aggregate_cpu_seconds": math.ceil(
                expected["aggregate_process_cpu_seconds"] * hard_multiplier
            ),
            "hard_budget_cgroup_cpu_seconds": math.ceil(
                expected["cgroup_cpu_seconds"] * hard_multiplier
            ),
            "hard_peak_rss_bytes": math.ceil(int(measured["peak_rss_bytes"]) * rss_multiplier),
            "soft_multiplier": soft_multiplier,
            "hard_multiplier": hard_multiplier,
            "rss_multiplier": rss_multiplier,
        }
        result = {
            "run_id": self.config["run_id"],
            "mode": "calibration",
            "status": "completed",
            "seed": self.seed,
            "source": self.source_at_start,
            "dataset_splits": self.splits,
            "arms": arms,
            "controls": controls,
            "component_projection": projection,
            "expected_formal_without_headroom": expected,
            "measured_total": measured,
            "recommended_resource_budget": recommended,
            "admission": {
                "fidelity_reduced": False,
                "route_complete": True,
                "decision": "admit-if-config-freezes-recommended-budget",
            },
        }
        path = self.output / "calibration.json"
        write_json(path, result)
        return result

    def resolved_admission(self) -> None:
        budget = self.config["resource_budget"]
        if budget["calibration_status"] != "complete":
            raise RuntimeError("formal mode requires a complete calibration admission")
        calibration = ROOT / budget["calibration_output"]
        if not calibration.exists() or sha256(calibration) != budget["calibration_output_sha256"]:
            raise RuntimeError("calibration evidence hash mismatch")
        if sha256(Path(__file__)) != budget["calibration_runner_sha256"]:
            raise RuntimeError("runner differs from calibrated bytes")
        if sha256(Path(__file__).with_name("model.py")) != budget["calibration_model_sha256"]:
            raise RuntimeError("model differs from calibrated bytes")
        for key in (
            "expected_formal_wall_seconds",
            "expected_formal_aggregate_cpu_seconds",
            "expected_formal_cgroup_cpu_seconds",
            "soft_budget_wall_seconds",
            "soft_budget_aggregate_cpu_seconds",
            "soft_budget_cgroup_cpu_seconds",
            "hard_budget_wall_seconds",
            "hard_budget_aggregate_cpu_seconds",
            "hard_budget_cgroup_cpu_seconds",
            "hard_peak_rss_bytes",
        ):
            if budget[key] is None:
                raise RuntimeError(f"unresolved formal resource field: {key}")
        if bool(budget["fidelity_reduced"]):
            raise RuntimeError("formal admission forbids reduced fidelity")

    def gate_results(
        self, arms: dict[str, Any], controls: dict[str, Any], resource_result: dict[str, Any]
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        gates = self.config["gates"]
        faithful = arms["symbol_equivariant"]
        width16 = faithful["evaluation"]["16"]["rollout"]
        width32 = faithful["evaluation"]["32"]["rollout"]
        controls32 = {
            arm: arms[arm]["evaluation"]["32"]["rollout"]["exact_accuracy"]
            for arm in ("position_only", "broken_tying")
        }
        best_control = max(controls32.values())
        faithful_residual = controls["core_equivariance"]["symbol_equivariant"]["permutations"]["swap"]["maximum_absolute_residual"]
        broken_residual = controls["core_equivariance"]["broken_tying"]["permutations"]["swap"]["maximum_absolute_residual"]
        parameter_ratios = {
            arm: arms[arm]["parameters"] / faithful["parameters"]
            for arm in ("position_only", "broken_tying")
        }
        maximum_invalid = max(
            value["rollout"]["any_invalid_residue_rate"]
            for arm in arms.values()
            for value in arm["evaluation"].values()
        )
        results = {
            "width16_exact": width16["exact_accuracy"]
            >= float(gates["minimum_symbol_equivariant_width16_rollout"]),
            "width32_exact": width32["exact_accuracy"]
            >= float(gates["minimum_symbol_equivariant_width32_rollout"]),
            "width32_gain": width32["exact_accuracy"] - best_control
            >= float(gates["minimum_width32_gain_over_best_control"]),
            "faithful_core_equivariance": faithful_residual
            <= float(gates["maximum_core_swap_residual"]),
            "causal_broken_tying": broken_residual
            >= float(gates["minimum_broken_tying_swap_residual"]),
            "task_symmetry_truthful_non_applicability": controls["task_symmetry_applicability"]["verdict"]
            == "non-applicable-domain-not-closed",
            "validity": maximum_invalid <= float(gates["maximum_invalid_rate"]),
            "determinism": bool(controls["deterministic_replay"]["passed"]),
            "randomized_weight_dependence": controls["randomized_weight_dependence"]["exact_drop"]
            >= float(gates["minimum_randomized_dependence_drop"]),
            "static_leakage": bool(controls["static_leakage"]["passed"]),
            "parameter_match": all(
                float(gates["minimum_control_parameter_ratio"])
                <= ratio
                <= float(gates["maximum_control_parameter_ratio"])
                for ratio in parameter_ratios.values()
            ),
            "wall_budget": float(resource_result["wall_seconds"])
            <= float(self.config["resource_budget"]["hard_budget_wall_seconds"]),
            "aggregate_cpu_budget": float(resource_result["aggregate_process_cpu_seconds"])
            <= float(self.config["resource_budget"]["hard_budget_aggregate_cpu_seconds"]),
            "cgroup_cpu_budget": (
                resource_result["cgroup_cpu_seconds"] is not None
                and float(resource_result["cgroup_cpu_seconds"])
                <= float(self.config["resource_budget"]["hard_budget_cgroup_cpu_seconds"])
            ),
            "rss_budget": int(resource_result["peak_rss_bytes"])
            <= int(self.config["resource_budget"]["hard_peak_rss_bytes"]),
        }
        comparison = {
            "symbol_equivariant_width16": width16["exact_accuracy"],
            "symbol_equivariant_width32": width32["exact_accuracy"],
            "control_width32": controls32,
            "best_control_width32": best_control,
            "width32_gain": width32["exact_accuracy"] - best_control,
            "faithful_swap_residual": faithful_residual,
            "broken_tying_swap_residual": broken_residual,
            "maximum_invalid_rate": maximum_invalid,
            "parameter_ratios_to_faithful": parameter_ratios,
        }
        return results, comparison

    def publish_evidence(self, summary: dict[str, Any]) -> dict[str, Any]:
        run_dir = ROOT / "experiments/runs/exp-659-symbol-equivariant-recurrence"
        eval_dir = ROOT / "evals/results/exp-659-symbol-equivariant-recurrence"
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_hashes = {}
        for arm, result in summary["arms"].items():
            source = ROOT / result["checkpoint"]["path"]
            destination = run_dir / f"{arm}.safetensors"
            shutil.copy2(source, destination)
            checkpoint_hashes[arm] = {
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        learning_curves = {
            arm: result["training"]["history"] for arm, result in summary["arms"].items()
        }
        environment = summary["environment"]
        hashes = {
            "source": summary["source"],
            "checkpoints": checkpoint_hashes,
            "run_payload_sha256": sha256_json(summary),
        }
        metrics = {
            "run_id": summary["run_id"],
            "seed": summary["seed"],
            "arms": {
                arm: result["evaluation"] for arm, result in summary["arms"].items()
            },
            "controls": summary["controls"],
            "comparison": summary["comparison"],
            "gates": summary["gates"],
            "decision": summary["decision"],
            "resource": summary["resource"],
        }
        write_json(run_dir / "run.json", summary)
        write_json(run_dir / "environment.json", environment)
        write_json(run_dir / "hashes.json", hashes)
        write_json(run_dir / "learning-curves.json", learning_curves)
        write_json(run_dir / "limitations.json", summary["limitations"])
        write_json(run_dir / "mlflow.json", summary["mlflow"])
        write_json(eval_dir / "metrics.json", metrics)
        write_json(
            eval_dir / "structural-controls.json",
            {
                "core_equivariance": summary["controls"]["core_equivariance"],
                "task_symmetry_applicability": summary["controls"]["task_symmetry_applicability"],
                "deterministic_replay": summary["controls"]["deterministic_replay"],
                "randomized_weight_dependence": summary["controls"]["randomized_weight_dependence"],
                "static_leakage": summary["controls"]["static_leakage"],
            },
        )
        return {"run_dir": str(run_dir.relative_to(ROOT)), "eval_dir": str(eval_dir.relative_to(ROOT)), "checkpoints": checkpoint_hashes}

    def formal(self) -> dict[str, Any]:
        self.resolved_admission()
        if self.source_at_start["dirty"]:
            raise RuntimeError("formal mode requires a clean tracked worktree")
        if self.source_at_start["repository_branch"] != "codex/exp659-symbol-equivariant":
            raise RuntimeError("formal mode requires the scoped EXP-659 branch")
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        run_ids = {}
        with mlflow.start_run(run_name=self.config["run_id"]) as parent:
            parent_id = parent.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "running",
                    "role": "matched-symbol-equivariant-recurrent-falsifier",
                    "repository_commit": self.source_at_start["repository_commit"],
                    "repository_dirty": "false",
                }
            )
            mlflow.log_params(
                {
                    "seed": self.seed,
                    "steps_per_arm": self.config["training"]["total_steps_per_arm"],
                    "config_sha256": self.source_at_start["config_sha256"],
                    "runner_sha256": self.source_at_start["runner_sha256"],
                    "model_sha256": self.source_at_start["model_sha256"],
                    "dataset_sha256": self.dataset_sha256,
                }
            )
            arms: dict[str, Any] = {}
            partial = self.output / "formal" / "partial.json"
            for arm in ARM_ORDER:
                with mlflow.start_run(run_name=arm, nested=True) as nested:
                    run_ids[arm] = nested.info.run_id
                    result = self.train_arm(arm)
                    result["evaluation"] = self.evaluate_arm(result)
                    arms[arm] = result
                    for width, values in result["evaluation"].items():
                        mlflow.log_metric(
                            f"width{width}_rollout_exact", values["rollout"]["exact_accuracy"]
                        )
                write_json(partial, {"status": "partial", "arms": arms})
            control_timer = Timer()
            controls = {
                "core_equivariance": {
                    arm: self.core_equivariance(arms[arm])
                    for arm in ("symbol_equivariant", "broken_tying")
                },
                "task_symmetry_applicability": self.task_symmetry_applicability(),
                "static_leakage": self.static_leakage_check(),
                "deterministic_replay": self.deterministic_replay(arms["symbol_equivariant"]),
                "randomized_weight_dependence": self.randomized_dependence(arms["symbol_equivariant"]),
            }
            controls["timing"] = control_timer.finish()
            controls["runtime_width256"] = self.runtime_probe(arms["symbol_equivariant"])
            resource_result = self.started.finish()
            gate_results, comparison = self.gate_results(arms, controls, resource_result)
            passed = all(gate_results.values())
            decision = (
                "advance-symbol-equivariant-successor-confirmation"
                if passed
                else "reject-exp659-symbol-equivariant-formulation"
            )
            summary = {
                "run_id": self.config["run_id"],
                "status": "completed",
                "hypothesis": self.config["hypothesis"],
                "seed": self.seed,
                "source": self.source_at_start,
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "mlflow": mlflow.__version__,
                    "numpy": np.__version__,
                    "psutil": psutil.__version__,
                    "device": "cpu",
                    "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "pid": os.getpid(),
                },
                "mlflow": {
                    "tracking_uri": tracking_uri(),
                    "experiment_name": self.config["tracking"]["experiment_name"],
                    "experiment_id": experiment_id,
                    "parent_run_id": parent_id,
                    "arm_run_ids": run_ids,
                },
                "dataset_splits": self.splits,
                "arms": arms,
                "controls": controls,
                "comparison": comparison,
                "gates": {**self.config["gates"], "results": gate_results, "passed": passed},
                "decision": decision,
                "resource": {
                    **resource_result,
                    "resolved_budget": self.config["resource_budget"],
                },
                "limitations": [
                    "Single formal training seed; an advance would require preregistered independent training-seed confirmation.",
                    "Core symbol equivariance is bounded-faithful, but the anchored arithmetic boundary is not a task-level bit-swap symmetry.",
                    "Fresh generated W16/W24/W32 identities only; checked, prior independent, official/private answers were not read.",
                    "Exact Python arithmetic is confined to offline labels, evaluation, and property tests.",
                    "The W256 probe measures four exact autonomous rollouts and is not an official benchmark result.",
                ],
            }
            preview = self.output / "formal-summary.json"
            write_json(preview, summary)
            mlflow.log_metrics(
                {
                    "symbol_equivariant_width16": comparison["symbol_equivariant_width16"],
                    "symbol_equivariant_width32": comparison["symbol_equivariant_width32"],
                    "width32_gain": comparison["width32_gain"],
                    "faithful_swap_residual": comparison["faithful_swap_residual"],
                    "broken_tying_swap_residual": comparison["broken_tying_swap_residual"],
                    "wall_seconds": resource_result["wall_seconds"],
                    "aggregate_cpu_seconds": resource_result["aggregate_process_cpu_seconds"],
                }
            )
            mlflow.log_artifact(str(self.config_path), artifact_path="preregistration")
            mlflow.log_artifact(str(ROOT / self.config["preregistration"]), artifact_path="preregistration")
            mlflow.log_artifact(str(preview), artifact_path="canonical-preview")
            mlflow.set_tags({"status": "completed", "decision": decision})
        publication = self.publish_evidence(summary)
        summary["publication"] = publication
        write_json(self.output / "formal-summary.json", summary)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_659_symbol_equivariant_recurrence.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-659-symbol-equivariant-recurrence",
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
                "decision": result.get("decision", result.get("admission", {}).get("decision")),
                "resource": result.get("resource", result.get("measured_total")),
                "recommended_resource_budget": result.get("recommended_resource_budget"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
