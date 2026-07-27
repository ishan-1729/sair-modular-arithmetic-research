"""Calibrate and execute the frozen EXP-650 library-backed Mamba-2 comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import resource
import statistics
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
from sympy import primepi
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_650_library_backed_mamba.model import (  # noqa: E402
    LibraryMamba2Hybrid,
    build_model,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

ARM_ORDER = ("mamba2_hybrid", "convgru_control", "attention_control")
SOURCE_PATHS = (
    "configs/exp_650_library_backed_mamba.yaml",
    "experiments/exp_650_library_backed_mamba/model.py",
    "experiments/exp_650_library_backed_mamba/run.py",
    "tests/test_exp_650_library_backed_mamba.py",
)


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
            "exp300_for_exp650",
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


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def process_cpu_seconds() -> float:
    times = psutil.Process().cpu_times()
    return float(times.user + times.system)


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.exists():
        return None
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        values[key] = int(value)
    usage = values.get("usage_usec")
    return None if usage is None else usage / 1_000_000.0


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def decode_bits(rows: Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in rows.tolist()
    ]


def transition_features(exp300, states, other, modulus, bits, width: int) -> Tensor:
    """Reuse EXP-300 planes while exposing the token-major library contract."""
    return exp300.features(states, other, modulus, bits, width).transpose(1, 2)


def make_split(config: dict[str, Any], exp300, width: int, seed: int):
    data = config["data"]
    minimum_bits = max(3, width - int(data["prime_bit_span"]) + 1)
    low = 1 << (minimum_bits - 1)
    high = 1 << width
    requested = int(data["train_prime_identities"]) + int(
        data["test_prime_identities"]
    )
    if width <= 32:
        available = int(primepi(high - 1) - primepi(low - 1))
        if requested > available:
            raise ValueError(
                f"width {width} requests {requested} identities from {available} primes"
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


def make_splits(config: dict[str, Any], exp300, seed: int):
    widths = sorted(
        set(config["data"]["train_widths"] + config["data"]["evaluation_widths"])
    )
    return {width: make_split(config, exp300, width, seed) for width in widths}


def trajectory_batch(sampler, width: int, batch_size: int, exp300):
    left, right, modulus = sampler.pairs(batch_size)
    states: list[int] = []
    repeated_right: list[int] = []
    repeated_modulus: list[int] = []
    input_bits: list[int] = []
    target_rows = []
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
            rows.append([(target >> index) & 1 for index in range(width)])
            state = target
        target_rows.extend(rows)
    return {
        "features": transition_features(
            exp300,
            states, repeated_right, repeated_modulus, input_bits, width
        ),
        "targets": torch.tensor(target_rows, dtype=torch.float32),
        "trajectories": batch_size,
        "transitions": batch_size * width,
    }


def make_cases(exp300, primes: list[int], seed: int, count: int):
    left, right, modulus = exp300.BigIntSampler(primes, seed).pairs(count)
    return {"left": left, "right": right, "modulus": modulus}


def prefix_loss(logits: Tensor, targets: Tensor, training: dict[str, Any]):
    per_bit = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mean = per_bit.mean()
    worst = per_bit.mean(0).max()
    loss = mean + float(training["worst_bit_weight"]) * worst
    exact = (logits.ge(0) == targets.bool()).all(-1)
    return loss, {
        "loss": float(loss.detach()),
        "mean_bce": float(mean.detach()),
        "worst_bit_bce": float(worst.detach()),
        "transition_exact": float(exact.float().mean()),
    }


def call_model(model: nn.Module, features: Tensor, direction: str = "forward"):
    if isinstance(model, LibraryMamba2Hybrid):
        return model(features, direction=direction)
    if direction != "forward":
        raise ValueError("direction interventions apply only to the Mamba-2 arm")
    return model(features)


@torch.inference_mode()
def evaluate_cases(
    model: nn.Module,
    exp300,
    cases: dict[str, list[int]],
    width: int,
    batch_size: int,
    direction: str = "forward",
):
    wall_started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    cgroup_started = cgroup_cpu_seconds()
    transition_correct = 0
    transition_total = 0
    final_predictions: list[int] = []
    all_prefix_correct = 0
    prefix_correct = 0
    invalid_cases = 0
    first_errors: list[int] = []
    for offset in range(0, len(cases["left"]), batch_size):
        left = cases["left"][offset : offset + batch_size]
        right = cases["right"][offset : offset + batch_size]
        modulus = cases["modulus"][offset : offset + batch_size]
        exact_states = [0 for _ in left]
        predicted_states = [0 for _ in left]
        exact_rows = [[] for _ in left]
        predicted_rows = [[] for _ in left]
        case_first = [None for _ in left]
        for position in range(width):
            bits = [(value >> (width - 1 - position)) & 1 for value in left]
            exact_targets = [
                (2 * state + bit * other) % prime
                for state, bit, other, prime in zip(
                    exact_states, bits, right, modulus, strict=True
                )
            ]
            teacher_features = transition_features(
                exp300, exact_states, right, modulus, bits, width
            )
            teacher_values = decode_bits(
                call_model(model, teacher_features, direction).ge(0)
            )
            transition_correct += sum(
                value == target
                for value, target in zip(teacher_values, exact_targets, strict=True)
            )
            transition_total += len(left)
            autonomous_features = transition_features(
                exp300, predicted_states, right, modulus, bits, width
            )
            predicted_states = decode_bits(
                call_model(model, autonomous_features, direction).ge(0)
            )
            for index, (predicted, target) in enumerate(
                zip(predicted_states, exact_targets, strict=True)
            ):
                exact_rows[index].append(target)
                predicted_rows[index].append(predicted)
                exact = predicted == target
                prefix_correct += int(exact)
                if not exact and case_first[index] is None:
                    case_first[index] = position
            exact_states = exact_targets
        final_predictions.extend(predicted_states)
        for predicted, expected, prime, row, first in zip(
            predicted_states,
            exact_states,
            modulus,
            predicted_rows,
            case_first,
            strict=True,
        ):
            all_prefix_correct += int(first is None)
            invalid_cases += int(any(value >= prime for value in row))
            if first is not None:
                first_errors.append(first)
    cases_count = len(cases["left"])
    final_expected = [
        (left * right) % prime
        for left, right, prime in zip(
            cases["left"], cases["right"], cases["modulus"], strict=True
        )
    ]
    final_correct = sum(
        predicted == expected
        for predicted, expected in zip(final_predictions, final_expected, strict=True)
    )
    cgroup_finished = cgroup_cpu_seconds()
    return {
        "cases": cases_count,
        "correct": final_correct,
        "exact_accuracy": final_correct / cases_count,
        "transition_cases": transition_total,
        "transition_correct": transition_correct,
        "transition_exact_accuracy": transition_correct / transition_total,
        "prefix_transition_exact_accuracy": prefix_correct / transition_total,
        "all_prefix_exact_accuracy": all_prefix_correct / cases_count,
        "any_invalid_residue_rate": invalid_cases / cases_count,
        "first_error": {
            "error_cases": len(first_errors),
            "mean_position": statistics.fmean(first_errors) if first_errors else None,
            "minimum_position": min(first_errors) if first_errors else None,
            "maximum_position": max(first_errors) if first_errors else None,
            "histogram": {
                str(position): first_errors.count(position)
                for position in sorted(set(first_errors))
            },
        },
        "wall_seconds": time.perf_counter() - wall_started,
        "process_cpu_seconds": process_cpu_seconds() - cpu_started,
        "cgroup_cpu_seconds": (
            None
            if cgroup_started is None or cgroup_finished is None
            else cgroup_finished - cgroup_started
        ),
        "predictions_sha256": sha256_json(final_predictions),
    }


def model_flops(model: nn.Module, width: int) -> int:
    sample = torch.zeros(1, width, 6)
    model.eval()
    with torch.profiler.profile(with_flops=True) as profile:
        with torch.no_grad():
            call_model(model, sample)
    return int(sum(event.flops or 0 for event in profile.key_averages()))


def save_checkpoint(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()},
        path,
    )
    try:
        recorded_path = path.relative_to(ROOT)
    except ValueError:
        recorded_path = path
    return {"path": str(recorded_path), "sha256": sha256(path)}


def run_arm(
    arm: str,
    config: dict[str, Any],
    exp300,
    splits,
    seed: int,
    steps: int,
    root_dir: Path,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(arm, max(config["data"]["evaluation_widths"]), config["models"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial = save_checkpoint(model, root_dir / arm / "initial.safetensors")
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(training["minimum_learning_rate"]),
    )
    width = int(config["data"]["train_widths"][0])
    sampler = exp300.BigIntSampler(
        splits[width]["train"],
        seed + int(config["data"]["training_pair_seed_offset"]) + width,
    )
    history = []
    update_wall_seconds = []
    update_process_cpu_seconds = []
    update_cgroup_cpu_seconds = []
    wall_started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    cgroup_started = cgroup_cpu_seconds()
    model.train()
    for step in range(1, steps + 1):
        step_wall = time.perf_counter()
        step_cpu = process_cpu_seconds()
        step_cgroup = cgroup_cpu_seconds()
        batch = trajectory_batch(
            sampler,
            width,
            int(training["batch_trajectories"]),
            exp300,
        )
        logits = call_model(model, batch["features"])
        loss, stats = prefix_loss(logits, batch["targets"], training)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
        )
        optimizer.step()
        scheduler.step()
        update_wall_seconds.append(time.perf_counter() - step_wall)
        update_process_cpu_seconds.append(process_cpu_seconds() - step_cpu)
        step_cgroup_finished = cgroup_cpu_seconds()
        if step_cgroup is not None and step_cgroup_finished is not None:
            update_cgroup_cpu_seconds.append(step_cgroup_finished - step_cgroup)
        if (
            step == 1
            or step == steps
            or step % int(training["log_every_steps"]) == 0
        ):
            history.append(
                {"step": step, "gradient_norm": gradient_norm, **stats}
            )
    training_wall = time.perf_counter() - wall_started
    training_cpu = process_cpu_seconds() - cpu_started
    cgroup_finished = cgroup_cpu_seconds()
    training_cgroup = (
        None
        if cgroup_started is None or cgroup_finished is None
        else cgroup_finished - cgroup_started
    )
    checkpoint = save_checkpoint(model, root_dir / arm / "trained.safetensors")
    reload_model = build_model(
        arm, max(config["data"]["evaluation_widths"]), config["models"]
    ).eval()
    reload_model.load_state_dict(load_file(ROOT / checkpoint["path"]))
    eval_cases = {}
    for eval_width in config["data"]["evaluation_widths"]:
        eval_cases[eval_width] = make_cases(
            exp300,
            splits[eval_width]["test"],
            seed + int(config["data"]["evaluation_pair_seed_offset"]) + eval_width,
            int(config["evaluation"]["cases_per_width"]),
        )
    determinism_cases = eval_cases[width]
    first_features = transition_features(
        exp300,
        [0] * int(config["evaluation"]["determinism_cases"]),
        determinism_cases["right"][: int(config["evaluation"]["determinism_cases"])],
        determinism_cases["modulus"][: int(config["evaluation"]["determinism_cases"])],
        [
            value >> (width - 1) & 1
            for value in determinism_cases["left"][: int(config["evaluation"]["determinism_cases"])]
        ],
        width,
    )
    model.eval()
    with torch.no_grad():
        original_logits = call_model(model, first_features)
        reload_logits = call_model(reload_model, first_features)
    reload_exact = bool(torch.equal(original_logits, reload_logits))
    evaluations = {
        str(eval_width): evaluate_cases(
            reload_model,
            exp300,
            eval_cases[eval_width],
            eval_width,
            int(config["evaluation"]["batch_size"]),
        )
        for eval_width in config["data"]["evaluation_widths"]
    }
    interventions = {}
    randomized = None
    if arm == "mamba2_hybrid":
        interventions = {
            direction: evaluate_cases(
                reload_model,
                exp300,
                eval_cases[width],
                width,
                int(config["evaluation"]["batch_size"]),
                direction=direction,
            )
            for direction in ("reverse", "bidirectional")
        }
        randomized_model = build_model(
            arm, max(config["data"]["evaluation_widths"]), config["models"]
        ).eval()
        randomized_model.load_state_dict(load_file(ROOT / initial["path"]))
        randomized_cases = {
            key: value[: int(config["evaluation"]["randomized_weight_cases"])]
            for key, value in eval_cases[width].items()
        }
        randomized = evaluate_cases(
            randomized_model,
            exp300,
            randomized_cases,
            width,
            int(config["evaluation"]["batch_size"]),
        )
    return {
        "arm": arm,
        "parameters": parameter_count,
        "optimizer": "AdamW",
        "training_updates": steps,
        "training_trajectories": steps * int(training["batch_trajectories"]),
        "training_transition_examples": steps
        * int(training["batch_trajectories"])
        * width,
        "training_wall_seconds": training_wall,
        "training_process_cpu_seconds": training_cpu,
        "training_cgroup_cpu_seconds": training_cgroup,
        "update_wall_seconds": update_wall_seconds,
        "update_process_cpu_seconds": update_process_cpu_seconds,
        "update_cgroup_cpu_seconds": update_cgroup_cpu_seconds,
        "history": history,
        "initial_checkpoint": initial,
        "checkpoint": checkpoint,
        "reload_exact": reload_exact,
        "reload_max_abs_difference": float(
            (original_logits - reload_logits).abs().max()
        ),
        "profiler_forward_flops_width16_batch1": model_flops(reload_model, width),
        "evaluations": evaluations,
        "interventions": interventions,
        "randomized_weight": randomized,
        "peak_rss_bytes": peak_rss_bytes(),
    }


def source_identity(config_path: Path):
    return {
        "repository_base_commit": git_output("rev-parse", "HEAD"),
        "source_hashes": {path: sha256(ROOT / path) for path in SOURCE_PATHS},
        "config_sha256": sha256(config_path),
        "generator_sha256": sha256(
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"
        ),
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def seed_for_root(config: dict[str, Any], root: int):
    if root == int(config["roots"]["calibration"]):
        return int(config["data"]["calibration_seed"])
    return int(config["data"]["formal_seeds"][root])


def preregister(config_path: Path, output_base: Path):
    config = yaml.safe_load(config_path.read_text())
    exp300 = load_exp300()
    identities = {}
    roots = [config["roots"]["calibration"], config["roots"]["viability"]]
    roots.extend(config["roots"]["confirmations"])
    for root in roots:
        seed = seed_for_root(config, int(root))
        splits = make_splits(config, exp300, seed)
        cases = {
            str(width): make_cases(
                exp300,
                splits[width]["test"],
                seed + int(config["data"]["evaluation_pair_seed_offset"]) + width,
                int(config["evaluation"]["cases_per_width"]),
            )
            for width in config["data"]["evaluation_widths"]
        }
        identities[str(root)] = {
            "seed": seed,
            "splits_sha256": sha256_json(splits),
            "evaluation_cases_sha256": sha256_json(cases),
        }
    payload = {
        "status": "frozen-before-scientific-training",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roots": config["roots"],
        "identities": identities,
        "source": source_identity(config_path),
        "gates": config["gates"],
        "resource_budget": config["resource_budget"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "models": config["models"],
        "limitations": [
            "Roots use fresh generated train/test prime identities and no checked, prior-independent, official, or private answers.",
            "Exact arithmetic appears only in offline labels and evaluation truth.",
            "Transformers Mamba2 uses its deterministic naive torch_forward CPU path, not CUDA/Triton fast kernels.",
        ],
    }
    output_base.mkdir(parents=True, exist_ok=True)
    path = output_base / "preregistration.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError("existing preregistration does not match frozen bytes")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preregistration": str(path), "sha256": sha256(path)}))


def validate_preregistration(config_path: Path, output_base: Path):
    path = output_base / "preregistration.json"
    frozen = json.loads(path.read_text())
    current = source_identity(config_path)
    for key in ("source_hashes", "config_sha256", "generator_sha256", "transformers_version"):
        if frozen["source"][key] != current[key]:
            raise RuntimeError(f"preregistered {key} changed")
    return frozen, sha256(path)


def variability_headroom(samples: list[float], config: dict[str, Any]):
    median = statistics.median(samples)
    p90 = sorted(samples)[max(0, math.ceil(0.9 * len(samples)) - 1)]
    observed = p90 / median if median else 1.0
    budget = config["resource_budget"]
    return min(
        float(budget["eta_variability_headroom_ceiling"]),
        max(float(budget["eta_variability_headroom_floor"]), observed),
    )


def calibration_projection(payload: dict[str, Any], config: dict[str, Any]):
    arms = payload["arms"]
    formal_steps = int(config["training"]["steps_per_arm"])
    calibration_steps = int(config["calibration"]["updates_per_arm"])
    all_wall = [value for arm in arms.values() for value in arm["update_wall_seconds"]]
    headroom = variability_headroom(all_wall, config)
    projected_training_wall = sum(
        arm["training_wall_seconds"] / calibration_steps * formal_steps
        for arm in arms.values()
    )
    projected_training_process_cpu = sum(
        arm["training_process_cpu_seconds"] / calibration_steps * formal_steps
        for arm in arms.values()
    )
    projected_training_cgroup_cpu = sum(
        (arm["training_cgroup_cpu_seconds"] or arm["training_process_cpu_seconds"])
        / calibration_steps
        * formal_steps
        for arm in arms.values()
    )
    measured_eval_wall = sum(
        metric["wall_seconds"]
        for arm in arms.values()
        for metric in arm["evaluations"].values()
    ) + sum(
        metric["wall_seconds"]
        for metric in arms["mamba2_hybrid"]["interventions"].values()
    ) + arms["mamba2_hybrid"]["randomized_weight"]["wall_seconds"]
    measured_eval_process_cpu = sum(
        metric["process_cpu_seconds"]
        for arm in arms.values()
        for metric in arm["evaluations"].values()
    ) + sum(
        metric["process_cpu_seconds"]
        for metric in arms["mamba2_hybrid"]["interventions"].values()
    ) + arms["mamba2_hybrid"]["randomized_weight"]["process_cpu_seconds"]
    measured_eval_cgroup_cpu = sum(
        metric["cgroup_cpu_seconds"] or metric["process_cpu_seconds"]
        for arm in arms.values()
        for metric in arm["evaluations"].values()
    ) + sum(
        metric["cgroup_cpu_seconds"] or metric["process_cpu_seconds"]
        for metric in arms["mamba2_hybrid"]["interventions"].values()
    ) + (
        arms["mamba2_hybrid"]["randomized_weight"]["cgroup_cpu_seconds"]
        or arms["mamba2_hybrid"]["randomized_weight"]["process_cpu_seconds"]
    )
    expected_wall = (projected_training_wall + measured_eval_wall) * headroom
    expected_process_cpu = (
        projected_training_process_cpu + measured_eval_process_cpu
    ) * headroom
    expected_cgroup_cpu = (
        projected_training_cgroup_cpu + measured_eval_cgroup_cpu
    ) * headroom
    budget = config["resource_budget"]
    peak = max(arm["peak_rss_bytes"] for arm in arms.values())
    checks = {
        "wall": expected_wall <= float(budget["admission_maximum_expected_wall_seconds"]),
        "cgroup_cpu": expected_cgroup_cpu
        <= float(budget["admission_maximum_expected_cgroup_cpu_seconds"]),
        "rss": peak <= int(budget["admission_maximum_peak_rss_bytes"]),
        "reload": all(arm["reload_exact"] for arm in arms.values()),
    }
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        "variability": {
            "update_wall_median_seconds": statistics.median(all_wall),
            "update_wall_min_seconds": min(all_wall),
            "update_wall_max_seconds": max(all_wall),
            "headroom_multiplier": headroom,
        },
        "projected_formal": {
            "training_wall_seconds": projected_training_wall,
            "evaluation_wall_seconds": measured_eval_wall,
            "expected_total_wall_seconds_with_headroom": expected_wall,
            "expected_process_cpu_seconds_with_headroom": expected_process_cpu,
            "expected_cgroup_cpu_seconds_with_headroom": expected_cgroup_cpu,
            "peak_rss_bytes": peak,
            "hard_runaway_ceiling_wall_seconds": expected_wall
            * float(budget["hard_runaway_multiplier"]),
        },
        "fidelity": {
            "formal_training_updates_per_arm": formal_steps,
            "evaluation_cases_per_width": int(config["evaluation"]["cases_per_width"]),
            "evaluation_widths": config["data"]["evaluation_widths"],
            "calibration_used_full_evaluation_contract": True,
        },
    }


def scientific_gate(payload: dict[str, Any], config: dict[str, Any]):
    arms = payload["arms"]
    hybrid = arms["mamba2_hybrid"]
    w16 = hybrid["evaluations"]["16"]
    w24 = hybrid["evaluations"]["24"]
    w32 = hybrid["evaluations"]["32"]
    controls = [arms[name]["evaluations"]["16"] for name in ARM_ORDER[1:]]
    gates = config["gates"]
    parameter_ratios = {
        name: arms[name]["parameters"] / hybrid["parameters"] for name in ARM_ORDER[1:]
    }
    checks = {
        "w16_viability": w16["exact_accuracy"]
        >= float(gates["minimum_w16_rollout_accuracy"]),
        "gain_over_convgru": w16["exact_accuracy"] - controls[0]["exact_accuracy"]
        >= float(gates["minimum_gain_over_each_control"]),
        "gain_over_attention": w16["exact_accuracy"] - controls[1]["exact_accuracy"]
        >= float(gates["minimum_gain_over_each_control"]),
        "w24_transfer": w24["exact_accuracy"]
        >= float(gates["minimum_w24_rollout_accuracy"]),
        "w32_transfer": w32["exact_accuracy"]
        >= float(gates["minimum_w32_rollout_accuracy"]),
        "trained_weight_dependence": w16["exact_accuracy"]
        - hybrid["randomized_weight"]["exact_accuracy"]
        >= float(gates["minimum_trained_gain_over_randomized_w16"]),
        "validity": max(
            arm["evaluations"][width]["any_invalid_residue_rate"]
            for arm in arms.values()
            for width in ("16", "24", "32")
        )
        <= float(gates["maximum_invalid_rate"]),
        "reload": all(arm["reload_exact"] for arm in arms.values()),
        "parameter_match": all(
            1 / float(gates["maximum_parameter_ratio"])
            <= ratio
            <= float(gates["maximum_parameter_ratio"])
            for ratio in parameter_ratios.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parameter_ratios_to_hybrid": parameter_ratios,
        "decision": "advance-to-confirmations" if all(checks.values()) else "reject",
    }


def execute(config_path: Path, output_base: Path, eval_base: Path, mode: str, root: int):
    config = yaml.safe_load(config_path.read_text())
    frozen, prereg_sha = validate_preregistration(config_path, output_base)
    expected_root = int(
        config["roots"]["calibration"] if mode == "calibration" else root
    )
    if root != expected_root:
        raise ValueError("root does not match requested mode")
    if mode == "formal" and root in config["roots"]["confirmations"]:
        viability = json.loads(
            (output_base / "roots" / str(config["roots"]["viability"]) / "result.json").read_text()
        )
        if not viability["gate"]["passed"]:
            raise RuntimeError("confirmation root is not authorized by the frozen viability gate")
    if mode == "formal":
        admission = json.loads((output_base / "admission.json").read_text())
        if not admission["admitted"]:
            raise RuntimeError("root 20261186 rejected scientific admission")
    root_dir = output_base / "roots" / str(root)
    result_path = root_dir / "result.json"
    if result_path.exists() and config["resource_budget"]["cache_completed_roots"]:
        cached = json.loads(result_path.read_text())
        if cached["preregistration_sha256"] != prereg_sha:
            raise RuntimeError("cached root has a different preregistration")
        print(json.dumps({"cached": True, "root": root, "result": str(result_path)}))
        return
    root_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_for_root(config, root)
    exp300 = load_exp300()
    splits = make_splits(config, exp300, seed)
    identity = frozen["identities"][str(root)]
    if sha256_json(splits) != identity["splits_sha256"]:
        raise RuntimeError("dataset identity differs from preregistration")
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    steps = int(
        config["calibration"]["updates_per_arm"]
        if mode == "calibration"
        else config["training"]["steps_per_arm"]
    )
    wall_started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    cgroup_started = cgroup_cpu_seconds()
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"exp650-{mode}-root-{root}") as active:
        mlflow.log_params(
            {
                "root": root,
                "mode": mode,
                "seed": seed,
                "steps_per_arm": steps,
                "transformers": importlib.metadata.version("transformers"),
            }
        )
        arms = {
            arm: run_arm(arm, config, exp300, splits, seed, steps, root_dir)
            for arm in ARM_ORDER
        }
        mlflow.log_metrics(
            {
                f"{arm}_w16_rollout": values["evaluations"]["16"]["exact_accuracy"]
                for arm, values in arms.items()
            }
        )
        mlflow_run_id = active.info.run_id
    cgroup_finished = cgroup_cpu_seconds()
    payload = {
        "run_id": config["run_id"],
        "root": root,
        "mode": mode,
        "status": "complete",
        "seed": seed,
        "preregistration_sha256": prereg_sha,
        "source": source_identity(config_path),
        "dataset": identity,
        "arms": arms,
        "resources": {
            "wall_seconds": time.perf_counter() - wall_started,
            "process_cpu_seconds": process_cpu_seconds() - cpu_started,
            "cgroup_cpu_seconds": (
                None
                if cgroup_started is None or cgroup_finished is None
                else cgroup_finished - cgroup_started
            ),
            "peak_rss_bytes": peak_rss_bytes(),
            "torch_threads": int(config["training"]["torch_threads"]),
        },
        "mlflow": {
            "experiment_id": experiment_id,
            "run_id": mlflow_run_id,
            "tracking_uri": tracking_uri(),
        },
        "limitations": [
            "This is an offline learned-transition experiment, not a submission candidate or private benchmark result.",
            "CPU execution uses the maintained Transformers naive torch_forward fallback; CUDA fast-path performance is not measured.",
            "The controls are parameter matched and use identical examples/updates/optimizer, but operator FLOPs and wall time differ and are disclosed.",
        ],
    }
    if mode == "calibration":
        payload["admission"] = calibration_projection(payload, config)
    else:
        payload["gate"] = scientific_gate(payload, config)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    eval_path = eval_base / "roots" / str(root) / "metrics.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_payload = {
        "run_id": config["run_id"],
        "root": root,
        "mode": mode,
        "arms": {
            arm: {
                "parameters": values["parameters"],
                "evaluations": values["evaluations"],
                "interventions": values["interventions"],
                "randomized_weight": values["randomized_weight"],
            }
            for arm, values in arms.items()
        },
        "decision": payload.get("gate", payload.get("admission")),
        "source_result_sha256": sha256(result_path),
    }
    eval_path.write_text(json.dumps(eval_payload, indent=2, sort_keys=True) + "\n")
    if mode == "calibration":
        admission_path = output_base / "admission.json"
        admission_path.write_text(
            json.dumps(payload["admission"], indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                "root": root,
                "mode": mode,
                "result": str(result_path),
                "result_sha256": sha256(result_path),
                "decision": payload.get("gate", payload.get("admission")),
            },
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_650_library_backed_mamba.yaml",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=ROOT / "experiments/runs/exp-650-library-backed-mamba",
    )
    parser.add_argument(
        "--eval-base",
        type=Path,
        default=ROOT / "evals/results/exp-650-library-backed-mamba",
    )
    parser.add_argument(
        "--mode", choices=("preregister", "calibration", "formal"), required=True
    )
    parser.add_argument("--root", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "preregister":
        if args.root is not None:
            raise ValueError("preregister freezes all roots; omit --root")
        preregister(args.config.resolve(), args.output_base.resolve())
        return
    if args.root is None:
        raise ValueError("--root is required for calibration/formal")
    execute(
        args.config.resolve(),
        args.output_base.resolve(),
        args.eval_base.resolve(),
        args.mode,
        args.root,
    )


if __name__ == "__main__":
    main()
