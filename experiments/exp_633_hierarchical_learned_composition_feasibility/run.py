"""Matched sequential-control versus hierarchical-composition feasibility rung."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
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
from safetensors.torch import save_file
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
        timeout=30,
    ).stdout.strip()


def make_splits(exp300, config: dict[str, Any], seed: int) -> dict[int, dict]:
    widths = sorted(set(config["training"]["widths"] + config["evaluation"]["widths"]))
    splits = {}
    for width in widths:
        train_count = config["data"].get("train_prime_identities_by_width", {}).get(
            width, config["data"]["train_prime_identities"]
        )
        test_count = config["data"].get("test_prime_identities_by_width", {}).get(
            width, config["data"]["test_prime_identities"]
        )
        minimum_bits = max(3, width - config["data"]["prime_bit_span"] + 1)
        low = 1 << (minimum_bits - 1)
        high = 1 << width
        available = int(primepi(high - 1) - primepi(low - 1))
        if train_count + test_count > available:
            raise ValueError(
                f"width {width} requests {train_count + test_count} unique primes "
                f"but the configured band contains only {available}"
            )
        local = {
            "data": {
                "prime_bits_min": minimum_bits,
                "prime_bits_max": width,
                "train_prime_identities": train_count,
                "test_prime_identities": test_count,
            }
        }
        splits[width] = exp300.bigint_prime_split(local, seed + 63300 + width)
    return splits


def trajectory_batch(exp300, tree, sampler, width: int, batch_size: int) -> dict[str, Any]:
    multiplier, multiplicand, modulus = sampler.pairs(batch_size)
    multiplier_bits = torch.tensor(
        [
            [(value >> (width - 1 - position)) & 1 for position in range(width)]
            for value in multiplier
        ],
        dtype=torch.float32,
    )
    states: list[int] = []
    repeated_multiplicand: list[int] = []
    repeated_modulus: list[int] = []
    input_bits: list[int] = []
    control_targets: list[int] = []
    for left, other, prime in zip(multiplier, multiplicand, modulus):
        state = 0
        for position in range(width):
            bit = (left >> (width - 1 - position)) & 1
            target = (2 * state + bit * other) % prime
            states.append(state)
            repeated_multiplicand.append(other)
            repeated_modulus.append(prime)
            input_bits.append(bit)
            control_targets.append(target)
            state = target

    node_values = [[int(bit) for bit in row] for row in multiplier_bits.tolist()]
    level_targets: list[torch.Tensor] = []
    segment_length = 1
    while len(node_values[0]) > 1:
        next_values: list[list[int]] = []
        target_values: list[list[int]] = []
        for row, other, prime in zip(node_values, multiplicand, modulus):
            combined_row: list[int] = []
            residue_row: list[int] = []
            for position in range(0, len(row), 2):
                segment = (row[position] << segment_length) | row[position + 1]
                combined_row.append(segment)
                residue_row.append((segment * other) % prime)
            next_values.append(combined_row)
            target_values.append(residue_row)
        level_targets.append(
            torch.stack(
                [
                    exp300.bits(values, width)
                    for values in target_values
                ],
                dim=0,
            )
        )
        node_values = next_values
        segment_length *= 2
    return {
        "multiplier": multiplier,
        "multiplicand": multiplicand,
        "modulus": modulus,
        "multiplier_bits": multiplier_bits,
        "static_features": tree.make_static_features(multiplicand, modulus, width),
        "tree_level_targets": level_targets,
        "states": states,
        "repeated_multiplicand": repeated_multiplicand,
        "repeated_modulus": repeated_modulus,
        "input_bits": input_bits,
        "control_targets": control_targets,
    }


def exact_bit_loss(
    logits: torch.Tensor, targets: torch.Tensor, worst_bit_weight: float
) -> tuple[torch.Tensor, dict[str, float]]:
    per_bit = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mean = per_bit.mean()
    worst = per_bit.reshape(-1, per_bit.shape[-1]).mean(0).max()
    loss = mean + worst_bit_weight * worst
    exact = (logits.ge(0) == targets.bool()).all(-1).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "mean_bce": float(mean.detach()),
        "worst_bit_bce": float(worst.detach()),
        "node_exact": float(exact.detach()),
    }


def arm_batch_loss(
    architecture: str,
    model: nn.Module,
    batch: dict[str, Any],
    width: int,
    exp300,
    worst_bit_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if architecture == "tree":
        root_logits, level_logits = model(
            batch["multiplier_bits"], batch["static_features"], return_all_levels=True
        )
        logits = torch.cat([item.flatten(0, 1) for item in level_logits], dim=0)
        targets = torch.cat(
            [item.flatten(0, 1) for item in batch["tree_level_targets"]], dim=0
        )
        loss, stats = exact_bit_loss(logits, targets, worst_bit_weight)
        root_target = batch["tree_level_targets"][-1][:, 0].bool()
        root_exact = (root_logits.ge(0) == root_target).all(-1).float().mean()
        return loss, {**stats, "root_exact": float(root_exact.detach())}
    logits, _ = model(
        exp300.features(
            batch["states"],
            batch["repeated_multiplicand"],
            batch["repeated_modulus"],
            batch["input_bits"],
            width,
        )
    )
    targets = exp300.bits(batch["control_targets"], width)
    loss, stats = exact_bit_loss(logits, targets, worst_bit_weight)
    final_rows = (logits.ge(0) == targets.bool()).all(-1).reshape(-1, width)[:, -1]
    return loss, {**stats, "root_exact": float(final_rows.float().mean())}


@torch.inference_mode()
def evaluate_tree_cases(model, tree, cases: dict[str, list[int]], width: int) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    predictions: list[int] = []
    batch_size = 64
    for offset in range(0, len(cases["multiplier"]), batch_size):
        left = cases["multiplier"][offset : offset + batch_size]
        other = cases["multiplicand"][offset : offset + batch_size]
        modulus = cases["modulus"][offset : offset + batch_size]
        multiplier_bits = torch.tensor(
            [
                [(value >> (width - 1 - position)) & 1 for position in range(width)]
                for value in left
            ],
            dtype=torch.float32,
        )
        logits = model(multiplier_bits, tree.make_static_features(other, modulus, width))
        predictions.extend(
            sum(int(bit) << position for position, bit in enumerate(row))
            for row in logits.ge(0).tolist()
        )
    expected = [
        (left * other) % prime
        for left, other, prime in zip(
            cases["multiplier"], cases["multiplicand"], cases["modulus"]
        )
    ]
    correct = sum(value == target for value, target in zip(predictions, expected))
    invalid = sum(value >= prime for value, prime in zip(predictions, cases["modulus"]))
    return {
        "cases": len(expected),
        "correct": correct,
        "exact_accuracy": correct / len(expected),
        "invalid_rate": invalid / len(expected),
        "predictions": predictions,
        "wall_seconds": time.perf_counter() - started,
        "aggregate_cpu_seconds": time.process_time() - cpu_started,
    }


def generated_cases(exp300, primes: list[int], seed: int, count: int) -> dict[str, list[int]]:
    left, right, modulus = exp300.BigIntSampler(primes, seed).pairs(count)
    return {"multiplier": left, "multiplicand": right, "modulus": modulus}


def build_model(architecture: str, width: int, config: dict[str, Any], cellular, tree):
    if architecture == "tree":
        return tree.HierarchicalCompositionTree(width=width, **config["tree_model"])
    return cellular.SharedDilatedConvGRU(width=width, **config["control_model"])


def train_steps(
    architecture: str,
    model: nn.Module,
    optimizer,
    scheduler,
    samplers: dict[int, Any],
    start_step: int,
    end_step: int,
    config: dict[str, Any],
    exp300,
    tree,
    history: list[dict[str, Any]],
    started: float,
    checkpoint_path: Path | None = None,
    hard_deadline: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    widths = config["training"]["widths"]
    peak_rss = psutil.Process().memory_info().rss
    model.train()
    for step in range(start_step, end_step + 1):
        width = widths[(step - 1) % len(widths)]
        model.width = width
        batch = trajectory_batch(
            exp300, tree, samplers[width], width, config["training"]["batch_size"]
        )
        loss, stats = arm_batch_loss(
            architecture,
            model,
            batch,
            width,
            exp300,
            config["training"]["worst_bit_weight"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
        )
        optimizer.step()
        scheduler.step()
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        report = step == 1 or step % 100 == 0 or step == end_step
        if report:
            item = {
                "step": step,
                "width": width,
                "elapsed_wall_seconds": time.perf_counter() - started,
                "gradient_norm": gradient,
                **stats,
            }
            history.append(item)
            if mlflow.active_run() is not None:
                mlflow.log_metrics(
                    {
                        "loss": stats["loss"],
                        "node_exact": stats["node_exact"],
                        "root_exact": stats["root_exact"],
                        "gradient_norm": gradient,
                    },
                    step=step,
                )
            print(json.dumps({"arm": architecture, **item}), flush=True)
        cadence = config["training"]["checkpoint_every_steps"]
        if checkpoint_path and (step % cadence == 0 or step == end_step):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "sampler_rng": {
                        width: sampler.rng.getstate() for width, sampler in samplers.items()
                    },
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                },
                checkpoint_path,
            )
            if hard_deadline is not None and time.perf_counter() > hard_deadline:
                raise RuntimeError(
                    f"hard runaway ceiling reached after preserving step {step}"
                )
    return history, peak_rss


def calibrate(
    config: dict[str, Any], output: Path, splits: dict[int, dict], exp300, cellular, tree
) -> dict[str, Any]:
    calibration = config["resource_budget"]
    updates = int(calibration["calibration_updates_per_arm"])
    inference_cases = int(calibration["calibration_inference_cases_per_width"])
    results: dict[str, Any] = {}
    for arm_index, architecture in enumerate(("control", "tree")):
        random.seed(config["seed"] + 900 + arm_index)
        np.random.seed(config["seed"] + 900 + arm_index)
        torch.manual_seed(config["seed"] + 900 + arm_index)
        model = build_model(
            architecture, max(config["training"]["widths"]), config, cellular, tree
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["steps"],
            eta_min=config["training"]["minimum_learning_rate"],
        )
        samplers = {
            width: exp300.BigIntSampler(
                splits[width]["train"], config["seed"] + 63400 + width + arm_index * 100
            )
            for width in config["training"]["widths"]
        }
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        history, peak = train_steps(
            architecture,
            model,
            optimizer,
            scheduler,
            samplers,
            1,
            updates,
            config,
            exp300,
            tree,
            [],
            wall_started,
        )
        training_wall = time.perf_counter() - wall_started
        training_cpu = time.process_time() - cpu_started
        model.eval()
        inference: dict[str, Any] = {}
        for width in config["evaluation"]["widths"]:
            model.width = width
            cases = generated_cases(
                exp300,
                splits[width]["train"],
                config["seed"] + 63500 + width + arm_index * 100,
                inference_cases,
            )
            if architecture == "tree":
                result = evaluate_tree_cases(model, tree, cases, width)
            else:
                before_wall = time.perf_counter()
                before_cpu = time.process_time()
                result = exp300.evaluate_rollouts(
                    model,
                    exp300.BigIntSampler(
                        splits[width]["train"],
                        config["seed"] + 63500 + width + arm_index * 100,
                    ),
                    inference_cases,
                )
                result["wall_seconds"] = time.perf_counter() - before_wall
                result["aggregate_cpu_seconds"] = time.process_time() - before_cpu
            result.pop("predictions", None)
            inference[str(width)] = result
        results[architecture] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_updates": updates,
            "training_wall_seconds": training_wall,
            "training_aggregate_cpu_seconds": training_cpu,
            "training_history": history,
            "inference": inference,
            "peak_rss_bytes": peak,
        }
    formal_steps = config["training"]["steps"]
    rollout_cases = config["evaluation"]["rollout_cases"]
    projected_training_wall = sum(
        arm["training_wall_seconds"] / updates * formal_steps for arm in results.values()
    )
    projected_training_cpu = sum(
        arm["training_aggregate_cpu_seconds"] / updates * formal_steps
        for arm in results.values()
    )
    projected_evaluation_wall = sum(
        arm["inference"][str(width)]["wall_seconds"] / inference_cases * rollout_cases
        for arm in results.values()
        for width in config["evaluation"]["widths"]
    )
    projected_evaluation_cpu = sum(
        arm["inference"][str(width)]["aggregate_cpu_seconds"]
        / inference_cases
        * rollout_cases
        for arm in results.values()
        for width in config["evaluation"]["widths"]
    )
    tree_width32_wall_per_case = (
        results["tree"]["inference"]["32"]["wall_seconds"] / inference_cases
    )
    projected_controls_wall = tree_width32_wall_per_case * (
        config["evaluation"]["randomized_width32_cases"]
        + 2 * config["evaluation"]["determinism_cases"]
    )
    expected_wall = projected_training_wall + projected_evaluation_wall + projected_controls_wall + 180
    expected_cpu = projected_training_cpu + projected_evaluation_cpu + projected_controls_wall * 4
    recommendation = {
        "expected_total_wall_seconds": math.ceil(expected_wall),
        "expected_total_aggregate_cpu_minutes": math.ceil(expected_cpu / 60),
        "soft_budget_wall_seconds": math.ceil(expected_wall * 1.35 / 60) * 60,
        "hard_runaway_ceiling_wall_seconds": math.ceil(expected_wall * 2.0 / 60) * 60,
        "hard_ceiling_aggregate_cpu_minutes_upper_bound": round(
            math.ceil(expected_wall * 2.0 / 60) * 60 * config["training"]["torch_threads"] / 60,
            2,
        ),
    }
    payload = {
        "run_id": config["run_id"],
        "status": "calibration-only",
        "seed": config["seed"],
        "training_identities_only": True,
        "arms": results,
        "projection": {
            "training_wall_seconds": projected_training_wall,
            "training_aggregate_cpu_seconds": projected_training_cpu,
            "evaluation_wall_seconds": projected_evaluation_wall,
            "evaluation_aggregate_cpu_seconds": projected_evaluation_cpu,
            "controls_wall_seconds": projected_controls_wall,
        },
        "recommendation": recommendation,
        "limitations": [
            "Calibration uses training identities and is not model-selection evidence.",
            "Thirty alternating updates include both formal training widths.",
            "The recommendation adds 180 seconds setup allowance and measured safety headroom.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload


def train_formal_arm(
    architecture: str,
    config: dict[str, Any],
    output: Path,
    splits: dict[int, dict],
    dataset_hash: str,
    source_hashes: dict[str, str],
    exp300,
    cellular,
    tree,
    run_started: float,
) -> dict[str, Any]:
    arm_dir = output / architecture
    arm_dir.mkdir(parents=True, exist_ok=True)
    cache_key = sha256_text(
        json.dumps(
            {**source_hashes, "dataset": dataset_hash, "architecture": architecture},
            sort_keys=True,
        )
    )
    cache_path = arm_dir / f"result-{cache_key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        checkpoint = ROOT / cached["result"]["checkpoint"]["path"]
        if cached.get("cache_key") == cache_key and checkpoint.exists():
            if sha256(checkpoint) == cached["result"]["checkpoint"]["sha256"]:
                return {**cached["result"], "cached": True}

    arm_offset = 0 if architecture == "control" else 1
    random.seed(config["seed"] + arm_offset)
    np.random.seed(config["seed"] + arm_offset)
    torch.manual_seed(config["seed"] + arm_offset)
    model = build_model(
        architecture, max(config["training"]["widths"]), config, cellular, tree
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["steps"],
        eta_min=config["training"]["minimum_learning_rate"],
    )
    samplers = {
        width: exp300.BigIntSampler(
            splits[width]["train"], config["seed"] + 63600 + width + arm_offset * 100
        )
        for width in config["training"]["widths"]
    }
    partial = arm_dir / f"training-{cache_key}.partial.pt"
    start_step = 1
    history: list[dict[str, Any]] = []
    if partial.exists():
        state = torch.load(partial, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        for width, rng_state in state["sampler_rng"].items():
            samplers[int(width)].rng.setstate(rng_state)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        start_step = int(state["step"]) + 1
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    if start_step <= config["training"]["steps"]:
        history, peak = train_steps(
            architecture,
            model,
            optimizer,
            scheduler,
            samplers,
            start_step,
            config["training"]["steps"],
            config,
            exp300,
            tree,
            history,
            wall_started,
            partial,
            run_started + config["resource_budget"]["hard_runaway_ceiling_wall_seconds"],
        )
    else:
        peak = psutil.Process().memory_info().rss
    training_wall = time.perf_counter() - wall_started
    training_cpu = time.process_time() - cpu_started
    checkpoint = arm_dir / "weights.safetensors"
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()},
        checkpoint,
    )
    model.eval()
    evaluation: dict[str, Any] = {}
    for width in config["evaluation"]["widths"]:
        model.width = width
        cases = generated_cases(
            exp300,
            splits[width]["test"],
            config["seed"] + 63700 + width,
            config["evaluation"]["rollout_cases"],
        )
        if architecture == "tree":
            item = evaluate_tree_cases(model, tree, cases, width)
        else:
            before_wall = time.perf_counter()
            before_cpu = time.process_time()
            item = exp300.evaluate_rollouts(
                model,
                exp300.BigIntSampler(
                    splits[width]["test"], config["seed"] + 63700 + width
                ),
                config["evaluation"]["rollout_cases"],
            )
            item["invalid_rate"] = item.pop("any_invalid_residue_rate")
            item["wall_seconds"] = time.perf_counter() - before_wall
            item["aggregate_cpu_seconds"] = time.process_time() - before_cpu
        item.pop("predictions", None)
        evaluation[str(width)] = item
    result = {
        "arm": architecture,
        "cached": False,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": {
            "steps": config["training"]["steps"],
            "trajectory_examples": config["training"]["steps"]
            * config["training"]["batch_size"],
            "history": history,
            "wall_seconds_this_invocation": training_wall,
            "aggregate_cpu_seconds_this_invocation": training_cpu,
        },
        "evaluation": evaluation,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "peak_rss_bytes": peak,
    }
    cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n")
    return result


def load_tree_checkpoint(config, checkpoint: Path, tree):
    from safetensors.torch import load_file

    model = tree.HierarchicalCompositionTree(
        width=max(config["training"]["widths"]), **config["tree_model"]
    )
    model.load_state_dict(load_file(checkpoint))
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_633_hierarchical_learned_composition_feasibility.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-633-hierarchical-learned-composition-feasibility",
    )
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    output = args.output.resolve()
    exp300 = load_module(
        "exp300_for_exp633", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"
    )
    cellular = load_module(
        "cellular_for_exp633", ROOT / "experiments/exp_220_width_independent_cellular/model.py"
    )
    tree = load_module(
        "tree_for_exp633",
        ROOT / "experiments/exp_633_hierarchical_learned_composition_feasibility/model.py",
    )
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    splits = make_splits(exp300, config, config["seed"])
    dataset_hash = sha256_text(json.dumps(splits, sort_keys=True))
    if args.calibrate_only:
        calibrate(config, output, splits, exp300, cellular, tree)
        return 0
    budget = config["resource_budget"]
    if budget.get("status") != "calibrated" or not budget.get(
        "hard_runaway_ceiling_wall_seconds"
    ):
        raise RuntimeError("formal run requires a documented measured-throughput budget")

    output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path)
    runner_hash = sha256(Path(__file__))
    model_hash = sha256(
        ROOT / "experiments/exp_633_hierarchical_learned_composition_feasibility/model.py"
    )
    source_hashes = {"config": config_hash, "runner": runner_hash, "model": model_hash}
    commit = git_output("rev-parse", "HEAD")
    dirty = bool(git_output("status", "--porcelain"))
    started = time.perf_counter()
    cpu_started = time.process_time()
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    results: dict[str, Any] = {}
    child_ids: dict[str, str] = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "status": "running",
                "role": "hierarchical-learned-composition-feasibility",
                "repository_commit": commit,
                "repository_dirty": str(dirty).lower(),
            }
        )
        mlflow.log_params(
            {
                "seed": config["seed"],
                "steps_per_arm": config["training"]["steps"],
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "model_sha256": model_hash,
                "dataset_sha256": dataset_hash,
                "budget_limit_unit": budget["limit_unit"],
                "hard_runaway_ceiling_wall_seconds": budget[
                    "hard_runaway_ceiling_wall_seconds"
                ],
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for architecture in ("control", "tree"):
            with mlflow.start_run(run_name=architecture, nested=True) as child:
                child_ids[architecture] = child.info.run_id
                results[architecture] = train_formal_arm(
                    architecture,
                    config,
                    output,
                    splits,
                    dataset_hash,
                    source_hashes,
                    exp300,
                    cellular,
                    tree,
                    started,
                )
                mlflow.log_metrics(
                    {
                        f"width{width}_accuracy": results[architecture]["evaluation"][
                            str(width)
                        ]["exact_accuracy"]
                        for width in config["evaluation"]["widths"]
                    }
                )
            elapsed = time.perf_counter() - started
            if elapsed > budget["hard_runaway_ceiling_wall_seconds"]:
                raise RuntimeError(
                    f"hard runaway ceiling reached after cached {architecture} arm: {elapsed:.1f}s"
                )

        trained_model = load_tree_checkpoint(
            config, ROOT / results["tree"]["checkpoint"]["path"], tree
        )
        random.seed(config["seed"] + 2)
        np.random.seed(config["seed"] + 2)
        torch.manual_seed(config["seed"] + 2)
        randomized_model = build_model(
            "tree", max(config["training"]["widths"]), config, cellular, tree
        )
        randomized_model.eval()
        random_cases = generated_cases(
            exp300,
            splits[32]["test"],
            config["seed"] + 63832,
            config["evaluation"]["randomized_width32_cases"],
        )
        trained_random_matched = evaluate_tree_cases(
            trained_model, tree, random_cases, 32
        )
        randomized = evaluate_tree_cases(randomized_model, tree, random_cases, 32)
        trained_random_matched.pop("predictions", None)
        randomized.pop("predictions", None)
        randomized_control = {
            "trained_matched": trained_random_matched,
            "randomized": randomized,
            "trained_gain": trained_random_matched["exact_accuracy"]
            - randomized["exact_accuracy"],
        }
        deterministic_cases = generated_cases(
            exp300,
            splits[32]["test"],
            config["seed"] + 63932,
            config["evaluation"]["determinism_cases"],
        )
        repeat_a = evaluate_tree_cases(trained_model, tree, deterministic_cases, 32)
        repeat_b = evaluate_tree_cases(trained_model, tree, deterministic_cases, 32)
        predictions_equal = repeat_a.pop("predictions") == repeat_b.pop("predictions")
        metrics_equal = {
            key: value
            for key, value in repeat_a.items()
            if key not in {"wall_seconds", "aggregate_cpu_seconds"}
        } == {
            key: value
            for key, value in repeat_b.items()
            if key not in {"wall_seconds", "aggregate_cpu_seconds"}
        }
        determinism = {
            "predictions_equal": predictions_equal,
            "metrics_equal": metrics_equal,
            "passed": predictions_equal and metrics_equal,
        }
        control16 = results["control"]["evaluation"]["16"]["exact_accuracy"]
        tree16 = results["tree"]["evaluation"]["16"]["exact_accuracy"]
        control32 = results["control"]["evaluation"]["32"]["exact_accuracy"]
        tree32 = results["tree"]["evaluation"]["32"]["exact_accuracy"]
        width16_gain = tree16 - control16
        width32_gain = tree32 - control32
        parameter_ratio = results["tree"]["parameters"] / results["control"]["parameters"]
        tree_eval16 = results["tree"]["evaluation"]["16"]
        tree_eval32 = results["tree"]["evaluation"]["32"]
        time_ratio = max(tree_eval32["wall_seconds"] / tree_eval16["wall_seconds"], 1.0)
        scaling_exponent = math.log(time_ratio, 2)
        projected_width128_per_case = (
            tree_eval32["wall_seconds"]
            / tree_eval32["cases"]
            * (128 / 32) ** scaling_exponent
        )
        projected_integrated = 36.9 + 110 * projected_width128_per_case
        gates = config["gates"]
        passed = (
            tree16 >= gates["minimum_tree_width16_accuracy"]
            and width16_gain >= gates["minimum_tree_width16_gain_over_control"]
            and tree32 >= gates["minimum_tree_width32_accuracy"]
            and width32_gain >= gates["minimum_tree_width32_gain_over_control"]
            and randomized_control["trained_gain"]
            >= gates["minimum_trained_gain_over_randomized_width32"]
            and max(
                results["tree"]["evaluation"][str(width)]["invalid_rate"]
                for width in config["evaluation"]["widths"]
            )
            <= gates["maximum_invalid_rate"]
            and parameter_ratio <= gates["maximum_parameter_ratio"]
            and projected_integrated <= gates["maximum_projected_integrated_seconds"]
            and determinism["passed"]
        )
        decision = (
            "advance-hierarchical-composition"
            if passed
            else "reject-hierarchical-composition-feasibility"
        )
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": config["tracking"]["experiment_name"],
                "experiment_id": experiment_id,
                "parent_run_id": parent_id,
                "arm_run_ids": child_ids,
            },
            "source": {
                "repository_commit": commit,
                "dirty": dirty,
                "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "model_sha256": model_hash,
                "dataset_sha256": dataset_hash,
            },
            "arms": results,
            "randomized_control": randomized_control,
            "determinism": determinism,
            "comparison": {
                "width16_control": control16,
                "width16_tree": tree16,
                "width16_gain": width16_gain,
                "width32_control": control32,
                "width32_tree": tree32,
                "width32_gain": width32_gain,
                "parameter_ratio": parameter_ratio,
                "runtime_scaling_exponent_16_to_32": scaling_exponent,
                "projected_integrated_seconds": projected_integrated,
            },
            "gates": {**gates, "passed": passed},
            "decision": decision,
            "resource_budget": budget,
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "aggregate_cpu_seconds": time.process_time() - cpu_started,
                "torch_threads": config["training"]["torch_threads"],
                "peak_rss_bytes": psutil.Process().memory_info().rss,
            },
            "limitations": [
                "Fresh generated feasibility evidence only; checked, independent, official, and private answers were not read.",
                "Exact bigint arithmetic is confined to offline node labels and evaluation truth.",
                "The arms match updates, trajectories, widths, and labeled states closely, but their network operations differ by design.",
                "Width-128 runtime is projected from measured width-16/32 scaling and would require direct confirmation after a research-gate pass.",
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics(
            {
                "width16_gain": width16_gain,
                "width32_gain": width32_gain,
                "trained_gain_over_randomized": randomized_control["trained_gain"],
                "projected_integrated_seconds": projected_integrated,
                "wall_seconds": summary["runtime"]["wall_seconds"],
                "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"],
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(
        json.dumps(
            {
                "decision": decision,
                "comparison": summary["comparison"],
                "gates": summary["gates"],
                "runtime": summary["runtime"],
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
