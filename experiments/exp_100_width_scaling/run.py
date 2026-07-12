"""Run the preregistered EXP-100 width-9/width-16 smoke comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

from model import (  # noqa: E402
    SharedContextHornerTCN,
    ints_to_bits,
    make_features,
    predict_state,
)


CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def primes_upto(limit: int) -> list[int]:
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for value in range(2, int(limit**0.5) + 1):
        if sieve[value]:
            start = value * value
            sieve[start : limit + 1 : value] = b"\x00" * (
                ((limit - start) // value) + 1
            )
    return [value for value, flag in enumerate(sieve) if flag]


def split_primes(
    minimum_bits: int,
    maximum_bits: int,
    seed: int,
) -> dict[str, list[int]]:
    all_primes = primes_upto((1 << maximum_bits) - 1)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for bits in range(minimum_bits, maximum_bits + 1):
        group = [prime for prime in all_primes if prime.bit_length() == bits]
        rng.shuffle(group)
        if len(group) < 3:
            splits["train"].extend(group[:-1])
            splits["test"].extend(group[-1:])
            continue
        test_count = max(1, round(len(group) * 0.10))
        validation_count = max(1, round(len(group) * 0.10))
        splits["test"].extend(group[:test_count])
        splits["validation"].extend(
            group[test_count : test_count + validation_count]
        )
        splits["train"].extend(group[test_count + validation_count :])
    return splits


class TransitionSampler:
    def __init__(
        self,
        width: int,
        minimum_bits: int,
        maximum_bits: int,
        primes: list[int],
        seed: int,
        composite_fraction: float,
        random_state_fraction: float,
    ) -> None:
        self.width = width
        self.minimum_bits = minimum_bits
        self.maximum_bits = maximum_bits
        self.primes = torch.tensor(primes, dtype=torch.int64)
        all_prime_set = set(primes_upto((1 << maximum_bits) - 1))
        low = max(2, 1 << (minimum_bits - 1))
        high = 1 << maximum_bits
        self.composites = torch.tensor(
            [value for value in range(low, high) if value not in all_prime_set],
            dtype=torch.int64,
        )
        self.generator = torch.Generator().manual_seed(seed)
        self.composite_fraction = composite_fraction
        self.random_state_fraction = random_state_fraction

    def pairs(self, size: int, composites: bool) -> tuple[Tensor, Tensor, Tensor]:
        modulus = self.primes[
            torch.randint(len(self.primes), (size,), generator=self.generator)
        ].clone()
        if composites and self.composite_fraction > 0:
            use_composite = (
                torch.rand(size, generator=self.generator)
                < self.composite_fraction
            )
            count = int(use_composite.sum())
            if count:
                modulus[use_composite] = self.composites[
                    torch.randint(
                        len(self.composites),
                        (count,),
                        generator=self.generator,
                    )
                ]
        multiplier = (
            torch.rand(size, generator=self.generator) * modulus.double()
        ).long()
        multiplicand = (
            torch.rand(size, generator=self.generator) * modulus.double()
        ).long()
        return multiplier, multiplicand, modulus

    def transitions(
        self, size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        multiplier, multiplicand, modulus = self.pairs(size, composites=True)
        step = torch.randint(
            0, self.width, (size,), generator=self.generator
        )
        prefix = multiplier >> (self.width - step)
        state = (prefix * multiplicand) % modulus
        bit = (multiplier >> (self.width - 1 - step)) & 1
        random_mask = (
            torch.rand(size, generator=self.generator)
            < self.random_state_fraction
        )
        count = int(random_mask.sum())
        if count:
            state[random_mask] = (
                torch.rand(count, generator=self.generator)
                * modulus[random_mask].double()
            ).long()
            bit[random_mask] = torch.randint(
                0, 2, (count,), generator=self.generator
            )
        raw = 2 * state + bit * multiplicand
        target = raw % modulus
        quotient = raw // modulus
        return state, multiplicand, modulus, bit, target, quotient


def transition_loss(
    logits: Tensor,
    quotient_logits: Tensor,
    target: Tensor,
    quotient: Tensor,
    width: int,
    worst_bit_weight: float,
    quotient_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    target_bits = ints_to_bits(target, width)
    per_bit = F.binary_cross_entropy_with_logits(
        logits, target_bits, reduction="none"
    )
    mean_bce = per_bit.mean()
    worst_bit = per_bit.max(1).values.mean()
    quotient_ce = F.cross_entropy(quotient_logits, quotient)
    loss = (
        mean_bce
        + worst_bit_weight * worst_bit
        + quotient_weight * quotient_ce
    )
    with torch.no_grad():
        exact = (logits >= 0).eq(target_bits.bool()).all(1).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "mean_bce": float(mean_bce.detach()),
        "worst_bit_bce": float(worst_bit.detach()),
        "quotient_ce": float(quotient_ce.detach()),
        "batch_exact": float(exact),
    }


@torch.inference_mode()
def evaluate_transitions(
    model: SharedContextHornerTCN,
    sampler: TransitionSampler,
    cases: int,
) -> dict[str, Any]:
    model.eval()
    correct = bit_correct = valid = processed = 0
    by_bits: dict[int, list[int]] = {}
    while processed < cases:
        size = min(4096, cases - processed)
        state, multiplicand, modulus, bit, target, _ = sampler.transitions(size)
        predicted, logits = predict_state(
            model, state, multiplicand, modulus, bit
        )
        exact = predicted.eq(target)
        correct += int(exact.sum())
        bit_correct += int(
            (logits >= 0).eq(ints_to_bits(target, model.width).bool()).sum()
        )
        valid += int((predicted < modulus).sum())
        for bit_length in range(2, model.width + 1):
            mask = modulus.ge(1 << (bit_length - 1)) & modulus.lt(1 << bit_length)
            count = int(mask.sum())
            if count:
                stats = by_bits.setdefault(bit_length, [0, 0])
                stats[0] += int(exact[mask].sum())
                stats[1] += count
        processed += size
    return {
        "cases": cases,
        "correct": correct,
        "exact_accuracy": correct / cases,
        "bit_accuracy": bit_correct / (cases * model.width),
        "valid_residue_rate": valid / cases,
        "by_modulus_bits": {
            str(bits): {
                "correct": values[0],
                "cases": values[1],
                "exact_accuracy": values[0] / values[1],
            }
            for bits, values in sorted(by_bits.items())
        },
    }


@torch.inference_mode()
def evaluate_rollouts(
    model: SharedContextHornerTCN,
    sampler: TransitionSampler,
    cases: int,
) -> dict[str, Any]:
    model.eval()
    correct = chain_correct = invalid_count = processed = 0
    first_error = [0] * model.width
    by_bits: dict[int, list[int]] = {}
    while processed < cases:
        size = min(1024, cases - processed)
        multiplier, multiplicand, modulus = sampler.pairs(
            size, composites=False
        )
        learned = torch.zeros(size, dtype=torch.int64)
        oracle = torch.zeros_like(learned)
        chain = torch.ones(size, dtype=torch.bool)
        seen_error = torch.zeros(size, dtype=torch.bool)
        invalid = torch.zeros(size, dtype=torch.bool)
        for step in range(model.width):
            bit = (multiplier >> (model.width - 1 - step)) & 1
            learned, _ = predict_state(
                model, learned, multiplicand, modulus, bit
            )
            oracle = (2 * oracle + bit * multiplicand) % modulus
            step_ok = learned.eq(oracle)
            newly_wrong = (~step_ok) & (~seen_error)
            first_error[step] += int(newly_wrong.sum())
            seen_error |= ~step_ok
            chain &= step_ok
            invalid |= learned >= modulus
        exact = learned.eq((multiplier * multiplicand) % modulus)
        correct += int(exact.sum())
        chain_correct += int(chain.sum())
        invalid_count += int(invalid.sum())
        for bit_length in range(2, model.width + 1):
            mask = modulus.ge(1 << (bit_length - 1)) & modulus.lt(1 << bit_length)
            count = int(mask.sum())
            if count:
                stats = by_bits.setdefault(bit_length, [0, 0])
                stats[0] += int(exact[mask].sum())
                stats[1] += count
        processed += size
    return {
        "cases": cases,
        "correct": correct,
        "exact_accuracy": correct / cases,
        "all_steps_correct": chain_correct,
        "all_steps_exact_accuracy": chain_correct / cases,
        "any_invalid_residue_rate": invalid_count / cases,
        "first_error_by_step": first_error,
        "by_modulus_bits": {
            str(bits): {
                "correct": values[0],
                "cases": values[1],
                "exact_accuracy": values[0] / values[1],
            }
            for bits, values in sorted(by_bits.items())
        },
    }


def train_arm(
    arm: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    seed = int(config["seed"])
    width = int(arm["width"])
    random.seed(seed + width)
    np.random.seed(seed + width)
    torch.manual_seed(seed + width)
    split = split_primes(
        int(arm["prime_bits_min"]),
        int(arm["prime_bits_max"]),
        seed + width,
    )
    training = config["training"]
    evaluation = config["evaluation"]
    train_sampler = TransitionSampler(
        width,
        int(arm["prime_bits_min"]),
        int(arm["prime_bits_max"]),
        split["train"],
        seed + width * 100 + 1,
        float(training["composite_fraction"]),
        float(training["random_state_fraction"]),
    )
    model = SharedContextHornerTCN(
        width=width,
        hidden=int(config["model"]["hidden"]),
        blocks=int(config["model"]["blocks"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["resource_cap"]["continuation_steps"]),
        eta_min=float(training["minimum_learning_rate"]),
    )
    test_seed = seed + width * 100 + 50_000
    untrained_transition = evaluate_transitions(
        model,
        TransitionSampler(
            width,
            int(arm["prime_bits_min"]),
            int(arm["prime_bits_max"]),
            split["test"],
            test_seed,
            0.0,
            float(training["random_state_fraction"]),
        ),
        min(2000, int(evaluation["heldout_transition_cases"])),
    )
    untrained_rollout = evaluate_rollouts(
        model,
        TransitionSampler(
            width,
            int(arm["prime_bits_min"]),
            int(arm["prime_bits_max"]),
            split["test"],
            test_seed + 1,
            0.0,
            0.0,
        ),
        min(1000, int(evaluation["rollout_cases"])),
    )
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    model.train()
    steps = int(training["steps"])
    for step in range(1, steps + 1):
        state, multiplicand, modulus, bit, target, quotient = (
            train_sampler.transitions(int(training["batch_size"]))
        )
        logits, quotient_logits = model(
            make_features(state, multiplicand, modulus, bit, width)
        )
        loss, stats = transition_loss(
            logits,
            quotient_logits,
            target,
            quotient,
            width,
            float(training["worst_bit_weight"]),
            float(training["quotient_weight"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == steps:
            peak_rss = max(peak_rss, process.memory_info().rss)
            record = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": gradient_norm,
                **stats,
            }
            history.append(record)
            mlflow.log_metrics(
                {
                    "train_loss": stats["loss"],
                    "train_batch_exact": stats["batch_exact"],
                    "train_bit_bce": stats["mean_bce"],
                    "gradient_norm": gradient_norm,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=step,
            )
            print(json.dumps({"arm": arm["name"], **record}), flush=True)
    training_seconds = time.perf_counter() - started
    transition = evaluate_transitions(
        model,
        TransitionSampler(
            width,
            int(arm["prime_bits_min"]),
            int(arm["prime_bits_max"]),
            split["test"],
            test_seed,
            0.0,
            float(training["random_state_fraction"]),
        ),
        int(evaluation["heldout_transition_cases"]),
    )
    rollout = evaluate_rollouts(
        model,
        TransitionSampler(
            width,
            int(arm["prime_bits_min"]),
            int(arm["prime_bits_max"]),
            split["test"],
            test_seed + 1,
            0.0,
            0.0,
        ),
        int(evaluation["rollout_cases"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "weights.safetensors"
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()},
        str(checkpoint),
        metadata={
            "experiment": config["run_id"],
            "arm": arm["name"],
            "width": str(width),
        },
    )
    split_path = output_dir / "prime-split.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    split_hash = hashlib.sha256(
        json.dumps(split, sort_keys=True).encode("utf-8")
    ).hexdigest()
    delta = history[-1]["batch_exact"] - history[-3]["batch_exact"]
    result = {
        "arm": arm,
        "parameters": model.parameter_count(),
        "prime_split": {
            "train": len(split["train"]),
            "validation": len(split["validation"]),
            "test": len(split["test"]),
            "sha256": split_hash,
        },
        "training": {
            "steps": steps,
            "examples": steps * int(training["batch_size"]),
            "seconds": training_seconds,
            "peak_rss_bytes": peak_rss,
            "final_100_step_batch_exact_delta": delta,
            "history": history,
        },
        "controls": {
            "untrained_transition": untrained_transition,
            "untrained_rollout": untrained_rollout,
        },
        "evaluation": {
            "heldout_transition": transition,
            "heldout_rollout": rollout,
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "limitations": [
            "Smoke-scale checkpoint selected only by preregistered heldout-prime gates.",
            "No checked public expected answers were read or evaluated.",
            "The width bands differ by construction; this tests scaling, not equal task difficulty.",
        ],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    mlflow.log_params(
        {
            "width": width,
            "prime_bits_min": arm["prime_bits_min"],
            "prime_bits_max": arm["prime_bits_max"],
            "parameters": result["parameters"],
            "train_primes": len(split["train"]),
            "test_primes": len(split["test"]),
            "prime_split_sha256": split_hash,
            "checkpoint_sha256": result["checkpoint"]["sha256"],
        }
    )
    mlflow.log_metrics(
        {
            "transition_exact": transition["exact_accuracy"],
            "transition_bit_accuracy": transition["bit_accuracy"],
            "rollout_exact": rollout["exact_accuracy"],
            "rollout_all_steps_exact": rollout["all_steps_exact_accuracy"],
            "rollout_invalid_rate": rollout["any_invalid_residue_rate"],
            "untrained_transition_exact": untrained_transition["exact_accuracy"],
            "untrained_rollout_exact": untrained_rollout["exact_accuracy"],
            "training_seconds": training_seconds,
            "peak_rss_bytes": peak_rss,
            "final_100_step_batch_exact_delta": delta,
        }
    )
    mlflow.log_artifact(str(checkpoint), artifact_path="checkpoint")
    mlflow.log_artifact(str(metrics_path), artifact_path="diagnostics")
    mlflow.log_artifact(str(split_path), artifact_path="diagnostics")
    return result


def gate_decision(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> str:
    gates = config["gates"]
    control = results["width9-control"]
    challenger = results["width16-challenger"]
    control_valid = (
        control["evaluation"]["heldout_transition"]["exact_accuracy"]
        >= float(gates["width9_valid_transition"])
        and control["evaluation"]["heldout_rollout"]["exact_accuracy"]
        >= float(gates["width9_valid_rollout"])
    )
    if not control_valid:
        return "invalid-shared-context-surface"
    strong = (
        challenger["evaluation"]["heldout_transition"]["exact_accuracy"]
        >= float(gates["width16_strong_transition"])
        and challenger["evaluation"]["heldout_rollout"]["exact_accuracy"]
        >= float(gates["width16_strong_rollout"])
    )
    if strong:
        return "strong-pass-full-independent-gate"
    continuation = (
        challenger["evaluation"]["heldout_transition"]["exact_accuracy"]
        >= float(gates["width16_continue_transition"])
        and challenger["evaluation"]["heldout_rollout"]["exact_accuracy"]
        >= float(gates["width16_continue_rollout"])
        and challenger["training"]["final_100_step_batch_exact_delta"] > 0
    )
    if continuation:
        return "continue-width16-to-1200"
    falsified = (
        challenger["evaluation"]["heldout_transition"]["exact_accuracy"]
        < float(gates["width16_falsify_transition"])
        and challenger["evaluation"]["heldout_rollout"]["exact_accuracy"]
        < float(gates["width16_falsify_rollout"])
        and challenger["training"]["final_100_step_batch_exact_delta"] <= 0
    )
    return "falsified-direct-scaling" if falsified else "inconclusive-revise"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_100_width16_shared_context_smoke.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-100-width16-shared-context-smoke",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    configure_mlflow(config["tracking"]["experiment_name"])
    git_commit = git_output("rev-parse", "HEAD")
    git_dirty = bool(git_output("status", "--porcelain"))
    source_hashes = {
        "config": sha256(config_path),
        "model": sha256(Path(__file__).with_name("model.py")),
        "runner": sha256(Path(__file__)),
        "data_checksums": sha256(ROOT / "data/checksums.sha256"),
    }
    started = time.perf_counter()
    arm_results: dict[str, dict[str, Any]] = {}
    arm_run_ids: dict[str, str] = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "hypothesis": config["hypothesis"],
                "status": "running",
                "role": "challenger-smoke",
                "baseline_run": "candidate-001-horner-tcn9",
                "ab_group": "EXP-100",
                "promotion_gate": "preregistered-width16-smoke",
                "repository_commit": git_commit,
                "repository_dirty": str(git_dirty).lower(),
                "challenge_commit": CHALLENGE_COMMIT,
                "config_path": str(config_path.relative_to(ROOT)),
            }
        )
        mlflow.log_params(
            {
                "seed": config["seed"],
                "steps": config["training"]["steps"],
                "batch_size": config["training"]["batch_size"],
                "hidden": config["model"]["hidden"],
                "blocks": config["model"]["blocks"],
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "device": "cpu",
                "torch_threads": config["training"]["torch_threads"],
                **{f"sha256_{key}": value for key, value in source_hashes.items()},
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(
            str(ROOT / "knowledge/autoresearch/ablations/exp-100-width16-shared-context.md"),
            artifact_path="preregistration",
        )
        for arm in config["comparison"]["arms"]:
            with mlflow.start_run(run_name=arm["name"], nested=True) as nested:
                arm_run_ids[arm["name"]] = nested.info.run_id
                mlflow.set_tags(
                    {
                        "canonical_run_id": config["run_id"],
                        "parent_run_id": parent_run_id,
                        "ab_group": arm["name"],
                        "status": "running",
                        "promotion_gate": "preregistered-width-scaling-smoke",
                    }
                )
                arm_results[arm["name"]] = train_arm(
                    arm,
                    config,
                    output / arm["name"],
                )
                mlflow.set_tag("status", "completed")
        decision = gate_decision(arm_results, config)
        elapsed = time.perf_counter() - started
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": config["tracking"]["experiment_name"],
                "parent_run_id": parent_run_id,
                "arm_run_ids": arm_run_ids,
            },
            "source": {
                "repository_commit": git_commit,
                "dirty": git_dirty,
                "challenge_commit": CHALLENGE_COMMIT,
                "sha256": source_hashes,
            },
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "numpy": np.__version__,
                "mlflow": mlflow.__version__,
                "platform": platform.platform(),
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
                "cpu_count": os.cpu_count(),
            },
            "resource": {
                "wall_seconds": elapsed,
                "cap_cpu_minutes": config["resource_cap"]["aggregate_cpu_minutes"],
            },
            "decision": decision,
            "results": arm_results,
            "limitations": [
                "Smoke-scale A/B; no public expected answers consulted.",
                "The prime bands differ because the ranking question is width scaling.",
                "A continuation decision is not candidate promotion.",
            ],
        }
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        mlflow.log_metrics(
            {
                "width9_transition_exact": arm_results["width9-control"]["evaluation"]["heldout_transition"]["exact_accuracy"],
                "width9_rollout_exact": arm_results["width9-control"]["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "width16_transition_exact": arm_results["width16-challenger"]["evaluation"]["heldout_transition"]["exact_accuracy"],
                "width16_rollout_exact": arm_results["width16-challenger"]["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "wall_seconds": elapsed,
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
