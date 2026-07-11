"""Evaluate a checkpoint on prime-disjoint transitions, rollouts, and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import POCConfig, load_config
from .data import TrajectorySampler, split_primes, training_moduli
from .evaluation import (
    evaluate_rollout,
    evaluate_transitions,
    randomized_copy,
    zero_baseline_rollout,
)
from .model import HornerTCN


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, requested_config: POCConfig) -> tuple[HornerTCN, POCConfig, list[int], list[int]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    embedded = POCConfig.from_mapping(checkpoint.get("config", requested_config.as_dict()))
    config = replace(
        embedded,
        run_id=requested_config.run_id,
        transition_cases=requested_config.transition_cases,
        rollout_cases=requested_config.rollout_cases,
        torch_threads=requested_config.torch_threads,
    )
    train_primes, heldout_primes = split_primes(config)
    train_values = list(
        checkpoint.get(
            "train_moduli",
            checkpoint.get(
                "train_primes",
                training_moduli(config, train_primes, heldout_primes),
            ),
        )
    )
    heldout_primes = list(checkpoint.get("heldout_primes", heldout_primes))
    model = HornerTCN(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config, train_values, heldout_primes


def evaluate_checkpoint(config_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    requested_config = load_config(config_path)
    torch.set_num_threads(requested_config.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    model, config, train_values, heldout_primes = load_checkpoint(
        checkpoint_path, requested_config
    )
    seen_sampler = TrajectorySampler(train_values, config, torch.device("cpu"), config.seed + 3)
    heldout_sampler = TrajectorySampler(heldout_primes, config, torch.device("cpu"), config.seed + 2)

    transition_seen = evaluate_transitions(model, seen_sampler, config)
    transition_heldout = evaluate_transitions(model, heldout_sampler, config)
    rollouts = {
        "seen_primes_16bit": evaluate_rollout(
            model, train_values, config, 16, config.seed + 200
        ),
        "heldout_primes_8bit": evaluate_rollout(
            model, heldout_primes, config, 8, config.seed + 108
        ),
        "heldout_primes_16bit": evaluate_rollout(
            model, heldout_primes, config, 16, config.seed + 116
        ),
        "heldout_primes_32bit": evaluate_rollout(
            model, heldout_primes, config, 32, config.seed + 132
        ),
    }
    per_prime: dict[str, Any] = {}
    for prime in heldout_primes:
        prime_sampler = TrajectorySampler(
            [prime], config, torch.device("cpu"), config.seed + 500 + prime
        )
        per_prime[str(prime)] = {
            "transition": evaluate_transitions(
                model, prime_sampler, config, cases=4_000
            ),
            "rollout_16bit": evaluate_rollout(
                model,
                [prime],
                config,
                16,
                config.seed + 600 + prime,
                cases=1_000,
            ),
        }
    repeat = evaluate_rollout(
        model, heldout_primes, config, 16, config.seed + 116
    )
    deterministic = repeat == rollouts["heldout_primes_16bit"]

    control = randomized_copy(model, config.seed + 9000)
    control_cases = min(4_000, config.transition_cases)
    control_rollout_cases = min(2_000, config.rollout_cases)
    control_sampler = TrajectorySampler(
        heldout_primes, config, torch.device("cpu"), config.seed + 9001
    )
    randomized_transition = evaluate_transitions(
        control, control_sampler, config, cases=control_cases
    )
    randomized_rollout = evaluate_rollout(
        control,
        heldout_primes,
        config,
        16,
        config.seed + 9002,
        cases=control_rollout_cases,
    )
    zero_rollout = zero_baseline_rollout(
        heldout_primes,
        config,
        16,
        config.seed + 9002,
        control_rollout_cases,
    )
    return {
        "run_id": config.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model": {
            "architecture": "non-causal dilated residual TCN",
            "parameters": model.parameter_count(),
            "width": config.width,
            "dilations": model.dilations,
            "hard_binary_feedback": True,
        },
        "split": {
            "strategy": "seeded width-stratified prime identity holdout",
            "train_moduli": train_values,
            "heldout_primes": heldout_primes,
            "composite_augmentation": config.augment_composite_moduli,
        },
        "evaluation": {
            "transition": {
                "seen_primes": transition_seen,
                "heldout_primes": transition_heldout,
            },
            "rollout": rollouts,
            "heldout_per_prime": per_prime,
        },
        "controls": {
            "deterministic_repeat_agreement": 1.0 if deterministic else 0.0,
            "randomized_weights": {
                "transition_heldout_primes": randomized_transition,
                "rollout_heldout_primes_16bit": randomized_rollout,
            },
            "constant_zero": {
                "rollout_heldout_primes_16bit": zero_rollout,
            },
        },
        "runtime": {
            "device": "cpu",
            "torch_threads": config.torch_threads,
            "evaluation_seconds": time.perf_counter() - started,
        },
        "limitations": [
            "Fixed eight-bit state canvas with training moduli limited to four through six bits.",
            "This is a research POC, not an official-interface or competition-scale submission.",
            "Public benchmark data was not used in training or checkpoint selection.",
            "Held-out-prime rollout degradation shows that one-step accuracy is not sufficient.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_checkpoint(args.config, args.checkpoint)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
