"""Train and evaluate the preregistered radix-4 cellular viability rung."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import psutil
import torch
import yaml
from safetensors.torch import load_file, save_file
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
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def decode(logits: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in logits.ge(0).tolist()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_630_radix4_cellular_transition.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-630-radix4-cellular-transition",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    source_path = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source_path) != config["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")

    exp300 = load_module(
        "exp300_for_exp630",
        ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
    )
    radix2_module = load_module(
        "radix2_for_exp630",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    radix4_module = load_module(
        "radix4_for_exp630",
        ROOT / "experiments/exp_630_radix4_cellular_transition/model.py",
    )
    data = load_module(
        "radix4_data_for_exp630",
        ROOT / "experiments/exp_630_radix4_cellular_transition/data.py",
    )
    transfer = load_module(
        "radix4_transfer_for_exp630",
        ROOT / "experiments/exp_630_radix4_cellular_transition/transfer.py",
    )
    seed = int(config["seed"])
    train_widths = [int(width) for width in config["training"]["widths"]]
    eval_widths = [int(width) for width in config["evaluation"]["widths"]]
    all_widths = sorted(set(train_widths + eval_widths))

    def split(width: int) -> dict[str, list[int]]:
        if width == 8:
            train_count = config["data"]["width8_train_prime_identities"]
            test_count = config["data"]["width8_test_prime_identities"]
            prime_bits_min = 2
        else:
            train_count = config["data"]["train_prime_identities"]
            test_count = config["data"]["test_prime_identities"]
            prime_bits_min = width - config["data"]["prime_bit_span"] + 1
        local = {
            "data": {
                "prime_bits_min": prime_bits_min,
                "prime_bits_max": width,
                "train_prime_identities": train_count,
                "test_prime_identities": test_count,
            }
        }
        return exp300.bigint_prime_split(local, seed + 63000 + width)

    splits = {width: split(width) for width in all_widths}
    split_hash = sha256_text(json.dumps(splits, sort_keys=True))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    process = psutil.Process()
    started = time.perf_counter()
    cpu_started = time.process_time()
    config_hash = sha256(config_path)
    runner_hash = sha256(Path(__file__))
    model_hash = sha256(Path(__file__).with_name("model.py"))
    data_hash = sha256(Path(__file__).with_name("data.py"))
    transfer_hash = sha256(Path(__file__).with_name("transfer.py"))
    repository_commit = git_output("rev-parse", "HEAD")

    def check_runaway() -> None:
        if (
            time.perf_counter() - started
            > config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]
        ):
            raise TimeoutError("EXP-630 reached its preregistered wall runaway ceiling")

    source = radix2_module.SharedDilatedConvGRU(
        width=max(all_widths),
        hidden=config["model"]["hidden"],
        cycles=config["model"]["cycles"],
    )
    source.load_state_dict(load_file(source_path))
    source.eval()
    challenger = radix4_module.Radix4SharedDilatedConvGRU(
        width=max(all_widths),
        hidden=config["model"]["hidden"],
        cycles=config["model"]["cycles"],
    )
    transfer_report = transfer.transfer_radix2_state(challenger, source.state_dict())

    def evaluate_transitions(model, sampler, width: int, cases: int) -> dict:
        seen = correct = valid = 0
        model.width = width
        with torch.inference_mode():
            while seen < cases:
                size = min(512, cases - seen)
                states, rights, moduli, digits, targets, _ = sampler.transitions(size)
                logits, _ = model(data.features(states, rights, moduli, digits, width))
                predictions = decode(logits)
                correct += sum(
                    predicted == expected
                    for predicted, expected in zip(predictions, targets)
                )
                valid += sum(
                    predicted < modulus
                    for predicted, modulus in zip(predictions, moduli)
                )
                seen += size
        return {
            "cases": seen,
            "correct": correct,
            "exact_accuracy": correct / seen,
            "valid_residue_rate": valid / seen,
        }

    def rollout_radix2(left, right, moduli, width: int) -> dict:
        states = [0] * len(left)
        teacher = [0] * len(left)
        all_steps = [True] * len(left)
        invalid = [False] * len(left)
        source.width = width
        wall = time.perf_counter()
        cpu = time.process_time()
        with torch.inference_mode():
            for position in range(width):
                input_bits = [
                    (value >> (width - 1 - position)) & 1 for value in left
                ]
                logits, _ = source(
                    exp300.features(states, right, moduli, input_bits, width)
                )
                states = decode(logits)
                teacher = [
                    (2 * value + bit * other) % modulus
                    for value, bit, other, modulus in zip(
                        teacher, input_bits, right, moduli
                    )
                ]
                all_steps = [
                    prior and predicted == expected
                    for prior, predicted, expected in zip(all_steps, states, teacher)
                ]
                invalid = [
                    prior or predicted >= modulus
                    for prior, predicted, modulus in zip(invalid, states, moduli)
                ]
        expected = [(a * b) % p for a, b, p in zip(left, right, moduli)]
        return {
            "predictions": states,
            "cases": len(left),
            "correct": sum(a == b for a, b in zip(states, expected)),
            "exact_accuracy": sum(a == b for a, b in zip(states, expected))
            / len(left),
            "all_steps_exact_accuracy": sum(all_steps) / len(left),
            "any_invalid_residue_rate": sum(invalid) / len(left),
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "recurrent_steps": width,
        }

    def rollout_radix4(left, right, moduli, width: int) -> dict:
        if width % 2:
            raise ValueError("radix-4 evaluation width must be even")
        states = [0] * len(left)
        teacher = [0] * len(left)
        all_steps = [True] * len(left)
        invalid = [False] * len(left)
        challenger.width = width
        wall = time.perf_counter()
        cpu = time.process_time()
        with torch.inference_mode():
            for position in range(width // 2):
                digits = [
                    (value >> (width - 2 * (position + 1))) & 3 for value in left
                ]
                logits, _ = challenger(
                    data.features(states, right, moduli, digits, width)
                )
                states = decode(logits)
                teacher = [
                    data.exact_transition(value, other, modulus, digit)[0]
                    for value, other, modulus, digit in zip(
                        teacher, right, moduli, digits
                    )
                ]
                all_steps = [
                    prior and predicted == expected
                    for prior, predicted, expected in zip(all_steps, states, teacher)
                ]
                invalid = [
                    prior or predicted >= modulus
                    for prior, predicted, modulus in zip(invalid, states, moduli)
                ]
        expected = [(a * b) % p for a, b, p in zip(left, right, moduli)]
        correct = sum(a == b for a, b in zip(states, expected))
        return {
            "predictions": states,
            "cases": len(left),
            "correct": correct,
            "exact_accuracy": correct / len(left),
            "all_steps_exact_accuracy": sum(all_steps) / len(left),
            "any_invalid_residue_rate": sum(invalid) / len(left),
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "recurrent_steps": width // 2,
        }

    optimizer = torch.optim.AdamW(
        challenger.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["steps"],
        eta_min=config["training"]["minimum_learning_rate"],
    )
    samplers = {
        width: data.Radix4Sampler(splits[width]["train"], seed + 63100 + width)
        for width in train_widths
    }
    partial_key = sha256_text(
        json.dumps(
            {
                "source": sha256(source_path),
                "config": config_hash,
                "runner": runner_hash,
                "model": model_hash,
                "data": data_hash,
                "transfer": transfer_hash,
                "split": split_hash,
            },
            sort_keys=True,
        )
    )
    partial_path = output / f"training-{partial_key}.partial.pt"
    completed = 0
    history: list[dict] = []
    cumulative_wall = 0.0
    cumulative_cpu = 0.0
    if partial_path.exists():
        partial = torch.load(partial_path, weights_only=False)
        if partial.get("cache_key") != partial_key:
            raise RuntimeError("training cache key mismatch")
        completed = partial["step"]
        history = partial["history"]
        cumulative_wall = partial["wall_seconds"]
        cumulative_cpu = partial["aggregate_cpu_seconds"]
        challenger.load_state_dict(partial["model"])
        optimizer.load_state_dict(partial["optimizer"])
        scheduler.load_state_dict(partial["scheduler"])
        for width in train_widths:
            samplers[width].rng.setstate(partial["sampler_rng_states"][width])

    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "status": "running",
                "role": "radix4-cellular-viability",
                "repository_commit": repository_commit,
                "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(),
            }
        )
        mlflow.log_params(
            {
                "seed": seed,
                "steps": config["training"]["steps"],
                "widths": str(train_widths),
                "source_checkpoint_sha256": sha256(source_path),
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "split_sha256": split_hash,
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(
            str(ROOT / config["preregistration"]), artifact_path="preregistration"
        )
        challenger.train()
        segment_wall = time.perf_counter()
        segment_cpu = time.process_time()
        peak_rss = process.memory_info().rss
        for step in range(completed + 1, config["training"]["steps"] + 1):
            width = train_widths[(step - 1) % len(train_widths)]
            states, rights, moduli, digits, targets, quotients = samplers[
                width
            ].transitions(config["training"]["batch_size"])
            challenger.width = width
            logits, quotient_logits = challenger(
                data.features(states, rights, moduli, digits, width)
            )
            loss, stats = exp300.transition_loss(
                logits,
                quotient_logits,
                targets,
                quotients,
                width,
                config["training"]["worst_bit_weight"],
                config["training"]["quotient_weight"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(
                nn.utils.clip_grad_norm_(
                    challenger.parameters(), config["training"]["gradient_clip"]
                )
            )
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 50 == 0 or step == config["training"]["steps"]:
                peak_rss = max(peak_rss, process.memory_info().rss)
                item = {
                    "step": step,
                    "width": width,
                    "elapsed_wall_seconds": cumulative_wall
                    + time.perf_counter()
                    - segment_wall,
                    "gradient_norm": gradient,
                    **stats,
                }
                history.append(item)
                mlflow.log_metrics(
                    {
                        "train_loss": stats["loss"],
                        "train_batch_exact": stats["batch_exact"],
                        "gradient_norm": gradient,
                    },
                    step=step,
                )
                print(json.dumps(item), flush=True)
            if (
                step % config["training"]["save_every_steps"] == 0
                or step == config["training"]["steps"]
            ):
                wall_total = cumulative_wall + time.perf_counter() - segment_wall
                cpu_total = cumulative_cpu + time.process_time() - segment_cpu
                torch.save(
                    {
                        "cache_key": partial_key,
                        "step": step,
                        "history": history,
                        "wall_seconds": wall_total,
                        "aggregate_cpu_seconds": cpu_total,
                        "model": challenger.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "sampler_rng_states": {
                            width: sampler.rng.getstate()
                            for width, sampler in samplers.items()
                        },
                    },
                    partial_path,
                )
                check_runaway()
        partial = torch.load(partial_path, weights_only=False)
        challenger.eval()
        checkpoint = output / "weights.safetensors"
        save_file(
            {
                key: value.detach().contiguous()
                for key, value in challenger.state_dict().items()
            },
            checkpoint,
        )
        evaluation = {}
        for width in eval_widths:
            transition = evaluate_transitions(
                challenger,
                data.Radix4Sampler(splits[width]["test"], seed + 63200 + width),
                width,
                config["evaluation"]["transition_cases"],
            )
            pair_sampler = data.Radix4Sampler(
                splits[width]["test"], seed + 63300 + width
            )
            pairs = pair_sampler.pairs(config["evaluation"]["rollout_cases"])
            control_rollout = rollout_radix2(*pairs, width)
            challenger_rollout = rollout_radix4(*pairs, width)
            control_rollout.pop("predictions")
            challenger_rollout.pop("predictions")
            evaluation[str(width)] = {
                "transition": transition,
                "radix2_control_rollout": control_rollout,
                "radix4_challenger_rollout": challenger_rollout,
                "radix4_to_radix2_wall_ratio": challenger_rollout["wall_seconds"]
                / control_rollout["wall_seconds"],
            }
        determinism_sampler = data.Radix4Sampler(
            splits[64]["test"], seed + 63464
        )
        determinism_pairs = determinism_sampler.pairs(
            config["evaluation"]["determinism_cases"]
        )
        first = rollout_radix4(*determinism_pairs, 64)
        second = rollout_radix4(*determinism_pairs, 64)
        determinism = {
            "cases": config["evaluation"]["determinism_cases"],
            "predictions_equal": first["predictions"] == second["predictions"],
            "correct_equal": first["correct"] == second["correct"],
            "invalid_equal": first["any_invalid_residue_rate"]
            == second["any_invalid_residue_rate"],
        }
        determinism["passed"] = all(
            determinism[key]
            for key in ("predictions_equal", "correct_equal", "invalid_equal")
        )
        gates = config["gates"]
        e16, e32, e64 = (evaluation[str(width)] for width in (16, 32, 64))
        passed = (
            e16["radix4_challenger_rollout"]["exact_accuracy"]
            >= gates["minimum_width16_rollout"]
            and e32["radix4_challenger_rollout"]["exact_accuracy"]
            >= gates["minimum_width32_rollout"]
            and e64["radix4_challenger_rollout"]["exact_accuracy"]
            >= gates["minimum_width64_rollout"]
            and e32["transition"]["exact_accuracy"]
            >= gates["minimum_width32_transition"]
            and e64["transition"]["exact_accuracy"]
            >= gates["minimum_width64_transition"]
            and e64["radix4_challenger_rollout"]["any_invalid_residue_rate"]
            <= gates["maximum_width64_invalid_rollout_rate"]
            and e64["radix4_to_radix2_wall_ratio"]
            <= gates["maximum_width64_radix4_to_radix2_wall_ratio"]
            and (determinism["passed"] or not gates["require_determinism"])
        )
        decision = (
            "advance-radix4-higher-width-scaling"
            if passed
            else "reject-radix4-cellular-viability"
        )
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": config["tracking"]["experiment_name"],
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
            },
            "source": {
                "repository_commit": repository_commit,
                "dirty": bool(git_output("status", "--porcelain")),
                "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "model_sha256": model_hash,
                "data_sha256": data_hash,
                "transfer_sha256": transfer_hash,
                "split_sha256": split_hash,
                "source_checkpoint_sha256": sha256(source_path),
            },
            "transfer": transfer_report,
            "training": {
                "steps": config["training"]["steps"],
                "history": history,
                "wall_seconds": partial["wall_seconds"],
                "aggregate_cpu_seconds": partial["aggregate_cpu_seconds"],
            },
            "models": {
                "radix2_parameters": source.parameter_count(),
                "radix4_parameters": challenger.parameter_count(),
                "radix4_checkpoint": {
                    "path": str(checkpoint.relative_to(ROOT)),
                    "sha256": sha256(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                },
            },
            "evaluation": evaluation,
            "determinism": determinism,
            "gates": {**gates, "passed": passed},
            "decision": decision,
            "resource_budget": config["resource_budget"],
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "aggregate_cpu_seconds": time.process_time() - cpu_started,
                "torch_threads": config["training"]["torch_threads"],
                "peak_rss_bytes": peak_rss,
            },
            "limitations": [
                "Architecture viability evidence only; no checked, consumed independent, or private answers were read.",
                "Radix and quotient-head size change together with autonomous horizon.",
                "Exact bigint arithmetic is confined to offline generation, labels, tests, and evaluation truth.",
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics(
            {
                "width16_rollout": e16["radix4_challenger_rollout"][
                    "exact_accuracy"
                ],
                "width32_rollout": e32["radix4_challenger_rollout"][
                    "exact_accuracy"
                ],
                "width64_rollout": e64["radix4_challenger_rollout"][
                    "exact_accuracy"
                ],
                "width64_transition": e64["transition"]["exact_accuracy"],
                "width64_runtime_ratio": e64["radix4_to_radix2_wall_ratio"],
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
                "evaluation": evaluation,
                "determinism": determinism,
                "checkpoint": summary["models"]["radix4_checkpoint"],
                "runtime": summary["runtime"],
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
