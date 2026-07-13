"""Matched uniform versus balanced first-error replay for the Tier-6 cell."""

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
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15
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
        default=ROOT / "configs/exp_629_tier6_balanced_first_error_replay.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-629-tier6-balanced-first-error-replay",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")

    exp300 = load_module(
        "exp300_for_exp629",
        ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
    )
    cellular = load_module(
        "cellular_for_exp629",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    replay = load_module(
        "replay_for_exp629",
        ROOT / "experiments/exp_629_tier6_balanced_first_error_replay/replay.py",
    )
    seed = int(config["seed"])
    widths = (64, 128)

    def split(width: int) -> dict[str, list[int]]:
        local = {
            "data": {
                "prime_bits_min": width - config["data"]["prime_bit_span"] + 1,
                "prime_bits_max": width,
                "train_prime_identities": config["data"]["train_prime_identities"],
                "test_prime_identities": config["data"]["test_prime_identities"],
            }
        }
        return exp300.bigint_prime_split(local, seed + 62900 + width)

    splits = {width: split(width) for width in widths}
    split_hash = sha256_text(json.dumps(splits, sort_keys=True))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    process = psutil.Process()
    started = time.perf_counter()
    cpu_started = time.process_time()
    config_hash = sha256(config_path)
    runner_hash = sha256(Path(__file__))
    replay_hash = sha256(Path(__file__).with_name("replay.py"))
    repository_commit = git_output("rev-parse", "HEAD")
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])

    def check_runaway() -> None:
        if (
            time.perf_counter() - started
            > config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]
        ):
            raise TimeoutError("EXP-629 reached its preregistered wall runaway ceiling")

    def new_model():
        model = cellular.SharedDilatedConvGRU(
            width=128,
            hidden=config["model"]["hidden"],
            cycles=config["model"]["cycles"],
        )
        model.load_state_dict(load_file(source))
        return model

    def transition_record(
        state: int,
        right: int,
        modulus: int,
        bit: int,
        target: int,
        quotient: int,
        position: int,
    ) -> dict[str, int]:
        return {
            "state": int(state),
            "right": int(right),
            "modulus": int(modulus),
            "bit": int(bit),
            "target": int(target),
            "quotient": int(quotient),
            "width": 128,
            "position": int(position),
        }

    mining_key = sha256_text(
        json.dumps(
            {
                "source": sha256(source),
                "split": split_hash,
                "mining": config["mining"],
                "runner": runner_hash,
                "replay": replay_hash,
            },
            sort_keys=True,
        )
    )
    dataset_path = output / f"replay-{mining_key}.json"
    mining_partial_path = output / f"replay-{mining_key}.partial.pt"

    def mine_replay() -> tuple[dict, dict]:
        mining_started = time.perf_counter()
        mining_cpu = time.process_time()
        if dataset_path.exists():
            dataset = json.loads(dataset_path.read_text())
            if dataset.get("cache_key") != mining_key:
                raise RuntimeError("replay cache key mismatch")
            return dataset, {
                "cached": True,
                "wall_seconds": 0.0,
                "aggregate_cpu_seconds": 0.0,
                "dataset_sha256": sha256(dataset_path),
                "trajectory_count": dataset["trajectory_count"],
                "first_error_count": len(dataset["hard"]),
            }

        sampler = exp300.BigIntSampler(splits[128]["train"], seed + 63028)
        completed = 0
        events: list[dict] = []
        cumulative_wall = 0.0
        cumulative_cpu = 0.0
        if mining_partial_path.exists():
            partial = torch.load(mining_partial_path, weights_only=False)
            if partial.get("cache_key") != mining_key:
                raise RuntimeError("mining partial cache key mismatch")
            completed = partial["completed"]
            events = partial["events"]
            cumulative_wall = partial["wall_seconds"]
            cumulative_cpu = partial["aggregate_cpu_seconds"]
            sampler.rng.setstate(partial["sampler_rng_state"])
        frozen = new_model().eval()
        segment_wall = time.perf_counter()
        segment_cpu = time.process_time()
        target_cases = config["mining"]["trajectories"]
        with torch.inference_mode():
            while completed < target_cases:
                size = min(config["mining"]["batch_size"], target_cases - completed)
                left, right, modulus = sampler.pairs(size)
                predicted = [0] * size
                teacher = [0] * size
                active = [True] * size
                for position in range(128):
                    bits = [(value >> (127 - position)) & 1 for value in left]
                    raw = [
                        2 * value + bit * other
                        for value, bit, other in zip(teacher, bits, right)
                    ]
                    targets = [value % prime for value, prime in zip(raw, modulus)]
                    quotients = [value // prime for value, prime in zip(raw, modulus)]
                    frozen.width = 128
                    logits, _ = frozen(
                        exp300.features(predicted, right, modulus, bits, 128)
                    )
                    guesses = decode(logits)
                    for row in range(size):
                        if active[row] and guesses[row] != targets[row]:
                            if predicted[row] != teacher[row]:
                                raise RuntimeError("first-error input is not teacher-exact")
                            hard = transition_record(
                                predicted[row],
                                right[row],
                                modulus[row],
                                bits[row],
                                targets[row],
                                quotients[row],
                                position,
                            )
                            control_position = (position + 64) % 128
                            control = replay.exact_transition_at(
                                left[row],
                                right[row],
                                modulus[row],
                                control_position,
                                128,
                            )
                            events.append({"hard": hard, "control": control})
                            active[row] = False
                    predicted = guesses
                    teacher = targets
                completed += size
                wall_total = cumulative_wall + time.perf_counter() - segment_wall
                cpu_total = cumulative_cpu + time.process_time() - segment_cpu
                torch.save(
                    {
                        "cache_key": mining_key,
                        "completed": completed,
                        "events": events,
                        "sampler_rng_state": sampler.rng.getstate(),
                        "wall_seconds": wall_total,
                        "aggregate_cpu_seconds": cpu_total,
                    },
                    mining_partial_path,
                )
                print(
                    json.dumps(
                        {
                            "phase": "mining",
                            "completed": completed,
                            "first_errors": len(events),
                            "wall_seconds": wall_total,
                            "aggregate_cpu_seconds": cpu_total,
                        }
                    ),
                    flush=True,
                )
                check_runaway()
        if len(events) < config["mining"]["minimum_first_errors"]:
            raise RuntimeError(
                f"mining yielded only {len(events)} first errors; minimum is "
                f"{config['mining']['minimum_first_errors']}"
            )
        dataset = {
            "cache_key": mining_key,
            "trajectory_count": completed,
            "hard": [event["hard"] for event in events],
            "control": [event["control"] for event in events],
        }
        dataset_path.write_text(json.dumps(dataset) + "\n")
        partial = torch.load(mining_partial_path, weights_only=False)
        runtime = {
            "cached": False,
            "wall_seconds": partial["wall_seconds"],
            "aggregate_cpu_seconds": partial["aggregate_cpu_seconds"],
            "dataset_sha256": sha256(dataset_path),
            "trajectory_count": completed,
            "first_error_count": len(events),
            "observed_yield": len(events) / completed,
            "current_invocation_wall_seconds": time.perf_counter() - mining_started,
            "current_invocation_aggregate_cpu_seconds": time.process_time() - mining_cpu,
        }
        mining_partial_path.unlink(missing_ok=True)
        return dataset, runtime

    def evaluate(model, width: int) -> dict:
        model.width = width
        return {
            "transition": exp300.evaluate_transitions(
                model,
                exp300.BigIntSampler(splits[width]["test"], seed + 63600 + width),
                config["evaluation"]["transition_cases"],
            ),
            "rollout": exp300.evaluate_rollouts(
                model,
                exp300.BigIntSampler(splits[width]["test"], seed + 63700 + width),
                config["evaluation"]["rollout_cases"],
            ),
        }

    def train_arm(name: str, replay_name: str, dataset: dict, dataset_hash: str) -> dict:
        arm_dir = output / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_key = sha256_text(
            json.dumps(
                {
                    "runner": runner_hash,
                    "config": config_hash,
                    "dataset": dataset_hash,
                    "arm": replay_name,
                },
                sort_keys=True,
            )
        )
        result_path = arm_dir / f"result-{arm_key}.json"
        checkpoint_path = arm_dir / "weights.safetensors"
        if result_path.exists() and checkpoint_path.exists():
            payload = json.loads(result_path.read_text())
            if payload.get("cache_key") == arm_key:
                return {**payload["result"], "cached": True}

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = new_model()
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
        sampler64 = exp300.BigIntSampler(splits[64]["train"], seed + 63164)
        sampler128 = exp300.BigIntSampler(splits[128]["train"], seed + 63128)
        replay_rng = random.Random(seed + 63200)
        partial_path = arm_dir / f"partial-{arm_key}.pt"
        completed = 0
        history: list[dict] = []
        cumulative_wall = 0.0
        cumulative_cpu = 0.0
        if partial_path.exists():
            partial = torch.load(partial_path, weights_only=False)
            if partial.get("cache_key") != arm_key:
                raise RuntimeError("arm partial cache key mismatch")
            completed = partial["step"]
            history = partial["history"]
            cumulative_wall = partial["wall_seconds"]
            cumulative_cpu = partial["aggregate_cpu_seconds"]
            model.load_state_dict(partial["model"])
            optimizer.load_state_dict(partial["optimizer"])
            scheduler.load_state_dict(partial["scheduler"])
            sampler64.rng.setstate(partial["sampler64_rng_state"])
            sampler128.rng.setstate(partial["sampler128_rng_state"])
            replay_rng.setstate(partial["replay_rng_state"])
        replay_pool = dataset[replay_name]
        model.train()
        segment_wall = time.perf_counter()
        segment_cpu = time.process_time()
        for step in range(completed + 1, config["training"]["steps"] + 1):
            width = (
                64
                if step % config["training"]["width64_every_steps"] == 0
                else 128
            )
            batch_size = config["training"]["batch_size"]
            if width == 64:
                values = sampler64.transitions(batch_size)
                states, rights, moduli, bits, targets, quotients = values
            else:
                replay_count = config["training"]["replay_cases_per_width128_batch"]
                uniform_count = batch_size - replay_count
                values = sampler128.transitions(uniform_count)
                states, rights, moduli, bits, targets, quotients = values
                rows = [
                    replay_pool[replay_rng.randrange(len(replay_pool))]
                    for _ in range(replay_count)
                ]
                states += [row["state"] for row in rows]
                rights += [row["right"] for row in rows]
                moduli += [row["modulus"] for row in rows]
                bits += [row["bit"] for row in rows]
                targets += [row["target"] for row in rows]
                quotients = torch.cat(
                    (quotients, torch.tensor([row["quotient"] for row in rows]))
                )
            model.width = width
            logits, quotient_logits = model(
                exp300.features(states, rights, moduli, bits, width)
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
                    model.parameters(), config["training"]["gradient_clip"]
                )
            )
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 25 == 0 or step == config["training"]["steps"]:
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
                        f"{name}_loss": stats["loss"],
                        f"{name}_batch_exact": stats["batch_exact"],
                        f"{name}_gradient_norm": gradient,
                    },
                    step=step,
                )
                print(json.dumps({"arm": name, **item}), flush=True)
            if (
                step % config["training"]["save_every_steps"] == 0
                or step == config["training"]["steps"]
            ):
                wall_total = cumulative_wall + time.perf_counter() - segment_wall
                cpu_total = cumulative_cpu + time.process_time() - segment_cpu
                torch.save(
                    {
                        "cache_key": arm_key,
                        "step": step,
                        "history": history,
                        "wall_seconds": wall_total,
                        "aggregate_cpu_seconds": cpu_total,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "sampler64_rng_state": sampler64.rng.getstate(),
                        "sampler128_rng_state": sampler128.rng.getstate(),
                        "replay_rng_state": replay_rng.getstate(),
                    },
                    partial_path,
                )
                check_runaway()
        partial = torch.load(partial_path, weights_only=False)
        model.eval()
        save_file(
            {key: value.detach().contiguous() for key, value in model.state_dict().items()},
            checkpoint_path,
        )
        evaluation_started = time.perf_counter()
        evaluation_cpu = time.process_time()
        evaluation = {str(width): evaluate(model, width) for width in widths}
        result = {
            "arm": name,
            "cached": False,
            "parameters": model.parameter_count(),
            "training": {"steps": config["training"]["steps"], "history": history},
            "evaluation": evaluation,
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
            "runtime": {
                "training_wall_seconds": partial["wall_seconds"],
                "training_aggregate_cpu_seconds": partial["aggregate_cpu_seconds"],
                "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
                "evaluation_aggregate_cpu_seconds": time.process_time() - evaluation_cpu,
                "peak_rss_bytes": process.memory_info().rss,
            },
        }
        result_path.write_text(
            json.dumps({"cache_key": arm_key, "result": result}, indent=2) + "\n"
        )
        partial_path.unlink(missing_ok=True)
        return result

    results: dict[str, dict] = {}
    arm_run_ids: dict[str, str] = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "status": "running",
                "role": "tier6-balanced-first-error-replay",
                "repository_commit": repository_commit,
                "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(),
            }
        )
        mlflow.log_params(
            {
                "seed": seed,
                "steps": config["training"]["steps"],
                "source_checkpoint_sha256": sha256(source),
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "split_sha256": split_hash,
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(
            str(ROOT / config["preregistration"]), artifact_path="preregistration"
        )
        dataset, mining = mine_replay()
        for name, replay_name in (
            ("uniform_replay_control", "control"),
            ("first_error_replay", "hard"),
        ):
            with mlflow.start_run(run_name=name, nested=True) as nested:
                arm_run_ids[name] = nested.info.run_id
                results[name] = train_arm(
                    name, replay_name, dataset, mining["dataset_sha256"]
                )
                evaluation = results[name]["evaluation"]
                mlflow.log_metrics(
                    {
                        "width64_rollout": evaluation["64"]["rollout"][
                            "exact_accuracy"
                        ],
                        "width128_transition": evaluation["128"]["transition"][
                            "exact_accuracy"
                        ],
                        "width128_rollout": evaluation["128"]["rollout"][
                            "exact_accuracy"
                        ],
                    }
                )
        control = results["uniform_replay_control"]["evaluation"]
        challenger = results["first_error_replay"]["evaluation"]
        gain = (
            challenger["128"]["rollout"]["exact_accuracy"]
            - control["128"]["rollout"]["exact_accuracy"]
        )
        gates = config["gates"]
        passed = (
            challenger["128"]["rollout"]["exact_accuracy"]
            >= gates["minimum_challenger_width128_rollout"]
            and gain >= gates["minimum_width128_gain_over_control"]
            and challenger["64"]["rollout"]["exact_accuracy"]
            >= gates["minimum_challenger_width64_rollout"]
            and challenger["128"]["transition"]["exact_accuracy"]
            >= gates["minimum_challenger_width128_transition"]
            and challenger["128"]["rollout"]["any_invalid_residue_rate"]
            <= gates["maximum_challenger_width128_invalid_rollout_rate"]
        )
        decision = (
            "advance-tier6-balanced-first-error-replay"
            if passed
            else "reject-tier6-balanced-first-error-replay"
        )
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": config["tracking"]["experiment_name"],
                "experiment_id": experiment_id,
                "parent_run_id": parent_run_id,
                "arm_run_ids": arm_run_ids,
            },
            "source": {
                "repository_commit": repository_commit,
                "dirty": bool(git_output("status", "--porcelain")),
                "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "replay_sha256": replay_hash,
                "split_sha256": split_hash,
                "source_checkpoint_sha256": sha256(source),
            },
            "mining": mining,
            "arms": results,
            "comparison": {
                "control_width128_rollout": control["128"]["rollout"][
                    "exact_accuracy"
                ],
                "challenger_width128_rollout": challenger["128"]["rollout"][
                    "exact_accuracy"
                ],
                "width128_rollout_gain": gain,
            },
            "gates": {**gates, "passed": passed},
            "decision": decision,
            "resource_budget": config["resource_budget"],
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "aggregate_cpu_seconds": time.process_time() - cpu_started,
                "torch_threads": config["training"]["torch_threads"],
                "peak_rss_bytes": process.memory_info().rss,
            },
            "limitations": [
                "Fresh generated selection evidence only; checked, consumed independent, and private answers were not read.",
                "Exact bigint arithmetic is confined to offline generation, labels, mining, tests, and evaluation truth.",
                "The paired replay records match trajectories but intentionally differ in Horner position and difficulty.",
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics(
            {
                "control_width128_rollout": summary["comparison"][
                    "control_width128_rollout"
                ],
                "challenger_width128_rollout": summary["comparison"][
                    "challenger_width128_rollout"
                ],
                "width128_rollout_gain": gain,
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
                "comparison": summary["comparison"],
                "mining": mining,
                "arms": {
                    name: {
                        "evaluation": arm["evaluation"],
                        "checkpoint": arm["checkpoint"],
                        "runtime": arm["runtime"],
                    }
                    for name, arm in results.items()
                },
                "runtime": summary["runtime"],
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
