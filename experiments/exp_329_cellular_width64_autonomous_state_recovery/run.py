"""Matched teacher-forced versus autonomous-state recovery training at width 64."""

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
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def decode_logits(logits: torch.Tensor, width: int) -> list[int]:
    return [sum(int(bit) << position for position, bit in enumerate(row)) for row in logits.ge(0).tolist()]


def teacher_states_before(multiplier: list[int], multiplicand: list[int], modulus: list[int], starts: list[int], width: int) -> list[int]:
    states: list[int] = []
    for left, right, prime, start in zip(multiplier, multiplicand, modulus, starts):
        state = 0
        for position in range(start):
            bit = (left >> (width - 1 - position)) & 1
            state = (2 * state + bit * right) % prime
        states.append(state)
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_329_cellular_width64_autonomous_state_recovery.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-329-cellular-width64-autonomous-state-recovery")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")

    exp300 = load_module("exp300_for_exp329", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    exp320 = load_module("exp320_for_exp329", ROOT / "experiments/exp_320_cellular_width64_curriculum/run.py")
    cellular = load_module("cellular_for_exp329", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    seed = int(config["seed"])

    def make_split(width: int) -> dict:
        minimum = 53 if width == 56 else 33
        local = {"data": {"prime_bits_min": minimum, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
        return exp300.bigint_prime_split(local, seed + 32900 + width)

    splits = {width: make_split(width) for width in (56, 64)}
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner_hash = sha256(Path(__file__))
    config_hash = sha256(config_path)
    dataset_hash = sha256_text(json.dumps(splits, sort_keys=True))
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    commit = git_output("rev-parse", "HEAD")
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    process = psutil.Process()

    def evaluate(model, width: int) -> dict:
        model.width = width
        return {
            "transition": exp300.evaluate_transitions(model, exp300.BigIntSampler(splits[width]["test"], seed + 33500 + width), config["evaluation"][f"width{width}_transition_cases"]),
            "rollout": exp300.evaluate_rollouts(model, exp300.BigIntSampler(splits[width]["test"], seed + 33600 + width), config["evaluation"][f"width{width}_rollout_cases"]),
        }

    def train_arm(name: str, recovery: bool) -> dict:
        arm_dir = output / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        cache_key = sha256_text(json.dumps({"arm": name, "source": sha256(source), "runner": runner_hash, "config": config_hash, "dataset": dataset_hash}, sort_keys=True))
        cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == cache_key and (ROOT / payload["result"]["checkpoint"]["path"]).exists():
                return {**payload["result"], "cached": True}

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = cellular.SharedDilatedConvGRU(width=64, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"])
        model.load_state_dict(load_file(source))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"])
        samplers = {width: exp300.BigIntSampler(splits[width]["train"], seed + 33100 + width) for width in (56, 64)}
        recovery_sampler = exp300.BigIntSampler(splits[64]["train"], seed + 33264)
        recovery_rng = random.Random(seed + 33364)
        selector = torch.Generator().manual_seed(seed + 3292)
        pool_size = config["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"]
        history = []
        arm_wall = time.perf_counter()
        arm_cpu = time.process_time()
        peak_rss = process.memory_info().rss
        model.train()
        for step in range(1, config["training"]["steps"] + 1):
            use_recovery = recovery and step % config["training"]["recovery_update_period"] == 0
            if not use_recovery:
                width = 56 if step % 2 else 64
                state, multiplicand, modulus, input_bit, target, quotient = exp320.mine(samplers[width].transitions(pool_size), config["training"]["batch_size"], config, selector)
                model.width = width
                logits, quotient_logits = model(exp300.features(state, multiplicand, modulus, input_bit, width))
                loss, stats = exp300.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"])
            else:
                width = 64
                batch = config["training"]["recovery_batch_size"]
                chunk = config["training"]["recovery_chunk_steps"]
                multiplier, multiplicand, modulus = recovery_sampler.pairs(batch)
                starts = [recovery_rng.randrange(0, width - chunk + 1) for _ in range(batch)]
                teacher = teacher_states_before(multiplier, multiplicand, modulus, starts, width)
                exposed = list(teacher)
                losses = []
                exact = []
                probability = config["training"]["scheduled_prediction_probability_start"] + (config["training"]["scheduled_prediction_probability_end"] - config["training"]["scheduled_prediction_probability_start"]) * (step - 1) / max(1, config["training"]["steps"] - 1)
                model.width = width
                for offset in range(chunk):
                    positions = [start + offset for start in starts]
                    input_bit = [(left >> (width - 1 - position)) & 1 for left, position in zip(multiplier, positions)]
                    raw = [2 * value + bit * right for value, bit, right in zip(teacher, input_bit, multiplicand)]
                    target = [value % prime for value, prime in zip(raw, modulus)]
                    quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long)
                    logits, quotient_logits = model(exp300.features(exposed, multiplicand, modulus, input_bit, width))
                    part, part_stats = exp300.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["recovery_quotient_weight"])
                    losses.append(part)
                    exact.append(part_stats["batch_exact"])
                    predicted = decode_logits(logits.detach(), width)
                    exposed = [guess if recovery_rng.random() < probability else truth for guess, truth in zip(predicted, target)]
                    teacher = target
                loss = torch.stack(losses).mean()
                stats = {"loss": float(loss.detach()), "batch_exact": sum(exact) / len(exact), "mean_bce": float("nan"), "worst_bit_bce": float("nan"), "quotient_ce": 0.0}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 100 == 0 or step == config["training"]["steps"]:
                peak_rss = max(peak_rss, process.memory_info().rss)
                record = {"step": step, "width": width, "mode": "recovery" if use_recovery else "transition", "elapsed_seconds": time.perf_counter() - arm_wall, "gradient_norm": gradient, **stats}
                history.append(record)
                mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient}, step=step)
                print(json.dumps({"arm": name, **record}), flush=True)
        model.eval()
        checkpoint = arm_dir / "weights.safetensors"
        save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint)
        result = {
            "arm": name,
            "recovery": recovery,
            "cached": False,
            "parameters": model.parameter_count(),
            "training": {"steps": config["training"]["steps"], "transition_equivalent_examples": config["training"]["steps"] * config["training"]["batch_size"], "history": history},
            "evaluation": {"width56": evaluate(model, 56), "width64": evaluate(model, 64)},
            "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
            "runtime": {"wall_seconds": time.perf_counter() - arm_wall, "aggregate_cpu_seconds": time.process_time() - arm_cpu, "peak_rss_bytes": peak_rss},
        }
        cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n")
        return result

    results = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "cellular-width64-autonomous-state-recovery-ab", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()})
        mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "source_checkpoint_sha256": sha256(source), "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash})
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        run_ids = {}
        for name, recovery in (("teacher-forced-control", False), ("state-recovery-challenger", True)):
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id
                results[name] = train_arm(name, recovery)
                arm_eval = results[name]["evaluation"]
                mlflow.log_metrics({"width56_rollout": arm_eval["width56"]["rollout"]["exact_accuracy"], "width64_transition": arm_eval["width64"]["transition"]["exact_accuracy"], "width64_rollout": arm_eval["width64"]["rollout"]["exact_accuracy"], "width64_invalid": arm_eval["width64"]["rollout"]["any_invalid_residue_rate"]})
                mlflow.set_tag("status", "completed")
        control = results["teacher-forced-control"]["evaluation"]
        challenger = results["state-recovery-challenger"]["evaluation"]
        comparison = {
            "control_width64_transition": control["width64"]["transition"]["exact_accuracy"],
            "control_width64_rollout": control["width64"]["rollout"]["exact_accuracy"],
            "challenger_width64_transition": challenger["width64"]["transition"]["exact_accuracy"],
            "challenger_width64_rollout": challenger["width64"]["rollout"]["exact_accuracy"],
            "width64_transition_change": challenger["width64"]["transition"]["exact_accuracy"] - control["width64"]["transition"]["exact_accuracy"],
            "width64_rollout_gain": challenger["width64"]["rollout"]["exact_accuracy"] - control["width64"]["rollout"]["exact_accuracy"],
            "challenger_width64_invalid": challenger["width64"]["rollout"]["any_invalid_residue_rate"],
            "challenger_width56_rollout": challenger["width56"]["rollout"]["exact_accuracy"],
        }
        gates = config["gates"]
        passed = comparison["challenger_width64_transition"] >= gates["minimum_challenger_width64_transition"] and comparison["challenger_width64_rollout"] >= gates["minimum_challenger_width64_rollout"] and comparison["width64_rollout_gain"] >= gates["minimum_width64_rollout_gain_over_control"] and comparison["width64_transition_change"] >= -gates["maximum_width64_transition_regression"] and comparison["challenger_width64_invalid"] <= gates["maximum_challenger_invalid_rollout_rate"] and comparison["challenger_width56_rollout"] >= gates["minimum_retained_width56_rollout"]
        decision = "advance-state-recovery-independent-tier5" if passed else "reject-cellular-autonomous-state-recovery"
        summary = {
            "run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"],
            "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids},
            "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash, "source_checkpoint_sha256": sha256(source)},
            "arms": results, "comparison": comparison, "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"],
            "runtime": {"wall_seconds": time.perf_counter() - wall_started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "torch_threads": config["training"]["torch_threads"]},
            "limitations": ["Generated identity-disjoint selection evidence only; checked, consumed independent, and private answers were not read.", "Recovery predictions are detached; gradients train recovery targets rather than backpropagating through autonomous state transitions.", "The challenger replaces one quarter of hard-mined updates, so trajectory exposure and training-distribution mixture change together."],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({**comparison, "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]})
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": comparison, "checkpoints": {name: value["checkpoint"] for name, value in results.items()}, "mlflow": summary["mlflow"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
