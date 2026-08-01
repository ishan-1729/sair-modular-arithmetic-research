"""Fine-tune EXP-202 on an official-like numeric prime distribution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from safetensors.torch import load_file
from sympy import nextprime


ROOT = Path(__file__).resolve().parents[2]
EXP100_DIR = ROOT / "experiments/exp_100_width_scaling"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP100_DIR))

import model as exp100_model  # noqa: E402
import run as exp100  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def official_numeric_prime_split(minimum_bits: int, maximum_bits: int, seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    low = 1 << minimum_bits
    high = 1 << maximum_bits
    counts = {"train": 2048, "validation": 512, "test": 512}
    seen: set[int] = set()
    values: list[int] = []
    while len(values) < sum(counts.values()):
        prime = int(nextprime(rng.randrange(low, high)))
        if prime < high and prime not in seen:
            seen.add(prime)
            values.append(prime)
    splits = {"train": [], "validation": [], "test": []}
    cursor = 0
    for name, count in counts.items():
        splits[name] = values[cursor : cursor + count]
        cursor += count
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_204_official_distribution_finetune.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-204-official-distribution-finetune")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source_path) != config["source_checkpoint_sha256"]:
        raise RuntimeError("frozen EXP-202 checkpoint hash mismatch")
    exp200 = load_module("exp200_for_exp204", ROOT / "experiments/exp_200_width32_transfer_ab/run.py")
    matched = exp200.matched_config(config)
    arm = next(item for item in matched["comparison"]["arms"] if item["name"] == "width32-transfer-hard-mining")
    arm = {**arm, "name": "width32-official-distribution-finetune"}
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original_constructor = exp100.SharedContextHornerTCN
    original_sampler = exp100.TransitionSampler
    original_split = exp100.split_primes
    active_sampler = None

    def loaded_constructor(width: int, hidden: int, blocks: int) -> exp100_model.SharedContextHornerTCN:
        model = original_constructor(width=width, hidden=hidden, blocks=blocks)
        model.load_state_dict(load_file(source_path))
        return model

    def sampler_factory(width: int, minimum_bits: int, maximum_bits: int, primes: list[int], seed: int, composite_fraction: float, random_state_fraction: float):
        nonlocal active_sampler
        if composite_fraction > 0:
            active_sampler = exp200.ScalableHardMiningSampler(width, minimum_bits, maximum_bits, primes, seed, composite_fraction, random_state_fraction, pool_factor=int(config["hard_mining"]["candidate_pool_factor"]), quotient2_fraction=float(config["hard_mining"]["quotient2_fraction"]), boundary_fraction=float(config["hard_mining"]["boundary_fraction"]), boundary_relative_margin=float(config["hard_mining"]["boundary_relative_margin"]))
            return active_sampler
        return exp200.ScalableTransitionSampler(width, minimum_bits, maximum_bits, primes, seed, composite_fraction, random_state_fraction)

    git_commit = git_output("rev-parse", "HEAD")
    git_dirty = bool(git_output("status", "--porcelain"))
    started = time.perf_counter()
    exp100.SharedContextHornerTCN = loaded_constructor
    exp100.TransitionSampler = sampler_factory
    exp100.split_primes = official_numeric_prime_split
    try:
        with mlflow.start_run(run_name=config["run_id"]) as parent:
            parent_run_id = parent.info.run_id
            mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "tier4-distribution-correction", "baseline_run": "exp-202-width32-2400", "ab_group": "EXP-204", "promotion_gate": "preregistered-official-distribution-paired", "repository_commit": git_commit, "repository_dirty": str(git_dirty).lower(), "challenge_commit": exp100.CHALLENGE_COMMIT})
            mlflow.log_params({"seed": config["seed"], "steps": config["training"]["steps"], "batch_size": config["training"]["batch_size"], "source_checkpoint_sha256": sha256(source_path), "prime_distribution": config["evaluation"]["prime_distribution"], "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "python": platform.python_version(), "pytorch": torch.__version__})
            mlflow.log_artifact(str(config_path), artifact_path="preregistration")
            mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-204-official-distribution-finetune.md"), artifact_path="preregistration")
            with mlflow.start_run(run_name=arm["name"], nested=True) as nested:
                arm_run_id = nested.info.run_id
                mlflow.set_tags({"canonical_run_id": config["run_id"], "parent_run_id": parent_run_id, "ab_group": arm["name"], "initialization": "exact-exp202", "status": "running"})
                result = exp100.train_arm(arm, matched, output / arm["name"])
                mlflow.set_tag("status", "completed")
            if active_sampler is None:
                raise RuntimeError("hard-mining sampler was not constructed")
            result["hard_mining"] = active_sampler.mining_stats()
            baseline_transition = result["controls"]["untrained_transition"]["exact_accuracy"]
            baseline_rollout = result["controls"]["untrained_rollout"]["exact_accuracy"]
            transition = result["evaluation"]["heldout_transition"]["exact_accuracy"]
            rollout = result["evaluation"]["heldout_rollout"]["exact_accuracy"]
            invalid = result["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]
            comparison = {"baseline_transition": baseline_transition, "baseline_rollout": baseline_rollout, "post_transition": transition, "post_rollout": rollout, "transition_gain": transition - baseline_transition, "rollout_gain": rollout - baseline_rollout, "post_invalid_rollout": invalid}
            gates = config["gates"]
            passed = transition >= gates["minimum_transition_accuracy"] and rollout >= gates["minimum_rollout_accuracy"] and comparison["rollout_gain"] >= gates["minimum_rollout_gain"] and invalid <= gates["maximum_invalid_rollout_rate"]
            decision = "advance-fresh-independent-tier4-seeds" if passed else "reject-simple-distribution-correction"
            summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_run_id, "arm_run_id": arm_run_id}, "source": {"repository_commit": git_commit, "dirty": git_dirty, "challenge_commit": exp100.CHALLENGE_COMMIT, "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "source_checkpoint_sha256": sha256(source_path)}, "result": result, "comparison": comparison, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Official-like heldout identities are synthetic and separate from consumed EXP-203 seeds.", "No checked public expected answers were read or evaluated."]}
            summary_path = output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            mlflow.log_metrics({**comparison, "wall_seconds": summary["resource"]["wall_seconds"]})
            mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
            mlflow.set_tags({"status": "completed", "decision": decision})
    finally:
        exp100.SharedContextHornerTCN = original_constructor
        exp100.TransitionSampler = original_sampler
        exp100.split_primes = original_split
    print(json.dumps({"decision": decision, "comparison": comparison, "checkpoint": result["checkpoint"], "mlflow": summary["mlflow"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
