"""Matched uniform replay versus online exact-failure mining at width 32."""

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

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from torch import nn


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
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def target_bits(target: torch.Tensor, width: int) -> torch.Tensor:
    shifts = torch.arange(width, dtype=torch.int64)
    return ((target[:, None] >> shifts[None, :]) & 1).float()


@torch.inference_mode()
def rank_pool(model, values: tuple[torch.Tensor, ...], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    state, multiplicand, modulus, bit, target, _quotient = values
    logits, _ = model(exp100_model.make_features(state, multiplicand, modulus, bit, width))
    bit_loss = F.binary_cross_entropy_with_logits(logits, target_bits(target, width), reduction="none").mean(1)
    predicted = ((logits >= 0).to(torch.int64) * (1 << torch.arange(width, dtype=torch.int64))[None, :]).sum(1)
    failures = predicted.ne(target)
    score = bit_loss + failures.float() * 100.0
    return score, failures


def select_indices(score: torch.Tensor, failures: torch.Tensor, size: int, hard_fraction: float, online: bool, generator: torch.Generator) -> torch.Tensor:
    order = torch.randperm(len(score), generator=generator)
    if not online:
        return order[:size]
    hard_count = round(size * hard_fraction)
    ranked = torch.argsort(score, descending=True, stable=True)
    hard = ranked[:hard_count]
    available = torch.ones(len(score), dtype=torch.bool)
    available[hard] = False
    replay_pool = torch.nonzero(available, as_tuple=False).flatten()
    replay = replay_pool[torch.randperm(len(replay_pool), generator=generator)[: size - hard_count]]
    selected = torch.cat((hard, replay))
    return selected[torch.randperm(len(selected), generator=generator)]


def train_arm(name: str, online: bool, config: dict, source: Path, split: dict, exp200, output: Path) -> dict:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    width = int(config["model"]["width"])
    model = exp100_model.SharedContextHornerTCN(**config["model"])
    model.load_state_dict(load_file(source))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["training"]["steps"]), eta_min=float(config["training"]["minimum_learning_rate"]))
    sampler = exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["train"], seed + 2501, 0.0, 0.15)
    selector = torch.Generator().manual_seed(seed + 2502)
    batch_size = int(config["training"]["batch_size"])
    pool_size = batch_size * int(config["training"]["candidate_pool_factor"])
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    selected_failures = 0
    pool_failures = 0
    selected_total = 0
    history = []
    started = time.perf_counter()
    model.train()
    for step in range(1, int(config["training"]["steps"]) + 1):
        pool = sampler.transitions(pool_size)
        model.eval()
        score, failures = rank_pool(model, pool, width)
        indices = select_indices(score, failures, batch_size, float(config["training"]["hard_fraction"]), online, selector)
        model.train()
        state, multiplicand, modulus, bit, target, quotient = (value[indices] for value in pool)
        logits, quotient_logits = model(exp100_model.make_features(state, multiplicand, modulus, bit, width))
        loss, stats = exp100.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        selected_failures += int(failures[indices].sum())
        pool_failures += int(failures.sum())
        selected_total += len(indices)
        if step == 1 or step % 50 == 0 or step == int(config["training"]["steps"]):
            peak_rss = max(peak_rss, process.memory_info().rss)
            record = {"step": step, "elapsed_seconds": time.perf_counter() - started, "loss": stats["loss"], "batch_exact": stats["batch_exact"], "gradient_norm": gradient_norm, "selected_failure_fraction": selected_failures / selected_total, "pool_failure_fraction": pool_failures / (step * pool_size)}
            history.append(record)
            mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "selected_failure_fraction": record["selected_failure_fraction"], "pool_failure_fraction": record["pool_failure_fraction"], "gradient_norm": gradient_norm}, step=step)
            print(json.dumps({"arm": name, **record}), flush=True)
    training_seconds = time.perf_counter() - started
    test_seed = seed + 75000
    model.eval()
    transition = exp100.evaluate_transitions(model, exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], test_seed, 0.0, 0.15), config["evaluation"]["transition_cases"])
    rollout = exp100.evaluate_rollouts(model, exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], test_seed + 1, 0.0, 0.0), config["evaluation"]["rollout_cases"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "weights.safetensors"
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint)
    return {"arm": name, "online_failure_mining": online, "parameters": model.parameter_count(), "training": {"steps": config["training"]["steps"], "seconds": training_seconds, "peak_rss_bytes": peak_rss, "selected_failure_fraction": selected_failures / selected_total, "pool_failure_fraction": pool_failures / (config["training"]["steps"] * pool_size), "history": history}, "evaluation": {"heldout_transition": transition, "heldout_rollout": rollout}, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_250_online_failure_mining.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-250-online-failure-mining")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")
    exp200 = load_module("exp200_for_exp250", ROOT / "experiments/exp_200_width32_transfer_ab/run.py")
    exp204 = load_module("exp204_for_exp250", ROOT / "experiments/exp_204_official_distribution_finetune/run.py")
    split = exp204.official_numeric_prime_split(config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], config["seed"] + 250)
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repository_commit = git_output("rev-parse", "HEAD")
    repository_dirty = bool(git_output("status", "--porcelain"))
    started = time.perf_counter()
    results = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "online-failure-mining-matched-ab", "repository_commit": repository_commit, "repository_dirty": str(repository_dirty).lower(), "challenge_commit": exp100.CHALLENGE_COMMIT})
        mlflow.log_params({"seed": config["seed"], "steps": config["training"]["steps"], "batch_size": config["training"]["batch_size"], "candidate_pool_factor": config["training"]["candidate_pool_factor"], "source_checkpoint_sha256": sha256(source), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "python": platform.python_version(), "pytorch": torch.__version__})
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-250-online-failure-mining.md"), artifact_path="preregistration")
        run_ids = {}
        for name, online in (("uniform-replay-control", False), ("online-failure-mining-challenger", True)):
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id
                mlflow.set_tags({"canonical_run_id": config["run_id"], "parent_run_id": parent_id, "ab_group": name, "status": "running"})
                results[name] = train_arm(name, online, config, source, split, exp200, output / name)
                mlflow.log_metrics({"transition_exact": results[name]["evaluation"]["heldout_transition"]["exact_accuracy"], "rollout_exact": results[name]["evaluation"]["heldout_rollout"]["exact_accuracy"], "invalid_rollout": results[name]["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]})
                mlflow.set_tag("status", "completed")
        control = results["uniform-replay-control"]
        challenger = results["online-failure-mining-challenger"]
        ct = control["evaluation"]["heldout_transition"]["exact_accuracy"]
        cr = control["evaluation"]["heldout_rollout"]["exact_accuracy"]
        tt = challenger["evaluation"]["heldout_transition"]["exact_accuracy"]
        tr = challenger["evaluation"]["heldout_rollout"]["exact_accuracy"]
        invalid = challenger["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]
        comparison = {"control_transition": ct, "control_rollout": cr, "challenger_transition": tt, "challenger_rollout": tr, "transition_gain": tt - ct, "rollout_gain": tr - cr, "challenger_invalid_rollout": invalid}
        gates = config["gates"]
        passed = comparison["transition_gain"] >= gates["minimum_transition_gain"] and comparison["rollout_gain"] >= gates["minimum_rollout_gain"] and tt >= gates["minimum_challenger_transition"] and tr >= gates["minimum_challenger_rollout"] and invalid <= gates["maximum_invalid_rollout_rate"]
        decision = "advance-online-mining-independent-tier4" if passed else "reject-online-failure-mining"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids}, "source": {"repository_commit": repository_commit, "dirty": repository_dirty, "challenge_commit": exp100.CHALLENGE_COMMIT, "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "source_checkpoint_sha256": sha256(source)}, "arms": results, "comparison": comparison, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Selection uses model predictions and exact offline labels only during training.", "Matched fresh official-like identities were used; no consumed independent seed or checked answer was read."]}
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({**comparison, "wall_seconds": summary["resource"]["wall_seconds"]})
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": comparison, "mining": {name: value["training"] for name, value in results.items()}, "checkpoints": {name: value["checkpoint"] for name, value in results.items()}, "mlflow": summary["mlflow"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
