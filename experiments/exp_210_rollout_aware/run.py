"""Matched one-step versus scheduled learned-state training at width 32."""

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


def decode_logits(logits: torch.Tensor, width: int) -> torch.Tensor:
    weights = 1 << torch.arange(width, dtype=torch.int64)
    return ((logits >= 0).to(torch.int64) * weights[None, :]).sum(1)


def teacher_state_before(multiplier: torch.Tensor, multiplicand: torch.Tensor, modulus: torch.Tensor, starts: torch.Tensor, width: int) -> torch.Tensor:
    state = torch.zeros(len(multiplier), dtype=torch.int64)
    for position in range(width):
        bit = (multiplier >> (width - 1 - position)) & 1
        next_state = (2 * state + bit * multiplicand) % modulus
        state = torch.where(position < starts, next_state, state)
    return state


def train_arm(name: str, rollout_aware: bool, config: dict, source: Path, split: dict, exp200, output: Path) -> dict:
    seed = int(config["seed"])
    width = int(config["model"]["width"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = exp100_model.SharedContextHornerTCN(**config["model"])
    model.load_state_dict(load_file(source))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["training"]["steps"]), eta_min=float(config["training"]["minimum_learning_rate"]))
    hard = exp200.ScalableHardMiningSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["train"], seed + 1, 0.15, 0.15, pool_factor=config["hard_mining"]["candidate_pool_factor"], quotient2_fraction=config["hard_mining"]["quotient2_fraction"], boundary_fraction=config["hard_mining"]["boundary_fraction"], boundary_relative_margin=config["hard_mining"]["boundary_relative_margin"])
    trajectory = exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["train"], seed + 2, 0.0, 0.0)
    history = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    model.train()
    total_steps = int(config["training"]["steps"])
    rollout_period = round(1 / float(config["training"]["rollout_step_fraction"]))
    for step in range(1, total_steps + 1):
        use_rollout = rollout_aware and step % rollout_period == 0
        if not use_rollout:
            state, multiplicand, modulus, bit, target, quotient = hard.transitions(int(config["training"]["transition_batch_size"]))
            logits, quotient_logits = model(exp100_model.make_features(state, multiplicand, modulus, bit, width))
            loss, stats = exp100.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"])
        else:
            batch = int(config["training"]["rollout_batch_size"])
            chunk = int(config["training"]["rollout_chunk_steps"])
            multiplier, multiplicand, modulus = trajectory.pairs(batch, composites=False)
            starts = torch.randint(0, width - chunk + 1, (batch,), generator=trajectory.generator)
            teacher = teacher_state_before(multiplier, multiplicand, modulus, starts, width)
            exposed = teacher.clone()
            losses = []
            exact = []
            probability = config["training"]["scheduled_prediction_probability_start"] + (config["training"]["scheduled_prediction_probability_end"] - config["training"]["scheduled_prediction_probability_start"]) * (step - 1) / max(1, total_steps - 1)
            for offset in range(chunk):
                positions = starts + offset
                bit = (multiplier >> (width - 1 - positions)) & 1
                target = (2 * teacher + bit * multiplicand) % modulus
                quotient = (2 * teacher + bit * multiplicand) // modulus
                logits, quotient_logits = model(exp100_model.make_features(exposed, multiplicand, modulus, bit, width))
                part, part_stats = exp100.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], 0.0)
                losses.append(part)
                exact.append(part_stats["batch_exact"])
                predicted = decode_logits(logits, width).detach()
                use_predicted = torch.rand(batch, generator=trajectory.generator) < probability
                exposed = torch.where(use_predicted, predicted, target)
                teacher = target
            loss = torch.stack(losses).mean()
            stats = {"loss": float(loss.detach()), "mean_bce": float("nan"), "worst_bit_bce": float("nan"), "quotient_ce": 0.0, "batch_exact": sum(exact) / len(exact)}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == total_steps:
            peak_rss = max(peak_rss, process.memory_info().rss)
            record = {"step": step, "elapsed_seconds": time.perf_counter() - started, "learning_rate": scheduler.get_last_lr()[0], "gradient_norm": gradient_norm, "mode": "rollout" if use_rollout else "transition", **stats}
            history.append(record)
            mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient_norm, "learning_rate": scheduler.get_last_lr()[0]}, step=step)
            print(json.dumps({"arm": name, **record}), flush=True)
    training_seconds = time.perf_counter() - started
    test_seed = seed + 50000
    transition = exp100.evaluate_transitions(model, exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], test_seed, 0.0, 0.15), config["evaluation"]["transition_cases"])
    rollout = exp100.evaluate_rollouts(model, exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], test_seed + 1, 0.0, 0.0), config["evaluation"]["rollout_cases"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "weights.safetensors"
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, checkpoint)
    return {"arm": name, "rollout_aware": rollout_aware, "parameters": model.parameter_count(), "training": {"steps": total_steps, "transition_equivalent_examples": total_steps * config["training"]["transition_batch_size"], "seconds": training_seconds, "peak_rss_bytes": peak_rss, "history": history}, "evaluation": {"heldout_transition": transition, "heldout_rollout": rollout}, "hard_mining": hard.mining_stats(), "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_210_rollout_aware_ab.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-210-rollout-aware")
    args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text())
    source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source hash mismatch")
    exp200 = load_module("exp200_for_exp210", ROOT / "experiments/exp_200_width32_transfer_ab/run.py")
    exp204 = load_module("exp204_for_exp210", ROOT / "experiments/exp_204_official_distribution_finetune/run.py")
    split = exp204.official_numeric_prime_split(config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], config["seed"] + 32)
    torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    git_commit = git_output("rev-parse", "HEAD"); git_dirty = bool(git_output("status", "--porcelain")); started = time.perf_counter(); results = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "rollout-aware-matched-ab", "ab_group": "EXP-210", "promotion_gate": "preregistered-scheduled-learned-state", "repository_commit": git_commit, "repository_dirty": str(git_dirty).lower(), "challenge_commit": exp100.CHALLENGE_COMMIT})
        mlflow.log_params({"seed": config["seed"], "steps": config["training"]["steps"], "transition_equivalent_examples": config["training"]["steps"] * config["training"]["transition_batch_size"], "source_checkpoint_sha256": sha256(source), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "python": platform.python_version(), "pytorch": torch.__version__})
        mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-210-rollout-aware-width32.md"), artifact_path="preregistration")
        run_ids = {}
        for name, aware in (("one-step-control", False), ("scheduled-state-challenger", True)):
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "parent_run_id": parent_id, "ab_group": name, "status": "running"})
                results[name] = train_arm(name, aware, config, source, split, exp200, output / name)
                mlflow.log_metrics({"transition_exact": results[name]["evaluation"]["heldout_transition"]["exact_accuracy"], "rollout_exact": results[name]["evaluation"]["heldout_rollout"]["exact_accuracy"], "invalid_rollout": results[name]["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]}); mlflow.set_tag("status", "completed")
        control = results["one-step-control"]; challenger = results["scheduled-state-challenger"]
        ct = control["evaluation"]["heldout_transition"]["exact_accuracy"]; cr = control["evaluation"]["heldout_rollout"]["exact_accuracy"]
        tt = challenger["evaluation"]["heldout_transition"]["exact_accuracy"]; tr = challenger["evaluation"]["heldout_rollout"]["exact_accuracy"]; invalid = challenger["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]
        comparison = {"control_transition": ct, "control_rollout": cr, "challenger_transition": tt, "challenger_rollout": tr, "transition_change": tt - ct, "rollout_gain": tr - cr, "challenger_invalid_rollout": invalid}
        gates = config["gates"]; passed = comparison["rollout_gain"] >= gates["minimum_rollout_gain_over_control"] and tr >= gates["minimum_challenger_rollout"] and tt >= gates["minimum_challenger_transition"] and comparison["transition_change"] >= -gates["maximum_transition_regression"] and invalid <= gates["maximum_invalid_rollout_rate"]
        decision = "advance-rollout-aware-independent-gate" if passed else "reject-scheduled-state-formulation"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids}, "source": {"repository_commit": git_commit, "dirty": git_dirty, "challenge_commit": exp100.CHALLENGE_COMMIT, "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "source_checkpoint_sha256": sha256(source)}, "arms": results, "comparison": comparison, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Matched official-like heldout identities; no consumed independent seed or checked answer was used.", "Hard scheduled predictions are detached, so learning occurs through recovery targets rather than backpropagation through time."]}
        summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({**comparison, "wall_seconds": summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": comparison, "checkpoints": {k: v["checkpoint"] for k, v in results.items()}, "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
