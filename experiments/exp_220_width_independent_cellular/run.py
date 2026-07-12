"""Run EXP-220 matched full-canvas versus width-independent cellular smoke test."""

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

spec = importlib.util.spec_from_file_location("cellular_model", Path(__file__).resolve().parent / "model.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load cellular model")
cellular = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cellular)


def load_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def train_model(name: str, constructor, config: dict, split: dict, exp200, output: Path) -> dict:
    seed = int(config["seed"])
    training = config["training"]
    width = int(training["width"])
    random.seed(seed + width)
    np.random.seed(seed + width)
    torch.manual_seed(seed + width)
    model = constructor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["steps"]), eta_min=float(training["minimum_learning_rate"]))
    sampler = exp200.ScalableHardMiningSampler(width, training["prime_bits_min"], training["prime_bits_max"], split["train"], seed + width * 100 + 1, training["composite_fraction"], training["random_state_fraction"], pool_factor=config["hard_mining"]["candidate_pool_factor"], quotient2_fraction=config["hard_mining"]["quotient2_fraction"], boundary_fraction=config["hard_mining"]["boundary_fraction"], boundary_relative_margin=config["hard_mining"]["boundary_relative_margin"])
    process = psutil.Process(); peak_rss = process.memory_info().rss; history = []; started = time.perf_counter(); model.train()
    for step in range(1, int(training["steps"]) + 1):
        state, multiplicand, modulus, bit, target, quotient = sampler.transitions(int(training["batch_size"]))
        features = exp100_model.make_features(state, multiplicand, modulus, bit, width)
        logits, quotient_logits = model(features)
        loss, stats = exp100.transition_loss(logits, quotient_logits, target, quotient, width, training["worst_bit_weight"], training["quotient_weight"])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"]))
        optimizer.step(); scheduler.step()
        if step == 1 or step % 50 == 0 or step == training["steps"]:
            peak_rss = max(peak_rss, process.memory_info().rss)
            record = {"step": step, "elapsed_seconds": time.perf_counter() - started, "learning_rate": scheduler.get_last_lr()[0], "gradient_norm": gradient_norm, **stats}
            history.append(record); print(json.dumps({"arm": name, **record}), flush=True)
            mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient_norm, "learning_rate": scheduler.get_last_lr()[0]}, step=step)
    training_seconds = time.perf_counter() - started
    test_seed = seed + width * 100 + 50000
    transition = exp100.evaluate_transitions(model, exp200.ScalableTransitionSampler(width, training["prime_bits_min"], training["prime_bits_max"], split["test"], test_seed, 0.0, training["random_state_fraction"]), config["evaluation"]["width16_transition_cases"])
    rollout = exp100.evaluate_rollouts(model, exp200.ScalableTransitionSampler(width, training["prime_bits_min"], training["prime_bits_max"], split["test"], test_seed + 1, 0.0, 0.0), config["evaluation"]["width16_rollout_cases"])
    output.mkdir(parents=True, exist_ok=True); checkpoint = output / "weights.safetensors"
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, str(checkpoint), metadata={"experiment": config["run_id"], "arm": name, "width_independent": str(name == "shared-cellular-challenger").lower()})
    return {"arm": name, "parameters": model.parameter_count(), "training": {"steps": training["steps"], "examples": training["steps"] * training["batch_size"], "seconds": training_seconds, "peak_rss_bytes": peak_rss, "history": history}, "evaluation": {"heldout_transition": transition, "heldout_rollout": rollout}, "hard_mining": sampler.mining_stats(), "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_220_width_independent_cellular.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-220-width-independent-cellular")
    args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text())
    exp200 = load_module("exp200_for_exp220", ROOT / "experiments/exp_200_width32_transfer_ab/run.py")
    training = config["training"]; width = training["width"]
    split16 = exp100.split_primes(training["prime_bits_min"], training["prime_bits_max"], config["seed"] + width)
    split32 = exp200.sampled_prime_split(config["evaluation"]["width32_prime_bits_min"], config["evaluation"]["width32_prime_bits_max"], config["seed"] + 32)
    torch.set_num_threads(training["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    constructors = {
        "full-canvas-control": lambda: exp100_model.SharedContextHornerTCN(width=width, hidden=config["control"]["hidden"], blocks=config["control"]["blocks"]),
        "shared-cellular-challenger": lambda: cellular.SharedDilatedConvGRU(width=width, hidden=config["challenger"]["hidden"], cycles=config["challenger"]["cycles"]),
    }
    git_commit = git_output("rev-parse", "HEAD"); git_dirty = bool(git_output("status", "--porcelain")); started = time.perf_counter(); results = {}; arm_ids = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "width-independent-architecture-ab", "ab_group": "EXP-220", "promotion_gate": "preregistered-width16-and-zero-shot-width32", "repository_commit": git_commit, "repository_dirty": str(git_dirty).lower(), "challenge_commit": exp100.CHALLENGE_COMMIT})
        mlflow.log_params({"seed": config["seed"], "steps": training["steps"], "batch_size": training["batch_size"], "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "model_sha256": sha256(Path(__file__).resolve().parent / "model.py"), "python": platform.python_version(), "pytorch": torch.__version__})
        mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-220-width-independent-cellular.md"), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/research/2026-07-12-width-independent-arithmetic.md"), artifact_path="research")
        for name, constructor in constructors.items():
            with mlflow.start_run(run_name=name, nested=True) as nested:
                arm_ids[name] = nested.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "parent_run_id": parent_id, "ab_group": name, "status": "running"})
                results[name] = train_model(name, constructor, config, split16, exp200, output / name)
                mlflow.log_metrics({"transition_exact": results[name]["evaluation"]["heldout_transition"]["exact_accuracy"], "rollout_exact": results[name]["evaluation"]["heldout_rollout"]["exact_accuracy"], "invalid_rollout": results[name]["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]}); mlflow.set_tag("status", "completed")
        state_dict = load_file(str(output / "shared-cellular-challenger" / "weights.safetensors"))
        challenger32 = cellular.SharedDilatedConvGRU(width=32, hidden=config["challenger"]["hidden"], cycles=config["challenger"]["cycles"]); challenger32.load_state_dict(state_dict); challenger32.eval()
        test_seed = config["seed"] + 32 * 100 + 50000
        zero_transition = exp100.evaluate_transitions(challenger32, exp200.ScalableTransitionSampler(32, config["evaluation"]["width32_prime_bits_min"], config["evaluation"]["width32_prime_bits_max"], split32["test"], test_seed, 0.0, training["random_state_fraction"]), config["evaluation"]["width32_transition_cases"])
        zero_rollout = exp100.evaluate_rollouts(challenger32, exp200.ScalableTransitionSampler(32, config["evaluation"]["width32_prime_bits_min"], config["evaluation"]["width32_prime_bits_max"], split32["test"], test_seed + 1, 0.0, 0.0), config["evaluation"]["width32_rollout_cases"])
        results["shared-cellular-challenger"]["evaluation"]["zero_shot_width32_transition"] = zero_transition; results["shared-cellular-challenger"]["evaluation"]["zero_shot_width32_rollout"] = zero_rollout
        control = results["full-canvas-control"]; challenger = results["shared-cellular-challenger"]
        cr = control["evaluation"]["heldout_rollout"]["exact_accuracy"]; ct = challenger["evaluation"]["heldout_transition"]["exact_accuracy"]; rr = challenger["evaluation"]["heldout_rollout"]["exact_accuracy"]
        comparison = {"control_width16_rollout": cr, "challenger_width16_transition": ct, "challenger_width16_rollout": rr, "width16_rollout_deficit": cr - rr, "zero_shot_width32_transition": zero_transition["exact_accuracy"], "zero_shot_width32_rollout": zero_rollout["exact_accuracy"], "parameter_ratio": challenger["parameters"] / control["parameters"], "training_time_ratio": challenger["training"]["seconds"] / control["training"]["seconds"]}
        gates = config["gates"]; passed = ct >= gates["minimum_challenger_width16_transition"] and rr >= gates["minimum_challenger_width16_rollout"] and comparison["width16_rollout_deficit"] <= gates["maximum_width16_rollout_deficit_to_control"] and comparison["zero_shot_width32_transition"] >= gates["minimum_zero_shot_width32_transition"] and comparison["zero_shot_width32_rollout"] >= gates["minimum_zero_shot_width32_rollout"] and comparison["parameter_ratio"] <= gates["maximum_parameter_ratio"] and comparison["training_time_ratio"] <= gates["maximum_training_time_ratio"]
        decision = "advance-cellular-mixed-width" if passed else "reject-shared-dilated-convgru"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": arm_ids}, "source": {"repository_commit": git_commit, "dirty": git_dirty, "challenge_commit": exp100.CHALLENGE_COMMIT, "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "model_sha256": sha256(Path(__file__).resolve().parent / "model.py")}, "arms": results, "comparison": comparison, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Architecture smoke test; no consumed independent seed or checked answer was used.", "Parameter and wall-time ratios are bounded but not exactly matched.", "Zero-shot width 32 tests length transfer without fine-tuning, not Tier-4 readiness."]}
        summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({**comparison, "wall_seconds": summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": comparison, "checkpoints": {k: v["checkpoint"] for k, v in results.items()}, "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
