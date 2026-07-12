"""Expand the successful width-independent cellular curriculum through width 24."""

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
import torch
import yaml
from safetensors.torch import load_file, save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_311_cellular_expanded_width_curriculum.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-311-cellular-expanded-width"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp310 = load_module("exp310_for_exp311", ROOT / "experiments/exp_310_chunk_shared_recurrent/run.py"); exp200 = load_module("exp200_for_exp311", ROOT / "experiments/exp_200_width32_transfer_ab/run.py"); exp100 = exp310.exp100; cellular = exp310.cellular; torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); seed = config["seed"]
    def split_and_sampler(width: int, train: bool, offset: int):
        minimum = max(2, width - config["training"]["prime_bit_span"] + 1)
        if width <= 16:
            split = exp100.split_primes(minimum, width, seed + width)
            sampler = exp100.TransitionSampler(width, minimum, width, split["train" if train else "test"], seed + width * 100 + offset, config["training"]["composite_fraction"] if train else 0.0, config["training"]["random_state_fraction"] if train else 0.0)
        else:
            split = exp200.sampled_prime_split(minimum, width, seed + width)
            sampler = exp200.ScalableTransitionSampler(width, minimum, width, split["train" if train else "test"], seed + width * 100 + offset, config["training"]["composite_fraction"] if train else 0.0, config["training"]["random_state_fraction"] if train else 0.0)
        return sampler
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); model = cellular.SharedDilatedConvGRU(width=24, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]); model.load_state_dict(load_file(source)); optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"]); samplers = {width: split_and_sampler(width, True, 1) for width in config["training"]["widths"]}; selector = torch.Generator().manual_seed(seed + 3112); widths = config["training"]["widths"]; started = time.perf_counter(); history = []; commit = git_output("rev-parse", "HEAD")
    def evaluate(checkpoint_model, width: int, transition_cases: int, rollout_cases: int):
        checkpoint_model.width = width; transition = exp100.evaluate_transitions(checkpoint_model, split_and_sampler(width, False, 50000), transition_cases); rollout = exp100.evaluate_rollouts(checkpoint_model, split_and_sampler(width, False, 50001), rollout_cases); return {"transition": transition, "rollout": rollout}
    source32 = evaluate(model, 32, config["evaluation"]["width32_transition_cases"], config["evaluation"]["width32_rollout_cases"])
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "expanded-width-cellular-curriculum", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": exp100.CHALLENGE_COMMIT}); mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "widths": str(widths), "source_checkpoint_sha256": sha256(source), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__))}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-311-cellular-expanded-width-curriculum.md"), artifact_path="preregistration"); model.train()
        for step in range(1, config["training"]["steps"] + 1):
            width = widths[(step - 1) % len(widths)]; state, multiplicand, modulus, bit, target, quotient = exp310.mine_batch(samplers[width], config["training"]["batch_size"], config, selector); logits, quotient_logits = model(exp310.exp100_model.make_features(state, multiplicand, modulus, bit, width)); loss, stats = exp100.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"]); optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 100 == 0 or step == config["training"]["steps"]:
                record = {"step": step, "width": width, "elapsed_seconds": time.perf_counter() - started, "gradient_norm": gradient, **stats}; history.append(record); mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient, "train_width": width}, step=step); print(json.dumps(record), flush=True)
        model.eval(); evaluation = {"width16": evaluate(model, 16, config["evaluation"]["width16_transition_cases"], config["evaluation"]["width16_rollout_cases"]), "width24": evaluate(model, 24, config["evaluation"]["width24_transition_cases"], config["evaluation"]["width24_rollout_cases"]), "width32": evaluate(model, 32, config["evaluation"]["width32_transition_cases"], config["evaluation"]["width32_rollout_cases"])}; output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); checkpoint = output / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); gain = evaluation["width32"]["transition"]["exact_accuracy"] - source32["transition"]["exact_accuracy"]; gates = config["gates"]; passed = evaluation["width16"]["rollout"]["exact_accuracy"] >= gates["minimum_retained_width16_rollout"] and evaluation["width24"]["transition"]["exact_accuracy"] >= gates["minimum_width24_transition"] and evaluation["width24"]["rollout"]["exact_accuracy"] >= gates["minimum_width24_rollout"] and gain >= gates["minimum_width32_transition_gain"] and evaluation["width32"]["transition"]["exact_accuracy"] >= gates["minimum_width32_transition"] and evaluation["width32"]["rollout"]["exact_accuracy"] >= gates["minimum_width32_rollout"] and evaluation["width32"]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_width32_invalid_rollout_rate"]; decision = "advance-cellular-direct-width32" if passed else "reject-expanded-width-cellular-curriculum"; summary = {"run_id": config["run_id"], "status": "completed", "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": exp100.CHALLENGE_COMMIT, "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "source_checkpoint_sha256": sha256(source)}, "source_width32": source32, "trained": {"parameters": model.parameter_count(), "training": {"steps": config["training"]["steps"], "history": history}, "evaluation": evaluation, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}, "comparison": {"width32_transition_gain": gain}, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Generated identity-disjoint architecture curriculum only; no consumed independent or checked answer was used.", "Width-32 evaluation is zero-shot length transfer, not Tier-4 readiness."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"width16_rollout": evaluation["width16"]["rollout"]["exact_accuracy"], "width24_transition": evaluation["width24"]["transition"]["exact_accuracy"], "width24_rollout": evaluation["width24"]["rollout"]["exact_accuracy"], "width32_transition": evaluation["width32"]["transition"]["exact_accuracy"], "width32_rollout": evaluation["width32"]["rollout"]["exact_accuracy"], "width32_transition_gain": gain, "wall_seconds": summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "source_width32": source32, "evaluation": evaluation, "width32_transition_gain": gain, "checkpoint": summary["trained"]["checkpoint"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
