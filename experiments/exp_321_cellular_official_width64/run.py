"""Direct official-numeric width-64 continuation of the cellular model."""

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
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_321_cellular_official_width64.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-321-cellular-official-width64"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp321", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); exp320 = load_module("exp320_for_exp321", ROOT / "experiments/exp_320_cellular_width64_curriculum/run.py"); cellular = load_module("cellular_for_exp321", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); seed = config["seed"]
    def split(width: int):
        minimum = 53 if width == 56 else 33; local = {"data": {"prime_bits_min": minimum, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}; return exp300.bigint_prime_split(local, seed + 32100 + width)
    splits = {width: split(width) for width in (56, 64)}; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); model = cellular.SharedDilatedConvGRU(width=64, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]); model.load_state_dict(load_file(source)); model.eval()
    def evaluate(width: int, transition_cases: int, rollout_cases: int): model.width = width; return {"transition": exp300.evaluate_transitions(model, exp300.BigIntSampler(splits[width]["test"], seed + 32500 + width), transition_cases), "rollout": exp300.evaluate_rollouts(model, exp300.BigIntSampler(splits[width]["test"], seed + 32600 + width), rollout_cases)}
    source64 = evaluate(64, config["evaluation"]["source_width64_transition_cases"], config["evaluation"]["source_width64_rollout_cases"]); samplers = {width: exp300.BigIntSampler(splits[width]["train"], seed + 32200 + width) for width in (56, 64)}; optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"]); selector = torch.Generator().manual_seed(seed + 3212); pool_size = config["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"]; started = time.perf_counter(); history = []; experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); commit = git_output("rev-parse", "HEAD")
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "cellular-official-width64", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"}); mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "source_checkpoint_sha256": sha256(source), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__))}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-321-cellular-official-width64.md"), artifact_path="preregistration"); model.train()
        for step in range(1, config["training"]["steps"] + 1):
            width = 56 if step % 2 else 64; state, multiplicand, modulus, input_bit, target, quotient = exp320.mine(samplers[width].transitions(pool_size), config["training"]["batch_size"], config, selector); logits, quotient_logits = model(exp300.features(state, multiplicand, modulus, input_bit, width)); loss, stats = exp300.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"]); optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 100 == 0 or step == config["training"]["steps"]:
                record = {"step": step, "width": width, "elapsed_seconds": time.perf_counter() - started, "gradient_norm": gradient, **stats}; history.append(record); mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient}, step=step); print(json.dumps(record), flush=True)
        model.eval(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); checkpoint = output / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); evaluation = {"width56": evaluate(56, config["evaluation"]["width56_transition_cases"], config["evaluation"]["width56_rollout_cases"]), "width64": evaluate(64, config["evaluation"]["width64_transition_cases"], config["evaluation"]["width64_rollout_cases"])}; gain = evaluation["width64"]["rollout"]["exact_accuracy"] - source64["rollout"]["exact_accuracy"]; gates = config["gates"]; passed = evaluation["width56"]["rollout"]["exact_accuracy"] >= gates["minimum_retained_width56_rollout"] and evaluation["width64"]["transition"]["exact_accuracy"] >= gates["minimum_width64_transition"] and evaluation["width64"]["rollout"]["exact_accuracy"] >= gates["minimum_width64_rollout"] and gain >= gates["minimum_width64_rollout_gain"] and evaluation["width64"]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_width64_invalid_rollout_rate"]; decision = "advance-cellular-independent-tier5" if passed else "reject-cellular-official-width64"; summary = {"run_id": config["run_id"], "status": "completed", "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "source_checkpoint_sha256": sha256(source)}, "source_width64": source64, "trained": {"parameters": model.parameter_count(), "training": {"steps": config["training"]["steps"], "history": history}, "evaluation": evaluation, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}, "comparison": {"width64_rollout_gain": gain}, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Python bigint arithmetic is confined to offline data, labels, mining, and evaluation truth.", "No checked, consumed independent, or private answers were read."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"source_width64_rollout": source64["rollout"]["exact_accuracy"], "width56_rollout": evaluation["width56"]["rollout"]["exact_accuracy"], "width64_transition": evaluation["width64"]["transition"]["exact_accuracy"], "width64_rollout": evaluation["width64"]["rollout"]["exact_accuracy"], "width64_rollout_gain": gain, "wall_seconds": summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "source_width64": source64, "evaluation": evaluation, "width64_rollout_gain": gain, "checkpoint": summary["trained"]["checkpoint"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
