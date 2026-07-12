"""Bigint-backed progressive cellular curriculum through width 56."""

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


def prime_split(exp300, config: dict, width: int) -> dict:
    local = {"data": {"prime_bits_min": width - config["training"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
    return exp300.bigint_prime_split(local, config["seed"] + 32000 + width)


def mine(values, size: int, config: dict, generator: torch.Generator):
    state, multiplicand, modulus, input_bit, target, quotient = values; raw = [2 * first + bit * second for first, bit, second in zip(state, input_bit, multiplicand)]; boundary = torch.tensor([min(abs(value - prime), abs(value - 2 * prime)) / prime <= config["hard_mining"]["boundary_relative_margin"] for value, prime in zip(raw, modulus)]); available = torch.ones(len(state), dtype=torch.bool)
    def draw(mask: torch.Tensor, count: int) -> torch.Tensor:
        candidates = torch.nonzero(mask & available, as_tuple=False).flatten(); chosen = candidates[torch.randperm(len(candidates), generator=generator)[:min(count, len(candidates))]]; available[chosen] = False; return chosen
    q2 = draw(quotient.eq(2), round(size * config["hard_mining"]["quotient2_fraction"])); near = draw(boundary, round(size * config["hard_mining"]["boundary_fraction"])); rest = draw(available, size - len(q2) - len(near)); indices = torch.cat((q2, near, rest)); indices = indices[torch.randperm(len(indices), generator=generator)]; positions = indices.tolist(); return ([state[index] for index in positions], [multiplicand[index] for index in positions], [modulus[index] for index in positions], [input_bit[index] for index in positions], [target[index] for index in positions], quotient[indices])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_320_cellular_width64_curriculum.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-320-cellular-width64-curriculum"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp320", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp320", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); seed = config["seed"]; splits = {width: prime_split(exp300, config, width) for width in [*config["training"]["widths"], 64]}; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); model = cellular.SharedDilatedConvGRU(width=64, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]); model.load_state_dict(load_file(source)); model.eval()
    def evaluate(width: int, transition_cases: int, rollout_cases: int):
        model.width = width; return {"transition": exp300.evaluate_transitions(model, exp300.BigIntSampler(splits[width]["test"], seed + 32500 + width), transition_cases), "rollout": exp300.evaluate_rollouts(model, exp300.BigIntSampler(splits[width]["test"], seed + 32600 + width), rollout_cases)}
    source64 = evaluate(64, config["evaluation"]["source_width64_transition_cases"], config["evaluation"]["source_width64_rollout_cases"]); samplers = {width: exp300.BigIntSampler(splits[width]["train"], seed + 32100 + width) for width in config["training"]["widths"]}; optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"]); selector = torch.Generator().manual_seed(seed + 3202); widths = config["training"]["widths"]; pool_size = config["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"]; started = time.perf_counter(); history = []; experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); commit = git_output("rev-parse", "HEAD")
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "bigint-cellular-width64-curriculum", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"}); mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "widths": str(widths), "source_checkpoint_sha256": sha256(source), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__))}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-320-cellular-width64-curriculum.md"), artifact_path="preregistration"); model.train()
        for step in range(1, config["training"]["steps"] + 1):
            width = widths[(step - 1) % len(widths)]; state, multiplicand, modulus, input_bit, target, quotient = mine(samplers[width].transitions(pool_size), config["training"]["batch_size"], config, selector); logits, quotient_logits = model(exp300.features(state, multiplicand, modulus, input_bit, width)); loss, stats = exp300.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"]); optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 100 == 0 or step == config["training"]["steps"]:
                record = {"step": step, "width": width, "elapsed_seconds": time.perf_counter() - started, "gradient_norm": gradient, **stats}; history.append(record); mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient}, step=step); print(json.dumps(record), flush=True)
        model.eval(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); checkpoint = output / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); evaluation = {"width32": evaluate(32, config["evaluation"]["width32_transition_cases"], config["evaluation"]["width32_rollout_cases"]), "width56": evaluate(56, config["evaluation"]["width56_transition_cases"], config["evaluation"]["width56_rollout_cases"]), "width64": evaluate(64, config["evaluation"]["width64_transition_cases"], config["evaluation"]["width64_rollout_cases"])}; gates = config["gates"]; passed = evaluation["width32"]["rollout"]["exact_accuracy"] >= gates["minimum_retained_width32_rollout"] and evaluation["width56"]["transition"]["exact_accuracy"] >= gates["minimum_width56_transition"] and evaluation["width56"]["rollout"]["exact_accuracy"] >= gates["minimum_width56_rollout"] and evaluation["width64"]["transition"]["exact_accuracy"] >= gates["minimum_zero_width64_transition"] and evaluation["width64"]["rollout"]["exact_accuracy"] >= gates["minimum_zero_width64_rollout"] and evaluation["width64"]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_zero_width64_invalid_rollout_rate"]; decision = "advance-cellular-direct-width64" if passed else "reject-cellular-width64-curriculum"; summary = {"run_id": config["run_id"], "status": "completed", "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "source_checkpoint_sha256": sha256(source)}, "source_width64": source64, "trained": {"parameters": model.parameter_count(), "training": {"steps": config["training"]["steps"], "history": history}, "evaluation": evaluation, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Python bigint arithmetic is confined to offline data, labels, hard mining, and evaluation truth.", "The cellular model receives and emits tensors only.", "No checked, consumed independent, or private answers were read."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"width32_rollout": evaluation["width32"]["rollout"]["exact_accuracy"], "width56_transition": evaluation["width56"]["transition"]["exact_accuracy"], "width56_rollout": evaluation["width56"]["rollout"]["exact_accuracy"], "width64_transition": evaluation["width64"]["transition"]["exact_accuracy"], "width64_rollout": evaluation["width64"]["rollout"]["exact_accuracy"], "wall_seconds": summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "source_width64": source64, "evaluation": evaluation, "checkpoint": summary["trained"]["checkpoint"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
