"""Matched mixed-width cellular versus chunk-shared recurrent experiment."""

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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(EXP100_DIR))
import model as exp100_model  # noqa: E402
import run as exp100  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


cellular = load_module("cellular_for_exp310", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
chunked = load_module("chunked_for_exp310", Path(__file__).resolve().parent / "model.py")


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def mine_batch(sampler, size: int, config: dict, generator: torch.Generator):
    values = sampler.transitions(size * config["hard_mining"]["candidate_pool_factor"]); state, multiplicand, modulus, bit, target, quotient = values; raw = 2 * state + bit * multiplicand; margin = torch.minimum((raw - modulus).abs(), (raw - 2 * modulus).abs()); boundary = margin.double() / modulus.double() <= config["hard_mining"]["boundary_relative_margin"]; available = torch.ones(len(state), dtype=torch.bool)
    def draw(mask: torch.Tensor, count: int) -> torch.Tensor:
        candidates = torch.nonzero(mask & available, as_tuple=False).flatten(); order = torch.randperm(len(candidates), generator=generator); chosen = candidates[order[:min(count, len(candidates))]]; available[chosen] = False; return chosen
    q2 = draw(quotient.eq(2), round(size * config["hard_mining"]["quotient2_fraction"])); near = draw(boundary, round(size * config["hard_mining"]["boundary_fraction"])); rest = draw(available, size - len(q2) - len(near)); selected = torch.cat((q2, near, rest)); selected = selected[torch.randperm(len(selected), generator=generator)]; return tuple(value[selected] for value in values)


def make_samplers(config: dict, seed: int):
    samplers = {}; splits = {}
    for width in config["training"]["widths"]:
        minimum = max(2, width - config["training"]["prime_bit_span"] + 1); split = exp100.split_primes(minimum, width, seed + width); splits[width] = (minimum, split); samplers[width] = exp100.TransitionSampler(width, minimum, width, split["train"], seed + width * 100 + 1, config["training"]["composite_fraction"], config["training"]["random_state_fraction"])
    return samplers, splits


def train_arm(name: str, constructor, config: dict, output: Path) -> dict:
    seed = config["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); model = constructor(16); optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"]); samplers, splits = make_samplers(config, seed); selector = torch.Generator().manual_seed(seed + 3102); process = psutil.Process(); peak = process.memory_info().rss; history = []; started = time.perf_counter(); widths = config["training"]["widths"]; model.train()
    for step in range(1, config["training"]["steps"] + 1):
        width = widths[(step - 1) % len(widths)]; state, multiplicand, modulus, bit, target, quotient = mine_batch(samplers[width], config["training"]["batch_size"], config, selector); logits, quotient_logits = model(exp100_model.make_features(state, multiplicand, modulus, bit, width)); loss, stats = exp100.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"]); optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
        if step == 1 or step % 100 == 0 or step == config["training"]["steps"]:
            peak = max(peak, process.memory_info().rss); record = {"step": step, "width": width, "elapsed_seconds": time.perf_counter() - started, "gradient_norm": gradient, **stats}; history.append(record); mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient, "train_width": width}, step=step); print(json.dumps({"arm": name, **record}), flush=True)
    training_seconds = time.perf_counter() - started; minimum16, split16 = splits[16]; model.eval(); test_seed = seed + 16 * 100 + 50000; transition = exp100.evaluate_transitions(model, exp100.TransitionSampler(16, minimum16, 16, split16["test"], test_seed, 0.0, config["training"]["random_state_fraction"]), config["evaluation"]["width16_transition_cases"]); rollout = exp100.evaluate_rollouts(model, exp100.TransitionSampler(16, minimum16, 16, split16["test"], test_seed + 1, 0.0, 0.0), config["evaluation"]["width16_rollout_cases"]); output.mkdir(parents=True, exist_ok=True); checkpoint = output / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); return {"arm": name, "parameters": model.parameter_count(), "training": {"steps": config["training"]["steps"], "seconds": training_seconds, "peak_rss_bytes": peak, "history": history}, "evaluation": {"width16_transition": transition, "width16_rollout": rollout}, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}


def evaluate_width32(constructor, checkpoint: Path, config: dict, exp200) -> dict:
    model = constructor(32); model.load_state_dict(load_file(checkpoint)); model.eval(); split = exp200.sampled_prime_split(config["evaluation"]["width32_prime_bits_min"], config["evaluation"]["width32_prime_bits_max"], config["seed"] + 32); seed = config["seed"] + 32 * 100 + 50000; transition = exp100.evaluate_transitions(model, exp200.ScalableTransitionSampler(32, 17, 32, split["test"], seed, 0.0, config["training"]["random_state_fraction"]), config["evaluation"]["width32_transition_cases"]); rollout = exp100.evaluate_rollouts(model, exp200.ScalableTransitionSampler(32, 17, 32, split["test"], seed + 1, 0.0, 0.0), config["evaluation"]["width32_rollout_cases"]); return {"transition": transition, "rollout": rollout}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_310_chunk_shared_recurrent.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-310-chunk-shared-recurrent"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); exp200 = load_module("exp200_for_exp310", ROOT / "experiments/exp_200_width32_transfer_ab/run.py"); experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); constructors = {"cellular-control": lambda width: cellular.SharedDilatedConvGRU(width=width, hidden=config["control"]["hidden"], cycles=config["control"]["cycles"]), "chunk-gru-challenger": lambda width: chunked.ChunkSharedBiGRU(width=width, chunk_size=config["challenger"]["chunk_size"], hidden=config["challenger"]["hidden"], cycles=config["challenger"]["cycles"])}; started = time.perf_counter(); commit = git_output("rev-parse", "HEAD"); results = {}; run_ids = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "matched-mixed-width-recurrent-ab", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": exp100.CHALLENGE_COMMIT}); mlflow.log_params({"seed": config["seed"], "steps": config["training"]["steps"], "widths": str(config["training"]["widths"]), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "model_sha256": sha256(Path(__file__).resolve().parent / "model.py"), "python": platform.python_version(), "pytorch": torch.__version__}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-310-chunk-shared-recurrent-reduction.md"), artifact_path="preregistration")
        for name, constructor in constructors.items():
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id; results[name] = train_arm(name, constructor, config, output / name); zero = evaluate_width32(constructor, output / name / "weights.safetensors", config, exp200); results[name]["evaluation"]["zero_shot_width32"] = zero; mlflow.log_metrics({"width16_transition": results[name]["evaluation"]["width16_transition"]["exact_accuracy"], "width16_rollout": results[name]["evaluation"]["width16_rollout"]["exact_accuracy"], "width32_transition": zero["transition"]["exact_accuracy"], "width32_rollout": zero["rollout"]["exact_accuracy"]})
        control = results["cellular-control"]; challenger = results["chunk-gru-challenger"]; ct = control["evaluation"]["width16_transition"]["exact_accuracy"]; cr = control["evaluation"]["width16_rollout"]["exact_accuracy"]; tt = challenger["evaluation"]["width16_transition"]["exact_accuracy"]; tr = challenger["evaluation"]["width16_rollout"]["exact_accuracy"]; zt = challenger["evaluation"]["zero_shot_width32"]["transition"]["exact_accuracy"]; zr = challenger["evaluation"]["zero_shot_width32"]["rollout"]["exact_accuracy"]; comparison = {"control_width16_transition": ct, "control_width16_rollout": cr, "challenger_width16_transition": tt, "challenger_width16_rollout": tr, "transition_gain": tt - ct, "rollout_gain": tr - cr, "challenger_zero_width32_transition": zt, "challenger_zero_width32_rollout": zr, "parameter_ratio": challenger["parameters"] / control["parameters"], "training_time_ratio": challenger["training"]["seconds"] / control["training"]["seconds"]}; gates = config["gates"]; passed = tt >= gates["minimum_challenger_width16_transition"] and tr >= gates["minimum_challenger_width16_rollout"] and comparison["transition_gain"] >= gates["minimum_transition_gain_over_control"] and comparison["rollout_gain"] >= gates["minimum_rollout_gain_over_control"] and zt >= gates["minimum_zero_shot_width32_transition"] and zr >= gates["minimum_zero_shot_width32_rollout"] and comparison["parameter_ratio"] <= gates["maximum_parameter_ratio"] and comparison["training_time_ratio"] <= gates["maximum_training_time_ratio"]; decision = "advance-chunk-gru-curriculum" if passed else "reject-chunk-shared-bidirectional-gru"; summary = {"run_id": config["run_id"], "status": "completed", "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": exp100.CHALLENGE_COMMIT, "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "model_sha256": sha256(Path(__file__).resolve().parent / "model.py")}, "arms": results, "comparison": comparison, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Mixed-width architecture smoke test; no consumed independent or checked answer was used.", "Zero-shot width 32 measures length transfer, not Tier-4 readiness."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({**comparison, "wall_seconds": summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": comparison, "checkpoints": {name: value["checkpoint"] for name, value in results.items()}, "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
