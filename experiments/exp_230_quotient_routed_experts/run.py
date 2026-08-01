"""Run EXP-230 matched single-head versus learned quotient-routed experts."""

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
from safetensors.torch import save_file
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


routed_module = load_module("routed_model", Path(__file__).resolve().parent / "model.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def build_control(config: dict, source: Path):
    model = exp100_model.SharedContextHornerTCN(width=config["model"]["width"], hidden=config["model"]["hidden"], blocks=config["model"]["blocks"])
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(source)))
    return model


def build_routed(config: dict, source: Path):
    model = routed_module.QuotientRoutedExperts(**config["model"])
    model.load_single_head_checkpoint(source)
    return model


def source_equivalence(config: dict, source: Path, exp200, split: dict) -> dict:
    control = build_control(config, source).eval(); routed = build_routed(config, source).eval()
    sampler = exp200.ScalableTransitionSampler(32, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], config["seed"] + 23001, 0.0, 0.15)
    state, multiplicand, modulus, bit, _, _ = sampler.transitions(512)
    features = exp100_model.make_features(state, multiplicand, modulus, bit, 32)
    with torch.inference_mode():
        control_logits, control_q = control(features); routed_logits, routed_q = routed(features)
    return {"maximum_output_logit_difference": float((control_logits - routed_logits).abs().max()), "maximum_router_logit_difference": float((control_q - routed_q).abs().max())}


def train_arm(name: str, routed: bool, config: dict, source: Path, split: dict, exp200, output: Path) -> dict:
    seed = int(config["seed"]); training = config["training"]; width = config["model"]["width"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_routed(config, source) if routed else build_control(config, source)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["steps"], eta_min=training["minimum_learning_rate"])
    sampler = exp200.ScalableHardMiningSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["train"], seed + 23002, training["composite_fraction"], training["random_state_fraction"], pool_factor=config["hard_mining"]["candidate_pool_factor"], quotient2_fraction=config["hard_mining"]["quotient2_fraction"], boundary_fraction=config["hard_mining"]["boundary_fraction"], boundary_relative_margin=config["hard_mining"]["boundary_relative_margin"])
    process = psutil.Process(); peak_rss = process.memory_info().rss; history = []; started = time.perf_counter(); model.train()
    for step in range(1, training["steps"] + 1):
        state, multiplicand, modulus, bit, target, quotient = sampler.transitions(training["batch_size"])
        features = exp100_model.make_features(state, multiplicand, modulus, bit, width)
        logits, quotient_logits = model(features, teacher_route=quotient) if routed else model(features)
        loss, stats = exp100.transition_loss(logits, quotient_logits, target, quotient, width, training["worst_bit_weight"], training["quotient_weight"])
        optimizer.zero_grad(set_to_none=True); loss.backward(); gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])); optimizer.step(); scheduler.step()
        if step == 1 or step % 50 == 0 or step == training["steps"]:
            peak_rss = max(peak_rss, process.memory_info().rss); record = {"step": step, "elapsed_seconds": time.perf_counter() - started, "learning_rate": scheduler.get_last_lr()[0], "gradient_norm": gradient_norm, **stats}; history.append(record); print(json.dumps({"arm": name, **record}), flush=True)
            mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient_norm, "learning_rate": scheduler.get_last_lr()[0]}, step=step)
    training_seconds = time.perf_counter() - started; model.eval(); test_seed = seed + 23050
    transition = exp100.evaluate_transitions(model, exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], test_seed, 0.0, training["random_state_fraction"]), config["evaluation"]["transition_cases"])
    rollout = exp100.evaluate_rollouts(model, exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], test_seed + 1, 0.0, 0.0), config["evaluation"]["rollout_cases"])
    output.mkdir(parents=True, exist_ok=True); checkpoint = output / "weights.safetensors"; save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(checkpoint), metadata={"experiment": config["run_id"], "arm": name})
    result = {"arm": name, "routed": routed, "parameters": model.parameter_count(), "training": {"steps": training["steps"], "examples": training["steps"] * training["batch_size"], "seconds": training_seconds, "peak_rss_bytes": peak_rss, "history": history}, "evaluation": {"heldout_transition": transition, "heldout_rollout": rollout}, "hard_mining": sampler.mining_stats(), "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}
    if routed:
        result["expert_pairwise_distance"] = model.expert_pairwise_distance()
    return result


@torch.inference_mode()
def router_diagnostics(model, config: dict, split: dict, exp200) -> dict:
    width = config["model"]["width"]; total = config["evaluation"]["router_cases"]; batch_size = 1024
    sampler = exp200.ScalableTransitionSampler(width, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], config["seed"] + 23051, 0.0, config["training"]["random_state_fraction"])
    correct_router = 0; predicted_exact = 0; oracle_exact = 0; cases = 0; weights = 1 << torch.arange(width, dtype=torch.int64)
    while cases < total:
        size = min(batch_size, total - cases); state, multiplicand, modulus, bit, target, quotient = sampler.transitions(size); features = exp100_model.make_features(state, multiplicand, modulus, bit, width)
        predicted_logits, router_logits = model(features); oracle_logits, _ = model(features, teacher_route=quotient)
        predicted = ((predicted_logits >= 0).to(torch.int64) * weights[None, :]).sum(1); oracle = ((oracle_logits >= 0).to(torch.int64) * weights[None, :]).sum(1)
        correct_router += int(router_logits.argmax(1).eq(quotient).sum()); predicted_exact += int(predicted.eq(target).sum()); oracle_exact += int(oracle.eq(target).sum()); cases += size
    return {"cases": cases, "router_correct": correct_router, "router_accuracy": correct_router / cases, "predicted_route_exact_accuracy": predicted_exact / cases, "oracle_route_exact_accuracy": oracle_exact / cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_230_quotient_routed_experts.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-230-quotient-routed-experts"); args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp200 = load_module("exp200_for_exp230", ROOT / "experiments/exp_200_width32_transfer_ab/run.py"); exp204 = load_module("exp204_for_exp230", ROOT / "experiments/exp_204_official_distribution_finetune/run.py")
    split = exp204.official_numeric_prime_split(config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], config["seed"] + 230)
    torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    equivalence = source_equivalence(config, source, exp200, split); git_commit = git_output("rev-parse", "HEAD"); git_dirty = bool(git_output("status", "--porcelain")); started = time.perf_counter(); results = {}; run_ids = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "compliant-neuro-symbolic-ab", "ab_group": "EXP-230", "promotion_gate": "preregistered-learned-quotient-routing", "repository_commit": git_commit, "repository_dirty": str(git_dirty).lower(), "challenge_commit": exp100.CHALLENGE_COMMIT})
        mlflow.log_params({"seed": config["seed"], "steps": config["training"]["steps"], "batch_size": config["training"]["batch_size"], "source_checkpoint_sha256": sha256(source), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "model_sha256": sha256(Path(__file__).resolve().parent / "model.py"), "python": platform.python_version(), "pytorch": torch.__version__}); mlflow.log_metrics(equivalence)
        mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-230-quotient-routed-experts.md"), artifact_path="preregistration")
        for name, routed in (("single-head-control", False), ("routed-expert-challenger", True)):
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "parent_run_id": parent_id, "ab_group": name, "status": "running"}); results[name] = train_arm(name, routed, config, source, split, exp200, output / name); mlflow.log_metrics({"transition_exact": results[name]["evaluation"]["heldout_transition"]["exact_accuracy"], "rollout_exact": results[name]["evaluation"]["heldout_rollout"]["exact_accuracy"], "invalid_rollout": results[name]["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]}); mlflow.set_tag("status", "completed")
        routed_model = routed_module.QuotientRoutedExperts(**config["model"]); routed_model.load_state_dict(__import__("safetensors.torch", fromlist=["load_file"]).load_file(str(output / "routed-expert-challenger" / "weights.safetensors"))); routed_model.eval()
        diagnostics = router_diagnostics(routed_model, config, split, exp200)
        routed_model.routing_mode = "expert0"; forced_expert0 = exp100.evaluate_rollouts(routed_model, exp200.ScalableTransitionSampler(32, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], config["seed"] + 23051, 0.0, 0.0), config["evaluation"]["rollout_cases"]); routed_model.routing_mode = "predicted"
        control = results["single-head-control"]; challenger = results["routed-expert-challenger"]; control_rollout = control["evaluation"]["heldout_rollout"]["exact_accuracy"]; routed_rollout = challenger["evaluation"]["heldout_rollout"]["exact_accuracy"]; routed_transition = challenger["evaluation"]["heldout_transition"]["exact_accuracy"]; invalid = challenger["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]
        comparison = {"control_rollout": control_rollout, "routed_transition": routed_transition, "routed_rollout": routed_rollout, "rollout_gain": routed_rollout - control_rollout, "router_accuracy": diagnostics["router_accuracy"], "oracle_route_transition": diagnostics["oracle_route_exact_accuracy"], "forced_expert0_rollout": forced_expert0["exact_accuracy"], "forced_expert0_rollout_drop": routed_rollout - forced_expert0["exact_accuracy"], "invalid_rollout": invalid, "parameter_ratio": challenger["parameters"] / control["parameters"], "training_time_ratio": challenger["training"]["seconds"] / control["training"]["seconds"]}
        gates = config["gates"]; passed = comparison["rollout_gain"] >= gates["minimum_rollout_gain_over_control"] and routed_rollout >= gates["minimum_routed_rollout"] and routed_transition >= gates["minimum_routed_transition"] and diagnostics["router_accuracy"] >= gates["minimum_router_accuracy"] and comparison["forced_expert0_rollout_drop"] >= gates["minimum_forced_expert0_rollout_drop"] and invalid <= gates["maximum_invalid_rollout_rate"] and comparison["parameter_ratio"] <= gates["maximum_parameter_ratio"] and comparison["training_time_ratio"] <= gates["maximum_training_time_ratio"]
        decision = "advance-routed-experts-independent-gate" if passed else "reject-quotient-routed-experts"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids}, "source": {"repository_commit": git_commit, "dirty": git_dirty, "challenge_commit": exp100.CHALLENGE_COMMIT, "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "model_sha256": sha256(Path(__file__).resolve().parent / "model.py"), "source_checkpoint_sha256": sha256(source)}, "source_equivalence": equivalence, "arms": results, "router_diagnostics": diagnostics, "forced_expert0_rollout": forced_expert0, "comparison": comparison, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Quotient classes are offline training supervision only; inference routing and all candidate outputs are learned.", "Forced routing is a diagnostic control and is not a submission mode.", "No consumed independent seed or checked answer was used."]}
        summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({**comparison, "wall_seconds": summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": comparison, "router_diagnostics": diagnostics, "checkpoints": {k: v["checkpoint"] for k, v in results.items()}, "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
