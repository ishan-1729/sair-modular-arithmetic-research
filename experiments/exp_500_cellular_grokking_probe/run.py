"""Three-seed finite-data grokking probe for the cellular transition family."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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
from safetensors.torch import save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
EXP100 = ROOT / "experiments/exp_100_width_scaling"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP100))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def dataset_sha256(data: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in data:
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_fixed(model, data, width: int, batch_size: int, base, cellular) -> dict:
    totals = {"cases": 0, "correct": 0, "bce_sum": 0.0, "quotient_sum": 0.0, "margin_sum": 0.0}
    bit_bce = torch.zeros(width)
    state, multiplicand, modulus, input_bit, target, quotient = data
    model.eval()
    for start in range(0, len(state), batch_size):
        stop = min(len(state), start + batch_size)
        sl = slice(start, stop)
        logits, quotient_logits = model(cellular.make_features(state[sl], multiplicand[sl], modulus[sl], input_bit[sl], width))
        target_bits = base.ints_to_bits(target[sl], width)
        per_bit = F.binary_cross_entropy_with_logits(logits, target_bits, reduction="none")
        signed = (target_bits.mul(2).sub(1)) * logits
        size = stop - start
        totals["cases"] += size
        totals["correct"] += int((logits.ge(0) == target_bits.bool()).all(1).sum())
        totals["bce_sum"] += float(per_bit.sum())
        totals["quotient_sum"] += float(F.cross_entropy(quotient_logits, quotient[sl], reduction="sum"))
        totals["margin_sum"] += float(signed.sum())
        bit_bce += per_bit.sum(0).cpu()
    cases = totals["cases"]
    mean_bce = totals["bce_sum"] / (cases * width)
    worst_bit_bce = float((bit_bce / cases).max())
    quotient_ce = totals["quotient_sum"] / cases
    return {"cases": cases, "correct": totals["correct"], "exact_accuracy": totals["correct"] / cases, "loss": mean_bce + 0.45 * worst_bit_bce + 0.10 * quotient_ce, "mean_bce": mean_bce, "worst_bit_bce": worst_bit_bce, "quotient_ce": quotient_ce, "mean_signed_margin": totals["margin_sum"] / (cases * width)}


def norm_metrics(model, initial: dict[str, torch.Tensor], weight_decay: float) -> dict:
    norm_sq = 0.0
    delta_sq = 0.0
    for name, parameter in model.named_parameters():
        value = parameter.detach().double()
        norm_sq += float(value.square().sum())
        delta_sq += float((value - initial[name].double()).square().sum())
    return {"parameter_l2_norm": math.sqrt(norm_sq), "distance_from_initial_l2": math.sqrt(delta_sq), "weight_decay_penalty": 0.5 * weight_decay * norm_sq}


def analyze_signature(curve: list[dict], signature: dict) -> dict:
    memorization = next((row for row in curve if row["train_exact"] >= signature["minimum_train_exact_at_memorization"] and row["heldout_exact"] <= signature["maximum_heldout_exact_at_memorization"] and row["train_exact"] - row["heldout_exact"] >= signature["minimum_train_heldout_gap_at_memorization"]), None)
    onset = None
    if memorization is not None:
        for row in curve:
            if row["rollout_exact"] is None or row["step"] < memorization["step"] + signature["minimum_delay_after_memorization_steps"]:
                continue
            if row["train_exact"] >= signature["minimum_late_train_exact"] and row["heldout_exact"] >= signature["minimum_late_heldout_exact"] and row["heldout_exact"] - memorization["heldout_exact"] >= signature["minimum_heldout_gain_after_memorization"] and row["rollout_exact"] >= signature["minimum_late_rollout_exact"] and row["rollout_exact"] - curve[0]["rollout_exact"] >= signature["minimum_rollout_gain_from_initial"]:
                onset = row
                break
    max_heldout = max(row["heldout_exact"] for row in curve)
    final = curve[-1]
    semi = memorization is not None and final["heldout_exact"] - memorization["heldout_exact"] >= signature["minimum_heldout_gain_after_memorization"] and onset is None
    ungrokking = max_heldout >= signature["minimum_late_heldout_exact"] and final["heldout_exact"] <= max_heldout - signature["minimum_heldout_gain_after_memorization"]
    return {"signature": onset is not None, "memorization_step": memorization["step"] if memorization else None, "memorization_heldout_exact": memorization["heldout_exact"] if memorization else None, "onset_step": onset["step"] if onset else None, "semi_grokking": semi, "ungrokking": ungrokking, "maximum_heldout_exact": max_heldout, "final_train_exact": final["train_exact"], "final_heldout_exact": final["heldout_exact"], "final_rollout_exact": final["rollout_exact"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_500_cellular_grokking_probe.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-500-cellular-grokking-probe")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    base = load_module("base_for_exp500", ROOT / "experiments/exp_100_width_scaling/run.py")
    cellular = load_module("cellular_for_exp500", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path)
    runner_hash = sha256(Path(__file__))
    cellular_hash = sha256(ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    commit = git_output("rev-parse", "HEAD")
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    process = psutil.Process()
    total_wall = time.perf_counter()
    total_cpu = time.process_time()
    results = {}
    nested_ids = {}

    def fixed_data(seed: int):
        data_config = config["data"]
        split = base.split_primes(data_config["prime_bits_min"], data_config["prime_bits_max"], seed + 50000)
        train_primes = split["train"]
        heldout_primes = split["validation"] + split["test"]
        train_sampler = base.TransitionSampler(config["model"]["width"], data_config["prime_bits_min"], data_config["prime_bits_max"], train_primes, seed + 50001, data_config["composite_fraction"], data_config["random_state_fraction"])
        heldout_sampler = base.TransitionSampler(config["model"]["width"], data_config["prime_bits_min"], data_config["prime_bits_max"], heldout_primes, seed + 50002, 0.0, data_config["random_state_fraction"])
        return train_sampler.transitions(data_config["train_examples"]), heldout_sampler.transitions(data_config["heldout_examples"]), train_primes, heldout_primes

    def train_arm(seed: int, arm: str, weight_decay: float, train_data, heldout_data, train_primes, heldout_primes) -> dict:
        arm_dir = output / str(seed) / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        data_hash = sha256_text(dataset_sha256(train_data) + dataset_sha256(heldout_data) + json.dumps({"train_primes": train_primes, "heldout_primes": heldout_primes}))
        cache_key = sha256_text(json.dumps({"seed": seed, "arm": arm, "weight_decay": weight_decay, "config": config_hash, "runner": runner_hash, "cellular": cellular_hash, "data": data_hash}, sort_keys=True))
        cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == cache_key and (ROOT / payload["result"]["checkpoint"]["path"]).exists():
                return {**payload["result"], "cached": True}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = cellular.SharedDilatedConvGRU(**config["model"])
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], betas=tuple(config["training"]["betas"]), eps=config["training"]["epsilon"], weight_decay=weight_decay)
        index_generator = torch.Generator().manual_seed(seed + 50003)
        checkpoints = arm_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        curve = []
        started = time.perf_counter(); cpu_started = time.process_time(); peak_rss = process.memory_info().rss; last_gradient = None; last_train_batch_loss = None

        def record(step: int) -> None:
            nonlocal peak_rss
            train_metrics = evaluate_fixed(model, train_data, config["model"]["width"], config["evaluation"]["evaluation_batch_size"], base, cellular)
            heldout_metrics = evaluate_fixed(model, heldout_data, config["model"]["width"], config["evaluation"]["evaluation_batch_size"], base, cellular)
            rollout_exact = None
            rollout_invalid = None
            if step % config["evaluation"]["rollout_interval_steps"] == 0:
                sampler = base.TransitionSampler(config["model"]["width"], config["data"]["prime_bits_min"], config["data"]["prime_bits_max"], heldout_primes, seed + 50004, 0.0, 0.0)
                rollout = base.evaluate_rollouts(model, sampler, config["evaluation"]["rollout_cases"])
                rollout_exact = rollout["exact_accuracy"]
                rollout_invalid = rollout["any_invalid_residue_rate"]
            norms = norm_metrics(model, initial, weight_decay)
            row = {"step": step, "elapsed_wall_seconds": time.perf_counter() - started, "train_exact": train_metrics["exact_accuracy"], "heldout_exact": heldout_metrics["exact_accuracy"], "train_loss": train_metrics["loss"], "heldout_loss": heldout_metrics["loss"], "train_margin": train_metrics["mean_signed_margin"], "heldout_margin": heldout_metrics["mean_signed_margin"], "rollout_exact": rollout_exact, "rollout_invalid": rollout_invalid, "last_gradient_norm": last_gradient, "last_train_batch_loss": last_train_batch_loss, **norms}
            curve.append(row)
            mlflow.log_metrics({key: value for key, value in row.items() if key not in {"step"} and value is not None}, step=step)
            print(json.dumps({"seed": seed, "arm": arm, **row}), flush=True)
            peak_rss = max(peak_rss, process.memory_info().rss)

        record(0)
        state, multiplicand, modulus, input_bit, target, quotient = train_data
        for step in range(1, config["training"]["steps"] + 1):
            model.train()
            scale = min(1.0, step / max(1, config["training"]["warmup_steps"]))
            optimizer.param_groups[0]["lr"] = config["training"]["learning_rate"] * scale
            indices = torch.randint(len(state), (config["training"]["batch_size"],), generator=index_generator)
            logits, quotient_logits = model(cellular.make_features(state[indices], multiplicand[indices], modulus[indices], input_bit[indices], config["model"]["width"]))
            loss, stats = base.transition_loss(logits, quotient_logits, target[indices], quotient[indices], config["model"]["width"], config["training"]["worst_bit_weight"], config["training"]["quotient_weight"])
            optimizer.zero_grad(set_to_none=True); loss.backward(); last_gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); last_train_batch_loss = stats["loss"]
            if step % config["evaluation"]["checkpoint_interval_steps"] == 0:
                save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, checkpoints / f"step-{step:05d}.safetensors", metadata={"seed": str(seed), "arm": arm, "step": str(step)})
            if step % config["evaluation"]["curve_interval_steps"] == 0:
                record(step)
        checkpoint = arm_dir / "weights.safetensors"
        save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, checkpoint, metadata={"seed": str(seed), "arm": arm, "steps": str(config["training"]["steps"])})
        analysis = analyze_signature(curve, config["signature"])
        curve_path = arm_dir / "curve.json"
        curve_path.write_text(json.dumps(curve, indent=2) + "\n")
        result = {"seed": seed, "arm": arm, "weight_decay": weight_decay, "cached": False, "dataset_sha256": data_hash, "train_prime_count": len(train_primes), "heldout_prime_count": len(heldout_primes), "parameters": model.parameter_count(), "curve": curve, "analysis": analysis, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "curve_path": str(curve_path.relative_to(ROOT)), "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "peak_rss_bytes": peak_rss}}
        cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n")
        return result

    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "cellular-grokking-multi-seed", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()})
        mlflow.log_params({"seeds": str(config["seeds"]), "steps": config["training"]["steps"], "train_examples": config["data"]["train_examples"], "heldout_examples": config["data"]["heldout_examples"], "config_sha256": config_hash, "runner_sha256": runner_hash, "cellular_model_sha256": cellular_hash})
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/research/2026-07-12-grokking-delayed-generalization.md"), artifact_path="research")
        for seed in config["seeds"]:
            train_data, heldout_data, train_primes, heldout_primes = fixed_data(seed)
            results[str(seed)] = {}
            for arm, weight_decay in config["training"]["weight_decays"].items():
                with mlflow.start_run(run_name=f"{seed}-{arm}", nested=True) as nested:
                    nested_ids[f"{seed}-{arm}"] = nested.info.run_id
                    result = train_arm(seed, arm, weight_decay, train_data, heldout_data, train_primes, heldout_primes)
                    results[str(seed)][arm] = result
                    mlflow.log_metrics({"signature": float(result["analysis"]["signature"]), "final_train_exact": result["analysis"]["final_train_exact"], "final_heldout_exact": result["analysis"]["final_heldout_exact"], "final_rollout_exact": result["analysis"]["final_rollout_exact"], "wall_seconds": result["runtime"]["wall_seconds"]})
                    mlflow.set_tag("status", "completed")
        counts = {arm: sum(int(results[str(seed)][arm]["analysis"]["signature"]) for seed in config["seeds"]) for arm in config["training"]["weight_decays"]}
        decay_count = counts["decay-1e-2"]
        control_count = counts["no-decay-control"]
        passed = decay_count >= config["gates"]["minimum_decay_signature_seeds"] and (not config["gates"]["decay_signature_must_exceed_control_count"] or decay_count > control_count)
        decision = "advance-grokking-mechanism-tier5-ab" if passed else "reject-cellular-grokking-signature"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "nested_run_ids": nested_ids}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "cellular_model_sha256": cellular_hash}, "results": results, "signature_counts": counts, "gates": {**config["gates"], "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - total_wall, "aggregate_cpu_seconds": time.process_time() - total_cpu, "torch_threads": config["training"]["torch_threads"]}, "limitations": ["Width-9 finite-data feasibility probe; a positive result would not itself establish Tier-5 improvement.", "Generated prime-disjoint evidence only; checked, consumed independent, and private answers were not read.", "The no-decay versus 1e-2 contrast is project-defined and is not claimed to reproduce every original paper hyperparameter."]}
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({"decay_signature_seeds": decay_count, "control_signature_seeds": control_count, "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]})
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "signature_counts": counts, "analyses": {seed: {arm: result["analysis"] for arm, result in arms.items()} for seed, arms in results.items()}, "mlflow": summary["mlflow"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
