"""Finite-data grokking feasibility for PyG max-MPNN and shared-context TCN."""

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
import psutil
import torch
import torch_geometric
import yaml
from safetensors.torch import save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
EXP100 = ROOT / "experiments/exp_100_width_scaling"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(EXP100))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def sha256_text(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest()
def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_510_multiarchitecture_grokking_feasibility.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-510-multiarchitecture-grokking-feasibility")
    args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text())
    if torch_geometric.__version__ != config["library"]["torch_geometric"]: raise RuntimeError("PyG version mismatch")
    base = load_module("base_for_exp510", ROOT / "experiments/exp_100_width_scaling/run.py")
    features = load_module("features_for_exp510", ROOT / "experiments/exp_100_width_scaling/model.py")
    graph = load_module("graph_for_exp510", ROOT / "experiments/exp_401_pyg_graph_message_passing/model.py")
    exp500 = load_module("exp500_for_exp510", ROOT / "experiments/exp_500_cellular_grokking_probe/run.py")
    torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    constructors = {
        "shared-context-tcn": lambda: features.SharedContextHornerTCN(**config["architectures"]["shared-context-tcn"]),
        "pyg-max-mpnn": lambda: graph.PyGRecurrentTypedMaxMPNN(**config["architectures"]["pyg-max-mpnn"]),
    }
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path); runner_hash = sha256(Path(__file__)); tcn_hash = sha256(ROOT / "experiments/exp_100_width_scaling/model.py"); graph_hash = sha256(ROOT / "experiments/exp_401_pyg_graph_message_passing/model.py"); commit = git_output("rev-parse", "HEAD")
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); process = psutil.Process(); total_wall = time.perf_counter(); total_cpu = time.process_time(); results = {}; nested_ids = {}

    def fixed_data(seed: int):
        data_config = config["data"]; split = base.split_primes(data_config["prime_bits_min"], data_config["prime_bits_max"], seed + 51000); train_primes = split["train"]; heldout_primes = split["validation"] + split["test"]
        train_sampler = base.TransitionSampler(9, data_config["prime_bits_min"], data_config["prime_bits_max"], train_primes, seed + 51001, data_config["composite_fraction"], data_config["random_state_fraction"])
        heldout_sampler = base.TransitionSampler(9, data_config["prime_bits_min"], data_config["prime_bits_max"], heldout_primes, seed + 51002, 0.0, data_config["random_state_fraction"])
        return train_sampler.transitions(data_config["train_examples"]), heldout_sampler.transitions(data_config["heldout_examples"]), train_primes, heldout_primes

    def train_arm(seed: int, architecture: str, constructor, train_data, heldout_data, train_primes, heldout_primes) -> dict:
        arm_dir = output / str(seed) / architecture; arm_dir.mkdir(parents=True, exist_ok=True)
        data_hash = sha256_text(exp500.dataset_sha256(train_data) + exp500.dataset_sha256(heldout_data) + json.dumps({"train_primes": train_primes, "heldout_primes": heldout_primes}))
        code_hash = tcn_hash if architecture == "shared-context-tcn" else graph_hash
        cache_key = sha256_text(json.dumps({"seed": seed, "architecture": architecture, "config": config_hash, "runner": runner_hash, "code": code_hash, "pyg": torch_geometric.__version__, "data": data_hash}, sort_keys=True)); cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == cache_key and (ROOT / payload["result"]["checkpoint"]["path"]).exists(): return {**payload["result"], "cached": True}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); model = constructor(); initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], betas=tuple(config["training"]["betas"]), eps=config["training"]["epsilon"], weight_decay=config["training"]["weight_decay"])
        index_generator = torch.Generator().manual_seed(seed + 51003); checkpoints = arm_dir / "checkpoints"; checkpoints.mkdir(exist_ok=True); curve = []; started = time.perf_counter(); cpu_started = time.process_time(); peak_rss = process.memory_info().rss; last_gradient = None; last_train_batch_loss = None

        def record(step: int) -> None:
            nonlocal peak_rss
            train_metrics = exp500.evaluate_fixed(model, train_data, 9, config["evaluation"]["evaluation_batch_size"], base, features); heldout_metrics = exp500.evaluate_fixed(model, heldout_data, 9, config["evaluation"]["evaluation_batch_size"], base, features)
            rollout_exact = None; rollout_invalid = None
            if step % config["evaluation"]["rollout_interval_steps"] == 0:
                sampler = base.TransitionSampler(9, config["data"]["prime_bits_min"], config["data"]["prime_bits_max"], heldout_primes, seed + 51004, 0.0, 0.0); rollout = base.evaluate_rollouts(model, sampler, config["evaluation"]["rollout_cases"]); rollout_exact = rollout["exact_accuracy"]; rollout_invalid = rollout["any_invalid_residue_rate"]
            norms = exp500.norm_metrics(model, initial, config["training"]["weight_decay"])
            row = {"step": step, "elapsed_wall_seconds": time.perf_counter() - started, "train_exact": train_metrics["exact_accuracy"], "heldout_exact": heldout_metrics["exact_accuracy"], "train_loss": train_metrics["loss"], "heldout_loss": heldout_metrics["loss"], "train_margin": train_metrics["mean_signed_margin"], "heldout_margin": heldout_metrics["mean_signed_margin"], "rollout_exact": rollout_exact, "rollout_invalid": rollout_invalid, "last_gradient_norm": last_gradient, "last_train_batch_loss": last_train_batch_loss, **norms}; curve.append(row)
            mlflow.log_metrics({key: value for key, value in row.items() if key != "step" and value is not None}, step=step); print(json.dumps({"seed": seed, "architecture": architecture, **row}), flush=True); peak_rss = max(peak_rss, process.memory_info().rss)

        record(0); state, multiplicand, modulus, input_bit, target, quotient = train_data
        for step in range(1, config["training"]["steps"] + 1):
            model.train(); optimizer.param_groups[0]["lr"] = config["training"]["learning_rate"] * min(1.0, step / max(1, config["training"]["warmup_steps"])); indices = torch.randint(len(state), (config["training"]["batch_size"],), generator=index_generator)
            logits, quotient_logits = model(features.make_features(state[indices], multiplicand[indices], modulus[indices], input_bit[indices], 9)); loss, stats = base.transition_loss(logits, quotient_logits, target[indices], quotient[indices], 9, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"])
            optimizer.zero_grad(set_to_none=True); loss.backward(); last_gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); last_train_batch_loss = stats["loss"]
            if step % config["evaluation"]["checkpoint_interval_steps"] == 0: save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, checkpoints / f"step-{step:05d}.safetensors", metadata={"seed": str(seed), "architecture": architecture, "step": str(step)})
            if step % config["evaluation"]["curve_interval_steps"] == 0: record(step)
        checkpoint = arm_dir / "weights.safetensors"; save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, checkpoint, metadata={"seed": str(seed), "architecture": architecture}); analysis = exp500.analyze_signature(curve, config["signature"]); curve_path = arm_dir / "curve.json"; curve_path.write_text(json.dumps(curve, indent=2) + "\n")
        result = {"seed": seed, "architecture": architecture, "weight_decay": config["training"]["weight_decay"], "cached": False, "dataset_sha256": data_hash, "train_prime_count": len(train_primes), "heldout_prime_count": len(heldout_primes), "parameters": model.parameter_count(), "curve": curve, "analysis": analysis, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "curve_path": str(curve_path.relative_to(ROOT)), "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "peak_rss_bytes": peak_rss}}
        cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n"); return result

    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "multiarchitecture-grokking-feasibility", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "pyg_version": torch_geometric.__version__})
        mlflow.log_params({"seeds": str(config["seeds"]), "steps": config["training"]["steps"], "weight_decay": config["training"]["weight_decay"], "config_sha256": config_hash, "runner_sha256": runner_hash, "tcn_sha256": tcn_hash, "graph_sha256": graph_hash, "pyg_version": torch_geometric.__version__}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for seed in config["seeds"]:
            train_data, heldout_data, train_primes, heldout_primes = fixed_data(seed); results[str(seed)] = {}
            for architecture, constructor in constructors.items():
                with mlflow.start_run(run_name=f"{seed}-{architecture}", nested=True) as nested:
                    nested_ids[f"{seed}-{architecture}"] = nested.info.run_id; result = train_arm(seed, architecture, constructor, train_data, heldout_data, train_primes, heldout_primes); results[str(seed)][architecture] = result; mlflow.log_metrics({"signature": float(result["analysis"]["signature"]), "final_train_exact": result["analysis"]["final_train_exact"], "final_heldout_exact": result["analysis"]["final_heldout_exact"], "final_rollout_exact": result["analysis"]["final_rollout_exact"], "wall_seconds": result["runtime"]["wall_seconds"]}); mlflow.set_tag("status", "completed")
        counts = {architecture: sum(int(results[str(seed)][architecture]["analysis"]["signature"]) for seed in config["seeds"]) for architecture in constructors}; eligible = [architecture for architecture, count in counts.items() if count >= config["gates"]["minimum_signature_seeds_per_architecture"]]; winner = max(eligible, key=lambda name: counts[name]) if eligible else None; decision = f"advance-{winner}-grokking-mechanism" if winner else "reject-multiarchitecture-grokking-signature"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "library": config["library"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "nested_run_ids": nested_ids}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "tcn_sha256": tcn_hash, "graph_sha256": graph_hash}, "results": results, "signature_counts": counts, "eligible": eligible, "winner": winner, "gates": config["gates"], "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - total_wall, "aggregate_cpu_seconds": time.process_time() - total_cpu, "torch_threads": config["training"]["torch_threads"]}, "limitations": ["Width-9 finite-data architecture feasibility only; no candidate promotion is authorized.", "Generated prime-disjoint evidence only; checked, consumed independent, and private answers were not read.", "Architectures are parameter-matched but not compute-matched; runtime is reported explicitly."]}
        summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"tcn_signature_seeds": counts["shared-context-tcn"], "graph_signature_seeds": counts["pyg-max-mpnn"], "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision, "winner": winner or "none"})
    print(json.dumps({"decision": decision, "signature_counts": counts, "analyses": {seed: {architecture: result["analysis"] for architecture, result in arms.items()} for seed, arms in results.items()}, "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
