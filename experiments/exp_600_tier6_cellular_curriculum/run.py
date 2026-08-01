"""Progressive cellular curriculum from the frozen Tier-5 checkpoint toward Tier 6."""

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
import yaml
from safetensors.torch import load_file, save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_600_tier6_cellular_curriculum.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-600-tier6-cellular-curriculum")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")

    exp300 = load_module("exp300_for_exp600", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    exp320 = load_module("exp320_for_exp600", ROOT / "experiments/exp_320_cellular_width64_curriculum/run.py")
    cellular = load_module("cellular_for_exp600", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    seed = int(config["seed"])
    widths = list(config["training"]["widths"])
    eval_widths = sorted({config["evaluation"]["retained_width"], config["evaluation"]["trained_width"], config["evaluation"]["target_width"]})
    all_widths = sorted(set(widths + eval_widths))

    def make_split(width: int) -> dict:
        local = {"data": {"prime_bits_min": width - config["training"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
        return exp300.bigint_prime_split(local, seed + 60000 + width)

    splits = {width: make_split(width) for width in all_widths}
    dataset_hash = sha256_text(json.dumps(splits, sort_keys=True))
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner_hash = sha256(Path(__file__))
    config_hash = sha256(config_path)
    started = time.perf_counter()
    cpu_started = time.process_time()
    process = psutil.Process()

    def new_model(width: int):
        model = cellular.SharedDilatedConvGRU(width=width, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"])
        model.load_state_dict(load_file(source))
        return model

    def evaluate(model, width: int) -> dict:
        model.width = width
        transition_sampler = exp300.BigIntSampler(splits[width]["test"], seed + 60500 + width)
        rollout_sampler = exp300.BigIntSampler(splits[width]["test"], seed + 60600 + width)
        return {
            "transition": exp300.evaluate_transitions(model, transition_sampler, config["evaluation"]["transition_cases"]),
            "rollout": exp300.evaluate_rollouts(model, rollout_sampler, config["evaluation"]["rollout_cases"]),
        }

    cache_key = sha256_text(json.dumps({"checkpoint": sha256(source), "config": config_hash, "dataset": dataset_hash, "width": config["evaluation"]["control_width"]}, sort_keys=True))
    control_cache = output / f"source-control-{cache_key}.json"
    if control_cache.exists():
        control = json.loads(control_cache.read_text())
    else:
        control_model = new_model(config["evaluation"]["control_width"])
        control_model.eval()
        control = evaluate(control_model, config["evaluation"]["control_width"])
        control_cache.write_text(json.dumps(control, indent=2) + "\n")

    model = new_model(max(widths))
    samplers = {width: exp300.BigIntSampler(splits[width]["train"], seed + 60100 + width) for width in widths}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"])
    selector = torch.Generator().manual_seed(seed + 6002)
    pool_size = config["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"]
    history = []
    peak_rss = process.memory_info().rss
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    commit = git_output("rev-parse", "HEAD")

    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "tier6-progressive-cellular-smoke", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"})
        mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "widths": str(widths), "source_checkpoint_sha256": sha256(source), "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash})
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        model.train()
        for step in range(1, config["training"]["steps"] + 1):
            width = widths[(step - 1) % len(widths)]
            values = samplers[width].transitions(pool_size)
            state, multiplicand, modulus, input_bit, target, quotient = exp320.mine(values, config["training"]["batch_size"], config, selector)
            model.width = width
            logits, quotient_logits = model(exp300.features(state, multiplicand, modulus, input_bit, width))
            loss, stats = exp300.transition_loss(logits, quotient_logits, target, quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 25 == 0 or step == config["training"]["steps"]:
                peak_rss = max(peak_rss, process.memory_info().rss)
                record = {"step": step, "width": width, "elapsed_wall_seconds": time.perf_counter() - started, "gradient_norm": gradient, **stats}
                history.append(record)
                mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient}, step=step)
                print(json.dumps(record), flush=True)

        model.eval()
        checkpoint = output / "weights.safetensors"
        save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint)
        trained = {str(width): evaluate(model, width) for width in eval_widths}
        target = trained[str(config["evaluation"]["target_width"])]
        source_rollout = control["rollout"]["exact_accuracy"]
        target_rollout = target["rollout"]["exact_accuracy"]
        gates = config["gates"]
        passed = (
            target["transition"]["exact_accuracy"] >= gates["minimum_width128_transition"]
            and target_rollout >= gates["minimum_width128_rollout"]
            and target_rollout - source_rollout >= gates["minimum_width128_gain_over_source"]
            and trained[str(config["evaluation"]["trained_width"])]["rollout"]["exact_accuracy"] >= gates["minimum_width112_rollout"]
            and trained[str(config["evaluation"]["retained_width"])]["rollout"]["exact_accuracy"] >= gates["minimum_retained_width64_rollout"]
            and target["rollout"]["any_invalid_residue_rate"] <= gates["maximum_width128_invalid_rollout_rate"]
        )
        decision = "advance-direct-width128" if passed else "reject-tier6-progressive-curriculum-smoke"
        summary = {
            "run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"],
            "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id},
            "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash, "source_checkpoint_sha256": sha256(source)},
            "control_width128": control,
            "trained": {"parameters": sum(p.numel() for p in model.parameters()), "history": history, "evaluation": trained, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}},
            "comparison": {"source_width128_rollout": source_rollout, "challenger_width128_rollout": target_rollout, "width128_rollout_gain": target_rollout - source_rollout},
            "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"],
            "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "peak_rss_bytes": peak_rss, "torch_threads": config["training"]["torch_threads"]},
            "limitations": ["Smoke-rung generated evidence only; no checked, consumed independent, or private answers were read.", "The profile's 20-rollout result is throughput context, not selection evidence.", "Python bigint arithmetic is confined to offline generation, labels, mining, and evaluation truth; the model receives bit tensors."],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({"source_width128_rollout": source_rollout, "challenger_width128_rollout": target_rollout, "width128_rollout_gain": target_rollout - source_rollout, "width112_rollout": trained[str(config["evaluation"]["trained_width"])]["rollout"]["exact_accuracy"], "retained_width64_rollout": trained[str(config["evaluation"]["retained_width"])]["rollout"]["exact_accuracy"], "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]})
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": summary["comparison"], "evaluation": trained, "checkpoint": summary["trained"]["checkpoint"], "runtime": summary["runtime"], "mlflow": summary["mlflow"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
