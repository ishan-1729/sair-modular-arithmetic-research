"""Matched cellular, GCN, GAT, and typed max-MPNN successive-halving smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import torch
import yaml

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
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_400_graph_message_passing_smoke.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-400-graph-message-passing-smoke")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    exp310 = load_module("exp310_for_exp400", ROOT / "experiments/exp_310_chunk_shared_recurrent/run.py")
    exp200 = load_module("exp200_for_exp400", ROOT / "experiments/exp_200_width32_transfer_ab/run.py")
    cellular = load_module("cellular_for_exp400", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    graph = load_module("graph_for_exp400", Path(__file__).resolve().parent / "model.py")
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    constructors = {
        "cellular-control": lambda width: cellular.SharedDilatedConvGRU(width=width, **config["architectures"]["cellular"]),
        "gcn": lambda width: graph.RecurrentGCN(width=width, **config["architectures"]["gcn"]),
        "gat": lambda width: graph.RecurrentGAT(width=width, **config["architectures"]["gat"]),
        "typed-max-mpnn": lambda width: graph.RecurrentTypedMPNN(width=width, **config["architectures"]["mpnn"]),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path)
    runner_hash = sha256(Path(__file__))
    model_hash = sha256(Path(__file__).resolve().parent / "model.py")
    commit = git_output("rev-parse", "HEAD")
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    results = {}
    run_ids = {}

    def run_arm(name: str, constructor) -> dict:
        arm_dir = output / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        cache_key = sha256_text(json.dumps({"name": name, "config": config_hash, "runner": runner_hash, "model": model_hash, "seed": config["seed"]}, sort_keys=True))
        cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == cache_key and (ROOT / payload["result"]["checkpoint"]["path"]).exists():
                return {**payload["result"], "cached": True}
        result = exp310.train_arm(name, constructor, config, arm_dir)
        result["evaluation"]["zero_shot_width32"] = exp310.evaluate_width32(constructor, arm_dir / "weights.safetensors", config, exp200)
        result["cached"] = False
        cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n")
        return result

    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "graph-message-passing-successive-halving", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()})
        mlflow.log_params({"seed": config["seed"], "steps": config["training"]["steps"], "widths": str(config["training"]["widths"]), "config_sha256": config_hash, "runner_sha256": runner_hash, "model_sha256": model_hash})
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for name, constructor in constructors.items():
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id
                results[name] = run_arm(name, constructor)
                evaluation = results[name]["evaluation"]
                mlflow.log_metrics({"width16_transition": evaluation["width16_transition"]["exact_accuracy"], "width16_rollout": evaluation["width16_rollout"]["exact_accuracy"], "width32_transition": evaluation["zero_shot_width32"]["transition"]["exact_accuracy"], "width32_rollout": evaluation["zero_shot_width32"]["rollout"]["exact_accuracy"], "parameters": results[name]["parameters"], "training_wall_seconds": results[name]["training"]["seconds"]})
                mlflow.set_tag("status", "completed")
        control = results["cellular-control"]
        control_rollout = control["evaluation"]["width16_rollout"]["exact_accuracy"]
        control_zero = control["evaluation"]["zero_shot_width32"]["transition"]["exact_accuracy"]
        comparisons = {}
        for name in ("gcn", "gat", "typed-max-mpnn"):
            arm = results[name]
            evaluation = arm["evaluation"]
            comparisons[name] = {
                "width16_transition": evaluation["width16_transition"]["exact_accuracy"],
                "width16_rollout": evaluation["width16_rollout"]["exact_accuracy"],
                "width16_rollout_gain": evaluation["width16_rollout"]["exact_accuracy"] - control_rollout,
                "zero_width32_transition": evaluation["zero_shot_width32"]["transition"]["exact_accuracy"],
                "zero_width32_transition_gain": evaluation["zero_shot_width32"]["transition"]["exact_accuracy"] - control_zero,
                "zero_width32_rollout": evaluation["zero_shot_width32"]["rollout"]["exact_accuracy"],
                "parameter_ratio": arm["parameters"] / control["parameters"],
                "training_time_ratio": arm["training"]["seconds"] / control["training"]["seconds"],
            }
        gates = config["gates"]
        eligible = [name for name, value in comparisons.items() if value["width16_transition"] >= gates["minimum_graph_width16_transition"] and value["width16_rollout"] >= gates["minimum_graph_width16_rollout"] and (value["width16_rollout_gain"] >= gates["minimum_width16_rollout_gain_over_control"] or value["zero_width32_transition_gain"] >= gates["minimum_zero_width32_transition_gain_over_control"]) and value["parameter_ratio"] <= gates["maximum_parameter_ratio"] and value["training_time_ratio"] <= gates["maximum_training_time_ratio"]]
        winner = max(eligible, key=lambda name: (comparisons[name]["width16_rollout"], comparisons[name]["zero_width32_transition"])) if eligible else None
        decision = f"advance-{winner}-full-budget" if winner else "reject-exp400-graph-smoke-rung"
        summary = {
            "run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"],
            "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids},
            "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "model_sha256": model_hash},
            "arms": results, "comparisons": comparisons, "eligible": eligible, "winner": winner, "gates": gates, "decision": decision, "resource_budget": config["resource_budget"],
            "runtime": {"wall_seconds": time.perf_counter() - wall_started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "torch_threads": config["training"]["torch_threads"]},
            "limitations": ["This is a declared 200-update successive-halving rung, not a full-budget graph falsification.", "Generated identity-disjoint evidence only; checked, consumed independent, and private answers were not read.", "Multiscale positional edges are fixed generic relations and do not encode arithmetic operations."],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({"wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"], "eligible_graph_arms": len(eligible)})
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision, "winner": winner or "none"})
    print(json.dumps({"decision": decision, "winner": winner, "eligible": eligible, "comparisons": comparisons, "mlflow": summary["mlflow"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
