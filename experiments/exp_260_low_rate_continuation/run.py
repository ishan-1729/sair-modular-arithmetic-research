"""Fresh-split low-rate continuation of the strongest uniform width-32 checkpoint."""

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
from safetensors.torch import load_file

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


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_260_low_rate_continuation.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-260-low-rate-continuation"); args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp200 = load_module("exp200_for_exp260", ROOT / "experiments/exp_200_width32_transfer_ab/run.py"); exp204 = load_module("exp204_for_exp260", ROOT / "experiments/exp_204_official_distribution_finetune/run.py"); exp250 = load_module("exp250_for_exp260", ROOT / "experiments/exp_250_online_failure_mining/run.py")
    split = exp204.official_numeric_prime_split(17, 32, config["seed"] + 260); torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); started = time.perf_counter(); output = args.output.resolve(); eval_seed = config["seed"] + 75000
    source_model = exp100_model.SharedContextHornerTCN(**config["model"]); source_model.load_state_dict(load_file(source)); source_model.eval()
    source_transition = exp100.evaluate_transitions(source_model, exp200.ScalableTransitionSampler(32, 17, 32, split["test"], eval_seed, 0.0, 0.15), config["evaluation"]["transition_cases"]); source_rollout = exp100.evaluate_rollouts(source_model, exp200.ScalableTransitionSampler(32, 17, 32, split["test"], eval_seed + 1, 0.0, 0.0), config["evaluation"]["rollout_cases"])
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "low-rate-uniform-continuation", "repository_commit": git_output("rev-parse", "HEAD"), "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": exp100.CHALLENGE_COMMIT}); mlflow.log_params({"seed": config["seed"], "steps": config["training"]["steps"], "source_checkpoint_sha256": sha256(source), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__))}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        trained = exp250.train_arm("low-rate-uniform-continuation", False, config, source, split, exp200, output / "trained")
        transition = trained["evaluation"]["heldout_transition"]; rollout = trained["evaluation"]["heldout_rollout"]; gain = rollout["exact_accuracy"] - source_rollout["exact_accuracy"]; gates = config["gates"]; passed = gain >= gates["minimum_rollout_gain"] and transition["exact_accuracy"] >= gates["minimum_transition"] and rollout["exact_accuracy"] >= gates["minimum_rollout"] and rollout["any_invalid_residue_rate"] <= gates["maximum_invalid_rollout_rate"]; decision = "advance-low-rate-independent-tier4" if passed else "reject-further-fixed-width-continuation"
        summary = {"run_id": config["run_id"], "status": "completed", "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": git_output("rev-parse", "HEAD"), "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": exp100.CHALLENGE_COMMIT, "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "source_checkpoint_sha256": sha256(source)}, "source_evaluation": {"transition": source_transition, "rollout": source_rollout}, "trained": trained, "comparison": {"transition_change": transition["exact_accuracy"] - source_transition["exact_accuracy"], "rollout_gain": gain}, "gates": {**gates, "passed": passed}, "decision": decision, "resource": {"wall_seconds": time.perf_counter() - started, "cpu_minute_cap": config["resource_cap"]["cpu_minutes"]}, "limitations": ["Fresh official-like identities only; no consumed independent or checked answer was read."]}; summary_path = output / "summary.json"; summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"source_transition": source_transition["exact_accuracy"], "source_rollout": source_rollout["exact_accuracy"], "trained_transition": transition["exact_accuracy"], "trained_rollout": rollout["exact_accuracy"], "rollout_gain": gain, "wall_seconds": summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "source_transition": source_transition["exact_accuracy"], "source_rollout": source_rollout["exact_accuracy"], "trained_transition": transition["exact_accuracy"], "trained_rollout": rollout["exact_accuracy"], "rollout_gain": gain, "checkpoint": trained["checkpoint"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
