"""Fresh-data interpolation test for two width-64 cellular checkpoints."""

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
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_325_cellular_width64_soup.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-325-cellular-width64-soup"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); path_a = (ROOT / config["checkpoint_a"]).resolve(); path_b = (ROOT / config["checkpoint_b"]).resolve()
    if sha256(path_a) != config["checkpoint_a_sha256"] or sha256(path_b) != config["checkpoint_b_sha256"]: raise RuntimeError("checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp325", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp325", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); local = {"data": config["data"]}; split = exp300.bigint_prime_split(local, config["seed"] + 325); state_a = load_file(path_a); state_b = load_file(path_b); torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); started = time.perf_counter(); results = {}; states = {}; commit = git_output("rev-parse", "HEAD")
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "cellular-width64-weight-soup", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()}); mlflow.log_params({"checkpoint_a_sha256": sha256(path_a), "checkpoint_b_sha256": sha256(path_b), "alphas": str(config["interpolation"]["alphas"]), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__))}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-325-cellular-width64-weight-soup.md"), artifact_path="preregistration")
        for alpha in config["interpolation"]["alphas"]:
            label = f"alpha-{alpha:.2f}"; averaged = {key: ((1.0 - alpha) * state_a[key] + alpha * state_b[key]).contiguous() for key in state_a}; model = cellular.SharedDilatedConvGRU(**config["architecture"]); model.load_state_dict(averaged); model.eval(); transition = exp300.evaluate_transitions(model, exp300.BigIntSampler(split["test"], config["seed"] + 32550), config["evaluation"]["transition_cases_per_alpha"]); rollout = exp300.evaluate_rollouts(model, exp300.BigIntSampler(split["test"], config["seed"] + 32551), config["evaluation"]["rollout_cases_per_alpha"]); results[label] = {"alpha": alpha, "transition": transition, "rollout": rollout}; states[label] = averaged; mlflow.log_metrics({f"{label}_transition": transition["exact_accuracy"], f"{label}_rollout": rollout["exact_accuracy"], f"{label}_invalid": rollout["any_invalid_residue_rate"]}); print(json.dumps({"arm": label, "transition": transition, "rollout": rollout}), flush=True)
        best_label, best = max(results.items(), key=lambda item: (item[1]["rollout"]["exact_accuracy"], item[1]["transition"]["exact_accuracy"])); endpoint = max(results["alpha-0.00"]["rollout"]["exact_accuracy"], results["alpha-1.00"]["rollout"]["exact_accuracy"]); gain = best["rollout"]["exact_accuracy"] - endpoint; gates = config["gates"]; passed = best["rollout"]["exact_accuracy"] >= gates["minimum_best_rollout"] and gain >= gates["minimum_gain_over_best_endpoint"] and best["transition"]["exact_accuracy"] >= gates["minimum_best_transition"] and best["rollout"]["any_invalid_residue_rate"] <= gates["maximum_best_invalid_rollout_rate"]; decision = "advance-cellular-soup-independent-tier5" if passed else "reject-cellular-width64-weight-soup"; output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); checkpoint = output / "weights.safetensors"; save_file(states[best_label], checkpoint, metadata={"alpha": str(best["alpha"]), "source_a": sha256(path_a), "source_b": sha256(path_b)}); summary = {"run_id": config["run_id"], "status": "completed", "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "checkpoint_a_sha256": sha256(path_a), "checkpoint_b_sha256": sha256(path_b)}, "arms": results, "selection": {"best_label": best_label, "best_alpha": best["alpha"], "gain_over_best_endpoint": gain}, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "gates": {**gates, "passed": passed}, "decision": decision, "runtime": {"wall_seconds": time.perf_counter() - started, "device": "cpu", "torch_threads": 4}, "limitations": ["Fresh generated selection data only; no checked, consumed independent, or private answers were read.", "Interpolation adds no inference parameters or operations."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"best_transition": best["transition"]["exact_accuracy"], "best_rollout": best["rollout"]["exact_accuracy"], "gain_over_endpoint": gain, "wall_seconds": summary["runtime"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision, "best_alpha": str(best["alpha"])})
    print(json.dumps({"decision": decision, "selection": summary["selection"], "best": best, "checkpoint": summary["checkpoint"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
