"""Evaluate a two-checkpoint cellular logit ensemble at width 64."""

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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


class LogitEnsemble(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module, alpha: float, width: int) -> None:
        super().__init__(); self.first = first; self.second = second; self.alpha = alpha; self.width = width
    def forward(self, features):
        first_logits, first_q = self.first(features); second_logits, second_q = self.second(features); return (1.0 - self.alpha) * first_logits + self.alpha * second_logits, (1.0 - self.alpha) * first_q + self.alpha * second_q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_327_cellular_width64_logit_ensemble.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-327-cellular-width64-logit-ensemble"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); path_a = (ROOT / config["checkpoint_a"]).resolve(); path_b = (ROOT / config["checkpoint_b"]).resolve()
    if sha256(path_a) != config["checkpoint_a_sha256"] or sha256(path_b) != config["checkpoint_b_sha256"]: raise RuntimeError("checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp327", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp327", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); split = exp300.bigint_prime_split({"data": config["data"]}, config["seed"] + 327); torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    def model(path: Path):
        result = cellular.SharedDilatedConvGRU(**config["architecture"]); result.load_state_dict(load_file(path)); result.eval(); return result
    first = model(path_a); second = model(path_b); ensemble = LogitEnsemble(model(path_a), model(path_b), config["ensemble"]["alpha_b"], 64); ensemble.eval(); arms = {"endpoint_a": first, "endpoint_b": second, "logit_ensemble": ensemble}; experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); started = time.perf_counter(); results = {}; commit = git_output("rev-parse", "HEAD")
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "cellular-width64-logit-ensemble", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()}); mlflow.log_params({"checkpoint_a_sha256": sha256(path_a), "checkpoint_b_sha256": sha256(path_b), "alpha_b": config["ensemble"]["alpha_b"], "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__))}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-327-cellular-width64-logit-ensemble.md"), artifact_path="preregistration")
        for label, active in arms.items():
            transition = exp300.evaluate_transitions(active, exp300.BigIntSampler(split["test"], config["seed"] + 32750), config["evaluation"]["transition_cases_per_arm"]); rollout = exp300.evaluate_rollouts(active, exp300.BigIntSampler(split["test"], config["seed"] + 32751), config["evaluation"]["rollout_cases_per_arm"]); results[label] = {"transition": transition, "rollout": rollout}; mlflow.log_metrics({f"{label}_transition": transition["exact_accuracy"], f"{label}_rollout": rollout["exact_accuracy"], f"{label}_invalid": rollout["any_invalid_residue_rate"]}); print(json.dumps({"arm": label, **results[label]}), flush=True)
        endpoint = max(results["endpoint_a"]["rollout"]["exact_accuracy"], results["endpoint_b"]["rollout"]["exact_accuracy"]); active = results["logit_ensemble"]; gain = active["rollout"]["exact_accuracy"] - endpoint; gates = config["gates"]; passed = active["transition"]["exact_accuracy"] >= gates["minimum_ensemble_transition"] and active["rollout"]["exact_accuracy"] >= gates["minimum_ensemble_rollout"] and gain >= gates["minimum_gain_over_best_endpoint"] and active["rollout"]["any_invalid_residue_rate"] <= gates["maximum_ensemble_invalid_rollout_rate"]; decision = "advance-cellular-ensemble-independent-tier5" if passed else "reject-cellular-width64-logit-ensemble"; output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); checkpoint = output / "weights.safetensors"; state_a = load_file(path_a); state_b = load_file(path_b); combined = {**{f"model_a.{key}": value.contiguous() for key, value in state_a.items()}, **{f"model_b.{key}": value.contiguous() for key, value in state_b.items()}}; save_file(combined, checkpoint, metadata={"alpha_b": str(config["ensemble"]["alpha_b"])}); summary = {"run_id": config["run_id"], "status": "completed", "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "checkpoint_a_sha256": sha256(path_a), "checkpoint_b_sha256": sha256(path_b)}, "arms": results, "comparison": {"gain_over_best_endpoint": gain}, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "gates": {**gates, "passed": passed}, "decision": decision, "runtime": {"wall_seconds": time.perf_counter() - started, "device": "cpu", "torch_threads": 4}, "limitations": ["Generated selection evidence only; no checked, consumed independent, or private answers were read.", "The ensemble doubles Tier-5 specialist parameters and forward compute."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"ensemble_transition": active["transition"]["exact_accuracy"], "ensemble_rollout": active["rollout"]["exact_accuracy"], "gain_over_endpoint": gain, "wall_seconds": summary["runtime"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": summary["comparison"], "arms": results, "checkpoint": summary["checkpoint"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
