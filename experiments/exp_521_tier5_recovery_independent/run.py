"""Evaluate the frozen recovery checkpoint on five untouched Tier-5 generator seeds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
import yaml
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]; UPSTREAM = ROOT / "sources/github/modular-arithmetic-challenge/src"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(UPSTREAM))
from modchallenge.config import TIERS  # noqa: E402
from modchallenge.testgen.generator import _generate_tier_cases  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def predict_cases(model, cases, exp300) -> dict:
    modulus = [int(case.p) for case in cases]; multiplier = [int(case.a) % prime for case, prime in zip(cases, modulus)]; multiplicand = [int(case.b) % prime for case, prime in zip(cases, modulus)]; expected = [int(case.expected) for case in cases]; state = [0] * len(cases); started = time.perf_counter()
    for step in range(model.width):
        input_bit = [(value >> (model.width - 1 - step)) & 1 for value in multiplier]; logits, _ = model(exp300.features(state, multiplicand, modulus, input_bit, model.width)); rows = logits.ge(0).tolist(); state = [sum(int(row[index]) << index for index in range(model.width)) for row in rows]
    elapsed = time.perf_counter() - started; correct = [predicted == truth for predicted, truth in zip(state, expected)]; invalid = [predicted >= prime for predicted, prime in zip(state, modulus)]; per_prime = defaultdict(lambda: [0, 0])
    for prime, ok in zip(modulus, correct): per_prime[prime][0] += int(ok); per_prime[prime][1] += 1
    return {"cases": len(cases), "correct": sum(correct), "exact_accuracy": sum(correct) / len(cases), "invalid_residues": sum(invalid), "inference_seconds": elapsed, "per_prime": {str(prime): {"cases": values[1], "correct": values[0], "exact_accuracy": values[0] / values[1]} for prime, values in sorted(per_prime.items())}, "errors": [{"index": index, "p": modulus[index], "predicted": state[index], "expected": expected[index]} for index, ok in enumerate(correct) if not ok]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_521_tier5_recovery_independent.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-521-tier5-recovery-independent.json"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); checkpoint = (ROOT / config["checkpoint"]).resolve()
    if sha256(checkpoint) != config["checkpoint_sha256"]: raise RuntimeError("frozen checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp521", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp521", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    torch.set_num_threads(config["resource_budget"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); model = cellular.SharedDilatedConvGRU(**config["architecture"]); model.load_state_dict(load_file(checkpoint)); model.eval(); experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); results = {}; started = time.perf_counter(); cpu_started = time.process_time()
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "cellular-independent-tier5-gate", "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "checked_public_used": "false"}); mlflow.log_params({"checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(config_path), "tier": 5, "seeds": len(config["evaluation"]["seeds"]), "cases_per_seed": config["evaluation"]["cases_per_seed"]}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for seed_text in config["evaluation"]["seeds"]:
            seed = seed_text.encode(); seed_hash = hashlib.sha256(seed).hexdigest(); cases = _generate_tier_cases(tier=TIERS[5], num_cases=config["evaluation"]["cases_per_seed"], num_primes=config["evaluation"]["primes_per_seed"], edge_cases=config["evaluation"]["edge_cases_per_seed"], seed=seed); results[seed_hash[:16]] = {"seed_sha256": seed_hash, **predict_cases(model, cases, exp300)}
        total_cases = sum(item["cases"] for item in results.values()); total_correct = sum(item["correct"] for item in results.values()); aggregate = total_correct / total_cases; passing_seeds = sum(item["exact_accuracy"] >= config["gates"]["per_seed_passing_accuracy"] for item in results.values()); minimum_seed = min(item["exact_accuracy"] for item in results.values()); passed = aggregate >= config["gates"]["minimum_aggregate_accuracy"] and passing_seeds >= config["gates"]["minimum_passing_seeds"] and minimum_seed >= config["gates"]["minimum_single_seed_accuracy"]; decision = "promote-recovery-tier5-integration" if passed else "reject-recovery-tier5-endpoint"
        summary = {"run_id": config["run_id"], "status": "completed", "checkpoint": {"path": config["checkpoint"], "sha256": sha256(checkpoint)}, "generator": {"implementation": "modchallenge.testgen.generator._generate_tier_cases", "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"}, "checked_public_used": False, "tier": 5, "seeds": results, "aggregate": {"cases": total_cases, "correct": total_correct, "exact_accuracy": aggregate, "passing_seeds": passing_seeds, "minimum_seed_accuracy": minimum_seed}, "gates": {**config["gates"], "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "device": "cpu", "torch_threads": config["resource_budget"]["torch_threads"]}, "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "limitations": ["Five official-generator seeds cannot reproduce the unknowable private seed.", "Tier 5 only; full candidate integration remains required.", "Checked and private Tier-5 answers were not read."]}; args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"tier5_exact": aggregate, "passing_seeds": passing_seeds, "minimum_seed_accuracy": minimum_seed, "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(args.output), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "aggregate": summary["aggregate"], "per_seed": {key: value["exact_accuracy"] for key, value in results.items()}, "mlflow_run_id": summary["mlflow"]["run_id"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
