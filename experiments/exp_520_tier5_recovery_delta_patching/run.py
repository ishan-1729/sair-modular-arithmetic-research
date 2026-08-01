"""Causal tensor-group patching for the strongest Tier-5 cellular checkpoint pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def sha256_text(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest()
def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def selected(key: str, selectors: list[str]) -> bool:
    return "*" in selectors or any(key == selector or key.startswith(selector + ".") for selector in selectors)


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / total; denominator = 1 + z * z / total; center = (p + z * z / (2 * total)) / denominator; radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator; return [center - radius, center + radius]


def one_sided_mcnemar(patch: list[bool], reference: list[bool]) -> dict:
    improved = sum(a and not b for a, b in zip(patch, reference)); worsened = sum(not a and b for a, b in zip(patch, reference)); discordant = improved + worsened
    if discordant == 0 or improved <= worsened: p_value = 1.0
    else: p_value = sum(math.comb(discordant, value) for value in range(improved, discordant + 1)) / (2 ** discordant)
    return {"improved": improved, "worsened": worsened, "discordant": discordant, "one_sided_exact_p": p_value}


@torch.inference_mode()
def detailed_rollouts(model, sampler, cases: int, exp300) -> dict:
    width = model.width; seen = 0; correct_flags = []; all_steps_flags = []; invalid_flags = []; final_predictions = []
    while seen < cases:
        size = min(128, cases - seen); multiplier, multiplicand, modulus = sampler.pairs(size); state = [0] * size; teacher = [0] * size; every = [True] * size; bad = [False] * size
        for position in range(width):
            input_bit = [(value >> (width - 1 - position)) & 1 for value in multiplier]; logits, _ = model(exp300.features(state, multiplicand, modulus, input_bit, width)); rows = logits.ge(0).tolist(); state = [sum(int(row[index]) << index for index in range(width)) for row in rows]; teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in zip(teacher, input_bit, multiplicand, modulus)]; every = [prior and predicted == expected for prior, predicted, expected in zip(every, state, teacher)]; bad = [prior or predicted >= prime for prior, predicted, prime in zip(bad, state, modulus)]
        expected = [(left * right) % prime for left, right, prime in zip(multiplier, multiplicand, modulus)]; correct_flags.extend(predicted == truth for predicted, truth in zip(state, expected)); all_steps_flags.extend(every); invalid_flags.extend(bad); final_predictions.extend(state); seen += size
    correct = sum(correct_flags); invalid = sum(invalid_flags); return {"cases": seen, "correct": correct, "exact_accuracy": correct / seen, "exact_accuracy_wilson95": wilson(correct, seen), "all_steps_correct": sum(all_steps_flags), "all_steps_exact_accuracy": sum(all_steps_flags) / seen, "any_invalid_residue_rate": invalid / seen, "correct_flags": correct_flags, "invalid_flags": invalid_flags, "final_predictions": final_predictions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_520_tier5_recovery_delta_patching.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-520-tier5-recovery-delta-patching"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text())
    control_path = (ROOT / config["control_checkpoint"]).resolve(); challenger_path = (ROOT / config["challenger_checkpoint"]).resolve()
    if sha256(control_path) != config["control_checkpoint_sha256"] or sha256(challenger_path) != config["challenger_checkpoint_sha256"]: raise RuntimeError("endpoint hash mismatch")
    exp300 = load_module("exp300_for_exp520", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp520", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    split = exp300.bigint_prime_split({"data": config["data"]}, config["seed"] + 520); dataset_hash = sha256_text(json.dumps(split["test"], sort_keys=True)); evaluator_hash = sha256(Path(__file__)); control_state = load_file(control_path); challenger_state = load_file(challenger_path)
    torch.set_num_threads(config["resource_budget"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); config_hash = sha256(config_path); commit = git_output("rev-parse", "HEAD"); experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); total_wall = time.perf_counter(); total_cpu = time.process_time(); results = {}; nested_ids = {}

    def evaluate_arm(label: str, selectors: list[str]) -> dict:
        arm_dir = output / label; arm_dir.mkdir(parents=True, exist_ok=True); state = {key: (challenger_state[key] if selected(key, selectors) else value).detach().contiguous() for key, value in control_state.items()}; checkpoint = arm_dir / "weights.safetensors"; save_file(state, checkpoint, metadata={"label": label, "selectors": json.dumps(selectors)})
        cache_key = sha256_text(json.dumps({"checkpoint": sha256(checkpoint), "evaluator": evaluator_hash, "dataset": dataset_hash, "seed": config["seed"], "evaluation": config["evaluation"]}, sort_keys=True)); cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == cache_key: return {**payload["result"], "cached": True}
        model = cellular.SharedDilatedConvGRU(**config["architecture"]); model.load_state_dict(state); model.eval(); started = time.perf_counter(); cpu_started = time.process_time(); transition = exp300.evaluate_transitions(model, exp300.BigIntSampler(split["test"], config["seed"] + 52050), config["evaluation"]["transition_cases_per_arm"]); rollout = detailed_rollouts(model, exp300.BigIntSampler(split["test"], config["seed"] + 52051), config["evaluation"]["rollout_cases_per_arm"], exp300)
        result = {"label": label, "selectors": selectors, "cached": False, "parameters": model.parameter_count(), "transition": transition, "rollout": rollout, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started}}; cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n"); return result

    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "hypothesis": config["hypothesis"], "status": "running", "role": "tier5-causal-delta-patching", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()}); mlflow.log_params({"seed": config["seed"], "control_checkpoint_sha256": sha256(control_path), "challenger_checkpoint_sha256": sha256(challenger_path), "config_sha256": config_hash, "runner_sha256": evaluator_hash, "dataset_sha256": dataset_hash}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["analysis"]), artifact_path="analysis")
        for label, selectors in config["patches"].items():
            with mlflow.start_run(run_name=label, nested=True) as nested:
                nested_ids[label] = nested.info.run_id; result = evaluate_arm(label, selectors); results[label] = result; mlflow.log_metrics({"transition_exact": result["transition"]["exact_accuracy"], "rollout_exact": result["rollout"]["exact_accuracy"], "invalid_rollout": result["rollout"]["any_invalid_residue_rate"], "wall_seconds": result["runtime"]["wall_seconds"]}); mlflow.set_tag("status", "completed")
        if results["quotient-negative-control"]["rollout"]["final_predictions"] != results["control"]["rollout"]["final_predictions"] or results["quotient-negative-control"]["rollout"]["invalid_flags"] != results["control"]["rollout"]["invalid_flags"]: raise RuntimeError("quotient negative control changed residue predictions")
        full = results["full-challenger"]; comparisons = {}
        patch_labels = [label for label in config["patches"] if label not in {"control", "quotient-negative-control", "full-challenger"}]
        for label in patch_labels:
            result = results[label]; paired = one_sided_mcnemar(result["rollout"]["correct_flags"], full["rollout"]["correct_flags"]); comparisons[label] = {"rollout_gain_over_full_challenger": result["rollout"]["exact_accuracy"] - full["rollout"]["exact_accuracy"], "paired": paired}
        gates = config["gates"]; eligible = [label for label in patch_labels if results[label]["transition"]["exact_accuracy"] >= gates["minimum_patch_transition"] and results[label]["rollout"]["exact_accuracy"] >= gates["minimum_patch_rollout"] and comparisons[label]["rollout_gain_over_full_challenger"] >= gates["minimum_rollout_gain_over_full_challenger"] and results[label]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_patch_invalid_rollout_rate"] and comparisons[label]["paired"]["one_sided_exact_p"] <= gates["maximum_one_sided_mcnemar_p"]]
        winner = max(eligible, key=lambda label: results[label]["rollout"]["exact_accuracy"]) if eligible else None; decision = f"advance-{winner}-independent-tier5" if winner else "reject-tier5-recovery-delta-patches"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "nested_run_ids": nested_ids}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": evaluator_hash, "dataset_sha256": dataset_hash, "control_checkpoint_sha256": sha256(control_path), "challenger_checkpoint_sha256": sha256(challenger_path)}, "results": results, "comparisons": comparisons, "negative_control_identical": True, "eligible": eligible, "winner": winner, "gates": gates, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - total_wall, "aggregate_cpu_seconds": time.process_time() - total_cpu, "torch_threads": config["resource_budget"]["torch_threads"]}, "limitations": ["Fresh generated Tier-5 selection evidence only; checked, consumed independent, and private answers were not read.", "Tensor-group patches test necessity/sufficiency at parameter-group resolution, not individual-channel mechanisms.", "The one-sided exact paired test is selection evidence and does not replace five independent seeds."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"control_rollout": results["control"]["rollout"]["exact_accuracy"], "full_challenger_rollout": full["rollout"]["exact_accuracy"], "eligible_patches": len(eligible), "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision, "winner": winner or "none"})
    print(json.dumps({"decision": decision, "winner": winner, "eligible": eligible, "arms": {label: {"transition": result["transition"], "rollout": {key: value for key, value in result["rollout"].items() if key not in {"correct_flags", "invalid_flags", "final_predictions"}}, "checkpoint": result["checkpoint"]} for label, result in results.items()}, "comparisons": comparisons, "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
