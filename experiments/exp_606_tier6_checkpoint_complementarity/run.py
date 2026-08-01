"""Fresh Tier-6 endpoint overlap, weight soup, and fixed logit ensemble."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path

import mlflow, torch, yaml
from safetensors.torch import load_file, save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha256_text(value): return hashlib.sha256(value.encode()).hexdigest()
def git_output(*args): return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


class LogitEnsemble(nn.Module):
    def __init__(self, first, second, alpha, width): super().__init__(); self.first = first; self.second = second; self.alpha = alpha; self.width = width
    def forward(self, features):
        left, left_q = self.first(features); right, right_q = self.second(features); return (1 - self.alpha) * left + self.alpha * right, (1 - self.alpha) * left_q + self.alpha * right_q


@torch.inference_mode()
def transition_on_values(model, exp300, values, width):
    state, right, modulus, bit, target, _ = values; started = time.perf_counter(); cpu = time.process_time(); model.width = width; logits, _ = model(exp300.features(state, right, modulus, bit, width)); predicted = logits.ge(0); expected = exp300.bits(target, width).bool(); correct = int((predicted == expected).all(1).sum()); decoded = [sum(int(value) << position for position, value in enumerate(row)) for row in predicted.tolist()]; return {"cases": len(target), "correct": correct, "exact_accuracy": correct / len(target), "valid_residue_rate": sum(value < prime for value, prime in zip(decoded, modulus)) / len(target), "wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu}


@torch.inference_mode()
def rollout_on_pairs(model, exp300, pairs, width):
    multiplier, right, modulus = pairs; state = [0] * len(modulus); teacher = [0] * len(modulus); every = [True] * len(modulus); invalid = [False] * len(modulus); started = time.perf_counter(); cpu = time.process_time(); model.width = width
    for position in range(width):
        bit = [(value >> (width - 1 - position)) & 1 for value in multiplier]; logits, _ = model(exp300.features(state, right, modulus, bit, width)); rows = logits.ge(0).tolist(); state = [sum(int(value) << index for index, value in enumerate(row)) for row in rows]; teacher = [(2 * value + flag * other) % prime for value, flag, other, prime in zip(teacher, bit, right, modulus)]; every = [prior and guess == truth for prior, guess, truth in zip(every, state, teacher)]; invalid = [prior or guess >= prime for prior, guess, prime in zip(invalid, state, modulus)]
    expected = [(left * other) % prime for left, other, prime in zip(multiplier, right, modulus)]; correctness = [guess == truth for guess, truth in zip(state, expected)]; return {"cases": len(modulus), "correct": sum(correctness), "exact_accuracy": sum(correctness) / len(modulus), "all_steps_exact_accuracy": sum(every) / len(modulus), "any_invalid_residue_rate": sum(invalid) / len(modulus), "correctness": correctness, "wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_606_tier6_checkpoint_complementarity.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-606-tier6-checkpoint-complementarity"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); path_a = (ROOT / config["checkpoint_a"]).resolve(); path_b = (ROOT / config["checkpoint_b"]).resolve()
    if sha256(path_a) != config["checkpoint_a_sha256"] or sha256(path_b) != config["checkpoint_b_sha256"]: raise RuntimeError("checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp606", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp606", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); seed = int(config["seed"]); torch.set_num_threads(config["resource_budget"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); started = time.perf_counter(); cpu_started = time.process_time()
    def model(path, width=128):
        result = cellular.SharedDilatedConvGRU(width=width, **config["architecture"]); result.load_state_dict(load_file(path)); result.eval(); return result
    state_a = load_file(path_a); state_b = load_file(path_b); alpha = config["combination"]["weight_soup_alpha_b"]; soup_state = {key: ((1 - alpha) * state_a[key] + alpha * state_b[key]).contiguous() for key in state_a}; soup_path = output / "weight-soup.safetensors"; save_file(soup_state, soup_path, metadata={"alpha_b": str(alpha), "checkpoint_a": sha256(path_a), "checkpoint_b": sha256(path_b)}); soup = cellular.SharedDilatedConvGRU(width=128, **config["architecture"]); soup.load_state_dict(soup_state); soup.eval(); ensemble = LogitEnsemble(model(path_a), model(path_b), config["combination"]["logit_ensemble_alpha_b"], 128); ensemble.eval(); arms = {"endpoint_a": model(path_a), "endpoint_b": model(path_b), "weight_soup": soup, "logit_ensemble": ensemble}
    splits = {}
    for width in config["evaluation"]["widths"]:
        local = {"data": {"prime_bits_min": width - config["data"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": 0, "test_prime_identities": config["data"]["test_prime_identities"]}}; splits[str(width)] = exp300.bigint_prime_split(local, seed + 60600 + width)["test"]
    dataset_hash = sha256_text(json.dumps(splits, sort_keys=True)); cases = {}
    for width in config["evaluation"]["widths"]:
        transition_sampler = exp300.BigIntSampler(splits[str(width)], seed + 60700 + width); rollout_sampler = exp300.BigIntSampler(splits[str(width)], seed + 60800 + width); cases[str(width)] = {"transition": transition_sampler.transitions(config["evaluation"]["transition_cases"]), "rollout": rollout_sampler.pairs(config["evaluation"]["rollout_cases"])}
    results = {name: {} for name in arms}
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); commit = git_output("rev-parse", "HEAD")
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "tier6-checkpoint-complementarity", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"}); mlflow.log_params({"checkpoint_a_sha256": sha256(path_a), "checkpoint_b_sha256": sha256(path_b), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "dataset_sha256": dataset_hash}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for name, active in arms.items():
            for width in config["evaluation"]["widths"]:
                active.width = width
                if isinstance(active, LogitEnsemble): active.first.width = width; active.second.width = width
                results[name][str(width)] = {"transition": transition_on_values(active, exp300, cases[str(width)]["transition"], width), "rollout": rollout_on_pairs(active, exp300, cases[str(width)]["rollout"], width)}
            mlflow.log_metrics({f"{name}_width64_rollout": results[name]["64"]["rollout"]["exact_accuracy"], f"{name}_width128_transition": results[name]["128"]["transition"]["exact_accuracy"], f"{name}_width128_rollout": results[name]["128"]["rollout"]["exact_accuracy"]}); print(json.dumps({"arm": name, "width64": results[name]["64"], "width128": results[name]["128"]}, default=lambda value: None), flush=True)
        correct_a = results["endpoint_a"]["128"]["rollout"]["correctness"]; correct_b = results["endpoint_b"]["128"]["rollout"]["correctness"]; union = sum(left or right for left, right in zip(correct_a, correct_b)) / len(correct_a); intersection = sum(left and right for left, right in zip(correct_a, correct_b)) / len(correct_a); fixed = max(("weight_soup", "logit_ensemble"), key=lambda name: results[name]["128"]["rollout"]["exact_accuracy"]); selected = results[fixed]; inference_seconds = selected["128"]["rollout"]["wall_seconds"] * 100 / config["evaluation"]["rollout_cases"] + 36.9; gates = config["gates"]; passed = union >= gates["minimum_oracle_union_width128_rollout"] and selected["128"]["rollout"]["exact_accuracy"] >= gates["minimum_fixed_combination_width128_rollout"] and selected["64"]["rollout"]["exact_accuracy"] >= gates["minimum_fixed_combination_width64_rollout"] and selected["128"]["transition"]["exact_accuracy"] >= gates["minimum_fixed_combination_width128_transition"] and selected["128"]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_fixed_combination_width128_invalid_rate"] and inference_seconds <= gates["maximum_projected_full_candidate_inference_seconds"]; decision = "advance-tier6-fixed-combination" if passed else "reject-tier6-checkpoint-combination"
        combined_path = output / "logit-ensemble.safetensors"; save_file({**{f"model_a.{key}": value.contiguous() for key, value in state_a.items()}, **{f"model_b.{key}": value.contiguous() for key, value in state_b.items()}}, combined_path, metadata={"alpha_b": str(config["combination"]["logit_ensemble_alpha_b"])}); clean_results = json.loads(json.dumps(results)); summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "dataset_sha256": dataset_hash, "checkpoint_a_sha256": sha256(path_a), "checkpoint_b_sha256": sha256(path_b)}, "arms": clean_results, "complementarity": {"width128_oracle_union": union, "width128_intersection": intersection, "endpoint_a_only": sum(left and not right for left, right in zip(correct_a, correct_b)) / len(correct_a), "endpoint_b_only": sum(right and not left for left, right in zip(correct_a, correct_b)) / len(correct_a)}, "selection": {"fixed_combination": fixed, "projected_full_candidate_inference_seconds": inference_seconds}, "artifacts": {"weight_soup": {"path": str(soup_path.relative_to(ROOT)), "sha256": sha256(soup_path)}, "logit_ensemble": {"path": str(combined_path.relative_to(ROOT)), "sha256": sha256(combined_path)}}, "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "torch_threads": config["resource_budget"]["torch_threads"]}, "limitations": ["Fresh generated selection evidence only; checked, independent, and private answers were not read.", "Oracle union is an unattainable upper bound and never an inference arm.", "Projected candidate runtime scales measured local rollout time and requires official-container confirmation if advanced."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"oracle_union": union, "selected_width128_rollout": selected["128"]["rollout"]["exact_accuracy"], "projected_full_candidate_inference_seconds": inference_seconds, "wall_seconds": summary["runtime"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "complementarity": summary["complementarity"], "selection": summary["selection"], "arms": {name: {width: {kind: {key: value for key, value in metrics.items() if key != "correctness"} for kind, metrics in result.items()} for width, result in widths.items()} for name, widths in results.items()}, "runtime": summary["runtime"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
