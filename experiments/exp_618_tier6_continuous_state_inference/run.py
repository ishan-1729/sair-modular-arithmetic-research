"""Paired hard versus continuous intermediate-state inference evaluation."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path

import mlflow, psutil, torch, yaml
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha256_text(value): return hashlib.sha256(value.encode()).hexdigest()
def git_output(*args): return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()
def decode(logits): return [sum(int(bit) << position for position, bit in enumerate(row)) for row in logits.ge(0).tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_618_tier6_continuous_state_inference.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-618-tier6-continuous-state-inference"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp618", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp618", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); seed = int(config["seed"]); widths = (64, 128)
    def make_split(width):
        local = {"data": {"prime_bits_min": width - config["data"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
        return exp300.bigint_prime_split(local, seed + 61800 + width)
    splits = {width: make_split(width) for width in widths}; dataset_hash = sha256_text(json.dumps(splits, sort_keys=True)); torch.set_num_threads(config["evaluation"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); process = psutil.Process(); started = time.perf_counter(); cpu_started = time.process_time(); config_hash = sha256(config_path); runner_hash = sha256(Path(__file__)); commit = git_output("rev-parse", "HEAD"); experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    model = cellular.SharedDilatedConvGRU(width=128, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]); model.load_state_dict(load_file(source)); model.eval()
    def soft_features(state_plane, right, modulus, bit, width):
        batch = len(right); bit_plane = torch.tensor(bit, dtype=torch.float32)[:, None].expand(batch, width); lsb = torch.zeros(batch, width); msb = torch.zeros(batch, width); lsb[:, 0] = 1; msb[:, -1] = 1; return torch.stack((state_plane, exp300.bits(right, width), exp300.bits(modulus, width), bit_plane, lsb, msb), dim=1).mul(2).sub(1)
    @torch.inference_mode()
    def paired_rollouts(width, cases, sampler_seed):
        model.width = width; seen = 0; hard_correct = soft_correct = hard_all = soft_all = hard_invalid = soft_invalid = hard_final_invalid = soft_final_invalid = 0; hard_only = soft_only = both = neither = 0; hard_predictions = []; soft_predictions = []; flags = []
        sampler = exp300.BigIntSampler(splits[width]["test"], sampler_seed)
        while seen < cases:
            size = min(config["evaluation"]["batch_size"], cases - seen); multiplier, right, modulus = sampler.pairs(size); hard_state = [0] * size; soft_state = torch.zeros(size, width); teacher = [0] * size; hard_every = [True] * size; soft_every = [True] * size; hard_bad = [False] * size; soft_bad = [False] * size; soft_logits = None
            for position in range(width):
                bit = [(value >> (width - 1 - position)) & 1 for value in multiplier]; hard_logits, _ = model(exp300.features(hard_state, right, modulus, bit, width)); hard_state = decode(hard_logits); soft_logits, _ = model(soft_features(soft_state, right, modulus, bit, width)); soft_state = torch.sigmoid(soft_logits); soft_threshold = decode(soft_logits); teacher = [(2 * value + flag * other) % prime for value, flag, other, prime in zip(teacher, bit, right, modulus)]; hard_every = [prior and guess == truth for prior, guess, truth in zip(hard_every, hard_state, teacher)]; soft_every = [prior and guess == truth for prior, guess, truth in zip(soft_every, soft_threshold, teacher)]; hard_bad = [prior or guess >= prime for prior, guess, prime in zip(hard_bad, hard_state, modulus)]; soft_bad = [prior or guess >= prime for prior, guess, prime in zip(soft_bad, soft_threshold, modulus)]
            assert soft_logits is not None; soft_final = decode(soft_logits); expected = [(left * other) % prime for left, other, prime in zip(multiplier, right, modulus)]
            for index, (hard_value, soft_value, truth) in enumerate(zip(hard_state, soft_final, expected)):
                hard_ok = hard_value == truth; soft_ok = soft_value == truth; hard_correct += hard_ok; soft_correct += soft_ok; both += hard_ok and soft_ok; hard_only += hard_ok and not soft_ok; soft_only += soft_ok and not hard_ok; neither += not hard_ok and not soft_ok; hard_predictions.append(hard_value); soft_predictions.append(soft_value); flags.append({"case": seen + index, "hard_correct": hard_ok, "continuous_correct": soft_ok})
            hard_all += sum(hard_every); soft_all += sum(soft_every); hard_invalid += sum(hard_bad); soft_invalid += sum(soft_bad); hard_final_invalid += sum(value >= prime for value, prime in zip(hard_state, modulus)); soft_final_invalid += sum(value >= prime for value, prime in zip(soft_final, modulus)); seen += size
        return {"hard": {"cases": seen, "correct": hard_correct, "exact_accuracy": hard_correct / seen, "all_steps_exact_accuracy": hard_all / seen, "any_thresholded_invalid_rate": hard_invalid / seen, "final_invalid_rate": hard_final_invalid / seen}, "continuous": {"cases": seen, "correct": soft_correct, "exact_accuracy": soft_correct / seen, "all_thresholded_steps_exact_accuracy": soft_all / seen, "any_thresholded_invalid_rate": soft_invalid / seen, "final_invalid_rate": soft_final_invalid / seen}, "paired": {"both_correct": both, "hard_only": hard_only, "continuous_only": soft_only, "neither_correct": neither, "accuracy_gain": (soft_correct - hard_correct) / seen}, "hard_predictions": hard_predictions, "continuous_predictions": soft_predictions, "flags": flags}
    results = {}; raw = {}; eval_started = time.perf_counter(); eval_cpu = time.process_time()
    for width in widths:
        paired = paired_rollouts(width, config["evaluation"]["rollout_cases"], seed + 61900 + width); raw[str(width)] = {"hard_predictions": paired.pop("hard_predictions"), "continuous_predictions": paired.pop("continuous_predictions"), "flags": paired.pop("flags")}; results[str(width)] = paired
    raw_path = output / "paired-predictions.json"; raw_path.write_text(json.dumps(raw) + "\n"); raw_hash = sha256(raw_path)
    transition = {str(width): exp300.evaluate_transitions(model, exp300.BigIntSampler(splits[width]["test"], seed + 62000 + width), config["evaluation"]["transition_cases"]) for width in widths}
    determinism = {}
    for width in widths:
        first = paired_rollouts(width, config["evaluation"]["determinism_cases"], seed + 62100 + width); second = paired_rollouts(width, config["evaluation"]["determinism_cases"], seed + 62100 + width); determinism[str(width)] = {"hard_predictions_equal": first["hard_predictions"] == second["hard_predictions"], "continuous_predictions_equal": first["continuous_predictions"] == second["continuous_predictions"]}
    evaluation_runtime = {"wall_seconds": time.perf_counter() - eval_started, "aggregate_cpu_seconds": time.process_time() - eval_cpu, "peak_rss_bytes": process.memory_info().rss}; gates = config["gates"]; width128 = results["128"]; passed = width128["continuous"]["exact_accuracy"] >= gates["minimum_continuous_width128_rollout"] and width128["paired"]["accuracy_gain"] >= gates["minimum_width128_gain"] and results["64"]["continuous"]["exact_accuracy"] >= gates["minimum_continuous_width64_rollout"] and width128["continuous"]["final_invalid_rate"] <= gates["maximum_continuous_invalid_rate"] and all(item["hard_predictions_equal"] and item["continuous_predictions_equal"] for item in determinism.values()); decision = "advance-tier6-continuous-state-inference" if passed else "reject-tier6-continuous-state-inference"
    with mlflow.start_run(run_name=config["run_id"]) as run:
        run_id = run.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "completed", "decision": decision, "role": "tier6-frozen-continuous-state-inference", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()}); mlflow.log_params({"seed": seed, "source_checkpoint_sha256": sha256(source), "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash}); mlflow.log_metrics({"hard_width128_rollout": width128["hard"]["exact_accuracy"], "continuous_width128_rollout": width128["continuous"]["exact_accuracy"], "width128_gain": width128["paired"]["accuracy_gain"], "wall_seconds": evaluation_runtime["wall_seconds"], "aggregate_cpu_seconds": evaluation_runtime["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration"); mlflow.log_artifact(str(raw_path), artifact_path="paired-evidence")
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run_id}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash, "source_checkpoint_sha256": sha256(source)}, "transition": transition, "rollouts": results, "determinism": determinism, "paired_predictions": {"path": str(raw_path.relative_to(ROOT)), "sha256": raw_hash}, "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "evaluation": evaluation_runtime, "torch_threads": config["evaluation"]["torch_threads"]}, "limitations": ["Frozen fresh generated selection evidence only; checked and private answers were not read.", "Continuous intermediate states are not residues; validity is reported for their thresholded diagnostic projections and final output.", "Exact bigint arithmetic is confined to offline input generation and evaluation truth."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
    print(json.dumps({"decision": decision, "rollouts": results, "transition": transition, "determinism": determinism, "runtime": summary["runtime"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
