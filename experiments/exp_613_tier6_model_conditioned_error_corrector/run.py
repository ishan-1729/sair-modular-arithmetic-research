"""Matched uniform versus model-conditioned training for a learned error corrector."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, random, subprocess, sys, time
from pathlib import Path

import mlflow, numpy as np, psutil, torch, yaml
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
def decode(logits): return [sum(int(bit) << position for position, bit in enumerate(row)) for row in logits.ge(0).tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_613_tier6_model_conditioned_error_corrector.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-613-tier6-model-conditioned-error-corrector"); args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp613", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp613", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); correction = load_module("correction_for_exp613", ROOT / "experiments/exp_613_tier6_model_conditioned_error_corrector/model.py")
    seed = int(config["seed"]); widths = (64, 128)
    def split(width):
        local = {"data": {"prime_bits_min": width - config["data"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
        return exp300.bigint_prime_split(local, seed + 61300 + width)
    splits = {width: split(width) for width in widths}; split_hash = sha256_text(json.dumps(splits, sort_keys=True)); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); process = psutil.Process(); runner_hash = sha256(Path(__file__)); config_hash = sha256(config_path); started = time.perf_counter(); cpu_started = time.process_time(); commit = git_output("rev-parse", "HEAD"); experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    def base_model():
        model = cellular.SharedDilatedConvGRU(width=128, hidden=config["base_model"]["hidden"], cycles=config["base_model"]["cycles"]); model.load_state_dict(load_file(source)); model.eval(); return model
    def corrected_model(): return correction.ErrorCorrectedCell(base_model(), correction.ConditionedCorrector(width=128, hidden=config["corrector_model"]["hidden"], cycles=config["corrector_model"]["cycles"]))
    def record(state, right, modulus, bit, target, quotient, width, margin=None):
        value = {"state": int(state), "right": int(right), "modulus": int(modulus), "bit": int(bit), "target": int(target), "quotient": int(quotient), "width": int(width)}
        if margin is not None: value["margin"] = float(margin)
        return value
    dataset_key = sha256_text(json.dumps({"source": sha256(source), "config": config_hash, "split": split_hash, "mining": config["mining"]}, sort_keys=True)); dataset_path = output / f"datasets-{dataset_key}.json"
    mining_started = time.perf_counter(); mining_cpu = time.process_time()
    if dataset_path.exists():
        datasets = json.loads(dataset_path.read_text()); mining_cached = True
        if datasets.get("cache_key") != dataset_key: raise RuntimeError("dataset cache key mismatch")
    else:
        mining_cached = False; frozen = base_model(); first_errors = []; low_margin = []; sampler = exp300.BigIntSampler(splits[128]["train"], seed + 61428); remaining = config["mining"]["trajectories"]
        with torch.no_grad():
            while remaining:
                size = min(config["mining"]["batch_size"], remaining); left, right, modulus = sampler.pairs(size); predicted = [0] * size; teacher = [0] * size; active = [True] * size; lows = [(float("inf"), None)] * size
                for position in range(128):
                    bits = [(value >> (127 - position)) & 1 for value in left]; raw = [2 * value + bit * other for value, bit, other in zip(teacher, bits, right)]; target = [value % prime for value, prime in zip(raw, modulus)]; quotient = [value // prime for value, prime in zip(raw, modulus)]; frozen.width = 128; logits, _ = frozen(exp300.features(predicted, right, modulus, bits, 128)); guesses = decode(logits); margins = logits.abs().amin(1).tolist()
                    for row in range(size):
                        item = record(predicted[row], right[row], modulus[row], bits[row], target[row], quotient[row], 128, margins[row])
                        if margins[row] < lows[row][0]: lows[row] = (margins[row], item)
                        if active[row] and guesses[row] != target[row]: first_errors.append(item); active[row] = False
                    predicted = guesses; teacher = target
                low_margin.extend(item for _, item in lows); remaining -= size
        if not first_errors: raise RuntimeError("mining produced no first errors")
        target_hard = config["datasets"]["width128_cases"] // 2; repeated_errors = [first_errors[index % len(first_errors)] for index in range(target_hard)]; selected_lows = sorted(low_margin, key=lambda item: item["margin"])[:target_hard]
        uniform128_values = exp300.BigIntSampler(splits[128]["train"], seed + 61528).transitions(config["datasets"]["width128_cases"]); uniform128 = [record(*(values[index] for values in uniform128_values[:5]), int(uniform128_values[5][index]), 128) for index in range(config["datasets"]["width128_cases"])]
        uniform64_values = exp300.BigIntSampler(splits[64]["train"], seed + 61564).transitions(config["datasets"]["width64_cases"]); uniform64 = [record(*(values[index] for values in uniform64_values[:5]), int(uniform64_values[5][index]), 64) for index in range(config["datasets"]["width64_cases"])]
        datasets = {"cache_key": dataset_key, "first_error_count": len(first_errors), "trajectory_count": config["mining"]["trajectories"], "control": {"64": uniform64, "128": uniform128}, "challenger": {"64": uniform64, "128": repeated_errors + selected_lows}}
        dataset_path.write_text(json.dumps(datasets) + "\n")
    mining_runtime = {"cached": mining_cached, "wall_seconds": time.perf_counter() - mining_started, "aggregate_cpu_seconds": time.process_time() - mining_cpu, "first_error_count": datasets["first_error_count"], "trajectory_count": datasets["trajectory_count"], "dataset_sha256": sha256(dataset_path)}
    torch.manual_seed(seed + 613); reference = base_model(); exact_candidate = corrected_model(); exact_initialization = {}
    with torch.no_grad():
        for width in widths:
            features = torch.randn(config["exact_initialization_cases"], 6, width); reference.width = width; exact_candidate.width = width; left = reference(features); right = exact_candidate(features); exact_initialization[str(width)] = {"logits_equal": torch.equal(left[0], right[0]), "quotient_equal": torch.equal(left[1], right[1]), "maximum_absolute_difference": max(float((left[0] - right[0]).abs().max()), float((left[1] - right[1]).abs().max()))}
    if not all(item["logits_equal"] and item["quotient_equal"] for item in exact_initialization.values()): raise RuntimeError("exact corrector initialization failed")
    def evaluate(model, width):
        model.width = width; return {"transition": exp300.evaluate_transitions(model, exp300.BigIntSampler(splits[width]["test"], seed + 61600 + width), config["evaluation"]["transition_cases"]), "rollout": exp300.evaluate_rollouts(model, exp300.BigIntSampler(splits[width]["test"], seed + 61700 + width), config["evaluation"]["rollout_cases"])}
    def train_arm(name, dataset_name):
        arm_dir = output / name; arm_dir.mkdir(parents=True, exist_ok=True); cache_key = sha256_text(json.dumps({"runner": runner_hash, "config": config_hash, "dataset": mining_runtime["dataset_sha256"], "arm": dataset_name}, sort_keys=True)); cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == cache_key and (ROOT / payload["result"]["checkpoint"]["path"]).exists(): return {**payload["result"], "cached": True}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); model = corrected_model(); optimizer = torch.optim.AdamW(model.corrector.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"]); rng = random.Random(seed + 61800); history = []; arm_wall = time.perf_counter(); arm_cpu = time.process_time(); peak = process.memory_info().rss; model.train()
        for step in range(1, config["training"]["steps"] + 1):
            width = 64 if step % 4 == 0 else 128; pool = datasets[dataset_name][str(width)]; rows = [pool[rng.randrange(len(pool))] for _ in range(config["training"]["batch_size"])]; model.width = width; logits, quotient_logits = model(exp300.features([row["state"] for row in rows], [row["right"] for row in rows], [row["modulus"] for row in rows], [row["bit"] for row in rows], width)); quotient = torch.tensor([row["quotient"] for row in rows], dtype=torch.long); loss, stats = exp300.transition_loss(logits, quotient_logits, [row["target"] for row in rows], quotient, width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"]); optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.corrector.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 50 == 0 or step == config["training"]["steps"]:
                peak = max(peak, process.memory_info().rss); item = {"step": step, "width": width, "elapsed_wall_seconds": time.perf_counter() - arm_wall, "gradient_norm": gradient, **stats}; history.append(item); mlflow.log_metrics({f"{name}_loss": stats["loss"], f"{name}_batch_exact": stats["batch_exact"]}, step=step); print(json.dumps({"arm": name, **item}), flush=True)
        model.eval(); checkpoint = arm_dir / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); result = {"arm": name, "cached": False, "parameters": model.parameter_count(), "trainable_parameters": model.trainable_parameter_count(), "training": {"steps": config["training"]["steps"], "history": history}, "evaluation": {str(width): evaluate(model, width) for width in widths}, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "runtime": {"wall_seconds": time.perf_counter() - arm_wall, "aggregate_cpu_seconds": time.process_time() - arm_cpu, "peak_rss_bytes": peak}}; cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n"); return result
    results = {}; run_ids = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "tier6-model-conditioned-error-corrector", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()}); mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "source_checkpoint_sha256": sha256(source), "config_sha256": config_hash, "runner_sha256": runner_hash, "split_sha256": split_hash}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for name, dataset_name in (("uniform_corrector_control", "control"), ("first_error_corrector", "challenger")):
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id; results[name] = train_arm(name, dataset_name); mlflow.log_metrics({"width64_rollout": results[name]["evaluation"]["64"]["rollout"]["exact_accuracy"], "width128_transition": results[name]["evaluation"]["128"]["transition"]["exact_accuracy"], "width128_rollout": results[name]["evaluation"]["128"]["rollout"]["exact_accuracy"]})
        control = results["uniform_corrector_control"]["evaluation"]; challenger = results["first_error_corrector"]["evaluation"]; gain = challenger["128"]["rollout"]["exact_accuracy"] - control["128"]["rollout"]["exact_accuracy"]; gates = config["gates"]; passed = challenger["128"]["rollout"]["exact_accuracy"] >= gates["minimum_challenger_width128_rollout"] and gain >= gates["minimum_width128_gain"] and challenger["64"]["rollout"]["exact_accuracy"] >= gates["minimum_challenger_width64_rollout"] and challenger["128"]["transition"]["exact_accuracy"] >= gates["minimum_challenger_width128_transition"] and challenger["128"]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_invalid_rate"]; decision = "advance-tier6-model-conditioned-corrector" if passed else "reject-tier6-model-conditioned-corrector"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "split_sha256": split_hash, "source_checkpoint_sha256": sha256(source)}, "exact_initialization": exact_initialization, "mining": mining_runtime, "arms": results, "comparison": {"control_width128_rollout": control["128"]["rollout"]["exact_accuracy"], "challenger_width128_rollout": challenger["128"]["rollout"]["exact_accuracy"], "width128_rollout_gain": gain}, "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "torch_threads": config["training"]["torch_threads"]}, "limitations": ["Fresh generated selection evidence only; checked and private answers were not read.", "Exact bigint arithmetic is confined to offline mining, labels, and evaluation truth.", "Mining compute is intentionally unmatched and disclosed; model architecture, updates, and training examples are matched."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"control_width128_rollout": summary["comparison"]["control_width128_rollout"], "challenger_width128_rollout": summary["comparison"]["challenger_width128_rollout"], "width128_rollout_gain": gain, "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": summary["comparison"], "mining": mining_runtime, "arms": {name: {"evaluation": arm["evaluation"], "checkpoint": arm["checkpoint"], "runtime": arm["runtime"]} for name, arm in results.items()}, "runtime": summary["runtime"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
