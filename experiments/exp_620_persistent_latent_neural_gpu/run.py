"""Matched decoded-state cellular versus persistent latent Neural GPU smoke."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, random, subprocess, sys, time
from pathlib import Path

import mlflow, numpy as np, psutil, torch, torch.nn.functional as F, yaml
from safetensors.torch import save_file
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_620_persistent_latent_neural_gpu.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-620-persistent-latent-neural-gpu"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); seed = int(config["seed"]); train_widths = tuple(config["training"]["width_schedule"]); eval_widths = tuple(config["evaluation"]["widths"])
    exp300 = load_module("exp300_for_exp620", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); cellular = load_module("cellular_for_exp620", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); persistent = load_module("persistent_for_exp620", ROOT / "experiments/exp_620_persistent_latent_neural_gpu/model.py")
    def make_split(width):
        local = {"data": {"prime_bits_min": max(3, width - config["data"]["prime_bit_span"] + 1), "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
        return exp300.bigint_prime_split(local, seed + 62000 + width)
    splits = {width: make_split(width) for width in sorted(set(train_widths + eval_widths))}; dataset_hash = sha256_text(json.dumps(splits, sort_keys=True)); torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); process = psutil.Process(); started = time.perf_counter(); cpu_started = time.process_time(); config_hash = sha256(config_path); runner_hash = sha256(Path(__file__)); commit = git_output("rev-parse", "HEAD"); experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    def trajectory_batch(sampler, width, batch):
        left, right, modulus = sampler.pairs(batch); prefix_targets = []; states = []; repeated_right = []; repeated_modulus = []; input_bits = []; flat_targets = []
        for multiplier, other, prime in zip(left, right, modulus):
            state = 0; rows = []
            for position in range(width):
                bit = (multiplier >> (width - 1 - position)) & 1; raw = 2 * state + bit * other; target = raw % prime; states.append(state); repeated_right.append(other); repeated_modulus.append(prime); input_bits.append(bit); flat_targets.append(target); state = target; rows.append([(target >> index) & 1 for index in range(width)])
            prefix_targets.append(rows)
        multiplier_bits = persistent.PersistentLatentNeuralGPU.bits(left, width).flip(1); target_bits = torch.tensor(prefix_targets, dtype=torch.float32); return {"left": left, "right": right, "modulus": modulus, "multiplier_bits": multiplier_bits, "target_bits": target_bits, "states": states, "repeated_right": repeated_right, "repeated_modulus": repeated_modulus, "input_bits": input_bits, "flat_targets": flat_targets}
    def bit_loss(logits, targets):
        per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none"); mean = per.mean(); worst = per.flatten(0, -2).mean(0).max(); loss = mean + config["training"]["worst_bit_weight"] * worst; exact = (logits.ge(0) == targets.bool()).all(-1).float(); return loss, {"loss": float(loss.detach()), "mean_bce": float(mean.detach()), "worst_bit_bce": float(worst.detach()), "prefix_exact": float(exact.mean()), "final_exact": float(exact[:, -1].mean())}
    @torch.inference_mode()
    def evaluate_persistent(model, width, cases):
        model.width = width; sampler = exp300.BigIntSampler(splits[width]["test"], seed + 62400 + width); seen = correct = all_prefix = invalid = 0
        while seen < cases:
            size = min(128, cases - seen); batch = trajectory_batch(sampler, width, size); logits = model(batch["multiplier_bits"], model.static_features(batch["right"], batch["modulus"], width), return_all=True); predicted = logits.ge(0); truth = batch["target_bits"].bool(); final_rows = predicted[:, -1].tolist(); values = [sum(int(bit) << index for index, bit in enumerate(row)) for row in final_rows]; expected = [(a * b) % prime for a, b, prime in zip(batch["left"], batch["right"], batch["modulus"])]; correct += sum(value == target for value, target in zip(values, expected)); all_prefix += int((predicted == truth).all(-1).all(-1).sum()); invalid += sum(value >= prime for value, prime in zip(values, batch["modulus"])); seen += size
        return {"cases": seen, "correct": correct, "exact_accuracy": correct / seen, "all_prefix_exact_accuracy": all_prefix / seen, "invalid_rate": invalid / seen}
    def train_arm(name, architecture):
        arm_dir = output / name; arm_dir.mkdir(parents=True, exist_ok=True); cache_key = sha256_text(json.dumps({"runner": runner_hash, "config": config_hash, "dataset": dataset_hash, "arm": architecture}, sort_keys=True)); cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == cache_key and (ROOT / payload["result"]["checkpoint"]["path"]).exists(): return {**payload["result"], "cached": True}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if architecture == "persistent": model = persistent.PersistentLatentNeuralGPU(width=max(train_widths), hidden=config["persistent_model"]["hidden"], microsteps=config["persistent_model"]["microsteps"])
        else: model = cellular.SharedDilatedConvGRU(width=max(train_widths), hidden=config["control_model"]["hidden"], cycles=config["control_model"]["cycles"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"]); samplers = {width: exp300.BigIntSampler(splits[width]["train"], seed + 62500 + width) for width in train_widths}; history = []; arm_wall = time.perf_counter(); arm_cpu = time.process_time(); peak = process.memory_info().rss; model.train()
        for step in range(1, config["training"]["steps"] + 1):
            width = train_widths[(step - 1) % len(train_widths)]; batch = trajectory_batch(samplers[width], width, config["training"]["batch_size"]); model.width = width
            if architecture == "persistent": logits = model(batch["multiplier_bits"], model.static_features(batch["right"], batch["modulus"], width), return_all=True); loss, stats = bit_loss(logits, batch["target_bits"])
            else:
                logits, _ = model(exp300.features(batch["states"], batch["repeated_right"], batch["repeated_modulus"], batch["input_bits"], width)); targets = exp300.bits(batch["flat_targets"], width); loss, flat_stats = bit_loss(logits, targets); exact = (logits.ge(0) == targets.bool()).all(-1).view(config["training"]["batch_size"], width); stats = {**flat_stats, "prefix_exact": float(exact.float().mean()), "final_exact": float(exact[:, -1].float().mean())}
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 100 == 0 or step == config["training"]["steps"]:
                peak = max(peak, process.memory_info().rss); item = {"step": step, "width": width, "elapsed_wall_seconds": time.perf_counter() - arm_wall, "gradient_norm": gradient, **stats}; history.append(item); mlflow.log_metrics({f"{name}_loss": stats["loss"], f"{name}_prefix_exact": stats["prefix_exact"], f"{name}_final_exact": stats["final_exact"]}, step=step); print(json.dumps({"arm": name, **item}), flush=True)
        model.eval(); checkpoint = arm_dir / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); evaluation = {}
        for width in eval_widths:
            if architecture == "persistent": evaluation[str(width)] = evaluate_persistent(model, width, config["evaluation"]["rollout_cases"])
            else:
                model.width = width; result = exp300.evaluate_rollouts(model, exp300.BigIntSampler(splits[width]["test"], seed + 62400 + width), config["evaluation"]["rollout_cases"]); evaluation[str(width)] = {"cases": result["cases"], "correct": result["correct"], "exact_accuracy": result["exact_accuracy"], "all_prefix_exact_accuracy": result["all_steps_exact_accuracy"], "invalid_rate": result["any_invalid_residue_rate"]}
        result = {"arm": name, "cached": False, "architecture": architecture, "parameters": model.parameter_count(), "training": {"steps": config["training"]["steps"], "trajectory_examples": config["training"]["steps"] * config["training"]["batch_size"], "history": history}, "evaluation": evaluation, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "runtime": {"wall_seconds": time.perf_counter() - arm_wall, "aggregate_cpu_seconds": time.process_time() - arm_cpu, "peak_rss_bytes": peak}}; cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n"); return result
    results = {}; run_ids = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "persistent-latent-neural-gpu-smoke", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()}); mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for name, architecture in (("decoded_cellular_control", "control"), ("persistent_neural_gpu", "persistent")):
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id; results[name] = train_arm(name, architecture); mlflow.log_metrics({f"width{width}_rollout": results[name]["evaluation"][str(width)]["exact_accuracy"] for width in eval_widths})
        control = results["decoded_cellular_control"]["evaluation"]; challenger = results["persistent_neural_gpu"]["evaluation"]; gates = config["gates"]; gain16 = challenger["16"]["exact_accuracy"] - control["16"]["exact_accuracy"]; passed = challenger["16"]["exact_accuracy"] >= gates["minimum_persistent_width16_rollout"] and gain16 >= gates["minimum_width16_gain"] and challenger["24"]["exact_accuracy"] >= gates["minimum_persistent_width24_rollout"] and challenger["32"]["exact_accuracy"] >= gates["minimum_persistent_width32_rollout"] and max(challenger[str(width)]["invalid_rate"] for width in eval_widths) <= gates["maximum_invalid_rate"]; decision = "advance-persistent-neural-gpu-curriculum" if passed else "reject-persistent-neural-gpu-smoke"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash}, "arms": results, "comparison": {"width16_control": control["16"]["exact_accuracy"], "width16_persistent": challenger["16"]["exact_accuracy"], "width16_gain": gain16}, "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "torch_threads": config["training"]["torch_threads"]}, "limitations": ["Fresh generated smoke evidence only; checked and private answers were not read.", "Trajectory counts and parameter counts are matched closely, but control and challenger network operations differ by design.", "Exact bigint arithmetic is confined to offline prefix labels and evaluation truth."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"width16_control": summary["comparison"]["width16_control"], "width16_persistent": summary["comparison"]["width16_persistent"], "width16_gain": gain16, "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": summary["comparison"], "arms": {name: {"parameters": arm["parameters"], "evaluation": arm["evaluation"], "checkpoint": arm["checkpoint"], "runtime": arm["runtime"]} for name, arm in results.items()}, "runtime": summary["runtime"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
