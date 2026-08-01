"""Matched base continuation versus persistent confidence-augmented state."""

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
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_619_tier6_confidence_augmented_state.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-619-tier6-confidence-augmented-state"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp619", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); exp320 = load_module("exp320_for_exp619", ROOT / "experiments/exp_320_cellular_width64_curriculum/run.py"); exp329 = load_module("exp329_for_exp619", ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py"); cellular = load_module("cellular_for_exp619", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); confidence_model = load_module("confidence_for_exp619", ROOT / "experiments/exp_619_tier6_confidence_augmented_state/model.py"); seed = int(config["seed"]); widths = (64, 128)
    def make_split(width):
        local = {"data": {"prime_bits_min": width - config["data"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
        return exp300.bigint_prime_split(local, seed + 61900 + width)
    splits = {width: make_split(width) for width in widths}; dataset_hash = sha256_text(json.dumps(splits, sort_keys=True)); torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); process = psutil.Process(); started = time.perf_counter(); cpu_started = time.process_time(); config_hash = sha256(config_path); runner_hash = sha256(Path(__file__)); commit = git_output("rev-parse", "HEAD"); experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    def base_model():
        model = cellular.SharedDilatedConvGRU(width=128, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]); model.load_state_dict(load_file(source)); return model
    def make_model(confidence): return confidence_model.ConfidenceAugmentedCell(base_model()) if confidence else base_model()
    def soft_features(state_plane, right, modulus, bit, width):
        batch = len(right); bit_plane = torch.tensor(bit, dtype=torch.float32)[:, None].expand(batch, width); lsb = torch.zeros(batch, width); msb = torch.zeros(batch, width); lsb[:, 0] = 1; msb[:, -1] = 1; return torch.stack((state_plane, exp300.bits(right, width), exp300.bits(modulus, width), bit_plane, lsb, msb), dim=1).mul(2).sub(1)
    def augmented_features(state_plane, right, modulus, bit, width, confidence): return torch.cat((soft_features(state_plane, right, modulus, bit, width), confidence), dim=1)
    @torch.inference_mode()
    def evaluate_augmented_transitions(model, width, cases):
        model.width = width; sampler = exp300.BigIntSampler(splits[width]["test"], seed + 62200 + width); seen = correct = valid = 0
        while seen < cases:
            size = min(512, cases - seen); state, right, modulus, bit, target, _ = sampler.transitions(size); features = augmented_features(exp300.bits(state, width), right, modulus, bit, width, torch.zeros(size, 2, width)); logits, _ = model(features); predicted = logits.ge(0); truth = exp300.bits(target, width).bool(); values = decode(logits); correct += int((predicted == truth).all(1).sum()); valid += sum(value < prime for value, prime in zip(values, modulus)); seen += size
        return {"cases": seen, "correct": correct, "exact_accuracy": correct / seen, "valid_residue_rate": valid / seen}
    @torch.inference_mode()
    def evaluate_augmented_rollouts(model, width, cases):
        model.width = width; sampler = exp300.BigIntSampler(splits[width]["test"], seed + 62300 + width); seen = correct = all_steps = invalid = 0
        while seen < cases:
            size = min(128, cases - seen); left, right, modulus = sampler.pairs(size); state = [0] * size; teacher = [0] * size; confidence = torch.zeros(size, 2, width); every = [True] * size; bad = [False] * size
            for position in range(width):
                bit = [(value >> (width - 1 - position)) & 1 for value in left]; logits, _ = model(augmented_features(exp300.bits(state, width), right, modulus, bit, width, confidence)); state = decode(logits); confidence = model.confidence_from_logits(logits); teacher = [(2 * value + flag * other) % prime for value, flag, other, prime in zip(teacher, bit, right, modulus)]; every = [prior and guess == truth for prior, guess, truth in zip(every, state, teacher)]; bad = [prior or guess >= prime for prior, guess, prime in zip(bad, state, modulus)]
            expected = [(a * b) % prime for a, b, prime in zip(left, right, modulus)]; correct += sum(a == b for a, b in zip(state, expected)); all_steps += sum(every); invalid += sum(bad); seen += size
        return {"cases": seen, "correct": correct, "exact_accuracy": correct / seen, "all_steps_exact_accuracy": all_steps / seen, "any_invalid_residue_rate": invalid / seen}
    def evaluate(model, confidence, width):
        if confidence: return {"transition": evaluate_augmented_transitions(model, width, config["evaluation"]["transition_cases"]), "rollout": evaluate_augmented_rollouts(model, width, config["evaluation"]["rollout_cases"])}
        model.width = width; return {"transition": exp300.evaluate_transitions(model, exp300.BigIntSampler(splits[width]["test"], seed + 62200 + width), config["evaluation"]["transition_cases"]), "rollout": exp300.evaluate_rollouts(model, exp300.BigIntSampler(splits[width]["test"], seed + 62300 + width), config["evaluation"]["rollout_cases"])}
    torch.manual_seed(seed + 619); reference = base_model().eval(); exact_candidate = make_model(True).eval(); exact_initialization = {}
    with torch.no_grad():
        for width in widths:
            base_features = torch.randn(config["exact_initialization_cases"], 6, width); confidence_features = torch.randn(config["exact_initialization_cases"], 2, width); reference.width = width; exact_candidate.width = width; expected = reference(base_features); actual = exact_candidate(torch.cat((base_features, confidence_features), dim=1)); exact_initialization[str(width)] = {"logits_equal": torch.equal(expected[0], actual[0]), "quotient_equal": torch.equal(expected[1], actual[1]), "maximum_absolute_difference": max(float((expected[0] - actual[0]).abs().max()), float((expected[1] - actual[1]).abs().max()))}
    if not all(item["logits_equal"] and item["quotient_equal"] for item in exact_initialization.values()): raise RuntimeError("confidence exact-init gate failed")
    def train_arm(name, use_confidence):
        arm_dir = output / name; arm_dir.mkdir(parents=True, exist_ok=True); cache_key = sha256_text(json.dumps({"source": sha256(source), "config": config_hash, "dataset": dataset_hash, "runner": runner_hash, "confidence": use_confidence}, sort_keys=True)); cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == cache_key and (ROOT / payload["result"]["checkpoint"]["path"]).exists(): return {**payload["result"], "cached": True}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = random.Random(seed + 71); selector = torch.Generator().manual_seed(seed + 72); model = make_model(use_confidence)
        if use_confidence:
            parameters = [model.confidence_stem.weight, model.confidence_gate_weight, model.confidence_candidate_weight]; optimizer = torch.optim.AdamW(parameters, lr=config["training"]["confidence_learning_rate"], weight_decay=config["training"]["weight_decay"]); eta_min = config["training"]["minimum_confidence_learning_rate"]
        else:
            parameters = list(model.parameters()); optimizer = torch.optim.AdamW(parameters, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); eta_min = config["training"]["minimum_learning_rate"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=eta_min); transition_sampler = exp300.BigIntSampler(splits[64]["train"], seed + 73); recovery_samplers = {width: exp300.BigIntSampler(splits[width]["train"], seed + 100 + width) for width in widths}; pool_size = config["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"]; history = []; arm_started = time.perf_counter(); arm_cpu = time.process_time(); peak = process.memory_info().rss; model.train()
        def recovery_loss(width):
            batch = config["training"]["recovery_batch_size"]; chunk = config["training"]["recovery_chunk_steps"]; left, right, modulus = recovery_samplers[width].pairs(batch); starts = [rng.randrange(0, width - chunk + 1) for _ in range(batch)]; teacher = exp329.teacher_states_before(left, right, modulus, starts, width); state_plane = exp300.bits(teacher, width); confidence = torch.zeros(batch, 2, width); losses = []; exact = []; model.width = width
            for offset in range(chunk):
                positions = [start + offset for start in starts]; bit = [(value >> (width - 1 - position)) & 1 for value, position in zip(left, positions)]; raw = [2 * state + flag * value for state, flag, value in zip(teacher, bit, right)]; target = [value % prime for value, prime in zip(raw, modulus)]; quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long); features = augmented_features(state_plane, right, modulus, bit, width, confidence) if use_confidence else soft_features(state_plane, right, modulus, bit, width); logits, qlogits = model(features); loss, stats = exp300.transition_loss(logits, qlogits, target, quotient, width, config["training"]["worst_bit_weight"], 0.0); losses.append(loss); exact.append(stats["batch_exact"]); probabilities = torch.sigmoid(logits); state_plane = logits.ge(0).float() + probabilities - probabilities.detach()
                if use_confidence: confidence = model.confidence_from_logits(logits)
                teacher = target
            loss = torch.stack(losses).mean(); return loss, {"loss": float(loss.detach()), "batch_exact": sum(exact) / len(exact)}
        schedule = config["training"]["mode_schedule"]
        for step in range(1, config["training"]["steps"] + 1):
            mode = schedule[(step - 1) % len(schedule)]
            if mode == "transition64":
                values = transition_sampler.transitions(pool_size); state, right, modulus, bit, target, quotient = exp320.mine(values, config["training"]["batch_size"], config, selector); model.width = 64; features = augmented_features(exp300.bits(state, 64), right, modulus, bit, 64, torch.zeros(len(state), 2, 64)) if use_confidence else exp300.features(state, right, modulus, bit, 64); logits, qlogits = model(features); loss, stats = exp300.transition_loss(logits, qlogits, target, quotient, 64, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"])
            else: loss, stats = recovery_loss(int(mode.removeprefix("recovery")))
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 50 == 0 or step == config["training"]["steps"]:
                peak = max(peak, process.memory_info().rss); item = {"step": step, "mode": mode, "elapsed_wall_seconds": time.perf_counter() - arm_started, "gradient_norm": gradient, "loss": stats["loss"], "batch_exact": stats["batch_exact"]}; history.append(item); mlflow.log_metrics({f"{name}_loss": stats["loss"], f"{name}_batch_exact": stats["batch_exact"]}, step=step); print(json.dumps({"arm": name, **item}), flush=True)
        model.eval(); checkpoint = arm_dir / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); activation = None
        if use_confidence: activation = {"stem_weight_rms": float(model.confidence_stem.weight.detach().square().mean().sqrt()), "gate_weight_rms": float(model.confidence_gate_weight.detach().square().mean().sqrt()), "candidate_weight_rms": float(model.confidence_candidate_weight.detach().square().mean().sqrt())}
        result = {"arm": name, "cached": False, "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "training": {"steps": config["training"]["steps"], "history": history}, "evaluation": {str(width): evaluate(model, use_confidence, width) for width in widths}, "activation": activation, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "runtime": {"wall_seconds": time.perf_counter() - arm_started, "aggregate_cpu_seconds": time.process_time() - arm_cpu, "peak_rss_bytes": peak}}; cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n"); return result
    results = {}; run_ids = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "tier6-confidence-augmented-state", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()}); mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "source_checkpoint_sha256": sha256(source), "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for name, enabled in (("hard_state_control", False), ("confidence_state_challenger", True)):
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id; results[name] = train_arm(name, enabled); mlflow.log_metrics({"width64_rollout": results[name]["evaluation"]["64"]["rollout"]["exact_accuracy"], "width128_transition": results[name]["evaluation"]["128"]["transition"]["exact_accuracy"], "width128_rollout": results[name]["evaluation"]["128"]["rollout"]["exact_accuracy"]})
        control = results["hard_state_control"]["evaluation"]; challenger = results["confidence_state_challenger"]["evaluation"]; gain = challenger["128"]["rollout"]["exact_accuracy"] - control["128"]["rollout"]["exact_accuracy"]; gates = config["gates"]; passed = challenger["128"]["rollout"]["exact_accuracy"] >= gates["minimum_challenger_width128_rollout"] and gain >= gates["minimum_width128_gain"] and challenger["64"]["rollout"]["exact_accuracy"] >= gates["minimum_challenger_width64_rollout"] and challenger["128"]["transition"]["exact_accuracy"] >= gates["minimum_challenger_width128_transition"] and challenger["128"]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_invalid_rate"] and results["confidence_state_challenger"]["activation"]["gate_weight_rms"] >= gates["minimum_confidence_gate_weight_rms"]; decision = "advance-tier6-confidence-augmented-state" if passed else "reject-tier6-confidence-augmented-state"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash, "source_checkpoint_sha256": sha256(source)}, "exact_initialization": exact_initialization, "arms": results, "comparison": {"control_width128_rollout": control["128"]["rollout"]["exact_accuracy"], "challenger_width128_rollout": challenger["128"]["rollout"]["exact_accuracy"], "width128_rollout_gain": gain}, "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "torch_threads": config["training"]["torch_threads"]}, "limitations": ["Fresh generated selection evidence only; checked and private answers were not read.", "Recovery chunks initialize confidence to zero at their random starts, while inference carries confidence from the first step.", "Exact bigint arithmetic is confined to offline generation and evaluation truth."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"control_width128_rollout": summary["comparison"]["control_width128_rollout"], "challenger_width128_rollout": summary["comparison"]["challenger_width128_rollout"], "width128_rollout_gain": gain, "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": summary["comparison"], "arms": {name: {"evaluation": arm["evaluation"], "activation": arm["activation"], "checkpoint": arm["checkpoint"], "runtime": arm["runtime"]} for name, arm in results.items()}, "runtime": summary["runtime"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
