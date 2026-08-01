"""Matched detached-hard versus differentiable-soft learner-state recovery."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_607_tier6_differentiable_soft_state.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-607-tier6-differentiable-soft-state"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp607", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"); exp320 = load_module("exp320_for_exp607", ROOT / "experiments/exp_320_cellular_width64_curriculum/run.py"); exp329 = load_module("exp329_for_exp607", ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py"); cellular = load_module("cellular_for_exp607", ROOT / "experiments/exp_220_width_independent_cellular/model.py"); widening = load_module("widening_for_exp607", ROOT / "experiments/exp_611_tier6_function_preserving_widening/model.py"); refinement = load_module("refinement_for_exp607", ROOT / "experiments/exp_614_tier6_global_recurrent_refinement/model.py"); lora = load_module("lora_for_exp607", ROOT / "experiments/exp_617_tier6_recurrent_core_lora/model.py"); seed = int(config["seed"]); widths = (64, 128)
    def make_split(width):
        local = {"data": {"prime_bits_min": width - config["data"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}; return exp300.bigint_prime_split(local, seed + 60700 + width)
    splits = {width: make_split(width) for width in widths}; dataset_hash = sha256_text(json.dumps(splits, sort_keys=True)); torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); config_hash = sha256(config_path); runner_hash = sha256(Path(__file__)); started = time.perf_counter(); cpu_started = time.process_time(); process = psutil.Process(); experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); commit = git_output("rev-parse", "HEAD")
    def new_model(arm=None):
        base = cellular.SharedDilatedConvGRU(width=128, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]); base.load_state_dict(load_file(source))
        lora_rank = (arm or {}).get("recurrent_lora_rank")
        if lora_rank is not None:
            return lora.RecurrentCoreLoRACell(base, rank=int(lora_rank), alpha=float((arm or {}).get("recurrent_lora_alpha", lora_rank)))
        refinement_passes = (arm or {}).get("global_refinement_passes")
        if refinement_passes is not None:
            block = refinement.GlobalRefinementBlock(hidden=config["model"]["hidden"], passes=int(refinement_passes))
            return refinement.GloballyRefinedCell(base, block)
        adapter_hidden = (arm or {}).get("residual_adapter_hidden")
        if adapter_hidden is None: return base
        adapter = cellular.SharedDilatedConvGRU(width=128, hidden=int(adapter_hidden), cycles=config["model"]["cycles"])
        return widening.ResidualWidenedCell(base, adapter)
    def soft_features(state_plane, right, modulus, bit, width):
        batch = len(right); bit_plane = torch.tensor(bit, dtype=torch.float32)[:, None].expand(batch, width); lsb = torch.zeros(batch, width); msb = torch.zeros(batch, width); lsb[:, 0] = 1; msb[:, -1] = 1; return torch.stack((state_plane, exp300.bits(right, width), exp300.bits(modulus, width), bit_plane, lsb, msb), dim=1).mul(2).sub(1)
    def evaluate(model, width):
        model.width = width; return {"transition": exp300.evaluate_transitions(model, exp300.BigIntSampler(splits[width]["test"], seed + 61100 + width), config["evaluation"]["transition_cases"]), "rollout": exp300.evaluate_rollouts(model, exp300.BigIntSampler(splits[width]["test"], seed + 61200 + width), config["evaluation"]["rollout_cases"])}
    def train_arm(name, arm):
        arm_dir = output / name; arm_dir.mkdir(parents=True, exist_ok=True); cache_key = sha256_text(json.dumps({"source": sha256(source), "config": config_hash, "dataset": dataset_hash, "runner": runner_hash, "arm": arm}, sort_keys=True)); cache_path = arm_dir / f"result-{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            merged_record = payload.get("result", {}).get("merged_checkpoint"); merged_valid = merged_record is None or (ROOT / merged_record["path"]).exists()
            if payload.get("cache_key") == cache_key and (ROOT / payload["result"]["checkpoint"]["path"]).exists() and merged_valid: return {**payload["result"], "cached": True}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = random.Random(seed + 71); corruption_rng = random.Random(seed + 74); selector = torch.Generator().manual_seed(seed + 72); model = new_model(arm); transition_sampler = exp300.BigIntSampler(splits[64]["train"], seed + 73); recovery_samplers = {width: exp300.BigIntSampler(splits[width]["train"], seed + 100 + width) for width in widths}
        lora_lr = arm.get("lora_learning_rate")
        refinement_lr = arm.get("refinement_learning_rate")
        adapter_lr = arm.get("adapter_learning_rate")
        if lora_lr is not None:
            optimizer = torch.optim.AdamW([model.gate_a, model.gate_b, model.candidate_a, model.candidate_b], lr=float(lora_lr), weight_decay=config["training"]["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=float(arm.get("minimum_lora_learning_rate", lora_lr * 0.1)))
        elif refinement_lr is not None:
            optimizer = torch.optim.AdamW([model.blend, *model.refinement.parameters()], lr=float(refinement_lr), weight_decay=config["training"]["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=float(arm.get("minimum_refinement_learning_rate", refinement_lr * 0.1)))
        elif adapter_lr is not None:
            optimizer = torch.optim.AdamW([{"params": model.base.parameters(), "lr": config["training"]["learning_rate"]}, {"params": model.adapter.parameters(), "lr": float(adapter_lr)}], weight_decay=config["training"]["weight_decay"])
            minimum_ratio = config["training"]["minimum_learning_rate"] / config["training"]["learning_rate"]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + np.cos(np.pi * step / config["training"]["steps"])))
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"])
        pool_size = config["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"]; history = []; arm_started = time.perf_counter(); arm_cpu = time.process_time(); peak = process.memory_info().rss
        def transition_objective(logits, qlogits, target, quotient, width, quotient_weight):
            loss, stats = exp300.transition_loss(logits, qlogits, target, quotient, width, config["training"]["worst_bit_weight"], quotient_weight)
            casewise_weight = float(arm.get("casewise_worst_bit_weight", 0.0)); target_bits = exp300.bits(target, width); casewise = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_bits, reduction="none").amax(dim=1).mean()
            if casewise_weight: loss = loss + casewise_weight * casewise
            stats["casewise_worst_bit_bce"] = float(casewise.detach()); stats["casewise_worst_bit_weight"] = casewise_weight; stats["loss"] = float(loss.detach()); return loss, stats
        def recovery_loss(width):
            batch = config["training"]["recovery_batch_size"]; chunk = config["training"]["recovery_chunk_steps"]; multiplier, right, modulus = recovery_samplers[width].pairs(batch); starts = [rng.randrange(0, width - chunk + 1) for _ in range(batch)]; teacher = exp329.teacher_states_before(multiplier, right, modulus, starts, width); state_plane = exp300.bits(teacher, width)
            corruption = arm.get("initial_state_corruption")
            if corruption and width in corruption["widths"]:
                state_plane = state_plane.clone()
                for row in range(batch):
                    count = 1 if corruption_rng.random() < corruption["one_bit_probability"] else 2
                    for position in corruption_rng.sample(range(width), count): state_plane[row, position] = 1.0 - state_plane[row, position]
            losses = []; exact = []; casewise = []; trajectory_vectors = []; model.width = width
            for offset in range(chunk):
                positions = [start + offset for start in starts]; bit = [(left >> (width - 1 - position)) & 1 for left, position in zip(multiplier, positions)]; raw = [2 * state + flag * value for state, flag, value in zip(teacher, bit, right)]; target = [value % prime for value, prime in zip(raw, modulus)]; quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long); logits, qlogits = model(soft_features(state_plane, right, modulus, bit, width)); part, stats = transition_objective(logits, qlogits, target, quotient, width, 0.0); losses.append(part); exact.append(stats["batch_exact"]); casewise.append(stats["casewise_worst_bit_bce"]); trajectory_vectors.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, exp300.bits(target, width), reduction="none").amax(dim=1))
                state_mode = arm.get("state_mode", "soft" if arm["differentiable_soft_state"] else "detached"); probabilities = torch.sigmoid(logits)
                if state_mode == "soft": state_plane = probabilities
                elif state_mode == "detached": state_plane = logits.ge(0).float().detach()
                elif state_mode == "straight_through": state_plane = logits.ge(0).float() + probabilities - probabilities.detach()
                else: raise ValueError(f"unknown state mode {state_mode}")
                teacher = target
            loss = torch.stack(losses).mean(); trajectory_worst = torch.stack(trajectory_vectors).amax(dim=0); fraction = float(arm.get("trajectory_cvar_fraction", 0.25)); tail_count = max(1, int(np.ceil(batch * fraction))); trajectory_cvar = torch.topk(trajectory_worst, tail_count).values.mean(); trajectory_weight = float(arm.get("trajectory_cvar_weight", 0.0))
            if trajectory_weight: loss = loss + trajectory_weight * trajectory_cvar
            return loss, {"loss": float(loss.detach()), "batch_exact": sum(exact) / len(exact), "casewise_worst_bit_bce": sum(casewise) / len(casewise), "trajectory_cvar": float(trajectory_cvar.detach()), "trajectory_cvar_weight": trajectory_weight}
        model.train(); schedule = config["training"]["mode_schedule"]
        for step in (range(1, config["training"]["steps"] + 1) if arm.get("train", True) else ()):
            mode = schedule[(step - 1) % len(schedule)]
            if mode == "transition64":
                values = transition_sampler.transitions(pool_size); state, right, modulus, bit, target, quotient = exp320.mine(values, config["training"]["batch_size"], config, selector); model.width = 64; logits, qlogits = model(exp300.features(state, right, modulus, bit, 64)); loss, full_stats = transition_objective(logits, qlogits, target, quotient, 64, config["training"]["quotient_weight"]); stats = {"loss": full_stats["loss"], "batch_exact": full_stats["batch_exact"], "casewise_worst_bit_bce": full_stats["casewise_worst_bit_bce"]}
            else: loss, stats = recovery_loss(int(mode.removeprefix("recovery")))
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 50 == 0 or step == config["training"]["steps"]:
                peak = max(peak, process.memory_info().rss); record = {"step": step, "mode": mode, "elapsed_wall_seconds": time.perf_counter() - arm_started, "gradient_norm": gradient, **stats}; history.append(record); mlflow.log_metrics({f"{name}_loss": stats["loss"], f"{name}_batch_exact": stats["batch_exact"]}, step=step); print(json.dumps({"arm": name, **record}), flush=True)
        model.eval(); checkpoint = arm_dir / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); merged_checkpoint = None; merge_verification = None
        if hasattr(model, "merged_base"):
            merged_checkpoint = arm_dir / "merged-weights.safetensors"; merged = model.merged_base().eval(); save_file({key: value.detach().contiguous() for key, value in merged.state_dict().items()}, merged_checkpoint); merge_verification = {}
            with torch.no_grad():
                for width in widths:
                    representative = torch.randn(256, 6, width); model.width = width; merged.width = width; adapted_outputs = model(representative); merged_outputs = merged(representative); merge_verification[str(width)] = {"maximum_logit_absolute_difference": float((adapted_outputs[0] - merged_outputs[0]).abs().max()), "maximum_quotient_absolute_difference": float((adapted_outputs[1] - merged_outputs[1]).abs().max()), "bit_predictions_equal": torch.equal(adapted_outputs[0].ge(0), merged_outputs[0].ge(0)), "quotient_predictions_equal": torch.equal(adapted_outputs[1].argmax(1), merged_outputs[1].argmax(1))}
        evaluation = {str(width): evaluate(model, width) for width in widths}; activation = None
        if hasattr(model, "adapter"):
            activation = {}
            with torch.no_grad():
                for width in widths:
                    values = exp300.BigIntSampler(splits[width]["test"], seed + 61300 + width).transitions(config.get("activation_diagnostic_cases", 1000)); state, right, modulus, bit, *_ = values; feature_batch = exp300.features(state, right, modulus, bit, width); model.width = width; model.base.width = width; model.adapter.width = width; base_logits, base_quotient = model.base(feature_batch); adapter_logits, adapter_quotient = model.adapter(feature_batch)
                    activation[str(width)] = {"cases": len(state), "adapter_logit_rms": float(adapter_logits.square().mean().sqrt()), "adapter_logit_max_abs": float(adapter_logits.abs().max()), "adapter_quotient_rms": float(adapter_quotient.square().mean().sqrt()), "bit_prediction_elements_changed": int((base_logits.ge(0) != (base_logits + adapter_logits).ge(0)).sum()), "quotient_predictions_changed": int((base_quotient.argmax(1) != (base_quotient + adapter_quotient).argmax(1)).sum())}
        elif hasattr(model, "refinement"):
            activation = {"blend_rms": float(torch.tanh(model.blend.detach()).square().mean().sqrt()), "blend_max_abs": float(torch.tanh(model.blend.detach()).abs().max()), "blend_nonzero_channels": int((model.blend.detach() != 0).sum())}
        elif hasattr(model, "delta_gate"):
            activation = {"gate_delta_rms": float(model.delta_gate().detach().square().mean().sqrt()), "gate_delta_max_abs": float(model.delta_gate().detach().abs().max()), "candidate_delta_rms": float(model.delta_candidate().detach().square().mean().sqrt()), "candidate_delta_max_abs": float(model.delta_candidate().detach().abs().max())}
        result = {"arm": name, "cached": False, "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "training": {"steps": config["training"]["steps"], "history": history, **arm}, "evaluation": evaluation, "activation": activation, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "merged_checkpoint": ({"path": str(merged_checkpoint.relative_to(ROOT)), "sha256": sha256(merged_checkpoint), "bytes": merged_checkpoint.stat().st_size} if merged_checkpoint else None), "merge_verification": merge_verification, "runtime": {"wall_seconds": time.perf_counter() - arm_started, "aggregate_cpu_seconds": time.process_time() - arm_cpu, "peak_rss_bytes": peak}}; cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n"); return result
    exact_initialization = None
    enhanced_arms = [arm for arm in config["arms"].values() if arm.get("residual_adapter_hidden") is not None or arm.get("global_refinement_passes") is not None or arm.get("recurrent_lora_rank") is not None]
    if enhanced_arms:
        torch.manual_seed(seed + 611)
        reference = new_model()
        candidate = new_model(enhanced_arms[0])
        exact_initialization = {}
        with torch.no_grad():
            for width in widths:
                representative = torch.randn(config.get("exact_initialization_cases", 256), 6, width)
                reference.width = width; candidate.width = width
                reference_logits, reference_quotient = reference(representative); candidate_logits, candidate_quotient = candidate(representative)
                exact_initialization[str(width)] = {"logits_equal": torch.equal(reference_logits, candidate_logits), "quotient_equal": torch.equal(reference_quotient, candidate_quotient), "maximum_logit_absolute_difference": float((reference_logits - candidate_logits).abs().max()), "maximum_quotient_absolute_difference": float((reference_quotient - candidate_quotient).abs().max())}
        if not all(item["logits_equal"] and item["quotient_equal"] for item in exact_initialization.values()): raise RuntimeError("function-preserving widening gate failed")
    results = {}; run_ids = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "tier6-differentiable-soft-state-ab", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"}); mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "source_checkpoint_sha256": sha256(source), "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        for name, arm in config["arms"].items():
            with mlflow.start_run(run_name=name, nested=True) as nested:
                run_ids[name] = nested.info.run_id; results[name] = train_arm(name, arm); mlflow.log_metrics({"width64_rollout": results[name]["evaluation"]["64"]["rollout"]["exact_accuracy"], "width128_transition": results[name]["evaluation"]["128"]["transition"]["exact_accuracy"], "width128_rollout": results[name]["evaluation"]["128"]["rollout"]["exact_accuracy"]}); mlflow.set_tag("status", "completed")
        comparison = config.get("comparison", {"control_arm": "detached_hard_state", "challenger_arm": "differentiable_soft_state"}); control_result = results[comparison["control_arm"]]; challenger_result = results[comparison["challenger_arm"]]; control = control_result["evaluation"]; challenger = challenger_result["evaluation"]; gain = challenger["128"]["rollout"]["exact_accuracy"] - control["128"]["rollout"]["exact_accuracy"]; gates = config["gates"]; passed = challenger["128"]["rollout"]["exact_accuracy"] >= gates["minimum_challenger_width128_rollout"] and gain >= gates["minimum_width128_gain_over_control"] and challenger["64"]["rollout"]["exact_accuracy"] >= gates["minimum_challenger_width64_rollout"] and challenger["128"]["transition"]["exact_accuracy"] >= gates["minimum_challenger_width128_transition"] and challenger["128"]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_challenger_width128_invalid_rollout_rate"]
        if gates.get("minimum_challenger_adapter_logit_rms") is not None: passed = passed and challenger_result["activation"]["128"]["adapter_logit_rms"] >= gates["minimum_challenger_adapter_logit_rms"]
        if gates.get("minimum_challenger_prediction_changes") is not None: passed = passed and (challenger_result["activation"]["128"]["bit_prediction_elements_changed"] + challenger_result["activation"]["128"]["quotient_predictions_changed"]) >= gates["minimum_challenger_prediction_changes"]
        if gates.get("minimum_challenger_blend_rms") is not None: passed = passed and challenger_result["activation"]["blend_rms"] >= gates["minimum_challenger_blend_rms"]
        if gates.get("minimum_challenger_gate_delta_rms") is not None: passed = passed and challenger_result["activation"]["gate_delta_rms"] >= gates["minimum_challenger_gate_delta_rms"]
        decision = config.get("pass_decision", "advance-tier6-soft-state") if passed else config.get("fail_decision", "reject-tier6-differentiable-soft-state")
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "parent_run_id": parent_id, "arm_run_ids": run_ids}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash, "source_checkpoint_sha256": sha256(source)}, "exact_initialization": exact_initialization, "arms": results, "comparison": {"control_width128_rollout": control["128"]["rollout"]["exact_accuracy"], "challenger_width128_rollout": challenger["128"]["rollout"]["exact_accuracy"], "width128_rollout_gain": gain}, "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "torch_threads": config["training"]["torch_threads"]}, "limitations": ["Matched fresh generated selection evidence only; checked, independent, and private answers were not read.", "Soft or straight-through states are training-only relaxations; evaluated inference remains thresholded and discrete.", "Enhanced arms may change parameter count and wall time; updates and data remain matched and unmatched variables are disclosed in the preregistration.", "Exact bigint arithmetic is confined to offline labels and evaluation truth."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"control_width128_rollout": summary["comparison"]["control_width128_rollout"], "challenger_width128_rollout": summary["comparison"]["challenger_width128_rollout"], "width128_rollout_gain": gain, "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": summary["comparison"], "arms": {name: {"evaluation": arm["evaluation"], "checkpoint": arm["checkpoint"], "runtime": arm["runtime"]} for name, arm in results.items()}, "runtime": summary["runtime"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
