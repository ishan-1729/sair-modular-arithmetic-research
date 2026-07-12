"""Matched frozen-source versus direct width-128 autonomous-state recovery."""

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
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_601_tier6_autonomous_state_recovery.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-601-tier6-autonomous-state-recovery"); args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp601", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    exp320 = load_module("exp320_for_exp601", ROOT / "experiments/exp_320_cellular_width64_curriculum/run.py")
    exp329 = load_module("exp329_for_exp601", ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py")
    cellular = load_module("cellular_for_exp601", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    seed = int(config["seed"]); transition_width = config["training"]["transition_width"]; recovery_width = config["training"]["recovery_width"]
    def split(width):
        local = {"data": {"prime_bits_min": width - config["data"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
        return exp300.bigint_prime_split(local, seed + 60100 + width)
    splits = {width: split(width) for width in (transition_width, recovery_width)}; dataset_hash = sha256_text(json.dumps(splits, sort_keys=True))
    torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); runner_hash = sha256(Path(__file__)); config_hash = sha256(config_path); started = time.perf_counter(); cpu_started = time.process_time(); process = psutil.Process()
    def new_model():
        model = cellular.SharedDilatedConvGRU(width=recovery_width, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]); model.load_state_dict(load_file(source)); return model
    def evaluate(model, width):
        model.width = width
        return {"transition": exp300.evaluate_transitions(model, exp300.BigIntSampler(splits[width]["test"], seed + 60500 + width), config["evaluation"]["transition_cases"]), "rollout": exp300.evaluate_rollouts(model, exp300.BigIntSampler(splits[width]["test"], seed + 60600 + width), config["evaluation"]["rollout_cases"])}
    cache_key = sha256_text(json.dumps({"source": sha256(source), "config": config_hash, "dataset": dataset_hash}, sort_keys=True)); cache_path = output / f"source-{cache_key}.json"
    if cache_path.exists(): control = json.loads(cache_path.read_text())
    else:
        control_model = new_model(); control_model.eval(); control = {str(width): evaluate(control_model, width) for width in (transition_width, recovery_width)}; cache_path.write_text(json.dumps(control, indent=2) + "\n")
    model = new_model(); optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"])
    transition_sampler = exp300.BigIntSampler(splits[transition_width]["train"], seed + 60264); recovery_sampler = exp300.BigIntSampler(splits[recovery_width]["train"], seed + 60328); selector = torch.Generator().manual_seed(seed + 6012); recovery_rng = random.Random(seed + 60428); pool_size = config["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"]
    history = []; peak_rss = process.memory_info().rss; experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); commit = git_output("rev-parse", "HEAD")
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "tier6-autonomous-state-recovery-smoke", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"}); mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "source_checkpoint_sha256": sha256(source), "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration"); model.train()
        for step in range(1, config["training"]["steps"] + 1):
            recovery = step % config["training"]["recovery_update_period"] == 0
            if not recovery:
                values = transition_sampler.transitions(pool_size); state, multiplicand, modulus, input_bit, target, quotient = exp320.mine(values, config["training"]["batch_size"], config, selector); model.width = transition_width; logits, quotient_logits = model(exp300.features(state, multiplicand, modulus, input_bit, transition_width)); loss, stats = exp300.transition_loss(logits, quotient_logits, target, quotient, transition_width, config["training"]["worst_bit_weight"], config["training"]["quotient_weight"])
            else:
                batch = config["training"]["recovery_batch_size"]; chunk = config["training"]["recovery_chunk_steps"]; multiplier, multiplicand, modulus = recovery_sampler.pairs(batch); starts = [recovery_rng.randrange(0, recovery_width - chunk + 1) for _ in range(batch)]; teacher = exp329.teacher_states_before(multiplier, multiplicand, modulus, starts, recovery_width); exposed = list(teacher); losses = []; exact = []; probability = config["training"]["scheduled_prediction_probability_start"] + (config["training"]["scheduled_prediction_probability_end"] - config["training"]["scheduled_prediction_probability_start"]) * (step - 1) / max(1, config["training"]["steps"] - 1); model.width = recovery_width
                for offset in range(chunk):
                    positions = [start + offset for start in starts]; input_bit = [(left >> (recovery_width - 1 - position)) & 1 for left, position in zip(multiplier, positions)]; raw = [2 * value + bit * right for value, bit, right in zip(teacher, input_bit, multiplicand)]; target = [value % prime for value, prime in zip(raw, modulus)]; quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long); logits, quotient_logits = model(exp300.features(exposed, multiplicand, modulus, input_bit, recovery_width)); part, part_stats = exp300.transition_loss(logits, quotient_logits, target, quotient, recovery_width, config["training"]["worst_bit_weight"], 0.0); losses.append(part); exact.append(part_stats["batch_exact"]); predicted = exp329.decode_logits(logits.detach(), recovery_width); exposed = [guess if recovery_rng.random() < probability else truth for guess, truth in zip(predicted, target)]; teacher = target
                loss = torch.stack(losses).mean(); stats = {"loss": float(loss.detach()), "batch_exact": sum(exact) / len(exact), "mean_bce": float("nan"), "worst_bit_bce": float("nan"), "quotient_ce": 0.0}
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 25 == 0 or step == config["training"]["steps"]:
                peak_rss = max(peak_rss, process.memory_info().rss); record = {"step": step, "mode": "recovery" if recovery else "transition", "elapsed_wall_seconds": time.perf_counter() - started, "gradient_norm": gradient, **stats}; history.append(record); mlflow.log_metrics({"train_loss": stats["loss"], "train_batch_exact": stats["batch_exact"], "gradient_norm": gradient}, step=step); print(json.dumps(record), flush=True)
        model.eval(); checkpoint = output / "weights.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); challenger = {str(width): evaluate(model, width) for width in (transition_width, recovery_width)}
        source128 = control[str(recovery_width)]["rollout"]["exact_accuracy"]; challenger128 = challenger[str(recovery_width)]["rollout"]["exact_accuracy"]; gates = config["gates"]; passed = challenger128 >= gates["minimum_width128_rollout"] and challenger128 - source128 >= gates["minimum_width128_gain_over_source"] and challenger[str(transition_width)]["rollout"]["exact_accuracy"] >= gates["minimum_retained_width64_rollout"] and challenger[str(recovery_width)]["transition"]["exact_accuracy"] >= gates["minimum_width128_transition"] and challenger[str(recovery_width)]["rollout"]["any_invalid_residue_rate"] <= gates["maximum_width128_invalid_rollout_rate"]; decision = "advance-tier6-recovery-full-budget" if passed else "reject-tier6-autonomous-state-recovery-smoke"
        summary = {"run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"], "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": commit, "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": config_hash, "runner_sha256": runner_hash, "dataset_sha256": dataset_hash, "source_checkpoint_sha256": sha256(source)}, "control": control, "challenger": {"parameters": sum(p.numel() for p in model.parameters()), "history": history, "evaluation": challenger, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}}, "comparison": {"source_width128_rollout": source128, "challenger_width128_rollout": challenger128, "width128_rollout_gain": challenger128 - source128}, "gates": {**gates, "passed": passed}, "decision": decision, "resource_budget": config["resource_budget"], "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "peak_rss_bytes": peak_rss, "torch_threads": config["training"]["torch_threads"]}, "limitations": ["Fresh generated smoke evidence only; checked, consumed independent, and private answers were not read.", "Recovery predictions are detached, matching the validated EXP-329 formulation.", "Python bigint arithmetic is confined to offline data, labels, and evaluation truth."]}; summary_path = output / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n"); mlflow.log_metrics({"source_width128_rollout": source128, "challenger_width128_rollout": challenger128, "width128_rollout_gain": challenger128 - source128, "retained_width64_rollout": challenger[str(transition_width)]["rollout"]["exact_accuracy"], "wall_seconds": summary["runtime"]["wall_seconds"], "aggregate_cpu_seconds": summary["runtime"]["aggregate_cpu_seconds"]}); mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "comparison": summary["comparison"], "control": control, "challenger": challenger, "checkpoint": summary["challenger"]["checkpoint"], "runtime": summary["runtime"], "mlflow": summary["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
