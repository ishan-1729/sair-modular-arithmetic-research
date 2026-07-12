"""Causal activation and channel-ablation analysis for the width-32 TCN."""

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
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
EXP100_DIR = ROOT / "experiments/exp_100_width_scaling"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(EXP100_DIR))
import model as exp100_model  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def decode(logits: torch.Tensor, width: int) -> torch.Tensor:
    weights = 1 << torch.arange(width, dtype=torch.int64)
    return ((logits >= 0).to(torch.int64) * weights[None, :]).sum(1)


@torch.inference_mode()
def evaluate(model, sampler, cases: int, ablated: list[int] | None = None) -> dict:
    width = model.width; batch_size = 1024; totals = {q: [0, 0] for q in range(3)}; correct = 0; seen = 0
    handle = None
    if ablated:
        def intervention(_module, _inputs, output):
            changed = output.clone(); changed[:, ablated, :] = 0.0; return changed
        handle = model.head_norm.register_forward_hook(intervention)
    try:
        while seen < cases:
            size = min(batch_size, cases - seen); state, multiplicand, modulus, bit, target, quotient = sampler.transitions(size); features = exp100_model.make_features(state, multiplicand, modulus, bit, width); logits, _ = model(features); matches = decode(logits, width).eq(target); correct += int(matches.sum())
            for q in range(3):
                mask = quotient.eq(q); totals[q][0] += int(matches[mask].sum()); totals[q][1] += int(mask.sum())
            seen += size
    finally:
        if handle is not None: handle.remove()
    return {"cases": seen, "correct": correct, "exact_accuracy": correct / seen, "by_quotient": {str(q): {"correct": values[0], "cases": values[1], "exact_accuracy": values[0] / values[1] if values[1] else None} for q, values in totals.items()}}


@torch.inference_mode()
def activation_statistics(model, sampler, cases: int) -> dict:
    hidden = model.hidden; groups = {"q0": [torch.zeros(hidden), 0], "q1": [torch.zeros(hidden), 0], "q2": [torch.zeros(hidden), 0], "correct": [torch.zeros(hidden), 0], "error": [torch.zeros(hidden), 0], "high_bits": [torch.zeros(hidden), 0], "lower_bits": [torch.zeros(hidden), 0]}; captured = []
    def capture(_module, _inputs, output): captured.append(output.detach())
    handle = model.head_norm.register_forward_hook(capture); seen = 0
    try:
        while seen < cases:
            size = min(1024, cases - seen); state, multiplicand, modulus, bit, target, quotient = sampler.transitions(size); captured.clear(); logits, _ = model(exp100_model.make_features(state, multiplicand, modulus, bit, model.width)); activations = F.silu(captured[0]).abs().mean(dim=-1).cpu(); matches = decode(logits, model.width).eq(target).cpu(); quotient_cpu = quotient.cpu(); high = torch.tensor([value.bit_length() >= 30 for value in modulus.tolist()])
            masks = {"q0": quotient_cpu.eq(0), "q1": quotient_cpu.eq(1), "q2": quotient_cpu.eq(2), "correct": matches, "error": ~matches, "high_bits": high, "lower_bits": ~high}
            for name, mask in masks.items():
                if int(mask.sum()): groups[name][0] += activations[mask].sum(dim=0); groups[name][1] += int(mask.sum())
            seen += size
    finally: handle.remove()
    means = {name: (values[0] / max(1, values[1])).tolist() for name, values in groups.items()}; counts = {name: values[1] for name, values in groups.items()}
    q2_selectivity = (torch.tensor(means["q2"]) - 0.5 * (torch.tensor(means["q0"]) + torch.tensor(means["q1"]))).tolist(); error_selectivity = (torch.tensor(means["error"]) - torch.tensor(means["correct"])).tolist(); high_selectivity = (torch.tensor(means["high_bits"]) - torch.tensor(means["lower_bits"])).tolist()
    return {"cases": seen, "counts": counts, "mean_absolute_activation": means, "q2_selectivity": q2_selectivity, "error_selectivity": error_selectivity, "high_bit_selectivity": high_selectivity}


def weight_statistics(model) -> dict:
    global_weights = model.global_head.weight.detach().reshape(model.width, model.hidden, model.width).abs().mean(dim=(0, 2)); quotient_weights = model.quotient_head.weight.detach().reshape(3, model.hidden, model.width).abs().mean(dim=(0, 2)); local_weights = model.local_head.weight.detach().abs().reshape(model.hidden)
    return {"local_head_abs": local_weights.tolist(), "global_head_mean_abs": global_weights.tolist(), "quotient_head_mean_abs": quotient_weights.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_240_channel_causal_analysis.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-240-channel-causal-analysis.json"); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text()); checkpoint = (ROOT / config["checkpoint"]).resolve()
    if sha256(checkpoint) != config["checkpoint_sha256"]: raise RuntimeError("checkpoint hash mismatch")
    exp200 = load_module("exp200_for_exp240", ROOT / "experiments/exp_200_width32_transfer_ab/run.py"); exp204 = load_module("exp204_for_exp240", ROOT / "experiments/exp_204_official_distribution_finetune/run.py")
    split = exp204.official_numeric_prime_split(config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], config["seed"] + 240); model = exp100_model.SharedContextHornerTCN(**config["model"]); model.load_state_dict(load_file(str(checkpoint))); model.eval(); torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"]); started = time.perf_counter(); eval_seed = config["seed"] + 24050
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "causal-channel-analysis", "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "repository_commit": git_output("rev-parse", "HEAD"), "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()}); mlflow.log_params({"checkpoint_sha256": sha256(checkpoint), "baseline_cases": config["evaluation"]["baseline_transition_cases"], "activation_cases": config["evaluation"]["activation_cases"], "ablation_cases": config["evaluation"]["cases_per_channel_ablation"], "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__))}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / "knowledge/autoresearch/ablations/exp-240-channel-causal-analysis.md"), artifact_path="preregistration")
        baseline = evaluate(model, exp200.ScalableTransitionSampler(32, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], eval_seed, 0.0, 0.15), config["evaluation"]["baseline_transition_cases"]); activations = activation_statistics(model, exp200.ScalableTransitionSampler(32, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], eval_seed + 1, 0.0, 0.15), config["evaluation"]["activation_cases"])
        ablation_baseline = evaluate(model, exp200.ScalableTransitionSampler(32, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], eval_seed + 2, 0.0, 0.15), config["evaluation"]["cases_per_channel_ablation"])
        ablations = []
        for channel in range(model.hidden):
            result = evaluate(model, exp200.ScalableTransitionSampler(32, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], eval_seed + 2, 0.0, 0.15), config["evaluation"]["cases_per_channel_ablation"], [channel]); q2_base = ablation_baseline["by_quotient"]["2"]["exact_accuracy"]; q2_value = result["by_quotient"]["2"]["exact_accuracy"]; ablations.append({"channel": channel, "exact_accuracy": result["exact_accuracy"], "overall_drop": ablation_baseline["exact_accuracy"] - result["exact_accuracy"], "q2_accuracy": q2_value, "q2_drop": q2_base - q2_value})
        ranked = sorted(ablations, key=lambda item: (item["overall_drop"], item["q2_drop"]), reverse=True); top = [item["channel"] for item in ranked[:config["evaluation"]["combined_ablation_channels"]]]; combined = evaluate(model, exp200.ScalableTransitionSampler(32, config["evaluation"]["prime_bits_min"], config["evaluation"]["prime_bits_max"], split["test"], eval_seed + 2, 0.0, 0.15), config["evaluation"]["cases_per_channel_ablation"], top)
        critical = [item for item in ablations if item["overall_drop"] >= config["selection"]["critical_overall_drop"] or item["q2_drop"] >= config["selection"]["critical_quotient2_drop"]]; sparse = len(critical) <= config["selection"]["maximum_sparse_critical_channels"]
        decision = "targeted-channel-redundancy-followup" if sparse and critical else "distributed-representation-no-channel-surgery"; payload = {"run_id": config["run_id"], "status": "completed", "checkpoint": {"path": config["checkpoint"], "sha256": sha256(checkpoint)}, "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id}, "source": {"repository_commit": git_output("rev-parse", "HEAD"), "dirty": bool(git_output("status", "--porcelain")), "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__))}, "baseline": baseline, "ablation_baseline": ablation_baseline, "activations": activations, "weights": weight_statistics(model), "single_channel_ablations": ablations, "ranked_channels": ranked, "combined_top_channels": {"channels": top, "result": combined, "overall_drop": ablation_baseline["exact_accuracy"] - combined["exact_accuracy"]}, "selection": {"critical_channels": critical, "critical_count": len(critical), "sparse": sparse}, "decision": decision, "runtime": {"wall_seconds": time.perf_counter() - started, "device": "cpu", "torch_threads": 4}, "limitations": ["Final-normalization channel ablation tests one causal site, not every internal use of a channel.", "Transition effects do not directly establish autonomous-rollout effects.", "Fresh generated identities were used; no consumed independent or checked answer was read."]}; args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n"); mlflow.log_metrics({"baseline_transition": baseline["exact_accuracy"], "ablation_baseline_transition": ablation_baseline["exact_accuracy"], "critical_channel_count": len(critical), "combined_top4_drop": payload["combined_top_channels"]["overall_drop"], "wall_seconds": payload["runtime"]["wall_seconds"]}); mlflow.log_artifact(str(args.output), artifact_path="canonical-preview"); mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({"decision": decision, "baseline": baseline, "critical_count": len(critical), "top_channels": ranked[:10], "combined": payload["combined_top_channels"], "mlflow": payload["mlflow"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
