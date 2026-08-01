"""Independent cellular-seed diversity and target-free routing feasibility."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def make_data(config: dict[str, Any], exp300, exp633) -> dict[str, Any]:
    base = {
        "training": {"widths": config["training"]["widths"]},
        "evaluation": {"widths": [config["evaluation"]["width"]]},
        "data": {
            "prime_bit_span": config["data"]["prime_bit_span"],
            "train_prime_identities": config["data"]["train_prime_identities"],
            "test_prime_identities": config["control"][
                "original_test_prime_identities_width16"
            ],
            "train_prime_identities_by_width": config["data"][
                "train_prime_identities_by_width"
            ],
            "test_prime_identities_by_width": config["data"][
                "test_prime_identities_by_width"
            ],
        },
    }
    original = exp633.make_splits(
        exp300, base, int(config["control"]["source_data_seed"])
    )
    expanded = json.loads(json.dumps(base))
    width = str(config["evaluation"]["width"])
    expanded["data"]["test_prime_identities_by_width"][width] = (
        config["control"]["original_test_prime_identities_width16"]
        + config["data"]["fresh_test_prime_identities_width16"]
    )
    # YAML integer keys become strings through the JSON copy; normalize them.
    expanded["data"]["train_prime_identities_by_width"] = {
        int(key): value
        for key, value in expanded["data"]["train_prime_identities_by_width"].items()
    }
    expanded["data"]["test_prime_identities_by_width"] = {
        int(key): value
        for key, value in expanded["data"]["test_prime_identities_by_width"].items()
    }
    wider = exp633.make_splits(
        exp300, expanded, int(config["control"]["source_data_seed"])
    )
    eval_width = int(config["evaluation"]["width"])
    old_count = int(config["control"]["original_test_prime_identities_width16"])
    if wider[eval_width]["train"] != original[eval_width]["train"]:
        raise RuntimeError("expanded split changed the frozen control training identities")
    if wider[eval_width]["test"][:old_count] != original[eval_width]["test"]:
        raise RuntimeError("expanded split changed the EXP-633 evaluation prefix")
    fresh = wider[eval_width]["test"][old_count:]
    if set(fresh) & set(wider[eval_width]["train"]):
        raise RuntimeError("fresh evaluation identities overlap training identities")
    if set(fresh) & set(original[eval_width]["test"]):
        raise RuntimeError("fresh evaluation identities overlap EXP-633 evaluation")
    return {
        "training_splits": original,
        "fresh_evaluation_primes": fresh,
        "original_evaluation_primes": original[eval_width]["test"],
    }


def generated_cases(exp300, primes: list[int], seed: int, count: int) -> dict[str, list[int]]:
    left, right, modulus = exp300.BigIntSampler(primes, seed).pairs(count)
    return {"left": left, "right": right, "modulus": modulus}


@torch.inference_mode()
def rollout_details(model, exp300, cases: dict[str, list[int]], width: int) -> dict[str, Any]:
    left = cases["left"]
    right = cases["right"]
    modulus = cases["modulus"]
    state = [0] * len(modulus)
    minimum_margin = [math.inf] * len(modulus)
    invalid = [False] * len(modulus)
    started = time.perf_counter()
    cpu_started = time.process_time()
    model.width = width
    model.eval()
    for position in range(width):
        input_bit = [(value >> (width - 1 - position)) & 1 for value in left]
        logits, _ = model(exp300.features(state, right, modulus, input_bit, width))
        step_margin = logits.abs().amin(dim=1).tolist()
        minimum_margin = [min(old, new) for old, new in zip(minimum_margin, step_margin)]
        rows = logits.ge(0).tolist()
        state = [
            sum(int(value) << bit for bit, value in enumerate(row)) for row in rows
        ]
        invalid = [
            prior or prediction >= prime
            for prior, prediction, prime in zip(invalid, state, modulus)
        ]
    expected = [(a * b) % p for a, b, p in zip(left, right, modulus)]
    correctness = [prediction == target for prediction, target in zip(state, expected)]
    return {
        "cases": len(expected),
        "correct": sum(correctness),
        "exact_accuracy": sum(correctness) / len(expected),
        "invalid_rate": sum(invalid) / len(expected),
        "predictions": state,
        "expected": expected,
        "correctness": correctness,
        "minimum_margin": minimum_margin,
        "wall_seconds": time.perf_counter() - started,
        "aggregate_cpu_seconds": time.process_time() - cpu_started,
    }


def wilson_interval(correct: int, cases: int, z: float = 1.959963984540054) -> list[float]:
    proportion = correct / cases
    denominator = 1 + z * z / cases
    center = (proportion + z * z / (2 * cases)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / cases + z * z / (4 * cases * cases)
    ) / denominator
    return [center - radius, center + radius]


def paired_comparison(control: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    if control["expected"] != challenger["expected"]:
        raise RuntimeError("paired arms do not share expected answers")
    expected = control["expected"]
    control_correct = control["correctness"]
    challenger_correct = challenger["correctness"]
    routed_predictions = [
        challenger_prediction
        if challenger_margin > control_margin
        else control_prediction
        for control_prediction, challenger_prediction, control_margin, challenger_margin in zip(
            control["predictions"],
            challenger["predictions"],
            control["minimum_margin"],
            challenger["minimum_margin"],
        )
    ]
    routed_correct = [value == target for value, target in zip(routed_predictions, expected)]
    union_correct = [left or right for left, right in zip(control_correct, challenger_correct)]
    cases = len(expected)
    best = max(control["exact_accuracy"], challenger["exact_accuracy"])
    union_accuracy = sum(union_correct) / cases
    routed_accuracy = sum(routed_correct) / cases
    return {
        "cases": cases,
        "control_only": sum(left and not right for left, right in zip(control_correct, challenger_correct)),
        "challenger_only": sum(right and not left for left, right in zip(control_correct, challenger_correct)),
        "intersection": sum(left and right for left, right in zip(control_correct, challenger_correct)),
        "prediction_disagreement": sum(
            left != right for left, right in zip(control["predictions"], challenger["predictions"])
        ),
        "oracle_union_correct": sum(union_correct),
        "oracle_union_exact_accuracy": union_accuracy,
        "oracle_union_wilson95": wilson_interval(sum(union_correct), cases),
        "oracle_union_gain_over_best_arm": union_accuracy - best,
        "routed_correct": sum(routed_correct),
        "routed_exact_accuracy": routed_accuracy,
        "routed_wilson95": wilson_interval(sum(routed_correct), cases),
        "routed_gain_over_best_arm": routed_accuracy - best,
        "challenger_route_count": sum(
            right > left
            for left, right in zip(control["minimum_margin"], challenger["minimum_margin"])
        ),
    }


def clean_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    clean = {
        key: value
        for key, value in result.items()
        if key not in {"predictions", "expected", "correctness", "minimum_margin"}
    }
    clean["wilson95"] = wilson_interval(result["correct"], result["cases"])
    return clean


def initialize_model(config: dict[str, Any], cellular, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return cellular.SharedDilatedConvGRU(
        width=max(config["training"]["widths"]), **config["model"]
    )


def load_control(config: dict[str, Any], cellular):
    path = (ROOT / config["control"]["checkpoint"]).resolve()
    if sha256(path) != config["control"]["checkpoint_sha256"]:
        raise RuntimeError("frozen EXP-633 control checkpoint hash mismatch")
    model = initialize_model(config, cellular, 0)
    model.load_state_dict(load_file(path))
    model.eval()
    return model, path


def calibration(config, output, data, exp300, exp633, cellular, tree) -> dict[str, Any]:
    updates = int(config["resource_budget"]["calibration_training_updates"])
    inference_cases = int(config["resource_budget"]["calibration_inference_cases"])
    challenger = initialize_model(config, cellular, int(config["challenger"]["seed"]))
    optimizer = torch.optim.AdamW(
        challenger.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["steps"],
        eta_min=config["training"]["minimum_learning_rate"],
    )
    samplers = {
        width: exp300.BigIntSampler(
            data["training_splits"][width]["train"],
            int(config["challenger"]["seed"]) + 63600 + width,
        )
        for width in config["training"]["widths"]
    }
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    history, peak = exp633.train_steps(
        "control",
        challenger,
        optimizer,
        scheduler,
        samplers,
        1,
        updates,
        config,
        exp300,
        tree,
        [],
        wall_started,
    )
    training_wall = time.perf_counter() - wall_started
    training_cpu = time.process_time() - cpu_started
    control, _ = load_control(config, cellular)
    random_model = initialize_model(config, cellular, int(config["challenger"]["seed"]) + 1000)
    train_cases = generated_cases(
        exp300,
        data["training_splits"][config["evaluation"]["width"]]["train"],
        config["evaluation"]["case_seed"] + 1,
        inference_cases,
    )
    evaluations = {
        "control": rollout_details(control, exp300, train_cases, config["evaluation"]["width"]),
        "challenger": rollout_details(challenger, exp300, train_cases, config["evaluation"]["width"]),
        "randomized": rollout_details(random_model, exp300, train_cases, config["evaluation"]["width"]),
    }
    per_case_wall = {
        name: result["wall_seconds"] / inference_cases for name, result in evaluations.items()
    }
    per_case_cpu = {
        name: result["aggregate_cpu_seconds"] / inference_cases
        for name, result in evaluations.items()
    }
    formal_steps = int(config["training"]["steps"])
    paired_cases = int(config["evaluation"]["rollout_cases"])
    randomized_cases = int(config["evaluation"]["randomized_cases"])
    determinism_cases = int(config["evaluation"]["determinism_cases"])
    projected_training_wall = training_wall / updates * formal_steps
    projected_training_cpu = training_cpu / updates * formal_steps
    projected_evaluation_wall = (
        paired_cases * (per_case_wall["control"] + per_case_wall["challenger"])
        + randomized_cases * per_case_wall["randomized"]
        + determinism_cases * 2 * (per_case_wall["control"] + per_case_wall["challenger"])
    )
    projected_evaluation_cpu = (
        paired_cases * (per_case_cpu["control"] + per_case_cpu["challenger"])
        + randomized_cases * per_case_cpu["randomized"]
        + determinism_cases * 2 * (per_case_cpu["control"] + per_case_cpu["challenger"])
    )
    setup_allowance = 120
    expected_wall = projected_training_wall + projected_evaluation_wall + setup_allowance
    expected_cpu = projected_training_cpu + projected_evaluation_cpu + setup_allowance * config["training"]["torch_threads"]
    soft = math.ceil(expected_wall * 1.35 / 60) * 60
    hard = math.ceil(expected_wall * 2.0 / 60) * 60
    payload = {
        "run_id": config["run_id"],
        "status": "calibration-only",
        "training_identities_only": True,
        "training": {
            "updates": updates,
            "wall_seconds": training_wall,
            "aggregate_cpu_seconds": training_cpu,
            "history": history,
            "peak_rss_bytes": peak,
        },
        "inference": {name: clean_evaluation(value) for name, value in evaluations.items()},
        "projection": {
            "training_wall_seconds": projected_training_wall,
            "training_aggregate_cpu_seconds": projected_training_cpu,
            "evaluation_wall_seconds": projected_evaluation_wall,
            "evaluation_aggregate_cpu_seconds": projected_evaluation_cpu,
            "setup_serialization_tracking_allowance_wall_seconds": setup_allowance,
        },
        "recommendation": {
            "expected_total_wall_seconds": math.ceil(expected_wall),
            "expected_total_aggregate_cpu_minutes": math.ceil(expected_cpu / 60),
            "soft_budget_wall_seconds": soft,
            "hard_runaway_ceiling_wall_seconds": hard,
            "hard_ceiling_aggregate_cpu_minutes_upper_bound": hard
            * config["training"]["torch_threads"]
            / 60,
        },
        "limitations": [
            "Calibration uses training identities only and is not model-selection evidence.",
            "Wall and aggregate CPU projections are recorded separately.",
            "The recommendation includes all formal cases, determinism repeats, and setup allowance.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload


def train_challenger(config, output, data, source_hashes, exp300, exp633, cellular, tree, run_started):
    cache_key = sha256_text(json.dumps(source_hashes, sort_keys=True))
    result_path = output / f"challenger-result-{cache_key}.json"
    checkpoint = output / "challenger" / "weights.safetensors"
    if result_path.exists() and checkpoint.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("cache_key") == cache_key and sha256(checkpoint) == cached["checkpoint_sha256"]:
            model = initialize_model(config, cellular, 0)
            model.load_state_dict(load_file(checkpoint))
            model.eval()
            return model, {**cached["training"], "cached": True}, checkpoint
    model = initialize_model(config, cellular, int(config["challenger"]["seed"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"]
    )
    samplers = {
        width: exp300.BigIntSampler(
            data["training_splits"][width]["train"], int(config["challenger"]["seed"]) + 63600 + width
        )
        for width in config["training"]["widths"]
    }
    partial = output / "challenger" / f"training-{cache_key}.partial.pt"
    start_step = 1
    history: list[dict[str, Any]] = []
    if partial.exists():
        state = torch.load(partial, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        for width, rng_state in state["sampler_rng"].items():
            samplers[int(width)].rng.setstate(rng_state)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        start_step = int(state["step"]) + 1
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    peak = psutil.Process().memory_info().rss
    if start_step <= config["training"]["steps"]:
        history, peak = exp633.train_steps(
            "control", model, optimizer, scheduler, samplers, start_step, config["training"]["steps"],
            config, exp300, tree, history, wall_started, partial,
            run_started + config["resource_budget"]["hard_runaway_ceiling_wall_seconds"],
        )
    training = {
        "steps": config["training"]["steps"],
        "trajectory_examples": config["training"]["steps"] * config["training"]["batch_size"],
        "wall_seconds_this_invocation": time.perf_counter() - wall_started,
        "aggregate_cpu_seconds_this_invocation": time.process_time() - cpu_started,
        "history": history,
        "peak_rss_bytes": peak,
        "cached": False,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint)
    result_path.write_text(json.dumps({
        "cache_key": cache_key,
        "checkpoint_sha256": sha256(checkpoint),
        "training": training,
    }, indent=2) + "\n")
    model.eval()
    return model, training, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_635_independent_cellular_diversity_feasibility.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-635-independent-cellular-diversity-feasibility")
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    output = args.output.resolve()
    exp300 = load_module("exp300_for_exp635", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    exp633 = load_module("exp633_for_exp635", ROOT / "experiments/exp_633_hierarchical_learned_composition_feasibility/run.py")
    cellular = load_module("cellular_for_exp635", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    tree = load_module("tree_for_exp635", ROOT / "experiments/exp_633_hierarchical_learned_composition_feasibility/model.py")
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = make_data(config, exp300, exp633)
    if args.calibrate_only:
        calibration(config, output, data, exp300, exp633, cellular, tree)
        return 0
    budget = config["resource_budget"]
    if budget.get("status") != "calibrated" or not budget.get("hard_runaway_ceiling_wall_seconds"):
        raise RuntimeError("formal run requires a documented measured-throughput budget")
    if git_output("status", "--porcelain"):
        raise RuntimeError("formal run requires a clean committed worktree")
    output.mkdir(parents=True, exist_ok=True)
    commit = git_output("rev-parse", "HEAD")
    source_hashes = {
        "repository_commit": commit,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "exp633_runner_sha256": sha256(ROOT / "experiments/exp_633_hierarchical_learned_composition_feasibility/run.py"),
        "cellular_model_sha256": sha256(ROOT / "experiments/exp_220_width_independent_cellular/model.py"),
        "control_checkpoint_sha256": config["control"]["checkpoint_sha256"],
        "training_identity_sha256": sha256_text(json.dumps(data["training_splits"], sort_keys=True)),
        "fresh_evaluation_prime_sha256": sha256_text(json.dumps(data["fresh_evaluation_primes"])),
    }
    started = time.perf_counter()
    cpu_started = time.process_time()
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({
            "canonical_run_id": config["run_id"], "status": "running",
            "role": "independent-cellular-diversity-feasibility", "repository_commit": commit,
            "repository_dirty": "false",
        })
        mlflow.log_params({
            "seed": config["challenger"]["seed"], "steps": config["training"]["steps"],
            "config_sha256": source_hashes["config_sha256"], "runner_sha256": source_hashes["runner_sha256"],
            "control_checkpoint_sha256": source_hashes["control_checkpoint_sha256"],
            "fresh_evaluation_prime_sha256": source_hashes["fresh_evaluation_prime_sha256"],
            "hard_runaway_ceiling_wall_seconds": budget["hard_runaway_ceiling_wall_seconds"],
        })
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        control, control_path = load_control(config, cellular)
        challenger, training, challenger_path = train_challenger(
            config, output, data, source_hashes, exp300, exp633, cellular, tree, started
        )
        cases = generated_cases(
            exp300, data["fresh_evaluation_primes"], config["evaluation"]["case_seed"],
            config["evaluation"]["rollout_cases"],
        )
        cases_hash = sha256_text(json.dumps(cases, sort_keys=True))
        control_eval = rollout_details(control, exp300, cases, config["evaluation"]["width"])
        challenger_eval = rollout_details(challenger, exp300, cases, config["evaluation"]["width"])
        comparison = paired_comparison(control_eval, challenger_eval)
        random_model = initialize_model(config, cellular, int(config["challenger"]["seed"]) + 1000)
        random_cases = {key: values[: config["evaluation"]["randomized_cases"]] for key, values in cases.items()}
        randomized_eval = rollout_details(random_model, exp300, random_cases, config["evaluation"]["width"])
        repeat_cases = {key: values[: config["evaluation"]["determinism_cases"]] for key, values in cases.items()}
        control_repeat_a = rollout_details(control, exp300, repeat_cases, config["evaluation"]["width"])
        control_repeat_b = rollout_details(control, exp300, repeat_cases, config["evaluation"]["width"])
        challenger_repeat_a = rollout_details(challenger, exp300, repeat_cases, config["evaluation"]["width"])
        challenger_repeat_b = rollout_details(challenger, exp300, repeat_cases, config["evaluation"]["width"])
        deterministic = all(
            first[key] == second[key]
            for first, second in ((control_repeat_a, control_repeat_b), (challenger_repeat_a, challenger_repeat_b))
            for key in ("predictions", "minimum_margin")
        )
        raw_path = output / "paired-evidence.json"
        raw_path.write_text(json.dumps({
            "cases_sha256": cases_hash,
            "expected": control_eval["expected"],
            "control": {key: control_eval[key] for key in ("predictions", "correctness", "minimum_margin")},
            "challenger": {key: challenger_eval[key] for key in ("predictions", "correctness", "minimum_margin")},
        }, indent=2) + "\n")
        gates = config["gates"]
        best = max(control_eval["exact_accuracy"], challenger_eval["exact_accuracy"])
        paired_wall = control_eval["wall_seconds"] + challenger_eval["wall_seconds"]
        gate_results = {
            "challenger_exact": challenger_eval["exact_accuracy"] >= gates["minimum_challenger_exact_accuracy"],
            "oracle_absolute": comparison["oracle_union_exact_accuracy"] >= gates["minimum_oracle_union_exact_accuracy"],
            "oracle_gain": comparison["oracle_union_gain_over_best_arm"] >= gates["minimum_oracle_union_gain_over_best_arm"],
            "routed_gain": comparison["routed_gain_over_best_arm"] >= gates["minimum_routed_gain_over_best_arm"],
            "trained_dependence": challenger_eval["exact_accuracy"] - randomized_eval["exact_accuracy"] >= gates["minimum_trained_gain_over_randomized"],
            "validity": max(control_eval["invalid_rate"], challenger_eval["invalid_rate"]) <= gates["maximum_invalid_rate"],
            "runtime": paired_wall <= gates["maximum_paired_width16_wall_seconds"],
            "determinism": deterministic if gates["require_exact_determinism"] else True,
        }
        passed = all(gate_results.values())
        decision = "advance-independent-cellular-lineage-to-tier6" if passed else "reject-independent-cellular-diversity-feasibility"
        summary = {
            "run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"],
            "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id},
            "source": {**source_hashes, "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60", "cases_sha256": cases_hash},
            "data": {"training_prime_count_width16": len(data["training_splits"][16]["train"]), "fresh_evaluation_prime_count_width16": len(data["fresh_evaluation_primes"]), "fresh_disjoint_from_training": True, "fresh_disjoint_from_exp633_evaluation": True},
            "training": training,
            "arms": {"control": clean_evaluation(control_eval), "challenger": clean_evaluation(challenger_eval), "randomized": clean_evaluation(randomized_eval)},
            "comparison": {**comparison, "best_arm_exact_accuracy": best},
            "determinism": {"cases_per_arm": config["evaluation"]["determinism_cases"], "exact_predictions_and_margins": deterministic},
            "gates": {"thresholds": gates, "results": gate_results, "passed": passed},
            "decision": decision,
            "artifacts": {
                "control_checkpoint": {"path": str(control_path.relative_to(ROOT)), "sha256": sha256(control_path)},
                "challenger_checkpoint": {"path": str(challenger_path.relative_to(ROOT)), "sha256": sha256(challenger_path), "bytes": challenger_path.stat().st_size},
                "paired_evidence": {"path": str(raw_path.relative_to(ROOT)), "sha256": sha256(raw_path)},
            },
            "resource_budget": budget,
            "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "paired_evaluation_wall_seconds": paired_wall, "torch_threads": config["training"]["torch_threads"]},
            "limitations": [
                "Fresh generated feasibility evidence only; checked, independent full-seed, official, and private answers were not read.",
                "Oracle union is diagnostic and not an inference result.",
                "Small-width diversity may not transfer to width 128.",
                "Raw minimum margins are target-free but are not calibrated across independent models.",
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({
            "control_exact": control_eval["exact_accuracy"], "challenger_exact": challenger_eval["exact_accuracy"],
            "oracle_union": comparison["oracle_union_exact_accuracy"], "routed_exact": comparison["routed_exact_accuracy"],
            "randomized_exact": randomized_eval["exact_accuracy"], "wall_seconds": summary["runtime"]["wall_seconds"],
        })
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({
        "decision": decision, "arms": summary["arms"], "comparison": comparison,
        "gates": summary["gates"], "runtime": summary["runtime"], "mlflow": summary["mlflow"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
