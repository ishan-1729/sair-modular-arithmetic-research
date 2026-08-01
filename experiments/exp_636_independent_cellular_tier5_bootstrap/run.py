"""Progress an independent cellular lineage from width 16 through width 64."""

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
from torch import nn

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


def validate_stages(config: dict[str, Any]) -> None:
    stages = config["training"]["stages"]
    if [stage["name"] for stage in stages] != ["bridge", "wide", "tier5", "recovery"]:
        raise ValueError("EXP-636 requires the frozen four-stage order")
    if stages[-1]["name"] != "recovery" or stages[-1]["widths"] != [56, 64]:
        raise ValueError("the final stage must be width-56/64 recovery")
    if sum(int(stage["steps"]) for stage in stages) != 3200:
        raise ValueError("the preregistered curriculum contains exactly 3,200 updates")


def make_splits(config: dict[str, Any], exp300) -> dict[int, dict[str, list[int]]]:
    widths = sorted({width for stage in config["training"]["stages"] for width in stage["widths"]})
    splits = {}
    for width in widths:
        local = {
            "data": {
                "prime_bits_min": max(3, width - config["data"]["prime_bit_span"] + 1),
                "prime_bits_max": width,
                "train_prime_identities": config["data"]["train_prime_identities"],
                "test_prime_identities": config["data"]["test_prime_identities"],
            }
        }
        splits[width] = exp300.bigint_prime_split(local, int(config["seed"]) + 63600 + width)
        if set(splits[width]["train"]) & set(splits[width]["test"]):
            raise RuntimeError(f"width-{width} train/test identity overlap")
    return splits


def initialize_model(config, cellular, checkpoint: Path):
    model = cellular.SharedDilatedConvGRU(width=64, **config["model"])
    model.load_state_dict(load_file(checkpoint))
    return model


def optimizer_for(model, config, stage):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=stage["steps"],
        eta_min=stage["minimum_learning_rate"],
    )
    return optimizer, scheduler


def save_partial(path, step, model, optimizer, scheduler, history, samplers, selector, recovery_rng=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "sampler_rng": {width: sampler.rng.getstate() for width, sampler in samplers.items()},
            "selector_rng": selector.get_state(),
            "recovery_rng": recovery_rng.getstate() if recovery_rng else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        path,
    )


def restore_partial(path, model, optimizer, scheduler, samplers, selector, recovery_rng=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    for width, value in state["sampler_rng"].items():
        key = int(width) if str(width).isdigit() else width
        samplers[key].rng.setstate(value)
    selector.set_state(state["selector_rng"])
    if recovery_rng is not None and state["recovery_rng"] is not None:
        recovery_rng.setstate(state["recovery_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    return int(state["step"]) + 1, state["history"]


def transition_update(model, width, sampler, selector, config, stage, exp300, exp320):
    pool = config["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"]
    values = exp320.mine(
        sampler.transitions(pool), config["training"]["batch_size"], config, selector
    )
    state, right, modulus, input_bit, target, quotient = values
    model.width = width
    logits, quotient_logits = model(exp300.features(state, right, modulus, input_bit, width))
    return exp300.transition_loss(
        logits,
        quotient_logits,
        target,
        quotient,
        width,
        config["training"]["worst_bit_weight"],
        config["training"]["quotient_weight"],
    )


def recovery_update(model, sampler, rng, step, config, stage, exp300, exp329):
    width = 64
    batch = config["training"]["recovery"]["batch_size"]
    chunk = config["training"]["recovery"]["chunk_steps"]
    left, right, modulus = sampler.pairs(batch)
    starts = [rng.randrange(0, width - chunk + 1) for _ in range(batch)]
    teacher = exp329.teacher_states_before(left, right, modulus, starts, width)
    exposed = list(teacher)
    losses = []
    exact = []
    recovery = config["training"]["recovery"]
    probability = recovery["scheduled_prediction_probability_start"] + (
        recovery["scheduled_prediction_probability_end"]
        - recovery["scheduled_prediction_probability_start"]
    ) * (step - 1) / max(1, stage["steps"] - 1)
    model.width = width
    for offset in range(chunk):
        positions = [start + offset for start in starts]
        input_bit = [
            (value >> (width - 1 - position)) & 1 for value, position in zip(left, positions)
        ]
        raw = [2 * value + bit * other for value, bit, other in zip(teacher, input_bit, right)]
        target = [value % prime for value, prime in zip(raw, modulus)]
        quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long)
        logits, quotient_logits = model(exp300.features(exposed, right, modulus, input_bit, width))
        part, stats = exp300.transition_loss(
            logits,
            quotient_logits,
            target,
            quotient,
            width,
            config["training"]["worst_bit_weight"],
            0.0,
        )
        losses.append(part)
        exact.append(stats["batch_exact"])
        predicted = exp329.decode_logits(logits.detach(), width)
        exposed = [guess if rng.random() < probability else truth for guess, truth in zip(predicted, target)]
        teacher = target
    loss = torch.stack(losses).mean()
    return loss, {
        "loss": float(loss.detach()),
        "batch_exact": sum(exact) / len(exact),
        "mean_bce": None,
        "worst_bit_bce": None,
        "quotient_ce": 0.0,
    }


def train_stage(
    config,
    stage,
    stage_index,
    source_checkpoint,
    output,
    splits,
    exp300,
    exp320,
    exp329,
    cellular,
    source_hashes,
    global_started,
    steps_override=None,
    resumable=True,
):
    stage_steps = int(stage["steps"] if steps_override is None else steps_override)
    random.seed(int(config["seed"]) + stage_index)
    np.random.seed(int(config["seed"]) + stage_index)
    torch.manual_seed(int(config["seed"]) + stage_index)
    model = initialize_model(config, cellular, source_checkpoint)
    optimizer, scheduler = optimizer_for(model, config, stage)
    samplers = {
        width: exp300.BigIntSampler(
            splits[width]["train"], int(config["seed"]) + 64000 + stage_index * 1000 + width
        )
        for width in stage["widths"]
    }
    selector = torch.Generator().manual_seed(int(config["seed"]) + 65000 + stage_index)
    recovery_rng = random.Random(int(config["seed"]) + 66000 + stage_index)
    if stage["name"] == "recovery":
        samplers["recovery"] = exp300.BigIntSampler(
            splits[64]["train"], int(config["seed"]) + 66664
        )
    stage_dir = output / f"stage-{stage_index + 1}-{stage['name']}"
    key = sha256_text(json.dumps({**source_hashes, "stage": stage, "source": sha256(source_checkpoint)}, sort_keys=True))
    result_path = stage_dir / f"result-{key}.json"
    checkpoint = stage_dir / "weights.safetensors"
    if resumable and result_path.exists() and checkpoint.exists():
        payload = json.loads(result_path.read_text())
        if payload.get("cache_key") == key and sha256(checkpoint) == payload["checkpoint_sha256"]:
            return checkpoint, {**payload["training"], "cached": True}, model
    partial = stage_dir / f"training-{key}.partial.pt"
    start_step = 1
    history = []
    if resumable and partial.exists():
        start_step, history = restore_partial(
            partial, model, optimizer, scheduler, samplers, selector, recovery_rng
        )
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    peak = psutil.Process().memory_info().rss
    model.train()
    for step in range(start_step, stage_steps + 1):
        use_recovery = stage["name"] == "recovery" and step % config["training"]["recovery"]["update_period"] == 0
        if use_recovery:
            width = 64
            loss, stats = recovery_update(
                model, samplers["recovery"], recovery_rng, step, config, stage, exp300, exp329
            )
        else:
            width = stage["widths"][(step - 1) % len(stage["widths"])]
            loss, stats = transition_update(
                model, width, samplers[width], selector, config, stage, exp300, exp320
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        peak = max(peak, psutil.Process().memory_info().rss)
        if step == 1 or step % 100 == 0 or step == stage_steps:
            item = {
                "step": step,
                "stage": stage["name"],
                "width": width,
                "mode": "recovery" if use_recovery else "transition",
                "elapsed_wall_seconds": time.perf_counter() - wall_started,
                "gradient_norm": gradient,
                **stats,
            }
            history.append(item)
            if mlflow.active_run() is not None:
                mlflow.log_metrics(
                    {f"{stage['name']}_loss": stats["loss"], f"{stage['name']}_batch_exact": stats["batch_exact"]},
                    step=step,
                )
            print(json.dumps(item), flush=True)
        cadence = config["training"]["checkpoint_every_steps"]
        if resumable and (step % cadence == 0 or step == stage_steps):
            save_partial(
                partial, step, model, optimizer, scheduler, history, samplers, selector, recovery_rng
            )
            if time.perf_counter() > global_started + config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]:
                raise RuntimeError(f"hard wall ceiling reached after preserving {stage['name']} step {step}")
    model.eval()
    training = {
        "stage": stage["name"],
        "steps": stage_steps,
        "history": history,
        "wall_seconds_this_invocation": time.perf_counter() - wall_started,
        "aggregate_cpu_seconds_this_invocation": time.process_time() - cpu_started,
        "peak_rss_bytes": peak,
        "cached": False,
    }
    if resumable:
        stage_dir.mkdir(parents=True, exist_ok=True)
        save_file({key_: value.detach().contiguous() for key_, value in model.state_dict().items()}, checkpoint)
        result_path.write_text(
            json.dumps({"cache_key": key, "checkpoint_sha256": sha256(checkpoint), "training": training}, indent=2) + "\n"
        )
        return checkpoint, training, model
    return source_checkpoint, training, model


def generated_cases(exp300, primes, seed, count):
    left, right, modulus = exp300.BigIntSampler(primes, seed).pairs(count)
    return {"left": left, "right": right, "modulus": modulus}


def calibrate(config, output, splits, modules, source):
    exp300, exp320, exp329, exp635, cellular = modules
    output.mkdir(parents=True, exist_ok=True)
    calibration_steps = int(config["resource_budget"]["calibration_updates_per_stage"])
    source_hashes = {"calibration": True}
    stage_results = []
    current = source
    models = []
    calibration_started = time.perf_counter()
    for index, stage in enumerate(config["training"]["stages"]):
        current, result, model = train_stage(
            config, stage, index, current, output, splits, exp300, exp320, exp329, cellular,
            source_hashes, calibration_started, steps_override=calibration_steps, resumable=False,
        )
        stage_results.append(result)
        models.append(model)
        temporary = output / "calibration-current.safetensors"
        save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, temporary)
        current = temporary
    incumbent = initialize_model(config, cellular, (ROOT / config["source"]["incumbent_checkpoint"]).resolve())
    source_model = initialize_model(config, cellular, source)
    independent = models[-1]
    cases_n = int(config["resource_budget"]["calibration_inference_cases"])
    cases = generated_cases(exp300, splits[64]["train"], config["evaluation"]["case_seed"] + 1, cases_n)
    inference = {
        "source": exp635.rollout_details(source_model, exp300, cases, 64),
        "incumbent": exp635.rollout_details(incumbent, exp300, cases, 64),
        "independent": exp635.rollout_details(independent, exp300, cases, 64),
    }
    transition_started = time.perf_counter()
    transition_cpu = time.process_time()
    transition = exp300.evaluate_transitions(
        independent,
        exp300.BigIntSampler(splits[64]["train"], config["evaluation"]["case_seed"] + 2),
        100,
    )
    transition_wall = time.perf_counter() - transition_started
    transition_cpu_seconds = time.process_time() - transition_cpu
    projected_training_wall = sum(
        result["wall_seconds_this_invocation"] / calibration_steps * stage["steps"]
        for result, stage in zip(stage_results, config["training"]["stages"])
    )
    projected_training_cpu = sum(
        result["aggregate_cpu_seconds_this_invocation"] / calibration_steps * stage["steps"]
        for result, stage in zip(stage_results, config["training"]["stages"])
    )
    per_case_wall = {name: result["wall_seconds"] / cases_n for name, result in inference.items()}
    per_case_cpu = {name: result["aggregate_cpu_seconds"] / cases_n for name, result in inference.items()}
    evaluation = config["evaluation"]
    projected_evaluation_wall = (
        evaluation["width64_rollout_cases"] * (per_case_wall["incumbent"] + per_case_wall["independent"])
        + evaluation["source_dependence_width64_cases"] * per_case_wall["source"]
        + evaluation["width32_rollout_cases"] * per_case_wall["independent"] / 2
        + 2 * evaluation["determinism_cases"] * (per_case_wall["incumbent"] + per_case_wall["independent"])
        + evaluation["width64_transition_cases"] * transition_wall / 100
    )
    projected_evaluation_cpu = (
        evaluation["width64_rollout_cases"] * (per_case_cpu["incumbent"] + per_case_cpu["independent"])
        + evaluation["source_dependence_width64_cases"] * per_case_cpu["source"]
        + evaluation["width32_rollout_cases"] * per_case_cpu["independent"] / 2
        + 2 * evaluation["determinism_cases"] * (per_case_cpu["incumbent"] + per_case_cpu["independent"])
        + evaluation["width64_transition_cases"] * transition_cpu_seconds / 100
    )
    setup = 180
    expected_wall = projected_training_wall + projected_evaluation_wall + setup
    expected_cpu = projected_training_cpu + projected_evaluation_cpu + setup * config["training"]["torch_threads"]
    soft = math.ceil(expected_wall * 1.35 / 60) * 60
    hard = math.ceil(expected_wall * 2.0 / 60) * 60
    payload = {
        "run_id": config["run_id"],
        "status": "calibration-only",
        "training_identities_only": True,
        "stages": stage_results,
        "inference": {name: exp635.clean_evaluation(value) for name, value in inference.items()},
        "transition_calibration": {**transition, "wall_seconds": transition_wall, "aggregate_cpu_seconds": transition_cpu_seconds},
        "projection": {
            "training_wall_seconds": projected_training_wall,
            "training_aggregate_cpu_seconds": projected_training_cpu,
            "evaluation_wall_seconds": projected_evaluation_wall,
            "evaluation_aggregate_cpu_seconds": projected_evaluation_cpu,
            "setup_serialization_tracking_allowance_wall_seconds": setup,
        },
        "recommendation": {
            "expected_total_wall_seconds": math.ceil(expected_wall),
            "expected_total_aggregate_cpu_minutes": math.ceil(expected_cpu / 60),
            "soft_budget_wall_seconds": soft,
            "hard_runaway_ceiling_wall_seconds": hard,
            "hard_ceiling_aggregate_cpu_minutes_upper_bound": hard * config["training"]["torch_threads"] / 60,
        },
        "limitations": [
            "Calibration uses training identities only and is not model-selection evidence.",
            "Each stage executes its real operations, optimizer, hard mining, and recovery mix.",
            "Wall and aggregate CPU projections are separate and include the full evaluation workload.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_636_independent_cellular_tier5_bootstrap.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-636-independent-cellular-tier5-bootstrap")
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    validate_stages(config)
    source = (ROOT / config["source"]["independent_checkpoint"]).resolve()
    incumbent_path = (ROOT / config["source"]["incumbent_checkpoint"]).resolve()
    if sha256(source) != config["source"]["independent_checkpoint_sha256"]:
        raise RuntimeError("independent source checkpoint hash mismatch")
    if sha256(incumbent_path) != config["source"]["incumbent_checkpoint_sha256"]:
        raise RuntimeError("incumbent checkpoint hash mismatch")
    exp300 = load_module("exp300_for_exp636", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    exp320 = load_module("exp320_for_exp636", ROOT / "experiments/exp_320_cellular_width64_curriculum/run.py")
    exp329 = load_module("exp329_for_exp636", ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py")
    exp635 = load_module("exp635_for_exp636", ROOT / "experiments/exp_635_independent_cellular_diversity_feasibility/run.py")
    cellular = load_module("cellular_for_exp636", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    modules = (exp300, exp320, exp329, exp635, cellular)
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    splits = make_splits(config, exp300)
    output = args.output.resolve()
    if args.calibrate_only:
        calibrate(config, output, splits, modules, source)
        return 0
    budget = config["resource_budget"]
    if budget.get("status") != "calibrated" or not budget.get("hard_runaway_ceiling_wall_seconds"):
        raise RuntimeError("formal run requires a documented measured-throughput budget")
    if git_output("status", "--porcelain"):
        raise RuntimeError("formal run requires a clean committed worktree")
    commit = git_output("rev-parse", "HEAD")
    source_hashes = {
        "repository_commit": commit,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "dataset_sha256": sha256_text(json.dumps(splits, sort_keys=True)),
        "source_checkpoint_sha256": sha256(source),
        "incumbent_checkpoint_sha256": sha256(incumbent_path),
    }
    started = time.perf_counter()
    cpu_started = time.process_time()
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({
            "canonical_run_id": config["run_id"], "status": "running",
            "role": "independent-cellular-tier5-bootstrap", "repository_commit": commit,
            "repository_dirty": "false",
        })
        mlflow.log_params({
            "seed": config["seed"], "total_steps": sum(stage["steps"] for stage in config["training"]["stages"]),
            "config_sha256": source_hashes["config_sha256"], "runner_sha256": source_hashes["runner_sha256"],
            "dataset_sha256": source_hashes["dataset_sha256"],
            "hard_runaway_ceiling_wall_seconds": budget["hard_runaway_ceiling_wall_seconds"],
        })
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        current = source
        stage_records = []
        independent = None
        for index, stage in enumerate(config["training"]["stages"]):
            current, record, independent = train_stage(
                config, stage, index, current, output, splits, exp300, exp320, exp329,
                cellular, source_hashes, started,
            )
            stage_records.append(record)
            if time.perf_counter() > started + budget["hard_runaway_ceiling_wall_seconds"]:
                raise RuntimeError(f"hard wall ceiling reached after preserving stage {stage['name']}")
        assert independent is not None
        independent = initialize_model(config, cellular, current)
        independent.eval()
        incumbent = initialize_model(config, cellular, incumbent_path)
        incumbent.eval()
        source_model = initialize_model(config, cellular, source)
        source_model.eval()
        width32_cases = generated_cases(
            exp300, splits[32]["test"], config["evaluation"]["case_seed"] + 32,
            config["evaluation"]["width32_rollout_cases"],
        )
        width32_eval = exp635.rollout_details(independent, exp300, width32_cases, 32)
        paired_cases = generated_cases(
            exp300, splits[64]["test"], config["evaluation"]["case_seed"] + 64,
            config["evaluation"]["width64_rollout_cases"],
        )
        incumbent_eval = exp635.rollout_details(incumbent, exp300, paired_cases, 64)
        independent_eval = exp635.rollout_details(independent, exp300, paired_cases, 64)
        comparison = exp635.paired_comparison(incumbent_eval, independent_eval)
        source_cases = {key: value[: config["evaluation"]["source_dependence_width64_cases"]] for key, value in paired_cases.items()}
        source_eval = exp635.rollout_details(source_model, exp300, source_cases, 64)
        transition = exp300.evaluate_transitions(
            independent,
            exp300.BigIntSampler(splits[64]["test"], config["evaluation"]["case_seed"] + 65),
            config["evaluation"]["width64_transition_cases"],
        )
        repeat_cases = {key: value[: config["evaluation"]["determinism_cases"]] for key, value in paired_cases.items()}
        repeats = []
        for model in (incumbent, independent):
            first = exp635.rollout_details(model, exp300, repeat_cases, 64)
            second = exp635.rollout_details(model, exp300, repeat_cases, 64)
            repeats.append(first["predictions"] == second["predictions"] and first["minimum_margin"] == second["minimum_margin"])
        deterministic = all(repeats)
        evidence_path = output / "paired-evidence.json"
        evidence_path.write_text(json.dumps({
            "cases_sha256": sha256_text(json.dumps(paired_cases, sort_keys=True)),
            "expected": incumbent_eval["expected"],
            "incumbent": {key: incumbent_eval[key] for key in ("predictions", "correctness", "minimum_margin")},
            "independent": {key: independent_eval[key] for key in ("predictions", "correctness", "minimum_margin")},
        }, indent=2) + "\n")
        gates = config["gates"]
        paired_wall = incumbent_eval["wall_seconds"] + independent_eval["wall_seconds"]
        gate_results = {
            "width32_retention": width32_eval["exact_accuracy"] >= gates["minimum_independent_width32_rollout"],
            "width64_transition": transition["exact_accuracy"] >= gates["minimum_independent_width64_transition"],
            "width64_rollout": independent_eval["exact_accuracy"] >= gates["minimum_independent_width64_rollout"],
            "trained_dependence": independent_eval["exact_accuracy"] - source_eval["exact_accuracy"] >= gates["minimum_trained_gain_over_width16_source_at_width64"],
            "oracle_absolute": comparison["oracle_union_exact_accuracy"] >= gates["minimum_oracle_union_width64"],
            "oracle_gain": comparison["oracle_union_gain_over_best_arm"] >= gates["minimum_oracle_union_gain_over_best_arm"],
            "routed_gain": comparison["routed_gain_over_best_arm"] >= gates["minimum_routed_gain_over_best_arm"],
            "validity": independent_eval["invalid_rate"] <= gates["maximum_independent_invalid_rate"],
            "runtime": paired_wall <= gates["maximum_paired_width64_wall_seconds"],
            "determinism": deterministic if gates["require_exact_determinism"] else True,
        }
        passed = all(gate_results.values())
        decision = "advance-independent-cellular-lineage-to-tier6" if passed else "reject-independent-cellular-tier5-bootstrap"
        summary = {
            "run_id": config["run_id"], "status": "completed", "hypothesis": config["hypothesis"],
            "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": config["tracking"]["experiment_name"], "experiment_id": experiment_id, "run_id": run.info.run_id},
            "source": {**source_hashes, "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"},
            "training": {"stages": stage_records, "total_steps": 3200},
            "evaluation": {
                "independent_width32": exp635.clean_evaluation(width32_eval),
                "incumbent_width64": exp635.clean_evaluation(incumbent_eval),
                "independent_width64": exp635.clean_evaluation(independent_eval),
                "source_width64": exp635.clean_evaluation(source_eval),
                "independent_width64_transition": transition,
            },
            "comparison": comparison,
            "determinism": {"cases_per_arm": config["evaluation"]["determinism_cases"], "passed": deterministic},
            "gates": {"thresholds": gates, "results": gate_results, "passed": passed},
            "decision": decision,
            "artifacts": {
                "checkpoint": {"path": str(current.relative_to(ROOT)), "sha256": sha256(current), "bytes": current.stat().st_size},
                "paired_evidence": {"path": str(evidence_path.relative_to(ROOT)), "sha256": sha256(evidence_path)},
            },
            "resource_budget": budget,
            "runtime": {"wall_seconds": time.perf_counter() - started, "aggregate_cpu_seconds": time.process_time() - cpu_started, "paired_evaluation_wall_seconds": paired_wall, "torch_threads": config["training"]["torch_threads"]},
            "limitations": [
                "Fresh generated selection evidence only; checked, full independent, official, and private answers were not read.",
                "Passing authorizes width-128 training but is not candidate promotion.",
                "Oracle union is diagnostic and not an inference result.",
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({
            "independent_width32": width32_eval["exact_accuracy"],
            "incumbent_width64": incumbent_eval["exact_accuracy"],
            "independent_width64": independent_eval["exact_accuracy"],
            "routed_width64": comparison["routed_exact_accuracy"],
            "oracle_width64": comparison["oracle_union_exact_accuracy"],
            "wall_seconds": summary["runtime"]["wall_seconds"],
        })
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(json.dumps({
        "decision": decision, "evaluation": summary["evaluation"], "comparison": comparison,
        "gates": summary["gates"], "runtime": summary["runtime"], "mlflow": summary["mlflow"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
