"""Preregister, calibrate, and execute EXP-661 LoopFormer roots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_661_loopformer_width_extrapolation.model import (  # noqa: E402
    DecodedCellularControl,
    LoopFormer,
    PlainLoopedTransformer,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

ARM_NAMES = (
    "loopformer",
    "plain_looped_transformer",
    "decoded_cellular_control",
    "no_time_step",
    "no_shortcut",
    "no_short_loss",
    "no_consistency",
)
SOURCE_PATHS = (
    "configs/exp_661_loopformer_width_extrapolation.yaml",
    "experiments/exp_661_loopformer_width_extrapolation/model.py",
    "experiments/exp_661_loopformer_width_extrapolation/run.py",
    "tests/test_exp_661_loopformer_width_extrapolation.py",
    "knowledge/autoresearch/research/2026-07-27-exp661-loopformer-mechanism.md",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def process_cpu_seconds() -> float:
    values = os.times()
    return values.user + values.system + values.children_user + values.children_system


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.exists():
        return None
    values = dict(line.split() for line in path.read_text().splitlines())
    return float(values["usage_usec"]) / 1_000_000.0


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def bits(values: list[int], width: int) -> Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def feature_tensor(cases: list[dict[str, int]], width: int) -> Tensor:
    left = bits([case["left"] for case in cases], width)
    right = bits([case["right"] for case in cases], width)
    modulus = bits([case["modulus"] for case in cases], width)
    lsb, msb = torch.zeros_like(left), torch.zeros_like(left)
    lsb[:, 0], msb[:, -1] = 1.0, 1.0
    return torch.stack((left, right, modulus, lsb, msb), dim=1).mul(2).sub(1)


def endpoint_targets(cases: list[dict[str, int]], width: int) -> Tensor:
    values = [(case["left"] * case["right"]) % case["modulus"] for case in cases]
    return bits(values, width)


def decode_rows(rows: Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in rows.tolist()
    ]


def make_splits(config: dict[str, Any], exp300: Any, root: int) -> dict[int, Any]:
    widths = sorted(set(config["data"]["training_widths"] + config["data"]["evaluation_widths"]))
    result = {}
    for width in widths:
        local = {
            "data": {
                "prime_bits_min": max(3, int(width) - int(config["data"]["prime_bit_span"]) + 1),
                "prime_bits_max": int(width),
                "train_prime_identities": int(config["data"]["train_prime_identities"]),
                "test_prime_identities": int(config["data"]["test_prime_identities"]),
            }
        }
        result[int(width)] = exp300.bigint_prime_split(
            local, root + int(config["data"]["split_seed_offset"]) + int(width)
        )
    return result


def make_cases(exp300: Any, identities: list[int], seed: int, count: int) -> list[dict[str, int]]:
    sampler = exp300.BigIntSampler(identities, seed)
    left, right, modulus = sampler.pairs(count)
    return [
        {"left": int(a), "right": int(b), "modulus": int(p)}
        for a, b, p in zip(left, right, modulus)
    ]


def short_schedule(step_index: int, root: int, maximum: int) -> list[float]:
    """Draw a deterministic discrete shortcut trajectory on the maximum grid."""
    budget = 1 + (step_index % (maximum - 1))
    rng = random.Random(root * 10_000 + step_index)
    cuts = sorted(rng.sample(range(1, maximum), budget - 1))
    points = [0, *cuts, maximum]
    return [(right - left) / maximum for left, right in zip(points, points[1:])]


def build_model(name: str, config: dict[str, Any]) -> LoopFormer:
    kwargs = {
        "hidden": int(config["model"]["hidden"]),
        "heads": int(config["model"]["heads"]),
        "ff_hidden": int(config["model"]["ff_hidden"]),
        "condition": int(config["model"]["condition"]),
        "max_width": int(config["model"]["max_width"]),
        "default_budget": int(config["model"]["trained_budget"]),
    }
    if name == "plain_looped_transformer":
        return PlainLoopedTransformer(**kwargs).cpu()
    if name == "decoded_cellular_control":
        return DecodedCellularControl(**kwargs).cpu()
    if name == "no_time_step":
        return LoopFormer(**kwargs, use_time_step=False).cpu()
    if name == "no_shortcut":
        return LoopFormer(**kwargs, use_shortcut=False).cpu()
    if name in {"loopformer", "no_short_loss", "no_consistency"}:
        return LoopFormer(**kwargs).cpu()
    raise KeyError(name)


def loss_weights(name: str, config: dict[str, Any]) -> tuple[float, float]:
    short = float(config["training"]["short_loss_weight"])
    consistency = float(config["training"]["consistency_loss_weight"])
    if name in {"plain_looped_transformer", "decoded_cellular_control"}:
        return 0.0, 0.0
    if name == "no_short_loss":
        short = 0.0
    if name == "no_consistency":
        consistency = 0.0
    return short, consistency


def curve_statistics(output: Any) -> dict[str, Any]:
    hidden_delta, hidden_cosine, logit_delta = [], [], []
    for prior, current in zip(output.hidden_curve, output.hidden_curve[1:]):
        hidden_delta.append(float((current - prior).square().mean().sqrt()))
        hidden_cosine.append(float(F.cosine_similarity(prior.flatten(1), current.flatten(1)).mean()))
    for prior, current in zip(output.logit_curve, output.logit_curve[1:]):
        logit_delta.append(float((current - prior).square().mean().sqrt()))
    return {
        "hidden_rms_delta": hidden_delta,
        "hidden_cosine": hidden_cosine,
        "logit_rms_delta": logit_delta,
    }


@torch.inference_mode()
def evaluate_model(
    model: LoopFormer,
    cases: list[dict[str, int]],
    width: int,
    config: dict[str, Any],
    *,
    budget: int,
    intervention: str = "none",
) -> dict[str, Any]:
    model.eval()
    total = correct = bit_correct = invalid = 0
    first_errors: list[int] = []
    curve = None
    batch_size = int(config["evaluation"]["batch_size"])
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        features = feature_tensor(batch, width)
        targets = endpoint_targets(batch, width).bool()
        output = model.loop(
            features,
            schedule=model.uniform_schedule(int(budget)),
            use_time_step=False if intervention == "no_time_step" else None,
            use_shortcut=False if intervention == "no_shortcut" else None,
        )
        if curve is None:
            curve = curve_statistics(output)
        predicted = output.logits.ge(0)
        exact = predicted.eq(targets).all(dim=-1)
        values = decode_rows(predicted)
        correct += int(exact.sum())
        bit_correct += int(predicted.eq(targets).sum())
        invalid += sum(value >= case["modulus"] for value, case in zip(values, batch))
        for row, truth in zip(predicted, targets):
            mismatch = row.ne(truth).nonzero()
            if len(mismatch):
                first_errors.append(int(mismatch[0, 0]))
        total += len(batch)
    return {
        "cases": total,
        "correct": correct,
        "exact_accuracy": correct / total,
        "bit_accuracy": bit_correct / (total * width),
        "invalid_rate": invalid / total,
        "first_error": {
            "error_cases": len(first_errors),
            "mean_lsb_position": statistics.mean(first_errors) if first_errors else None,
        },
        "curves": curve,
    }


def save_checkpoint(path: Path, model: LoopFormer) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def exact_reload(name: str, config: dict[str, Any], path: Path, reference: LoopFormer, cases: list[dict[str, int]]) -> bool:
    loaded = build_model(name, config)
    loaded.load_state_dict(load_file(path), strict=True)
    features = feature_tensor(cases[:16], int(config["data"]["train_width"]))
    reference.eval(), loaded.eval()
    with torch.inference_mode():
        expected = reference(features, budget=int(config["model"]["trained_budget"]))
        observed = loaded(features, budget=int(config["model"]["trained_budget"]))
    return torch.equal(expected, observed)


def run_arm(
    name: str,
    config: dict[str, Any],
    exp300: Any,
    splits: dict[int, Any],
    evaluation_cases: dict[int, list[dict[str, int]]],
    root: int,
    steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    random.seed(root + 661)
    np.random.seed(root + 661)
    torch.manual_seed(root + 661)
    model = build_model(name, config)
    initial_identity = sha256_json(
        {
            key: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
            for key, value in model.state_dict().items()
        }
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, steps),
        eta_min=float(config["training"]["minimum_learning_rate"]),
    )
    widths = list(map(int, config["data"]["training_widths"]))
    samplers = {
        width: exp300.BigIntSampler(
            splits[width]["train"], root + int(config["data"]["train_pair_seed_offset"]) + width
        )
        for width in widths
    }
    short_weight, consistency_weight = loss_weights(name, config)
    update_wall, update_cpu, update_cgroup = [], [], []
    history: list[dict[str, Any]] = []
    examples = calls = 0
    process = psutil.Process()
    training_wall_started = time.perf_counter()
    training_cpu_started = process_cpu_seconds()
    training_cgroup_started = cgroup_cpu_seconds()
    for step_index in range(steps):
        width = widths[step_index % len(widths)]
        sampler = samplers[width]
        left, right, modulus = sampler.pairs(int(config["training"]["batch_size"]))
        cases = [
            {"left": int(a), "right": int(b), "modulus": int(p)}
            for a, b, p in zip(left, right, modulus)
        ]
        features, targets = feature_tensor(cases, width), endpoint_targets(cases, width)
        full_schedule = model.uniform_schedule(int(config["model"]["trained_budget"]))
        shortcut = short_schedule(step_index, root, int(config["model"]["trained_budget"]))
        wall_started, cpu_started, cg_started = time.perf_counter(), process_cpu_seconds(), cgroup_cpu_seconds()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        full = model.loop(features, schedule=full_schedule)
        short = model.loop(features, schedule=shortcut)
        full_loss = F.binary_cross_entropy_with_logits(full.logits, targets)
        short_loss = F.binary_cross_entropy_with_logits(short.logits, targets)
        consistency = F.mse_loss(short.hidden, full.hidden.detach())
        loss = full_loss + short_weight * short_loss + consistency_weight * consistency
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        cg_finished = cgroup_cpu_seconds()
        update_wall.append(time.perf_counter() - wall_started)
        update_cpu.append(process_cpu_seconds() - cpu_started)
        update_cgroup.append(0.0 if cg_started is None or cg_finished is None else cg_finished - cg_started)
        examples += len(cases)
        calls += full.call_count + short.call_count
        if step_index == 0 or (step_index + 1) % int(config["training"]["log_every_steps"]) == 0 or step_index + 1 == steps:
            history.append(
                {
                    "step": step_index + 1,
                    "width": width,
                    "short_budget": len(shortcut),
                    "loss": float(loss.detach()),
                    "full_loss": float(full_loss.detach()),
                    "short_loss": float(short_loss.detach()),
                    "consistency_loss": float(consistency.detach()),
                }
            )
    training_cgroup_finished = cgroup_cpu_seconds()
    checkpoint_path = output_dir / name / "checkpoint.safetensors"
    checkpoint = save_checkpoint(checkpoint_path, model)
    evaluations = {
        str(width): {
            str(budget): evaluate_model(model, evaluation_cases[width], width, config, budget=int(budget))
            for budget in config["evaluation"]["fixed_budgets"]
        }
        for width in map(int, config["data"]["evaluation_widths"])
    }
    reload_exact = exact_reload(
        name,
        config,
        checkpoint_path,
        model,
        evaluation_cases[int(config["data"]["train_width"])],
    )
    return {
        "name": name,
        "parameters": model.parameter_count(),
        "initial_state_sha256": initial_identity,
        "checkpoint": checkpoint,
        "reload_exact": reload_exact,
        "training": {
            "optimizer_updates": steps,
            "trajectory_examples": examples,
            "recurrent_block_calls": calls,
            "full_short_forward_pairs": steps,
            "short_loss_weight": short_weight,
            "consistency_loss_weight": consistency_weight,
            "history": history,
        },
        "evaluation": evaluations,
        "resources": {
            "training_wall_seconds": time.perf_counter() - training_wall_started,
            "training_process_cpu_seconds": process_cpu_seconds() - training_cpu_started,
            "training_cgroup_cpu_seconds": None if training_cgroup_started is None or training_cgroup_finished is None else training_cgroup_finished - training_cgroup_started,
            "update_wall_seconds": update_wall,
            "update_process_cpu_seconds": update_cpu,
            "update_cgroup_cpu_seconds": update_cgroup,
            "peak_rss_bytes": max(peak_rss_bytes(), int(process.memory_info().rss)),
        },
        "_model": model,
    }


def primary_diagnostics(arms: dict[str, dict[str, Any]], config: dict[str, Any], cases: dict[int, list[dict[str, int]]], root: int) -> dict[str, Any]:
    primary = arms["loopformer"]["_model"]
    width = int(config["data"]["train_width"])
    budget = int(config["model"]["trained_budget"])
    interventions = {
        name: evaluate_model(primary, cases[width], width, config, budget=budget, intervention=name)
        for name in config["evaluation"]["inference_interventions"]
    }
    torch.manual_seed(root + 9_999)
    randomized = build_model("loopformer", config)
    return {
        "inference_interventions": interventions,
        "randomized_weights": evaluate_model(randomized, cases[width], width, config, budget=budget),
        "fixed_user_budget": True,
        "learned_exit": False,
    }


def strip_models(arm: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arm.items() if key != "_model"}


def scientific_gate(arms: dict[str, dict[str, Any]], diagnostics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    trained = str(config["model"]["trained_budget"])
    deeper = str(max(config["evaluation"]["fixed_budgets"]))
    primary = arms["loopformer"]
    exact = {width: primary["evaluation"][width][trained]["exact_accuracy"] for width in ("16", "24", "32")}
    controls = ("plain_looped_transformer", "decoded_cellular_control")
    checks: dict[str, bool] = {
        "w16_exact": exact["16"] >= float(gates["minimum_primary_w16_exact_accuracy"]),
        "w24_exact": exact["24"] >= float(gates["minimum_primary_w24_exact_accuracy"]),
        "w32_exact": exact["32"] >= float(gates["minimum_primary_w32_exact_accuracy"]),
        "deeper_budget_retention": exact["16"] - primary["evaluation"]["16"][deeper]["exact_accuracy"] <= float(gates["maximum_deeper_budget_degradation"]),
        "zero_invalid": all(primary["evaluation"][width][budget]["invalid_rate"] == 0.0 for width in ("16", "24", "32") for budget in primary["evaluation"][width]),
        "reload_exact": all(arm["reload_exact"] for arm in arms.values()),
        "parameter_match": max(arm["parameters"] for arm in arms.values()) / min(arm["parameters"] for arm in arms.values()) <= float(gates["maximum_parameter_ratio"]),
        "example_update_call_match": len({(arm["training"]["optimizer_updates"], arm["training"]["trajectory_examples"], arm["training"]["recurrent_block_calls"]) for arm in arms.values()}) == 1,
        "trained_weight_dependence": exact["16"] - diagnostics["randomized_weights"]["exact_accuracy"] >= float(gates["minimum_trained_gain_over_randomized_w16"]),
    }
    transfer = float(gates["minimum_transfer_gain_over_each_control"])
    for control in controls:
        for width in ("24", "32"):
            checks[f"{width}_gain_over_{control}"] = exact[width] - arms[control]["evaluation"][width][trained]["exact_accuracy"] >= transfer
    component = float(gates["minimum_component_causal_drop_w16"])
    for name in ("no_time_step", "no_shortcut", "no_short_loss", "no_consistency"):
        checks[f"trained_{name}_effect"] = exact["16"] - arms[name]["evaluation"]["16"][trained]["exact_accuracy"] >= component
    for name in ("no_time_step", "no_shortcut"):
        checks[f"inference_{name}_effect"] = exact["16"] - diagnostics["inference_interventions"][name]["exact_accuracy"] >= component
    return {"passed": all(checks.values()), "checks": checks, "primary_exact_at_trained_budget": exact}


def calibration_admission(arms: dict[str, dict[str, Any]], route_wall: float, route_cpu: float, route_cgroup: float | None, config: dict[str, Any]) -> dict[str, Any]:
    expected_steps = int(config["training"]["steps_per_arm"])
    calibration_steps = int(config["calibration"]["representative_updates_per_route"])
    training_wall = sum(arm["resources"]["training_wall_seconds"] for arm in arms.values())
    training_cpu = sum(arm["resources"]["training_process_cpu_seconds"] for arm in arms.values())
    training_cgroup = sum((arm["resources"]["training_cgroup_cpu_seconds"] or 0.0) for arm in arms.values())
    fixed_wall, fixed_cpu = max(0.0, route_wall - training_wall), max(0.0, route_cpu - training_cpu)
    fixed_cgroup = None if route_cgroup is None else max(0.0, route_cgroup - training_cgroup)
    projected_wall = fixed_wall + sum(arm["resources"]["training_wall_seconds"] / calibration_steps * expected_steps for arm in arms.values())
    projected_cpu = fixed_cpu + sum(arm["resources"]["training_process_cpu_seconds"] / calibration_steps * expected_steps for arm in arms.values())
    projected_cgroup = None if route_cgroup is None else fixed_cgroup + sum((arm["resources"]["training_cgroup_cpu_seconds"] or 0.0) / calibration_steps * expected_steps for arm in arms.values())
    ratios = []
    for arm in arms.values():
        samples = arm["resources"]["update_wall_seconds"]
        ratios.append(max(samples) / max(statistics.mean(samples), 1e-9))
    headroom = min(float(config["resource_policy"]["variability_headroom_ceiling"]), max(float(config["resource_policy"]["variability_headroom_floor"]), max(ratios)))

    def guards(expected: float | None) -> dict[str, Any] | None:
        if expected is None:
            return None
        eta = expected * headroom
        return {
            "expected": expected,
            "variability_headroom": headroom,
            "eta": eta,
            "soft": eta * float(config["resource_policy"]["soft_guard_multiplier"]),
            "hard": eta * float(config["resource_policy"]["hard_guard_multiplier"]),
        }

    checks = {
        "all_route_classes": set(arms) == set(config["calibration"]["route_classes"]),
        "all_reload_exact": all(arm["reload_exact"] for arm in arms.values()),
        "rss": max(arm["resources"]["peak_rss_bytes"] for arm in arms.values()) <= int(config["resource_policy"]["admission_maximum_peak_rss_bytes"]),
        "finite_projection": math.isfinite(projected_wall) and math.isfinite(projected_cpu) and (projected_cgroup is None or math.isfinite(projected_cgroup)),
    }
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        "observed_complete_route": {"wall_seconds": route_wall, "process_cpu_seconds": route_cpu, "cgroup_cpu_seconds": route_cgroup, "peak_rss_bytes": max(arm["resources"]["peak_rss_bytes"] for arm in arms.values())},
        "variability_ratios": ratios,
        "projected_formal_route": {"wall_seconds": guards(projected_wall), "process_cpu_seconds": guards(projected_cpu), "cgroup_cpu_seconds": guards(projected_cgroup)},
        "projected_confirmation_route": {"wall_seconds": guards(projected_wall), "process_cpu_seconds": guards(projected_cpu), "cgroup_cpu_seconds": guards(projected_cgroup)},
        "arbitrary_static_wall_cap": None,
    }


def fixture_identity() -> dict[str, Any]:
    path = ROOT.parent / "t054" / "fixture-manifest.json"
    manifest = json.loads(path.read_text())
    required = {
        "all_sources_regular": True,
        "all_destinations_regular": True,
        "no_symlinks": True,
        "destination_hardlink_count_one": True,
        "source_destination_inode_distinct": True,
        "reflink_disabled": True,
        "primary_source_unchanged": True,
        "answers_or_credentials_copied": False,
    }
    if manifest["properties"] != required:
        raise RuntimeError("authenticated fixture properties changed")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "schema_version": manifest["schema_version"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "entries_tsv_sha256": manifest["entries_tsv_sha256"],
        "properties": manifest["properties"],
        "reuse_only_no_bytes_republished": True,
    }


def source_identity(config_path: Path) -> dict[str, Any]:
    generator = ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"
    return {
        "base_commit": git_output("rev-parse", "HEAD"),
        "config_sha256": sha256(config_path),
        "source_hashes": {path: sha256(ROOT / path) for path in SOURCE_PATHS},
        "generator_sha256": sha256(generator),
        "pdf_sha256": sha256(ROOT / "research/papers/arxiv-2602.11451.pdf"),
        "fixture_manifest": fixture_identity(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
    }


def root_identity(config: dict[str, Any], exp300: Any, root: int) -> dict[str, Any]:
    splits = make_splits(config, exp300, root)
    cases = {
        str(width): make_cases(
            exp300,
            splits[int(width)]["test"],
            root + int(config["data"]["evaluation_pair_seed_offset"]) + int(width),
            int(config["evaluation"]["cases_per_width"]),
        )
        for width in config["data"]["evaluation_widths"]
    }
    return {
        "root": root,
        "splits_sha256": sha256_json(splits),
        "evaluation_cases_sha256": sha256_json(cases),
        "training_pair_seeds": {str(width): root + int(config["data"]["train_pair_seed_offset"]) + int(width) for width in config["data"]["training_widths"]},
        "evaluation_pair_seeds": {str(width): root + int(config["data"]["evaluation_pair_seed_offset"]) + int(width) for width in config["data"]["evaluation_widths"]},
    }


def preregister(config_path: Path, output_base: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    exp300 = load_module("exp300_for_exp661_prereg", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    roots = [int(config["roots"]["calibration"]), int(config["roots"]["viability"]), *map(int, config["roots"]["confirmations"])]
    payload = {
        "schema_version": "exp661-preregistration-v1",
        "run_id": config["run_id"],
        "source": source_identity(config_path),
        "roots": {str(root): root_identity(config, exp300, root) for root in roots},
        "root_status": {str(root): "unopened" for root in roots},
        "gates": config["gates"],
        "confirmation": config["confirmation"],
        "resource_policy": config["resource_policy"],
        "budgets": config["evaluation"]["fixed_budgets"],
        "route_classes": config["calibration"]["route_classes"],
        "stopping": "Any viability conjunct miss rejects and leaves confirmation roots unopened.",
        "scientific_surfaces_excluded": ["checked", "prior-independent", "official", "private", "submission", "benchmark answers"],
    }
    output_base.mkdir(parents=True, exist_ok=True)
    path = output_base / "preregistration.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preregistration": str(path), "sha256": sha256(path), "roots": roots}, sort_keys=True))


def validate_preregistration(config_path: Path, output_base: Path) -> tuple[dict[str, Any], str]:
    path = output_base / "preregistration.json"
    frozen = json.loads(path.read_text())
    current = source_identity(config_path)
    for key in ("config_sha256", "source_hashes", "generator_sha256", "pdf_sha256", "fixture_manifest", "torch_version"):
        if frozen["source"][key] != current[key]:
            raise RuntimeError(f"preregistered {key} changed")
    return frozen, sha256(path)


def execute(config_path: Path, output_base: Path, eval_base: Path, mode: str, root: int) -> None:
    config = yaml.safe_load(config_path.read_text())
    frozen, prereg_sha = validate_preregistration(config_path, output_base)
    calibration_root = int(config["roots"]["calibration"])
    viability_root = int(config["roots"]["viability"])
    confirmation_roots = list(map(int, config["roots"]["confirmations"]))
    if mode == "calibration" and root != calibration_root:
        raise ValueError("calibration root mismatch")
    if mode == "formal" and root not in [viability_root, *confirmation_roots]:
        raise ValueError("formal root mismatch")
    if mode == "formal":
        admission = json.loads((output_base / "admission.json").read_text())
        if not admission["admitted"]:
            raise RuntimeError("calibration rejected formal execution")
    if root in confirmation_roots:
        viability = json.loads((output_base / "roots" / str(viability_root) / "result.json").read_text())
        if not viability["gate"]["passed"]:
            raise RuntimeError("confirmation root not authorized")
    result_path = output_base / "roots" / str(root) / "result.json"
    if result_path.exists() and bool(config["resource_policy"]["cache_completed_roots"]):
        cached = json.loads(result_path.read_text())
        if cached["preregistration_sha256"] != prereg_sha:
            raise RuntimeError("cached root preregistration mismatch")
        print(json.dumps({"cached": True, "root": root, "result": str(result_path)}))
        return
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    exp300 = load_module("exp300_for_exp661", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    splits = make_splits(config, exp300, root)
    identity = frozen["roots"][str(root)]
    if sha256_json(splits) != identity["splits_sha256"]:
        raise RuntimeError("split identity mismatch")
    cases = {
        int(width): make_cases(
            exp300,
            splits[int(width)]["test"],
            root + int(config["data"]["evaluation_pair_seed_offset"]) + int(width),
            int(config["evaluation"]["cases_per_width"]),
        )
        for width in config["data"]["evaluation_widths"]
    }
    if sha256_json({str(key): value for key, value in cases.items()}) != identity["evaluation_cases_sha256"]:
        raise RuntimeError("evaluation identity mismatch")
    steps = int(config["calibration"]["representative_updates_per_route"]) if mode == "calibration" else int(config["training"]["steps_per_arm"])
    root_dir = result_path.parent
    root_dir.mkdir(parents=True, exist_ok=True)
    wall_started, cpu_started, cgroup_started = time.perf_counter(), process_cpu_seconds(), cgroup_cpu_seconds()
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    arms: dict[str, dict[str, Any]] = {}
    with mlflow.start_run(run_name=f"exp661-{mode}-root-{root}") as active:
        mlflow.log_params({"root": root, "mode": mode, "steps_per_arm": steps, "preregistration_sha256": prereg_sha})
        for name in ARM_NAMES:
            arms[name] = run_arm(name, config, exp300, splits, cases, root, steps, root_dir)
            mlflow.log_metric(f"{name}_w16_b8_exact", arms[name]["evaluation"]["16"][str(config["model"]["trained_budget"])]["exact_accuracy"])
        diagnostics = primary_diagnostics(arms, config, cases, root)
        mlflow_run_id = active.info.run_id
    cgroup_finished = cgroup_cpu_seconds()
    route_wall, route_cpu = time.perf_counter() - wall_started, process_cpu_seconds() - cpu_started
    route_cgroup = None if cgroup_started is None or cgroup_finished is None else cgroup_finished - cgroup_started
    durable = {name: strip_models(arm) for name, arm in arms.items()}
    payload: dict[str, Any] = {
        "run_id": config["run_id"],
        "root": root,
        "mode": mode,
        "status": "complete",
        "preregistration_sha256": prereg_sha,
        "source": source_identity(config_path),
        "dataset": identity,
        "arms": durable,
        "primary_diagnostics": diagnostics,
        "resources": {"wall_seconds": route_wall, "process_cpu_seconds": route_cpu, "cgroup_cpu_seconds": route_cgroup, "peak_rss_bytes": peak_rss_bytes(), "torch_threads": int(config["training"]["torch_threads"])},
        "mlflow": {"experiment_id": experiment_id, "run_id": mlflow_run_id, "tracking_uri": tracking_uri()},
        "limitations": [
            "Fresh generated held-out-prime evidence only; checked, prior-independent, official, private, benchmark, and submission answers were not read.",
            "Bit-token endpoint BCE and generic learned positions are SAIR adaptations; this is not LoopViT, language-corpus scaling, or an instance-adaptive schedule policy.",
            "Budget 16 extrapolates beyond the paper-compatible trained maximum L=8 and is diagnostic rather than a source reproduction claim.",
            "Exact modular arithmetic is confined to offline labels, tests, and evaluator truth.",
        ],
    }
    if mode == "calibration":
        payload["admission"] = calibration_admission(arms, route_wall, route_cpu, route_cgroup, config)
        (output_base / "admission.json").write_text(json.dumps(payload["admission"], indent=2, sort_keys=True) + "\n")
    else:
        payload["gate"] = scientific_gate(arms, diagnostics, config)
        if root in confirmation_roots:
            payload["confirmation"] = {
                "viability_reproduced": payload["gate"]["passed"],
                "component_verdicts": {key: value for key, value in payload["gate"]["checks"].items() if "effect" in key},
                "fixed_user_budget_verified": diagnostics["fixed_user_budget"] and not diagnostics["learned_exit"],
            }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    eval_path = eval_base / "roots" / str(root) / "metrics.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps({"run_id": config["run_id"], "root": root, "mode": mode, "arms": durable, "primary_diagnostics": diagnostics, "decision": payload.get("gate", payload.get("admission")), "source_result_sha256": sha256(result_path)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"root": root, "mode": mode, "result": str(result_path), "result_sha256": sha256(result_path), "decision": payload.get("gate", payload.get("admission"))}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_661_loopformer_width_extrapolation.yaml")
    parser.add_argument("--output-base", type=Path, default=ROOT / "experiments/runs/exp-661-loopformer-width-extrapolation")
    parser.add_argument("--eval-base", type=Path, default=ROOT / "evals/results/exp-661-loopformer-width-extrapolation")
    parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True)
    parser.add_argument("--root", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preregister":
        if args.root is not None:
            raise ValueError("preregister freezes all roots; omit --root")
        preregister(args.config.resolve(), args.output_base.resolve())
    else:
        if args.root is None:
            raise ValueError("--root is required")
        execute(args.config.resolve(), args.output_base.resolve(), args.eval_base.resolve(), args.mode, int(args.root))


if __name__ == "__main__":
    main()
