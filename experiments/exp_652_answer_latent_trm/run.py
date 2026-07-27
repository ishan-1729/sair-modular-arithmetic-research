"""Freeze, calibrate, and execute EXP-652 answer-plus-latent TRM roots."""

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
from copy import deepcopy
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_652_answer_latent_trm.model import (  # noqa: E402
    AnswerLatentTRM,
    DecodedCellularControl,
    ExponentialMovingAverage,
    NoLatentControl,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

ARM_CLASSES = {
    "answer_latent_trm": AnswerLatentTRM,
    "decoded_cellular_control": DecodedCellularControl,
    "no_latent_control": NoLatentControl,
    "no_deep_supervision": AnswerLatentTRM,
    "no_q_loss": AnswerLatentTRM,
    "no_ema": AnswerLatentTRM,
}
SOURCE_PATHS = (
    "configs/exp_652_answer_latent_trm.yaml",
    "experiments/exp_652_answer_latent_trm/model.py",
    "experiments/exp_652_answer_latent_trm/run.py",
    "tests/test_exp_652_answer_latent_trm.py",
    "knowledge/autoresearch/research/2026-07-27-exp652-trm-mechanism.md",
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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def process_cpu_seconds() -> float:
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


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
    lsb = torch.zeros_like(left)
    msb = torch.zeros_like(left)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack((left, right, modulus, lsb, msb), dim=1).mul(2.0).sub(1.0)


def endpoint_targets(cases: list[dict[str, int]], width: int) -> Tensor:
    values = [
        (case["left"] * case["right"]) % case["modulus"] for case in cases
    ]
    return bits(values, width)


def decode_rows(rows: Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in rows.tolist()
    ]


def make_splits(config: dict[str, Any], exp300: Any, root: int) -> dict[int, Any]:
    result = {}
    for width in config["data"]["evaluation_widths"]:
        local = {
            "data": {
                "prime_bits_min": max(
                    3, width - int(config["data"]["prime_bit_span"]) + 1
                ),
                "prime_bits_max": width,
                "train_prime_identities": int(
                    config["data"]["train_prime_identities"]
                ),
                "test_prime_identities": int(config["data"]["test_prime_identities"]),
            }
        }
        result[int(width)] = exp300.bigint_prime_split(
            local, root + int(config["data"]["split_seed_offset"]) + int(width)
        )
    return result


def make_cases(
    exp300: Any, identities: list[int], seed: int, count: int
) -> list[dict[str, int]]:
    sampler = exp300.BigIntSampler(identities, seed)
    left, right, modulus = sampler.pairs(count)
    return [
        {"left": int(a), "right": int(b), "modulus": int(p)}
        for a, b, p in zip(left, right, modulus)
    ]


def build_model(name: str, config: dict[str, Any]) -> AnswerLatentTRM:
    cls = ARM_CLASSES[name]
    model = cls(
        hidden=int(config["model"]["hidden"]),
        latent_updates=int(config["model"]["latent_updates"]),
        outer_recursions=int(config["model"]["outer_recursions"]),
    )
    return model.cpu()


def selected_logits(output: Any, learned_halt: bool, threshold: float) -> tuple[Tensor, Tensor]:
    if not learned_halt:
        return output.logits, output.q_logit
    logits = torch.stack(output.answer_curve, dim=1)
    q_logits = torch.stack(output.q_curve, dim=1)
    halt = q_logits.sigmoid().ge(threshold)
    first = torch.full(
        (logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device
    )
    for index in range(logits.shape[1]):
        first = torch.where(halt[:, index] & first.eq(logits.shape[1] - 1), index, first)
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, first], q_logits[rows, first]


@torch.inference_mode()
def evaluate_model(
    model: AnswerLatentTRM,
    cases: list[dict[str, int]],
    width: int,
    config: dict[str, Any],
    *,
    outer_recursions: int | None = None,
    state_intervention: str = "none",
    learned_halt: bool = False,
    supervision_steps: int | None = None,
    random_seed: int = 0,
) -> dict[str, Any]:
    model.eval()
    batch_size = int(config["evaluation"]["batch_size"])
    total = correct = bit_correct = invalid = 0
    first_errors: list[int] = []
    q_probabilities: list[float] = []
    q_targets: list[float] = []
    generator = torch.Generator().manual_seed(random_seed)
    steps = (
        int(config["model"]["supervision_steps"])
        if supervision_steps is None
        else int(supervision_steps)
    )
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        features = feature_tensor(batch, width)
        targets = endpoint_targets(batch, width).bool()
        output = model.deep_recursion(
            features,
            supervision_steps=steps,
            outer_recursions=outer_recursions,
            state_intervention=state_intervention,
            learned_halt=False,
            halt_threshold=float(config["evaluation"]["q_halt_threshold"]),
            generator=generator,
        )
        logits, q_logit = selected_logits(
            output,
            learned_halt,
            float(config["evaluation"]["q_halt_threshold"]),
        )
        predicted = logits.ge(0)
        exact = predicted.eq(targets).all(dim=-1)
        values = decode_rows(predicted)
        correct += int(exact.sum())
        bit_correct += int(predicted.eq(targets).sum())
        invalid += sum(value >= case["modulus"] for value, case in zip(values, batch))
        for row, truth in zip(predicted, targets):
            mismatch = row.ne(truth).nonzero()
            if len(mismatch):
                first_errors.append(int(mismatch[0, 0]))
        q_probabilities.extend(q_logit.sigmoid().tolist())
        q_targets.extend(exact.float().tolist())
        total += len(batch)
    brier = statistics.mean(
        (probability - target) ** 2
        for probability, target in zip(q_probabilities, q_targets)
    )
    q_accuracy = statistics.mean(
        (probability >= 0.5) == bool(target)
        for probability, target in zip(q_probabilities, q_targets)
    )
    return {
        "cases": total,
        "correct": correct,
        "exact_accuracy": correct / total,
        "bit_accuracy": bit_correct / (total * width),
        "invalid_rate": invalid / total,
        "first_error": {
            "error_cases": len(first_errors),
            "mean_lsb_position": (
                statistics.mean(first_errors) if first_errors else None
            ),
        },
        "q": {"brier": brier, "threshold_accuracy": q_accuracy},
    }


def save_state(path: Path, state: dict[str, Tensor]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().contiguous() for key, value in state.items()}, path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def exact_reload(
    name: str,
    config: dict[str, Any],
    checkpoint: Path,
    reference: AnswerLatentTRM,
    cases: list[dict[str, int]],
    width: int,
) -> bool:
    loaded = build_model(name, config)
    loaded.load_state_dict(load_file(checkpoint), strict=True)
    features = feature_tensor(cases[:16], width)
    reference.eval()
    loaded.eval()
    with torch.inference_mode():
        expected = reference.deep_recursion(
            features, supervision_steps=int(config["model"]["supervision_steps"])
        ).logits
        observed = loaded.deep_recursion(
            features, supervision_steps=int(config["model"]["supervision_steps"])
        ).logits
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
    random.seed(root + 652)
    np.random.seed(root + 652)
    torch.manual_seed(root + 652)
    model = build_model(name, config)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_identity = sha256_json(
        {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in initial_state.items()}
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
    ema = ExponentialMovingAverage(model, float(config["model"]["ema_decay"]))
    width = int(config["data"]["train_width"])
    sampler = exp300.BigIntSampler(
        splits[width]["train"], root + int(config["data"]["train_pair_seed_offset"])
    )
    supervision_steps = 1 if name == "no_deep_supervision" else int(config["model"]["supervision_steps"])
    q_weight = 0.0 if name == "no_q_loss" else float(config["training"]["q_loss_weight"])
    update_wall_seconds: list[float] = []
    update_process_cpu_seconds: list[float] = []
    update_cgroup_cpu_seconds: list[float] = []
    history: list[dict[str, Any]] = []
    process = psutil.Process()
    peak = process.memory_info().rss
    wall_started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    cgroup_started = cgroup_cpu_seconds()
    model.train()
    for step in range(1, steps + 1):
        left, right, modulus = sampler.pairs(int(config["training"]["batch_size"]))
        cases = [
            {"left": int(a), "right": int(b), "modulus": int(p)}
            for a, b, p in zip(left, right, modulus)
        ]
        features = feature_tensor(cases, width)
        targets = endpoint_targets(cases, width)
        step_wall = time.perf_counter()
        step_cpu = process_cpu_seconds()
        step_cgroup = cgroup_cpu_seconds()
        output = model.deep_recursion(features, supervision_steps=supervision_steps)
        endpoint_losses = [
            F.binary_cross_entropy_with_logits(logits, targets)
            for logits in output.answer_curve
        ]
        q_losses = []
        for logits, q_logit in zip(output.answer_curve, output.q_curve):
            correctness = logits.detach().ge(0).eq(targets.bool()).all(dim=-1).float()
            q_losses.append(F.binary_cross_entropy_with_logits(q_logit, correctness))
        loss = torch.stack(endpoint_losses).mean() + q_weight * torch.stack(q_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
        )
        optimizer.step()
        scheduler.step()
        if name != "no_ema":
            ema.update(model)
        update_wall_seconds.append(time.perf_counter() - step_wall)
        update_process_cpu_seconds.append(process_cpu_seconds() - step_cpu)
        finished_cgroup = cgroup_cpu_seconds()
        if step_cgroup is not None and finished_cgroup is not None:
            update_cgroup_cpu_seconds.append(finished_cgroup - step_cgroup)
        peak = max(peak, process.memory_info().rss)
        if step == 1 or step % int(config["training"]["log_every_steps"]) == 0 or step == steps:
            exact = output.logits.detach().ge(0).eq(targets.bool()).all(dim=-1).float().mean()
            item = {
                "step": step,
                "loss": float(loss.detach()),
                "batch_exact_accuracy": float(exact),
                "gradient_norm": gradient,
                "elapsed_wall_seconds": time.perf_counter() - wall_started,
            }
            history.append(item)
            print(json.dumps({"arm": name, **item}), flush=True)
    if name == "no_ema":
        ema.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
        ema.updates = steps
    wall_finished = time.perf_counter()
    cpu_finished = process_cpu_seconds()
    cgroup_finished = cgroup_cpu_seconds()
    raw_path = output_dir / name / "raw.safetensors"
    ema_path = output_dir / name / "ema.safetensors"
    raw_checkpoint = save_state(raw_path, model.state_dict())
    ema_checkpoint = save_state(ema_path, ema.shadow)
    ema_model = ema.model_copy(model).eval()
    model.eval()
    raw_evaluation = {
        str(eval_width): evaluate_model(
            model, evaluation_cases[eval_width], eval_width, config, random_seed=root + eval_width
        )
        for eval_width in config["data"]["evaluation_widths"]
    }
    ema_evaluation = {
        str(eval_width): evaluate_model(
            ema_model,
            evaluation_cases[eval_width],
            eval_width,
            config,
            random_seed=root + eval_width,
        )
        for eval_width in config["data"]["evaluation_widths"]
    }
    raw_reload = exact_reload(name, config, raw_path, model, evaluation_cases[width], width)
    ema_reload = exact_reload(name, config, ema_path, ema_model, evaluation_cases[width], width)
    return {
        "name": name,
        "architecture": model.architecture,
        "parameters": model.parameter_count(),
        "initial_state_sha256": initial_identity,
        "training": {
            "optimizer_updates": steps,
            "trajectory_examples": steps * int(config["training"]["batch_size"]),
            "supervision_steps": supervision_steps,
            "shared_network_calls_per_example": supervision_steps
            * int(config["model"]["outer_recursions"])
            * (int(config["model"]["latent_updates"]) + 1),
            "backpropagated_shared_calls_per_example": supervision_steps
            * (int(config["model"]["latent_updates"]) + 1),
            "q_loss_weight": q_weight,
            "ema_updates": ema.updates,
            "history": history,
        },
        "checkpoints": {"raw": raw_checkpoint, "ema": ema_checkpoint},
        "reload_exact": {"raw": raw_reload, "ema": ema_reload},
        "evaluation": {"raw": raw_evaluation, "ema": ema_evaluation},
        "resources": {
            "training_wall_seconds": wall_finished - wall_started,
            "training_process_cpu_seconds": cpu_finished - cpu_started,
            "training_cgroup_cpu_seconds": (
                None
                if cgroup_started is None or cgroup_finished is None
                else cgroup_finished - cgroup_started
            ),
            "update_wall_seconds": update_wall_seconds,
            "update_process_cpu_seconds": update_process_cpu_seconds,
            "update_cgroup_cpu_seconds": update_cgroup_cpu_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "_raw_model": model,
        "_ema_model": ema_model,
    }


def primary_diagnostics(
    arm: dict[str, Any],
    config: dict[str, Any],
    cases: dict[int, list[dict[str, int]]],
    root: int,
) -> dict[str, Any]:
    model = arm["_ema_model"]
    width = int(config["data"]["train_width"])
    depth_curve = {
        str(depth): evaluate_model(
            model,
            cases[width],
            width,
            config,
            outer_recursions=int(depth),
            random_seed=root + depth,
        )
        for depth in config["evaluation"]["fixed_depths"]
    }
    interventions = {
        intervention: evaluate_model(
            model,
            cases[width],
            width,
            config,
            state_intervention=intervention,
            random_seed=root + 1000 + index,
        )
        for index, intervention in enumerate(config["evaluation"]["interventions"])
    }
    q_halt = evaluate_model(
        model,
        cases[width],
        width,
        config,
        learned_halt=True,
        supervision_steps=int(config["model"]["supervision_steps"]),
        random_seed=root + 2000,
    )
    randomized = deepcopy(model)
    torch.manual_seed(root + 9999)
    randomized.reset_parameters()
    randomized_result = evaluate_model(
        randomized, cases[width], width, config, random_seed=root + 3000
    )
    return {
        "recursion_depth_curve": depth_curve,
        "state_interventions": interventions,
        "q_halt": q_halt,
        "randomized_weights": randomized_result,
    }


def strip_models(arm: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arm.items() if not key.startswith("_")}


def variability_headroom(samples: list[float], config: dict[str, Any]) -> float:
    median = statistics.median(samples)
    p90 = sorted(samples)[max(0, math.ceil(0.9 * len(samples)) - 1)]
    observed = p90 / median if median else 1.0
    policy = config["resource_policy"]
    return min(
        float(policy["variability_headroom_ceiling"]),
        max(float(policy["variability_headroom_floor"]), observed),
    )


def calibration_admission(
    arms: dict[str, dict[str, Any]],
    route_wall_seconds: float,
    route_process_cpu_seconds: float,
    route_cgroup_cpu_seconds: float | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    samples = [
        sample
        for arm in arms.values()
        for sample in arm["resources"]["update_wall_seconds"]
    ]
    headroom = variability_headroom(samples, config)
    calibration_steps = int(config["calibration"]["representative_updates_per_route"])
    formal_steps = int(config["training"]["steps_per_arm"])
    projected_training_wall = sum(
        arm["resources"]["training_wall_seconds"] / calibration_steps * formal_steps
        for arm in arms.values()
    )
    projected_training_process = sum(
        arm["resources"]["training_process_cpu_seconds"] / calibration_steps * formal_steps
        for arm in arms.values()
    )
    projected_training_cgroup = sum(
        (arm["resources"]["training_cgroup_cpu_seconds"] or arm["resources"]["training_process_cpu_seconds"])
        / calibration_steps
        * formal_steps
        for arm in arms.values()
    )
    measured_training_wall = sum(
        arm["resources"]["training_wall_seconds"] for arm in arms.values()
    )
    measured_training_process = sum(
        arm["resources"]["training_process_cpu_seconds"] for arm in arms.values()
    )
    measured_training_cgroup = sum(
        arm["resources"]["training_cgroup_cpu_seconds"] or arm["resources"]["training_process_cpu_seconds"]
        for arm in arms.values()
    )
    evaluation_wall = max(0.0, route_wall_seconds - measured_training_wall)
    evaluation_process = max(0.0, route_process_cpu_seconds - measured_training_process)
    evaluation_cgroup = max(
        0.0,
        (route_cgroup_cpu_seconds or route_process_cpu_seconds) - measured_training_cgroup,
    )
    expected_wall = (projected_training_wall + evaluation_wall) * headroom
    expected_process = (projected_training_process + evaluation_process) * headroom
    expected_cgroup = (projected_training_cgroup + evaluation_cgroup) * headroom
    policy = config["resource_policy"]
    peak = max(arm["resources"]["peak_rss_bytes"] for arm in arms.values())
    checks = {
        "all_route_classes": set(arms) == set(config["calibration"]["route_classes"]),
        "finite_positive_projection": all(
            math.isfinite(value) and value > 0.0
            for value in (expected_wall, expected_process, expected_cgroup)
        ),
        "rss": peak <= int(policy["admission_maximum_peak_rss_bytes"]),
        "raw_reload": all(arm["reload_exact"]["raw"] for arm in arms.values()),
        "ema_reload": all(arm["reload_exact"]["ema"] for arm in arms.values()),
    }
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        "variability": {
            "samples": len(samples),
            "median_update_wall_seconds": statistics.median(samples),
            "min_update_wall_seconds": min(samples),
            "max_update_wall_seconds": max(samples),
            "headroom_multiplier": headroom,
        },
        "projected_confirmation_route": {
            "expected_wall_seconds": expected_wall,
            "soft_wall_seconds": expected_wall * float(policy["soft_guard_multiplier"]),
            "hard_wall_seconds": expected_wall * float(policy["hard_guard_multiplier"]),
            "expected_process_cpu_seconds": expected_process,
            "soft_process_cpu_seconds": expected_process * float(policy["soft_guard_multiplier"]),
            "hard_process_cpu_seconds": expected_process * float(policy["hard_guard_multiplier"]),
            "expected_cgroup_cpu_seconds": expected_cgroup,
            "soft_cgroup_cpu_seconds": expected_cgroup * float(policy["soft_guard_multiplier"]),
            "hard_cgroup_cpu_seconds": expected_cgroup * float(policy["hard_guard_multiplier"]),
            "peak_rss_bytes": peak,
        },
        "fidelity": {
            "formal_updates_per_arm": formal_steps,
            "cases_per_width": int(config["evaluation"]["cases_per_width"]),
            "widths": config["data"]["evaluation_widths"],
            "all_potential_formal_routes_profiled": True,
            "evaluation_contract_reduced": False,
        },
    }


def scientific_gate(
    arms: dict[str, dict[str, Any]], diagnostics: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    primary = arms["answer_latent_trm"]["evaluation"]["ema"]
    controls = [
        arms[name]["evaluation"]["ema"]
        for name in ("decoded_cellular_control", "no_latent_control")
    ]
    gates = config["gates"]
    w16 = primary["16"]["exact_accuracy"]
    latent_drop = w16 - max(
        diagnostics["state_interventions"][name]["exact_accuracy"]
        for name in ("reset_z", "randomize_z", "swap_y_z")
    )
    parameter_ratios = {
        name: arms[name]["parameters"] / arms["answer_latent_trm"]["parameters"]
        for name in ("decoded_cellular_control", "no_latent_control")
    }
    checks = {
        "w16": w16 >= float(gates["minimum_primary_w16_exact_accuracy"]),
        "w24": primary["24"]["exact_accuracy"] >= float(gates["minimum_primary_w24_exact_accuracy"]),
        "w32": primary["32"]["exact_accuracy"] >= float(gates["minimum_primary_w32_exact_accuracy"]),
        "gain_over_decoded_w24": primary["24"]["exact_accuracy"] - controls[0]["24"]["exact_accuracy"]
        >= float(gates["minimum_transfer_gain_over_each_control"]),
        "gain_over_no_latent_w24": primary["24"]["exact_accuracy"] - controls[1]["24"]["exact_accuracy"]
        >= float(gates["minimum_transfer_gain_over_each_control"]),
        "gain_over_decoded_w32": primary["32"]["exact_accuracy"] - controls[0]["32"]["exact_accuracy"]
        >= float(gates["minimum_transfer_gain_over_each_control"]),
        "gain_over_no_latent_w32": primary["32"]["exact_accuracy"] - controls[1]["32"]["exact_accuracy"]
        >= float(gates["minimum_transfer_gain_over_each_control"]),
        "trained_weight_dependence": w16 - diagnostics["randomized_weights"]["exact_accuracy"]
        >= float(gates["minimum_trained_gain_over_randomized_w16"]),
        "latent_causality": latent_drop >= float(gates["minimum_latent_causal_drop_w16"]),
        "validity": max(
            result[width]["invalid_rate"]
            for arm in arms.values()
            for result in (arm["evaluation"]["raw"], arm["evaluation"]["ema"])
            for width in ("16", "24", "32")
        ) <= float(gates["maximum_invalid_rate"]),
        "raw_reload": all(arm["reload_exact"]["raw"] for arm in arms.values()),
        "ema_reload": all(arm["reload_exact"]["ema"] for arm in arms.values()),
        "parameter_match": all(
            1 / float(gates["maximum_parameter_ratio"]) <= ratio <= float(gates["maximum_parameter_ratio"])
            for ratio in parameter_ratios.values()
        ),
        "example_update_match": len(
            {
                (arm["training"]["optimizer_updates"], arm["training"]["trajectory_examples"])
                for arm in arms.values()
            }
        ) == 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "latent_causal_drop_w16": latent_drop,
        "parameter_ratios_to_primary": parameter_ratios,
        "decision": "advance-to-confirmations" if all(checks.values()) else "reject",
    }


def confirmation_verdict(arms: dict[str, dict[str, Any]], gate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    primary = arms["answer_latent_trm"]["evaluation"]["ema"]["16"]["exact_accuracy"]
    minimum = float(config["confirmation"]["minimum_component_causal_drop_w16"])
    drops = {
        name: primary - arms[name]["evaluation"]["ema"]["16"]["exact_accuracy"]
        for name in config["confirmation"]["component_controls"]
    }
    return {
        "viability_reproduced": gate["passed"],
        "component_drops_w16": drops,
        "component_causal": {name: drop >= minimum for name, drop in drops.items()},
    }


def source_identity(config_path: Path) -> dict[str, Any]:
    return {
        "repository_base_commit": git_output("rev-parse", "HEAD"),
        "repository_dirty": bool(git_output("status", "--porcelain")),
        "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
        "source_hashes": {path: sha256(ROOT / path) for path in SOURCE_PATHS},
        "config_sha256": sha256(config_path),
        "generator_sha256": sha256(ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def seed_for_root(config: dict[str, Any], root: int) -> int:
    return int(root)


def preregister(config_path: Path, output_base: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    exp300 = load_module("exp300_for_exp652", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    roots = [config["roots"]["calibration"], config["roots"]["viability"], *config["roots"]["confirmations"]]
    identities = {}
    for root in roots:
        root = int(root)
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
        identities[str(root)] = {
            "seed": seed_for_root(config, root),
            "splits_sha256": sha256_json(splits),
            "evaluation_cases_sha256": sha256_json(cases),
        }
    payload = {
        "status": "frozen-before-scientific-training",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roots": config["roots"],
        "identities": identities,
        "source": source_identity(config_path),
        "model": config["model"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "gates": config["gates"],
        "confirmation": config["confirmation"],
        "resource_policy": config["resource_policy"],
        "mechanism_scope": {
            "family": "ST-M010.a TRM",
            "sair_adaptation": True,
            "tab_trm_reproduced": False,
            "exact_arithmetic_scope": "offline-labels-tests-evaluator-only",
        },
    }
    output_base.mkdir(parents=True, exist_ok=True)
    path = output_base / "preregistration.json"
    if path.exists():
        raise RuntimeError("preregistration already exists")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preregistration": str(path), "sha256": sha256(path)}))


def validate_preregistration(config_path: Path, output_base: Path) -> tuple[dict[str, Any], str]:
    path = output_base / "preregistration.json"
    frozen = json.loads(path.read_text())
    current = source_identity(config_path)
    for key in ("source_hashes", "config_sha256", "generator_sha256", "torch_version"):
        if frozen["source"][key] != current[key]:
            raise RuntimeError(f"preregistered {key} changed")
    return frozen, sha256(path)


def execute(config_path: Path, output_base: Path, eval_base: Path, mode: str, root: int) -> None:
    config = yaml.safe_load(config_path.read_text())
    frozen, prereg_sha = validate_preregistration(config_path, output_base)
    if mode == "calibration" and root != int(config["roots"]["calibration"]):
        raise ValueError("calibration root mismatch")
    if mode == "formal" and root not in [int(config["roots"]["viability"]), *map(int, config["roots"]["confirmations"])]:
        raise ValueError("formal root mismatch")
    if mode == "formal":
        admission = json.loads((output_base / "admission.json").read_text())
        if not admission["admitted"]:
            raise RuntimeError("calibration rejected formal execution")
    if root in map(int, config["roots"]["confirmations"]):
        viability = json.loads((output_base / "roots" / str(config["roots"]["viability"]) / "result.json").read_text())
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
    exp300 = load_module("exp300_for_exp652", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    splits = make_splits(config, exp300, root)
    identity = frozen["identities"][str(root)]
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
    names = (
        list(config["calibration"]["route_classes"])
        if mode == "calibration" or root in map(int, config["roots"]["confirmations"])
        else ["answer_latent_trm", "decoded_cellular_control", "no_latent_control"]
    )
    steps = (
        int(config["calibration"]["representative_updates_per_route"])
        if mode == "calibration"
        else int(config["training"]["steps_per_arm"])
    )
    root_dir = result_path.parent
    root_dir.mkdir(parents=True, exist_ok=True)
    wall_started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    cgroup_started = cgroup_cpu_seconds()
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    arms: dict[str, dict[str, Any]] = {}
    with mlflow.start_run(run_name=f"exp652-{mode}-root-{root}") as active:
        mlflow.log_params({"root": root, "mode": mode, "seed": root, "steps_per_arm": steps, "preregistration_sha256": prereg_sha})
        for name in names:
            arms[name] = run_arm(name, config, exp300, splits, cases, root, steps, root_dir)
            mlflow.log_metrics({f"{name}_ema_w16": arms[name]["evaluation"]["ema"]["16"]["exact_accuracy"]})
        diagnostics = primary_diagnostics(arms["answer_latent_trm"], config, cases, root)
        mlflow_run_id = active.info.run_id
    cgroup_finished = cgroup_cpu_seconds()
    route_wall = time.perf_counter() - wall_started
    route_cpu = process_cpu_seconds() - cpu_started
    route_cgroup = None if cgroup_started is None or cgroup_finished is None else cgroup_finished - cgroup_started
    durable_arms = {name: strip_models(arm) for name, arm in arms.items()}
    payload: dict[str, Any] = {
        "run_id": config["run_id"],
        "root": root,
        "mode": mode,
        "status": "complete",
        "seed": seed_for_root(config, root),
        "preregistration_sha256": prereg_sha,
        "source": source_identity(config_path),
        "dataset": identity,
        "arms": durable_arms,
        "primary_diagnostics": diagnostics,
        "resources": {
            "wall_seconds": route_wall,
            "process_cpu_seconds": route_cpu,
            "cgroup_cpu_seconds": route_cgroup,
            "peak_rss_bytes": peak_rss_bytes(),
            "torch_threads": int(config["training"]["torch_threads"]),
        },
        "mlflow": {"experiment_id": experiment_id, "run_id": mlflow_run_id, "tracking_uri": tracking_uri()},
        "limitations": [
            "Fresh generated held-out-prime evidence only; checked, prior-independent, official, private, benchmark, and submission answers were not read.",
            "The local two-layer convolutional shared network is a declared SAIR adaptation of TRM, not Tab-TRM or byte-level reproduction of the paper implementation.",
            "Exact modular arithmetic is confined to offline endpoint labels, tests, and evaluator truth.",
        ],
    }
    if mode == "calibration":
        payload["admission"] = calibration_admission(arms, route_wall, route_cpu, route_cgroup, config)
        (output_base / "admission.json").write_text(json.dumps(payload["admission"], indent=2, sort_keys=True) + "\n")
    else:
        primary_names = {name: arms[name] for name in ("answer_latent_trm", "decoded_cellular_control", "no_latent_control")}
        payload["gate"] = scientific_gate(primary_names, diagnostics, config)
        if root in map(int, config["roots"]["confirmations"]):
            payload["confirmation"] = confirmation_verdict(arms, payload["gate"], config)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    eval_path = eval_base / "roots" / str(root) / "metrics.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(
        json.dumps(
            {
                "run_id": config["run_id"],
                "root": root,
                "mode": mode,
                "arms": {name: {"parameters": arm["parameters"], "evaluation": arm["evaluation"]} for name, arm in durable_arms.items()},
                "primary_diagnostics": diagnostics,
                "decision": payload.get("gate", payload.get("admission")),
                "source_result_sha256": sha256(result_path),
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    print(json.dumps({"root": root, "mode": mode, "result": str(result_path), "result_sha256": sha256(result_path), "decision": payload.get("gate", payload.get("admission"))}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_652_answer_latent_trm.yaml")
    parser.add_argument("--output-base", type=Path, default=ROOT / "experiments/runs/exp-652-answer-latent-trm")
    parser.add_argument("--eval-base", type=Path, default=ROOT / "evals/results/exp-652-answer-latent-trm")
    parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True)
    parser.add_argument("--root", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preregister":
        if args.root is not None:
            raise ValueError("preregister freezes all roots; omit --root")
        preregister(args.config.resolve(), args.output_base.resolve())
        return
    if args.root is None:
        raise ValueError("--root is required")
    execute(args.config.resolve(), args.output_base.resolve(), args.eval_base.resolve(), args.mode, int(args.root))


if __name__ == "__main__":
    main()
