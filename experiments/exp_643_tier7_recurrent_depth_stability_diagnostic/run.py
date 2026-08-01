"""Diagnose frozen recurrent-depth stability at widths 128 and 256 for EXP-643."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mlflow
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file


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


def load_exp300():
    previous_path = sys.path.copy()
    previous_model = sys.modules.get("model")
    had_model = "model" in sys.modules
    try:
        return load_module(
            "exp300_for_exp643",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
    finally:
        sys.path[:] = previous_path
        if had_model:
            sys.modules["model"] = previous_model
        else:
            sys.modules.pop("model", None)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def process_cpu_seconds() -> float:
    times = psutil.Process().cpu_times()
    return float(times.user + times.system)


def peak_rss_bytes() -> int:
    # Linux ru_maxrss is KiB. EXP-643 is required to run through Ubuntu WSL.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def decode_bits(rows: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in rows.tolist()
    ]


def make_split(config: dict[str, Any], exp300, width: int, seed: int) -> dict[str, list[int]]:
    fresh = config["fresh_evidence"]
    local = {
        "data": {
            "prime_bits_min": width - int(fresh["prime_bit_span"]) + 1,
            "prime_bits_max": width,
            "train_prime_identities": int(fresh["train_prime_identities"]),
            "test_prime_identities": int(fresh["test_prime_identities"]),
        }
    }
    return exp300.bigint_prime_split(
        local, seed + int(fresh["split_seed_offset"]) + width
    )


def _head_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    return model.local_head(F.silu(model.output_norm(hidden))).squeeze(1)


def instrumented_forward(
    model, features: torch.Tensor, teacher_bits: torch.Tensor | None = None
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor | float | int]]]:
    """Run the exact ordinary recurrence while exposing per-visit diagnostics."""
    hidden = model.stem(features)
    previous_logits = _head_logits(model, hidden)
    previous_prediction = previous_logits.ge(0)
    visits: list[dict[str, torch.Tensor | float | int]] = []
    schedule = model.dilation_schedule(features.shape[-1])
    for visit_index, dilation in enumerate(schedule):
        normalized = model.state_norm(hidden)
        combined = torch.cat((normalized, features), dim=1)
        gates = torch.sigmoid(
            F.conv1d(
                combined,
                model.gate_weight,
                model.gate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, features), dim=1)
        candidate = torch.tanh(
            F.conv1d(
                candidate_inputs,
                model.candidate_weight,
                model.candidate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        next_hidden = model.recurrent_update(hidden, features, dilation)
        logits = _head_logits(model, next_hidden)
        prediction = logits.ge(0)
        record: dict[str, torch.Tensor | float | int] = {
            "visit_index": visit_index,
            "dilation": dilation,
            "hidden_rms": next_hidden.square().mean((1, 2)).sqrt(),
            "state_delta_rms": (next_hidden - hidden).square().mean((1, 2)).sqrt(),
            "normalized_mean": normalized.mean((1, 2)),
            "normalized_std": normalized.std((1, 2), unbiased=False),
            "update_mean": update.mean((1, 2)),
            "update_saturation": ((update < 0.05) | (update > 0.95)).float().mean((1, 2)),
            "reset_mean": reset.mean((1, 2)),
            "reset_saturation": ((reset < 0.05) | (reset > 0.95)).float().mean((1, 2)),
            "candidate_rms": candidate.square().mean((1, 2)).sqrt(),
            "logit_delta_rms": (logits - previous_logits).square().mean(1).sqrt(),
            "output_flip_rate": (prediction != previous_prediction).float().mean(1),
            "minimum_absolute_margin": logits.abs().amin(1),
            "prediction": prediction,
        }
        if teacher_bits is not None:
            record["exact_to_teacher"] = (prediction == teacher_bits).all(1)
        visits.append(record)
        hidden = next_hidden
        previous_logits = logits
        previous_prediction = prediction
    return previous_logits, visits


def jacobian_gain_proxy(
    model,
    features: torch.Tensor,
    visit_fractions: list[float],
    directions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return fixed random-direction lower bounds on local recurrent Jacobian gain."""
    schedule = model.dilation_schedule(features.shape[-1])
    selected = sorted(
        {
            min(len(schedule) - 1, max(0, round(fraction * (len(schedule) - 1))))
            for fraction in visit_fractions
        }
    )
    generator = torch.Generator(device=features.device).manual_seed(seed)
    hidden = model.stem(features).detach()
    results: list[dict[str, Any]] = []
    for visit_index, dilation in enumerate(schedule):
        if visit_index in selected:
            gains = []
            base = hidden.detach()
            for _ in range(directions):
                direction = torch.randn(
                    base.shape, generator=generator, dtype=base.dtype, device=base.device
                )
                direction = direction / direction.norm().clamp_min(1e-12)
                _, tangent = torch.func.jvp(
                    lambda state: model.recurrent_update(state, features, dilation),
                    (base,),
                    (direction,),
                )
                gains.append(float(tangent.norm()))
            results.append(
                {
                    "visit_index": visit_index,
                    "dilation": dilation,
                    "direction_gains": gains,
                    "maximum_random_direction_gain": max(gains),
                    "mean_random_direction_gain": sum(gains) / len(gains),
                }
            )
        # Keep ordinary tensors here because a later selected visit must support JVP.
        with torch.no_grad():
            hidden = model.recurrent_update(hidden, features, dilation)
    return results


VISIT_METRICS = (
    "hidden_rms",
    "state_delta_rms",
    "normalized_mean",
    "normalized_std",
    "update_mean",
    "update_saturation",
    "reset_mean",
    "reset_saturation",
    "candidate_rms",
    "logit_delta_rms",
    "output_flip_rate",
    "minimum_absolute_margin",
)


class VisitAccumulator:
    def __init__(self, schedule: list[int]) -> None:
        self.schedule = schedule
        self.sums = {
            group: [{metric: 0.0 for metric in VISIT_METRICS} for _ in schedule]
            for group in ("success", "failure")
        }
        self.counts = {
            group: [0 for _ in schedule] for group in ("success", "failure")
        }
        self.transitions = 0
        self.errors = 0
        self.overthinking = 0
        self.late_repairs = 0
        self.final_visit_converged = 0

    def add(self, visits: list[dict[str, Any]]) -> None:
        exact = torch.stack([visit["exact_to_teacher"] for visit in visits], dim=0)
        final_exact = exact[-1]
        earlier_exact = exact[:-1].any(0) if len(visits) > 1 else torch.zeros_like(final_exact)
        earlier_wrong = (~exact[:-1]).any(0) if len(visits) > 1 else torch.zeros_like(final_exact)
        self.transitions += int(final_exact.numel())
        self.errors += int((~final_exact).sum())
        self.overthinking += int((earlier_exact & ~final_exact).sum())
        self.late_repairs += int((earlier_wrong & final_exact).sum())
        if len(visits) > 1:
            converged = (visits[-1]["prediction"] == visits[-2]["prediction"]).all(1)
            self.final_visit_converged += int(converged.sum())
        else:
            self.final_visit_converged += int(final_exact.numel())
        for group, mask in (("success", final_exact), ("failure", ~final_exact)):
            count = int(mask.sum())
            if count == 0:
                continue
            for visit_index, visit in enumerate(visits):
                self.counts[group][visit_index] += count
                for metric in VISIT_METRICS:
                    self.sums[group][visit_index][metric] += float(visit[metric][mask].sum())

    def summary(self) -> dict[str, Any]:
        profiles: dict[str, list[dict[str, Any]]] = {}
        for group in ("success", "failure"):
            profiles[group] = []
            for visit_index, dilation in enumerate(self.schedule):
                count = self.counts[group][visit_index]
                profiles[group].append(
                    {
                        "visit_index": visit_index,
                        "dilation": dilation,
                        "count": count,
                        **{
                            metric: self.sums[group][visit_index][metric] / count
                            if count
                            else None
                            for metric in VISIT_METRICS
                        },
                    }
                )
        late_start = max(0, math.floor(0.75 * len(self.schedule)))

        def late(group: str, metric: str) -> float | None:
            values = [
                item[metric]
                for item in profiles[group][late_start:]
                if item[metric] is not None
            ]
            return sum(values) / len(values) if values else None

        return {
            "transitions": self.transitions,
            "errors": self.errors,
            "transition_exact_accuracy": 1.0 - self.errors / self.transitions,
            "overthinking_transitions": self.overthinking,
            "overthinking_fraction_of_errors": self.overthinking / self.errors
            if self.errors
            else 0.0,
            "late_repair_transitions": self.late_repairs,
            "final_visit_convergence_rate": self.final_visit_converged / self.transitions,
            "late_visit_start_index": late_start,
            "late_dynamics": {
                group: {
                    "output_flip_rate": late(group, "output_flip_rate"),
                    "state_delta_rms": late(group, "state_delta_rms"),
                    "logit_delta_rms": late(group, "logit_delta_rms"),
                }
                for group in ("success", "failure")
            },
            "visit_profiles": profiles,
        }


@torch.inference_mode()
def evaluate_transitions(
    model,
    exp300,
    sampler,
    cases: int,
    check: Callable[[], None],
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    seen = correct = valid = 0
    while seen < cases:
        size = min(128, cases - seen)
        state, multiplicand, modulus, input_bit, target, _ = sampler.transitions(size)
        logits, _ = model(exp300.features(state, multiplicand, modulus, input_bit, model.width))
        predicted = logits.ge(0)
        target_bits = exp300.bits(target, model.width).bool()
        correct += int((predicted == target_bits).all(1).sum())
        values = decode_bits(predicted)
        valid += sum(value < prime for value, prime in zip(values, modulus, strict=True))
        seen += size
        check()
    return {
        "cases": seen,
        "correct": correct,
        "exact_accuracy": correct / seen,
        "valid_residue_rate": valid / seen,
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "aggregate_cpu_seconds": process_cpu_seconds() - cpu_started,
        },
    }


@torch.inference_mode()
def evaluate_rollouts(
    model,
    exp300,
    sampler,
    cases: int,
    batch_size: int,
    check: Callable[[], None],
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    records: list[dict[str, Any]] = []
    while len(records) < cases:
        size = min(batch_size, cases - len(records))
        multiplier, multiplicand, modulus = sampler.pairs(size)
        state = [0] * size
        teacher = [0] * size
        first_error: list[int | None] = [None] * size
        invalid = [False] * size
        for position in range(model.width):
            input_bit = [
                (value >> (model.width - 1 - position)) & 1 for value in multiplier
            ]
            logits, _ = model(
                exp300.features(state, multiplicand, modulus, input_bit, model.width)
            )
            state = decode_bits(logits.ge(0))
            teacher = [
                (2 * value + bit * other) % prime
                for value, bit, other, prime in zip(
                    teacher, input_bit, multiplicand, modulus, strict=True
                )
            ]
            for index, (prediction, expected, prime) in enumerate(
                zip(state, teacher, modulus, strict=True)
            ):
                if first_error[index] is None and prediction != expected:
                    first_error[index] = position
                invalid[index] = invalid[index] or prediction >= prime
            check()
        targets = [
            left * right % prime
            for left, right, prime in zip(multiplier, multiplicand, modulus, strict=True)
        ]
        records.extend(
            {
                "multiplier": multiplier[index],
                "multiplicand": multiplicand[index],
                "modulus": modulus[index],
                "prediction": state[index],
                "target": targets[index],
                "exact": state[index] == targets[index],
                "all_steps_exact": first_error[index] is None,
                "first_error_position": first_error[index],
                "any_invalid_residue": invalid[index],
            }
            for index in range(size)
        )
    correct = sum(record["exact"] for record in records)
    all_steps = sum(record["all_steps_exact"] for record in records)
    invalid_count = sum(record["any_invalid_residue"] for record in records)
    positions = [
        record["first_error_position"]
        for record in records
        if record["first_error_position"] is not None
    ]
    return {
        "cases": cases,
        "correct": correct,
        "exact_accuracy": correct / cases,
        "all_steps_exact_accuracy": all_steps / cases,
        "any_invalid_residue_rate": invalid_count / cases,
        "first_error": {
            "error_cases": len(positions),
            "mean_position": sum(positions) / len(positions) if positions else None,
            "minimum_position": min(positions) if positions else None,
            "maximum_position": max(positions) if positions else None,
            "histogram": {
                str(position): positions.count(position) for position in sorted(set(positions))
            },
        },
        "case_records": records,
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "aggregate_cpu_seconds": process_cpu_seconds() - cpu_started,
        },
    }


@torch.inference_mode()
def evaluate_instrumented_rollouts(
    model,
    exp300,
    sampler,
    cases: int,
    check: Callable[[], None],
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    multiplier, multiplicand, modulus = sampler.pairs(cases)
    state = [0] * cases
    teacher = [0] * cases
    first_error: list[int | None] = [None] * cases
    invalid = [False] * cases
    accumulator = VisitAccumulator(model.dilation_schedule(model.width))
    parity = True
    maximum_logit_difference = 0.0
    for position in range(model.width):
        input_bit = [
            (value >> (model.width - 1 - position)) & 1 for value in multiplier
        ]
        next_teacher = [
            (2 * value + bit * other) % prime
            for value, bit, other, prime in zip(
                teacher, input_bit, multiplicand, modulus, strict=True
            )
        ]
        feature_tensor = exp300.features(
            state, multiplicand, modulus, input_bit, model.width
        )
        ordinary, _ = model(feature_tensor)
        instrumented, visits = instrumented_forward(
            model, feature_tensor, exp300.bits(next_teacher, model.width).bool()
        )
        difference = float((ordinary - instrumented).abs().max())
        maximum_logit_difference = max(maximum_logit_difference, difference)
        parity = parity and torch.equal(ordinary.ge(0), instrumented.ge(0))
        accumulator.add(visits)
        state = decode_bits(instrumented.ge(0))
        teacher = next_teacher
        for index, (prediction, expected, prime) in enumerate(
            zip(state, teacher, modulus, strict=True)
        ):
            if first_error[index] is None and prediction != expected:
                first_error[index] = position
            invalid[index] = invalid[index] or prediction >= prime
        check()
    targets = [
        left * right % prime
        for left, right, prime in zip(multiplier, multiplicand, modulus, strict=True)
    ]
    records = [
        {
            "multiplier": multiplier[index],
            "multiplicand": multiplicand[index],
            "modulus": modulus[index],
            "prediction": state[index],
            "target": targets[index],
            "exact": state[index] == targets[index],
            "first_error_position": first_error[index],
            "any_invalid_residue": invalid[index],
        }
        for index in range(cases)
    ]
    return {
        "cases": cases,
        "exact_accuracy": sum(record["exact"] for record in records) / cases,
        "any_invalid_residue_rate": sum(record["any_invalid_residue"] for record in records)
        / cases,
        "ordinary_threshold_parity": parity,
        "maximum_ordinary_logit_difference": maximum_logit_difference,
        "transition_dynamics": accumulator.summary(),
        "case_records": records,
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "aggregate_cpu_seconds": process_cpu_seconds() - cpu_started,
        },
    }


def evaluate_spectral(
    model,
    exp300,
    sampler,
    transitions: int,
    directions: int,
    fractions: list[float],
    seed: int,
    check: Callable[[], None],
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    cases = []
    for index in range(transitions):
        state, multiplicand, modulus, input_bit, _target, _ = sampler.transitions(1)
        feature_tensor = exp300.features(
            state, multiplicand, modulus, input_bit, model.width
        )
        cases.append(
            {
                "case_index": index,
                "visits": jacobian_gain_proxy(
                    model,
                    feature_tensor,
                    visit_fractions=fractions,
                    directions=directions,
                    seed=seed + index,
                ),
            }
        )
        check()
    return {
        "transitions": transitions,
        "directions_per_visit": directions,
        "interpretation": "random-direction lower-bound proxy, not spectral norm",
        "cases": cases,
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "aggregate_cpu_seconds": process_cpu_seconds() - cpu_started,
        },
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        # Keep canonical JSON finite; the caller handles a positive-over-zero gate.
        return None
    return numerator / denominator


def _late_spectral_max(cell: dict[str, Any]) -> float:
    values = [
        visit["maximum_random_direction_gain"]
        for case in cell["spectral"]["cases"]
        for visit in case["visits"]
        if visit["visit_index"] >= cell["schedule_visits"] // 2
    ]
    return max(values)


def diagnose(cells: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["decision_rule"]
    seed_names = list(config["fresh_evidence"]["formal_seeds"])
    per_seed: dict[str, Any] = {}
    for seed_name in seed_names:
        control = cells[f"{seed_name}:128"]
        target = cells[f"{seed_name}:256"]
        control_dynamics = control["instrumented_rollout"]["transition_dynamics"]
        target_dynamics = target["instrumented_rollout"]["transition_dynamics"]
        control_overthink = control_dynamics["overthinking_fraction_of_errors"]
        target_overthink = target_dynamics["overthinking_fraction_of_errors"]
        overthink_ratio = _safe_ratio(target_overthink, control_overthink)
        target_late = target_dynamics["late_dynamics"]
        flip_ratio = _safe_ratio(
            target_late["failure"]["output_flip_rate"],
            target_late["success"]["output_flip_rate"],
        )
        delta_ratio = _safe_ratio(
            target_late["failure"]["state_delta_rms"],
            target_late["success"]["state_delta_rms"],
        )
        control_spectral = _late_spectral_max(control)
        target_spectral = _late_spectral_max(target)
        spectral_ratio = _safe_ratio(target_spectral, control_spectral)
        retention = (
            control["transition"]["exact_accuracy"]
            >= float(gates["width128_minimum_transition_accuracy"])
            and control["rollout"]["exact_accuracy"]
            >= float(gates["width128_minimum_rollout_accuracy"])
        )
        overthinking_ratio_pass = (
            (control_overthink == 0.0 and target_overthink > 0.0)
            or (
                overthink_ratio is not None
                and overthink_ratio >= float(gates["overthinking_minimum_width_ratio"])
            )
        )
        overthinking_axis = (
            target_overthink >= float(gates["overthinking_minimum_error_fraction"])
            and overthinking_ratio_pass
        )
        dynamics_axis = any(
            ratio is not None
            and ratio >= float(gates["failure_success_late_dynamics_ratio"])
            for ratio in (flip_ratio, delta_ratio)
        )
        spectral_axis = (
            target_spectral >= float(gates["spectral_minimum_late_gain"])
            and spectral_ratio is not None
            and spectral_ratio >= float(gates["spectral_minimum_width_ratio"])
        )
        per_seed[seed_name] = {
            "width128_retention": retention,
            "width128_overthinking_fraction_of_errors": control_overthink,
            "width256_overthinking_fraction_of_errors": target_overthink,
            "overthinking_width_ratio": overthink_ratio,
            "overthinking_ratio_pass": overthinking_ratio_pass,
            "failure_success_late_flip_ratio": flip_ratio,
            "failure_success_late_state_delta_ratio": delta_ratio,
            "width128_late_spectral_gain": control_spectral,
            "width256_late_spectral_gain": target_spectral,
            "spectral_width_ratio": spectral_ratio,
            "overthinking_axis": overthinking_axis,
            "failure_dynamics_axis": dynamics_axis,
            "spectral_corroboration_axis": spectral_axis,
        }
    retention_reproduced = all(item["width128_retention"] for item in per_seed.values())
    overthinking_reproduced = all(item["overthinking_axis"] for item in per_seed.values())
    dynamics_reproduced = all(item["failure_dynamics_axis"] for item in per_seed.values())
    spectral_reproduced = all(
        item["spectral_corroboration_axis"] for item in per_seed.values()
    )
    supported = retention_reproduced and (
        overthinking_reproduced or (dynamics_reproduced and spectral_reproduced)
    )
    return {
        "per_seed": per_seed,
        "width128_retention_reproduced": retention_reproduced,
        "overthinking_axis_reproduced": overthinking_reproduced,
        "failure_dynamics_axis_reproduced": dynamics_reproduced,
        "spectral_corroboration_reproduced": spectral_reproduced,
        "stability_primary_cause_supported": supported,
        "decision": "advance-one-matched-stability-intervention"
        if supported
        else "reject-stability-as-primary-cause-and-change-architecture-track",
    }


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.exp300 = load_exp300()
        self.cellular = load_module(
            "cellular_for_exp643",
            ROOT / "experiments/exp_220_width_independent_cellular/model.py",
        )
        self.source_path = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source_path) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")
        torch.set_num_threads(int(self.config["evaluation"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(self.config["seed"]))
        self.started = time.perf_counter()
        self.cpu_started = process_cpu_seconds()

    def source(self) -> dict[str, Any]:
        preregistration = ROOT / self.config["preregistration"]
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
            "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(preregistration),
            "cell_model_sha256": sha256(
                ROOT / "experiments/exp_220_width_independent_cellular/model.py"
            ),
            "generator_runner_sha256": sha256(
                ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"
            ),
            "source_checkpoint_sha256": sha256(self.source_path),
        }

    def check_budget(self) -> None:
        budget = self.config["resource_budget"]
        prefix = "calibration_" if self.mode == "calibration" else ""
        wall_ceiling = budget[f"{prefix}hard_runaway_ceiling_wall_seconds"]
        cpu_ceiling = budget[f"{prefix}hard_runaway_ceiling_aggregate_cpu_seconds"]
        if wall_ceiling is None or cpu_ceiling is None:
            raise RuntimeError(f"{self.mode} resource budget is not resolved")
        if time.perf_counter() - self.started > float(wall_ceiling):
            raise TimeoutError(f"EXP-643 {self.mode} wall runaway ceiling reached")
        if process_cpu_seconds() - self.cpu_started > float(cpu_ceiling):
            raise TimeoutError(f"EXP-643 {self.mode} aggregate CPU runaway ceiling reached")

    def model(self, width: int):
        model = self.cellular.SharedDilatedConvGRU(
            width=width,
            hidden=int(self.config["model"]["hidden"]),
            cycles=int(self.config["model"]["cycles"]),
        )
        model.load_state_dict(load_file(self.source_path))
        return model.eval()

    def run_cell(self, seed_name: str, seed: int, width: int) -> dict[str, Any]:
        config = self.config
        fresh = config["fresh_evidence"]
        counts = config["calibration"] if self.mode == "calibration" else config["evaluation"]
        suffix = "_per_width" if self.mode == "calibration" else "_per_width_seed"
        transition_cases = int(counts[f"transition_cases{suffix}"])
        rollout_cases = int(counts[f"rollout_cases{suffix}"])
        instrumented_cases = int(counts[f"instrumented_rollout_cases{suffix}"])
        spectral_transitions = int(counts[f"spectral_transitions{suffix}"])
        directions = int(counts["spectral_directions_per_visit"])
        model = self.model(width)
        split = make_split(config, self.exp300, width, seed)
        split_hash = sha256_text(json.dumps(split, sort_keys=True))
        check = self.check_budget
        transition = evaluate_transitions(
            model,
            self.exp300,
            self.exp300.BigIntSampler(
                split["test"], seed + int(fresh["transition_seed_offset"]) + width
            ),
            transition_cases,
            check,
        )
        rollout = evaluate_rollouts(
            model,
            self.exp300,
            self.exp300.BigIntSampler(
                split["test"], seed + int(fresh["rollout_seed_offset"]) + width
            ),
            rollout_cases,
            int(config["evaluation"]["rollout_batch_size"][width]),
            check,
        )
        instrumented = evaluate_instrumented_rollouts(
            model,
            self.exp300,
            self.exp300.BigIntSampler(
                split["test"], seed + int(fresh["diagnostic_seed_offset"]) + width
            ),
            instrumented_cases,
            check,
        )
        spectral = evaluate_spectral(
            model,
            self.exp300,
            self.exp300.BigIntSampler(
                split["test"], seed + int(fresh["spectral_seed_offset"]) + width
            ),
            spectral_transitions,
            directions,
            [float(value) for value in config["evaluation"]["spectral_visit_fractions"]],
            seed + int(fresh["spectral_seed_offset"]) + width,
            check,
        )
        return {
            "seed_name": seed_name,
            "seed": seed,
            "width": width,
            "schedule": model.dilation_schedule(width),
            "schedule_visits": len(model.dilation_schedule(width)),
            "split_sha256": split_hash,
            "split_identity_counts": {
                "train": len(split["train"]),
                "test": len(split["test"]),
            },
            "transition": transition,
            "rollout": rollout,
            "instrumented_rollout": instrumented,
            "spectral": spectral,
            "peak_rss_bytes_after_cell": peak_rss_bytes(),
        }

    def calibration(self) -> dict[str, Any]:
        seed = int(self.config["fresh_evidence"]["calibration_seed"])
        cells: dict[str, Any] = {}
        for width in self.config["fresh_evidence"]["widths"]:
            width = int(width)
            cells[str(width)] = self.run_cell("calibration", seed, width)
        calibration = self.config["calibration"]
        evaluation = self.config["evaluation"]
        formal_seed_count = len(self.config["fresh_evidence"]["formal_seeds"])
        projected_wall = 0.0
        projected_cpu = 0.0
        component_projection: dict[str, Any] = {}
        component_keys = {
            "transition": (
                "transition_cases_per_width_seed",
                "transition_cases_per_width",
            ),
            "rollout": ("rollout_cases_per_width_seed", "rollout_cases_per_width"),
            "instrumented_rollout": (
                "instrumented_rollout_cases_per_width_seed",
                "instrumented_rollout_cases_per_width",
            ),
            "spectral": (
                "spectral_transitions_per_width_seed",
                "spectral_transitions_per_width",
            ),
        }
        for width, cell in cells.items():
            component_projection[width] = {}
            for component, (formal_key, calibration_key) in component_keys.items():
                scale = float(evaluation[formal_key]) / float(calibration[calibration_key])
                if component == "spectral":
                    scale *= float(evaluation["spectral_directions_per_visit"]) / float(
                        calibration["spectral_directions_per_visit"]
                    )
                scale *= formal_seed_count
                wall = float(cell[component]["timing"]["wall_seconds"]) * scale
                cpu = float(cell[component]["timing"]["aggregate_cpu_seconds"]) * scale
                projected_wall += wall
                projected_cpu += cpu
                component_projection[width][component] = {
                    "scale_to_formal": scale,
                    "projected_wall_seconds": wall,
                    "projected_aggregate_cpu_seconds": cpu,
                }
        multiplier = float(self.config["resource_budget"]["throughput_safety_multiplier"])
        setup_wall = max(60.0, projected_wall * 0.05)
        setup_cpu = max(30.0, projected_cpu * 0.02)
        soft_wall = math.ceil(projected_wall * multiplier + setup_wall)
        soft_cpu = math.ceil(projected_cpu * multiplier + setup_cpu)
        hard_wall = math.ceil(max(soft_wall * 1.5, soft_wall + 600.0))
        hard_cpu = math.ceil(max(soft_cpu * 1.5, soft_cpu + 1800.0))
        result = {
            "run_id": self.config["run_id"],
            "mode": "calibration",
            "status": "completed",
            "calibration_seed": seed,
            "source": self.source(),
            "cells": cells,
            "component_projection": component_projection,
            "measured_total": {
                "wall_seconds": time.perf_counter() - self.started,
                "aggregate_cpu_seconds": process_cpu_seconds() - self.cpu_started,
                "peak_rss_bytes": peak_rss_bytes(),
            },
            "projected_formal_without_headroom": {
                "wall_seconds": projected_wall,
                "aggregate_cpu_seconds": projected_cpu,
            },
            "recommended_resource_budget": {
                "projected_formal_wall_seconds": math.ceil(projected_wall),
                "projected_formal_aggregate_cpu_seconds": math.ceil(projected_cpu),
                "setup_and_serialisation_headroom_wall_seconds": math.ceil(setup_wall),
                "setup_and_serialisation_headroom_aggregate_cpu_seconds": math.ceil(
                    setup_cpu
                ),
                "throughput_safety_multiplier": multiplier,
                "soft_budget_wall_seconds": soft_wall,
                "soft_budget_aggregate_cpu_seconds": soft_cpu,
                "hard_runaway_ceiling_wall_seconds": hard_wall,
                "hard_runaway_ceiling_aggregate_cpu_seconds": hard_cpu,
            },
            "limitations": [
                "Calibration stream is excluded from formal diagnosis.",
                "Projection scales measured representative components; formal ceilings retain 50 percent throughput headroom plus setup allowance.",
                "Aggregate CPU is process user plus system time and is distinct from wall time.",
            ],
        }
        path = self.output / "calibration.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result

    def formal(self) -> dict[str, Any]:
        budget = self.config["resource_budget"]
        if budget["calibration_status"] != "complete":
            raise RuntimeError("formal mode requires a committed complete calibration budget")
        source = self.source()
        if source["dirty"]:
            raise RuntimeError("formal mode requires a clean tracked worktree")
        cache_key = sha256_text(
            json.dumps(
                {
                    "source": source,
                    "fresh_evidence": self.config["fresh_evidence"],
                    "evaluation": self.config["evaluation"],
                    "decision_rule": self.config["decision_rule"],
                },
                sort_keys=True,
            )
        )
        final_path = self.output / "summary.json"
        if budget["cache_completed_evidence"] and final_path.exists():
            cached = json.loads(final_path.read_text())
            if cached.get("cache_key") == cache_key and cached.get("status") == "completed":
                return cached
        partial_path = self.output / "partial.json"
        cells: dict[str, Any] = {}
        if budget["preserve_partial_evidence"] and partial_path.exists():
            partial = json.loads(partial_path.read_text())
            if partial.get("cache_key") == cache_key:
                cells = partial.get("cells", {})
        for seed_name, seed in self.config["fresh_evidence"]["formal_seeds"].items():
            for width_value in self.config["fresh_evidence"]["widths"]:
                width = int(width_value)
                key = f"{seed_name}:{width}"
                if key not in cells:
                    cells[key] = self.run_cell(seed_name, int(seed), width)
                    partial_path.write_text(
                        json.dumps(
                            {
                                "run_id": self.config["run_id"],
                                "cache_key": cache_key,
                                "status": "partial",
                                "cells": cells,
                            },
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        diagnosis = diagnose(cells, self.config)
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=self.config["run_id"]) as run:
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "completed",
                    "decision": diagnosis["decision"],
                    "role": "frozen-recurrent-depth-stability-diagnostic",
                    "repository_commit": source["repository_commit"],
                    "repository_dirty": "false",
                }
            )
            mlflow.log_params(
                {
                    "seed": self.config["seed"],
                    "formal_seeds": json.dumps(
                        self.config["fresh_evidence"]["formal_seeds"], sort_keys=True
                    ),
                    "widths": json.dumps(self.config["fresh_evidence"]["widths"]),
                    "checkpoint_sha256": source["source_checkpoint_sha256"],
                    "config_sha256": source["config_sha256"],
                    "runner_sha256": source["runner_sha256"],
                }
            )
            for key, cell in cells.items():
                metric_key = key.replace(":", "_")
                mlflow.log_metrics(
                    {
                        f"{metric_key}_transition_exact": cell["transition"][
                            "exact_accuracy"
                        ],
                        f"{metric_key}_rollout_exact": cell["rollout"]["exact_accuracy"],
                        f"{metric_key}_overthinking": cell["instrumented_rollout"][
                            "transition_dynamics"
                        ]["overthinking_fraction_of_errors"],
                        f"{metric_key}_late_spectral_gain": _late_spectral_max(cell),
                    }
                )
            result = {
                "run_id": self.config["run_id"],
                "cache_key": cache_key,
                "status": "completed",
                "mlflow": {
                    "tracking_uri": tracking_uri(),
                    "experiment_name": self.config["tracking"]["experiment_name"],
                    "experiment_id": experiment_id,
                    "run_id": run.info.run_id,
                },
                "source": source,
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "mlflow": mlflow.__version__,
                    "torch_threads": torch.get_num_threads(),
                },
                "cells": cells,
                "diagnosis": diagnosis,
                "resource": {
                    "wall_seconds": time.perf_counter() - self.started,
                    "aggregate_cpu_seconds": process_cpu_seconds() - self.cpu_started,
                    "peak_rss_bytes": peak_rss_bytes(),
                    "resolved_budget": budget,
                },
                "limitations": [
                    "Fresh generated evidence is not an official private score or a completion-gate seed.",
                    "The Jacobian statistic is a random-direction lower-bound gain proxy, not a spectral norm.",
                    "A supported association only authorizes a matched causal intervention.",
                    "Python exact arithmetic is confined to generation, teachers, and evaluation.",
                ],
            }
            final_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            mlflow.log_artifact(str(final_path), artifact_path="canonical-preview")
            mlflow.log_artifact(str(self.config_path), artifact_path="preregistration")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_643_tier7_recurrent_depth_stability_diagnostic.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-643-tier7-recurrent-depth-stability-diagnostic",
    )
    parser.add_argument("--mode", choices=("calibration", "formal"), required=True)
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode)
    result = experiment.calibration() if args.mode == "calibration" else experiment.formal()
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "mode": args.mode,
                "status": result["status"],
                "decision": result.get("diagnosis", {}).get("decision"),
                "resource": result.get("measured_total", result.get("resource")),
                "recommended_resource_budget": result.get("recommended_resource_budget"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
