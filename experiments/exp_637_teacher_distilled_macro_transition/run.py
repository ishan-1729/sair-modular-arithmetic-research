"""Calibrate, train, and evaluate EXP-637 at frozen fidelity."""

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
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


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
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def decode(logits: Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in logits.ge(0).tolist()
    ]


def masked_binary_distillation(
    student_logits: Tensor,
    teacher_logits: Tensor,
    exact_teacher_mask: Tensor,
    temperature: float,
) -> Tensor:
    """Temperature-scaled binary KL surrogate on teacher-correct cases only."""
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    if not bool(exact_teacher_mask.any()):
        return student_logits.sum() * 0.0
    teacher_probability = torch.sigmoid(teacher_logits.detach() / temperature)
    per_bit = F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        teacher_probability,
        reduction="none",
    )
    return per_bit[exact_teacher_mask].mean() * (temperature**2)


def paired_counts(first: list[int], second: list[int], targets: list[int]) -> dict:
    first_ok = [prediction == target for prediction, target in zip(first, targets)]
    second_ok = [prediction == target for prediction, target in zip(second, targets)]
    return {
        "first_only": sum(a and not b for a, b in zip(first_ok, second_ok)),
        "second_only": sum(b and not a for a, b in zip(first_ok, second_ok)),
        "intersection": sum(a and b for a, b in zip(first_ok, second_ok)),
        "neither": sum(not a and not b for a, b in zip(first_ok, second_ok)),
        "prediction_disagreement": sum(a != b for a, b in zip(first, second)),
    }


class Experiment:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = int(self.config["seed"])
        self.process = psutil.Process()
        self.exp300 = load_module(
            "exp300_for_exp637",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
        self.teacher_module = load_module(
            "teacher_for_exp637",
            ROOT / "experiments/exp_220_width_independent_cellular/model.py",
        )
        self.direct_module = load_module(
            "direct_for_exp637",
            ROOT / "experiments/exp_630_radix4_cellular_transition/model.py",
        )
        self.direct_data = load_module(
            "direct_data_for_exp637",
            ROOT / "experiments/exp_630_radix4_cellular_transition/data.py",
        )
        self.model_module = load_module(
            "model_for_exp637", Path(__file__).with_name("model.py")
        )
        self.data = load_module(
            "data_for_exp637", Path(__file__).with_name("data.py")
        )
        self.teacher_path = (ROOT / self.config["teacher_checkpoint"]).resolve()
        self.direct_path = (
            ROOT / self.config["direct_control_checkpoint"]
        ).resolve()
        if sha256(self.teacher_path) != self.config["teacher_checkpoint_sha256"]:
            raise RuntimeError("teacher checkpoint hash mismatch")
        if sha256(self.direct_path) != self.config["direct_control_checkpoint_sha256"]:
            raise RuntimeError("direct-control checkpoint hash mismatch")
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.train_widths = [int(v) for v in self.config["training"]["widths"]]
        self.eval_widths = [int(v) for v in self.config["evaluation"]["widths"]]
        self.all_widths = sorted(set(self.train_widths + self.eval_widths))
        self.splits = {width: self.make_split(width) for width in self.all_widths}
        self.split_hash = sha256_text(json.dumps(self.splits, sort_keys=True))
        self.teacher = self.make_teacher()
        self.direct = self.make_direct()

    def make_split(self, width: int) -> dict[str, list[int]]:
        if width == 8:
            train_count = self.config["data"]["width8_train_prime_identities"]
            test_count = self.config["data"]["width8_test_prime_identities"]
            prime_bits_min = 2
        else:
            train_count = self.config["data"]["train_prime_identities"]
            test_count = self.config["data"]["test_prime_identities"]
            prime_bits_min = width - self.config["data"]["prime_bit_span"] + 1
        local = {
            "data": {
                "prime_bits_min": prime_bits_min,
                "prime_bits_max": width,
                "train_prime_identities": train_count,
                "test_prime_identities": test_count,
            }
        }
        # Deliberately identical to EXP-630 for a matched cached-checkpoint contrast.
        return self.exp300.bigint_prime_split(local, self.seed + 63000 + width)

    def make_teacher(self):
        model = self.teacher_module.SharedDilatedConvGRU(
            width=max(self.all_widths),
            hidden=int(self.config["model"]["hidden"]),
            cycles=int(self.config["model"]["teacher_cycles"]),
        )
        model.load_state_dict(load_file(self.teacher_path))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def make_direct(self):
        model = self.direct_module.Radix4SharedDilatedConvGRU(
            width=max(self.all_widths),
            hidden=int(self.config["model"]["hidden"]),
            cycles=int(self.config["model"]["teacher_cycles"]),
        )
        model.load_state_dict(load_file(self.direct_path))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def make_student(self):
        model = self.model_module.TeacherDistilledMacroTransition(
            hidden=int(self.config["model"]["hidden"]),
            phase1_reverse_dilations=bool(
                self.config["model"]["phase1_reverse_dilations"]
            ),
            phase2_reverse_dilations=bool(
                self.config["model"]["phase2_reverse_dilations"]
            ),
            intermediate_temperature=float(
                self.config["model"]["intermediate_temperature"]
            ),
        )
        report = self.model_module.transfer_teacher_state(
            model, self.teacher.state_dict()
        )
        return model, report

    def teacher_logits(
        self,
        states: list[int],
        rights: list[int],
        moduli: list[int],
        input_bits: list[int],
        width: int,
    ) -> tuple[Tensor, Tensor]:
        self.teacher.width = width
        with torch.inference_mode():
            return self.teacher(
                self.data.features(states, rights, moduli, input_bits, width)
            )

    def training_objective(
        self,
        student,
        batch: tuple,
        width: int,
    ) -> tuple[Tensor, dict[str, float]]:
        (
            states,
            rights,
            moduli,
            digits,
            intermediates,
            finals,
            q1,
            q2,
        ) = batch
        phase1_features, phase2_context = self.data.macro_features(
            states, rights, moduli, digits, width
        )
        high_bits = [digit >> 1 for digit in digits]
        low_bits = [digit & 1 for digit in digits]
        teacher1, _teacher_q1 = self.teacher_logits(
            states, rights, moduli, high_bits, width
        )
        teacher2, _teacher_q2 = self.teacher_logits(
            intermediates, rights, moduli, low_bits, width
        )
        final_logits, intermediate_logits, q1_logits, q2_logits = student(
            phase1_features, phase2_context
        )
        teacher_forced_logits, teacher_forced_q2_logits = student.forward_phase2(
            self.data.features(intermediates, rights, moduli, low_bits, width)
        )
        final_loss, final_stats = self.exp300.transition_loss(
            final_logits,
            q2_logits,
            finals,
            q2,
            width,
            float(self.config["training"]["worst_bit_weight"]),
            float(self.config["training"]["quotient_weight"]),
        )
        intermediate_loss, intermediate_stats = self.exp300.transition_loss(
            intermediate_logits,
            q1_logits,
            intermediates,
            q1,
            width,
            float(self.config["training"]["worst_bit_weight"]),
            float(self.config["training"]["quotient_weight"]),
        )
        teacher_forced_loss, teacher_forced_stats = self.exp300.transition_loss(
            teacher_forced_logits,
            teacher_forced_q2_logits,
            finals,
            q2,
            width,
            float(self.config["training"]["worst_bit_weight"]),
            float(self.config["training"]["quotient_weight"]),
        )
        intermediate_targets = self.data.bits(intermediates, width).bool()
        final_targets = self.data.bits(finals, width).bool()
        teacher1_correct = teacher1.ge(0).eq(intermediate_targets).all(1)
        teacher2_correct = teacher2.ge(0).eq(final_targets).all(1)
        temperature = float(
            self.config["training"]["teacher_distillation_temperature"]
        )
        distill1 = masked_binary_distillation(
            intermediate_logits, teacher1, teacher1_correct, temperature
        )
        distill2_autonomous = masked_binary_distillation(
            final_logits, teacher2, teacher2_correct, temperature
        )
        distill2_teacher_forced = masked_binary_distillation(
            teacher_forced_logits, teacher2, teacher2_correct, temperature
        )
        distillation = distill1 + 0.5 * (
            distill2_autonomous + distill2_teacher_forced
        )
        loss = (
            final_loss
            + float(self.config["training"]["intermediate_hard_weight"])
            * intermediate_loss
            + float(self.config["training"]["teacher_forced_final_weight"])
            * teacher_forced_loss
            + float(self.config["training"]["teacher_distillation_weight"])
            * distillation
        )
        return loss, {
            "loss": float(loss.detach()),
            "final_batch_exact": final_stats["batch_exact"],
            "intermediate_batch_exact": intermediate_stats["batch_exact"],
            "teacher_forced_batch_exact": teacher_forced_stats["batch_exact"],
            "final_loss": final_stats["loss"],
            "intermediate_loss": intermediate_stats["loss"],
            "teacher_forced_loss": teacher_forced_stats["loss"],
            "distillation_loss": float(distillation.detach()),
            "teacher1_exact": float(teacher1_correct.float().mean()),
            "teacher2_exact": float(teacher2_correct.float().mean()),
        }

    def transition_student(
        self, model, batch: tuple, width: int, keep_logits: bool = False
    ) -> dict[str, Any]:
        states, rights, moduli, digits, intermediates, finals, _q1, _q2 = batch
        correct = intermediate_correct = valid = intermediate_valid = 0
        predictions: list[int] = []
        intermediate_predictions: list[int] = []
        logit_records: list[list[list[float]]] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        with torch.inference_mode():
            for start in range(0, len(states), 512):
                stop = min(start + 512, len(states))
                first, second = self.data.macro_features(
                    states[start:stop],
                    rights[start:stop],
                    moduli[start:stop],
                    digits[start:stop],
                    width,
                )
                final_logits, intermediate_logits, _q1l, _q2l = model(first, second)
                final_values = decode(final_logits)
                intermediate_values = decode(intermediate_logits)
                predictions.extend(final_values)
                intermediate_predictions.extend(intermediate_values)
                correct += sum(
                    value == target
                    for value, target in zip(final_values, finals[start:stop])
                )
                intermediate_correct += sum(
                    value == target
                    for value, target in zip(
                        intermediate_values, intermediates[start:stop]
                    )
                )
                valid += sum(
                    value < modulus
                    for value, modulus in zip(final_values, moduli[start:stop])
                )
                intermediate_valid += sum(
                    value < modulus
                    for value, modulus in zip(
                        intermediate_values, moduli[start:stop]
                    )
                )
                if keep_logits:
                    logit_records.append(
                        [intermediate_logits.tolist(), final_logits.tolist()]
                    )
        cases = len(states)
        result: dict[str, Any] = {
            "cases": cases,
            "correct": correct,
            "exact_accuracy": correct / cases,
            "intermediate_correct": intermediate_correct,
            "intermediate_exact_accuracy": intermediate_correct / cases,
            "valid_residue_rate": valid / cases,
            "intermediate_valid_residue_rate": intermediate_valid / cases,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "predictions": predictions,
            "intermediate_predictions": intermediate_predictions,
        }
        if keep_logits:
            result["logits"] = logit_records
        return result

    def transition_direct(self, batch: tuple, width: int) -> dict[str, Any]:
        states, rights, moduli, digits, _intermediates, finals, _q1, _q2 = batch
        predictions: list[int] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        self.direct.width = width
        with torch.inference_mode():
            for start in range(0, len(states), 512):
                stop = min(start + 512, len(states))
                logits, _ = self.direct(
                    self.direct_data.features(
                        states[start:stop],
                        rights[start:stop],
                        moduli[start:stop],
                        digits[start:stop],
                        width,
                    )
                )
                predictions.extend(decode(logits))
        correct = sum(a == b for a, b in zip(predictions, finals))
        return {
            "cases": len(states),
            "correct": correct,
            "exact_accuracy": correct / len(states),
            "valid_residue_rate": sum(
                value < modulus for value, modulus in zip(predictions, moduli)
            )
            / len(states),
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "predictions": predictions,
        }

    def transition_teacher(self, batch: tuple, width: int) -> dict[str, Any]:
        states, rights, moduli, digits, _intermediates, finals, _q1, _q2 = batch
        predictions: list[int] = []
        intermediate_predictions: list[int] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        self.teacher.width = width
        with torch.inference_mode():
            for start in range(0, len(states), 512):
                stop = min(start + 512, len(states))
                high = [digit >> 1 for digit in digits[start:stop]]
                low = [digit & 1 for digit in digits[start:stop]]
                first_logits, _ = self.teacher(
                    self.data.features(
                        states[start:stop],
                        rights[start:stop],
                        moduli[start:stop],
                        high,
                        width,
                    )
                )
                first_values = decode(first_logits)
                second_logits, _ = self.teacher(
                    self.data.features(
                        first_values,
                        rights[start:stop],
                        moduli[start:stop],
                        low,
                        width,
                    )
                )
                intermediate_predictions.extend(first_values)
                predictions.extend(decode(second_logits))
        correct = sum(a == b for a, b in zip(predictions, finals))
        return {
            "cases": len(states),
            "correct": correct,
            "exact_accuracy": correct / len(states),
            "valid_residue_rate": sum(
                value < modulus for value, modulus in zip(predictions, moduli)
            )
            / len(states),
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "predictions": predictions,
            "intermediate_predictions": intermediate_predictions,
        }

    def rollout_student(
        self,
        model,
        left: list[int],
        right: list[int],
        moduli: list[int],
        width: int,
        keep_logits: bool = False,
    ) -> dict[str, Any]:
        states = [0] * len(left)
        exact_states = [0] * len(left)
        every_macro = [True] * len(left)
        every_intermediate = [True] * len(left)
        invalid = [False] * len(left)
        intermediate_invalid = [False] * len(left)
        logit_records: list[list[list[float]]] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        with torch.inference_mode():
            for position in range(width // 2):
                digits = [
                    (value >> (width - 2 * (position + 1))) & 3 for value in left
                ]
                first, second = self.data.macro_features(
                    states, right, moduli, digits, width
                )
                final_logits, intermediate_logits, _q1, _q2 = model(first, second)
                intermediate_values = decode(intermediate_logits)
                states = decode(final_logits)
                exact_intermediates: list[int] = []
                next_exact: list[int] = []
                for state, other, modulus, digit in zip(
                    exact_states, right, moduli, digits
                ):
                    intermediate, _q1v, final, _q2v = self.data.exact_macro_transition(
                        state, other, modulus, digit
                    )
                    exact_intermediates.append(intermediate)
                    next_exact.append(final)
                every_intermediate = [
                    prior and predicted == expected
                    for prior, predicted, expected in zip(
                        every_intermediate, intermediate_values, exact_intermediates
                    )
                ]
                every_macro = [
                    prior and predicted == expected
                    for prior, predicted, expected in zip(
                        every_macro, states, next_exact
                    )
                ]
                intermediate_invalid = [
                    prior or predicted >= modulus
                    for prior, predicted, modulus in zip(
                        intermediate_invalid, intermediate_values, moduli
                    )
                ]
                invalid = [
                    prior or predicted >= modulus
                    for prior, predicted, modulus in zip(invalid, states, moduli)
                ]
                exact_states = next_exact
                if keep_logits:
                    logit_records.append(
                        [intermediate_logits.tolist(), final_logits.tolist()]
                    )
        expected = [(a * b) % p for a, b, p in zip(left, right, moduli)]
        correct = sum(a == b for a, b in zip(states, expected))
        result: dict[str, Any] = {
            "predictions": states,
            "cases": len(left),
            "correct": correct,
            "exact_accuracy": correct / len(left),
            "all_macro_steps_exact_accuracy": sum(every_macro) / len(left),
            "all_intermediate_steps_exact_accuracy": sum(every_intermediate)
            / len(left),
            "any_invalid_residue_rate": sum(invalid) / len(left),
            "any_invalid_intermediate_rate": sum(intermediate_invalid) / len(left),
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "macro_steps": width // 2,
            "dilation_sweeps_per_macro": 2,
        }
        if keep_logits:
            result["logits"] = logit_records
        return result

    def rollout_direct(
        self,
        left: list[int],
        right: list[int],
        moduli: list[int],
        width: int,
    ) -> dict[str, Any]:
        states = [0] * len(left)
        exact_states = [0] * len(left)
        every = [True] * len(left)
        invalid = [False] * len(left)
        wall = time.perf_counter()
        cpu = time.process_time()
        self.direct.width = width
        with torch.inference_mode():
            for position in range(width // 2):
                digits = [
                    (value >> (width - 2 * (position + 1))) & 3 for value in left
                ]
                logits, _ = self.direct(
                    self.direct_data.features(states, right, moduli, digits, width)
                )
                states = decode(logits)
                exact_states = [
                    self.data.exact_macro_transition(state, other, modulus, digit)[2]
                    for state, other, modulus, digit in zip(
                        exact_states, right, moduli, digits
                    )
                ]
                every = [
                    prior and predicted == expected
                    for prior, predicted, expected in zip(every, states, exact_states)
                ]
                invalid = [
                    prior or predicted >= modulus
                    for prior, predicted, modulus in zip(invalid, states, moduli)
                ]
        expected = [(a * b) % p for a, b, p in zip(left, right, moduli)]
        correct = sum(a == b for a, b in zip(states, expected))
        return {
            "predictions": states,
            "cases": len(left),
            "correct": correct,
            "exact_accuracy": correct / len(left),
            "all_macro_steps_exact_accuracy": sum(every) / len(left),
            "any_invalid_residue_rate": sum(invalid) / len(left),
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "macro_steps": width // 2,
        }

    def rollout_teacher(
        self,
        left: list[int],
        right: list[int],
        moduli: list[int],
        width: int,
    ) -> dict[str, Any]:
        states = [0] * len(left)
        exact_states = [0] * len(left)
        every = [True] * len(left)
        invalid = [False] * len(left)
        wall = time.perf_counter()
        cpu = time.process_time()
        self.teacher.width = width
        with torch.inference_mode():
            for position in range(width):
                input_bits = [
                    (value >> (width - 1 - position)) & 1 for value in left
                ]
                logits, _ = self.teacher(
                    self.data.features(states, right, moduli, input_bits, width)
                )
                states = decode(logits)
                exact_states = [
                    self.data.exact_radix2_transition(
                        state, other, modulus, input_bit
                    )[0]
                    for state, other, modulus, input_bit in zip(
                        exact_states, right, moduli, input_bits
                    )
                ]
                every = [
                    prior and predicted == expected
                    for prior, predicted, expected in zip(every, states, exact_states)
                ]
                invalid = [
                    prior or predicted >= modulus
                    for prior, predicted, modulus in zip(invalid, states, moduli)
                ]
        expected = [(a * b) % p for a, b, p in zip(left, right, moduli)]
        correct = sum(a == b for a, b in zip(states, expected))
        return {
            "predictions": states,
            "cases": len(left),
            "correct": correct,
            "exact_accuracy": correct / len(left),
            "all_steps_exact_accuracy": sum(every) / len(left),
            "any_invalid_residue_rate": sum(invalid) / len(left),
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "recurrent_steps": width,
            "dilation_sweeps_per_step": 2,
        }

    @staticmethod
    def strip_predictions(result: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in result.items()
            if key not in {"predictions", "intermediate_predictions", "logits"}
        }

    def warm_rollout_arms(
        self, student, initialized, pairs: tuple[list[int], list[int], list[int]], width: int
    ) -> None:
        warm = tuple(values[: min(2, len(values))] for values in pairs)
        self.rollout_teacher(*warm, width)
        self.rollout_direct(*warm, width)
        self.rollout_student(student, *warm, width)
        self.rollout_student(initialized, *warm, width)

    def calibrate(self) -> dict[str, Any]:
        student, _ = self.make_student()
        student.train()
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        update_profiles: dict[str, dict[str, float]] = {}
        profile_updates = int(
            self.config["resource_budget"]["profile_updates_per_width"]
        )
        profile_batch = int(self.config["resource_budget"]["profile_batch_size"])
        for width in self.config["resource_budget"]["profile_widths"]:
            width = int(width)
            sampler = self.data.MacroSampler(
                self.splits[width]["train"], self.seed + 637100 + width
            )
            wall = time.perf_counter()
            cpu = time.process_time()
            for _ in range(profile_updates):
                loss, _stats = self.training_objective(
                    student, sampler.transitions(profile_batch), width
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    student.parameters(),
                    float(self.config["training"]["gradient_clip"]),
                )
                optimizer.step()
            update_profiles[str(width)] = {
                "updates": profile_updates,
                "wall_seconds": time.perf_counter() - wall,
                "aggregate_cpu_seconds": time.process_time() - cpu,
            }
        student.eval()
        initialized, _ = self.make_student()
        initialized.eval()
        evaluation_profiles: dict[str, Any] = {}
        transition_cases = int(
            self.config["resource_budget"]["profile_transition_cases_per_width"]
        )
        rollout_cases = int(
            self.config["resource_budget"]["profile_rollout_cases_per_width"]
        )
        projected_eval_wall = 0.0
        projected_eval_cpu = 0.0
        for width in self.eval_widths:
            transition_batch = self.data.MacroSampler(
                self.splits[width]["train"], self.seed + 637200 + width
            ).transitions(transition_cases)
            transition_results = {
                "student": self.transition_student(student, transition_batch, width),
                "initialized": self.transition_student(
                    initialized, transition_batch, width
                ),
                "direct": self.transition_direct(transition_batch, width),
                "radix2": self.transition_teacher(transition_batch, width),
            }
            pairs = self.data.MacroSampler(
                self.splits[width]["train"], self.seed + 637300 + width
            ).pairs(rollout_cases)
            self.warm_rollout_arms(student, initialized, pairs, width)
            rollout_results = {
                "student": self.rollout_student(student, *pairs, width),
                "initialized": self.rollout_student(initialized, *pairs, width),
                "direct": self.rollout_direct(*pairs, width),
                "radix2": self.rollout_teacher(*pairs, width),
            }
            evaluation_profiles[str(width)] = {
                "transition": {
                    key: self.strip_predictions(value)
                    for key, value in transition_results.items()
                },
                "rollout": {
                    key: self.strip_predictions(value)
                    for key, value in rollout_results.items()
                },
            }
            for arm, value in transition_results.items():
                target_cases = (
                    int(
                        self.config["evaluation"][
                            "initialized_control_transition_cases"
                        ]
                    )
                    if arm == "initialized"
                    else int(self.config["evaluation"]["transition_cases"])
                )
                projected_eval_wall += value["wall_seconds"] / transition_cases * target_cases
                projected_eval_cpu += (
                    value["aggregate_cpu_seconds"] / transition_cases * target_cases
                )
            for arm, value in rollout_results.items():
                target_cases = (
                    int(self.config["evaluation"]["initialized_control_rollout_cases"])
                    if arm == "initialized"
                    else int(self.config["evaluation"]["rollout_cases"])
                )
                projected_eval_wall += value["wall_seconds"] / rollout_cases * target_cases
                projected_eval_cpu += (
                    value["aggregate_cpu_seconds"] / rollout_cases * target_cases
                )
        determinism_cases = int(self.config["evaluation"]["determinism_cases"])
        width64_student = evaluation_profiles["64"]["rollout"]["student"]
        projected_eval_wall += (
            2
            * width64_student["wall_seconds"]
            / rollout_cases
            * determinism_cases
        )
        projected_eval_cpu += (
            2
            * width64_student["aggregate_cpu_seconds"]
            / rollout_cases
            * determinism_cases
        )
        average_update_wall = sum(
            value["wall_seconds"] / value["updates"]
            for value in update_profiles.values()
        ) / len(update_profiles)
        average_update_cpu = sum(
            value["aggregate_cpu_seconds"] / value["updates"]
            for value in update_profiles.values()
        ) / len(update_profiles)
        projected_training_wall = average_update_wall * int(
            self.config["training"]["steps"]
        )
        projected_training_cpu = average_update_cpu * int(
            self.config["training"]["steps"]
        )
        setup = float(
            self.config["resource_budget"]["setup_tracking_artifact_wall_seconds"]
        )
        expected_wall = math.ceil(projected_training_wall + projected_eval_wall + setup)
        expected_cpu_minutes = math.ceil(
            (projected_training_cpu + projected_eval_cpu + setup * 2.0) / 60.0
        )
        recommendation = {
            "projected_training_wall_seconds": projected_training_wall,
            "projected_training_aggregate_cpu_seconds": projected_training_cpu,
            "projected_evaluation_wall_seconds": projected_eval_wall,
            "projected_evaluation_aggregate_cpu_seconds": projected_eval_cpu,
            "expected_total_wall_seconds": expected_wall,
            "expected_total_aggregate_cpu_minutes": expected_cpu_minutes,
            "soft_budget_wall_seconds": math.ceil(expected_wall * 1.35 / 60.0) * 60,
            "hard_runaway_ceiling_wall_seconds": math.ceil(
                expected_wall * 2.0 / 60.0
            )
            * 60,
            "hard_ceiling_aggregate_cpu_minutes_upper_bound": math.ceil(
                (math.ceil(expected_wall * 2.0 / 60.0) * 60 * 4) / 60.0
            ),
        }
        result = {
            "run_id": self.config["run_id"],
            "mode": "training-identity-only-calibration",
            "seed": self.seed,
            "source_hashes": {
                "teacher": sha256(self.teacher_path),
                "direct_control": sha256(self.direct_path),
                "config": sha256(self.config_path),
                "runner": sha256(Path(__file__)),
                "model": sha256(Path(__file__).with_name("model.py")),
                "data": sha256(Path(__file__).with_name("data.py")),
                "split": self.split_hash,
            },
            "update_profiles": update_profiles,
            "evaluation_profiles": evaluation_profiles,
            "recommendation": recommendation,
            "fidelity_fields_changed": False,
            "heldout_identities_read": False,
            "peak_rss_bytes": self.process.memory_info().rss,
        }
        path = self.output / "calibration.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return result

    def train(self, student, partial_path: Path, partial_key: str):
        training = self.config["training"]
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training["steps"]),
            eta_min=float(training["minimum_learning_rate"]),
        )
        samplers = {
            width: self.data.MacroSampler(
                self.splits[width]["train"], self.seed + 63100 + width
            )
            for width in self.train_widths
        }
        completed = 0
        history: list[dict[str, Any]] = []
        cumulative_wall = 0.0
        cumulative_cpu = 0.0
        if partial_path.exists():
            partial = torch.load(partial_path, weights_only=False)
            if partial.get("cache_key") != partial_key:
                raise RuntimeError("training cache key mismatch")
            completed = int(partial["step"])
            history = partial["history"]
            cumulative_wall = float(partial["wall_seconds"])
            cumulative_cpu = float(partial["aggregate_cpu_seconds"])
            student.load_state_dict(partial["model"])
            optimizer.load_state_dict(partial["optimizer"])
            scheduler.load_state_dict(partial["scheduler"])
            for width in self.train_widths:
                samplers[width].rng.setstate(partial["sampler_rng_states"][width])
            random.setstate(partial["python_rng_state"])
            np.random.set_state(partial["numpy_rng_state"])
            torch.set_rng_state(partial["torch_rng_state"])
        segment_wall = time.perf_counter()
        segment_cpu = time.process_time()
        peak_rss = self.process.memory_info().rss
        student.train()
        for step in range(completed + 1, int(training["steps"]) + 1):
            width = self.train_widths[(step - 1) % len(self.train_widths)]
            loss, stats = self.training_objective(
                student,
                samplers[width].transitions(int(training["batch_size"])),
                width,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(
                nn.utils.clip_grad_norm_(
                    student.parameters(), float(training["gradient_clip"])
                )
            )
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 50 == 0 or step == int(training["steps"]):
                peak_rss = max(peak_rss, self.process.memory_info().rss)
                item = {
                    "step": step,
                    "width": width,
                    "elapsed_wall_seconds": cumulative_wall
                    + time.perf_counter()
                    - segment_wall,
                    "gradient_norm": gradient,
                    **stats,
                }
                history.append(item)
                mlflow.log_metrics(
                    {
                        "train_loss": stats["loss"],
                        "final_batch_exact": stats["final_batch_exact"],
                        "intermediate_batch_exact": stats[
                            "intermediate_batch_exact"
                        ],
                        "teacher_forced_batch_exact": stats[
                            "teacher_forced_batch_exact"
                        ],
                        "gradient_norm": gradient,
                    },
                    step=step,
                )
                print(json.dumps(item), flush=True)
            if step % int(training["save_every_steps"]) == 0 or step == int(
                training["steps"]
            ):
                wall_total = cumulative_wall + time.perf_counter() - segment_wall
                cpu_total = cumulative_cpu + time.process_time() - segment_cpu
                torch.save(
                    {
                        "cache_key": partial_key,
                        "step": step,
                        "history": history,
                        "wall_seconds": wall_total,
                        "aggregate_cpu_seconds": cpu_total,
                        "model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "sampler_rng_states": {
                            width: sampler.rng.getstate()
                            for width, sampler in samplers.items()
                        },
                        "python_rng_state": random.getstate(),
                        "numpy_rng_state": np.random.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                    partial_path,
                )
                if wall_total > float(
                    self.config["resource_budget"][
                        "hard_runaway_ceiling_wall_seconds"
                    ]
                ):
                    raise TimeoutError("EXP-637 reached its calibrated wall ceiling")
        partial = torch.load(partial_path, weights_only=False)
        student.eval()
        return partial, peak_rss

    def evaluate(self, student, initialized, evaluation_path: Path, cache_key: str):
        cached: dict[str, Any] = {
            "cache_key": cache_key,
            "widths": {},
            "evaluation_wall_seconds": 0.0,
            "evaluation_aggregate_cpu_seconds": 0.0,
        }
        if evaluation_path.exists():
            loaded = json.loads(evaluation_path.read_text())
            if loaded.get("cache_key") != cache_key:
                raise RuntimeError("evaluation cache key mismatch")
            cached = loaded
        for width in self.eval_widths:
            key = str(width)
            if key in cached["widths"]:
                continue
            sampler = self.data.MacroSampler(
                self.splits[width]["test"], self.seed + 63200 + width
            )
            batch = sampler.transitions(
                int(self.config["evaluation"]["transition_cases"])
            )
            student_transition = self.transition_student(student, batch, width)
            direct_transition = self.transition_direct(batch, width)
            teacher_transition = self.transition_teacher(batch, width)
            init_cases = int(
                self.config["evaluation"]["initialized_control_transition_cases"]
            )
            init_batch = tuple(
                value[:init_cases] if isinstance(value, list) else value[:init_cases]
                for value in batch
            )
            initialized_transition = self.transition_student(
                initialized, init_batch, width
            )
            trained_matched_init = {
                "cases": init_cases,
                "correct": sum(
                    prediction == target
                    for prediction, target in zip(
                        student_transition["predictions"][:init_cases],
                        batch[5][:init_cases],
                    )
                ),
            }
            trained_matched_init["exact_accuracy"] = (
                trained_matched_init["correct"] / init_cases
            )
            pair_sampler = self.data.MacroSampler(
                self.splits[width]["test"], self.seed + 63300 + width
            )
            pairs = pair_sampler.pairs(int(self.config["evaluation"]["rollout_cases"]))
            self.warm_rollout_arms(student, initialized, pairs, width)
            teacher_rollout = self.rollout_teacher(*pairs, width)
            direct_rollout = self.rollout_direct(*pairs, width)
            student_rollout = self.rollout_student(student, *pairs, width)
            init_rollout_cases = int(
                self.config["evaluation"]["initialized_control_rollout_cases"]
            )
            initialized_rollout = self.rollout_student(
                initialized,
                *(values[:init_rollout_cases] for values in pairs),
                width,
            )
            value = {
                "transition": {
                    "student": self.strip_predictions(student_transition),
                    "direct_control": self.strip_predictions(direct_transition),
                    "radix2_control": self.strip_predictions(teacher_transition),
                    "initialized_control": self.strip_predictions(
                        initialized_transition
                    ),
                    "trained_matched_initialized_cases": trained_matched_init,
                    "student_vs_direct": paired_counts(
                        student_transition["predictions"],
                        direct_transition["predictions"],
                        batch[5],
                    ),
                },
                "rollout": {
                    "student": self.strip_predictions(student_rollout),
                    "direct_control": self.strip_predictions(direct_rollout),
                    "radix2_control": self.strip_predictions(teacher_rollout),
                    "initialized_control": self.strip_predictions(initialized_rollout),
                    "student_vs_direct": paired_counts(
                        student_rollout["predictions"],
                        direct_rollout["predictions"],
                        [(a * b) % p for a, b, p in zip(*pairs)],
                    ),
                    "student_to_radix2_wall_ratio": student_rollout["wall_seconds"]
                    / teacher_rollout["wall_seconds"],
                },
            }
            arm_values = [
                student_transition,
                direct_transition,
                teacher_transition,
                initialized_transition,
                student_rollout,
                direct_rollout,
                teacher_rollout,
                initialized_rollout,
            ]
            cached["evaluation_wall_seconds"] += sum(
                float(item["wall_seconds"]) for item in arm_values
            )
            cached["evaluation_aggregate_cpu_seconds"] += sum(
                float(item["aggregate_cpu_seconds"]) for item in arm_values
            )
            cached["widths"][key] = value
            evaluation_path.write_text(json.dumps(cached, indent=2) + "\n")
            if cached["evaluation_wall_seconds"] > float(
                self.config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]
            ):
                raise TimeoutError("EXP-637 evaluation reached calibrated wall ceiling")
        if "determinism" not in cached:
            determinism_cases = int(self.config["evaluation"]["determinism_cases"])
            determinism_sampler = self.data.MacroSampler(
                self.splits[64]["test"], self.seed + 63464
            )
            determinism_pairs = determinism_sampler.pairs(determinism_cases)
            first = self.rollout_student(
                student, *determinism_pairs, 64, keep_logits=True
            )
            second = self.rollout_student(
                student, *determinism_pairs, 64, keep_logits=True
            )
            determinism = {
                "cases": determinism_cases,
                "predictions_equal": first["predictions"] == second["predictions"],
                "logits_equal": first["logits"] == second["logits"],
                "correct_equal": first["correct"] == second["correct"],
                "invalid_equal": first["any_invalid_residue_rate"]
                == second["any_invalid_residue_rate"],
            }
            determinism["passed"] = all(
                determinism[key]
                for key in (
                    "predictions_equal",
                    "logits_equal",
                    "correct_equal",
                    "invalid_equal",
                )
            )
            cached["evaluation_wall_seconds"] += first["wall_seconds"] + second[
                "wall_seconds"
            ]
            cached["evaluation_aggregate_cpu_seconds"] += first[
                "aggregate_cpu_seconds"
            ] + second["aggregate_cpu_seconds"]
            cached["determinism"] = determinism
            evaluation_path.write_text(json.dumps(cached, indent=2) + "\n")
        return cached

    def formal(self) -> dict[str, Any]:
        budget = self.config["resource_budget"]
        required_budget_fields = (
            "projected_training_wall_seconds",
            "projected_training_aggregate_cpu_seconds",
            "projected_evaluation_wall_seconds",
            "projected_evaluation_aggregate_cpu_seconds",
            "expected_total_wall_seconds",
            "expected_total_aggregate_cpu_minutes",
            "soft_budget_wall_seconds",
            "hard_runaway_ceiling_wall_seconds",
            "hard_ceiling_aggregate_cpu_minutes_upper_bound",
        )
        if budget.get("calibration_status") != "complete" or any(
            budget.get(key) is None for key in required_budget_fields
        ):
            raise RuntimeError("formal EXP-637 execution requires documented calibration")
        started = time.perf_counter()
        cpu_started = time.process_time()
        repository_commit = git_output("rev-parse", "HEAD")
        repository_dirty = bool(git_output("status", "--porcelain"))
        config_hash = sha256(self.config_path)
        runner_hash = sha256(Path(__file__))
        model_hash = sha256(Path(__file__).with_name("model.py"))
        data_hash = sha256(Path(__file__).with_name("data.py"))
        identity = {
            "teacher": sha256(self.teacher_path),
            "direct": sha256(self.direct_path),
            "config": config_hash,
            "runner": runner_hash,
            "model": model_hash,
            "data": data_hash,
            "split": self.split_hash,
        }
        partial_key = sha256_text(json.dumps(identity, sort_keys=True))
        partial_path = self.output / f"training-{partial_key}.partial.pt"
        student, transfer_report = self.make_student()
        initialized, _ = self.make_student()
        initialized.eval()
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=self.config["run_id"]) as run:
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "running",
                    "role": "teacher-distilled-macro-transition",
                    "repository_commit": repository_commit,
                    "repository_dirty": str(repository_dirty).lower(),
                }
            )
            mlflow.log_params(
                {
                    "seed": self.seed,
                    "steps": self.config["training"]["steps"],
                    "widths": str(self.train_widths),
                    "teacher_checkpoint_sha256": sha256(self.teacher_path),
                    "direct_control_checkpoint_sha256": sha256(self.direct_path),
                    "config_sha256": config_hash,
                    "runner_sha256": runner_hash,
                    "split_sha256": self.split_hash,
                }
            )
            mlflow.log_artifact(str(self.config_path), artifact_path="preregistration")
            mlflow.log_artifact(
                str(ROOT / self.config["preregistration"]),
                artifact_path="preregistration",
            )
            partial, peak_rss = self.train(student, partial_path, partial_key)
            checkpoint = self.output / "weights.safetensors"
            save_file(
                {
                    key: value.detach().contiguous()
                    for key, value in student.state_dict().items()
                },
                checkpoint,
            )
            checkpoint_hash = sha256(checkpoint)
            evaluation_key = sha256_text(
                json.dumps(
                    {
                        "training": partial_key,
                        "checkpoint": checkpoint_hash,
                        "evaluation": self.config["evaluation"],
                    },
                    sort_keys=True,
                )
            )
            evaluation_path = self.output / f"evaluation-{evaluation_key}.partial.json"
            evaluation = self.evaluate(
                student, initialized, evaluation_path, evaluation_key
            )
            gates = self.config["gates"]
            e16 = evaluation["widths"]["16"]
            e32 = evaluation["widths"]["32"]
            e64 = evaluation["widths"]["64"]
            trained_init_gain = (
                e64["transition"]["trained_matched_initialized_cases"][
                    "exact_accuracy"
                ]
                - e64["transition"]["initialized_control"]["exact_accuracy"]
            )
            direct_gain = (
                e64["transition"]["student"]["exact_accuracy"]
                - e64["transition"]["direct_control"]["exact_accuracy"]
            )
            gate_results = {
                "width16_rollout": e16["rollout"]["student"]["exact_accuracy"]
                >= float(gates["minimum_width16_rollout"]),
                "width32_rollout": e32["rollout"]["student"]["exact_accuracy"]
                >= float(gates["minimum_width32_rollout"]),
                "width64_rollout": e64["rollout"]["student"]["exact_accuracy"]
                >= float(gates["minimum_width64_rollout"]),
                "width32_transition": e32["transition"]["student"][
                    "exact_accuracy"
                ]
                >= float(gates["minimum_width32_transition"]),
                "width64_transition": e64["transition"]["student"][
                    "exact_accuracy"
                ]
                >= float(gates["minimum_width64_transition"]),
                "direct_gain": direct_gain
                >= float(gates["minimum_width64_transition_gain_over_direct"]),
                "initialized_gain": trained_init_gain
                >= float(
                    gates["minimum_width64_transition_gain_over_initialized"]
                ),
                "validity": e64["rollout"]["student"][
                    "any_invalid_residue_rate"
                ]
                <= float(gates["maximum_width64_invalid_rollout_rate"]),
                "runtime": e64["rollout"]["student_to_radix2_wall_ratio"]
                <= float(gates["maximum_width64_student_to_radix2_wall_ratio"]),
                "determinism": evaluation["determinism"]["passed"]
                or not bool(gates["require_determinism"]),
            }
            passed = all(gate_results.values())
            decision = (
                "advance-distilled-macro-ablation"
                if passed
                else "reject-teacher-distilled-macro-transition"
            )
            summary = {
                "run_id": self.config["run_id"],
                "status": "completed",
                "hypothesis": self.config["hypothesis"],
                "mlflow": {
                    "tracking_uri": tracking_uri(),
                    "experiment_name": self.config["tracking"]["experiment_name"],
                    "experiment_id": experiment_id,
                    "run_id": run.info.run_id,
                },
                "source": {
                    "repository_commit": repository_commit,
                    "dirty": repository_dirty,
                    "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                    "config_sha256": config_hash,
                    "runner_sha256": runner_hash,
                    "model_sha256": model_hash,
                    "data_sha256": data_hash,
                    "split_sha256": self.split_hash,
                    "teacher_checkpoint_sha256": sha256(self.teacher_path),
                    "direct_control_checkpoint_sha256": sha256(self.direct_path),
                },
                "transfer": transfer_report,
                "training": {
                    "steps": self.config["training"]["steps"],
                    "history": partial["history"],
                    "wall_seconds": partial["wall_seconds"],
                    "aggregate_cpu_seconds": partial["aggregate_cpu_seconds"],
                },
                "models": {
                    "teacher_parameters": self.teacher.parameter_count(),
                    "direct_parameters": self.direct.parameter_count(),
                    "student_parameters": student.parameter_count(),
                    "student_checkpoint": {
                        "path": str(checkpoint.relative_to(ROOT)),
                        "sha256": checkpoint_hash,
                        "bytes": checkpoint.stat().st_size,
                    },
                },
                "evaluation": evaluation,
                "comparison": {
                    "width64_transition_gain_over_direct": direct_gain,
                    "width64_transition_gain_over_initialized": trained_init_gain,
                },
                "gates": {"thresholds": gates, "results": gate_results, "passed": passed},
                "decision": decision,
                "resource_budget": budget,
                "runtime": {
                    "wall_seconds": time.perf_counter() - started,
                    "aggregate_cpu_seconds": time.process_time() - cpu_started,
                    "torch_threads": self.config["training"]["torch_threads"],
                    "peak_rss_bytes": peak_rss,
                },
                "limitations": [
                    "Matched selection evidence only; no checked, consumed independent, official, or private answer was read.",
                    "The viability rung changes macro architecture and supervision together.",
                    "Exact bigint arithmetic is confined to offline generation, labels, tests, and evaluation truth.",
                ],
            }
            summary_path = self.output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            mlflow.log_metrics(
                {
                    "width16_rollout": e16["rollout"]["student"][
                        "exact_accuracy"
                    ],
                    "width32_rollout": e32["rollout"]["student"][
                        "exact_accuracy"
                    ],
                    "width64_rollout": e64["rollout"]["student"][
                        "exact_accuracy"
                    ],
                    "width64_transition": e64["transition"]["student"][
                        "exact_accuracy"
                    ],
                    "width64_runtime_ratio": e64["rollout"][
                        "student_to_radix2_wall_ratio"
                    ],
                    "wall_seconds": summary["runtime"]["wall_seconds"],
                    "aggregate_cpu_seconds": summary["runtime"][
                        "aggregate_cpu_seconds"
                    ],
                }
            )
            mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
            mlflow.set_tags({"status": "completed", "decision": decision})
        print(
            json.dumps(
                {
                    "decision": decision,
                    "comparison": summary["comparison"],
                    "gates": summary["gates"],
                    "checkpoint": summary["models"]["student_checkpoint"],
                    "runtime": summary["runtime"],
                    "mlflow": summary["mlflow"],
                },
                indent=2,
            )
        )
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_637_teacher_distilled_macro_transition.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-637-teacher-distilled-macro-transition",
    )
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output)
    if args.calibrate:
        experiment.calibrate()
    else:
        experiment.formal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
