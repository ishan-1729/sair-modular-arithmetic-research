"""Calibrate, train, and evaluate EXP-639 at frozen fidelity."""

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


def straight_through_bits(logits: Tensor) -> Tensor:
    probabilities = torch.sigmoid(logits)
    return logits.ge(0).float() + probabilities - probabilities.detach()


class Experiment:
    widths = (64, 128)

    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = int(self.config["seed"])
        self.process = psutil.Process()
        self.exp300 = load_module(
            "exp300_for_exp639",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
        self.exp329 = load_module(
            "exp329_for_exp639",
            ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py",
        )
        self.cellular = load_module(
            "cellular_for_exp639",
            ROOT / "experiments/exp_220_width_independent_cellular/model.py",
        )
        self.macro_data = load_module(
            "macro_data_for_exp639",
            ROOT / "experiments/exp_637_teacher_distilled_macro_transition/data.py",
        )
        self.shared = load_module(
            "shared_for_exp639",
            ROOT / "experiments/exp_637_teacher_distilled_macro_transition/run.py",
        )
        self.source_path = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source_path) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.splits = {width: self.make_split(width) for width in self.widths}
        self.split_hash = sha256_text(json.dumps(self.splits, sort_keys=True))
        self.temperature = float(self.config["model"]["intermediate_temperature"])

    def make_split(self, width: int) -> dict[str, list[int]]:
        data = self.config["data"]
        local = {
            "data": {
                "prime_bits_min": width - int(data["prime_bit_span"]) + 1,
                "prime_bits_max": width,
                "train_prime_identities": int(data["train_prime_identities"]),
                "test_prime_identities": int(data["test_prime_identities"]),
            }
        }
        return self.exp300.bigint_prime_split(local, self.seed + 63900 + width)

    def new_model(self):
        model = self.cellular.SharedDilatedConvGRU(
            width=128,
            hidden=int(self.config["model"]["hidden"]),
            cycles=int(self.config["model"]["cycles"]),
        )
        model.load_state_dict(load_file(self.source_path))
        return model

    def soft_features(
        self,
        state_plane: Tensor,
        rights: list[int],
        moduli: list[int],
        input_bits: list[int],
        width: int,
    ) -> Tensor:
        batch = len(rights)
        bit_plane = torch.tensor(input_bits, dtype=torch.float32)[:, None].expand(
            batch, width
        )
        lsb = torch.zeros(batch, width)
        msb = torch.zeros(batch, width)
        lsb[:, 0] = 1.0
        msb[:, -1] = 1.0
        return torch.stack(
            (
                state_plane,
                self.exp300.bits(rights, width),
                self.exp300.bits(moduli, width),
                bit_plane,
                lsb,
                msb,
            ),
            dim=1,
        ).mul(2.0).sub(1.0)

    def macro_forward(
        self,
        model,
        state_plane: Tensor,
        rights: list[int],
        moduli: list[int],
        digits: list[int],
        width: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        high = [digit >> 1 for digit in digits]
        low = [digit & 1 for digit in digits]
        model.width = width
        intermediate_logits, q1_logits = model(
            self.soft_features(state_plane, rights, moduli, high, width)
        )
        intermediate_plane = torch.sigmoid(intermediate_logits / self.temperature)
        final_logits, q2_logits = model(
            self.soft_features(intermediate_plane, rights, moduli, low, width)
        )
        return final_logits, intermediate_logits, q1_logits, q2_logits

    def macro_loss(
        self,
        model,
        state_plane: Tensor,
        rights: list[int],
        moduli: list[int],
        digits: list[int],
        intermediates: list[int],
        finals: list[int],
        q1: Tensor,
        q2: Tensor,
        width: int,
        quotient_weight: float,
    ) -> tuple[Tensor, dict[str, float], Tensor]:
        final_logits, intermediate_logits, q1_logits, q2_logits = self.macro_forward(
            model, state_plane, rights, moduli, digits, width
        )
        final_loss, final_stats = self.exp300.transition_loss(
            final_logits,
            q2_logits,
            finals,
            q2,
            width,
            float(self.config["training"]["worst_bit_weight"]),
            quotient_weight,
        )
        intermediate_loss, intermediate_stats = self.exp300.transition_loss(
            intermediate_logits,
            q1_logits,
            intermediates,
            q1,
            width,
            float(self.config["training"]["worst_bit_weight"]),
            quotient_weight,
        )
        loss = final_loss + float(
            self.config["training"]["intermediate_loss_weight"]
        ) * intermediate_loss
        return (
            loss,
            {
                "loss": float(loss.detach()),
                "final_batch_exact": final_stats["batch_exact"],
                "intermediate_batch_exact": intermediate_stats["batch_exact"],
                "final_loss": final_stats["loss"],
                "intermediate_loss": intermediate_stats["loss"],
            },
            final_logits,
        )

    def transition_update_loss(self, model, sampler) -> tuple[Tensor, dict[str, float]]:
        batch = sampler.transitions(int(self.config["training"]["batch_size"]))
        states, rights, moduli, digits, intermediates, finals, q1, q2 = batch
        state_plane = self.exp300.bits(states, 64)
        loss, stats, _ = self.macro_loss(
            model,
            state_plane,
            rights,
            moduli,
            digits,
            intermediates,
            finals,
            q1,
            q2,
            64,
            float(self.config["training"]["quotient_weight"]),
        )
        return loss, stats

    def recovery_update_loss(
        self,
        model,
        sampler,
        start_rng: random.Random,
        width: int,
    ) -> tuple[Tensor, dict[str, float]]:
        batch_size = int(self.config["training"]["recovery_batch_size"])
        chunk = int(self.config["training"]["recovery_macro_steps"])
        left, rights, moduli = sampler.pairs(batch_size)
        maximum_start = width // 2 - chunk
        starts = [start_rng.randrange(0, maximum_start + 1) for _ in range(batch_size)]
        bit_starts = [2 * start for start in starts]
        exact_states = self.exp329.teacher_states_before(
            left, rights, moduli, bit_starts, width
        )
        state_plane = self.exp300.bits(exact_states, width)
        losses: list[Tensor] = []
        final_exact: list[float] = []
        intermediate_exact: list[float] = []
        for offset in range(chunk):
            macro_positions = [start + offset for start in starts]
            digits = [
                (value >> (width - 2 * (position + 1))) & 3
                for value, position in zip(left, macro_positions)
            ]
            intermediates: list[int] = []
            finals: list[int] = []
            q1_values: list[int] = []
            q2_values: list[int] = []
            for state, right, modulus, digit in zip(
                exact_states, rights, moduli, digits
            ):
                intermediate, q1, final, q2 = self.macro_data.exact_macro_transition(
                    state, right, modulus, digit
                )
                intermediates.append(intermediate)
                finals.append(final)
                q1_values.append(q1)
                q2_values.append(q2)
            loss, stats, final_logits = self.macro_loss(
                model,
                state_plane,
                rights,
                moduli,
                digits,
                intermediates,
                finals,
                torch.tensor(q1_values, dtype=torch.long),
                torch.tensor(q2_values, dtype=torch.long),
                width,
                0.0,
            )
            losses.append(loss)
            final_exact.append(stats["final_batch_exact"])
            intermediate_exact.append(stats["intermediate_batch_exact"])
            state_plane = straight_through_bits(final_logits)
            exact_states = finals
        mean_loss = torch.stack(losses).mean()
        return mean_loss, {
            "loss": float(mean_loss.detach()),
            "final_batch_exact": sum(final_exact) / len(final_exact),
            "intermediate_batch_exact": sum(intermediate_exact)
            / len(intermediate_exact),
        }

    def mode_loss(self, model, samplers: dict, start_rng: random.Random, mode: str):
        if mode == "transition64":
            return self.transition_update_loss(model, samplers[mode])
        if mode in {"recovery64", "recovery128"}:
            return self.recovery_update_loss(
                model, samplers[mode], start_rng, int(mode.removeprefix("recovery"))
            )
        raise ValueError(f"unknown mode {mode}")

    def make_training_samplers(self) -> tuple[dict, random.Random]:
        samplers = {
            "transition64": self.macro_data.MacroSampler(
                self.splits[64]["train"], self.seed + 639164
            ),
            "recovery64": self.macro_data.MacroSampler(
                self.splits[64]["train"], self.seed + 639264
            ),
            "recovery128": self.macro_data.MacroSampler(
                self.splits[128]["train"], self.seed + 639328
            ),
        }
        return samplers, random.Random(self.seed + 639400)

    def transition_continuous(
        self, model, batch: tuple, width: int
    ) -> dict[str, Any]:
        states, rights, moduli, digits, intermediates, finals, _q1, _q2 = batch
        predictions: list[int] = []
        intermediate_predictions: list[int] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        with torch.inference_mode():
            for start in range(0, len(states), 512):
                stop = min(start + 512, len(states))
                final_logits, intermediate_logits, _q1l, _q2l = self.macro_forward(
                    model,
                    self.exp300.bits(states[start:stop], width),
                    rights[start:stop],
                    moduli[start:stop],
                    digits[start:stop],
                    width,
                )
                predictions.extend(self.shared.decode(final_logits))
                intermediate_predictions.extend(self.shared.decode(intermediate_logits))
        cases = len(states)
        correct = sum(a == b for a, b in zip(predictions, finals))
        return {
            "cases": cases,
            "correct": correct,
            "exact_accuracy": correct / cases,
            "intermediate_correct": sum(
                a == b for a, b in zip(intermediate_predictions, intermediates)
            ),
            "intermediate_exact_accuracy": sum(
                a == b for a, b in zip(intermediate_predictions, intermediates)
            )
            / cases,
            "valid_residue_rate": sum(
                prediction < modulus
                for prediction, modulus in zip(predictions, moduli)
            )
            / cases,
            "predictions": predictions,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
        }

    def transition_hard(self, model, batch: tuple, width: int) -> dict[str, Any]:
        states, rights, moduli, digits, _intermediates, finals, _q1, _q2 = batch
        predictions: list[int] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        model.width = width
        with torch.inference_mode():
            for start in range(0, len(states), 512):
                stop = min(start + 512, len(states))
                high = [digit >> 1 for digit in digits[start:stop]]
                low = [digit & 1 for digit in digits[start:stop]]
                first_logits, _ = model(
                    self.exp300.features(
                        states[start:stop],
                        rights[start:stop],
                        moduli[start:stop],
                        high,
                        width,
                    )
                )
                first = self.shared.decode(first_logits)
                second_logits, _ = model(
                    self.exp300.features(
                        first,
                        rights[start:stop],
                        moduli[start:stop],
                        low,
                        width,
                    )
                )
                predictions.extend(self.shared.decode(second_logits))
        cases = len(states)
        correct = sum(a == b for a, b in zip(predictions, finals))
        return {
            "cases": cases,
            "correct": correct,
            "exact_accuracy": correct / cases,
            "valid_residue_rate": sum(
                prediction < modulus
                for prediction, modulus in zip(predictions, moduli)
            )
            / cases,
            "predictions": predictions,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
        }

    def rollout_hard(
        self,
        model,
        left: list[int],
        rights: list[int],
        moduli: list[int],
        width: int,
    ) -> dict[str, Any]:
        states = [0] * len(left)
        exact_states = [0] * len(left)
        every = [True] * len(left)
        invalid = [False] * len(left)
        wall = time.perf_counter()
        cpu = time.process_time()
        model.width = width
        with torch.inference_mode():
            for position in range(width):
                input_bits = [
                    (value >> (width - 1 - position)) & 1 for value in left
                ]
                logits, _ = model(
                    self.exp300.features(states, rights, moduli, input_bits, width)
                )
                states = self.shared.decode(logits)
                exact_states = [
                    self.macro_data.exact_radix2_transition(
                        state, right, modulus, input_bit
                    )[0]
                    for state, right, modulus, input_bit in zip(
                        exact_states, rights, moduli, input_bits
                    )
                ]
                every = [
                    prior and prediction == target
                    for prior, prediction, target in zip(every, states, exact_states)
                ]
                invalid = [
                    prior or prediction >= modulus
                    for prior, prediction, modulus in zip(invalid, states, moduli)
                ]
        expected = [(a * b) % p for a, b, p in zip(left, rights, moduli)]
        correct = sum(a == b for a, b in zip(states, expected))
        return {
            "cases": len(left),
            "correct": correct,
            "exact_accuracy": correct / len(left),
            "all_steps_exact_accuracy": sum(every) / len(left),
            "any_invalid_residue_rate": sum(invalid) / len(left),
            "predictions": states,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "recurrent_steps": width,
        }

    def rollout_continuous(
        self,
        model,
        left: list[int],
        rights: list[int],
        moduli: list[int],
        width: int,
        keep_logits: bool = False,
    ) -> dict[str, Any]:
        states = [0] * len(left)
        exact_states = [0] * len(left)
        every_macro = [True] * len(left)
        every_intermediate = [True] * len(left)
        invalid = [False] * len(left)
        logit_records: list[list[list[float]]] = []
        wall = time.perf_counter()
        cpu = time.process_time()
        with torch.inference_mode():
            for position in range(width // 2):
                digits = [
                    (value >> (width - 2 * (position + 1))) & 3 for value in left
                ]
                final_logits, intermediate_logits, _q1, _q2 = self.macro_forward(
                    model,
                    self.exp300.bits(states, width),
                    rights,
                    moduli,
                    digits,
                    width,
                )
                states = self.shared.decode(final_logits)
                intermediate_predictions = self.shared.decode(intermediate_logits)
                exact_intermediates: list[int] = []
                next_exact: list[int] = []
                for state, right, modulus, digit in zip(
                    exact_states, rights, moduli, digits
                ):
                    intermediate, _q1v, final, _q2v = (
                        self.macro_data.exact_macro_transition(
                            state, right, modulus, digit
                        )
                    )
                    exact_intermediates.append(intermediate)
                    next_exact.append(final)
                every_intermediate = [
                    prior and prediction == target
                    for prior, prediction, target in zip(
                        every_intermediate,
                        intermediate_predictions,
                        exact_intermediates,
                    )
                ]
                every_macro = [
                    prior and prediction == target
                    for prior, prediction, target in zip(
                        every_macro, states, next_exact
                    )
                ]
                invalid = [
                    prior or prediction >= modulus
                    for prior, prediction, modulus in zip(invalid, states, moduli)
                ]
                exact_states = next_exact
                if keep_logits:
                    logit_records.append(
                        [intermediate_logits.tolist(), final_logits.tolist()]
                    )
        expected = [(a * b) % p for a, b, p in zip(left, rights, moduli)]
        correct = sum(a == b for a, b in zip(states, expected))
        result: dict[str, Any] = {
            "cases": len(left),
            "correct": correct,
            "exact_accuracy": correct / len(left),
            "all_macro_steps_exact_accuracy": sum(every_macro) / len(left),
            "all_intermediate_steps_exact_accuracy": sum(every_intermediate)
            / len(left),
            "any_invalid_residue_rate": sum(invalid) / len(left),
            "predictions": states,
            "wall_seconds": time.perf_counter() - wall,
            "aggregate_cpu_seconds": time.process_time() - cpu,
            "macro_steps": width // 2,
        }
        if keep_logits:
            result["logits"] = logit_records
        return result

    @staticmethod
    def strip(result: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in result.items()
            if key not in {"predictions", "logits"}
        }

    def calibrate(self) -> dict[str, Any]:
        model = self.new_model()
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        samplers, start_rng = self.make_training_samplers()
        profiles: dict[str, dict[str, float]] = {}
        updates = int(self.config["resource_budget"]["profile_updates_per_mode"])
        for mode in ("transition64", "recovery64", "recovery128"):
            wall = time.perf_counter()
            cpu = time.process_time()
            for _ in range(updates):
                loss, _ = self.mode_loss(model, samplers, start_rng, mode)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(self.config["training"]["gradient_clip"]),
                )
                optimizer.step()
            profiles[mode] = {
                "updates": updates,
                "wall_seconds": time.perf_counter() - wall,
                "aggregate_cpu_seconds": time.process_time() - cpu,
            }
        model.eval()
        source = self.new_model().eval()
        evaluation_profiles: dict[str, Any] = {}
        transition_cases = int(
            self.config["resource_budget"]["profile_transition_cases_per_width"]
        )
        rollout_cases = int(
            self.config["resource_budget"]["profile_rollout_cases_per_width"]
        )
        projected_eval_wall = 0.0
        projected_eval_cpu = 0.0
        for width in self.widths:
            batch = self.macro_data.MacroSampler(
                self.splits[width]["train"], self.seed + 639500 + width
            ).transitions(transition_cases)
            transition_results = {
                "hard": self.transition_hard(source, batch, width),
                "initialized": self.transition_continuous(source, batch, width),
                "trained_shaped": self.transition_continuous(model, batch, width),
            }
            pairs = self.macro_data.MacroSampler(
                self.splits[width]["train"], self.seed + 639600 + width
            ).pairs(rollout_cases)
            warm = tuple(values[:2] for values in pairs)
            self.rollout_hard(source, *warm, width)
            self.rollout_continuous(source, *warm, width)
            self.rollout_continuous(model, *warm, width)
            rollout_results = {
                "hard": self.rollout_hard(source, *pairs, width),
                "initialized": self.rollout_continuous(source, *pairs, width),
                "trained_shaped": self.rollout_continuous(model, *pairs, width),
            }
            evaluation_profiles[str(width)] = {
                "transition": {
                    key: self.strip(value) for key, value in transition_results.items()
                },
                "rollout": {
                    key: self.strip(value) for key, value in rollout_results.items()
                },
            }
            for value in transition_results.values():
                projected_eval_wall += (
                    value["wall_seconds"]
                    / transition_cases
                    * int(self.config["evaluation"]["transition_cases"])
                )
                projected_eval_cpu += (
                    value["aggregate_cpu_seconds"]
                    / transition_cases
                    * int(self.config["evaluation"]["transition_cases"])
                )
            for value in rollout_results.values():
                projected_eval_wall += (
                    value["wall_seconds"]
                    / rollout_cases
                    * int(self.config["evaluation"]["rollout_cases"])
                )
                projected_eval_cpu += (
                    value["aggregate_cpu_seconds"]
                    / rollout_cases
                    * int(self.config["evaluation"]["rollout_cases"])
                )
        width128_profile = evaluation_profiles["128"]["rollout"]["trained_shaped"]
        determinism_cases = int(self.config["evaluation"]["determinism_cases"])
        projected_eval_wall += (
            2
            * width128_profile["wall_seconds"]
            / rollout_cases
            * determinism_cases
        )
        projected_eval_cpu += (
            2
            * width128_profile["aggregate_cpu_seconds"]
            / rollout_cases
            * determinism_cases
        )
        mode_counts = {"transition64": 100, "recovery64": 100, "recovery128": 200}
        projected_training_wall = sum(
            profiles[mode]["wall_seconds"] / updates * count
            for mode, count in mode_counts.items()
        )
        projected_training_cpu = sum(
            profiles[mode]["aggregate_cpu_seconds"] / updates * count
            for mode, count in mode_counts.items()
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
            "soft_budget_wall_seconds": math.ceil(expected_wall * 1.35 / 60) * 60,
            "hard_runaway_ceiling_wall_seconds": math.ceil(expected_wall * 2.0 / 60)
            * 60,
            "hard_ceiling_aggregate_cpu_minutes_upper_bound": math.ceil(
                expected_wall * 2.0 * 4.0 / 60.0
            ),
        }
        result = {
            "run_id": self.config["run_id"],
            "mode": "training-identity-only-calibration",
            "seed": self.seed,
            "source_hashes": self.source_hashes(),
            "update_profiles": profiles,
            "mode_counts": mode_counts,
            "evaluation_profiles": evaluation_profiles,
            "recommendation": recommendation,
            "heldout_identities_read": False,
            "fidelity_fields_changed": False,
            "peak_rss_bytes": self.process.memory_info().rss,
        }
        path = self.output / "calibration.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return result

    def source_hashes(self) -> dict[str, str]:
        return {
            "checkpoint": sha256(self.source_path),
            "config": sha256(self.config_path),
            "runner": sha256(Path(__file__)),
            "macro_data": sha256(
                ROOT / "experiments/exp_637_teacher_distilled_macro_transition/data.py"
            ),
            "cellular_model": sha256(
                ROOT / "experiments/exp_220_width_independent_cellular/model.py"
            ),
            "split": self.split_hash,
        }

    def train(self, model, partial_path: Path, cache_key: str):
        training = self.config["training"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training["steps"]),
            eta_min=float(training["minimum_learning_rate"]),
        )
        samplers, start_rng = self.make_training_samplers()
        completed = 0
        history: list[dict[str, Any]] = []
        cumulative_wall = cumulative_cpu = 0.0
        if partial_path.exists():
            partial = torch.load(partial_path, weights_only=False)
            if partial.get("cache_key") != cache_key:
                raise RuntimeError("training cache key mismatch")
            completed = int(partial["step"])
            history = partial["history"]
            cumulative_wall = float(partial["wall_seconds"])
            cumulative_cpu = float(partial["aggregate_cpu_seconds"])
            model.load_state_dict(partial["model"])
            optimizer.load_state_dict(partial["optimizer"])
            scheduler.load_state_dict(partial["scheduler"])
            for name, sampler in samplers.items():
                sampler.rng.setstate(partial["sampler_rng_states"][name])
            start_rng.setstate(partial["start_rng_state"])
            random.setstate(partial["python_rng_state"])
            np.random.set_state(partial["numpy_rng_state"])
            torch.set_rng_state(partial["torch_rng_state"])
        schedule = list(training["mode_schedule"])
        segment_wall = time.perf_counter()
        segment_cpu = time.process_time()
        peak_rss = self.process.memory_info().rss
        model.train()
        for step in range(completed + 1, int(training["steps"]) + 1):
            mode = schedule[(step - 1) % len(schedule)]
            loss, stats = self.mode_loss(model, samplers, start_rng, mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
            )
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 25 == 0 or step == int(training["steps"]):
                peak_rss = max(peak_rss, self.process.memory_info().rss)
                record = {
                    "step": step,
                    "mode": mode,
                    "elapsed_wall_seconds": cumulative_wall
                    + time.perf_counter()
                    - segment_wall,
                    "gradient_norm": gradient,
                    **stats,
                }
                history.append(record)
                mlflow.log_metrics(
                    {
                        "train_loss": stats["loss"],
                        "final_batch_exact": stats["final_batch_exact"],
                        "intermediate_batch_exact": stats[
                            "intermediate_batch_exact"
                        ],
                        "gradient_norm": gradient,
                    },
                    step=step,
                )
                print(json.dumps(record), flush=True)
            if step % int(training["save_every_steps"]) == 0 or step == int(
                training["steps"]
            ):
                wall_total = cumulative_wall + time.perf_counter() - segment_wall
                cpu_total = cumulative_cpu + time.process_time() - segment_cpu
                torch.save(
                    {
                        "cache_key": cache_key,
                        "step": step,
                        "history": history,
                        "wall_seconds": wall_total,
                        "aggregate_cpu_seconds": cpu_total,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "sampler_rng_states": {
                            name: sampler.rng.getstate()
                            for name, sampler in samplers.items()
                        },
                        "start_rng_state": start_rng.getstate(),
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
                    raise TimeoutError("EXP-639 reached its calibrated wall ceiling")
        partial = torch.load(partial_path, weights_only=False)
        model.eval()
        return partial, peak_rss

    def evaluate(self, trained, source, cache_path: Path, cache_key: str, training_wall: float):
        cached: dict[str, Any] = {
            "cache_key": cache_key,
            "widths": {},
            "active_wall_seconds": 0.0,
            "arm_wall_seconds": 0.0,
            "arm_aggregate_cpu_seconds": 0.0,
        }
        if cache_path.exists():
            loaded = json.loads(cache_path.read_text())
            if loaded.get("cache_key") != cache_key:
                raise RuntimeError("evaluation cache key mismatch")
            cached = loaded
        prior_active = float(cached["active_wall_seconds"])
        segment_started = time.perf_counter()

        def save() -> None:
            cached["active_wall_seconds"] = prior_active + (
                time.perf_counter() - segment_started
            )
            cache_path.write_text(json.dumps(cached, indent=2) + "\n")
            if training_wall + cached["active_wall_seconds"] > float(
                self.config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]
            ):
                raise TimeoutError("EXP-639 evaluation reached calibrated wall ceiling")

        for width in self.widths:
            key = str(width)
            if key in cached["widths"]:
                continue
            batch = self.macro_data.MacroSampler(
                self.splits[width]["test"], self.seed + 639700 + width
            ).transitions(int(self.config["evaluation"]["transition_cases"]))
            hard_transition = self.transition_hard(source, batch, width)
            initialized_transition = self.transition_continuous(source, batch, width)
            trained_transition = self.transition_continuous(trained, batch, width)
            pairs = self.macro_data.MacroSampler(
                self.splits[width]["test"], self.seed + 639800 + width
            ).pairs(int(self.config["evaluation"]["rollout_cases"]))
            warm = tuple(values[:2] for values in pairs)
            self.rollout_hard(source, *warm, width)
            self.rollout_continuous(source, *warm, width)
            self.rollout_continuous(trained, *warm, width)
            hard_rollout = self.rollout_hard(source, *pairs, width)
            initialized_rollout = self.rollout_continuous(source, *pairs, width)
            trained_rollout = self.rollout_continuous(trained, *pairs, width)
            targets = [(a * b) % p for a, b, p in zip(*pairs)]
            cached["widths"][key] = {
                "transition": {
                    "hard_frozen": self.strip(hard_transition),
                    "continuous_initialized": self.strip(initialized_transition),
                    "continuous_trained": self.strip(trained_transition),
                    "trained_vs_hard": self.shared.paired_counts(
                        trained_transition["predictions"],
                        hard_transition["predictions"],
                        batch[5],
                    ),
                    "trained_vs_initialized": self.shared.paired_counts(
                        trained_transition["predictions"],
                        initialized_transition["predictions"],
                        batch[5],
                    ),
                },
                "rollout": {
                    "hard_frozen": self.strip(hard_rollout),
                    "continuous_initialized": self.strip(initialized_rollout),
                    "continuous_trained": self.strip(trained_rollout),
                    "trained_vs_hard": self.shared.paired_counts(
                        trained_rollout["predictions"],
                        hard_rollout["predictions"],
                        targets,
                    ),
                    "trained_vs_initialized": self.shared.paired_counts(
                        trained_rollout["predictions"],
                        initialized_rollout["predictions"],
                        targets,
                    ),
                    "trained_to_hard_wall_ratio": trained_rollout["wall_seconds"]
                    / hard_rollout["wall_seconds"],
                },
            }
            for result in (
                hard_transition,
                initialized_transition,
                trained_transition,
                hard_rollout,
                initialized_rollout,
                trained_rollout,
            ):
                cached["arm_wall_seconds"] += result["wall_seconds"]
                cached["arm_aggregate_cpu_seconds"] += result[
                    "aggregate_cpu_seconds"
                ]
            save()
        if "determinism" not in cached:
            width = 128
            pairs = self.macro_data.MacroSampler(
                self.splits[width]["test"], self.seed + 639900 + width
            ).pairs(int(self.config["evaluation"]["determinism_cases"]))
            first = self.rollout_continuous(trained, *pairs, width, keep_logits=True)
            second = self.rollout_continuous(trained, *pairs, width, keep_logits=True)
            determinism = {
                "cases": len(pairs[0]),
                "predictions_equal": first["predictions"] == second["predictions"],
                "logits_equal": first["logits"] == second["logits"],
                "correct_equal": first["correct"] == second["correct"],
                "validity_equal": first["any_invalid_residue_rate"]
                == second["any_invalid_residue_rate"],
            }
            determinism["passed"] = all(
                determinism[key]
                for key in (
                    "predictions_equal",
                    "logits_equal",
                    "correct_equal",
                    "validity_equal",
                )
            )
            cached["determinism"] = determinism
            for result in (first, second):
                cached["arm_wall_seconds"] += result["wall_seconds"]
                cached["arm_aggregate_cpu_seconds"] += result[
                    "aggregate_cpu_seconds"
                ]
            save()
        return cached

    def formal(self) -> dict[str, Any]:
        budget = self.config["resource_budget"]
        required = (
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
            budget.get(key) is None for key in required
        ):
            raise RuntimeError("formal EXP-639 execution requires documented calibration")
        started = time.perf_counter()
        cpu_started = time.process_time()
        commit = git_output("rev-parse", "HEAD")
        dirty = bool(git_output("status", "--porcelain"))
        hashes = self.source_hashes()
        partial_key = sha256_text(json.dumps(hashes, sort_keys=True))
        partial_path = self.output / f"training-{partial_key}.partial.pt"
        trained = self.new_model()
        source = self.new_model().eval()
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=self.config["run_id"]) as run:
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "running",
                    "role": "full-depth-soft-boundary-adaptation",
                    "repository_commit": commit,
                    "repository_dirty": str(dirty).lower(),
                }
            )
            mlflow.log_params(
                {
                    "seed": self.seed,
                    "steps": self.config["training"]["steps"],
                    "mode_schedule": str(self.config["training"]["mode_schedule"]),
                    "source_checkpoint_sha256": hashes["checkpoint"],
                    "config_sha256": hashes["config"],
                    "runner_sha256": hashes["runner"],
                    "split_sha256": hashes["split"],
                }
            )
            mlflow.log_artifact(str(self.config_path), artifact_path="preregistration")
            mlflow.log_artifact(
                str(ROOT / self.config["preregistration"]),
                artifact_path="preregistration",
            )
            partial, peak_rss = self.train(trained, partial_path, partial_key)
            checkpoint = self.output / "weights.safetensors"
            save_file(
                {
                    key: value.detach().contiguous()
                    for key, value in trained.state_dict().items()
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
                trained,
                source,
                evaluation_path,
                evaluation_key,
                float(partial["wall_seconds"]),
            )
            e64 = evaluation["widths"]["64"]
            e128 = evaluation["widths"]["128"]
            trained128 = e128["rollout"]["continuous_trained"]["exact_accuracy"]
            better_frozen = max(
                e128["rollout"]["hard_frozen"]["exact_accuracy"],
                e128["rollout"]["continuous_initialized"]["exact_accuracy"],
            )
            gain = trained128 - better_frozen
            gates = self.config["gates"]
            gate_results = {
                "width128_rollout": trained128
                >= float(gates["minimum_trained_width128_rollout"]),
                "gain": gain
                >= float(gates["minimum_width128_gain_over_better_frozen_arm"]),
                "width64_retention": e64["rollout"]["continuous_trained"][
                    "exact_accuracy"
                ]
                >= float(gates["minimum_trained_width64_rollout"]),
                "width128_transition": e128["transition"]["continuous_trained"][
                    "exact_accuracy"
                ]
                >= float(gates["minimum_trained_width128_transition"]),
                "validity": e128["rollout"]["continuous_trained"][
                    "any_invalid_residue_rate"
                ]
                <= float(gates["maximum_trained_width128_invalid_rollout_rate"]),
                "runtime": e128["rollout"]["trained_to_hard_wall_ratio"]
                <= float(gates["maximum_trained_to_hard_width128_wall_ratio"]),
                "determinism": evaluation["determinism"]["passed"]
                or not bool(gates["require_determinism"]),
            }
            passed = all(gate_results.values())
            decision = (
                "advance-soft-boundary-tier6-independent-confirmation"
                if passed
                else "reject-full-depth-soft-boundary-adaptation"
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
                    "repository_commit": commit,
                    "dirty": dirty,
                    "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                    **{f"{key}_sha256": value for key, value in hashes.items()},
                },
                "training": {
                    "steps": self.config["training"]["steps"],
                    "history": partial["history"],
                    "wall_seconds": partial["wall_seconds"],
                    "aggregate_cpu_seconds": partial["aggregate_cpu_seconds"],
                },
                "model": {
                    "parameters": trained.parameter_count(),
                    "checkpoint": {
                        "path": str(checkpoint.relative_to(ROOT)),
                        "sha256": checkpoint_hash,
                        "bytes": checkpoint.stat().st_size,
                    },
                },
                "evaluation": evaluation,
                "comparison": {
                    "better_frozen_width128_rollout": better_frozen,
                    "trained_width128_rollout": trained128,
                    "width128_gain": gain,
                },
                "gates": {"thresholds": gates, "results": gate_results, "passed": passed},
                "decision": decision,
                "resource_budget": budget,
                "runtime": {
                    "wall_seconds": time.perf_counter() - started,
                    "aggregate_cpu_seconds": time.process_time() - cpu_started,
                    "peak_rss_bytes": peak_rss,
                    "torch_threads": self.config["training"]["torch_threads"],
                },
                "limitations": [
                    "Fresh generated selection evidence only; checked, independent-full, official, and private answers were not read.",
                    "The adaptation changes macro supervision and outer-state credit together.",
                    "Exact bigint arithmetic is confined to offline data, labels, starts, and evaluation truth.",
                ],
            }
            summary_path = self.output / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            mlflow.log_metrics(
                {
                    "hard_width128_rollout": e128["rollout"]["hard_frozen"][
                        "exact_accuracy"
                    ],
                    "initialized_width128_rollout": e128["rollout"][
                        "continuous_initialized"
                    ]["exact_accuracy"],
                    "trained_width128_rollout": trained128,
                    "width128_gain": gain,
                    "trained_width64_rollout": e64["rollout"][
                        "continuous_trained"
                    ]["exact_accuracy"],
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
                    "checkpoint": summary["model"]["checkpoint"],
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
        default=ROOT / "configs/exp_639_full_depth_soft_boundary_adaptation.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-639-full-depth-soft-boundary-adaptation",
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
