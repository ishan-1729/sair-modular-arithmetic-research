"""Calibrate and run the matched EXP-647 commutative-consistency experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
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
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def scientific_config_sha256(config: dict[str, Any]) -> str:
    """Hash scientific fields while allowing a post-calibration budget resolution."""

    scientific = copy.deepcopy(config)
    scientific.pop("resource_budget", None)
    return sha256_text(canonical_json(scientific))


def by_width(mapping: dict[Any, Any], width: int) -> Any:
    if width in mapping:
        return mapping[width]
    return mapping[str(width)]


def decode_logits(logits: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in logits.ge(0).tolist()
    ]


def soft_features(
    state_plane: torch.Tensor,
    multiplicands: list[int],
    moduli: list[int],
    input_bits: list[int],
    width: int,
    bits_fn,
) -> torch.Tensor:
    batch = len(multiplicands)
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
            bits_fn(multiplicands, width),
            bits_fn(moduli, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul(2.0).sub(1.0)


def confidence_teacher_loss(
    original_logits: torch.Tensor,
    swapped_logits: torch.Tensor,
    original_minimum_margins: torch.Tensor,
    swapped_minimum_margins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teach the lower-margin endpoint from the higher-margin endpoint.

    The confidence comparison and teacher probabilities are detached. Ties retain
    the original as teacher, matching the frozen EXP-640--642 convention.
    """

    swapped_is_teacher = (
        swapped_minimum_margins.detach() > original_minimum_margins.detach()
    )
    mask = swapped_is_teacher[:, None]
    student_logits = torch.where(mask, original_logits, swapped_logits)
    teacher_probabilities = torch.where(
        mask,
        torch.sigmoid(swapped_logits.detach()),
        torch.sigmoid(original_logits.detach()),
    )
    return (
        F.binary_cross_entropy_with_logits(student_logits, teacher_probabilities),
        swapped_is_teacher,
    )


def paired_rollout_summary(
    original_predictions: list[int],
    swapped_predictions: list[int],
    original_margins: list[float],
    swapped_margins: list[float],
    targets: list[int],
) -> dict[str, Any]:
    if not (
        len(original_predictions)
        == len(swapped_predictions)
        == len(original_margins)
        == len(swapped_margins)
        == len(targets)
    ):
        raise ValueError("paired rollout vectors must have equal length")
    original_correct = [
        prediction == target
        for prediction, target in zip(original_predictions, targets, strict=True)
    ]
    swapped_correct = [
        prediction == target
        for prediction, target in zip(swapped_predictions, targets, strict=True)
    ]
    use_swapped = [
        swapped > original
        for original, swapped in zip(original_margins, swapped_margins, strict=True)
    ]
    selected = [
        swapped if choose else original
        for original, swapped, choose in zip(
            original_predictions, swapped_predictions, use_swapped, strict=True
        )
    ]
    selected_correct = [
        prediction == target
        for prediction, target in zip(selected, targets, strict=True)
    ]
    cases = len(targets)
    return {
        "cases": cases,
        "intersection_correct": sum(
            left and right
            for left, right in zip(original_correct, swapped_correct, strict=True)
        ),
        "original_only_correct": sum(
            left and not right
            for left, right in zip(original_correct, swapped_correct, strict=True)
        ),
        "swapped_only_correct": sum(
            right and not left
            for left, right in zip(original_correct, swapped_correct, strict=True)
        ),
        "neither_correct": sum(
            not left and not right
            for left, right in zip(original_correct, swapped_correct, strict=True)
        ),
        "oracle_union_exact_accuracy": sum(
            left or right
            for left, right in zip(original_correct, swapped_correct, strict=True)
        )
        / cases,
        "selected_correct": sum(selected_correct),
        "selected_exact_accuracy": sum(selected_correct) / cases,
        "selected_swapped": sum(use_swapped),
        "prediction_disagreement": sum(
            left != right
            for left, right in zip(
                original_predictions, swapped_predictions, strict=True
            )
        ),
        "prediction_disagreement_rate": sum(
            left != right
            for left, right in zip(
                original_predictions, swapped_predictions, strict=True
            )
        )
        / cases,
        "use_swapped": use_swapped,
        "selected_predictions": selected,
    }


def paired_arm_comparison(
    control_predictions: list[int],
    challenger_predictions: list[int],
    targets: list[int],
) -> dict[str, Any]:
    control_correct = [
        prediction == target
        for prediction, target in zip(control_predictions, targets, strict=True)
    ]
    challenger_correct = [
        prediction == target
        for prediction, target in zip(challenger_predictions, targets, strict=True)
    ]
    control_only = sum(
        left and not right
        for left, right in zip(control_correct, challenger_correct, strict=True)
    )
    challenger_only = sum(
        right and not left
        for left, right in zip(control_correct, challenger_correct, strict=True)
    )
    discordant = control_only + challenger_only
    if discordant:
        tail = sum(
            math.comb(discordant, index) for index in range(min(control_only, challenger_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "cases": len(targets),
        "both_correct": sum(
            left and right
            for left, right in zip(control_correct, challenger_correct, strict=True)
        ),
        "control_only_correct": control_only,
        "challenger_only_correct": challenger_only,
        "neither_correct": sum(
            not left and not right
            for left, right in zip(control_correct, challenger_correct, strict=True)
        ),
        "discordant_cases": discordant,
        "exact_two_sided_mcnemar_p": p_value,
    }


def resolve_budget(
    config: dict[str, Any], measured_arms: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    schedule_counts = Counter(int(width) for width in config["training"]["width_schedule"])
    repetitions = int(config["training"]["steps"]) // len(
        config["training"]["width_schedule"]
    )
    remainder = int(config["training"]["steps"]) % len(
        config["training"]["width_schedule"]
    )
    formal_counts = {
        width: count * repetitions for width, count in schedule_counts.items()
    }
    for width in config["training"]["width_schedule"][:remainder]:
        formal_counts[int(width)] = formal_counts.get(int(width), 0) + 1

    projected_wall = 0.0
    projected_cpu = 0.0
    components: dict[str, Any] = {}
    profile_transition_cases = int(
        config["calibration"]["transition_cases_per_width_per_arm"]
    )
    profile_rollout_cases = int(
        config["calibration"]["paired_rollout_cases_per_width_per_arm"]
    )
    for arm_name, arm in measured_arms.items():
        arm_wall = 0.0
        arm_cpu = 0.0
        for width, formal_updates in formal_counts.items():
            timing = arm["training"]["per_width_timing"][str(width)]
            profile_updates = timing["updates"]
            arm_wall += timing["wall_seconds"] / profile_updates * formal_updates
            arm_cpu += (
                timing["aggregate_cpu_seconds"] / profile_updates * formal_updates
            )
        for width in (64, 128, 256):
            formal_transition_cases = int(
                by_width(config["evaluation"]["transition_cases"], width)
            )
            formal_rollout_cases = int(
                by_width(config["evaluation"]["paired_rollout_cases"], width)
            )
            transition_timing = arm["evaluation"][str(width)]["transition"]["timing"]
            original_timing = arm["evaluation"][str(width)]["original"]["timing"]
            swapped_timing = arm["evaluation"][str(width)]["swapped"]["timing"]
            transition_scale = formal_transition_cases / profile_transition_cases
            rollout_scale = formal_rollout_cases / profile_rollout_cases
            arm_wall += transition_timing["wall_seconds"] * transition_scale
            arm_cpu += transition_timing["aggregate_cpu_seconds"] * transition_scale
            arm_wall += (
                original_timing["wall_seconds"] + swapped_timing["wall_seconds"]
            ) * rollout_scale
            arm_cpu += (
                original_timing["aggregate_cpu_seconds"]
                + swapped_timing["aggregate_cpu_seconds"]
            ) * rollout_scale
        if arm_name == "confidence_consistency":
            det_cases = int(config["evaluation"]["determinism_cases"])
            original = arm["evaluation"]["128"]["original"]["timing"]
            det_scale = (2 * det_cases) / profile_rollout_cases
            arm_wall += original["wall_seconds"] * det_scale
            arm_cpu += original["aggregate_cpu_seconds"] * det_scale
        projected_wall += arm_wall
        projected_cpu += arm_cpu
        components[arm_name] = {
            "projected_wall_seconds": arm_wall,
            "projected_aggregate_cpu_seconds": arm_cpu,
        }

    resource = config["resource_budget"]
    wall_setup = float(resource["setup_and_serialization_headroom_wall_seconds"])
    cpu_setup = float(
        resource["setup_and_serialization_headroom_aggregate_cpu_seconds"]
    )
    safety = float(resource["throughput_safety_multiplier"])
    soft_wall = math.ceil(projected_wall * safety + wall_setup)
    soft_cpu = math.ceil(projected_cpu * safety + cpu_setup)
    hard_multiplier = float(resource["hard_ceiling_multiplier_over_soft"])
    return {
        "formal_update_counts_by_width_per_arm": {
            str(width): count for width, count in sorted(formal_counts.items())
        },
        "component_projection": components,
        "projected_formal_wall_seconds": math.ceil(projected_wall),
        "projected_formal_aggregate_cpu_seconds": math.ceil(projected_cpu),
        "setup_and_serialization_headroom_wall_seconds": wall_setup,
        "setup_and_serialization_headroom_aggregate_cpu_seconds": cpu_setup,
        "throughput_safety_multiplier": safety,
        "soft_budget_wall_seconds": soft_wall,
        "soft_budget_aggregate_cpu_seconds": soft_cpu,
        "hard_runaway_ceiling_wall_seconds": math.ceil(soft_wall * hard_multiplier),
        "hard_runaway_ceiling_aggregate_cpu_seconds": math.ceil(
            soft_cpu * hard_multiplier
        ),
        "fidelity_reduced": False,
    }


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = output.resolve()
        self.mode = mode
        self.output.mkdir(parents=True, exist_ok=True)
        self.source = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")
        self.exp300 = load_module(
            "exp300_for_exp647",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
        self.cellular = load_module(
            "cellular_for_exp647",
            ROOT / "experiments/exp_220_width_independent_cellular/model.py",
        )
        self.seed = int(self.config["seed"])
        self.widths = (64, 128, 256)
        self.runner_hash = sha256(Path(__file__))
        self.config_hash = sha256(self.config_path)
        self.scientific_hash = scientific_config_sha256(self.config)
        self.preregistration = (ROOT / self.config["preregistration"]).resolve()
        self.preregistration_hash = sha256(self.preregistration)
        self.repository_commit = git_output("rev-parse", "HEAD")
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.splits = {width: self.make_split(width) for width in self.widths}
        self.evaluation_pairs = {
            width: self.exp300.BigIntSampler(
                self.splits[width]["test"], self.seed + 647800 + width
            ).pairs(
                int(by_width(self.config["evaluation"]["paired_rollout_cases"], width))
            )
            for width in self.widths
        }
        self.dataset_hash = sha256_text(
            canonical_json(
                {
                    "splits": self.splits,
                    "evaluation_pairs": self.evaluation_pairs,
                }
            )
        )

    def make_split(self, width: int) -> dict[str, list[int]]:
        local = {
            "data": {
                "prime_bits_min": width - int(self.config["data"]["prime_bit_span"]) + 1,
                "prime_bits_max": width,
                "train_prime_identities": int(
                    self.config["data"]["train_prime_identities"]
                ),
                "test_prime_identities": int(
                    self.config["data"]["test_prime_identities"]
                ),
            }
        }
        return self.exp300.bigint_prime_split(local, self.seed + 64700 + width)

    def new_model(self):
        model = self.cellular.SharedDilatedConvGRU(
            width=128,
            hidden=int(self.config["model"]["hidden"]),
            cycles=int(self.config["model"]["cycles"]),
        )
        model.load_state_dict(load_file(self.source))
        return model

    def check_runaway(self) -> None:
        if self.mode != "formal":
            return
        resolved = self.config["resource_budget"]["resolved"]
        wall = time.perf_counter() - self.started
        cpu = time.process_time() - self.cpu_started
        if wall > float(resolved["hard_runaway_ceiling_wall_seconds"]):
            raise TimeoutError("EXP-647 reached its formal wall runaway ceiling")
        if cpu > float(
            resolved["hard_runaway_ceiling_aggregate_cpu_seconds"]
        ):
            raise TimeoutError("EXP-647 reached its formal aggregate CPU ceiling")

    def training_update(
        self,
        model,
        optimizer,
        width: int,
        left: list[int],
        right: list[int],
        moduli: list[int],
        consistency_weight: float,
    ) -> dict[str, Any]:
        batch = len(left)
        original_state = torch.zeros(batch, width)
        swapped_state = torch.zeros(batch, width)
        original_teacher = [0] * batch
        swapped_teacher = [0] * batch
        original_minimum_margin = torch.full((batch,), float("inf"))
        swapped_minimum_margin = torch.full((batch,), float("inf"))
        supervised_losses: list[torch.Tensor] = []
        original_exact_sum = 0.0
        swapped_exact_sum = 0.0
        final_original_logits = None
        final_swapped_logits = None
        final_consistency = None
        swapped_teacher_rows = None
        chunk = int(self.config["training"]["bptt_chunk_steps"])
        optimizer.zero_grad(set_to_none=True)
        model.width = width
        for position in range(width):
            original_bits = [
                (value >> (width - 1 - position)) & 1 for value in left
            ]
            swapped_bits = [
                (value >> (width - 1 - position)) & 1 for value in right
            ]
            original_raw = [
                2 * state + bit * value
                for state, bit, value in zip(
                    original_teacher, original_bits, right, strict=True
                )
            ]
            swapped_raw = [
                2 * state + bit * value
                for state, bit, value in zip(
                    swapped_teacher, swapped_bits, left, strict=True
                )
            ]
            original_targets = [
                value % modulus
                for value, modulus in zip(original_raw, moduli, strict=True)
            ]
            swapped_targets = [
                value % modulus
                for value, modulus in zip(swapped_raw, moduli, strict=True)
            ]
            original_quotients = torch.tensor(
                [
                    value // modulus
                    for value, modulus in zip(original_raw, moduli, strict=True)
                ],
                dtype=torch.long,
            )
            swapped_quotients = torch.tensor(
                [
                    value // modulus
                    for value, modulus in zip(swapped_raw, moduli, strict=True)
                ],
                dtype=torch.long,
            )
            original_logits, original_qlogits = model(
                soft_features(
                    original_state,
                    right,
                    moduli,
                    original_bits,
                    width,
                    self.exp300.bits,
                )
            )
            swapped_logits, swapped_qlogits = model(
                soft_features(
                    swapped_state,
                    left,
                    moduli,
                    swapped_bits,
                    width,
                    self.exp300.bits,
                )
            )
            original_loss, original_stats = self.exp300.transition_loss(
                original_logits,
                original_qlogits,
                original_targets,
                original_quotients,
                width,
                float(self.config["training"]["worst_bit_weight"]),
                float(self.config["training"]["quotient_weight"]),
            )
            swapped_loss, swapped_stats = self.exp300.transition_loss(
                swapped_logits,
                swapped_qlogits,
                swapped_targets,
                swapped_quotients,
                width,
                float(self.config["training"]["worst_bit_weight"]),
                float(self.config["training"]["quotient_weight"]),
            )
            supervised_losses.append((original_loss + swapped_loss) * 0.5)
            original_exact_sum += original_stats["batch_exact"]
            swapped_exact_sum += swapped_stats["batch_exact"]
            original_minimum_margin = torch.minimum(
                original_minimum_margin,
                original_logits.detach().abs().amin(dim=1),
            )
            swapped_minimum_margin = torch.minimum(
                swapped_minimum_margin,
                swapped_logits.detach().abs().amin(dim=1),
            )
            original_probability = torch.sigmoid(original_logits)
            swapped_probability = torch.sigmoid(swapped_logits)
            original_state = (
                original_logits.ge(0).float()
                + original_probability
                - original_probability.detach()
            )
            swapped_state = (
                swapped_logits.ge(0).float()
                + swapped_probability
                - swapped_probability.detach()
            )
            original_teacher = original_targets
            swapped_teacher = swapped_targets
            final_original_logits = original_logits
            final_swapped_logits = swapped_logits
            boundary = (position + 1) % chunk == 0 or position + 1 == width
            if boundary:
                block_fraction = len(supervised_losses) / width
                block_loss = torch.stack(supervised_losses).mean() * block_fraction
                if position + 1 == width:
                    final_consistency, swapped_teacher_rows = confidence_teacher_loss(
                        final_original_logits,
                        final_swapped_logits,
                        original_minimum_margin,
                        swapped_minimum_margin,
                    )
                    block_loss = block_loss + consistency_weight * final_consistency
                block_loss.backward()
                supervised_losses.clear()
                original_state = original_state.detach()
                swapped_state = swapped_state.detach()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(
                model.parameters(), float(self.config["training"]["gradient_clip"])
            )
        )
        optimizer.step()
        assert final_consistency is not None and swapped_teacher_rows is not None
        assert final_original_logits is not None and final_swapped_logits is not None
        final_targets = self.exp300.bits(
            [
                (a * b) % modulus
                for a, b, modulus in zip(left, right, moduli, strict=True)
            ],
            width,
        ).bool()
        return {
            "width": width,
            "gradient_norm": gradient_norm,
            "mean_original_prefix_exact": original_exact_sum / width,
            "mean_swapped_prefix_exact": swapped_exact_sum / width,
            "final_original_exact": float(
                (final_original_logits.detach().ge(0) == final_targets)
                .all(dim=1)
                .float()
                .mean()
            ),
            "final_swapped_exact": float(
                (final_swapped_logits.detach().ge(0) == final_targets)
                .all(dim=1)
                .float()
                .mean()
            ),
            "confidence_consistency_bce": float(final_consistency.detach()),
            "swapped_teacher_fraction": float(swapped_teacher_rows.float().mean()),
        }

    def train_arm(
        self,
        arm_name: str,
        arm: dict[str, Any],
        arm_output: Path,
        schedule: list[int],
        use_cache: bool,
    ) -> dict[str, Any]:
        arm_output.mkdir(parents=True, exist_ok=True)
        cache_key = sha256_text(
            canonical_json(
                {
                    "arm": arm,
                    "schedule": schedule,
                    "scientific_config": self.scientific_hash,
                    "source": sha256(self.source),
                    "runner": self.runner_hash,
                    "dataset": self.dataset_hash,
                }
            )
        )
        checkpoint = arm_output / "weights.safetensors"
        cache_path = arm_output / f"training-{cache_key}.json"
        if use_cache and cache_path.exists() and checkpoint.exists():
            payload = json.loads(cache_path.read_text())
            if (
                payload.get("cache_key") == cache_key
                and payload["result"]["checkpoint"]["sha256"] == sha256(checkpoint)
            ):
                return {**payload["result"], "cached": True}

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        model = self.new_model()
        initial_checkpoint = arm_output / "initial.safetensors"
        save_file(
            {
                key: value.detach().contiguous()
                for key, value in model.state_dict().items()
            },
            initial_checkpoint,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(schedule),
            eta_min=float(self.config["training"]["minimum_learning_rate"]),
        )
        samplers = {
            width: self.exp300.BigIntSampler(
                self.splits[width]["train"], self.seed + 647100 + width
            )
            for width in self.widths
        }
        arm_started = time.perf_counter()
        arm_cpu_started = time.process_time()
        history: list[dict[str, Any]] = []
        per_width_timing = {
            str(width): {
                "updates": 0,
                "wall_seconds": 0.0,
                "aggregate_cpu_seconds": 0.0,
            }
            for width in self.widths
        }
        consistency_weight = float(arm["confidence_consistency_weight"])
        model.train()
        for step, width in enumerate(schedule, start=1):
            update_started = time.perf_counter()
            update_cpu_started = time.process_time()
            left, right, moduli = samplers[width].pairs(
                int(self.config["training"]["batch_size"])
            )
            stats = self.training_update(
                model,
                optimizer,
                width,
                left,
                right,
                moduli,
                consistency_weight,
            )
            scheduler.step()
            timing = per_width_timing[str(width)]
            timing["updates"] += 1
            timing["wall_seconds"] += time.perf_counter() - update_started
            timing["aggregate_cpu_seconds"] += time.process_time() - update_cpu_started
            log_every = int(self.config["training"]["log_every_steps"])
            if step == 1 or step % log_every == 0 or step == len(schedule):
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
                record = {
                    "step": step,
                    "elapsed_wall_seconds": time.perf_counter() - arm_started,
                    "learning_rate": scheduler.get_last_lr()[0],
                    **stats,
                }
                history.append(record)
                if mlflow.active_run() is not None:
                    mlflow.log_metrics(
                        {
                            "train_confidence_consistency_bce": stats[
                                "confidence_consistency_bce"
                            ],
                            "train_final_original_exact": stats[
                                "final_original_exact"
                            ],
                            "train_final_swapped_exact": stats["final_swapped_exact"],
                            "gradient_norm": stats["gradient_norm"],
                        },
                        step=step,
                    )
                print(json.dumps({"arm": arm_name, **record}), flush=True)
            self.check_runaway()
        model.eval()
        save_file(
            {
                key: value.detach().contiguous()
                for key, value in model.state_dict().items()
            },
            checkpoint,
        )
        result = {
            "arm": arm_name,
            "cached": False,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "confidence_consistency_weight": consistency_weight,
            "initial_checkpoint": {
                "path": str(initial_checkpoint.relative_to(ROOT)),
                "sha256": sha256(initial_checkpoint),
                "bytes": initial_checkpoint.stat().st_size,
            },
            "training": {
                "steps": len(schedule),
                "schedule": schedule,
                "trajectory_pairs": len(schedule)
                * int(self.config["training"]["batch_size"]),
                "oriented_transition_examples": sum(schedule)
                * int(self.config["training"]["batch_size"])
                * 2,
                "history": history,
                "per_width_timing": per_width_timing,
                "wall_seconds": time.perf_counter() - arm_started,
                "aggregate_cpu_seconds": time.process_time() - arm_cpu_started,
            },
            "checkpoint": {
                "path": str(checkpoint.relative_to(ROOT)),
                "sha256": sha256(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
            "peak_rss_bytes_after_training": self.peak_rss,
        }
        cache_path.write_text(
            json.dumps({"cache_key": cache_key, "result": result}, indent=2) + "\n"
        )
        return result

    def rollout_orientation(
        self,
        model,
        multipliers: list[int],
        multiplicands: list[int],
        moduli: list[int],
        width: int,
    ) -> dict[str, Any]:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        predictions: list[int] = []
        margins: list[float] = []
        invalid_flags: list[bool] = []
        all_prefix_flags: list[bool] = []
        first_errors: list[int | None] = []
        batch_size = int(self.config["evaluation"]["rollout_batch_size"])
        model.width = width
        with torch.inference_mode():
            if multipliers:
                warm_size = min(2, len(multipliers))
                warm_bits = [
                    (value >> (width - 1)) & 1 for value in multipliers[:warm_size]
                ]
                model(
                    self.exp300.features(
                        [0] * warm_size,
                        multiplicands[:warm_size],
                        moduli[:warm_size],
                        warm_bits,
                        width,
                    )
                )
            for start in range(0, len(multipliers), batch_size):
                end = min(start + batch_size, len(multipliers))
                left = multipliers[start:end]
                right = multiplicands[start:end]
                prime = moduli[start:end]
                size = len(left)
                state = [0] * size
                teacher = [0] * size
                minimum_margin = torch.full((size,), float("inf"))
                invalid = [False] * size
                every = [True] * size
                first_error: list[int | None] = [None] * size
                for position in range(width):
                    bits = [
                        (value >> (width - 1 - position)) & 1 for value in left
                    ]
                    logits, _ = model(
                        self.exp300.features(state, right, prime, bits, width)
                    )
                    state = decode_logits(logits)
                    teacher = [
                        (2 * value + bit * other) % modulus
                        for value, bit, other, modulus in zip(
                            teacher, bits, right, prime, strict=True
                        )
                    ]
                    minimum_margin = torch.minimum(
                        minimum_margin, logits.abs().amin(dim=1).cpu()
                    )
                    for row, (guess, truth, modulus) in enumerate(
                        zip(state, teacher, prime, strict=True)
                    ):
                        if guess != truth:
                            every[row] = False
                            if first_error[row] is None:
                                first_error[row] = position
                        if guess >= modulus:
                            invalid[row] = True
                predictions.extend(state)
                margins.extend(float(value) for value in minimum_margin.tolist())
                invalid_flags.extend(invalid)
                all_prefix_flags.extend(every)
                first_errors.extend(first_error)
        targets = [
            (left * right) % modulus
            for left, right, modulus in zip(
                multipliers, multiplicands, moduli, strict=True
            )
        ]
        errors = [position for position in first_errors if position is not None]
        cases = len(targets)
        return {
            "cases": cases,
            "correct": sum(
                prediction == target
                for prediction, target in zip(predictions, targets, strict=True)
            ),
            "exact_accuracy": sum(
                prediction == target
                for prediction, target in zip(predictions, targets, strict=True)
            )
            / cases,
            "all_prefix_exact_accuracy": sum(all_prefix_flags) / cases,
            "any_invalid_residue_rate": sum(invalid_flags) / cases,
            "mean_first_error_position": (
                sum(errors) / len(errors) if errors else None
            ),
            "predictions": predictions,
            "minimum_margins": margins,
            "invalid": invalid_flags,
            "first_errors": first_errors,
            "timing": {
                "wall_seconds": time.perf_counter() - wall_started,
                "aggregate_cpu_seconds": time.process_time() - cpu_started,
            },
        }

    @staticmethod
    def compact_rollout(result: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in result.items()
            if key
            not in {"predictions", "minimum_margins", "invalid", "first_errors"}
        }

    def evaluate_arm(
        self,
        arm_name: str,
        training_result: dict[str, Any],
        cases_override: dict[str, int] | None,
        use_cache: bool,
        run_determinism: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        checkpoint = ROOT / training_result["checkpoint"]["path"]
        evaluation_key = sha256_text(
            canonical_json(
                {
                    "checkpoint": sha256(checkpoint),
                    "scientific_config": self.scientific_hash,
                    "runner": self.runner_hash,
                    "dataset": self.dataset_hash,
                    "cases_override": cases_override,
                    "determinism": run_determinism,
                }
            )
        )
        arm_dir = checkpoint.parent
        cache_path = arm_dir / f"evaluation-{evaluation_key}.json"
        if use_cache and cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_key") == evaluation_key:
                return payload["summary"], payload["evidence"]

        model = self.new_model()
        model.load_state_dict(load_file(checkpoint))
        model.eval()
        summary: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        for width in self.widths:
            if cases_override:
                transition_cases = int(cases_override["transition"])
                rollout_cases = int(cases_override["rollout"])
            else:
                transition_cases = int(
                    by_width(self.config["evaluation"]["transition_cases"], width)
                )
                rollout_cases = int(
                    by_width(
                        self.config["evaluation"]["paired_rollout_cases"], width
                    )
                )
            transition_started = time.perf_counter()
            transition_cpu_started = time.process_time()
            model.width = width
            transition = self.exp300.evaluate_transitions(
                model,
                self.exp300.BigIntSampler(
                    self.splits[width]["test"], self.seed + 647700 + width
                ),
                transition_cases,
            )
            transition["timing"] = {
                "wall_seconds": time.perf_counter() - transition_started,
                "aggregate_cpu_seconds": time.process_time()
                - transition_cpu_started,
            }
            all_left, all_right, all_moduli = self.evaluation_pairs[width]
            left = all_left[:rollout_cases]
            right = all_right[:rollout_cases]
            moduli = all_moduli[:rollout_cases]
            targets = [
                (a * b) % modulus
                for a, b, modulus in zip(left, right, moduli, strict=True)
            ]
            original = self.rollout_orientation(model, left, right, moduli, width)
            swapped = self.rollout_orientation(model, right, left, moduli, width)
            paired = paired_rollout_summary(
                original["predictions"],
                swapped["predictions"],
                original["minimum_margins"],
                swapped["minimum_margins"],
                targets,
            )
            summary[str(width)] = {
                "transition": transition,
                "original": self.compact_rollout(original),
                "swapped": self.compact_rollout(swapped),
                "paired": {
                    key: value
                    for key, value in paired.items()
                    if key not in {"use_swapped", "selected_predictions"}
                },
            }
            evidence[str(width)] = {
                "left": left,
                "right": right,
                "moduli": moduli,
                "targets": targets,
                "original": {
                    "predictions": original["predictions"],
                    "minimum_margins": original["minimum_margins"],
                    "invalid": original["invalid"],
                    "first_errors": original["first_errors"],
                },
                "swapped": {
                    "predictions": swapped["predictions"],
                    "minimum_margins": swapped["minimum_margins"],
                    "invalid": swapped["invalid"],
                    "first_errors": swapped["first_errors"],
                },
                "selected": {
                    "predictions": paired["selected_predictions"],
                    "use_swapped": paired["use_swapped"],
                },
            }
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.check_runaway()

        determinism = None
        if run_determinism:
            width = int(self.config["evaluation"]["determinism_width"])
            cases = int(self.config["evaluation"]["determinism_cases"])
            left, right, moduli = self.evaluation_pairs[width]
            first = self.rollout_orientation(
                model, left[:cases], right[:cases], moduli[:cases], width
            )
            second = self.rollout_orientation(
                model, left[:cases], right[:cases], moduli[:cases], width
            )
            determinism = {
                "width": width,
                "cases": cases,
                "predictions_equal": first["predictions"] == second["predictions"],
                "margins_equal": first["minimum_margins"]
                == second["minimum_margins"],
                "invalid_equal": first["invalid"] == second["invalid"],
                "passed": first["predictions"] == second["predictions"]
                and first["minimum_margins"] == second["minimum_margins"]
                and first["invalid"] == second["invalid"],
                "first_timing": first["timing"],
                "second_timing": second["timing"],
            }
        summary["determinism"] = determinism
        cache_path.write_text(
            json.dumps(
                {"cache_key": evaluation_key, "summary": summary, "evidence": evidence},
                indent=2,
            )
            + "\n"
        )
        return summary, evidence

    def validate_formal_preconditions(self) -> dict[str, Any]:
        resource = self.config["resource_budget"]
        if resource.get("calibration_status") != "complete" or not resource.get(
            "resolved"
        ):
            raise RuntimeError("formal execution requires a committed complete calibration")
        if git_output("status", "--porcelain"):
            raise RuntimeError("formal execution requires a clean repository")
        calibration_path = (ROOT / resource["calibration_output"]).resolve()
        if not calibration_path.exists():
            raise RuntimeError("calibration artifact is missing")
        if sha256(calibration_path) != resource["resolved"][
            "calibration_output_sha256"
        ]:
            raise RuntimeError("calibration artifact hash mismatch")
        calibration = json.loads(calibration_path.read_text())
        if calibration["source"]["runner_sha256"] != self.runner_hash:
            raise RuntimeError("runner changed after calibration")
        if calibration["source"]["scientific_config_sha256"] != self.scientific_hash:
            raise RuntimeError("scientific config changed after calibration")
        if calibration["source"]["source_checkpoint_sha256"] != sha256(self.source):
            raise RuntimeError("source checkpoint changed after calibration")
        return calibration

    def calibrate(self) -> dict[str, Any]:
        if git_output("status", "--porcelain"):
            raise RuntimeError("calibration requires a clean repository")
        calibration_dir = self.output / "calibration"
        measured_arms: dict[str, Any] = {}
        updates = int(self.config["calibration"]["updates_per_width_per_arm"])
        schedule = [
            width for width in self.widths for _ in range(updates)
        ]
        profile_cases = {
            "transition": int(
                self.config["calibration"]["transition_cases_per_width_per_arm"]
            ),
            "rollout": int(
                self.config["calibration"]["paired_rollout_cases_per_width_per_arm"]
            ),
        }
        for arm_name, arm in self.config["arms"].items():
            training = self.train_arm(
                arm_name,
                arm,
                calibration_dir / arm_name,
                schedule,
                use_cache=False,
            )
            evaluation, _ = self.evaluate_arm(
                arm_name,
                training,
                cases_override=profile_cases,
                use_cache=False,
                run_determinism=False,
            )
            measured_arms[arm_name] = {"training": training["training"], "evaluation": evaluation}
        resolved = resolve_budget(self.config, measured_arms)
        result = {
            "run_id": self.config["run_id"],
            "status": "complete",
            "role": "representative-throughput-calibration-not-model-evidence",
            "source": {
                "repository_commit": self.repository_commit,
                "repository_dirty": False,
                "runner_sha256": self.runner_hash,
                "config_sha256": self.config_hash,
                "scientific_config_sha256": self.scientific_hash,
                "preregistration_sha256": self.preregistration_hash,
                "source_checkpoint_sha256": sha256(self.source),
                "dataset_sha256": self.dataset_hash,
            },
            "profile": {
                "updates_per_width_per_arm": updates,
                "transition_cases_per_width_per_arm": profile_cases["transition"],
                "paired_rollout_cases_per_width_per_arm": profile_cases["rollout"],
                "measured_arms": measured_arms,
            },
            "resolved_budget": resolved,
            "runtime": {
                "wall_seconds": time.perf_counter() - self.started,
                "aggregate_cpu_seconds": time.process_time() - self.cpu_started,
                "peak_rss_bytes": self.peak_rss,
            },
            "limitations": [
                "Calibration weights are discarded and provide no accuracy evidence.",
                "Projection is componentwise linear in measured updates and cases with explicit safety and setup headroom.",
                "Formal fidelity is unchanged.",
            ],
        }
        output_path = (ROOT / self.config["resource_budget"]["calibration_output"]).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"calibration": result["resolved_budget"], "runtime": result["runtime"]}, indent=2))
        return result

    def formal(self) -> dict[str, Any]:
        calibration = self.validate_formal_preconditions()
        formal_dir = self.output / "formal"
        schedule = [
            int(self.config["training"]["width_schedule"][(step - 1) % len(self.config["training"]["width_schedule"])])
            for step in range(1, int(self.config["training"]["steps"]) + 1)
        ]
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        results: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        run_ids: dict[str, str] = {}
        with mlflow.start_run(run_name=self.config["run_id"]) as parent:
            parent_id = parent.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "status": "running",
                    "role": "tier7-commutative-endpoint-consistency-ab",
                    "repository_commit": self.repository_commit,
                    "repository_dirty": "false",
                    "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                }
            )
            mlflow.log_params(
                {
                    "seed": self.seed,
                    "steps": len(schedule),
                    "source_checkpoint_sha256": sha256(self.source),
                    "config_sha256": self.config_hash,
                    "scientific_config_sha256": self.scientific_hash,
                    "runner_sha256": self.runner_hash,
                    "dataset_sha256": self.dataset_hash,
                }
            )
            mlflow.log_artifact(str(self.config_path), artifact_path="preregistration")
            mlflow.log_artifact(str(self.preregistration), artifact_path="preregistration")
            for arm_name, arm in self.config["arms"].items():
                with mlflow.start_run(run_name=arm_name, nested=True) as nested:
                    run_ids[arm_name] = nested.info.run_id
                    training = self.train_arm(
                        arm_name,
                        arm,
                        formal_dir / arm_name,
                        schedule,
                        use_cache=True,
                    )
                    evaluation, arm_evidence = self.evaluate_arm(
                        arm_name,
                        training,
                        cases_override=None,
                        use_cache=True,
                        run_determinism=arm_name == "confidence_consistency",
                    )
                    results[arm_name] = {**training, "evaluation": evaluation}
                    evidence[arm_name] = arm_evidence
                    mlflow.log_metrics(
                        {
                            "width64_original_rollout": evaluation["64"]["original"]["exact_accuracy"],
                            "width128_original_rollout": evaluation["128"]["original"]["exact_accuracy"],
                            "width256_original_rollout": evaluation["256"]["original"]["exact_accuracy"],
                            "width256_transition": evaluation["256"]["transition"]["exact_accuracy"],
                        }
                    )
                    mlflow.set_tag("status", "completed")
        control = results["paired_rollout_control"]["evaluation"]
        challenger = results["confidence_consistency"]["evaluation"]
        paired_comparison = {
            str(width): paired_arm_comparison(
                evidence["paired_rollout_control"][str(width)]["original"][
                    "predictions"
                ],
                evidence["confidence_consistency"][str(width)]["original"][
                    "predictions"
                ],
                evidence["confidence_consistency"][str(width)]["targets"],
            )
            for width in self.widths
        }
        comparison = {
            "control_width128_original_rollout": control["128"]["original"]["exact_accuracy"],
            "challenger_width128_original_rollout": challenger["128"]["original"]["exact_accuracy"],
            "width128_original_gain": challenger["128"]["original"]["exact_accuracy"]
            - control["128"]["original"]["exact_accuracy"],
            "control_width256_original_rollout": control["256"]["original"]["exact_accuracy"],
            "challenger_width256_original_rollout": challenger["256"]["original"]["exact_accuracy"],
            "width256_original_gain": challenger["256"]["original"]["exact_accuracy"]
            - control["256"]["original"]["exact_accuracy"],
            "challenger_width64_original_rollout": challenger["64"]["original"]["exact_accuracy"],
            "challenger_width128_transition": challenger["128"]["transition"]["exact_accuracy"],
            "challenger_width256_transition": challenger["256"]["transition"]["exact_accuracy"],
            "challenger_original_invalid_rates": {
                str(width): challenger[str(width)]["original"]["any_invalid_residue_rate"]
                for width in self.widths
            },
            "paired_arm_comparison": paired_comparison,
        }
        incremental_runtime = (
            challenger["256"]["original"]["timing"]["wall_seconds"]
            / challenger["256"]["original"]["cases"]
            * 100
        )
        gates = self.config["gates"]
        gate_results = {
            "width128_absolute": comparison["challenger_width128_original_rollout"]
            >= float(gates["minimum_challenger_width128_original_rollout"]),
            "width128_gain": comparison["width128_original_gain"]
            >= float(gates["minimum_width128_original_gain_over_control"]),
            "width256_absolute": comparison["challenger_width256_original_rollout"]
            >= float(gates["minimum_challenger_width256_original_rollout"]),
            "width256_gain": comparison["width256_original_gain"]
            >= float(gates["minimum_width256_original_gain_over_control"]),
            "width64_retention": comparison["challenger_width64_original_rollout"]
            >= float(gates["minimum_challenger_width64_original_rollout"]),
            "width128_transition": comparison["challenger_width128_transition"]
            >= float(gates["minimum_challenger_width128_transition"]),
            "width256_transition": comparison["challenger_width256_transition"]
            >= float(gates["minimum_challenger_width256_transition"]),
            "validity": max(comparison["challenger_original_invalid_rates"].values())
            <= float(gates["maximum_challenger_invalid_rate"]),
            "determinism": bool(challenger["determinism"]["passed"]),
            "runtime": incremental_runtime
            <= float(gates["maximum_incremental_width256_100_case_wall_seconds"]),
        }
        accuracy_keys = {
            "width128_absolute",
            "width128_gain",
            "width256_absolute",
            "width256_gain",
            "width64_retention",
            "width128_transition",
            "width256_transition",
            "validity",
            "determinism",
        }
        accuracy_passed = all(gate_results[key] for key in accuracy_keys)
        passed = accuracy_passed and gate_results["runtime"]
        if passed:
            decision = "advance-commutative-consistency-independent-tier7"
        elif accuracy_passed:
            decision = "accuracy-supported-runtime-blocked-optimize-exact-inference"
        else:
            decision = "reject-current-commutative-consistency-formulation"
        raw_evidence_path = formal_dir / "paired-evidence.json"
        raw_evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        summary = {
            "run_id": self.config["run_id"],
            "status": "completed",
            "hypothesis": self.config["hypothesis"],
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": self.config["tracking"]["experiment_name"],
                "experiment_id": experiment_id,
                "parent_run_id": parent_id,
                "arm_run_ids": run_ids,
            },
            "source": {
                "repository_commit": self.repository_commit,
                "dirty": bool(git_output("status", "--porcelain")),
                "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                "config_sha256": self.config_hash,
                "scientific_config_sha256": self.scientific_hash,
                "runner_sha256": self.runner_hash,
                "preregistration_sha256": self.preregistration_hash,
                "dataset_sha256": self.dataset_hash,
                "source_checkpoint_sha256": sha256(self.source),
            },
            "arms": results,
            "comparison": comparison,
            "runtime_probe": {
                "width": 256,
                "measured_cases": challenger["256"]["original"]["cases"],
                "measured_wall_seconds": challenger["256"]["original"]["timing"]["wall_seconds"],
                "measured_aggregate_cpu_seconds": challenger["256"]["original"]["timing"]["aggregate_cpu_seconds"],
                "projected_100_case_wall_seconds": incremental_runtime,
            },
            "gates": {
                **gates,
                "results": gate_results,
                "accuracy_passed": accuracy_passed,
                "passed": passed,
            },
            "decision": decision,
            "resource": {
                "wall_seconds": time.perf_counter() - self.started,
                "aggregate_cpu_seconds": time.process_time() - self.cpu_started,
                "peak_rss_bytes": self.peak_rss,
                "resolved_budget": self.config["resource_budget"]["resolved"],
                "calibration_output_sha256": sha256(
                    (ROOT / self.config["resource_budget"]["calibration_output"]).resolve()
                ),
            },
            "artifacts": {
                "paired_evidence": {
                    "path": str(raw_evidence_path.relative_to(ROOT)),
                    "sha256": sha256(raw_evidence_path),
                }
            },
            "calibration": calibration["resolved_budget"],
            "limitations": [
                "One matched training seed is a selection rung, not independent confirmation.",
                "Fresh generated identity-disjoint evidence only; checked, prior independent-full, official, and private answers were not read.",
                "The final consistency teacher is target-free but confidence can still be wrong; paired exact scoring measures that failure directly.",
                "Exact bigint arithmetic is confined to offline labels and evaluation truth.",
            ],
        }
        summary_path = self.output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        with mlflow.start_run(run_id=parent_id):
            mlflow.log_metrics(
                {
                    "width128_original_gain": comparison["width128_original_gain"],
                    "width256_original_gain": comparison["width256_original_gain"],
                    "challenger_width256_original_rollout": comparison[
                        "challenger_width256_original_rollout"
                    ],
                    "incremental_width256_100_case_wall_seconds": incremental_runtime,
                    "wall_seconds": summary["resource"]["wall_seconds"],
                    "aggregate_cpu_seconds": summary["resource"][
                        "aggregate_cpu_seconds"
                    ],
                }
            )
            mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
            mlflow.log_artifact(
                str(raw_evidence_path), artifact_path="canonical-preview"
            )
            mlflow.set_tags({"status": "completed", "decision": decision})
        print(
            json.dumps(
                {
                    "decision": decision,
                    "comparison": comparison,
                    "runtime_probe": summary["runtime_probe"],
                    "gates": summary["gates"],
                    "resource": summary["resource"],
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
        default=ROOT / "configs/exp_647_operand_swap_consistency.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-647-operand-swap-consistency",
    )
    parser.add_argument("--mode", choices=("calibration", "formal"), required=True)
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode)
    if args.mode == "calibration":
        experiment.calibrate()
    else:
        experiment.formal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
