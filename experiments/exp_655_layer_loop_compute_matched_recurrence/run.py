"""Calibrate and execute the full-fidelity EXP-655 dense schedule falsifier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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
from experiments.exp_655_layer_loop_compute_matched_recurrence.model import (  # noqa: E402
    ARM_ORDER,
    PRIMARY_ARMS,
    LayerLoopReasoner,
    state_bytes,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


CONFIG_PATH = ROOT / "configs/exp_655_layer_loop_compute_matched_recurrence.yaml"


@dataclass(frozen=True)
class RootSpec:
    root: int
    init_seed: int
    train_seed: int
    eval_seed: int


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file():
        return None
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = line.split()
            values[key] = int(value)
    except (OSError, ValueError):
        return None
    return values.get("usage_usec", 0) / 1_000_000.0


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class Timer:
    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cpu = process_cpu_seconds()
        self.cgroup = cgroup_cpu_seconds()

    def finish(self) -> dict[str, float | int | None]:
        current_cgroup = cgroup_cpu_seconds()
        return {
            "wall_seconds": time.perf_counter() - self.wall,
            "process_cpu_seconds": process_cpu_seconds() - self.cpu,
            "cgroup_cpu_seconds": (
                None
                if self.cgroup is None or current_cgroup is None
                else current_cgroup - self.cgroup
            ),
            "peak_rss_bytes": peak_rss_bytes(),
        }


def bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def decode_bits(rows: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in rows.tolist()
    ]


def features(
    state: list[int],
    multiplicand: list[int],
    modulus: list[int],
    input_bit: list[int],
    width: int,
) -> torch.Tensor:
    batch = len(state)
    bit_plane = torch.tensor(input_bit, dtype=torch.float32)[:, None].expand(
        batch, width
    )
    lsb = torch.zeros(batch, width)
    msb = torch.zeros(batch, width)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (
            bits(state, width),
            bits(multiplicand, width),
            bits(modulus, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul(2.0).sub(1.0)


def transition_loss(
    logits: torch.Tensor,
    quotient_logits: torch.Tensor,
    targets: list[int],
    quotients: torch.Tensor,
    width: int,
    training: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    target = bits(targets, width)
    per_bit = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mean = per_bit.mean()
    worst = per_bit.mean(0).max()
    quotient = F.cross_entropy(quotient_logits, quotients)
    loss = (
        mean
        + float(training["worst_bit_weight"]) * worst
        + float(training["quotient_weight"]) * quotient
    )
    exact = (logits.ge(0) == target.bool()).all(1).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "batch_exact": float(exact.detach()),
        "mean_bce": float(mean.detach()),
        "worst_bit_bce": float(worst.detach()),
        "quotient_ce": float(quotient.detach()),
    }


def hierarchical_paired_bootstrap(
    deltas: np.ndarray, *, seed: int, resamples: int = 100_000
) -> dict[str, Any]:
    """Equal-cluster hierarchical bootstrap with bounded-memory chunks."""
    if deltas.ndim != 2 or deltas.shape[0] != 3 or deltas.shape[1] != 300:
        raise ValueError("bootstrap requires exactly three clusters of 300 pairs")
    if resamples != 100_000:
        raise ValueError("EXP-655 requires exactly 100,000 resamples")
    timer = Timer()
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    chunk = 1_000
    for start in range(0, resamples, chunk):
        count = min(chunk, resamples - start)
        clusters = rng.integers(0, 3, size=(count, 3), dtype=np.int16)
        cluster_means = np.empty((count, 3), dtype=np.float64)
        for cluster_slot in range(3):
            identities = rng.integers(0, 300, size=(count, 300), dtype=np.int16)
            selected_clusters = clusters[:, cluster_slot]
            cluster_means[:, cluster_slot] = deltas[
                selected_clusters[:, None], identities
            ].mean(axis=1)
        values[start : start + count] = cluster_means.mean(axis=1)
    return {
        "clusters": 3,
        "paired_identities_per_cluster": 300,
        "resamples": resamples,
        "seed": seed,
        "mean": float(values.mean()),
        "one_sided_95_lower_bound": float(
            np.percentile(values, 5.0, method="linear")
        ),
        "distribution_sha256": sha256_bytes(values.tobytes()),
        "timing": timer.finish(),
    }


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.exp300 = load_module(
            "exp300_for_exp655",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        torch.use_deterministic_algorithms(True)
        self.started = Timer()
        self.source_at_start = self.source()

    def source(self) -> dict[str, Any]:
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "repository_branch": git_output("branch", "--show-current"),
            "dirty": bool(git_output("status", "--porcelain", "--untracked-files=normal")),
            "challenge_commit": self.config["source_contract"]["challenge_commit"],
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "model_sha256": sha256(Path(__file__).with_name("model.py")),
            "fixtures_sha256": sha256(Path(__file__).with_name("fixtures.py")),
            "preregistration_sha256": sha256(ROOT / self.config["preregistration"]),
            "paper_sha256": sha256(ROOT / self.config["paper"]["path"]),
        }

    def roots(self) -> list[RootSpec]:
        if self.mode == "calibration":
            values = self.config["calibration"]
            return [
                RootSpec(
                    root=int(values["root"]),
                    init_seed=int(values["init_seed"]),
                    train_seed=int(values["train_seed"]),
                    eval_seed=int(values["eval_seed"]),
                )
            ]
        return [RootSpec(**{key: int(value) for key, value in item.items()}) for item in self.config["data"]["formal_roots"]]

    def stages(self) -> list[tuple[int, int]]:
        if self.mode == "calibration":
            return [
                (int(width), int(steps))
                for width, steps in self.config["calibration"]["stage_steps"].items()
            ]
        return [
            (int(item["width"]), int(item["steps"]))
            for item in self.config["training"]["stages"]
        ]

    def make_model(self, arm: str) -> LayerLoopReasoner:
        values = self.config["model"]
        return LayerLoopReasoner(
            arm,
            hidden=int(values["hidden"]),
            heads=int(values["heads"]),
            stored_blocks=int(values["stored_blocks"]),
            expansion=int(values["expansion"]),
        )

    def make_split(self, width: int, seed: int) -> dict[str, list[int]]:
        minimum_bits = max(
            3, width - int(self.config["data"]["prime_bit_span"]) + 1
        )
        local = {
            "data": {
                "prime_bits_min": minimum_bits,
                "prime_bits_max": width,
                "train_prime_identities": int(
                    self.config["data"]["train_prime_identities"]
                ),
                "test_prime_identities": int(
                    self.config["data"]["test_prime_identities"]
                ),
            }
        }
        return self.exp300.bigint_prime_split(
            local, seed + int(self.config["data"]["split_seed_offset"]) + width
        )

    def root_splits(self, root: RootSpec) -> dict[int, dict[str, dict[str, list[int]]]]:
        widths = {width for width, _ in self.stages()}
        widths.update(int(value) for value in self.config["evaluation"]["widths"])
        widths.add(int(self.config["evaluation"]["runtime_probe_width"]))
        return {
            width: {
                "train": self.make_split(width, root.train_seed),
                "evaluation": self.make_split(width, root.eval_seed),
            }
            for width in sorted(widths)
        }

    def sampler(self, primes: list[int], seed: int):
        return self.exp300.BigIntSampler(primes, seed)

    def freeze_identity_manifest(
        self,
        root: RootSpec,
        splits: dict[int, dict[str, dict[str, list[int]]]],
    ) -> dict[str, Any]:
        cases = int(self.config["evaluation"]["rollout_cases"][16])
        identities: dict[str, list[dict[str, int]]] = {}
        for width in (16, 32):
            sampler = self.sampler(
                splits[width]["evaluation"]["test"],
                root.eval_seed
                + int(self.config["data"]["evaluation_pair_seed_offset"])
                + width,
            )
            multiplier, multiplicand, modulus = sampler.pairs(cases)
            rows = [
                {
                    "multiplier": int(left),
                    "multiplicand": int(right),
                    "modulus": int(prime),
                }
                for left, right, prime in zip(
                    multiplier, multiplicand, modulus, strict=True
                )
            ]
            if len({(row["multiplier"], row["multiplicand"], row["modulus"]) for row in rows}) != cases:
                raise RuntimeError("evaluation identity manifest contains duplicates")
            if set(splits[width]["evaluation"]["train"]).intersection(
                row["modulus"] for row in rows
            ):
                raise RuntimeError("evaluation identities are not train-prime-disjoint")
            identities[str(width)] = rows
        runtime_width = int(self.config["evaluation"]["runtime_probe_width"])
        runtime_cases = int(self.config["evaluation"]["runtime_probe_cases"])
        runtime_sampler = self.sampler(
            splits[runtime_width]["evaluation"]["test"],
            root.eval_seed
            + int(self.config["data"]["evaluation_pair_seed_offset"])
            + runtime_width,
        )
        left, right, modulus = runtime_sampler.pairs(runtime_cases)
        runtime_rows = [
            {"multiplier": int(a), "multiplicand": int(b), "modulus": int(p)}
            for a, b, p in zip(left, right, modulus, strict=True)
        ]
        unhashed = {
            "schema_version": "exp655-evaluation-input-identities-v1",
            "root": asdict(root),
            "label_values_present": False,
            "train_prime_disjoint": True,
            "identities": identities,
            "runtime_width256_identities": runtime_rows,
        }
        payload = {**unhashed, "identity_sha256": sha256_json(unhashed)}
        path = self.output / "identities" / f"root-{root.root}.json"
        write_json(path, payload)
        payload["path"] = str(path.relative_to(ROOT))
        payload["file_sha256"] = sha256(path)
        return payload

    def primary_initial_state(self, root: RootSpec) -> tuple[dict[str, torch.Tensor], str]:
        torch.manual_seed(root.init_seed)
        base = self.make_model("layer_loop_r2")
        state = {
            key: value.detach().clone() for key, value in base.state_dict().items()
        }
        return state, sha256_bytes(state_bytes(base))

    def initial_model(
        self, arm: str, root: RootSpec, primary_state: dict[str, torch.Tensor]
    ) -> LayerLoopReasoner:
        if arm != "untied_equal_compute":
            model = self.make_model(arm)
            model.load_state_dict(copy.deepcopy(primary_state))
            return model
        torch.manual_seed(root.init_seed + 10_000)
        return self.make_model(arm)

    def validation_fixture(
        self,
        width: int,
        root: RootSpec,
        splits: dict[int, dict[str, dict[str, list[int]]]],
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        sampler = self.sampler(
            splits[width]["evaluation"]["test"], root.eval_seed + 2_000 + width
        )
        count = int(self.config["training"]["validation_cases"])
        state, multiplicand, modulus, input_bit, targets, quotients = sampler.transitions(
            count
        )
        return (
            features(state, multiplicand, modulus, input_bit, width),
            targets,
            quotients,
        )

    @torch.inference_mode()
    def validate_fixture(
        self,
        model: LayerLoopReasoner,
        fixture: tuple[torch.Tensor, list[int], torch.Tensor],
        width: int,
    ) -> dict[str, float]:
        tensor, targets, quotients = fixture
        logits, quotient_logits = model(tensor)
        target = bits(targets, width).bool()
        return {
            "exact_accuracy": float(
                (logits.ge(0) == target).all(1).float().mean()
            ),
            "bit_accuracy": float((logits.ge(0) == target).float().mean()),
            "quotient_accuracy": float(
                (quotient_logits.argmax(1) == quotients).float().mean()
            ),
        }

    def check_budget(self, *, preserved_checkpoint: bool) -> None:
        if not preserved_checkpoint:
            return
        elapsed = self.started.finish()
        if self.mode == "calibration":
            budget = {
                "wall_seconds": self.config["resource_budget"][
                    "calibration_operational_runaway_wall_seconds"
                ],
                "process_cpu_seconds": self.config["resource_budget"][
                    "calibration_operational_runaway_process_cpu_seconds"
                ],
                "cgroup_cpu_seconds": self.config["resource_budget"][
                    "calibration_operational_runaway_cgroup_cpu_seconds"
                ],
                "peak_rss_bytes": None,
            }
        else:
            budget = self.load_admission()["hard_budget"]
        for key in ("wall_seconds", "process_cpu_seconds", "cgroup_cpu_seconds"):
            limit = budget.get(key)
            observed = elapsed.get(key)
            if limit is not None and observed is not None and float(observed) > float(limit):
                raise TimeoutError(f"EXP-655 {self.mode} {key} runaway guard exceeded")
        rss = budget.get("peak_rss_bytes")
        if rss is not None and peak_rss_bytes() > int(rss):
            raise MemoryError("EXP-655 peak RSS guard exceeded")

    def train_arm(
        self,
        arm: str,
        root: RootSpec,
        splits: dict[int, dict[str, dict[str, list[int]]]],
        primary_state: dict[str, torch.Tensor],
        primary_initial_sha256: str,
    ) -> dict[str, Any]:
        stages = self.stages()
        total_steps = sum(steps for _, steps in stages)
        batch_size = int(
            self.config["calibration"]["batch_size"]
            if self.mode == "calibration"
            else self.config["training"]["batch_size"]
        )
        source_identity = {
            "source": self.source_at_start,
            "mode": self.mode,
            "root": asdict(root),
            "arm": arm,
            "stages": stages,
            "batch_size": batch_size,
        }
        cache_key = sha256_json(source_identity)
        arm_dir = self.output / self.mode / f"root-{root.root}" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        result_path = arm_dir / f"result-{cache_key}.json"
        if (
            self.mode == "formal"
            and bool(self.config["resource_budget"]["reuse_hash_valid_completed_arms"])
            and result_path.is_file()
        ):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            result = payload.get("result", {})
            checkpoint = ROOT / result.get("checkpoint", {}).get("path", "missing")
            if (
                payload.get("cache_key") == cache_key
                and checkpoint.is_file()
                and sha256(checkpoint) == result["checkpoint"]["sha256"]
            ):
                return {**result, "cached": True}

        random.seed(root.train_seed)
        np.random.seed(root.train_seed)
        torch.manual_seed(root.init_seed)
        model = self.initial_model(arm, root, primary_state)
        initial_state_sha256 = sha256_bytes(state_bytes(model))
        if arm in PRIMARY_ARMS and initial_state_sha256 != primary_initial_sha256:
            raise RuntimeError("primary initial states are not byte-identical")
        initial_path = arm_dir / "initial.safetensors"
        save_file(
            {
                key: value.detach().contiguous()
                for key, value in model.state_dict().items()
            },
            initial_path,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(self.config["training"]["minimum_learning_rate"]),
        )
        samplers = {
            width: self.sampler(
                splits[width]["train"]["train"],
                root.train_seed
                + int(self.config["data"]["training_pair_seed_offset"])
                + width,
            )
            for width, _ in stages
        }
        fixtures = {
            width: self.validation_fixture(width, root, splits)
            for width, _ in stages
        }
        interval = (
            max(1, total_steps // 5)
            if self.mode == "calibration"
            else int(self.config["training"]["validation_interval"])
        )
        history: list[dict[str, Any]] = []
        stage_checkpoints: list[dict[str, Any]] = []
        timer = Timer()
        step = 0
        model.train()
        for width, stage_steps in stages:
            stage_timer = Timer()
            for stage_step in range(1, stage_steps + 1):
                step += 1
                state, multiplicand, modulus, input_bit, targets, quotients = samplers[
                    width
                ].transitions(batch_size)
                logits, quotient_logits = model(
                    features(state, multiplicand, modulus, input_bit, width)
                )
                loss, stats = transition_loss(
                    logits,
                    quotient_logits,
                    targets,
                    quotients,
                    width,
                    self.config["training"],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = float(
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(self.config["training"]["gradient_clip"]),
                    )
                )
                optimizer.step()
                scheduler.step()
                if (
                    step == 1
                    or step % interval == 0
                    or stage_step == stage_steps
                    or step == total_steps
                ):
                    model.eval()
                    validation = {
                        str(candidate_width): self.validate_fixture(
                            model, fixtures[candidate_width], candidate_width
                        )
                        for candidate_width, _ in stages
                        if candidate_width <= width
                    }
                    model.train()
                    record = {
                        "step": step,
                        "stage_width": width,
                        "stage_step": stage_step,
                        "elapsed_wall_seconds": timer.finish()["wall_seconds"],
                        "gradient_norm": gradient,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "train": stats,
                        "validation": validation,
                    }
                    history.append(record)
                    print(json.dumps({"root": root.root, "arm": arm, **record}), flush=True)
            checkpoint_path = arm_dir / f"stage-{width}-step-{step}.safetensors"
            persistence = Timer()
            save_file(
                {
                    key: value.detach().contiguous()
                    for key, value in model.state_dict().items()
                },
                checkpoint_path,
            )
            stage_checkpoints.append(
                {
                    "width": width,
                    "step": step,
                    "path": str(checkpoint_path.relative_to(ROOT)),
                    "sha256": sha256(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                    "stage_timing": stage_timer.finish(),
                    "persistence_timing": persistence.finish(),
                }
            )
            self.check_budget(preserved_checkpoint=True)
        final_path = arm_dir / "weights.safetensors"
        persistence = Timer()
        save_file(
            {
                key: value.detach().contiguous()
                for key, value in model.state_dict().items()
            },
            final_path,
        )
        result = {
            "root": root.root,
            "arm": arm,
            "cached": False,
            "parameters": model.parameter_count(),
            "complete_block_calls_per_transition": model.complete_block_calls(),
            "initial_checkpoint": {
                "path": str(initial_path.relative_to(ROOT)),
                "sha256": sha256(initial_path),
                "state_sha256": initial_state_sha256,
                "bytes": initial_path.stat().st_size,
            },
            "checkpoint": {
                "path": str(final_path.relative_to(ROOT)),
                "sha256": sha256(final_path),
                "bytes": final_path.stat().st_size,
                "persistence_timing": persistence.finish(),
            },
            "stage_checkpoints": stage_checkpoints,
            "training": {
                "stages": [
                    {"width": width, "steps": count} for width, count in stages
                ],
                "steps": total_steps,
                "batch_size": batch_size,
                "optimizer": "AdamW",
                "history": history,
                "timing": timer.finish(),
                "executed_complete_block_calls": (
                    total_steps
                    * batch_size
                    * model.complete_block_calls()
                ),
            },
        }
        write_json(result_path, {"cache_key": cache_key, "result": result})
        return result

    def load_model(self, arm_result: dict[str, Any]) -> LayerLoopReasoner:
        model = self.make_model(arm_result["arm"])
        model.load_state_dict(load_file(ROOT / arm_result["checkpoint"]["path"]))
        return model.eval()

    @torch.inference_mode()
    def evaluate_rollout(
        self,
        model: LayerLoopReasoner,
        identities: list[dict[str, int]],
        width: int,
        *,
        include_truth: bool,
        schedule: str | None = None,
        block_order: tuple[int, ...] | None = None,
        visits: int | None = None,
        reset_before_visit: int | None = None,
    ) -> dict[str, Any]:
        timer = Timer()
        multiplier = [row["multiplier"] for row in identities]
        multiplicand = [row["multiplicand"] for row in identities]
        modulus = [row["modulus"] for row in identities]
        state = [0] * len(identities)
        teacher = [0] * len(identities)
        every = [True] * len(identities)
        first = [-1] * len(identities)
        invalid = [False] * len(identities)
        prefix_correct = 0
        for position in range(width):
            input_bit = [
                (value >> (width - 1 - position)) & 1 for value in multiplier
            ]
            logits, _ = model(
                features(state, multiplicand, modulus, input_bit, width),
                schedule=schedule,
                block_order=block_order,
                visits=visits,
                reset_before_visit=reset_before_visit,
            )
            state = decode_bits(logits.ge(0))
            invalid = [
                old or predicted >= prime
                for old, predicted, prime in zip(
                    invalid, state, modulus, strict=True
                )
            ]
            if include_truth:
                teacher = [
                    (2 * current + bit * other) % prime
                    for current, bit, other, prime in zip(
                        teacher, input_bit, multiplicand, modulus, strict=True
                    )
                ]
                for index, (predicted, expected) in enumerate(
                    zip(state, teacher, strict=True)
                ):
                    exact = predicted == expected
                    prefix_correct += int(exact)
                    if every[index] and not exact:
                        every[index] = False
                        first[index] = position
        correct_vector: list[int] | None = None
        correct = None
        if include_truth:
            expected = [
                (left * right) % prime
                for left, right, prime in zip(
                    multiplier, multiplicand, modulus, strict=True
                )
            ]
            correct_vector = [
                int(predicted == target)
                for predicted, target in zip(state, expected, strict=True)
            ]
            correct = sum(correct_vector)
        return {
            "cases": len(identities),
            "correct": correct,
            "exact_accuracy": (
                None if correct is None else correct / len(identities)
            ),
            "correct_by_identity": correct_vector,
            "all_prefix_exact_accuracy": (
                None if not include_truth else sum(every) / len(identities)
            ),
            "prefix_transition_exact_accuracy": (
                None
                if not include_truth
                else prefix_correct / (len(identities) * width)
            ),
            "any_invalid_residue_rate": sum(invalid) / len(identities),
            "first_error": (
                None
                if not include_truth
                else {
                    "error_cases": sum(value >= 0 for value in first),
                    "histogram": {
                        str(value): first.count(value)
                        for value in sorted(set(first))
                        if value >= 0
                    },
                }
            ),
            "prediction_sha256": sha256_json(state),
            "timing": timer.finish(),
        }

    def noncommutation(self, model: LayerLoopReasoner, seed: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(seed)
        state = torch.randn(4, 11, model.hidden, generator=generator)
        with torch.inference_mode():
            b1_then_b2 = model.blocks[1](model.blocks[0](state))
            b2_then_b1 = model.blocks[0](model.blocks[1](state))
        residual = (b1_then_b2 - b2_then_b1).abs()
        return {
            "maximum_absolute_residual": float(residual.max()),
            "mean_absolute_residual": float(residual.mean()),
            "noncommuting": bool(float(residual.max()) > 1e-7),
        }

    def malformed_and_leakage(self, model: LayerLoopReasoner) -> dict[str, Any]:
        rejected = 0
        for tensor in (
            torch.zeros(2, 8),
            torch.zeros(2, 5, 8),
            torch.full((2, 6, 8), float("nan")),
        ):
            try:
                model(tensor)
            except ValueError:
                rejected += 1
        source = Path(__file__).with_name("model.py").read_text(encoding="utf-8").lower()
        forbidden = (
            "expected_answer",
            "barrett",
            "montgomery",
            "crt_recomb",
            "torch.remainder",
            "fmod(",
            "data/checked",
            "data/private",
            "submission/",
        )
        hits = [value for value in forbidden if value in source]
        return {
            "malformed_cases": 3,
            "malformed_rejected": rejected,
            "malformed_passed": rejected == 3,
            "static_forbidden_hits": hits,
            "leakage_passed": not hits,
        }

    def randomized_model(self, arm: str, seed: int) -> tuple[LayerLoopReasoner, str]:
        torch.manual_seed(seed)
        model = self.make_model(arm).eval()
        return model, sha256_bytes(state_bytes(model))

    def evaluate_root(
        self,
        root: RootSpec,
        arm_results: dict[str, dict[str, Any]],
        identity_manifest: dict[str, Any],
        splits: dict[int, dict[str, dict[str, list[int]]]],
        *,
        resource_only: bool,
        root_index: int,
    ) -> dict[str, Any]:
        identities = identity_manifest["identities"]
        models = {arm: self.load_model(result) for arm, result in arm_results.items()}
        evaluations: dict[str, Any] = {}
        for arm, model in models.items():
            evaluations[arm] = {
                str(width): self.evaluate_rollout(
                    model,
                    identities[str(width)],
                    width,
                    include_truth=not resource_only,
                )
                for width in (16, 32)
            }
            self.check_budget(preserved_checkpoint=True)

        interventions: dict[str, Any] = {}
        intervention_cases = int(self.config["evaluation"]["intervention_cases"])
        for arm in PRIMARY_ARMS:
            model = models[arm]
            opposite = (
                "model_loop_r2" if arm == "layer_loop_r2" else "layer_loop_r2"
            )
            local = random.Random(root.eval_seed + root_index + (0 if arm == PRIMARY_ARMS[0] else 100))
            order = list(model.schedule())
            local.shuffle(order)
            interventions[arm] = {}
            for width in (16, 32):
                full_rows = identities[str(width)]
                rows = full_rows[:intervention_cases]
                interventions[arm][str(width)] = {
                    "opposite_schedule": self.evaluate_rollout(
                        model,
                        full_rows,
                        width,
                        include_truth=not resource_only,
                        schedule=opposite,
                    ),
                    "block_randomization": self.evaluate_rollout(
                        model,
                        rows,
                        width,
                        include_truth=not resource_only,
                        block_order=tuple(order),
                    ),
                    "visit_truncation": self.evaluate_rollout(
                        model,
                        rows,
                        width,
                        include_truth=not resource_only,
                        visits=model.complete_block_calls() - 1,
                    ),
                    "hidden_reset": self.evaluate_rollout(
                        model,
                        rows,
                        width,
                        include_truth=not resource_only,
                        reset_before_visit=model.complete_block_calls() // 2,
                    ),
                }
        repeat_cases = int(self.config["evaluation"]["deterministic_repeat_cases"])
        deterministic = {}
        randomized = {}
        for arm_index, arm in enumerate(PRIMARY_ARMS):
            repeat_rows = identities["16"][:repeat_cases]
            first = self.evaluate_rollout(
                models[arm], repeat_rows, 16, include_truth=not resource_only
            )
            second = self.evaluate_rollout(
                models[arm], repeat_rows, 16, include_truth=not resource_only
            )
            deterministic[arm] = {
                "cases": repeat_cases,
                "first_prediction_sha256": first["prediction_sha256"],
                "second_prediction_sha256": second["prediction_sha256"],
                "passed": first["prediction_sha256"] == second["prediction_sha256"],
            }
            random_seed = (
                65571000 + root_index if arm_index == 0 else 65572000 + root_index
            )
            randomized_model, randomized_sha = self.randomized_model(arm, random_seed)
            randomized_result = self.evaluate_rollout(
                randomized_model,
                identities["16"][: int(self.config["evaluation"]["randomized_weight_cases"])],
                16,
                include_truth=not resource_only,
            )
            trained = evaluations[arm]["16"]["exact_accuracy"]
            randomized_accuracy = randomized_result["exact_accuracy"]
            randomized[arm] = {
                "seed": random_seed,
                "state_sha256": randomized_sha,
                "evaluation": randomized_result,
                "trained_minus_randomized": (
                    None
                    if trained is None or randomized_accuracy is None
                    else trained - randomized_accuracy
                ),
                "all_trainable_parameters_reinitialized": True,
            }

        runtime_identities = identity_manifest["runtime_width256_identities"]
        runtime = {
            arm: self.evaluate_rollout(
                model,
                runtime_identities,
                int(self.config["evaluation"]["runtime_probe_width"]),
                include_truth=False,
            )
            for arm, model in models.items()
        }
        structural = {
            "noncommutation": {
                arm: self.noncommutation(models[arm], root.eval_seed + 9_000)
                for arm in PRIMARY_ARMS
            },
            "malformed_and_leakage": self.malformed_and_leakage(
                models["layer_loop_r2"]
            ),
            "primary_initial_state_sha256": {
                arm: arm_results[arm]["initial_checkpoint"]["state_sha256"]
                for arm in PRIMARY_ARMS
            },
            "primary_initial_state_identical": len(
                {
                    arm_results[arm]["initial_checkpoint"]["state_sha256"]
                    for arm in PRIMARY_ARMS
                }
            )
            == 1,
        }
        return {
            "root": asdict(root),
            "resource_only": resource_only,
            "identity": {
                "path": identity_manifest["path"],
                "identity_sha256": identity_manifest["identity_sha256"],
                "file_sha256": identity_manifest["file_sha256"],
                "frozen_before_evaluation_labels": True,
            },
            "arms": arm_results,
            "evaluation": evaluations,
            "interventions": interventions,
            "deterministic_repeat": deterministic,
            "randomized_weight_dependence": randomized,
            "runtime_width256": runtime,
            "structural_controls": structural,
        }

    def calibration(self) -> dict[str, Any]:
        if self.source_at_start["dirty"]:
            raise RuntimeError("calibration requires clean committed source bytes")
        root = self.roots()[0]
        splits = self.root_splits(root)
        identity = self.freeze_identity_manifest(root, splits)
        primary_state, primary_sha = self.primary_initial_state(root)
        arm_results: dict[str, dict[str, Any]] = {}
        for arm in ARM_ORDER:
            arm_results[arm] = self.train_arm(
                arm, root, splits, primary_state, primary_sha
            )
        route = self.evaluate_root(
            root,
            arm_results,
            identity,
            splits,
            resource_only=True,
            root_index=0,
        )
        synthetic = np.zeros((3, 300), dtype=np.float64)
        bootstrap = {
            str(width): hierarchical_paired_bootstrap(
                synthetic,
                seed=int(self.config["evaluation"]["bootstrap_seeds"][width]),
                resamples=int(self.config["calibration"]["full_bootstrap_resamples"]),
            )
            for width in (16, 32)
        }
        measured = self.started.finish()
        training_timing = [
            arm_results[arm]["training"]["timing"] for arm in ARM_ORDER
        ]
        calibration_steps = sum(steps for _, steps in self.stages())
        formal_steps = int(self.config["training"]["total_steps_per_arm"])
        roots = len(self.config["data"]["formal_roots"])
        scale = formal_steps / calibration_steps
        expected: dict[str, int] = {}
        for key in ("wall_seconds", "process_cpu_seconds", "cgroup_cpu_seconds"):
            if measured[key] is None or any(item[key] is None for item in training_timing):
                raise RuntimeError(f"calibration did not measure {key}")
            training = sum(float(item[key]) for item in training_timing)
            nontraining = max(0.0, float(measured[key]) - training)
            expected[key] = math.ceil((training * scale + nontraining) * roots)
        per_step_wall = [
            float(item["wall_seconds"]) / calibration_steps
            for item in training_timing
        ]
        mean = float(np.mean(per_step_wall))
        coefficient = float(np.std(per_step_wall) / mean) if mean else 0.0
        variability = {
            "arm_wall_seconds_per_step": dict(zip(ARM_ORDER, per_step_wall, strict=True)),
            "minimum": min(per_step_wall),
            "maximum": max(per_step_wall),
            "mean": mean,
            "coefficient_of_variation": coefficient,
        }
        soft_multiplier = max(
            float(self.config["resource_budget"]["soft_multiplier"]),
            1.0
            + coefficient
            + float(self.config["resource_budget"]["variability_floor"]),
        )
        hard_multiplier = max(
            float(self.config["resource_budget"]["hard_multiplier"]),
            soft_multiplier + 0.5,
        )
        soft = {key: math.ceil(value * soft_multiplier) for key, value in expected.items()}
        hard = {key: math.ceil(value * hard_multiplier) for key, value in expected.items()}
        hard["peak_rss_bytes"] = math.ceil(
            int(measured["peak_rss_bytes"])
            * float(self.config["resource_budget"]["rss_multiplier"])
        )
        calibration = {
            "schema_version": "exp655-root20261178-resource-calibration-v1",
            "run_id": self.config["run_id"],
            "status": "completed-resource-only",
            "root": asdict(root),
            "source": self.source_at_start,
            "full_route": {
                "all_arms": list(ARM_ORDER),
                "stage_steps_per_arm": calibration_steps,
                "evaluation_cases_per_arm": {"16": 300, "32": 300},
                "opposite_schedule_full_cases_per_primary": {"16": 300, "32": 300},
                "intervention_cases": int(self.config["evaluation"]["intervention_cases"]),
                "randomized_weight_cases_per_primary": int(self.config["evaluation"]["randomized_weight_cases"]),
                "bootstrap_resamples_per_endpoint": int(self.config["calibration"]["full_bootstrap_resamples"]),
                "runtime_width256_cases_per_arm": int(self.config["evaluation"]["runtime_probe_cases"]),
                "checkpoint_reload": True,
                "persistence": True,
                "scientific_metrics_persisted": False,
            },
            "identity": route["identity"],
            "arm_resources": {
                arm: {
                    "parameters": arm_results[arm]["parameters"],
                    "checkpoint_sha256": arm_results[arm]["checkpoint"]["sha256"],
                    "training_timing": arm_results[arm]["training"]["timing"],
                    "stage_timing": [item["stage_timing"] for item in arm_results[arm]["stage_checkpoints"]],
                    "calibration_checkpoint_forbidden_for_formal": True,
                }
                for arm in ARM_ORDER
            },
            "evaluation_resource_receipt": {
                "prediction_hashes": {
                    arm: {
                        width: route["evaluation"][arm][width]["prediction_sha256"]
                        for width in ("16", "32")
                    }
                    for arm in ARM_ORDER
                },
                "interventions_executed": True,
                "randomized_weights_executed": True,
                "deterministic_repeat_executed": True,
                "malformed_and_leakage_executed": True,
                "noncommutation_executed": True,
                "bootstrap": bootstrap,
                "runtime_width256": route["runtime_width256"],
            },
            "measured_total": measured,
            "variability": variability,
            "projection": {
                "formal_roots": roots,
                "calibration_training_steps_per_arm": calibration_steps,
                "formal_training_steps_per_arm": formal_steps,
                "training_scale": scale,
                "expected_without_headroom": expected,
                "soft_multiplier": soft_multiplier,
                "hard_multiplier": hard_multiplier,
                "soft_budget": soft,
                "hard_budget": hard,
            },
            "limitations": [
                "Root 20261178 is resource-only; no accuracy or mechanism verdict is persisted.",
                "Formal projection scales measured per-arm training and retains the measured full evaluation/control route for each of three roots.",
                "Wall, process CPU, cgroup CPU, and peak RSS are independently named and are not substituted for one another.",
            ],
        }
        calibration_path = self.output / "calibration.json"
        write_json(calibration_path, calibration)
        admission = {
            "schema_version": "exp655-dynamic-resource-admission-v1",
            "run_id": self.config["run_id"],
            "decision": "admit-full-three-root-falsifier",
            "reason": "complete representative route measured; derived hard guards preserve frozen fidelity and the matched schedule decision has high information value",
            "fidelity_reduced": False,
            "calibration_path": str(calibration_path.relative_to(ROOT)),
            "calibration_sha256": sha256(calibration_path),
            "source": self.source_at_start,
            "expected_without_headroom": expected,
            "soft_budget": soft,
            "hard_budget": hard,
            "variability": variability,
            "calibration_checkpoints_reusable": False,
        }
        admission_path = ROOT / self.config["resource_budget"]["calibration_admission_path"]
        calibration["admission"] = {
            "path": str(admission_path.relative_to(ROOT)),
            "decision": admission["decision"],
            "note": "The separately persisted admission binds this final calibration by SHA-256.",
        }
        write_json(calibration_path, calibration)
        admission["calibration_sha256"] = sha256(calibration_path)
        write_json(admission_path, admission)
        return calibration

    def load_admission(self) -> dict[str, Any]:
        path = ROOT / self.config["resource_budget"]["calibration_admission_path"]
        if not path.is_file():
            raise RuntimeError("formal execution requires dynamic calibration admission")
        admission = json.loads(path.read_text(encoding="utf-8"))
        calibration = ROOT / admission.get("calibration_path", "missing")
        if not calibration.is_file() or sha256(calibration) != admission.get("calibration_sha256"):
            raise RuntimeError("calibration admission evidence hash mismatch")
        expected_source = admission.get("source", {})
        for key in (
            "repository_commit",
            "config_sha256",
            "runner_sha256",
            "model_sha256",
            "fixtures_sha256",
        ):
            if expected_source.get(key) != self.source_at_start.get(key):
                raise RuntimeError(f"calibrated source identity changed: {key}")
        if (
            admission.get("decision") != "admit-full-three-root-falsifier"
            or admission.get("fidelity_reduced") is not False
            or admission.get("calibration_checkpoints_reusable") is not False
        ):
            raise RuntimeError("dynamic resource admission is not valid")
        required = {"wall_seconds", "process_cpu_seconds", "cgroup_cpu_seconds", "peak_rss_bytes"}
        if set(admission.get("hard_budget", {})) != required:
            raise RuntimeError("dynamic hard budget is incomplete")
        return admission

    def gate_results(
        self,
        roots: list[dict[str, Any]],
        bootstraps: dict[str, dict[str, Any]],
        resource_result: dict[str, Any],
        admission: dict[str, Any],
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        gates = self.config["gates"]
        width16_deltas = []
        width32_deltas = []
        primary_width16 = []
        primary_width32 = []
        invalid_rates = []
        randomized_drops = []
        deterministic = []
        noncommutation = []
        initial_identity = []
        for root in roots:
            evaluation = root["evaluation"]
            layer16 = evaluation["layer_loop_r2"]["16"]["exact_accuracy"]
            model16 = evaluation["model_loop_r2"]["16"]["exact_accuracy"]
            layer32 = evaluation["layer_loop_r2"]["32"]["exact_accuracy"]
            model32 = evaluation["model_loop_r2"]["32"]["exact_accuracy"]
            width16_deltas.append(layer16 - model16)
            width32_deltas.append(layer32 - model32)
            primary_width16.append((layer16, model16))
            primary_width32.append((layer32, model32))
            for arm in PRIMARY_ARMS:
                for width in ("16", "32"):
                    invalid_rates.append(
                        evaluation[arm][width]["any_invalid_residue_rate"]
                    )
                randomized_drops.append(
                    root["randomized_weight_dependence"][arm][
                        "trained_minus_randomized"
                    ]
                )
                deterministic.append(root["deterministic_repeat"][arm]["passed"])
                noncommutation.append(
                    root["structural_controls"]["noncommutation"][arm][
                        "noncommuting"
                    ]
                )
            initial_identity.append(
                root["structural_controls"]["primary_initial_state_identical"]
            )
        hard = admission["hard_budget"]
        results = {
            "primary_width16_each_root": all(
                min(values) >= float(gates["minimum_primary_width16_each_root"])
                for values in primary_width16
            ),
            "any_primary_width32_each_root": all(
                max(values) >= float(gates["minimum_any_primary_width32_each_root"])
                for values in primary_width32
            ),
            "width32_delta_each_root": all(
                value >= float(gates["minimum_layer_loop_width32_delta_each_root"])
                for value in width32_deltas
            ),
            "width32_mean_delta": float(np.mean(width32_deltas))
            >= float(gates["minimum_mean_layer_loop_width32_delta"]),
            "width32_bootstrap_lower": bootstraps["32"][
                "one_sided_95_lower_bound"
            ]
            > float(gates["minimum_width32_one_sided_95_lower_bound"]),
            "width16_delta_each_root": all(
                value >= float(gates["minimum_layer_loop_width16_delta_each_root"])
                for value in width16_deltas
            ),
            "width16_bootstrap_lower": bootstraps["16"][
                "one_sided_95_lower_bound"
            ]
            > float(gates["minimum_width16_one_sided_95_lower_bound"]),
            "validity": max(invalid_rates) <= float(gates["maximum_invalid_rate"]),
            "randomized_weight_dependence": min(randomized_drops)
            >= float(gates["minimum_trained_minus_randomized_width16"]),
            "determinism": all(deterministic),
            "complete_block_noncommutation": all(noncommutation),
            "primary_initial_identity": all(initial_identity),
            "malformed_and_leakage": all(
                root["structural_controls"]["malformed_and_leakage"][
                    "malformed_passed"
                ]
                and root["structural_controls"]["malformed_and_leakage"][
                    "leakage_passed"
                ]
                for root in roots
            ),
            "wall_budget": float(resource_result["wall_seconds"])
            <= float(hard["wall_seconds"]),
            "process_cpu_budget": float(resource_result["process_cpu_seconds"])
            <= float(hard["process_cpu_seconds"]),
            "cgroup_cpu_budget": resource_result["cgroup_cpu_seconds"] is not None
            and float(resource_result["cgroup_cpu_seconds"])
            <= float(hard["cgroup_cpu_seconds"]),
            "rss_budget": int(resource_result["peak_rss_bytes"])
            <= int(hard["peak_rss_bytes"]),
            "full_denominators": all(
                root["evaluation"][arm][width]["cases"] == 300
                for root in roots
                for arm in ARM_ORDER
                for width in ("16", "32")
            ),
        }
        comparison = {
            "primary_width16_by_root": primary_width16,
            "primary_width32_by_root": primary_width32,
            "width16_layer_minus_model_by_root": width16_deltas,
            "width32_layer_minus_model_by_root": width32_deltas,
            "width16_unweighted_mean_delta": float(np.mean(width16_deltas)),
            "width32_unweighted_mean_delta": float(np.mean(width32_deltas)),
            "bootstrap": bootstraps,
            "maximum_invalid_rate": max(invalid_rates),
            "minimum_trained_minus_randomized": min(randomized_drops),
        }
        return results, comparison

    def copy_checkpoint_manifest(
        self, roots: list[dict[str, Any]], run_dir: Path
    ) -> dict[str, Any]:
        destination = run_dir / "checkpoints"
        destination.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {}
        for root in roots:
            root_id = str(root["root"]["root"])
            manifest[root_id] = {}
            for arm in ARM_ORDER:
                source_record = root["arms"][arm]["checkpoint"]
                source = ROOT / source_record["path"]
                target = destination / f"root-{root_id}-{arm}.safetensors"
                shutil.copy2(source, target)
                manifest[root_id][arm] = {
                    "path": str(target.relative_to(ROOT)),
                    "sha256": sha256(target),
                    "bytes": target.stat().st_size,
                    "source_sha256": source_record["sha256"],
                }
                if manifest[root_id][arm]["sha256"] != source_record["sha256"]:
                    raise RuntimeError("durable checkpoint copy hash mismatch")
        if sum(len(value) for value in manifest.values()) != 12:
            raise RuntimeError("formal result must contain exactly twelve checkpoints")
        return manifest

    def publish(
        self,
        summary: dict[str, Any],
        calibration: dict[str, Any],
        admission: dict[str, Any],
    ) -> dict[str, Any]:
        run_dir = ROOT / "experiments/runs" / self.config["run_id"]
        eval_dir = ROOT / "evals/results" / self.config["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = self.copy_checkpoint_manifest(summary["roots"], run_dir)
        durable = {**summary, "checkpoint_manifest": checkpoints}
        write_json(run_dir / "run.json", durable)
        write_json(run_dir / "environment.json", summary["environment"])
        write_json(run_dir / "calibration.json", calibration)
        write_json(run_dir / "admission.json", admission)
        write_json(run_dir / "mlflow.json", summary["mlflow"])
        write_json(run_dir / "limitations.json", {"limitations": summary["limitations"]})
        write_json(
            run_dir / "verification.json",
            {
                "source": summary["source"],
                "identity_manifests": [root["identity"] for root in summary["roots"]],
                "checkpoint_manifest": checkpoints,
                "gates": summary["gates"],
                "resource": summary["resource"],
                "bootstrap": summary["comparison"]["bootstrap"],
                "twelve_checkpoint_hashes_present": True,
            },
        )
        write_json(
            eval_dir / "metrics.json",
            {
                "run_id": summary["run_id"],
                "roots": [
                    {
                        "root": root["root"],
                        "evaluation": root["evaluation"],
                        "interventions": root["interventions"],
                        "randomized_weight_dependence": root[
                            "randomized_weight_dependence"
                        ],
                    }
                    for root in summary["roots"]
                ],
                "comparison": summary["comparison"],
                "gates": summary["gates"],
                "decision": summary["decision"],
            },
        )
        write_json(
            eval_dir / "decision.json",
            {
                "run_id": summary["run_id"],
                "decision": summary["decision"],
                "all_conjunctive_gates_passed": summary["gates"]["passed"],
                "gate_results": summary["gates"]["results"],
                "comparison": summary["comparison"],
                "bounded_faithful_scope": ["ST-M043.a", "ST-M009.a"],
                "not_candidate_benchmark_submission_or_goal_completion": True,
            },
        )
        return {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "eval_dir": str(eval_dir.relative_to(ROOT)),
            "checkpoint_count": 12,
        }

    def formal(self) -> dict[str, Any]:
        if self.source_at_start["dirty"]:
            raise RuntimeError("formal mode requires clean tracked source bytes")
        if self.source_at_start["repository_branch"] != self.config["source_contract"]["branch"]:
            raise RuntimeError("formal mode requires the frozen scoped branch")
        admission = self.load_admission()
        calibration_path = ROOT / admission["calibration_path"]
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        specs = self.roots()
        prepared = []
        for spec in specs:
            splits = self.root_splits(spec)
            identity = self.freeze_identity_manifest(spec, splits)
            prepared.append((spec, splits, identity))
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        mlflow_ids: dict[str, Any] = {"root_runs": {}, "arm_runs": {}}
        root_summaries = []
        with mlflow.start_run(run_name=self.config["run_id"]) as parent:
            mlflow_ids["parent_run_id"] = parent.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": self.config["run_id"],
                    "role": "full-three-root-dense-layer-loop-schedule-falsifier",
                    "repository_commit": self.source_at_start["repository_commit"],
                    "repository_dirty": "false",
                }
            )
            for root_index, (spec, splits, identity) in enumerate(prepared, start=1):
                with mlflow.start_run(
                    run_name=f"root-{spec.root}", nested=True
                ) as root_run:
                    mlflow_ids["root_runs"][str(spec.root)] = root_run.info.run_id
                    primary_state, primary_sha = self.primary_initial_state(spec)
                    arms: dict[str, dict[str, Any]] = {}
                    for arm in ARM_ORDER:
                        with mlflow.start_run(
                            run_name=f"root-{spec.root}-{arm}", nested=True
                        ) as arm_run:
                            mlflow_ids["arm_runs"][f"{spec.root}:{arm}"] = arm_run.info.run_id
                            arms[arm] = self.train_arm(
                                arm,
                                spec,
                                splits,
                                primary_state,
                                primary_sha,
                            )
                            mlflow.log_params(
                                {
                                    "root": spec.root,
                                    "arm": arm,
                                    "steps": self.config["training"]["total_steps_per_arm"],
                                    "checkpoint_sha256": arms[arm]["checkpoint"]["sha256"],
                                }
                            )
                    root_summary = self.evaluate_root(
                        spec,
                        arms,
                        identity,
                        splits,
                        resource_only=False,
                        root_index=root_index,
                    )
                    for arm in ARM_ORDER:
                        for width in ("16", "32"):
                            mlflow.log_metric(
                                f"{arm}_width{width}_exact",
                                root_summary["evaluation"][arm][width]["exact_accuracy"],
                            )
                    write_json(
                        self.output / "formal" / f"root-{spec.root}-summary.json",
                        root_summary,
                    )
                    root_summaries.append(root_summary)
                    self.check_budget(preserved_checkpoint=True)
            bootstraps = {}
            for width in (16, 32):
                deltas = np.asarray(
                    [
                        np.asarray(
                            root["evaluation"]["layer_loop_r2"][str(width)][
                                "correct_by_identity"
                            ],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            root["evaluation"]["model_loop_r2"][str(width)][
                                "correct_by_identity"
                            ],
                            dtype=np.float64,
                        )
                        for root in root_summaries
                    ]
                )
                bootstraps[str(width)] = hierarchical_paired_bootstrap(
                    deltas,
                    seed=int(self.config["evaluation"]["bootstrap_seeds"][width]),
                    resamples=int(self.config["evaluation"]["bootstrap_resamples"]),
                )
            resource_result = self.started.finish()
            gate_values, comparison = self.gate_results(
                root_summaries, bootstraps, resource_result, admission
            )
            passed = all(gate_values.values())
            decision = (
                "advance-dense-layer-loop-schedule-to-separately-judged-width64-confirmation"
                if passed
                else "reject-exp655-dense-layer-loop-schedule-formulation"
            )
            mlflow.log_metrics(
                {
                    "width32_mean_delta": comparison[
                        "width32_unweighted_mean_delta"
                    ],
                    "width32_bootstrap_lower": bootstraps["32"][
                        "one_sided_95_lower_bound"
                    ],
                    "width16_bootstrap_lower": bootstraps["16"][
                        "one_sided_95_lower_bound"
                    ],
                    "wall_seconds": resource_result["wall_seconds"],
                    "process_cpu_seconds": resource_result["process_cpu_seconds"],
                }
            )
            mlflow.set_tags({"status": "completed", "decision": decision})
        summary = {
            "schema_version": "exp655-full-three-root-formal-v1",
            "run_id": self.config["run_id"],
            "status": "completed",
            "hypothesis": self.config["hypothesis"],
            "source": self.source_at_start,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "mlflow": mlflow.__version__,
                "psutil": psutil.__version__,
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "pid": os.getpid(),
            },
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": self.config["tracking"]["experiment_name"],
                "experiment_id": experiment_id,
                **mlflow_ids,
            },
            "roots": root_summaries,
            "comparison": comparison,
            "gates": {
                **self.config["gates"],
                "results": gate_values,
                "passed": passed,
            },
            "decision": decision,
            "resource": {
                **resource_result,
                "dynamic_admission": admission,
            },
            "limitations": [
                "This is bounded-faithful evidence for dense recurrence ordering (ST-M043.a/ST-M009.a), not a reproduction of Loopie MoE, adaptive routing, capacity recycling, or post-training.",
                "Only fresh generated train-prime-disjoint W16/W32 identities were evaluated; checked, prior-independent, official, and private answers were not read.",
                "Exact Python arithmetic is confined to offline training labels and evaluation.",
                "A pass authorizes only a separately judged width64 confirmation; it cannot promote a candidate, benchmark, submission, or Goal completion.",
            ],
        }
        preview = self.output / "formal-summary.json"
        write_json(preview, summary)
        summary["publication"] = self.publish(summary, calibration, admission)
        write_json(preview, summary)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-655-layer-loop-compute-matched-recurrence",
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
                "decision": result.get("decision", result.get("admission", {}).get("decision")),
                "resource": result.get("resource", result.get("measured_total")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
