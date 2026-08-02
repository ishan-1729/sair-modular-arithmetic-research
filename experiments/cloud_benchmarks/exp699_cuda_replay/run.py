"""Replay the completed EXP699 teacher-cosine arm on CPU or CUDA.

This is a non-scientific hardware replay.  It preserves the frozen data order,
training work, generated evaluation cases, and paired-evidence route while
recording device and dependency identity.  A matching replay can justify using
the device for a future fresh root; it never replaces the consumed CPU root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import mlflow
import numpy as np
import psutil
import safetensors
import sympy
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
REPLAY_CONFIG_PATH = ROOT / "configs/exp_699_cuda_hardware_replay.yaml"
LOCAL_REFERENCE_PATH = BENCHMARK_DIR / "local-reference.json"
REQUIREMENTS = {
    "pypi": BENCHMARK_DIR / "requirements.txt",
    "pytorch_cuda": BENCHMARK_DIR / "requirements-torch-cu124.txt",
}
sys.path.insert(0, str(ROOT))

from experiments.exp_699_h92_w256_backbone_factorial_complete import run as source  # noqa: E402

WIDTHS = (128, 256)
ARM = "teacher_cosine"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        timeout=30,
    ).stdout.strip()


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


class Deadline:
    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("deadline must be positive")
        self.started = time.monotonic()
        self.seconds = float(seconds)

    def check(self, stage: str) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= self.seconds:
            raise TimeoutError(
                f"EXP699 CUDA replay exceeded {self.seconds:.3f}s during {stage}"
            )

    def remaining(self) -> float:
        return max(0.0, self.seconds - (time.monotonic() - self.started))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA replay requested but torch.cuda.is_available() is false")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        device = torch.device("cuda", index)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported replay device: {device}")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_torch(config: dict[str, Any], device: torch.device) -> None:
    torch.set_num_threads(int(config["replay"]["torch_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(config["source"]["formal_root"]) + 698)


class DeviceTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        synchronize(device)
        self.timer = source.Timer()

    def finish(self) -> dict[str, Any]:
        synchronize(self.device)
        result = source.resource_snapshot(self.timer)
        if self.device.type == "cuda":
            result["cuda_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            result["cuda_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device)
            )
        return result


class DeviceExp300:
    """Device-aware adapter around the exact EXP300 sampler and tensor semantics."""

    def __init__(self, parent: Any, device: torch.device, deadline: Deadline) -> None:
        self.parent = parent
        self.device = device
        self.deadline = deadline
        self.BigIntSampler = parent.BigIntSampler

    def bits(self, values: list[int], width: int) -> torch.Tensor:
        return self.parent.bits(values, width).to(self.device)

    def features(
        self,
        state: list[int],
        multiplicand: list[int],
        modulus: list[int],
        input_bit: list[int],
        width: int,
    ) -> torch.Tensor:
        return self.parent.features(
            state, multiplicand, modulus, input_bit, width
        ).to(self.device)

    def transition_loss(
        self,
        logits: torch.Tensor,
        quotient_logits: torch.Tensor,
        targets: list[int],
        quotients: torch.Tensor,
        width: int,
        worst_weight: float,
        quotient_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target = self.bits(targets, width)
        quotients = quotients.to(self.device)
        per_bit = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        mean = per_bit.mean()
        worst = per_bit.mean(0).max()
        quotient = F.cross_entropy(quotient_logits, quotients)
        loss = mean + worst_weight * worst + quotient_weight * quotient
        exact = (logits.ge(0) == target.bool()).all(1).float().mean()
        return loss, {
            "loss": float(loss.detach()),
            "batch_exact": float(exact),
            "mean_bce": float(mean.detach()),
            "worst_bit_bce": float(worst.detach()),
            "quotient_ce": float(quotient.detach()),
        }

    @torch.inference_mode()
    def evaluate_transitions(self, model: Any, sampler: Any, cases: int) -> dict:
        seen = correct = valid = 0
        while seen < cases:
            self.deadline.check("transition-evaluation")
            size = min(512, cases - seen)
            state, multiplicand, modulus, input_bit, target, _ = sampler.transitions(
                size
            )
            logits, _ = model(
                self.features(state, multiplicand, modulus, input_bit, model.width)
            )
            predicted = logits.ge(0)
            target_tensor = self.bits(target, model.width).bool()
            correct += int((predicted == target_tensor).all(1).sum().item())
            rows = predicted.cpu().tolist()
            predicted_values = [
                sum(int(row[position]) << position for position in range(model.width))
                for row in rows
            ]
            valid += sum(
                value < prime
                for value, prime in zip(predicted_values, modulus, strict=True)
            )
            seen += size
        return {
            "cases": seen,
            "correct": correct,
            "exact_accuracy": correct / seen,
            "valid_residue_rate": valid / seen,
        }


def soft_features(
    adapter: DeviceExp300,
    state_plane: torch.Tensor,
    multiplicand: list[int],
    modulus: list[int],
    input_bit: list[int],
    width: int,
) -> torch.Tensor:
    batch = len(multiplicand)
    device = state_plane.device
    bit_plane = torch.tensor(
        input_bit, dtype=torch.float32, device=device
    )[:, None].expand(batch, width)
    lsb = torch.zeros(batch, width, device=device)
    msb = torch.zeros(batch, width, device=device)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (
            state_plane,
            adapter.bits(multiplicand, width),
            adapter.bits(modulus, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul(2.0).sub(1.0)


def train_update(
    config: dict[str, Any],
    adapter: DeviceExp300,
    model: Any,
    optimizer: torch.optim.Optimizer,
    left: list[int],
    right: list[int],
    modulus: list[int],
    starts: list[int],
    arm: str,
    width: int,
) -> dict[str, float]:
    horizon = int(config["training"]["horizon"])
    prior_width = source.base.WIDTH
    source.base.WIDTH = width
    try:
        teacher = source.base.teacher_prefix(left, right, modulus, starts)
        state_plane = adapter.bits(teacher, width)
        losses: list[torch.Tensor] = []
        exact: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        model.width = width
        for offset in range(horizon):
            flags = [
                (value >> (width - 1 - start - offset)) & 1
                for value, start in zip(left, starts, strict=True)
            ]
            raw = [
                2 * state + flag * other
                for state, flag, other in zip(teacher, flags, right, strict=True)
            ]
            target = [
                value % prime for value, prime in zip(raw, modulus, strict=True)
            ]
            quotient = torch.tensor(
                [
                    value // prime
                    for value, prime in zip(raw, modulus, strict=True)
                ],
                dtype=torch.long,
                device=adapter.device,
            )
            if arm == "teacher_forced":
                input_features = adapter.features(
                    teacher, right, modulus, flags, width
                )
            else:
                input_features = soft_features(
                    adapter, state_plane, right, modulus, flags, width
                )
            logits, quotient_logits = model(input_features)
            loss, stats = adapter.transition_loss(
                logits,
                quotient_logits,
                target,
                quotient,
                width,
                float(config["training"]["worst_bit_weight"]),
                float(config["training"]["quotient_weight"]),
            )
            losses.append(loss)
            exact.append(float(stats["batch_exact"]))
            state_plane = source.base.hard_straight_through(logits)
            teacher = target
        objective = torch.stack(losses).mean()
        objective.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip"])
            )
        )
        optimizer.step()
        return {
            "loss": float(objective.detach()),
            "gradient_norm": gradient,
            "mean_exact": float(np.mean(exact)),
        }
    finally:
        source.base.WIDTH = prior_width


@torch.inference_mode()
def rollout(
    adapter: DeviceExp300,
    model: Any,
    left: list[int],
    right: list[int],
    modulus: list[int],
    width: int,
) -> tuple[list[int], torch.Tensor, list[int | None], list[bool]]:
    batch = len(left)
    left_bits = adapter.bits(left, width)
    invariant = torch.empty(
        (batch, 6, width), dtype=torch.float32, device=adapter.device
    )
    invariant[:, 1, :] = adapter.bits(right, width).mul(2).sub(1)
    invariant[:, 2, :] = adapter.bits(modulus, width).mul(2).sub(1)
    invariant[:, 4, :].fill_(-1)
    invariant[:, 4, 0] = 1
    invariant[:, 5, :].fill_(-1)
    invariant[:, 5, -1] = 1
    state_bits = torch.zeros(
        (batch, width), dtype=torch.bool, device=adapter.device
    )
    teacher = [0] * batch
    margins = torch.full((batch,), math.inf)
    first_error: list[int | None] = [None] * batch
    invalid = [False] * batch
    model.width = width
    for position in range(width):
        adapter.deadline.check(f"rollout-w{width}-position-{position}")
        invariant[:, 0, :] = state_bits.float().mul(2).sub(1)
        invariant[:, 3, :] = left_bits[:, width - 1 - position, None].mul(2).sub(1)
        logits, _ = model(invariant)
        state_bits = logits.ge(0)
        state = source.base.decode_boolean(state_bits.cpu())
        flags = [
            int(value)
            for value in left_bits[:, width - 1 - position].cpu().tolist()
        ]
        teacher = [
            (2 * value + flag * other) % prime
            for value, flag, other, prime in zip(
                teacher, flags, right, modulus, strict=True
            )
        ]
        margins = torch.minimum(margins, logits.abs().amin(dim=1).cpu())
        for row, (guess, truth, prime) in enumerate(
            zip(state, teacher, modulus, strict=True)
        ):
            if guess != truth and first_error[row] is None:
                first_error[row] = position
            invalid[row] = invalid[row] or guess >= prime
    return source.base.decode_boolean(state_bits.cpu()), margins, first_error, invalid


def interval(config: dict[str, Any], correct: int, cases: int) -> dict[str, float]:
    alpha = (1.0 - float(config["evaluation"]["confidence_level"])) / int(
        config["evaluation"]["multiplicity_family"]
    )
    return source.interval(correct, cases, alpha)


@torch.inference_mode()
def evaluate(
    config: dict[str, Any],
    adapter: DeviceExp300,
    model: Any,
    split: dict[str, Any],
    root: int,
    width: int,
    transitions: int,
    rollouts: int,
) -> dict[str, Any]:
    model.width = width
    timer = DeviceTimer(adapter.device)
    transition = adapter.evaluate_transitions(
        model,
        adapter.BigIntSampler(
            split["test"],
            root + int(config["data"]["test_seed_offset"]) + width,
        ),
        transitions,
    )
    sampler = adapter.BigIntSampler(
        split["test"],
        root + int(config["data"]["test_seed_offset"]) + width + 1,
    )
    left, right, modulus = sampler.pairs(rollouts)
    prediction, margins, first, invalid = rollout(
        adapter, model, left, right, modulus, width
    )
    target = [
        (a * b) % prime
        for a, b, prime in zip(left, right, modulus, strict=True)
    ]
    correct = sum(
        guess == truth for guess, truth in zip(prediction, target, strict=True)
    )
    errors = [value for value in first if value is not None]
    return {
        "width": width,
        "transition": transition,
        "rollout": {
            "cases": rollouts,
            "correct": correct,
            "exact_accuracy": correct / rollouts,
            "invalids": sum(invalid),
            "any_invalid_residue_rate": sum(invalid) / rollouts,
            "mean_first_error_position": (
                None if not errors else float(np.mean(errors))
            ),
            "first_error_sha256": source.canonical_hash(first),
            "prediction_sha256": source.canonical_hash(prediction),
            "target_sha256": source.canonical_hash(target),
            "minimum_margin_sha256": hashlib.sha256(
                margins.numpy().tobytes()
            ).hexdigest(),
            "confidence_99_time_uniform_multiplicity_controlled": interval(
                config, correct, rollouts
            ),
        },
        "resource": timer.finish(),
    }


@torch.inference_mode()
def paired_evidence(
    config: dict[str, Any],
    adapter: DeviceExp300,
    model: Any,
    split: dict[str, Any],
    root: int,
    width: int,
    transitions: int,
    rollouts: int,
) -> dict[str, Any]:
    timer = DeviceTimer(adapter.device)
    sampler = adapter.BigIntSampler(
        split["test"],
        root + int(config["data"]["test_seed_offset"]) + width,
    )
    transition_rows: list[list[int]] = []
    seen = transition_correct = transition_valid = 0
    while seen < transitions:
        adapter.deadline.check("paired-transition-evaluation")
        size = min(512, transitions - seen)
        state, multiplicand, modulus, input_bit, target, _ = sampler.transitions(size)
        logits, _ = model(
            adapter.features(state, multiplicand, modulus, input_bit, width)
        )
        predicted_bits = logits.ge(0)
        predicted = [
            sum(int(bit) << position for position, bit in enumerate(row))
            for row in predicted_bits.cpu().tolist()
        ]
        transition_rows.extend(
            [s, m, p, b, truth, guess]
            for s, m, p, b, truth, guess in zip(
                state,
                multiplicand,
                modulus,
                input_bit,
                target,
                predicted,
                strict=True,
            )
        )
        transition_correct += sum(
            truth == guess
            for truth, guess in zip(target, predicted, strict=True)
        )
        transition_valid += sum(
            guess < prime
            for guess, prime in zip(predicted, modulus, strict=True)
        )
        seen += size
    pair_sampler = adapter.BigIntSampler(
        split["test"],
        root + int(config["data"]["test_seed_offset"]) + width + 1,
    )
    left, right, modulus = pair_sampler.pairs(rollouts)
    prediction, margins, first, invalid = rollout(
        adapter, model, left, right, modulus, width
    )
    target = [
        (a * b) % prime
        for a, b, prime in zip(left, right, modulus, strict=True)
    ]
    rollout_rows = [
        [a, b, prime, truth, guess, error, bool(bad)]
        for a, b, prime, truth, guess, error, bad in zip(
            left,
            right,
            modulus,
            target,
            prediction,
            first,
            invalid,
            strict=True,
        )
    ]
    return {
        "transition": {
            "cases": transitions,
            "correct": transition_correct,
            "valid": transition_valid,
            "case_sha256": source.canonical_hash(
                [row[:5] for row in transition_rows]
            ),
            "prediction_sha256": source.canonical_hash(
                [row[5] for row in transition_rows]
            ),
            "paired_vectors_sha256": source.canonical_hash(transition_rows),
            "paired_vectors": transition_rows,
        },
        "rollout": {
            "cases": rollouts,
            "case_sha256": source.canonical_hash(
                [row[:4] for row in rollout_rows]
            ),
            "prediction_sha256": source.canonical_hash(prediction),
            "target_sha256": source.canonical_hash(target),
            "first_error_sha256": source.canonical_hash(first),
            "minimum_margin_sha256": hashlib.sha256(
                margins.numpy().tobytes()
            ).hexdigest(),
            "paired_vectors_sha256": source.canonical_hash(rollout_rows),
            "paired_vectors": rollout_rows,
        },
        "resource": timer.finish(),
    }


def evaluate_with_evidence(
    config: dict[str, Any],
    adapter: DeviceExp300,
    model: Any,
    split: dict[str, Any],
    root: int,
    width: int,
    transitions: int,
    rollouts: int,
) -> dict[str, Any]:
    result = evaluate(
        config, adapter, model, split, root, width, transitions, rollouts
    )
    paired = paired_evidence(
        config, adapter, model, split, root, width, transitions, rollouts
    )
    if (
        paired["transition"]["correct"] != result["transition"]["correct"]
        or paired["rollout"]["prediction_sha256"]
        != result["rollout"]["prediction_sha256"]
    ):
        raise RuntimeError("paired evidence replay changed primary evaluation semantics")
    result["transition"].update(
        {
            "case_sha256": paired["transition"]["case_sha256"],
            "prediction_sha256": paired["transition"]["prediction_sha256"],
            "paired_vectors_sha256": paired["transition"][
                "paired_vectors_sha256"
            ],
            "confidence_99_time_uniform_multiplicity_controlled": interval(
                config, result["transition"]["correct"], transitions
            ),
        }
    )
    result["rollout"].update(
        {
            "case_sha256": paired["rollout"]["case_sha256"],
            "paired_vectors_sha256": paired["rollout"]["paired_vectors_sha256"],
        }
    )
    result["paired_vectors"] = {
        "transition": paired["transition"]["paired_vectors"],
        "rollout": paired["rollout"]["paired_vectors"],
    }
    result["paired_evidence_resource"] = paired["resource"]
    return result


def cpu_state_dict(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }


def train_arm(
    config: dict[str, Any],
    adapter: DeviceExp300,
    checkpoint: Path,
    splits: dict[int, Any],
    root: int,
    arm: str,
    updates: int,
    output: Path,
    checkpoint_updates: set[int],
    curve_every_updates: int,
) -> tuple[Any, dict[str, Any]]:
    random.seed(root + 698)
    np.random.seed(root + 698)
    torch.manual_seed(root + 698)
    if adapter.device.type == "cuda":
        torch.cuda.manual_seed_all(root + 698)
        torch.cuda.reset_peak_memory_stats(adapter.device)
    model = source.new_model(config, checkpoint).to(adapter.device)
    initial = source.state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = source.make_scheduler(config, optimizer, arm, updates)
    samplers = {
        width: adapter.BigIntSampler(
            splits[width]["train"],
            root + int(config["data"]["training_seed_offset"]) + width,
        )
        for width in WIDTHS
    }
    starts = {
        width: random.Random(
            root + int(config["data"]["training_seed_offset"]) + width + 1
        )
        for width in WIDTHS
    }
    timer = DeviceTimer(adapter.device)
    curves: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    batch = int(config["training"]["batch_size"])
    horizon = int(config["training"]["horizon"])
    data_order = hashlib.sha256()
    width_order: list[int] = []
    model.train()
    for update in range(1, updates + 1):
        adapter.deadline.check(f"training-update-{update}")
        width = int(
            config["training"].get(
                "profile_width",
                source.balanced_width(
                    update, int(config["training"]["w128_only_updates"])
                ),
            )
        )
        left, right, modulus = samplers[width].pairs(batch)
        offsets = [
            starts[width].randrange(0, width - horizon + 1) for _ in left
        ]
        data_order.update(
            json.dumps(
                [width, left, right, modulus, offsets], separators=(",", ":")
            ).encode()
        )
        width_order.append(width)
        state_arm = (
            "teacher_forced"
            if arm.startswith("teacher_")
            else "learner_straight_through"
        )
        stats = train_update(
            config,
            adapter,
            model,
            optimizer,
            left,
            right,
            modulus,
            offsets,
            state_arm,
            width,
        )
        scheduler.step()
        if update % curve_every_updates == 0 or update == updates:
            row = {
                "update": update,
                "width": width,
                "learning_rate": scheduler.get_last_lr()[0],
                **stats,
            }
            curves.append(row)
            print(json.dumps({"arm": arm, **row}), flush=True)
        if update in checkpoint_updates or update == updates:
            path = output / "arms" / arm / f"update-{update}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(cpu_state_dict(model), path)
            checkpoints.append(
                {
                    "update": update,
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    final = output / "arms" / arm / "model.safetensors"
    save_file(cpu_state_dict(model), final)
    reloaded = source.new_model(config, checkpoint)
    reloaded.load_state_dict(load_file(final))
    exact = all(
        torch.equal(left.cpu(), right.cpu())
        for left, right in zip(
            model.state_dict().values(), reloaded.state_dict().values(), strict=True
        )
    )
    return model.eval(), {
        "arm": arm,
        "initial_state_sha256": initial,
        "parameters": model.parameter_count(),
        "training": {
            "updates": updates,
            "examples": updates * batch * horizon,
            "model_calls": updates * horizon,
            "width_order_sha256": source.canonical_hash(width_order),
            "data_order_sha256": data_order.hexdigest(),
            "curves": curves,
            "resource": timer.finish(),
        },
        "checkpoints": checkpoints,
        "final_checkpoint": {
            "path": str(final),
            "sha256": sha256(final),
            "bytes": final.stat().st_size,
        },
        "final_state_sha256": source.state_hash(model),
        "reload_exact": exact,
    }


def dependency_versions() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "torch_cuda_runtime": str(torch.version.cuda),
        "numpy": np.__version__,
        "mlflow": mlflow.__version__,
        "psutil": psutil.__version__,
        "PyYAML": yaml.__version__,
        "safetensors": safetensors.__version__,
        "sympy": sympy.__version__,
    }


def validate_environment(
    replay_config: dict[str, Any], device: torch.device, profile: str
) -> dict[str, str]:
    versions = dependency_versions()
    expected = {
        "numpy": "2.5.1",
        "mlflow": "3.14.0",
        "psutil": "7.0.0",
        "PyYAML": "6.0.3",
        "safetensors": "0.8.0",
        "sympy": "1.13.1",
    }
    mismatches = {
        name: {"expected": value, "actual": versions[name]}
        for name, value in expected.items()
        if versions[name] != value
    }
    if device.type == "cuda":
        cuda_expected = {
            "torch": replay_config["environment"]["torch_cuda"],
            "torch_cuda_runtime": replay_config["environment"]["cuda_runtime"],
        }
        mismatches.update(
            {
                name: {"expected": value, "actual": versions[name]}
                for name, value in cuda_expected.items()
                if versions[name] != value
            }
        )
    elif profile == "full" and not versions["torch"].startswith("2.5.1"):
        mismatches["torch"] = {
            "expected": "2.5.1 CPU variant",
            "actual": versions["torch"],
        }
    if mismatches:
        raise RuntimeError(f"dependency identity mismatch: {mismatches}")
    return versions


def source_identity(
    replay_config: dict[str, Any], checkpoint: Path
) -> dict[str, Any]:
    paths = {
        "config": ROOT / replay_config["source"]["config"],
        "runner": ROOT / replay_config["source"]["runner"],
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    expected = {
        "config": replay_config["source"]["config_sha256"],
        "runner": replay_config["source"]["runner_sha256"],
    }
    mismatches = {
        name: {"expected": expected[name], "actual": value}
        for name, value in actual.items()
        if value != expected[name]
    }
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != replay_config["source"]["checkpoint_sha256"]:
        mismatches["checkpoint"] = {
            "expected": replay_config["source"]["checkpoint_sha256"],
            "actual": checkpoint_hash,
        }
    if mismatches:
        raise RuntimeError(f"source identity mismatch: {mismatches}")
    return {
        "repository_commit": git_output("rev-parse", "HEAD"),
        "repository_branch": git_output("branch", "--show-current"),
        "repository_dirty": bool(git_output("status", "--porcelain")),
        "challenge_commit": git_output(
            "-C", "sources/github/modular-arithmetic-challenge", "rev-parse", "HEAD"
        ),
        "source_sha256": actual,
        "source_checkpoint_sha256": checkpoint_hash,
        "replay_runner_sha256": sha256(Path(__file__)),
        "replay_config_sha256": sha256(REPLAY_CONFIG_PATH),
        "requirements_sha256": {
            name: sha256(path) for name, path in REQUIREMENTS.items()
        },
    }


def environment(device: torch.device, versions: dict[str, str]) -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    result: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "device": str(device),
        "os_cpu_count": os.cpu_count(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in (read_optional("/proc/cpuinfo") or "").splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        "cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
        "cpuset_cpus_effective": read_optional(
            "/sys/fs/cgroup/cpuset.cpus.effective"
        ),
        "memory_max": read_optional("/sys/fs/cgroup/memory.max"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "dependencies": versions,
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
    return result


def route_resource(record: dict[str, Any], evaluations: dict[str, Any]) -> dict:
    resources = [record["training"]["resource"]]
    for width in ("128", "256"):
        resources.extend(
            [
                evaluations[width]["resource"],
                evaluations[width]["paired_evidence_resource"],
            ]
        )
    keys = ("wall_seconds", "process_cpu_seconds", "cgroup_cpu_seconds")
    result = {
        f"route_{key}": (
            None
            if any(item.get(key) is None for item in resources)
            else sum(float(item[key]) for item in resources)
        )
        for key in keys
    }
    result["peak_rss_bytes"] = max(int(item["peak_rss_bytes"]) for item in resources)
    cuda_allocated = [
        int(item.get("cuda_peak_allocated_bytes", 0)) for item in resources
    ]
    cuda_reserved = [
        int(item.get("cuda_peak_reserved_bytes", 0)) for item in resources
    ]
    if any(cuda_allocated):
        result["cuda_peak_allocated_bytes"] = max(cuda_allocated)
        result["cuda_peak_reserved_bytes"] = max(cuda_reserved)
    return result


def compare_reference(
    reference: dict[str, Any],
    record: dict[str, Any],
    evaluations: dict[str, Any],
    resource: dict[str, Any],
    minimum_speedup: float,
) -> dict[str, Any]:
    actual_work = {
        "updates": record["training"]["updates"],
        "examples": record["training"]["examples"],
        "model_calls": record["training"]["model_calls"],
        "transition_cases_per_width": evaluations["128"]["transition"]["cases"],
        "w128_rollout_cases": evaluations["128"]["rollout"]["cases"],
        "w256_rollout_cases": evaluations["256"]["rollout"]["cases"],
    }
    work_identity = actual_work == reference["work"]
    case_checks: dict[str, bool] = {}
    correct_checks: dict[str, bool] = {}
    prediction_checks: dict[str, bool] = {}
    for width in ("128", "256"):
        for kind in ("transition", "rollout"):
            key = f"{width}.{kind}"
            expected = reference["evaluations"][width][kind]
            actual = evaluations[width][kind]
            case_checks[key] = actual["case_sha256"] == expected["case_sha256"]
            correct_checks[key] = (
                actual["cases"] == expected["cases"]
                and actual["correct"] == expected["correct"]
            )
            prediction_checks[key] = (
                actual["prediction_sha256"] == expected["prediction_sha256"]
            )
    cloud_wall = float(resource["route_wall_seconds"])
    local_wall = float(reference["resource"]["route_wall_seconds"])
    speedup = local_wall / cloud_wall
    return {
        "work": {"passed": work_identity, "expected": reference["work"], "actual": actual_work},
        "case_identity": {"passed": all(case_checks.values()), "checks": case_checks},
        "exact_correct_counts": {
            "passed": all(correct_checks.values()),
            "checks": correct_checks,
        },
        "prediction_hash_identity": {
            "passed": all(prediction_checks.values()),
            "checks": prediction_checks,
            "gating": False,
        },
        "performance": {
            "local_route_wall_seconds": local_wall,
            "replay_route_wall_seconds": cloud_wall,
            "speedup": speedup,
            "minimum_required_speedup": minimum_speedup,
            "passed": speedup >= minimum_speedup,
        },
        "eligible_for_future_fresh_cuda_root": (
            work_identity
            and all(case_checks.values())
            and all(correct_checks.values())
            and speedup >= minimum_speedup
        ),
        "scientific_root_replaced": False,
    }


def profile_values(
    replay_config: dict[str, Any], profile: str
) -> dict[str, int]:
    selected = replay_config["replay"] if profile == "full" else replay_config["smoke"]
    return {
        "updates": int(selected["updates"]),
        "transitions": int(selected["transition_cases_per_width"]),
        "w128_rollouts": int(selected["w128_rollout_cases"]),
        "w256_rollouts": int(selected["w256_rollout_cases"]),
    }


def admitted_wall_seconds(
    requested_wall: float, hourly_rate: float | None, maximum_compute_usd: float
) -> float:
    if hourly_rate is None:
        return float(requested_wall)
    if hourly_rate <= 0:
        raise ValueError("hourly rate must be positive")
    return min(float(requested_wall), maximum_compute_usd / hourly_rate * 3600.0)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    replay_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_config_path = ROOT / replay_config["source"]["config"]
    scientific_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    device = resolve_device(args.device or replay_config["replay"]["device"])
    if args.profile == "full" and device.type == "cpu" and not args.allow_cpu_full_replay:
        raise RuntimeError(
            "full CPU replay is disabled; use CUDA or explicitly pass "
            "--allow-cpu-full-replay"
        )
    if (
        args.profile == "full"
        and replay_config["guardrails"]["require_hourly_rate_for_full_replay"]
        and args.hourly_rate_usd is None
    ):
        raise RuntimeError("--hourly-rate-usd is required for a full replay")
    requested_wall = (
        replay_config["guardrails"]["maximum_wall_seconds"]
        if args.max_wall_seconds is None
        else args.max_wall_seconds
    )
    requested_compute_usd = (
        replay_config["guardrails"]["maximum_compute_usd"]
        if args.max_compute_usd is None
        else args.max_compute_usd
    )
    maximum_wall = min(
        float(requested_wall),
        float(replay_config["guardrails"]["maximum_wall_seconds"]),
    )
    maximum_compute_usd = min(
        float(requested_compute_usd),
        float(replay_config["guardrails"]["maximum_compute_usd"]),
    )
    admitted = admitted_wall_seconds(
        maximum_wall, args.hourly_rate_usd, maximum_compute_usd
    )
    deadline = Deadline(admitted)
    checkpoint = (
        args.source_checkpoint.resolve()
        if args.source_checkpoint is not None
        else Path(scientific_config["source_checkpoint"]).resolve()
    )
    if not checkpoint.is_file():
        raise RuntimeError(
            "source checkpoint is unavailable; pass --source-checkpoint after "
            "copying the hash-verified EXP609 weights to the host"
        )
    configure_torch(replay_config, device)
    versions = validate_environment(replay_config, device, args.profile)
    identity = source_identity(replay_config, checkpoint)
    environment_payload = environment(device, versions)
    values = profile_values(replay_config, args.profile)
    preflight = {
        "schema_version": "exp699-cuda-preflight-v1",
        "status": "admitted",
        "classification": replay_config["classification"],
        "profile": args.profile,
        "identity": identity,
        "environment": environment_payload,
        "guardrails": {
            "requested_maximum_wall_seconds": maximum_wall,
            "admitted_wall_seconds": admitted,
            "maximum_compute_usd": maximum_compute_usd,
            "hourly_rate_usd": args.hourly_rate_usd,
            "maximum_compute_charge_usd": (
                None
                if args.hourly_rate_usd is None
                else args.hourly_rate_usd * admitted / 3600.0
            ),
            "provider_resource_termination_external": True,
        },
        "work": values,
        "source_checkpoint": str(checkpoint),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", preflight)
    if args.dry_run:
        return {**preflight, "status": "dry-run"}
    exp300_parent = source.base.load_exp300()
    adapter = DeviceExp300(exp300_parent, device, deadline)
    root = int(replay_config["source"]["formal_root"])
    splits = {
        width: source.make_split(scientific_config, exp300_parent, width, root)
        for width in WIDTHS
    }
    run_config = copy.deepcopy(scientific_config)
    if args.profile == "smoke":
        run_config["training"]["checkpoint_updates"] = [values["updates"]]
        run_config["training"]["curve_every_updates"] = values["updates"]
    checkpoint_updates = set(
        map(
            int,
            (
                replay_config["replay"]["checkpoint_updates"]
                if args.profile == "full"
                else [values["updates"]]
            ),
        )
    )
    curve_every = int(
        replay_config["replay"]["curve_every_updates"]
        if args.profile == "full"
        else values["updates"]
    )
    started_at = datetime.now(timezone.utc).isoformat()
    outer_timer = DeviceTimer(device)
    model, record = train_arm(
        run_config,
        adapter,
        checkpoint,
        splits,
        root,
        ARM,
        values["updates"],
        output,
        checkpoint_updates,
        curve_every,
    )
    evaluations = {
        "128": evaluate_with_evidence(
            run_config,
            adapter,
            model,
            splits[128],
            root,
            128,
            values["transitions"],
            values["w128_rollouts"],
        ),
        "256": evaluate_with_evidence(
            run_config,
            adapter,
            model,
            splits[256],
            root,
            256,
            values["transitions"],
            values["w256_rollouts"],
        ),
    }
    outer_resource = outer_timer.finish()
    scoped_resource = route_resource(record, evaluations)
    actual_cost = (
        None
        if args.hourly_rate_usd is None
        else args.hourly_rate_usd
        * float(scoped_resource["route_wall_seconds"])
        / 3600.0
    )
    comparison = None
    decision = "smoke-completed"
    if args.profile == "full":
        reference = json.loads(LOCAL_REFERENCE_PATH.read_text(encoding="utf-8"))
        comparison = compare_reference(
            reference,
            record,
            evaluations,
            scoped_resource,
            float(replay_config["guardrails"]["minimum_required_speedup"]),
        )
        decision = (
            "eligible-for-future-fresh-cuda-root"
            if comparison["eligible_for_future_fresh_cuda_root"]
            else "timing-only-hardware-replay"
        )
    receipt = {
        "schema_version": "exp699-cuda-hardware-replay-v1",
        "run_id": replay_config["run_id"],
        "status": "completed",
        "classification": replay_config["classification"],
        "profile": args.profile,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "environment": environment_payload,
        "guardrails": {
            **preflight["guardrails"],
            "remaining_wall_seconds_at_completion": deadline.remaining(),
            "actual_compute_charge_usd": actual_cost,
        },
        "record": record,
        "evaluations": evaluations,
        "resource": {
            **scoped_resource,
            "outer_route": outer_resource,
        },
        "comparison": comparison,
        "decision": decision,
        "limitations": replay_config["limitations"],
    }
    write_json(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPLAY_CONFIG_PATH)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-699-cuda-hardware-replay",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hourly-rate-usd", type=float)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-compute-usd", type=float)
    parser.add_argument("--allow-cpu-full-replay", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = execute(args)
    except Exception as error:
        failure = {
            "schema_version": "exp699-cuda-hardware-replay-failure-v1",
            "status": "failed",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "profile": args.profile,
            "device": args.device,
        }
        write_json(args.output.resolve() / "failure.json", failure)
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "profile": result["profile"],
                "decision": result.get("decision"),
                "resource": result.get("resource"),
                "comparison": result.get("comparison"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
