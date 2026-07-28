"""Execute EXP686's matched W256 straight-through specialist experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
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

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU  # noqa: E402
from experiments.exp_664_sprm_boundary_perturbation.run import (  # noqa: E402
    Timer,
    bits,
    features,
    git_output,
    make_split,
    peak_rss_bytes,
    sha256,
    write_json,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

RUN_ID = "exp-686-w256-straight-through-specialist"
CONFIG_PATH = ROOT / "configs/exp_686_w256_straight_through_specialist.yaml"
ARM_ORDER = (
    "learner_straight_through",
    "teacher_forced",
    "frozen_source",
    "randomized_control",
    "reset_control",
)
TRAINED_ARMS = ARM_ORDER[:2]
WIDTH = 256
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"


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
            "exp300_for_exp686",
            ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
        )
    finally:
        sys.path[:] = previous_path
        if had_model:
            sys.modules["model"] = previous_model
        else:
            sys.modules.pop("model", None)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def configure_torch(config: dict[str, Any]) -> None:
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def soft_features(
    state_plane: torch.Tensor,
    multiplicand: list[int],
    modulus: list[int],
    input_bit: list[int],
    width: int,
) -> torch.Tensor:
    batch = len(multiplicand)
    bit_plane = torch.tensor(input_bit, dtype=torch.float32)[:, None].expand(batch, width)
    lsb = torch.zeros(batch, width)
    msb = torch.zeros(batch, width)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (state_plane, bits(multiplicand, width), bits(modulus, width), bit_plane, lsb, msb),
        dim=1,
    ).mul(2.0).sub(1.0)


def hard_straight_through(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    return logits.ge(0).float() + probability - probability.detach()


def decode_boolean(rows: torch.Tensor) -> list[int]:
    if rows.dtype is not torch.bool:
        raise TypeError("decode_boolean requires bool rows")
    return [sum(int(bit) << position for position, bit in enumerate(row)) for row in rows.tolist()]


def new_model(config: dict[str, Any], checkpoint: Path, *, randomized_seed: int | None = None) -> SharedDilatedConvGRU:
    if randomized_seed is not None:
        torch.manual_seed(randomized_seed)
    model = SharedDilatedConvGRU(
        width=WIDTH,
        hidden=int(config["model"]["hidden"]),
        cycles=int(config["model"]["cycles"]),
    )
    if randomized_seed is None:
        model.load_state_dict(load_file(checkpoint))
    return model


def teacher_prefix(
    left: list[int], right: list[int], modulus: list[int], starts: list[int]
) -> list[int]:
    state = [0] * len(left)
    maximum = max(starts, default=0)
    for position in range(maximum):
        for row in range(len(left)):
            if position < starts[row]:
                flag = (left[row] >> (WIDTH - 1 - position)) & 1
                state[row] = (2 * state[row] + flag * right[row]) % modulus[row]
    return state


def train_update(
    config: dict[str, Any],
    exp300,
    model: SharedDilatedConvGRU,
    optimizer: torch.optim.Optimizer,
    left: list[int],
    right: list[int],
    modulus: list[int],
    starts: list[int],
    arm: str,
) -> dict[str, float]:
    horizon = int(config["training"]["horizon"])
    teacher = teacher_prefix(left, right, modulus, starts)
    state_plane = bits(teacher, WIDTH)
    losses: list[torch.Tensor] = []
    exact: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    model.width = WIDTH
    for offset in range(horizon):
        flags = [(value >> (WIDTH - 1 - start - offset)) & 1 for value, start in zip(left, starts, strict=True)]
        raw = [2 * state + flag * other for state, flag, other in zip(teacher, flags, right, strict=True)]
        target = [value % prime for value, prime in zip(raw, modulus, strict=True)]
        quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus, strict=True)], dtype=torch.long)
        if arm == "teacher_forced":
            input_features = features(teacher, right, modulus, flags, WIDTH)
        else:
            input_features = soft_features(state_plane, right, modulus, flags, WIDTH)
        logits, quotient_logits = model(input_features)
        loss, stats = exp300.transition_loss(
            logits,
            quotient_logits,
            target,
            quotient,
            WIDTH,
            float(config["training"]["worst_bit_weight"]),
            float(config["training"]["quotient_weight"]),
        )
        losses.append(loss)
        exact.append(float(stats["batch_exact"]))
        state_plane = hard_straight_through(logits)
        teacher = target
    objective = torch.stack(losses).mean()
    objective.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"])))
    optimizer.step()
    return {"loss": float(objective.detach()), "gradient_norm": gradient, "mean_exact": float(np.mean(exact))}


def train_arm(
    config: dict[str, Any],
    exp300,
    checkpoint: Path,
    split: dict[str, list[int]],
    root: int,
    arm: str,
    updates: int,
    output: Path,
) -> tuple[SharedDilatedConvGRU, dict[str, Any]]:
    arm_dir = output / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(root + 686)
    random.seed(root + 686)
    np.random.seed(root + 686)
    randomized = root + 686900 if arm == "randomized_control" else None
    model = new_model(config, checkpoint, randomized_seed=randomized)
    initial_hash = canonical_hash({name: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest() for name, value in model.state_dict().items()})
    history: list[dict[str, Any]] = []
    timer = Timer()
    supervision_calls = optimizer_steps = examples = 0
    if arm in TRAINED_ARMS:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=updates,
            eta_min=float(config["training"]["minimum_learning_rate"]),
        )
        sampler = exp300.BigIntSampler(split["train"], root + int(config["data"]["training_seed_offset"]))
        start_rng = random.Random(root + int(config["data"]["training_seed_offset"]) + 1)
        batch_size = int(config["training"]["batch_size"])
        horizon = int(config["training"]["horizon"])
        model.train()
        for update in range(1, updates + 1):
            tick = Timer()
            left, right, modulus = sampler.pairs(batch_size)
            starts = [start_rng.randrange(0, WIDTH - horizon + 1) for _ in range(batch_size)]
            stats = train_update(config, exp300, model, optimizer, left, right, modulus, starts, arm)
            scheduler.step()
            supervision_calls += horizon
            optimizer_steps += 1
            examples += batch_size * horizon
            record = {"update": update, "learning_rate": scheduler.get_last_lr()[0], "timing": tick.finish(), **stats}
            history.append(record)
            if update == 1 or update == updates or update % int(config["training"]["log_every_updates"]) == 0:
                print(json.dumps({"arm": arm, **record}), flush=True)
    model.eval()
    model_path = arm_dir / "model.safetensors"
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, model_path)
    reloaded = new_model(config, checkpoint, randomized_seed=randomized)
    reloaded.load_state_dict(load_file(model_path))
    reload_exact = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    record = {
        "arm": arm,
        "trained": arm in TRAINED_ARMS,
        "initial_state_sha256": initial_hash,
        "parameters": model.parameter_count(),
        "training": {
            "updates": optimizer_steps,
            "supervision_calls": supervision_calls,
            "examples": examples,
            "calls_per_update": int(config["training"]["horizon"]) if arm in TRAINED_ARMS else 0,
            "history": history,
            "resource": timer.finish(),
        },
        "checkpoint": {"path": str(model_path), "sha256": sha256(model_path), "bytes": model_path.stat().st_size},
        "reload_exact": reload_exact,
    }
    return model.eval(), record


@torch.inference_mode()
def legacy_rollout(model: SharedDilatedConvGRU, left: list[int], right: list[int], modulus: list[int]) -> tuple[list[int], torch.Tensor, list[int | None], list[bool]]:
    state = [0] * len(left)
    teacher = [0] * len(left)
    margins = torch.full((len(left),), math.inf)
    first_error: list[int | None] = [None] * len(left)
    invalid = [False] * len(left)
    model.width = WIDTH
    for position in range(WIDTH):
        flags = [(value >> (WIDTH - 1 - position)) & 1 for value in left]
        logits, _ = model(features(state, right, modulus, flags, WIDTH))
        state = decode_boolean(logits.ge(0))
        teacher = [(2 * value + flag * other) % prime for value, flag, other, prime in zip(teacher, flags, right, modulus, strict=True)]
        margins = torch.minimum(margins, logits.abs().amin(dim=1).cpu())
        for row, (guess, truth, prime) in enumerate(zip(state, teacher, modulus, strict=True)):
            if guess != truth and first_error[row] is None:
                first_error[row] = position
            invalid[row] = invalid[row] or guess >= prime
    return state, margins, first_error, invalid


@torch.inference_mode()
def tensor_native_rollout(model: SharedDilatedConvGRU, left: list[int], right: list[int], modulus: list[int]) -> tuple[list[int], torch.Tensor, list[int | None], list[bool]]:
    batch = len(left)
    left_bits = bits(left, WIDTH)
    invariant = torch.empty((batch, 6, WIDTH), dtype=torch.float32)
    invariant[:, 1, :] = bits(right, WIDTH).mul(2).sub(1)
    invariant[:, 2, :] = bits(modulus, WIDTH).mul(2).sub(1)
    invariant[:, 4, :].fill_(-1); invariant[:, 4, 0] = 1
    invariant[:, 5, :].fill_(-1); invariant[:, 5, -1] = 1
    state_bits = torch.zeros((batch, WIDTH), dtype=torch.bool)
    teacher = [0] * batch
    margins = torch.full((batch,), math.inf)
    first_error: list[int | None] = [None] * batch
    invalid = [False] * batch
    model.width = WIDTH
    for position in range(WIDTH):
        invariant[:, 0, :] = state_bits.float().mul(2).sub(1)
        invariant[:, 3, :] = left_bits[:, WIDTH - 1 - position, None].mul(2).sub(1)
        logits, _ = model(invariant)
        state_bits = logits.ge(0)
        state = decode_boolean(state_bits)
        flags = [int(value) for value in left_bits[:, WIDTH - 1 - position].tolist()]
        teacher = [(2 * value + flag * other) % prime for value, flag, other, prime in zip(teacher, flags, right, modulus, strict=True)]
        margins = torch.minimum(margins, logits.abs().amin(dim=1).cpu())
        for row, (guess, truth, prime) in enumerate(zip(state, teacher, modulus, strict=True)):
            if guess != truth and first_error[row] is None:
                first_error[row] = position
            invalid[row] = invalid[row] or guess >= prime
    return decode_boolean(state_bits), margins, first_error, invalid


def evaluate_model(config: dict[str, Any], exp300, model: SharedDilatedConvGRU, split: dict[str, list[int]], root: int, transitions: int, rollouts: int) -> dict[str, Any]:
    model.eval(); model.width = WIDTH
    transition_timer = Timer()
    transition = exp300.evaluate_transitions(
        model,
        exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"])),
        transitions,
    )
    transition["resource"] = transition_timer.finish()
    left, right, modulus = exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + 1).pairs(rollouts)
    rollout_timer = Timer()
    prediction, margins, first_error, invalid = tensor_native_rollout(model, left, right, modulus)
    targets = [(a * b) % prime for a, b, prime in zip(left, right, modulus, strict=True)]
    errors = [value for value in first_error if value is not None]
    rollout = {
        "cases": rollouts,
        "correct": sum(a == b for a, b in zip(prediction, targets, strict=True)),
        "exact_accuracy": sum(a == b for a, b in zip(prediction, targets, strict=True)) / rollouts,
        "invalids": sum(invalid),
        "any_invalid_residue_rate": sum(invalid) / rollouts,
        "mean_first_error_position": None if not errors else float(np.mean(errors)),
        "prediction_sha256": canonical_hash(prediction),
        "minimum_margin_sha256": hashlib.sha256(margins.numpy().tobytes()).hexdigest(),
        "resource": rollout_timer.finish(),
    }
    return {"transition": transition, "rollout": rollout}


def parity_probe(model: SharedDilatedConvGRU, cases: tuple[list[int], list[int], list[int]]) -> dict[str, Any]:
    legacy = legacy_rollout(model, *cases)
    native = tensor_native_rollout(model, *cases)
    output_equal = legacy[0] == native[0]
    margin_equal = torch.equal(legacy[1], native[1])
    if not output_equal or not margin_equal:
        raise RuntimeError("legacy/tensor-native output or minimum-margin parity failed")
    return {
        "cases": len(cases[0]),
        "exact_outputs": output_equal,
        "exact_minimum_margins": margin_equal,
        "output_sha256": canonical_hash(legacy[0]),
        "minimum_margin_sha256": hashlib.sha256(legacy[1].numpy().tobytes()).hexdigest(),
    }


def candidate_inputs(root: int) -> list[tuple[str, str, str, int]]:
    source = ROOT / "sources/github/modular-arithmetic-challenge/src"
    sys.path.insert(0, str(source))
    try:
        from modchallenge.config import EvalConfig
        from modchallenge.testgen.generator import generate_private_test_set
        test_set = generate_private_test_set(hashlib.sha256(f"exp686-candidate-{root}".encode()).digest(), EvalConfig())
        return [(case.a, case.b, case.p, tier.tier_id) for tier in test_set.tiers for case in tier.cases]
    finally:
        sys.path.remove(str(source))


def candidate_runtime(config: dict[str, Any], model: SharedDilatedConvGRU, root: int, limit: int | None = None) -> dict[str, Any]:
    rows = candidate_inputs(root)
    if limit is not None:
        wide_first = [row for row in rows if 128 < int(row[2]).bit_length() <= 256]
        other = [row for row in rows if not (128 < int(row[2]).bit_length() <= 256)]
        rows = (wide_first + other)[:limit]
    candidate_module = load_module("candidate005_for_exp686", ROOT / "submission/model.py")
    incumbent = candidate_module.Tier2HornerModel(); incumbent.load(str(ROOT / "submission"))
    timer = Timer()
    lower = [(a, b, p) for a, b, p, _ in rows if int(p).bit_length() <= 128]
    wide = [(a, b, p) for a, b, p, _ in rows if 128 < int(p).bit_length() <= 256]
    upper = [row for row in rows if int(row[2]).bit_length() > 256]
    lower_outputs = incumbent.predict_digits_batch(lower) if lower else []
    if wide:
        modulus = [int(row[2]) for row in wide]
        left = [int(row[0]) % p for row, p in zip(wide, modulus, strict=True)]
        right = [int(row[1]) % p for row, p in zip(wide, modulus, strict=True)]
        wide_outputs = tensor_native_rollout(model, left, right, modulus)[0]
    else:
        wide_outputs = []
    resource = timer.finish()
    return {
        "cases": len(rows),
        "complete_1100_case_route": len(rows) == 1100,
        "lower_incumbent_cases": len(lower),
        "w256_specialist_cases": len(wide),
        "above_w256_zero_fallback_cases": len(upper),
        "output_count": len(lower_outputs) + len(wide_outputs) + len(upper),
        "labels_available_to_timed_route": False,
        "resource": resource,
    }


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if self.config["run_id"] != RUN_ID:
            raise RuntimeError("unexpected run_id")
        self.output = output.resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode; self.root = root
        self.checkpoint = (ROOT / self.config["source_checkpoint"]).resolve()
        self.authenticate(); configure_torch(self.config); self.exp300 = load_exp300()
        if mode != "smoke":
            expected = int(self.config["roots"]["calibration"] if mode == "calibration" else self.config["roots"]["formal"] if mode == "formal" else -1)
            allowed = {expected} if mode != "confirmation" else set(map(int, self.config["roots"]["confirmations"]))
            self.root = expected if root is None and mode != "confirmation" else root
            if self.root is None or int(self.root) not in allowed:
                raise ValueError(f"root {self.root} is not frozen for {mode}")
            self.root = int(self.root)
            self.split = make_split(self.config, self.exp300, WIDTH, self.root)

    def authenticate(self) -> None:
        if not self.checkpoint.is_file() or sha256(self.checkpoint) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("EXP609 source checkpoint hash mismatch")
        for name, binding in self.config["source_contract"].items():
            path = ROOT / binding["path"]
            if sha256(path) != binding["sha256"]:
                raise RuntimeError(f"source contract mismatch: {name}")
        challenge = git_output("-C", "sources/github/modular-arithmetic-challenge", "rev-parse", "HEAD")
        if challenge != CHALLENGE_COMMIT:
            raise RuntimeError("challenge commit mismatch")

    def identity(self) -> dict[str, Any]:
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "challenge_commit": CHALLENGE_COMMIT,
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__)),
            "test_sha256": sha256(ROOT / "tests/test_exp_686_w256_straight_through_specialist.py"),
            "source_checkpoint_sha256": sha256(self.checkpoint),
            "source_contract": {name: binding["sha256"] for name, binding in self.config["source_contract"].items()},
        }

    def root_dir(self) -> Path:
        return self.output / self.mode / f"root-{self.root}"

    def open_root(self) -> Path:
        path = self.root_dir()
        if path.exists():
            raise RuntimeError(f"root {self.root} already opened")
        path.mkdir(parents=True)
        write_json(path / "root-opened.json", {"mode": self.mode, "root": self.root, "source": self.identity()})
        return path

    def train_routes(self, output: Path, updates: int) -> tuple[dict[str, SharedDilatedConvGRU], dict[str, Any]]:
        models: dict[str, SharedDilatedConvGRU] = {}; records: dict[str, Any] = {}
        for arm in ARM_ORDER:
            model, record = train_arm(self.config, self.exp300, self.checkpoint, self.split, int(self.root), arm, updates, output)
            models[arm] = model; records[arm] = record
            write_json(output / "progress.json", {"completed_arms": list(records), "records": records})
        if records["learner_straight_through"]["initial_state_sha256"] != records["teacher_forced"]["initial_state_sha256"]:
            raise RuntimeError("trained arms are not byte-identically initialized")
        for key in ("updates", "supervision_calls", "examples", "calls_per_update"):
            if records["learner_straight_through"]["training"][key] != records["teacher_forced"]["training"][key]:
                raise RuntimeError(f"matched training invariant failed: {key}")
        return models, records

    def evaluate_routes(self, models: dict[str, SharedDilatedConvGRU], transitions: int, rollouts: int, arms: tuple[str, ...] = ARM_ORDER) -> dict[str, Any]:
        return {arm: evaluate_model(self.config, self.exp300, models[arm], self.split, int(self.root), transitions, rollouts) for arm in arms}

    def admission(self, records: dict[str, Any], evaluations: dict[str, Any], runtime: dict[str, Any], calibration_resource: dict[str, Any], updates: int, transitions: int, rollouts: int, parity: dict[str, Any]) -> dict[str, Any]:
        update_scale = int(self.config["training"]["updates"]) / updates
        training_wall = sum(float(records[arm]["training"]["resource"]["wall_seconds"]) for arm in TRAINED_ARMS) * update_scale
        training_cpu = sum(float(records[arm]["training"]["resource"]["process_cpu_seconds"]) for arm in TRAINED_ARMS) * update_scale
        evaluation_wall = evaluation_cpu = 0.0
        for arm in ARM_ORDER:
            evaluation_wall += float(evaluations[arm]["transition"]["resource"]["wall_seconds"]) * int(self.config["evaluation"]["transition_cases_per_arm"]) / transitions
            evaluation_wall += float(evaluations[arm]["rollout"]["resource"]["wall_seconds"]) * int(self.config["evaluation"]["rollout_cases_per_arm"]) / rollouts
            evaluation_cpu += float(evaluations[arm]["transition"]["resource"]["process_cpu_seconds"]) * int(self.config["evaluation"]["transition_cases_per_arm"]) / transitions
            evaluation_cpu += float(evaluations[arm]["rollout"]["resource"]["process_cpu_seconds"]) * int(self.config["evaluation"]["rollout_cases_per_arm"]) / rollouts
        expected_wall = training_wall + evaluation_wall + float(runtime["resource"]["wall_seconds"])
        expected_cpu = training_cpu + evaluation_cpu + float(runtime["resource"]["process_cpu_seconds"])
        update_walls = [float(item["timing"]["wall_seconds"]) for arm in TRAINED_ARMS for item in records[arm]["training"]["history"]]
        variability = max(update_walls) / max(float(np.mean(update_walls)), 1e-12)
        policy = self.config["resource_policy"]
        headroom = min(float(policy["variability_headroom_ceiling"]), max(float(policy["variability_headroom_floor"]), variability))
        expected_rss = int(calibration_resource["peak_rss_bytes"] * headroom)
        admission = {
            "projected_training_wall_seconds": training_wall,
            "projected_evaluation_wall_seconds": evaluation_wall,
            "expected_wall_seconds": expected_wall,
            "expected_process_cpu_seconds": expected_cpu,
            "headroom": headroom,
            "eta_wall_seconds": expected_wall * headroom,
            "soft_guard_wall_seconds": math.ceil(expected_wall * headroom * float(policy["soft_guard_multiplier"])),
            "hard_guard_wall_seconds": math.ceil(expected_wall * headroom * float(policy["hard_guard_multiplier"])),
            "expected_peak_rss_bytes": expected_rss,
            "candidate_actual_wall_seconds": runtime["resource"]["wall_seconds"],
            "candidate_guarded_wall_seconds": float(runtime["resource"]["wall_seconds"]) * headroom,
            "fidelity_reduced": bool(policy["fidelity_reduced"]),
        }
        admission["checks"] = {
            "complete_routes": set(records) == set(ARM_ORDER) == set(evaluations),
            "matched_training": all(records[TRAINED_ARMS[0]]["training"][key] == records[TRAINED_ARMS[1]]["training"][key] for key in ("updates", "supervision_calls", "examples", "calls_per_update")),
            "reload_exact": all(record["reload_exact"] for record in records.values()),
            "tensor_native_output_parity": parity["exact_outputs"],
            "tensor_native_minimum_margin_parity": parity["exact_minimum_margins"],
            "candidate_complete_1100": runtime["complete_1100_case_route"] and runtime["output_count"] == 1100,
            "candidate_runtime": admission["candidate_guarded_wall_seconds"] <= float(self.config["gates"]["maximum_candidate_wall_seconds"]),
            "expected_value": expected_wall <= float(policy["maximum_expected_formal_wall_seconds"]),
            "memory": expected_rss <= int(policy["maximum_peak_rss_bytes"]),
            "no_fidelity_reduction": not admission["fidelity_reduced"],
        }
        admission["admitted"] = all(admission["checks"].values())
        return admission

    def smoke(self) -> dict[str, Any]:
        smoke_root = 686000
        previous_root = self.root
        previous_split = getattr(self, "split", None)
        self.root = smoke_root; self.split = make_split(self.config, self.exp300, WIDTH, smoke_root)
        with tempfile.TemporaryDirectory(prefix="exp686-smoke-", dir=self.output) as directory:
            path = Path(directory); models, records = self.train_routes(path, 1)
            evaluations = self.evaluate_routes(models, 1, 1)
            parity_cases = self.exp300.BigIntSampler(self.split["test"], smoke_root + 9).pairs(1)
            parity = parity_probe(models["learner_straight_through"], parity_cases)
            runtime = candidate_runtime(self.config, models["learner_straight_through"], smoke_root, limit=1)
            synthetic_runtime = {**runtime, "complete_1100_case_route": True, "cases": 1100, "output_count": 1100, "resource": {**runtime["resource"], "wall_seconds": float(runtime["resource"]["wall_seconds"]) * 1100}}
            resource = Timer().finish()
            admission = self.admission(records, evaluations, synthetic_runtime, resource, 1, 1, 1, parity)
            terminals = []
            for mode in ("calibration", "formal", "confirmation"):
                terminal = path / f"{mode}-terminal.json"
                write_json(terminal, {"schema_version": "exp686-smoke-terminal-v1", "mode": mode, "root": None, "status": "completed", "scientific": False, "records": records, "evaluations": evaluations, "admission": admission})
                json.loads(terminal.read_text()); terminals.append(terminal.name)
            result = {
                "schema_version": "exp686-rootless-all-mode-smoke-v1", "status": "completed", "root": None,
                "all_modes": ["smoke", "calibration", "formal", "confirmation"], "one_update_each_trained_arm": all(records[arm]["training"]["updates"] == 1 for arm in TRAINED_ARMS),
                "one_case_every_route": all(evaluations[arm][kind]["cases"] == 1 for arm in ARM_ORDER for kind in ("transition", "rollout")),
                "reload_exact": all(record["reload_exact"] for record in records.values()), "matched_training": admission["checks"]["matched_training"],
                "parity": parity, "candidate_route_executed": runtime["output_count"] == 1, "admission_math_exercised": bool(admission["checks"]), "terminal_serializations": terminals,
            }
        self.root = previous_root; self.split = previous_split
        write_json(self.output / "smoke.json", result)
        return result

    def check_calibration_guard(self, timer: Timer) -> None:
        resource = timer.finish(); policy = self.config["resource_policy"]
        if float(resource["wall_seconds"]) > float(policy["calibration_hard_wall_seconds"]):
            raise RuntimeError("root94 hard wall guard exceeded")
        if float(resource["process_cpu_seconds"]) > float(policy["calibration_hard_process_cpu_seconds"]):
            raise RuntimeError("root94 hard process CPU guard exceeded")

    def calibration(self) -> dict[str, Any]:
        root_dir = self.open_root(); timer = Timer()
        updates = int(self.config["training"]["calibration_updates"])
        transitions = int(self.config["calibration"]["transition_cases_per_arm"]); rollouts = int(self.config["calibration"]["rollout_cases_per_arm"])
        models, records = self.train_routes(root_dir, updates); self.check_calibration_guard(timer)
        evaluations = self.evaluate_routes(models, transitions, rollouts); self.check_calibration_guard(timer)
        parity_cases = self.exp300.BigIntSampler(self.split["test"], int(self.root) + 44).pairs(int(self.config["calibration"]["parity_cases"]))
        parity = parity_probe(models["learner_straight_through"], parity_cases); self.check_calibration_guard(timer)
        runtime = candidate_runtime(self.config, models["learner_straight_through"], int(self.root)); self.check_calibration_guard(timer)
        resource = timer.finish(); admission = self.admission(records, evaluations, runtime, resource, updates, transitions, rollouts, parity)
        decision = "admit-root-20261295" if admission["admitted"] else "park-exp686-at-root20261294-resource-admission"
        result = {"schema_version": "exp686-calibration-v1", "run_id": RUN_ID, "status": "completed", "mode": self.mode, "root": self.root, "decision": decision, "source": self.identity(), "records": records, "evaluations": evaluations, "parity": parity, "candidate_runtime": runtime, "resource": resource, "admission": admission, "limitations": self.limitations(True)}
        write_json(root_dir / "root-summary.json", result); write_json(self.output / "calibration.json", result)
        return result

    def admitted_calibration(self) -> dict[str, Any]:
        path = self.output / "calibration.json"
        if not path.is_file():
            raise RuntimeError("science requires root20261294 admission")
        value = json.loads(path.read_text())
        if value["decision"] != "admit-root-20261295" or not value["admission"]["admitted"] or value["source"]["source_checkpoint_sha256"] != sha256(self.checkpoint):
            raise RuntimeError("science requires matching admitted calibration")
        if value["source"]["runner_sha256"] != sha256(Path(__file__)) or value["source"]["config_sha256"] != sha256(self.config_path):
            raise RuntimeError("calibration code/config identity mismatch")
        return value

    def require_predecessor(self) -> None:
        if self.mode != "confirmation":
            return
        roots = list(map(int, self.config["roots"]["confirmations"])); index = roots.index(int(self.root))
        predecessor = int(self.config["roots"]["formal"]) if index == 0 else roots[index - 1]
        predecessor_mode = "formal" if index == 0 else "confirmation"
        path = self.output / predecessor_mode / f"root-{predecessor}" / "root-summary.json"
        expected = "advance-to-confirmations" if index == 0 else "confirmation-pass"
        if not path.is_file() or json.loads(path.read_text())["decision"] != expected:
            raise RuntimeError(f"confirmation requires {expected} at root {predecessor}")

    def science(self) -> dict[str, Any]:
        calibration = self.admitted_calibration(); self.require_predecessor(); root_dir = self.open_root(); timer = Timer()
        models, records = self.train_routes(root_dir, int(self.config["training"]["updates"]))
        transitions = int(self.config["evaluation"]["transition_cases_per_arm"]); rollouts = int(self.config["evaluation"]["rollout_cases_per_arm"])
        primary = self.evaluate_routes(models, transitions, rollouts, ("learner_straight_through",))
        evaluations = dict(primary); p = primary["learner_straight_through"]
        gates: dict[str, bool] = {
            "primary_rollout": p["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_rollout"]),
            "primary_transition": p["transition"]["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_transition"]),
            "invalid_rate": p["rollout"]["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rate"]),
            "reload_exact": records["learner_straight_through"]["reload_exact"],
        }
        stage = "w256-primary"
        if all(gates.values()):
            evaluations.update(self.evaluate_routes(models, transitions, rollouts, ARM_ORDER[1:]))
            teacher = evaluations["teacher_forced"]["rollout"]["exact_accuracy"]
            frozen = evaluations["frozen_source"]["rollout"]["exact_accuracy"]
            randomized = evaluations["randomized_control"]["rollout"]["exact_accuracy"]
            reset = evaluations["reset_control"]["rollout"]["exact_accuracy"]
            score = p["rollout"]["exact_accuracy"]
            gates.update({
                "gain_over_teacher_forced": score - teacher >= float(self.config["gates"]["minimum_gain_over_teacher_forced"]),
                "gain_over_frozen_source": score - frozen >= float(self.config["gates"]["minimum_gain_over_frozen_source"]),
                "trained_weight_dependence": score - randomized >= float(self.config["gates"]["minimum_trained_weight_drop"]),
                "reset_matches_frozen": reset == frozen,
                "matched_training": all(records[TRAINED_ARMS[0]]["training"][key] == records[TRAINED_ARMS[1]]["training"][key] for key in ("updates", "supervision_calls", "examples", "calls_per_update")),
            }); stage = "w256-controls"
        runtime = candidate_runtime(self.config, models["learner_straight_through"], int(self.root))
        gates["candidate_runtime"] = float(runtime["resource"]["wall_seconds"]) * float(calibration["admission"]["headroom"]) <= float(self.config["gates"]["maximum_candidate_wall_seconds"])
        gates["confirmation_threshold"] = p["rollout"]["exact_accuracy"] >= float(self.config["gates"]["confirmation_admission_rollout"])
        passed_mechanism = all(value for key, value in gates.items() if key != "confirmation_threshold")
        passed_confirmation = passed_mechanism and gates["confirmation_threshold"]
        if self.mode == "formal":
            decision = "advance-to-confirmations" if passed_confirmation else "reject-exp686-w256-straight-through-specialist"
        else:
            decision = "confirmation-pass" if passed_confirmation else "reject-exp686-confirmation"
        result = {"schema_version": "exp686-terminal-v1", "run_id": RUN_ID, "status": "completed", "mode": self.mode, "root": self.root, "stage": stage, "decision": decision, "mechanism_gate_passed": passed_mechanism, "confirmation_gate_passed": passed_confirmation, "source": self.identity(), "records": records, "evaluations": evaluations, "gates": gates, "candidate_runtime": runtime, "dynamic_admission": calibration["admission"], "resource": timer.finish(), "limitations": self.limitations(False)}
        write_json(root_dir / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result)
        return result

    @staticmethod
    def limitations(calibration: bool) -> list[str]:
        return [
            "Calibration accuracy is resource and integrity evidence, not scientific evidence." if calibration else "Only fresh generated identities are evaluated; no checked, independent, official, or private benchmark is opened.",
            "Exact arithmetic is confined to offline labels, generated input reduction already sanctioned by Candidate005, and evaluation truth; trained SharedDilatedConvGRU weights produce specialist outputs.",
            "Tensor-native state caching changes no model parameters, calls, outputs, or minimum margins relative to the legacy route.",
            "The specialist is measured only as an intermediate research route; no candidate, submission, release, or incumbent promotion is authorized.",
        ]


def log_result(experiment: Experiment, result: dict[str, Any]) -> dict[str, Any]:
    if experiment.mode == "smoke":
        return result
    experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{RUN_ID}-{experiment.mode}-{experiment.root}") as active:
        mlflow.set_tags({"canonical_run_id": RUN_ID, "mode": experiment.mode, "root": str(experiment.root), "decision": result["decision"], "repository_commit": git_output("rev-parse", "HEAD"), "candidate_registration": "false"})
        mlflow.log_dict(result, f"{experiment.mode}-result.json")
        run_id = active.info.run_id
    record = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run_id, "mode": experiment.mode, "root": experiment.root}
    write_json(experiment.output / f"{experiment.mode}-mlflow.json", record)
    result["mlflow"] = record
    write_json(experiment.root_dir() / "root-summary.json", result)
    if experiment.mode == "calibration":
        write_json(experiment.output / "calibration.json", result)
    else:
        write_json(experiment.output / "terminal-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-686-w256-straight-through-specialist")
    parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmation"), required=True)
    parser.add_argument("--root", type=int)
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root)
    result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science()
    result = log_result(experiment, result)
    print(json.dumps({"mode": args.mode, "root": result.get("root"), "status": result["status"], "decision": result.get("decision"), "gates": result.get("gates"), "admission": result.get("admission", {}).get("checks")}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
