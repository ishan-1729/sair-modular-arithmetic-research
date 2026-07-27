"""Calibrate and execute EXP-651's paired visit/dilation adapter comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
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

from experiments.exp_651_visit_dilation_adapters.model import VisitDilationAdapterCell  # noqa: E402
from experiments.exp_655_layer_loop_compute_matched_recurrence.fixtures import verified_copy  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


@dataclass(frozen=True)
class RootSpec:
    root: int
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


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file():
        return None
    values = dict(line.split() for line in path.read_text(encoding="utf-8").splitlines())
    return float(values["usage_usec"]) / 1_000_000.0 if "usage_usec" in values else None


class Timer:
    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cpu = time.process_time()
        self.cgroup = cgroup_cpu_seconds()
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss

    def sample(self) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)

    def result(self) -> dict[str, float | int | None]:
        self.sample()
        current = cgroup_cpu_seconds()
        return {
            "wall_seconds": time.perf_counter() - self.wall,
            "process_cpu_seconds": time.process_time() - self.cpu,
            "cgroup_cpu_seconds": None if self.cgroup is None or current is None else current - self.cgroup,
            "peak_rss_bytes": self.peak,
        }


RESOURCE_KEYS = ("wall_seconds", "process_cpu_seconds", "cgroup_cpu_seconds")


def aggregate_resources(resources: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate sequential phase receipts without conflating wall, CPU, and RSS."""
    result: dict[str, float | int | None] = {}
    for key in RESOURCE_KEYS:
        values = [item[key] for item in resources]
        result[key] = None if any(value is None for value in values) else sum(float(value) for value in values)
    result["peak_rss_bytes"] = max(int(item["peak_rss_bytes"]) for item in resources)
    return result


def paired_bootstrap(deltas: np.ndarray, *, seed: int, resamples: int) -> dict[str, Any]:
    if deltas.ndim != 2 or deltas.shape[0] != 3:
        raise ValueError("paired bootstrap requires exactly three root rows")
    if resamples != 100_000:
        raise ValueError("formal bootstrap requires exactly 100,000 resamples")
    rng = np.random.default_rng(seed)
    roots, identities = deltas.shape
    distribution = np.empty(resamples, dtype=np.float64)
    chunk = 1000
    for start in range(0, resamples, chunk):
        count = min(chunk, resamples - start)
        selected_roots = rng.integers(0, roots, size=(count, roots))
        selected_ids = rng.integers(0, identities, size=(count, roots, identities))
        sampled = deltas[selected_roots[:, :, None], selected_ids]
        distribution[start : start + count] = sampled.mean(axis=(1, 2))
    return {
        "clusters": roots,
        "paired_identities_per_cluster": identities,
        "resamples": resamples,
        "mean": float(deltas.mean()),
        "one_sided_95_lower_bound": float(np.percentile(distribution, 5.0, method="linear")),
        "distribution_sha256": hashlib.sha256(distribution.tobytes()).hexdigest(),
    }


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.exp300 = load_module("exp300_for_exp651", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
        self.exp329 = load_module("exp329_for_exp651", ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py")
        self.cellular = load_module("cellular_for_exp651", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.checkpoint = self.materialize_checkpoint()

    def materialize_checkpoint(self) -> Path:
        destination = self.output / "fixtures/exp609-weights.safetensors"
        expected = self.config["source_contract"]["checkpoint_sha256"]
        candidates = [
            ROOT / ".scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors",
            Path("/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"),
            Path("/mnt/d/Projects/SAIR_mod_challenge_exp655_t048_wt/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"),
        ]
        source = next((item for item in candidates if item.is_file() and sha256(item) == expected), None)
        if source is None:
            raise RuntimeError("checksum-authenticated EXP609 checkpoint source is unavailable")
        verified_copy(source, destination, expected)
        return destination

    def source_identity(self) -> dict[str, Any]:
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "config_sha256": sha256(self.config_path),
            "model_sha256": sha256(Path(__file__).with_name("model.py")),
            "runner_sha256": sha256(Path(__file__)),
            "test_sha256": sha256(ROOT / "tests/test_exp_651_visit_dilation_adapters.py"),
            "preregistration_sha256": sha256(ROOT / self.config["preregistration"]),
            "checkpoint_sha256": sha256(self.checkpoint),
            "challenge_commit": self.config["source_contract"]["challenge_commit"],
        }

    def assert_frozen_clean(self) -> None:
        paths = [
            str(self.config_path.relative_to(ROOT)),
            "experiments/exp_651_visit_dilation_adapters",
            "tests/test_exp_651_visit_dilation_adapters.py",
            self.config["preregistration"],
        ]
        if git_output("status", "--porcelain", "--", *paths):
            raise RuntimeError("execution requires committed frozen source/config/test/preregistration bytes")

    def specs(self) -> list[RootSpec]:
        if self.mode == "calibration":
            return [RootSpec(**self.config["data"]["calibration"])]
        return [RootSpec(**item) for item in self.config["data"]["formal_roots"]]

    def split(self, width: int, root: int) -> dict[str, list[int]]:
        local = {"data": {
            "prime_bits_min": width - int(self.config["data"]["prime_bit_span"]) + 1,
            "prime_bits_max": width,
            "train_prime_identities": int(self.config["data"]["train_prime_identities"]),
            "test_prime_identities": int(self.config["data"]["test_prime_identities"]),
        }}
        return self.exp300.bigint_prime_split(local, int(self.config["data"]["split_seed_offset"]) + root + width)

    def base(self) -> nn.Module:
        values = self.config["model"]
        model = self.cellular.SharedDilatedConvGRU(width=128, hidden=int(values["hidden"]), cycles=int(values["cycles"]))
        model.load_state_dict(load_file(self.checkpoint))
        return model

    def model(self, arm_name: str) -> nn.Module:
        arm = self.config["arms"][arm_name]
        base = self.base()
        if arm["kind"] == "shared":
            return base
        return VisitDilationAdapterCell(
            base,
            kind=arm["kind"],
            rank=int(arm["rank"]),
            alpha=float(arm["alpha"]),
            max_dilation_levels=int(self.config["model"]["max_dilation_levels"]),
        )

    def exact_initialization(self) -> dict[str, Any]:
        torch.manual_seed(651)
        reference = self.base().eval()
        results: dict[str, Any] = {}
        for arm_name in ("global_adapter_control", "visit_dilation_adapter"):
            candidate = self.model(arm_name).eval()
            arm_result = {}
            with torch.no_grad():
                for width in self.config["evaluation"]["widths"]:
                    features = torch.randn(64, 6, int(width))
                    reference.width = int(width)
                    candidate.width = int(width)
                    expected = reference(features)
                    observed = candidate(features)
                    arm_result[str(width)] = {
                        "logits_equal": torch.equal(expected[0], observed[0]),
                        "quotient_equal": torch.equal(expected[1], observed[1]),
                        "maximum_logit_absolute_difference": float((expected[0] - observed[0]).abs().max()),
                        "maximum_quotient_absolute_difference": float((expected[1] - observed[1]).abs().max()),
                    }
            results[arm_name] = arm_result
        if not all(item["logits_equal"] and item["quotient_equal"] for arm in results.values() for item in arm.values()):
            raise RuntimeError("exact source-logit initialization parity failed")
        return results

    def recovery_loss(self, model: nn.Module, sampler: Any, rng: random.Random, width: int) -> torch.Tensor:
        training = self.config["training"]
        batch = int(training["recovery_batch_size"])
        chunk = int(training["recovery_chunk_steps"])
        multiplier, right, modulus = sampler.pairs(batch)
        starts = [rng.randrange(0, width - chunk + 1) for _ in range(batch)]
        teacher = self.exp329.teacher_states_before(multiplier, right, modulus, starts, width)
        state_plane = self.exp300.bits(teacher, width)
        losses = []
        model.width = width
        for offset in range(chunk):
            positions = [start + offset for start in starts]
            bit = [(left >> (width - 1 - position)) & 1 for left, position in zip(multiplier, positions)]
            raw = [2 * state + flag * value for state, flag, value in zip(teacher, bit, right)]
            target = [value % prime for value, prime in zip(raw, modulus)]
            quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long)
            logits, qlogits = model(self.exp300.features_from_soft_state(state_plane, right, modulus, bit, width)) if hasattr(self.exp300, "features_from_soft_state") else model(self._soft_features(state_plane, right, modulus, bit, width))
            loss, _ = self.exp300.transition_loss(logits, qlogits, target, quotient, width, float(training["worst_bit_weight"]), 0.0)
            losses.append(loss)
            probabilities = torch.sigmoid(logits)
            state_plane = logits.ge(0).float() + probabilities - probabilities.detach()
            teacher = target
        return torch.stack(losses).mean()

    def _soft_features(self, state_plane: torch.Tensor, right: list[int], modulus: list[int], bit: list[int], width: int) -> torch.Tensor:
        batch = len(right)
        bit_plane = torch.tensor(bit, dtype=torch.float32)[:, None].expand(batch, width)
        lsb = torch.zeros(batch, width)
        msb = torch.zeros(batch, width)
        lsb[:, 0] = 1
        msb[:, -1] = 1
        return torch.stack((state_plane, self.exp300.bits(right, width), self.exp300.bits(modulus, width), bit_plane, lsb, msb), dim=1).mul(2).sub(1)

    @torch.inference_mode()
    def evaluate_rollouts(self, model: nn.Module, sampler: Any, cases: int, width: int, **forward_kwargs: Any) -> dict[str, Any]:
        seen = correct = all_steps = invalid = 0
        exact_vector: list[int] = []
        first_errors: list[int | None] = []
        while seen < cases:
            size = min(64, cases - seen)
            multiplier, multiplicand, modulus = sampler.pairs(size)
            state = [0] * size
            teacher = [0] * size
            every = [True] * size
            bad = [False] * size
            first: list[int | None] = [None] * size
            for position in range(width):
                input_bit = [(value >> (width - 1 - position)) & 1 for value in multiplier]
                logits, _ = model(self.exp300.features(state, multiplicand, modulus, input_bit, width), **forward_kwargs)
                rows = logits.ge(0).tolist()
                state = [sum(int(row[index]) << index for index in range(width)) for row in rows]
                teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in zip(teacher, input_bit, multiplicand, modulus)]
                for row, (predicted, expected) in enumerate(zip(state, teacher)):
                    if predicted != expected and first[row] is None:
                        first[row] = position
                every = [prior and predicted == expected for prior, predicted, expected in zip(every, state, teacher)]
                bad = [prior or predicted >= prime for prior, predicted, prime in zip(bad, state, modulus)]
            expected = [(left * right) % prime for left, right, prime in zip(multiplier, multiplicand, modulus)]
            vector = [int(predicted == target) for predicted, target in zip(state, expected)]
            exact_vector.extend(vector)
            first_errors.extend(first)
            correct += sum(vector)
            all_steps += sum(every)
            invalid += sum(bad)
            seen += size
        failures = [value for value in first_errors if value is not None]
        return {
            "cases": seen,
            "correct": correct,
            "exact_accuracy": correct / seen,
            "all_steps_exact_accuracy": all_steps / seen,
            "any_invalid_residue_rate": invalid / seen,
            "mean_first_error_position": None if not failures else float(np.mean(failures)),
            "exact_vector": exact_vector,
        }

    def optimizer(self, model: nn.Module, arm_name: str, steps: int):
        arm = self.config["arms"][arm_name]
        parameters = model.adapter_parameters() if isinstance(model, VisitDilationAdapterCell) else list(model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=float(arm["learning_rate"]), weight_decay=float(self.config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=float(arm["minimum_learning_rate"]))
        return parameters, optimizer, scheduler

    def train_arm(self, spec: RootSpec, arm_name: str, steps: int, timer: Timer) -> tuple[nn.Module, dict[str, Any]]:
        random.seed(spec.train_seed)
        np.random.seed(spec.train_seed)
        torch.manual_seed(spec.train_seed)
        model = self.model(arm_name)
        parameters, optimizer, scheduler = self.optimizer(model, arm_name, steps)
        rng = random.Random(spec.train_seed + 7)
        transition_sampler = self.exp300.BigIntSampler(self.split(64, spec.root)["train"], spec.train_seed + 11)
        recovery_samplers = {width: self.exp300.BigIntSampler(self.split(width, spec.root)["train"], spec.train_seed + width) for width in (64, 128)}
        history = []
        schedule = self.config["training"]["mode_schedule"]
        model.train()
        for step in range(1, steps + 1):
            mode = schedule[(step - 1) % len(schedule)]
            if mode == "transition64":
                values = transition_sampler.transitions(int(self.config["training"]["batch_size"]))
                state, right, modulus, bit, target, quotient = values
                model.width = 64
                logits, qlogits = model(self.exp300.features(state, right, modulus, bit, 64))
                loss, stats = self.exp300.transition_loss(logits, qlogits, target, quotient, 64, float(self.config["training"]["worst_bit_weight"]), float(self.config["training"]["quotient_weight"]))
            else:
                width = int(mode.removeprefix("recovery"))
                loss = self.recovery_loss(model, recovery_samplers[width], rng, width)
                stats = {"loss": float(loss.detach())}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(parameters, float(self.config["training"]["gradient_clip"])))
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 50 == 0 or step == steps:
                timer.sample()
                record = {"step": step, "mode": mode, "loss": float(loss.detach()), "gradient_norm": gradient}
                history.append(record)
                print(json.dumps({"root": spec.root, "arm": arm_name, **record}), flush=True)
        return model.eval(), {"steps": steps, "history": history}

    def evaluate_arm(self, model: nn.Module, spec: RootSpec, arm_name: str) -> dict[str, Any]:
        evaluation: dict[str, Any] = {}
        for width in self.config["evaluation"]["widths"]:
            width = int(width)
            model.width = width
            split = self.split(width, spec.root)
            transition = self.exp300.evaluate_transitions(model, self.exp300.BigIntSampler(split["test"], spec.eval_seed + 1000 + width), int(self.config["evaluation"]["transition_cases"][width]))
            rollout = self.evaluate_rollouts(model, self.exp300.BigIntSampler(split["test"], spec.eval_seed + 2000 + width), int(self.config["evaluation"]["rollout_cases"][width]), width)
            evaluation[str(width)] = {"transition": transition, "rollout": rollout}
        checkpoint = self.output / self.mode / f"root-{spec.root}" / f"{arm_name}.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint)
        reloaded = self.model(arm_name)
        reloaded.load_state_dict(load_file(checkpoint))
        reloaded.eval()
        probe = torch.randn(int(self.config["evaluation"]["deterministic_repeat_cases"]), 6, 128)
        model.width = reloaded.width = 128
        with torch.no_grad():
            first = model(probe)
            second = reloaded(probe)
        deterministic = torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
        result: dict[str, Any] = {
            "evaluation": evaluation,
            "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
            "parameters": sum(value.numel() for value in model.parameters()),
            "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
            "deterministic_reload": deterministic,
        }
        if isinstance(model, VisitDilationAdapterCell):
            deltas = torch.cat([model.delta_gate(slot).flatten() for slot in range(model.slot_count)] + [model.delta_candidate(slot).flatten() for slot in range(model.slot_count)])
            intervention_width = 128
            split = self.split(intervention_width, spec.root)
            cases = int(self.config["evaluation"]["intervention_cases"])
            seed = spec.eval_seed + 3000
            ordinary = self.evaluate_rollouts(model, self.exp300.BigIntSampler(split["test"], seed), cases, intervention_width)
            zeroed = self.evaluate_rollouts(model, self.exp300.BigIntSampler(split["test"], seed), cases, intervention_width, adapter_enabled=False)
            permutation = tuple(reversed(range(model.slot_count)))
            permuted = self.evaluate_rollouts(model, self.exp300.BigIntSampler(split["test"], seed), cases, intervention_width, slot_permutation=permutation)
            randomized = copy.deepcopy(model)
            generator = torch.Generator().manual_seed(spec.eval_seed + (31 if arm_name == "visit_dilation_adapter" else 32))
            with torch.no_grad():
                for parameter in randomized.adapter_parameters():
                    flat = parameter.flatten()
                    order = torch.randperm(flat.numel(), generator=generator)
                    flat.copy_(flat[order])
            random_eval = self.evaluate_rollouts(randomized, self.exp300.BigIntSampler(split["test"], seed), cases, intervention_width)
            result["activation_and_interventions"] = {
                "adapter_delta_rms": float(deltas.square().mean().sqrt()),
                "adapter_delta_max_abs": float(deltas.abs().max()),
                "active_slots": {str(width): model.active_slot_count(int(width)) for width in self.config["evaluation"]["widths"]},
                "ordinary_exact": ordinary["exact_accuracy"],
                "zeroed_exact": zeroed["exact_accuracy"],
                "permuted_exact": permuted["exact_accuracy"],
                "randomized_exact": random_eval["exact_accuracy"],
                "zeroing_changed_predictions": sum(a != b for a, b in zip(ordinary["exact_vector"], zeroed["exact_vector"])),
                "permutation_changed_predictions": sum(a != b for a, b in zip(ordinary["exact_vector"], permuted["exact_vector"])),
                "trained_minus_randomized_w128": ordinary["exact_accuracy"] - random_eval["exact_accuracy"],
            }
        return result

    def _completed_root(self, spec: RootSpec, steps: int) -> dict[str, Any] | None:
        if self.mode != "formal":
            return None
        path = self.output / self.mode / f"root-{spec.root}-summary.json"
        if not path.is_file():
            return None
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary["source"] != self.source_identity():
                return None
            if summary["root"] != spec.root or summary["train_seed"] != spec.train_seed or summary["eval_seed"] != spec.eval_seed:
                return None
            if summary["steps"] != steps or set(summary["arms"]) != set(self.config["arms"]):
                return None
            for arm in summary["arms"].values():
                checkpoint = ROOT / arm["checkpoint"]["path"]
                if not checkpoint.is_file() or sha256(checkpoint) != arm["checkpoint"]["sha256"]:
                    return None
            return summary
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _partial_root_arms(self, spec: RootSpec, steps: int) -> dict[str, Any]:
        if self.mode != "formal":
            return {}
        path = self.output / self.mode / f"root-{spec.root}-progress.json"
        if not path.is_file():
            return {}
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
            if progress["source"] != self.source_identity() or progress["root"] != spec.root or progress["steps"] != steps:
                return {}
            arms = progress["arms"]
            if not set(arms).issubset(self.config["arms"]):
                return {}
            for arm in arms.values():
                checkpoint = ROOT / arm["checkpoint"]["path"]
                if not checkpoint.is_file() or sha256(checkpoint) != arm["checkpoint"]["sha256"]:
                    return {}
            return arms
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {}

    def run_root(self, spec: RootSpec, steps: int, timer: Timer) -> dict[str, Any]:
        completed = self._completed_root(spec, steps)
        if completed is not None:
            print(json.dumps({"root": spec.root, "status": "reused-hash-valid-completed-root"}), flush=True)
            return completed
        arms = self._partial_root_arms(spec, steps)
        if arms:
            print(json.dumps({"root": spec.root, "status": "reused-hash-valid-completed-arms", "arms": sorted(arms)}), flush=True)
        for arm_name in self.config["arms"]:
            if arm_name in arms:
                continue
            arm_timer = Timer()
            training_timer = Timer()
            model, training = self.train_arm(spec, arm_name, steps, timer)
            training["runtime"] = training_timer.result()
            evaluation_timer = Timer()
            result = self.evaluate_arm(model, spec, arm_name)
            result["evaluation_runtime"] = evaluation_timer.result()
            result["training"] = training
            result["runtime"] = arm_timer.result()
            arms[arm_name] = result
            write_json(
                self.output / self.mode / f"root-{spec.root}-progress.json",
                {"schema_version": "exp651-root-progress-v1", "source": self.source_identity(), "root": spec.root, "steps": steps, "arms": arms},
            )
        root_resource = aggregate_resources([arm["runtime"] for arm in arms.values()])
        summary = {
            "schema_version": "exp651-completed-root-v1",
            "source": self.source_identity(),
            "root": spec.root,
            "train_seed": spec.train_seed,
            "eval_seed": spec.eval_seed,
            "steps": steps,
            "arms": arms,
            "resource": root_resource,
        }
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", summary)
        return summary

    def calibration(self) -> dict[str, Any]:
        self.assert_frozen_clean()
        exact = self.exact_initialization()
        timer = Timer()
        spec = self.specs()[0]
        root = self.run_root(spec, int(self.config["training"]["calibration_steps"]), timer)
        resource = timer.result()
        calibration_steps = int(self.config["training"]["calibration_steps"])
        formal_steps = int(self.config["training"]["steps"])
        roots = len(self.config["data"]["formal_roots"])
        scale = formal_steps / calibration_steps
        training_resource = aggregate_resources([arm["training"]["runtime"] for arm in root["arms"].values()])
        evaluation_resource = aggregate_resources([arm["evaluation_runtime"] for arm in root["arms"].values()])
        expected: dict[str, float | int] = {}
        overhead: dict[str, float] = {}
        for key in RESOURCE_KEYS:
            measured = resource[key]
            training_value = training_resource[key]
            evaluation_value = evaluation_resource[key]
            if measured is None or training_value is None or evaluation_value is None:
                raise RuntimeError(f"calibration did not measure {key}")
            overhead[key] = max(0.0, float(measured) - float(training_value) - float(evaluation_value))
            expected[key] = overhead[key] + roots * (float(training_value) * scale + float(evaluation_value))
        expected["peak_rss_bytes"] = int(resource["peak_rss_bytes"])
        budget = self.config["resource_budget"]
        variability = max(float(budget["variability_floor"]), max(float(arm["runtime"]["wall_seconds"]) for arm in root["arms"].values()) / max(1e-9, np.mean([float(arm["runtime"]["wall_seconds"]) for arm in root["arms"].values()])) - 1.0)
        soft = {key: (int(math.ceil(value * float(budget["rss_multiplier"]))) if key == "peak_rss_bytes" else float(value) * float(budget["soft_multiplier"]) * (1.0 + variability)) for key, value in expected.items()}
        hard = {key: (int(math.ceil(value * float(budget["rss_multiplier"]))) if key == "peak_rss_bytes" else float(value) * float(budget["hard_multiplier"]) * (1.0 + variability)) for key, value in expected.items()}
        calibration = {
            "schema_version": "exp651-root20261182-calibration-v1",
            "source": self.source_identity(),
            "exact_initialization": exact,
            "root": root,
            "measured_resource": resource,
            "projection": {
                "calibration_steps_per_arm": calibration_steps,
                "formal_steps_per_arm": formal_steps,
                "formal_roots": roots,
                "measured_training_resource": training_resource,
                "measured_evaluation_resource": evaluation_resource,
                "measured_one_time_overhead": overhead,
                "only_training_scaled_by_step_ratio": True,
                "variability_headroom": variability,
            },
            "expected_budget": expected,
            "soft_budget": soft,
            "hard_budget": hard,
            "fidelity_reduced": False,
            "limitations": ["Calibration scientific values and checkpoints are resource-only and forbidden from formal reuse.", "Wall, process CPU, cgroup CPU, and RSS are independently measured and named."],
        }
        calibration_path = self.output / "calibration.json"
        write_json(calibration_path, calibration)
        operational = {
            "wall_seconds": float(budget["calibration_operational_runaway_wall_seconds"]),
            "process_cpu_seconds": float(budget["calibration_operational_runaway_process_cpu_seconds"]),
            "cgroup_cpu_seconds": float(budget["calibration_operational_runaway_cgroup_cpu_seconds"]),
        }
        admission_checks = {key: float(hard[key]) <= ceiling for key, ceiling in operational.items()}
        decision = "admit-full-three-root-falsifier" if all(admission_checks.values()) else "reject-formal-route-resource-projection"
        admission = {
            "schema_version": "exp651-dynamic-resource-admission-v1",
            "decision": decision,
            "source": calibration["source"],
            "calibration_path": str(calibration_path.relative_to(ROOT)),
            "calibration_sha256": sha256(calibration_path),
            "expected_budget": expected,
            "soft_budget": soft,
            "hard_budget": hard,
            "operational_runaway_ceiling": operational,
            "admission_checks": admission_checks,
            "fidelity_reduced": False,
            "calibration_checkpoints_reusable": False,
        }
        write_json(ROOT / budget["admission_path"], admission)
        return admission

    def load_admission(self) -> dict[str, Any]:
        path = ROOT / self.config["resource_budget"]["admission_path"]
        if not path.is_file():
            raise RuntimeError("formal execution requires calibration admission")
        admission = json.loads(path.read_text(encoding="utf-8"))
        calibration = ROOT / admission["calibration_path"]
        if sha256(calibration) != admission["calibration_sha256"]:
            raise RuntimeError("calibration admission hash mismatch")
        if admission["source"] != self.source_identity():
            raise RuntimeError("formal source identity differs from calibration")
        if admission["decision"] != "admit-full-three-root-falsifier" or admission["fidelity_reduced"] is not False:
            raise RuntimeError("invalid dynamic admission")
        return admission

    def gates(self, roots: list[dict[str, Any]], bootstraps: dict[str, Any], resource: dict[str, Any], admission: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
        primary = self.config["primary_arm"]
        global_name = "global_adapter_control"
        deltas: dict[str, list[float]] = {}
        for width in self.config["evaluation"]["widths"]:
            key = str(width)
            deltas[key] = [root["arms"][primary]["evaluation"][key]["rollout"]["exact_accuracy"] - root["arms"][global_name]["evaluation"][key]["rollout"]["exact_accuracy"] for root in roots]
        gates = self.config["gates"]
        checks = {
            "w128_each_root": min(deltas["128"]) >= float(gates["minimum_each_root_w128_gain_vs_global"]),
            "w128_mean": float(np.mean(deltas["128"])) >= float(gates["minimum_mean_w128_gain_vs_global"]),
            "w128_bootstrap": bootstraps["128"]["one_sided_95_lower_bound"] > float(gates["minimum_w128_one_sided_95_lower_bound"]),
            "w256_each_root": min(deltas["256"]) >= float(gates["minimum_each_root_w256_gain_vs_global"]),
            "w256_mean": float(np.mean(deltas["256"])) >= float(gates["minimum_mean_w256_gain_vs_global"]),
            "w256_bootstrap": bootstraps["256"]["one_sided_95_lower_bound"] > float(gates["minimum_w256_one_sided_95_lower_bound"]),
            "w64_retention_each_root": min(deltas["64"]) >= float(gates["minimum_each_root_w64_gain_vs_global"]),
            "w64_bootstrap": bootstraps["64"]["one_sided_95_lower_bound"] >= float(gates["minimum_w64_one_sided_95_lower_bound"]),
            "validity": all(root["arms"][arm]["evaluation"][str(width)]["rollout"]["any_invalid_residue_rate"] <= float(gates["maximum_invalid_rate"]) for root in roots for arm in self.config["arms"] for width in self.config["evaluation"]["widths"]),
            "determinism": all(root["arms"][arm]["deterministic_reload"] for root in roots for arm in self.config["arms"]),
            "activation": all(root["arms"][primary]["activation_and_interventions"]["adapter_delta_rms"] >= float(gates["minimum_adapter_delta_rms"]) for root in roots),
            "zeroing": all(root["arms"][primary]["activation_and_interventions"]["zeroing_changed_predictions"] >= int(gates["minimum_zeroing_changed_predictions"]) for root in roots),
            "permutation": all(root["arms"][primary]["activation_and_interventions"]["permutation_changed_predictions"] >= int(gates["minimum_permutation_changed_predictions"]) for root in roots),
            "randomized_weight_dependence": all(root["arms"][primary]["activation_and_interventions"]["trained_minus_randomized_w128"] >= float(gates["minimum_trained_minus_randomized_w128"]) for root in roots),
            "resource": all(resource[key] is not None and float(resource[key]) <= float(admission["hard_budget"][key]) for key in ("wall_seconds", "process_cpu_seconds", "cgroup_cpu_seconds", "peak_rss_bytes")),
        }
        return checks, {"visit_minus_global_rollout_deltas": deltas, "means": {key: float(np.mean(value)) for key, value in deltas.items()}, "bootstraps": bootstraps}

    def formal(self) -> dict[str, Any]:
        self.assert_frozen_clean()
        admission = self.load_admission()
        timer = Timer()
        exact = self.exact_initialization()
        roots = []
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        mlflow_ids: dict[str, Any] = {"root_runs": {}, "arm_runs": {}}
        with mlflow.start_run(run_name=self.config["run_id"]) as parent:
            mlflow_ids["parent_run_id"] = parent.info.run_id
            mlflow.set_tags({"canonical_run_id": self.config["run_id"], "status": "running", "scope": "fresh-held-out-generated-only", "repository_commit": git_output("rev-parse", "HEAD"), "checkpoint_sha256": self.config["source_contract"]["checkpoint_sha256"]})
            mlflow.log_dict(self.config, "resolved-config.json")
            for spec in self.specs():
                with mlflow.start_run(run_name=f"root-{spec.root}", nested=True) as root_run:
                    mlflow_ids["root_runs"][str(spec.root)] = root_run.info.run_id
                    root = self.run_root(spec, int(self.config["training"]["steps"]), timer)
                    roots.append(root)
                    mlflow.log_dict(root, f"root-{spec.root}-summary.json")
                    mlflow_ids["arm_runs"][str(spec.root)] = {}
                    for arm_name, arm in root["arms"].items():
                        with mlflow.start_run(run_name=f"root-{spec.root}-{arm_name}", nested=True) as arm_run:
                            mlflow_ids["arm_runs"][str(spec.root)][arm_name] = arm_run.info.run_id
                            mlflow.set_tags({"root": str(spec.root), "arm": arm_name, "checkpoint_sha256": arm["checkpoint"]["sha256"]})
                            mlflow.log_params({"steps": int(self.config["training"]["steps"]), "train_seed": spec.train_seed, "eval_seed": spec.eval_seed})
                            for width in self.config["evaluation"]["widths"]:
                                endpoint = arm["evaluation"][str(width)]
                                mlflow.log_metrics({f"w{width}_transition_exact": endpoint["transition"]["exact_accuracy"], f"w{width}_rollout_exact": endpoint["rollout"]["exact_accuracy"], f"w{width}_invalid_rate": endpoint["rollout"]["any_invalid_residue_rate"]})
                            mlflow.log_dict(arm, f"root-{spec.root}-{arm_name}.json")
            bootstraps = {}
            for width in self.config["evaluation"]["widths"]:
                key = str(width)
                matrix = np.asarray([[a - b for a, b in zip(root["arms"]["visit_dilation_adapter"]["evaluation"][key]["rollout"]["exact_vector"], root["arms"]["global_adapter_control"]["evaluation"][key]["rollout"]["exact_vector"])] for root in roots], dtype=np.float64)
                bootstraps[key] = paired_bootstrap(matrix, seed=int(self.config["evaluation"]["bootstrap_seeds"][width]), resamples=int(self.config["evaluation"]["bootstrap_resamples"]))
            session_resource = timer.result()
            root_resource = aggregate_resources([root["resource"] for root in roots])
            resource = {
                key: None if root_resource[key] is None or session_resource[key] is None else max(float(root_resource[key]), float(session_resource[key]))
                for key in RESOURCE_KEYS
            }
            resource["peak_rss_bytes"] = max(int(root_resource["peak_rss_bytes"]), int(session_resource["peak_rss_bytes"]))
            checks, comparison = self.gates(roots, bootstraps, resource, admission)
            decision = "promote-exp651-to-candidate-shaped-confirmation" if all(checks.values()) else "reject-exp651-visit-dilation-adapter-formulation"
            mlflow.log_metrics({f"mean_w{width}_gain": comparison["means"][str(width)] for width in self.config["evaluation"]["widths"]})
            mlflow.set_tags({"status": "completed", "decision": decision})
        summary = {
            "schema_version": "exp651-full-three-root-formal-v1",
            "run_id": self.config["run_id"],
            "status": "completed",
            "decision": decision,
            "source": self.source_identity(),
            "exact_initialization": exact,
            "roots": roots,
            "comparison": comparison,
            "gates": checks,
            "resource": resource,
            "dynamic_admission": admission,
            "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": self.config["tracking"]["experiment_name"], "experiment_id": experiment_id, **mlflow_ids},
            "limitations": ["Fresh held-out generated prime identities only; checked, prior-independent, official/private answers were not read.", "This is a bounded-faithful RRT analogue without SVD-teacher initialization or joint base uptraining.", "A pass authorizes confirmation only and cannot replace Candidate005."],
        }
        write_json(self.output / "formal-summary.json", summary)
        run_dir = ROOT / "experiments/runs/exp-651-visit-dilation-adapters"
        eval_dir = ROOT / "evals/results/exp-651-visit-dilation-adapters"
        write_json(run_dir / "run.json", summary)
        write_json(run_dir / "environment.json", {"python": sys.version, "torch": torch.__version__, "mlflow": mlflow.__version__, "platform": sys.platform, "torch_threads": torch.get_num_threads(), "repository_commit": git_output("rev-parse", "HEAD")})
        write_json(run_dir / "calibration.json", json.loads((self.output / "calibration.json").read_text(encoding="utf-8")))
        write_json(run_dir / "admission.json", admission)
        write_json(eval_dir / "metrics.json", summary)
        write_json(eval_dir / "decision.json", {"run_id": self.config["run_id"], "decision": decision, "gates": checks, "comparison": comparison, "resource": resource})
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_651_visit_dilation_adapters.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-651-visit-dilation-adapters")
    parser.add_argument("--mode", choices=("calibration", "formal"), required=True)
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode)
    result = experiment.calibration() if args.mode == "calibration" else experiment.formal()
    print(json.dumps({"mode": args.mode, "decision": result.get("decision"), "resource": result.get("resource"), "gates": result.get("gates")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
