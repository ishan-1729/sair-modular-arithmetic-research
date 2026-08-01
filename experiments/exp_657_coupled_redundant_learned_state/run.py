"""Run the staged EXP-657 coupled redundant learned-state experiment.

The only launch permitted by the scaffold package is the resource-only L1
calibration.  Formal mode is fail-closed behind a separately committed
authorization manifest and is intentionally not opened by this task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import mlflow
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from sympy import nextprime
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from model import CoupledRedundantLearnedState, StepOutput  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/exp_657_coupled_redundant_learned_state.yaml"
DEFAULT_RUN_DIR = ROOT / "experiments/runs/exp-657-coupled-redundant-learned-state"
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_yaml(path: Path, value: Any) -> None:
    atomic_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


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
    times = psutil.Process().cpu_times()
    return float(times.user + times.system)


def peak_rss_bytes() -> int:
    # T010 requires Ubuntu WSL. Linux ru_maxrss is expressed in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def combined_rusage_cpu() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime)


def set_determinism(seed: int, threads: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Safe when a test process already established the same interop pool.
        pass
    torch.use_deterministic_algorithms(True)


# All functions with the offline_ prefix are data, label, diagnostic, or
# evaluation utilities. They are never called from the model inference graph.
def offline_bits(values: Iterable[int], width: int) -> Tensor:
    rows = [
        [(int(value) >> position) & 1 for position in range(width)]
        for value in values
    ]
    return torch.tensor(rows, dtype=torch.float32)


def offline_state_belief(values: Iterable[int], width: int) -> Tensor:
    return offline_bits(values, width).unsqueeze(1).mul(2.0).sub(1.0)


def offline_context(
    multiplicands: list[int],
    moduli: list[int],
    input_bits: list[int],
    width: int,
) -> Tensor:
    batch = len(multiplicands)
    bit_plane = torch.tensor(input_bits, dtype=torch.float32)[:, None].expand(batch, width)
    lsb = torch.zeros((batch, width), dtype=torch.float32)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    planes = torch.stack(
        (
            offline_bits(multiplicands, width),
            offline_bits(moduli, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    )
    return planes.mul(2.0).sub(1.0)


def offline_decode(logits: Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in logits.ge(0).tolist()
    ]


def tensor_digest(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class OfflineSampler:
    """Deterministic offline generator with exact arithmetic confined here."""

    def __init__(self, primes: list[int], seed: int) -> None:
        self.primes = list(primes)
        self.rng = random.Random(int(seed))

    def transitions(
        self, size: int
    ) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
        states: list[int] = []
        multiplicands: list[int] = []
        moduli: list[int] = []
        input_bits: list[int] = []
        targets: list[int] = []
        for _ in range(int(size)):
            modulus = self.rng.choice(self.primes)
            state = self.rng.randrange(modulus)
            multiplicand = self.rng.randrange(modulus)
            input_bit = self.rng.randrange(2)
            raw = 2 * state + input_bit * multiplicand
            states.append(state)
            multiplicands.append(multiplicand)
            moduli.append(modulus)
            input_bits.append(input_bit)
            targets.append(raw % modulus)
        return states, multiplicands, moduli, input_bits, targets

    def pairs(self, size: int) -> tuple[list[int], list[int], list[int]]:
        moduli = [self.rng.choice(self.primes) for _ in range(int(size))]
        multipliers = [self.rng.randrange(modulus) for modulus in moduli]
        multiplicands = [self.rng.randrange(modulus) for modulus in moduli]
        return multipliers, multiplicands, moduli


def make_prime_split(config: dict[str, Any], width: int, seed: int) -> dict[str, list[int]]:
    data = config["data"]
    rng = random.Random(seed + int(data["split_seed_offset"]) + width)
    low = 1 << (width - int(data["prime_bit_span"]))
    high = 1 << width
    total = int(data["train_prime_identities"]) + int(data["test_prime_identities"])
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < total:
        value = int(nextprime(rng.randrange(low, high)))
        if value < high and value not in seen:
            values.append(value)
            seen.add(value)
    cut = int(data["train_prime_identities"])
    return {"train": values[:cut], "test": values[cut:]}


def build_splits(config: dict[str, Any], seed: int) -> dict[int, dict[str, list[int]]]:
    return {int(width): make_prime_split(config, int(width), seed) for width in config["data"]["widths"]}


def model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return {
        "hidden": int(model["hidden"]),
        "syndrome_hidden": int(model["syndrome_hidden"]),
        "cycles": int(model["cycles"]),
        "context_channels": int(model["input_context_channels"]),
        "correction_scale": float(model["correction_scale"]),
    }


def new_model(config: dict[str, Any], persistence: str, seed: int) -> CoupledRedundantLearnedState:
    torch.manual_seed(int(seed))
    return CoupledRedundantLearnedState(persistence=persistence, **model_kwargs(config))


def matched_geometry(config: dict[str, Any], seed: int) -> dict[str, Any]:
    control = new_model(config, "single", seed)
    challenger = new_model(config, "redundant", seed)
    control_signature = control.parameter_signature()
    challenger_signature = challenger.parameter_signature()
    if control_signature != challenger_signature:
        raise RuntimeError("matched arms have different parameter geometry")
    if control.parameter_count() != challenger.parameter_count():
        raise RuntimeError("matched arms have different parameter counts")
    if not control.branch_parameters_are_distinct() or not challenger.branch_parameters_are_distinct():
        raise RuntimeError("a redundant channel aliases another channel")
    return {
        "parameter_count_each_arm": control.parameter_count(),
        "parameter_count_difference": 0,
        "parameter_signature_sha256": sha256_text(canonical_json(control_signature)),
        "forward_step_graph": "identical-except-persistent-state-feedback-rule",
        "separately_parameterized_channel_a_and_b": True,
    }


def supervised_loss(
    output: StepOutput,
    targets: Tensor,
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    target_bool = targets.bool()
    fused = F.binary_cross_entropy_with_logits(output.fused_logits, targets)
    channel_a = F.binary_cross_entropy_with_logits(output.channel_a_logits, targets)
    channel_b = F.binary_cross_entropy_with_logits(output.channel_b_logits, targets)
    error_labels = torch.stack(
        (
            output.channel_a_logits.detach().ge(0).ne(target_bool),
            output.channel_b_logits.detach().ge(0).ne(target_bool),
        ),
        dim=1,
    ).float()
    syndrome = F.binary_cross_entropy_with_logits(output.syndrome_error_logits, error_labels)
    residual_a = torch.sigmoid(output.channel_a_logits) - targets
    residual_b = torch.sigmoid(output.channel_b_logits) - targets
    residual_a = residual_a - residual_a.mean(dim=1, keepdim=True)
    residual_b = residual_b - residual_b.mean(dim=1, keepdim=True)
    numerator = (residual_a * residual_b).sum(dim=1)
    denominator = residual_a.square().sum(dim=1).sqrt() * residual_b.square().sum(dim=1).sqrt()
    decorrelation = (numerator / denominator.clamp_min(1e-8)).square().mean()
    model = config["model"]
    total = (
        fused
        + float(model["channel_auxiliary_weight"]) * (channel_a + channel_b) * 0.5
        + float(model["syndrome_supervision_weight"]) * syndrome
        + float(model["residual_decorrelation_weight"]) * decorrelation
    )
    return total, {
        "fused_loss": fused.detach(),
        "channel_loss": ((channel_a + channel_b) * 0.5).detach(),
        "syndrome_loss": syndrome.detach(),
        "decorrelation_loss": decorrelation.detach(),
    }


@dataclass
class NaturalErrorBatch:
    state_a: Tensor
    state_b: Tensor
    context: Tensor
    target_bits: Tensor
    candidate_pairs: int
    candidate_bit_steps: int


def materialize_natural_error_batch(batch: NaturalErrorBatch) -> NaturalErrorBatch:
    """Copy inference-mode mining outputs into ordinary autograd-safe tensors."""

    materialized = NaturalErrorBatch(
        state_a=batch.state_a.clone().detach(),
        state_b=batch.state_b.clone().detach(),
        context=batch.context.clone().detach(),
        target_bits=batch.target_bits.clone().detach(),
        candidate_pairs=batch.candidate_pairs,
        candidate_bit_steps=batch.candidate_bit_steps,
    )
    tensors = (
        materialized.state_a,
        materialized.state_b,
        materialized.context,
        materialized.target_bits,
    )
    if any(tensor.is_inference() for tensor in tensors):
        raise RuntimeError("natural-error replay tensor remained inference-owned")
    return materialized


@torch.inference_mode()
def mine_natural_error_batch(
    model: CoupledRedundantLearnedState,
    sampler: OfflineSampler,
    *,
    width: int,
    needed: int,
    maximum_candidate_pairs: int,
) -> NaturalErrorBatch | None:
    """Mine only a model's naturally observed first errors and their prior states."""

    states_a: list[Tensor] = []
    states_b: list[Tensor] = []
    contexts: list[Tensor] = []
    target_rows: list[Tensor] = []
    candidate_pairs = 0
    candidate_bit_steps = 0
    model.eval()
    while len(states_a) < needed and candidate_pairs < maximum_candidate_pairs:
        size = min(16, maximum_candidate_pairs - candidate_pairs)
        multipliers, multiplicands, moduli = sampler.pairs(size)
        candidate_pairs += size
        state_a, state_b = model.initial_state(size, width)
        teacher = [0 for _ in range(size)]
        active = [True for _ in range(size)]
        for position in range(width):
            if not any(active) or len(states_a) >= needed:
                break
            input_bits = [
                (left >> (width - 1 - position)) & 1 for left in multipliers
            ]
            context = offline_context(multiplicands, moduli, input_bits, width)
            prior_a = state_a
            prior_b = state_b
            output = model.forward_step(state_a, state_b, context)
            predicted = offline_decode(output.fused_logits)
            next_teacher = [
                (2 * current + bit * other) % modulus
                for current, bit, other, modulus in zip(
                    teacher, input_bits, multiplicands, moduli
                )
            ]
            candidate_bit_steps += sum(active)
            target_bits = offline_bits(next_teacher, width)
            for index, is_active in enumerate(active):
                if is_active and predicted[index] != next_teacher[index]:
                    states_a.append(prior_a[index : index + 1].detach().clone())
                    states_b.append(prior_b[index : index + 1].detach().clone())
                    contexts.append(context[index : index + 1].detach().clone())
                    target_rows.append(target_bits[index : index + 1].detach().clone())
                    active[index] = False
                    if len(states_a) >= needed:
                        break
            state_a = output.next_state_a.detach()
            state_b = output.next_state_b.detach()
            teacher = next_teacher
    if len(states_a) < needed:
        return None
    return NaturalErrorBatch(
        state_a=torch.cat(states_a[:needed], dim=0),
        state_b=torch.cat(states_b[:needed], dim=0),
        context=torch.cat(contexts[:needed], dim=0),
        target_bits=torch.cat(target_rows[:needed], dim=0),
        candidate_pairs=candidate_pairs,
        candidate_bit_steps=candidate_bit_steps,
    )


def train_one_arm(
    *,
    name: str,
    persistence: str,
    config: dict[str, Any],
    splits: dict[int, dict[str, list[int]]],
    seed: int,
    calibration: bool,
    measure_callback: Callable[[str, str, int, Callable[[], Any]], tuple[Any, dict[str, Any]]],
) -> tuple[CoupledRedundantLearnedState, list[dict[str, Any]], bool]:
    section = config["calibration"]["training"] if calibration else config["formal_protocol"]["training"]
    formal_training = config["formal_protocol"]["training"]
    model = new_model(config, persistence, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(formal_training["learning_rate"]),
        weight_decay=float(formal_training["weight_decay"]),
    )
    steps = int(section["updates_per_arm"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(formal_training["minimum_learning_rate"]),
    )
    widths = [int(width) for width in config["data"]["widths"]]
    transition_samplers = {
        width: OfflineSampler(
            splits[width]["train"],
            seed + int(config["data"]["training_seed_offset"]) + width,
        )
        for width in widths
    }
    recovery_samplers = {
        width: OfflineSampler(
            splits[width]["train"],
            seed + int(config["data"]["natural_error_seed_offset"]) + width,
        )
        for width in widths
    }
    records: list[dict[str, Any]] = []
    natural_errors_available = True
    model.train()
    for step in range(1, steps + 1):
        width = widths[(step - 1) % len(widths)]
        recovery = step % int(section["recovery_update_period"]) == 0

        def update() -> dict[str, Any]:
            nonlocal natural_errors_available
            model.train()
            if recovery:
                mined = mine_natural_error_batch(
                    model,
                    recovery_samplers[width],
                    width=width,
                    needed=int(section["recovery_batch_size"]),
                    maximum_candidate_pairs=int(
                        section["maximum_natural_error_candidate_pairs_per_update"]
                    ),
                )
                if mined is None:
                    natural_errors_available = False
                    raise RuntimeError(
                        f"natural first-error cases unavailable for {name} at width {width}"
                    )
                mined = materialize_natural_error_batch(mined)
                output = model.forward_step(mined.state_a, mined.state_b, mined.context)
                targets = mined.target_bits
                candidate_bit_steps = mined.candidate_bit_steps
            else:
                values = transition_samplers[width].transitions(int(section["batch_size"]))
                state, multiplicand, modulus, input_bit, target = values
                belief = offline_state_belief(state, width)
                context = offline_context(multiplicand, modulus, input_bit, width)
                output = model.forward_step(belief, belief.clone(), context)
                targets = offline_bits(target, width)
                candidate_bit_steps = 0
            loss, losses = supervised_loss(output, targets, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(
                model.parameters(), float(formal_training["gradient_clip"])
            )
            if not torch.isfinite(gradient):
                raise RuntimeError(f"non-finite gradient for {name}")
            optimizer.step()
            scheduler.step()
            return {
                "digest": tensor_digest(
                    output.fused_logits,
                    output.channel_a_logits,
                    output.channel_b_logits,
                    losses["fused_loss"].reshape(1),
                ),
                "candidate_bit_steps": candidate_bit_steps,
            }

        result, record = measure_callback(
            "training_update",
            "updates",
            1,
            update,
        )
        record.update(
            {
                "arm": name,
                "width": width,
                "update_kind": "natural-first-error-recovery" if recovery else "clean-transition",
                "update_index": step,
                "output_digest": result["digest"],
                "natural_error_candidate_bit_steps": int(result["candidate_bit_steps"]),
            }
        )
        records.append(record)
    model.eval()
    return model, records, natural_errors_available


def measure_component(
    component: str,
    unit_kind: str,
    units: int,
    function: Callable[[], Any],
) -> tuple[Any, dict[str, Any]]:
    wall_started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    result = function()
    cpu_seconds = process_cpu_seconds() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    return result, {
        "component": component,
        "unit_kind": unit_kind,
        "units": int(units),
        "monotonic_wall_seconds": wall_seconds,
        "process_user_plus_system_aggregate_cpu_seconds": cpu_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
    }


@torch.inference_mode()
def resource_transition_component(
    model: CoupledRedundantLearnedState,
    sampler: OfflineSampler,
    width: int,
    cases: int,
) -> dict[str, Any]:
    state, multiplicand, modulus, input_bit, target = sampler.transitions(cases)
    belief = offline_state_belief(state, width)
    output = model.forward_step(
        belief,
        belief.clone(),
        offline_context(multiplicand, modulus, input_bit, width),
    )
    target_bits = offline_bits(target, width)
    comparison = output.fused_logits.ge(0).eq(target_bits.bool())
    return {
        "digest": tensor_digest(output.fused_logits, target_bits, comparison),
        "cases": cases,
    }


@torch.inference_mode()
def resource_rollout_component(
    model: CoupledRedundantLearnedState,
    sampler: OfflineSampler,
    width: int,
    cases: int,
    batch_size: int,
    *,
    ablation: str = "none",
) -> dict[str, Any]:
    digest = hashlib.sha256()
    processed = 0
    while processed < cases:
        size = min(batch_size, cases - processed)
        multipliers, multiplicands, moduli = sampler.pairs(size)
        state_a, state_b = model.initial_state(size, width)
        teacher = [0 for _ in range(size)]
        for position in range(width):
            input_bits = [
                (left >> (width - 1 - position)) & 1 for left in multipliers
            ]
            output = model.forward_step(
                state_a,
                state_b,
                offline_context(multiplicands, moduli, input_bits, width),
                ablation=ablation,
            )
            teacher = [
                (2 * current + bit * other) % modulus
                for current, bit, other, modulus in zip(
                    teacher, input_bits, multiplicands, moduli
                )
            ]
            target_bits = offline_bits(teacher, width)
            comparison = output.fused_logits.ge(0).eq(target_bits.bool())
            digest.update(
                bytes.fromhex(
                    tensor_digest(
                        output.fused_logits,
                        output.channel_a_logits,
                        output.channel_b_logits,
                        comparison,
                    )
                )
            )
            state_a = output.next_state_a
            state_b = output.next_state_b
        processed += size
    return {"digest": digest.hexdigest(), "case_bit_steps": cases * width}


@torch.inference_mode()
def resource_recovery_component(
    control: CoupledRedundantLearnedState,
    challenger: CoupledRedundantLearnedState,
    sampler: OfflineSampler,
    width: int,
    threshold: int,
    maximum_candidates: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    found = 0
    scanned = 0
    while found < threshold and scanned < maximum_candidates:
        size = min(32, maximum_candidates - scanned)
        state, multiplicand, modulus, input_bit, target = sampler.transitions(size)
        belief = offline_state_belief(state, width)
        context = offline_context(multiplicand, modulus, input_bit, width)
        target_bits = offline_bits(target, width).bool()
        control_output = control.forward_step(belief, belief.clone(), context)
        challenger_output = challenger.forward_step(belief, belief.clone(), context)
        control_case_exact = control_output.fused_logits.ge(0).eq(target_bits).all(1)
        challenger_case_exact = challenger_output.fused_logits.ge(0).eq(target_bits).all(1)
        natural_error = ~control_case_exact
        digest.update(
            bytes.fromhex(
                tensor_digest(
                    natural_error,
                    challenger_case_exact,
                    control_output.fused_logits,
                    challenger_output.fused_logits,
                )
            )
        )
        found += int(natural_error.sum())
        scanned += size
    return {
        "digest": digest.hexdigest(),
        "threshold_met": found >= threshold,
        "candidate_cases": scanned,
        "counts_released": False,
    }


@torch.inference_mode()
def resource_collapse_component(
    model: CoupledRedundantLearnedState,
    sampler: OfflineSampler,
    width: int,
    cases: int,
) -> dict[str, Any]:
    state, multiplicand, modulus, input_bit, target = sampler.transitions(cases)
    belief = offline_state_belief(state, width)
    output = model.forward_step(
        belief,
        belief.clone(),
        offline_context(multiplicand, modulus, input_bit, width),
    )
    target_bits = offline_bits(target, width).bool()
    probability_a = torch.sigmoid(output.channel_a_logits)
    probability_b = torch.sigmoid(output.channel_b_logits)
    centered_a = probability_a - probability_a.mean()
    centered_b = probability_b - probability_b.mean()
    diagnostics = torch.stack(
        (
            (probability_a - probability_b).abs().mean(),
            output.channel_a_logits.ge(0).ne(output.channel_b_logits.ge(0)).float().mean(),
            F.cosine_similarity(centered_a.flatten(), centered_b.flatten(), dim=0),
            output.channel_a_logits.ge(0).ne(target_bits).all(1).float().mean(),
            output.channel_b_logits.ge(0).ne(target_bits).all(1).float().mean(),
        )
    )
    return {
        "digest": tensor_digest(output.channel_a_logits, output.channel_b_logits, diagnostics),
        "cases": cases,
    }


@torch.inference_mode()
def resource_width256_component(
    model: CoupledRedundantLearnedState,
    seed: int,
    cases: int,
) -> dict[str, Any]:
    width = 256
    rng = random.Random(seed)
    lower = 1 << (width - 1)
    moduli = [lower + (rng.getrandbits(width - 2) << 1) + 1 for _ in range(cases)]
    multipliers = [rng.randrange(modulus) for modulus in moduli]
    multiplicands = [rng.randrange(modulus) for modulus in moduli]
    state_a, state_b = model.initial_state(cases, width)
    digest = hashlib.sha256()
    for position in range(width):
        input_bits = [
            (left >> (width - 1 - position)) & 1 for left in multipliers
        ]
        output = model.forward_step(
            state_a,
            state_b,
            offline_context(multiplicands, moduli, input_bits, width),
        )
        digest.update(bytes.fromhex(tensor_digest(output.fused_logits)))
        state_a = output.next_state_a
        state_b = output.next_state_b
    return {"digest": digest.hexdigest(), "case_bit_steps": cases * width}


def rate_max(phases: list[dict[str, Any]], component: str, field: str) -> float:
    matching = [phase for phase in phases if phase["component"] == component]
    if not matching:
        raise RuntimeError(f"calibration did not measure component {component}")
    return max(float(phase[field]) / int(phase["units"]) for phase in matching)


def resolve_dynamic_budget(
    config: dict[str, Any],
    phases: list[dict[str, Any]],
    outer: dict[str, Any],
    checkpoint_bytes: int,
) -> dict[str, Any]:
    formal = config["formal_protocol"]
    training = formal["training"]
    evaluation = formal["evaluation"]
    widths = [int(width) for width in config["data"]["widths"]]
    workload = {
        "training_update": 2 * int(training["updates_per_arm"]),
        "transition_case": 2 * len(widths) * int(evaluation["transition_cases_per_width_arm"]),
        "rollout_case_bit_step": 2
        * int(evaluation["rollout_cases_per_width_arm"])
        * sum(widths),
        "natural_recovery_candidate_bit_step": int(
            evaluation["maximum_held_out_natural_error_candidates"]
        )
        * max(widths),
        "ablation_case_bit_step": 3
        * int(evaluation["causal_ablation_rollout_cases_per_mode"])
        * max(widths),
        "collapse_transition_case": int(evaluation["collapse_transition_cases"]),
        "width256_case_bit_step": int(config["calibration"]["evaluation"]["width256_runtime_projection_cases"])
        * 256,
        "checkpoint_byte": 2 * checkpoint_bytes,
    }
    component_map = {
        "training_update": "training_update",
        "transition_case": "transition_evaluation",
        "rollout_case_bit_step": "rollout_evaluation",
        "natural_recovery_candidate_bit_step": "natural_recovery_evaluation",
        "ablation_case_bit_step": "causal_ablation_evaluation",
        "collapse_transition_case": "collapse_diagnostic",
        "width256_case_bit_step": "width256_runtime_shape",
        "checkpoint_byte": "checkpoint_serialization",
    }
    clocks = (
        "monotonic_wall_seconds",
        "process_user_plus_system_aggregate_cpu_seconds",
    )
    rate_tables: dict[str, dict[str, float]] = {clock: {} for clock in clocks}
    variable: dict[str, float] = {clock: 0.0 for clock in clocks}
    contributions: dict[str, dict[str, float]] = {clock: {} for clock in clocks}
    for workload_key, units in workload.items():
        component = component_map[workload_key]
        for clock in clocks:
            rate = rate_max(phases, component, clock)
            rate_tables[clock][workload_key] = rate
            contribution = rate * units
            contributions[clock][workload_key] = contribution
            variable[clock] += contribution

    phase_wall = sum(float(phase[clocks[0]]) for phase in phases)
    phase_cpu = sum(float(phase[clocks[1]]) for phase in phases)
    observed_setup = {
        clocks[0]: max(0.0, float(outer[clocks[0]]) - phase_wall),
        clocks[1]: max(0.0, float(outer[clocks[1]]) - phase_cpu),
    }
    policy = config["resource_policy"]
    setup_policy = policy["explicit_setup_headroom"]
    headroom = {
        clocks[0]: max(
            float(setup_policy["minimum_wall_seconds"]),
            observed_setup[clocks[0]] * float(setup_policy["measured_setup_fraction"]),
        ),
        clocks[1]: max(
            float(setup_policy["minimum_aggregate_cpu_seconds"]),
            observed_setup[clocks[1]] * float(setup_policy["measured_setup_fraction"]),
        ),
    }
    multiplier = float(policy["throughput_safety_multiplier"])
    hard_multiplier = float(policy["hard_runaway_multiplier_over_soft"])
    projected = {clock: observed_setup[clock] + variable[clock] for clock in clocks}
    soft = {
        clock: observed_setup[clock] + multiplier * variable[clock] + headroom[clock]
        for clock in clocks
    }
    hard = {clock: soft[clock] * hard_multiplier for clock in clocks}

    width256_phases = [phase for phase in phases if phase["component"] == "width256_runtime_shape"]
    width256_cases = int(config["calibration"]["evaluation"]["width256_runtime_projection_cases"])
    projected_cases = int(config["promotion_gates"]["width256_runtime_projection"]["projected_cases"])
    width256_projection = {
        clock: max(float(phase[clock]) / width256_cases for phase in width256_phases)
        * projected_cases
        * multiplier
        for clock in clocks
    }
    return {
        "policy": "maximum-measured-matching-component-rates-with-separate-clocks",
        "formal_workload": workload,
        "maximum_measured_rates": rate_tables,
        "projected_component_contributions": contributions,
        "observed_unassigned_setup_and_publication": observed_setup,
        "explicit_setup_headroom": headroom,
        "throughput_safety_multiplier": multiplier,
        "projected": projected,
        "soft_budget": soft,
        "hard_runaway_ceiling": hard,
        "peak_rss_guard_bytes": math.ceil(
            float(outer["peak_rss_bytes"]) * float(policy["peak_rss_guard_multiplier"])
        ),
        "credible_width256_100_case_projection": width256_projection,
        "fidelity_reduced": False,
        "fixed_short_wall_cap_used": False,
    }


def environment_receipt() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mlflow": mlflow.__version__,
        "psutil": psutil.__version__,
        "safetensors": __import__("safetensors").__version__,
        "sympy": __import__("sympy").__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def calibration_worker(config_path: Path, output: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    calibration_root = int(config["roots"]["calibration"])
    if calibration_root != 20261173:
        raise RuntimeError("T012 permits only calibration root 20261173")
    if output.name != f"calibration-root-{calibration_root}":
        raise RuntimeError("calibration output is not the preregistered root")
    output.mkdir(parents=True, exist_ok=True)
    seed = calibration_root
    set_determinism(seed, int(config["formal_protocol"]["training"]["torch_threads"]))
    worker_wall_started = time.perf_counter()
    worker_cpu_started = process_cpu_seconds()
    splits = build_splits(config, seed)
    split_hash = sha256_text(canonical_json(splits))
    geometry = matched_geometry(config, seed)
    phases: list[dict[str, Any]] = []
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    source = {
        "repository_commit": git_output("rev-parse", "HEAD"),
        "repository_dirty": bool(git_output("status", "--porcelain")),
        "challenge_commit": CHALLENGE_COMMIT,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(HERE / "model.py"),
        "preregistration_sha256": sha256(ROOT / config["preregistration"]),
        "dataset_split_sha256": split_hash,
    }
    mlflow_payload: dict[str, Any]
    with mlflow.start_run(run_name=f"{config['run_id']}-calibration-root-{calibration_root}") as run:
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "stage": "L1-resource-only",
                "root": str(calibration_root),
                "status": "running",
                "scientific_metrics_released": "false",
                "formal_root_opened": "false",
                "repository_commit": source["repository_commit"],
                "repository_dirty": str(source["repository_dirty"]).lower(),
            }
        )
        mlflow.log_params(
            {
                "root": seed,
                "config_sha256": source["config_sha256"],
                "runner_sha256": source["runner_sha256"],
                "model_sha256": source["model_sha256"],
                "dataset_split_sha256": split_hash,
                "parameter_count_each_arm": geometry["parameter_count_each_arm"],
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")

        control, control_records, control_errors = train_one_arm(
            name="single-state-control",
            persistence="single",
            config=config,
            splits=splits,
            seed=seed,
            calibration=True,
            measure_callback=measure_component,
        )
        challenger, challenger_records, challenger_errors = train_one_arm(
            name="coupled-redundant-challenger",
            persistence="redundant",
            config=config,
            splits=splits,
            seed=seed,
            calibration=True,
            measure_callback=measure_component,
        )
        phases.extend(control_records)
        phases.extend(challenger_records)
        widths = [int(width) for width in config["data"]["widths"]]
        calibration_eval = config["calibration"]["evaluation"]
        for arm_name, arm_index, model in (
            ("single-state-control", 0, control),
            ("coupled-redundant-challenger", 1, challenger),
        ):
            for width in widths:
                transition_cases = int(calibration_eval["transition_cases_per_width_arm"])
                transition_sampler = OfflineSampler(
                    splits[width]["test"],
                    seed + int(config["data"]["evaluation_seed_offset"]) + 1000 * arm_index + width,
                )
                result, record = measure_component(
                    "transition_evaluation",
                    "transition_cases",
                    transition_cases,
                    lambda m=model, s=transition_sampler, w=width, c=transition_cases: resource_transition_component(m, s, w, c),
                )
                record.update({"arm": arm_name, "width": width, "output_digest": result["digest"]})
                phases.append(record)

                rollout_cases = int(calibration_eval["rollout_cases_per_width_arm"])
                rollout_sampler = OfflineSampler(
                    splits[width]["test"],
                    seed + int(config["data"]["evaluation_seed_offset"]) + 2000 * arm_index + 100 + width,
                )
                result, record = measure_component(
                    "rollout_evaluation",
                    "case_bit_steps",
                    rollout_cases * width,
                    lambda m=model, s=rollout_sampler, w=width, c=rollout_cases: resource_rollout_component(
                        m,
                        s,
                        w,
                        c,
                        int(calibration_eval["rollout_batch_size"]),
                    ),
                )
                record.update({"arm": arm_name, "width": width, "output_digest": result["digest"]})
                phases.append(record)

        recovery_width = max(widths)
        recovery_sampler = OfflineSampler(
            splits[recovery_width]["test"],
            seed + int(config["data"]["held_out_recovery_seed_offset"]) + recovery_width,
        )
        recovery_result, recovery_record = measure_component(
            "natural_recovery_evaluation",
            "candidate_bit_steps",
            int(calibration_eval["maximum_held_out_natural_error_candidates"]) * recovery_width,
            lambda: resource_recovery_component(
                control,
                challenger,
                recovery_sampler,
                recovery_width,
                int(calibration_eval["held_out_natural_error_threshold"]),
                int(calibration_eval["maximum_held_out_natural_error_candidates"]),
            ),
        )
        # Charge the actual worst possible scanned bit work conservatively, not a
        # favorable error count that would disclose a selection statistic.
        recovery_record["units"] = max(1, int(recovery_result["candidate_cases"]) * recovery_width)
        recovery_record["output_digest"] = recovery_result["digest"]
        phases.append(recovery_record)
        if not recovery_result["threshold_met"]:
            raise RuntimeError("held-out natural control errors unavailable at width 32")

        ablation_cases = int(calibration_eval["causal_ablation_rollout_cases_per_mode"])
        for mode_index, mode in enumerate(("none", "drop_a", "drop_b")):
            sampler = OfflineSampler(
                splits[recovery_width]["test"],
                seed + int(config["data"]["evaluation_seed_offset"]) + 3000 + mode_index,
            )
            result, record = measure_component(
                "causal_ablation_evaluation",
                "case_bit_steps",
                ablation_cases * recovery_width,
                lambda s=sampler, a=mode: resource_rollout_component(
                    challenger,
                    s,
                    recovery_width,
                    ablation_cases,
                    int(calibration_eval["rollout_batch_size"]),
                    ablation=a,
                ),
            )
            record.update({"ablation": mode, "width": recovery_width, "output_digest": result["digest"]})
            phases.append(record)

        collapse_cases = int(calibration_eval["collapse_transition_cases"])
        collapse_sampler = OfflineSampler(
            splits[recovery_width]["test"],
            seed + int(config["data"]["evaluation_seed_offset"]) + 4000,
        )
        collapse_result, collapse_record = measure_component(
            "collapse_diagnostic",
            "transition_cases",
            collapse_cases,
            lambda: resource_collapse_component(
                challenger, collapse_sampler, recovery_width, collapse_cases
            ),
        )
        collapse_record.update({"width": recovery_width, "output_digest": collapse_result["digest"]})
        phases.append(collapse_record)

        width256_cases = int(calibration_eval["width256_runtime_projection_cases"])
        width256_result, width256_record = measure_component(
            "width256_runtime_shape",
            "case_bit_steps",
            width256_cases * 256,
            lambda: resource_width256_component(challenger, seed + 657256, width256_cases),
        )
        width256_record["output_digest"] = width256_result["digest"]
        phases.append(width256_record)

        checkpoints: dict[str, Any] = {}
        checkpoint_bytes = 0
        for name, model in (
            ("single-state-control", control),
            ("coupled-redundant-challenger", challenger),
        ):
            checkpoint = output / f"{name}-calibration-weights.safetensors"

            def serialize(m: CoupledRedundantLearnedState = model, p: Path = checkpoint) -> dict[str, Any]:
                save_file(
                    {key: value.detach().contiguous() for key, value in m.state_dict().items()},
                    p,
                )
                loaded = load_file(p)
                for key, expected in m.state_dict().items():
                    if key not in loaded or not torch.equal(expected.detach().cpu(), loaded[key]):
                        raise RuntimeError(f"checkpoint reload mismatch: {key}")
                return {"sha256": sha256(p), "bytes": p.stat().st_size}

            serialized, record = measure_component(
                "checkpoint_serialization",
                "bytes",
                max(1, geometry["parameter_count_each_arm"] * 4),
                serialize,
            )
            record["units"] = int(serialized["bytes"])
            record["arm"] = name
            record["output_digest"] = serialized["sha256"]
            phases.append(record)
            checkpoint_bytes = max(checkpoint_bytes, int(serialized["bytes"]))
            checkpoints[name] = {
                "path": str(checkpoint.relative_to(ROOT)),
                "sha256": serialized["sha256"],
                "bytes": serialized["bytes"],
                "scientific_checkpoint": False,
            }

        for index, phase in enumerate(phases):
            mlflow.log_metrics(
                {
                    "resource_wall_seconds": float(phase["monotonic_wall_seconds"]),
                    "resource_aggregate_cpu_seconds": float(
                        phase["process_user_plus_system_aggregate_cpu_seconds"]
                    ),
                    "resource_peak_rss_bytes": float(phase["peak_rss_bytes"]),
                },
                step=index,
            )
        mlflow.set_tags(
            {
                "status": "completed-resource-only",
                "decision": "calibration-complete-formal-still-prohibited",
            }
        )
        mlflow_payload = {
            "tracking_uri": tracking_uri(),
            "experiment_name": config["tracking"]["experiment_name"],
            "experiment_id": experiment_id,
            "run_id": run.info.run_id,
            "status": "FINISHED",
            "metrics_scope": "resource-only-no-accuracy-or-selection",
        }

    environment = environment_receipt()
    environment_path = output / "environment.json"
    mlflow_path = output / "mlflow.json"
    atomic_json(environment_path, environment)
    atomic_json(mlflow_path, mlflow_payload)
    core = {
        "schema_version": "exp657-resource-calibration-core-v1",
        "run_id": config["run_id"],
        "stage": "L1",
        "root": calibration_root,
        "status": "completed-resource-only-child",
        "source": source,
        "geometry": geometry,
        "workload": {
            "every_arm_trained": True,
            "all_planned_evaluation_components_exercised": True,
            "natural_first_error_training_available_for_every_arm": bool(
                control_errors and challenger_errors
            ),
            "held_out_natural_control_error_threshold_met": bool(
                recovery_result["threshold_met"]
            ),
            "natural_error_counts_released": False,
            "generic_or_random_synthetic_corruption_used": False,
            "cache_reuse": False,
            "fidelity_reduced": False,
        },
        "phases": phases,
        "checkpoints": checkpoints,
        "checkpoint_bytes_for_projection": checkpoint_bytes,
        "mlflow": mlflow_payload,
        "worker_boundary": {
            "monotonic_wall_seconds": time.perf_counter() - worker_wall_started,
            "process_user_plus_system_aggregate_cpu_seconds": process_cpu_seconds()
            - worker_cpu_started,
            "peak_rss_bytes": peak_rss_bytes(),
            "scope": "child-entry-through-pre-core-publication-sample",
        },
        "boundary": {
            "scientific_accuracy_or_selection_reported": False,
            "model_update_for_scientific_claim": False,
            "formal_root_opened": False,
            "checked_independent_or_private_answers_read": False,
            "candidate_or_submission_changed": False,
        },
        "limitations": [
            "Calibration weights exist only to measure representative update, serialization, and evaluation-shape work; they are not scientific checkpoints.",
            "Natural-error availability is released only as a preregistered threshold boolean; no correct/error numerator, accuracy, or selection metric is released.",
            "The child boundary excludes parent-observed process exit and final receipt publication; the outer completion receipt is authoritative for those events.",
        ],
    }
    atomic_json(output / "calibration-core.json", core)
    return 0


def calibration_parent(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    calibration_root = int(config["roots"]["calibration"])
    output = (ROOT / config["calibration"]["output_dir"]).resolve()
    expected = (DEFAULT_RUN_DIR / f"calibration-root-{calibration_root}").resolve()
    if output != expected:
        raise RuntimeError("calibration output path differs from T012 preregistration")
    completion_path = output / "completion.json"
    if completion_path.exists():
        raise RuntimeError(
            f"calibration root {calibration_root} is already consumed and non-retryable"
        )
    output.mkdir(parents=True, exist_ok=True)
    attempt_state = {
        "schema_version": "exp657-attempt-state-v1",
        "run_id": config["run_id"],
        "root": calibration_root,
        "stage": "L1-resource-only",
        "status": "running",
        "formal_root_opened": False,
        "started_at_unix_ns": time.time_ns(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(HERE / "model.py"),
    }
    atomic_json(output / "attempt-state.json", attempt_state)
    outer_wall_started = time.perf_counter()
    outer_cpu_started = combined_rusage_cpu()
    command = [
        sys.executable,
        str(Path(__file__)),
        "--config",
        str(config_path),
        "--mode",
        "calibration",
        "--internal-calibration-worker",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    outer = {
        "monotonic_wall_seconds": time.perf_counter() - outer_wall_started,
        "process_user_plus_system_aggregate_cpu_seconds": combined_rusage_cpu()
        - outer_cpu_started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024),
        "child_exit_code": int(completed.returncode),
        "scope": "parent-prelaunch-through-observed-child-process-exit",
    }
    atomic_text(output / "worker-stdout.txt", completed.stdout)
    atomic_text(output / "worker-stderr.txt", completed.stderr)
    if completed.returncode != 0:
        failure = {
            "schema_version": "exp657-calibration-completion-v1",
            "run_id": config["run_id"],
            "root": calibration_root,
            "stage": "L1-resource-only",
            "status": "failed-child-exited",
            "outer_resource_boundary": outer,
            "scientific_accuracy_or_selection_reported": False,
            "formal_root_opened": False,
            "stderr_sha256": sha256(output / "worker-stderr.txt"),
        }
        atomic_json(completion_path, failure)
        attempt_state.update({"status": "consumed-failed", "completion_sha256": sha256(completion_path)})
        atomic_json(output / "attempt-state.json", attempt_state)
        sys.stderr.write(completed.stderr)
        return completed.returncode

    core_path = output / "calibration-core.json"
    if not core_path.is_file():
        raise RuntimeError("calibration child exited zero without a core receipt")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    budget = resolve_dynamic_budget(
        config,
        core["phases"],
        outer,
        int(core["checkpoint_bytes_for_projection"]),
    )
    executed_config = dict(config)
    executed_config["resolved_resource_budget"] = budget
    executed_config["execution_boundary"] = {
        "root": calibration_root,
        "stage": "L1-resource-only",
        "formal_roots_opened": [],
        "scientific_metrics_released": False,
    }
    executed_config_path = output / "executed-config.yaml"
    atomic_yaml(executed_config_path, executed_config)
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"completion.json", "calibration.json", "attempt-state.json"}:
            artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    completion = {
        "schema_version": "exp657-calibration-completion-v1",
        "run_id": config["run_id"],
        "root": calibration_root,
        "stage": "L1-resource-only",
        "status": "completed-child-exited-zero",
        "outer_resource_boundary": outer,
        "artifacts_before_completion": artifacts,
        "scientific_accuracy_or_selection_reported": False,
        "formal_root_opened": False,
        "completion_observed_after_child_exit": True,
        "post_sample_exclusions": [
            "completion-json-serialization-and-atomic-replace",
            "canonical-calibration-receipt-serialization-and-atomic-replace",
            "attempt-state-terminal-rewrite",
            "cli-summary-stdout",
            "parent-return-interpreter-teardown-and-exit",
        ],
    }
    atomic_json(completion_path, completion)
    receipt = {
        **core,
        "schema_version": "exp657-resource-calibration-v1",
        "status": "consumed-complete-non-scientific",
        "outer_resource_boundary": outer,
        "resolved_resource_budget": budget,
        "executed_config": {
            "path": str(executed_config_path.relative_to(ROOT)),
            "sha256": sha256(executed_config_path),
        },
        "completion": {
            "path": str(completion_path.relative_to(ROOT)),
            "sha256": sha256(completion_path),
            "child_exit_code": completed.returncode,
        },
        "next_authorization_boundary": {
            "stage": "L2-first-formal",
            "minimum_root": int(config["roots"]["first_formal"]),
            "authorization_required": True,
            "authorization_exists": False,
            "L3_status": "prohibited-until-L2-promotion-and-separate-authorization",
        },
    }
    receipt_path = output / "calibration.json"
    atomic_json(receipt_path, receipt)
    attempt_state.update(
        {
            "status": "consumed-complete-non-scientific",
            "completion_sha256": sha256(completion_path),
            "calibration_receipt_sha256": sha256(receipt_path),
            "finished_at_unix_ns": time.time_ns(),
        }
    )
    atomic_json(output / "attempt-state.json", attempt_state)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "root": calibration_root,
                "outer_resource_boundary": outer,
                "resolved_resource_budget": budget,
                "receipt_sha256": sha256(receipt_path),
                "scientific_accuracy_or_selection_reported": False,
                "formal_root_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_formal_authorization(config_path: Path, root: int | None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    first_formal = int(config["roots"]["first_formal"])
    if root is None or root < first_formal:
        raise RuntimeError(
            f"formal root must be explicitly supplied and at least {first_formal}"
        )
    authorization_path = ROOT / config["stages"]["L2"]["authorization_file"]
    if not authorization_path.is_file():
        raise RuntimeError("formal execution is prohibited: authorization file is absent")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    required = {
        "experiment": config["run_id"],
        "stage": "L2-first-formal",
        "root": root,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(HERE / "model.py"),
        "calibration_receipt_sha256": sha256(
            DEFAULT_RUN_DIR
            / f"calibration-root-{int(config['roots']['calibration'])}/calibration.json"
        ),
        "launch": True,
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise RuntimeError(f"formal authorization mismatch for {key}")
    return authorization


def best_attainable_rate(numerator: int, observed: int, total: int) -> float:
    if not (0 <= numerator <= observed <= total):
        raise ValueError("invalid attainable-bound counts")
    return (numerator + total - observed) / total


def best_attainable_paired_difference(
    positive_numerator: int,
    adverse_numerator: int,
    observed: int,
    total: int,
) -> float:
    if not (0 <= positive_numerator <= observed <= total):
        raise ValueError("invalid positive paired counts")
    if not (0 <= adverse_numerator <= observed):
        raise ValueError("invalid adverse paired counts")
    return (positive_numerator + total - observed - adverse_numerator) / total


def formal_entry(config_path: Path, root: int | None) -> int:
    validate_formal_authorization(config_path, root)
    # The calibration package deliberately ends at the authorization boundary.
    # A later Worker must bind the calibrated budgets and formal evidence
    # publication path before this branch may train.  Keeping this fail-closed is
    # preferable to silently opening a scientific root from calibration code.
    raise RuntimeError(
        "formal authorization validated, but T010 calibration package forbids opening L2; "
        "a separately authorized formal Worker package is required"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("calibration", "formal"), required=True)
    parser.add_argument("--root", type=int)
    parser.add_argument("--internal-calibration-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if args.mode == "formal":
        if args.internal_calibration_worker:
            raise RuntimeError("internal calibration worker cannot enter formal mode")
        return formal_entry(config_path, args.root)
    if args.root is not None:
        raise RuntimeError("calibration root is frozen in the preregistration")
    if args.internal_calibration_worker:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output = (ROOT / config["calibration"]["output_dir"]).resolve()
        return calibration_worker(config_path, output)
    return calibration_parent(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
