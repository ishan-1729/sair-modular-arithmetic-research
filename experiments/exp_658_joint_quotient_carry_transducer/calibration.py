"""Inert, contract-complete EXP-658 L1 resource-calibration scaffold.

This module deliberately contains no authorized execution path.  It imports the
frozen EXP-658 L0 model, label/data, and run modules and exposes pure contract
resolution plus tiny unit-scale mechanics.  A later committed Judge package
must add and bind authorization before any root, output directory, MLflow run,
Candidate-005 invocation, optimizer-backed workload, or durable receipt exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from sympy import __version__ as SYMPY_VERSION
from sympy import nextprime
from torch import Tensor, nn

try:
    from . import data as frozen_data
    from . import model as frozen_model
    from . import run as frozen_run
except ImportError:  # pragma: no cover - direct-script import path
    import data as frozen_data
    import model as frozen_model
    import run as frozen_run


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/exp_658_joint_quotient_carry_transducer.yaml"
PREREGISTRATION_PATH = (
    ROOT
    / "knowledge/autoresearch/ablations/exp-658-joint-quotient-carry-transducer.md"
)
CALIBRATION_TEST_PATH = ROOT / "tests/test_exp658_joint_quotient_carry_calibration.py"
RUN_ID = "exp-658-joint-quotient-carry-transducer"
BASE_SEED = 20261176
SOURCE_CHECKPOINT_SHA256 = (
    "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
)
L0_BASE_COMMIT = "fc4e0605388812f8da619956063c5d49d9b8d2e3"
AUTHORIZATION_SCHEMA = "exp658-l1-resource-calibration-authorization-v1"
RESOURCE_RECEIPT_SCHEMA = "exp658-l1-resource-calibration-receipt-v1"
PRIOR_MEASUREMENT_AUTHORIZATION_SCHEMA = (
    "exp658-prior-phase-resource-measurement-authorization-v1"
)
MEASUREMENT_RECORD_SCHEMA = "exp658-cumulative-resource-measurement-v1"
PINNED_TORCH_VERSION = "2.5.1+cpu"
PINNED_SYMPY_VERSION = "1.13.1"
PINNED_CANDIDATE_IMAGE = "modchallenge-sandbox:82510bba00a1"
EXP300_SPLIT_PATH = (
    ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"
)
EXP300_SPLIT_SHA256 = (
    "9725303f4151554cb1f877cd92a2db6df911726f27f93fe5798c6d6e9706506f"
)
EXPECTED_SPLIT_SHA256 = {
    32: "f014bd47e7eeb0f3e3f8c43c7422cb154d9ce9fd93bc840cb6eaf473633a67bd",
    64: "f49a8a17c3e3a40b8ee96d2ed5d8648038af79b46ed7e645d06b83e870025fd7",
}
EXPECTED_WIDTH256_STREAM_SHA256 = (
    "623b459ecca48cd83dbebdd64c57afccf4d4dfd264da763f956c818e565582ee"
)
ADAMW_DEFAULTS = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
}
SEED_NAMESPACES = frozenset(
    {
        "split",
        "init",
        "train",
        "eval_transition",
        "eval_rollout",
        "rare_q",
        "rare_carry",
        "profile_warmup",
        "profile_fresh",
    }
)
FROZEN_L0_HASHES = {
    "init": (
        "experiments/exp_658_joint_quotient_carry_transducer/__init__.py",
        "e323af4b994d471feef56a86ee0d993d316672e8786b5961ef686cddbb57547a",
    ),
    "model": (
        "experiments/exp_658_joint_quotient_carry_transducer/model.py",
        "44ccbdbada2110313c982cff211e7e9fcc28c13a6f0ca8f793817a3f99c65dda",
    ),
    "data": (
        "experiments/exp_658_joint_quotient_carry_transducer/data.py",
        "c998e47e0867ed97f7ce176c8854835bf8b687cd33ff018bfef70443327eefba",
    ),
    "run": (
        "experiments/exp_658_joint_quotient_carry_transducer/run.py",
        "da669dc0771bc69708f96e2e23558b3f756fd9af8758f3240571a3c12a211d15",
    ),
    "l0_test": (
        "tests/test_exp658_joint_quotient_carry_transducer.py",
        "8c6a4b3352e2229d3fded622637f350b56f6a99ec42771f664cb5a98f33ebbd6",
    ),
}
STRATA = ("natural", "q2", "boundary", "rare_carry")
SIGNED_CARRY_CLASS_MAP = {-2: 0, -1: 1, 0: 2, 1: 3}
APPORTIONMENT = {
    32: (
        (64, 26, 19, 19),
        (64, 26, 19, 19),
        (64, 26, 19, 19),
        (64, 26, 19, 19),
        (64, 26, 19, 19),
        (64, 25, 20, 19),
        (64, 25, 20, 19),
        (64, 25, 19, 20),
    ),
    64: (
        (64, 26, 19, 19),
        (64, 26, 19, 19),
        (64, 26, 19, 19),
        (64, 26, 19, 19),
        (64, 26, 19, 19),
        (64, 25, 19, 20),
        (64, 25, 19, 20),
        (64, 25, 20, 19),
    ),
}


class ContractError(RuntimeError):
    """Raised when the frozen T028 resource-only contract is inconsistent."""


class AuthorizationError(RuntimeError):
    """Raised before every side effect while the scaffold remains inert."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_frozen_l0_hashes(root: Path = ROOT) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, (relative, expected) in FROZEN_L0_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise ContractError(f"frozen {name} path is absent: {relative}")
        actual = frozen_run.sha256(path)
        if actual != expected:
            raise ContractError(f"frozen {name} SHA-256 mismatch")
        observed[name] = actual
    return observed


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = frozen_run.load_config(path)
    frozen_run.validate_l0_contract(config)
    return config


def derive_seed(namespace: str, *parts: object, base_seed: int = BASE_SEED) -> int:
    """Implement T028's exact domain-separated low-63-bit seed function."""

    if namespace not in SEED_NAMESPACES:
        raise ValueError(f"unsupported EXP-658 seed namespace: {namespace}")
    suffix = "|".join(str(part) for part in parts)
    material = f"exp658|{base_seed}|{namespace}|{suffix}".encode("utf-8")
    first_eight = hashlib.sha256(material).digest()[:8]
    return int.from_bytes(first_eight, "big") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class IdentitySplit:
    width: int
    minimum_bits: int
    maximum_bits: int
    train: tuple[int, ...]
    test: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class Width256IdentityStream:
    width: int
    minimum_bits: int
    maximum_bits: int
    seed: int
    moduli: tuple[int, ...]
    warmup: tuple[int, ...]
    fresh_by_repeat: Mapping[int, tuple[int, ...]]
    sha256: str


def _next_prime_strictly_above(value: int) -> int:
    if SYMPY_VERSION != PINNED_SYMPY_VERSION:
        raise ContractError("pinned SymPy version changed")
    return int(nextprime(value))


def frozen_identity_split(width: int) -> IdentitySplit:
    """Reproduce EXP-300/609 ``bigint_prime_split`` without importing MLflow."""

    if width not in (32, 64):
        raise ValueError("L1 identity splits are restricted to widths 32 and 64")
    minimum_bits = width - 7
    maximum_bits = width
    low = 1 << (width - 8)
    high = 1 << width
    rng = random.Random(derive_seed("split", width))
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < 640:
        prime = _next_prime_strictly_above(rng.randrange(low, high))
        if prime < high and prime not in seen:
            seen.add(prime)
            values.append(prime)
    ordered = tuple(values)
    if any(
        not minimum_bits <= value.bit_length() <= maximum_bits
        or not frozen_data.is_prime(value)
        for value in ordered
    ):
        raise ContractError("frozen split contains a non-prime or out-of-span identity")
    train = ordered[:512]
    test = ordered[512:]
    if set(train).intersection(test):
        raise ContractError("train/test identities overlap")
    payload = {
        "width": width,
        "algorithm_sha256": EXP300_SPLIT_SHA256,
        "minimum_bits": minimum_bits,
        "maximum_bits": maximum_bits,
        "split_seed": derive_seed("split", width),
        "train": train,
        "test": test,
    }
    result = IdentitySplit(
        width=width,
        minimum_bits=minimum_bits,
        maximum_bits=maximum_bits,
        train=train,
        test=test,
        sha256=sha256_bytes(canonical_json_bytes(payload)),
    )
    if result.sha256 != EXPECTED_SPLIT_SHA256[width]:
        raise ContractError("frozen EXP-609 split hash does not match preregistration")
    return result


def width256_heldout_identity_stream() -> Width256IdentityStream:
    """Derive the one canonical, domain-separated 4+3x8 W256 prime stream."""

    width = 256
    minimum_bits = width - 7
    maximum_bits = width
    low = 1 << (width - 8)
    high = 1 << width
    seed = derive_seed("profile_warmup", "heldout-width256-prime-stream", width)
    rng = random.Random(seed)
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < 28:
        prime = _next_prime_strictly_above(rng.randrange(low, high))
        if prime < high and prime not in seen:
            seen.add(prime)
            values.append(prime)
    moduli = tuple(values)
    warmup = moduli[:4]
    fresh = {
        repeat: moduli[4 + (repeat - 1) * 8 : 4 + repeat * 8]
        for repeat in (1, 2, 3)
    }
    if (
        any(
            not minimum_bits <= value.bit_length() <= maximum_bits
            or value <= 0
            or not frozen_data.is_prime(value)
            for value in moduli
        )
        or len(set(moduli)) != 28
        or set(warmup).intersection(
            value for rows in fresh.values() for value in rows
        )
        or len({value for rows in fresh.values() for value in rows}) != 24
    ):
        raise ContractError("width-256 held-out prime stream is invalid")
    payload = {
        "width": width,
        "minimum_bits": minimum_bits,
        "maximum_bits": maximum_bits,
        "stream_seed": seed,
        "count": len(moduli),
        "moduli": moduli,
        "warmup": warmup,
        "fresh": {str(key): value for key, value in fresh.items()},
    }
    digest = sha256_bytes(canonical_json_bytes(payload))
    if digest != EXPECTED_WIDTH256_STREAM_SHA256:
        raise ContractError("width-256 held-out stream hash changed")
    return Width256IdentityStream(
        width=width,
        minimum_bits=minimum_bits,
        maximum_bits=maximum_bits,
        seed=seed,
        moduli=moduli,
        warmup=warmup,
        fresh_by_repeat=fresh,
        sha256=digest,
    )


def stream_seed_manifest(repeats: Sequence[int] = (0, 1, 2, 3)) -> dict[str, int]:
    manifest: dict[str, int] = {}
    for repeat in repeats:
        for width in (32, 64):
            manifest[f"init/r{repeat}"] = derive_seed("init", repeat)
            manifest[f"train/r{repeat}/w{width}"] = derive_seed(
                "train", repeat, width
            )
            manifest[f"transition/r{repeat}/w{width}"] = derive_seed(
                "eval_transition", repeat, width
            )
            manifest[f"rollout/r{repeat}/w{width}"] = derive_seed(
                "eval_rollout", repeat, width
            )
    for width in (32, 64):
        manifest[f"rare_q/w{width}"] = derive_seed("rare_q", width)
        manifest[f"rare_carry/w{width}"] = derive_seed("rare_carry", width)
    manifest["profile/warmup/w256"] = derive_seed("profile_warmup", 256)
    manifest["profile/heldout-prime-stream/w256"] = derive_seed(
        "profile_warmup", "heldout-width256-prime-stream", 256
    )
    for repeat in (1, 2, 3):
        manifest[f"profile/fresh{repeat}/w256"] = derive_seed(
            "profile_fresh", repeat, 256
        )
    if len(set(manifest.values())) != len(manifest):
        raise ContractError("domain-separated seed streams collide")
    return manifest


@dataclass(frozen=True, slots=True)
class BatchQuota:
    global_step: int
    width: int
    width_appearance: int
    natural: int
    q2: int
    boundary: int
    rare_carry: int

    @property
    def total(self) -> int:
        return self.natural + self.q2 + self.boundary + self.rare_carry


def training_schedule(cycles: int = 1) -> tuple[BatchQuota, ...]:
    """Return 16 alternating batches per cycle; L2 is exactly 25 cycles."""

    if cycles < 1:
        raise ValueError("cycle count must be positive")
    appearances = {32: 0, 64: 0}
    schedule: list[BatchQuota] = []
    for global_step in range(1, 16 * cycles + 1):
        width = 32 if global_step % 2 else 64
        appearances[width] += 1
        quota = APPORTIONMENT[width][(appearances[width] - 1) % 8]
        item = BatchQuota(global_step, width, appearances[width], *quota)
        if item.total != 128:
            raise ContractError("apportioned training batch is not 128 rows")
        schedule.append(item)
    return tuple(schedule)


def apportionment_totals(schedule: Sequence[BatchQuota]) -> dict[str, int]:
    return {
        stratum: sum(getattr(item, stratum) for item in schedule)
        for stratum in STRATA
    } | {"total": sum(item.total for item in schedule)}


@dataclass(frozen=True, slots=True)
class StratumCandidate:
    identity: tuple[int, ...]
    width: int
    state: int
    multiplicand: int
    modulus: int
    operand_bit: int
    raw_u: int
    quotient_label: int
    signed_carry_labels: tuple[int, ...]

    @classmethod
    def from_transition(
        cls,
        identity: tuple[int, ...],
        transition: frozen_data.LegalTransition,
        width: int,
    ) -> "StratumCandidate":
        trace = frozen_data.quotient_carry_trace(transition, width)
        return cls(
            identity=identity,
            width=width,
            state=transition.state,
            multiplicand=transition.multiplicand,
            modulus=transition.modulus,
            operand_bit=transition.operand_bit,
            raw_u=2 * transition.state
            + transition.operand_bit * transition.multiplicand,
            quotient_label=trace.quotient,
            signed_carry_labels=trace.carry_out,
        )

    def validate_raw_labels(self) -> None:
        if not self.identity:
            raise ContractError("stratum identity cannot be empty")
        transition = frozen_data.LegalTransition(
            self.state,
            self.multiplicand,
            self.modulus,
            self.operand_bit,
        )
        trace = frozen_data.quotient_carry_trace(transition, self.width)
        expected_u = 2 * self.state + self.operand_bit * self.multiplicand
        if self.raw_u != expected_u:
            raise ContractError("raw u does not match the legal transition")
        if self.quotient_label != trace.quotient:
            raise ContractError("quotient label does not match raw u and p")
        if self.signed_carry_labels != trace.carry_out:
            raise ContractError("signed-carry labels do not match exact recurrence")

    @property
    def q2_eligible(self) -> bool:
        return self.quotient_label == 2

    @property
    def boundary_eligible(self) -> bool:
        margin = min(
            abs(self.raw_u - self.modulus),
            abs(self.raw_u - 2 * self.modulus),
        )
        return 20 * margin <= self.modulus

    @property
    def rare_carry_eligible(self) -> bool:
        return any(value in (-2, 1) for value in self.signed_carry_labels)


@dataclass(frozen=True, slots=True)
class StratumSelection:
    selected: Mapping[str, tuple[tuple[int, ...], ...]]
    requested: Mapping[str, int]
    realized_before_fallback: Mapping[str, int]
    fallback_natural: int
    natural_semantic_overlap: Mapping[str, int]


def select_training_strata(
    candidates: Sequence[StratumCandidate],
    quota: BatchQuota,
    *,
    attempt_allowance: int,
) -> StratumSelection:
    """Apply exact precedence and natural-only training deficit fallback."""

    if attempt_allowance < quota.total:
        raise ValueError("attempt allowance cannot be smaller than the batch")
    pool = tuple(candidates[:attempt_allowance])
    for item in pool:
        item.validate_raw_labels()
    if len({item.identity for item in pool}) != len(pool):
        raise ContractError("candidate stream contains duplicate tuple identities")
    requested = {stratum: getattr(quota, stratum) for stratum in STRATA}
    chosen: dict[str, list[tuple[int, ...]]] = {stratum: [] for stratum in STRATA}
    used: set[tuple[int, ...]] = set()

    def take(stratum: str, predicate: Callable[[StratumCandidate], bool]) -> None:
        for item in pool:
            if len(chosen[stratum]) >= requested[stratum]:
                break
            if item.identity not in used and predicate(item):
                chosen[stratum].append(item.identity)
                used.add(item.identity)

    take("natural", lambda _item: True)
    take("q2", lambda item: item.q2_eligible)
    take(
        "boundary",
        lambda item: item.boundary_eligible and not item.q2_eligible,
    )
    take(
        "rare_carry",
        lambda item: item.rare_carry_eligible
        and not item.q2_eligible
        and not item.boundary_eligible,
    )
    realized = {name: len(values) for name, values in chosen.items()}
    deficit = quota.total - sum(realized.values())
    if deficit:
        for item in pool:
            if deficit == 0:
                break
            if item.identity not in used:
                chosen["natural"].append(item.identity)
                used.add(item.identity)
                deficit -= 1
    if deficit:
        raise ContractError("deterministic attempt allowance cannot fill batch")
    overlap = {
        "q2": sum(
            item.q2_eligible for item in pool if item.identity in chosen["natural"]
        ),
        "boundary": sum(
            item.boundary_eligible
            for item in pool
            if item.identity in chosen["natural"]
        ),
        "rare_carry": sum(
            item.rare_carry_eligible
            for item in pool
            if item.identity in chosen["natural"]
        ),
    }
    return StratumSelection(
        selected={name: tuple(values) for name, values in chosen.items()},
        requested=requested,
        realized_before_fallback=realized,
        fallback_natural=sum(requested.values()) - sum(realized.values()),
        natural_semantic_overlap=overlap,
    )


def select_rare_without_fallback(
    candidates: Sequence[str], *, count: int
) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) < count:
        raise ContractError("rare evaluation deficit has no fallback")
    return unique[:count]


@dataclass(frozen=True, slots=True)
class OwnedCase:
    identity: tuple[int, ...]
    source_identity: tuple[int, ...]
    width: int
    modulus: int
    repeat: int
    arm: str
    label: int | None = None
    state: int | None = None
    multiplier: int | None = None
    multiplicand: int | None = None
    operand_bit: int | None = None
    position: int | None = None


@dataclass(frozen=True, slots=True)
class L1IdentityManifest:
    training: tuple[OwnedCase, ...]
    transitions: tuple[OwnedCase, ...]
    normal_rollouts: tuple[OwnedCase, ...]
    zero_quotient_rollouts: tuple[OwnedCase, ...]
    zero_carry_rollouts: tuple[OwnedCase, ...]
    rare_quotient: tuple[OwnedCase, ...]
    rare_carry: tuple[OwnedCase, ...]
    determinism_repeat: tuple[OwnedCase, ...]
    profile_warmup: tuple[OwnedCase, ...]
    profile_fresh: tuple[OwnedCase, ...]


def _case_payload(case: OwnedCase) -> dict[str, Any]:
    return {
        "identity": case.identity,
        "source_identity": case.source_identity,
        "width": case.width,
        "modulus": case.modulus,
        "repeat": case.repeat,
        "arm": case.arm,
        "label": case.label,
        "state": case.state,
        "multiplier": case.multiplier,
        "multiplicand": case.multiplicand,
        "operand_bit": case.operand_bit,
        "position": case.position,
    }


def _draw_source_transition(
    rng: random.Random, moduli: Sequence[int]
) -> tuple[int, int, int, int]:
    """Draw exactly one EXP-300 ``BigIntSampler.transitions`` raw tuple."""

    modulus = rng.choice(moduli)
    return (
        rng.randrange(modulus),
        rng.randrange(modulus),
        modulus,
        rng.randrange(2),
    )


def _owned_transition(
    raw: tuple[int, int, int, int],
    *,
    width: int,
    repeat: int,
    arm: str,
    label: int | None = None,
    position: int | None = None,
) -> OwnedCase:
    state, multiplicand, modulus, operand_bit = raw
    identity = raw if position is None else (*raw, position)
    return OwnedCase(
        identity=identity,
        source_identity=raw,
        width=width,
        modulus=modulus,
        repeat=repeat,
        arm=arm,
        label=label,
        state=state,
        multiplicand=multiplicand,
        operand_bit=operand_bit,
        position=position,
    )


def _source_rollout_pairs(
    rng: random.Random,
    moduli: Sequence[int],
    count: int,
    *,
    excluded: set[tuple[int, ...]],
) -> tuple[tuple[int, int, int], ...]:
    """Reproduce batched ``BigIntSampler.pairs`` and skip duplicate raw tuples."""

    rows: list[tuple[int, int, int]] = []
    while len(rows) < count:
        batch_size = count - len(rows)
        selected_moduli = [rng.choice(moduli) for _ in range(batch_size)]
        multipliers = [rng.randrange(modulus) for modulus in selected_moduli]
        multiplicands = [rng.randrange(modulus) for modulus in selected_moduli]
        for multiplier, multiplicand, modulus in zip(
            multipliers, multiplicands, selected_moduli, strict=True
        ):
            raw = (multiplier, multiplicand, modulus)
            if raw in excluded:
                continue
            excluded.add(raw)
            rows.append(raw)
    return tuple(rows)


def _owned_rollout(
    raw: tuple[int, int, int],
    *,
    width: int,
    repeat: int,
    arm: str,
) -> OwnedCase:
    multiplier, multiplicand, modulus = raw
    return OwnedCase(
        identity=raw,
        source_identity=raw,
        width=width,
        modulus=modulus,
        repeat=repeat,
        arm=arm,
        multiplier=multiplier,
        multiplicand=multiplicand,
    )


@lru_cache(maxsize=None)
def canonical_l1_identity_manifest(
    deterministic_attempt_allowance: int,
) -> L1IdentityManifest:
    """Regenerate every L1 raw case from T028 seeds and the source sampler.

    The numeric attempt allowance remains a later authorization input; it has no
    default.  It controls how many deterministic source draws are available to
    each stratified training batch, without changing any denominator.
    """

    if (
        type(deterministic_attempt_allowance) is not int
        or deterministic_attempt_allowance < 128
    ):
        raise ContractError("future authorization must supply a numeric attempt allowance")
    splits = {width: frozen_identity_split(width) for width in (32, 64)}
    transition_sources: set[tuple[int, ...]] = set()
    rollout_sources: set[tuple[int, ...]] = set()
    training: list[OwnedCase] = []
    transitions: list[OwnedCase] = []
    normal_rollouts: list[OwnedCase] = []
    zero_quotient_rollouts: list[OwnedCase] = []
    zero_carry_rollouts: list[OwnedCase] = []
    normal_by_repeat_width: dict[tuple[int, int], tuple[OwnedCase, ...]] = {}

    for repeat in (0, 1, 2, 3):
        for width in (32, 64):
            split = splits[width]
            train_rng = random.Random(derive_seed("train", repeat, width))
            for quota in (
                row for row in training_schedule() if row.width == width
            ):
                pool: list[StratumCandidate] = []
                attempted: set[tuple[int, ...]] = set()
                while len(pool) < deterministic_attempt_allowance:
                    raw = _draw_source_transition(train_rng, split.train)
                    if raw in attempted or raw in transition_sources:
                        continue
                    attempted.add(raw)
                    transition = frozen_data.LegalTransition(*raw)
                    pool.append(
                        StratumCandidate.from_transition(raw, transition, width)
                    )
                selected = select_training_strata(
                    pool,
                    quota,
                    attempt_allowance=deterministic_attempt_allowance,
                )
                by_identity = {item.identity: item for item in pool}
                for stratum in STRATA:
                    for identity in selected.selected[stratum]:
                        raw = tuple(identity)
                        if raw in transition_sources:
                            raise ContractError("canonical training stream overlaps")
                        transition_sources.add(raw)
                        training.append(
                            _owned_transition(
                                raw, width=width, repeat=repeat, arm="shared"
                            )
                        )
                        if by_identity[identity].width != width:
                            raise ContractError("canonical training width changed")

            transition_rng = random.Random(
                derive_seed("eval_transition", repeat, width)
            )
            while len(
                [
                    row
                    for row in transitions
                    if row.repeat == repeat and row.width == width
                ]
            ) < 512:
                raw = _draw_source_transition(transition_rng, split.test)
                if raw in transition_sources:
                    continue
                transition_sources.add(raw)
                transitions.append(
                    _owned_transition(raw, width=width, repeat=repeat, arm="shared")
                )

            rollout_rng = random.Random(derive_seed("eval_rollout", repeat, width))
            raw_rollouts = _source_rollout_pairs(
                rollout_rng, split.test, 32, excluded=rollout_sources
            )
            owned = tuple(
                _owned_rollout(raw, width=width, repeat=repeat, arm="shared")
                for raw in raw_rollouts
            )
            normal_rollouts.extend(owned)
            normal_by_repeat_width[(repeat, width)] = owned
            zero_quotient_rollouts.extend(
                OwnedCase(**(_case_payload(row) | {"arm": "challenger"}))
                for row in owned
            )
            zero_carry_rollouts.extend(
                OwnedCase(**(_case_payload(row) | {"arm": "challenger"}))
                for row in owned
            )

    rare_quotient: list[OwnedCase] = []
    rare_carry: list[OwnedCase] = []
    rare_q_sources: set[tuple[int, ...]] = set()
    for width in (32, 64):
        split = splits[width]
        remaining_q = {0: 256, 1: 256, 2: 256}
        q_rng = random.Random(derive_seed("rare_q", width))
        while any(remaining_q.values()):
            raw = _draw_source_transition(q_rng, split.test)
            if raw in transition_sources or raw in rare_q_sources:
                continue
            trace = frozen_data.quotient_carry_trace(
                frozen_data.LegalTransition(*raw), width
            )
            label = trace.quotient
            if remaining_q[label] <= 0:
                continue
            rare_q_sources.add(raw)
            rare_quotient.append(
                _owned_transition(
                    raw,
                    width=width,
                    repeat=3,
                    arm="challenger",
                    label=label,
                )
            )
            remaining_q[label] -= 1

        remaining_carry = {-2: 256, -1: 256, 0: 256, 1: 256}
        carry_rng = random.Random(derive_seed("rare_carry", width))
        carry_items: set[tuple[int, ...]] = set()
        while any(remaining_carry.values()):
            raw = _draw_source_transition(carry_rng, split.test)
            if raw in transition_sources or raw in rare_q_sources:
                continue
            trace = frozen_data.quotient_carry_trace(
                frozen_data.LegalTransition(*raw), width
            )
            for position, label in enumerate(trace.carry_out):
                identity = (*raw, position)
                if remaining_carry[label] <= 0 or identity in carry_items:
                    continue
                carry_items.add(identity)
                rare_carry.append(
                    _owned_transition(
                        raw,
                        width=width,
                        repeat=3,
                        arm="challenger",
                        label=label,
                        position=position,
                    )
                )
                remaining_carry[label] -= 1

    determinism_repeat = tuple(
        OwnedCase(**(_case_payload(row) | {"arm": "challenger"}))
        for width in (32, 64)
        for row in normal_by_repeat_width[(3, width)]
    )
    heldout = width256_heldout_identity_stream()
    profile_warmup: list[OwnedCase] = []
    profile_rng = random.Random(derive_seed("profile_warmup", 256))
    for modulus in heldout.warmup:
        raw = (profile_rng.randrange(modulus), profile_rng.randrange(modulus), modulus)
        if raw in rollout_sources:
            raise ContractError("canonical width-256 warmup stream overlaps")
        rollout_sources.add(raw)
        profile_warmup.append(
            _owned_rollout(raw, width=256, repeat=0, arm="challenger")
        )
    profile_fresh: list[OwnedCase] = []
    for repeat in (1, 2, 3):
        profile_rng = random.Random(derive_seed("profile_fresh", repeat, 256))
        for modulus in heldout.fresh_by_repeat[repeat]:
            raw = (
                profile_rng.randrange(modulus),
                profile_rng.randrange(modulus),
                modulus,
            )
            if raw in rollout_sources:
                raise ContractError("canonical width-256 fresh stream overlaps")
            rollout_sources.add(raw)
            profile_fresh.append(
                _owned_rollout(
                    raw, width=256, repeat=repeat, arm="challenger"
                )
            )
    return L1IdentityManifest(
        training=tuple(training),
        transitions=tuple(transitions),
        normal_rollouts=tuple(normal_rollouts),
        zero_quotient_rollouts=tuple(zero_quotient_rollouts),
        zero_carry_rollouts=tuple(zero_carry_rollouts),
        rare_quotient=tuple(rare_quotient),
        rare_carry=tuple(rare_carry),
        determinism_repeat=determinism_repeat,
        profile_warmup=tuple(profile_warmup),
        profile_fresh=tuple(profile_fresh),
    )


def _recomputed_rare_label(case: OwnedCase, *, kind: str) -> int:
    if kind not in {"quotient", "carry"}:
        raise ValueError("rare label kind must be quotient or carry")
    if any(
        value is None
        for value in (case.state, case.multiplicand, case.operand_bit)
    ):
        raise ContractError("rare case lacks raw transition fields")
    transition = frozen_data.LegalTransition(
        int(case.state),
        int(case.multiplicand),
        case.modulus,
        int(case.operand_bit),
    )
    trace = frozen_data.quotient_carry_trace(transition, case.width)
    raw_identity = (
        int(case.state),
        int(case.multiplicand),
        case.modulus,
        int(case.operand_bit),
    )
    if tuple(case.source_identity) != raw_identity:
        raise ContractError("rare source identity does not bind raw transition")
    if kind == "quotient":
        if case.position is not None or tuple(case.identity) != raw_identity:
            raise ContractError("rare quotient case must not carry a position")
        computed = trace.quotient
    else:
        if (
            type(case.position) is not int
            or case.position < 0
            or case.position >= case.width
        ):
            raise ContractError("rare carry position is outside the transition")
        if tuple(case.identity) != (*raw_identity, case.position):
            raise ContractError("rare carry identity is not the exact raw item")
        computed = trace.carry_out[case.position]
    if case.label != computed:
        raise ContractError(f"rare {kind} label does not match raw fields")
    return computed


def validate_l1_identity_manifest(
    manifest: L1IdentityManifest,
    *,
    deterministic_attempt_allowance: int,
) -> dict[str, Any]:
    """Validate and seed-regenerate exact complete-stream ownership."""

    splits = {width: frozen_identity_split(width) for width in (32, 64)}
    train_moduli = {width: set(split.train) for width, split in splits.items()}
    test_moduli = {width: set(split.test) for width, split in splits.items()}
    heldout_width256 = width256_heldout_identity_stream()
    heldout = set(heldout_width256.moduli)

    named = {
        "training": manifest.training,
        "transitions": manifest.transitions,
        "normal_rollouts": manifest.normal_rollouts,
        "zero_quotient_rollouts": manifest.zero_quotient_rollouts,
        "zero_carry_rollouts": manifest.zero_carry_rollouts,
        "rare_quotient": manifest.rare_quotient,
        "rare_carry": manifest.rare_carry,
        "determinism_repeat": manifest.determinism_repeat,
        "profile_warmup": manifest.profile_warmup,
        "profile_fresh": manifest.profile_fresh,
    }
    transition_streams = {"training", "transitions", "rare_quotient", "rare_carry"}
    rollout_streams = {
        "normal_rollouts",
        "zero_quotient_rollouts",
        "zero_carry_rollouts",
        "determinism_repeat",
        "profile_warmup",
        "profile_fresh",
    }
    for stream, rows in named.items():
        for row in rows:
            if not row.identity or not row.source_identity:
                raise ContractError(f"{stream} contains an empty identity")
            if stream in transition_streams:
                if any(
                    value is None
                    for value in (row.state, row.multiplicand, row.operand_bit)
                ) or row.multiplier is not None:
                    raise ContractError(f"{stream} lacks an exact raw transition")
                raw_transition = (
                    int(row.state),
                    int(row.multiplicand),
                    row.modulus,
                    int(row.operand_bit),
                )
                expected_identity = (
                    (*raw_transition, row.position)
                    if stream == "rare_carry"
                    else raw_transition
                )
                if (
                    tuple(row.source_identity) != raw_transition
                    or tuple(row.identity) != expected_identity
                ):
                    raise ContractError(
                        f"{stream} raw identity has a prohibited suffix or mutation"
                    )
                frozen_data.quotient_carry_trace(
                    frozen_data.LegalTransition(*raw_transition), row.width
                )
            elif stream in rollout_streams:
                if (
                    type(row.multiplier) is not int
                    or type(row.multiplicand) is not int
                    or row.state is not None
                    or row.operand_bit is not None
                    or row.position is not None
                    or row.label is not None
                ):
                    raise ContractError(f"{stream} lacks an exact raw rollout tuple")
                raw_rollout = (row.multiplier, row.multiplicand, row.modulus)
                if (
                    tuple(row.source_identity) != raw_rollout
                    or tuple(row.identity) != raw_rollout
                    or not 0 <= row.multiplier < row.modulus
                    or not 0 <= row.multiplicand < row.modulus
                ):
                    raise ContractError(
                        f"{stream} raw identity has a prohibited suffix or mutation"
                    )

    def exact_rows(
        rows: Sequence[OwnedCase],
        *,
        repeat: int,
        width: int,
        count: int,
        arm: str,
    ) -> tuple[OwnedCase, ...]:
        selected = tuple(
            row for row in rows if row.repeat == repeat and row.width == width
        )
        if len(selected) != count or any(row.arm != arm for row in selected):
            raise ContractError(
                f"identity count/owner mismatch for repeat {repeat} width {width}"
            )
        return selected

    generated_base: list[OwnedCase] = []
    for repeat in (0, 1, 2, 3):
        for width in (32, 64):
            training = exact_rows(
                manifest.training,
                repeat=repeat,
                width=width,
                count=1024,
                arm="shared",
            )
            transitions = exact_rows(
                manifest.transitions,
                repeat=repeat,
                width=width,
                count=512,
                arm="shared",
            )
            normal = exact_rows(
                manifest.normal_rollouts,
                repeat=repeat,
                width=width,
                count=32,
                arm="shared",
            )
            zero_q = exact_rows(
                manifest.zero_quotient_rollouts,
                repeat=repeat,
                width=width,
                count=32,
                arm="challenger",
            )
            zero_carry = exact_rows(
                manifest.zero_carry_rollouts,
                repeat=repeat,
                width=width,
                count=32,
                arm="challenger",
            )
            if tuple(row.source_identity for row in zero_q) != tuple(
                row.source_identity for row in normal
            ) or tuple(row.source_identity for row in zero_carry) != tuple(
                row.source_identity for row in normal
            ):
                raise ContractError("interventions are not paired to normal rollouts")
            if any(row.modulus not in train_moduli[width] for row in training):
                raise ContractError("training stream uses a non-train identity")
            if any(
                row.modulus not in test_moduli[width]
                for row in (*transitions, *normal, *zero_q, *zero_carry)
            ):
                raise ContractError("evaluation stream uses a non-test identity")
            generated_base.extend((*training, *transitions, *normal))

    if (
        len(manifest.training) != 4 * 2 * 1024
        or len(manifest.transitions) != 4 * 2 * 512
        or len(manifest.normal_rollouts) != 4 * 2 * 32
        or len(manifest.zero_quotient_rollouts) != 4 * 2 * 32
        or len(manifest.zero_carry_rollouts) != 4 * 2 * 32
    ):
        raise ContractError("complete repeat streams contain extra records")

    base_sources = [row.source_identity for row in generated_base]
    if len(base_sources) != len(set(base_sources)):
        raise ContractError("tuple identity repeats across the complete repeat streams")

    rare_sources: set[tuple[int, ...]] = set()
    for width in (32, 64):
        q_rows = exact_rows(
            manifest.rare_quotient,
            repeat=3,
            width=width,
            count=3 * 256,
            arm="challenger",
        )
        carry_rows = exact_rows(
            manifest.rare_carry,
            repeat=3,
            width=width,
            count=4 * 256,
            arm="challenger",
        )
        if any(row.modulus not in test_moduli[width] for row in (*q_rows, *carry_rows)):
            raise ContractError("rare diagnostic uses a non-test identity")
        q_labels = tuple(
            _recomputed_rare_label(row, kind="quotient") for row in q_rows
        )
        carry_labels = tuple(
            _recomputed_rare_label(row, kind="carry") for row in carry_rows
        )
        if {label: q_labels.count(label) for label in (0, 1, 2)} != {
            0: 256,
            1: 256,
            2: 256,
        }:
            raise ContractError("rare quotient class denominators are not exact")
        if {
            label: carry_labels.count(label)
            for label in (-2, -1, 0, 1)
        } != {-2: 256, -1: 256, 0: 256, 1: 256}:
            raise ContractError("rare carry class denominators are not exact")
        if len({row.identity for row in q_rows}) != len(q_rows):
            raise ContractError("rare quotient transitions are not unique")
        if len({row.identity for row in carry_rows}) != len(carry_rows):
            raise ContractError("rare carry transition-position items are not unique")
        carry_positions = [
            (row.source_identity, row.position) for row in carry_rows
        ]
        if len(carry_positions) != len(set(carry_positions)):
            raise ContractError("rare carry contains duplicate transition positions")
        q_sources = {row.source_identity for row in q_rows}
        carry_sources = {row.source_identity for row in carry_rows}
        if q_sources.intersection(carry_sources):
            raise ContractError("rare quotient/carry diagnostic sources overlap")
        if (q_sources | carry_sources).intersection(base_sources):
            raise ContractError("rare diagnostic overlaps a complete repeat stream")
        rare_sources.update(q_sources | carry_sources)

        deterministic = exact_rows(
            manifest.determinism_repeat,
            repeat=3,
            width=width,
            count=32,
            arm="challenger",
        )
        normal_repeat3 = exact_rows(
            manifest.normal_rollouts,
            repeat=3,
            width=width,
            count=32,
            arm="shared",
        )
        if tuple(row.source_identity for row in deterministic) != tuple(
            row.source_identity for row in normal_repeat3
        ):
            raise ContractError("determinism pass does not reuse repeat-3 normal cases")

    if (
        len(manifest.rare_quotient) != 2 * 3 * 256
        or len(manifest.rare_carry) != 2 * 4 * 256
        or len(manifest.determinism_repeat) != 2 * 32
    ):
        raise ContractError("repeat-3 diagnostics contain extra records")

    warmup = tuple(manifest.profile_warmup)
    fresh = tuple(manifest.profile_fresh)
    if (
        len(warmup) != 4
        or len(fresh) != 24
        or tuple(row.modulus for row in warmup) != heldout_width256.warmup
        or any(
            row.width != 256
            or row.repeat != 0
            or row.arm != "challenger"
            or row.modulus not in heldout
            for row in warmup
        )
    ):
        raise ContractError("width-256 profile warmup ownership is invalid")
    for repeat in (1, 2, 3):
        repeat_rows = exact_rows(
            fresh, repeat=repeat, width=256, count=8, arm="challenger"
        )
        if tuple(row.modulus for row in repeat_rows) != heldout_width256.fresh_by_repeat[
            repeat
        ]:
            raise ContractError("width-256 fresh profile stream order changed")
    if any(row.modulus not in heldout for row in fresh):
        raise ContractError("width-256 fresh profile uses a non-held-out identity")
    profile_sources = [row.source_identity for row in (*warmup, *fresh)]
    if len(profile_sources) != 28 or len(set(profile_sources)) != 28:
        raise ContractError("width-256 warmup/fresh profile cases overlap")

    canonical = canonical_l1_identity_manifest(deterministic_attempt_allowance)
    if manifest != canonical:
        raise ContractError(
            "identity manifest does not match canonical seed-to-case regeneration"
        )

    payload = {
        name: [_case_payload(row) for row in rows]
        for name, rows in named.items()
    }
    return {
        "split_sha256_by_width": {
            "32": splits[32].sha256,
            "64": splits[64].sha256,
        },
        "stream_manifest_sha256": sha256_bytes(canonical_json_bytes(payload)),
        "width256_stream_sha256": heldout_width256.sha256,
        "checked_or_private_answers_read": False,
        "repeat_zero_is_warmup_only": True,
        "rare_owner": "fresh-repeat-3-challenger",
        "profile_owner": "exact-reloaded-challenger",
    }


@dataclass(frozen=True, slots=True)
class ImmutableBatch:
    features: Tensor
    residue_targets: Tensor
    quotient_targets: Tensor
    carry_targets: Tensor
    sha256: str


def immutable_batch(
    features: Tensor,
    residue_targets: Tensor,
    quotient_targets: Tensor,
    carry_targets: Tensor,
) -> ImmutableBatch:
    tensors = tuple(
        tensor.detach().clone().contiguous()
        for tensor in (features, residue_targets, quotient_targets, carry_targets)
    )
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return ImmutableBatch(*tensors, sha256=digest.hexdigest())


@dataclass(frozen=True, slots=True)
class LossTerms:
    residue_mean: Tensor
    residue_worst_position: Tensor
    residue: Tensor
    quotient: Tensor
    carry: Tensor
    total: Tensor


def exact_loss_terms(
    output: frozen_model.TransducerOutput,
    batch: ImmutableBatch,
    *,
    quotient_weight: float,
    carry_weight: float,
) -> LossTerms:
    per_bit = F.binary_cross_entropy_with_logits(
        output.residue_logits, batch.residue_targets, reduction="none"
    )
    residue_mean = per_bit.mean()
    residue_worst = per_bit.mean(dim=0).max()
    residue = residue_mean + 0.45 * residue_worst
    quotient = F.cross_entropy(
        output.quotient_logits, batch.quotient_targets, reduction="mean"
    )
    carry = F.cross_entropy(
        output.signed_carry_logits,
        batch.carry_targets,
        reduction="mean",
    )
    total = residue + quotient_weight * quotient + carry_weight * carry
    return LossTerms(residue_mean, residue_worst, residue, quotient, carry, total)


def deterministic_cpu_preflight(
    model: nn.Module,
    *,
    amp_enabled: bool,
) -> dict[str, Any]:
    """Read and enforce the exact source-resolved CPU execution identity."""

    identity = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "precision": str(torch.get_default_dtype()).removeprefix("torch."),
        "amp_enabled": bool(amp_enabled),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
    }
    expected = {
        "torch_version": PINNED_TORCH_VERSION,
        "device": "cpu",
        "precision": "float32",
        "amp_enabled": False,
        "deterministic_algorithms": True,
        "intra_op_threads": 4,
        "inter_op_threads": 1,
    }
    if identity != expected:
        raise ContractError("deterministic CPU runtime identity mismatch")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ContractError("model is not entirely CPU FP32")
    return identity


def optimizer_and_scheduler(
    model: nn.Module,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.CosineAnnealingLR]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-6,
        betas=ADAMW_DEFAULTS["betas"],
        eps=ADAMW_DEFAULTS["eps"],
        weight_decay=2e-5,
        amsgrad=ADAMW_DEFAULTS["amsgrad"],
        maximize=ADAMW_DEFAULTS["maximize"],
        foreach=ADAMW_DEFAULTS["foreach"],
        capturable=ADAMW_DEFAULTS["capturable"],
        differentiable=ADAMW_DEFAULTS["differentiable"],
        fused=ADAMW_DEFAULTS["fused"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=400, eta_min=5e-7
    )
    return optimizer, scheduler


def validate_optimizer_contract(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 1:
        raise ContractError("optimizer must be one independent AdamW group")
    group = optimizer.param_groups[0]
    if (
        optimizer.defaults.get("lr") != 5e-6
        or group.get("initial_lr") != 5e-6
        or tuple(float(value) for value in scheduler.base_lrs) != (5e-6,)
    ):
        raise ContractError("AdamW initial/base learning rate changed")
    expected_current_lr = 5e-7 + 0.5 * (5e-6 - 5e-7) * (
        1.0 + math.cos(math.pi * scheduler.last_epoch / 400)
    )
    observed_current_lr = float(group["lr"])
    if (
        not math.isclose(observed_current_lr, expected_current_lr, rel_tol=0.0, abs_tol=1e-18)
        or tuple(float(value) for value in scheduler.get_last_lr())
        != (observed_current_lr,)
        or tuple(float(value) for value in scheduler._last_lr)
        != (observed_current_lr,)
    ):
        raise ContractError("AdamW/cosine current learning rate changed")
    expected = {
        "lr": observed_current_lr,
        "betas": ADAMW_DEFAULTS["betas"],
        "eps": ADAMW_DEFAULTS["eps"],
        "weight_decay": 2e-5,
        "amsgrad": ADAMW_DEFAULTS["amsgrad"],
        "maximize": ADAMW_DEFAULTS["maximize"],
        "foreach": ADAMW_DEFAULTS["foreach"],
        "capturable": ADAMW_DEFAULTS["capturable"],
        "differentiable": ADAMW_DEFAULTS["differentiable"],
        "fused": ADAMW_DEFAULTS["fused"],
    }
    if any(group.get(key) != value for key, value in expected.items()):
        raise ContractError("AdamW source-resolved defaults changed")
    model_parameters = list(model.parameters())
    if len(group["params"]) != len(model_parameters) or any(
        left is not right
        for left, right in zip(group["params"], model_parameters, strict=True)
    ):
        raise ContractError("AdamW group does not contain every parameter in order")
    if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        raise ContractError("scheduler is not CosineAnnealingLR")
    if scheduler.T_max != 400 or scheduler.eta_min != 5e-7:
        raise ContractError("cosine scheduler contract changed")


def optimizer_name_topology(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[tuple[str, ...], ...]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return tuple(
        tuple(names[id(parameter)] for parameter in group["params"])
        for group in optimizer.param_groups
    )


def independently_initialized_pair(
    factory: Callable[[], nn.Module], *, repeat: int
) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(derive_seed("init", repeat))
    control = factory()
    challenger = factory()
    challenger.load_state_dict(control.state_dict(), strict=True)
    for left, right in zip(control.parameters(), challenger.parameters()):
        if left.data_ptr() == right.data_ptr() or not torch.equal(left, right):
            raise ContractError("matched initialization is not exact and independent")
    return control, challenger


def frozen_matched_pair(
    config: dict[str, Any], *, repeat: int
) -> tuple[frozen_model.JointQuotientCarryTransducer, frozen_model.JointQuotientCarryTransducer]:
    local = dict(config)
    local["seed"] = derive_seed("init", repeat)
    return frozen_run.build_matched_arms(local)


@dataclass(frozen=True, slots=True)
class PairedUpdateAudit:
    arm_order: tuple[str, str]
    event_order: tuple[str, ...]
    batch_unchanged: bool
    initialization_independent: bool
    structure_equal: bool
    finite: bool
    finite_gradients: bool
    auxiliary_connected_difference: bool
    rng_after_forward_equal: bool
    forward_results_independent: bool
    entry_value_equality_required: bool
    learning_rates: Mapping[str, float]
    optimizer_step_counts: Mapping[str, tuple[int, ...]]
    scheduler_last_epochs: Mapping[str, int]


def _structural_signature(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            name,
            tuple(parameter.shape),
            str(parameter.dtype),
            str(parameter.device),
            bool(parameter.requires_grad),
        )
        for name, parameter in model.named_parameters()
    )


def _optimizer_step_counts(
    optimizer: torch.optim.Optimizer,
) -> tuple[int, ...]:
    counts = {
        int(state["step"].item())
        for state in optimizer.state.values()
        if "step" in state
    }
    return tuple(sorted(counts))


_AUXILIARY_PARAMETER_PREFIXES = (
    "quotient_head",
    "signed_carry_head",
    "quotient_message_embedding",
    "signed_carry_message_embedding",
    "message_gate_weight",
    "message_candidate_weight",
)


def _auxiliary_connected_difference(control: nn.Module, challenger: nn.Module) -> bool:
    right = challenger.state_dict()
    return any(
        name.startswith(_AUXILIARY_PARAMETER_PREFIXES)
        and not torch.equal(value, right[name])
        for name, value in control.state_dict().items()
    )


def _finite_parameters_and_gradients(models: Sequence[nn.Module]) -> tuple[bool, bool]:
    finite_parameters = all(
        bool(torch.isfinite(parameter).all())
        for model in models
        for parameter in model.parameters()
    )
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for model in models
        for parameter in model.parameters()
    )
    return finite_parameters, finite_gradients


def _independent_optimizer_state(
    control: torch.optim.Optimizer,
    challenger: torch.optim.Optimizer,
) -> bool:
    if control is challenger:
        return False
    left_tensors = [
        value
        for state in control.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    ]
    right_pointers = {
        value.data_ptr()
        for state in challenger.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    }
    return all(value.data_ptr() not in right_pointers for value in left_tensors)


def paired_update(
    control: nn.Module,
    challenger: nn.Module,
    control_optimizer: torch.optim.Optimizer,
    challenger_optimizer: torch.optim.Optimizer,
    control_scheduler: torch.optim.lr_scheduler.LRScheduler,
    challenger_scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch: ImmutableBatch,
    *,
    repeat: int,
    global_step: int,
) -> PairedUpdateAudit:
    """Execute exactly one tiny matched update for unit verification only."""

    if global_step < 1:
        raise ValueError("global_step is one-indexed")
    if control_optimizer is challenger_optimizer or control_scheduler is challenger_scheduler:
        raise ContractError("paired arms do not own independent optimizer state")
    if tuple(control.state_dict()) != tuple(challenger.state_dict()):
        raise ContractError("paired state-dictionary keys differ")
    values_equal = all(
        torch.equal(value, challenger.state_dict()[name])
        for name, value in control.state_dict().items()
    )
    independent = all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(control.parameters(), challenger.parameters(), strict=True)
    )
    structure_equal = _structural_signature(control) == _structural_signature(challenger)
    topology_equal = optimizer_name_topology(
        control, control_optimizer
    ) == optimizer_name_topology(challenger, challenger_optimizer)
    validate_optimizer_contract(control, control_optimizer, control_scheduler)
    validate_optimizer_contract(challenger, challenger_optimizer, challenger_scheduler)
    pre_steps = {
        "control": _optimizer_step_counts(control_optimizer),
        "challenger": _optimizer_step_counts(challenger_optimizer),
    }
    expected_pre_steps = () if global_step == 1 else (global_step - 1,)
    if pre_steps != {
        "control": expected_pre_steps,
        "challenger": expected_pre_steps,
    }:
        raise ContractError("optimizer step counts do not match global_step")
    if (
        control_scheduler.last_epoch != global_step - 1
        or challenger_scheduler.last_epoch != global_step - 1
    ):
        raise ContractError("scheduler epoch does not match global_step")
    if control_optimizer.param_groups[0]["lr"] != challenger_optimizer.param_groups[0]["lr"]:
        raise ContractError("paired learning rates differ")
    entry_auxiliary_difference = _auxiliary_connected_difference(control, challenger)
    if global_step == 1 and not values_equal:
        raise ContractError("step 1 did not start from exact matched values")
    if global_step > 1 and (values_equal or not entry_auxiliary_difference):
        raise ContractError("post-step-1 auxiliary divergence was not preserved")
    pre_finite, pre_gradients_finite = _finite_parameters_and_gradients(
        (control, challenger)
    )
    if not all(
        (
            independent,
            structure_equal,
            topology_equal,
            _independent_optimizer_state(control_optimizer, challenger_optimizer),
            pre_finite,
            pre_gradients_finite,
        )
    ):
        raise ContractError("paired geometry, storage, topology, or finiteness failed")
    before = batch.sha256
    order = (
        ("control", "challenger")
        if (repeat + global_step) % 2 == 0
        else ("challenger", "control")
    )
    models = {"control": control, "challenger": challenger}
    optimizers = {"control": control_optimizer, "challenger": challenger_optimizer}
    schedulers = {"control": control_scheduler, "challenger": challenger_scheduler}
    weights = {"control": (0.0, 0.0), "challenger": (0.10, 0.10)}
    rng_before = torch.get_rng_state().clone()
    rng_after: Tensor | None = None
    rng_after_forward: dict[str, Tensor] = {}
    forward_results: dict[str, frozen_model.TransducerOutput] = {}
    events: list[str] = []
    for arm in order:
        torch.set_rng_state(rng_before.clone())
        optimizer = optimizers[arm]
        optimizer.zero_grad(set_to_none=True)
        events.append(f"{arm}:zero-grad")
        output = models[arm](batch.features)
        forward_results[arm] = output
        rng_after_forward[arm] = torch.get_rng_state().clone()
        events.append(f"{arm}:forward")
        terms = exact_loss_terms(
            output,
            batch,
            quotient_weight=weights[arm][0],
            carry_weight=weights[arm][1],
        )
        events.append(f"{arm}:all-three-losses")
        terms.total.backward()
        events.append(f"{arm}:backward")
        if not _finite_parameters_and_gradients((models[arm],))[1]:
            raise ContractError("non-finite gradient before clipping")
        total_norm = nn.utils.clip_grad_norm_(models[arm].parameters(), max_norm=2.0)
        if not bool(torch.isfinite(total_norm)):
            raise ContractError("non-finite global gradient norm")
        events.append(f"{arm}:clip")
        optimizer.step()
        events.append(f"{arm}:optimizer-step")
        schedulers[arm].step()
        events.append(f"{arm}:scheduler-step")
        if rng_after is None:
            rng_after = torch.get_rng_state().clone()
    if rng_after is not None:
        torch.set_rng_state(rng_after)
    after_batch = immutable_batch(
        batch.features,
        batch.residue_targets,
        batch.quotient_targets,
        batch.carry_targets,
    ).sha256
    structure_equal = _structural_signature(control) == _structural_signature(challenger)
    finite, finite_gradients = _finite_parameters_and_gradients((control, challenger))
    auxiliary_difference = _auxiliary_connected_difference(control, challenger)
    result_independence = all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(
            forward_results["control"],
            forward_results["challenger"],
            strict=True,
        )
    )
    post_steps = {
        "control": _optimizer_step_counts(control_optimizer),
        "challenger": _optimizer_step_counts(challenger_optimizer),
    }
    expected_post_steps = (global_step,)
    scheduler_epochs = {
        "control": control_scheduler.last_epoch,
        "challenger": challenger_scheduler.last_epoch,
    }
    if post_steps != {
        "control": expected_post_steps,
        "challenger": expected_post_steps,
    } or scheduler_epochs != {
        "control": global_step,
        "challenger": global_step,
    }:
        raise ContractError("post-update optimizer/scheduler counts diverged")
    if not auxiliary_difference:
        raise ContractError("auxiliary-connected divergence did not persist")
    return PairedUpdateAudit(
        arm_order=order,
        event_order=tuple(events),
        batch_unchanged=before == after_batch,
        initialization_independent=independent,
        structure_equal=structure_equal
        and optimizer_name_topology(control, control_optimizer)
        == optimizer_name_topology(challenger, challenger_optimizer)
        and _independent_optimizer_state(control_optimizer, challenger_optimizer),
        finite=finite,
        finite_gradients=finite_gradients,
        auxiliary_connected_difference=auxiliary_difference,
        rng_after_forward_equal=torch.equal(
            rng_after_forward["control"], rng_after_forward["challenger"]
        ),
        forward_results_independent=result_independence,
        entry_value_equality_required=global_step == 1,
        learning_rates={
            "control": float(control_optimizer.param_groups[0]["lr"]),
            "challenger": float(challenger_optimizer.param_groups[0]["lr"]),
        },
        optimizer_step_counts=post_steps,
        scheduler_last_epochs=scheduler_epochs,
    )


def rollout_bit_positions(width: int) -> tuple[int, ...]:
    if width < 1:
        raise ValueError("rollout width must be positive")
    return tuple(range(width - 1, -1, -1))


def decode_little_endian_for_next_feature(
    predicted_bits: Sequence[bool], modulus: int
) -> tuple[int, bool]:
    """Offline evaluation decode; retain, flag, and never repair invalid state."""

    value = sum(int(bit) << position for position, bit in enumerate(predicted_bits))
    return value, value >= modulus


@dataclass(frozen=True, slots=True)
class CheckpointIntegrityResult:
    reloaded_model: frozen_model.JointQuotientCarryTransducer
    repeat: int
    arm: str
    ordinal: int
    receipt: Mapping[str, Any]


def checkpoint_roundtrip(
    model: frozen_model.JointQuotientCarryTransducer,
    path: Path,
    probe: Tensor,
    *,
    repeat: int,
    arm: str,
    ordinal: int,
) -> CheckpointIntegrityResult:
    """One exact save/hash/reload/fixed-probe cycle in a caller-owned temp path."""

    if repeat != 3 or (arm, ordinal) not in {("control", 1), ("challenger", 2)}:
        raise ContractError("checkpoint is not bound to repeat 3 and arm ordinal")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, str(temporary))
    os.replace(temporary, path)
    reloaded = frozen_model.JointQuotientCarryTransducer(
        hidden=model.hidden, cycles=model.cycles
    )
    reloaded.load_state_dict(load_file(str(path)), strict=True)
    tensors_equal = all(
        torch.equal(value, reloaded.state_dict()[name])
        for name, value in model.state_dict().items()
    )
    model.eval()
    reloaded.eval()
    with torch.inference_mode():
        left = model(probe)
        right = reloaded(probe)
    probe_equal = all(torch.equal(a, b) for a, b in zip(left, right))
    if not tensors_equal or not probe_equal:
        raise ContractError("checkpoint reload equivalence failed")
    return CheckpointIntegrityResult(
        reloaded_model=reloaded,
        repeat=repeat,
        arm=arm,
        ordinal=ordinal,
        receipt={
            "path_name": path.name,
            "sha256": frozen_run.sha256(path),
            "bytes": path.stat().st_size,
            "tensor_equality": True,
            "fixed_probe_equality": True,
            "scientific_checkpoint": False,
            "repeat": repeat,
            "arm": arm,
            "ordinal": ordinal,
        },
    )


def exact_ordinal_profile_challenger(
    control: CheckpointIntegrityResult,
    challenger: CheckpointIntegrityResult,
) -> frozen_model.JointQuotientCarryTransducer:
    """Return the already reloaded challenger; never reload or select by result."""

    if control.reloaded_model is challenger.reloaded_model:
        raise ContractError("control/challenger checkpoint handles alias")
    if (control.repeat, control.arm, control.ordinal) != (3, "control", 1) or (
        challenger.repeat,
        challenger.arm,
        challenger.ordinal,
    ) != (3, "challenger", 2):
        raise ContractError("profile checkpoint ordinal binding changed")
    return challenger.reloaded_model


def l1_physical_counters() -> dict[str, Any]:
    core = {
        "shared_training_batches_generated": 16,
        "shared_training_rows_generated": 2048,
        "training_updates": {"control": 16, "challenger": 16},
        "shared_transition_cases_generated": {32: 512, 64: 512},
        "transition_forward_cases": {
            "control": {32: 512, 64: 512},
            "challenger": {32: 512, 64: 512},
        },
        "shared_normal_rollout_cases_generated": {32: 32, 64: 32},
        "normal_rollout_cases": {
            "control": {32: 32, 64: 32},
            "challenger": {32: 32, 64: 32},
        },
        "intervention_baseline_cases_reused": {32: 32, 64: 32},
        "extra_zero_quotient_rollout_cases": {32: 32, 64: 32},
        "extra_zero_carry_rollout_cases": {32: 32, 64: 32},
    }
    return {
        "warmup_repeat_0": {**core, "rate_or_evidence_denominator": False},
        "fresh_repeats": {
            repeat: {**core, "rate_or_evidence_denominator": True}
            for repeat in (1, 2, 3)
        },
        "post_repeat_3": {
            "rare_owner": "challenger",
            "q_shared_generated_per_width": 3 * 256,
            "q_forward_cases_per_width_per_arm": {
                "control": 3 * 256,
                "challenger": 3 * 256,
            },
            "carry_shared_generated_per_width": 4 * 256,
            "carry_forward_items_per_width_per_arm": {
                "control": 4 * 256,
                "challenger": 4 * 256,
            },
            "q_carry_datasets_disjoint": True,
            "determinism_pass1_reused_per_width": 32,
            "determinism_extra_cases_per_width": 32,
            "checkpoint_cycles": {"control": 1, "challenger": 1},
            "selected_checkpoint": "reloaded-challenger-by-ordinal",
            "profile_warmup_cases": 4,
            "profile_fresh_batches": 3,
            "profile_cases_per_fresh_batch": 8,
        },
    }


@dataclass(frozen=True, slots=True)
class PhysicalUnit:
    key: str
    rate_key: str
    phase: str
    repeat: int
    source: str
    units: int
    width: int | None = None
    arm: str = "shared"
    eligible_for_rate: bool = True


REGISTERED_RATE_KEYS = frozenset(
    {
        "training_shared_generation_batch",
        "training_arm_update",
        "transition_shared_generation_case",
        "transition_arm_forward_case",
        "rollout_shared_generation_case",
        "rollout_arm_forward_case",
        "intervention_extra_rollout_case",
        "q_shared_generation_case",
        "q_arm_forward_case",
        "carry_shared_generation_item",
        "carry_arm_forward_item",
        "determinism_extra_rollout_case",
        "checkpoint_cycle",
        "width256_profile_case",
        "incumbent_full_1100_case",
        "fixed_setup",
        "fixed_split",
        "fixed_model",
        "fixed_optimizer_control",
        "fixed_optimizer_challenger",
        "fixed_tracker",
    }
)
REGISTERED_PHASES = frozenset(
    {
        "L1-fixed",
        "L1-warmup",
        "L1-fresh",
        "L1-post-repeat3",
        "L1-profile-warmup",
        "L1-profile-fresh",
        "L1-incumbent-rerun",
        "L2-projected",
        "L2-projected-fixed",
    }
)


def _physical_unit(
    key: str,
    rate_key: str,
    phase: str,
    repeat: int,
    source: str,
    units: int,
    *,
    width: int | None = None,
    arm: str = "shared",
    eligible_for_rate: bool = True,
) -> PhysicalUnit:
    return PhysicalUnit(
        key=key,
        rate_key=rate_key,
        phase=phase,
        repeat=repeat,
        source=source,
        units=units,
        width=width,
        arm=arm,
        eligible_for_rate=eligible_for_rate,
    )


def _validate_physical_manifest(manifest: Sequence[PhysicalUnit]) -> None:
    keys = [item.key for item in manifest]
    if len(keys) != len(set(keys)) or not keys:
        raise ContractError("physical manifest keys are empty or overlap")
    for item in manifest:
        if item.units <= 0 or item.repeat < 0:
            raise ContractError("physical manifest contains invalid units/repeat")
        if item.source not in {
            "shared-generator",
            "control-model",
            "challenger-model",
            "checkpoint-io",
            "candidate005-sandbox",
            "scaffold-runtime",
        }:
            raise ContractError("physical manifest source is not registered")
        if (
            item.rate_key not in REGISTERED_RATE_KEYS
            or item.phase not in REGISTERED_PHASES
            or item.arm not in {"shared", "control", "challenger"}
            or not item.key.startswith(("l1-", "l2-"))
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in item.key
            )
        ):
            raise ContractError("physical manifest identifier is not registered")
    forbidden_reuse = (
        "intervention_baseline",
        "determinism_pass1",
        "shared_model_execution",
    )
    if any(token in item.key for token in forbidden_reuse for item in manifest):
        raise ContractError("reused or shared model work was double-counted")


def warmup_l1_physical_manifest() -> tuple[PhysicalUnit, ...]:
    units: list[PhysicalUnit] = []
    for component, source, arm in (
        ("setup", "scaffold-runtime", "shared"),
        ("split", "shared-generator", "shared"),
        ("model", "scaffold-runtime", "shared"),
        ("optimizer_control", "scaffold-runtime", "control"),
        ("optimizer_challenger", "scaffold-runtime", "challenger"),
        ("tracker", "scaffold-runtime", "shared"),
    ):
        units.append(
            _physical_unit(
                f"l1-fixed-{component.replace('_', '-')}",
                f"fixed_{component}",
                "L1-fixed",
                0,
                source,
                1,
                arm=arm,
            )
        )
    for width in (32, 64):
        prefix = f"l1-warmup-r0-w{width}"
        units.extend(
            (
                _physical_unit(prefix + "-train-generate", "training_shared_generation_batch", "L1-warmup", 0, "shared-generator", 8, width=width),
                _physical_unit(prefix + "-control-update", "training_arm_update", "L1-warmup", 0, "control-model", 8, width=width, arm="control"),
                _physical_unit(prefix + "-challenger-update", "training_arm_update", "L1-warmup", 0, "challenger-model", 8, width=width, arm="challenger"),
                _physical_unit(prefix + "-transition-generate", "transition_shared_generation_case", "L1-warmup", 0, "shared-generator", 512, width=width),
                _physical_unit(prefix + "-control-transition", "transition_arm_forward_case", "L1-warmup", 0, "control-model", 512, width=width, arm="control"),
                _physical_unit(prefix + "-challenger-transition", "transition_arm_forward_case", "L1-warmup", 0, "challenger-model", 512, width=width, arm="challenger"),
                _physical_unit(prefix + "-rollout-generate", "rollout_shared_generation_case", "L1-warmup", 0, "shared-generator", 32, width=width),
                _physical_unit(prefix + "-control-rollout", "rollout_arm_forward_case", "L1-warmup", 0, "control-model", 32, width=width, arm="control"),
                _physical_unit(prefix + "-challenger-rollout", "rollout_arm_forward_case", "L1-warmup", 0, "challenger-model", 32, width=width, arm="challenger"),
                _physical_unit(prefix + "-zero-q", "intervention_extra_rollout_case", "L1-warmup", 0, "challenger-model", 32, width=width, arm="challenger"),
                _physical_unit(prefix + "-zero-carry", "intervention_extra_rollout_case", "L1-warmup", 0, "challenger-model", 32, width=width, arm="challenger"),
            )
        )
    result = tuple(units)
    _validate_physical_manifest(result)
    return result


def remaining_l1_physical_manifest(cache_verdict: "CacheVerdict") -> tuple[PhysicalUnit, ...]:
    if not isinstance(cache_verdict, CacheVerdict) or not cache_verdict.evidence_validated:
        raise ContractError("remaining-L1 manifest requires a validated cache verdict")
    units: list[PhysicalUnit] = []
    for repeat in (1, 2, 3):
        for width in (32, 64):
            prefix = f"l1-r{repeat}-w{width}"
            units.extend(
                (
                    _physical_unit(prefix + "-train-generate", "training_shared_generation_batch", "L1-fresh", repeat, "shared-generator", 8, width=width),
                    _physical_unit(prefix + "-control-update", "training_arm_update", "L1-fresh", repeat, "control-model", 8, width=width, arm="control"),
                    _physical_unit(prefix + "-challenger-update", "training_arm_update", "L1-fresh", repeat, "challenger-model", 8, width=width, arm="challenger"),
                    _physical_unit(prefix + "-transition-generate", "transition_shared_generation_case", "L1-fresh", repeat, "shared-generator", 512, width=width),
                    _physical_unit(prefix + "-control-transition", "transition_arm_forward_case", "L1-fresh", repeat, "control-model", 512, width=width, arm="control"),
                    _physical_unit(prefix + "-challenger-transition", "transition_arm_forward_case", "L1-fresh", repeat, "challenger-model", 512, width=width, arm="challenger"),
                    _physical_unit(prefix + "-rollout-generate", "rollout_shared_generation_case", "L1-fresh", repeat, "shared-generator", 32, width=width),
                    _physical_unit(prefix + "-control-rollout", "rollout_arm_forward_case", "L1-fresh", repeat, "control-model", 32, width=width, arm="control"),
                    _physical_unit(prefix + "-challenger-rollout", "rollout_arm_forward_case", "L1-fresh", repeat, "challenger-model", 32, width=width, arm="challenger"),
                    _physical_unit(prefix + "-zero-q", "intervention_extra_rollout_case", "L1-fresh", repeat, "challenger-model", 32, width=width, arm="challenger"),
                    _physical_unit(prefix + "-zero-carry", "intervention_extra_rollout_case", "L1-fresh", repeat, "challenger-model", 32, width=width, arm="challenger"),
                )
            )
    for width in (32, 64):
        prefix = f"l1-post-r3-w{width}"
        units.extend(
            (
                _physical_unit(prefix + "-q-generate", "q_shared_generation_case", "L1-post-repeat3", 3, "shared-generator", 768, width=width),
                _physical_unit(prefix + "-control-q-forward", "q_arm_forward_case", "L1-post-repeat3", 3, "control-model", 768, width=width, arm="control"),
                _physical_unit(prefix + "-q-forward", "q_arm_forward_case", "L1-post-repeat3", 3, "challenger-model", 768, width=width, arm="challenger"),
                _physical_unit(prefix + "-carry-generate", "carry_shared_generation_item", "L1-post-repeat3", 3, "shared-generator", 1024, width=width),
                _physical_unit(prefix + "-control-carry-forward", "carry_arm_forward_item", "L1-post-repeat3", 3, "control-model", 1024, width=width, arm="control"),
                _physical_unit(prefix + "-carry-forward", "carry_arm_forward_item", "L1-post-repeat3", 3, "challenger-model", 1024, width=width, arm="challenger"),
                _physical_unit(prefix + "-determinism-extra", "determinism_extra_rollout_case", "L1-post-repeat3", 3, "challenger-model", 32, width=width, arm="challenger"),
            )
        )
    units.extend(
        (
            _physical_unit("l1-checkpoint-control", "checkpoint_cycle", "L1-post-repeat3", 3, "checkpoint-io", 1, arm="control"),
            _physical_unit("l1-checkpoint-challenger", "checkpoint_cycle", "L1-post-repeat3", 3, "checkpoint-io", 1, arm="challenger"),
            _physical_unit("l1-profile-warmup", "width256_profile_case", "L1-profile-warmup", 0, "challenger-model", 4, width=256, arm="challenger", eligible_for_rate=False),
        )
    )
    for repeat in (1, 2, 3):
        units.append(
            _physical_unit(
                f"l1-profile-fresh-r{repeat}",
                "width256_profile_case",
                "L1-profile-fresh",
                repeat,
                "challenger-model",
                8,
                width=256,
                arm="challenger",
            )
        )
    if cache_verdict.full_1100_required:
        units.append(
            _physical_unit(
                "l1-candidate005-full1100",
                "incumbent_full_1100_case",
                "L1-incumbent-rerun",
                0,
                "candidate005-sandbox",
                1100,
            )
        )
    result = tuple(units)
    _validate_physical_manifest(result)
    return result


def l2_physical_manifest(cache_verdict: "CacheVerdict") -> tuple[PhysicalUnit, ...]:
    if not isinstance(cache_verdict, CacheVerdict) or not cache_verdict.evidence_validated:
        raise ContractError("L2 manifest requires a validated cache verdict")
    units: list[PhysicalUnit] = []
    for width in (32, 64):
        prefix = f"l2-w{width}"
        units.extend(
            (
                _physical_unit(prefix + "-train-generate", "training_shared_generation_batch", "L2-projected", 0, "shared-generator", 200, width=width),
                _physical_unit(prefix + "-control-update", "training_arm_update", "L2-projected", 0, "control-model", 200, width=width, arm="control"),
                _physical_unit(prefix + "-challenger-update", "training_arm_update", "L2-projected", 0, "challenger-model", 200, width=width, arm="challenger"),
                _physical_unit(prefix + "-transition-generate", "transition_shared_generation_case", "L2-projected", 0, "shared-generator", 20000, width=width),
                _physical_unit(prefix + "-control-transition", "transition_arm_forward_case", "L2-projected", 0, "control-model", 20000, width=width, arm="control"),
                _physical_unit(prefix + "-challenger-transition", "transition_arm_forward_case", "L2-projected", 0, "challenger-model", 20000, width=width, arm="challenger"),
                _physical_unit(prefix + "-rollout-generate", "rollout_shared_generation_case", "L2-projected", 0, "shared-generator", 1000, width=width),
                _physical_unit(prefix + "-control-rollout", "rollout_arm_forward_case", "L2-projected", 0, "control-model", 1000, width=width, arm="control"),
                _physical_unit(prefix + "-challenger-rollout", "rollout_arm_forward_case", "L2-projected", 0, "challenger-model", 1000, width=width, arm="challenger"),
                _physical_unit(prefix + "-q-generate", "q_shared_generation_case", "L2-projected", 0, "shared-generator", 6000, width=width),
                _physical_unit(prefix + "-control-q", "q_arm_forward_case", "L2-projected", 0, "control-model", 6000, width=width, arm="control"),
                _physical_unit(prefix + "-challenger-q", "q_arm_forward_case", "L2-projected", 0, "challenger-model", 6000, width=width, arm="challenger"),
                _physical_unit(prefix + "-carry-generate", "carry_shared_generation_item", "L2-projected", 0, "shared-generator", 8000, width=width),
                _physical_unit(prefix + "-control-carry", "carry_arm_forward_item", "L2-projected", 0, "control-model", 8000, width=width, arm="control"),
                _physical_unit(prefix + "-challenger-carry", "carry_arm_forward_item", "L2-projected", 0, "challenger-model", 8000, width=width, arm="challenger"),
                _physical_unit(prefix + "-zero-q", "intervention_extra_rollout_case", "L2-projected", 0, "challenger-model", 500, width=width, arm="challenger"),
                _physical_unit(prefix + "-zero-carry", "intervention_extra_rollout_case", "L2-projected", 0, "challenger-model", 500, width=width, arm="challenger"),
                _physical_unit(prefix + "-determinism-extra", "determinism_extra_rollout_case", "L2-projected", 0, "challenger-model", 100, width=width, arm="challenger"),
            )
        )
    for component, source, arm in (
        ("checkpoint-control", "checkpoint-io", "control"),
        ("checkpoint-challenger", "checkpoint-io", "challenger"),
        ("setup", "scaffold-runtime", "shared"),
        ("split", "shared-generator", "shared"),
        ("model", "scaffold-runtime", "shared"),
        ("optimizer-control", "scaffold-runtime", "control"),
        ("optimizer-challenger", "scaffold-runtime", "challenger"),
        ("tracker", "scaffold-runtime", "shared"),
    ):
        rate_key = (
            "checkpoint_cycle"
            if component.startswith("checkpoint")
            else "fixed_" + component.replace("-", "_")
        )
        units.append(
            _physical_unit(
                "l2-fixed-" + component,
                rate_key,
                "L2-projected-fixed",
                0,
                source,
                1,
                arm=arm,
            )
        )
    units.append(
        _physical_unit(
            "l2-width256-profile",
            "width256_profile_case",
            "L2-projected-fixed",
            0,
            "challenger-model",
            28,
            width=256,
            arm="challenger",
        )
    )
    if cache_verdict.full_1100_required:
        units.append(
            _physical_unit(
                "l2-candidate005-full1100",
                "incumbent_full_1100_case",
                "L2-projected-fixed",
                0,
                "candidate005-sandbox",
                1100,
            )
        )
    result = tuple(units)
    _validate_physical_manifest(result)
    return result


def l2_projection_units(cache_verdict: "CacheVerdict") -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in l2_physical_manifest(cache_verdict):
        totals[item.rate_key] = totals.get(item.rate_key, 0) + item.units
    return totals


@dataclass(frozen=True, slots=True, order=True)
class CoverageClass:
    rate_key: str
    width: int | None
    arm: str
    source: str


# No width or arm collapse is preregistered. Physical units with the same
# literal width/arm/source naturally share a class; every other class is exact.
EXPLICIT_COVERAGE_EQUIVALENCES: Mapping[str, tuple[CoverageClass, ...]] = (
    MappingProxyType({})
)


def physical_coverage_class(physical: PhysicalUnit) -> CoverageClass:
    _validate_physical_manifest((physical,))
    return CoverageClass(
        rate_key=physical.rate_key,
        width=physical.width,
        arm=physical.arm,
        source=physical.source,
    )


def coverage_class_id(value: CoverageClass) -> str:
    width = "none" if value.width is None else str(value.width)
    return (
        f"{value.rate_key}|width={width}|arm={value.arm}|source={value.source}"
    )


def canonical_unresolved_measurement_manifest(
    cache_verdict: "CacheVerdict",
) -> tuple[PhysicalUnit, ...]:
    """Return exact remaining units whose classes have no eligible warmup rate."""

    warmup_classes = {
        physical_coverage_class(item)
        for item in warmup_l1_physical_manifest()
        if item.eligible_for_rate
    }
    result = tuple(
        item
        for item in remaining_l1_physical_manifest(cache_verdict)
        if item.eligible_for_rate
        and physical_coverage_class(item) not in warmup_classes
    )
    _validate_physical_manifest(result)
    return result


def l2_batch_counters() -> dict[str, Any]:
    return {
        "training": {"per_arm": 400, "per_width_per_arm": 200},
        "transition": {
            "per_width_per_arm": 20_000,
            "batch_size": 128,
            "batches": 157,
            "final_partial": 32,
        },
        "rollout": {
            "per_width_per_arm": 1_000,
            "batch_size": 32,
            "batches": 32,
            "final_partial": 8,
        },
        "q_per_class": {
            "cases": 2_000,
            "classes": 3,
            "batch_size": 256,
            "batches_per_class": 8,
            "final_partial": 208,
        },
        "carry_per_class": {
            "items": 2_000,
            "classes": 4,
            "batch_size": 256,
            "batches_per_class": 8,
            "final_partial": 208,
        },
        "intervention_per_mode_width": {
            "baseline_reused": 500,
            "extra": 500,
        },
        "determinism_per_width": {"pass1_reused": 100, "extra": 100},
    }


@dataclass(frozen=True, slots=True)
class MeterSnapshot:
    monotonic_wall_seconds: float
    process_user_plus_system_aggregate_cpu_seconds: float
    simultaneous_tree_rss_bytes: int


@dataclass(frozen=True, slots=True)
class MeterProvenance:
    wall_meter: str
    aggregate_cpu_meter: str
    peak_rss_meter: str
    meter_source_sha256: str


@dataclass(frozen=True, slots=True)
class ResourceObservation:
    physical_key: str
    rate_key: str
    phase: str
    repeat: int
    source: str
    units: int
    width: int | None
    arm: str
    monotonic_wall_seconds: float
    process_user_plus_system_aggregate_cpu_seconds: float
    peak_simultaneous_tree_rss_bytes: int
    provenance: MeterProvenance
    inner_monotonic_wall_seconds: float | None = None
    environment_sha256: str | None = None
    prior_authorization_sha256: str | None = None
    measurement_record_sha256: str | None = None


def _cgroup_v2_directory() -> Path:
    try:
        records = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ContractError("cgroup-v2 membership is unavailable") from error
    matches = [line.split("::", 1)[1] for line in records if line.startswith("0::")]
    if len(matches) != 1:
        raise ContractError("cgroup-v2 membership is not unique")
    path = (Path("/sys/fs/cgroup") / matches[0].lstrip("/")).resolve()
    if not path.is_dir():
        raise ContractError("cgroup-v2 directory is unavailable")
    return path


def meter_implementation_binding(root: Path = ROOT) -> dict[str, Any]:
    implementation = Path(__file__).resolve()
    return {
        "path": str(implementation.relative_to(root.resolve())).replace("\\", "/"),
        "sha256": frozen_run.sha256(implementation),
        "wall_meter": "time.perf_counter",
        "aggregate_cpu_meter": "cgroup-tree-user-plus-system",
        "peak_rss_meter": "cgroup-memory.peak",
        "cpu_counter_cumulative": True,
        "rss_counter_cumulative_high_water": True,
    }


def cumulative_resource_meter_snapshot() -> MeterSnapshot:
    """Read the bound cumulative cgroup-v2 CPU and memory high-water meters."""

    cgroup = _cgroup_v2_directory()
    try:
        cpu_fields = {
            key: int(value)
            for key, value in (
                line.split(maxsplit=1)
                for line in (cgroup / "cpu.stat").read_text(encoding="utf-8").splitlines()
            )
        }
        peak_rss = int((cgroup / "memory.peak").read_text(encoding="utf-8").strip())
    except (OSError, ValueError, KeyError) as error:
        raise ContractError("cumulative cgroup resource meter is unavailable") from error
    if "user_usec" not in cpu_fields or "system_usec" not in cpu_fields:
        raise ContractError("cgroup CPU meter lacks user/system counters")
    return MeterSnapshot(
        monotonic_wall_seconds=time.perf_counter(),
        process_user_plus_system_aggregate_cpu_seconds=(
            cpu_fields["user_usec"] + cpu_fields["system_usec"]
        )
        / 1_000_000.0,
        simultaneous_tree_rss_bytes=peak_rss,
    )


def live_resource_environment() -> dict[str, Any]:
    """Read the live four-CPU/cgroup/tmpfs identity used by authorization."""

    if not hasattr(os, "sched_getaffinity"):
        raise ContractError("live CPU affinity is unavailable")
    affinity = tuple(sorted(os.sched_getaffinity(0)))
    cgroup = _cgroup_v2_directory()
    try:
        quota_text, period_text = (cgroup / "cpu.max").read_text(
            encoding="utf-8"
        ).split()
        memory_text = (cgroup / "memory.max").read_text(encoding="utf-8").strip()
        tmpfs = os.statvfs("/dev/shm")
    except (OSError, ValueError) as error:
        raise ContractError("live cgroup resource limits are unavailable") from error
    if quota_text == "max" or memory_text == "max":
        raise ContractError("live CPU or memory limit is unbounded")
    cpu_limit = int(quota_text) / int(period_text)
    return {
        "torch_version": torch.__version__,
        "device": "cpu",
        "default_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "affinity_cpus": affinity,
        "cpu_limit": cpu_limit,
        "memory_limit_bytes": int(memory_text),
        "tmpfs_limit_bytes": int(tmpfs.f_blocks * tmpfs.f_frsize),
        "meter_implementation": meter_implementation_binding(),
    }


def canonical_meter_provenance() -> MeterProvenance:
    binding = meter_implementation_binding()
    return MeterProvenance(
        wall_meter=binding["wall_meter"],
        aggregate_cpu_meter=binding["aggregate_cpu_meter"],
        peak_rss_meter=binding["peak_rss_meter"],
        meter_source_sha256=binding["sha256"],
    )


def resource_observation(
    physical_unit: PhysicalUnit,
    before: MeterSnapshot,
    after: MeterSnapshot,
    *,
    provenance: MeterProvenance,
    inner_monotonic_wall_seconds: float | None = None,
    environment_sha256: str | None = None,
    prior_authorization_sha256: str | None = None,
    measurement_record_sha256: str | None = None,
) -> ResourceObservation:
    wall = after.monotonic_wall_seconds - before.monotonic_wall_seconds
    cpu = (
        after.process_user_plus_system_aggregate_cpu_seconds
        - before.process_user_plus_system_aggregate_cpu_seconds
    )
    observation = ResourceObservation(
        physical_key=physical_unit.key,
        rate_key=physical_unit.rate_key,
        phase=physical_unit.phase,
        repeat=physical_unit.repeat,
        source=physical_unit.source,
        units=physical_unit.units,
        width=physical_unit.width,
        arm=physical_unit.arm,
        monotonic_wall_seconds=wall,
        process_user_plus_system_aggregate_cpu_seconds=cpu,
        peak_simultaneous_tree_rss_bytes=max(
            before.simultaneous_tree_rss_bytes,
            after.simultaneous_tree_rss_bytes,
        ),
        provenance=provenance,
        inner_monotonic_wall_seconds=inner_monotonic_wall_seconds,
        environment_sha256=environment_sha256,
        prior_authorization_sha256=prior_authorization_sha256,
        measurement_record_sha256=measurement_record_sha256,
    )
    _validate_observation_values(observation)
    return observation


def _is_positive_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_observation_values(observation: ResourceObservation) -> None:
    if not _is_positive_finite(observation.monotonic_wall_seconds):
        raise ContractError("resource wall timing is non-positive or non-finite")
    if not _is_positive_finite(
        observation.process_user_plus_system_aggregate_cpu_seconds
    ):
        raise ContractError("resource aggregate CPU is non-positive or non-finite")
    if (
        type(observation.peak_simultaneous_tree_rss_bytes) is not int
        or observation.peak_simultaneous_tree_rss_bytes <= 0
    ):
        raise ContractError("resource peak RSS is invalid")
    provenance = observation.provenance
    if provenance != canonical_meter_provenance():
        raise ContractError("resource observation meter implementation is not bound")
    profile_fresh = observation.phase == "L1-profile-fresh"
    if profile_fresh:
        if (
            not _is_positive_finite(observation.inner_monotonic_wall_seconds)
            or observation.inner_monotonic_wall_seconds
            > observation.monotonic_wall_seconds
        ):
            raise ContractError("profile inner/outer wall timings are invalid")
    elif observation.inner_monotonic_wall_seconds is not None:
        raise ContractError("inner wall timing is only valid for fresh profile batches")


def validate_resource_observations(
    observations: Sequence[ResourceObservation],
    observed_manifest: Sequence[PhysicalUnit],
    *,
    require_path_backed: bool = False,
    expected_environment_sha256: str | None = None,
    expected_prior_authorization_sha256: str | None = None,
) -> None:
    _validate_physical_manifest(observed_manifest)
    expected = {item.key: item for item in observed_manifest}
    if len(observations) != len(expected):
        raise ContractError("resource observation count does not match physical manifest")
    if {item.physical_key for item in observations} != set(expected):
        raise ContractError("resource observations are missing or unexpected")
    if len({item.physical_key for item in observations}) != len(observations):
        raise ContractError("resource observation is duplicated")
    for observation in observations:
        _validate_observation_values(observation)
        if require_path_backed and (
            not _is_sha256(observation.environment_sha256)
            or not _is_sha256(observation.prior_authorization_sha256)
            or not _is_sha256(observation.measurement_record_sha256)
        ):
            raise ContractError("resource observation lacks path-backed authorization")
        if (
            expected_environment_sha256 is not None
            and observation.environment_sha256 != expected_environment_sha256
        ):
            raise ContractError("resource observation environment binding changed")
        if (
            expected_prior_authorization_sha256 is not None
            and observation.prior_authorization_sha256
            != expected_prior_authorization_sha256
        ):
            raise ContractError("resource observation prior authorization changed")
        physical = expected[observation.physical_key]
        binding = (
            observation.rate_key,
            observation.phase,
            observation.repeat,
            observation.source,
            observation.units,
            observation.width,
            observation.arm,
        )
        expected_binding = (
            physical.rate_key,
            physical.phase,
            physical.repeat,
            physical.source,
            physical.units,
            physical.width,
            physical.arm,
        )
        if binding != expected_binding:
            raise ContractError("resource observation provenance binding mismatch")


def _snapshot_payload(snapshot: MeterSnapshot) -> dict[str, Any]:
    return {
        "monotonic_wall_seconds": snapshot.monotonic_wall_seconds,
        "aggregate_cpu_seconds": (
            snapshot.process_user_plus_system_aggregate_cpu_seconds
        ),
        "simultaneous_tree_rss_bytes": snapshot.simultaneous_tree_rss_bytes,
    }


def _snapshot_from_payload(payload: Mapping[str, Any]) -> MeterSnapshot:
    _require_exact_keys(
        payload,
        {
            "monotonic_wall_seconds",
            "aggregate_cpu_seconds",
            "simultaneous_tree_rss_bytes",
        },
        "raw meter snapshot",
    )
    return MeterSnapshot(
        monotonic_wall_seconds=payload["monotonic_wall_seconds"],
        process_user_plus_system_aggregate_cpu_seconds=payload[
            "aggregate_cpu_seconds"
        ],
        simultaneous_tree_rss_bytes=payload["simultaneous_tree_rss_bytes"],
    )


def _physical_unit_from_payload(payload: Mapping[str, Any]) -> PhysicalUnit:
    _require_exact_keys(
        payload,
        {
            "key",
            "rate_key",
            "phase",
            "repeat",
            "source",
            "units",
            "width",
            "arm",
            "eligible_for_rate",
        },
        "measurement physical unit",
    )
    if type(payload["eligible_for_rate"]) is not bool:
        raise ContractError("measurement physical rate eligibility is invalid")
    result = PhysicalUnit(
        key=payload["key"],
        rate_key=payload["rate_key"],
        phase=payload["phase"],
        repeat=payload["repeat"],
        source=payload["source"],
        units=payload["units"],
        width=payload["width"],
        arm=payload["arm"],
        eligible_for_rate=payload["eligible_for_rate"],
    )
    _validate_physical_manifest((result,))
    return result


def _derived_observation_payload(observation: ResourceObservation) -> dict[str, Any]:
    return {
        "physical_key": observation.physical_key,
        "rate_key": observation.rate_key,
        "phase": observation.phase,
        "repeat": observation.repeat,
        "source": observation.source,
        "units": observation.units,
        "width": observation.width,
        "arm": observation.arm,
        "monotonic_wall_seconds": observation.monotonic_wall_seconds,
        "aggregate_cpu_seconds": (
            observation.process_user_plus_system_aggregate_cpu_seconds
        ),
        "peak_rss_bytes": observation.peak_simultaneous_tree_rss_bytes,
        "provenance": {
            field: getattr(observation.provenance, field)
            for field in observation.provenance.__dataclass_fields__
        },
        "inner_monotonic_wall_seconds": observation.inner_monotonic_wall_seconds,
        "environment_sha256": observation.environment_sha256,
        "prior_authorization_sha256": observation.prior_authorization_sha256,
    }


OPERATION_DESCRIPTOR_KEYS = {
    "operation_id",
    "physical_key",
    "rate_key",
    "width",
    "arm",
    "source",
    "units",
    "input_manifest_sha256",
    "environment_sha256",
}


@dataclass(frozen=True, slots=True)
class CanonicalOperationRegistration:
    operation_id: str
    physical_key: str
    rate_key: str
    width: int | None
    arm: str
    source: str
    units: int


# This is deliberately empty. Adding an entry and its internal implementation is
# a separately reviewed code change; config/authorization callers cannot register
# an operation or provide a callback.
CANONICAL_OPERATION_REGISTRY: Mapping[
    str, CanonicalOperationRegistration
] = MappingProxyType({})


def _derive_operation_descriptors(
    manifest: Sequence[PhysicalUnit],
    *,
    registry: Mapping[str, CanonicalOperationRegistration],
    input_manifest_sha256: str,
    environment_sha256: str,
) -> list[dict[str, Any]]:
    _validate_physical_manifest(manifest)
    if not _is_sha256(input_manifest_sha256) or not _is_sha256(environment_sha256):
        raise ContractError("measurement operation descriptor identity is malformed")
    by_physical: dict[str, CanonicalOperationRegistration] = {}
    for operation_id, registration in registry.items():
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or operation_id != registration.operation_id
            or registration.physical_key in by_physical
        ):
            raise ContractError("canonical operation registry is malformed")
        by_physical[registration.physical_key] = registration
    missing = [item.key for item in manifest if item.key not in by_physical]
    if missing:
        raise ContractError(
            f"closed canonical operation registry lacks physical units: {missing}"
        )
    descriptors: list[dict[str, Any]] = []
    for physical in manifest:
        registration = by_physical[physical.key]
        expected = (
            physical.key,
            physical.rate_key,
            physical.width,
            physical.arm,
            physical.source,
            physical.units,
        )
        observed = (
            registration.physical_key,
            registration.rate_key,
            registration.width,
            registration.arm,
            registration.source,
            registration.units,
        )
        if observed != expected:
            raise ContractError("canonical operation registry physical binding changed")
        descriptors.append(
            {
                "operation_id": registration.operation_id,
                "physical_key": physical.key,
                "rate_key": physical.rate_key,
                "width": physical.width,
                "arm": physical.arm,
                "source": physical.source,
                "units": physical.units,
                "input_manifest_sha256": input_manifest_sha256,
                "environment_sha256": environment_sha256,
            }
        )
    return descriptors


def _validate_operation_descriptors(
    descriptors: Sequence[Mapping[str, Any]],
    manifest: Sequence[PhysicalUnit],
    *,
    registry: Mapping[str, CanonicalOperationRegistration],
    environment_sha256: str,
    expected_input_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    if any(not isinstance(item, Mapping) for item in descriptors):
        raise ContractError("operation descriptor is not a mapping")
    for descriptor in descriptors:
        _require_exact_keys(descriptor, OPERATION_DESCRIPTOR_KEYS, "operation descriptor")
    expected = _derive_operation_descriptors(
        manifest,
        registry=registry,
        input_manifest_sha256=expected_input_manifest_sha256,
        environment_sha256=environment_sha256,
    )
    if [dict(item) for item in descriptors] != expected:
        raise ContractError("caller-authored operation manifest is prohibited")
    return {item["physical_key"]: item for item in expected}


def _prior_measurement_authorization_mechanics(
    config: Mapping[str, Any],
    cache_verdict: "CacheVerdict",
    *,
    registry: Mapping[str, CanonicalOperationRegistration],
) -> dict[str, Any]:
    environment = canonical_authorized_environment(config)
    environment_sha256 = sha256_bytes(canonical_json_bytes(environment))
    manifest = (
        *warmup_l1_physical_manifest(),
        *canonical_unresolved_measurement_manifest(cache_verdict),
    )
    manifest_payload = _physical_manifest_payload(manifest)
    manifest_sha256 = sha256_bytes(canonical_json_bytes(manifest_payload))
    descriptors = _derive_operation_descriptors(
        manifest,
        registry=registry,
        input_manifest_sha256=manifest_sha256,
        environment_sha256=environment_sha256,
    )
    payload = {
        "schema_version": PRIOR_MEASUREMENT_AUTHORIZATION_SCHEMA,
        "run_id": RUN_ID,
        "stage": "prior-phase-resource-measurement",
        "cache_evidence_sha256": cache_verdict.evidence_sha256,
        "environment_sha256": environment_sha256,
        "meter_implementation_sha256": meter_implementation_binding()["sha256"],
        "physical_manifest": manifest_payload,
        "physical_manifest_sha256": manifest_sha256,
        "input_manifest_sha256": manifest_sha256,
        "operation_descriptors": descriptors,
        "operation_descriptors_sha256": sha256_bytes(
            canonical_json_bytes(descriptors)
        ),
        "scientific_values_authorized": False,
    }
    return json.loads(canonical_json_bytes(payload))


def _validate_prior_measurement_authorization(
    authorization: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    cache_verdict: "CacheVerdict",
    registry: Mapping[str, CanonicalOperationRegistration],
    environment_sha256: str,
) -> None:
    _require_exact_keys(
        authorization,
        {
            "schema_version",
            "run_id",
            "stage",
            "cache_evidence_sha256",
            "environment_sha256",
            "meter_implementation_sha256",
            "physical_manifest",
            "physical_manifest_sha256",
            "input_manifest_sha256",
            "operation_descriptors",
            "operation_descriptors_sha256",
            "scientific_values_authorized",
        },
        "prior measurement authorization",
    )
    manifest_payload = authorization["physical_manifest"]
    if not isinstance(manifest_payload, list):
        raise ContractError("prior measurement manifest is not a list")
    expected = _prior_measurement_authorization_mechanics(
        config,
        cache_verdict,
        registry=registry,
    )
    if (
        environment_sha256 != expected["environment_sha256"]
        or dict(authorization) != expected
    ):
        raise ContractError("prior measurement authorization binding changed")


def authorized_measurement_record(
    physical_key: str,
    operation_id: str,
    *,
    prior_authorization_path: Path | None,
) -> dict[str, Any]:
    """Fail-closed production stub for a future internal-operation package."""

    del prior_authorization_path
    registration = CANONICAL_OPERATION_REGISTRY.get(operation_id)
    if registration is None or registration.physical_key != physical_key:
        raise AuthorizationError(
            "EXP-658 canonical operation registry is empty/unregistered"
        )
    raise AuthorizationError(
        "EXP-658 internal canonical operation executor is not implemented"
    )


def _measurement_record_from_snapshots_mechanics(
    physical_unit: PhysicalUnit,
    before: MeterSnapshot,
    after: MeterSnapshot,
    *,
    operation_descriptor: Mapping[str, Any],
    environment_sha256: str,
    prior_authorization_sha256: str,
    inner_monotonic_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Pure record-shape mechanics; it performs no operation or measurement."""

    _validate_physical_manifest((physical_unit,))
    _require_exact_keys(
        operation_descriptor,
        OPERATION_DESCRIPTOR_KEYS,
        "operation descriptor",
    )
    if operation_descriptor["physical_key"] != physical_unit.key:
        raise ContractError("measurement operation descriptor changed")
    observation = resource_observation(
        physical_unit,
        before,
        after,
        provenance=canonical_meter_provenance(),
        inner_monotonic_wall_seconds=inner_monotonic_wall_seconds,
        environment_sha256=environment_sha256,
        prior_authorization_sha256=prior_authorization_sha256,
    )
    unhashed = {
        "schema_version": MEASUREMENT_RECORD_SCHEMA,
        "factory": meter_implementation_binding()
        | {"callable": "authorized_measurement_record"},
        "physical_unit": _physical_manifest_payload((physical_unit,))[0],
        "operation_descriptor": dict(operation_descriptor),
        "raw_before": _snapshot_payload(before),
        "raw_after": _snapshot_payload(after),
        "observation": _derived_observation_payload(observation),
        "environment_sha256": environment_sha256,
        "prior_authorization_sha256": prior_authorization_sha256,
    }
    return unhashed | {
        "record_sha256": sha256_bytes(canonical_json_bytes(unhashed))
    }


def _validate_measurement_record_payload(
    payload: Mapping[str, Any],
    *,
    environment_sha256: str,
    prior_authorization_sha256: str,
    authorized_operation_descriptors: Mapping[str, Mapping[str, Any]],
) -> ResourceObservation:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "factory",
            "physical_unit",
            "operation_descriptor",
            "raw_before",
            "raw_after",
            "observation",
            "environment_sha256",
            "prior_authorization_sha256",
            "record_sha256",
        },
        "measurement record",
    )
    unhashed = dict(payload)
    recorded_sha256 = unhashed.pop("record_sha256")
    if (
        payload["schema_version"] != MEASUREMENT_RECORD_SCHEMA
        or payload["factory"]
        != meter_implementation_binding()
        | {"callable": "authorized_measurement_record"}
        or payload["environment_sha256"] != environment_sha256
        or payload["prior_authorization_sha256"]
        != prior_authorization_sha256
        or recorded_sha256 != sha256_bytes(canonical_json_bytes(unhashed))
    ):
        raise ContractError("measurement record binding changed")
    physical = _physical_unit_from_payload(payload["physical_unit"])
    if (
        physical.key not in authorized_operation_descriptors
        or payload["operation_descriptor"]
        != dict(authorized_operation_descriptors[physical.key])
    ):
        raise ContractError("measurement operation descriptor changed")
    before = _snapshot_from_payload(payload["raw_before"])
    after = _snapshot_from_payload(payload["raw_after"])
    recomputed = resource_observation(
        physical,
        before,
        after,
        provenance=canonical_meter_provenance(),
        inner_monotonic_wall_seconds=payload["observation"].get(
            "inner_monotonic_wall_seconds"
        ),
        environment_sha256=environment_sha256,
        prior_authorization_sha256=prior_authorization_sha256,
    )
    if payload["observation"] != _derived_observation_payload(recomputed):
        raise ContractError("measurement derived observation changed")
    return ResourceObservation(
        **{
            field: getattr(recomputed, field)
            for field in recomputed.__dataclass_fields__
            if field != "measurement_record_sha256"
        },
        measurement_record_sha256=recorded_sha256,
    )


def _resolve_dynamic_budget_from_observations(
    observations: Sequence[ResourceObservation],
    observed_manifest: Sequence[PhysicalUnit],
    projection_manifest: Sequence[PhysicalUnit],
    *,
    projection_stage: str,
    cache_verdict: "CacheVerdict",
    require_canonical_stage: bool = True,
) -> dict[str, Any]:
    """Project serial physical work from maximum matching mean rates."""

    if projection_stage not in {"remaining-L1", "L2"}:
        raise ValueError("projection stage must be remaining-L1 or L2")
    if not isinstance(cache_verdict, CacheVerdict) or not cache_verdict.evidence_validated:
        raise ContractError("dynamic budget requires validated CacheVerdict")
    canonical_observed = (
        warmup_l1_physical_manifest()
        if projection_stage == "remaining-L1"
        else (
            *warmup_l1_physical_manifest(),
            *remaining_l1_physical_manifest(cache_verdict),
        )
    )
    canonical_projection = (
        remaining_l1_physical_manifest(cache_verdict)
        if projection_stage == "remaining-L1"
        else l2_physical_manifest(cache_verdict)
    )
    if require_canonical_stage and tuple(observed_manifest) != tuple(canonical_observed):
        raise ContractError("observed stage manifest is not exactly canonical")
    if require_canonical_stage and tuple(projection_manifest) != tuple(canonical_projection):
        raise ContractError("projection stage/cache manifest is not exactly canonical")
    validate_resource_observations(observations, observed_manifest)
    _validate_physical_manifest(projection_manifest)
    observed_by_key = {item.key: item for item in observed_manifest}
    grouped: dict[CoverageClass, list[ResourceObservation]] = {}
    for item in observations:
        physical = observed_by_key[item.physical_key]
        if physical.eligible_for_rate:
            if projection_stage == "L2" and item.phase == "L1-warmup":
                if not item.rate_key.startswith("fixed_"):
                    continue
            grouped.setdefault(physical_coverage_class(physical), []).append(item)
    projection_units: dict[CoverageClass, int] = {}
    for item in projection_manifest:
        coverage = physical_coverage_class(item)
        projection_units[coverage] = projection_units.get(coverage, 0) + item.units
    missing = set(projection_units).difference(grouped)
    if missing:
        raise ContractError(
            "missing physical resource coverage: "
            f"{sorted(coverage_class_id(item) for item in missing)}"
        )
    wall_rates = {
        coverage_class_id(component): max(
            item.monotonic_wall_seconds / item.units for item in rows
        )
        for component, rows in grouped.items()
    }
    cpu_rates = {
        coverage_class_id(component): max(
            item.process_user_plus_system_aggregate_cpu_seconds / item.units
            for item in rows
        )
        for component, rows in grouped.items()
    }
    expected_wall = sum(
        wall_rates[coverage_class_id(name)] * units
        for name, units in projection_units.items()
    )
    expected_cpu = sum(
        cpu_rates[coverage_class_id(name)] * units
        for name, units in projection_units.items()
    )
    maximum_rss = max(item.peak_simultaneous_tree_rss_bytes for item in observations)
    return {
        "expected": {
            "monotonic_wall_seconds": expected_wall,
            "process_user_plus_system_aggregate_cpu_seconds": expected_cpu,
        },
        "soft_budget": {
            "monotonic_wall_seconds": 1.25 * expected_wall,
            "process_user_plus_system_aggregate_cpu_seconds": 1.25
            * expected_cpu,
        },
        "hard_runaway_ceiling": {
            "monotonic_wall_seconds": 1.75 * expected_wall,
            "process_user_plus_system_aggregate_cpu_seconds": 1.75
            * expected_cpu,
        },
        "peak_simultaneous_tree_rss_guard_bytes": math.ceil(1.25 * maximum_rss),
        "maximum_matching_mean_rates": {
            "monotonic_wall_seconds": wall_rates,
            "process_user_plus_system_aggregate_cpu_seconds": cpu_rates,
        },
        "observed_manifest_sha256": sha256_bytes(
            canonical_json_bytes(_physical_manifest_payload(observed_manifest))
        ),
        "projection_manifest_sha256": sha256_bytes(
            canonical_json_bytes(_physical_manifest_payload(projection_manifest))
        ),
        "projection_stage": projection_stage,
        "shared_generation_charged_once": True,
        "model_execution_shared": False,
        "wall_cpu_conflated": False,
        "summed_serial_peak_rss": False,
        "fixed_short_wall_cap_used": False,
        "fidelity_reduced": False,
    }


@dataclass(frozen=True, slots=True)
class ComparableRateEvidence:
    rate_key: str
    coverage_class: str
    width: int | None
    arm: str
    source: str
    experiment: str
    authorization_path: str
    authorization_sha256: str
    record_path: str
    record_file_sha256: str
    record_sha256: str
    physical_key: str
    observed_units: int
    wall_seconds_per_unit: float
    aggregate_cpu_seconds_per_unit: float
    peak_rss_bytes: int
    comparability: str


def _path_backed_json(
    root: Path, relative: str, expected_sha256: str, *, label: str
) -> Mapping[str, Any]:
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not path.is_file()
        or not _is_sha256(expected_sha256)
        or frozen_run.sha256(path) != expected_sha256
    ):
        raise ContractError(f"{label} comparable evidence path/hash mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} comparable evidence is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ContractError(f"{label} comparable evidence is not a mapping")
    return payload


def _load_prior_measurement_authorization(
    config: Mapping[str, Any],
    *,
    cache_verdict: "CacheVerdict",
    registry: Mapping[str, CanonicalOperationRegistration],
    evidence_root: Path,
    environment_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    measurement = config["L1"]["budget"]["prior_phase_measurement_evidence"]
    _require_exact_keys(
        measurement,
        {"authorization_path", "authorization_sha256", "warmup_records"},
        "prior-phase measurement evidence",
    )
    if (
        not isinstance(measurement["authorization_path"], str)
        or not _is_sha256(measurement["authorization_sha256"])
    ):
        raise ContractError("prior-phase measurement evidence is absent")
    authorization = _path_backed_json(
        evidence_root,
        measurement["authorization_path"],
        measurement["authorization_sha256"],
        label="prior measurement authorization",
    )
    authorization_path = (evidence_root / measurement["authorization_path"]).resolve()
    if authorization_path.read_bytes() != canonical_json_bytes(authorization):
        raise ContractError("prior measurement authorization is not canonical JSON")
    _validate_prior_measurement_authorization(
        authorization,
        config=config,
        cache_verdict=cache_verdict,
        registry=registry,
        environment_sha256=environment_sha256,
    )
    if sha256_bytes(canonical_json_bytes(authorization)) != measurement[
        "authorization_sha256"
    ]:
        raise ContractError("prior measurement authorization content hash changed")
    return authorization, measurement["authorization_sha256"]


def _load_measurement_descriptor(
    descriptor: Mapping[str, Any],
    *,
    evidence_root: Path,
    environment_sha256: str,
    prior_authorization_sha256: str,
    authorized_operation_descriptors: Mapping[str, Mapping[str, Any]],
) -> ResourceObservation:
    _require_exact_keys(
        descriptor,
        {
            "physical_key",
            "rate_key",
            "record_path",
            "record_file_sha256",
            "record_sha256",
        },
        "measurement descriptor",
    )
    if (
        not isinstance(descriptor["physical_key"], str)
        or not descriptor["physical_key"]
        or descriptor["rate_key"] not in REGISTERED_RATE_KEYS
        or not isinstance(descriptor["record_path"], str)
        or not descriptor["record_path"]
        or not _is_sha256(descriptor["record_file_sha256"])
        or not _is_sha256(descriptor["record_sha256"])
    ):
        raise ContractError("measurement descriptor is malformed")
    record = _path_backed_json(
        evidence_root,
        descriptor["record_path"],
        descriptor["record_file_sha256"],
        label=f"{descriptor['rate_key']} exact measurement",
    )
    record_path = (evidence_root / descriptor["record_path"]).resolve()
    if record_path.read_bytes() != canonical_json_bytes(record):
        raise ContractError("exact measurement record is not canonical JSON")
    observation = _validate_measurement_record_payload(
        record,
        environment_sha256=environment_sha256,
        prior_authorization_sha256=prior_authorization_sha256,
        authorized_operation_descriptors=authorized_operation_descriptors,
    )
    if (
        observation.physical_key != descriptor["physical_key"]
        or observation.rate_key != descriptor["rate_key"]
        or observation.measurement_record_sha256 != descriptor["record_sha256"]
    ):
        raise ContractError("exact measurement descriptor binding changed")
    return observation


def missing_exact_remaining_l1_rate_keys(
    cache_verdict: "CacheVerdict",
) -> tuple[str, ...]:
    warmup_rates = {
        item.rate_key
        for item in warmup_l1_physical_manifest()
        if item.eligible_for_rate
    }
    projection_rates = {
        item.rate_key for item in remaining_l1_physical_manifest(cache_verdict)
    }
    return tuple(sorted(projection_rates - warmup_rates))


def _load_comparable_resource_rates_mechanics(
    config: Mapping[str, Any],
    cache_verdict: "CacheVerdict",
    *,
    registry: Mapping[str, CanonicalOperationRegistration],
    evidence_root: Path,
) -> dict[CoverageClass, tuple[ComparableRateEvidence, ...]]:
    """Load canonical unresolved records grouped by exact coverage class."""

    provenance = config["L1"]["budget"]["comparable_source_provenance"]
    _validate_comparable_resource_provenance(provenance, root=evidence_root)
    exp609_record = provenance["EXP609"]
    exp643_record = provenance["EXP643"]
    exp609_identity = _path_backed_json(
        evidence_root,
        exp609_record["identity_path"],
        exp609_record["identity_sha256"],
        label="EXP609 identity",
    )
    exp609_rates = _path_backed_json(
        evidence_root,
        exp609_record["rate_path"],
        exp609_record["rate_sha256"],
        label="EXP609 rate",
    )
    exp643_identity = _path_backed_json(
        evidence_root,
        exp643_record["identity_path"],
        exp643_record["identity_sha256"],
        label="EXP643 identity",
    )
    exp643_rates = _path_backed_json(
        evidence_root,
        exp643_record["rate_path"],
        exp643_record["rate_sha256"],
        label="EXP643 rate",
    )
    if (
        exp609_identity.get("run_id")
        != "exp-609-tier6-straight-through-confirmation"
        or exp609_identity.get("status") != "completed"
        or exp609_identity.get("artifacts", {}).get("checkpoint_sha256")
        != SOURCE_CHECKPOINT_SHA256
        or exp609_identity.get("artifacts", {}).get("metrics")
        != exp609_record["rate_path"]
        or exp609_rates.get("run_id")
        != "exp-609-tier6-straight-through-confirmation"
        or exp609_rates.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
        or exp643_identity.get("run_id")
        != "exp-643-tier7-recurrent-depth-stability-diagnostic"
        or exp643_identity.get("mode") != "calibration"
        or exp643_identity.get("status") != "completed"
        or exp643_identity.get("source", {}).get("source_checkpoint_sha256")
        != SOURCE_CHECKPOINT_SHA256
        or exp643_rates != exp643_identity
    ):
        raise ContractError("comparable evidence identity is not lineage matched")
    # These two files are coarse lineage context only.  Their aggregate values
    # are deliberately never normalized, duplicated, or exposed as component rates.
    exact = config["L1"]["budget"]["exact_component_rate_evidence"]
    if not isinstance(exact, list):
        raise ContractError("exact component-rate evidence is not a list")
    if not exact:
        return {}
    if any(not isinstance(item, Mapping) for item in exact):
        raise ContractError("exact component-rate descriptor is not a mapping")
    unresolved_manifest = canonical_unresolved_measurement_manifest(cache_verdict)
    expected_physical_keys = [item.key for item in unresolved_manifest]
    if [item.get("physical_key") for item in exact] != expected_physical_keys:
        raise ContractError(
            "exact component records do not cover canonical unresolved physical units"
        )
    environment = canonical_authorized_environment(config)
    environment_sha256 = sha256_bytes(canonical_json_bytes(environment))
    prior_authorization, prior_sha256 = _load_prior_measurement_authorization(
        config,
        cache_verdict=cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
        environment_sha256=environment_sha256,
    )
    prior_manifest = tuple(
        _physical_unit_from_payload(item)
        for item in prior_authorization["physical_manifest"]
    )
    operation_descriptors = _validate_operation_descriptors(
        prior_authorization["operation_descriptors"],
        prior_manifest,
        registry=registry,
        environment_sha256=environment_sha256,
        expected_input_manifest_sha256=prior_authorization[
            "input_manifest_sha256"
        ],
    )
    grouped: dict[CoverageClass, list[ComparableRateEvidence]] = {}
    record_hashes: set[str] = set()
    for item in exact:
        if not isinstance(item, Mapping):
            raise ContractError("exact component-rate descriptor is not a mapping")
        _require_exact_keys(
            item,
            {
                "physical_key",
                "rate_key",
                "experiment",
                "record_path",
                "record_file_sha256",
                "record_sha256",
            },
            "exact component-rate descriptor",
        )
        descriptor = {
            key: item[key]
            for key in (
                "physical_key",
                "rate_key",
                "record_path",
                "record_file_sha256",
                "record_sha256",
            )
        }
        observation = _load_measurement_descriptor(
            descriptor,
            evidence_root=evidence_root,
            environment_sha256=environment_sha256,
            prior_authorization_sha256=prior_sha256,
            authorized_operation_descriptors=operation_descriptors,
        )
        if (
            item["record_sha256"] in record_hashes
            or not isinstance(item["experiment"], str)
            or not item["experiment"]
        ):
            raise ContractError("exact component-rate evidence is not one-to-one")
        record_hashes.add(item["record_sha256"])
        coverage = physical_coverage_class(
            next(
                physical
                for physical in unresolved_manifest
                if physical.key == observation.physical_key
            )
        )
        grouped.setdefault(coverage, []).append(ComparableRateEvidence(
            rate_key=item["rate_key"],
            coverage_class=coverage_class_id(coverage),
            width=coverage.width,
            arm=coverage.arm,
            source=coverage.source,
            experiment=item["experiment"],
            authorization_path=config["L1"]["budget"][
                "prior_phase_measurement_evidence"
            ]["authorization_path"],
            authorization_sha256=prior_sha256,
            record_path=item["record_path"],
            record_file_sha256=item["record_file_sha256"],
            record_sha256=item["record_sha256"],
            physical_key=observation.physical_key,
            observed_units=observation.units,
            wall_seconds_per_unit=(
                observation.monotonic_wall_seconds / observation.units
            ),
            aggregate_cpu_seconds_per_unit=(
                observation.process_user_plus_system_aggregate_cpu_seconds
                / observation.units
            ),
            peak_rss_bytes=observation.peak_simultaneous_tree_rss_bytes,
            comparability="exact-width-arm-source-prior-phase-authorized-component",
        ))
    return {key: tuple(value) for key, value in grouped.items()}


def _comparable_rate_payload(value: ComparableRateEvidence) -> dict[str, Any]:
    return {
        field: getattr(value, field)
        for field in value.__dataclass_fields__
    }


def _resource_observation_payload(value: ResourceObservation) -> dict[str, Any]:
    return _derived_observation_payload(value) | {
        "measurement_record_sha256": value.measurement_record_sha256
    }


def _load_authorized_warmup_observations_mechanics(
    config: Mapping[str, Any],
    cache_verdict: "CacheVerdict",
    *,
    registry: Mapping[str, CanonicalOperationRegistration],
    evidence_root: Path,
) -> tuple[tuple[ResourceObservation, ...], dict[str, Any]]:
    environment = canonical_authorized_environment(config)
    environment_sha256 = sha256_bytes(canonical_json_bytes(environment))
    prior_authorization, prior_sha256 = _load_prior_measurement_authorization(
        config,
        cache_verdict=cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
        environment_sha256=environment_sha256,
    )
    prior_manifest = tuple(
        _physical_unit_from_payload(item)
        for item in prior_authorization["physical_manifest"]
    )
    operation_descriptors = _validate_operation_descriptors(
        prior_authorization["operation_descriptors"],
        prior_manifest,
        registry=registry,
        environment_sha256=environment_sha256,
        expected_input_manifest_sha256=prior_authorization[
            "input_manifest_sha256"
        ],
    )
    measurement = config["L1"]["budget"]["prior_phase_measurement_evidence"]
    descriptors = measurement["warmup_records"]
    if not isinstance(descriptors, list):
        raise ContractError("warmup measurement descriptors are not a list")
    observations = tuple(
        _load_measurement_descriptor(
            descriptor,
            evidence_root=evidence_root,
            environment_sha256=environment_sha256,
            prior_authorization_sha256=prior_sha256,
            authorized_operation_descriptors=operation_descriptors,
        )
        for descriptor in descriptors
    )
    manifest = warmup_l1_physical_manifest()
    validate_resource_observations(
        observations,
        manifest,
        require_path_backed=True,
        expected_environment_sha256=environment_sha256,
        expected_prior_authorization_sha256=prior_sha256,
    )
    if len({item.measurement_record_sha256 for item in observations}) != len(
        observations
    ):
        raise ContractError("warmup measurement records are not one-to-one")
    binding = {
        "authorization_path": measurement["authorization_path"],
        "authorization_sha256": prior_sha256,
        "record_descriptors": [dict(item) for item in descriptors],
        "record_descriptors_sha256": sha256_bytes(
            canonical_json_bytes(descriptors)
        ),
        "environment_sha256": environment_sha256,
        "prior_authorization_manifest_sha256": prior_authorization[
            "physical_manifest_sha256"
        ],
    }
    return observations, binding


def _resolve_remaining_l1_budget_mechanics(
    config: Mapping[str, Any],
    cache_verdict: "CacheVerdict",
    *,
    registry: Mapping[str, CanonicalOperationRegistration],
    evidence_root: Path,
) -> dict[str, Any]:
    """Recompute the one canonical post-warmup L1 projection and envelope."""

    if not isinstance(cache_verdict, CacheVerdict) or not cache_verdict.evidence_validated:
        raise ContractError("remaining-L1 resolver requires validated CacheVerdict")
    warmup_manifest = warmup_l1_physical_manifest()
    projection_manifest = remaining_l1_physical_manifest(cache_verdict)
    unresolved_manifest = canonical_unresolved_measurement_manifest(cache_verdict)
    exact_descriptors = config["L1"]["budget"]["exact_component_rate_evidence"]
    if not isinstance(exact_descriptors, list):
        raise ContractError("exact component-rate evidence is not a list")
    present_physical = {
        item.get("physical_key")
        for item in exact_descriptors
        if isinstance(item, Mapping)
    }
    missing_units = [
        item for item in unresolved_manifest if item.key not in present_physical
    ]
    if missing_units:
        categories = sorted({item.rate_key for item in missing_units})
        coverage = sorted(
            {coverage_class_id(physical_coverage_class(item)) for item in missing_units}
        )
        raise ContractError(
            "missing exact one-to-one component rates: "
            f"categories={categories}; coverage={coverage}; "
            f"physical_units={[item.key for item in missing_units]}; "
            f"closed_registry_entries={len(registry)}"
        )
    comparable = _load_comparable_resource_rates_mechanics(
        config,
        cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
    )
    warmup_observations, warmup_evidence = (
        _load_authorized_warmup_observations_mechanics(
            config,
            cache_verdict,
            registry=registry,
            evidence_root=evidence_root,
        )
    )
    observed_rates: dict[CoverageClass, list[tuple[float, float]]] = {}
    for item in warmup_observations:
        physical = next(
            physical
            for physical in warmup_manifest
            if physical.key == item.physical_key
        )
        if not physical.eligible_for_rate:
            continue
        observed_rates.setdefault(physical_coverage_class(physical), []).append(
            (
                item.monotonic_wall_seconds / item.units,
                item.process_user_plus_system_aggregate_cpu_seconds / item.units,
            )
        )
    projection_units: dict[CoverageClass, int] = {}
    for item in projection_manifest:
        coverage = physical_coverage_class(item)
        projection_units[coverage] = projection_units.get(coverage, 0) + item.units
    used_comparable: dict[CoverageClass, tuple[ComparableRateEvidence, ...]] = {}
    selected_rates: dict[str, dict[str, Any]] = {}
    for coverage in sorted(projection_units, key=coverage_class_id):
        identifier = coverage_class_id(coverage)
        if coverage in observed_rates:
            rows = observed_rates[coverage]
            wall_rate = max(item[0] for item in rows)
            cpu_rate = max(item[1] for item in rows)
            selected_rates[identifier] = {
                "source": "validated-warmup-maximum-matching-mean",
                "record_count": len(rows),
                "wall_seconds_per_unit": wall_rate,
                "aggregate_cpu_seconds_per_unit": cpu_rate,
            }
        else:
            if coverage not in comparable:
                raise ContractError(
                    f"missing path-backed comparable coverage for {identifier}"
                )
            evidence = comparable[coverage]
            used_comparable[coverage] = evidence
            selected_rates[identifier] = {
                "source": "validated-exact-maximum-matching-mean",
                "experiments": sorted({item.experiment for item in evidence}),
                "record_count": len(evidence),
                "wall_seconds_per_unit": max(
                    item.wall_seconds_per_unit for item in evidence
                ),
                "aggregate_cpu_seconds_per_unit": max(
                    item.aggregate_cpu_seconds_per_unit for item in evidence
                ),
            }
    completed_wall = sum(item.monotonic_wall_seconds for item in warmup_observations)
    completed_cpu = sum(
        item.process_user_plus_system_aggregate_cpu_seconds
        for item in warmup_observations
    )
    remaining_wall = sum(
        selected_rates[coverage_class_id(key)]["wall_seconds_per_unit"] * units
        for key, units in projection_units.items()
    )
    remaining_cpu = sum(
        selected_rates[coverage_class_id(key)]["aggregate_cpu_seconds_per_unit"]
        * units
        for key, units in projection_units.items()
    )
    maximum_rss = max(
        [item.peak_simultaneous_tree_rss_bytes for item in warmup_observations]
        + [
            item.peak_rss_bytes
            for rows in used_comparable.values()
            for item in rows
        ]
    )
    expected_wall = completed_wall + remaining_wall
    expected_cpu = completed_cpu + remaining_cpu
    envelope = {
        "expected_wall_seconds": expected_wall,
        "expected_aggregate_cpu_seconds": expected_cpu,
        "maximum_observed_rss_bytes": maximum_rss,
        "soft_wall_seconds": 1.25 * expected_wall,
        "soft_aggregate_cpu_seconds": 1.25 * expected_cpu,
        "hard_wall_seconds": 1.75 * expected_wall,
        "hard_aggregate_cpu_seconds": 1.75 * expected_cpu,
        "rss_guard_bytes": math.ceil(1.25 * maximum_rss),
    }
    warmup_payload = [_resource_observation_payload(item) for item in warmup_observations]
    comparable_payload = {
        coverage_class_id(key): [_comparable_rate_payload(value) for value in rows]
        for key, rows in sorted(
            used_comparable.items(), key=lambda item: coverage_class_id(item[0])
        )
    }
    derivation = {
        "stage": "remaining-L1-after-warmup",
        "cache_evidence_sha256": cache_verdict.evidence_sha256,
        "environment_sha256": warmup_evidence["environment_sha256"],
        "warmup_observations_sha256": sha256_bytes(canonical_json_bytes(warmup_payload)),
        "warmup_measurement_evidence_sha256": sha256_bytes(
            canonical_json_bytes(warmup_evidence)
        ),
        "warmup_manifest_sha256": sha256_bytes(
            canonical_json_bytes(_physical_manifest_payload(warmup_manifest))
        ),
        "projection_manifest_sha256": sha256_bytes(
            canonical_json_bytes(_physical_manifest_payload(projection_manifest))
        ),
        "comparable_evidence_sha256": sha256_bytes(
            canonical_json_bytes(comparable_payload)
        ),
        "projection_units": {
            coverage_class_id(key): value
            for key, value in sorted(
                projection_units.items(), key=lambda item: coverage_class_id(item[0])
            )
        },
        "selected_rates": selected_rates,
        "completed_warmup": {
            "wall_seconds": completed_wall,
            "aggregate_cpu_seconds": completed_cpu,
        },
        "remaining_projection": {
            "wall_seconds": remaining_wall,
            "aggregate_cpu_seconds": remaining_cpu,
        },
        "numeric_resource_envelope": envelope,
        "wall_cpu_conflated": False,
        "summed_serial_peak_rss": False,
        "fixed_short_wall_cap_used": False,
        "fidelity_reduced": False,
    }
    derivation["derivation_sha256"] = sha256_bytes(canonical_json_bytes(derivation))
    return derivation


def resolve_remaining_l1_budget() -> dict[str, Any]:
    """Resolve the production remaining-L1 envelope from tracked evidence only."""

    config = load_config(CONFIG_PATH)
    cache_verdict = validate_incumbent_cache(config, root=ROOT)
    return _resolve_remaining_l1_budget_mechanics(
        config,
        cache_verdict,
        registry=CANONICAL_OPERATION_REGISTRY,
        evidence_root=ROOT,
    )


def _validate_budget_derivation_record(
    derivation: Mapping[str, Any],
    *,
    workload: Mapping[str, Any],
    config: Mapping[str, Any],
    cache_verdict: "CacheVerdict",
    registry: Mapping[str, CanonicalOperationRegistration],
    evidence_root: Path,
    environment_sha256: str,
) -> None:
    _require_exact_keys(
        derivation,
        {
            "stage",
            "cache_evidence_sha256",
            "environment_sha256",
            "warmup_observations_sha256",
            "warmup_measurement_evidence_sha256",
            "warmup_manifest_sha256",
            "projection_manifest_sha256",
            "comparable_evidence_sha256",
            "projection_units",
            "selected_rates",
            "completed_warmup",
            "remaining_projection",
            "numeric_resource_envelope",
            "wall_cpu_conflated",
            "summed_serial_peak_rss",
            "fixed_short_wall_cap_used",
            "fidelity_reduced",
            "derivation_sha256",
        },
        "remaining-L1 budget derivation",
    )
    unhashed = dict(derivation)
    recorded_hash = unhashed.pop("derivation_sha256")
    if (
        derivation["stage"] != "remaining-L1-after-warmup"
        or derivation["cache_evidence_sha256"] != cache_verdict.evidence_sha256
        or derivation["environment_sha256"] != environment_sha256
        or derivation["warmup_manifest_sha256"]
        != workload["warmup_manifest_sha256"]
        or derivation["projection_manifest_sha256"]
        != workload["remaining_l1_manifest_sha256"]
        or not _is_sha256(derivation["warmup_observations_sha256"])
        or not _is_sha256(derivation["warmup_measurement_evidence_sha256"])
        or not _is_sha256(derivation["comparable_evidence_sha256"])
        or recorded_hash != sha256_bytes(canonical_json_bytes(unhashed))
        or derivation["wall_cpu_conflated"] is not False
        or derivation["summed_serial_peak_rss"] is not False
        or derivation["fixed_short_wall_cap_used"] is not False
        or derivation["fidelity_reduced"] is not False
    ):
        raise ContractError("remaining-L1 budget derivation binding changed")
    _validate_numeric_resource_envelope(derivation["numeric_resource_envelope"])
    canonical = _resolve_remaining_l1_budget_mechanics(
        config,
        cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
    )
    if dict(derivation) != canonical:
        raise ContractError("remaining-L1 budget derivation is not canonical")


@dataclass(frozen=True, slots=True)
class CacheVerdict:
    valid: bool
    incumbent_wall_seconds: float | None
    incremental_allowance_seconds: float | None
    reason: str
    evidence_validated: bool
    full_1100_required: bool
    evidence_sha256: str


def _validated_cache_verdict(
    *,
    valid: bool,
    reason: str,
    evidence: Mapping[str, Any],
    incumbent_wall_seconds: float | None = None,
) -> CacheVerdict:
    allowance = (
        None
        if incumbent_wall_seconds is None
        else 285.0 - incumbent_wall_seconds
    )
    return CacheVerdict(
        valid=valid,
        incumbent_wall_seconds=incumbent_wall_seconds,
        incremental_allowance_seconds=allowance,
        reason=reason,
        evidence_validated=True,
        full_1100_required=not valid,
        evidence_sha256=sha256_bytes(canonical_json_bytes(evidence)),
    )


def validate_incumbent_cache(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> CacheVerdict:
    cached = config["L1"]["deployed_path_profile"]["cached_candidate005"]
    artifacts = cached["path_backed_artifacts"]
    expected_artifact_keys = {
        "submission_weights",
        "submission_manifest",
        "submission_model",
        "sandbox_record",
        "sandbox_runner",
        "dockerfile",
        "official_generator",
        "official_primes",
        "environment_record",
    }
    if set(artifacts) != expected_artifact_keys:
        raise ContractError("Candidate005 path-backed artifact schema is not closed")
    root_resolved = root.resolve()
    evidence: dict[str, Any] = {"artifacts": {}}
    paths: dict[str, Path] = {}
    for key in sorted(expected_artifact_keys):
        record = artifacts[key]
        if set(record) != {"path", "sha256"} or not _is_sha256(record["sha256"]):
            raise ContractError("Candidate005 artifact record is malformed")
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root_resolved):
            raise ContractError("Candidate005 artifact escapes repository root")
        paths[key] = path
        if not path.is_file():
            evidence["artifacts"][key] = "missing"
            return _validated_cache_verdict(
                valid=False,
                reason=f"{key}-missing",
                evidence=evidence,
            )
        actual = frozen_run.sha256(path)
        evidence["artifacts"][key] = actual
        if actual != record["sha256"]:
            return _validated_cache_verdict(
                valid=False,
                reason=f"{key}-hash-mismatch",
                evidence=evidence,
            )
    try:
        sandbox = json.loads(paths["sandbox_record"].read_text(encoding="utf-8"))
        environment = json.loads(
            paths["environment_record"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return _validated_cache_verdict(
            valid=False,
            reason="candidate005-record-parse-failure",
            evidence=evidence | {"parse_error": type(error).__name__},
        )
    sandbox_expected = cached["sandbox_record_expectation"]
    environment_expected = cached["environment_record_expectation"]
    if set(sandbox_expected) != {
        "challenge_commit",
        "dockerfile_sha256",
        "manifest_sha256",
        "model_sha256",
        "weights_sha256",
        "engine_command",
        "engine_version_sha256",
        "image",
        "image_id",
        "seed",
        "total",
        "timeout_seconds",
        "memory",
        "cpus",
        "tmpfs_size",
        "result_status",
        "inference_wall_seconds",
    }:
        raise ContractError("Candidate005 sandbox expectation schema is not closed")
    if set(environment_expected) != {
        "platform",
        "distribution",
        "python",
        "pytorch",
        "device",
        "torch_threads",
    }:
        raise ContractError("Candidate005 environment expectation schema is not closed")
    parsed_sandbox = {key: sandbox.get(key) for key in sandbox_expected}
    parsed_environment = {key: environment.get(key) for key in environment_expected}
    evidence.update(
        {
            "sandbox": parsed_sandbox,
            "environment": parsed_environment,
        }
    )
    if parsed_sandbox != dict(sandbox_expected):
        return _validated_cache_verdict(
            valid=False,
            reason="sandbox-record-content-mismatch",
            evidence=evidence,
        )
    if parsed_environment != dict(environment_expected):
        return _validated_cache_verdict(
            valid=False,
            reason="environment-record-content-mismatch",
            evidence=evidence,
        )
    if sandbox["image"] != PINNED_CANDIDATE_IMAGE:
        return _validated_cache_verdict(
            valid=False,
            reason="candidate005-image-not-pinned",
            evidence=evidence,
        )
    if hashlib.sha256(
        (sandbox["engine_version_output"] + "\n").encode()
    ).hexdigest() != sandbox["engine_version_sha256"]:
        return _validated_cache_verdict(
            valid=False,
            reason="engine-version-output-hash-mismatch",
            evidence=evidence,
        )
    incumbent = float(cached["incumbent_wall_seconds"])
    if not _is_positive_finite(incumbent) or sandbox["inference_wall_seconds"] != incumbent:
        return _validated_cache_verdict(
            valid=False,
            reason="incumbent-wall-mismatch",
            evidence=evidence,
        )
    allowance = 285.0 - incumbent
    if allowance != float(cached["incremental_allowance_seconds"]):
        raise ContractError("cached incumbent allowance is internally inconsistent")
    return _validated_cache_verdict(
        valid=True,
        incumbent_wall_seconds=incumbent,
        reason="exact-path-backed-cache-match",
        evidence=evidence,
    )


def candidate005_full_1100_rerun_command(output_dir: Path) -> tuple[str, ...]:
    return (
        "python",
        "tools/run_official_sandbox.py",
        "--engine",
        "podman",
        "--image",
        PINNED_CANDIDATE_IMAGE,
        "--submission",
        "submission",
        "--output-dir",
        str(output_dir),
        "--seed",
        "1c82674f2826d26d7c5de0bf1933457ddc9ce824c30acff4a8a37578b3a566e4",
        "--total",
        "1100",
        "--timeout",
        "300",
        "--memory",
        "8g",
        "--cpus",
        "4",
        "--tmpfs-size",
        "2g",
    )


def runtime_gate(
    incumbent_wall_seconds: float | None,
    challenger_batch_wall_seconds: Sequence[float],
    *,
    exact_100_case_wall_seconds: float | None = None,
) -> dict[str, Any]:
    if len(challenger_batch_wall_seconds) != 3 or any(
        not _is_positive_finite(value) for value in challenger_batch_wall_seconds
    ):
        raise ValueError(
            "exactly three positive finite fresh eight-case walls are required"
        )
    if exact_100_case_wall_seconds is not None and not _is_positive_finite(
        exact_100_case_wall_seconds
    ):
        raise ValueError("exact 100-case wall must be positive and finite")
    if (
        incumbent_wall_seconds is None
        or not _is_positive_finite(incumbent_wall_seconds)
        or incumbent_wall_seconds >= 285.0
    ):
        return {"decision": "fail", "reason": "incumbent-unavailable-or-ineligible"}
    allowance = 285.0 - incumbent_wall_seconds
    upper = 1.25 * 100 * max(challenger_batch_wall_seconds) / 8
    lower = 100 * min(challenger_batch_wall_seconds) / 8
    if upper <= allowance:
        decision = "pass"
        reason = "upper-bound-within-incremental-allowance"
    elif lower > allowance:
        decision = "fail"
        reason = "lower-bound-exceeds-incremental-allowance"
    elif exact_100_case_wall_seconds is None:
        decision = "straddle-requires-exact-100-case-profile"
        reason = "valid-bounds-straddle-allowance"
    elif 1.10 * exact_100_case_wall_seconds <= allowance:
        decision = "pass"
        reason = "exact-100-case-headroom-within-allowance"
    else:
        decision = "fail"
        reason = "exact-100-case-headroom-exceeds-allowance"
    return {
        "decision": decision,
        "reason": reason,
        "I_same_environment": incumbent_wall_seconds,
        "A": allowance,
        "U100": upper,
        "L100": lower,
        "T100": exact_100_case_wall_seconds,
        "aggregate_cpu_substituted_for_wall": False,
    }


def decision_safe_stop(
    *,
    at_preserved_checkpoint: bool,
    wall_seconds: float,
    aggregate_cpu_seconds: float,
    peak_rss_bytes: int,
    hard_wall_seconds: float,
    hard_aggregate_cpu_seconds: float,
    rss_guard_bytes: int,
    exact_impossibility: bool = False,
) -> bool:
    if not at_preserved_checkpoint:
        return False
    return exact_impossibility or any(
        (
            wall_seconds > hard_wall_seconds,
            aggregate_cpu_seconds > hard_aggregate_cpu_seconds,
            peak_rss_bytes > rss_guard_bytes,
        )
    )


def scaffold_hashes(root: Path = ROOT) -> dict[str, str]:
    paths = {
        "config": CONFIG_PATH.relative_to(ROOT),
        "calibration": Path(__file__).resolve().relative_to(ROOT),
        "preregistration": PREREGISTRATION_PATH.relative_to(ROOT),
        "calibration_test": CALIBRATION_TEST_PATH.relative_to(ROOT),
    }
    hashes = {name: frozen_run.sha256(root / path) for name, path in paths.items()}
    hashes.update(verify_frozen_l0_hashes(root))
    config = load_config(root / paths["config"])
    source = root / config["source"]["checkpoint"]
    if frozen_run.sha256(source) != SOURCE_CHECKPOINT_SHA256:
        raise ContractError("source checkpoint SHA-256 mismatch")
    hashes["source_checkpoint"] = SOURCE_CHECKPOINT_SHA256
    split_source = root / EXP300_SPLIT_PATH.relative_to(ROOT)
    if frozen_run.sha256(split_source) != EXP300_SPLIT_SHA256:
        raise ContractError("EXP-300 split source SHA-256 mismatch")
    hashes["exp300_split_source"] = EXP300_SPLIT_SHA256
    artifacts = config["L1"]["deployed_path_profile"]["cached_candidate005"][
        "path_backed_artifacts"
    ]
    for name, record in sorted(artifacts.items()):
        path = root / record["path"]
        if not path.is_file() or frozen_run.sha256(path) != record["sha256"]:
            raise ContractError(f"Candidate005 scaffold artifact mismatch: {name}")
        hashes[f"candidate005_{name}"] = record["sha256"]
    return hashes


def _physical_manifest_payload(manifest: Sequence[PhysicalUnit]) -> list[dict[str, Any]]:
    return [
        {
            "key": item.key,
            "rate_key": item.rate_key,
            "phase": item.phase,
            "repeat": item.repeat,
            "source": item.source,
            "units": item.units,
            "width": item.width,
            "arm": item.arm,
            "eligible_for_rate": item.eligible_for_rate,
        }
        for item in manifest
    ]


def workload_binding(
    config: Mapping[str, Any],
    cache_verdict: CacheVerdict,
    *,
    deterministic_attempt_allowance: int,
    root: Path = ROOT,
) -> dict[str, Any]:
    if type(deterministic_attempt_allowance) is not int or deterministic_attempt_allowance < 128:
        raise ContractError("future authorization must supply a numeric attempt allowance")
    if validate_incumbent_cache(config, root=root) != cache_verdict:
        raise ContractError("workload CacheVerdict is not live/path-backed")
    identity_manifest = canonical_l1_identity_manifest(
        deterministic_attempt_allowance
    )
    identity_binding = validate_l1_identity_manifest(
        identity_manifest,
        deterministic_attempt_allowance=deterministic_attempt_allowance,
    )
    if set(identity_binding) != {
        "split_sha256_by_width",
        "stream_manifest_sha256",
        "width256_stream_sha256",
        "checked_or_private_answers_read",
        "repeat_zero_is_warmup_only",
        "rare_owner",
        "profile_owner",
    }:
        raise ContractError("identity binding schema is not closed")
    if identity_binding["split_sha256_by_width"] != {
        "32": EXPECTED_SPLIT_SHA256[32],
        "64": EXPECTED_SPLIT_SHA256[64],
    } or not _is_sha256(identity_binding["stream_manifest_sha256"]) or (
        identity_binding["width256_stream_sha256"]
        != EXPECTED_WIDTH256_STREAM_SHA256
    ):
        raise ContractError("identity binding hashes do not match frozen splits")
    if (
        identity_binding["checked_or_private_answers_read"] is not False
        or identity_binding["repeat_zero_is_warmup_only"] is not True
        or identity_binding["rare_owner"] != "fresh-repeat-3-challenger"
        or identity_binding["profile_owner"] != "exact-reloaded-challenger"
    ):
        raise ContractError("identity ownership binding mismatch")
    warmup = warmup_l1_physical_manifest()
    remaining = remaining_l1_physical_manifest(cache_verdict)
    l2 = l2_physical_manifest(cache_verdict)
    return {
        "seed_manifest": stream_seed_manifest(),
        "identity_binding": dict(identity_binding),
        "deterministic_attempt_allowance": deterministic_attempt_allowance,
        "apportionment": {
            "width32": APPORTIONMENT[32],
            "width64": APPORTIONMENT[64],
            "l1_totals": apportionment_totals(training_schedule()),
            "l2_totals": apportionment_totals(training_schedule(25)),
        },
        "l1_physical": l1_physical_counters(),
        "warmup_manifest_sha256": sha256_bytes(
            canonical_json_bytes(_physical_manifest_payload(warmup))
        ),
        "remaining_l1_manifest_sha256": sha256_bytes(
            canonical_json_bytes(_physical_manifest_payload(remaining))
        ),
        "l2_manifest_sha256": sha256_bytes(
            canonical_json_bytes(_physical_manifest_payload(l2))
        ),
        "l2_projection_units": l2_projection_units(cache_verdict),
        "l2_batches": l2_batch_counters(),
        "runtime_gate": config["L1"]["runtime_feasibility"],
        "cache_verdict": {
            "valid": cache_verdict.valid,
            "incumbent_wall_seconds": cache_verdict.incumbent_wall_seconds,
            "incremental_allowance_seconds": cache_verdict.incremental_allowance_seconds,
            "reason": cache_verdict.reason,
            "evidence_validated": cache_verdict.evidence_validated,
            "full_1100_required": cache_verdict.full_1100_required,
            "evidence_sha256": cache_verdict.evidence_sha256,
        },
    }


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ContractError(f"{label} schema is not closed")


def _validate_numeric_resource_envelope(envelope: Mapping[str, Any]) -> None:
    _require_exact_keys(
        envelope,
        {
            "expected_wall_seconds",
            "expected_aggregate_cpu_seconds",
            "maximum_observed_rss_bytes",
            "soft_wall_seconds",
            "soft_aggregate_cpu_seconds",
            "hard_wall_seconds",
            "hard_aggregate_cpu_seconds",
            "rss_guard_bytes",
        },
        "numeric resource envelope",
    )
    for key in (
        "expected_wall_seconds",
        "expected_aggregate_cpu_seconds",
        "soft_wall_seconds",
        "soft_aggregate_cpu_seconds",
        "hard_wall_seconds",
        "hard_aggregate_cpu_seconds",
    ):
        if not _is_positive_finite(envelope[key]):
            raise ContractError("numeric resource envelope is not positive/finite")
    for key in ("maximum_observed_rss_bytes", "rss_guard_bytes"):
        if type(envelope[key]) is not int or envelope[key] <= 0:
            raise ContractError("numeric RSS envelope is invalid")
    if (
        envelope["soft_wall_seconds"]
        != 1.25 * envelope["expected_wall_seconds"]
        or envelope["soft_aggregate_cpu_seconds"]
        != 1.25 * envelope["expected_aggregate_cpu_seconds"]
        or envelope["hard_wall_seconds"]
        != 1.75 * envelope["expected_wall_seconds"]
        or envelope["hard_aggregate_cpu_seconds"]
        != 1.75 * envelope["expected_aggregate_cpu_seconds"]
        or envelope["rss_guard_bytes"]
        != math.ceil(1.25 * envelope["maximum_observed_rss_bytes"])
    ):
        raise ContractError("numeric resource envelope headroom formulas changed")


def _binary_size_bytes(value: str) -> int:
    if not isinstance(value, str) or len(value) < 2:
        raise ContractError("recorded resource size is malformed")
    suffix = value[-1].lower()
    factors = {"k": 1024, "m": 1024**2, "g": 1024**3}
    try:
        amount = int(value[:-1])
    except ValueError as error:
        raise ContractError("recorded resource size is malformed") from error
    if suffix not in factors or amount <= 0:
        raise ContractError("recorded resource size is malformed")
    return amount * factors[suffix]


def canonical_authorized_environment(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cached = config["L1"]["deployed_path_profile"]["cached_candidate005"]
    sandbox = cached["sandbox_record_expectation"]
    live = live_resource_environment()
    _require_exact_keys(
        live,
        {
            "torch_version",
            "device",
            "default_dtype",
            "deterministic_algorithms",
            "intra_op_threads",
            "inter_op_threads",
            "affinity_cpus",
            "cpu_limit",
            "memory_limit_bytes",
            "tmpfs_limit_bytes",
            "meter_implementation",
        },
        "live resource environment",
    )
    expected_resources = {
        "cpu_limit": float(sandbox["cpus"]),
        "memory_limit_bytes": _binary_size_bytes(sandbox["memory"]),
        "tmpfs_limit_bytes": _binary_size_bytes(sandbox["tmpfs_size"]),
    }
    affinity = tuple(live["affinity_cpus"])
    if (
        {
            key: live[key]
            for key in (
                "torch_version",
                "device",
                "default_dtype",
                "deterministic_algorithms",
                "intra_op_threads",
                "inter_op_threads",
            )
        }
        != {
            "torch_version": PINNED_TORCH_VERSION,
            "device": "cpu",
            "default_dtype": "float32",
            "deterministic_algorithms": True,
            "intra_op_threads": 4,
            "inter_op_threads": 1,
        }
        or
        len(affinity) != 4
        or len(set(affinity)) != 4
        or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
        or {key: live[key] for key in expected_resources} != expected_resources
        or live["meter_implementation"] != meter_implementation_binding()
    ):
        raise ContractError("live four-CPU/resource/meter identity mismatch")
    return {
        "current_live_probe": {
            "status": "current-live-probed",
            "torch_version": live["torch_version"],
            "device": live["device"],
            "default_dtype": live["default_dtype"],
            "amp_enabled": False,
            "deterministic_algorithms": live["deterministic_algorithms"],
            "intra_op_threads": live["intra_op_threads"],
            "inter_op_threads": live["inter_op_threads"],
            "affinity_cpus": list(affinity),
            "cpu_limit": live["cpu_limit"],
            "memory_limit_bytes": live["memory_limit_bytes"],
            "tmpfs_limit_bytes": live["tmpfs_limit_bytes"],
            "meter_implementation": dict(live["meter_implementation"]),
        },
        "recorded_cache_provenance": {
            "status": "recorded-only-not-current-environment-proof",
            "timeout_seconds": int(sandbox["timeout_seconds"]),
            "candidate_image": PINNED_CANDIDATE_IMAGE,
            "candidate_image_id": sandbox["image_id"],
            "engine_version_sha256": sandbox["engine_version_sha256"],
        },
    }


def _validate_authorized_environment(
    environment: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _require_exact_keys(
        environment,
        {"current_live_probe", "recorded_cache_provenance"},
        "authorized environment",
    )
    if dict(environment) != canonical_authorized_environment(config):
        raise ContractError("authorized live environment binding mismatch")


def _validate_comparable_resource_provenance(
    provenance: Mapping[str, Any], *, root: Path = ROOT
) -> None:
    _require_exact_keys(provenance, {"EXP609", "EXP643"}, "comparable provenance")
    expected_roles = {
        "EXP609": "lineage-matched-training-and-evaluation",
        "EXP643": "lineage-matched-width256-calibration",
    }
    for name, role in expected_roles.items():
        record = provenance[name]
        _require_exact_keys(
            record,
            {
                "identity_path",
                "identity_sha256",
                "rate_path",
                "rate_sha256",
                "role",
            },
            f"{name} provenance",
        )
        if (
            record["role"] != role
            or not _is_sha256(record["identity_sha256"])
            or not _is_sha256(record["rate_sha256"])
        ):
            raise ContractError(f"{name} comparable provenance is malformed")
        for path_key, hash_key in (
            ("identity_path", "identity_sha256"),
            ("rate_path", "rate_sha256"),
        ):
            path = (root / record[path_key]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ContractError(f"{name} comparable provenance path is invalid")
            if frozen_run.sha256(path) != record[hash_key]:
                raise ContractError(f"{name} comparable provenance hash mismatch")


def independently_verify_unused_root_output(
    root_id: int,
    output: str,
    *,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    if type(root_id) is not int or root_id <= 0 or not output:
        raise AuthorizationError("root/output identity is invalid")
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = repository_root / output_path
    output_path = output_path.resolve()
    if output_path.exists():
        raise AuthorizationError("authorized output path is already used")
    token = str(root_id)
    occurrences: list[str] = []
    stateful_roots = (
        repository_root / "experiments/runs",
        repository_root / "evals/results",
        repository_root / ".scratch",
        repository_root / "mlruns",
    )
    for stateful in stateful_roots:
        if not stateful.exists():
            continue
        for path in stateful.rglob("*"):
            if token in path.name:
                occurrences.append(str(path.relative_to(repository_root)))
            if path.is_file() and (
                path.name == "run.json"
                or "authorization" in path.name
                or "provenance" in path.name
            ):
                try:
                    if token in path.read_text(encoding="utf-8"):
                        occurrences.append(str(path.relative_to(repository_root)))
                except (OSError, UnicodeDecodeError):
                    continue
    history = subprocess.run(
        [
            "git",
            "log",
            "--all",
            f"-S{token}",
            "--format=%H",
            "--",
            "experiments/runs",
            "evals/results",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    if occurrences or history:
        raise AuthorizationError("authorized root is present in workspace or history")
    evidence = {
        "root": root_id,
        "output": str(output_path),
        "output_absent": True,
        "workspace_occurrences": 0,
        "git_history_occurrences": 0,
        "checked_scopes": [
            "experiments/runs",
            "evals/results",
            ".scratch",
            "mlruns",
        ],
    }
    return evidence | {
        "evidence_sha256": sha256_bytes(canonical_json_bytes(evidence))
    }


def _authorization_expectation_mechanics(
    *,
    config: Mapping[str, Any],
    scaffold_commit: str,
    root: int,
    output: str,
    deterministic_attempt_allowance: int,
    registry: Mapping[str, CanonicalOperationRegistration],
    evidence_root: Path = ROOT,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    if not scaffold_commit or root <= 0 or not output:
        raise ValueError("authorization expectation requires commit/root/output")
    if (
        len(scaffold_commit) != 40
        or any(character not in "0123456789abcdef" for character in scaffold_commit)
        or scaffold_commit == L0_BASE_COMMIT
    ):
        raise ValueError("authorization requires a later 40-hex scaffold commit")
    if config.get("run_id") != RUN_ID:
        raise ContractError("authorization config identity changed")
    cache_verdict = validate_incumbent_cache(config, root=repository_root)
    environment = canonical_authorized_environment(config)
    environment_sha256 = sha256_bytes(canonical_json_bytes(environment))
    comparable_resource_provenance = config["L1"]["budget"][
        "comparable_source_provenance"
    ]
    _validate_comparable_resource_provenance(
        comparable_resource_provenance, root=evidence_root
    )
    workload = workload_binding(
        config,
        cache_verdict,
        deterministic_attempt_allowance=deterministic_attempt_allowance,
        root=repository_root,
    )
    budget_derivation = _resolve_remaining_l1_budget_mechanics(
        config,
        cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
    )
    _validate_budget_derivation_record(
        budget_derivation,
        workload=workload,
        config=config,
        cache_verdict=cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
        environment_sha256=environment_sha256,
    )
    numeric_resource_envelope = budget_derivation["numeric_resource_envelope"]
    _validate_numeric_resource_envelope(numeric_resource_envelope)
    cached_identity = config["L1"]["deployed_path_profile"]["cached_candidate005"]
    recorded = environment["recorded_cache_provenance"]
    if (
        recorded["candidate_image_id"]
        != cached_identity["sandbox_record_expectation"]["image_id"]
        or recorded["engine_version_sha256"]
        != cached_identity["sandbox_record_expectation"]["engine_version_sha256"]
    ):
        raise ContractError("authorization Candidate005 environment identity changed")
    hashes = scaffold_hashes(repository_root)
    expected_workload_keys = {
        "seed_manifest",
        "identity_binding",
        "deterministic_attempt_allowance",
        "apportionment",
        "l1_physical",
        "warmup_manifest_sha256",
        "remaining_l1_manifest_sha256",
        "l2_manifest_sha256",
        "l2_projection_units",
        "l2_batches",
        "runtime_gate",
        "cache_verdict",
    }
    _require_exact_keys(workload, expected_workload_keys, "authorized workload")
    if (
        type(workload["deterministic_attempt_allowance"]) is not int
        or workload["deterministic_attempt_allowance"] < 128
    ):
        raise ContractError("authorization lacks numeric attempt allowance")
    unused = independently_verify_unused_root_output(
        root, output, repository_root=repository_root
    )
    _, warmup_evidence = _load_authorized_warmup_observations_mechanics(
        config,
        cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
    )
    authorized_operation_manifest = remaining_l1_physical_manifest(cache_verdict)
    authorized_operation_manifest_payload = _physical_manifest_payload(
        authorized_operation_manifest
    )
    authorized_operation_descriptors = _derive_operation_descriptors(
        authorized_operation_manifest,
        registry=registry,
        input_manifest_sha256=sha256_bytes(canonical_json_bytes(workload)),
        environment_sha256=environment_sha256,
    )
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "run_id": RUN_ID,
        "stage": "L1",
        "purpose": "resource-only-calibration",
        "parent_l0_commit": L0_BASE_COMMIT,
        "scaffold_commit": scaffold_commit,
        "clean_pushed_scaffold_required": True,
        "root": root,
        "output": output,
        "unused_root_output_proof": unused,
        "hashes": dict(hashes),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "split_sha256_by_width": workload["identity_binding"][
            "split_sha256_by_width"
        ],
        "width256_stream_sha256": workload["identity_binding"][
            "width256_stream_sha256"
        ],
        "workload": dict(workload),
        "workload_sha256": sha256_bytes(canonical_json_bytes(workload)),
        "deterministic_attempt_allowance": workload[
            "deterministic_attempt_allowance"
        ],
        "environment": dict(environment),
        "environment_sha256": environment_sha256,
        "warmup_measurement_evidence": warmup_evidence,
        "warmup_observations_sha256": budget_derivation[
            "warmup_observations_sha256"
        ],
        "authorized_operation_manifest": authorized_operation_manifest_payload,
        "authorized_operation_manifest_sha256": sha256_bytes(
            canonical_json_bytes(authorized_operation_manifest_payload)
        ),
        "authorized_operation_descriptors": [
            dict(item) for item in authorized_operation_descriptors
        ],
        "authorized_operation_descriptors_sha256": sha256_bytes(
            canonical_json_bytes(authorized_operation_descriptors)
        ),
        "budget_derivation": dict(budget_derivation),
        "budget_derivation_sha256": budget_derivation["derivation_sha256"],
        "numeric_resource_envelope": dict(numeric_resource_envelope),
        "comparable_resource_provenance": {
            key: dict(value)
            for key, value in comparable_resource_provenance.items()
        },
        "exact_component_rate_evidence": [
            dict(value)
            for value in config["L1"]["budget"][
                "exact_component_rate_evidence"
            ]
        ],
        "cache_verdict": {
            "valid": cache_verdict.valid,
            "reason": cache_verdict.reason,
            "evidence_sha256": cache_verdict.evidence_sha256,
            "full_1100_required": cache_verdict.full_1100_required,
        },
        "scientific_values_authorized": False,
        "fidelity_reduction_authorized": False,
    }
    return json.loads(canonical_json_bytes(payload))


def authorization_expectation(
    *,
    scaffold_commit: str,
    root: int,
    output: str,
    deterministic_attempt_allowance: int,
) -> dict[str, Any]:
    """Build the production expectation from tracked config and fixed roots."""

    return _authorization_expectation_mechanics(
        config=load_config(CONFIG_PATH),
        scaffold_commit=scaffold_commit,
        root=root,
        output=output,
        deterministic_attempt_allowance=deterministic_attempt_allowance,
        registry=CANONICAL_OPERATION_REGISTRY,
        evidence_root=ROOT,
        repository_root=ROOT,
    )


def _validate_authorization_candidate_mechanics(
    manifest_path: Path | None,
    *,
    config: Mapping[str, Any] | None = None,
    registry: Mapping[str, CanonicalOperationRegistration] = CANONICAL_OPERATION_REGISTRY,
    evidence_root: Path = ROOT,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.is_file():
        raise AuthorizationError("later Judge authorization is absent")
    try:
        raw = manifest_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorizationError("authorization is malformed") from error
    if raw != canonical_json_bytes(payload):
        raise AuthorizationError("authorization is not canonical JSON")
    expected_top = {
        "schema_version",
        "run_id",
        "stage",
        "purpose",
        "parent_l0_commit",
        "scaffold_commit",
        "clean_pushed_scaffold_required",
        "root",
        "output",
        "unused_root_output_proof",
        "hashes",
        "source_checkpoint_sha256",
        "split_sha256_by_width",
        "width256_stream_sha256",
        "workload",
        "workload_sha256",
        "deterministic_attempt_allowance",
        "environment",
        "environment_sha256",
        "warmup_measurement_evidence",
        "warmup_observations_sha256",
        "authorized_operation_manifest",
        "authorized_operation_manifest_sha256",
        "authorized_operation_descriptors",
        "authorized_operation_descriptors_sha256",
        "budget_derivation",
        "budget_derivation_sha256",
        "numeric_resource_envelope",
        "comparable_resource_provenance",
        "exact_component_rate_evidence",
        "cache_verdict",
        "scientific_values_authorized",
        "fidelity_reduction_authorized",
    }
    try:
        _require_exact_keys(payload, expected_top, "authorization")
        if (
            type(payload["root"]) is not int
            or type(payload["deterministic_attempt_allowance"]) is not int
            or not isinstance(payload["output"], str)
            or not isinstance(payload["scaffold_commit"], str)
        ):
            raise ContractError("authorization identifying fields are malformed")
    except ContractError as error:
        raise AuthorizationError(str(error)) from error
    scaffold_git_state = git_read_only_state()
    scaffold_commit = payload["scaffold_commit"]
    if bool(scaffold_git_state.get("dirty")):
        raise AuthorizationError("scaffold Git state is dirty")
    if (
        scaffold_git_state.get("head") != scaffold_commit
        or scaffold_git_state.get("upstream") != scaffold_commit
    ):
        raise AuthorizationError("scaffold commit is not the clean pushed HEAD")
    config = (
        load_config(repository_root / CONFIG_PATH.relative_to(ROOT))
        if config is None
        else config
    )
    try:
        expected = _authorization_expectation_mechanics(
            config=config,
            scaffold_commit=payload["scaffold_commit"],
            root=payload["root"],
            output=payload["output"],
            deterministic_attempt_allowance=payload[
                "deterministic_attempt_allowance"
            ],
            registry=registry,
            evidence_root=evidence_root,
            repository_root=repository_root,
        )
    except (ContractError, AuthorizationError, ValueError) as error:
        raise AuthorizationError(str(error)) from error
    if payload != expected:
        raise AuthorizationError("authorization does not match canonical recomputation")
    return payload


def _fixed_repository_path(path: Path | None, *, label: str) -> Path:
    if path is None:
        raise AuthorizationError(f"{label} is absent")
    if not isinstance(path, Path):
        raise AuthorizationError(f"{label} path is malformed")
    candidate = path if path.is_absolute() else ROOT / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise AuthorizationError(f"{label} is outside the fixed repository root")
    return resolved


def validate_authorization_candidate(manifest_path: Path | None) -> dict[str, Any]:
    """Validate one candidate against tracked config and fixed repository evidence."""

    fixed_manifest = _fixed_repository_path(
        manifest_path,
        label="later Judge authorization",
    )
    return _validate_authorization_candidate_mechanics(
        fixed_manifest,
        config=load_config(CONFIG_PATH),
        registry=CANONICAL_OPERATION_REGISTRY,
        evidence_root=ROOT,
        repository_root=ROOT,
    )


def _load_authorized_l1_observations(
    record_descriptors: Sequence[Mapping[str, Any]],
    *,
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
    cache_verdict: "CacheVerdict",
    registry: Mapping[str, CanonicalOperationRegistration],
    evidence_root: Path,
) -> tuple[ResourceObservation, ...]:
    """Reload and authenticate every warmup and later L1 measurement record."""

    if not isinstance(record_descriptors, (list, tuple)):
        raise ContractError("measurement record descriptors are not a sequence")
    if any(not isinstance(item, Mapping) for item in record_descriptors):
        raise ContractError("measurement record descriptor is not a mapping")
    environment_sha256 = authorization["environment_sha256"]
    if environment_sha256 != sha256_bytes(
        canonical_json_bytes(canonical_authorized_environment(config))
    ):
        raise ContractError("measurement record environment is not live/canonical")
    warmup_manifest = warmup_l1_physical_manifest()
    remaining_manifest = remaining_l1_physical_manifest(cache_verdict)
    canonical_manifest = (*warmup_manifest, *remaining_manifest)
    canonical_keys = [item.key for item in canonical_manifest]
    if [item.get("physical_key") for item in record_descriptors] != canonical_keys:
        raise ContractError("measurement record descriptors are not exactly canonical")
    if len(
        {
            (item.get("record_path"), item.get("record_file_sha256"), item.get("record_sha256"))
            for item in record_descriptors
        }
    ) != len(record_descriptors):
        raise ContractError("measurement records are not one-to-one")

    warmup_evidence = authorization["warmup_measurement_evidence"]
    warmup_descriptors = warmup_evidence["record_descriptors"]
    if (
        not isinstance(warmup_descriptors, list)
        or list(record_descriptors[: len(warmup_manifest)])
        != warmup_descriptors
        or warmup_evidence["record_descriptors_sha256"]
        != sha256_bytes(canonical_json_bytes(warmup_descriptors))
    ):
        raise ContractError("warmup record descriptor binding changed")
    prior_authorization, prior_sha256 = _load_prior_measurement_authorization(
        config,
        cache_verdict=cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
        environment_sha256=environment_sha256,
    )
    if prior_sha256 != warmup_evidence["authorization_sha256"]:
        raise ContractError("warmup record prior authorization changed")
    prior_manifest = tuple(
        _physical_unit_from_payload(item)
        for item in prior_authorization["physical_manifest"]
    )
    prior_operation_descriptors = _validate_operation_descriptors(
        prior_authorization["operation_descriptors"],
        prior_manifest,
        registry=registry,
        environment_sha256=environment_sha256,
        expected_input_manifest_sha256=prior_authorization[
            "input_manifest_sha256"
        ],
    )

    remaining_payload = _physical_manifest_payload(remaining_manifest)
    if (
        authorization["authorized_operation_manifest"] != remaining_payload
        or authorization["authorized_operation_manifest_sha256"]
        != sha256_bytes(canonical_json_bytes(remaining_payload))
        or authorization["authorized_operation_descriptors_sha256"]
        != sha256_bytes(
            canonical_json_bytes(authorization["authorized_operation_descriptors"])
        )
    ):
        raise ContractError("later measurement operation manifest changed")
    later_operation_descriptors = _validate_operation_descriptors(
        authorization["authorized_operation_descriptors"],
        remaining_manifest,
        registry=registry,
        environment_sha256=environment_sha256,
        expected_input_manifest_sha256=authorization["workload_sha256"],
    )
    authorization_sha256 = sha256_bytes(canonical_json_bytes(authorization))
    observations: list[ResourceObservation] = []
    warmup_count = len(warmup_manifest)
    for index, descriptor in enumerate(record_descriptors):
        is_warmup = index < warmup_count
        observations.append(
            _load_measurement_descriptor(
                descriptor,
                evidence_root=evidence_root,
                environment_sha256=environment_sha256,
                prior_authorization_sha256=(
                    prior_sha256 if is_warmup else authorization_sha256
                ),
                authorized_operation_descriptors=(
                    prior_operation_descriptors
                    if is_warmup
                    else later_operation_descriptors
                ),
            )
        )
    validate_resource_observations(
        observations,
        canonical_manifest,
        require_path_backed=True,
        expected_environment_sha256=environment_sha256,
    )
    return tuple(observations)


def _post_l1_dynamic_budget_derivation(
    observations: Sequence[ResourceObservation],
    record_descriptors: Sequence[Mapping[str, Any]],
    *,
    authorization: Mapping[str, Any],
    cache_verdict: "CacheVerdict",
) -> dict[str, Any]:
    observed_manifest = (
        *warmup_l1_physical_manifest(),
        *remaining_l1_physical_manifest(cache_verdict),
    )
    projection_manifest = l2_physical_manifest(cache_verdict)
    budget = _resolve_dynamic_budget_from_observations(
        observations,
        observed_manifest,
        projection_manifest,
        projection_stage="L2",
        cache_verdict=cache_verdict,
    )
    descriptor_payload = [dict(item) for item in record_descriptors]
    observation_payload = [
        _resource_observation_payload(item) for item in observations
    ]
    projection_payload = _physical_manifest_payload(projection_manifest)
    payload = {
        "stage": "post-L1-to-L2-dynamic-budget",
        "authorization_sha256": sha256_bytes(canonical_json_bytes(authorization)),
        "environment_sha256": authorization["environment_sha256"],
        "cache_evidence_sha256": cache_verdict.evidence_sha256,
        "projection_manifest": projection_payload,
        "projection_manifest_sha256": sha256_bytes(
            canonical_json_bytes(projection_payload)
        ),
        "record_descriptors": descriptor_payload,
        "record_descriptors_sha256": sha256_bytes(
            canonical_json_bytes(descriptor_payload)
        ),
        "record_sha256s": [item["record_sha256"] for item in descriptor_payload],
        "observations": observation_payload,
        "observations_sha256": sha256_bytes(
            canonical_json_bytes(observation_payload)
        ),
        "maximum_matching_mean_rates": budget["maximum_matching_mean_rates"],
        "expected": budget["expected"],
        "maximum_observed_rss_bytes": max(
            item.peak_simultaneous_tree_rss_bytes for item in observations
        ),
        "multipliers": {"soft": 1.25, "hard": 1.75, "rss_guard": 1.25},
        "soft_budget": budget["soft_budget"],
        "hard_runaway_ceiling": budget["hard_runaway_ceiling"],
        "peak_simultaneous_tree_rss_guard_bytes": budget[
            "peak_simultaneous_tree_rss_guard_bytes"
        ],
        "observed_manifest_sha256": budget["observed_manifest_sha256"],
        "shared_generation_charged_once": True,
        "model_execution_shared": False,
        "wall_cpu_conflated": False,
        "summed_serial_peak_rss": False,
        "fixed_short_wall_cap_used": False,
        "fidelity_reduced": False,
    }
    payload["derivation_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def resolve_dynamic_budget(
    authorization_path: Path | None,
    record_descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve the production L2 envelope from fixed-root measurement records."""

    authorization = validate_authorization_candidate(authorization_path)
    config = load_config(CONFIG_PATH)
    cache_verdict = validate_incumbent_cache(config, root=ROOT)
    observations = _load_authorized_l1_observations(
        record_descriptors,
        authorization=authorization,
        config=config,
        cache_verdict=cache_verdict,
        registry=CANONICAL_OPERATION_REGISTRY,
        evidence_root=ROOT,
    )
    return _post_l1_dynamic_budget_derivation(
        observations,
        record_descriptors,
        authorization=authorization,
        cache_verdict=cache_verdict,
    )


@dataclass(slots=True)
class SideEffectProbe:
    output_allocations: int = 0
    mlflow_starts: int = 0
    model_builds: int = 0
    optimizer_builds: int = 0
    candidate_runs: int = 0
    workload_runs: int = 0

    def untouched(self) -> bool:
        return not any(
            (
                self.output_allocations,
                self.mlflow_starts,
                self.model_builds,
                self.optimizer_builds,
                self.candidate_runs,
                self.workload_runs,
            )
        )


def inert_entrypoint(
    manifest_path: Path | None,
    *,
    side_effects: SideEffectProbe | None = None,
) -> None:
    """Validate candidate bytes, then still reject: this scaffold authorizes no launch."""

    validate_authorization_candidate(manifest_path)
    if side_effects is not None and not side_effects.untouched():
        raise ContractError("side effect occurred before inert boundary")
    raise AuthorizationError(
        "EXP-658 L1 is an inert scaffold; a later committed Judge execution package is required"
    )


def _validate_resource_only_receipt_mechanics(
    payload: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
    registry: Mapping[str, CanonicalOperationRegistration],
    root: Path,
    evidence_root: Path,
) -> None:
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ContractError("resource receipt authorization binding is invalid")
    live_cache_verdict = validate_incumbent_cache(config, root=root)
    _validate_authorized_environment(authorization["environment"], config)
    if authorization["environment_sha256"] != sha256_bytes(
        canonical_json_bytes(authorization["environment"])
    ):
        raise ContractError("resource receipt authorization environment changed")
    canonical_workload = workload_binding(
        config,
        live_cache_verdict,
        deterministic_attempt_allowance=authorization[
            "deterministic_attempt_allowance"
        ],
        root=root,
    )
    canonical_budget = _resolve_remaining_l1_budget_mechanics(
        config,
        live_cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
    )
    _, canonical_warmup_evidence = _load_authorized_warmup_observations_mechanics(
        config,
        live_cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
    )
    if (
        authorization["workload"]
        != json.loads(canonical_json_bytes(canonical_workload))
        or authorization["workload_sha256"]
        != sha256_bytes(canonical_json_bytes(canonical_workload))
        or authorization["budget_derivation"] != canonical_budget
        or authorization["budget_derivation_sha256"]
        != canonical_budget["derivation_sha256"]
        or authorization["numeric_resource_envelope"]
        != canonical_budget["numeric_resource_envelope"]
        or authorization["warmup_measurement_evidence"]
        != canonical_warmup_evidence
    ):
        raise ContractError("resource receipt authorization is not canonical")
    canonical_manifest = (
        *warmup_l1_physical_manifest(),
        *remaining_l1_physical_manifest(live_cache_verdict),
    )
    canonical_manifest_payload = _physical_manifest_payload(canonical_manifest)
    expected_bindings = {
        "authorization_sha256": sha256_bytes(canonical_json_bytes(authorization)),
        "scaffold_hashes_sha256": sha256_bytes(
            canonical_json_bytes(authorization["hashes"])
        ),
        "workload_sha256": authorization["workload_sha256"],
        "budget_derivation_sha256": authorization["budget_derivation_sha256"],
        "cache_evidence_sha256": live_cache_verdict.evidence_sha256,
        "meter_implementation_sha256": meter_implementation_binding(root)["sha256"],
        "environment_sha256": authorization["environment_sha256"],
        "warmup_observations_sha256": authorization[
            "warmup_observations_sha256"
        ],
        "warmup_measurement_evidence_sha256": sha256_bytes(
            canonical_json_bytes(authorization["warmup_measurement_evidence"])
        ),
    }
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "bindings",
            "boundary",
            "identities",
            "counts",
            "hashes",
            "shapes",
            "integrity",
            "determinism",
            "resources",
            "cache_verdict",
            "command",
            "budgets",
            "post_l1_budget_derivation",
        },
        "resource receipt",
    )
    if payload["schema_version"] != RESOURCE_RECEIPT_SCHEMA:
        raise ContractError("resource receipt schema version changed")
    _require_exact_keys(
        payload["bindings"], set(expected_bindings), "resource receipt bindings"
    )
    if payload["bindings"] != expected_bindings:
        raise ContractError("resource receipt authorization/workload/budget binding changed")
    boundary = payload["boundary"]
    _require_exact_keys(
        boundary,
        {
            "stage",
            "resource_only",
            "scientific_values_present",
            "fidelity_reduced",
        },
        "resource receipt boundary",
    )
    if boundary != {
        "stage": "L1",
        "resource_only": True,
        "scientific_values_present": False,
        "fidelity_reduced": False,
    }:
        raise ContractError("resource receipt boundary is not resource-only")

    identities = payload["identities"]
    _require_exact_keys(
        identities,
        {
            "split_sha256_by_width",
            "stream_manifest_sha256",
            "width256_stream_sha256",
        },
        "resource receipt identities",
    )
    expected_identities = authorization["workload"]["identity_binding"]
    if identities != {
        "split_sha256_by_width": expected_identities["split_sha256_by_width"],
        "stream_manifest_sha256": expected_identities["stream_manifest_sha256"],
        "width256_stream_sha256": expected_identities["width256_stream_sha256"],
    }:
        raise ContractError("resource receipt identity hashes are invalid")

    counts = payload["counts"]
    _require_exact_keys(
        counts,
        {
            "physical_manifest_sha256",
            "physical_manifest",
            "completed_physical_units",
            "completed_components",
        },
        "resource receipt counts",
    )
    if not _is_sha256(counts["physical_manifest_sha256"]) or not isinstance(
        counts["physical_manifest"], list
    ) or any(
        type(counts[key]) is not int or counts[key] < 0
        for key in ("completed_physical_units", "completed_components")
    ):
        raise ContractError("resource receipt counts are invalid")
    manifest_keys = {
        "key",
        "rate_key",
        "phase",
        "repeat",
        "source",
        "units",
        "width",
        "arm",
        "eligible_for_rate",
    }
    physical_manifest: list[PhysicalUnit] = []
    for item in counts["physical_manifest"]:
        if not isinstance(item, Mapping):
            raise ContractError("resource receipt physical manifest item is invalid")
        _require_exact_keys(item, manifest_keys, "resource receipt physical manifest")
        if type(item["eligible_for_rate"]) is not bool:
            raise ContractError("resource receipt physical rate eligibility is invalid")
        physical_manifest.append(
            PhysicalUnit(
                key=item["key"],
                rate_key=item["rate_key"],
                phase=item["phase"],
                repeat=item["repeat"],
                source=item["source"],
                units=item["units"],
                width=item["width"],
                arm=item["arm"],
                eligible_for_rate=item["eligible_for_rate"],
            )
        )
    _validate_physical_manifest(physical_manifest)
    if _physical_manifest_payload(physical_manifest) != canonical_manifest_payload:
        raise ContractError("resource receipt physical manifest is not exactly canonical")
    if counts["physical_manifest_sha256"] != sha256_bytes(
        canonical_json_bytes(_physical_manifest_payload(physical_manifest))
    ) or counts["completed_components"] != len(physical_manifest) or counts[
        "completed_physical_units"
    ] != sum(item.units for item in physical_manifest):
        raise ContractError("resource receipt physical manifest hash mismatch")

    hashes = payload["hashes"]
    _require_exact_keys(
        hashes,
        {
            "scaffold_binding_sha256",
            "control_checkpoint_sha256",
            "challenger_checkpoint_sha256",
        },
        "resource receipt hashes",
    )
    if any(not _is_sha256(value) for value in hashes.values()) or hashes[
        "scaffold_binding_sha256"
    ] != expected_bindings["scaffold_hashes_sha256"]:
        raise ContractError("resource receipt artifact hash is invalid")

    shapes = payload["shapes"]
    _require_exact_keys(
        shapes,
        {"control_parameters", "challenger_parameters", "state_key_count"},
        "resource receipt shapes",
    )
    if (
        shapes["control_parameters"] != 103522
        or shapes["challenger_parameters"] != 103522
        or type(shapes["state_key_count"]) is not int
        or shapes["state_key_count"] <= 0
    ):
        raise ContractError("resource receipt shapes changed")

    integrity = payload["integrity"]
    _require_exact_keys(
        integrity,
        {
            "frozen_hashes_exact",
            "source_parity_exact",
            "control_reload_exact",
            "challenger_reload_exact",
            "profile_uses_exact_reloaded_challenger",
        },
        "resource receipt integrity",
    )
    if any(type(value) is not bool for value in integrity.values()):
        raise ContractError("resource receipt integrity values must be booleans")

    determinism = payload["determinism"]
    _require_exact_keys(
        determinism,
        {"width32_bit_exact", "width64_bit_exact"},
        "resource receipt determinism",
    )
    if any(type(value) is not bool for value in determinism.values()):
        raise ContractError("resource receipt determinism values must be booleans")

    resources = payload["resources"]
    _require_exact_keys(
        resources,
        {"record_descriptors", "wall_cpu_distinct", "rss_scope"},
        "resource receipt resources",
    )
    if (
        resources["wall_cpu_distinct"] is not True
        or resources["rss_scope"] != "cgroup-memory.peak"
    ):
        raise ContractError("resource receipt meter boundary changed")
    if not isinstance(resources["record_descriptors"], list):
        raise ContractError("resource receipt record descriptors must be a list")
    parsed_observations = _load_authorized_l1_observations(
        resources["record_descriptors"],
        authorization=authorization,
        config=config,
        cache_verdict=live_cache_verdict,
        registry=registry,
        evidence_root=evidence_root,
    )
    canonical_post_l1_derivation = _post_l1_dynamic_budget_derivation(
        parsed_observations,
        resources["record_descriptors"],
        authorization=authorization,
        cache_verdict=live_cache_verdict,
    )
    if payload["post_l1_budget_derivation"] != canonical_post_l1_derivation:
        raise ContractError("post-L1 dynamic budget derivation is not canonical")
    warmup_keys = {item.key for item in warmup_l1_physical_manifest()}
    warmup_observations = tuple(
        item for item in parsed_observations if item.physical_key in warmup_keys
    )
    warmup_sha256 = sha256_bytes(
        canonical_json_bytes(
            [_resource_observation_payload(item) for item in warmup_observations]
        )
    )
    prior_warmup_authorization_sha256 = authorization[
        "warmup_measurement_evidence"
    ]["authorization_sha256"]
    authorization_sha256 = expected_bindings["authorization_sha256"]
    if (
        warmup_sha256 != authorization["warmup_observations_sha256"]
        or warmup_sha256
        != authorization["budget_derivation"]["warmup_observations_sha256"]
        or any(
            item.prior_authorization_sha256
            != prior_warmup_authorization_sha256
            for item in warmup_observations
        )
        or any(
            item.prior_authorization_sha256 != authorization_sha256
            for item in parsed_observations
            if item.physical_key not in warmup_keys
        )
    ):
        raise ContractError("resource receipt warmup/environment binding changed")

    cache = payload["cache_verdict"]
    _require_exact_keys(
        cache,
        {
            "valid",
            "reason",
            "evidence_sha256",
            "full_1100_required",
        },
        "resource receipt cache verdict",
    )
    if (
        type(cache["valid"]) is not bool
        or type(cache["full_1100_required"]) is not bool
        or cache["full_1100_required"] is cache["valid"]
        or not _is_sha256(cache["evidence_sha256"])
        or not isinstance(cache["reason"], str)
        or cache["reason"] not in {
            "exact-path-backed-cache-match",
            "candidate005-record-parse-failure",
            "sandbox-record-content-mismatch",
            "environment-record-content-mismatch",
            "candidate005-image-not-pinned",
            "engine-version-output-hash-mismatch",
            "incumbent-wall-mismatch",
            *{
                f"{name}-{suffix}"
                for name in (
                    "submission_weights",
                    "submission_manifest",
                    "submission_model",
                    "sandbox_record",
                    "sandbox_runner",
                    "dockerfile",
                    "official_generator",
                    "official_primes",
                    "environment_record",
                )
                for suffix in ("missing", "hash-mismatch")
            },
        }
    ):
        raise ContractError("resource receipt cache verdict is invalid")
    expected_cache = {
        "valid": live_cache_verdict.valid,
        "reason": live_cache_verdict.reason,
        "evidence_sha256": live_cache_verdict.evidence_sha256,
        "full_1100_required": live_cache_verdict.full_1100_required,
    }
    if cache != expected_cache or authorization["cache_verdict"] != expected_cache:
        raise ContractError("resource receipt cache is not the live path-backed verdict")

    command = payload["command"]
    _require_exact_keys(
        command,
        {"candidate005_full_1100"},
        "resource receipt command",
    )
    candidate_command = command["candidate005_full_1100"]
    if candidate_command is not None:
        if not isinstance(candidate_command, list) or "--output-dir" not in candidate_command:
            raise ContractError("resource receipt Candidate005 command is malformed")
        output_index = candidate_command.index("--output-dir") + 1
        if output_index >= len(candidate_command) or tuple(candidate_command) != candidate005_full_1100_rerun_command(
            Path(candidate_command[output_index])
        ):
            raise ContractError("resource receipt Candidate005 command is not pinned")
    if cache["full_1100_required"] != (candidate_command is not None):
        raise ContractError("resource receipt command/cache relationship changed")

    _validate_numeric_resource_envelope(payload["budgets"])
    if payload["budgets"] != authorization["numeric_resource_envelope"]:
        raise ContractError("resource receipt budget derivation binding changed")


def validate_resource_only_receipt(
    receipt_path: Path | None,
    *,
    authorization_path: Path | None,
) -> None:
    """Validate a fixed-root canonical receipt and reload every record it names."""

    fixed_receipt = _fixed_repository_path(
        receipt_path,
        label="resource receipt",
    )
    if not fixed_receipt.is_file():
        raise ContractError("resource receipt is absent from fixed evidence root")
    try:
        raw = fixed_receipt.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("resource receipt is malformed") from error
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise ContractError("resource receipt is not canonical JSON")
    authorization = validate_authorization_candidate(authorization_path)
    _validate_resource_only_receipt_mechanics(
        payload,
        authorization=authorization,
        config=load_config(CONFIG_PATH),
        registry=CANONICAL_OPERATION_REGISTRY,
        root=ROOT,
        evidence_root=ROOT,
    )


def git_read_only_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "upstream": run("rev-parse", "@{upstream}"),
        "dirty": bool(run("status", "--porcelain")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=None)
    arguments = parser.parse_args()
    # No valid expectation is constructible in this task: numeric budgets,
    # unused root/output, committed scaffold, and Judge authorization are absent.
    if arguments.authorization is None:
        raise AuthorizationError(
            "EXP-658 L1 execution is prohibited: later Judge authorization is absent"
        )
    raise AuthorizationError(
        "EXP-658 L1 execution is prohibited: the repaired scaffold is inert"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuthorizationError, ContractError) as error:
        raise SystemExit(str(error)) from error
