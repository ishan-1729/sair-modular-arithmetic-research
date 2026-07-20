"""Fail-closed implementation and resource calibration for EXP-648.

This package intentionally exposes only two roots: a scaffold-test root and the
non-scientific calibration root.  Corpus construction and every scientific
evaluation mode remain unavailable until a later Judge authorizes a separately
frozen runner.
"""

from __future__ import annotations

# Start before third-party imports so launcher/import/setup cost is explicit.
_MODULE_IMPORT_WALL_STARTED = __import__("time").perf_counter()
_MODULE_IMPORT_PROCESS_CPU_STARTED = __import__("time").process_time()

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import mlflow
import numpy as np
import psutil
import scipy
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from scipy.stats import beta
from sympy import nextprime
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


RUN_ID = "exp-648-structured-trajectory-consistency"
TEST_ONLY_ROOT = 648_900_001
CALIBRATION_ROOT = 20_261_151
CORPUS_ROOT = 20_261_152
SCREEN_ROOTS = frozenset((20_261_153, 20_261_154))
CONFIRMATION_ROOTS = frozenset((20_261_155, 20_261_156, 20_261_157, 20_261_158))
EXP649_FORMAL_ROOT = 20_261_149
FORBIDDEN_NUMERIC_ROOTS = frozenset(
    (CORPUS_ROOT, EXP649_FORMAL_ROOT, *SCREEN_ROOTS, *CONFIRMATION_ROOTS)
)
FORBIDDEN_SURFACE_MODES = frozenset(
    (
        "corpus",
        "screen",
        "confirmation",
        "checked",
        "checked-public",
        "independent",
        "official",
        "official-private",
        "private",
        "formal",
    )
)
ALLOWED_MODE_ROOTS = {
    "test": frozenset((TEST_ONLY_ROOT,)),
    "calibration": frozenset((CALIBRATION_ROOT,)),
}
DEPENDENCY_SOURCE_PATHS = {
    "cellular_model": "experiments/exp_220_width_independent_cellular/model.py",
    "exp300_data_evaluator": "experiments/exp_300_width64_bit_tensor_transfer/run.py",
}
REQUIRED_SCIENTIFIC_INTEGRITY_KEYS = frozenset(
    (
        "source_checkpoint_hash_valid",
        "identical_initialization_valid",
        "parameter_count_valid",
        "paired_data_order_valid",
        "seed_roles_valid",
        "checkpoint_hashes_valid",
        "evaluator_hash_valid",
        "dependency_manifest_valid",
        "matched_work_valid",
        "control_exact_repeat_deterministic",
        "challenger_exact_repeat_deterministic",
    )
)
REQUIRED_CALIBRATION_INTEGRITY_BOOLEAN_KEYS = frozenset(
    (
        "source_checkpoint_hash_valid",
        "identical_initialization_valid",
        "parameter_count_valid",
        "optimizer_states_fresh",
        "frozen_runtime_valid",
        "dependency_manifest_valid",
        "corpus_semantic_verification_valid",
        "matched_arm_work_valid",
        "ordered_batches_valid",
        "same_evaluation_inputs_valid",
        "state_corruption_absent",
        "source_checkpoint_unchanged",
    )
)
PRIOR_SUPERSEDED_CALIBRATION = {
    "evidence_sha256": "f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4",
    "wall_seconds": 1478.023299573,
    "aggregate_cpu_seconds": 6085.320000000001,
    "eligible_for_projection_rates": False,
    "classified_as_reused_repeat": False,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    return canonical_json(value).encode("utf-8")


def dependency_source_manifest() -> dict[str, Any]:
    sources = {
        name: {"path": relative, "sha256": sha256(ROOT / relative)}
        for name, relative in sorted(DEPENDENCY_SOURCE_PATHS.items())
    }
    payload = {
        "schema_version": "exp648-executable-dependency-manifest-v1",
        "sources": sources,
    }
    return {**payload, "manifest_sha256": sha256_text(canonical_json(payload))}


def verify_dependency_source_manifest(manifest: dict[str, Any]) -> None:
    expected = dependency_source_manifest()
    if manifest != expected:
        raise RuntimeError("EXP-648 executable dependency source manifest mismatch")
    for record in manifest["sources"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError("EXP-648 executable dependency source hash mismatch")


def load_verified_dependency_module(
    module_name: str, dependency_name: str, manifest: dict[str, Any]
):
    verify_dependency_source_manifest(manifest)
    if dependency_name not in DEPENDENCY_SOURCE_PATHS:
        raise RuntimeError(f"unknown EXP-648 dependency {dependency_name!r}")
    record = manifest["sources"][dependency_name]
    if record["path"] != DEPENDENCY_SOURCE_PATHS[dependency_name]:
        raise RuntimeError("EXP-648 dependency path changed")
    return load_module(module_name, ROOT / record["path"])


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def process_cpu_seconds(process: psutil.Process | None = None) -> float:
    times = (process or psutil.Process()).cpu_times()
    return float(times.user + times.system)


def process_peak_rss_bytes(process: psutil.Process | None = None) -> int:
    current = (process or psutil.Process()).memory_info().rss
    # Linux ru_maxrss is KiB and is a true process high-water mark.
    high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return max(int(current), high_water)


def state_dict_sha256(model_or_state: nn.Module | dict[str, torch.Tensor]) -> str:
    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else model_or_state
    )
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(canonical_json(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def optimizer_is_fresh(optimizer: torch.optim.Optimizer) -> bool:
    return len(optimizer.state) == 0


def authorize_root(mode: str, root_seed: int) -> int:
    normalized = str(mode).strip().lower()
    seed = int(root_seed)
    if normalized in FORBIDDEN_SURFACE_MODES:
        raise PermissionError(f"EXP-648 mode {normalized!r} is prohibited")
    allowed = ALLOWED_MODE_ROOTS.get(normalized)
    if allowed is None:
        raise PermissionError(f"EXP-648 mode {normalized!r} is not implemented")
    if seed not in allowed:
        if seed in FORBIDDEN_NUMERIC_ROOTS:
            raise PermissionError(f"EXP-648 root {seed} is frozen and prohibited")
        raise PermissionError(
            f"EXP-648 mode {normalized!r} permits only {sorted(allowed)}, not {seed}"
        )
    return seed


def derive_seed(root_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"exp648:{int(root_seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def scientific_protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config)
    for key in ("status", "phase", "resource_protocol", "authorization"):
        payload.pop(key, None)
    payload.get("corpus", {}).pop("resolved_corpus_mining_ceiling", None)
    return payload


def resource_protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config["resource_protocol"])
    payload.pop("status", None)
    payload.pop("resolved", None)
    projections = payload.get("projections", {})
    projections.pop("resolved", None)
    return payload


def scientific_protocol_sha256(config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(scientific_protocol_payload(config)))


def resource_protocol_sha256(config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(resource_protocol_payload(config)))


def workload_payload(config: dict[str, Any]) -> dict[str, Any]:
    calibration = config["resource_protocol"]["calibration"]
    return {
        "root": int(calibration["root"]),
        "warmups": int(calibration["warmups"]),
        "timed_repeats": int(calibration["timed_repeats"]),
        "representative_corpus_mining": calibration["representative_corpus_mining"],
        "startup_and_model_loading": calibration["startup_and_model_loading"],
        "k8_updates_per_arm": int(calibration["k8_updates_per_arm"]),
        "checkpoint_writes_per_arm": int(calibration["checkpoint_writes_per_arm"]),
        "transition_cases_per_arm": {
            str(key): int(value)
            for key, value in calibration["transition_cases_per_arm"].items()
        },
        "full_rollout_cases_per_arm": {
            str(key): int(value)
            for key, value in calibration["full_rollout_cases_per_arm"].items()
        },
        "hash_and_serialization_work": calibration["hash_and_serialization_work"],
        "window_length": int(config["corpus"]["window_length"]),
        "batch_size": int(config["training"]["batch_size"]),
        "torch_threads": int(calibration["torch_threads"]),
        "torch_interop_threads": int(calibration["torch_interop_threads"]),
        "complete_units": bool(calibration["each_warmup_and_repeat_is_complete"]),
    }


def workload_sha256(config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(workload_payload(config)))


def validate_frozen_config(config: dict[str, Any]) -> None:
    if config.get("run_id") != RUN_ID:
        raise RuntimeError("EXP-648 run id changed")
    if config["source_checkpoint_sha256"] != (
        "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
    ):
        raise RuntimeError("EXP-648 source checkpoint pin changed")
    if config["source_model"] != {
        "hidden": 92,
        "cycles": 2,
        "parameter_count": 99_824,
        "inference_architecture_changes": False,
        "inference_semantics_changes": False,
        "parameter_count_changes": False,
    }:
        raise RuntimeError("EXP-648 source model contract changed")
    configured_dependencies = config.get("executable_dependency_sources")
    if configured_dependencies != dependency_source_manifest():
        raise RuntimeError("EXP-648 executable dependency manifest changed")
    integrity_protocol = config.get("scientific_integrity_protocol", {})
    if integrity_protocol.get("schema") != "exact-required-key-set-no-missing-no-extra":
        raise RuntimeError("EXP-648 scientific integrity schema changed")
    if set(integrity_protocol.get("required_keys", ())) != REQUIRED_SCIENTIFIC_INTEGRITY_KEYS:
        raise RuntimeError("EXP-648 scientific integrity required keys changed")
    if integrity_protocol.get("value_contract") != "every-required-value-is-exact-bool-true":
        raise RuntimeError("EXP-648 scientific integrity value contract changed")
    seeds = config["seed_roles"]
    expected = {
        "calibration_root": CALIBRATION_ROOT,
        "corpus_root": CORPUS_ROOT,
        "screen_training_root": 20_261_153,
        "screen_evaluation_root": 20_261_154,
        "confirmation_1_training_root": 20_261_155,
        "confirmation_1_evaluation_root": 20_261_156,
        "confirmation_2_training_root": 20_261_157,
        "confirmation_2_evaluation_root": 20_261_158,
        "scaffold_test_only_root": TEST_ONLY_ROOT,
    }
    observed = {name: int(seeds[name]["seed"]) for name in expected}
    if observed != expected:
        raise RuntimeError("EXP-648 seed role mapping changed")
    if int(config["resource_protocol"]["calibration"]["root"]) != CALIBRATION_ROOT:
        raise RuntimeError("EXP-648 calibration root changed")
    if config["corpus"]["width"] != 256 or config["corpus"]["window_length"] != 8:
        raise RuntimeError("EXP-648 corpus geometry changed")
    if config["corpus"]["matched_pairs"] != 512:
        raise RuntimeError("EXP-648 matched corpus count changed")
    if config["corpus"]["matching"]["fallback_or_relaxation"] != "prohibited":
        raise RuntimeError("EXP-648 matching relaxation is prohibited")
    if config["corpus"]["synthetic_state_corruption"] != "prohibited":
        raise RuntimeError("EXP-648 state corruption is prohibited")
    if config["corpus"]["state_bit_flips_or_injections"] != "prohibited":
        raise RuntimeError("EXP-648 state corruption is prohibited")
    if config["matched_arms"]["control"]["endpoint_loss_weight"] != 0.0:
        raise RuntimeError("EXP-648 control coefficient changed")
    if config["matched_arms"]["challenger"]["endpoint_loss_weight"] != 0.1:
        raise RuntimeError("EXP-648 challenger coefficient changed")
    training = config["training"]
    required_training = {
        "batch_size": 16,
        "failure_windows_per_batch": 8,
        "matched_success_windows_per_batch": 8,
        "autonomous_unroll_steps": 8,
        "torch_threads": 4,
        "torch_interop_threads": 1,
        "deterministic_algorithms": True,
    }
    for key, value in required_training.items():
        if training[key] != value:
            raise RuntimeError(f"EXP-648 frozen training field changed: {key}")
    calibration = config["resource_protocol"]["calibration"]
    if int(calibration["warmups"]) != 1 or int(calibration["timed_repeats"]) != 3:
        raise RuntimeError("EXP-648 calibration repetition count changed")
    if int(calibration["representative_corpus_mining"]["matched_pairs"]) != 64:
        raise RuntimeError("EXP-648 calibration mining count changed")
    if int(calibration["k8_updates_per_arm"]) != 20:
        raise RuntimeError("EXP-648 calibration update count changed")
    if {int(k): int(v) for k, v in calibration["transition_cases_per_arm"].items()} != {
        128: 200,
        256: 200,
    }:
        raise RuntimeError("EXP-648 calibration transition workload changed")
    if {int(k): int(v) for k, v in calibration["full_rollout_cases_per_arm"].items()} != {
        64: 20,
        128: 20,
        256: 20,
    }:
        raise RuntimeError("EXP-648 calibration rollout workload changed")
    if bool(config["resource_protocol"]["fidelity_may_be_reduced_to_fit_budget"]):
        raise RuntimeError("EXP-648 fidelity reduction is prohibited")
    if bool(config["resource_protocol"]["counts_or_work_may_be_reduced_to_fit_budget"]):
        raise RuntimeError("EXP-648 work reduction is prohibited")


def configure_frozen_runtime(config: dict[str, Any]) -> dict[str, Any]:
    threads = int(config["training"]["torch_threads"])
    interop = int(config["training"]["torch_interop_threads"])
    deterministic = bool(config["training"]["deterministic_algorithms"])
    torch.set_num_threads(threads)
    current_interop = torch.get_num_interop_threads()
    if current_interop != interop:
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError as error:
            if torch.get_num_interop_threads() != interop:
                raise RuntimeError("cannot establish frozen interop thread count") from error
    torch.use_deterministic_algorithms(deterministic)
    observed = {
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    expected = {
        "torch_threads": threads,
        "torch_interop_threads": interop,
        "deterministic_algorithms": deterministic,
    }
    if observed != expected:
        raise RuntimeError(f"frozen runtime mismatch: {observed!r} != {expected!r}")
    return observed


def load_verified_source_model(
    config: dict[str, Any], *, mode: str, root_seed: int
) -> nn.Module:
    """Load the frozen model only after a mode/root authorization check."""

    authorize_root(mode, root_seed)
    validate_frozen_config(config)
    source = (ROOT / config["source_checkpoint"]).resolve()
    if not source.is_file() or sha256(source) != config["source_checkpoint_sha256"]:
        raise RuntimeError("EXP-648 source checkpoint hash mismatch")
    dependency_manifest = dependency_source_manifest()
    cellular = load_verified_dependency_module(
        f"cellular_for_exp648_{mode}_{root_seed}",
        "cellular_model",
        dependency_manifest,
    )
    model = cellular.SharedDilatedConvGRU(
        width=256,
        hidden=int(config["source_model"]["hidden"]),
        cycles=int(config["source_model"]["cycles"]),
    )
    model.load_state_dict(load_file(source))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != int(config["source_model"]["parameter_count"]):
        raise RuntimeError("EXP-648 source parameter count mismatch")
    return model


def composite_cache_key(key_fields: dict[str, Any], repeat_index: int) -> str:
    return sha256_text(
        canonical_json({"repeat_index": int(repeat_index), **key_fields})
    )


def calibration_evidence_core_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the non-circular durable evidence hash scope.

    The post-resolution config link and the self hash are deliberately excluded:
    the resolved config points to this core hash, while the final artifact points
    back to the resolved config's exact file hash.
    """

    payload = copy.deepcopy(artifact)
    payload.pop("calibration_evidence_core_sha256", None)
    payload.pop("calibration_evidence_core_hash_scope", None)
    payload.get("source", {}).pop("resolved_config_sha256", None)
    return payload


def calibration_evidence_core_sha256(artifact: dict[str, Any]) -> str:
    return sha256_text(canonical_json(calibration_evidence_core_payload(artifact)))


def environment_payload(config: dict[str, Any]) -> dict[str, Any]:
    runtime = configure_frozen_runtime(config)
    distributions = {}
    for name in ("mlflow", "numpy", "psutil", "safetensors", "scipy", "sympy", "torch"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = "unavailable"
    return {
        "schema_version": "exp648-environment-v1",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_device": "cpu",
        **runtime,
        "dependencies": distributions,
    }


def exact_teacher_step(state: int, multiplicand: int, modulus: int, input_bit: int) -> tuple[int, int]:
    """Offline-only exact transition used for generated labels and evaluation."""

    raw = 2 * int(state) + int(input_bit) * int(multiplicand)
    return raw % int(modulus), raw // int(modulus)


def exact_teacher_source_sha256() -> str:
    return sha256_text(inspect.getsource(exact_teacher_step))


def decode_logits(logits: torch.Tensor) -> list[int]:
    rows = logits.detach().ge(0).tolist()
    return [
        sum(int(bit) << position for position, bit in enumerate(row)) for row in rows
    ]


def hard_straight_through(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    return logits.ge(0).to(probabilities.dtype) + probabilities - probabilities.detach()


def soft_features(
    state_plane: torch.Tensor,
    multiplicands: Sequence[int],
    moduli: Sequence[int],
    input_bits: Sequence[int],
    width: int,
    bits_fn: Callable[[list[int], int], torch.Tensor],
) -> torch.Tensor:
    batch = len(multiplicands)
    bit_plane = torch.tensor(input_bits, dtype=torch.float32)[:, None].expand(batch, width)
    lsb = torch.zeros(batch, width)
    msb = torch.zeros(batch, width)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (
            state_plane,
            bits_fn(list(multiplicands), width),
            bits_fn(list(moduli), width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul(2.0).sub(1.0)


def normalized_modulus_band(modulus: int) -> str:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    bit_length = int(modulus).bit_length()
    scaled_eighths = (int(modulus) * 8) // (1 << bit_length)
    if not 4 <= scaled_eighths <= 7:
        raise AssertionError("normalized modulus is outside [0.5, 1.0)")
    return f"b{scaled_eighths - 4}"


def first_error_index(
    predicted_states: Sequence[int], expected_states: Sequence[int], *, success_sentinel: int = 257
) -> int:
    if len(predicted_states) != len(expected_states):
        raise ValueError("predicted and expected state sequences must have equal length")
    for index, (prediction, expected) in enumerate(
        zip(predicted_states, expected_states, strict=True)
    ):
        if prediction != expected:
            return index
    return success_sentinel


def paired_drift_counts(
    control_exact: Sequence[bool], challenger_exact: Sequence[bool]
) -> dict[str, int]:
    if len(control_exact) != len(challenger_exact):
        raise ValueError("paired exactness vectors must have equal length")
    recovered = sum(
        (not left) and right
        for left, right in zip(control_exact, challenger_exact, strict=True)
    )
    introduced = sum(
        left and (not right)
        for left, right in zip(control_exact, challenger_exact, strict=True)
    )
    return {
        "cases": len(control_exact),
        "recovered_control_wrong_challenger_exact": int(recovered),
        "introduced_drift_control_exact_challenger_wrong": int(introduced),
    }


class Identity:
    def __init__(
        self, identity_index: int, multiplier: int, multiplicand: int, modulus: int
    ) -> None:
        self.identity_index = int(identity_index)
        self.multiplier = int(multiplier)
        self.multiplicand = int(multiplicand)
        self.modulus = int(modulus)

    @property
    def modulus_bit_length(self) -> int:
        return self.modulus.bit_length()

    @property
    def band(self) -> str:
        return normalized_modulus_band(self.modulus)


def window_match_key(window: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(window["window_start_index"]),
        int(window["modulus_bit_length"]),
        str(window["normalized_modulus_band"]),
    )


def validate_window(window: dict[str, Any], *, kind: str, k: int = 8) -> None:
    if kind not in ("failure", "success"):
        raise ValueError(kind)
    start = int(window["window_start_index"])
    if not 0 <= start <= 248:
        raise RuntimeError("window does not contain a complete K=8 prefix")
    if int(window["window_length"]) != k:
        raise RuntimeError("window length changed")
    if len(window["teacher_states"]) != k + 1:
        raise RuntimeError("teacher state window is incomplete")
    if len(window["natural_states"]) != k + 1:
        raise RuntimeError("natural state window is incomplete")
    if len(window["input_bits"]) != k or len(window["quotients"]) != k:
        raise RuntimeError("window transition labels are incomplete")
    modulus = int(window["modulus"])
    multiplier = int(window["multiplier"])
    multiplicand = int(window["multiplicand"])
    if int(window["modulus_bit_length"]) != modulus.bit_length():
        raise RuntimeError("window modulus bit length is inconsistent")
    if str(window["normalized_modulus_band"]) != normalized_modulus_band(modulus):
        raise RuntimeError("window normalized modulus band is inconsistent")
    expected_bits = [
        (multiplier >> (256 - 1 - position)) & 1
        for position in range(start, start + k)
    ]
    if [int(value) for value in window["input_bits"]] != expected_bits:
        raise RuntimeError("window input-bit labels are inconsistent")
    for offset, bit in enumerate(expected_bits):
        expected, quotient = exact_teacher_step(
            int(window["teacher_states"][offset]), multiplicand, modulus, bit
        )
        if expected != int(window["teacher_states"][offset + 1]):
            raise RuntimeError("window exact teacher transition is inconsistent")
        if quotient != int(window["quotients"][offset]):
            raise RuntimeError("window quotient label is inconsistent")
    if window["teacher_states"][0] != window["natural_states"][0]:
        raise RuntimeError("window does not begin from an exact prefix")
    natural_exact = [
        left == right
        for left, right in zip(
            window["natural_states"][1:], window["teacher_states"][1:], strict=True
        )
    ]
    if kind == "failure" and natural_exact[0]:
        raise RuntimeError("failure window does not begin at the first wrong transition")
    if kind == "success" and not all(natural_exact):
        raise RuntimeError("success window is not exact for all K transitions")
    if kind == "failure" and int(window["first_error_index"]) != start:
        raise RuntimeError("failure window start is not the zero-based first error")
    if kind == "success" and int(window["first_error_index"]) < start + k:
        raise RuntimeError("success window lacks complete-K eligibility")


def match_windows(
    failures: Sequence[dict[str, Any]],
    successes: Sequence[dict[str, Any]],
    target_pairs: int,
) -> list[dict[str, Any]]:
    """Apply the frozen first-unused exact-stratum matching without relaxation."""

    if target_pairs <= 0:
        raise ValueError("target_pairs must be positive")
    used: set[tuple[int, int]] = set()
    pairs: list[dict[str, Any]] = []
    for failure in failures:
        validate_window(failure, kind="failure")
        key = window_match_key(failure)
        selected = None
        for success in successes:
            validate_window(success, kind="success")
            identity_start = (
                int(success["identity_index"]),
                int(success["window_start_index"]),
            )
            if identity_start in used:
                continue
            if window_match_key(success) == key:
                selected = success
                used.add(identity_start)
                break
        if selected is None:
            continue
        pairs.append(
            {
                "pair_index": len(pairs),
                "matching_key": {
                    "window_start_index": key[0],
                    "modulus_bit_length": key[1],
                    "normalized_modulus_band": key[2],
                },
                "failure": failure,
                "success": selected,
            }
        )
        if len(pairs) == target_pairs:
            return pairs
    raise RuntimeError(
        f"structured matching shortfall: {len(pairs)} of {target_pairs}; relaxation prohibited"
    )


def _prime_in_band(rng: random.Random, bit_length: int, band_index: int) -> int:
    if not 0 <= band_index <= 3:
        raise ValueError("band index must be in [0, 3]")
    scale = 1 << bit_length
    lower = scale * (4 + band_index) // 8
    upper = scale * (5 + band_index) // 8
    while True:
        candidate = rng.randrange(lower, upper - 512) | 1
        prime = int(nextprime(candidate))
        if prime < upper and prime.bit_length() == bit_length:
            return prime


def generate_identities(
    mode: str, root_seed: int, start: int, count: int
) -> list[Identity]:
    """Generate deterministic training identities under an already-authorized root."""

    authorize_root(mode, root_seed)
    rng = random.Random(derive_seed(root_seed, f"identity-block:{start}:{count}"))
    identities = []
    for offset in range(count):
        index = start + offset
        modulus = _prime_in_band(rng, 256, index % 4)
        identities.append(
            Identity(
                identity_index=index,
                multiplier=rng.randrange(modulus),
                multiplicand=rng.randrange(modulus),
                modulus=modulus,
            )
        )
    return identities


def _identity_payload(identity: Identity) -> dict[str, Any]:
    return {
        "identity_index": identity.identity_index,
        "multiplier": identity.multiplier,
        "multiplicand": identity.multiplicand,
        "modulus": identity.modulus,
        "modulus_bit_length": identity.modulus_bit_length,
        "normalized_modulus_band": identity.band,
    }


def _make_window(
    identity: Identity,
    start: int,
    predicted_states: Sequence[int],
    teacher_states: Sequence[int],
    first_error: int,
    k: int = 8,
) -> dict[str, Any]:
    input_bits = [
        (identity.multiplier >> (256 - 1 - position)) & 1
        for position in range(start, start + k)
    ]
    quotients = []
    for offset, bit in enumerate(input_bits):
        expected, quotient = exact_teacher_step(
            teacher_states[start + offset],
            identity.multiplicand,
            identity.modulus,
            bit,
        )
        if expected != teacher_states[start + offset + 1]:
            raise AssertionError("teacher trajectory is inconsistent")
        quotients.append(quotient)
    return {
        **_identity_payload(identity),
        "window_start_index": start,
        "window_length": k,
        "first_error_index": first_error,
        "teacher_states": list(teacher_states[start : start + k + 1]),
        "natural_states": list(predicted_states[start : start + k + 1]),
        "input_bits": input_bits,
        "quotients": quotients,
    }


@torch.inference_mode()
def rollout_identity_batch(
    model: nn.Module,
    exp300,
    identities: Sequence[Identity],
    guard: Callable[[str], None],
) -> list[dict[str, Any]]:
    width = 256
    model.width = width
    batch = len(identities)
    multiplicands = [identity.multiplicand for identity in identities]
    moduli = [identity.modulus for identity in identities]
    predicted = [0] * batch
    teacher = [0] * batch
    predicted_history = [[0] for _ in identities]
    teacher_history = [[0] for _ in identities]
    first_errors: list[int | None] = [None] * batch
    for position in range(width):
        input_bits = [
            (identity.multiplier >> (width - 1 - position)) & 1
            for identity in identities
        ]
        logits, _ = model(exp300.features(predicted, multiplicands, moduli, input_bits, width))
        predicted = decode_logits(logits)
        teacher = [
            exact_teacher_step(state, value, modulus, bit)[0]
            for state, value, modulus, bit in zip(
                teacher, multiplicands, moduli, input_bits, strict=True
            )
        ]
        for index, (prediction, expected) in enumerate(
            zip(predicted, teacher, strict=True)
        ):
            predicted_history[index].append(prediction)
            teacher_history[index].append(expected)
            if first_errors[index] is None and prediction != expected:
                first_errors[index] = position
        if (position + 1) % 4 == 0 or position + 1 == width:
            guard("calibration:mining:trajectory-rollout")
    records = []
    for index, identity in enumerate(identities):
        error = first_errors[index] if first_errors[index] is not None else 257
        records.append(
            {
                "identity": identity,
                "predicted_states": predicted_history[index],
                "teacher_states": teacher_history[index],
                "first_error_index": error,
            }
        )
    return records


def trajectory_windows(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identity: Identity = record["identity"]
    predicted = record["predicted_states"]
    teacher = record["teacher_states"]
    first_error = int(record["first_error_index"])
    failures = []
    if first_error <= 248:
        failure = _make_window(identity, first_error, predicted, teacher, first_error)
        validate_window(failure, kind="failure")
        failures.append(failure)
    successes = []
    final_eligible_start = min(248, first_error - 8)
    for start in range(max(-1, final_eligible_start) + 1):
        success = _make_window(identity, start, predicted, teacher, first_error)
        validate_window(success, kind="success")
        successes.append(success)
    return failures, successes


def verify_calibration_corpus_payload(
    payload: dict[str, Any],
    *,
    mode: str,
    root_seed: int,
    replayed_trajectories: Sequence[dict[str, Any]] | None,
    target_pairs: int = 64,
) -> dict[str, Any]:
    """Strongly verify labels, matching, and full source-trajectory linkage."""

    authorize_root(mode, root_seed)
    if replayed_trajectories is None:
        raise RuntimeError("strong corpus verification requires independent model replay")
    if payload.get("schema_version") != "exp648-calibration-corpus-v1":
        raise RuntimeError("calibration corpus schema mismatch")
    if int(payload.get("root", -1)) != int(root_seed):
        raise RuntimeError("calibration corpus root mismatch")
    if int(payload.get("target_pairs", -1)) != int(target_pairs):
        raise RuntimeError("calibration corpus target count mismatch")
    trajectories = payload.get("trajectory_manifest")
    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise RuntimeError("calibration corpus trajectory manifest is missing")
    if len(trajectories) != int(payload.get("identity_trajectories", -1)):
        raise RuntimeError("calibration identity trajectory count mismatch")
    if len(replayed_trajectories) != len(trajectories):
        raise RuntimeError("calibration replay trajectory count mismatch")

    reconstructed_manifest = []
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for expected_index, (stored, replayed) in enumerate(
        zip(trajectories, replayed_trajectories, strict=True)
    ):
        identity = Identity(
            identity_index=int(stored["identity_index"]),
            multiplier=int(stored["multiplier"]),
            multiplicand=int(stored["multiplicand"]),
            modulus=int(stored["modulus"]),
        )
        if identity.identity_index != expected_index:
            raise RuntimeError("calibration identity order is not contiguous")
        identity_payload = _identity_payload(identity)
        if any(stored.get(key) != value for key, value in identity_payload.items()):
            raise RuntimeError("calibration identity semantic fields are inconsistent")
        reconstructed_manifest.append(identity_payload)
        predicted = [int(value) for value in stored["complete_predicted_states"]]
        teacher = [int(value) for value in stored["complete_teacher_states"]]
        if len(predicted) != 257 or len(teacher) != 257:
            raise RuntimeError("calibration complete trajectory length mismatch")
        if predicted[0] != 0 or teacher[0] != 0:
            raise RuntimeError("calibration trajectory does not begin at exact zero prefix")
        if any(value < 0 or value >= (1 << 256) for value in predicted):
            raise RuntimeError("calibration natural trajectory contains an invalid bit state")
        recomputed_teacher = [0]
        for position in range(256):
            bit = (identity.multiplier >> (256 - 1 - position)) & 1
            next_state, _ = exact_teacher_step(
                recomputed_teacher[-1], identity.multiplicand, identity.modulus, bit
            )
            recomputed_teacher.append(next_state)
        if teacher != recomputed_teacher:
            raise RuntimeError("calibration complete exact teacher trajectory is inconsistent")
        global_first_error = first_error_index(predicted[1:], teacher[1:])
        if int(stored["first_error_index"]) != global_first_error:
            raise RuntimeError("calibration stored global first error is inconsistent")

        replay_identity = replayed.get("identity")
        replay_identity_payload = (
            _identity_payload(replay_identity)
            if isinstance(replay_identity, Identity)
            else {
                key: replayed[key]
                for key in identity_payload
            }
        )
        if replay_identity_payload != identity_payload:
            raise RuntimeError("calibration replay identity mismatch")
        replay_predicted = [int(value) for value in replayed["predicted_states"]]
        replay_teacher = [int(value) for value in replayed["teacher_states"]]
        if replay_predicted != predicted or replay_teacher != teacher:
            raise RuntimeError("calibration full source trajectory replay mismatch")
        if int(replayed["first_error_index"]) != global_first_error:
            raise RuntimeError("calibration replay first error mismatch")

        record = {
            "identity": identity,
            "predicted_states": predicted,
            "teacher_states": teacher,
            "first_error_index": global_first_error,
        }
        local_failures, local_successes = trajectory_windows(record)
        failures.extend(local_failures)
        successes.extend(local_successes)

    if payload.get("identity_manifest") != reconstructed_manifest:
        raise RuntimeError("calibration identity manifest linkage mismatch")
    stored_pairs = payload.get("pairs")
    if not isinstance(stored_pairs, list) or len(stored_pairs) != target_pairs:
        raise RuntimeError("calibration paired corpus count mismatch")
    expected_pairs = match_windows(failures, successes, target_pairs)
    if canonical_json(stored_pairs) != canonical_json(expected_pairs):
        raise RuntimeError("calibration paired windows fail exact deterministic rematching")
    used_success_windows = [
        (
            int(pair["success"]["identity_index"]),
            int(pair["success"]["window_start_index"]),
        )
        for pair in stored_pairs
    ]
    if len(used_success_windows) != len(set(used_success_windows)):
        raise RuntimeError("calibration success window reuse detected")
    for pair_index, pair in enumerate(stored_pairs):
        if int(pair["pair_index"]) != pair_index:
            raise RuntimeError("calibration pair order mismatch")
        if window_match_key(pair["failure"]) != window_match_key(pair["success"]):
            raise RuntimeError("calibration pair matching field mismatch")
    return {
        "identity_trajectories": len(trajectories),
        "matched_pairs": len(stored_pairs),
        "identity_manifest_sha256": sha256_text(canonical_json(reconstructed_manifest)),
        "trajectory_manifest_sha256": sha256_text(canonical_json(trajectories)),
        "paired_corpus_sha256": sha256_text(canonical_json(stored_pairs)),
        "full_model_replay_verified": True,
        "semantic_labels_verified": True,
        "deterministic_matching_verified": True,
    }


def clean_transition_loss(
    logits: torch.Tensor,
    quotient_logits: torch.Tensor,
    target_bits: torch.Tensor,
    quotients: torch.Tensor,
    *,
    worst_bit_weight: float = 0.45,
    quotient_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    per_bit = F.binary_cross_entropy_with_logits(logits, target_bits, reduction="none")
    mean_bit_bce = per_bit.mean()
    worst_position_mean_bce = per_bit.mean(dim=0).max()
    quotient_cross_entropy = F.cross_entropy(quotient_logits, quotients)
    loss = (
        mean_bit_bce
        + worst_bit_weight * worst_position_mean_bce
        + quotient_weight * quotient_cross_entropy
    )
    return loss, {
        "mean_bit_bce": float(mean_bit_bce.detach()),
        "worst_position_mean_bce": float(worst_position_mean_bce.detach()),
        "quotient_cross_entropy": float(quotient_cross_entropy.detach()),
    }


def endpoint_residue_loss(
    logits: torch.Tensor,
    target_bits: torch.Tensor,
    *,
    worst_bit_weight: float = 0.45,
) -> tuple[torch.Tensor, dict[str, float]]:
    per_bit = F.binary_cross_entropy_with_logits(logits, target_bits, reduction="none")
    mean_bit_bce = per_bit.mean()
    worst_position_mean_bce = per_bit.mean(dim=0).max()
    loss = mean_bit_bce + worst_bit_weight * worst_position_mean_bce
    return loss, {
        "mean_bit_bce": float(mean_bit_bce.detach()),
        "worst_position_mean_bce": float(worst_position_mean_bce.detach()),
    }


def combine_arm_loss(
    base_clean_loss: torch.Tensor,
    endpoint_loss: torch.Tensor,
    endpoint_loss_weight: float,
) -> torch.Tensor:
    if float(endpoint_loss_weight) not in (0.0, 0.1):
        raise RuntimeError("EXP-648 endpoint coefficient must be exactly 0.0 or 0.1")
    return base_clean_loss + float(endpoint_loss_weight) * endpoint_loss


def matched_work_payload(config: dict[str, Any], pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pair_hashes = [sha256_text(canonical_json(pair)) for pair in pairs]
    return {
        "pair_hashes_in_order": pair_hashes,
        "batch_size": int(config["training"]["batch_size"]),
        "failure_windows_per_batch": int(config["training"]["failure_windows_per_batch"]),
        "success_windows_per_batch": int(
            config["training"]["matched_success_windows_per_batch"]
        ),
        "k": int(config["training"]["autonomous_unroll_steps"]),
        "clean_branch": config["training"]["clean_branch"],
        "autonomous_state_forward": config["training"]["autonomous_state_forward"],
        "autonomous_state_backward": config["training"]["autonomous_state_backward"],
        "optimizer": config["training"]["optimizer"],
        "learning_rate": float(config["training"]["learning_rate"]),
        "minimum_learning_rate": float(config["training"]["minimum_learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "gradient_clip": float(config["training"]["gradient_clip"]),
    }


def _delta(challenger: dict[str, Any], control: dict[str, Any], key: str) -> float:
    return float(challenger[key]) - float(control[key])


def validate_scientific_integrity(integrity: dict[str, Any]) -> bool:
    if not isinstance(integrity, dict):
        return False
    if set(integrity) != REQUIRED_SCIENTIFIC_INTEGRITY_KEYS:
        return False
    return all(type(integrity[key]) is bool and integrity[key] for key in integrity)


def screen_gate_results(
    control: dict[str, Any],
    challenger: dict[str, Any],
    integrity: dict[str, bool],
    gates: dict[str, Any],
) -> dict[str, Any]:
    # Decimal preregistration thresholds are inclusive.  A tiny arithmetic
    # tolerance prevents binary floating subtraction from turning exact decimal
    # equality (for example 0.94 - 0.95 == -0.01) into a false miss.
    tolerance = 1e-12
    results = {
        "width256_rollout_gain": _delta(challenger, control, "width256_rollout_accuracy")
        >= float(gates["minimum_width256_rollout_gain"]) - tolerance,
        "first_error_mean_delta": _delta(
            challenger, control, "mean_clipped_first_error_index"
        )
        >= float(gates["minimum_first_error_mean_delta"]) - tolerance,
        "width64_retention": _delta(challenger, control, "width64_rollout_accuracy")
        >= float(gates["minimum_width64_rollout_delta"]) - tolerance,
        "width128_retention": _delta(challenger, control, "width128_rollout_accuracy")
        >= float(gates["minimum_width128_rollout_delta"]) - tolerance,
        "challenger_width256_transition": float(
            challenger["width256_transition_accuracy"]
        )
        >= float(gates["minimum_challenger_width256_transition_accuracy"]) - tolerance,
        "challenger_invalid_rollout": float(challenger["invalid_rollout_rate"])
        <= float(gates["maximum_challenger_invalid_rollout_rate"]) + tolerance,
        "control_exact_repeat_and_determinism": (
            validate_scientific_integrity(integrity)
            and integrity["control_exact_repeat_deterministic"]
        ),
        "challenger_exact_repeat_and_determinism": (
            validate_scientific_integrity(integrity)
            and integrity["challenger_exact_repeat_deterministic"]
        ),
        "integrity": validate_scientific_integrity(integrity),
    }
    return {**results, "all_passed": all(results.values())}


def confirmation_pair_gate_results(
    control: dict[str, Any],
    challenger: dict[str, Any],
    integrity: dict[str, bool],
    gates: dict[str, Any],
) -> dict[str, Any]:
    translated = {
        "minimum_width256_rollout_gain": gates["minimum_width256_rollout_gain"],
        "minimum_first_error_mean_delta": gates["minimum_first_error_mean_delta"],
        "minimum_width64_rollout_delta": gates["minimum_width64_rollout_delta"],
        "minimum_width128_rollout_delta": gates["minimum_width128_rollout_delta"],
        "minimum_challenger_width256_transition_accuracy": gates[
            "minimum_challenger_width256_transition_accuracy"
        ],
        "maximum_challenger_invalid_rollout_rate": gates[
            "maximum_challenger_invalid_rollout_rate"
        ],
    }
    return screen_gate_results(control, challenger, integrity, translated)


def pooled_conditional_clopper_pearson(
    recovered: int,
    introduced_drift: int,
    *,
    denominator: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    b = int(recovered)
    c = int(introduced_drift)
    n = int(denominator)
    if b < 0 or c < 0 or n <= 0 or b + c > n:
        raise ValueError("invalid discordant-case counts")
    if b + c == 0:
        lower = 0.0
        net = 0.0
    else:
        lower = 0.0 if b == 0 else float(beta.ppf(alpha, b, c + 1))
        net = (2.0 * lower - 1.0) * (b + c) / n
    return {
        "recovered_b": b,
        "introduced_drift_c": c,
        "discordant": b + c,
        "denominator": n,
        "alpha": float(alpha),
        "one_sided_clopper_pearson_lower": lower,
        "net_exact_gain_lower": net,
        "strict_positive_gate_passed": net > 0.0,
    }


class ResourceTiming:
    def __init__(
        self, wall_seconds: float, aggregate_cpu_seconds: float, peak_rss_bytes: int
    ) -> None:
        self.wall_seconds = float(wall_seconds)
        self.aggregate_cpu_seconds = float(aggregate_cpu_seconds)
        self.peak_rss_bytes = int(peak_rss_bytes)

    def payload(self) -> dict[str, Any]:
        return {
            "wall_seconds": self.wall_seconds,
            "aggregate_cpu_seconds": self.aggregate_cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


class Measurement:
    def __init__(self, process: psutil.Process):
        self.process = process
        self.wall_started = time.perf_counter()
        self.cpu_started = process_cpu_seconds(process)
        self.peak_rss = process_peak_rss_bytes(process)

    def sample(self) -> None:
        self.peak_rss = max(self.peak_rss, process_peak_rss_bytes(self.process))

    def stop(self) -> ResourceTiming:
        self.sample()
        return ResourceTiming(
            wall_seconds=time.perf_counter() - self.wall_started,
            aggregate_cpu_seconds=process_cpu_seconds(self.process) - self.cpu_started,
            peak_rss_bytes=self.peak_rss,
        )


class CalibrationRunawayGuard:
    def __init__(
        self,
        process: psutil.Process,
        *,
        wall_ceiling: float,
        cpu_ceiling: float,
        reused_wall_seconds: float = 0.0,
        reused_cpu_seconds: float = 0.0,
        prior_superseded_wall_seconds: float = 0.0,
        prior_superseded_cpu_seconds: float = 0.0,
        active_started_wall: float | None = None,
        active_started_cpu: float | None = None,
    ) -> None:
        self.process = process
        self.wall_ceiling = float(wall_ceiling)
        self.cpu_ceiling = float(cpu_ceiling)
        self.reused_wall_seconds = float(reused_wall_seconds)
        self.reused_cpu_seconds = float(reused_cpu_seconds)
        self.prior_superseded_wall_seconds = float(prior_superseded_wall_seconds)
        self.prior_superseded_cpu_seconds = float(prior_superseded_cpu_seconds)
        self.started_wall = (
            time.perf_counter() if active_started_wall is None else float(active_started_wall)
        )
        self.started_cpu = (
            process_cpu_seconds(process)
            if active_started_cpu is None
            else float(active_started_cpu)
        )
        self.peak_rss = process_peak_rss_bytes(process)
        self.checks: list[dict[str, Any]] = []

    def charged(self) -> tuple[float, float]:
        return (
            self.prior_superseded_wall_seconds
            + self.reused_wall_seconds
            + time.perf_counter()
            - self.started_wall,
            self.prior_superseded_cpu_seconds
            + self.reused_cpu_seconds
            + process_cpu_seconds(self.process)
            - self.started_cpu,
        )

    def active(self) -> tuple[float, float]:
        return (
            time.perf_counter() - self.started_wall,
            process_cpu_seconds(self.process) - self.started_cpu,
        )

    def __call__(self, stage: str) -> None:
        wall, cpu = self.charged()
        self.peak_rss = max(self.peak_rss, process_peak_rss_bytes(self.process))
        self.checks.append({"stage": stage, "charged_wall_seconds": wall, "charged_cpu_seconds": cpu})
        if wall >= self.wall_ceiling:
            raise TimeoutError(
                f"EXP-648 calibration wall planning ceiling reached during {stage}"
            )
        if cpu >= self.cpu_ceiling:
            raise TimeoutError(
                f"EXP-648 calibration aggregate CPU planning ceiling reached during {stage}"
            )

    def add_reused(self, wall_seconds: float, cpu_seconds: float) -> None:
        self.reused_wall_seconds += float(wall_seconds)
        self.reused_cpu_seconds += float(cpu_seconds)
        self("calibration:cache-reuse-charge")


def _max_repeat_component(
    repeats: Sequence[dict[str, Any]], path: Sequence[str], resource: str
) -> float:
    values = []
    for repeat in repeats:
        value: Any = repeat
        for key in path:
            value = value[key]
        values.append(float(value[resource]))
    return max(values)


def _max_slower_arm_rate(
    repeats: Sequence[dict[str, Any]],
    component: str,
    units_key: str,
    resource: str,
    *,
    width: int | None = None,
) -> float:
    rates = []
    for repeat in repeats:
        for arm in ("control", "challenger"):
            value = repeat["components"][component][arm]
            if width is not None:
                value = value[str(width)]
            rates.append(float(value["timing"][resource]) / int(value[units_key]))
    return max(rates)


def validate_calibration_repeat(record: dict[str, Any], config: dict[str, Any]) -> None:
    if not record.get("complete"):
        raise RuntimeError("calibration repeat is incomplete")
    workload = record["workload"]
    if workload != workload_payload(config):
        raise RuntimeError("calibration repeat workload mismatch")
    components = record["components"]
    mining = components["mining"]
    if mining["matched_pairs"] != 64 or mining["windows"] != 128:
        raise RuntimeError("calibration mining workload is incomplete")
    if components["corpus_integrity_validation"]["windows"] != 128:
        raise RuntimeError("calibration corpus integrity work is incomplete")
    for arm in ("control", "challenger"):
        training = components["training"][arm]
        if training["updates"] != 20 or training["batch_size"] != 16 or training["k"] != 8:
            raise RuntimeError("calibration training workload is incomplete")
        if components["checkpoint"][arm]["writes"] != 1:
            raise RuntimeError("calibration checkpoint workload is incomplete")
        for width in (128, 256):
            if components["transitions"][arm][str(width)]["cases"] != 200:
                raise RuntimeError("calibration transition workload is incomplete")
        for width in (64, 128, 256):
            if components["rollouts"][arm][str(width)]["cases"] != 20:
                raise RuntimeError("calibration rollout workload is incomplete")
    if record["integrity"]["matched_arm_work_sha256_equal"] is not True:
        raise RuntimeError("calibration arms did not receive matched work")
    if record["integrity"]["identical_initial_tensor_sha256"] is not True:
        raise RuntimeError("calibration arm initialization mismatch")
    checks = record.get("integrity_checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CALIBRATION_INTEGRITY_BOOLEAN_KEYS:
        raise RuntimeError("calibration integrity boolean schema mismatch")
    if not all(type(checks[key]) is bool and checks[key] for key in checks):
        raise RuntimeError("calibration integrity boolean check failed")
    if record["fidelity_reduced"] is not False:
        raise RuntimeError("calibration fidelity was reduced")
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        if float(record["total_timing"][resource]) < 0:
            raise RuntimeError("calibration timing is invalid")
    component_peaks: list[int] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if set(("wall_seconds", "aggregate_cpu_seconds", "peak_rss_bytes")) <= set(value):
                component_peaks.append(int(value["peak_rss_bytes"]))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(record["components"])
    if int(record["total_timing"]["peak_rss_bytes"]) < max(component_peaks, default=0):
        raise RuntimeError("calibration total peak RSS is below a component high-water mark")
    manifest = record.get("integrity", {}).get("dependency_source_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("calibration dependency source manifest is missing")
    verify_dependency_source_manifest(manifest)
    fields = record.get("cache_key_fields")
    if not isinstance(fields, dict):
        raise RuntimeError("calibration cache key fields are missing")
    if fields.get("dependency_manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError("calibration dependency cache key mismatch")
    if fields.get("workload_sha256") != workload_sha256(config):
        raise RuntimeError("calibration workload cache key mismatch")


def validate_cached_repeat_record(
    record: dict[str, Any],
    config: dict[str, Any],
    *,
    repeat_index: int,
    expected_cache_key_fields: dict[str, Any],
) -> None:
    validate_calibration_repeat(record, config)
    if type(record.get("repeat_index")) is not int or record["repeat_index"] != repeat_index:
        raise RuntimeError("cached repeat index mismatch")
    if record.get("cache_key_fields") != expected_cache_key_fields:
        raise RuntimeError("cached repeat key fields mismatch")
    expected_key = composite_cache_key(expected_cache_key_fields, repeat_index)
    if record.get("cache_key") != expected_key:
        raise RuntimeError("cached repeat composite key mismatch")
    unit_record = record.get("unit_record")
    if not isinstance(unit_record, dict) or set(unit_record) != {"path", "sha256"}:
        raise RuntimeError("cached repeat unit-record link mismatch")
    unit_path = ROOT / unit_record["path"]
    if not unit_path.is_file() or sha256(unit_path) != unit_record["sha256"]:
        raise RuntimeError("cached repeat unit-record hash mismatch")
    for arm in ("control", "challenger"):
        checkpoint = record["components"]["checkpoint"][arm]
        path = ROOT / checkpoint["path"]
        if not path.is_file() or sha256(path) != checkpoint["sha256"]:
            raise RuntimeError("cached repeat checkpoint hash mismatch")
    mining = record["components"]["mining"]
    corpus_path = ROOT / mining["corpus_path"]
    if not corpus_path.is_file() or sha256(corpus_path) != mining["corpus_sha256"]:
        raise RuntimeError("cached repeat corpus hash mismatch")
    serialization = record["components"]["hash_serialization"]
    serialization_path = ROOT / serialization["path"]
    if not serialization_path.is_file() or sha256(serialization_path) != serialization["sha256"]:
        raise RuntimeError("cached repeat serialization hash mismatch")


def resolve_dynamic_budgets(
    config: dict[str, Any],
    repeats: Sequence[dict[str, Any]],
    launcher_setup_timing: dict[str, Any],
) -> dict[str, Any]:
    if len(repeats) != 3:
        raise RuntimeError("budget resolution requires three complete timed repeats")
    for repeat in repeats:
        validate_calibration_repeat(repeat, config)

    wall = "wall_seconds"
    cpu = "aggregate_cpu_seconds"
    resources = (wall, cpu)
    rates: dict[str, Any] = {}
    for resource in resources:
        rates[resource] = {
            "launcher_import_config_environment_dependency_setup": float(
                launcher_setup_timing[resource]
            ),
            "startup_per_unit": _max_repeat_component(
                repeats, ("components", "startup_model_loading", "timing"), resource
            ),
            "mining_per_matched_pair": _max_repeat_component(
                repeats, ("components", "mining", "timing"), resource
            )
            / 64.0,
            "corpus_integrity_per_window": _max_repeat_component(
                repeats,
                ("components", "corpus_integrity_validation", "timing"),
                resource,
            )
            / 128.0,
            "training_per_update_slower_arm": _max_slower_arm_rate(
                repeats, "training", "updates", resource
            ),
            "checkpoint_per_write_slower_arm": _max_slower_arm_rate(
                repeats, "checkpoint", "writes", resource
            ),
            "transition_per_case_slower_arm": {
                str(width): _max_slower_arm_rate(
                    repeats, "transitions", "cases", resource, width=width
                )
                for width in (128, 256)
            },
            "rollout_per_case_slower_arm": {
                str(width): _max_slower_arm_rate(
                    repeats, "rollouts", "cases", resource, width=width
                )
                for width in (64, 128, 256)
            },
            "hash_serialization_per_unit": _max_repeat_component(
                repeats, ("components", "hash_serialization", "timing"), resource
            ),
        }

    max_identities = max(int(repeat["components"]["mining"]["identity_trajectories"]) for repeat in repeats)
    identity_rate = max_identities / 64.0
    required_pairs = int(config["corpus"]["matched_pairs"])
    projected_identities = math.ceil(identity_rate * required_pairs)
    soft_identities = math.ceil(1.25 * projected_identities)
    hard_identity_ceiling = math.ceil(1.5 * soft_identities)

    def projected(resource: str, phase: str) -> tuple[float, dict[str, float]]:
        rate = rates[resource]
        components: dict[str, float] = {
            "launcher_import_config_environment_dependency_setup": rate[
                "launcher_import_config_environment_dependency_setup"
            ],
            "startup_model_loading": rate["startup_per_unit"],
        }
        if phase == "screen":
            components["corpus_mining_512_pairs_1024_windows"] = (
                rate["mining_per_matched_pair"] * required_pairs
            )
            updates = int(config["screen"]["updates_per_arm"])
            evaluation = config["screen"]["evaluation"]
        else:
            components["frozen_corpus_integrity_1024_windows"] = (
                rate["corpus_integrity_per_window"] * required_pairs * 2
            )
            updates = int(config["confirmation"]["updates_per_arm_per_pair"])
            evaluation = config["confirmation"]["evaluation_per_pair"]
        components["training_both_arms"] = rate["training_per_update_slower_arm"] * updates * 2
        components["checkpoint_both_arms"] = rate["checkpoint_per_write_slower_arm"] * 2
        for width in (128, 256):
            cases = int(evaluation["transition_cases"][width])
            components[f"transition_width{width}_both_arms"] = (
                rate["transition_per_case_slower_arm"][str(width)] * cases * 2
            )
        for width in (64, 128, 256):
            cases = int(evaluation["full_rollout_cases"][width])
            components[f"rollout_width{width}_both_arms"] = (
                rate["rollout_per_case_slower_arm"][str(width)] * cases * 2
            )
        repeat_cases = int(evaluation["exact_repeat_cases"])
        repeat_width = int(evaluation["exact_repeat_width"])
        components["exact_repeat_both_arms"] = (
            rate["rollout_per_case_slower_arm"][str(repeat_width)] * repeat_cases * 2
        )
        components["hash_and_serialization"] = rate["hash_serialization_per_unit"]
        return sum(components.values()), components

    resolved_phases: dict[str, Any] = {}
    for phase in ("screen", "confirmation_pair"):
        resource_records = {}
        for resource in resources:
            projection, components = projected(resource, phase)
            rates_for_resource = rates[resource]
            overhead_component = (
                rates_for_resource[
                    "launcher_import_config_environment_dependency_setup"
                ]
                + rates_for_resource["startup_per_unit"]
                + 2 * rates_for_resource["checkpoint_per_write_slower_arm"]
                + rates_for_resource["hash_serialization_per_unit"]
            )
            explicit_headroom = 2.0 * overhead_component
            soft = 1.25 * projection + explicit_headroom
            hard = 1.5 * soft
            resource_records[resource] = {
                "component_projection": components,
                "projected_component_seconds": projection,
                "explicit_measured_setup_headroom_seconds": explicit_headroom,
                "soft_budget_seconds": soft,
                "hard_runaway_ceiling_seconds": hard,
            }
        resolved_phases[phase] = resource_records

    confirmation_total = {}
    for resource in resources:
        pair = resolved_phases["confirmation_pair"][resource]
        projection = 2.0 * pair["projected_component_seconds"]
        explicit_headroom = 2.0 * pair["explicit_measured_setup_headroom_seconds"]
        soft = 1.25 * projection + explicit_headroom
        confirmation_total[resource] = {
            "projected_component_seconds": projection,
            "explicit_measured_setup_headroom_seconds": explicit_headroom,
            "soft_budget_seconds": soft,
            "hard_runaway_ceiling_seconds": 1.5 * soft,
        }
    resolved_phases["confirmation_total"] = confirmation_total

    corpus_ceiling = {
        "declared_work_unit": "complete_width256_source_identity_trajectory",
        "calibration_matched_pairs": 64,
        "maximum_complete_repeat_identity_trajectories": max_identities,
        "maximum_identity_trajectories_per_matched_pair": identity_rate,
        "required_matched_pairs": required_pairs,
        "projected_identity_trajectories": projected_identities,
        "soft_identity_trajectory_budget": soft_identities,
        "hard_identity_trajectory_ceiling": hard_identity_ceiling,
        "matching_relaxation_allowed": False,
        "count_reduction_allowed": False,
    }
    for resource in resources:
        rate = rates[resource]
        projection = (
            rate["launcher_import_config_environment_dependency_setup"]
            + rate["startup_per_unit"]
            + rate["mining_per_matched_pair"] * required_pairs
            + rate["corpus_integrity_per_window"] * required_pairs * 2
            + rate["hash_serialization_per_unit"]
        )
        overhead = 2.0 * (
            rate["launcher_import_config_environment_dependency_setup"]
            + rate["startup_per_unit"]
            + rate["hash_serialization_per_unit"]
        )
        soft = 1.25 * projection + overhead
        corpus_ceiling[resource] = {
            "projected_seconds": projection,
            "explicit_measured_setup_headroom_seconds": overhead,
            "soft_budget_seconds": soft,
            "hard_runaway_ceiling_seconds": 1.5 * soft,
        }

    return {
        "schema_version": "exp648-resolved-resource-v1",
        "rate_selection": "maximum-complete-repeat-and-slower-arm",
        "launcher_import_setup_timing": launcher_setup_timing,
        "complete_timed_repeats": len(repeats),
        "component_rates": rates,
        "corpus_mining_ceiling": corpus_ceiling,
        **resolved_phases,
        "planning_hard_ceiling_wall_seconds": float(
            config["resource_protocol"]["calibration"][
                "initial_planning_hard_ceiling_wall_seconds"
            ]
        ),
        "planning_hard_ceiling_aggregate_cpu_seconds": float(
            config["resource_protocol"]["calibration"][
                "initial_planning_hard_ceiling_aggregate_cpu_seconds"
            ]
        ),
        "fidelity_reduced": False,
    }


def _shape_signature(model: nn.Module) -> dict[str, Any]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(model.state_dict().items())
    }


class CalibrationExperiment:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        validate_frozen_config(self.config)
        self.root_seed = authorize_root(
            "calibration", self.config["resource_protocol"]["calibration"]["root"]
        )
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.source_path = (ROOT / self.config["source_checkpoint"]).resolve()
        if not self.source_path.is_file():
            raise RuntimeError("EXP-648 source checkpoint is missing")
        if sha256(self.source_path) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("EXP-648 source checkpoint hash mismatch")
        self.process = psutil.Process()
        self.runtime = configure_frozen_runtime(self.config)
        self.environment = environment_payload(self.config)
        self.environment_hash = sha256_text(canonical_json(self.environment))
        self.dependency_manifest = dependency_source_manifest()
        verify_dependency_source_manifest(self.dependency_manifest)
        self.runner_hash = sha256(Path(__file__))
        self.executed_config_hash = sha256(self.config_path)
        self.scientific_hash = scientific_protocol_sha256(self.config)
        self.resource_hash = resource_protocol_sha256(self.config)
        self.workload_hash = workload_sha256(self.config)
        self.preregistration_path = (ROOT / self.config["preregistration"]).resolve()
        self.preregistration_hash = sha256(self.preregistration_path)
        self.teacher_hash = exact_teacher_source_sha256()
        self.evaluator_hash = sha256_text(
            canonical_json(
                {
                    "runner_sha256": self.runner_hash,
                    "dependency_manifest_sha256": self.dependency_manifest[
                        "manifest_sha256"
                    ],
                    "evaluation_functions": [
                        inspect.getsource(self.evaluate_transitions_non_scientific),
                        inspect.getsource(self.evaluate_rollouts_non_scientific),
                    ],
                }
            )
        )
        self.dataset_spec = {
            "schema_version": "exp648-calibration-dataset-v1",
            "root": self.root_seed,
            "role": "resource-calibration-only",
            "width": 256,
            "matched_pairs": 64,
            "window_length": 8,
            "generator": "deterministic-balanced-normalized-band-prime-identities",
            "contains_benchmark_identities": False,
            "contains_checked_independent_or_private_inputs": False,
            "contains_labels_or_oracle_for_scientific_scoring": False,
        }
        self.dataset_spec_hash = sha256_text(canonical_json(self.dataset_spec))
        self.exp300 = load_verified_dependency_module(
            "exp300_for_exp648",
            "exp300_data_evaluator",
            self.dependency_manifest,
        )
        self.cellular = load_verified_dependency_module(
            "cellular_for_exp648",
            "cellular_model",
            self.dependency_manifest,
        )
        setup_wall = time.perf_counter() - _MODULE_IMPORT_WALL_STARTED
        setup_cpu_elapsed = process_cpu_seconds(self.process) - _MODULE_IMPORT_PROCESS_CPU_STARTED
        self.launcher_setup_timing = {
            "wall_seconds": setup_wall,
            "aggregate_cpu_seconds": setup_cpu_elapsed,
            "cumulative_process_cpu_seconds_at_setup_complete": process_cpu_seconds(
                self.process
            ),
            "peak_rss_bytes": process_peak_rss_bytes(self.process),
            "scope": (
                "top-of-module-through-import-config-environment-and-verified-"
                "dynamic-dependency-loading"
            ),
        }
        calibration = self.config["resource_protocol"]["calibration"]
        self.guard = CalibrationRunawayGuard(
            self.process,
            wall_ceiling=float(calibration["initial_planning_hard_ceiling_wall_seconds"]),
            cpu_ceiling=float(
                calibration["initial_planning_hard_ceiling_aggregate_cpu_seconds"]
            ),
            prior_superseded_wall_seconds=float(
                PRIOR_SUPERSEDED_CALIBRATION["wall_seconds"]
            ),
            prior_superseded_cpu_seconds=float(
                PRIOR_SUPERSEDED_CALIBRATION["aggregate_cpu_seconds"]
            ),
            active_started_wall=_MODULE_IMPORT_WALL_STARTED,
            active_started_cpu=_MODULE_IMPORT_PROCESS_CPU_STARTED,
        )

    def cache_key_fields(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": sha256(self.source_path),
            "evaluator_sha256": self.evaluator_hash,
            "dataset_sha256": self.dataset_spec_hash,
            "config_sha256": self.executed_config_hash,
            "seed": self.root_seed,
            "environment_sha256": self.environment_hash,
            "runner_sha256": self.runner_hash,
            "scientific_protocol_sha256": self.scientific_hash,
            "resource_protocol_sha256": self.resource_hash,
            "workload_sha256": self.workload_hash,
            "dependency_manifest_sha256": self.dependency_manifest["manifest_sha256"],
        }

    def cache_key(self, repeat_index: int) -> str:
        return composite_cache_key(self.cache_key_fields(), repeat_index)

    def new_model(self) -> nn.Module:
        model = self.cellular.SharedDilatedConvGRU(
            width=256,
            hidden=int(self.config["source_model"]["hidden"]),
            cycles=int(self.config["source_model"]["cycles"]),
        )
        model.load_state_dict(load_file(self.source_path))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters != int(self.config["source_model"]["parameter_count"]):
            raise RuntimeError(
                f"EXP-648 source parameter mismatch: {parameters} != 99824"
            )
        return model

    def load_verified_models_and_optimizers(
        self, update_count: int
    ) -> tuple[nn.Module, dict[str, nn.Module], dict[str, torch.optim.Optimizer], dict[str, Any]]:
        source_model = self.new_model().eval()
        arms = {"control": self.new_model(), "challenger": self.new_model()}
        initial_hashes = {name: state_dict_sha256(model) for name, model in arms.items()}
        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError("EXP-648 arms do not have identical initial tensors")
        source_state_hash = state_dict_sha256(source_model)
        if source_state_hash != initial_hashes["control"]:
            raise RuntimeError("EXP-648 arm does not exactly match raw source checkpoint")
        if _shape_signature(arms["control"]) != _shape_signature(arms["challenger"]):
            raise RuntimeError("EXP-648 arm shapes differ")
        optimizers = {
            name: torch.optim.AdamW(
                model.parameters(),
                lr=float(self.config["training"]["learning_rate"]),
                weight_decay=float(self.config["training"]["weight_decay"]),
            )
            for name, model in arms.items()
        }
        if not all(optimizer_is_fresh(optimizer) for optimizer in optimizers.values()):
            raise RuntimeError("EXP-648 optimizer state is not fresh")
        integrity = {
            "source_checkpoint_sha256": sha256(self.source_path),
            "source_state_dict_sha256": source_state_hash,
            "arm_initial_tensor_sha256": initial_hashes,
            "identical_initial_tensor_sha256": True,
            "shape_signature_sha256": sha256_text(
                canonical_json(_shape_signature(source_model))
            ),
            "parameter_count": sum(
                parameter.numel() for parameter in source_model.parameters()
            ),
            "optimizer_states_fresh": True,
            "optimizer_expected_updates": int(update_count),
            "torch_runtime": self.runtime,
        }
        return source_model, arms, optimizers, integrity

    def mine_pairs(self, source_model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        target_pairs = int(
            self.config["resource_protocol"]["calibration"][
                "representative_corpus_mining"
            ]["matched_pairs"]
        )
        failures: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []
        trajectory_manifest: list[dict[str, Any]] = []
        identity_index = 0
        batch_size = 32
        while True:
            identities = generate_identities(
                "calibration", self.root_seed, identity_index, batch_size
            )
            trajectories = rollout_identity_batch(
                source_model, self.exp300, identities, self.guard
            )
            for trajectory in trajectories:
                local_failures, local_successes = trajectory_windows(trajectory)
                failures.extend(local_failures)
                successes.extend(local_successes)
                trajectory_manifest.append(
                    {
                        **_identity_payload(trajectory["identity"]),
                        "first_error_index": int(trajectory["first_error_index"]),
                        "complete_predicted_states": trajectory["predicted_states"],
                        "complete_teacher_states": trajectory["teacher_states"],
                    }
                )
            identity_index += batch_size
            try:
                pairs = match_windows(failures, successes, target_pairs)
            except RuntimeError as error:
                if "structured matching shortfall" not in str(error):
                    raise
            else:
                break
            self.guard("calibration:mining:matching-batch-boundary")
        identity_manifest = [
            {
                key: record[key]
                for key in (
                    "identity_index",
                    "multiplier",
                    "multiplicand",
                    "modulus",
                    "modulus_bit_length",
                    "normalized_modulus_band",
                )
            }
            for record in trajectory_manifest
        ]
        payload = {
            "schema_version": "exp648-calibration-corpus-v1",
            "role": "resource-calibration-only-non-scientific",
            "root": self.root_seed,
            "target_pairs": target_pairs,
            "identity_trajectories": identity_index,
            "identity_manifest": identity_manifest,
            "trajectory_manifest": trajectory_manifest,
            "pairs": pairs,
            "matching": {
                "exact_fields": [
                    "window_start_index",
                    "modulus_bit_length",
                    "normalized_modulus_band",
                ],
                "failure_order": "root-generated-identity-and-trajectory-order",
                "success_tie_order": "first-unused-eligible-success-in-root-order",
                "reuse": False,
                "relaxation": False,
            },
            "synthetic_state_corruption": False,
            "state_bit_flips_or_injections": False,
        }
        return pairs, payload

    @staticmethod
    def ordered_batch_windows(
        pairs: Sequence[dict[str, Any]], update_index: int
    ) -> list[dict[str, Any]]:
        if len(pairs) < 8:
            raise RuntimeError("at least eight matched pairs are required")
        selected = [pairs[(update_index * 8 + offset) % len(pairs)] for offset in range(8)]
        failures = [pair["failure"] for pair in selected]
        successes = [pair["success"] for pair in selected]
        windows = failures + successes
        if len(windows) != 16:
            raise AssertionError("matched batch composition changed")
        return windows

    def training_update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        windows: Sequence[dict[str, Any]],
        endpoint_loss_weight: float,
    ) -> dict[str, Any]:
        width = 256
        k = 8
        if len(windows) != 16:
            raise RuntimeError("EXP-648 training batch must contain exactly 16 windows")
        model.width = width
        model.train()
        optimizer.zero_grad(set_to_none=True)
        multiplicands = [int(window["multiplicand"]) for window in windows]
        moduli = [int(window["modulus"]) for window in windows]
        autonomous_state = self.exp300.bits(
            [int(window["teacher_states"][0]) for window in windows], width
        )
        clean_loss_sum = 0.0
        endpoint_logits = None
        clean_forward_calls = 0
        autonomous_forward_calls = 0
        for offset in range(k):
            input_bits = [int(window["input_bits"][offset]) for window in windows]
            clean_states = [int(window["teacher_states"][offset]) for window in windows]
            targets = [int(window["teacher_states"][offset + 1]) for window in windows]
            quotients = torch.tensor(
                [int(window["quotients"][offset]) for window in windows],
                dtype=torch.long,
            )
            clean_logits, clean_qlogits = model(
                self.exp300.features(
                    clean_states, multiplicands, moduli, input_bits, width
                )
            )
            clean_loss, _ = clean_transition_loss(
                clean_logits,
                clean_qlogits,
                self.exp300.bits(targets, width),
                quotients,
                worst_bit_weight=float(self.config["loss"]["worst_bit_weight"]),
                quotient_weight=float(self.config["loss"]["quotient_weight"]),
            )
            # Exact gradient-equivalent accumulation of mean(K clean losses), while
            # retaining only the autonomous K-step graph.
            (clean_loss / k).backward()
            clean_loss_sum += float(clean_loss.detach())
            clean_forward_calls += 1
            autonomous_logits, _ = model(
                soft_features(
                    autonomous_state,
                    multiplicands,
                    moduli,
                    input_bits,
                    width,
                    self.exp300.bits,
                )
            )
            autonomous_state = hard_straight_through(autonomous_logits)
            endpoint_logits = autonomous_logits
            autonomous_forward_calls += 1
            self.guard("calibration:training:k8-step")
        assert endpoint_logits is not None
        endpoint_targets = self.exp300.bits(
            [int(window["teacher_states"][-1]) for window in windows], width
        )
        endpoint_loss, endpoint_stats = endpoint_residue_loss(
            endpoint_logits,
            endpoint_targets,
            worst_bit_weight=float(self.config["loss"]["worst_bit_weight"]),
        )
        # Even the 0.0 control traverses the identical endpoint graph.
        weighted_endpoint = endpoint_loss * float(endpoint_loss_weight)
        weighted_endpoint.backward()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(
                model.parameters(), float(self.config["training"]["gradient_clip"])
            )
        )
        optimizer.step()
        return {
            "base_clean_loss": clean_loss_sum / k,
            "endpoint_loss": float(endpoint_loss.detach()),
            "total_loss": clean_loss_sum / k
            + float(endpoint_loss_weight) * float(endpoint_loss.detach()),
            "endpoint_loss_weight": float(endpoint_loss_weight),
            "gradient_norm": gradient_norm,
            "clean_forward_calls": clean_forward_calls,
            "autonomous_forward_calls": autonomous_forward_calls,
            "hard_forward_state": True,
            "sigmoid_straight_through_backward": True,
            "synthetic_state_corruption": False,
            **{f"endpoint_{key}": value for key, value in endpoint_stats.items()},
        }

    def train_arm(
        self,
        arm_name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        pairs: Sequence[dict[str, Any]],
        updates: int,
    ) -> dict[str, Any]:
        weight = float(self.config["matched_arms"][arm_name]["endpoint_loss_weight"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=updates,
            eta_min=float(self.config["training"]["minimum_learning_rate"]),
        )
        history = []
        ordered_batch_hashes = []
        for update_index in range(updates):
            windows = self.ordered_batch_windows(pairs, update_index)
            ordered_batch_hashes.append(sha256_text(canonical_json(windows)))
            stats = self.training_update(model, optimizer, windows, weight)
            scheduler.step()
            history.append(
                {
                    "update": update_index + 1,
                    "learning_rate": scheduler.get_last_lr()[0],
                    **stats,
                }
            )
            self.guard("calibration:training:update-boundary")
        return {
            "arm": arm_name,
            "endpoint_loss_weight": weight,
            "updates": updates,
            "batch_size": 16,
            "failure_windows_per_batch": 8,
            "success_windows_per_batch": 8,
            "k": 8,
            "ordered_batch_sha256": sha256_text(canonical_json(ordered_batch_hashes)),
            "matched_work_sha256": sha256_text(
                canonical_json(matched_work_payload(self.config, pairs))
            ),
            "history": history,
            "clean_forward_calls": updates * 8,
            "autonomous_forward_calls": updates * 8,
            "synthetic_state_corruption": False,
        }

    def generate_transition_cases(self, width: int, count: int) -> dict[str, Any]:
        authorize_root("calibration", self.root_seed)
        rng = random.Random(derive_seed(self.root_seed, f"transition:{width}:{count}"))
        primes = [_prime_in_band(rng, width, index % 4) for index in range(16)]
        states = []
        multiplicands = []
        moduli = []
        input_bits = []
        targets = []
        quotients = []
        for _ in range(count):
            modulus = rng.choice(primes)
            state = rng.randrange(modulus)
            multiplicand = rng.randrange(modulus)
            bit = rng.randrange(2)
            target, quotient = exact_teacher_step(state, multiplicand, modulus, bit)
            states.append(state)
            multiplicands.append(multiplicand)
            moduli.append(modulus)
            input_bits.append(bit)
            targets.append(target)
            quotients.append(quotient)
        return {
            "state": states,
            "multiplicand": multiplicands,
            "modulus": moduli,
            "input_bit": input_bits,
            "target": targets,
            "quotient": quotients,
        }

    def generate_rollout_cases(self, width: int, count: int) -> dict[str, Any]:
        authorize_root("calibration", self.root_seed)
        rng = random.Random(derive_seed(self.root_seed, f"rollout:{width}:{count}"))
        primes = [_prime_in_band(rng, width, index % 4) for index in range(16)]
        moduli = [rng.choice(primes) for _ in range(count)]
        return {
            "multiplier": [rng.randrange(modulus) for modulus in moduli],
            "multiplicand": [rng.randrange(modulus) for modulus in moduli],
            "modulus": moduli,
        }

    @torch.inference_mode()
    def evaluate_transitions_non_scientific(
        self, model: nn.Module, width: int, cases: dict[str, Any]
    ) -> dict[str, Any]:
        model.eval()
        model.width = width
        count = len(cases["state"])
        integrity_rows = []
        for start in range(0, count, 128):
            end = min(count, start + 128)
            logits, quotient_logits = model(
                self.exp300.features(
                    cases["state"][start:end],
                    cases["multiplicand"][start:end],
                    cases["modulus"][start:end],
                    cases["input_bit"][start:end],
                    width,
                )
            )
            predictions = decode_logits(logits)
            quotient_predictions = quotient_logits.argmax(dim=1).tolist()
            # Execute exact evaluation comparisons, but retain only a digest so the
            # calibration cannot be mistaken for scientific accuracy evidence.
            integrity_rows.extend(
                {
                    "prediction": prediction,
                    "quotient_prediction": int(quotient_prediction),
                    "target_comparison": prediction == target,
                    "quotient_comparison": int(quotient_prediction) == quotient,
                    "valid_residue": prediction < modulus,
                }
                for prediction, quotient_prediction, target, quotient, modulus in zip(
                    predictions,
                    quotient_predictions,
                    cases["target"][start:end],
                    cases["quotient"][start:end],
                    cases["modulus"][start:end],
                    strict=True,
                )
            )
            self.guard(f"calibration:transition-evaluation:width{width}:batch")
        return {
            "cases": count,
            "output_and_discarded_comparison_sha256": sha256_text(
                canonical_json(integrity_rows)
            ),
            "scientific_metrics_reported": False,
        }

    @torch.inference_mode()
    def evaluate_rollouts_non_scientific(
        self, model: nn.Module, width: int, cases: dict[str, Any]
    ) -> dict[str, Any]:
        model.eval()
        model.width = width
        multiplier = list(cases["multiplier"])
        multiplicand = list(cases["multiplicand"])
        moduli = list(cases["modulus"])
        count = len(multiplier)
        state = [0] * count
        teacher = [0] * count
        first_errors: list[int | None] = [None] * count
        invalid = [False] * count
        prefix_digest = hashlib.sha256()
        for position in range(width):
            input_bits = [
                (value >> (width - 1 - position)) & 1 for value in multiplier
            ]
            logits, _ = model(
                self.exp300.features(state, multiplicand, moduli, input_bits, width)
            )
            state = decode_logits(logits)
            teacher = [
                exact_teacher_step(previous, value, modulus, bit)[0]
                for previous, value, modulus, bit in zip(
                    teacher, multiplicand, moduli, input_bits, strict=True
                )
            ]
            prefix_digest.update(canonical_json(state).encode("utf-8"))
            for index, (prediction, expected, modulus) in enumerate(
                zip(state, teacher, moduli, strict=True)
            ):
                if first_errors[index] is None and prediction != expected:
                    first_errors[index] = position
                invalid[index] = invalid[index] or prediction >= modulus
            if (position + 1) % 4 == 0 or position + 1 == width:
                self.guard(f"calibration:rollout-evaluation:width{width}:step")
        discarded_comparisons = [
            {
                "final_exact": prediction == expected,
                "first_error": 257 if error is None else error,
                "invalid": bad,
            }
            for prediction, expected, error, bad in zip(
                state, teacher, first_errors, invalid, strict=True
            )
        ]
        return {
            "cases": count,
            "prefix_output_sha256": prefix_digest.hexdigest(),
            "output_and_discarded_comparison_sha256": sha256_text(
                canonical_json(discarded_comparisons)
            ),
            "scientific_metrics_reported": False,
        }

    def execute_complete_unit(
        self,
        unit_name: str,
        *,
        repeat_index: int | None,
    ) -> dict[str, Any]:
        unit_dir = self.output / "units" / unit_name
        unit_dir.mkdir(parents=True, exist_ok=True)
        total_measurement = Measurement(self.process)
        self.guard(f"calibration:{unit_name}:start")

        startup_measurement = Measurement(self.process)
        update_count = int(
            self.config["resource_protocol"]["calibration"]["k8_updates_per_arm"]
        )
        source_model, arms, optimizers, model_integrity = (
            self.load_verified_models_and_optimizers(update_count)
        )
        transition_inputs = {
            str(width): self.generate_transition_cases(width, 200)
            for width in (128, 256)
        }
        rollout_inputs = {
            str(width): self.generate_rollout_cases(width, 20)
            for width in (64, 128, 256)
        }
        evaluation_input_hash = sha256_text(
            canonical_json(
                {"transitions": transition_inputs, "rollouts": rollout_inputs}
            )
        )
        startup_timing = startup_measurement.stop().payload()
        self.guard(f"calibration:{unit_name}:startup-complete")

        mining_measurement = Measurement(self.process)
        pairs, corpus_payload = self.mine_pairs(source_model)
        corpus_path = unit_dir / "calibration-corpus.json"
        corpus_bytes = canonical_json_bytes(corpus_payload, pretty=False)
        corpus_path.write_bytes(corpus_bytes)
        corpus_hash = sha256_bytes(corpus_bytes)
        identity_manifest_hash = sha256_text(
            canonical_json(corpus_payload["identity_manifest"])
        )
        trajectory_manifest_hash = sha256_text(
            canonical_json(corpus_payload["trajectory_manifest"])
        )
        pair_hash = sha256_text(canonical_json(pairs))
        mining_timing = mining_measurement.stop().payload()
        self.guard(f"calibration:{unit_name}:mining-complete")

        integrity_measurement = Measurement(self.process)
        reparsed = json.loads(corpus_path.read_text(encoding="utf-8"))
        if sha256(corpus_path) != corpus_hash:
            raise RuntimeError("calibration corpus hash changed during validation")
        replayed_trajectories: list[dict[str, Any]] = []
        stored_trajectories = reparsed["trajectory_manifest"]
        for start in range(0, len(stored_trajectories), 32):
            replay_identities = [
                Identity(
                    identity_index=int(record["identity_index"]),
                    multiplier=int(record["multiplier"]),
                    multiplicand=int(record["multiplicand"]),
                    modulus=int(record["modulus"]),
                )
                for record in stored_trajectories[start : start + 32]
            ]
            replayed_trajectories.extend(
                rollout_identity_batch(
                    source_model, self.exp300, replay_identities, self.guard
                )
            )
        semantic_verification = verify_calibration_corpus_payload(
            reparsed,
            mode="calibration",
            root_seed=self.root_seed,
            replayed_trajectories=replayed_trajectories,
            target_pairs=64,
        )
        corpus_integrity_timing = integrity_measurement.stop().payload()
        self.guard(f"calibration:{unit_name}:corpus-integrity-complete")

        training_components: dict[str, Any] = {}
        arm_work_hashes: dict[str, str] = {}
        for arm_name in ("control", "challenger"):
            measurement = Measurement(self.process)
            training = self.train_arm(
                arm_name,
                arms[arm_name],
                optimizers[arm_name],
                pairs,
                update_count,
            )
            training_components[arm_name] = {
                **training,
                "timing": measurement.stop().payload(),
            }
            arm_work_hashes[arm_name] = training["matched_work_sha256"]
            self.guard(f"calibration:{unit_name}:training:{arm_name}:complete")
        if len(set(arm_work_hashes.values())) != 1:
            raise RuntimeError("EXP-648 arms received different matched work")
        if (
            training_components["control"]["ordered_batch_sha256"]
            != training_components["challenger"]["ordered_batch_sha256"]
        ):
            raise RuntimeError("EXP-648 arm batch order differs")

        checkpoint_components: dict[str, Any] = {}
        for arm_name in ("control", "challenger"):
            measurement = Measurement(self.process)
            checkpoint_path = unit_dir / arm_name / "weights.safetensors"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    name: tensor.detach().contiguous()
                    for name, tensor in arms[arm_name].state_dict().items()
                },
                checkpoint_path,
            )
            checkpoint_components[arm_name] = {
                "writes": 1,
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "timing": measurement.stop().payload(),
            }
            self.guard(f"calibration:{unit_name}:checkpoint:{arm_name}:complete")

        transition_components: dict[str, Any] = {"control": {}, "challenger": {}}
        for arm_name in ("control", "challenger"):
            for width in (128, 256):
                measurement = Measurement(self.process)
                result = self.evaluate_transitions_non_scientific(
                    arms[arm_name], width, transition_inputs[str(width)]
                )
                transition_components[arm_name][str(width)] = {
                    **result,
                    "timing": measurement.stop().payload(),
                }
                self.guard(
                    f"calibration:{unit_name}:transition:{arm_name}:width{width}:complete"
                )

        rollout_components: dict[str, Any] = {"control": {}, "challenger": {}}
        for arm_name in ("control", "challenger"):
            for width in (64, 128, 256):
                measurement = Measurement(self.process)
                result = self.evaluate_rollouts_non_scientific(
                    arms[arm_name], width, rollout_inputs[str(width)]
                )
                rollout_components[arm_name][str(width)] = {
                    **result,
                    "timing": measurement.stop().payload(),
                }
                self.guard(
                    f"calibration:{unit_name}:rollout:{arm_name}:width{width}:complete"
                )

        serialization_measurement = Measurement(self.process)
        serialization_payload = {
            "unit_name": unit_name,
            "repeat_index": repeat_index,
            "cache_key_fields": self.cache_key_fields(),
            "executable_dependency_manifest": self.dependency_manifest,
            "corpus_artifact_sha256": corpus_hash,
            "identity_manifest_sha256": identity_manifest_hash,
            "trajectory_manifest_sha256": trajectory_manifest_hash,
            "paired_corpus_sha256": pair_hash,
            "evaluation_input_sha256": evaluation_input_hash,
            "checkpoint_sha256": {
                arm: checkpoint_components[arm]["sha256"]
                for arm in ("control", "challenger")
            },
            "transition_output_sha256": {
                arm: {
                    width: record["output_and_discarded_comparison_sha256"]
                    for width, record in transition_components[arm].items()
                }
                for arm in ("control", "challenger")
            },
            "rollout_output_sha256": {
                arm: {
                    width: record["output_and_discarded_comparison_sha256"]
                    for width, record in rollout_components[arm].items()
                }
                for arm in ("control", "challenger")
            },
        }
        serialization_path = unit_dir / "declared-hashing-serialization.json"
        serialization_path.write_bytes(canonical_json_bytes(serialization_payload, pretty=True))
        serialization_sha = sha256(serialization_path)
        hash_serialization_timing = serialization_measurement.stop().payload()
        self.guard(f"calibration:{unit_name}:hash-serialization-complete")

        total_timing = total_measurement.stop().payload()
        record = {
            "schema_version": "exp648-calibration-unit-v1",
            "complete": True,
            "unit_name": unit_name,
            "repeat_index": repeat_index,
            "cache_key": self.cache_key(repeat_index) if repeat_index is not None else None,
            "cache_key_fields": self.cache_key_fields(),
            "workload": workload_payload(self.config),
            "components": {
                "startup_model_loading": {"units": 1, "timing": startup_timing},
                "mining": {
                    "matched_pairs": 64,
                    "windows": 128,
                    "identity_trajectories": int(corpus_payload["identity_trajectories"]),
                    "corpus_path": str(corpus_path.relative_to(ROOT)),
                    "corpus_sha256": corpus_hash,
                    "identity_manifest_sha256": identity_manifest_hash,
                    "trajectory_manifest_sha256": trajectory_manifest_hash,
                    "paired_corpus_sha256": pair_hash,
                    "timing": mining_timing,
                },
                "corpus_integrity_validation": {
                    "windows": 128,
                    "matched_pairs": 64,
                    "hash_valid": True,
                    "matching_valid": True,
                    "semantic_verification": semantic_verification,
                    "timing": corpus_integrity_timing,
                },
                "training": training_components,
                "checkpoint": checkpoint_components,
                "transitions": transition_components,
                "rollouts": rollout_components,
                "hash_serialization": {
                    "units": 1,
                    "path": str(serialization_path.relative_to(ROOT)),
                    "sha256": serialization_sha,
                    "timing": hash_serialization_timing,
                },
            },
            "integrity": {
                **model_integrity,
                "matched_arm_work_sha256": arm_work_hashes,
                "matched_arm_work_sha256_equal": len(set(arm_work_hashes.values())) == 1,
                "ordered_batch_sha256_equal": (
                    training_components["control"]["ordered_batch_sha256"]
                    == training_components["challenger"]["ordered_batch_sha256"]
                ),
                "evaluation_input_sha256": evaluation_input_hash,
                "same_evaluation_inputs_for_both_arms": True,
                "state_corruption_or_injection": False,
                "source_checkpoint_unchanged": sha256(self.source_path)
                == self.config["source_checkpoint_sha256"],
                "dependency_source_manifest": self.dependency_manifest,
            },
            "integrity_checks": {
                "source_checkpoint_hash_valid": sha256(self.source_path)
                == self.config["source_checkpoint_sha256"],
                "identical_initialization_valid": model_integrity[
                    "identical_initial_tensor_sha256"
                ],
                "parameter_count_valid": model_integrity["parameter_count"]
                == int(self.config["source_model"]["parameter_count"]),
                "optimizer_states_fresh": model_integrity["optimizer_states_fresh"],
                "frozen_runtime_valid": model_integrity["torch_runtime"] == self.runtime,
                "dependency_manifest_valid": self.dependency_manifest
                == dependency_source_manifest(),
                "corpus_semantic_verification_valid": bool(
                    semantic_verification["full_model_replay_verified"]
                    and semantic_verification["semantic_labels_verified"]
                    and semantic_verification["deterministic_matching_verified"]
                ),
                "matched_arm_work_valid": len(set(arm_work_hashes.values())) == 1,
                "ordered_batches_valid": (
                    training_components["control"]["ordered_batch_sha256"]
                    == training_components["challenger"]["ordered_batch_sha256"]
                ),
                "same_evaluation_inputs_valid": True,
                "state_corruption_absent": True,
                "source_checkpoint_unchanged": sha256(self.source_path)
                == self.config["source_checkpoint_sha256"],
            },
            "total_timing": total_timing,
            "non_scientific": True,
            "labels_or_oracle_used_for_scientific_scoring": False,
            "scientific_accuracy_reported": False,
            "fidelity_reduced": False,
        }
        validate_calibration_repeat(record, self.config)
        unit_record_path = unit_dir / "unit-record.json"
        unit_record_path.write_bytes(canonical_json_bytes(record, pretty=True))
        record["unit_record"] = {
            "path": str(unit_record_path.relative_to(ROOT)),
            "sha256": sha256(unit_record_path),
        }
        return record

    def _cache_path(self, repeat_index: int) -> Path:
        return self.output / "repeat-cache" / (
            f"repeat-{repeat_index}-{self.cache_key(repeat_index)}.json"
        )

    def load_cached_repeat(self, repeat_index: int) -> dict[str, Any] | None:
        cache_path = self._cache_path(repeat_index)
        if not cache_path.is_file():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        record = payload.get("record")
        if payload.get("schema_version") != "exp648-complete-repeat-cache-v1":
            return None
        if payload.get("cache_key") != self.cache_key(repeat_index):
            return None
        if not isinstance(record, dict):
            return None
        try:
            validate_cached_repeat_record(
                record,
                self.config,
                repeat_index=repeat_index,
                expected_cache_key_fields=self.cache_key_fields(),
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None
        return {**record, "reused_from_cache": True, "cache_path": str(cache_path.relative_to(ROOT))}

    def save_repeat_cache(self, repeat_index: int, record: dict[str, Any]) -> Path:
        cache_path = self._cache_path(repeat_index)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "exp648-complete-repeat-cache-v1",
            "cache_key": self.cache_key(repeat_index),
            "record": record,
        }
        cache_path.write_bytes(canonical_json_bytes(payload, pretty=True))
        return cache_path

    def run(self) -> dict[str, Any]:
        repository_commit = git_output("rev-parse", "HEAD")
        repository_dirty = bool(git_output("status", "--porcelain"))
        cached = {
            repeat_index: self.load_cached_repeat(repeat_index)
            for repeat_index in range(1, 4)
        }

        # A recovery always pays for a fresh full-fidelity warmup before reusing any
        # complete repeat.  The timestamp makes freshness explicit and preserves it.
        warmup_name = f"warmup-{time.time_ns()}"
        warmup = self.execute_complete_unit(warmup_name, repeat_index=None)
        self.guard("calibration:fresh-complete-warmup-finished")

        repeats = []
        reused_indices = []
        measured_indices = []
        for repeat_index in range(1, 4):
            cached_record = cached[repeat_index]
            if cached_record is not None:
                self.guard.add_reused(
                    float(cached_record["total_timing"]["wall_seconds"]),
                    float(cached_record["total_timing"]["aggregate_cpu_seconds"]),
                )
                repeats.append(cached_record)
                reused_indices.append(repeat_index)
                continue
            record = self.execute_complete_unit(
                f"timed-repeat-{repeat_index}-{self.cache_key(repeat_index)[:12]}",
                repeat_index=repeat_index,
            )
            cache_path = self.save_repeat_cache(repeat_index, record)
            record["cache_path"] = str(cache_path.relative_to(ROOT))
            record["reused_from_cache"] = False
            repeats.append(record)
            measured_indices.append(repeat_index)
            self.guard(f"calibration:timed-repeat-{repeat_index}:preserved")

        repeats.sort(key=lambda item: int(item["repeat_index"]))
        budgets = resolve_dynamic_budgets(
            self.config, repeats, self.launcher_setup_timing
        )
        environment_path = self.output / "environment.json"
        environment_path.write_bytes(canonical_json_bytes(self.environment, pretty=True))
        environment_file_hash = sha256(environment_path)
        self.guard("calibration:final-bookkeeping-start")
        active_wall, active_cpu = self.guard.active()
        charged_wall, charged_cpu = self.guard.charged()
        final_core = {
            "schema_version": "exp648-calibration-evidence-v1",
            "run_id": RUN_ID,
            "status": "complete-non-scientific-resource-calibration",
            "role": "full-fidelity-resource-calibration-only",
            "authorization_boundary": {
                "implemented_modes": ["test", "calibration"],
                "calibration_root": self.root_seed,
                "corpus_root_opened": False,
                "screen_roots_opened": False,
                "confirmation_roots_opened": False,
                "checked_independent_or_private_opened": False,
                "exp649_formal_root_opened": False,
                "corpus_screen_confirmation_authorized": False,
            },
            "source": {
                "repository_commit": repository_commit,
                "repository_dirty_during_calibration": repository_dirty,
                "executed_pre_resolution_config_sha256": self.executed_config_hash,
                "resolved_config_sha256": "pending-post-calibration-resolution",
                "runner_sha256": self.runner_hash,
                "scientific_protocol_sha256": self.scientific_hash,
                "resource_protocol_sha256": self.resource_hash,
                "workload_sha256": self.workload_hash,
                "preregistration_sha256": self.preregistration_hash,
                "source_checkpoint_sha256": sha256(self.source_path),
                "exact_teacher_source_sha256": self.teacher_hash,
                "evaluator_sha256": self.evaluator_hash,
                "environment_payload_canonical_sha256": self.environment_hash,
                "environment_file_bytes_sha256": environment_file_hash,
                "executable_dependency_manifest": self.dependency_manifest,
                "calibration_dataset_spec_sha256": self.dataset_spec_hash,
            },
            "environment": self.environment,
            "launcher_import_config_environment_dynamic_module_setup": self.launcher_setup_timing,
            "dataset": self.dataset_spec,
            "workload": workload_payload(self.config),
            "warmup": warmup,
            "timed_repeats": repeats,
            "recovery": {
                "fresh_complete_warmup_performed": True,
                "reused_complete_repeat_indices": reused_indices,
                "measured_missing_repeat_indices": measured_indices,
                "reused_complete_repeat_count": len(reused_indices),
                "reused_wall_seconds_charged": self.guard.reused_wall_seconds,
                "reused_aggregate_cpu_seconds_charged": self.guard.reused_cpu_seconds,
                "prior_superseded_calibration": PRIOR_SUPERSEDED_CALIBRATION,
                "prior_superseded_wall_seconds_charged": self.guard.prior_superseded_wall_seconds,
                "prior_superseded_aggregate_cpu_seconds_charged": self.guard.prior_superseded_cpu_seconds,
                "active_session_wall_seconds": active_wall,
                "active_session_aggregate_cpu_seconds": active_cpu,
                "new_active_plus_reused_wall_seconds": active_wall
                + self.guard.reused_wall_seconds,
                "new_active_plus_reused_aggregate_cpu_seconds": active_cpu
                + self.guard.reused_cpu_seconds,
                "cumulative_prior_active_reused_wall_seconds": charged_wall,
                "cumulative_prior_active_reused_aggregate_cpu_seconds": charged_cpu,
                "remaining_planning_wall_headroom_seconds": self.guard.wall_ceiling
                - charged_wall,
                "remaining_planning_aggregate_cpu_headroom_seconds": self.guard.cpu_ceiling
                - charged_cpu,
            },
            "resolved_resource_budgets": budgets,
            "integrity": {
                "all_complete_repeats_valid": all(
                    repeat.get("complete") is True for repeat in repeats
                ),
                "all_repeat_cache_keys_match": all(
                    repeat["cache_key_fields"] == self.cache_key_fields()
                    for repeat in repeats
                ),
                "all_repeat_workloads_exact": all(
                    repeat["workload"] == workload_payload(self.config)
                    for repeat in repeats
                ),
                "source_checkpoint_unchanged": sha256(self.source_path)
                == self.config["source_checkpoint_sha256"],
                "dependency_source_manifest_valid": self.dependency_manifest
                == dependency_source_manifest(),
                "all_repeat_integrity_booleans_valid": all(
                    set(repeat["integrity_checks"])
                    == REQUIRED_CALIBRATION_INTEGRITY_BOOLEAN_KEYS
                    and all(repeat["integrity_checks"].values())
                    for repeat in repeats
                ),
                "total_peak_rss_covers_all_component_peaks": True,
                "fidelity_reduced": False,
            },
            "non_scientific_declaration": {
                "scientific_evaluation_not_performed": True,
                "exact_accuracy_not_reported": True,
                "labels_and_exact_teacher_use": "offline-calibration-workload-only-not-selection-evidence",
                "candidate_registration": False,
                "promotion_decision": False,
            },
            "limitations": [
                "This is throughput and integrity evidence only; it reports no scientific exact-answer accuracy.",
                "Only calibration root 20261151 was opened; corpus, screen, confirmation, checked, independent, private, and EXP-649 formal roots remained unopened.",
                "The resolved budgets project declared work from maximum complete-repeat component rates and the slower arm; actual later work requires the frozen hashes and separate Judge authorization.",
                "Corpus-mining throughput is stochastic in identity work even under a deterministic root, so both identity-trajectory and separate wall/aggregate-CPU ceilings are retained.",
                "No fidelity, count, repeat, diagnostic, or evaluation work was reduced to fit a wall-clock cap.",
            ],
            "fidelity_reduced": False,
        }
        evidence_core_hash = calibration_evidence_core_sha256(final_core)
        final = {
            **final_core,
            "calibration_evidence_core_sha256": evidence_core_hash,
            "calibration_evidence_core_hash_scope": (
                "canonical-artifact-excluding-self-hash-hash-scope-and-"
                "source.resolved_config_sha256"
            ),
        }
        raw_path = self.output / "calibration.json"
        raw_path.write_bytes(canonical_json_bytes(final, pretty=True))
        return {
            **final,
            "raw_calibration": {
                "path": str(raw_path.relative_to(ROOT)),
                "sha256": sha256(raw_path),
                "bytes": raw_path.stat().st_size,
            },
            "raw_environment": {
                "path": str(environment_path.relative_to(ROOT)),
                "sha256": sha256(environment_path),
            },
        }


def log_calibration_mlflow(config: dict[str, Any], artifact: dict[str, Any], raw_path: Path) -> dict[str, str] | None:
    try:
        experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{RUN_ID}-calibration-only") as run:
            mlflow.set_tags(
                {
                    "canonical_run_id": RUN_ID,
                    "status": "complete-non-scientific-resource-calibration",
                    "role": "calibration-only-non-scientific",
                    "candidate_registration": "false",
                    "scientific_accuracy_reported": "false",
                }
            )
            mlflow.log_params(
                {
                    "calibration_root": CALIBRATION_ROOT,
                    "runner_sha256": artifact["source"]["runner_sha256"],
                    "executed_config_sha256": artifact["source"][
                        "executed_pre_resolution_config_sha256"
                    ],
                    "scientific_protocol_sha256": artifact["source"][
                        "scientific_protocol_sha256"
                    ],
                    "resource_protocol_sha256": artifact["source"][
                        "resource_protocol_sha256"
                    ],
                }
            )
            mlflow.log_metrics(
                {
                    "charged_wall_seconds": artifact["recovery"][
                        "cumulative_prior_active_reused_wall_seconds"
                    ],
                    "charged_aggregate_cpu_seconds": artifact["recovery"][
                        "cumulative_prior_active_reused_aggregate_cpu_seconds"
                    ],
                    "peak_rss_bytes": max(
                        [artifact["warmup"]["total_timing"]["peak_rss_bytes"]]
                        + [
                            repeat["total_timing"]["peak_rss_bytes"]
                            for repeat in artifact["timed_repeats"]
                        ]
                    ),
                }
            )
            mlflow.log_artifact(str(raw_path), artifact_path="calibration-only")
            return {
                "tracking_uri": tracking_uri(),
                "experiment_id": str(experiment_id),
                "run_id": run.info.run_id,
            }
    except Exception as error:  # Tracking is complementary; raw evidence is canonical.
        return {"error": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_648_structured_trajectory_consistency.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-648-structured-trajectory-consistency",
    )
    parser.add_argument("--mode", choices=("calibration",), required=True)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    if args.mode != "calibration":
        raise PermissionError("only non-scientific calibration is implemented")
    experiment = CalibrationExperiment(args.config, args.output)
    artifact = experiment.run()
    raw_path = args.output.resolve() / "calibration.json"
    mlflow_record = None
    if not args.no_mlflow:
        mlflow_record = log_calibration_mlflow(experiment.config, artifact, raw_path)
        mlflow_path = args.output.resolve() / "mlflow.json"
        mlflow_path.write_bytes(canonical_json_bytes(mlflow_record, pretty=True))
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "raw_calibration": artifact["raw_calibration"],
                "recovery": artifact["recovery"],
                "resolved_resource_budgets": artifact["resolved_resource_budgets"],
                "mlflow": mlflow_record,
                "scientific_accuracy_reported": False,
                "fidelity_reduced": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
