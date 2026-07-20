"""Frozen corpus constructor and fail-closed scientific runner for EXP-648.

The only executable scientific operation authorized in this revision is corpus
construction with root 20261152.  Screen code is complete but cannot pass its
authorization boundary until a later Judge creates the exact canonical
authorization manifest derived from the frozen corpus manifest.
"""

from __future__ import annotations

# The launcher budget begins before the frozen calibration module imports any
# third-party package.  process_time is recorded only as a bootstrap marker;
# all durable CPU accounting below uses psutil user+system process time.
_MODULE_WALL_STARTED = __import__("time").perf_counter()
_MODULE_PROCESS_CPU_STARTED = __import__("time").process_time()

import argparse
import copy
import gzip
import hashlib
import importlib.util
import inspect
import io
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "exp-648-structured-trajectory-consistency"
TEST_ONLY_ROOT = 648_900_001
CORPUS_ROOT = 20_261_152
CALIBRATION_ROOT = 20_261_151
SCREEN_TRAINING_ROOT = 20_261_153
SCREEN_EVALUATION_ROOT = 20_261_154
CONFIRMATION_ROOTS = frozenset((20_261_155, 20_261_156, 20_261_157, 20_261_158))
EXP649_FORMAL_ROOT = 20_261_149

CALIBRATION_RUNNER = ROOT / "experiments/exp_648_structured_trajectory_consistency/run.py"
CALIBRATION_RUNNER_SHA256 = (
    "f8cdc278b67013c6ca920833bc26473ad938e15749eb85811fc00016b6bb5672"
)


def _bootstrap_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if not CALIBRATION_RUNNER.is_file():
    raise RuntimeError("frozen EXP-648 calibration module is missing")
if _bootstrap_sha256(CALIBRATION_RUNNER) != CALIBRATION_RUNNER_SHA256:
    raise RuntimeError("frozen EXP-648 calibration module hash mismatch")


def _load_calibration_module():
    spec = importlib.util.spec_from_file_location(
        "exp648_frozen_calibration_for_scientific", CALIBRATION_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen EXP-648 calibration module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cal = _load_calibration_module()

# Third-party objects are taken from the already hash-verified calibration
# module, keeping one frozen implementation surface for model/data primitives.
mlflow = cal.mlflow
psutil = cal.psutil
torch = cal.torch
yaml = cal.yaml
nn = cal.nn
load_file = cal.load_file
save_file = cal.save_file

CONFIG_PATH = ROOT / "configs/exp_648_structured_trajectory_consistency.yaml"
EXECUTED_CONFIG_PATH = (
    ROOT / "experiments/runs/exp-648-structured-trajectory-consistency/executed-config.yaml"
)
CALIBRATION_EVIDENCE_PATH = (
    ROOT / "experiments/runs/exp-648-structured-trajectory-consistency/calibration.json"
)
CALIBRATION_ENVIRONMENT_PATH = (
    ROOT / "experiments/runs/exp-648-structured-trajectory-consistency/environment.json"
)
PREREGISTRATION_PATH = (
    ROOT / "knowledge/autoresearch/ablations/exp-648-structured-trajectory-consistency.md"
)
SOURCE_CHECKPOINT_PATH = (
    ROOT
    / ".scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"
)
DATA_DIR = ROOT / "data/generated/exp-648-structured-trajectory-consistency"
TRACKED_CORPUS_DIR = (
    ROOT / "experiments/runs/exp-648-structured-trajectory-consistency/corpus"
)
SCRATCH_CORPUS_DIR = ROOT / ".scratch/exp-648-structured-trajectory-consistency/corpus"
SCREEN_AUTHORIZATION_PATH = (
    ROOT
    / "experiments/runs/exp-648-structured-trajectory-consistency/screen/judge-authorization.json"
)

PINNED = {
    "repository_head": "9f2226873a66e1938dbf30d87acbc876e528c622",
    "resolved_config_sha256": "7ec4eb7b31a93fc7d6cdcb8967abbaeed25f674248296ca6f63b395b51c63263",
    "executed_config_sha256": "28bbbaaedf10929bdd133a764ddfe688e8b9a98f0196b8c768930895b5407127",
    "calibration_evidence_sha256": "716852e1fea2ad21401016cdeb3e5e9ef5e529ca263b91a6079904e6075770db",
    "calibration_evidence_core_sha256": "45f34f3c7e436af2e7468060d734937780c3cda36a758cd841a378484f2aceff",
    "calibration_environment_file_sha256": "d5754d46d762643e9488a85924208ccd83a959206a14c7b7041cea7d332819c9",
    "calibration_environment_payload_sha256": "f85b286eeea048674dc157e55ba913c3afd97788f20471ea4828061573943a6a",
    "dependency_manifest_sha256": "a07662d9562e154494f8db189e19beaa5b10a050d89e88905bee675c09bceecc",
    "source_checkpoint_sha256": "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd",
    "preregistration_sha256": "49336918d9a893dfb53427b6372a4d8a507e54e79ea77c539aaa507aba0a9d60",
    "scientific_protocol_sha256": "f09a235072f36c5703a8fb58c0eb62b8de0070adbee599df94eb650edf206086",
    "resource_protocol_sha256": "f0800dff74038df7465abff8360fe572f603bdab4af26f547b130286e169b198",
    "calibration_workload_sha256": "ab7279be43dce8a56cd5e64a91918b1d4845cb33f8bb0e22fb0840c05adcc835",
    "exact_teacher_source_sha256": "ea10a1483bbabb2c9f5901b5f019ef7ad9ab77a525ae15c9307dc97630e62aef",
    "calibration_evaluator_sha256": "a926268b51c935534a21edfe52fe25ec6fa3721ee5e4cae2d144cbce5408560c",
}

CORPUS_TARGET_PAIRS = 512
WINDOW_LENGTH = 8
IDENTITY_BATCH_SIZE = 32
PROJECTED_IDENTITIES = 768
SOFT_IDENTITY_BUDGET = 960
HARD_IDENTITY_CEILING = 1440
PROJECTED_WALL_SECONDS = 2420.4370715340046
SOFT_WALL_SECONDS = 3032.646309429507
HARD_WALL_SECONDS = 4548.96946414426
PROJECTED_CPU_SECONDS = 9669.030946256997
SOFT_CPU_SECONDS = 12101.790575335246
HARD_CPU_SECONDS = 18152.68586300287
QUARTER_PAIRS = 128
QUARTER_SLOWDOWN_LIMIT = 1.20

REQUIRED_CORPUS_INTEGRITY_KEYS = frozenset(
    (
        "authorization_boundary_valid",
        "calibration_runner_hash_valid",
        "resolved_config_hash_valid",
        "executed_config_hash_valid",
        "calibration_evidence_hash_valid",
        "calibration_evidence_core_hash_valid",
        "calibration_environment_hash_valid",
        "environment_payload_hash_valid",
        "dependency_manifest_hash_valid",
        "source_checkpoint_hash_valid",
        "source_weights_unchanged",
        "preregistration_hash_valid",
        "scientific_protocol_hash_valid",
        "resource_protocol_hash_valid",
        "calibration_workload_hash_valid",
        "exact_teacher_hash_valid",
        "repository_head_valid",
        "repository_upstream_valid",
        "identity_geometry_valid",
        "identity_order_valid",
        "identity_uniqueness_valid",
        "complete_source_trajectories_valid",
        "independent_source_replay_valid",
        "teacher_transitions_and_quotients_valid",
        "first_error_and_k_eligibility_valid",
        "window_slice_linkage_valid",
        "exact_matching_fields_valid",
        "deterministic_first_unused_matching_valid",
        "success_window_nonreuse_valid",
        "target_pair_count_valid",
        "natural_trajectories_only",
        "benchmark_identities_absent",
        "gzip_determinism_valid",
        "raw_tracked_byte_parity_valid",
        "cache_keys_valid",
        "reused_resource_charge_valid",
        "first_quarter_guard_valid",
        "hard_resource_guards_valid",
        "no_model_update",
        "no_scientific_accuracy_or_selection_metrics",
        "mlflow_receipt_valid",
    )
)


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    return canonical_json(value).encode("utf-8")


def deterministic_gzip_bytes(value: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(value)
    return buffer.getvalue()


def deterministic_gunzip_bytes(value: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(value), mode="rb") as handle:
        return handle.read()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_mirrored(relative: Path, payload: bytes, *, family: str) -> tuple[Path, Path]:
    if family == "data":
        tracked = DATA_DIR / relative
        raw = SCRATCH_CORPUS_DIR / "data" / relative
    elif family == "evidence":
        tracked = TRACKED_CORPUS_DIR / relative
        raw = SCRATCH_CORPUS_DIR / "evidence" / relative
    else:
        raise ValueError(family)
    _atomic_write(raw, payload)
    _atomic_write(tracked, payload)
    if raw.read_bytes() != tracked.read_bytes():
        raise RuntimeError(f"raw/tracked byte parity failed for {relative}")
    return raw, tracked


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def verify_allowed_worktree() -> None:
    allowed = (
        "experiments/exp_648_structured_trajectory_consistency/scientific.py",
        "tests/test_exp648_structured_trajectory_scientific.py",
        "data/generated/exp-648-structured-trajectory-consistency/",
        "experiments/runs/exp-648-structured-trajectory-consistency/corpus/",
        ".scratch/exp-648-structured-trajectory-consistency/corpus/",
        ".scratch/mlflow/",
    )
    for line in git_output("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if not any(path == prefix or path.startswith(prefix) for prefix in allowed):
            raise RuntimeError(f"worktree change outside Worker allowlist: {path}")


def verify_pinned_foundation(config: dict[str, Any]) -> dict[str, str]:
    checks = {
        "calibration_runner_sha256": sha256(CALIBRATION_RUNNER),
        "resolved_config_sha256": sha256(CONFIG_PATH),
        "executed_config_sha256": sha256(EXECUTED_CONFIG_PATH),
        "calibration_evidence_sha256": sha256(CALIBRATION_EVIDENCE_PATH),
        "calibration_environment_file_sha256": sha256(CALIBRATION_ENVIRONMENT_PATH),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT_PATH),
        "preregistration_sha256": sha256(PREREGISTRATION_PATH),
        "repository_head": git_output("rev-parse", "HEAD"),
        "repository_upstream": git_output("rev-parse", "@{upstream}"),
        "scientific_protocol_sha256": cal.scientific_protocol_sha256(config),
        "resource_protocol_sha256": cal.resource_protocol_sha256(config),
        "calibration_workload_sha256": cal.workload_sha256(config),
        "exact_teacher_source_sha256": cal.exact_teacher_source_sha256(),
    }
    expected = {
        key: PINNED[key]
        for key in checks
        if key in PINNED
    }
    expected["calibration_runner_sha256"] = CALIBRATION_RUNNER_SHA256
    expected["repository_upstream"] = PINNED["repository_head"]
    for key, value in expected.items():
        if checks[key] != value:
            raise RuntimeError(f"frozen foundation mismatch for {key}: {checks[key]} != {value}")
    evidence = json.loads(CALIBRATION_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if evidence.get("calibration_evidence_core_sha256") != PINNED[
        "calibration_evidence_core_sha256"
    ]:
        raise RuntimeError("calibration evidence core hash mismatch")
    if cal.calibration_evidence_core_sha256(evidence) != PINNED[
        "calibration_evidence_core_sha256"
    ]:
        raise RuntimeError("calibration evidence core recomputation mismatch")
    environment = json.loads(CALIBRATION_ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    if sha256_bytes(canonical_json_bytes(environment)) != PINNED[
        "calibration_environment_payload_sha256"
    ]:
        raise RuntimeError("calibration environment payload hash mismatch")
    manifest = cal.dependency_source_manifest()
    cal.verify_dependency_source_manifest(manifest)
    if manifest["manifest_sha256"] != PINNED["dependency_manifest_sha256"]:
        raise RuntimeError("dependency manifest hash mismatch")
    cal.validate_frozen_config(config)
    verify_allowed_worktree()
    return checks


def expected_screen_authorization_payload(corpus_manifest_sha256: str) -> dict[str, Any]:
    if len(corpus_manifest_sha256) != 64:
        raise ValueError("corpus manifest hash must be SHA-256")
    return {
        "schema_version": "exp648-screen-judge-authorization-v1",
        "experiment_id": RUN_ID,
        "decision": "authorized-after-corpus-freeze",
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "scientific_runner_sha256": sha256(Path(__file__)),
        "screen_training_root": SCREEN_TRAINING_ROOT,
        "screen_evaluation_root": SCREEN_EVALUATION_ROOT,
        "confirmation_authorized": False,
        "checked_independent_official_authorized": False,
    }


def verify_screen_authorization(path: Path = SCREEN_AUTHORIZATION_PATH) -> dict[str, Any]:
    # Missing is checked before reading the corpus manifest or touching a root.
    if not path.is_file():
        raise PermissionError("EXP-648 screen Judge authorization manifest is absent")
    manifest_path = DATA_DIR / "corpus-manifest.json"
    if not manifest_path.is_file():
        raise PermissionError("EXP-648 frozen corpus manifest is absent")
    expected = expected_screen_authorization_payload(sha256(manifest_path))
    expected_bytes = canonical_json_bytes(expected, pretty=True)
    observed_bytes = path.read_bytes()
    expected_hash = sha256_bytes(expected_bytes)
    observed_hash = sha256_bytes(observed_bytes)
    if observed_hash != expected_hash or observed_bytes != expected_bytes:
        raise PermissionError("EXP-648 screen Judge authorization hash mismatch")
    return expected


def authorize_root(
    mode: str,
    root_seed: int,
    *,
    authorization_path: Path = SCREEN_AUTHORIZATION_PATH,
) -> int:
    normalized = str(mode).strip().lower()
    seed = int(root_seed)
    if normalized == "test":
        allowed = frozenset((TEST_ONLY_ROOT,))
    elif normalized == "corpus":
        allowed = frozenset((CORPUS_ROOT,))
    elif normalized in ("screen", "screen-training", "screen-evaluation"):
        # Authorization happens before comparing or using a screen root.
        verify_screen_authorization(authorization_path)
        allowed = (
            frozenset((SCREEN_TRAINING_ROOT, SCREEN_EVALUATION_ROOT))
            if normalized == "screen"
            else frozenset(
                (
                    SCREEN_TRAINING_ROOT
                    if normalized == "screen-training"
                    else SCREEN_EVALUATION_ROOT,
                )
            )
        )
    else:
        raise PermissionError(f"EXP-648 mode {normalized!r} is prohibited")
    if seed not in allowed:
        raise PermissionError(
            f"EXP-648 mode {normalized!r} permits only {sorted(allowed)}, not {seed}"
        )
    return seed


def generate_identities(
    mode: str, root_seed: int, start: int, count: int
) -> list[Any]:
    authorize_root(mode, root_seed)
    if start < 0 or count <= 0:
        raise ValueError("identity range must be nonnegative and nonempty")
    rng = random.Random(cal.derive_seed(root_seed, f"identity-block:{start}:{count}"))
    identities = []
    for offset in range(count):
        index = start + offset
        modulus = cal._prime_in_band(rng, 256, index % 4)
        identity = cal.Identity(
            index,
            rng.randrange(modulus),
            rng.randrange(modulus),
            modulus,
        )
        if identity.modulus_bit_length != 256 or identity.band != f"b{index % 4}":
            raise RuntimeError("corpus identity geometry mismatch")
        identities.append(identity)
    return identities


def exact_match_windows(
    failures: Sequence[dict[str, Any]],
    successes: Sequence[dict[str, Any]],
    target_pairs: int,
) -> list[dict[str, Any]]:
    """Linear-time implementation of frozen first-unused exact matching."""

    if target_pairs <= 0:
        raise ValueError("target_pairs must be positive")
    buckets: dict[tuple[int, int, str], deque[dict[str, Any]]] = defaultdict(deque)
    seen_success: set[tuple[int, int]] = set()
    for success in successes:
        cal.validate_window(success, kind="success", k=WINDOW_LENGTH)
        identity_start = (
            int(success["identity_index"]),
            int(success["window_start_index"]),
        )
        if identity_start in seen_success:
            raise RuntimeError("duplicate success window in matching input")
        seen_success.add(identity_start)
        buckets[cal.window_match_key(success)].append(success)
    pairs = []
    for failure in failures:
        cal.validate_window(failure, kind="failure", k=WINDOW_LENGTH)
        key = cal.window_match_key(failure)
        candidates = buckets.get(key)
        if not candidates:
            continue
        success = candidates.popleft()
        pairs.append(
            {
                "pair_index": len(pairs),
                "matching_key": {
                    "window_start_index": key[0],
                    "modulus_bit_length": key[1],
                    "normalized_modulus_band": key[2],
                },
                "failure": failure,
                "success": success,
            }
        )
        if len(pairs) == target_pairs:
            return pairs
    raise RuntimeError(
        f"structured matching shortfall: {len(pairs)} of {target_pairs}; relaxation prohibited"
    )


def trajectory_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **cal._identity_payload(record["identity"]),
        "first_error_index": int(record["first_error_index"]),
        "complete_predicted_states": [int(value) for value in record["predicted_states"]],
        "complete_teacher_states": [int(value) for value in record["teacher_states"]],
    }


def verify_trajectory_record(record: dict[str, Any]) -> dict[str, Any]:
    identity = cal.Identity(
        int(record["identity_index"]),
        int(record["multiplier"]),
        int(record["multiplicand"]),
        int(record["modulus"]),
    )
    if identity.modulus_bit_length != 256:
        raise RuntimeError("corpus modulus is not exactly 256 bits")
    if record.get("normalized_modulus_band") != identity.band:
        raise RuntimeError("corpus normalized modulus band mismatch")
    predicted = [int(value) for value in record["complete_predicted_states"]]
    teacher = [int(value) for value in record["complete_teacher_states"]]
    if len(predicted) != 257 or len(teacher) != 257:
        raise RuntimeError("corpus trajectory length mismatch")
    if predicted[0] != 0 or teacher[0] != 0:
        raise RuntimeError("corpus trajectory lacks the exact zero prefix")
    if any(value < 0 or value >= (1 << 256) for value in predicted):
        raise RuntimeError("corpus source trajectory contains an invalid bit state")
    recomputed = [0]
    quotient_digest_rows = []
    for position in range(256):
        bit = (identity.multiplier >> (255 - position)) & 1
        state, quotient = cal.exact_teacher_step(
            recomputed[-1], identity.multiplicand, identity.modulus, bit
        )
        recomputed.append(state)
        quotient_digest_rows.append(quotient)
    if teacher != recomputed:
        raise RuntimeError("corpus exact teacher trajectory mismatch")
    first_error = cal.first_error_index(predicted[1:], teacher[1:])
    if int(record["first_error_index"]) != first_error:
        raise RuntimeError("corpus first-error index mismatch")
    native = {
        "identity": identity,
        "predicted_states": predicted,
        "teacher_states": teacher,
        "first_error_index": first_error,
    }
    failures, successes = cal.trajectory_windows(native)
    return {
        "native": native,
        "failures": failures,
        "successes": successes,
        "quotient_sha256": sha256_bytes(canonical_json_bytes(quotient_digest_rows)),
    }


def verify_replay_pair(
    stored: Sequence[dict[str, Any]], replayed: Sequence[dict[str, Any]]
) -> None:
    if len(stored) != len(replayed):
        raise RuntimeError("source replay batch length mismatch")
    for left, right in zip(stored, replayed, strict=True):
        left_payload = trajectory_record_payload(left)
        right_payload = trajectory_record_payload(right)
        if canonical_json(left_payload) != canonical_json(right_payload):
            raise RuntimeError("independent full source trajectory replay mismatch")
        verify_trajectory_record(left_payload)


def corpus_workload_payload() -> dict[str, Any]:
    return {
        "schema_version": "exp648-corpus-workload-v1",
        "mode": "corpus",
        "root": CORPUS_ROOT,
        "width": 256,
        "identity_batch_size": IDENTITY_BATCH_SIZE,
        "target_failure_windows": CORPUS_TARGET_PAIRS,
        "target_success_windows": CORPUS_TARGET_PAIRS,
        "target_pairs": CORPUS_TARGET_PAIRS,
        "window_length": WINDOW_LENGTH,
        "full_source_replays_per_identity": 2,
        "matching": "exact-stratum-first-unused-no-reuse-no-relaxation",
        "deterministic_gzip_mtime": 0,
        "screen_training_or_evaluation": False,
    }


def scientific_evaluator_sha256() -> str:
    functions = (
        verify_trajectory_record,
        verify_replay_pair,
        evaluate_transitions,
        evaluate_rollouts,
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "calibration_runner_sha256": CALIBRATION_RUNNER_SHA256,
                "dependency_manifest_sha256": PINNED["dependency_manifest_sha256"],
                "functions": [inspect.getsource(function) for function in functions],
            }
        )
    )


class CorpusGuard:
    def __init__(self, process: Any) -> None:
        self.process = process
        self.started_wall = _MODULE_WALL_STARTED
        # psutil CPU is authoritative; process_time only establishes that timing
        # began before the heavyweight import and is retained in evidence.
        self.started_cpu = max(
            0.0,
            cal.process_cpu_seconds(process)
            - (time.process_time() - _MODULE_PROCESS_CPU_STARTED),
        )
        self.reused_wall_seconds = 0.0
        self.reused_cpu_seconds = 0.0
        self.peak_rss_bytes = cal.process_peak_rss_bytes(process)
        self.checks: list[dict[str, Any]] = []
        self.soft_wall_crossed = False
        self.soft_cpu_crossed = False
        self.quarter_checked = False

    def active(self) -> tuple[float, float]:
        return (
            time.perf_counter() - self.started_wall,
            cal.process_cpu_seconds(self.process) - self.started_cpu,
        )

    def charged(self) -> tuple[float, float]:
        wall, cpu = self.active()
        return wall + self.reused_wall_seconds, cpu + self.reused_cpu_seconds

    def add_reused(self, timing: dict[str, Any], identity_count: int) -> None:
        self.reused_wall_seconds += float(timing["wall_seconds"])
        self.reused_cpu_seconds += float(timing["aggregate_cpu_seconds"])
        self.check("corpus:cache-reuse-charge", identity_count)

    def check(self, stage: str, identity_count: int) -> None:
        wall, cpu = self.charged()
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, cal.process_peak_rss_bytes(self.process)
        )
        self.soft_wall_crossed = self.soft_wall_crossed or wall > SOFT_WALL_SECONDS
        self.soft_cpu_crossed = self.soft_cpu_crossed or cpu > SOFT_CPU_SECONDS
        self.checks.append(
            {
                "stage": stage,
                "identity_trajectories": int(identity_count),
                "charged_wall_seconds": wall,
                "charged_aggregate_cpu_seconds": cpu,
            }
        )
        if identity_count > HARD_IDENTITY_CEILING:
            raise TimeoutError("EXP-648 corpus hard identity ceiling reached")
        if wall >= HARD_WALL_SECONDS:
            raise TimeoutError(f"EXP-648 corpus hard wall ceiling reached during {stage}")
        if cpu >= HARD_CPU_SECONDS:
            raise TimeoutError(f"EXP-648 corpus hard CPU ceiling reached during {stage}")

    def check_quarter(self, matched_pairs: int, identity_count: int) -> None:
        if self.quarter_checked or matched_pairs < QUARTER_PAIRS:
            return
        self.check("corpus:first-quarter:verified", identity_count)
        wall, cpu = self.charged()
        fraction = QUARTER_PAIRS / CORPUS_TARGET_PAIRS
        projected_wall = PROJECTED_WALL_SECONDS * fraction
        projected_cpu = PROJECTED_CPU_SECONDS * fraction
        self.quarter_checked = True
        if wall > QUARTER_SLOWDOWN_LIMIT * projected_wall:
            raise RuntimeError(
                "EXP-648 corpus first-quarter wall throughput is >20% slower; re-budget required"
            )
        if cpu > QUARTER_SLOWDOWN_LIMIT * projected_cpu:
            raise RuntimeError(
                "EXP-648 corpus first-quarter CPU throughput is >20% slower; re-budget required"
            )

    def timing(self) -> dict[str, Any]:
        active_wall, active_cpu = self.active()
        charged_wall, charged_cpu = self.charged()
        return {
            "launcher_plus_active_wall_seconds": active_wall,
            "launcher_plus_active_aggregate_cpu_seconds": active_cpu,
            "reused_wall_seconds_charged": self.reused_wall_seconds,
            "reused_aggregate_cpu_seconds_charged": self.reused_cpu_seconds,
            "charged_wall_seconds": charged_wall,
            "charged_aggregate_cpu_seconds": charged_cpu,
            "peak_rss_bytes": self.peak_rss_bytes,
            "soft_wall_crossed_telemetry_only": self.soft_wall_crossed,
            "soft_cpu_crossed_telemetry_only": self.soft_cpu_crossed,
            "first_quarter_guard_checked": self.quarter_checked,
        }


def cache_key_fields(
    *,
    config: dict[str, Any],
    environment_payload_sha256: str,
    scientific_runner_sha256: str,
) -> dict[str, Any]:
    dataset = {
        "schema_version": "exp648-corpus-dataset-v1",
        "root": CORPUS_ROOT,
        "role": "generated-training-split-corpus-only",
        "width": 256,
        "target_pairs": CORPUS_TARGET_PAIRS,
        "window_length": WINDOW_LENGTH,
        "contains_benchmark_identities": False,
    }
    return {
        "checkpoint_sha256": PINNED["source_checkpoint_sha256"],
        "evaluator_sha256": scientific_evaluator_sha256(),
        "dependency_manifest_sha256": PINNED["dependency_manifest_sha256"],
        "dataset_sha256": sha256_bytes(canonical_json_bytes(dataset)),
        "config_sha256": sha256(CONFIG_PATH),
        "executed_config_sha256": sha256(EXECUTED_CONFIG_PATH),
        "root": CORPUS_ROOT,
        "environment_sha256": environment_payload_sha256,
        "calibration_runner_sha256": CALIBRATION_RUNNER_SHA256,
        "scientific_runner_sha256": scientific_runner_sha256,
        "scientific_protocol_sha256": cal.scientific_protocol_sha256(config),
        "resource_protocol_sha256": cal.resource_protocol_sha256(config),
        "workload_sha256": sha256_bytes(canonical_json_bytes(corpus_workload_payload())),
        "repository_head": PINNED["repository_head"],
    }


def batch_cache_key(fields: dict[str, Any], start: int, count: int) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {"cache_key_fields": fields, "start": int(start), "count": int(count)}
        )
    )


def batch_cache_path(fields: dict[str, Any], start: int, count: int) -> Path:
    root = int(fields["root"])
    authorize_root("test" if root == TEST_ONLY_ROOT else "corpus", root)
    key = batch_cache_key(fields, start, count)
    return SCRATCH_CORPUS_DIR / "cache" / f"batch-{start:04d}-{count}-{key}.json.gz"


def validate_batch_cache(
    record: dict[str, Any], fields: dict[str, Any], start: int, count: int
) -> None:
    root = int(record.get("root", -1))
    mode = "test" if root == TEST_ONLY_ROOT else "corpus"
    authorize_root(mode, root)
    required = {
        "schema_version",
        "role",
        "root",
        "start",
        "count",
        "cache_key_fields",
        "cache_key",
        "trajectories",
        "independent_replay_sha256",
        "timing",
        "complete",
        "core_sha256",
    }
    if set(record) != required:
        raise RuntimeError("corpus batch cache schema mismatch")
    if record["schema_version"] != "exp648-corpus-batch-cache-v1":
        raise RuntimeError("corpus batch cache version mismatch")
    if record["role"] != "generated-training-split-corpus-only":
        raise RuntimeError("corpus batch cache role mismatch")
    if int(record["start"]) != start or int(record["count"]) != count:
        raise RuntimeError("corpus batch cache range mismatch")
    if record["cache_key_fields"] != fields:
        raise RuntimeError("corpus batch cache key fields mismatch")
    if record["cache_key"] != batch_cache_key(fields, start, count):
        raise RuntimeError("corpus batch composite cache key mismatch")
    if record["complete"] is not True:
        raise RuntimeError("corpus batch cache is incomplete")
    core = copy.deepcopy(record)
    core.pop("core_sha256")
    if record["core_sha256"] != sha256_bytes(canonical_json_bytes(core)):
        raise RuntimeError("corpus batch cache content hash mismatch")
    trajectories = record["trajectories"]
    if not isinstance(trajectories, list) or len(trajectories) != count:
        raise RuntimeError("corpus batch trajectory count mismatch")
    expected_identities = generate_identities(mode, root, start, count)
    replay_digest = []
    for offset, (trajectory, identity) in enumerate(
        zip(trajectories, expected_identities, strict=True)
    ):
        if int(trajectory["identity_index"]) != start + offset:
            raise RuntimeError("corpus cached identity order mismatch")
        if cal._identity_payload(identity) != {
            key: trajectory[key]
            for key in (
                "identity_index",
                "multiplier",
                "multiplicand",
                "modulus",
                "modulus_bit_length",
                "normalized_modulus_band",
            )
        }:
            raise RuntimeError("corpus cached identity regeneration mismatch")
        verify_trajectory_record(trajectory)
        replay_digest.append(trajectory)
    if sha256_bytes(canonical_json_bytes(replay_digest)) != record[
        "independent_replay_sha256"
    ]:
        raise RuntimeError("corpus cached replay digest mismatch")
    timing = record["timing"]
    if set(timing) != {
        "wall_seconds",
        "aggregate_cpu_seconds",
        "peak_rss_bytes",
        "mining_wall_seconds",
        "mining_aggregate_cpu_seconds",
        "replay_wall_seconds",
        "replay_aggregate_cpu_seconds",
    }:
        raise RuntimeError("corpus batch cache timing schema mismatch")
    if any(float(timing[key]) < 0 for key in timing if key != "peak_rss_bytes"):
        raise RuntimeError("corpus batch cache timing is invalid")


def validate_integrity_checks(checks: dict[str, Any]) -> bool:
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CORPUS_INTEGRITY_KEYS:
        raise RuntimeError("EXP-648 corpus integrity required-key schema mismatch")
    if not all(type(checks[key]) is bool and checks[key] for key in checks):
        raise RuntimeError("EXP-648 corpus integrity boolean failed")
    return True


class CorpusExperiment:
    def __init__(self) -> None:
        self.root_seed = authorize_root("corpus", CORPUS_ROOT)
        self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.foundation = verify_pinned_foundation(self.config)
        self.process = psutil.Process()
        self.runtime = cal.configure_frozen_runtime(self.config)
        self.environment_base = cal.environment_payload(self.config)
        self.environment_payload_sha256 = sha256_bytes(
            canonical_json_bytes(self.environment_base)
        )
        if self.environment_payload_sha256 != PINNED[
            "calibration_environment_payload_sha256"
        ]:
            raise RuntimeError("current environment differs from calibrated environment")
        self.scientific_runner_sha256 = sha256(Path(__file__))
        self.test_sha256 = (
            sha256(ROOT / "tests/test_exp648_structured_trajectory_scientific.py")
            if (ROOT / "tests/test_exp648_structured_trajectory_scientific.py").is_file()
            else None
        )
        self.dependency_manifest = cal.dependency_source_manifest()
        cal.verify_dependency_source_manifest(self.dependency_manifest)
        self.exp300 = cal.load_verified_dependency_module(
            "exp300_for_exp648_scientific_corpus",
            "exp300_data_evaluator",
            self.dependency_manifest,
        )
        self.cellular = cal.load_verified_dependency_module(
            "cellular_for_exp648_scientific_corpus",
            "cellular_model",
            self.dependency_manifest,
        )
        self.cache_fields = cache_key_fields(
            config=self.config,
            environment_payload_sha256=self.environment_payload_sha256,
            scientific_runner_sha256=self.scientific_runner_sha256,
        )
        self.guard = CorpusGuard(self.process)
        self.source_file_hash_before = sha256(SOURCE_CHECKPOINT_PATH)
        self.source_state_hash_before = cal.state_dict_sha256(self.new_source_model())

    def new_source_model(self):
        authorize_root("corpus", self.root_seed)
        if sha256(SOURCE_CHECKPOINT_PATH) != PINNED["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")
        model = self.cellular.SharedDilatedConvGRU(
            width=256,
            hidden=int(self.config["source_model"]["hidden"]),
            cycles=int(self.config["source_model"]["cycles"]),
        )
        model.load_state_dict(load_file(SOURCE_CHECKPOINT_PATH))
        if sum(parameter.numel() for parameter in model.parameters()) != 99_824:
            raise RuntimeError("source model parameter count mismatch")
        return model.eval()

    def _batch_guard(self, identity_count: int) -> Callable[[str], None]:
        def check(stage: str) -> None:
            self.guard.check(stage.replace("calibration:", "corpus:"), identity_count)

        return check

    def load_or_create_batch(self, start: int, count: int) -> tuple[list[dict[str, Any]], bool]:
        authorize_root("corpus", self.root_seed)
        if start + count > HARD_IDENTITY_CEILING:
            raise TimeoutError("next complete corpus batch would exceed hard identity ceiling")
        path = batch_cache_path(self.cache_fields, start, count)
        if path.is_file():
            encoded = path.read_bytes()
            decoded = deterministic_gunzip_bytes(encoded)
            if deterministic_gzip_bytes(decoded) != encoded:
                raise RuntimeError("corpus batch cache gzip encoding is not canonical")
            record = json.loads(decoded)
            validate_batch_cache(record, self.cache_fields, start, count)
            self.guard.add_reused(record["timing"], start + count)
            return record["trajectories"], True

        identities = generate_identities("corpus", self.root_seed, start, count)
        model = self.new_source_model()
        total_measurement = cal.Measurement(self.process)
        mining = cal.Measurement(self.process)
        mined = cal.rollout_identity_batch(
            model, self.exp300, identities, self._batch_guard(start + count)
        )
        mining_timing = mining.stop().payload()
        self.guard.check("corpus:mining:identity-batch-boundary", start + count)

        replay_model = self.new_source_model()
        replay_measurement = cal.Measurement(self.process)
        replayed = cal.rollout_identity_batch(
            replay_model, self.exp300, identities, self._batch_guard(start + count)
        )
        replay_timing = replay_measurement.stop().payload()
        verify_replay_pair(mined, replayed)
        self.guard.check("corpus:semantic-replay:identity-batch-boundary", start + count)
        total_timing = total_measurement.stop().payload()
        trajectories = [trajectory_record_payload(record) for record in mined]
        record = {
            "schema_version": "exp648-corpus-batch-cache-v1",
            "role": "generated-training-split-corpus-only",
            "root": self.root_seed,
            "start": start,
            "count": count,
            "cache_key_fields": self.cache_fields,
            "cache_key": batch_cache_key(self.cache_fields, start, count),
            "trajectories": trajectories,
            "independent_replay_sha256": sha256_bytes(canonical_json_bytes(trajectories)),
            "timing": {
                "wall_seconds": total_timing["wall_seconds"],
                "aggregate_cpu_seconds": total_timing["aggregate_cpu_seconds"],
                "peak_rss_bytes": total_timing["peak_rss_bytes"],
                "mining_wall_seconds": mining_timing["wall_seconds"],
                "mining_aggregate_cpu_seconds": mining_timing[
                    "aggregate_cpu_seconds"
                ],
                "replay_wall_seconds": replay_timing["wall_seconds"],
                "replay_aggregate_cpu_seconds": replay_timing[
                    "aggregate_cpu_seconds"
                ],
            },
            "complete": True,
        }
        core = copy.deepcopy(record)
        record["core_sha256"] = sha256_bytes(canonical_json_bytes(core))
        encoded = deterministic_gzip_bytes(canonical_json_bytes(record))
        _atomic_write(path, encoded)
        validate_batch_cache(record, self.cache_fields, start, count)
        self.guard.check("corpus:cache:complete-batch-written", start + count)
        return trajectories, False

    def mine(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        trajectories: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []
        cache_records = []
        pairs: list[dict[str, Any]] | None = None
        start = 0
        quarter_seen = False
        while pairs is None:
            batch, reused = self.load_or_create_batch(start, IDENTITY_BATCH_SIZE)
            trajectories.extend(batch)
            for record in batch:
                verified = verify_trajectory_record(record)
                failures.extend(verified["failures"])
                successes.extend(verified["successes"])
            cache_records.append(
                {
                    "start": start,
                    "count": IDENTITY_BATCH_SIZE,
                    "path": batch_cache_path(
                        self.cache_fields, start, IDENTITY_BATCH_SIZE
                    ).relative_to(ROOT).as_posix(),
                    "sha256": sha256(
                        batch_cache_path(self.cache_fields, start, IDENTITY_BATCH_SIZE)
                    ),
                    "reused": reused,
                }
            )
            start += IDENTITY_BATCH_SIZE
            self.guard.check("corpus:matching:batch-boundary", start)
            try:
                quarter_pairs = exact_match_windows(
                    failures, successes, QUARTER_PAIRS
                )
            except RuntimeError as error:
                if "structured matching shortfall" not in str(error):
                    raise
            else:
                if not quarter_seen:
                    quarter_seen = True
                    # Every completed cache batch already has an independent full
                    # source replay, so these 128 pairs have corresponding replay,
                    # teacher, eligibility, and matching verification.
                    if len(quarter_pairs) != QUARTER_PAIRS:
                        raise AssertionError("first-quarter pair count changed")
                    self.guard.check_quarter(len(quarter_pairs), start)
            try:
                pairs = exact_match_windows(
                    failures, successes, CORPUS_TARGET_PAIRS
                )
            except RuntimeError as error:
                if "structured matching shortfall" not in str(error):
                    raise
                pairs = None
        if start > HARD_IDENTITY_CEILING:
            raise TimeoutError("corpus identity ceiling exceeded")
        return trajectories, failures, successes, {
            "pairs": pairs,
            "identity_trajectories": start,
            "cache_records": cache_records,
        }

    def serialize(
        self,
        trajectories: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        successes: list[dict[str, Any]],
        mining: dict[str, Any],
    ) -> dict[str, Any]:
        self.guard.check("corpus:serialization:start", len(trajectories))
        identities = [
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
            for record in trajectories
        ]
        identity_payload = {
            "schema_version": "exp648-identity-manifest-v1",
            "role": "generated-training-split-corpus-only",
            "root": self.root_seed,
            "width": 256,
            "identity_count": len(identities),
            "balanced_band_order": ["b0", "b1", "b2", "b3"],
            "contains_benchmark_identities": False,
            "identities": identities,
        }
        trajectory_payload = {
            "schema_version": "exp648-trajectory-manifest-v1",
            "role": "generated-training-split-corpus-only",
            "root": self.root_seed,
            "width": 256,
            "trajectory_count": len(trajectories),
            "source_checkpoint_sha256": PINNED["source_checkpoint_sha256"],
            "full_independent_source_replay_verified": True,
            "trajectories": trajectories,
        }
        pair_payload = {
            "schema_version": "exp648-paired-corpus-v1",
            "role": "generated-training-split-corpus-only",
            "root": self.root_seed,
            "width": 256,
            "window_length": WINDOW_LENGTH,
            "pair_count": len(mining["pairs"]),
            "matching": {
                "exact_fields": [
                    "window_start_index",
                    "modulus_bit_length",
                    "normalized_modulus_band",
                ],
                "failure_order": "corpus-root-generated-identity-and-trajectory-order",
                "success_tie_order": "first-unused-eligible-success-in-corpus-root-order",
                "success_window_reuse": False,
                "replacement": False,
                "relaxation": False,
                "shortened_window": False,
            },
            "synthetic_state_corruption": False,
            "state_bit_flips_or_injections": False,
            "pairs": mining["pairs"],
        }
        identity_bytes = canonical_json_bytes(identity_payload, pretty=True)
        trajectory_canonical = canonical_json_bytes(trajectory_payload)
        pair_canonical = canonical_json_bytes(pair_payload)
        trajectory_gzip = deterministic_gzip_bytes(trajectory_canonical)
        pair_gzip = deterministic_gzip_bytes(pair_canonical)
        write_mirrored(Path("identity-manifest.json"), identity_bytes, family="data")
        write_mirrored(
            Path("trajectory-manifest.json.gz"), trajectory_gzip, family="data"
        )
        write_mirrored(Path("paired-corpus.json.gz"), pair_gzip, family="data")
        data_hashes = {
            "identity_manifest_file_sha256": sha256_bytes(identity_bytes),
            "identity_manifest_canonical_sha256": sha256_bytes(
                canonical_json_bytes(identity_payload)
            ),
            "trajectory_manifest_canonical_sha256": sha256_bytes(
                trajectory_canonical
            ),
            "trajectory_manifest_gzip_sha256": sha256_bytes(trajectory_gzip),
            "paired_corpus_canonical_sha256": sha256_bytes(pair_canonical),
            "paired_corpus_gzip_sha256": sha256_bytes(pair_gzip),
        }
        corpus_manifest = {
            "schema_version": "exp648-corpus-manifest-v1",
            "experiment_id": RUN_ID,
            "role": "generated-training-split-corpus-only",
            "root": self.root_seed,
            "width": 256,
            "window_length": WINDOW_LENGTH,
            "identity_trajectories": len(trajectories),
            "failure_windows_available": len(failures),
            "success_windows_available": len(successes),
            "failure_windows_frozen": CORPUS_TARGET_PAIRS,
            "success_windows_frozen": CORPUS_TARGET_PAIRS,
            "matched_pairs": CORPUS_TARGET_PAIRS,
            "data_artifacts": {
                "identity_manifest": "data/generated/exp-648-structured-trajectory-consistency/identity-manifest.json",
                "trajectory_manifest": "data/generated/exp-648-structured-trajectory-consistency/trajectory-manifest.json.gz",
                "paired_corpus": "data/generated/exp-648-structured-trajectory-consistency/paired-corpus.json.gz",
            },
            "hashes": data_hashes,
            "data_artifact_set_hash_scope": "canonical-path-to-file-sha256-map-excluding-this-manifest",
            "data_artifact_set_sha256": sha256_bytes(
                canonical_json_bytes(
                    {
                        "identity-manifest.json": data_hashes[
                            "identity_manifest_file_sha256"
                        ],
                        "trajectory-manifest.json.gz": data_hashes[
                            "trajectory_manifest_gzip_sha256"
                        ],
                        "paired-corpus.json.gz": data_hashes[
                            "paired_corpus_gzip_sha256"
                        ],
                    }
                )
            ),
            "provenance": {
                **self.cache_fields,
                "preregistration_sha256": PINNED["preregistration_sha256"],
                "calibration_evidence_sha256": PINNED[
                    "calibration_evidence_sha256"
                ],
                "calibration_evidence_core_sha256": PINNED[
                    "calibration_evidence_core_sha256"
                ],
                "test_sha256": self.test_sha256,
            },
            "constraints": {
                "natural_trajectories_only": True,
                "injection_corruption_or_masking": False,
                "matching_relaxation": False,
                "benchmark_identities": False,
                "model_update": False,
                "scientific_accuracy_or_selection_metrics": False,
            },
        }
        manifest_bytes = canonical_json_bytes(corpus_manifest, pretty=True)
        write_mirrored(Path("corpus-manifest.json"), manifest_bytes, family="data")
        self.guard.check("corpus:serialization:data-complete", len(trajectories))
        return {
            "identity": identity_payload,
            "trajectory": trajectory_payload,
            "paired": pair_payload,
            "manifest": corpus_manifest,
            "corpus_manifest_sha256": sha256_bytes(manifest_bytes),
            "data_hashes": data_hashes,
        }

    def write_environment(self) -> dict[str, Any]:
        payload = {
            "schema_version": "exp648-corpus-environment-v1",
            "calibrated_environment": self.environment_base,
            "calibrated_environment_payload_sha256": self.environment_payload_sha256,
            "scientific_runner_sha256": self.scientific_runner_sha256,
            "calibration_runner_sha256": CALIBRATION_RUNNER_SHA256,
            "repository_head": git_output("rev-parse", "HEAD"),
            "repository_upstream": git_output("rev-parse", "@{upstream}"),
        }
        write_mirrored(
            Path("environment.json"), canonical_json_bytes(payload, pretty=True), family="evidence"
        )
        return payload

    def run(self) -> dict[str, Any]:
        authorize_root("corpus", self.root_seed)
        trajectories, failures, successes, mining = self.mine()
        serialized = self.serialize(trajectories, failures, successes, mining)
        environment = self.write_environment()
        source_file_hash_after = sha256(SOURCE_CHECKPOINT_PATH)
        source_state_hash_after = cal.state_dict_sha256(self.new_source_model())
        if source_file_hash_after != self.source_file_hash_before:
            raise RuntimeError("source checkpoint bytes changed during corpus construction")
        if source_state_hash_after != self.source_state_hash_before:
            raise RuntimeError("source weights changed during corpus construction")
        self.guard.check("corpus:final-bookkeeping", len(trajectories))
        timing = self.guard.timing()
        run_payload = {
            "schema_version": "exp648-corpus-run-v1",
            "experiment_id": RUN_ID,
            "status": "complete-corpus-only-awaiting-screen-judge",
            "role": "generated-training-split-corpus-only",
            "root": self.root_seed,
            "workload": corpus_workload_payload(),
            "counts": {
                "identity_trajectories": len(trajectories),
                "matched_pairs": CORPUS_TARGET_PAIRS,
                "failure_windows": CORPUS_TARGET_PAIRS,
                "success_windows": CORPUS_TARGET_PAIRS,
                "window_length": WINDOW_LENGTH,
            },
            "timing": timing,
            "budgets": {
                "identity_trajectories": {
                    "projected": PROJECTED_IDENTITIES,
                    "soft": SOFT_IDENTITY_BUDGET,
                    "hard": HARD_IDENTITY_CEILING,
                },
                "wall_seconds": {
                    "projected": PROJECTED_WALL_SECONDS,
                    "soft": SOFT_WALL_SECONDS,
                    "hard": HARD_WALL_SECONDS,
                },
                "aggregate_cpu_seconds": {
                    "projected": PROJECTED_CPU_SECONDS,
                    "soft": SOFT_CPU_SECONDS,
                    "hard": HARD_CPU_SECONDS,
                },
                "soft_budgets_are_telemetry_only": True,
                "fidelity_reduced": False,
            },
            "cache": {
                "key_fields": self.cache_fields,
                "complete_batches": mining["cache_records"],
                "reused_complete_batches": sum(
                    bool(record["reused"]) for record in mining["cache_records"]
                ),
                "deterministic_resume_order": True,
                "reused_wall_and_cpu_charged": True,
            },
            "hashes": {
                **serialized["data_hashes"],
                "corpus_manifest_sha256": serialized["corpus_manifest_sha256"],
                "source_checkpoint_file_before_sha256": self.source_file_hash_before,
                "source_checkpoint_file_after_sha256": source_file_hash_after,
                "source_state_before_sha256": self.source_state_hash_before,
                "source_state_after_sha256": source_state_hash_after,
                "environment_canonical_sha256": sha256_bytes(
                    canonical_json_bytes(environment)
                ),
            },
            "authorization_boundary": {
                "corpus_root_opened": True,
                "corpus_root": self.root_seed,
                "calibration_root_opened_by_this_runner": False,
                "screen_roots_opened": False,
                "confirmation_roots_opened": False,
                "exp649_formal_root_opened": False,
                "checked_public_prior_independent_official_private_opened": False,
                "screen_execution_authorized": False,
            },
            "non_scientific_declaration": {
                "training_data_construction_only": True,
                "model_update_performed": False,
                "evaluation_performed": False,
                "scientific_accuracy_reported": False,
                "selection_metric_reported": False,
                "candidate_registration": False,
            },
            "limitations": [
                "This run constructs generated training-split corpus data only and is not scientific accuracy evidence.",
                "Screen execution remains fail-closed pending a separate exact-hash Judge authorization manifest.",
                "No checked, public, prior-independent, official-private, confirmation, or EXP-649 formal surface was opened.",
                "Soft resource thresholds are telemetry; hard count, wall, and process user+system CPU ceilings are runaway guards and did not reduce fidelity.",
            ],
        }
        write_mirrored(
            Path("run.json"), canonical_json_bytes(run_payload, pretty=True), family="evidence"
        )
        return run_payload


def _load_corpus_pairs() -> list[dict[str, Any]]:
    payload = json.loads(
        deterministic_gunzip_bytes((DATA_DIR / "paired-corpus.json.gz").read_bytes())
    )
    if payload.get("schema_version") != "exp648-paired-corpus-v1":
        raise RuntimeError("frozen corpus schema mismatch")
    if int(payload.get("root", -1)) != CORPUS_ROOT:
        raise RuntimeError("frozen corpus root mismatch")
    if int(payload.get("pair_count", -1)) != CORPUS_TARGET_PAIRS:
        raise RuntimeError("frozen corpus pair count mismatch")
    return payload["pairs"]


def ordered_batch_windows(
    pairs: Sequence[dict[str, Any]], update_index: int
) -> list[dict[str, Any]]:
    if len(pairs) != CORPUS_TARGET_PAIRS:
        raise RuntimeError("screen requires the complete frozen 512-pair corpus")
    selected = [pairs[(update_index * 8 + offset) % len(pairs)] for offset in range(8)]
    return [pair["failure"] for pair in selected] + [pair["success"] for pair in selected]


def screen_training_update(
    model: Any,
    optimizer: Any,
    windows: Sequence[dict[str, Any]],
    endpoint_loss_weight: float,
    config: dict[str, Any],
    exp300: Any,
) -> dict[str, Any]:
    if endpoint_loss_weight not in (0.0, 0.1):
        raise RuntimeError("EXP-648 screen endpoint coefficient changed")
    if len(windows) != 16:
        raise RuntimeError("EXP-648 screen training batch must contain 16 windows")
    model.width = 256
    model.train()
    optimizer.zero_grad(set_to_none=True)
    multiplicands = [int(window["multiplicand"]) for window in windows]
    moduli = [int(window["modulus"]) for window in windows]
    autonomous_state = exp300.bits(
        [int(window["teacher_states"][0]) for window in windows], 256
    )
    clean_total = 0.0
    endpoint_logits = None
    for offset in range(WINDOW_LENGTH):
        input_bits = [int(window["input_bits"][offset]) for window in windows]
        clean_states = [int(window["teacher_states"][offset]) for window in windows]
        targets = [int(window["teacher_states"][offset + 1]) for window in windows]
        quotients = torch.tensor(
            [int(window["quotients"][offset]) for window in windows], dtype=torch.long
        )
        clean_logits, clean_qlogits = model(
            exp300.features(clean_states, multiplicands, moduli, input_bits, 256)
        )
        clean_loss, _ = cal.clean_transition_loss(
            clean_logits,
            clean_qlogits,
            exp300.bits(targets, 256),
            quotients,
            worst_bit_weight=float(config["loss"]["worst_bit_weight"]),
            quotient_weight=float(config["loss"]["quotient_weight"]),
        )
        (clean_loss / WINDOW_LENGTH).backward()
        clean_total += float(clean_loss.detach())
        endpoint_logits, _ = model(
            cal.soft_features(
                autonomous_state,
                multiplicands,
                moduli,
                input_bits,
                256,
                exp300.bits,
            )
        )
        autonomous_state = cal.hard_straight_through(endpoint_logits)
    assert endpoint_logits is not None
    endpoint_loss, endpoint_stats = cal.endpoint_residue_loss(
        endpoint_logits,
        exp300.bits([int(window["teacher_states"][-1]) for window in windows], 256),
        worst_bit_weight=float(config["loss"]["worst_bit_weight"]),
    )
    (endpoint_loss * endpoint_loss_weight).backward()
    gradient_norm = float(
        nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip"])
        )
    )
    optimizer.step()
    return {
        "base_clean_loss": clean_total / WINDOW_LENGTH,
        "endpoint_loss": float(endpoint_loss.detach()),
        "endpoint_loss_weight": endpoint_loss_weight,
        "gradient_norm": gradient_norm,
        **{f"endpoint_{key}": value for key, value in endpoint_stats.items()},
    }


def generate_screen_transition_cases(
    root_seed: int, width: int, count: int, authorization_path: Path
) -> dict[str, Any]:
    authorize_root(
        "screen-evaluation", root_seed, authorization_path=authorization_path
    )
    rng = random.Random(cal.derive_seed(root_seed, f"transition:{width}:{count}"))
    primes = [cal._prime_in_band(rng, width, index % 4) for index in range(16)]
    payload = {key: [] for key in ("state", "multiplicand", "modulus", "input_bit", "target", "quotient")}
    for _ in range(count):
        modulus = rng.choice(primes)
        state = rng.randrange(modulus)
        multiplicand = rng.randrange(modulus)
        bit = rng.randrange(2)
        target, quotient = cal.exact_teacher_step(state, multiplicand, modulus, bit)
        for key, value in (
            ("state", state),
            ("multiplicand", multiplicand),
            ("modulus", modulus),
            ("input_bit", bit),
            ("target", target),
            ("quotient", quotient),
        ):
            payload[key].append(value)
    return payload


def generate_screen_rollout_cases(
    root_seed: int, width: int, count: int, authorization_path: Path
) -> dict[str, Any]:
    authorize_root(
        "screen-evaluation", root_seed, authorization_path=authorization_path
    )
    rng = random.Random(cal.derive_seed(root_seed, f"rollout:{width}:{count}"))
    primes = [cal._prime_in_band(rng, width, index % 4) for index in range(16)]
    moduli = [rng.choice(primes) for _ in range(count)]
    return {
        "multiplier": [rng.randrange(modulus) for modulus in moduli],
        "multiplicand": [rng.randrange(modulus) for modulus in moduli],
        "modulus": moduli,
    }


@torch.inference_mode()
def evaluate_transitions(model: Any, width: int, cases: dict[str, Any], exp300: Any) -> dict[str, Any]:
    model.eval()
    model.width = width
    predictions = []
    quotient_predictions = []
    for start in range(0, len(cases["state"]), 128):
        end = min(len(cases["state"]), start + 128)
        logits, quotient_logits = model(
            exp300.features(
                cases["state"][start:end],
                cases["multiplicand"][start:end],
                cases["modulus"][start:end],
                cases["input_bit"][start:end],
                width,
            )
        )
        predictions.extend(cal.decode_logits(logits))
        quotient_predictions.extend(quotient_logits.argmax(dim=1).tolist())
    exact = [
        prediction == target
        for prediction, target in zip(predictions, cases["target"], strict=True)
    ]
    quotient_exact = [
        int(prediction) == int(target)
        for prediction, target in zip(
            quotient_predictions, cases["quotient"], strict=True
        )
    ]
    valid = [
        prediction < modulus
        for prediction, modulus in zip(predictions, cases["modulus"], strict=True)
    ]
    return {
        "cases": len(exact),
        "exact_accuracy": sum(exact) / len(exact),
        "quotient_accuracy": sum(quotient_exact) / len(quotient_exact),
        "valid_residue_rate": sum(valid) / len(valid),
        "exact_vector": exact,
        "output_sha256": sha256_bytes(canonical_json_bytes(predictions)),
    }


@torch.inference_mode()
def evaluate_rollouts(model: Any, width: int, cases: dict[str, Any], exp300: Any) -> dict[str, Any]:
    model.eval()
    model.width = width
    multiplier = list(cases["multiplier"])
    multiplicand = list(cases["multiplicand"])
    moduli = list(cases["modulus"])
    state = [0] * len(multiplier)
    teacher = [0] * len(multiplier)
    first_errors: list[int | None] = [None] * len(multiplier)
    invalid = [False] * len(multiplier)
    for position in range(width):
        bits = [(value >> (width - 1 - position)) & 1 for value in multiplier]
        logits, _ = model(exp300.features(state, multiplicand, moduli, bits, width))
        state = cal.decode_logits(logits)
        teacher = [
            cal.exact_teacher_step(previous, value, modulus, bit)[0]
            for previous, value, modulus, bit in zip(
                teacher, multiplicand, moduli, bits, strict=True
            )
        ]
        for index, (prediction, expected, modulus) in enumerate(
            zip(state, teacher, moduli, strict=True)
        ):
            if first_errors[index] is None and prediction != expected:
                first_errors[index] = position
            invalid[index] = invalid[index] or prediction >= modulus
    exact = [left == right for left, right in zip(state, teacher, strict=True)]
    clipped = [width + 1 if value is None else value for value in first_errors]
    return {
        "cases": len(exact),
        "exact_accuracy": sum(exact) / len(exact),
        "exact_vector": exact,
        "mean_clipped_first_error_index": sum(clipped) / len(clipped),
        "invalid_rollout_rate": sum(invalid) / len(invalid),
        "output_sha256": sha256_bytes(canonical_json_bytes(state)),
    }


class ScreenExperiment:
    """Complete screen path; construction itself requires exact Judge bytes."""

    def __init__(self, authorization_path: Path = SCREEN_AUTHORIZATION_PATH) -> None:
        # This must remain the first operation: no config/data/model/root work may
        # happen before exact authorization has passed.
        self.authorization = verify_screen_authorization(authorization_path)
        self.authorization_path = authorization_path
        self.training_root = authorize_root(
            "screen-training",
            SCREEN_TRAINING_ROOT,
            authorization_path=authorization_path,
        )
        self.evaluation_root = authorize_root(
            "screen-evaluation",
            SCREEN_EVALUATION_ROOT,
            authorization_path=authorization_path,
        )
        self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        verify_pinned_foundation(self.config)
        self.runtime = cal.configure_frozen_runtime(self.config)
        self.dependencies = cal.dependency_source_manifest()
        cal.verify_dependency_source_manifest(self.dependencies)
        self.evaluator_sha256 = scientific_evaluator_sha256()
        self.exp300 = cal.load_verified_dependency_module(
            "exp300_for_exp648_screen", "exp300_data_evaluator", self.dependencies
        )
        self.cellular = cal.load_verified_dependency_module(
            "cellular_for_exp648_screen", "cellular_model", self.dependencies
        )

    def new_model(self):
        authorize_root(
            "screen-training",
            self.training_root,
            authorization_path=self.authorization_path,
        )
        model = self.cellular.SharedDilatedConvGRU(width=256, hidden=92, cycles=2)
        model.load_state_dict(load_file(SOURCE_CHECKPOINT_PATH))
        return model

    def run(self) -> dict[str, Any]:
        pairs = _load_corpus_pairs()
        models = {"control": self.new_model(), "challenger": self.new_model()}
        initial = {name: cal.state_dict_sha256(model) for name, model in models.items()}
        if len(set(initial.values())) != 1:
            raise RuntimeError("screen matched-arm initialization mismatch")
        optimizers = {
            name: torch.optim.AdamW(
                model.parameters(),
                lr=float(self.config["training"]["learning_rate"]),
                weight_decay=float(self.config["training"]["weight_decay"]),
            )
            for name, model in models.items()
        }
        histories = {}
        batch_hashes = []
        updates = int(self.config["screen"]["updates_per_arm"])
        for arm in ("control", "challenger"):
            authorize_root(
                "screen-training",
                self.training_root,
                authorization_path=self.authorization_path,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[arm],
                T_max=updates,
                eta_min=float(self.config["training"]["minimum_learning_rate"]),
            )
            history = []
            local_hashes = []
            for update in range(updates):
                windows = ordered_batch_windows(pairs, update)
                local_hashes.append(sha256_bytes(canonical_json_bytes(windows)))
                history.append(
                    screen_training_update(
                        models[arm],
                        optimizers[arm],
                        windows,
                        float(self.config["matched_arms"][arm]["endpoint_loss_weight"]),
                        self.config,
                        self.exp300,
                    )
                )
                scheduler.step()
            histories[arm] = history
            batch_hashes.append(local_hashes)
        if batch_hashes[0] != batch_hashes[1]:
            raise RuntimeError("screen arms did not receive identical batch order")

        evaluation = {"transitions": {}, "rollouts": {}}
        spec = self.config["screen"]["evaluation"]
        for width, count in spec["transition_cases"].items():
            cases = generate_screen_transition_cases(
                self.evaluation_root,
                int(width),
                int(count),
                self.authorization_path,
            )
            evaluation["transitions"][str(width)] = {
                arm: evaluate_transitions(model, int(width), cases, self.exp300)
                for arm, model in models.items()
            }
        for width, count in spec["full_rollout_cases"].items():
            cases = generate_screen_rollout_cases(
                self.evaluation_root,
                int(width),
                int(count),
                self.authorization_path,
            )
            evaluation["rollouts"][str(width)] = {
                arm: evaluate_rollouts(model, int(width), cases, self.exp300)
                for arm, model in models.items()
            }
        repeat_cases = generate_screen_rollout_cases(
            self.evaluation_root,
            int(spec["exact_repeat_width"]),
            int(spec["exact_repeat_cases"]),
            self.authorization_path,
        )
        exact_repeat = {}
        for arm, model in models.items():
            first = evaluate_rollouts(
                model, int(spec["exact_repeat_width"]), repeat_cases, self.exp300
            )
            second = evaluate_rollouts(
                model, int(spec["exact_repeat_width"]), repeat_cases, self.exp300
            )
            exact_repeat[arm] = {
                "first": first,
                "second": second,
                "deterministic": first["output_sha256"] == second["output_sha256"],
            }
        width256_drift = cal.paired_drift_counts(
            evaluation["rollouts"]["256"]["control"]["exact_vector"],
            evaluation["rollouts"]["256"]["challenger"]["exact_vector"],
        )
        arm_metrics = {}
        for arm in ("control", "challenger"):
            arm_metrics[arm] = {
                "width64_rollout_accuracy": evaluation["rollouts"]["64"][arm][
                    "exact_accuracy"
                ],
                "width128_rollout_accuracy": evaluation["rollouts"]["128"][arm][
                    "exact_accuracy"
                ],
                "width256_rollout_accuracy": evaluation["rollouts"]["256"][arm][
                    "exact_accuracy"
                ],
                "mean_clipped_first_error_index": evaluation["rollouts"]["256"][
                    arm
                ]["mean_clipped_first_error_index"],
                "width256_transition_accuracy": evaluation["transitions"]["256"][
                    arm
                ]["exact_accuracy"],
                "invalid_rollout_rate": evaluation["rollouts"]["256"][arm][
                    "invalid_rollout_rate"
                ],
            }
        checkpoint_dir = (
            ROOT / ".scratch/exp-648-structured-trajectory-consistency/screen/checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_hashes = {}
        for arm, model in models.items():
            path = checkpoint_dir / f"{arm}.safetensors"
            save_file(model.state_dict(), path)
            checkpoint_hashes[arm] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "state_dict_sha256": cal.state_dict_sha256(model),
            }
        integrity = {
            "source_checkpoint_hash_valid": sha256(SOURCE_CHECKPOINT_PATH)
            == PINNED["source_checkpoint_sha256"],
            "identical_initialization_valid": len(set(initial.values())) == 1,
            "parameter_count_valid": all(
                sum(parameter.numel() for parameter in model.parameters()) == 99_824
                for model in models.values()
            ),
            "paired_data_order_valid": batch_hashes[0] == batch_hashes[1],
            "seed_roles_valid": (
                self.training_root == SCREEN_TRAINING_ROOT
                and self.evaluation_root == SCREEN_EVALUATION_ROOT
            ),
            "checkpoint_hashes_valid": all(
                sha256(ROOT / value["path"]) == value["sha256"]
                for value in checkpoint_hashes.values()
            ),
            "evaluator_hash_valid": scientific_evaluator_sha256()
            == self.evaluator_sha256,
            "dependency_manifest_valid": self.dependencies["manifest_sha256"]
            == PINNED["dependency_manifest_sha256"],
            "matched_work_valid": batch_hashes[0] == batch_hashes[1],
            "control_exact_repeat_deterministic": exact_repeat["control"][
                "deterministic"
            ],
            "challenger_exact_repeat_deterministic": exact_repeat["challenger"][
                "deterministic"
            ],
        }
        if not cal.validate_scientific_integrity(integrity):
            raise RuntimeError("EXP-648 screen scientific integrity failed")
        gates = cal.screen_gate_results(
            arm_metrics["control"],
            arm_metrics["challenger"],
            integrity,
            self.config["screen"]["gates"],
        )
        return {
            "schema_version": "exp648-screen-result-v1",
            "training_root": self.training_root,
            "evaluation_root": self.evaluation_root,
            "initial_tensor_sha256": initial,
            "ordered_batch_sha256": sha256_bytes(canonical_json_bytes(batch_hashes[0])),
            "histories": histories,
            "evaluation": evaluation,
            "exact_repeat": exact_repeat,
            "arm_metrics": arm_metrics,
            "paired_width256_drift_counts": width256_drift,
            "checkpoint_hashes": checkpoint_hashes,
            "scientific_integrity": integrity,
            "gates": gates,
            "decision": "pass-intermediate-requires-separate-confirmation-judge"
            if gates["all_passed"]
            else "reject-frozen-screen",
        }


def _read_data_artifacts() -> dict[str, Any]:
    identity_bytes = (DATA_DIR / "identity-manifest.json").read_bytes()
    trajectory_gzip = (DATA_DIR / "trajectory-manifest.json.gz").read_bytes()
    pair_gzip = (DATA_DIR / "paired-corpus.json.gz").read_bytes()
    manifest_bytes = (DATA_DIR / "corpus-manifest.json").read_bytes()
    if deterministic_gzip_bytes(deterministic_gunzip_bytes(trajectory_gzip)) != trajectory_gzip:
        raise RuntimeError("trajectory manifest gzip is not deterministic")
    if deterministic_gzip_bytes(deterministic_gunzip_bytes(pair_gzip)) != pair_gzip:
        raise RuntimeError("paired corpus gzip is not deterministic")
    return {
        "identity_bytes": identity_bytes,
        "trajectory_gzip": trajectory_gzip,
        "pair_gzip": pair_gzip,
        "manifest_bytes": manifest_bytes,
        "identity": json.loads(identity_bytes),
        "trajectory": json.loads(deterministic_gunzip_bytes(trajectory_gzip)),
        "paired": json.loads(deterministic_gunzip_bytes(pair_gzip)),
        "manifest": json.loads(manifest_bytes),
    }


def verify_corpus_artifacts(*, require_receipts: bool = True) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    verify_pinned_foundation(config)
    artifacts = _read_data_artifacts()
    identities = artifacts["identity"]["identities"]
    trajectories = artifacts["trajectory"]["trajectories"]
    pairs = artifacts["paired"]["pairs"]
    if len(identities) != len(trajectories):
        raise RuntimeError("identity/trajectory manifest count mismatch")
    failures = []
    successes = []
    for expected_index, (identity, trajectory) in enumerate(
        zip(identities, trajectories, strict=True)
    ):
        if int(identity["identity_index"]) != expected_index:
            raise RuntimeError("identity manifest order mismatch")
        if identity != {
            key: trajectory[key]
            for key in (
                "identity_index",
                "multiplier",
                "multiplicand",
                "modulus",
                "modulus_bit_length",
                "normalized_modulus_band",
            )
        }:
            raise RuntimeError("identity/trajectory linkage mismatch")
        verified = verify_trajectory_record(trajectory)
        failures.extend(verified["failures"])
        successes.extend(verified["successes"])
    expected_pairs = exact_match_windows(failures, successes, CORPUS_TARGET_PAIRS)
    if canonical_json(expected_pairs) != canonical_json(pairs):
        raise RuntimeError("serialized paired corpus fails deterministic rematching")
    used = [
        (int(pair["success"]["identity_index"]), int(pair["success"]["window_start_index"]))
        for pair in pairs
    ]
    if len(used) != len(set(used)):
        raise RuntimeError("serialized paired corpus reuses a success window")
    manifest = artifacts["manifest"]
    hashes = manifest["hashes"]
    observed = {
        "identity_manifest_file_sha256": sha256_bytes(artifacts["identity_bytes"]),
        "identity_manifest_canonical_sha256": sha256_bytes(
            canonical_json_bytes(artifacts["identity"])
        ),
        "trajectory_manifest_canonical_sha256": sha256_bytes(
            deterministic_gunzip_bytes(artifacts["trajectory_gzip"])
        ),
        "trajectory_manifest_gzip_sha256": sha256_bytes(artifacts["trajectory_gzip"]),
        "paired_corpus_canonical_sha256": sha256_bytes(
            deterministic_gunzip_bytes(artifacts["pair_gzip"])
        ),
        "paired_corpus_gzip_sha256": sha256_bytes(artifacts["pair_gzip"]),
    }
    if hashes != observed:
        raise RuntimeError("corpus manifest artifact hash mismatch")
    for name in (
        "identity-manifest.json",
        "trajectory-manifest.json.gz",
        "paired-corpus.json.gz",
        "corpus-manifest.json",
    ):
        if (DATA_DIR / name).read_bytes() != (SCRATCH_CORPUS_DIR / "data" / name).read_bytes():
            raise RuntimeError(f"raw/tracked corpus parity mismatch for {name}")
    if require_receipts:
        for name in ("run.json", "environment.json", "integrity.json", "mlflow.json"):
            tracked = TRACKED_CORPUS_DIR / name
            raw = SCRATCH_CORPUS_DIR / "evidence" / name
            if not tracked.is_file() or not raw.is_file() or tracked.read_bytes() != raw.read_bytes():
                raise RuntimeError(f"raw/tracked evidence parity mismatch for {name}")
        integrity = json.loads((TRACKED_CORPUS_DIR / "integrity.json").read_text())
        validate_integrity_checks(integrity["checks"])
        mlflow_receipt = json.loads((TRACKED_CORPUS_DIR / "mlflow.json").read_text())
        if mlflow_receipt.get("verified", {}).get("no_accuracy_or_selection_metrics") is not True:
            raise RuntimeError("MLflow receipt scientific-metric boundary mismatch")
    return {
        "identity_trajectories": len(trajectories),
        "matched_pairs": len(pairs),
        "corpus_manifest_sha256": sha256_bytes(artifacts["manifest_bytes"]),
        "semantic_verification": True,
        "deterministic_rematching": True,
        "gzip_determinism": True,
        "raw_tracked_byte_parity": True,
    }


def _mlflow_artifact_sources() -> list[tuple[Path, str]]:
    return [
        (DATA_DIR / "identity-manifest.json", "corpus"),
        (DATA_DIR / "trajectory-manifest.json.gz", "corpus"),
        (DATA_DIR / "paired-corpus.json.gz", "corpus"),
        (DATA_DIR / "corpus-manifest.json", "corpus"),
        (TRACKED_CORPUS_DIR / "run.json", "corpus-evidence"),
        (TRACKED_CORPUS_DIR / "environment.json", "corpus-evidence"),
    ]


def log_and_verify_mlflow(
    run_payload: dict[str, Any], *, guard: CorpusGuard | None = None
) -> dict[str, Any]:
    identity_count = int(run_payload["counts"]["identity_trajectories"])
    if guard is not None:
        guard.check("corpus:mlflow:preflight", identity_count)
        run_payload["timing"] = guard.timing()
        write_mirrored(
            Path("run.json"),
            canonical_json_bytes(run_payload, pretty=True),
            family="evidence",
        )
    experiment_id = cal.configure_mlflow(run_payload.get("tracking_experiment", "sair-modular-arithmetic-wsl"))
    client = cal.mlflow.tracking.MlflowClient(tracking_uri=cal.tracking_uri())
    corpus_manifest_sha = run_payload["hashes"]["corpus_manifest_sha256"]
    matches = client.search_runs(
        experiment_ids=[str(experiment_id)],
        filter_string=(
            f"tags.canonical_run_id = '{RUN_ID}' and "
            "tags.role = 'corpus-only-non-scientific-training-data'"
        ),
        max_results=20,
    )
    matching = [
        run
        for run in matches
        if run.data.tags.get("corpus_manifest_sha256") == corpus_manifest_sha
    ]
    if len(matching) > 1:
        raise RuntimeError("duplicate completed EXP-648 corpus MLflow runs")
    reused = False
    if matching:
        active = matching[0]
        if active.info.status != "FINISHED":
            raise RuntimeError("matching EXP-648 corpus MLflow run is not FINISHED")
        run_id = active.info.run_id
        reused = True
    else:
        with mlflow.start_run(run_name=f"{RUN_ID}-corpus-only") as active_run:
            run_id = active_run.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": RUN_ID,
                    "status": "complete-corpus-only-awaiting-screen-judge",
                    "role": "corpus-only-non-scientific-training-data",
                    "root_role": "corpus-root-generated-training-split-only",
                    "corpus_root": str(CORPUS_ROOT),
                    "corpus_manifest_sha256": corpus_manifest_sha,
                    "model_update": "false",
                    "benchmark_opened": "false",
                    "scientific_accuracy_reported": "false",
                    "selection_metric_reported": "false",
                    "candidate_registration": "false",
                }
            )
            mlflow.log_params(
                {
                    "corpus_root": CORPUS_ROOT,
                    "target_pairs": CORPUS_TARGET_PAIRS,
                    "window_length": WINDOW_LENGTH,
                    "source_checkpoint_sha256": PINNED["source_checkpoint_sha256"],
                    "scientific_runner_sha256": run_payload["cache"]["key_fields"][
                        "scientific_runner_sha256"
                    ],
                    "calibration_runner_sha256": CALIBRATION_RUNNER_SHA256,
                    "corpus_manifest_sha256": corpus_manifest_sha,
                }
            )
            if guard is not None:
                guard.check("corpus:mlflow:tags-and-params", identity_count)
            timing = run_payload["timing"]
            mlflow.log_metrics(
                {
                    "identity_trajectories": run_payload["counts"][
                        "identity_trajectories"
                    ],
                    "matched_pairs": run_payload["counts"]["matched_pairs"],
                    "charged_wall_seconds": timing["charged_wall_seconds"],
                    "charged_aggregate_cpu_seconds": timing[
                        "charged_aggregate_cpu_seconds"
                    ],
                    "launcher_plus_active_wall_seconds": timing[
                        "launcher_plus_active_wall_seconds"
                    ],
                    "launcher_plus_active_aggregate_cpu_seconds": timing[
                        "launcher_plus_active_aggregate_cpu_seconds"
                    ],
                    "reused_wall_seconds_charged": timing[
                        "reused_wall_seconds_charged"
                    ],
                    "reused_aggregate_cpu_seconds_charged": timing[
                        "reused_aggregate_cpu_seconds_charged"
                    ],
                    "peak_rss_bytes": float(timing["peak_rss_bytes"]),
                }
            )
            for source, artifact_path in _mlflow_artifact_sources():
                mlflow.log_artifact(str(source), artifact_path=artifact_path)
                if guard is not None:
                    guard.check(
                        f"corpus:mlflow:artifact:{source.name}", identity_count
                    )

    run = client.get_run(run_id)
    if guard is not None:
        guard.check("corpus:mlflow:query", identity_count)
    if run.info.status != "FINISHED":
        raise RuntimeError("EXP-648 corpus MLflow run did not finish")
    forbidden_metric_fragments = ("accuracy", "selection", "exact", "gain", "delta")
    if any(
        fragment in metric.lower()
        for metric in run.data.metrics
        for fragment in forbidden_metric_fragments
    ):
        raise RuntimeError("scientific accuracy or selection metric leaked to corpus MLflow run")
    required_tags = {
        "canonical_run_id": RUN_ID,
        "role": "corpus-only-non-scientific-training-data",
        "corpus_manifest_sha256": corpus_manifest_sha,
        "model_update": "false",
        "benchmark_opened": "false",
        "scientific_accuracy_reported": "false",
        "selection_metric_reported": "false",
    }
    if any(run.data.tags.get(key) != value for key, value in required_tags.items()):
        raise RuntimeError("EXP-648 corpus MLflow tag verification failed")
    verify_dir = SCRATCH_CORPUS_DIR / "mlflow-verification" / run_id
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    verified_artifacts = []
    for source, artifact_path in _mlflow_artifact_sources():
        remote = f"{artifact_path}/{source.name}"
        downloaded = Path(client.download_artifacts(run_id, remote, str(verify_dir)))
        if sha256(downloaded) != sha256(source):
            raise RuntimeError(f"MLflow artifact hash mismatch for {remote}")
        verified_artifacts.append(
            {
                "path": remote,
                "source_path": source.relative_to(ROOT).as_posix(),
                "source_sha256": sha256(source),
                "downloaded_sha256": sha256(downloaded),
                "verified": True,
            }
        )
        if guard is not None:
            guard.check(
                f"corpus:mlflow:verify-artifact:{source.name}", identity_count
            )
    if guard is not None:
        guard.check("corpus:mlflow:complete", identity_count)
    return {
        "schema_version": "exp648-corpus-mlflow-receipt-v1",
        "experiment_id": str(experiment_id),
        "experiment_name": "sair-modular-arithmetic-wsl",
        "tracking_uri": cal.tracking_uri(),
        "run_id": run_id,
        "run_name": run.info.run_name,
        "run_status": run.info.status,
        "reused_matching_completed_run": reused,
        "verified": {
            "tags": True,
            "params": True,
            "metrics": dict(run.data.metrics),
            "no_accuracy_or_selection_metrics": True,
            "artifacts": verified_artifacts,
            "final_resource_timing": guard.timing() if guard is not None else None,
        },
        "limitations": [
            "This MLflow run records corpus construction resources and provenance only, not scientific accuracy or selection evidence.",
            "No model update or benchmark evaluation was performed.",
        ],
    }


def finalize_corpus_evidence(
    run_payload: dict[str, Any], *, guard: CorpusGuard | None = None
) -> dict[str, Any]:
    mlflow_receipt = log_and_verify_mlflow(run_payload, guard=guard)
    write_mirrored(
        Path("mlflow.json"),
        canonical_json_bytes(mlflow_receipt, pretty=True),
        family="evidence",
    )
    pre_integrity_files = {
        "data/identity-manifest.json": sha256(DATA_DIR / "identity-manifest.json"),
        "data/trajectory-manifest.json.gz": sha256(DATA_DIR / "trajectory-manifest.json.gz"),
        "data/paired-corpus.json.gz": sha256(DATA_DIR / "paired-corpus.json.gz"),
        "data/corpus-manifest.json": sha256(DATA_DIR / "corpus-manifest.json"),
        "evidence/run.json": sha256(TRACKED_CORPUS_DIR / "run.json"),
        "evidence/environment.json": sha256(TRACKED_CORPUS_DIR / "environment.json"),
        "evidence/mlflow.json": sha256(TRACKED_CORPUS_DIR / "mlflow.json"),
    }
    checks = {key: True for key in REQUIRED_CORPUS_INTEGRITY_KEYS}
    validate_integrity_checks(checks)
    integrity = {
        "schema_version": "exp648-corpus-integrity-v1",
        "checks": checks,
        "artifact_set_hash_scope": (
            "canonical-path-to-file-sha256-map-of-all-required-corpus-artifacts-"
            "except-this-self-referential-integrity-receipt"
        ),
        "artifact_hashes": pre_integrity_files,
        "artifact_set_sha256": sha256_bytes(canonical_json_bytes(pre_integrity_files)),
        "scientific_runner_sha256": sha256(Path(__file__)),
        "scientific_test_sha256": sha256(
            ROOT / "tests/test_exp648_structured_trajectory_scientific.py"
        ),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT_PATH),
        "repository_head": git_output("rev-parse", "HEAD"),
        "corpus_manifest_sha256": sha256(DATA_DIR / "corpus-manifest.json"),
    }
    write_mirrored(
        Path("integrity.json"),
        canonical_json_bytes(integrity, pretty=True),
        family="evidence",
    )
    verification = verify_corpus_artifacts(require_receipts=True)
    return {
        "run": run_payload,
        "mlflow": mlflow_receipt,
        "integrity": integrity,
        "verification": verification,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("corpus", "screen", "verify-corpus"), required=True
    )
    parser.add_argument("--root", type=int)
    parser.add_argument("--authorization-manifest", type=Path, default=SCREEN_AUTHORIZATION_PATH)
    args = parser.parse_args(argv)
    if args.mode == "screen":
        # Fail before constructing ScreenExperiment or reading any root/data/model.
        verify_screen_authorization(args.authorization_manifest)
        root = SCREEN_TRAINING_ROOT if args.root is None else args.root
        authorize_root("screen", root, authorization_path=args.authorization_manifest)
        result = ScreenExperiment(args.authorization_manifest).run()
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if args.mode == "verify-corpus":
        if args.root is not None:
            raise PermissionError("read-only corpus verification accepts no root")
        result = verify_corpus_artifacts(require_receipts=True)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    root = CORPUS_ROOT if args.root is None else args.root
    authorize_root("corpus", root)
    if root != CORPUS_ROOT:
        raise PermissionError("EXP-648 corpus root changed")
    experiment = CorpusExperiment()
    run_payload = experiment.run()
    result = finalize_corpus_evidence(run_payload, guard=experiment.guard)
    print(
        json.dumps(
            {
                "status": run_payload["status"],
                "root": CORPUS_ROOT,
                "identity_trajectories": run_payload["counts"][
                    "identity_trajectories"
                ],
                "matched_pairs": run_payload["counts"]["matched_pairs"],
                "timing": run_payload["timing"],
                "corpus_manifest_sha256": result["verification"][
                    "corpus_manifest_sha256"
                ],
                "mlflow_run_id": result["mlflow"]["run_id"],
                "scientific_accuracy_reported": False,
                "screen_executed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
