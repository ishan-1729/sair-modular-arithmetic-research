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
import ast
import contextlib
import copy
import fcntl
import gzip
import hashlib
import importlib.util
import inspect
import io
import json
import math
import random
import re
import os
import shutil
import stat
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
HISTORICAL_READINESS_CALIBRATION_ROOT = 20_261_161
PREREGISTERED_READINESS_CALIBRATION_ROOT = 20_261_162
CONSUMED_REJECTED_READINESS_CALIBRATION_ROOTS = frozenset(
    (20_261_159, 20_261_160, HISTORICAL_READINESS_CALIBRATION_ROOT)
)


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

ROOT_20261159_SUPERSEDED_CHARGE = {
    "root": 20_261_159,
    "authorization_sha256": "d3aa35008f646edf5171771056cea523e60d0415083d58970cf5efadc7492a96",
    "receipt_sha256": "cd27fabbd47683677c9b6465db00444445569ce32a0b8e8a595a42d11da82981",
    "core_sha256": "dae882cefc44cdff89ca4b3a4c18a6ebd55e1415a62fb9bd323e1f1b65f0dd6f",
    "resource_completion_sha256": "9eeb228450ba8e684adbe34bb95ad057036ffdedc643fe3942a0aaf846cf1ab0",
    "wall_seconds": 3107.4547687239974,
    "aggregate_cpu_seconds": 12442.915271225,
    "eligible_for_projection_rates": False,
    "eligible_for_launch": False,
    "eligible_for_cache_reuse": False,
    "classification": "consumed-complete-components-but-chronology-nonconforming",
}
ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE = {
    "root": 20_261_160,
    "wall_seconds": 5294.4,
    "aggregate_cpu_seconds": 21000.0,
    "semantics": "conservative-accounting-allowance-never-an-exact-internal-observation",
    "exact_internal_observation": False,
    "completed_work_claimed": False,
}
READINESS_COMPLETED_HISTORY_CHARGE = {
    "wall_seconds": 9007.853947323996,
    "aggregate_cpu_seconds": 36721.035879589996,
}
READINESS_CARRIED_CHARGE = {
    "wall_seconds": 14302.253947323995,
    "aggregate_cpu_seconds": 57721.035879589996,
}
READINESS_CUMULATIVE_RUNAWAY_CEILINGS = {
    "wall_seconds": 25000.0,
    "aggregate_cpu_seconds": 100000.0,
    "scope": "independent-cumulative-runaway-protection-only-never-per-attempt-or-fidelity-cap",
}
READINESS_SUPERSEDED_CALIBRATIONS = (
    *cal.PRIOR_SUPERSEDED_CALIBRATIONS,
    ROOT_20261159_SUPERSEDED_CHARGE,
)
READINESS_SUPERSEDED_CALIBRATION_CHARGE = {
    "wall_seconds": float(cal.PRIOR_SUPERSEDED_CALIBRATION_CHARGE["wall_seconds"])
    + ROOT_20261159_SUPERSEDED_CHARGE["wall_seconds"],
    "aggregate_cpu_seconds": float(
        cal.PRIOR_SUPERSEDED_CALIBRATION_CHARGE["aggregate_cpu_seconds"]
    )
    + ROOT_20261159_SUPERSEDED_CHARGE["aggregate_cpu_seconds"],
}

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
TRACKED_SCREEN_DIR = ROOT / "experiments/runs/exp-648-structured-trajectory-consistency/screen"
SCRATCH_SCREEN_DIR = ROOT / ".scratch/exp-648-structured-trajectory-consistency/screen"
SCREEN_READINESS_CALIBRATION_PATH = TRACKED_SCREEN_DIR / "readiness-calibration.json"
READINESS_RESOURCE_COMPLETION_PATH = (
    TRACKED_SCREEN_DIR / "readiness-calibration-resource-completion.json"
)
READINESS_CALIBRATION_AUTHORIZATION_PATH = TRACKED_SCREEN_DIR / "readiness-calibration-authorization.json"
READINESS_SCAFFOLD_PROVENANCE_PATH = (
    ROOT / "experiments/runs/exp-648-structured-trajectory-consistency/readiness-scaffold/provenance.json"
)
ROOT_20261159_SUPERSEDED_DIR = (
    TRACKED_SCREEN_DIR / "superseded/root-20261159"
)
ROOT_20261159_CLASSIFICATION_PATH = (
    ROOT_20261159_SUPERSEDED_DIR / "classification.json"
)
ROOT_20261160_SUPERSEDED_DIR = (
    TRACKED_SCREEN_DIR / "superseded/root-20261160"
)
ROOT_20261160_CLASSIFICATION_PATH = (
    ROOT_20261160_SUPERSEDED_DIR / "classification.json"
)
MAX_AUTHORIZATION_BYTES = 65_536

FROZEN_CORPUS_FILES = {
    "data/identity-manifest.json": "b7143c486979c7667a24e5c3419774b5420240fc1bffe9e2cc039526adf97ae7",
    "data/trajectory-manifest.json.gz": "22de663d23b8b799b28104b3b62597fadd7d37fd64395f61643ca4fad91d910a",
    "data/paired-corpus.json.gz": "51a61aa3f98bfa7cebfe98f5e9ba620219bc1ef9624177b0b947341abbae86c0",
    "data/corpus-manifest.json": "1bf059fb97cc31d4aac3075c53a4d16918afddc8cff6a35143953a44449df12d",
    "evidence/run.json": "de34592118392d2b1a158697b78a35c0d0e130ead36f33e2069e8715d2056a09",
    "evidence/environment.json": "cab308ec1c6148ee4e5a68764809dc2be5114f23026cf4f4dbc4d4572e89a957",
    "evidence/integrity.json": "a0a00d09379cf77dab8a0c89c1b6796fffb9f238021409ab9f473b102ecd24fe",
    "evidence/mlflow.json": "9fbac562324f141d76f8b7e255ff54490072f7e72ffa777e0570466cea1d4a3f",
}

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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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


def screen_workload_payload(
    *, scientific_runner_sha256: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config_override or yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": "exp648-screen-workload-v2",
        "training": {"updates_per_arm": 200, "arms": 2, "batch_size": 16, "window_length": 8},
        "evaluation": json.loads(canonical_json(config["screen"]["evaluation"])),
        "frozen_corpus_semantic_replay": {"identity_trajectories": 736, "full_source_replays": 1},
        "checkpoint_before_evaluation": {"writes": 2, "reload_verify": True},
        "recoverable_evaluation_units": "paired-arm-width-mode",
        "recoverable_evaluation_unit_count": 7,
        "successful_attempt_ledger_atomic_writes": 213,
        "successful_atomic_evidence_writes": 19,
        "evidence_and_mlflow_verification": True,
        "fidelity_reduction_allowed": False,
        "scientific_runner_sha256": scientific_runner_sha256 or sha256(Path(__file__)),
    }


def _git_blob(commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT, check=False, capture_output=True)
    if result.returncode != 0:
        raise PermissionError(f"approved base lacks {relative}")
    return result.stdout


def _git_blob_sha256(commit: str, relative: str) -> str:
    return sha256_bytes(_git_blob(commit, relative))


def _git_json(commit: str, relative: str) -> dict[str, Any]:
    raw = _git_blob(commit, relative)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PermissionError(f"approved base has malformed JSON at {relative}") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value, pretty=True):
        raise PermissionError(f"approved base has noncanonical JSON at {relative}")
    return value


def _git_function_source(commit: str, relative: str, name: str) -> str:
    source = _git_blob(commit, relative).decode("utf-8")
    tree = ast.parse(source)
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise PermissionError(f"approved base function binding missing or ambiguous: {name}")
    node = matches[0]
    return "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def _base_protocol_hashes(commit: str, config: dict[str, Any]) -> tuple[str, str]:
    scientific = copy.deepcopy(config)
    for key in ("status", "phase", "resource_protocol", "authorization"):
        scientific.pop(key, None)
    scientific.get("corpus", {}).pop("resolved_corpus_mining_ceiling", None)
    resource = copy.deepcopy(config["resource_protocol"])
    resource.pop("status", None)
    resource.pop("resolved", None)
    resource.get("projections", {}).pop("resolved", None)
    return (
        sha256_bytes(canonical_json_bytes(scientific)),
        sha256_bytes(canonical_json_bytes(resource)),
    )


def _base_exact_teacher_hash(commit: str) -> str:
    relative = CALIBRATION_RUNNER.relative_to(ROOT).as_posix()
    return sha256_bytes(_git_function_source(commit, relative, "exact_teacher_step").encode())


def _base_evaluator_hash(commit: str, dependency_manifest_sha256: str) -> str:
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    functions = [
        _git_function_source(commit, relative, name)
        for name in (
            "verify_trajectory_record", "verify_replay_pair",
            "evaluate_transitions", "evaluate_rollouts",
        )
    ]
    return sha256_bytes(canonical_json_bytes({
        "calibration_runner_sha256": _git_blob_sha256(
            commit, CALIBRATION_RUNNER.relative_to(ROOT).as_posix()
        ),
        "dependency_manifest_sha256": dependency_manifest_sha256,
        "functions": functions,
    }))


def validate_one_child_authorization_lifecycle(
    *, auth_relative: str, approved_base_commit: str, launch: bool,
) -> str:
    auth_commit = git_output("log", "-1", "--format=%H", "--", auth_relative)
    if not auth_commit:
        raise PermissionError("authorization file has no committed history")
    if subprocess.run(["git", "merge-base", "--is-ancestor", approved_base_commit, auth_commit], cwd=ROOT).returncode != 0:
        raise PermissionError("approved authorization base is not an ancestor")
    if git_output("rev-list", "--count", f"{approved_base_commit}..{auth_commit}") != "1":
        raise PermissionError("authorization must be one child commit above its base")
    if git_output("diff", "--name-only", f"{approved_base_commit}..{auth_commit}").splitlines() != [auth_relative]:
        raise PermissionError("authorization child commit changes files outside its manifest")
    if launch:
        if git_output("rev-parse", "HEAD") != auth_commit:
            raise PermissionError("launch HEAD is not the authorization child commit")
        if git_output("rev-parse", "HEAD") != git_output("rev-parse", "@{upstream}"):
            raise PermissionError("launch authorization commit is not upstream")
        if git_output("status", "--porcelain=v1", "--untracked-files=all"):
            raise PermissionError("authorization launch requires a clean worktree")
    return auth_commit


def _validate_readiness_authorization_source(
    provenance: dict[str, Any], approved_base_commit: str
) -> dict[str, Any]:
    """Validate the frozen approved-base source without rebuilding its semantics."""
    source = provenance.get("readiness_calibration_authorization_source")
    required_source_keys = {
        "schema_version", "approved_base_file_bindings", "screen_workload",
        "screen_workload_sha256", "authorization_template",
        "authorization_template_sha256",
    }
    if not isinstance(source, dict) or set(source) != required_source_keys:
        raise PermissionError("approved-base authorization source key schema mismatch")
    if source["schema_version"] != "exp648-readiness-authorization-source-v1":
        raise PermissionError("approved-base authorization source schema mismatch")

    bindings = source["approved_base_file_bindings"]
    expected_binding_names = {
        "scientific_runner", "scientific_test", "resolved_config",
        "calibration_runner", "executed_config", "runtime_environment",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_names:
        raise PermissionError("approved-base direct file binding schema mismatch")
    expected_paths = {
        "scientific_runner": "experiments/exp_648_structured_trajectory_consistency/scientific.py",
        "scientific_test": "tests/test_exp648_structured_trajectory_scientific.py",
        "resolved_config": "configs/exp_648_structured_trajectory_consistency.yaml",
        "calibration_runner": "experiments/exp_648_structured_trajectory_consistency/run.py",
        "executed_config": "experiments/runs/exp-648-structured-trajectory-consistency/executed-config.yaml",
        "runtime_environment": "experiments/runs/exp-648-structured-trajectory-consistency/environment.json",
    }
    observed_blobs: dict[str, bytes] = {}
    for name, expected_path in expected_paths.items():
        binding = bindings.get(name)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise PermissionError(f"approved-base {name} binding schema mismatch")
        if binding["path"] != expected_path or not isinstance(binding["sha256"], str):
            raise PermissionError(f"approved-base {name} binding path/hash mismatch")
        raw = _git_blob(approved_base_commit, expected_path)
        observed_blobs[name] = raw
        if sha256_bytes(raw) != binding["sha256"]:
            raise PermissionError(f"approved-base {name} direct hash mismatch")

    workload = source["screen_workload"]
    required_workload_keys = {
        "schema_version", "training", "evaluation",
        "frozen_corpus_semantic_replay", "checkpoint_before_evaluation",
        "recoverable_evaluation_units", "recoverable_evaluation_unit_count",
        "successful_attempt_ledger_atomic_writes",
        "successful_atomic_evidence_writes", "evidence_and_mlflow_verification",
        "fidelity_reduction_allowed", "scientific_runner_sha256",
    }
    if not isinstance(workload, dict) or set(workload) != required_workload_keys:
        raise PermissionError("approved-base screen workload key schema mismatch")
    if workload["schema_version"] != "exp648-screen-workload-v2":
        raise PermissionError("approved-base screen workload schema mismatch")
    workload_sha256 = sha256_bytes(canonical_json_bytes(workload))
    if source["screen_workload_sha256"] != workload_sha256:
        raise PermissionError("approved-base screen workload hash mismatch")
    if provenance.get("screen_workload_sha256") != workload_sha256:
        raise PermissionError("approved-base provenance workload binding mismatch")
    if workload["scientific_runner_sha256"] != bindings["scientific_runner"]["sha256"]:
        raise PermissionError("approved-base workload runner binding mismatch")

    template = source["authorization_template"]
    required_template_keys = {
        "schema_version", "experiment_id", "decision", "scientific_runner_sha256",
        "scientific_test_sha256", "calibration_runner_sha256",
        "resolved_config_sha256", "executed_config_sha256",
        "source_checkpoint_sha256", "dependency_manifest_sha256",
        "approved_runtime_environment", "evaluator_sha256",
        "exact_teacher_source_sha256", "scientific_protocol_sha256",
        "resource_protocol_sha256", "screen_workload_sha256",
        "corpus_manifest_sha256", "screen_roots_authorized",
        "scientific_accuracy_or_selection_authorized",
    }
    if not isinstance(template, dict) or set(template) != required_template_keys:
        raise PermissionError("approved-base authorization template key schema mismatch")
    if template["schema_version"] != "exp648-screen-readiness-calibration-authorization-v1":
        raise PermissionError("approved-base authorization template schema mismatch")
    if template["decision"] != "authorize-non-scientific-screen-readiness-calibration-only":
        raise PermissionError("approved-base authorization template decision mismatch")
    if template["screen_roots_authorized"] is not False or template["scientific_accuracy_or_selection_authorized"] is not False:
        raise PermissionError("approved-base authorization template scope expanded")
    template_sha256 = sha256_bytes(canonical_json_bytes(template))
    if source["authorization_template_sha256"] != template_sha256:
        raise PermissionError("approved-base authorization template hash mismatch")

    template_file_bindings = {
        "scientific_runner_sha256": bindings["scientific_runner"]["sha256"],
        "scientific_test_sha256": bindings["scientific_test"]["sha256"],
        "calibration_runner_sha256": bindings["calibration_runner"]["sha256"],
        "resolved_config_sha256": bindings["resolved_config"]["sha256"],
        "executed_config_sha256": bindings["executed_config"]["sha256"],
    }
    for key, expected in template_file_bindings.items():
        if template.get(key) != expected:
            raise PermissionError(f"approved-base template/provenance binding mismatch for {key}")
    if template.get("screen_workload_sha256") != workload_sha256:
        raise PermissionError("approved-base template/provenance workload mismatch")
    try:
        environment = json.loads(observed_blobs["runtime_environment"])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PermissionError("approved-base runtime environment is malformed") from error
    if (
        not isinstance(environment, dict)
        or observed_blobs["runtime_environment"] != canonical_json_bytes(environment, pretty=True)
        or template.get("approved_runtime_environment") != environment
    ):
        raise PermissionError("approved-base template/runtime environment binding mismatch")
    if provenance.get("live_scaffold") != {
        "focused_scientific_test_sha256": bindings["scientific_test"]["sha256"],
        "resolved_config_sha256": bindings["resolved_config"]["sha256"],
        "scientific_runner_sha256": bindings["scientific_runner"]["sha256"],
    }:
        raise PermissionError("approved-base template/live scaffold binding mismatch")
    if template.get("experiment_id") != provenance.get("experiment_id"):
        raise PermissionError("approved-base template/provenance experiment mismatch")
    for key, value in template.items():
        if key.endswith("_sha256") and (not isinstance(value, str) or len(value) != 64):
            raise PermissionError(f"approved-base authorization template hash invalid for {key}")
    return copy.deepcopy(template)


def readiness_calibration_authorization_base(
    root_seed: int, approved_base_commit: str
) -> dict[str, Any]:
    """Return one frozen approved-base template plus its only two dynamic values."""
    provenance = _git_json(
        approved_base_commit,
        "experiments/runs/exp-648-structured-trajectory-consistency/readiness-scaffold/provenance.json",
    )
    template = _validate_readiness_authorization_source(
        provenance, approved_base_commit
    )
    return {
        **template,
        "calibration_root": int(root_seed),
        "approved_base_commit": approved_base_commit,
    }


def _read_authenticated_committed_manifest(
    path: Path, *, expected_schema: str, committed_bytes: bytes | None = None
) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise PermissionError(f"{expected_schema} manifest is absent")
    size = path.stat().st_size
    if size <= 0 or size > MAX_AUTHORIZATION_BYTES:
        raise PermissionError("authorization manifest size is invalid")
    observed = path.read_bytes()
    try:
        payload = json.loads(observed)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PermissionError("authorization manifest is malformed") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != expected_schema:
        raise PermissionError("authorization manifest schema mismatch")
    if observed != canonical_json_bytes(payload, pretty=True):
        raise PermissionError("authorization manifest is noncanonical")
    if committed_bytes is None:
        try:
            relative = path.resolve().relative_to(ROOT).as_posix()
        except ValueError as error:
            raise PermissionError("authorization manifest must be inside repository") from error
        result = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT,
            check=False, capture_output=True,
        )
        if result.returncode != 0:
            raise PermissionError("authorization manifest is not committed at HEAD")
        committed_bytes = result.stdout
    if committed_bytes != observed:
        raise PermissionError("authorization bytes differ from committed Git blob")
    return payload, observed


def readiness_calibration_publication_state(
    *, live_receipt: Path = SCREEN_READINESS_CALIBRATION_PATH,
    superseded_classification: Path = ROOT_20261159_CLASSIFICATION_PATH,
) -> dict[str, Any]:
    if live_receipt.is_file():
        return {
            "status": "live-receipt-present",
            "live_receipt_sha256": sha256(live_receipt),
            "superseded_root_20261159_preserved": superseded_classification.is_file(),
        }
    if not superseded_classification.is_file():
        return {
            "status": "no-live-receipt-and-no-superseded-classification",
            "live_receipt_sha256": None,
            "superseded_root_20261159_preserved": False,
        }
    raw = superseded_classification.read_bytes()
    try:
        classification = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PermissionError("superseded readiness classification is malformed") from error
    if raw != canonical_json_bytes(classification, pretty=True):
        raise PermissionError("superseded readiness classification is noncanonical")
    required = {
        "schema_version": "exp648-readiness-superseded-classification-v1",
        "root": 20_261_159,
        "status": "consumed-complete-components-but-chronology-nonconforming",
        "superseded": True,
        "eligible_for_launch": False,
        "eligible_for_projection_rates": False,
        "eligible_for_cache_reuse": False,
    }
    for key, expected in required.items():
        if classification.get(key) != expected:
            raise PermissionError(
                f"superseded readiness classification mismatch for {key}"
            )
    archived_receipt = ROOT_20261159_SUPERSEDED_DIR / "readiness-calibration.json"
    if (
        not archived_receipt.is_file()
        or sha256(archived_receipt)
        != ROOT_20261159_SUPERSEDED_CHARGE["receipt_sha256"]
    ):
        raise PermissionError("superseded readiness receipt preservation mismatch")
    return {
        "status": "superseded-only-no-live-receipt",
        "live_receipt_sha256": None,
        "superseded_root_20261159_preserved": True,
        "classification_sha256": sha256(superseded_classification),
        "superseded_receipt_sha256": sha256(archived_receipt),
    }


def verify_readiness_calibration_authorization(
    path: Path = READINESS_CALIBRATION_AUTHORIZATION_PATH, *,
    committed_bytes: bytes | None = None, allow_test_root: bool = False,
    launch: bool = True,
) -> dict[str, Any]:
    # Authentication and cheap structural checks happen before any config,
    # environment, model, corpus, output-directory, RNG, root, or MLflow action.
    payload, observed = _read_authenticated_committed_manifest(
        path,
        expected_schema="exp648-screen-readiness-calibration-authorization-v1",
        committed_bytes=committed_bytes,
    )
    expected_keys = {
        "schema_version", "experiment_id", "decision", "calibration_root",
        "approved_base_commit", "scientific_runner_sha256", "scientific_test_sha256",
        "calibration_runner_sha256", "resolved_config_sha256", "executed_config_sha256",
        "source_checkpoint_sha256", "dependency_manifest_sha256", "approved_runtime_environment", "evaluator_sha256",
        "exact_teacher_source_sha256", "scientific_protocol_sha256",
        "resource_protocol_sha256", "screen_workload_sha256", "corpus_manifest_sha256",
        "screen_roots_authorized", "scientific_accuracy_or_selection_authorized",
    }
    if set(payload) != expected_keys:
        raise PermissionError("calibration authorization key schema mismatch")
    root = int(payload["calibration_root"])
    prohibited = {
        CALIBRATION_ROOT, CORPUS_ROOT, SCREEN_TRAINING_ROOT, SCREEN_EVALUATION_ROOT,
        EXP649_FORMAL_ROOT, *CONFIRMATION_ROOTS,
        *CONSUMED_REJECTED_READINESS_CALIBRATION_ROOTS,
    }
    allowed_root = (
        root == PREREGISTERED_READINESS_CALIBRATION_ROOT
        or (root == HISTORICAL_READINESS_CALIBRATION_ROOT and not launch)
        or (root == TEST_ONLY_ROOT and allow_test_root)
    )
    historically_verifiable = root == HISTORICAL_READINESS_CALIBRATION_ROOT and not launch
    if (root in prohibited and not historically_verifiable) or root <= 0 or not allowed_root:
        raise PermissionError("calibration authorization root is reused, test-only, or prohibited")
    if payload["decision"] != "authorize-non-scientific-screen-readiness-calibration-only":
        raise PermissionError("calibration authorization decision mismatch")
    if payload["screen_roots_authorized"] is not False or payload["scientific_accuracy_or_selection_authorized"] is not False:
        raise PermissionError("calibration authorization scope expanded")
    # Only after the committed bytes authenticate do we inspect relevant files.
    base = str(payload["approved_base_commit"])
    expected = readiness_calibration_authorization_base(root, base)
    if payload != expected:
        raise PermissionError("calibration authorization relevant-file binding mismatch")
    auth_relative = path.resolve().relative_to(ROOT).as_posix()
    validate_one_child_authorization_lifecycle(
        auth_relative=auth_relative, approved_base_commit=base, launch=launch
    )
    return {**payload, "authorization_sha256": sha256_bytes(observed)}


def verify_screen_readiness_calibration(
    path: Path = SCREEN_READINESS_CALIBRATION_PATH,
) -> dict[str, Any]:
    if not path.is_file():
        lifecycle = readiness_calibration_publication_state(live_receipt=path)
        raise PermissionError(
            "fresh post-repair full-fidelity screen calibration is absent; "
            f"publication lifecycle is {lifecycle['status']}"
        )
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != canonical_json_bytes(payload, pretty=True):
        raise PermissionError("screen readiness calibration receipt is noncanonical")
    required = {
        "schema_version": "exp648-screen-readiness-calibration-v1",
        "status": "complete-non-scientific-full-fidelity-resource-calibration",
        "scientific_runner_sha256": sha256(Path(__file__)),
        "resolved_config_sha256": sha256(CONFIG_PATH),
        "screen_workload_sha256": sha256_bytes(canonical_json_bytes(screen_workload_payload())),
        "full_fidelity": True,
        "warmups": 1,
        "timed_repeats": 3,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise PermissionError(f"screen readiness calibration mismatch for {key}")
    root = int(payload.get("calibration_root", -1))
    if root <= 0 or root in {
        CALIBRATION_ROOT, CORPUS_ROOT, SCREEN_TRAINING_ROOT, SCREEN_EVALUATION_ROOT,
        EXP649_FORMAL_ROOT, *CONFIRMATION_ROOTS,
        *CONSUMED_REJECTED_READINESS_CALIBRATION_ROOTS,
    }:
        raise PermissionError("fresh screen calibration root is absent, reused, or prohibited")
    if not isinstance(payload.get("calibration_authorization_sha256"), str) or len(payload["calibration_authorization_sha256"]) != 64:
        raise PermissionError("screen calibration lacks separate authorization binding")
    budgets = payload.get("resolved_screen_budgets", {})
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        record = budgets.get(resource, {})
        if not (0 < float(record.get("projected_seconds", 0)) < float(record.get("soft_budget_seconds", 0)) < float(record.get("hard_runaway_ceiling_seconds", 0))):
            raise PermissionError(f"invalid dynamic {resource} budget")
    if int(budgets.get("peak_rss_guard_bytes", 0)) <= 0:
        raise PermissionError("invalid peak RSS guard")
    quarter = budgets.get("first_quarter_checkpoint", {})
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        if float(quarter.get(resource, {}).get("expected_elapsed_seconds", 0)) <= 0:
            raise PermissionError(f"missing measured first-quarter {resource} expectation")
    for key in ("calibration_evidence_core_sha256", "environment_sha256", "mlflow_receipt_sha256"):
        if not isinstance(payload.get(key), str) or len(payload[key]) != 64:
            raise PermissionError(f"screen calibration lacks {key}")
    try:
        validate_readiness_calibration_receipt(payload)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise PermissionError("screen readiness calibration receipt failed recomputation") from error
    companion = {
        "core": TRACKED_SCREEN_DIR / "readiness-calibration-core.json",
        "environment": TRACKED_SCREEN_DIR / "readiness-calibration-environment.json",
        "mlflow": TRACKED_SCREEN_DIR / "readiness-calibration-mlflow.json",
        "completion": TRACKED_SCREEN_DIR / "readiness-calibration-completion.json",
        "attempt_state": TRACKED_SCREEN_DIR / "readiness-calibration-attempt-state.json",
        "launch_ledger": TRACKED_SCREEN_DIR / "readiness-calibration-launch-ledger.json",
        "resource_completion": READINESS_RESOURCE_COMPLETION_PATH,
        "executed_config": TRACKED_SCREEN_DIR / "readiness-calibration-executed-config.yaml",
    }
    if any(not value.is_file() for value in companion.values()):
        raise PermissionError("screen readiness calibration companion evidence is incomplete")
    if sha256(companion["core"]) != payload["calibration_evidence_core_sha256"]:
        raise PermissionError("screen readiness calibration core hash mismatch")
    if sha256(companion["environment"]) != payload["environment_sha256"]:
        raise PermissionError("screen readiness calibration environment hash mismatch")
    if sha256(companion["mlflow"]) != payload["mlflow_receipt_sha256"]:
        raise PermissionError("screen readiness calibration MLflow hash mismatch")
    for key, item in companion.items():
        if key == "executed_config":
            continue
        parsed = json.loads(item.read_bytes())
        if item.read_bytes() != canonical_json_bytes(parsed, pretty=True):
            raise PermissionError("screen readiness calibration companion is noncanonical")
    completion = json.loads(companion["completion"].read_text(encoding="utf-8"))
    expected_artifacts = {
        "readiness-calibration.json": sha256(path),
        "readiness-calibration-core.json": sha256(companion["core"]),
        "readiness-calibration-environment.json": sha256(companion["environment"]),
        "readiness-calibration-mlflow.json": sha256(companion["mlflow"]),
        "readiness-calibration-executed-config.yaml": sha256(companion["executed_config"]),
    }
    if set(completion) != {"schema_version", "attempt_id", "authorization_sha256", "artifacts", "complete"} or completion.get("schema_version") != "exp648-readiness-calibration-completion-v1" or completion.get("complete") is not True or completion.get("artifacts") != expected_artifacts:
        raise PermissionError("screen readiness calibration completion seal mismatch")
    authorization = verify_readiness_calibration_authorization(
        READINESS_CALIBRATION_AUTHORIZATION_PATH, launch=False
    )
    if authorization["authorization_sha256"] != payload["calibration_authorization_sha256"]:
        raise PermissionError("screen readiness calibration authorization hash mismatch")
    core = json.loads(companion["core"].read_text(encoding="utf-8"))
    if completion["attempt_id"] != core.get("attempt_id") or completion["authorization_sha256"] != authorization["authorization_sha256"]:
        raise PermissionError("screen readiness completion attempt/auth mismatch")
    if core.get("authorization") != authorization:
        raise PermissionError("screen readiness calibration core provenance mismatch")
    try:
        validate_core_receipt_release_equality(core, payload)
    except RuntimeError as error:
        raise PermissionError("screen readiness calibration core/receipt mismatch") from error
    executed_config = yaml.safe_load(companion["executed_config"].read_text(encoding="utf-8"))
    if int(executed_config["resource_protocol"]["calibration"]["root"]) != int(payload["calibration_root"]):
        raise PermissionError("screen readiness executed root config mismatch")
    try:
        for repeat in payload["repeats"]:
            cal.validate_calibration_repeat(repeat, executed_config)
        cal.verify_dependency_source_manifest(cal.dependency_source_manifest())
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise PermissionError("screen readiness detailed repeat/dependency verification failed") from error
    environment = json.loads(companion["environment"].read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cal.configure_frozen_runtime(config)
    current_environment = cal.environment_payload(config)
    if environment != current_environment or core.get("environment") != environment:
        raise PermissionError("screen launch environment differs from fresh calibration")
    mlflow_receipt = json.loads(companion["mlflow"].read_text(encoding="utf-8"))
    if not all(mlflow_receipt.get(key) is True for key in (
        "exact_tags_verified", "exact_params_verified", "exact_metrics_verified"
    )) or mlflow_receipt.get("status") != "FINISHED":
        raise PermissionError("screen readiness calibration MLflow verification mismatch")
    verify_readiness_calibration_mlflow_at_launch(
        payload, mlflow_receipt, companion["core"], companion["environment"], companion["executed_config"]
    )
    inner_paths = {
        "readiness-calibration.json": path,
        "readiness-calibration-core.json": companion["core"],
        "readiness-calibration-environment.json": companion["environment"],
        "readiness-calibration-mlflow.json": companion["mlflow"],
        "readiness-calibration-executed-config.yaml": companion["executed_config"],
        "readiness-calibration-completion.json": companion["completion"],
        "readiness-calibration-attempt-state.json": companion["attempt_state"],
        "readiness-calibration-launch-ledger.json": companion["launch_ledger"],
    }
    try:
        outer = load_and_validate_readiness_resource_completion(
            companion["resource_completion"], inner_paths=inner_paths,
            expected_attempt_id=core["attempt_id"],
            expected_authorization_sha256=authorization["authorization_sha256"],
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise PermissionError("screen readiness observed resource completion mismatch") from error
    verified = copy.deepcopy(payload)
    verified["prepublication_provisional_recovery_charges"] = verified.pop("recovery_charges")
    verified["recovery_charges"] = outer["recovery_charges"]
    verified["observed_resource_completion"] = outer
    return verified


def _calibration_component_rates(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(repeats) != 3:
        raise RuntimeError("readiness calibration requires exactly three timed repeats")
    rates: dict[str, Any] = {}
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        rates[resource] = {
            "startup": max(float(r["components"]["startup_model_loading"]["timing"][resource]) for r in repeats),
            "mining_per_pair": max(float(r["components"]["mining"]["timing"][resource]) / 64 for r in repeats),
            "semantic_replay_per_window": max(float(r["components"]["corpus_integrity_validation"]["timing"][resource]) / 128 for r in repeats),
            "actual_source_replay_per_complete_identity": max(float(r["components"]["actual_source_replay"]["timing"][resource]) / 64 for r in repeats),
            "training_per_update_slower_arm": max(
                float(r["components"]["training"][arm]["timing"][resource]) / 20
                for r in repeats for arm in ("control", "challenger")
            ),
            "checkpoint_per_write_slower_arm": max(
                float(r["components"]["checkpoint"][arm]["timing"][resource])
                for r in repeats for arm in ("control", "challenger")
            ),
            "transition_per_case": {
                str(width): max(
                    float(r["components"]["transitions"][arm][str(width)]["timing"][resource]) / 200
                    for r in repeats for arm in ("control", "challenger")
                ) for width in (128, 256)
            },
            "rollout_per_case": {
                str(width): max(
                    float(r["components"]["rollouts"][arm][str(width)]["timing"][resource]) / 20
                    for r in repeats for arm in ("control", "challenger")
                ) for width in (64, 128, 256)
            },
            "serialization": max(float(r["components"]["hash_serialization"]["timing"][resource]) for r in repeats),
            "readiness_infrastructure_complete_unit": max(float(r["components"]["readiness_infrastructure"]["timing"][resource]) for r in repeats),
            "attempt_ledger_per_write": max(float(r["components"]["readiness_infrastructure"]["component_timing"]["attempt_ledger"][resource]) / 213 for r in repeats),
            "evaluation_cache_per_unit": max(float(r["components"]["readiness_infrastructure"]["component_timing"]["evaluation_cache"][resource]) / 7 for r in repeats),
            "atomic_evidence_per_write": max(float(r["components"]["readiness_infrastructure"]["component_timing"]["atomic_evidence"][resource]) / 19 for r in repeats),
            "atomic_checkpoint_reload_both": max(float(r["components"]["readiness_infrastructure"]["component_timing"]["atomic_checkpoints"][resource]) for r in repeats),
            "mlflow_remote_six_artifact_probe": max(float(r["components"]["mlflow_probe"]["timing"][resource]) for r in repeats),
        }
    return rates


def resolve_readiness_screen_budgets(
    repeats: Sequence[dict[str, Any]], launcher: dict[str, Any],
    warmup: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rates = _calibration_component_rates(repeats)
    resolved: dict[str, Any] = {}
    quarter: dict[str, Any] = {}
    components_by_resource: dict[str, Any] = {}
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        rate = rates[resource]
        components = {
            "launcher_import_authorization_environment": float(launcher[resource]),
            "startup": rate["startup"],
            "frozen_corpus_source_replay": rate["actual_source_replay_per_complete_identity"] * 736,
            "training_both_arms": rate["training_per_update_slower_arm"] * 200 * 2,
            "checkpoints_both_arms": rate["checkpoint_per_write_slower_arm"] * 2,
            "transitions_both_arms": sum(rate["transition_per_case"][str(w)] * 2000 * 2 for w in (128, 256)),
            "rollouts_both_arms": sum(rate["rollout_per_case"][str(w)] * 200 * 2 for w in (64, 128, 256)),
            "exact_repeat_both_passes_both_arms": rate["rollout_per_case"]["256"] * 20 * 2 * 2,
            "attempt_ledger_213_writes": rate["attempt_ledger_per_write"] * 213,
            "evaluation_cache_7_write_validate_units": rate["evaluation_cache_per_unit"] * 7,
            "atomic_evidence_19_writes": rate["atomic_evidence_per_write"] * 19,
            "atomic_checkpoint_reload_verification": rate["atomic_checkpoint_reload_both"],
            "verified_remote_mlflow_six_artifact_run": rate["mlflow_remote_six_artifact_probe"],
        }
        projection = sum(components.values())
        setup_headroom = 2 * (
            components["launcher_import_authorization_environment"] + components["startup"]
            + components["checkpoints_both_arms"] + components["atomic_checkpoint_reload_verification"]
            + rate["serialization"] + rate["atomic_evidence_per_write"]
        )
        soft = 1.25 * projection + setup_headroom
        hard = 1.5 * soft
        resolved[resource] = {
            "projected_seconds": projection,
            "explicit_measured_setup_headroom_seconds": setup_headroom,
            "soft_budget_seconds": soft,
            "hard_runaway_ceiling_seconds": hard,
        }
        quarter_expected = (
            components["launcher_import_authorization_environment"] + components["startup"]
            + components["frozen_corpus_source_replay"]
            + rate["training_per_update_slower_arm"] * 50 * 2
            + rate["attempt_ledger_per_write"] * 54
        )
        quarter[resource] = {"expected_elapsed_seconds": quarter_expected}
        components_by_resource[resource] = components
    peak_candidates = [int(launcher["peak_rss_bytes"])] + [
        int(r["total_timing"]["peak_rss_bytes"]) for r in repeats
    ]
    if warmup is not None:
        peak_candidates.append(int(warmup["total_timing"]["peak_rss_bytes"]))
    peak = max(peak_candidates)
    resolved["peak_rss_guard_bytes"] = max(peak + 512 * 1024 * 1024, math.ceil(peak * 1.25))
    resolved["peak_rss_formula"] = "max(max_complete_repeat_peak_rss+512MiB,1.25*max_complete_repeat_peak_rss)"
    resolved["first_quarter_checkpoint"] = quarter
    return resolved, {
        "rates": rates, "projected_components": components_by_resource,
        "rss_peak_inputs_bytes": peak_candidates,
        "rss_peak_max_bytes": peak,
        "superseded_rate_exclusions": {
            "root_20261159": {
                "classification": ROOT_20261159_SUPERSEDED_CHARGE["classification"],
                "receipt_sha256": ROOT_20261159_SUPERSEDED_CHARGE["receipt_sha256"],
                "eligible_for_projection_rates": False,
                "eligible_for_launch": False,
                "eligible_for_cache_reuse": False,
                "wall_seconds_preserved_for_conservative_planning_context_only": (
                    ROOT_20261159_SUPERSEDED_CHARGE["wall_seconds"]
                ),
                "aggregate_cpu_seconds_preserved_for_conservative_planning_context_only": (
                    ROOT_20261159_SUPERSEDED_CHARGE["aggregate_cpu_seconds"]
                ),
            },
            "rate_source": "only the three chronology-conforming current-root augmented repeats",
        },
        "timing_aggregation_contract": {
            "complete_unit_total": "base CalibrationExperiment total + actual source replay + MLflow probe + readiness infrastructure",
            "launcher": "excluded from each warmup/repeat total; charged once in projection and active process accounting",
            "peak_rss": "maximum of launcher, complete warmup, and every complete repeat peak; never summed",
        },
    }


READINESS_RELEASE_CRITICAL_FIELDS = (
    "warmup",
    "repeats",
    "component_resolution",
    "resolved_screen_budgets",
    "recovery_charges",
    "full_fidelity",
    "environment",
    "boundary",
    "chronology",
)


def aggregate_readiness_timing(
    base_timing: dict[str, Any], *additional_timings: dict[str, Any]
) -> dict[str, Any]:
    records = (base_timing, *additional_timings)
    return {
        "wall_seconds": sum(float(item["wall_seconds"]) for item in records),
        "aggregate_cpu_seconds": sum(
            float(item["aggregate_cpu_seconds"]) for item in records
        ),
        "peak_rss_bytes": max(int(item["peak_rss_bytes"]) for item in records),
    }


def validate_complete_readiness_timing(record: dict[str, Any]) -> None:
    nested = record.get("components", {})
    components = (
        nested
        if all(key in nested for key in ("actual_source_replay", "mlflow_probe", "readiness_infrastructure"))
        else record
    )
    expected = aggregate_readiness_timing(
        record["base_total_timing_before_readiness_additions"],
        components["actual_source_replay"]["timing"],
        components["mlflow_probe"]["timing"],
        components["readiness_infrastructure"]["timing"],
    )
    observed = record["total_timing"]
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        if not math.isclose(
            float(observed[resource]), float(expected[resource]), rel_tol=0, abs_tol=1e-9
        ):
            raise RuntimeError(f"readiness complete {resource} timing omission")
    if int(observed["peak_rss_bytes"]) != int(expected["peak_rss_bytes"]):
        raise RuntimeError("readiness complete peak RSS timing omission")


def validate_core_receipt_release_equality(
    core: dict[str, Any], receipt: dict[str, Any]
) -> None:
    for field in READINESS_RELEASE_CRITICAL_FIELDS:
        if core.get(field) != receipt.get(field):
            raise RuntimeError(f"readiness core/receipt mismatch for {field}")


def readiness_accounting_decomposition() -> dict[str, Any]:
    completed_wall = READINESS_COMPLETED_HISTORY_CHARGE["wall_seconds"]
    completed_cpu = READINESS_COMPLETED_HISTORY_CHARGE["aggregate_cpu_seconds"]
    prior_wall = READINESS_SUPERSEDED_CALIBRATION_CHARGE["wall_seconds"]
    prior_cpu = READINESS_SUPERSEDED_CALIBRATION_CHARGE["aggregate_cpu_seconds"]
    return {
        "completed_history_wall_seconds": completed_wall,
        "completed_history_aggregate_cpu_seconds": completed_cpu,
        "completed_prior_superseded_wall_seconds": prior_wall,
        "completed_prior_superseded_aggregate_cpu_seconds": prior_cpu,
        "completed_accepted_active_wall_seconds": completed_wall - prior_wall,
        "completed_accepted_active_aggregate_cpu_seconds": completed_cpu - prior_cpu,
        "failed_attempt_conservative_allowance_wall_seconds": (
            ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["wall_seconds"]
        ),
        "failed_attempt_conservative_allowance_aggregate_cpu_seconds": (
            ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["aggregate_cpu_seconds"]
        ),
        "failed_attempt_allowance_exact_internal_observation": False,
        "failed_attempt_allowance_completed_work_claimed": False,
        "carried_charge_wall_seconds": READINESS_CARRIED_CHARGE["wall_seconds"],
        "carried_charge_aggregate_cpu_seconds": READINESS_CARRIED_CHARGE[
            "aggregate_cpu_seconds"
        ],
    }


def observed_readiness_recovery(
    *, active_wall: float, active_cpu: float, reused_wall: float,
    reused_cpu: float, peak_rss: int,
) -> dict[str, Any]:
    accounting = readiness_accounting_decomposition()
    wall_total = (
        accounting["carried_charge_wall_seconds"]
        + active_wall + reused_wall
    )
    cpu_total = (
        accounting["carried_charge_aggregate_cpu_seconds"]
        + active_cpu + reused_cpu
    )
    return {
        "scope": "observed-launch-to-pre-outer-seal-sample-not-through-process-exit",
        "accounting_formula": "completed_history + failed_attempt_conservative_allowance + observed_pre_outer_seal_active + current_reused",
        "prior_superseded_calibrations": list(READINESS_SUPERSEDED_CALIBRATIONS),
        **accounting,
        "carried_charge_applied_exactly_once": True,
        "current_readiness_observed_pre_outer_seal_active_wall_seconds": active_wall,
        "current_readiness_observed_pre_outer_seal_active_aggregate_cpu_seconds": active_cpu,
        "current_readiness_reused_wall_seconds_charged": reused_wall,
        "current_readiness_reused_aggregate_cpu_seconds_charged": reused_cpu,
        "cumulative_carried_observed_active_reused_wall_seconds": wall_total,
        "cumulative_carried_observed_active_reused_aggregate_cpu_seconds": cpu_total,
        "observed_pre_outer_seal_peak_rss_bytes": int(peak_rss),
    }


def validate_observed_readiness_recovery(recovery: dict[str, Any]) -> None:
    if recovery.get("scope") != "observed-launch-to-pre-outer-seal-sample-not-through-process-exit":
        raise RuntimeError("readiness observed recovery scope mismatch")
    if recovery.get("prior_superseded_calibrations") != list(READINESS_SUPERSEDED_CALIBRATIONS):
        raise RuntimeError("readiness observed superseded evidence mismatch")
    accounting = readiness_accounting_decomposition()
    for key, expected in accounting.items():
        observed = recovery.get(key)
        if type(expected) is bool:
            if observed is not expected:
                raise RuntimeError(f"readiness observed accounting mismatch for {key}")
        elif not math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"readiness observed accounting mismatch for {key}")
    if recovery.get("carried_charge_applied_exactly_once") is not True:
        raise RuntimeError("readiness observed carried charge exact-once marker missing")
    for resource in ("wall", "aggregate_cpu"):
        active = float(recovery.get(f"current_readiness_observed_pre_outer_seal_active_{resource}_seconds", -1))
        reused = float(recovery.get(f"current_readiness_reused_{resource}_seconds_charged", -1))
        completed = float(recovery[f"completed_history_{resource}_seconds"])
        allowance = float(
            recovery[f"failed_attempt_conservative_allowance_{resource}_seconds"]
        )
        carried = float(recovery[f"carried_charge_{resource}_seconds"])
        total = float(recovery.get(f"cumulative_carried_observed_active_reused_{resource}_seconds", -1))
        if min(active, reused, completed, allowance, carried, total) < 0:
            raise RuntimeError(f"readiness observed cumulative {resource} is negative")
        if not math.isclose(carried, completed + allowance, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"readiness observed carried {resource} mismatch")
        if not math.isclose(
            total, carried + active + reused, rel_tol=0, abs_tol=1e-6
        ):
            raise RuntimeError(f"readiness observed cumulative {resource} mismatch")
    if int(recovery.get("observed_pre_outer_seal_peak_rss_bytes", 0)) <= 0:
        raise RuntimeError("readiness observed RSS missing")


READINESS_POST_SAMPLE_OPERATIONS = (
    "construct recovery arithmetic from already captured samples",
    "construct and validate the outer resource-completion payload",
    "canonical JSON serialization of the outer payload",
    "ensure the outer receipt parent directory exists",
    "create and open the atomic temporary file",
    "write the canonical outer payload bytes",
    "flush the temporary file",
    "fsync the temporary file",
    "close the temporary file",
    "atomically replace the outer receipt path with the temporary file",
    "open the parent directory",
    "fsync the parent directory",
    "close the parent directory",
    "return from write_readiness_resource_completion",
    "return from execute_readiness_calibration",
    "construct and serialize the CLI summary",
    "stdout write and flush",
    "return from main",
    "SystemExit construction and stack unwind",
    "Python interpreter teardown and process exit",
)


READINESS_INNER_ARTIFACT_NAMES = {
    "readiness-calibration.json",
    "readiness-calibration-core.json",
    "readiness-calibration-environment.json",
    "readiness-calibration-mlflow.json",
    "readiness-calibration-executed-config.yaml",
    "readiness-calibration-completion.json",
    "readiness-calibration-attempt-state.json",
    "readiness-calibration-launch-ledger.json",
}


def prevalidate_readiness_inner_lifecycle(
    *, inner_paths: dict[str, Path], expected_attempt_id: str,
    expected_authorization_sha256: str,
    expected_inner_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Hash and validate every non-self-referential inner artifact before sampling."""
    if set(inner_paths) != READINESS_INNER_ARTIFACT_NAMES:
        raise RuntimeError("readiness resource completion inner artifact schema mismatch")
    hashes = {name: sha256(path) for name, path in inner_paths.items()}
    if expected_inner_hashes is not None and hashes != expected_inner_hashes:
        raise RuntimeError("readiness resource completion inner artifact hash mismatch")
    parsed: dict[str, dict[str, Any]] = {}
    for name, path in inner_paths.items():
        if path.suffix != ".json":
            continue
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(
                f"readiness inner artifact is invalid JSON: {name}"
            ) from error
        if raw != canonical_json_bytes(value, pretty=True):
            raise RuntimeError(f"readiness inner artifact is noncanonical: {name}")
        parsed[name] = value
    completion = parsed["readiness-calibration-completion.json"]
    attempt_state = parsed["readiness-calibration-attempt-state.json"]
    launch_ledger = parsed["readiness-calibration-launch-ledger.json"]
    if (
        completion.get("schema_version")
        != "exp648-readiness-calibration-completion-v1"
        or completion.get("attempt_id") != expected_attempt_id
        or completion.get("authorization_sha256")
        != expected_authorization_sha256
        or completion.get("complete") is not True
    ):
        raise RuntimeError("readiness resource completion inner completion seal mismatch")
    if (
        launch_ledger != attempt_state
        or attempt_state.get("schema_version")
        != "exp648-readiness-calibration-attempt-v1"
        or attempt_state.get("attempt_id") != expected_attempt_id
        or attempt_state.get("authorization_sha256")
        != expected_authorization_sha256
        or attempt_state.get("phase") != "sealed"
        or attempt_state.get("complete") is not True
        or attempt_state.get("completion_manifest_sha256")
        != hashes["readiness-calibration-completion.json"]
    ):
        raise RuntimeError("readiness resource completion final attempt state mismatch")
    return hashes


def validate_readiness_resource_completion(
    payload: dict[str, Any], *, inner_paths: dict[str, Path],
    expected_attempt_id: str, expected_authorization_sha256: str,
    prevalidated_inner_hashes: dict[str, str] | None = None,
) -> None:
    required = {
        "schema_version", "status", "attempt_id", "authorization_sha256",
        "inner_artifacts", "inner_lifecycle_complete_before_resource_sample",
        "resource_boundary", "resource_observation", "measurement_semantics",
        "recovery_charges", "post_sample_operations",
    }
    if set(payload) != required or payload.get("schema_version") != "exp648-readiness-resource-completion-v2":
        raise RuntimeError("readiness resource completion schema mismatch")
    if payload.get("status") != "complete-observed-launch-to-pre-outer-seal-sample-resource-evidence":
        raise RuntimeError("readiness resource completion status mismatch")
    if payload.get("attempt_id") != expected_attempt_id or payload.get("authorization_sha256") != expected_authorization_sha256:
        raise RuntimeError("readiness resource completion attempt/auth mismatch")
    if payload.get("inner_lifecycle_complete_before_resource_sample") is not True:
        raise RuntimeError("readiness resource completion boundary is early")
    if payload.get("resource_boundary") != "observed launch-to-pre-outer-seal samples captured after inner MLflow verification, tracked publication, completion manifest, final sealed attempt state, and all feasible non-self-referential inner hashing and lifecycle validation; not through function return, CLI output, interpreter teardown, or process exit":
        raise RuntimeError("readiness resource completion boundary mismatch")
    if payload.get("post_sample_operations") != list(READINESS_POST_SAMPLE_OPERATIONS):
        raise RuntimeError("readiness resource completion post-sample operation list mismatch")
    semantics = payload.get("measurement_semantics")
    if semantics != {
        "wall_seconds": "external monotonic perf-counter elapsed seconds",
        "aggregate_cpu_seconds": "process user seconds plus process system seconds elapsed",
        "peak_rss_bytes": "Linux process high-water RSS bytes sampled separately",
        "wall_and_cpu_are_not_interchangeable_or_combined": True,
        "post_sample_operations_included_in_observed_totals": False,
    }:
        raise RuntimeError("readiness resource completion measurement semantics mismatch")
    expected_hashes = (
        prevalidate_readiness_inner_lifecycle(
            inner_paths=inner_paths,
            expected_attempt_id=expected_attempt_id,
            expected_authorization_sha256=expected_authorization_sha256,
            expected_inner_hashes=payload.get("inner_artifacts"),
        )
        if prevalidated_inner_hashes is None
        else prevalidated_inner_hashes
    )
    if payload.get("inner_artifacts") != expected_hashes:
        raise RuntimeError("readiness resource completion inner artifact hash mismatch")
    observation = payload.get("resource_observation", {})
    required_observation = {
        "sampling_order", "launch_wall_perf_counter", "wall_sample_perf_counter",
        "launch_process_user_plus_system_seconds", "process_user_seconds_at_sample",
        "process_system_seconds_at_sample", "observed_wall_seconds",
        "observed_aggregate_cpu_seconds", "observed_peak_rss_bytes",
    }
    if set(observation) != required_observation or observation.get("sampling_order") != [
        "process user/system CPU", "process peak RSS", "monotonic wall"
    ]:
        raise RuntimeError("readiness resource completion raw observation schema mismatch")
    expected_wall = float(observation["wall_sample_perf_counter"]) - float(observation["launch_wall_perf_counter"])
    expected_cpu = (
        float(observation["process_user_seconds_at_sample"])
        + float(observation["process_system_seconds_at_sample"])
        - float(observation["launch_process_user_plus_system_seconds"])
    )
    if not math.isclose(float(observation["observed_wall_seconds"]), expected_wall, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError("readiness observed wall is not monotonic wall elapsed")
    if not math.isclose(float(observation["observed_aggregate_cpu_seconds"]), expected_cpu, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError("readiness observed CPU is not process user plus system elapsed")
    recovery = payload["recovery_charges"]
    validate_observed_readiness_recovery(recovery)
    if not math.isclose(
        float(recovery["current_readiness_observed_pre_outer_seal_active_wall_seconds"]),
        expected_wall, rel_tol=0, abs_tol=1e-9,
    ):
        raise RuntimeError("readiness observed recovery wall/sample mismatch")
    if not math.isclose(
        float(recovery["current_readiness_observed_pre_outer_seal_active_aggregate_cpu_seconds"]),
        expected_cpu, rel_tol=0, abs_tol=1e-9,
    ):
        raise RuntimeError("readiness observed recovery CPU/sample mismatch")
    if int(recovery["observed_pre_outer_seal_peak_rss_bytes"]) != int(observation["observed_peak_rss_bytes"]):
        raise RuntimeError("readiness observed recovery RSS/sample mismatch")


def write_readiness_resource_completion(
    *, inner_paths: dict[str, Path], attempt_id: str,
    authorization_sha256: str, launch_wall_started: float,
    launch_cpu_started: float, launch_peak_rss: int,
    reused_wall: float, reused_cpu: float,
    output_path: Path = READINESS_RESOURCE_COMPLETION_PATH,
) -> dict[str, Any]:
    inner_hashes = prevalidate_readiness_inner_lifecycle(
        inner_paths=inner_paths,
        expected_attempt_id=attempt_id,
        expected_authorization_sha256=authorization_sha256,
    )
    process = psutil.Process()
    cpu = process.cpu_times()
    peak_rss = max(launch_peak_rss, cal.process_peak_rss_bytes(process))
    wall_sample = time.perf_counter()
    observed_wall = wall_sample - launch_wall_started
    observed_cpu = float(cpu.user + cpu.system) - launch_cpu_started
    observation = {
        "sampling_order": [
            "process user/system CPU", "process peak RSS", "monotonic wall"
        ],
        "launch_wall_perf_counter": launch_wall_started,
        "wall_sample_perf_counter": wall_sample,
        "launch_process_user_plus_system_seconds": launch_cpu_started,
        "process_user_seconds_at_sample": float(cpu.user),
        "process_system_seconds_at_sample": float(cpu.system),
        "observed_wall_seconds": observed_wall,
        "observed_aggregate_cpu_seconds": observed_cpu,
        "observed_peak_rss_bytes": peak_rss,
    }
    recovery = observed_readiness_recovery(
        active_wall=observed_wall,
        active_cpu=observed_cpu,
        reused_wall=reused_wall,
        reused_cpu=reused_cpu,
        peak_rss=peak_rss,
    )
    payload = {
        "schema_version": "exp648-readiness-resource-completion-v2",
        "status": "complete-observed-launch-to-pre-outer-seal-sample-resource-evidence",
        "attempt_id": attempt_id,
        "authorization_sha256": authorization_sha256,
        "inner_artifacts": inner_hashes,
        "inner_lifecycle_complete_before_resource_sample": True,
        "resource_boundary": "observed launch-to-pre-outer-seal samples captured after inner MLflow verification, tracked publication, completion manifest, final sealed attempt state, and all feasible non-self-referential inner hashing and lifecycle validation; not through function return, CLI output, interpreter teardown, or process exit",
        "resource_observation": observation,
        "measurement_semantics": {
            "wall_seconds": "external monotonic perf-counter elapsed seconds",
            "aggregate_cpu_seconds": "process user seconds plus process system seconds elapsed",
            "peak_rss_bytes": "Linux process high-water RSS bytes sampled separately",
            "wall_and_cpu_are_not_interchangeable_or_combined": True,
            "post_sample_operations_included_in_observed_totals": False,
        },
        "recovery_charges": recovery,
        "post_sample_operations": list(READINESS_POST_SAMPLE_OPERATIONS),
    }
    validate_readiness_resource_completion(
        payload, inner_paths=inner_paths,
        expected_attempt_id=attempt_id,
        expected_authorization_sha256=authorization_sha256,
        prevalidated_inner_hashes=inner_hashes,
    )
    _atomic_write(output_path, canonical_json_bytes(payload, pretty=True))
    return payload


def load_and_validate_readiness_resource_completion(
    path: Path, *, inner_paths: dict[str, Path], expected_attempt_id: str,
    expected_authorization_sha256: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != canonical_json_bytes(payload, pretty=True):
        raise RuntimeError("readiness resource completion is noncanonical")
    validate_readiness_resource_completion(
        payload, inner_paths=inner_paths,
        expected_attempt_id=expected_attempt_id,
        expected_authorization_sha256=expected_authorization_sha256,
    )
    return payload


def validate_readiness_calibration_receipt(
    payload: dict[str, Any], *, base_artifact: dict[str, Any] | None = None
) -> None:
    required = {
        "schema_version", "status", "calibration_root", "calibration_authorization_sha256",
        "scientific_runner_sha256", "resolved_config_sha256", "screen_workload_sha256",
        "calibration_evidence_core_sha256", "environment_sha256", "mlflow_receipt_sha256",
        "warmups", "timed_repeats", "warmup", "repeats", "component_resolution",
        "resolved_screen_budgets", "recovery_charges", "full_fidelity",
        "environment", "boundary", "chronology",
    }
    if set(payload) != required:
        raise RuntimeError("readiness calibration receipt key schema mismatch")
    if payload["schema_version"] != "exp648-screen-readiness-calibration-v1" or payload["warmups"] != 1 or payload["timed_repeats"] != 3:
        raise RuntimeError("readiness calibration repetition/schema mismatch")
    repeats = payload["repeats"]
    if len(repeats) != 3 or any(r.get("complete") is not True for r in repeats):
        raise RuntimeError("readiness calibration repeats incomplete")
    validate_readiness_chronology(payload)
    for repeat in repeats:
        validate_complete_readiness_timing(repeat)
        components = repeat.get("components", {})
        if components.get("startup_model_loading", {}).get("units") != 1:
            raise RuntimeError("readiness calibration startup workload mismatch")
        if components.get("mining", {}).get("matched_pairs") != 64 or components["mining"].get("windows") != 128:
            raise RuntimeError("readiness calibration mining workload mismatch")
        if components.get("corpus_integrity_validation", {}).get("windows") != 128:
            raise RuntimeError("readiness calibration replay workload mismatch")
        for arm in ("control", "challenger"):
            if components.get("training", {}).get(arm, {}).get("updates") != 20:
                raise RuntimeError("readiness calibration training workload mismatch")
            if components.get("checkpoint", {}).get(arm, {}).get("writes") != 1:
                raise RuntimeError("readiness calibration checkpoint workload mismatch")
            for width in (128, 256):
                if components.get("transitions", {}).get(arm, {}).get(str(width), {}).get("cases") != 200:
                    raise RuntimeError("readiness calibration transition workload mismatch")
            for width in (64, 128, 256):
                if components.get("rollouts", {}).get(arm, {}).get(str(width), {}).get("cases") != 20:
                    raise RuntimeError("readiness calibration rollout workload mismatch")
        infra = components.get("readiness_infrastructure", {})
        if infra.get("atomic_checkpoint_writes_and_reload_verifications") != 2 or infra.get("atomic_attempt_ledger_writes") != 213 or infra.get("evaluation_cache_write_validate_units") != 7 or infra.get("atomic_evidence_writes") != 19 or set(infra.get("component_timing", {})) != {"atomic_checkpoints", "attempt_ledger", "evaluation_cache", "atomic_evidence"} or infra.get("complete") is not True:
            raise RuntimeError("readiness calibration infrastructure workload mismatch")
        replay = components.get("actual_source_replay", {})
        if replay.get("complete_identities") != 64 or replay.get("steps_per_identity") != 256 or replay.get("complete_steps") != 16_384 or replay.get("actual_rollout_identity_batch_path") is not True or replay.get("complete") is not True:
            raise RuntimeError("readiness calibration actual source replay workload mismatch")
        probe = components.get("mlflow_probe", {})
        if probe.get("complete") is not True or probe.get("artifact_count") != 6 or probe.get("remote_verified") is not True:
            raise RuntimeError("readiness calibration MLflow probe workload mismatch")
        if repeat.get("fidelity_reduced") is not False or repeat.get("scientific_accuracy_reported") is not False:
            raise RuntimeError("readiness calibration repeat scientific/fidelity boundary mismatch")
    warmup = payload.get("warmup", {})
    validate_complete_readiness_timing(warmup)
    warmup_replay = warmup.get("actual_source_replay", {})
    if warmup_replay.get("complete_identities") != 64 or warmup_replay.get("complete_steps") != 16_384 or warmup_replay.get("complete") is not True:
        raise RuntimeError("readiness calibration warmup source replay mismatch")
    warmup_probe = payload.get("warmup", {}).get("mlflow_probe", {})
    if warmup_probe.get("complete") is not True or warmup_probe.get("artifact_count") != 6 or warmup_probe.get("remote_verified") is not True:
        raise RuntimeError("readiness calibration warmup MLflow probe mismatch")
    warmup_infra = warmup.get("readiness_infrastructure", {})
    if warmup_infra.get("complete") is not True:
        raise RuntimeError("readiness calibration warmup infrastructure mismatch")
    expected_budgets, expected_components = resolve_readiness_screen_budgets(
        repeats, payload["warmup"]["launcher_timing"], payload["warmup"]
    )
    if payload["resolved_screen_budgets"] != expected_budgets or payload["component_resolution"] != expected_components:
        raise RuntimeError("readiness calibration budgets/component rates do not recompute")
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        b = expected_budgets[resource]
        if not math.isclose(b["soft_budget_seconds"], 1.25 * b["projected_seconds"] + b["explicit_measured_setup_headroom_seconds"], rel_tol=0, abs_tol=1e-9):
            raise RuntimeError("readiness calibration soft-budget formula mismatch")
        if not math.isclose(b["hard_runaway_ceiling_seconds"], 1.5 * b["soft_budget_seconds"], rel_tol=0, abs_tol=1e-9):
            raise RuntimeError("readiness calibration hard-budget formula mismatch")
    expected_boundary = {
        "calibration_only": True,
        "screen_roots_opened": False,
        "confirmation_roots_opened": False,
        "checked_independent_official_private_opened": False,
        "scientific_accuracy_or_selection_reported": False,
        "model_update_performed": False,
        "fidelity_reduced": False,
    }
    if payload["full_fidelity"] is not True or payload["boundary"] != expected_boundary:
        raise RuntimeError("readiness calibration fidelity/scientific boundary mismatch")
    recovery = payload["recovery_charges"]
    if recovery.get("scope") != "prepublication-provisional-snapshot-before-inner-publication":
        raise RuntimeError("readiness calibration provisional recovery scope mismatch")
    if recovery.get("prior_superseded_calibrations") != list(READINESS_SUPERSEDED_CALIBRATIONS):
        raise RuntimeError("readiness calibration superseded evidence list mismatch")
    accounting = readiness_accounting_decomposition()
    for key, expected in accounting.items():
        observed = recovery.get(key)
        if type(expected) is bool:
            if observed is not expected:
                raise RuntimeError(f"readiness provisional accounting mismatch for {key}")
        elif not math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"readiness provisional accounting mismatch for {key}")
    if recovery.get("carried_charge_applied_exactly_once") is not True:
        raise RuntimeError("readiness provisional carried exact-once marker missing")
    expected_orchestration = {
        "chronology_contract_sha256": READINESS_CHRONOLOGY_CONTRACT_SHA256,
        "execution_order": [name for name, _ in READINESS_AUGMENTED_UNIT_ORDER],
        "old_base_only_cache_loaded_or_saved": False,
        "fresh_augmented_warmup_before_repeat_cache_reuse": True,
        "carried_charge_applied_exactly_once": True,
        "guard_carried_wall_seconds": (
            READINESS_CARRIED_CHARGE["wall_seconds"]
        ),
        "guard_carried_aggregate_cpu_seconds": (
            READINESS_CARRIED_CHARGE["aggregate_cpu_seconds"]
        ),
        "guard_configured_wall_ceiling_seconds": READINESS_CUMULATIVE_RUNAWAY_CEILINGS[
            "wall_seconds"
        ],
        "guard_configured_aggregate_cpu_ceiling_seconds": READINESS_CUMULATIVE_RUNAWAY_CEILINGS[
            "aggregate_cpu_seconds"
        ],
        "reused_complete_augmented_repeat_indices": [
            int(record["repeat_index"])
            for record in repeats
            if record.get("reused_from_augmented_cache") is True
        ],
        "all_augmented_units_validated_and_sealed_before_next_start": True,
    }
    if recovery.get("augmented_unit_orchestration") != expected_orchestration:
        raise RuntimeError("readiness augmented orchestration recovery mismatch")
    for resource in ("wall", "aggregate_cpu"):
        active = float(recovery.get(f"prepublication_provisional_active_{resource}_seconds", 0))
        reused = float(recovery.get(f"prepublication_provisional_reused_{resource}_seconds_charged", 0))
        completed = float(recovery[f"completed_history_{resource}_seconds"])
        allowance = float(recovery[f"failed_attempt_conservative_allowance_{resource}_seconds"])
        carried = float(recovery[f"carried_charge_{resource}_seconds"])
        cumulative = float(recovery.get(f"prepublication_provisional_cumulative_carried_active_reused_{resource}_seconds", -1))
        if min(active, reused, completed, allowance, carried, cumulative) < 0:
            raise RuntimeError(f"readiness calibration negative {resource} accounting")
        if not math.isclose(carried, completed + allowance, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"readiness calibration carried {resource} mismatch")
        if not math.isclose(cumulative, carried + active + reused, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(f"readiness calibration cumulative {resource} charge mismatch")
    if int(recovery.get("prepublication_provisional_process_peak_rss_bytes", 0)) <= 0:
        raise RuntimeError("readiness calibration final process peak RSS missing")
    if base_artifact is not None:
        for repeat in repeats:
            cal.validate_calibration_repeat(repeat, base_artifact["config_for_validation"])


def _readiness_mlflow_expected(payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    tags = {
        "canonical_run_id": RUN_ID, "role": "screen-readiness-calibration-only",
        "authorization_sha256": payload["calibration_authorization_sha256"],
        "scientific_accuracy_reported": "false", "candidate_registration": "false",
    }
    params = {
        "calibration_root": str(payload["calibration_root"]),
        "scientific_runner_sha256": payload["scientific_runner_sha256"],
        "screen_workload_sha256": payload["screen_workload_sha256"],
        "calibration_evidence_core_sha256": payload["calibration_evidence_core_sha256"],
    }
    metrics = {
        "projected_wall_seconds": float(payload["resolved_screen_budgets"]["wall_seconds"]["projected_seconds"]),
        "projected_aggregate_cpu_seconds": float(payload["resolved_screen_budgets"]["aggregate_cpu_seconds"]["projected_seconds"]),
        "peak_rss_guard_bytes": float(payload["resolved_screen_budgets"]["peak_rss_guard_bytes"]),
    }
    return tags, params, metrics


def select_finished_or_supersede_partial_mlflow_runs(
    client: Any, matches: Sequence[Any], *, reason: str
) -> Any | None:
    finished = [run for run in matches if run.info.status == "FINISHED"]
    if len(finished) > 1:
        raise RuntimeError(f"duplicate FINISHED {reason} MLflow runs")
    terminal = {"FINISHED", "FAILED", "KILLED"}
    active = {"RUNNING", "SCHEDULED"}
    for run in matches:
        if run.info.status == "FINISHED":
            continue
        if run.info.status in active:
            client.set_terminated(run.info.run_id, status="KILLED")
        client.set_tag(run.info.run_id, "superseded_for_recovery", "true")
        client.set_tag(run.info.run_id, "supersession_reason", reason)
        refreshed = client.get_run(run.info.run_id)
        if refreshed.info.status not in terminal:
            raise RuntimeError(f"superseded {reason} MLflow run remains active")
    return finished[0] if finished else None


def log_and_verify_readiness_calibration_mlflow(
    payload: dict[str, Any], artifact_paths: Sequence[Path]
) -> dict[str, Any]:
    experiment_id = cal.configure_mlflow("sair-modular-arithmetic-wsl")
    client = cal.mlflow.tracking.MlflowClient(tracking_uri=cal.tracking_uri())
    tags, params, metrics = _readiness_mlflow_expected(payload)
    matches = client.search_runs(
        experiment_ids=[str(experiment_id)],
        filter_string=(f"tags.canonical_run_id = '{RUN_ID}' and tags.role = 'screen-readiness-calibration-only'"),
        max_results=50,
    )
    matches = [r for r in matches if r.data.tags.get("authorization_sha256") == tags["authorization_sha256"]]
    finished = select_finished_or_supersede_partial_mlflow_runs(
        client, matches, reason="readiness-calibration-exact-binding"
    )
    reused = finished is not None
    if finished is not None:
        run_id = finished.info.run_id
    else:
        with mlflow.start_run(run_name=f"{RUN_ID}-screen-readiness-calibration") as active:
            run_id = active.info.run_id
            mlflow.set_tags(tags)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for path in artifact_paths:
                mlflow.log_artifact(str(path), artifact_path="readiness-calibration")
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise RuntimeError("readiness calibration MLflow run is not FINISHED")
    custom_tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    if custom_tags != tags or dict(run.data.params) != params or set(run.data.metrics) != set(metrics):
        raise RuntimeError("readiness calibration MLflow tags/params/metrics mismatch")
    for key, expected in metrics.items():
        if not math.isclose(float(run.data.metrics[key]), expected, rel_tol=1e-12, abs_tol=1e-9):
            raise RuntimeError(f"readiness calibration MLflow metric mismatch: {key}")
    verify_dir = SCRATCH_SCREEN_DIR / "readiness-calibration-mlflow" / run_id
    artifacts = []
    for source in artifact_paths:
        downloaded = Path(client.download_artifacts(run_id, f"readiness-calibration/{source.name}", str(verify_dir)))
        if sha256(downloaded) != sha256(source):
            raise RuntimeError(f"readiness calibration MLflow artifact mismatch: {source.name}")
        artifacts.append({"path": source.name, "sha256": sha256(source), "verified": True})
    return {
        "schema_version": "exp648-screen-readiness-calibration-mlflow-v1",
        "run_id": run_id, "status": "FINISHED", "reused": reused,
        "expected_tags": tags, "expected_params": params, "expected_metrics": metrics,
        "exact_tags_verified": True, "exact_params_verified": True,
        "exact_metrics_verified": True, "artifacts": artifacts,
    }


def verify_readiness_calibration_mlflow_at_launch(
    payload: dict[str, Any], receipt: dict[str, Any], core: Path,
    environment: Path, executed_config: Path,
) -> None:
    client = cal.mlflow.tracking.MlflowClient(tracking_uri=cal.tracking_uri())
    run = client.get_run(receipt["run_id"])
    tags, params, metrics = _readiness_mlflow_expected(payload)
    custom_tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    if run.info.status != "FINISHED" or custom_tags != tags or dict(run.data.params) != params or set(run.data.metrics) != set(metrics):
        raise PermissionError("fresh calibration MLflow launch requery schema mismatch")
    if any(not math.isclose(float(run.data.metrics[k]), v, rel_tol=1e-12, abs_tol=1e-9) for k, v in metrics.items()):
        raise PermissionError("fresh calibration MLflow launch metric mismatch")
    verify_dir = SCRATCH_SCREEN_DIR / "launch-mlflow-reverify" / receipt["run_id"]
    for source, remote_name in ((core, "calibration-core.json"), (environment, "environment.json"), (executed_config, "executed-config.yaml")):
        downloaded = Path(client.download_artifacts(receipt["run_id"], f"readiness-calibration/{remote_name}", str(verify_dir)))
        if sha256(downloaded) != sha256(source):
            raise PermissionError(f"fresh calibration MLflow launch artifact mismatch: {remote_name}")


def measure_readiness_infrastructure_unit(
    repeat: dict[str, Any], output: Path, process: Any
) -> dict[str, Any]:
    measurement = cal.Measurement(process)
    checkpoint_measurement = cal.Measurement(process)
    checkpoint_receipts = {}
    for arm in ("control", "challenger"):
        source = ROOT / repeat["components"]["checkpoint"][arm]["path"]
        state = load_file(source)
        destination = output / arm / "atomic-reload.safetensors"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        save_file(state, temporary)
        os.replace(temporary, destination)
        reloaded = load_file(destination)
        if cal.state_dict_sha256(state) != cal.state_dict_sha256(reloaded):
            raise RuntimeError("readiness calibration atomic checkpoint reload mismatch")
        checkpoint_receipts[arm] = {"sha256": sha256(destination), "state_dict_sha256": cal.state_dict_sha256(reloaded)}
    checkpoint_timing = checkpoint_measurement.stop().payload()
    ledger_measurement = cal.Measurement(process)
    ledger_dir = output / "ledger"
    for index in range(213):
        _atomic_write(ledger_dir / "attempt-state.json", canonical_json_bytes({
            "schema_version": "exp648-calibration-ledger-work-v1", "paired_update_or_unit": index,
            "checkpoint_hashes": checkpoint_receipts, "complete": index == 212,
        }, pretty=True))
    ledger_timing = ledger_measurement.stop().payload()
    cache_measurement = cal.Measurement(process)
    cases = {"state": [0, 1], "target": [0, 1], "quotient": [0, 0], "modulus": [3, 3]}
    result = {
        "cases": 2, "exact_accuracy": 1.0, "quotient_accuracy": 1.0,
        "valid_residue_rate": 1.0, "exact_vector": [True, True],
        "output_sha256": sha256_bytes(canonical_json_bytes([0, 1])),
        "predictions": [0, 1], "quotient_predictions": [0, 0],
    }
    for index in range(7):
        cache = output / "evaluation-cache" / f"unit-{index}.json"
        record = {"unit": index, "cases": cases, "results": {"control": result, "challenger": result}}
        _atomic_write(cache, canonical_json_bytes(record, pretty=True))
        loaded = json.loads(cache.read_text(encoding="utf-8"))
        validate_evaluation_results(unit="transition-128", width=128, cases=loaded["cases"], results=loaded["results"])
    cache_timing = cache_measurement.stop().payload()
    evidence_measurement = cal.Measurement(process)
    for index in range(19):
        _atomic_write(output / "evidence" / f"artifact-{index}.json", canonical_json_bytes({"index": index, "checkpoint_hashes": checkpoint_receipts}, pretty=True))
    evidence_timing = evidence_measurement.stop().payload()
    timing = measurement.stop().payload()
    return {
        "atomic_checkpoint_writes_and_reload_verifications": 2,
        "atomic_attempt_ledger_writes": 213,
        "evaluation_cache_write_validate_units": 7,
        "atomic_evidence_writes": 19,
        "component_timing": {"atomic_checkpoints": checkpoint_timing, "attempt_ledger": ledger_timing, "evaluation_cache": cache_timing, "atomic_evidence": evidence_timing},
        "checkpoint_receipts": checkpoint_receipts,
        "timing": timing, "complete": True,
    }


def measure_actual_source_replay_unit(
    experiment: Any, output: Path, process: Any
) -> dict[str, Any]:
    artifacts = _read_data_artifacts()
    identities_payload = artifacts["identity"]["identities"][:64]
    expected = artifacts["trajectory"]["trajectories"][:64]
    identities = [cal.Identity(
        int(row["identity_index"]), int(row["multiplier"]),
        int(row["multiplicand"]), int(row["modulus"]),
    ) for row in identities_payload]
    source_model = experiment.load_verified_models_and_optimizers(20)[0].eval()
    measurement = cal.Measurement(process)
    observed = []
    for start in range(0, 64, 32):
        rows = cal.rollout_identity_batch(source_model, experiment.exp300, identities[start:start + 32], experiment.guard)
        observed.extend(trajectory_record_payload(row) for row in rows)
    timing = measurement.stop().payload()
    if canonical_json(observed) != canonical_json(expected):
        raise RuntimeError("readiness calibration actual source replay mismatch")
    payload = {
        "schema_version": "exp648-readiness-source-replay-v1",
        "complete_identities": 64, "steps_per_identity": 256, "complete_steps": 64 * 256,
        "identity_sha256": sha256_bytes(canonical_json_bytes(identities_payload)),
        "trajectory_sha256": sha256_bytes(canonical_json_bytes(observed)),
        "actual_rollout_identity_batch_path": True, "complete": True, "timing": timing,
    }
    _atomic_write(output / "source-replay.json", canonical_json_bytes(payload, pretty=True))
    return payload


def measure_readiness_mlflow_probe(
    *, authorization_sha256: str, unit: str, output: Path, process: Any
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index in range(6):
        path = output / f"probe-{index}.json"
        _atomic_write(path, canonical_json_bytes({"unit": unit, "index": index}, pretty=True))
        artifacts.append(path)
    measurement = cal.Measurement(process)
    experiment_id = cal.configure_mlflow("sair-modular-arithmetic-wsl")
    with mlflow.start_run(run_name=f"{RUN_ID}-readiness-probe-{unit}") as active:
        run_id = active.info.run_id
        mlflow.set_tags({"canonical_run_id": RUN_ID, "role": "readiness-calibration-mlflow-probe", "authorization_sha256": authorization_sha256})
        mlflow.log_params({"unit": unit, "artifact_count": 6})
        mlflow.log_metrics({"probe_resource_only": 1.0})
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path="probe")
    client = cal.mlflow.tracking.MlflowClient(tracking_uri=cal.tracking_uri())
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise RuntimeError("readiness calibration MLflow probe did not finish")
    verify = output / "download"
    for path in artifacts:
        downloaded = Path(client.download_artifacts(run_id, f"probe/{path.name}", str(verify)))
        if sha256(downloaded) != sha256(path):
            raise RuntimeError("readiness calibration MLflow probe artifact mismatch")
    timing = measurement.stop().payload()
    client.set_tag(run_id, "superseded_for_calibration_probe_only", "true")
    return {"complete": True, "run_id": run_id, "artifact_count": 6, "remote_verified": True, "timing": timing}


READINESS_AUGMENTED_UNIT_ORDER = (
    ("warmup", None),
    ("repeat-1", 1),
    ("repeat-2", 2),
    ("repeat-3", 3),
)
READINESS_CHRONOLOGY_CONTRACT_SHA256 = sha256_bytes(canonical_json_bytes({
    "schema_version": "exp648-readiness-augmented-chronology-contract-v1",
    "order": [name for name, _ in READINESS_AUGMENTED_UNIT_ORDER],
    "unit_order": "base->source-replay->mlflow-probe->readiness-infrastructure->validate->atomic-record->seal",
    "next_start_requires_predecessor_seal": True,
    "fresh_augmented_warmup_before_repeat_cache_reuse": True,
    "old_base_only_cache_forbidden": True,
}))
READINESS_RECORD_METADATA_KEYS = frozenset((
    "readiness_chronology", "reused_from_augmented_cache", "augmented_cache_path",
))


def _path_for_receipt(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _bare_augmented_record(record: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(record)
    for key in READINESS_RECORD_METADATA_KEYS:
        result.pop(key, None)
    return result


def apply_readiness_carried_charge_once(guard: Any) -> None:
    if getattr(guard, "readiness_carried_charge_applied", False):
        raise RuntimeError("readiness cumulative carried charge applied more than once")
    if not math.isclose(
        float(guard.wall_ceiling),
        READINESS_CUMULATIVE_RUNAWAY_CEILINGS["wall_seconds"],
        rel_tol=0,
        abs_tol=1e-9,
    ) or not math.isclose(
        float(guard.cpu_ceiling),
        READINESS_CUMULATIVE_RUNAWAY_CEILINGS["aggregate_cpu_seconds"],
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("readiness guard ceilings differ from configured cumulative ceilings")
    initial_wall = float(guard.prior_superseded_wall_seconds)
    initial_cpu = float(guard.prior_superseded_cpu_seconds)
    if not math.isclose(
        initial_wall,
        float(cal.PRIOR_SUPERSEDED_CALIBRATION_CHARGE["wall_seconds"]),
        rel_tol=0,
        abs_tol=1e-9,
    ) or not math.isclose(
        initial_cpu,
        float(cal.PRIOR_SUPERSEDED_CALIBRATION_CHARGE["aggregate_cpu_seconds"]),
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("readiness guard initial completed-history charge mismatch")
    guard.readiness_initial_completed_history_wall_seconds = initial_wall
    guard.readiness_initial_completed_history_cpu_seconds = initial_cpu
    guard.prior_superseded_wall_seconds = READINESS_CARRIED_CHARGE["wall_seconds"]
    guard.prior_superseded_cpu_seconds = READINESS_CARRIED_CHARGE[
        "aggregate_cpu_seconds"
    ]
    guard.readiness_carried_charge_applied = True
    guard.readiness_carried_charge_applied_exactly_once = True
    guard("readiness:cumulative-carried-charge-applied-once")


ROOT_20261160_PLANNING_CACHE_BINDINGS = {
    "root": 20_261_160,
    "authorization_sha256": "9e1f63a9fe3c6613f8c02dea523147901da2aea862a3dec58d53c0cb7a054984",
    "scientific_runner_sha256": "1690fc3ba81c7f1747faa944594d78a896519da85eda9fabd772d5b71e7e1e3c",
    "calibration_runner_sha256": CALIBRATION_RUNNER_SHA256,
    "config_sha256": "2c6f04d3f3302c5ec7e0019176b4076cc66d56f8b7d63977dee1b4f54bc24c7c",
    "resource_protocol_sha256": "f66c6df75238070e4af162eadff48f4dbbe5f04439a6ad42f77333074b82e52b",
    "workload_sha256": "11151e62af2ee6eb6ac02fb32a290f6dd2155b95a97220033c30bace7b3b7c15",
    "corpus_provenance_amendment_sha256": "9f2b3ab9f82a083adca05e21f08f23cfe0fdcedfce47afa656be89e87701def9",
    "chronology_contract_sha256": "54c9e4384526a3f42f293489d951dcf4de45bc462aabbc01c8abb5d76720b623",
}
ROOT_20261160_PLANNING_CACHE_FILES = {
    1: (
        "repeat-1-3207e01b0318af83234079e8488a9431c66fa69e174dd9ad449bf798f1525788.json",
        "98a69670a229c15de7193c5878571fcc28a6d013e9b3ae23a7ac52be92624d80",
    ),
    2: (
        "repeat-2-0676d6cbc44c56f65daeb35d5d26d757047bbb9f937c12805a387bf68b16d59e.json",
        "8e8115d2b41974c20adfad734a847ec2ac64e07f5636325301b8793ffb1adc68",
    ),
}


def verify_root20261160_planning_caches(
    *, scratch_screen: Path = SCRATCH_SCREEN_DIR,
) -> dict[str, Any]:
    """Verify historical cache bytes without making them executable or reusable."""
    cache_root = scratch_screen / "augmented-repeat-cache/root-20261160"
    attempt = (
        scratch_screen
        / "readiness-calibration-attempts/cal-9e1f63a9fe3c6613"
    )
    verified = []
    for repeat_index, (name, expected_sha) in sorted(
        ROOT_20261160_PLANNING_CACHE_FILES.items()
    ):
        path = cache_root / name
        raw = path.read_bytes()
        payload = json.loads(raw)
        if raw != canonical_json_bytes(payload, pretty=True):
            raise RuntimeError("root20261160 planning cache is noncanonical")
        if sha256_bytes(raw) != expected_sha:
            raise RuntimeError("root20261160 planning cache byte hash mismatch")
        fields = payload.get("cache_key_fields", {})
        for key, expected in ROOT_20261160_PLANNING_CACHE_BINDINGS.items():
            if fields.get(key) != expected:
                raise RuntimeError(
                    f"root20261160 planning cache binding mismatch for {key}"
                )
        if fields.get("seed") != 20_261_160:
            raise RuntimeError("root20261160 planning cache seed mismatch")
        ordinal = f"{repeat_index:02d}-repeat-{repeat_index}"
        seal_path = attempt / "augmented-chronology" / f"{ordinal}-seal.json"
        unit_path = attempt / "augmented-units" / ordinal / "augmented-unit.json"
        seal_raw = seal_path.read_bytes()
        seal = json.loads(seal_raw)
        if seal_raw != canonical_json_bytes(seal, pretty=True):
            raise RuntimeError("root20261160 source seal is noncanonical")
        if payload.get("source_seal_sha256") != sha256_bytes(seal_raw):
            raise RuntimeError("root20261160 planning cache source seal mismatch")
        chronology = payload.get("source_chronology", {})
        if chronology.get("seal") != seal:
            raise RuntimeError("root20261160 planning cache chronology/seal mismatch")
        unit_raw = unit_path.read_bytes()
        unit = json.loads(unit_raw)
        if unit_raw != canonical_json_bytes(unit, pretty=True):
            raise RuntimeError("root20261160 source augmented record is noncanonical")
        if payload.get("record") != _bare_augmented_record(unit):
            raise RuntimeError("root20261160 planning cache/source record mismatch")
        verified.append({
            "repeat_index": repeat_index,
            "cache_sha256": expected_sha,
            "source_seal_sha256": sha256_bytes(seal_raw),
            "source_augmented_record_sha256": sha256_bytes(unit_raw),
        })
    return {
        "root": 20_261_160,
        "status": "planning-evidence-only-executable-reuse-permanently-denied",
        "verified": verified,
        "current_binding_match_or_reuse_claimed": False,
    }


def readiness_augmented_cache_key_fields(
    experiment: Any, *, authorization_sha256: str,
) -> dict[str, Any]:
    base = experiment.cache_key_fields()
    return {
        "schema_version": "exp648-readiness-augmented-cache-key-v1",
        "root": int(experiment.root_seed),
        "authorization_sha256": authorization_sha256,
        "scientific_runner_sha256": sha256(Path(__file__)),
        "calibration_runner_sha256": sha256(CALIBRATION_RUNNER),
        "checkpoint_sha256": base["checkpoint_sha256"],
        "environment_sha256": base["environment_sha256"],
        "evaluator_sha256": base["evaluator_sha256"],
        "dataset_sha256": base["dataset_sha256"],
        "config_sha256": base["config_sha256"],
        "seed": base["seed"],
        "dependency_manifest_sha256": base["dependency_manifest_sha256"],
        "scientific_protocol_sha256": base["scientific_protocol_sha256"],
        "resource_protocol_sha256": base["resource_protocol_sha256"],
        "workload_sha256": base["workload_sha256"],
        "frozen_corpus_file_sha256": copy.deepcopy(FROZEN_CORPUS_FILES),
        "corpus_provenance_amendment_sha256": sha256(
            TRACKED_CORPUS_DIR / "provenance-amendment.json"
        ),
        "chronology_contract_sha256": READINESS_CHRONOLOGY_CONTRACT_SHA256,
    }


def readiness_augmented_cache_key(
    key_fields: dict[str, Any], repeat_index: int,
) -> str:
    return sha256_bytes(canonical_json_bytes({
        "key_fields": key_fields, "repeat_index": int(repeat_index),
    }))


def base_calibration_cache_key_fields_from_augmented(
    key_fields: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_sha256": key_fields["checkpoint_sha256"],
        "evaluator_sha256": key_fields["evaluator_sha256"],
        "dataset_sha256": key_fields["dataset_sha256"],
        "config_sha256": key_fields["config_sha256"],
        "seed": key_fields["seed"],
        "environment_sha256": key_fields["environment_sha256"],
        "runner_sha256": key_fields["calibration_runner_sha256"],
        "scientific_protocol_sha256": key_fields["scientific_protocol_sha256"],
        "resource_protocol_sha256": key_fields["resource_protocol_sha256"],
        "workload_sha256": key_fields["workload_sha256"],
        "dependency_manifest_sha256": key_fields["dependency_manifest_sha256"],
    }


def _validate_readiness_chronology_reference(
    reference: dict[str, Any], record: dict[str, Any], *,
    expected_root: int, expected_authorization_sha256: str,
    expected_ordinal: int, expected_unit: str,
    expected_repeat_index: int | None,
    expected_predecessor_seal_sha256: str | None,
) -> None:
    if not isinstance(reference, dict):
        raise RuntimeError("readiness chronology reference is not an object")
    if set(reference) != {
        "ordinal", "unit", "repeat_index", "start", "start_sha256",
        "seal", "seal_sha256",
    }:
        raise RuntimeError("readiness chronology reference schema mismatch")
    start = reference["start"]
    seal = reference["seal"]
    if not isinstance(start, dict) or set(start) != {
        "schema_version", "event", "root", "authorization_sha256",
        "ordinal", "unit", "repeat_index", "predecessor_seal_sha256",
        "monotonic_ns", "complete",
    }:
        raise RuntimeError("readiness chronology start schema mismatch")
    if not isinstance(seal, dict) or set(seal) != {
        "schema_version", "event", "root", "authorization_sha256",
        "ordinal", "unit", "repeat_index", "predecessor_seal_sha256",
        "start_sha256", "augmented_record_sha256",
        "reused_from_augmented_cache", "source_cache_sha256",
        "monotonic_ns", "complete",
    }:
        raise RuntimeError("readiness chronology seal schema mismatch")
    if reference["ordinal"] != expected_ordinal or reference["unit"] != expected_unit:
        raise RuntimeError("readiness chronology ordinal/unit mismatch")
    if reference["repeat_index"] != expected_repeat_index:
        raise RuntimeError("readiness chronology repeat index mismatch")
    if type(reference["ordinal"]) is not int:
        raise RuntimeError("readiness chronology ordinal type mismatch")
    if reference["start_sha256"] != sha256_bytes(canonical_json_bytes(start, pretty=True)):
        raise RuntimeError("readiness chronology start hash mismatch")
    if reference["seal_sha256"] != sha256_bytes(canonical_json_bytes(seal, pretty=True)):
        raise RuntimeError("readiness chronology seal hash mismatch")
    common = {
        "root": expected_root,
        "authorization_sha256": expected_authorization_sha256,
        "ordinal": expected_ordinal,
        "unit": expected_unit,
        "repeat_index": expected_repeat_index,
        "predecessor_seal_sha256": expected_predecessor_seal_sha256,
    }
    for key, value in common.items():
        if start.get(key) != value or seal.get(key) != value:
            raise RuntimeError(f"readiness chronology binding mismatch for {key}")
    if start.get("schema_version") != "exp648-readiness-augmented-unit-start-v1" or start.get("event") != "start" or start.get("complete") is not False:
        raise RuntimeError("readiness chronology start event mismatch")
    if seal.get("schema_version") != "exp648-readiness-augmented-unit-seal-v1" or seal.get("event") != "seal" or seal.get("complete") is not True:
        raise RuntimeError("readiness chronology seal event mismatch")
    if seal.get("start_sha256") != reference["start_sha256"]:
        raise RuntimeError("readiness chronology seal/start linkage mismatch")
    reused = seal.get("reused_from_augmented_cache")
    source_cache_sha256 = seal.get("source_cache_sha256")
    if type(reused) is not bool:
        raise RuntimeError("readiness chronology cache-reuse marker mismatch")
    if reused:
        if not isinstance(source_cache_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", source_cache_sha256
        ) is None:
            raise RuntimeError("readiness chronology reused cache hash is absent")
    elif source_cache_sha256 is not None:
        raise RuntimeError("readiness chronology fresh unit claims a source cache")
    if "reused_from_augmented_cache" in record:
        if record["reused_from_augmented_cache"] is not reused:
            raise RuntimeError("readiness record/chronology cache-reuse mismatch")
        cache_path = record.get("augmented_cache_path")
        if reused and (not isinstance(cache_path, str) or not cache_path):
            raise RuntimeError("readiness reused record lacks its augmented cache path")
    if type(start.get("monotonic_ns")) is not int or type(
        seal.get("monotonic_ns")
    ) is not int:
        raise RuntimeError("readiness chronology monotonic type mismatch")
    start_ns = start["monotonic_ns"]
    seal_ns = seal["monotonic_ns"]
    if start_ns < 0 or seal_ns <= start_ns:
        raise RuntimeError("readiness chronology monotonic start/seal order mismatch")
    bare = _bare_augmented_record(record)
    if seal.get("augmented_record_sha256") != sha256_bytes(
        canonical_json_bytes(bare, pretty=True)
    ):
        raise RuntimeError("readiness chronology augmented record hash mismatch")
    validate_complete_readiness_timing(bare)


def validate_readiness_chronology(payload: dict[str, Any]) -> None:
    chronology = payload.get("chronology")
    if not isinstance(chronology, dict) or set(chronology) != {
        "schema_version", "root", "authorization_sha256", "contract_sha256",
        "order", "units", "complete",
    }:
        raise RuntimeError("readiness augmented chronology is absent or malformed")
    if chronology.get("schema_version") != "exp648-readiness-augmented-chronology-v1" or chronology.get("complete") is not True:
        raise RuntimeError("readiness augmented chronology is incomplete")
    root = int(payload["calibration_root"])
    authorization_sha256 = str(payload["calibration_authorization_sha256"])
    if chronology.get("root") != root or chronology.get("authorization_sha256") != authorization_sha256:
        raise RuntimeError("readiness augmented chronology root/auth mismatch")
    if chronology.get("contract_sha256") != READINESS_CHRONOLOGY_CONTRACT_SHA256:
        raise RuntimeError("readiness augmented chronology contract mismatch")
    expected_order = [name for name, _ in READINESS_AUGMENTED_UNIT_ORDER]
    if chronology.get("order") != expected_order:
        raise RuntimeError("readiness augmented chronology unit order mismatch")
    records = [payload["warmup"], *payload["repeats"]]
    units = chronology.get("units")
    if not isinstance(units, list) or len(units) != len(READINESS_AUGMENTED_UNIT_ORDER):
        raise RuntimeError("readiness augmented chronology unit count mismatch")
    predecessor: str | None = None
    predecessor_seal_ns = -1
    for ordinal, ((unit, repeat_index), record, reference) in enumerate(
        zip(READINESS_AUGMENTED_UNIT_ORDER, records, units, strict=True)
    ):
        if record.get("readiness_chronology") != reference:
            raise RuntimeError("readiness record/top-level chronology mismatch")
        _validate_readiness_chronology_reference(
            reference, record,
            expected_root=root,
            expected_authorization_sha256=authorization_sha256,
            expected_ordinal=ordinal,
            expected_unit=unit,
            expected_repeat_index=repeat_index,
            expected_predecessor_seal_sha256=predecessor,
        )
        start_ns = int(reference["start"]["monotonic_ns"])
        if ordinal and start_ns <= predecessor_seal_ns:
            raise RuntimeError("readiness next augmented unit started before predecessor seal")
        predecessor = reference["seal_sha256"]
        predecessor_seal_ns = int(reference["seal"]["monotonic_ns"])


def _write_readiness_chronology_start(
    *, attempt: Path, root: int, authorization_sha256: str,
    ordinal: int, unit: str, repeat_index: int | None,
    predecessor: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_unit, expected_repeat = READINESS_AUGMENTED_UNIT_ORDER[ordinal]
    if (unit, repeat_index) != (expected_unit, expected_repeat):
        raise RuntimeError("readiness chronology attempted an out-of-order unit")
    predecessor_sha = None if predecessor is None else predecessor["seal_sha256"]
    predecessor_ns = -1 if predecessor is None else int(predecessor["seal"]["monotonic_ns"])
    if ordinal == 0 and predecessor is not None or ordinal > 0 and predecessor is None:
        raise RuntimeError("readiness chronology predecessor presence mismatch")
    if predecessor is not None and predecessor_sha != sha256_bytes(
        canonical_json_bytes(predecessor["seal"], pretty=True)
    ):
        raise RuntimeError("readiness chronology predecessor seal hash mismatch")
    started_ns = time.monotonic_ns()
    if started_ns <= predecessor_ns:
        raise RuntimeError("readiness next augmented unit started before predecessor seal")
    payload = {
        "schema_version": "exp648-readiness-augmented-unit-start-v1",
        "event": "start", "root": root,
        "authorization_sha256": authorization_sha256,
        "ordinal": ordinal, "unit": unit, "repeat_index": repeat_index,
        "predecessor_seal_sha256": predecessor_sha,
        "monotonic_ns": started_ns, "complete": False,
    }
    path = attempt / "augmented-chronology" / f"{ordinal:02d}-{unit}-start.json"
    _atomic_write(path, canonical_json_bytes(payload, pretty=True))
    return {
        "payload": payload,
        "sha256": sha256(path),
        "path": _path_for_receipt(path),
    }


def _write_readiness_chronology_seal(
    *, attempt: Path, root: int, authorization_sha256: str,
    ordinal: int, unit: str, repeat_index: int | None,
    predecessor_seal_sha256: str | None,
    start: dict[str, Any], record_path: Path,
    reused_from_augmented_cache: bool,
    source_cache_sha256: str | None,
) -> dict[str, Any]:
    sealed_ns = time.monotonic_ns()
    if sealed_ns <= int(start["payload"]["monotonic_ns"]):
        raise RuntimeError("readiness chronology seal did not follow start")
    payload = {
        "schema_version": "exp648-readiness-augmented-unit-seal-v1",
        "event": "seal", "root": root,
        "authorization_sha256": authorization_sha256,
        "ordinal": ordinal, "unit": unit, "repeat_index": repeat_index,
        "predecessor_seal_sha256": predecessor_seal_sha256,
        "start_sha256": start["sha256"],
        "augmented_record_sha256": sha256(record_path),
        "reused_from_augmented_cache": reused_from_augmented_cache,
        "source_cache_sha256": source_cache_sha256,
        "monotonic_ns": sealed_ns, "complete": True,
    }
    path = attempt / "augmented-chronology" / f"{ordinal:02d}-{unit}-seal.json"
    _atomic_write(path, canonical_json_bytes(payload, pretty=True))
    reference = {
        "ordinal": ordinal, "unit": unit, "repeat_index": repeat_index,
        "start": start["payload"], "start_sha256": start["sha256"],
        "seal": payload, "seal_sha256": sha256(path),
    }
    return reference


def validate_augmented_repeat_cache_payload(
    payload: dict[str, Any], *, expected_key_fields: dict[str, Any],
    repeat_index: int, config: dict[str, Any],
) -> None:
    if set(payload) != {
        "schema_version", "cache_key", "cache_key_fields", "repeat_index",
        "record", "source_chronology", "source_seal_sha256",
        "cache_created_monotonic_ns",
    } or payload.get("schema_version") != "exp648-readiness-augmented-repeat-cache-v1":
        raise RuntimeError("readiness augmented repeat cache schema mismatch")
    if payload.get("cache_key_fields") != expected_key_fields or payload.get("repeat_index") != repeat_index:
        raise RuntimeError("readiness augmented repeat cache binding mismatch")
    expected_key = readiness_augmented_cache_key(expected_key_fields, repeat_index)
    if payload.get("cache_key") != expected_key:
        raise RuntimeError("readiness augmented repeat cache key mismatch")
    record = payload.get("record")
    reference = payload.get("source_chronology")
    if not isinstance(record, dict) or record.get("complete") is not True:
        raise RuntimeError("readiness augmented repeat cache is incomplete")
    if record.get("reused_from_augmented_cache") is not None or record.get("readiness_chronology") is not None:
        raise RuntimeError("readiness augmented cache stores a non-bare record")
    if not isinstance(reference, dict):
        raise RuntimeError("readiness augmented cache chronology is absent")
    source_start = reference.get("start")
    if not isinstance(source_start, dict):
        raise RuntimeError("readiness augmented cache chronology start is absent")
    _validate_readiness_chronology_reference(
        reference, record,
        expected_root=int(expected_key_fields["root"]),
        expected_authorization_sha256=str(expected_key_fields["authorization_sha256"]),
        expected_ordinal=repeat_index,
        expected_unit=f"repeat-{repeat_index}",
        expected_repeat_index=repeat_index,
        expected_predecessor_seal_sha256=source_start.get(
            "predecessor_seal_sha256"
        ),
    )
    if reference["seal"].get("reused_from_augmented_cache") is not False:
        raise RuntimeError("readiness augmented cache source was not freshly measured")
    if payload.get("source_seal_sha256") != reference["seal_sha256"]:
        raise RuntimeError("readiness augmented cache source seal mismatch")
    if int(payload.get("cache_created_monotonic_ns", -1)) <= int(reference["seal"]["monotonic_ns"]):
        raise RuntimeError("readiness augmented cache predates its source seal")
    validate_complete_readiness_timing(record)
    cal.validate_cached_repeat_record(
        record,
        config,
        repeat_index=repeat_index,
        expected_cache_key_fields=(
            base_calibration_cache_key_fields_from_augmented(expected_key_fields)
        ),
    )


def _augmented_repeat_cache_path(
    *, root: int, repeat_index: int, key: str,
) -> Path:
    return (
        SCRATCH_SCREEN_DIR / "augmented-repeat-cache" / f"root-{root}"
        / f"repeat-{repeat_index}-{key}.json"
    )


def load_augmented_repeat_cache(
    *, experiment: Any, authorization_sha256: str, repeat_index: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path, str] | None:
    key_fields = readiness_augmented_cache_key_fields(
        experiment, authorization_sha256=authorization_sha256
    )
    key = readiness_augmented_cache_key(key_fields, repeat_index)
    path = _augmented_repeat_cache_path(
        root=int(experiment.root_seed), repeat_index=repeat_index, key=key
    )
    if not path.is_file():
        return None
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("readiness augmented repeat cache is malformed") from error
    if raw != canonical_json_bytes(payload, pretty=True):
        raise RuntimeError("readiness augmented repeat cache is noncanonical")
    validate_augmented_repeat_cache_payload(
        payload, expected_key_fields=key_fields,
        repeat_index=repeat_index, config=config,
    )
    return payload, path, sha256(path)


def save_augmented_repeat_cache(
    *, experiment: Any, authorization_sha256: str, repeat_index: int,
    config: dict[str, Any], record: dict[str, Any],
) -> tuple[Path, str]:
    bare = _bare_augmented_record(record)
    reference = record["readiness_chronology"]
    key_fields = readiness_augmented_cache_key_fields(
        experiment, authorization_sha256=authorization_sha256
    )
    key = readiness_augmented_cache_key(key_fields, repeat_index)
    path = _augmented_repeat_cache_path(
        root=int(experiment.root_seed), repeat_index=repeat_index, key=key
    )
    created_ns = time.monotonic_ns()
    if created_ns <= int(reference["seal"]["monotonic_ns"]):
        raise RuntimeError("readiness augmented cache write did not follow source seal")
    payload = {
        "schema_version": "exp648-readiness-augmented-repeat-cache-v1",
        "cache_key": key, "cache_key_fields": key_fields,
        "repeat_index": repeat_index, "record": bare,
        "source_chronology": reference,
        "source_seal_sha256": reference["seal_sha256"],
        "cache_created_monotonic_ns": created_ns,
    }
    validate_augmented_repeat_cache_payload(
        payload, expected_key_fields=key_fields,
        repeat_index=repeat_index, config=config,
    )
    _atomic_write(path, canonical_json_bytes(payload, pretty=True))
    return path, sha256(path)


def _execute_one_augmented_readiness_unit(
    *, experiment: Any, attempt: Path, authorization_sha256: str,
    config: dict[str, Any], ordinal: int, unit: str,
    repeat_index: int | None, predecessor: dict[str, Any] | None,
    launcher_timing: dict[str, Any] | None,
    stage_context: dict[str, Any],
) -> dict[str, Any]:
    stage_context["stage"] = f"{unit}:chronology-start"
    start = _write_readiness_chronology_start(
        attempt=attempt, root=int(experiment.root_seed),
        authorization_sha256=authorization_sha256,
        ordinal=ordinal, unit=unit, repeat_index=repeat_index,
        predecessor=predecessor,
    )
    predecessor_sha = None if predecessor is None else predecessor["seal_sha256"]
    output = attempt / "augmented-units" / f"{ordinal:02d}-{unit}"
    record_path = output / "augmented-unit.json"
    stage_context["stage"] = f"{unit}:cache-lookup-and-validation"
    cached = None if repeat_index is None else load_augmented_repeat_cache(
        experiment=experiment, authorization_sha256=authorization_sha256,
        repeat_index=repeat_index, config=config,
    )
    if cached is not None:
        cache_payload, cache_path, cache_sha = cached
        record = copy.deepcopy(cache_payload["record"])
        stage_context["stage"] = f"{unit}:cache-reuse-charge"
        experiment.guard.add_reused(
            float(record["total_timing"]["wall_seconds"]),
            float(record["total_timing"]["aggregate_cpu_seconds"]),
        )
        stage_context["stage"] = f"{unit}:cache-acceptance-guard"
        experiment.guard(f"readiness:{unit}:complete-augmented-cache-accepted")
        stage_context["stage"] = f"{unit}:cached-record-publication"
        _atomic_write(record_path, canonical_json_bytes(record, pretty=True))
        experiment.guard(f"readiness:{unit}:cached-augmented-record-published")
        stage_context["stage"] = f"{unit}:reused-chronology-seal"
        reference = _write_readiness_chronology_seal(
            attempt=attempt, root=int(experiment.root_seed),
            authorization_sha256=authorization_sha256,
            ordinal=ordinal, unit=unit, repeat_index=repeat_index,
            predecessor_seal_sha256=predecessor_sha,
            start=start, record_path=record_path,
            reused_from_augmented_cache=True,
            source_cache_sha256=cache_sha,
        )
        experiment.guard(f"readiness:{unit}:reused-augmented-unit-sealed")
        return {
            **record,
            "readiness_chronology": reference,
            "reused_from_augmented_cache": True,
            "augmented_cache_path": _path_for_receipt(cache_path),
        }

    base_name = f"readiness-{unit}-{time.time_ns()}"
    stage_context["stage"] = f"{unit}:base-complete-unit"
    record = copy.deepcopy(experiment.execute_complete_unit(
        base_name, repeat_index=repeat_index
    ))
    experiment.guard(f"readiness:{unit}:base-complete")
    record["base_total_timing_before_readiness_additions"] = copy.deepcopy(
        record["total_timing"]
    )
    stage_context["stage"] = f"{unit}:source-replay"
    replay = measure_actual_source_replay_unit(
        experiment, output / "source-replay", psutil.Process()
    )
    experiment.guard(f"readiness:{unit}:source-replay-complete")
    stage_context["stage"] = f"{unit}:mlflow-probe"
    probe = measure_readiness_mlflow_probe(
        authorization_sha256=authorization_sha256, unit=unit,
        output=output / "mlflow-probe", process=psutil.Process(),
    )
    experiment.guard(f"readiness:{unit}:mlflow-probe-complete")
    stage_context["stage"] = f"{unit}:readiness-infrastructure"
    infrastructure = measure_readiness_infrastructure_unit(
        record, output / "readiness-infrastructure", psutil.Process()
    )
    experiment.guard(f"readiness:{unit}:infrastructure-complete")
    if repeat_index is None:
        record["actual_source_replay"] = replay
        record["mlflow_probe"] = probe
        record["readiness_infrastructure"] = infrastructure
        if launcher_timing is None:
            raise RuntimeError("readiness warmup lacks launcher timing")
        record["launcher_timing"] = launcher_timing
    else:
        record["components"]["actual_source_replay"] = replay
        record["components"]["mlflow_probe"] = probe
        record["components"]["readiness_infrastructure"] = infrastructure
    record["total_timing"] = aggregate_readiness_timing(
        record["base_total_timing_before_readiness_additions"],
        replay["timing"], probe["timing"], infrastructure["timing"],
    )
    stage_context["stage"] = f"{unit}:augmented-validation"
    validate_complete_readiness_timing(record)
    cal.validate_calibration_repeat(record, config)
    experiment.guard(f"readiness:{unit}:augmented-validation-complete")
    stage_context["stage"] = f"{unit}:augmented-record-publication"
    _atomic_write(record_path, canonical_json_bytes(record, pretty=True))
    experiment.guard(f"readiness:{unit}:augmented-record-published")
    stage_context["stage"] = f"{unit}:fresh-chronology-seal"
    reference = _write_readiness_chronology_seal(
        attempt=attempt, root=int(experiment.root_seed),
        authorization_sha256=authorization_sha256,
        ordinal=ordinal, unit=unit, repeat_index=repeat_index,
        predecessor_seal_sha256=predecessor_sha,
        start=start, record_path=record_path,
        reused_from_augmented_cache=False, source_cache_sha256=None,
    )
    experiment.guard(f"readiness:{unit}:fresh-augmented-unit-sealed")
    result = {
        **record,
        "readiness_chronology": reference,
        "reused_from_augmented_cache": False,
        "augmented_cache_path": None,
    }
    if repeat_index is not None:
        stage_context["stage"] = f"{unit}:augmented-cache-publication"
        cache_path, _cache_sha = save_augmented_repeat_cache(
            experiment=experiment, authorization_sha256=authorization_sha256,
            repeat_index=repeat_index, config=config, record=result,
        )
        result["augmented_cache_path"] = _path_for_receipt(cache_path)
        experiment.guard(f"readiness:{unit}:augmented-cache-preserved-after-seal")
    return result


def execute_synchronous_augmented_readiness_units(
    *, experiment: Any, attempt: Path, authorization_sha256: str,
    config: dict[str, Any], launcher_timing: dict[str, Any],
    stage_context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    predecessor = None
    for ordinal, (unit, repeat_index) in enumerate(READINESS_AUGMENTED_UNIT_ORDER):
        record = _execute_one_augmented_readiness_unit(
            experiment=experiment, attempt=attempt,
            authorization_sha256=authorization_sha256, config=config,
            ordinal=ordinal, unit=unit, repeat_index=repeat_index,
            predecessor=predecessor,
            launcher_timing=launcher_timing if ordinal == 0 else None,
            stage_context=stage_context,
        )
        records.append(record)
        predecessor = record["readiness_chronology"]
    chronology = {
        "schema_version": "exp648-readiness-augmented-chronology-v1",
        "root": int(experiment.root_seed),
        "authorization_sha256": authorization_sha256,
        "contract_sha256": READINESS_CHRONOLOGY_CONTRACT_SHA256,
        "order": [name for name, _ in READINESS_AUGMENTED_UNIT_ORDER],
        "units": [record["readiness_chronology"] for record in records],
        "complete": True,
    }
    payload = {
        "calibration_root": int(experiment.root_seed),
        "calibration_authorization_sha256": authorization_sha256,
        "warmup": records[0], "repeats": records[1:],
        "chronology": chronology,
    }
    stage_context["stage"] = "all-units:chronology-validation"
    validate_readiness_chronology(payload)
    experiment.guard("readiness:all-synchronous-augmented-units-sealed")
    return records[0], records[1:], chronology


def _read_regular_file_no_follow(path: Path, *, label: str) -> bytes:
    if not os.path.lexists(path):
        raise RuntimeError(f"{label} disappeared during validation")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{label} is not a no-follow regular file")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} cannot be opened without following links") from error
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"{label} filesystem identity changed")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_exclusive_readiness_attempt_directory(path: Path) -> None:
    if not os.path.lexists(path):
        raise RuntimeError("claimed readiness attempt directory disappeared")
    observed = os.lstat(path)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise RuntimeError("claimed readiness attempt path is not a no-follow directory")


def _parsed_readiness_attempt_directories(root: Path, stem: str) -> list[tuple[int, Path]]:
    if not os.path.lexists(root):
        return []
    root_state = os.lstat(root)
    if stat.S_ISLNK(root_state.st_mode) or not stat.S_ISDIR(root_state.st_mode):
        raise RuntimeError("readiness attempt root is not a no-follow directory")
    pattern = re.compile(rf"^{re.escape(stem)}(?:-retry-(\d{{3,}}))?$")
    parsed: list[tuple[int, Path]] = []
    with os.scandir(root) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if not entry.name.startswith(stem):
                continue
            match = pattern.fullmatch(entry.name)
            if (
                match is None
                or entry.is_symlink()
                or not entry.is_dir(follow_symlinks=False)
            ):
                raise RuntimeError("malformed, linked, or non-directory readiness attempt path")
            suffix = 0 if match.group(1) is None else int(match.group(1))
            parsed.append((suffix, root / entry.name))
    if len({suffix for suffix, _ in parsed}) != len(parsed):
        raise RuntimeError("colliding readiness attempt suffix")
    return parsed


def _read_readiness_attempt_state(path: Path) -> dict[str, Any]:
    raw = _read_regular_file_no_follow(path, label="readiness attempt state")
    try:
        state = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("existing readiness attempt state is malformed") from error
    if not isinstance(state, dict) or raw != canonical_json_bytes(state, pretty=True):
        raise RuntimeError("existing readiness attempt state is noncanonical")
    base_keys = {
        "schema_version", "attempt_id", "authorization_sha256", "phase", "complete",
    }
    allowed_keys = base_keys | {
        "completion_manifest_sha256", "exception_receipt_sha256",
        "failure_publication_error",
    }
    if not base_keys.issubset(state) or not set(state).issubset(allowed_keys):
        raise RuntimeError("existing readiness attempt state key schema mismatch")
    if state["schema_version"] != "exp648-readiness-calibration-attempt-v1":
        raise RuntimeError("existing readiness attempt state schema mismatch")
    if not isinstance(state["authorization_sha256"], str) or len(state["authorization_sha256"]) != 64:
        raise RuntimeError("existing readiness attempt authorization binding is invalid")
    if state["phase"] not in {
        "authorized", "sealed", "failed", "failure-publication-failed",
    } or not isinstance(state["complete"], bool):
        raise RuntimeError("existing readiness attempt phase is invalid")
    if (state["phase"] == "sealed") is not state["complete"]:
        raise RuntimeError("existing readiness attempt terminal state is inconsistent")
    if state["phase"] == "sealed":
        digest = state.get("completion_manifest_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("sealed readiness attempt completion binding is invalid")
        if set(state) != base_keys | {"completion_manifest_sha256"}:
            raise RuntimeError("sealed readiness attempt state key schema mismatch")
    elif state["phase"] == "failed":
        digest = state.get("exception_receipt_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("failed readiness attempt receipt binding is invalid")
        if set(state) != base_keys | {"exception_receipt_sha256"}:
            raise RuntimeError("failed readiness attempt state key schema mismatch")
    elif state["phase"] == "failure-publication-failed":
        failure = state.get("failure_publication_error")
        if (
            not isinstance(failure, dict)
            or set(failure) != {"type", "module"}
            or not all(isinstance(value, str) and value for value in failure.values())
            or set(state) != base_keys | {"failure_publication_error"}
        ):
            raise RuntimeError("failed-publication attempt state binding is invalid")
    elif set(state) != base_keys:
        raise RuntimeError("incomplete readiness attempt has a completion binding")
    return state


def _validate_failed_attempt_receipt(
    directory: Path, state: dict[str, Any]
) -> dict[str, Any]:
    path = directory / "catchable-exception-boundary.json"
    raw = _read_regular_file_no_follow(path, label="catchable exception receipt")
    try:
        receipt = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("catchable exception receipt is malformed") from error
    if not isinstance(receipt, dict) or raw != canonical_json_bytes(receipt, pretty=True):
        raise RuntimeError("catchable exception receipt is noncanonical")
    if sha256_bytes(raw) != state["exception_receipt_sha256"]:
        raise RuntimeError("failed readiness attempt receipt hash mismatch")
    context = receipt.get("context", {})
    if (
        receipt.get("schema_version") != "exp648-catchable-exception-boundary-v1"
        or receipt.get("status") != "consumed-incomplete-non-scientific"
        or receipt.get("complete") is not False
        or context.get("attempt_id") != state["attempt_id"]
        or context.get("authorization_sha256") != state["authorization_sha256"]
    ):
        raise RuntimeError("failed readiness attempt receipt/state binding mismatch")
    return receipt


def assert_readiness_authorization_not_sealed_or_published(
    *, authorization_sha256: str, attempts_root: Path, tracked_root: Path,
) -> list[Path]:
    stem = f"cal-{authorization_sha256[:16]}"
    prior = [item for _, item in _parsed_readiness_attempt_directories(attempts_root, stem)]
    for directory in prior:
        state_path = directory / "attempt-state.json"
        receipt_path = directory / "catchable-exception-boundary.json"
        if os.path.lexists(state_path):
            state = _read_readiness_attempt_state(state_path)
            if state["attempt_id"] != directory.name:
                raise RuntimeError("existing readiness attempt ID binding mismatch")
            if state["authorization_sha256"] != authorization_sha256:
                raise RuntimeError("existing readiness attempt provenance mismatch")
            if state["phase"] == "failed":
                _validate_failed_attempt_receipt(directory, state)
                raise PermissionError(
                    "failed readiness calibration attempt cannot be relaunched"
                )
            if state["phase"] == "failure-publication-failed":
                if os.path.lexists(receipt_path):
                    _read_regular_file_no_follow(
                        receipt_path, label="unexpected catchable exception receipt"
                    )
                    raise RuntimeError(
                        "failed-publication state unexpectedly has an exception receipt"
                    )
                raise PermissionError(
                    "failed-publication readiness attempt cannot be relaunched"
                )
            if os.path.lexists(receipt_path):
                _read_regular_file_no_follow(
                    receipt_path, label="unbound catchable exception receipt"
                )
                raise RuntimeError(
                    "catchable exception receipt exists without failed attempt state"
                )
            if state["complete"] or state["phase"] == "sealed":
                raise PermissionError(
                    "sealed readiness calibration attempt cannot be relaunched"
                )
        elif os.path.lexists(receipt_path):
            _read_regular_file_no_follow(
                receipt_path, label="orphan catchable exception receipt"
            )
            raise RuntimeError("catchable exception receipt lacks attempt state")
        seal_path = directory / "completion-seal.json"
        if os.path.lexists(seal_path):
            raw = _read_regular_file_no_follow(
                seal_path, label="readiness attempt completion seal"
            )
            try:
                seal = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError("readiness attempt completion seal is malformed") from error
            if not isinstance(seal, dict) or raw != canonical_json_bytes(seal, pretty=True):
                raise RuntimeError("readiness attempt completion seal is noncanonical")
            if seal.get("authorization_sha256") != authorization_sha256:
                raise RuntimeError("readiness attempt completion seal binding mismatch")
            raise PermissionError("sealed readiness calibration attempt cannot be relaunched")
    tracked_names = (
        "readiness-calibration.json",
        "readiness-calibration-core.json",
        "readiness-calibration-environment.json",
        "readiness-calibration-mlflow.json",
        "readiness-calibration-executed-config.yaml",
        "readiness-calibration-completion.json",
        "readiness-calibration-resource-completion.json",
        "readiness-calibration-attempt-state.json",
        "readiness-calibration-launch-ledger.json",
    )
    for name in tracked_names:
        path = tracked_root / name
        if os.path.lexists(path):
            _read_regular_file_no_follow(path, label="tracked readiness publication")
            raise PermissionError(
                "live tracked readiness output singleton blocks calibration launch"
            )
    return prior


def allocate_readiness_calibration_attempt(
    *, authorization_sha256: str, attempts_root: Path, tracked_root: Path,
) -> tuple[str, Path, list[Path]]:
    prior = assert_readiness_authorization_not_sealed_or_published(
        authorization_sha256=authorization_sha256,
        attempts_root=attempts_root,
        tracked_root=tracked_root,
    )
    stem = f"cal-{authorization_sha256[:16]}"
    attempts_root.mkdir(parents=True, exist_ok=True)
    parsed = _parsed_readiness_attempt_directories(attempts_root, stem)
    suffix = 0 if not parsed else max(value for value, _ in parsed) + 1
    while True:
        attempt_id = stem if suffix == 0 else f"{stem}-retry-{suffix:03d}"
        directory = attempts_root / attempt_id
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError:
            assert_readiness_authorization_not_sealed_or_published(
                authorization_sha256=authorization_sha256,
                attempts_root=attempts_root,
                tracked_root=tracked_root,
            )
            parsed = _parsed_readiness_attempt_directories(attempts_root, stem)
            suffix = max(value for value, _ in parsed) + 1
            continue
        _verify_exclusive_readiness_attempt_directory(directory)
        assert_readiness_authorization_not_sealed_or_published(
            authorization_sha256=authorization_sha256,
            attempts_root=attempts_root,
            tracked_root=tracked_root,
        )
        _verify_exclusive_readiness_attempt_directory(directory)
        return attempt_id, directory, prior


@contextlib.contextmanager
def readiness_calibration_authorization_lock(
    *, authorization_sha256: str, scratch_root: Path,
):
    lock_root = scratch_root / "readiness-calibration-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    root_state = os.lstat(lock_root)
    if stat.S_ISLNK(root_state.st_mode) or not stat.S_ISDIR(root_state.st_mode):
        raise RuntimeError("readiness lock root is not a no-follow directory")
    lock_path = lock_root / f"cal-{authorization_sha256[:16]}.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError("readiness authorization lock is unsafe") from error
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_nlink != 1
        ):
            raise RuntimeError("readiness authorization lock identity is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PermissionError(
                "concurrent readiness calibration authorization is already active"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def execute_readiness_calibration(
    authorization: dict[str, Any], *, launch_wall_started: float,
    launch_cpu_started: float, launch_peak_rss: int,
) -> dict[str, Any]:
    auth_sha = authorization["authorization_sha256"]
    with readiness_calibration_authorization_lock(
        authorization_sha256=auth_sha, scratch_root=SCRATCH_SCREEN_DIR,
    ):
        return _execute_readiness_calibration_locked(
            authorization,
            launch_wall_started=launch_wall_started,
            launch_cpu_started=launch_cpu_started,
            launch_peak_rss=launch_peak_rss,
        )


class CatchableExceptionReceiptPublicationError(RuntimeError):
    """Fail-closed signal when an attempt failure receipt cannot be published."""

    def __init__(self, original: Exception, publication: Exception) -> None:
        super().__init__("catchable-exception boundary receipt publication failed")
        self.original_exception = original
        self.publication_exception = publication


def _attempt_artifact_boundary(attempt: Path) -> dict[str, Any]:
    receipt_name = "catchable-exception-boundary.json"
    files = sorted(
        path for path in attempt.rglob("*")
        if path.is_file() and path.name != receipt_name
    )
    manifest = {
        path.relative_to(attempt).as_posix(): sha256(path) for path in files
    }
    manifest_bytes = "".join(
        f"{digest}  {name}\n" for name, digest in manifest.items()
    ).encode("utf-8")
    seals = sorted((attempt / "augmented-chronology").glob("*-seal.json"))
    last_seal = None
    if seals:
        seal_path = seals[-1]
        seal_raw = seal_path.read_bytes()
        seal = json.loads(seal_raw)
        if seal_raw != canonical_json_bytes(seal, pretty=True):
            raise RuntimeError("last completed chronology seal is noncanonical")
        last_seal = {
            "path": seal_path.relative_to(attempt).as_posix(),
            "sha256": sha256_bytes(seal_raw),
            "ordinal": seal.get("ordinal"),
            "unit": seal.get("unit"),
            "augmented_record_sha256": seal.get("augmented_record_sha256"),
        }
    return {
        "scope": "all-attempt-files-present-before-exception-receipt-publication",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "sha256sum_manifest_bytes": len(manifest_bytes),
        "sha256sum_manifest_sha256": sha256_bytes(manifest_bytes),
        "files": manifest,
        "last_completed_chronology_seal": last_seal,
    }


def write_catchable_exception_boundary_receipt(
    *, attempt: Path, attempt_id: str, root_seed: int,
    authorization_sha256: str, stage: str, error: Exception,
    launch_wall_started: float, launch_cpu_started: float,
    launch_peak_rss: int, last_guard_check: dict[str, Any] | None,
) -> dict[str, Any]:
    artifact_boundary = _attempt_artifact_boundary(attempt)
    process = psutil.Process()
    cpu = process.cpu_times()
    wall_sample = time.perf_counter()
    observation = {
        "monotonic_wall_seconds": wall_sample - launch_wall_started,
        "process_user_plus_system_cpu_seconds": (
            float(cpu.user + cpu.system) - launch_cpu_started
        ),
        "peak_rss_bytes": max(
            int(launch_peak_rss), cal.process_peak_rss_bytes(process)
        ),
        "wall_sample_perf_counter": wall_sample,
        "launch_wall_perf_counter": launch_wall_started,
        "process_user_seconds_at_sample": float(cpu.user),
        "process_system_seconds_at_sample": float(cpu.system),
        "launch_process_user_plus_system_seconds": launch_cpu_started,
    }
    context = {
        "stage": stage,
        "root": int(root_seed),
        "authorization_sha256": authorization_sha256,
        "attempt_id": attempt_id,
    }
    payload = {
        "schema_version": "exp648-catchable-exception-boundary-v1",
        "status": "consumed-incomplete-non-scientific",
        "context": context,
        "exception": {
            "type": type(error).__name__,
            "module": type(error).__module__,
            "message": str(error),
        },
        "resource_observation": observation,
        "artifact_boundary": artifact_boundary,
        "last_completed_chronology_seal": artifact_boundary[
            "last_completed_chronology_seal"
        ],
        "last_guard_check": copy.deepcopy(last_guard_check),
        "coverage": {
            "ends_at": "catchable-exception-boundary-before-receipt-serialization",
            "excluded": [
                "receipt serialization",
                "receipt atomic write and durability sync",
                "terminal failed attempt-state serialization and atomic write",
                "exception re-raise",
                "stdout or stderr publication",
                "stack unwind",
                "context-manager and interpreter teardown",
                "process exit",
            ],
            "post_exit_coverage_claimed": False,
        },
        "complete": False,
        "scientific_accuracy_or_selection_reported": False,
    }
    _atomic_write(
        attempt / "catchable-exception-boundary.json",
        canonical_json_bytes(payload, pretty=True),
    )
    return payload


def write_terminal_failed_attempt_state(
    *, attempt: Path, attempt_id: str, authorization_sha256: str,
    exception_receipt_sha256: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", exception_receipt_sha256) is None:
        raise RuntimeError("terminal failed attempt receipt hash is invalid")
    payload = {
        "schema_version": "exp648-readiness-calibration-attempt-v1",
        "attempt_id": attempt_id,
        "authorization_sha256": authorization_sha256,
        "phase": "failed",
        "complete": False,
        "exception_receipt_sha256": exception_receipt_sha256,
    }
    _atomic_write(
        attempt / "attempt-state.json", canonical_json_bytes(payload, pretty=True)
    )
    return payload


def _execute_readiness_calibration_locked(
    authorization: dict[str, Any], *, launch_wall_started: float,
    launch_cpu_started: float, launch_peak_rss: int,
) -> dict[str, Any]:
    root_seed = int(authorization["calibration_root"])
    auth_sha = authorization["authorization_sha256"]
    attempt_id, attempt, _prior_attempts = allocate_readiness_calibration_attempt(
        authorization_sha256=auth_sha,
        attempts_root=SCRATCH_SCREEN_DIR / "readiness-calibration-attempts",
        tracked_root=TRACKED_SCREEN_DIR,
    )
    _atomic_write(attempt / "attempt-state.json", canonical_json_bytes({
        "schema_version": "exp648-readiness-calibration-attempt-v1",
        "attempt_id": attempt_id, "authorization_sha256": auth_sha,
        "phase": "authorized", "complete": False,
    }, pretty=True))
    stage_context: dict[str, Any] = {
        "stage": "authorized-attempt-created",
        "guard": None,
    }
    try:
        return _execute_readiness_calibration_attempt_body(
            authorization,
            attempt_id=attempt_id,
            attempt=attempt,
            stage_context=stage_context,
            launch_wall_started=launch_wall_started,
            launch_cpu_started=launch_cpu_started,
            launch_peak_rss=launch_peak_rss,
        )
    except Exception as error:
        guard = stage_context.get("guard")
        last_guard_check = None
        if guard is not None and getattr(guard, "checks", None):
            last_guard_check = copy.deepcopy(guard.checks[-1])
        exception_context = {
            "stage": stage_context["stage"],
            "root": root_seed,
            "authorization_sha256": auth_sha,
            "attempt_id": attempt_id,
            "last_guard_check": last_guard_check,
        }
        try:
            error.exp648_attempt_context = exception_context
        except Exception:
            # Exception subclasses need not permit arbitrary attributes; that
            # must never prevent the attempt-bound failure receipt itself.
            pass
        try:
            write_catchable_exception_boundary_receipt(
                attempt=attempt,
                attempt_id=attempt_id,
                root_seed=root_seed,
                authorization_sha256=auth_sha,
                stage=stage_context["stage"],
                error=error,
                launch_wall_started=launch_wall_started,
                launch_cpu_started=launch_cpu_started,
                launch_peak_rss=launch_peak_rss,
                last_guard_check=last_guard_check,
            )
        except Exception as publication_error:
            emergency = {
                "schema_version": "exp648-readiness-calibration-attempt-v1",
                "attempt_id": attempt_id,
                "authorization_sha256": auth_sha,
                "phase": "failure-publication-failed",
                "complete": False,
                "failure_publication_error": {
                    "type": type(publication_error).__name__,
                    "module": type(publication_error).__module__,
                },
            }
            try:
                _atomic_write(
                    attempt / "attempt-state.json",
                    canonical_json_bytes(emergency, pretty=True),
                )
            except Exception:
                pass
            raise CatchableExceptionReceiptPublicationError(
                error, publication_error
            ) from publication_error
        receipt_path = attempt / "catchable-exception-boundary.json"
        try:
            write_terminal_failed_attempt_state(
                attempt=attempt,
                attempt_id=attempt_id,
                authorization_sha256=auth_sha,
                exception_receipt_sha256=sha256(receipt_path),
            )
        except Exception as publication_error:
            raise CatchableExceptionReceiptPublicationError(
                error, publication_error
            ) from publication_error
        raise


def _execute_readiness_calibration_attempt_body(
    authorization: dict[str, Any], *, attempt_id: str, attempt: Path,
    stage_context: dict[str, Any], launch_wall_started: float,
    launch_cpu_started: float, launch_peak_rss: int,
) -> dict[str, Any]:
    root_seed = int(authorization["calibration_root"])
    auth_sha = authorization["authorization_sha256"]
    stage_context["stage"] = "executed-config-publication"
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["resource_protocol"]["calibration"]["root"] = root_seed
    config["seed_roles"]["calibration_root"]["seed"] = root_seed
    config_path = attempt / "executed-config.yaml"
    _atomic_write(config_path, yaml.safe_dump(config, sort_keys=False).encode("utf-8"))

    original_root = cal.CALIBRATION_ROOT
    original_allowed = cal.ALLOWED_MODE_ROOTS["calibration"]
    original_preserve = cal.preserve_executed_config
    def preserve_local(source: Path, raw_output: Path, tracked_output_directory: Path | None = None) -> dict[str, Any]:
        payload = source.read_bytes()
        raw = raw_output / "executed-config.yaml"
        tracked = attempt / "evidence" / "executed-config.yaml"
        _atomic_write(raw, payload); _atomic_write(tracked, payload)
        digest = sha256_bytes(payload)
        return {"sha256": digest, "bytes": len(payload), "raw_path": raw.relative_to(ROOT).as_posix(), "raw_sha256": digest, "tracked_path": tracked.relative_to(ROOT).as_posix(), "tracked_sha256": digest, "raw_tracked_byte_identical": True, "preserved_before_root_authorization": True}
    try:
        stage_context["stage"] = "calibration-constructor"
        cal.CALIBRATION_ROOT = root_seed
        cal.ALLOWED_MODE_ROOTS["calibration"] = frozenset((root_seed,))
        cal.preserve_executed_config = preserve_local
        experiment = cal.CalibrationExperiment(config_path, attempt / "raw")
        stage_context["guard"] = experiment.guard
        stage_context["stage"] = "cumulative-charge-application"
        apply_readiness_carried_charge_once(experiment.guard)
        setup_process = psutil.Process()
        setup_cpu = setup_process.cpu_times()
        prewarmup_launcher = {
            "wall_seconds": time.perf_counter() - launch_wall_started,
            "aggregate_cpu_seconds": float(setup_cpu.user + setup_cpu.system) - launch_cpu_started,
            "peak_rss_bytes": max(launch_peak_rss, cal.process_peak_rss_bytes(setup_process)),
            "captured_before_warmup_and_repeats": True,
        }
        stage_context["stage"] = "synchronous-augmented-unit-execution"
        warmup, repeats, chronology = execute_synchronous_augmented_readiness_units(
            experiment=experiment,
            attempt=attempt,
            authorization_sha256=auth_sha,
            config=config,
            launcher_timing=prewarmup_launcher,
            stage_context=stage_context,
        )
    finally:
        cal.CALIBRATION_ROOT = original_root
        cal.ALLOWED_MODE_ROOTS["calibration"] = original_allowed
        cal.preserve_executed_config = original_preserve
    stage_context["stage"] = "dynamic-budget-resolution"
    launcher = prewarmup_launcher
    budgets, component_resolution = resolve_readiness_screen_budgets(
        repeats, launcher, warmup
    )
    environment = experiment.environment
    environment_path = attempt / "environment.json"
    _atomic_write(environment_path, canonical_json_bytes(environment, pretty=True))
    final_process = psutil.Process()
    final_cpu = final_process.cpu_times()
    active_wall = time.perf_counter() - launch_wall_started
    active_cpu = float(final_cpu.user + final_cpu.system) - launch_cpu_started
    final_peak_rss = max(launch_peak_rss, cal.process_peak_rss_bytes(final_process))
    accounting = readiness_accounting_decomposition()
    reused_wall = float(experiment.guard.reused_wall_seconds)
    reused_cpu = float(experiment.guard.reused_cpu_seconds)
    recovery = {
        "scope": "prepublication-provisional-snapshot-before-inner-publication",
        "accounting_formula": "completed_history + failed_attempt_conservative_allowance + prepublication_provisional_active + prepublication_provisional_reused",
        "augmented_unit_orchestration": {
            "chronology_contract_sha256": READINESS_CHRONOLOGY_CONTRACT_SHA256,
            "execution_order": [name for name, _ in READINESS_AUGMENTED_UNIT_ORDER],
            "old_base_only_cache_loaded_or_saved": False,
            "fresh_augmented_warmup_before_repeat_cache_reuse": True,
            "carried_charge_applied_exactly_once": bool(
                getattr(
                    experiment.guard, "readiness_carried_charge_applied_exactly_once", False
                )
            ),
            "guard_carried_wall_seconds": float(
                experiment.guard.prior_superseded_wall_seconds
            ),
            "guard_carried_aggregate_cpu_seconds": float(
                experiment.guard.prior_superseded_cpu_seconds
            ),
            "guard_configured_wall_ceiling_seconds": float(
                experiment.guard.wall_ceiling
            ),
            "guard_configured_aggregate_cpu_ceiling_seconds": float(
                experiment.guard.cpu_ceiling
            ),
            "reused_complete_augmented_repeat_indices": [
                int(record["repeat_index"])
                for record in repeats
                if record.get("reused_from_augmented_cache") is True
            ],
            "all_augmented_units_validated_and_sealed_before_next_start": True,
        },
        "prior_superseded_calibrations": list(READINESS_SUPERSEDED_CALIBRATIONS),
        **accounting,
        "carried_charge_applied_exactly_once": True,
        "prepublication_provisional_active_wall_seconds": active_wall,
        "prepublication_provisional_active_aggregate_cpu_seconds": active_cpu,
        "prepublication_provisional_reused_wall_seconds_charged": reused_wall,
        "prepublication_provisional_reused_aggregate_cpu_seconds_charged": reused_cpu,
        "prepublication_provisional_cumulative_carried_active_reused_wall_seconds": READINESS_CARRIED_CHARGE["wall_seconds"] + active_wall + reused_wall,
        "prepublication_provisional_cumulative_carried_active_reused_aggregate_cpu_seconds": READINESS_CARRIED_CHARGE["aggregate_cpu_seconds"] + active_cpu + reused_cpu,
        "prepublication_provisional_process_peak_rss_bytes": final_peak_rss,
        "active_snapshot_boundary": "after synchronous base-to-augmentation validation and atomic chronology sealing for the warmup and all three repeats, including setup and launcher; before immutable evidence publication",
    }
    boundary = {
        "calibration_only": True,
        "screen_roots_opened": False,
        "confirmation_roots_opened": False,
        "checked_independent_official_private_opened": False,
        "scientific_accuracy_or_selection_reported": False,
        "model_update_performed": False,
        "fidelity_reduced": False,
    }
    core = {
        "schema_version": "exp648-screen-readiness-calibration-core-v1",
        "authorization": authorization, "attempt_id": attempt_id,
        "warmup": warmup, "repeats": repeats,
        "chronology": chronology,
        "component_resolution": component_resolution,
        "resolved_screen_budgets": budgets, "environment": environment,
        "recovery_charges": recovery,
        "full_fidelity": True, "boundary": boundary,
    }
    core_path = attempt / "calibration-core.json"
    _atomic_write(core_path, canonical_json_bytes(core, pretty=True))
    core_sha = sha256(core_path)
    receipt_stub = {
        "schema_version": "exp648-screen-readiness-calibration-v1",
        "status": "complete-non-scientific-full-fidelity-resource-calibration",
        "calibration_root": root_seed, "calibration_authorization_sha256": auth_sha,
        "scientific_runner_sha256": sha256(Path(__file__)),
        "resolved_config_sha256": sha256(CONFIG_PATH),
        "screen_workload_sha256": sha256_bytes(canonical_json_bytes(screen_workload_payload())),
        "calibration_evidence_core_sha256": core_sha,
        "environment_sha256": sha256(environment_path), "mlflow_receipt_sha256": "pending",
        "warmups": 1, "timed_repeats": 3,
        "warmup": warmup,
        "repeats": repeats, "chronology": chronology,
        "component_resolution": component_resolution,
        "resolved_screen_budgets": budgets, "recovery_charges": recovery,
        "full_fidelity": True, "environment": environment, "boundary": boundary,
    }
    validate_core_receipt_release_equality(core, receipt_stub)
    # MLflow keys use the immutable core, not the pending final receipt.
    stage_context["stage"] = "mlflow-probe-publication-and-verification"
    mlflow_receipt = log_and_verify_readiness_calibration_mlflow(receipt_stub, [core_path, environment_path, config_path])
    mlflow_path = attempt / "mlflow.json"
    _atomic_write(mlflow_path, canonical_json_bytes(mlflow_receipt, pretty=True))
    receipt = {**receipt_stub, "mlflow_receipt_sha256": sha256(mlflow_path)}
    validate_readiness_calibration_receipt(receipt)
    receipt_path = attempt / "readiness-calibration.json"
    _atomic_write(receipt_path, canonical_json_bytes(receipt, pretty=True))
    stage_context["stage"] = "tracked-inner-artifact-publication"
    for source, name in ((receipt_path, "readiness-calibration.json"), (core_path, "readiness-calibration-core.json"), (environment_path, "readiness-calibration-environment.json"), (mlflow_path, "readiness-calibration-mlflow.json"), (config_path, "readiness-calibration-executed-config.yaml")):
        _atomic_write(TRACKED_SCREEN_DIR / name, source.read_bytes())
    manifest = {
        "schema_version": "exp648-readiness-calibration-completion-v1",
        "attempt_id": attempt_id, "authorization_sha256": auth_sha,
        "artifacts": {name: sha256(TRACKED_SCREEN_DIR / name) for name in ("readiness-calibration.json", "readiness-calibration-core.json", "readiness-calibration-environment.json", "readiness-calibration-mlflow.json", "readiness-calibration-executed-config.yaml")},
        "complete": True,
    }
    _atomic_write(TRACKED_SCREEN_DIR / "readiness-calibration-completion.json", canonical_json_bytes(manifest, pretty=True))
    final_attempt_state = {
        "schema_version": "exp648-readiness-calibration-attempt-v1", "attempt_id": attempt_id,
        "authorization_sha256": auth_sha, "phase": "sealed", "complete": True,
        "completion_manifest_sha256": sha256(TRACKED_SCREEN_DIR / "readiness-calibration-completion.json"),
    }
    final_attempt_state_bytes = canonical_json_bytes(final_attempt_state, pretty=True)
    _atomic_write(attempt / "attempt-state.json", final_attempt_state_bytes)
    tracked_attempt_state = TRACKED_SCREEN_DIR / "readiness-calibration-attempt-state.json"
    _atomic_write(tracked_attempt_state, final_attempt_state_bytes)
    _atomic_write(
        TRACKED_SCREEN_DIR / "readiness-calibration-launch-ledger.json",
        final_attempt_state_bytes,
    )
    inner_paths = {
        name: TRACKED_SCREEN_DIR / name
        for name in (
            "readiness-calibration.json",
            "readiness-calibration-core.json",
            "readiness-calibration-environment.json",
            "readiness-calibration-mlflow.json",
            "readiness-calibration-executed-config.yaml",
            "readiness-calibration-completion.json",
            "readiness-calibration-attempt-state.json",
            "readiness-calibration-launch-ledger.json",
        )
    }
    # All feasible inner hashing and lifecycle checks happen before the sample.
    # The remaining explicitly enumerated operations are outside observed totals.
    stage_context["stage"] = "outer-resource-completion-publication"
    write_readiness_resource_completion(
        inner_paths=inner_paths,
        attempt_id=attempt_id,
        authorization_sha256=auth_sha,
        launch_wall_started=launch_wall_started,
        launch_cpu_started=launch_cpu_started,
        launch_peak_rss=launch_peak_rss,
        reused_wall=reused_wall,
        reused_cpu=reused_cpu,
    )
    return receipt


def screen_readiness_binding(approved_base_commit: str) -> dict[str, Any]:
    config_rel = CONFIG_PATH.relative_to(ROOT).as_posix()
    config = yaml.safe_load(_git_blob(approved_base_commit, config_rel))
    dependency_manifest = config["executable_dependency_sources"]
    dependency_hash = dependency_manifest["manifest_sha256"]
    scientific_hash, resource_hash = _base_protocol_hashes(approved_base_commit, config)
    calibration_rel = SCREEN_READINESS_CALIBRATION_PATH.relative_to(ROOT).as_posix()
    calibration = _git_json(approved_base_commit, calibration_rel)
    resource_completion_rel = READINESS_RESOURCE_COMPLETION_PATH.relative_to(ROOT).as_posix()
    resource_completion = _git_json(approved_base_commit, resource_completion_rel)
    scaffold = _git_json(
        approved_base_commit,
        READINESS_SCAFFOLD_PROVENANCE_PATH.relative_to(ROOT).as_posix(),
    )
    frozen_paths = {
        "data/identity-manifest.json": DATA_DIR / "identity-manifest.json",
        "data/trajectory-manifest.json.gz": DATA_DIR / "trajectory-manifest.json.gz",
        "data/paired-corpus.json.gz": DATA_DIR / "paired-corpus.json.gz",
        "data/corpus-manifest.json": DATA_DIR / "corpus-manifest.json",
        "evidence/run.json": TRACKED_CORPUS_DIR / "run.json",
        "evidence/environment.json": TRACKED_CORPUS_DIR / "environment.json",
        "evidence/integrity.json": TRACKED_CORPUS_DIR / "integrity.json",
        "evidence/mlflow.json": TRACKED_CORPUS_DIR / "mlflow.json",
    }
    frozen_hashes = {
        name: _git_blob_sha256(approved_base_commit, path.relative_to(ROOT).as_posix())
        for name, path in frozen_paths.items()
    }
    runner_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    return {
        "approved_base_commit": approved_base_commit,
        "corpus_manifest_sha256": frozen_hashes["data/corpus-manifest.json"],
        "historical_corpus_commit": scaffold["historical_corpus_commit"],
        "frozen_corpus_file_sha256": frozen_hashes,
        "corpus_provenance_amendment_sha256": _git_blob_sha256(approved_base_commit, (TRACKED_CORPUS_DIR / "provenance-amendment.json").relative_to(ROOT).as_posix()),
        "scientific_runner_sha256": _git_blob_sha256(approved_base_commit, runner_rel),
        "scientific_test_sha256": _git_blob_sha256(approved_base_commit, "tests/test_exp648_structured_trajectory_scientific.py"),
        "calibration_runner_sha256": _git_blob_sha256(approved_base_commit, CALIBRATION_RUNNER.relative_to(ROOT).as_posix()),
        "resolved_config_sha256": _git_blob_sha256(approved_base_commit, config_rel),
        "executed_config_sha256": _git_blob_sha256(approved_base_commit, EXECUTED_CONFIG_PATH.relative_to(ROOT).as_posix()),
        "screen_workload_sha256": calibration["screen_workload_sha256"],
        "readiness_calibration_sha256": _git_blob_sha256(approved_base_commit, calibration_rel),
        "readiness_calibration_root": int(calibration["calibration_root"]),
        "readiness_calibration_core_sha256": calibration["calibration_evidence_core_sha256"],
        "readiness_calibration_environment_sha256": calibration["environment_sha256"],
        "readiness_calibration_mlflow_receipt_sha256": calibration["mlflow_receipt_sha256"],
        "readiness_resource_completion_sha256": _git_blob_sha256(approved_base_commit, resource_completion_rel),
        "readiness_observed_recovery_sha256": sha256_bytes(
            canonical_json_bytes(resource_completion["recovery_charges"])
        ),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "source_model": copy.deepcopy(config["source_model"]),
        "dependency_manifest_sha256": dependency_hash,
        "dependency_sources": copy.deepcopy(dependency_manifest["sources"]),
        "preregistration_sha256": _git_blob_sha256(approved_base_commit, PREREGISTRATION_PATH.relative_to(ROOT).as_posix()),
        "scientific_protocol_sha256": scientific_hash,
        "resource_protocol_sha256": resource_hash,
        "exact_teacher_source_sha256": _base_exact_teacher_hash(approved_base_commit),
        "evaluator_sha256": _base_evaluator_hash(approved_base_commit, dependency_hash),
        "runtime": {
            "torch_threads": int(config["training"]["torch_threads"]),
            "torch_interop_threads": int(config["training"]["torch_interop_threads"]),
            "deterministic_algorithms": bool(config["training"]["deterministic_algorithms"]),
        },
    }


def expected_screen_authorization_payload(
    corpus_manifest_sha256: str, *, binding: dict[str, Any] | None = None
) -> dict[str, Any]:
    if len(corpus_manifest_sha256) != 64:
        raise ValueError("corpus manifest hash must be SHA-256")
    if binding is None:
        raise ValueError("screen authorization requires an explicit approved-base binding")
    if binding.get("corpus_manifest_sha256") != corpus_manifest_sha256:
        raise ValueError("authorization binding corpus hash mismatch")
    return {
        "schema_version": "exp648-screen-judge-authorization-v1",
        "experiment_id": RUN_ID,
        "decision": "authorized-after-corpus-freeze",
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "screen_readiness_binding": binding,
        "screen_readiness_binding_sha256": sha256_bytes(canonical_json_bytes(binding)),
        "screen_training_root": SCREEN_TRAINING_ROOT,
        "screen_evaluation_root": SCREEN_EVALUATION_ROOT,
        "confirmation_authorized": False,
        "checked_independent_official_authorized": False,
    }


def verify_screen_authorization(
    path: Path = SCREEN_AUTHORIZATION_PATH, *, committed_bytes: bytes | None = None,
    launch: bool = True,
) -> dict[str, Any]:
    # Authenticate the small canonical committed object before expensive binding reads.
    observed, raw = _read_authenticated_committed_manifest(
        path, expected_schema="exp648-screen-judge-authorization-v1",
        committed_bytes=committed_bytes,
    )
    required_keys = {
        "schema_version", "experiment_id", "decision", "corpus_manifest_sha256",
        "screen_readiness_binding", "screen_readiness_binding_sha256",
        "screen_training_root", "screen_evaluation_root", "confirmation_authorized",
        "checked_independent_official_authorized",
    }
    if set(observed) != required_keys:
        raise PermissionError("screen authorization key schema mismatch")
    if observed.get("screen_readiness_binding_sha256") != sha256_bytes(
        canonical_json_bytes(observed.get("screen_readiness_binding"))
    ):
        raise PermissionError("screen authorization embedded binding hash mismatch")
    embedded = observed.get("screen_readiness_binding", {})
    base = str(embedded.get("approved_base_commit", ""))
    binding = screen_readiness_binding(base)
    expected = expected_screen_authorization_payload(
        binding["corpus_manifest_sha256"], binding=binding
    )
    expected_bytes = canonical_json_bytes(expected, pretty=True)
    observed_bytes = raw
    expected_hash = sha256_bytes(expected_bytes)
    observed_hash = sha256_bytes(observed_bytes)
    if observed_hash != expected_hash or observed_bytes != expected_bytes:
        raise PermissionError("EXP-648 screen Judge authorization hash mismatch")
    auth_relative = path.resolve().relative_to(ROOT).as_posix()
    validate_one_child_authorization_lifecycle(
        auth_relative=auth_relative, approved_base_commit=base, launch=launch
    )
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


def reauthorize_screen_root(
    mode: str, root_seed: int, authorization: dict[str, Any], path: Path
) -> int:
    normalized = str(mode).strip().lower()
    if normalized not in ("screen-training", "screen-evaluation"):
        raise PermissionError("lightweight screen reauthorization accepts one exact role")
    expected_bytes = canonical_json_bytes(authorization, pretty=True)
    if not path.is_file() or path.read_bytes() != expected_bytes:
        raise PermissionError("screen authorization bytes changed after verification")
    expected_seed = (
        int(authorization["screen_training_root"])
        if normalized == "screen-training"
        else int(authorization["screen_evaluation_root"])
    )
    if int(root_seed) != expected_seed:
        raise PermissionError(f"screen {normalized} root mismatch")
    return expected_seed


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
    pairs: Sequence[dict[str, Any]], update_index: int, *, order: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    if len(pairs) != CORPUS_TARGET_PAIRS:
        raise RuntimeError("screen requires the complete frozen 512-pair corpus")
    indices = list(range(len(pairs))) if order is None else list(order)
    if len(indices) != len(pairs) or sorted(indices) != list(range(len(pairs))):
        raise RuntimeError("screen pair order must be a complete permutation")
    selected = [pairs[indices[(update_index * 8 + offset) % len(pairs)]] for offset in range(8)]
    return [pair["failure"] for pair in selected] + [pair["success"] for pair in selected]


def frozen_root_pair_order(pair_count: int, corpus_manifest_sha256: str) -> list[int]:
    if pair_count != CORPUS_TARGET_PAIRS:
        raise RuntimeError("screen order requires frozen pair count")
    rng = random.Random(
        cal.derive_seed(SCREEN_TRAINING_ROOT, f"frozen-pair-order:{corpus_manifest_sha256}")
    )
    order = list(range(pair_count))
    rng.shuffle(order)
    return order


def screen_training_update(
    model: Any,
    optimizer: Any,
    windows: Sequence[dict[str, Any]],
    endpoint_loss_weight: float,
    config: dict[str, Any],
    exp300: Any,
    *,
    matched_work: dict[str, Any] | None = None,
    arm: str | None = None,
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
    if matched_work is not None:
        if arm not in ("control", "challenger"):
            raise RuntimeError("matched-work training arm is invalid")
        for category, count in (
            ("clean_forward_steps", WINDOW_LENGTH),
            ("autonomous_forward_steps", WINDOW_LENGTH),
            ("endpoint_loss_computations", 1),
            ("clean_backward_calls", WINDOW_LENGTH),
            ("endpoint_backward_calls", 1),
            ("gradient_clip_calls", 1),
            ("optimizer_steps", 1),
        ):
            record_matched_work(matched_work, arm, category, count)
        record_actual_matched_work(
            matched_work, arm, "training_update_calls", 1
        )
    return {
        "base_clean_loss": clean_total / WINDOW_LENGTH,
        "endpoint_loss": float(endpoint_loss.detach()),
        "endpoint_loss_weight": endpoint_loss_weight,
        "gradient_norm": gradient_norm,
        **{f"endpoint_{key}": value for key, value in endpoint_stats.items()},
    }


def generate_screen_transition_cases(
    root_seed: int, width: int, count: int, authorization_path: Path,
    authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if authorization is None:
        authorize_root("screen-evaluation", root_seed, authorization_path=authorization_path)
    else:
        reauthorize_screen_root("screen-evaluation", root_seed, authorization, authorization_path)
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
    root_seed: int, width: int, count: int, authorization_path: Path,
    authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if authorization is None:
        authorize_root("screen-evaluation", root_seed, authorization_path=authorization_path)
    else:
        reauthorize_screen_root("screen-evaluation", root_seed, authorization, authorization_path)
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
        "predictions": predictions,
        "quotient_predictions": quotient_predictions,
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
    prediction_trace = [list(state)]
    teacher_trace = [list(teacher)]
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
        prediction_trace.append(list(state))
        teacher_trace.append(list(teacher))
    exact = [left == right for left, right in zip(state, teacher, strict=True)]
    clipped = [width + 1 if value is None else value for value in first_errors]
    return {
        "cases": len(exact),
        "exact_accuracy": sum(exact) / len(exact),
        "exact_vector": exact,
        "mean_clipped_first_error_index": sum(clipped) / len(clipped),
        "invalid_rollout_rate": sum(invalid) / len(invalid),
        "output_sha256": sha256_bytes(canonical_json_bytes(state)),
        "prediction_trace_sha256": sha256_bytes(
            canonical_json_bytes(prediction_trace)
        ),
        "predictions": state,
        "teacher_targets": teacher,
        "first_error_indices": clipped,
        "any_intermediate_invalid_vector": invalid,
        "prediction_trace": prediction_trace,
        "teacher_trace": teacher_trace,
    }


def prediction_trajectory_repeat_deterministic(
    first: dict[str, Any], second: dict[str, Any]
) -> bool:
    """Compare complete ordered predicted-state traces, never endpoint summaries."""
    for payload in (first, second):
        trace = payload.get("prediction_trace")
        digest = payload.get("prediction_trace_sha256")
        if not isinstance(trace, list) or digest != sha256_bytes(
            canonical_json_bytes(trace)
        ):
            raise RuntimeError("exact-repeat prediction trajectory digest mismatch")
    return first["prediction_trace_sha256"] == second["prediction_trace_sha256"]


class ScreenGuard:
    """Independent wall/aggregate-CPU/RSS guards from fresh measured budgets."""

    def __init__(
        self, calibration: dict[str, Any], process: Any, *,
        started_wall: float | None = None, started_cpu: float | None = None,
        initial_peak_rss: int | None = None,
    ) -> None:
        self.process = process
        self.started_wall = time.perf_counter() if started_wall is None else float(started_wall)
        cpu = process.cpu_times()
        self.started_cpu = float(cpu.user + cpu.system) if started_cpu is None else float(started_cpu)
        self.reused_wall = 0.0
        self.reused_cpu = 0.0
        self.peak_rss = max(cal.process_peak_rss_bytes(process), int(initial_peak_rss or 0))
        budgets = calibration["resolved_screen_budgets"]
        self.projected_wall = float(budgets["wall_seconds"]["projected_seconds"])
        self.soft_wall = float(budgets["wall_seconds"]["soft_budget_seconds"])
        self.hard_wall = float(budgets["wall_seconds"]["hard_runaway_ceiling_seconds"])
        self.projected_cpu = float(budgets["aggregate_cpu_seconds"]["projected_seconds"])
        self.soft_cpu = float(budgets["aggregate_cpu_seconds"]["soft_budget_seconds"])
        self.hard_cpu = float(budgets["aggregate_cpu_seconds"]["hard_runaway_ceiling_seconds"])
        self.hard_rss = int(budgets["peak_rss_guard_bytes"])
        quarter = budgets["first_quarter_checkpoint"]
        self.expected_quarter_wall = float(quarter["wall_seconds"]["expected_elapsed_seconds"])
        self.expected_quarter_cpu = float(quarter["aggregate_cpu_seconds"]["expected_elapsed_seconds"])
        self.first_quarter: dict[str, Any] | None = None
        self.soft_crossings = {"wall": None, "aggregate_cpu": None}

    def timing(self) -> dict[str, Any]:
        cpu = self.process.cpu_times()
        active_wall = time.perf_counter() - self.started_wall
        active_cpu = float(cpu.user + cpu.system) - self.started_cpu
        self.peak_rss = max(self.peak_rss, cal.process_peak_rss_bytes(self.process))
        return {
            "active_wall_seconds": active_wall,
            "active_aggregate_cpu_seconds": active_cpu,
            "reused_wall_seconds_charged": self.reused_wall,
            "reused_aggregate_cpu_seconds_charged": self.reused_cpu,
            "charged_wall_seconds": active_wall + self.reused_wall,
            "charged_aggregate_cpu_seconds": active_cpu + self.reused_cpu,
            "peak_rss_bytes": self.peak_rss,
            "first_quarter": self.first_quarter,
            "budgets": {
                "wall_seconds": {"projected": self.projected_wall, "soft": self.soft_wall, "hard": self.hard_wall},
                "aggregate_cpu_seconds": {"projected": self.projected_cpu, "soft": self.soft_cpu, "hard": self.hard_cpu},
                "peak_rss_guard_bytes": self.hard_rss,
            },
            "soft_crossings": self.soft_crossings,
        }

    def check(self, stage: str) -> None:
        timing = self.timing()
        if timing["charged_wall_seconds"] > self.soft_wall and self.soft_crossings["wall"] is None:
            self.soft_crossings["wall"] = {"stage": stage, "seconds": timing["charged_wall_seconds"]}
        if timing["charged_aggregate_cpu_seconds"] > self.soft_cpu and self.soft_crossings["aggregate_cpu"] is None:
            self.soft_crossings["aggregate_cpu"] = {"stage": stage, "seconds": timing["charged_aggregate_cpu_seconds"]}
        if timing["charged_wall_seconds"] > self.hard_wall:
            raise TimeoutError(f"screen wall runaway ceiling exceeded during {stage}")
        if timing["charged_aggregate_cpu_seconds"] > self.hard_cpu:
            raise TimeoutError(f"screen aggregate CPU runaway ceiling exceeded during {stage}")
        if timing["peak_rss_bytes"] > self.hard_rss:
            raise MemoryError(f"screen RSS guard exceeded during {stage}")

    def add_reused(self, timing: dict[str, Any], stage: str) -> None:
        self.reused_wall += float(timing["wall_seconds"])
        self.reused_cpu += float(timing["aggregate_cpu_seconds"])
        self.peak_rss = max(self.peak_rss, int(timing.get("peak_rss_bytes", 0)))
        self.check(stage)

    def check_first_quarter(self, completed_arm_updates: int, total_arm_updates: int) -> dict[str, Any]:
        if completed_arm_updates * 4 != total_arm_updates:
            raise RuntimeError("screen first-quarter checkpoint is not exactly one quarter")
        timing = self.timing()
        record = {
            "completed_arm_updates": completed_arm_updates,
            "total_arm_updates": total_arm_updates,
            "charged_wall_seconds": timing["charged_wall_seconds"],
            "charged_aggregate_cpu_seconds": timing["charged_aggregate_cpu_seconds"],
            "projected_full_wall_seconds_from_checkpoint": timing["charged_wall_seconds"] * 4,
            "projected_full_aggregate_cpu_seconds_from_checkpoint": timing["charged_aggregate_cpu_seconds"] * 4,
            "measured_expected_quarter_wall_seconds": self.expected_quarter_wall,
            "measured_expected_quarter_aggregate_cpu_seconds": self.expected_quarter_cpu,
            "wall_within_20_percent_of_measured_projection": timing["charged_wall_seconds"] <= self.expected_quarter_wall * 1.20,
            "aggregate_cpu_within_20_percent_of_measured_projection": timing["charged_aggregate_cpu_seconds"] <= self.expected_quarter_cpu * 1.20,
        }
        self.first_quarter = record
        if not record["wall_within_20_percent_of_measured_projection"]:
            raise TimeoutError("screen first-quarter wall throughput requires re-budget")
        if not record["aggregate_cpu_within_20_percent_of_measured_projection"]:
            raise TimeoutError("screen first-quarter aggregate CPU throughput requires re-budget")
        return record


def _parsed_attempt_directories(root: Path, stem: str) -> list[tuple[int, Path]]:
    if not root.exists():
        return []
    matching = sorted(item for item in root.iterdir() if item.name.startswith(stem))
    parsed: list[tuple[int, Path]] = []
    pattern = re.compile(rf"^{re.escape(stem)}(?:-retry-(\d{{3,}}))?$")
    for item in matching:
        match = pattern.fullmatch(item.name)
        if match is None or not item.is_dir():
            raise RuntimeError("malformed or colliding attempt path")
        suffix = 0 if match.group(1) is None else int(match.group(1))
        parsed.append((suffix, item))
    if len({suffix for suffix, _ in parsed}) != len(parsed):
        raise RuntimeError("colliding parsed attempt suffix")
    return parsed


def allocate_exclusive_attempt_directory(root: Path, stem: str) -> tuple[str, Path, list[Path]]:
    root.mkdir(parents=True, exist_ok=True)
    parsed = _parsed_attempt_directories(root, stem)
    suffix = 0 if not parsed else max(value for value, _ in parsed) + 1
    attempt_id = stem if suffix == 0 else f"{stem}-retry-{suffix:03d}"
    directory = root / attempt_id
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError("exclusive attempt allocation collision") from error
    return attempt_id, directory, [item for _, item in parsed]


class ScreenAttemptLedger:
    def __init__(self, *, authorization_sha256: str, binding_sha256: str, guard: ScreenGuard, environment_sha256: str) -> None:
        stem = f"screen-{authorization_sha256[:16]}-{binding_sha256[:12]}"
        root = SCRATCH_SCREEN_DIR / "attempts"
        existing = [item for _, item in _parsed_attempt_directories(root, stem)]
        for directory in existing:
            state_path = directory / "attempt-state.json"
            if not state_path.is_file():
                raise RuntimeError("existing screen attempt lacks durable state")
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            if prior.get("authorization_sha256") != authorization_sha256 or prior.get("binding_sha256") != binding_sha256:
                raise RuntimeError("existing screen attempt provenance mismatch")
        self.attempt_id, self.directory, allocated_existing = allocate_exclusive_attempt_directory(
            root, stem
        )
        if [item.name for item in allocated_existing] != [item.name for item in existing]:
            raise RuntimeError("screen attempt set changed during exclusive allocation")
        self.guard = guard
        self.state = {
            "schema_version": "exp648-screen-attempt-state-v1",
            "attempt_id": self.attempt_id, "authorization_sha256": authorization_sha256,
            "binding_sha256": binding_sha256, "environment_sha256": environment_sha256,
            "phase": "authorized", "completed_updates": {"control": 0, "challenger": 0},
            "completed_evaluation_units": [], "cache_receipts": [],
            "verified_checkpoints": {}, "first_quarter": None,
            "gates_computed": False, "complete": False,
            "prior_attempts_preserved": [item.name for item in existing],
            "training_resume_supported": False,
            "budgets": {
                "projected_wall_seconds": guard.projected_wall, "hard_wall_seconds": guard.hard_wall,
                "projected_aggregate_cpu_seconds": guard.projected_cpu, "hard_aggregate_cpu_seconds": guard.hard_cpu,
                "hard_rss_bytes": guard.hard_rss,
            },
        }
        self.persist()

    def persist(self) -> None:
        self.state["final_timing"] = self.guard.timing()
        _atomic_write(self.directory / "attempt-state.json", canonical_json_bytes(self.state, pretty=True))

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.persist()


def create_screen_launch_ledger(
    authorization: dict[str, Any], path: Path, *, launch_wall_started: float,
    launch_cpu_started: float, launch_peak_rss: int,
) -> Path:
    auth_sha = sha256(path)
    binding_sha = authorization["screen_readiness_binding_sha256"]
    root = SCRATCH_SCREEN_DIR / "launch-attempts"
    stem = f"launch-{auth_sha[:16]}-{binding_sha[:12]}"
    attempt_id, directory, existing = allocate_exclusive_attempt_directory(root, stem)
    payload = {
        "schema_version": "exp648-screen-launch-state-v1", "attempt_id": attempt_id,
        "authorization_sha256": auth_sha, "binding_sha256": binding_sha,
        "phase": "authorization-verified-constructor-pending", "complete": False,
        "module_wall_elapsed_seconds": time.perf_counter() - _MODULE_WALL_STARTED,
        "module_process_cpu_elapsed_seconds": time.process_time() - _MODULE_PROCESS_CPU_STARTED,
        "launch_wall_started_perf_counter": launch_wall_started,
        "launch_process_cpu_started_seconds": launch_cpu_started,
        "launch_initial_peak_rss_bytes": launch_peak_rss,
        "prior_launch_attempts_preserved": [item.name for item in existing],
    }
    _atomic_write(directory / "launch-state.json", canonical_json_bytes(payload, pretty=True))
    return directory / "launch-state.json"


def transition_screen_launch_ledger(
    path: Path, *, phase: str, complete: bool,
    scientific_attempt_id: str | None = None,
    completion_pointer_path: Path | None = None,
) -> dict[str, Any]:
    raw = path.read_bytes()
    state = json.loads(raw)
    if raw != canonical_json_bytes(state, pretty=True):
        raise RuntimeError("screen launch ledger is noncanonical")
    process = psutil.Process()
    cpu = process.cpu_times()
    timing = {
        "wall_seconds": time.perf_counter() - float(state["launch_wall_started_perf_counter"]),
        "aggregate_cpu_seconds": float(cpu.user + cpu.system) - float(state["launch_process_cpu_started_seconds"]),
        "peak_rss_bytes": max(
            int(state["launch_initial_peak_rss_bytes"]),
            cal.process_peak_rss_bytes(process),
        ),
        "sampled_fresh_at_transition": True,
    }
    updated = {
        **state,
        "phase": phase,
        "complete": bool(complete),
        "scientific_attempt_id": scientific_attempt_id,
        "latest_transition_timing": timing,
    }
    if completion_pointer_path is not None:
        pointer = json.loads(completion_pointer_path.read_bytes())
        seal = completion_pointer_path.parent / "attempts" / pointer["attempt_id"] / "completion-seal.json"
        updated["completion_pointer_sha256"] = sha256(completion_pointer_path)
        updated["completion_seal_sha256"] = sha256(seal)
        updated["scientific_attempt_state_sha256"] = sha256(
            seal.parent / "attempt-state.json"
        )
    _atomic_write(path, canonical_json_bytes(updated, pretty=True))
    return updated


def execute_authorized_screen_launch(
    *, authorization: dict[str, Any], authorization_path: Path,
    launch_state_path: Path, root: int, launch_wall_started: float,
    launch_cpu_started: float, launch_peak_rss: int,
    experiment_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    experiment = None
    factory = experiment_factory or ScreenExperiment
    try:
        reauthorize_screen_root(
            "screen-training", root, authorization, authorization_path
        )
        experiment = factory(
            authorization_path,
            preverified_authorization=authorization,
            launch_wall_started=launch_wall_started,
            launch_cpu_started=launch_cpu_started,
            launch_peak_rss=launch_peak_rss,
        )
        launch_raw = launch_state_path.read_bytes()
        launch_state = json.loads(launch_raw)
        if launch_raw != canonical_json_bytes(launch_state, pretty=True):
            raise RuntimeError("screen launch ledger became noncanonical after construction")
        launch_attempt_id = launch_state.get("attempt_id")
        if not isinstance(launch_attempt_id, str) or not launch_attempt_id:
            raise RuntimeError("screen launch ledger lacks launch attempt ID")
        experiment.ledger.update(launch_attempt_id=launch_attempt_id)
        transition_screen_launch_ledger(
            launch_state_path, phase="constructor-complete", complete=False,
            scientific_attempt_id=experiment.ledger.attempt_id,
        )
        result = experiment.run()
        pointer = TRACKED_SCREEN_DIR / "completion.json"
        transition_screen_launch_ledger(
            launch_state_path, phase="complete-sealed", complete=True,
            scientific_attempt_id=experiment.ledger.attempt_id,
            completion_pointer_path=pointer,
        )
        verify_screen_completion_seal(pointer, launch_state_path=launch_state_path)
        return result
    except BaseException as error:
        preserve_screen_failure(
            "authorized-screen", error, experiment, launch_state_path
        )
        raise


def write_screen_file(relative: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    encoded = canonical_json_bytes(payload, pretty=True)
    scratch = SCRATCH_SCREEN_DIR / "evidence" / relative
    tracked = TRACKED_SCREEN_DIR / relative
    _atomic_write(scratch, encoded)
    _atomic_write(tracked, encoded)
    if sha256(scratch) != sha256(tracked):
        raise RuntimeError(f"screen atomic evidence parity failed for {relative}")
    return scratch, tracked


def atomic_save_checkpoint(model: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError("screen checkpoint path is immutable once created")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(model.state_dict(), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    file_hash = sha256(path)
    loaded = load_file(path)
    loaded_state_hash = cal.state_dict_sha256(loaded)
    live_state_hash = cal.state_dict_sha256(model)
    if loaded_state_hash != live_state_hash or sha256(path) != file_hash:
        raise RuntimeError("atomically saved screen checkpoint failed reload/hash verification")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": file_hash,
        "state_dict_sha256": live_state_hash,
        "reload_state_dict_sha256": loaded_state_hash,
        "saved_and_verified_before_evaluation": True,
    }


def prepare_checkpoints_for_evaluation(
    models: dict[str, Any], checkpoint_dir: Path,
    event_hook: Callable[[str], None] | None = None,
    *, attempt_id: str | None = None,
    receipt_hook: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if attempt_id is not None and checkpoint_dir.name != "checkpoints":
        raise RuntimeError("screen checkpoint directory must be attempt-local")
    if attempt_id is not None and checkpoint_dir.parent.name != attempt_id:
        raise RuntimeError("screen checkpoint directory does not bind the attempt ID")
    receipts = {}
    for arm in ("control", "challenger"):
        receipts[arm] = atomic_save_checkpoint(models[arm], checkpoint_dir / f"{arm}.safetensors")
        receipts[arm]["attempt_id"] = attempt_id
        if receipt_hook is not None:
            receipt_hook(arm, copy.deepcopy(receipts[arm]))
        if event_hook is not None:
            event_hook(f"checkpoint-verified:{arm}")
    if set(receipts) != {"control", "challenger"} or not all(
        value.get("saved_and_verified_before_evaluation") is True for value in receipts.values()
    ):
        raise RuntimeError("both screen checkpoints must verify before evaluation")
    return receipts


def evaluation_cache_key_fields(
    *, binding: dict[str, Any], checkpoint_hashes: dict[str, Any], environment_sha256: str
) -> dict[str, Any]:
    return {
        **binding,
        "checkpoint_sha256": {arm: value["sha256"] for arm, value in checkpoint_hashes.items()},
        "environment_sha256": environment_sha256,
        "evaluator_sha256": scientific_evaluator_sha256(),
        "evaluation_root": SCREEN_EVALUATION_ROOT,
    }


def validate_evaluation_results(
    *, unit: str, width: int, cases: dict[str, Any], results: dict[str, Any]
) -> None:
    if set(results) != {"control", "challenger"}:
        raise RuntimeError(f"screen cached arm schema mismatch for {unit}")
    mode = "transition" if unit.startswith("transition-") else "rollout"
    expected_count = len(cases["state"] if mode == "transition" else cases["multiplier"])
    for arm, result in results.items():
        expected_keys = (
            {"cases", "exact_accuracy", "quotient_accuracy", "valid_residue_rate", "exact_vector", "output_sha256", "predictions", "quotient_predictions"}
            if mode == "transition" else
            {"cases", "exact_accuracy", "exact_vector", "mean_clipped_first_error_index", "invalid_rollout_rate", "output_sha256", "prediction_trace_sha256", "predictions", "teacher_targets", "first_error_indices", "any_intermediate_invalid_vector", "prediction_trace", "teacher_trace"}
        )
        if set(result) != expected_keys:
            raise RuntimeError(f"screen cached result key schema mismatch for {unit}/{arm}")
        predictions = result.get("predictions")
        exact = result.get("exact_vector")
        if not isinstance(predictions, list) or not isinstance(exact, list) or len(predictions) != expected_count or len(exact) != expected_count:
            raise RuntimeError(f"screen cached result length mismatch for {unit}/{arm}")
        if any(type(value) is not int or value < 0 for value in predictions) or any(type(value) is not bool for value in exact):
            raise RuntimeError(f"screen cached result type/range mismatch for {unit}/{arm}")
        if result.get("cases") != expected_count or result.get("output_sha256") != sha256_bytes(canonical_json_bytes(predictions)):
            raise RuntimeError(f"screen cached result count/output hash mismatch for {unit}/{arm}")
        accuracy = sum(exact) / expected_count
        if not math.isfinite(float(result.get("exact_accuracy", math.nan))) or not math.isclose(float(result["exact_accuracy"]), accuracy, abs_tol=1e-15):
            raise RuntimeError(f"screen cached exact accuracy mismatch for {unit}/{arm}")
        if mode == "transition":
            recomputed = [left == right for left, right in zip(predictions, cases["target"], strict=True)]
            quotient = result.get("quotient_predictions")
            if recomputed != exact or not isinstance(quotient, list) or len(quotient) != expected_count:
                raise RuntimeError(f"screen cached transition semantics mismatch for {unit}/{arm}")
            qacc = sum(int(left) == int(right) for left, right in zip(quotient, cases["quotient"], strict=True)) / expected_count
            valid = sum(pred < modulus for pred, modulus in zip(predictions, cases["modulus"], strict=True)) / expected_count
            if not math.isclose(float(result["quotient_accuracy"]), qacc, abs_tol=1e-15) or not math.isclose(float(result["valid_residue_rate"]), valid, abs_tol=1e-15):
                raise RuntimeError(f"screen cached transition rates mismatch for {unit}/{arm}")
        else:
            teacher = result.get("teacher_targets")
            first = result.get("first_error_indices")
            if not isinstance(teacher, list) or not isinstance(first, list) or len(teacher) != expected_count or len(first) != expected_count:
                raise RuntimeError(f"screen cached rollout semantic arrays mismatch for {unit}/{arm}")
            if [a == b for a, b in zip(predictions, teacher, strict=True)] != exact:
                raise RuntimeError(f"screen cached rollout exact vector mismatch for {unit}/{arm}")
            prediction_trace = result.get("prediction_trace")
            teacher_trace = result.get("teacher_trace")
            invalid_vector = result.get("any_intermediate_invalid_vector")
            if not isinstance(prediction_trace, list) or not isinstance(teacher_trace, list) or len(prediction_trace) != width + 1 or len(teacher_trace) != width + 1:
                raise RuntimeError(f"screen cached rollout trace length mismatch for {unit}/{arm}")
            for trace_name, trace in (
                ("prediction", prediction_trace), ("teacher", teacher_trace)
            ):
                if any(
                    not isinstance(row, list)
                    or len(row) != expected_count
                    or any(type(value) is not int or value < 0 for value in row)
                    for row in trace
                ):
                    raise RuntimeError(
                        f"screen cached rollout {trace_name} trace row mismatch for {unit}/{arm}"
                    )
            if prediction_trace[0] != [0] * expected_count:
                raise RuntimeError(f"screen cached rollout initial prediction mismatch for {unit}/{arm}")
            if prediction_trace[-1] != predictions or teacher_trace[-1] != teacher:
                raise RuntimeError(f"screen cached rollout endpoint trace mismatch for {unit}/{arm}")
            if result.get("prediction_trace_sha256") != sha256_bytes(
                canonical_json_bytes(prediction_trace)
            ):
                raise RuntimeError(
                    f"screen cached rollout prediction trace hash mismatch for {unit}/{arm}"
                )
            recomputed_teacher = [[0] * expected_count]
            recomputed_first: list[int | None] = [None] * expected_count
            recomputed_invalid = [False] * expected_count
            for position in range(width):
                bits = [(value >> (width - 1 - position)) & 1 for value in cases["multiplier"]]
                next_teacher = [
                    cal.exact_teacher_step(previous, multiplicand, modulus, bit)[0]
                    for previous, multiplicand, modulus, bit in zip(
                        recomputed_teacher[-1], cases["multiplicand"], cases["modulus"], bits, strict=True
                    )
                ]
                recomputed_teacher.append(next_teacher)
                for index, (prediction, target, modulus) in enumerate(zip(prediction_trace[position + 1], next_teacher, cases["modulus"], strict=True)):
                    if recomputed_first[index] is None and prediction != target:
                        recomputed_first[index] = position
                    recomputed_invalid[index] = recomputed_invalid[index] or prediction >= modulus
            recomputed_clipped = [width + 1 if value is None else value for value in recomputed_first]
            if teacher_trace != recomputed_teacher or first != recomputed_clipped or invalid_vector != recomputed_invalid:
                raise RuntimeError(f"screen cached rollout trace semantics mismatch for {unit}/{arm}")
            mean_first = sum(recomputed_clipped) / expected_count
            invalid = sum(recomputed_invalid) / expected_count
            if any(type(value) is not int or value < 0 or value > width + 1 for value in first):
                raise RuntimeError(f"screen cached first-error range mismatch for {unit}/{arm}")
            if not math.isclose(float(result["mean_clipped_first_error_index"]), mean_first, abs_tol=1e-15) or not math.isclose(float(result["invalid_rollout_rate"]), invalid, abs_tol=1e-15):
                raise RuntimeError(f"screen cached rollout rates mismatch for {unit}/{arm}")


def compact_evaluation_results(value: Any) -> Any:
    bulky = {
        "predictions", "quotient_predictions", "teacher_targets", "first_error_indices",
        "any_intermediate_invalid_vector", "prediction_trace", "teacher_trace",
    }
    if isinstance(value, dict):
        return {key: compact_evaluation_results(item) for key, item in value.items() if key not in bulky}
    if isinstance(value, list):
        return [compact_evaluation_results(item) for item in value]
    return value


MATCHED_WORK_CATEGORIES = (
    "clean_forward_steps",
    "autonomous_forward_steps",
    "endpoint_loss_computations",
    "clean_backward_calls",
    "endpoint_backward_calls",
    "gradient_clip_calls",
    "optimizer_steps",
    "scheduler_steps",
    "checkpoint_writes",
    "checkpoint_reload_verifications",
    "checkpoint_byte_hashes",
    "checkpoint_receipt_serializations",
    "transition_evaluation_cases",
    "rollout_evaluation_cases",
    "rollout_evaluation_steps",
    "evaluation_result_hashes",
    "evaluation_result_serializations",
)

MATCHED_ACTUAL_OPERATION_CATEGORIES = (
    "training_update_calls",
    "scheduler_step_calls",
    "checkpoint_save_reload_hash_calls",
    "checkpoint_receipt_persist_calls",
    "fresh_evaluator_calls",
    "fresh_result_hashes",
    "fresh_result_serializations",
    "cache_record_validations",
    "cache_per_arm_receipt_validations",
    "cached_result_digest_validations",
    "cache_reuse_operations",
)


def create_matched_work_accounting() -> dict[str, Any]:
    """Create independent per-arm work ledgers; data-order hashes never enter them."""
    return {
        "schema_version": "exp648-matched-work-accounting-v1",
        "categories": list(MATCHED_WORK_CATEGORIES),
        "actual_operation_categories": list(MATCHED_ACTUAL_OPERATION_CATEGORIES),
        "arms": {
            arm: {
                "counts": {category: 0 for category in MATCHED_WORK_CATEGORIES},
                "ordered_events": [],
                "actual_counts": {
                    category: 0 for category in MATCHED_ACTUAL_OPERATION_CATEGORIES
                },
                "ordered_actual_operations": [],
            }
            for arm in ("control", "challenger")
        },
    }


def record_matched_work(
    accounting: dict[str, Any], arm: str, category: str, count: int = 1,
    *, unit: str | None = None,
) -> None:
    if arm not in ("control", "challenger") or category not in MATCHED_WORK_CATEGORIES:
        raise RuntimeError("matched-work event schema mismatch")
    if type(count) is not int or count < 0:
        raise RuntimeError("matched-work count must be a nonnegative integer")
    event = {"category": category, "count": count, "unit": unit}
    accounting["arms"][arm]["counts"][category] += count
    accounting["arms"][arm]["ordered_events"].append(event)


def record_actual_matched_work(
    accounting: dict[str, Any], arm: str, operation: str, count: int = 1,
    *, unit: str | None = None, evidence_sha256: str | None = None,
) -> None:
    if (
        arm not in ("control", "challenger")
        or operation not in MATCHED_ACTUAL_OPERATION_CATEGORIES
        or type(count) is not int
        or count < 0
    ):
        raise RuntimeError("matched-work actual operation schema mismatch")
    if evidence_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256):
        raise RuntimeError("matched-work actual evidence hash mismatch")
    event = {
        "operation": operation,
        "count": count,
        "unit": unit,
        "evidence_sha256": evidence_sha256,
    }
    record = accounting["arms"][arm]
    record["actual_counts"][operation] += count
    record["ordered_actual_operations"].append(event)


def finalize_matched_work(accounting: dict[str, Any]) -> dict[str, Any]:
    if (
        accounting.get("schema_version") != "exp648-matched-work-accounting-v1"
        or accounting.get("categories") != list(MATCHED_WORK_CATEGORIES)
        or accounting.get("actual_operation_categories")
        != list(MATCHED_ACTUAL_OPERATION_CATEGORIES)
        or set(accounting.get("arms", {})) != {"control", "challenger"}
    ):
        raise RuntimeError("matched-work accounting schema mismatch")
    per_arm = {}
    for arm in ("control", "challenger"):
        record = accounting["arms"][arm]
        if set(record) != {
            "counts", "ordered_events", "actual_counts",
            "ordered_actual_operations",
        }:
            raise RuntimeError("matched-work arm schema mismatch")
        if set(record["counts"]) != set(MATCHED_WORK_CATEGORIES):
            raise RuntimeError("matched-work category set mismatch")
        recomputed = {category: 0 for category in MATCHED_WORK_CATEGORIES}
        for event in record["ordered_events"]:
            if (
                set(event) != {"category", "count", "unit"}
                or event["category"] not in recomputed
                or type(event["count"]) is not int
                or event["count"] < 0
            ):
                raise RuntimeError("matched-work ordered event mismatch")
            recomputed[event["category"]] += event["count"]
        if recomputed != record["counts"]:
            raise RuntimeError("matched-work counts do not recompute")
        recomputed_actual = {
            category: 0 for category in MATCHED_ACTUAL_OPERATION_CATEGORIES
        }
        operation_projection = []
        bound_actual_events = []
        for event in record["ordered_actual_operations"]:
            if (
                set(event)
                != {"operation", "count", "unit", "evidence_sha256"}
                or event["operation"] not in recomputed_actual
                or type(event["count"]) is not int
                or event["count"] < 0
                or (
                    event["evidence_sha256"] is not None
                    and not re.fullmatch(r"[0-9a-f]{64}", event["evidence_sha256"])
                )
            ):
                raise RuntimeError("matched-work actual event mismatch")
            recomputed_actual[event["operation"]] += event["count"]
            operation_projection.append({
                key: event[key] for key in ("operation", "count", "unit")
            })
            bound_actual_events.append(event)
        if recomputed_actual != record["actual_counts"]:
            raise RuntimeError("matched-work actual counts do not recompute")
        training_calls = recomputed_actual["training_update_calls"]
        checkpoint_calls = recomputed_actual[
            "checkpoint_save_reload_hash_calls"
        ]
        evaluation_paths = (
            recomputed_actual["fresh_evaluator_calls"]
            + recomputed_actual["cache_reuse_operations"]
        )
        declared_to_actual_valid = bool(
            recomputed["clean_forward_steps"]
            == recomputed["autonomous_forward_steps"]
            == recomputed["clean_backward_calls"]
            == WINDOW_LENGTH * training_calls
            and recomputed["endpoint_loss_computations"]
            == recomputed["endpoint_backward_calls"]
            == recomputed["gradient_clip_calls"]
            == recomputed["optimizer_steps"]
            == training_calls
            and recomputed["scheduler_steps"]
            == recomputed_actual["scheduler_step_calls"]
            and recomputed["checkpoint_writes"]
            == recomputed["checkpoint_reload_verifications"]
            == recomputed["checkpoint_byte_hashes"]
            == checkpoint_calls
            and recomputed["checkpoint_receipt_serializations"]
            == recomputed_actual["checkpoint_receipt_persist_calls"]
            and recomputed["evaluation_result_hashes"]
            == recomputed["evaluation_result_serializations"]
            == evaluation_paths
            and recomputed_actual["fresh_result_hashes"]
            == recomputed_actual["fresh_result_serializations"]
            == recomputed_actual["fresh_evaluator_calls"]
            and recomputed_actual["cache_record_validations"]
            == recomputed_actual["cache_per_arm_receipt_validations"]
            == recomputed_actual["cached_result_digest_validations"]
            == recomputed_actual["cache_reuse_operations"]
        )
        per_arm[arm] = {
            "counts": recomputed,
            "ordered_event_count": len(record["ordered_events"]),
            "ordered_work_sha256": sha256_bytes(
                canonical_json_bytes(record["ordered_events"])
            ),
            "total_declared_work_units": sum(recomputed.values()),
            "actual_operation_counts": recomputed_actual,
            "actual_operation_event_count": len(operation_projection),
            "actual_operation_schedule_sha256": sha256_bytes(
                canonical_json_bytes(operation_projection)
            ),
            "actual_operation_evidence_sha256": sha256_bytes(
                canonical_json_bytes(bound_actual_events)
            ),
            "declared_to_actual_cross_binding_valid": declared_to_actual_valid,
        }
    required_positive = set(MATCHED_WORK_CATEGORIES)
    complete = all(
        per_arm[arm]["counts"][category] > 0
        for arm in ("control", "challenger")
        for category in required_positive
    )
    effective_equal = all(
        per_arm["control"][key] == per_arm["challenger"][key]
        for key in (
            "counts", "ordered_event_count", "ordered_work_sha256",
            "total_declared_work_units",
        )
    )
    actual_equal = all(
        per_arm["control"][key] == per_arm["challenger"][key]
        for key in (
            "actual_operation_counts", "actual_operation_event_count",
            "actual_operation_schedule_sha256",
        )
    )
    actual_paths_valid = True
    for arm in ("control", "challenger"):
        counts = per_arm[arm]["actual_operation_counts"]
        actual_paths_valid = actual_paths_valid and all(
            counts[key] > 0
            for key in (
                "training_update_calls", "scheduler_step_calls",
                "checkpoint_save_reload_hash_calls",
                "checkpoint_receipt_persist_calls",
            )
        )
        fresh = counts["fresh_evaluator_calls"]
        cached = counts["cache_reuse_operations"]
        actual_paths_valid = actual_paths_valid and fresh + cached > 0
        if fresh:
            actual_paths_valid = actual_paths_valid and (
                counts["fresh_result_hashes"] == fresh
                and counts["fresh_result_serializations"] == fresh
            )
        if cached:
            actual_paths_valid = actual_paths_valid and (
                counts["cache_record_validations"] == cached
                and counts["cache_per_arm_receipt_validations"] == cached
                and counts["cached_result_digest_validations"] == cached
            )
    declared_to_actual_cross_binding_valid = all(
        per_arm[arm]["declared_to_actual_cross_binding_valid"]
        for arm in ("control", "challenger")
    )
    return {
        "schema_version": "exp648-matched-work-receipt-v1",
        "independent_from_paired_data_order": True,
        "per_arm": per_arm,
        "all_declared_categories_observed": complete,
        "per_arm_effective_counts_and_ordered_digests_equal": effective_equal,
        "per_arm_actual_operation_counts_and_schedules_equal": actual_equal,
        "truthful_fresh_or_valid_cache_paths": actual_paths_valid,
        "declared_to_actual_cross_binding_valid": declared_to_actual_cross_binding_valid,
        "valid": bool(
            complete and effective_equal and actual_equal
            and actual_paths_valid and declared_to_actual_cross_binding_valid
        ),
    }


def evaluation_cache_work_receipts(
    *, unit: str, width: int, cases_sha256: str,
    expected_count: int, results: dict[str, Any],
) -> dict[str, Any]:
    arms = {}
    for arm in ("control", "challenger"):
        result_sha = sha256_bytes(canonical_json_bytes(results[arm]))
        core = {
            "schema_version": "exp648-evaluation-cache-arm-work-v1",
            "arm": arm,
            "unit": unit,
            "width": width,
            "cases": expected_count,
            "cases_sha256": cases_sha256,
            "execution_path": "fresh-explicit-evaluator-call",
            "result_canonical_sha256": result_sha,
            "output_sha256": results[arm]["output_sha256"],
        }
        arms[arm] = {
            **core,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(core)),
        }
    return {
        "schema_version": "exp648-evaluation-cache-work-receipts-v1",
        "arms": arms,
    }


def validate_evaluation_cache_work_receipts(
    *, receipts: dict[str, Any], unit: str, width: int,
    cases_sha256: str, expected_count: int, results: dict[str, Any],
) -> None:
    if (
        not isinstance(receipts, dict)
        or set(receipts) != {"schema_version", "arms"}
        or receipts["schema_version"]
        != "exp648-evaluation-cache-work-receipts-v1"
        or set(receipts.get("arms", {})) != {"control", "challenger"}
    ):
        raise RuntimeError(f"screen cached matched-work receipt schema mismatch for {unit}")
    for arm in ("control", "challenger"):
        receipt = receipts["arms"][arm]
        core = {
            "schema_version": "exp648-evaluation-cache-arm-work-v1",
            "arm": arm,
            "unit": unit,
            "width": width,
            "cases": expected_count,
            "cases_sha256": cases_sha256,
            "execution_path": "fresh-explicit-evaluator-call",
            "result_canonical_sha256": sha256_bytes(
                canonical_json_bytes(results[arm])
            ),
            "output_sha256": results[arm]["output_sha256"],
        }
        if receipt != {
            **core, "receipt_sha256": sha256_bytes(canonical_json_bytes(core))
        }:
            raise RuntimeError(
                f"screen cached matched-work receipt/result mismatch for {unit}/{arm}"
            )


def load_or_evaluate_unit(
    *, unit: str, cases: dict[str, Any], models: dict[str, Any], evaluator: Callable[..., dict[str, Any]],
    width: int, exp300: Any, key_fields: dict[str, Any], guard: ScreenGuard,
    matched_work: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    cases_sha = sha256_bytes(canonical_json_bytes(cases))
    key = sha256_bytes(canonical_json_bytes({"unit": unit, "cases_sha256": cases_sha, "key_fields": key_fields}))
    path = SCRATCH_SCREEN_DIR / "evaluation-cache" / f"{unit}-{key}.json"
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        core = copy.deepcopy(record)
        observed_core = core.pop("core_sha256", None)
        if (
            record.get("schema_version") != "exp648-screen-evaluation-cache-v1"
            or record.get("complete") is not True
            or record.get("unit") != unit
            or record.get("cases_sha256") != cases_sha
            or record.get("key_fields") != key_fields
            or observed_core != sha256_bytes(canonical_json_bytes(core))
        ):
            raise RuntimeError(f"screen evaluation recovery provenance mismatch for {unit}")
        expected_count = len(
            cases["state"]
            if unit.startswith("transition-") else cases["multiplier"]
        )
        timing = record.get("timing", {})
        if set(timing) != {"wall_seconds", "aggregate_cpu_seconds", "peak_rss_bytes"} or any(
            not isinstance(timing[key], (int, float)) or not math.isfinite(float(timing[key])) or float(timing[key]) < 0
            for key in timing
        ):
            raise RuntimeError(f"screen evaluation cache timing mismatch for {unit}")
        validate_evaluation_results(unit=unit, width=width, cases=cases, results=record["results"])
        validate_evaluation_cache_work_receipts(
            receipts=record.get("matched_work_receipts"), unit=unit,
            width=width, cases_sha256=cases_sha, expected_count=expected_count,
            results=record["results"],
        )
        guard.add_reused(timing, f"evaluation-cache:{unit}")
        if matched_work is not None:
            record_evaluation_matched_work(
                matched_work, unit, width, expected_count=expected_count
            )
            for arm in ("control", "challenger"):
                receipt = record["matched_work_receipts"]["arms"][arm]
                for operation in (
                    "cache_record_validations",
                    "cache_per_arm_receipt_validations",
                    "cached_result_digest_validations",
                    "cache_reuse_operations",
                ):
                    record_actual_matched_work(
                        matched_work, arm, operation, 1, unit=unit,
                        evidence_sha256=receipt["receipt_sha256"],
                    )
        return record["results"], True
    expected_count = len(
        cases["state"] if unit.startswith("transition-") else cases["multiplier"]
    )
    measurement = cal.Measurement(guard.process)
    results = {}
    for arm in ("control", "challenger"):
        results[arm] = evaluator(models[arm], width, cases, exp300)
        if matched_work is not None:
            record_actual_matched_work(
                matched_work, arm, "fresh_evaluator_calls", 1, unit=unit
            )
    validate_evaluation_results(unit=unit, width=width, cases=cases, results=results)
    timing = measurement.stop().payload()
    work_receipts = evaluation_cache_work_receipts(
        unit=unit, width=width, cases_sha256=cases_sha,
        expected_count=expected_count, results=results,
    )
    record = {
        "schema_version": "exp648-screen-evaluation-cache-v1", "complete": True,
        "unit": unit, "cases_sha256": cases_sha, "key_fields": key_fields,
        "results": results,
        "matched_work_receipts": work_receipts,
        "timing": {"wall_seconds": timing["wall_seconds"], "aggregate_cpu_seconds": timing["aggregate_cpu_seconds"], "peak_rss_bytes": timing["peak_rss_bytes"]},
    }
    record["core_sha256"] = sha256_bytes(canonical_json_bytes(record))
    _atomic_write(path, canonical_json_bytes(record, pretty=True))
    if matched_work is not None:
        record_evaluation_matched_work(
            matched_work, unit, width, expected_count=expected_count
        )
        for arm in ("control", "challenger"):
            receipt_sha = work_receipts["arms"][arm]["receipt_sha256"]
            record_actual_matched_work(
                matched_work, arm, "fresh_result_hashes", 1, unit=unit,
                evidence_sha256=receipt_sha,
            )
            record_actual_matched_work(
                matched_work, arm, "fresh_result_serializations", 1, unit=unit,
                evidence_sha256=receipt_sha,
            )
    guard.check(f"evaluation:{unit}")
    return results, False


def record_evaluation_matched_work(
    accounting: dict[str, Any], unit: str, width: int, *, expected_count: int
) -> None:
    category = (
        "transition_evaluation_cases"
        if unit.startswith("transition-")
        else "rollout_evaluation_cases"
    )
    for arm in ("control", "challenger"):
        record_matched_work(accounting, arm, category, expected_count, unit=unit)
        if category == "rollout_evaluation_cases":
            record_matched_work(
                accounting, arm, "rollout_evaluation_steps",
                expected_count * width, unit=unit,
            )
        record_matched_work(accounting, arm, "evaluation_result_hashes", 1, unit=unit)
        record_matched_work(
            accounting, arm, "evaluation_result_serializations", 1, unit=unit
        )


def replay_frozen_corpus_from_source(
    *, model: Any, exp300: Any, guard: ScreenGuard
) -> dict[str, Any]:
    artifacts = _read_data_artifacts()
    identities = artifacts["identity"]["identities"]
    expected = artifacts["trajectory"]["trajectories"]
    observed: list[dict[str, Any]] = []
    for start in range(0, len(identities), IDENTITY_BATCH_SIZE):
        batch = [
            cal.Identity(
                int(row["identity_index"]), int(row["multiplier"]),
                int(row["multiplicand"]), int(row["modulus"]),
            )
            for row in identities[start : start + IDENTITY_BATCH_SIZE]
        ]
        mined = cal.rollout_identity_batch(
            model, exp300, batch, lambda stage: guard.check(f"semantic-replay:{stage}")
        )
        observed.extend(trajectory_record_payload(row) for row in mined)
        guard.check(f"semantic-replay:batch:{start}")
    if canonical_json(observed) != canonical_json(expected):
        raise RuntimeError("frozen corpus source-model semantic replay mismatch")
    return {
        "identity_trajectories": len(observed),
        "trajectory_manifest_sha256": FROZEN_CORPUS_FILES["data/trajectory-manifest.json.gz"],
        "observed_canonical_sha256": sha256_bytes(canonical_json_bytes(observed)),
        "exact_complete_source_replay": True,
    }


class ScreenExperiment:
    """Complete screen path; construction itself requires exact Judge bytes."""

    def __init__(
        self, authorization_path: Path = SCREEN_AUTHORIZATION_PATH, *,
        preverified_authorization: dict[str, Any] | None = None,
        launch_wall_started: float | None = None,
        launch_cpu_started: float | None = None,
        launch_peak_rss: int | None = None,
    ) -> None:
        # This must remain the first operation: no config/data/model/root work may
        # happen before exact authorization has passed.
        if preverified_authorization is None:
            self.authorization = verify_screen_authorization(authorization_path)
        else:
            expected_bytes = canonical_json_bytes(preverified_authorization, pretty=True)
            if authorization_path.read_bytes() != expected_bytes:
                raise PermissionError("preverified screen authorization bytes changed")
            self.authorization = preverified_authorization
        self.authorization_path = authorization_path
        self.training_root = reauthorize_screen_root(
            "screen-training", SCREEN_TRAINING_ROOT, self.authorization, authorization_path
        )
        self.evaluation_root = reauthorize_screen_root(
            "screen-evaluation", SCREEN_EVALUATION_ROOT, self.authorization, authorization_path
        )
        self.binding = self.authorization["screen_readiness_binding"]
        self.calibration = verify_screen_readiness_calibration()
        self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cal.validate_frozen_config(self.config)
        if sha256(SOURCE_CHECKPOINT_PATH) != PINNED["source_checkpoint_sha256"]:
            raise RuntimeError("screen source checkpoint hash mismatch")
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
        self.process = psutil.Process()
        self.guard = ScreenGuard(
            self.calibration, self.process, started_wall=launch_wall_started,
            started_cpu=launch_cpu_started, initial_peak_rss=launch_peak_rss,
        )
        self.environment = cal.environment_payload(self.config)
        self.environment_sha256 = sha256_bytes(canonical_json_bytes(self.environment))
        self.ledger = ScreenAttemptLedger(
            authorization_sha256=sha256(self.authorization_path),
            binding_sha256=self.authorization["screen_readiness_binding_sha256"],
            guard=self.guard, environment_sha256=self.environment_sha256,
        )

    def new_model(self):
        reauthorize_screen_root(
            "screen-training", self.training_root, self.authorization, self.authorization_path
        )
        model = self.cellular.SharedDilatedConvGRU(width=256, hidden=92, cycles=2)
        model.load_state_dict(load_file(SOURCE_CHECKPOINT_PATH))
        return model

    def run(self) -> dict[str, Any]:
        corpus_verification = verify_corpus_artifacts(require_receipts=True)
        self.ledger.update(phase="corpus-verified")
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
        semantic_replay = replay_frozen_corpus_from_source(
            model=self.new_model(), exp300=self.exp300, guard=self.guard
        )
        self.ledger.update(phase="training", frozen_corpus_semantic_replay=semantic_replay)
        histories = {"control": [], "challenger": []}
        batch_hashes = {"control": [], "challenger": []}
        matched_work = create_matched_work_accounting()
        updates = int(self.config["screen"]["updates_per_arm"])
        order = frozen_root_pair_order(
            len(pairs), corpus_verification["corpus_manifest_sha256"]
        )
        schedulers = {
            arm: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[arm],
                T_max=updates,
                eta_min=float(self.config["training"]["minimum_learning_rate"]),
            )
            for arm in ("control", "challenger")
        }
        for update in range(updates):
            windows = ordered_batch_windows(pairs, update, order=order)
            batch_hash = sha256_bytes(canonical_json_bytes(windows))
            for arm in ("control", "challenger"):
                reauthorize_screen_root(
                    "screen-training", self.training_root, self.authorization, self.authorization_path
                )
                batch_hashes[arm].append(batch_hash)
                histories[arm].append(
                    screen_training_update(
                        models[arm],
                        optimizers[arm],
                        windows,
                        float(self.config["matched_arms"][arm]["endpoint_loss_weight"]),
                        self.config,
                        self.exp300,
                        matched_work=matched_work,
                        arm=arm,
                    )
                )
                schedulers[arm].step()
                record_matched_work(matched_work, arm, "scheduler_steps", 1)
                record_actual_matched_work(
                    matched_work, arm, "scheduler_step_calls", 1
                )
                self.guard.check(f"training:{arm}:update:{update}")
            self.ledger.state["completed_updates"] = {
                "control": update + 1, "challenger": update + 1
            }
            self.ledger.persist()
            if update == updates // 4 - 1:
                telemetry = self.guard.check_first_quarter(
                    completed_arm_updates=2 * (update + 1), total_arm_updates=2 * updates
                )
                write_screen_file("first-quarter-telemetry.json", {
                    "schema_version": "exp648-screen-first-quarter-v1",
                    "authorization_sha256": sha256(self.authorization_path),
                    "telemetry": telemetry,
                })
                self.ledger.update(first_quarter=telemetry)
        if batch_hashes["control"] != batch_hashes["challenger"]:
            raise RuntimeError("screen arms did not receive identical batch order")

        # Both durable, hash-verified checkpoints exist before the first
        # evaluation input is generated or evaluator is invoked.
        checkpoint_dir = self.ledger.directory / "checkpoints"
        def persist_checkpoint_receipt(arm: str, receipt: dict[str, Any]) -> None:
            self.ledger.state["verified_checkpoints"][arm] = receipt
            self.ledger.persist()
            for category in (
                "checkpoint_writes", "checkpoint_reload_verifications",
                "checkpoint_byte_hashes", "checkpoint_receipt_serializations",
            ):
                record_matched_work(matched_work, arm, category, 1)
            record_actual_matched_work(
                matched_work, arm, "checkpoint_save_reload_hash_calls", 1,
                evidence_sha256=receipt["sha256"],
            )
            record_actual_matched_work(
                matched_work, arm, "checkpoint_receipt_persist_calls", 1,
                evidence_sha256=sha256_bytes(canonical_json_bytes(receipt)),
            )
        checkpoint_hashes = prepare_checkpoints_for_evaluation(
            models, checkpoint_dir, attempt_id=self.ledger.attempt_id,
            receipt_hook=persist_checkpoint_receipt,
        )
        write_screen_file("checkpoints.json", {
            "schema_version": "exp648-screen-checkpoints-v1",
            "authorization_sha256": sha256(self.authorization_path),
            "checkpoint_hashes": checkpoint_hashes,
            "evaluation_started": False,
        })
        self.ledger.update(phase="evaluation", verified_checkpoints=checkpoint_hashes)
        key_fields = evaluation_cache_key_fields(
            binding=self.binding, checkpoint_hashes=checkpoint_hashes,
            environment_sha256=self.environment_sha256,
        )
        evaluation = {"transitions": {}, "rollouts": {}}
        reused_units = []
        spec = self.config["screen"]["evaluation"]
        for width, count in spec["transition_cases"].items():
            cases = generate_screen_transition_cases(
                self.evaluation_root,
                int(width),
                int(count),
                self.authorization_path,
                self.authorization,
            )
            result, reused = load_or_evaluate_unit(
                unit=f"transition-{width}", cases=cases, models=models,
                evaluator=evaluate_transitions, width=int(width), exp300=self.exp300,
                key_fields=key_fields, guard=self.guard,
                matched_work=matched_work,
            )
            evaluation["transitions"][str(width)] = result
            reused_units.append({"unit": f"transition-{width}", "reused": reused})
            self.ledger.state["completed_evaluation_units"].append(f"transition-{width}")
            self.ledger.state["cache_receipts"].append(reused_units[-1])
            self.ledger.persist()
        for width, count in spec["full_rollout_cases"].items():
            cases = generate_screen_rollout_cases(
                self.evaluation_root,
                int(width),
                int(count),
                self.authorization_path,
                self.authorization,
            )
            result, reused = load_or_evaluate_unit(
                unit=f"rollout-{width}", cases=cases, models=models,
                evaluator=evaluate_rollouts, width=int(width), exp300=self.exp300,
                key_fields=key_fields, guard=self.guard,
                matched_work=matched_work,
            )
            evaluation["rollouts"][str(width)] = result
            reused_units.append({"unit": f"rollout-{width}", "reused": reused})
            self.ledger.state["completed_evaluation_units"].append(f"rollout-{width}")
            self.ledger.state["cache_receipts"].append(reused_units[-1])
            self.ledger.persist()
        repeat_cases = generate_screen_rollout_cases(
            self.evaluation_root,
            int(spec["exact_repeat_width"]),
            int(spec["exact_repeat_cases"]),
            self.authorization_path,
            self.authorization,
        )
        repeat_first, reused_first = load_or_evaluate_unit(
            unit="exact-repeat-first-256", cases=repeat_cases, models=models,
            evaluator=evaluate_rollouts, width=int(spec["exact_repeat_width"]),
            exp300=self.exp300, key_fields=key_fields, guard=self.guard,
            matched_work=matched_work,
        )
        repeat_second, reused_second = load_or_evaluate_unit(
            unit="exact-repeat-second-256", cases=repeat_cases, models=models,
            evaluator=evaluate_rollouts, width=int(spec["exact_repeat_width"]),
            exp300=self.exp300, key_fields=key_fields, guard=self.guard,
            matched_work=matched_work,
        )
        reused_units.extend([
            {"unit": "exact-repeat-first-256", "reused": reused_first},
            {"unit": "exact-repeat-second-256", "reused": reused_second},
        ])
        self.ledger.state["completed_evaluation_units"].extend(["exact-repeat-first-256", "exact-repeat-second-256"])
        self.ledger.state["cache_receipts"].extend(reused_units[-2:])
        self.ledger.persist()
        exact_repeat = {}
        for arm in models:
            first = repeat_first[arm]
            second = repeat_second[arm]
            exact_repeat[arm] = {
                "first": first,
                "second": second,
                "deterministic": prediction_trajectory_repeat_deterministic(
                    first, second
                ),
            }
        matched_work_receipt = finalize_matched_work(matched_work)
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
        integrity = {
            "source_checkpoint_hash_valid": sha256(SOURCE_CHECKPOINT_PATH)
            == PINNED["source_checkpoint_sha256"],
            "identical_initialization_valid": len(set(initial.values())) == 1,
            "parameter_count_valid": all(
                sum(parameter.numel() for parameter in model.parameters()) == 99_824
                for model in models.values()
            ),
            "paired_data_order_valid": batch_hashes["control"] == batch_hashes["challenger"],
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
            "matched_work_valid": matched_work_receipt["valid"],
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
        self.ledger.update(phase="finalization", gates_computed=False)
        result = {
            "schema_version": "exp648-screen-result-v1",
            "training_root": self.training_root,
            "evaluation_root": self.evaluation_root,
            "initial_tensor_sha256": initial,
            "authorization_sha256": sha256(self.authorization_path),
            "screen_readiness_binding": self.binding,
            "corpus_verification": corpus_verification,
            "frozen_corpus_semantic_replay": semantic_replay,
            "ordered_pair_permutation_sha256": sha256_bytes(canonical_json_bytes(order)),
            "ordered_batch_sha256": sha256_bytes(canonical_json_bytes(batch_hashes["control"])),
            "matched_work": matched_work_receipt,
            "histories": histories,
            "evaluation": compact_evaluation_results(evaluation),
            "exact_repeat": compact_evaluation_results(exact_repeat),
            "arm_metrics": arm_metrics,
            "paired_width256_drift_counts": width256_drift,
            "checkpoint_hashes": checkpoint_hashes,
            "evaluation_cache": {"key_fields": key_fields, "units": reused_units},
            "timing": self.guard.timing(),
            "scientific_integrity": integrity,
            "gates": gates,
            "decision": "pass-intermediate-requires-separate-confirmation-judge"
            if gates["all_passed"]
            else "reject-frozen-screen",
        }
        return finalize_screen_evidence(result, self.environment, self.guard, self.ledger)


def finalize_screen_evidence(
    result: dict[str, Any], environment: dict[str, Any], guard: ScreenGuard,
    ledger: ScreenAttemptLedger,
) -> dict[str, Any]:
    launch_attempt_id = ledger.state.get("launch_attempt_id")
    if not isinstance(launch_attempt_id, str) or not launch_attempt_id:
        raise RuntimeError("screen scientific attempt lacks reciprocal launch attempt ID")
    orchestration = {
        "launch_attempt_id": launch_attempt_id,
        "scientific_attempt_id": ledger.attempt_id,
        "linkage_direction": "scientific-evidence-to-launch-attempt-id-without-launch-ledger-hash",
    }
    result["sealed_orchestration"] = orchestration
    metrics = {
        "schema_version": "exp648-screen-metrics-v1",
        "role": "frozen-preregistered-screen-only",
        "arm_metrics": result["arm_metrics"],
        "paired_width256_drift_counts": result["paired_width256_drift_counts"],
        "gates": result["gates"],
        "candidate_promotion_metric": False,
        "benchmark_metric": False,
    }
    integrity = {
        "schema_version": "exp648-screen-integrity-v1",
        "authorization_sha256": result["authorization_sha256"],
        "sealed_orchestration": orchestration,
        "checks": {
            **result["scientific_integrity"],
            "frozen_corpus_historical_bytes_valid": result["corpus_verification"]["historical_execution_provenance_verified"],
            "frozen_corpus_source_semantic_replay_valid": result["frozen_corpus_semantic_replay"]["exact_complete_source_replay"],
            "checkpoints_saved_before_evaluation": all(v["saved_and_verified_before_evaluation"] for v in result["checkpoint_hashes"].values()),
            "dynamic_wall_cpu_rss_guards_valid": True,
            "first_quarter_numeric_telemetry_serialized": result["timing"]["first_quarter"] is not None,
            "evaluation_cache_provenance_bound": True,
            "checked_independent_official_private_unopened": True,
            "candidate_005_unchanged": True,
        },
    }
    if not all(type(value) is bool and value for value in integrity["checks"].values()):
        raise RuntimeError("screen final integrity failed")
    result["timing"] = guard.timing()
    write_screen_file("run.json", result)
    write_screen_file("environment.json", {
        "schema_version": "exp648-screen-environment-v1",
        "payload": environment,
        "payload_sha256": sha256_bytes(canonical_json_bytes(environment)),
    })
    write_screen_file("metrics.json", metrics)
    evidence_hashes = {
        name: sha256(TRACKED_SCREEN_DIR / name)
        for name in ("run.json", "environment.json", "metrics.json", "checkpoints.json", "first-quarter-telemetry.json")
    }
    integrity["evidence_hashes"] = evidence_hashes
    integrity["evidence_set_sha256"] = sha256_bytes(canonical_json_bytes(evidence_hashes))
    write_screen_file("integrity.json", integrity)

    experiment_id = cal.configure_mlflow("sair-modular-arithmetic-wsl")
    required_tags = {
        "canonical_run_id": RUN_ID, "role": "frozen-preregistered-screen",
        "authorization_sha256": result["authorization_sha256"],
        "binding_sha256": result["screen_readiness_binding"]["screen_workload_sha256"],
        "checked_public_opened": "false", "independent_opened": "false",
        "official_private_opened": "false", "candidate_registration": "false",
    }
    required_params = {
        "training_root": str(SCREEN_TRAINING_ROOT), "evaluation_root": str(SCREEN_EVALUATION_ROOT),
        "scientific_runner_sha256": sha256(Path(__file__)),
        "screen_workload_sha256": result["screen_readiness_binding"]["screen_workload_sha256"],
        "corpus_manifest_sha256": result["corpus_verification"]["corpus_manifest_sha256"],
        "control_checkpoint_sha256": result["checkpoint_hashes"]["control"]["sha256"],
        "challenger_checkpoint_sha256": result["checkpoint_hashes"]["challenger"]["sha256"],
    }
    flat_metrics = {
        f"{arm}_{key}": float(value)
        for arm, values in result["arm_metrics"].items()
        for key, value in values.items()
    }
    flat_metrics.update({f"gate_{key}": float(value) for key, value in result["gates"].items() if type(value) in (bool, int, float)})
    client = cal.mlflow.tracking.MlflowClient(tracking_uri=cal.tracking_uri())
    matches = client.search_runs(
        experiment_ids=[str(experiment_id)],
        filter_string=(f"tags.canonical_run_id = '{RUN_ID}' and tags.role = 'frozen-preregistered-screen'"),
        max_results=50,
    )
    matches = [r for r in matches if r.data.tags.get("authorization_sha256") == result["authorization_sha256"]]
    finished = select_finished_or_supersede_partial_mlflow_runs(
        client, matches, reason="screen-exact-auth-binding-checkpoints"
    )
    reused_mlflow = finished is not None
    if finished is not None:
        mlflow_run_id = finished.info.run_id
    else:
        with mlflow.start_run(run_name=f"{RUN_ID}-screen") as active:
            mlflow_run_id = active.info.run_id
            mlflow.set_tags(required_tags)
            mlflow.log_params(required_params)
            mlflow.log_metrics(flat_metrics)
            for name in ("run.json", "environment.json", "metrics.json", "integrity.json", "checkpoints.json", "first-quarter-telemetry.json"):
                mlflow.log_artifact(str(TRACKED_SCREEN_DIR / name), artifact_path="screen-evidence")
                guard.check(f"mlflow:{name}")
    observed = client.get_run(mlflow_run_id)
    if observed.info.status != "FINISHED":
        raise RuntimeError("screen MLflow run did not finish")
    custom_tags = {k: v for k, v in observed.data.tags.items() if not k.startswith("mlflow.")}
    if custom_tags != required_tags or dict(observed.data.params) != required_params or set(observed.data.metrics) != set(flat_metrics):
        raise RuntimeError("screen MLflow exact tag/param/metric schema verification failed")
    if any(not math.isclose(float(observed.data.metrics[key]), value, rel_tol=1e-12, abs_tol=1e-9) for key, value in flat_metrics.items()):
        raise RuntimeError("screen MLflow metric value verification failed")
    verify_dir = SCRATCH_SCREEN_DIR / "mlflow-verification" / mlflow_run_id
    verified_artifacts = []
    for name in ("run.json", "environment.json", "metrics.json", "integrity.json", "checkpoints.json", "first-quarter-telemetry.json"):
        downloaded = Path(client.download_artifacts(mlflow_run_id, f"screen-evidence/{name}", str(verify_dir)))
        source = TRACKED_SCREEN_DIR / name
        if sha256(downloaded) != sha256(source):
            raise RuntimeError(f"screen MLflow artifact hash mismatch for {name}")
        verified_artifacts.append({"path": name, "sha256": sha256(source), "verified": True})
        guard.check(f"mlflow-verify:{name}")
    receipt = {
        "schema_version": "exp648-screen-mlflow-receipt-v1",
        "run_id": mlflow_run_id, "status": observed.info.status,
        "reused": reused_mlflow,
        "tags_verified": True, "params_verified": True, "metrics_verified": True,
        "expected_tags": required_tags, "expected_params": required_params,
        "expected_metrics": flat_metrics,
        "artifacts": verified_artifacts, "final_resource_timing": guard.timing(),
        "benchmark_metrics_logged": False, "candidate_registration_performed": False,
    }
    write_screen_file("mlflow.json", receipt)
    result["mlflow"] = receipt
    write_screen_file("final-timing.json", {
        "schema_version": "exp648-screen-final-timing-v1",
        "workload_timing": result["timing"],
        "through_remote_verification_preseal_timing": guard.timing(),
        "excluded_tail": "bounded local atomic copies, completion-seal write, and completion-pointer publication",
        "mlflow_run_id": receipt["run_id"],
    })
    attempt_tracked = TRACKED_SCREEN_DIR / "attempts" / ledger.attempt_id
    names = (
        "first-quarter-telemetry.json", "checkpoints.json", "run.json", "environment.json",
        "metrics.json", "integrity.json", "mlflow.json", "final-timing.json",
    )
    for name in names:
        _atomic_write(attempt_tracked / name, (TRACKED_SCREEN_DIR / name).read_bytes())
    checkpoint_artifacts = {}
    for arm in ("control", "challenger"):
        receipt = result["checkpoint_hashes"][arm]
        if receipt.get("attempt_id") != ledger.attempt_id:
            raise RuntimeError("screen checkpoint receipt is not bound to this attempt")
        source = ROOT / receipt["path"]
        if not source.is_file() or sha256(source) != receipt["sha256"]:
            raise RuntimeError("screen checkpoint bytes changed before sealing")
        relative = f"checkpoints/{arm}.safetensors"
        destination = attempt_tracked / relative
        _atomic_write(destination, source.read_bytes())
        if sha256(destination) != receipt["sha256"]:
            raise RuntimeError("sealed screen checkpoint copy hash mismatch")
        checkpoint_artifacts[relative] = receipt["sha256"]
    ledger.update(phase="sealed", gates_computed=True, complete=True)
    _atomic_write(attempt_tracked / "attempt-state.json", (ledger.directory / "attempt-state.json").read_bytes())
    hashes = {
        name: sha256(attempt_tracked / name)
        for name in (*names, "attempt-state.json")
    }
    hashes.update(checkpoint_artifacts)
    seal = {
        "schema_version": "exp648-screen-completion-seal-v1",
        "attempt_id": ledger.attempt_id, "authorization_sha256": result["authorization_sha256"],
        "artifacts": hashes, "complete": True,
    }
    _atomic_write(attempt_tracked / "completion-seal.json", canonical_json_bytes(seal, pretty=True))
    pointer = {
        "schema_version": "exp648-screen-completion-pointer-v1",
        "attempt_id": ledger.attempt_id,
        "completion_seal_sha256": sha256(attempt_tracked / "completion-seal.json"),
    }
    _atomic_write(TRACKED_SCREEN_DIR / "completion.json", canonical_json_bytes(pointer, pretty=True))
    verify_screen_completion_seal(TRACKED_SCREEN_DIR / "completion.json")
    return result


def verify_screen_completion_seal(
    pointer_path: Path, *, launch_state_path: Path | None = None,
) -> dict[str, Any]:
    pointer_raw = pointer_path.read_bytes()
    pointer = json.loads(pointer_raw)
    if pointer_raw != canonical_json_bytes(pointer, pretty=True) or set(pointer) != {"schema_version", "attempt_id", "completion_seal_sha256"}:
        raise RuntimeError("screen completion pointer schema/canonical mismatch")
    attempt = pointer_path.parent / "attempts" / pointer["attempt_id"]
    seal_path = attempt / "completion-seal.json"
    if not seal_path.is_file() or sha256(seal_path) != pointer["completion_seal_sha256"]:
        raise RuntimeError("screen completion seal pointer hash mismatch")
    seal_raw = seal_path.read_bytes()
    seal = json.loads(seal_raw)
    if seal_raw != canonical_json_bytes(seal, pretty=True) or set(seal) != {"schema_version", "attempt_id", "authorization_sha256", "artifacts", "complete"} or seal.get("complete") is not True:
        raise RuntimeError("screen completion seal schema mismatch")
    if seal.get("attempt_id") != pointer["attempt_id"]:
        raise RuntimeError("screen completion seal attempt ID mismatch")
    required = {"first-quarter-telemetry.json", "checkpoints.json", "run.json", "environment.json", "metrics.json", "integrity.json", "mlflow.json", "final-timing.json", "attempt-state.json", "checkpoints/control.safetensors", "checkpoints/challenger.safetensors"}
    if set(seal["artifacts"]) != required:
        raise RuntimeError("screen completion artifact set mismatch")
    for name, digest in seal["artifacts"].items():
        if not (attempt / name).is_file() or sha256(attempt / name) != digest:
            raise RuntimeError(f"screen sealed artifact mismatch: {name}")
    attempt_state = json.loads((attempt / "attempt-state.json").read_bytes())
    run = json.loads((attempt / "run.json").read_bytes())
    integrity = json.loads((attempt / "integrity.json").read_bytes())
    checkpoint_receipt = json.loads((attempt / "checkpoints.json").read_bytes())
    authorization_sha256 = seal.get("authorization_sha256")
    if (
        attempt_state.get("phase") != "sealed"
        or attempt_state.get("complete") is not True
        or attempt_state.get("gates_computed") is not True
    ):
        raise RuntimeError("screen sealed attempt state is not terminal-complete")
    if (
        not isinstance(authorization_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", authorization_sha256)
        or attempt_state.get("authorization_sha256") != authorization_sha256
        or run.get("authorization_sha256") != authorization_sha256
        or integrity.get("authorization_sha256") != authorization_sha256
        or checkpoint_receipt.get("authorization_sha256") != authorization_sha256
    ):
        raise RuntimeError("screen sealed authorization binding mismatch")
    for arm in ("control", "challenger"):
        receipt = checkpoint_receipt.get("checkpoint_hashes", {}).get(arm, {})
        if (
            receipt.get("attempt_id") != pointer["attempt_id"]
            or receipt.get("sha256")
            != seal["artifacts"][f"checkpoints/{arm}.safetensors"]
        ):
            raise RuntimeError("screen sealed checkpoint receipt/bytes mismatch")
    launch_attempt_id = attempt_state.get("launch_attempt_id")
    expected_orchestration = {
        "launch_attempt_id": launch_attempt_id,
        "scientific_attempt_id": pointer["attempt_id"],
        "linkage_direction": "scientific-evidence-to-launch-attempt-id-without-launch-ledger-hash",
    }
    if (
        not isinstance(launch_attempt_id, str)
        or not launch_attempt_id
        or attempt_state.get("attempt_id") != pointer["attempt_id"]
        or run.get("sealed_orchestration") != expected_orchestration
        or integrity.get("sealed_orchestration") != expected_orchestration
    ):
        raise RuntimeError("screen sealed reciprocal launch/scientific linkage mismatch")
    if launch_state_path is not None:
        launch_raw = launch_state_path.read_bytes()
        launch = json.loads(launch_raw)
        if launch_raw != canonical_json_bytes(launch, pretty=True):
            raise RuntimeError("screen terminal launch ledger is noncanonical")
        if (
            launch.get("phase") != "complete-sealed"
            or launch.get("complete") is not True
            or launch.get("attempt_id") != launch_attempt_id
            or launch.get("scientific_attempt_id") != pointer["attempt_id"]
            or launch.get("completion_pointer_sha256") != sha256(pointer_path)
            or launch.get("completion_seal_sha256") != sha256(seal_path)
            or launch.get("scientific_attempt_state_sha256") != sha256(attempt / "attempt-state.json")
        ):
            raise RuntimeError("screen terminal reciprocal launch/scientific pair mismatch")
    return seal


def inventory_attempt_checkpoint_artifacts(
    checkpoint_dir: Path, receipts: dict[str, Any]
) -> dict[str, Any]:
    """Inventory every direct checkpoint artifact without following unsafe links."""
    artifacts = []
    unsafe_entries = []
    if os.path.lexists(checkpoint_dir) and checkpoint_dir.is_symlink():
        return {
            "checkpoint_directory": checkpoint_dir.as_posix(),
            "directory_present": True,
            "artifacts": [],
            "unsafe_entries": [{
                "name": checkpoint_dir.name,
                "kind": "checkpoint-directory-symlink",
            }],
            "all_entries_regular_files": False,
            "all_artifacts_receipt_bound": False,
        }
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        return {
            "checkpoint_directory": checkpoint_dir.as_posix(),
            "directory_present": True,
            "artifacts": [],
            "unsafe_entries": [{
                "name": checkpoint_dir.name,
                "kind": "checkpoint-directory-is-not-a-directory",
            }],
            "all_entries_regular_files": False,
            "all_artifacts_receipt_bound": False,
        }
    if checkpoint_dir.is_dir():
        for entry in sorted(checkpoint_dir.iterdir(), key=lambda value: value.name):
            mode = entry.lstat().st_mode
            if stat.S_ISREG(mode):
                digest = sha256(entry)
                matching = [
                    arm for arm, receipt in receipts.items()
                    if isinstance(receipt, dict)
                    and Path(str(receipt.get("path", ""))).name == entry.name
                ]
                if len(matching) == 0:
                    binding_status = "unbound-no-receipt"
                elif len(matching) > 1:
                    binding_status = "ambiguous-multiple-receipts"
                elif receipts[matching[0]].get("sha256") != digest:
                    binding_status = "receipt-hash-mismatch"
                else:
                    binding_status = "receipt-bound"
                artifacts.append({
                    "name": entry.name,
                    "sha256": digest,
                    "bytes": entry.stat(follow_symlinks=False).st_size,
                    "matching_receipt_arms": matching,
                    "exactly_one_matching_arm_receipt": len(matching) == 1,
                    "binding_status": binding_status,
                    "receipt_binds_bytes": binding_status == "receipt-bound",
                })
            else:
                kind = (
                    "symlink" if stat.S_ISLNK(mode)
                    else "directory" if stat.S_ISDIR(mode)
                    else "non-regular"
                )
                unsafe_entries.append({"name": entry.name, "kind": kind})
    return {
        "checkpoint_directory": checkpoint_dir.as_posix(),
        "directory_present": checkpoint_dir.is_dir(),
        "artifacts": artifacts,
        "unsafe_entries": unsafe_entries,
        "all_entries_regular_files": not unsafe_entries,
        "all_artifacts_receipt_bound": bool(artifacts) and all(
            artifact["binding_status"] == "receipt-bound"
            for artifact in artifacts
        ),
    }


def preserve_screen_failure(
    stage: str, error: BaseException, experiment: ScreenExperiment | None = None,
    launch_state_path: Path | None = None,
) -> None:
    # Called only after exact authorization has passed.  It preserves completed
    # work without fabricating gates or a completed scientific result.
    phase = experiment.ledger.state["phase"] if experiment is not None else stage
    if isinstance(error, (TimeoutError, MemoryError)):
        if phase in ("training", "corpus-verified"):
            status = "training-resource-overrun"
        elif phase == "evaluation":
            status = "evaluation-resource-overrun"
        else:
            status = "finalization-resource-overrun"
    elif phase in ("finalization", "sealed"):
        status = "mlflow-or-finalization-error"
    else:
        status = "generic-error"
    launch_state = (
        json.loads(launch_state_path.read_text(encoding="utf-8"))
        if launch_state_path is not None and launch_state_path.is_file() else {}
    )
    state = experiment.ledger.state if experiment is not None else launch_state
    failure_process = psutil.Process()
    failure_cpu = failure_process.cpu_times()
    launch_failure_timing = {
        "wall_seconds": time.perf_counter()
        - float(launch_state.get("launch_wall_started_perf_counter", _MODULE_WALL_STARTED)),
        "aggregate_cpu_seconds": float(failure_cpu.user + failure_cpu.system)
        - float(launch_state.get("launch_process_cpu_started_seconds", 0.0)),
        "peak_rss_bytes": max(
            int(launch_state.get("launch_initial_peak_rss_bytes", 0)),
            cal.process_peak_rss_bytes(failure_process),
        ),
        "sampled_fresh_at_exception_boundary": True,
    }
    if experiment is not None:
        experiment.ledger.state["final_timing"] = experiment.guard.timing()
        experiment.ledger.persist()
        state = experiment.ledger.state
    checkpoint_directory = (
        experiment.ledger.directory / "checkpoints"
        if experiment is not None else SCRATCH_SCREEN_DIR / "absent-attempt/checkpoints"
    )
    checkpoint_inventory = inventory_attempt_checkpoint_artifacts(
        checkpoint_directory, state.get("verified_checkpoints", {})
    )
    preserved_checkpoints = {}
    for arm, receipt in state.get("verified_checkpoints", {}).items():
        path_text = receipt.get("path") if isinstance(receipt, dict) else None
        path = ROOT / path_text if isinstance(path_text, str) else None
        preserved_checkpoints[arm] = {
            "receipt": receipt,
            "bytes_present": bool(path is not None and path.is_file()),
            "observed_sha256": sha256(path) if path is not None and path.is_file() else None,
            "receipt_hash_matches_preserved_bytes": bool(
                path is not None
                and path.is_file()
                and sha256(path) == receipt.get("sha256")
            ),
        }
    payload = {
        "schema_version": "exp648-screen-incomplete-v1",
        "status": status, "stage": stage, "phase": phase,
        "attempt_id": state.get("attempt_id"),
        "launch_attempt_id": launch_state.get("attempt_id"),
        "authorization_sha256": state.get("authorization_sha256"),
        "error_type": type(error).__name__, "error": str(error),
        "budgets": state.get("budgets"), "final_timing": state.get("final_timing"),
        "launch_failure_timing": launch_failure_timing,
        "completed_updates": state.get("completed_updates", {}),
        "completed_evaluation_units": state.get("completed_evaluation_units", []),
        "cache_receipts": state.get("cache_receipts", []),
        "verified_checkpoints": state.get("verified_checkpoints", {}),
        "preserved_checkpoint_bytes": preserved_checkpoints,
        "attempt_checkpoint_artifact_inventory": checkpoint_inventory,
        "first_quarter": state.get("first_quarter"),
        "environment_sha256": state.get("environment_sha256"),
        "recovery_instructions": "Reuse only complete hash-validated checkpoints/evaluation units under the same committed authorization and binding; otherwise start a new attempt.",
        "gates_computed": False, "candidate_changed": False,
        "checked_independent_official_private_opened": False,
    }
    if experiment is not None:
        experiment.ledger.state.update({"phase": "incomplete", "complete": False, "gates_computed": False})
        experiment.ledger.persist()
        _atomic_write(experiment.ledger.directory / "incomplete.json", canonical_json_bytes(payload, pretty=True))
    if launch_state_path is not None:
        terminal = transition_screen_launch_ledger(
            launch_state_path,
            phase="terminal-constructor-failure" if experiment is None else "terminal-scientific-attempt-failure",
            complete=False,
            scientific_attempt_id=(experiment.ledger.attempt_id if experiment is not None else None),
        )
        terminal["failure"] = payload
        _atomic_write(launch_state_path, canonical_json_bytes(terminal, pretty=True))
    write_screen_file("incomplete.json", payload)


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
    # This verifies the immutable historical execution as history.  The live
    # runner/HEAD are deliberately not asserted to have executed these bytes.
    paths = {
        "data/identity-manifest.json": DATA_DIR / "identity-manifest.json",
        "data/trajectory-manifest.json.gz": DATA_DIR / "trajectory-manifest.json.gz",
        "data/paired-corpus.json.gz": DATA_DIR / "paired-corpus.json.gz",
        "data/corpus-manifest.json": DATA_DIR / "corpus-manifest.json",
        "evidence/run.json": TRACKED_CORPUS_DIR / "run.json",
        "evidence/environment.json": TRACKED_CORPUS_DIR / "environment.json",
        "evidence/integrity.json": TRACKED_CORPUS_DIR / "integrity.json",
        "evidence/mlflow.json": TRACKED_CORPUS_DIR / "mlflow.json",
    }
    for name, expected_hash in FROZEN_CORPUS_FILES.items():
        if not paths[name].is_file() or sha256(paths[name]) != expected_hash:
            raise RuntimeError(f"immutable historical corpus byte mismatch: {name}")
    amendment_path = TRACKED_CORPUS_DIR / "provenance-amendment.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment["execution"]["repository_head"] != PINNED["repository_head"]:
        raise RuntimeError("historical execution HEAD amendment mismatch")
    if amendment["execution"]["scoped_executed_bytes"]["scientific_runner"]["sha256"] != "83181524cfa4cdfb7a90b020f41cf32c0d598628f97379086bbd81f6172fd9e2":
        raise RuntimeError("historical runner amendment mismatch")
    historical = subprocess.run(
        ["git", "show", "8d960fa747b64bffa3b60581ae4ff518436a8887:experiments/exp_648_structured_trajectory_consistency/scientific.py"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    if sha256_bytes(historical) != "83181524cfa4cdfb7a90b020f41cf32c0d598628f97379086bbd81f6172fd9e2":
        raise RuntimeError("immutable corpus commit does not preserve executed runner bytes")
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
    scratch_parity: bool | None = None
    scratch_pairs = []
    for name in (
        "identity-manifest.json",
        "trajectory-manifest.json.gz",
        "paired-corpus.json.gz",
        "corpus-manifest.json",
    ):
        raw_path = SCRATCH_CORPUS_DIR / "data" / name
        if raw_path.is_file():
            scratch_pairs.append(raw_path.read_bytes() == (DATA_DIR / name).read_bytes())
    if scratch_pairs:
        scratch_parity = all(scratch_pairs) and len(scratch_pairs) == 4
        if not scratch_parity:
            raise RuntimeError("optional scratch corpus parity mismatch")
    if require_receipts:
        for name in ("run.json", "environment.json", "integrity.json", "mlflow.json"):
            tracked = TRACKED_CORPUS_DIR / name
            raw = SCRATCH_CORPUS_DIR / "evidence" / name
            if not tracked.is_file():
                raise RuntimeError(f"durable committed evidence is absent for {name}")
            if raw.is_file() and tracked.read_bytes() != raw.read_bytes():
                raise RuntimeError(f"optional scratch evidence parity mismatch for {name}")
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
        "scratch_raw_tracked_byte_parity_when_present": scratch_parity,
        "historical_execution_provenance_verified": True,
        "historical_corpus_commit": "8d960fa747b64bffa3b60581ae4ff518436a8887",
        "current_verifier_sha256": sha256(Path(__file__)),
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
        "--mode", choices=("corpus", "screen", "verify-corpus", "readiness-calibration"), required=True
    )
    parser.add_argument("--root", type=int)
    parser.add_argument("--authorization-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "readiness-calibration":
        authorization_path = args.authorization_manifest or READINESS_CALIBRATION_AUTHORIZATION_PATH
        authorization = verify_readiness_calibration_authorization(authorization_path)
        process = psutil.Process()
        cpu = process.cpu_times()
        launch_cpu_started = float(cpu.user + cpu.system) - (
            time.process_time() - _MODULE_PROCESS_CPU_STARTED
        )
        if args.root is not None and int(args.root) != int(authorization["calibration_root"]):
            raise PermissionError("CLI calibration root differs from committed authorization")
        result = execute_readiness_calibration(
            authorization, launch_wall_started=_MODULE_WALL_STARTED,
            launch_cpu_started=launch_cpu_started,
            launch_peak_rss=cal.process_peak_rss_bytes(process),
        )
        print(json.dumps({
            "status": result["status"], "calibration_root": result["calibration_root"],
            "resolved_screen_budgets": result["resolved_screen_budgets"],
            "scientific_accuracy_or_selection_reported": False,
        }, indent=2), flush=True)
        return 0
    if args.mode == "screen":
        authorization_path = args.authorization_manifest or SCREEN_AUTHORIZATION_PATH
        # Fail before constructing ScreenExperiment or reading any root/data/model/environment.
        authorization = verify_screen_authorization(authorization_path)
        process = psutil.Process()
        launch_wall_started = _MODULE_WALL_STARTED
        cpu = process.cpu_times()
        launch_cpu_started = float(cpu.user + cpu.system) - (
            time.process_time() - _MODULE_PROCESS_CPU_STARTED
        )
        launch_peak_rss = cal.process_peak_rss_bytes(process)
        launch_state_path = create_screen_launch_ledger(
            authorization, authorization_path,
            launch_wall_started=launch_wall_started,
            launch_cpu_started=launch_cpu_started,
            launch_peak_rss=launch_peak_rss,
        )
        root = SCREEN_TRAINING_ROOT if args.root is None else args.root
        result = execute_authorized_screen_launch(
            authorization=authorization,
            authorization_path=authorization_path,
            launch_state_path=launch_state_path,
            root=root,
            launch_wall_started=launch_wall_started,
            launch_cpu_started=launch_cpu_started,
            launch_peak_rss=launch_peak_rss,
        )
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
