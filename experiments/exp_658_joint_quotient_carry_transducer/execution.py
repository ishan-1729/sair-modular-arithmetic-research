"""Authorized EXP-658 T045 resource-calibration and decision runner.

Every operation is internal and selected from the immutable physical manifest.
L1 persists resources and integrity only; it never releases predictions or
scientific metrics.  A binding admission failure therefore yields a durable
park decision without weakening the frozen L2 experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from . import calibration as cal
from . import data as exact_data
from . import model as model_module


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "evals/results/exp-658-joint-quotient-carry-transducer"
AUTHORIZATION_SCHEMA = "exp658-t045-source-authorization-v1"
CALIBRATION_SCHEMA = "exp658-t045-l1-calibration-v1"
DECISION_SCHEMA = "exp658-t045-decision-v1"
ATTEMPT_ALLOWANCE = 512
EXECUTABLE_PATHS = (
    "configs/exp_658_joint_quotient_carry_transducer.yaml",
    "experiments/exp_658_joint_quotient_carry_transducer/__init__.py",
    "experiments/exp_658_joint_quotient_carry_transducer/calibration.py",
    "experiments/exp_658_joint_quotient_carry_transducer/data.py",
    "experiments/exp_658_joint_quotient_carry_transducer/execution.py",
    "experiments/exp_658_joint_quotient_carry_transducer/model.py",
    "experiments/exp_658_joint_quotient_carry_transducer/run.py",
    "tests/test_exp658_joint_quotient_carry_calibration.py",
    "tests/test_exp658_joint_quotient_carry_execution.py",
    "tests/test_exp658_joint_quotient_carry_transducer.py",
)


class ExecutionError(RuntimeError):
    """Fail-closed T045 execution boundary."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def executable_hashes() -> dict[str, str]:
    hashes = {}
    for relative in EXECUTABLE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise ExecutionError(f"missing executable identity path: {relative}")
        hashes[relative] = sha256_path(path)
    source = ROOT / cal.load_config()["source"]["checkpoint"]
    if not source.is_file() or sha256_path(source) != cal.SOURCE_CHECKPOINT_SHA256:
        raise ExecutionError("frozen EXP609 source checkpoint is absent or changed")
    hashes[str(source.relative_to(ROOT)).replace("\\", "/")] = sha256_path(source)
    return hashes


def prepare_authorization(path: Path) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    dirty = [line[3:] for line in _git("status", "--porcelain").splitlines()]
    evidence_prefixes = (
        "experiments/runs/exp-658-joint-quotient-carry-transducer/",
        "evals/results/exp-658-joint-quotient-carry-transducer/",
    )
    if head != upstream or any(not path.startswith(evidence_prefixes) for path in dirty):
        raise ExecutionError("source HEAD must be pushed and executable bytes clean")
    config = cal.load_config()
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "run_id": cal.RUN_ID,
        "stage": "T045-L1-then-conditional-L2",
        "source_commit": head,
        "upstream_commit": upstream,
        "executable_hashes": executable_hashes(),
        "source_checkpoint_sha256": cal.SOURCE_CHECKPOINT_SHA256,
        "base_seed": cal.BASE_SEED,
        "attempt_allowance": ATTEMPT_ALLOWANCE,
        "warmup_repeat": 0,
        "fresh_repeats": [1, 2, 3],
        "l2_denominators": cal.l2_batch_counters(),
        "cache_verdict": asdict(cal.validate_incumbent_cache(config, root=ROOT)),
        "scientific_values_authorized_in_l1": False,
        "l2_automatic_only_if_all_admission_gates_pass": True,
    }
    payload["authorization_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    write_canonical(path, payload)
    return payload


def load_authorization(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != canonical_bytes(payload):
        raise ExecutionError("authorization is not canonical JSON")
    recorded = payload.pop("authorization_sha256", None)
    observed = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    payload["authorization_sha256"] = recorded
    if recorded != observed or payload.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ExecutionError("authorization hash/schema mismatch")
    if payload["source_commit"] != _git("rev-parse", "HEAD"):
        raise ExecutionError("authorization source commit is not current HEAD")
    if payload["executable_hashes"] != executable_hashes():
        raise ExecutionError("authorized executable identity changed")
    return payload


def configure_runtime() -> dict[str, Any]:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    deterministic = cal.deterministic_cpu_preflight(
        model_module.JointQuotientCarryTransducer(hidden=4, cycles=1),
        amp_enabled=False,
    )
    live = cal.live_resource_environment()
    expected = {
        "affinity_cpus": (0, 1, 2, 3),
        "cpu_limit": 4.0,
        "memory_limit_bytes": 8 * 1024**3,
        "tmpfs_limit_bytes": 2 * 1024**3,
    }
    for key, expected_value in expected.items():
        observed = tuple(live[key]) if key == "affinity_cpus" else live[key]
        if observed != expected_value:
            raise ExecutionError(f"live pinned resource mismatch: {key}")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "deterministic": deterministic,
        "resources": live,
    }


def _snapshot_payload(value: cal.MeterSnapshot) -> dict[str, Any]:
    return {
        "wall": value.monotonic_wall_seconds,
        "aggregate_cpu": value.process_user_plus_system_aggregate_cpu_seconds,
        "rss": value.simultaneous_tree_rss_bytes,
    }


@dataclass
class Segment:
    before: cal.MeterSnapshot
    after: cal.MeterSnapshot
    inner_wall: float | None = None


class Recorder:
    """Closed fixed-operation meter supporting interleaved matched arms."""

    def __init__(self, manifest: Sequence[cal.PhysicalUnit]) -> None:
        self.manifest = tuple(manifest)
        self.by_key = {item.key: item for item in self.manifest}
        self.segments: dict[str, list[Segment]] = defaultdict(list)

    def measure(self, key: str, operation: Callable[[], Any], *, inner: bool = False) -> Any:
        if key not in self.by_key:
            raise ExecutionError(f"operation is absent from canonical manifest: {key}")
        before = cal.cumulative_resource_meter_snapshot()
        inner_start = time.perf_counter() if inner else None
        result = operation()
        inner_wall = time.perf_counter() - inner_start if inner_start is not None else None
        after = cal.cumulative_resource_meter_snapshot()
        self.segments[key].append(Segment(before, after, inner_wall))
        return result

    def record(self, key: str) -> dict[str, Any]:
        physical = self.by_key[key]
        rows = self.segments.get(key, [])
        if not rows:
            raise ExecutionError(f"canonical operation was not executed: {key}")
        wall = sum(row.after.monotonic_wall_seconds - row.before.monotonic_wall_seconds for row in rows)
        cpu = sum(
            row.after.process_user_plus_system_aggregate_cpu_seconds
            - row.before.process_user_plus_system_aggregate_cpu_seconds
            for row in rows
        )
        rss = max(row.after.simultaneous_tree_rss_bytes for row in rows)
        if wall <= 0 or cpu <= 0 or rss <= 0:
            raise ExecutionError(f"non-positive measured resource for {key}")
        payload = {
            "physical_unit": asdict(physical),
            "operation_id": "exp658-fixed-" + key,
            "segments": [
                {
                    "before": _snapshot_payload(row.before),
                    "after": _snapshot_payload(row.after),
                    "inner_wall": row.inner_wall,
                }
                for row in rows
            ],
            "derived": {
                "monotonic_wall_seconds": wall,
                "aggregate_cpu_seconds": cpu,
                "peak_rss_bytes": rss,
                "wall_seconds_per_unit": wall / physical.units,
                "aggregate_cpu_seconds_per_unit": cpu / physical.units,
            },
            "callback_surface": False,
        }
        payload["record_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        return payload

    def finalize(self) -> list[dict[str, Any]]:
        missing = set(self.by_key).difference(self.segments)
        if missing:
            raise ExecutionError(f"unexecuted canonical operations: {sorted(missing)}")
        return [self.record(item.key) for item in self.manifest]


def _cases(rows: Iterable[cal.OwnedCase], *, repeat: int, width: int) -> list[cal.OwnedCase]:
    return [row for row in rows if row.repeat == repeat and row.width == width]


def make_batch(rows: Sequence[cal.OwnedCase]) -> cal.ImmutableBatch:
    if not rows:
        raise ExecutionError("cannot materialize an empty batch")
    width = rows[0].width
    transitions = [
        exact_data.LegalTransition(row.state, row.multiplicand, row.modulus, row.operand_bit)
        for row in rows
    ]
    traces = [exact_data.quotient_carry_trace(row, width) for row in transitions]
    return cal.immutable_batch(
        exact_data.feature_tensor(transitions, width),
        torch.tensor([row.target_bits for row in traces], dtype=torch.float32),
        torch.tensor([row.quotient for row in traces], dtype=torch.long),
        torch.tensor([row.signed_carry_class_indices for row in traces], dtype=torch.long),
    )


def _model_forward(model: torch.nn.Module, rows: Sequence[cal.OwnedCase]) -> Any:
    batch = make_batch(rows)
    with torch.inference_mode():
        return model(batch.features)


def _rollout(
    model: torch.nn.Module,
    rows: Sequence[cal.OwnedCase],
    *,
    ablation: str = "none",
) -> list[int]:
    if not rows:
        return []
    width = rows[0].width
    states = [0] * len(rows)
    with torch.inference_mode():
        for position in cal.rollout_bit_positions(width):
            transitions = [
                exact_data.LegalTransition(
                    state=state,
                    multiplicand=row.multiplicand,
                    modulus=row.modulus,
                    operand_bit=(row.multiplier >> position) & 1,
                )
                for state, row in zip(states, rows, strict=True)
            ]
            bits = model(
                exact_data.feature_tensor(transitions, width),
                message_ablation=ablation,
            ).residue_logits >= 0
            states = [
                cal.decode_little_endian_for_next_feature(
                    bits[index].tolist(), row.modulus
                )[0]
                for index, row in enumerate(rows)
            ]
    return states


def _single_arm_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch: cal.ImmutableBatch,
    *,
    auxiliary_weight: float,
    rng: torch.Tensor,
) -> None:
    torch.set_rng_state(rng.clone())
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.features)
    terms = cal.exact_loss_terms(
        output,
        batch,
        quotient_weight=auxiliary_weight,
        carry_weight=auxiliary_weight,
    )
    terms.total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    if not bool(torch.isfinite(norm)):
        raise ExecutionError("non-finite gradient norm")
    optimizer.step()
    scheduler.step()


def _build_repeat(config: dict[str, Any], repeat: int):
    control, challenger = cal.frozen_matched_pair(config, repeat=repeat)
    control_optimizer, control_scheduler = cal.optimizer_and_scheduler(control)
    challenger_optimizer, challenger_scheduler = cal.optimizer_and_scheduler(challenger)
    return (
        control,
        challenger,
        control_optimizer,
        challenger_optimizer,
        control_scheduler,
        challenger_scheduler,
    )


def _train_repeat(
    recorder: Recorder,
    manifest: cal.L1IdentityManifest,
    config: dict[str, Any],
    repeat: int,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    prefix = "l1-warmup-r0" if repeat == 0 else f"l1-r{repeat}"
    control, challenger, co, ho, cs, hs = _build_repeat(config, repeat)
    batches: dict[int, list[cal.ImmutableBatch]] = {}
    for width in (32, 64):
        rows = _cases(manifest.training, repeat=repeat, width=width)
        batches[width] = recorder.measure(
            f"{prefix}-w{width}-train-generate",
            lambda rows=rows: [
                make_batch(rows[index : index + 128])
                for index in range(0, len(rows), 128)
            ],
        )
    appearance = {32: 0, 64: 0}
    for quota in cal.training_schedule():
        width = quota.width
        batch = batches[width][appearance[width]]
        appearance[width] += 1
        rng = torch.get_rng_state().clone()
        order = (
            ("control", "challenger")
            if (repeat + quota.global_step) % 2 == 0
            else ("challenger", "control")
        )
        for arm in order:
            if arm == "control":
                recorder.measure(
                    f"{prefix}-w{width}-control-update",
                    lambda: _single_arm_update(
                        control, co, cs, batch, auxiliary_weight=0.0, rng=rng
                    ),
                )
            else:
                recorder.measure(
                    f"{prefix}-w{width}-challenger-update",
                    lambda: _single_arm_update(
                        challenger, ho, hs, batch, auxiliary_weight=0.10, rng=rng
                    ),
                )
        if not cal._auxiliary_connected_difference(control, challenger):
            raise ExecutionError("paired auxiliary divergence absent")
    if cal.optimizer_name_topology(control, co) != cal.optimizer_name_topology(
        challenger, ho
    ):
        raise ExecutionError("matched optimizer topology diverged")
    return control, challenger


def _evaluate_repeat_resources(
    recorder: Recorder,
    manifest: cal.L1IdentityManifest,
    models: tuple[torch.nn.Module, torch.nn.Module],
    repeat: int,
) -> dict[tuple[int, str], list[int]]:
    control, challenger = models
    prefix = "l1-warmup-r0" if repeat == 0 else f"l1-r{repeat}"
    outputs: dict[tuple[int, str], list[int]] = {}
    for width in (32, 64):
        transitions = _cases(manifest.transitions, repeat=repeat, width=width)
        recorder.measure(
            f"{prefix}-w{width}-transition-generate",
            lambda: make_batch(transitions),
        )
        for arm, model in (("control", control), ("challenger", challenger)):
            recorder.measure(
                f"{prefix}-w{width}-{arm}-transition",
                lambda model=model: _model_forward(model, transitions),
            )
        rollouts = _cases(manifest.normal_rollouts, repeat=repeat, width=width)
        recorder.measure(
            f"{prefix}-w{width}-rollout-generate",
            lambda: tuple(row.identity for row in rollouts),
        )
        for arm, model in (("control", control), ("challenger", challenger)):
            outputs[(width, arm)] = recorder.measure(
                f"{prefix}-w{width}-{arm}-rollout",
                lambda model=model: _rollout(model, rollouts),
            )
        recorder.measure(
            f"{prefix}-w{width}-zero-q",
            lambda: _rollout(
                challenger, rollouts, ablation="zero-quotient-message"
            ),
        )
        recorder.measure(
            f"{prefix}-w{width}-zero-carry",
            lambda: _rollout(
                challenger, rollouts, ablation="zero-signed-carry-message"
            ),
        )
    return outputs


def _post_repeat3(
    recorder: Recorder,
    manifest: cal.L1IdentityManifest,
    models: tuple[torch.nn.Module, torch.nn.Module],
    normal: Mapping[tuple[int, str], list[int]],
    output: Path,
) -> tuple[list[float], dict[str, Any]]:
    control, challenger = models
    for width in (32, 64):
        q_rows = [row for row in manifest.rare_quotient if row.width == width]
        recorder.measure(
            f"l1-post-r3-w{width}-q-generate", lambda: make_batch(q_rows)
        )
        recorder.measure(
            f"l1-post-r3-w{width}-control-q-forward",
            lambda: _model_forward(control, q_rows),
        )
        recorder.measure(
            f"l1-post-r3-w{width}-q-forward",
            lambda: _model_forward(challenger, q_rows),
        )
        c_rows = [row for row in manifest.rare_carry if row.width == width]
        recorder.measure(
            f"l1-post-r3-w{width}-carry-generate", lambda: make_batch(c_rows)
        )
        recorder.measure(
            f"l1-post-r3-w{width}-control-carry-forward",
            lambda: _model_forward(control, c_rows),
        )
        recorder.measure(
            f"l1-post-r3-w{width}-carry-forward",
            lambda: _model_forward(challenger, c_rows),
        )
        repeated = recorder.measure(
            f"l1-post-r3-w{width}-determinism-extra",
            lambda: _rollout(
                challenger,
                _cases(manifest.determinism_repeat, repeat=3, width=width),
            ),
        )
        if repeated != normal[(width, "challenger")]:
            raise ExecutionError("L1 rollout determinism failed")
    probe = make_batch(_cases(manifest.transitions, repeat=3, width=32)[:2]).features
    checkpoint_dir = output / "l1-checkpoints"
    control_result = recorder.measure(
        "l1-checkpoint-control",
        lambda: cal.checkpoint_roundtrip(
            control,
            checkpoint_dir / "control.safetensors",
            probe,
            repeat=3,
            arm="control",
            ordinal=1,
        ),
    )
    challenger_result = recorder.measure(
        "l1-checkpoint-challenger",
        lambda: cal.checkpoint_roundtrip(
            challenger,
            checkpoint_dir / "challenger.safetensors",
            probe,
            repeat=3,
            arm="challenger",
            ordinal=2,
        ),
    )
    selected = cal.exact_ordinal_profile_challenger(
        control_result, challenger_result
    )
    recorder.measure(
        "l1-profile-warmup",
        lambda: _rollout(selected, manifest.profile_warmup),
        inner=True,
    )
    fresh_walls = []
    for repeat in (1, 2, 3):
        key = f"l1-profile-fresh-r{repeat}"
        recorder.measure(
            key,
            lambda repeat=repeat: _rollout(
                selected,
                _cases(manifest.profile_fresh, repeat=repeat, width=256),
            ),
            inner=True,
        )
        fresh_walls.append(
            sum(
                row.after.monotonic_wall_seconds
                - row.before.monotonic_wall_seconds
                for row in recorder.segments[key]
            )
        )
    return fresh_walls, {
        "control_checkpoint_sha256": control_result.receipt["checkpoint_sha256"],
        "challenger_checkpoint_sha256": challenger_result.receipt[
            "checkpoint_sha256"
        ],
        "control_reload_exact": control_result.receipt["tensor_equality"],
        "challenger_reload_exact": challenger_result.receipt["tensor_equality"],
        "profile_selected_by_ordinal": True,
    }


def _fixed_operations(recorder: Recorder, config: dict[str, Any]) -> None:
    recorder.measure("l1-fixed-setup", lambda: configure_runtime())
    recorder.measure(
        "l1-fixed-split",
        lambda: (
            cal.frozen_identity_split(32),
            cal.frozen_identity_split(64),
            cal.width256_heldout_identity_stream(),
        ),
    )
    pair = recorder.measure(
        "l1-fixed-model", lambda: cal.frozen_matched_pair(config, repeat=0)
    )
    recorder.measure(
        "l1-fixed-optimizer-control", lambda: cal.optimizer_and_scheduler(pair[0])
    )
    recorder.measure(
        "l1-fixed-optimizer-challenger",
        lambda: cal.optimizer_and_scheduler(pair[1]),
    )
    recorder.measure(
        "l1-fixed-tracker", lambda: {"backend": "json-plus-exported-mlflow"}
    )


def derive_l2_budget(
    records: Sequence[Mapping[str, Any]], cache: cal.CacheVerdict
) -> dict[str, Any]:
    rates: dict[
        tuple[str, int | None, str, str], list[tuple[float, float]]
    ] = defaultdict(list)
    maximum_rss = 0
    for record in records:
        physical = record["physical_unit"]
        derived = record["derived"]
        maximum_rss = max(maximum_rss, int(derived["peak_rss_bytes"]))
        if physical["eligible_for_rate"]:
            key = (
                physical["rate_key"],
                physical["width"],
                physical["arm"],
                physical["source"],
            )
            rates[key].append(
                (
                    derived["wall_seconds_per_unit"],
                    derived["aggregate_cpu_seconds_per_unit"],
                )
            )
    expected_wall = 0.0
    expected_cpu = 0.0
    selected: dict[str, dict[str, float]] = {}
    for physical in cal.l2_physical_manifest(cache):
        key = (physical.rate_key, physical.width, physical.arm, physical.source)
        if key not in rates:
            raise ExecutionError(f"missing exact L2 rate coverage: {key}")
        wall = max(row[0] for row in rates[key])
        cpu = max(row[1] for row in rates[key])
        expected_wall += wall * physical.units
        expected_cpu += cpu * physical.units
        selected[cal.coverage_class_id(cal.CoverageClass(*key))] = {
            "maximum_wall_seconds_per_unit": wall,
            "maximum_aggregate_cpu_seconds_per_unit": cpu,
        }
    result = {
        "expected": {
            "wall_seconds": expected_wall,
            "aggregate_cpu_seconds": expected_cpu,
        },
        "soft": {
            "wall_seconds": 1.25 * expected_wall,
            "aggregate_cpu_seconds": 1.25 * expected_cpu,
        },
        "hard": {
            "wall_seconds": 1.75 * expected_wall,
            "aggregate_cpu_seconds": 1.75 * expected_cpu,
        },
        "maximum_observed_rss_bytes": maximum_rss,
        "rss_guard_bytes": math.ceil(1.25 * maximum_rss),
        "maximum_matching_rates": selected,
        "headroom": {"soft": 1.25, "hard": 1.75, "rss": 1.25},
        "wall_cpu_distinct": True,
        "fidelity_reduced": False,
        "fixed_short_wall_cap": None,
    }
    result["derivation_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def _l2_training_batches() -> tuple[list[cal.ImmutableBatch], str]:
    splits = {width: cal.frozen_identity_split(width) for width in (32, 64)}
    rngs = {
        width: __import__("random").Random(cal.derive_seed("train", "L2", width))
        for width in (32, 64)
    }
    used: set[tuple[int, ...]] = set()
    batches: list[cal.ImmutableBatch] = []
    identities: list[tuple[int, ...]] = []
    for quota in cal.training_schedule(cycles=25):
        pool: list[cal.StratumCandidate] = []
        attempted: set[tuple[int, ...]] = set()
        while len(pool) < ATTEMPT_ALLOWANCE:
            raw = cal._draw_source_transition(rngs[quota.width], splits[quota.width].train)
            if raw in attempted or raw in used:
                continue
            attempted.add(raw)
            pool.append(
                cal.StratumCandidate.from_transition(
                    raw, exact_data.LegalTransition(*raw), quota.width
                )
            )
        selection = cal.select_training_strata(
            pool, quota, attempt_allowance=ATTEMPT_ALLOWANCE
        )
        selected = [
            identity
            for stratum in cal.STRATA
            for identity in selection.selected[stratum]
        ]
        if len(selected) != 128 or any(identity in used for identity in selected):
            raise ExecutionError("L2 training identity ownership failed")
        used.update(selected)
        identities.extend(selected)
        rows = [
            cal._owned_transition(
                tuple(identity), width=quota.width, repeat=4, arm="shared"
            )
            for identity in selected
        ]
        batches.append(make_batch(rows))
    return batches, hashlib.sha256(canonical_bytes(identities)).hexdigest()


def _l2_rare_rows(
    *, width: int, kind: str
) -> tuple[dict[int, list[cal.OwnedCase]], str]:
    split = cal.frozen_identity_split(width)
    namespace = "rare_q" if kind == "quotient" else "rare_carry"
    rng = __import__("random").Random(cal.derive_seed(namespace, "L2", width))
    classes = (0, 1, 2) if kind == "quotient" else (-2, -1, 0, 1)
    rows: dict[int, list[cal.OwnedCase]] = {label: [] for label in classes}
    seen: set[tuple[int, ...]] = set()
    while any(len(values) < 2000 for values in rows.values()):
        raw = cal._draw_source_transition(rng, split.test)
        trace = exact_data.quotient_carry_trace(
            exact_data.LegalTransition(*raw), width
        )
        if kind == "quotient":
            label = trace.quotient
            identity = raw
            if len(rows[label]) >= 2000 or identity in seen:
                continue
            seen.add(identity)
            rows[label].append(
                cal._owned_transition(
                    raw, width=width, repeat=4, arm="challenger", label=label
                )
            )
        else:
            for position, label in enumerate(trace.carry_out):
                identity = (*raw, position)
                if len(rows[label]) >= 2000 or identity in seen:
                    continue
                seen.add(identity)
                rows[label].append(
                    cal._owned_transition(
                        raw,
                        width=width,
                        repeat=4,
                        arm="challenger",
                        label=label,
                        position=position,
                    )
                )
    payload = {
        str(label): [row.identity for row in values]
        for label, values in rows.items()
    }
    return rows, hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _rare_futility(
    control: torch.nn.Module,
    challenger: torch.nn.Module,
    *,
    width: int,
    kind: str,
) -> dict[str, Any]:
    rows_by_class, identity_sha256 = _l2_rare_rows(width=width, kind=kind)
    class_map = cal.SIGNED_CARRY_CLASS_MAP if kind == "carry" else None
    result: dict[str, Any] = {}
    for label, rows in rows_by_class.items():
        errors = {"control": 0, "challenger": 0}
        seen = 0
        for start in range(0, 2000, 256):
            batch_rows = rows[start : start + 256]
            batch = make_batch(batch_rows)
            with torch.inference_mode():
                outputs = {
                    "control": control(batch.features),
                    "challenger": challenger(batch.features),
                }
            for arm, output in outputs.items():
                if kind == "quotient":
                    predicted = output.quotient_logits.argmax(dim=1)
                    truth = torch.full_like(predicted, int(label))
                else:
                    positions = torch.tensor(
                        [row.position for row in batch_rows], dtype=torch.long
                    )
                    selected = output.signed_carry_logits[
                        torch.arange(len(batch_rows)), :, positions
                    ]
                    predicted = selected.argmax(dim=1)
                    truth = torch.full_like(predicted, class_map[int(label)])
                errors[arm] += int((predicted != truth).sum().item())
            seen += len(batch_rows)
            if errors["challenger"] >= 3:
                result[str(label)] = {
                    "evaluated": seen,
                    "frozen_denominator": 2000,
                    "errors": errors,
                    "decision_safe_stop": "third-challenger-error",
                    "pass": False,
                }
                return {
                    "kind": kind,
                    "width": width,
                    "identity_sha256": identity_sha256,
                    "classes": result,
                    "binding_pass": False,
                }
        result[str(label)] = {
            "evaluated": 2000,
            "frozen_denominator": 2000,
            "errors": errors,
            "decision_safe_stop": None,
            "pass": errors["challenger"] <= 2,
        }
    return {
        "kind": kind,
        "width": width,
        "identity_sha256": identity_sha256,
        "classes": result,
        "binding_pass": all(value["pass"] for value in result.values()),
    }


def run_l2(
    output: Path,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    start = cal.cumulative_resource_meter_snapshot()
    config = cal.load_config()
    control, challenger, co, ho, cs, hs = _build_repeat(config, 4)
    batches, training_identity_sha256 = _l2_training_batches()
    for global_step, batch in enumerate(batches, 1):
        rng = torch.get_rng_state().clone()
        order = (
            ("control", "challenger")
            if (4 + global_step) % 2 == 0
            else ("challenger", "control")
        )
        for arm in order:
            if arm == "control":
                _single_arm_update(
                    control, co, cs, batch, auxiliary_weight=0.0, rng=rng
                )
            else:
                _single_arm_update(
                    challenger, ho, hs, batch, auxiliary_weight=0.10, rng=rng
                )
    checkpoint_dir = output / "l2-checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes = {}
    for arm, model in (("control", control), ("challenger", challenger)):
        path = checkpoint_dir / f"{arm}.safetensors"
        save_file(model.state_dict(), str(path))
        checkpoint_hashes[arm] = sha256_path(path)
        reloaded = model_module.JointQuotientCarryTransducer(
            hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]
        )
        reloaded.load_state_dict(load_file(str(path)), strict=True)
        if any(
            not torch.equal(value, reloaded.state_dict()[name])
            for name, value in model.state_dict().items()
        ):
            raise ExecutionError("L2 checkpoint reload mismatch")
    after_training = cal.cumulative_resource_meter_snapshot()
    budget = calibration["resources"]["budget"]
    training_resources = {
        "wall_seconds": after_training.monotonic_wall_seconds
        - start.monotonic_wall_seconds,
        "aggregate_cpu_seconds": (
            after_training.process_user_plus_system_aggregate_cpu_seconds
            - start.process_user_plus_system_aggregate_cpu_seconds
        ),
        "peak_rss_bytes": after_training.simultaneous_tree_rss_bytes,
    }
    resource_failed = cal.decision_safe_stop(
        at_preserved_checkpoint=True,
        wall_seconds=training_resources["wall_seconds"],
        aggregate_cpu_seconds=training_resources["aggregate_cpu_seconds"],
        peak_rss_bytes=training_resources["peak_rss_bytes"],
        hard_wall_seconds=budget["hard"]["wall_seconds"],
        hard_aggregate_cpu_seconds=budget["hard"]["aggregate_cpu_seconds"],
        rss_guard_bytes=budget["rss_guard_bytes"],
    )
    diagnostics = []
    stop_reason = "resource-hard-ceiling" if resource_failed else None
    if not resource_failed:
        for width in (32, 64):
            for kind in ("quotient", "carry"):
                diagnostic = _rare_futility(
                    control, challenger, width=width, kind=kind
                )
                diagnostics.append(diagnostic)
                if not diagnostic["binding_pass"]:
                    stop_reason = f"{kind}-class-third-error-width{width}"
                    break
            if stop_reason:
                break
    if stop_reason is None:
        raise ExecutionError(
            "rare-class gates unexpectedly passed; remaining frozen L2 gates must run"
        )
    end = cal.cumulative_resource_meter_snapshot()
    result = {
        "stage": "L2",
        "decision": "reject",
        "reason": stop_reason,
        "updates_completed_per_arm": 400,
        "batch_size": 128,
        "training_identity_sha256": training_identity_sha256,
        "checkpoint_sha256": checkpoint_hashes,
        "rare_diagnostics": diagnostics,
        "resources": {
            "wall_seconds": end.monotonic_wall_seconds - start.monotonic_wall_seconds,
            "aggregate_cpu_seconds": (
                end.process_user_plus_system_aggregate_cpu_seconds
                - start.process_user_plus_system_aggregate_cpu_seconds
            ),
            "peak_rss_bytes": end.simultaneous_tree_rss_bytes,
            "calibrated_hard": budget["hard"],
            "calibrated_rss_guard_bytes": budget["rss_guard_bytes"],
        },
        "decision_safe_stop": True,
        "unexecuted_denominators": (
            "authorized by preregistered exact third-error or resource stop"
        ),
    }
    result["l2_result_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    write_canonical(output / "l2-result.json", result)
    return result


def run_l1(authorization_path: Path, output: Path) -> dict[str, Any]:
    authorization = load_authorization(authorization_path)
    environment = configure_runtime()
    config = cal.load_config()
    cache = cal.validate_incumbent_cache(config, root=ROOT)
    manifest = cal.canonical_l1_identity_manifest(ATTEMPT_ALLOWANCE)
    identity = cal.validate_l1_identity_manifest(
        manifest, deterministic_attempt_allowance=ATTEMPT_ALLOWANCE
    )
    physical = (
        *cal.warmup_l1_physical_manifest(),
        *cal.remaining_l1_physical_manifest(cache),
    )
    recorder = Recorder(physical)
    _fixed_operations(recorder, config)
    final_models = None
    final_normal = None
    for repeat in (0, 1, 2, 3):
        models = _train_repeat(recorder, manifest, config, repeat)
        normal = _evaluate_repeat_resources(
            recorder, manifest, models, repeat
        )
        if repeat == 3:
            final_models, final_normal = models, normal
    assert final_models is not None and final_normal is not None
    fresh_profile_walls, integrity = _post_repeat3(
        recorder, manifest, final_models, final_normal, output
    )
    records = recorder.finalize()
    records_dir = output / "l1-resource-records"
    descriptors = []
    for record in records:
        key = record["physical_unit"]["key"]
        path = records_dir / f"{key}.json"
        descriptors.append(
            {
                "physical_key": key,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "file_sha256": write_canonical(path, record),
                "record_sha256": record["record_sha256"],
            }
        )
    budget = derive_l2_budget(records, cache)
    gate = cal.runtime_gate(156.0 if cache.valid else None, fresh_profile_walls)
    result = {
        "schema_version": CALIBRATION_SCHEMA,
        "authorization_sha256": authorization["authorization_sha256"],
        "source_commit": authorization["source_commit"],
        "environment": environment,
        "identity": identity,
        "record_descriptors": descriptors,
        "record_count": len(records),
        "physical_units": sum(
            row["physical_unit"]["units"] for row in records
        ),
        "integrity": integrity,
        "resources": {
            "budget": budget,
            "fresh_width256_eight_case_wall_seconds": fresh_profile_walls,
            "runtime_gate": gate,
            "wall_meter": "time.perf_counter",
            "aggregate_cpu_meter": "cgroup-tree-user-plus-system",
            "rss_meter": "cgroup-memory.peak",
        },
        "boundary": {
            "scientific_values_present": False,
            "warmup_excluded_from_rate_denominators": True,
            "full_fidelity": True,
        },
    }
    result["calibration_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    write_canonical(output / "l1-calibration.json", result)
    return result


def persist_decision(
    output: Path, calibration: Mapping[str, Any]
) -> dict[str, Any]:
    gate = calibration["resources"]["runtime_gate"]
    if gate["decision"] == "pass":
        l2_result = run_l2(output, calibration)
        decision = l2_result["decision"]
        reason = "L2:" + l2_result["reason"]
        stage = "T045-L2"
        next_decision = "Select the next materially distinct experiment family"
    else:
        l2_result = None
        decision = "park"
        reason = "binding-deployed-runtime-admission-failure:" + gate["reason"]
        stage = "T045-L1-admission"
        next_decision = "Select a materially different high-information experiment family"
    payload = {
        "schema_version": DECISION_SCHEMA,
        "run_id": cal.RUN_ID,
        "stage": stage,
        "decision": decision,
        "reason": reason,
        "calibration_sha256": calibration["calibration_sha256"],
        "source_commit": calibration["source_commit"],
        "runtime_gate": gate,
        "dynamic_budget": calibration["resources"]["budget"],
        "l2_result_sha256": (
            None if l2_result is None else l2_result["l2_result_sha256"]
        ),
        "limitations": [
            "L1 contains no scientific accuracy, recall, prediction, or causal values.",
            "The park decision does not falsify the quotient/carry hypothesis.",
            "No width-256, checked, independent, private, candidate, or goal evidence was produced.",
        ],
        "next_material_decision": next_decision,
    }
    payload["decision_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    write_canonical(output / "decision.json", payload)
    write_canonical(EVAL_ROOT / "decision.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-authorization")
    prepare.add_argument("--path", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--authorization", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare-authorization":
        prepare_authorization(arguments.path.resolve())
        return 0
    calibration = run_l1(
        arguments.authorization.resolve(), arguments.output.resolve()
    )
    decision = persist_decision(arguments.output.resolve(), calibration)
    print(
        json.dumps(
            {"decision": decision["decision"], "reason": decision["reason"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
