from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.exp_658_joint_quotient_carry_transducer import calibration as cal
from experiments.exp_658_joint_quotient_carry_transducer import execution as exe


def _snap(wall: float, cpu: float, rss: int) -> cal.MeterSnapshot:
    return cal.MeterSnapshot(wall, cpu, rss)


def test_fixed_recorder_is_closed_and_separates_wall_cpu(
    monkeypatch, tmp_path: Path
) -> None:
    physical = cal.warmup_l1_physical_manifest()[0]
    snapshots = iter((_snap(1.0, 2.0, 100), _snap(3.0, 7.0, 200)))
    monkeypatch.setattr(
        cal, "cumulative_resource_meter_snapshot", lambda: next(snapshots)
    )
    recorder = exe.Recorder((physical,))
    assert recorder.measure(physical.key, lambda: 17) == 17
    record = recorder.finalize()[0]
    assert record["operation_id"] == "exp658-fixed-" + physical.key
    assert record["callback_surface"] is False
    assert record["derived"] == {
        "monotonic_wall_seconds": 2.0,
        "aggregate_cpu_seconds": 5.0,
        "peak_rss_bytes": 200,
        "wall_seconds_per_unit": 2.0,
        "aggregate_cpu_seconds_per_unit": 5.0,
    }
    path = tmp_path / "record.json"
    assert exe.write_canonical(path, record) == exe.sha256_path(path)
    assert path.read_bytes() == exe.canonical_bytes(json.loads(path.read_bytes()))
    with pytest.raises(exe.ExecutionError, match="absent from canonical"):
        recorder.measure("caller-injected-operation", lambda: None)


def test_batch_and_rollout_use_learned_path() -> None:
    row = cal.OwnedCase(
        identity=(1, 2, 5, 1),
        source_identity=(1, 2, 5, 1),
        width=4,
        modulus=5,
        repeat=0,
        arm="shared",
        state=1,
        multiplicand=2,
        operand_bit=1,
    )
    batch = exe.make_batch((row,))
    assert tuple(batch.features.shape) == (1, 6, 4)
    assert tuple(batch.carry_targets.shape) == (1, 4)
    model = cal.frozen_model.JointQuotientCarryTransducer(hidden=4, cycles=1)
    rollout = cal.OwnedCase(
        identity=(3, 2, 5),
        source_identity=(3, 2, 5),
        width=4,
        modulus=5,
        repeat=0,
        arm="shared",
        multiplier=3,
        multiplicand=2,
    )
    result = exe._rollout(model, (rollout,))
    assert len(result) == 1 and 0 <= result[0] < 16


def test_dynamic_budget_uses_maximum_matching_rates() -> None:
    cache = cal.validate_incumbent_cache(cal.load_config())
    observed = (
        *cal.warmup_l1_physical_manifest(),
        *cal.remaining_l1_physical_manifest(cache),
    )
    records = []
    for index, physical in enumerate(observed):
        rate = 2.0 if index == 0 else 1.0
        records.append(
            {
                "physical_unit": asdict(physical),
                "derived": {
                    "wall_seconds_per_unit": rate,
                    "aggregate_cpu_seconds_per_unit": rate * 3,
                    "peak_rss_bytes": 400,
                },
            }
        )
    budget = exe.derive_l2_budget(records, cache)
    assert budget["soft"]["wall_seconds"] == 1.25 * budget["expected"][
        "wall_seconds"
    ]
    assert budget["hard"]["aggregate_cpu_seconds"] == 1.75 * budget[
        "expected"
    ]["aggregate_cpu_seconds"]
    assert budget["rss_guard_bytes"] == 500
    assert budget["fixed_short_wall_cap"] is None
    assert budget["fidelity_reduced"] is False


def test_authorization_hash_rejects_co_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "authorization.json"
    payload = {
        "schema_version": exe.AUTHORIZATION_SCHEMA,
        "source_commit": "f" * 40,
        "executable_hashes": {},
    }
    payload["authorization_sha256"] = exe.hashlib.sha256(
        exe.canonical_bytes(payload)
    ).hexdigest()
    exe.write_canonical(path, payload)
    monkeypatch.setattr(exe, "_git", lambda *args: "f" * 40)
    monkeypatch.setattr(exe, "executable_hashes", lambda: {})
    assert exe.load_authorization(path)["source_commit"] == "f" * 40
    changed = json.loads(path.read_text())
    changed["source_commit"] = "e" * 40
    path.write_bytes(exe.canonical_bytes(changed))
    with pytest.raises(exe.ExecutionError, match="hash/schema"):
        exe.load_authorization(path)
