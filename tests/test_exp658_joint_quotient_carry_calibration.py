"""Adversarial inert-contract tests for the repaired EXP-658 L1 scaffold."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from sympy import __version__ as sympy_version
from sympy import isprime, nextprime

from experiments.exp_658_joint_quotient_carry_transducer import calibration as cal
from experiments.exp_658_joint_quotient_carry_transducer import data as exp658_data
from experiments.exp_658_joint_quotient_carry_transducer import model as exp658_model


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/exp_658_joint_quotient_carry_transducer"
ATTEMPT_ALLOWANCE = 512


def _reviewed_registry(
    manifest: tuple[cal.PhysicalUnit, ...],
) -> dict[str, cal.CanonicalOperationRegistration]:
    return {
        f"reviewed-operation:{physical.key}": cal.CanonicalOperationRegistration(
            operation_id=f"reviewed-operation:{physical.key}",
            physical_key=physical.key,
            rate_key=physical.rate_key,
            width=physical.width,
            arm=physical.arm,
            source=physical.source,
            units=physical.units,
        )
        for physical in manifest
    }


def _expected_seed(namespace: str, *parts: object) -> int:
    suffix = "|".join(str(part) for part in parts)
    material = f"exp658|20261176|{namespace}|{suffix}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _independent_split(width: int) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    rng = random.Random(_expected_seed("split", width))
    low = 1 << (width - 8)
    high = 1 << width
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < 640:
        prime = int(nextprime(rng.randrange(low, high)))
        if prime < high and prime not in seen:
            values.append(prime)
            seen.add(prime)
    payload = {
        "algorithm_sha256": cal.EXP300_SPLIT_SHA256,
        "width": width,
        "minimum_bits": width - 7,
        "maximum_bits": width,
        "split_seed": _expected_seed("split", width),
        "train": values[:512],
        "test": values[512:],
    }
    digest = cal.sha256_bytes(cal.canonical_json_bytes(payload))
    return tuple(values[:512]), tuple(values[512:]), digest


def _independent_width256_stream() -> tuple[
    tuple[int, ...], tuple[int, ...], dict[int, tuple[int, ...]], str
]:
    assert sympy_version == "1.13.1"
    material = (
        "exp658|20261176|profile_warmup|"
        "heldout-width256-prime-stream|256"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    rng = random.Random(seed)
    low, high = 1 << 248, 1 << 256
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < 28:
        prime = int(nextprime(rng.randrange(low, high)))
        if prime < high and prime not in seen:
            values.append(prime)
            seen.add(prime)
    moduli = tuple(values)
    warmup = moduli[:4]
    fresh = {
        repeat: moduli[4 + (repeat - 1) * 8 : 4 + repeat * 8]
        for repeat in (1, 2, 3)
    }
    payload = {
        "width": 256,
        "minimum_bits": 249,
        "maximum_bits": 256,
        "stream_seed": seed,
        "count": len(moduli),
        "moduli": moduli,
        "warmup": warmup,
        "fresh": {str(key): value for key, value in fresh.items()},
    }
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    return moduli, warmup, fresh, hashlib.sha256(raw).hexdigest()


def _tiny_batch(seed: int, *, width: int = 6, rows: int = 2) -> cal.ImmutableBatch:
    generator = torch.Generator().manual_seed(seed)
    return cal.immutable_batch(
        torch.rand((rows, 6, width), generator=generator),
        torch.randint(0, 2, (rows, width), generator=generator).float(),
        torch.randint(0, 3, (rows,), generator=generator),
        torch.randint(0, 4, (rows, width), generator=generator),
    )


def _stratum_candidates() -> tuple[cal.StratumCandidate, ...]:
    candidates: list[cal.StratumCandidate] = []
    for modulus in (251, 241, 239, 233):
        for state in range(modulus):
            for multiplicand in range(modulus):
                for operand_bit in (0, 1):
                    transition = exp658_data.LegalTransition(
                        state, multiplicand, modulus, operand_bit
                    )
                    identity = (state, multiplicand, modulus, operand_bit)
                    candidates.append(
                        cal.StratumCandidate.from_transition(identity, transition, 8)
                    )
                if len(candidates) >= 70_000:
                    return tuple(candidates)
    return tuple(candidates)


def _build_identity_manifest() -> tuple[
    cal.L1IdentityManifest,
    dict[int, cal.IdentitySplit],
    cal.Width256IdentityStream,
]:
    splits = {width: cal.frozen_identity_split(width) for width in (32, 64)}
    heldout = cal.width256_heldout_identity_stream()
    return cal.canonical_l1_identity_manifest(ATTEMPT_ALLOWANCE), splits, heldout


def _provenance() -> cal.MeterProvenance:
    return cal.canonical_meter_provenance()


def _observation(
    physical: cal.PhysicalUnit,
    *,
    wall_per_unit: float = 2.0,
    cpu_per_unit: float = 3.0,
    rss: int = 200,
) -> cal.ResourceObservation:
    wall = wall_per_unit * physical.units
    return cal.resource_observation(
        physical,
        cal.MeterSnapshot(10.0, 20.0, 100),
        cal.MeterSnapshot(10.0 + wall, 20.0 + cpu_per_unit * physical.units, rss),
        provenance=_provenance(),
        inner_monotonic_wall_seconds=(
            wall * 0.8 if physical.phase == "L1-profile-fresh" else None
        ),
    )


def _live_resource_environment() -> dict:
    return {
        "torch_version": "2.5.1+cpu",
        "device": "cpu",
        "default_dtype": "float32",
        "deterministic_algorithms": True,
        "intra_op_threads": 4,
        "inter_op_threads": 1,
        "affinity_cpus": (0, 1, 2, 3),
        "cpu_limit": 4.0,
        "memory_limit_bytes": 8 * 1024**3,
        "tmpfs_limit_bytes": 2 * 1024**3,
        "meter_implementation": cal.meter_implementation_binding(),
    }


def _write_canonical_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cal.canonical_json_bytes(payload))


def _install_exact_measurement_evidence(
    config: dict, tmp_path: Path, monkeypatch
) -> tuple[dict, Path, dict[str, cal.CanonicalOperationRegistration]]:
    configured = copy.deepcopy(config)
    for record in configured["L1"]["budget"][
        "comparable_source_provenance"
    ].values():
        for key in ("identity_path", "rate_path"):
            source = ROOT / record[key]
            target = tmp_path / record[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)

    monkeypatch.setattr(cal, "live_resource_environment", _live_resource_environment)
    cache = cal.validate_incumbent_cache(config)
    warmup = cal.warmup_l1_physical_manifest()
    remaining = cal.remaining_l1_physical_manifest(cache)
    exact_units = cal.canonical_unresolved_measurement_manifest(cache)
    authorized_manifest = (*warmup, *exact_units)
    registry = _reviewed_registry((*warmup, *remaining))
    environment = cal.canonical_authorized_environment(configured)
    environment_sha256 = cal.sha256_bytes(cal.canonical_json_bytes(environment))
    prior = cal._prior_measurement_authorization_mechanics(
        configured,
        cache,
        registry=registry,
    )
    assert prior["physical_manifest"] == cal._physical_manifest_payload(
        authorized_manifest
    )
    descriptor_by_physical = {
        item["physical_key"]: item for item in prior["operation_descriptors"]
    }
    prior_path = Path("evidence/prior-measurement-authorization.json")
    _write_canonical_json(tmp_path / prior_path, prior)
    prior_sha256 = cal.frozen_run.sha256(tmp_path / prior_path)
    measurement_config = configured["L1"]["budget"][
        "prior_phase_measurement_evidence"
    ]
    measurement_config["authorization_path"] = prior_path.as_posix()
    measurement_config["authorization_sha256"] = prior_sha256

    warmup_descriptors: list[dict] = []
    exact_descriptors: list[dict] = []
    tick = 0
    for physical in authorized_manifest:
        tick += 1
        wall = float(physical.units + tick / 1000)
        cpu = float(physical.units * 2 + tick / 1000)
        record = cal._measurement_record_from_snapshots_mechanics(
            physical,
            cal.MeterSnapshot(1000.0 + tick, 2000.0 + tick, 1000 + tick),
            cal.MeterSnapshot(
                1000.0 + tick + wall,
                2000.0 + tick + cpu,
                2000 + tick,
            ),
            operation_descriptor=descriptor_by_physical[physical.key],
            environment_sha256=environment_sha256,
            prior_authorization_sha256=cal.sha256_bytes(
                cal.canonical_json_bytes(prior)
            ),
            inner_monotonic_wall_seconds=(
                wall * 0.5 if physical.phase == "L1-profile-fresh" else None
            ),
        )
        record_path = Path("evidence/records") / f"{physical.key}.json"
        _write_canonical_json(tmp_path / record_path, record)
        descriptor = {
            "physical_key": physical.key,
            "rate_key": physical.rate_key,
            "record_path": record_path.as_posix(),
            "record_file_sha256": cal.frozen_run.sha256(tmp_path / record_path),
            "record_sha256": record["record_sha256"],
        }
        if physical in warmup:
            warmup_descriptors.append(descriptor)
        else:
            exact_descriptors.append(
                descriptor | {"experiment": "TEMP-EXACT-MECHANICS"}
            )
    measurement_config["warmup_records"] = warmup_descriptors
    configured["L1"]["budget"]["exact_component_rate_evidence"] = (
        exact_descriptors
    )
    return configured, tmp_path, registry


def _fake_unused_proof(root_id: int, output: str, **_kwargs) -> dict:
    evidence = {
        "root": root_id,
        "output": str(Path(output)),
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
        "evidence_sha256": cal.sha256_bytes(cal.canonical_json_bytes(evidence))
    }


def test_zero_root_config_frozen_hashes_and_path_backed_cache() -> None:
    config = cal.load_config()
    assert config["run_control"] == {
        "active_stage": "L0",
        "calibration_root": None,
        "scientific_roots": [],
        "authorization_file": None,
        "mlflow": "prohibited",
        "evidence_registration": "prohibited",
        "registry_edit": "prohibited",
        "governance_edit": "prohibited",
        "submission_edit": "prohibited",
    }
    assert config["L1"]["scaffold"]["execution"] == "prohibited"
    assert config["L1"]["apportionment"]["deterministic_attempt_allowance"] == (
        "required-numeric-future-authorization-input-no-default"
    )
    profile = config["L1"]["deployed_path_profile"]
    assert profile["profiled_model"] == "exp658-selected-reloaded-challenger-only"
    assert profile["candidate005_role"].endswith(
        "pinned-full1100-incumbent-measurement-only"
    )
    assert not {"routes", "microprofile_routes", "legacy_L0_routes_field_interpretation"}.intersection(
        profile
    )
    assert cal.verify_frozen_l0_hashes() == {
        name: expected for name, (_path, expected) in cal.FROZEN_L0_HASHES.items()
    }
    hashes = cal.scaffold_hashes()
    assert len(hashes) == 20
    assert hashes["source_checkpoint"] == cal.SOURCE_CHECKPOINT_SHA256
    assert hashes["exp300_split_source"] == cal.EXP300_SPLIT_SHA256
    verdict = cal.validate_incumbent_cache(config)
    assert verdict.valid and verdict.evidence_validated
    assert not verdict.full_1100_required
    assert verdict.incumbent_wall_seconds == 156.0
    assert verdict.incremental_allowance_seconds == 129.0
    assert verdict.reason == "exact-path-backed-cache-match"
    assert len(verdict.evidence_sha256) == 64
    width256 = cal.width256_heldout_identity_stream()
    assert width256.sha256 == cal.EXPECTED_WIDTH256_STREAM_SHA256
    assert len(width256.warmup) == 4
    assert {key: len(value) for key, value in width256.fresh_by_repeat.items()} == {
        1: 8,
        2: 8,
        3: 8,
    }


def test_config_and_preregistration_have_no_legacy_route_or_value_equality_contract() -> None:
    config_text = cal.CONFIG_PATH.read_text(encoding="utf-8")
    preregistration = cal.PREREGISTRATION_PATH.read_text(encoding="utf-8")
    for stale in (
        "routes:",
        "microprofile_routes:",
        "legacy_L0_routes_field_interpretation:",
        "parameter_equality: exact",
    ):
        assert stale not in config_text
    assert "state-dictionary keys, shapes, values, initialization" not in preregistration
    assert "Exact values are required only at initialization" in preregistration
    assert "values are expected to diverge under auxiliary supervision" in preregistration
    assert "Candidate 005 is never substituted with a microprofile" in preregistration


@pytest.mark.parametrize("width", [32, 64])
def test_frozen_split_exactly_matches_independent_sympy_reproduction(width: int) -> None:
    train, test, digest = _independent_split(width)
    split = cal.frozen_identity_split(width)
    assert split.train == train
    assert split.test == test
    assert split.sha256 == digest == cal.EXPECTED_SPLIT_SHA256[width]
    assert len(train) == 512 and len(test) == 128
    assert len(set((*train, *test))) == 640
    assert not set(train).intersection(test)
    assert all(exp658_data.is_prime(value) for value in (*train, *test))
    assert all(width - 7 <= value.bit_length() <= width for value in (*train, *test))
    assert cal.derive_seed("split", width) == _expected_seed("split", width)


def test_seed_streams_apportionment_and_exact_25_cycle_extension() -> None:
    for namespace, parts in (
        ("init", (3,)),
        ("train", (2, 64)),
        ("eval_transition", (1, 32)),
        ("eval_rollout", (1, 64)),
        ("rare_q", (32,)),
        ("rare_carry", (64,)),
        ("profile_warmup", (256,)),
        ("profile_fresh", (3, 256)),
    ):
        assert cal.derive_seed(namespace, *parts) == _expected_seed(namespace, *parts)
    manifest = cal.stream_seed_manifest()
    assert len(manifest) == len(set(manifest.values())) == 37
    width256 = cal.width256_heldout_identity_stream()
    independent_moduli, independent_warmup, independent_fresh, independent_hash = (
        _independent_width256_stream()
    )
    assert width256.moduli == independent_moduli
    assert width256.warmup == independent_warmup
    assert width256.fresh_by_repeat == independent_fresh
    assert width256.sha256 == independent_hash == cal.EXPECTED_WIDTH256_STREAM_SHA256
    assert width256.seed == _expected_seed(
        "profile_warmup", "heldout-width256-prime-stream", 256
    )
    assert all(
        value > 0
        and 249 <= value.bit_length() <= 256
        and isprime(value)
        for value in width256.moduli
    )
    l1 = cal.training_schedule()
    assert len(l1) == 16
    assert tuple(row.width for row in l1) == (32, 64) * 8
    assert cal.apportionment_totals(l1) == {
        "natural": 1024,
        "q2": 410,
        "boundary": 307,
        "rare_carry": 307,
        "total": 2048,
    }
    l2 = cal.training_schedule(25)
    assert len(l2) == 400
    assert sum(row.width == 32 for row in l2) == 200
    assert sum(row.width == 64 for row in l2) == 200
    assert cal.apportionment_totals(l2) == {
        key: value * 25 for key, value in cal.apportionment_totals(l1).items()
    }


def test_raw_stratum_predicates_precedence_fallback_and_tamper_rejection() -> None:
    candidates = _stratum_candidates()
    natural = list(candidates[:64])
    used = {item.identity for item in natural}

    def take(predicate, count: int) -> list[cal.StratumCandidate]:
        selected = []
        for item in candidates[64:]:
            if item.identity not in used and predicate(item):
                selected.append(item)
                used.add(item.identity)
                if len(selected) == count:
                    return selected
        raise AssertionError("insufficient exact raw candidates")

    q2 = take(lambda item: item.q2_eligible, 26)
    boundary = take(
        lambda item: item.boundary_eligible and not item.q2_eligible, 19
    )
    rare = take(
        lambda item: item.rare_carry_eligible
        and not item.q2_eligible
        and not item.boundary_eligible,
        19,
    )
    quota = cal.training_schedule()[0]
    selection = cal.select_training_strata(
        (*natural, *q2, *boundary, *rare), quota, attempt_allowance=128
    )
    assert {name: len(rows) for name, rows in selection.selected.items()} == {
        "natural": 64,
        "q2": 26,
        "boundary": 19,
        "rare_carry": 19,
    }
    assert len(set().union(*(set(rows) for rows in selection.selected.values()))) == 128
    assert selection.fallback_natural == 0

    near = cal.StratumCandidate.from_transition(
        (125, 0, 251, 0), exp658_data.LegalTransition(125, 0, 251, 0), 8
    )
    outside = cal.StratumCandidate.from_transition(
        (119, 0, 251, 0), exp658_data.LegalTransition(119, 0, 251, 0), 8
    )
    assert near.raw_u == 250 and near.boundary_eligible
    assert outside.raw_u == 238 and not outside.boundary_eligible
    q2_case = cal.StratumCandidate.from_transition(
        (200, 200, 251, 1), exp658_data.LegalTransition(200, 200, 251, 1), 8
    )
    assert q2_case.raw_u == 600 and q2_case.quotient_label == 2 and q2_case.q2_eligible
    assert cal.SIGNED_CARRY_CLASS_MAP == {-2: 0, -1: 1, 0: 2, 1: 3}

    with pytest.raises(cal.ContractError, match="raw u"):
        replace(near, raw_u=near.raw_u + 1).validate_raw_labels()
    with pytest.raises(cal.ContractError, match="quotient"):
        replace(near, quotient_label=2).validate_raw_labels()
    tampered_carries = list(near.signed_carry_labels)
    tampered_carries[0] = 1 if tampered_carries[0] != 1 else 0
    with pytest.raises(cal.ContractError, match="signed-carry"):
        replace(near, signed_carry_labels=tuple(tampered_carries)).validate_raw_labels()
    with pytest.raises(cal.ContractError, match="duplicate"):
        cal.select_training_strata(
            (*natural, *q2, *boundary, *rare, natural[0]),
            quota,
            attempt_allowance=129,
        )

    fallback_pool = tuple(candidates[:128])
    fallback = cal.select_training_strata(
        fallback_pool, quota, attempt_allowance=128
    )
    assert sum(len(rows) for rows in fallback.selected.values()) == 128
    assert fallback.fallback_natural >= 0
    with pytest.raises(cal.ContractError, match="no fallback"):
        cal.select_rare_without_fallback(("a", "a", "b"), count=3)


def test_complete_stream_ownership_and_all_reuse_contracts() -> None:
    manifest, splits, heldout = _build_identity_manifest()
    binding = cal.validate_l1_identity_manifest(
        manifest, deterministic_attempt_allowance=ATTEMPT_ALLOWANCE
    )
    assert binding["split_sha256_by_width"] == {
        "32": cal.EXPECTED_SPLIT_SHA256[32],
        "64": cal.EXPECTED_SPLIT_SHA256[64],
    }
    assert len(binding["stream_manifest_sha256"]) == 64
    assert binding["width256_stream_sha256"] == cal.EXPECTED_WIDTH256_STREAM_SHA256
    assert binding["checked_or_private_answers_read"] is False
    assert binding["rare_owner"] == "fresh-repeat-3-challenger"
    assert binding["profile_owner"] == "exact-reloaded-challenger"

    duplicate_training = list(manifest.training)
    duplicate_training[1] = replace(
        duplicate_training[1],
        identity=duplicate_training[0].identity,
        source_identity=duplicate_training[0].source_identity,
        modulus=duplicate_training[0].modulus,
    )
    with pytest.raises(cal.ContractError, match="complete repeat|prohibited suffix"):
        cal.validate_l1_identity_manifest(
            replace(manifest, training=tuple(duplicate_training)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )

    wrong_owner = list(manifest.transitions)
    wrong_owner_source = (0, 0, splits[32].train[0], 0)
    wrong_owner[0] = replace(
        wrong_owner[0],
        identity=wrong_owner_source,
        source_identity=wrong_owner_source,
        modulus=splits[32].train[0],
        state=0,
        multiplicand=0,
        operand_bit=0,
    )
    with pytest.raises(cal.ContractError, match="non-test"):
        cal.validate_l1_identity_manifest(
            replace(manifest, transitions=tuple(wrong_owner)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )

    unpaired = list(manifest.zero_quotient_rollouts)
    unpaired[0] = replace(
        unpaired[0],
        identity=unpaired[1].identity,
        source_identity=unpaired[1].source_identity,
        modulus=unpaired[1].modulus,
    )
    with pytest.raises(cal.ContractError, match="not paired|prohibited suffix"):
        cal.validate_l1_identity_manifest(
            replace(manifest, zero_quotient_rollouts=tuple(unpaired)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )

    arbitrary = list(manifest.rare_quotient)
    arbitrary[0] = replace(arbitrary[0], label=(arbitrary[0].label + 1) % 3)
    with pytest.raises(cal.ContractError, match="does not match raw"):
        cal.validate_l1_identity_manifest(
            replace(manifest, rare_quotient=tuple(arbitrary)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )

    suffixed_q = list(manifest.rare_quotient)
    suffixed_q[0] = replace(
        suffixed_q[0], identity=(*suffixed_q[0].identity, 999)
    )
    with pytest.raises(cal.ContractError, match="prohibited suffix"):
        cal.validate_l1_identity_manifest(
            replace(manifest, rare_quotient=tuple(suffixed_q)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )
    suffixed_carry = list(manifest.rare_carry)
    suffixed_carry[0] = replace(
        suffixed_carry[0],
        source_identity=(*suffixed_carry[0].source_identity, 777),
    )
    with pytest.raises(cal.ContractError, match="prohibited suffix"):
        cal.validate_l1_identity_manifest(
            replace(manifest, rare_carry=tuple(suffixed_carry)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )

    rare_overlap = list(manifest.rare_carry)
    target_label = rare_overlap[0].label
    q_source = next(
        row
        for row in manifest.rare_quotient
        if target_label
        in exp658_data.quotient_carry_trace(
            exp658_data.LegalTransition(
                row.state, row.multiplicand, row.modulus, row.operand_bit
            ),
            row.width,
        ).carry_out
    )
    trace = exp658_data.quotient_carry_trace(
        exp658_data.LegalTransition(
            q_source.state,
            q_source.multiplicand,
            q_source.modulus,
            q_source.operand_bit,
        ),
        q_source.width,
    )
    position = trace.carry_out.index(target_label)
    rare_overlap[0] = replace(
        rare_overlap[0],
        source_identity=q_source.source_identity,
        identity=(*q_source.source_identity, position),
        modulus=q_source.modulus,
        state=q_source.state,
        multiplicand=q_source.multiplicand,
        operand_bit=q_source.operand_bit,
        position=position,
    )
    with pytest.raises(cal.ContractError, match="diagnostic sources overlap"):
        cal.validate_l1_identity_manifest(
            replace(manifest, rare_carry=tuple(rare_overlap)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )

    duplicate_positions = list(manifest.rare_carry)
    duplicate_index = next(
        index
        for index, row in enumerate(duplicate_positions[1:], start=1)
        if row.label == duplicate_positions[0].label
    )
    duplicate_positions[duplicate_index] = replace(
        duplicate_positions[duplicate_index],
        identity=duplicate_positions[0].identity,
        source_identity=duplicate_positions[0].source_identity,
        modulus=duplicate_positions[0].modulus,
        label=duplicate_positions[0].label,
        state=duplicate_positions[0].state,
        multiplicand=duplicate_positions[0].multiplicand,
        operand_bit=duplicate_positions[0].operand_bit,
        position=duplicate_positions[0].position,
    )
    with pytest.raises(cal.ContractError, match="not unique|duplicate"):
        cal.validate_l1_identity_manifest(
            replace(manifest, rare_carry=tuple(duplicate_positions)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )

    determinism = list(manifest.determinism_repeat)
    determinism[0] = replace(
        determinism[0],
        source_identity=determinism[1].source_identity,
        identity=determinism[1].identity,
        modulus=determinism[1].modulus,
    )
    with pytest.raises(cal.ContractError, match="does not reuse|prohibited suffix"):
        cal.validate_l1_identity_manifest(
            replace(manifest, determinism_repeat=tuple(determinism)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )

    profile = list(manifest.profile_fresh)
    profile[0] = replace(
        profile[0],
        identity=manifest.profile_warmup[0].identity,
        source_identity=manifest.profile_warmup[0].source_identity,
        modulus=manifest.profile_warmup[0].modulus,
    )
    with pytest.raises(
        cal.ContractError,
        match="stream order|profile cases overlap|prohibited suffix",
    ):
        cal.validate_l1_identity_manifest(
            replace(manifest, profile_fresh=tuple(profile)),
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )
    different_valid_stream = cal.canonical_l1_identity_manifest(
        ATTEMPT_ALLOWANCE + 1
    )
    with pytest.raises(cal.ContractError, match="seed-to-case regeneration"):
        cal.validate_l1_identity_manifest(
            different_valid_stream,
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )


def test_exact_loss_and_source_resolved_adamw_defaults() -> None:
    residue_logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0]])
    quotient_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    carry_logits = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 10
    raw_carry = ((-2, -1, 0, 1), (1, 0, -1, -2))
    carry_targets = torch.tensor(
        [[cal.SIGNED_CARRY_CLASS_MAP[value] for value in row] for row in raw_carry]
    )
    batch = cal.immutable_batch(
        torch.zeros((2, 6, 2)),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([2, 0]),
        carry_targets,
    )
    output = exp658_model.TransducerOutput(residue_logits, quotient_logits, carry_logits)
    terms = cal.exact_loss_terms(output, batch, quotient_weight=0.10, carry_weight=0.10)
    per_bit = F.binary_cross_entropy_with_logits(
        residue_logits, batch.residue_targets, reduction="none"
    )
    expected_residue = per_bit.mean() + 0.45 * per_bit.mean(dim=0).max()
    expected_q = F.cross_entropy(quotient_logits, batch.quotient_targets)
    expected_carry = F.cross_entropy(carry_logits, carry_targets)
    assert torch.equal(terms.residue, expected_residue)
    assert torch.equal(terms.quotient, expected_q)
    assert torch.equal(terms.carry, expected_carry)
    assert torch.equal(
        terms.total, expected_residue + 0.10 * expected_q + 0.10 * expected_carry
    )

    model = exp658_model.JointQuotientCarryTransducer(hidden=8, cycles=1)
    optimizer, scheduler = cal.optimizer_and_scheduler(model)
    cal.validate_optimizer_contract(model, optimizer, scheduler)
    group = optimizer.param_groups[0]
    assert {
        key: group[key] for key in cal.ADAMW_DEFAULTS
    } == cal.ADAMW_DEFAULTS
    assert optimizer.defaults["lr"] == group["initial_lr"] == scheduler.base_lrs[0] == 5e-6
    assert scheduler.T_max == 400 and scheduler.eta_min == 5e-7
    optimizer.param_groups[0]["betas"] = (0.8, 0.999)
    with pytest.raises(cal.ContractError, match="defaults"):
        cal.validate_optimizer_contract(model, optimizer, scheduler)
    for target in ("initial", "base", "current"):
        optimizer, scheduler = cal.optimizer_and_scheduler(model)
        if target == "initial":
            optimizer.param_groups[0]["initial_lr"] = 7e-6
        elif target == "base":
            scheduler.base_lrs[0] = 7e-6
        else:
            optimizer.param_groups[0]["lr"] = 7e-6
            scheduler._last_lr[0] = 7e-6
        with pytest.raises(cal.ContractError, match="learning rate"):
            cal.validate_optimizer_contract(model, optimizer, scheduler)


def test_deterministic_cpu_preflight_passes_only_exact_pinned_runtime() -> None:
    code = """
import json
import torch
from experiments.exp_658_joint_quotient_carry_transducer import calibration as c
from experiments.exp_658_joint_quotient_carry_transducer import model as m
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
print(json.dumps(c.deterministic_cpu_preflight(m.JointQuotientCarryTransducer(hidden=4, cycles=1), amp_enabled=False), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    identity = json.loads(completed.stdout)
    assert identity == {
        "amp_enabled": False,
        "deterministic_algorithms": True,
        "device": "cpu",
        "inter_op_threads": 1,
        "intra_op_threads": 4,
        "precision": "float32",
        "torch_version": "2.5.1+cpu",
    }
    with pytest.raises(cal.ContractError, match="runtime identity"):
        cal.deterministic_cpu_preflight(
            exp658_model.JointQuotientCarryTransducer(hidden=4, cycles=1),
            amp_enabled=True,
        )


def _paired_fixture(repeat: int = 1):
    class StochasticTransducer(exp658_model.JointQuotientCarryTransducer):
        def forward(self, features: torch.Tensor, *, message_ablation: str = "none"):
            return super().forward(
                features + torch.rand_like(features) * 1e-4,
                message_ablation=message_ablation,
            )

    factory = lambda: StochasticTransducer(hidden=4, cycles=1)
    control, challenger = cal.independently_initialized_pair(factory, repeat=repeat)
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


def _paired_step(fixture, *, repeat: int, step: int) -> cal.PairedUpdateAudit:
    return cal.paired_update(
        *fixture,
        _tiny_batch(65800 + step, width=4),
        repeat=repeat,
        global_step=step,
    )


def test_two_step_pairing_allows_only_initial_value_equality() -> None:
    fixture = _paired_fixture(repeat=1)
    first = _paired_step(fixture, repeat=1, step=1)
    assert first.entry_value_equality_required
    assert first.arm_order == ("control", "challenger")
    assert first.optimizer_step_counts == {"control": (1,), "challenger": (1,)}
    assert first.scheduler_last_epochs == {"control": 1, "challenger": 1}
    assert first.auxiliary_connected_difference
    assert first.rng_after_forward_equal
    assert first.forward_results_independent
    assert first.structure_equal and first.finite and first.finite_gradients

    second = _paired_step(fixture, repeat=1, step=2)
    assert not second.entry_value_equality_required
    assert second.arm_order == ("challenger", "control")
    assert second.optimizer_step_counts == {"control": (2,), "challenger": (2,)}
    assert second.scheduler_last_epochs == {"control": 2, "challenger": 2}
    assert second.auxiliary_connected_difference
    assert second.structure_equal and second.finite and second.finite_gradients

    broken = _paired_fixture(repeat=2)
    _paired_step(broken, repeat=2, step=1)
    broken[1].load_state_dict(broken[0].state_dict(), strict=True)
    with pytest.raises(cal.ContractError, match="divergence"):
        _paired_step(broken, repeat=2, step=2)


def test_lightweight_full_16_step_pairing_preserves_all_invariants() -> None:
    fixture = _paired_fixture(repeat=3)
    audits = [
        _paired_step(fixture, repeat=3, step=step) for step in range(1, 17)
    ]
    assert [audit.arm_order for audit in audits] == [
        ("control", "challenger") if (3 + step) % 2 == 0 else ("challenger", "control")
        for step in range(1, 17)
    ]
    assert audits[-1].optimizer_step_counts == {
        "control": (16,),
        "challenger": (16,),
    }
    assert audits[-1].scheduler_last_epochs == {"control": 16, "challenger": 16}
    assert all(
        audit.batch_unchanged
        and audit.structure_equal
        and audit.finite
        and audit.finite_gradients
        and audit.auxiliary_connected_difference
        and audit.rng_after_forward_equal
        and audit.forward_results_independent
        for audit in audits
    )
    expected_lr = 5e-7 + (5e-6 - 5e-7) * (1 + math.cos(16 * math.pi / 400)) / 2
    assert audits[-1].learning_rates == pytest.approx(
        {"control": expected_lr, "challenger": expected_lr}
    )


def test_exact_reloaded_challenger_handle_and_rollout_contract(tmp_path: Path) -> None:
    control = exp658_model.JointQuotientCarryTransducer(hidden=4, cycles=1)
    challenger = exp658_model.JointQuotientCarryTransducer(hidden=4, cycles=1)
    challenger.load_state_dict(control.state_dict())
    probe = torch.zeros((2, 6, 4))
    control_result = cal.checkpoint_roundtrip(
        control,
        tmp_path / "control.safetensors",
        probe,
        repeat=3,
        arm="control",
        ordinal=1,
    )
    challenger_result = cal.checkpoint_roundtrip(
        challenger,
        tmp_path / "challenger.safetensors",
        probe,
        repeat=3,
        arm="challenger",
        ordinal=2,
    )
    selected = cal.exact_ordinal_profile_challenger(control_result, challenger_result)
    assert selected is challenger_result.reloaded_model
    assert selected is not challenger
    assert challenger_result.receipt["tensor_equality"] is True
    assert challenger_result.receipt["fixed_probe_equality"] is True
    assert challenger_result.receipt["repeat"] == 3
    assert challenger_result.receipt["arm"] == "challenger"
    assert challenger_result.receipt["ordinal"] == 2
    with pytest.raises(cal.ContractError, match="repeat 3"):
        cal.checkpoint_roundtrip(
            exp658_model.JointQuotientCarryTransducer(hidden=4, cycles=1),
            tmp_path / "wrong.safetensors",
            probe,
            repeat=2,
            arm="challenger",
            ordinal=2,
        )
    assert cal.rollout_bit_positions(8) == tuple(range(7, -1, -1))
    assert cal.decode_little_endian_for_next_feature((True, False, True), 5) == (
        5,
        True,
    )


def test_strict_physical_manifests_and_cache_verdict_control_projection() -> None:
    config = cal.load_config()
    valid = cal.validate_incumbent_cache(config)
    warmup = cal.warmup_l1_physical_manifest()
    remaining = cal.remaining_l1_physical_manifest(valid)
    l2 = cal.l2_physical_manifest(valid)
    assert (len(warmup), len(remaining), len(l2)) == (28, 86, 45)
    assert len({item.key for item in (*warmup, *remaining, *l2)}) == len(
        (*warmup, *remaining, *l2)
    )
    assert not any(
        token in item.key
        for item in (*warmup, *remaining, *l2)
        for token in ("intervention_baseline", "determinism_pass1")
    )
    assert cal.l2_projection_units(valid) == {
        "training_shared_generation_batch": 400,
        "training_arm_update": 800,
        "transition_shared_generation_case": 40_000,
        "transition_arm_forward_case": 80_000,
        "rollout_shared_generation_case": 2_000,
        "rollout_arm_forward_case": 4_000,
        "q_shared_generation_case": 12_000,
        "q_arm_forward_case": 24_000,
        "carry_shared_generation_item": 16_000,
        "carry_arm_forward_item": 32_000,
        "intervention_extra_rollout_case": 2_000,
        "determinism_extra_rollout_case": 200,
        "checkpoint_cycle": 2,
        "fixed_setup": 1,
        "fixed_split": 1,
        "fixed_model": 1,
        "fixed_optimizer_control": 1,
        "fixed_optimizer_challenger": 1,
        "fixed_tracker": 1,
        "width256_profile_case": 28,
    }
    changed = copy.deepcopy(config)
    changed["L1"]["deployed_path_profile"]["cached_candidate005"][
        "path_backed_artifacts"
    ]["submission_weights"]["sha256"] = "0" * 64
    invalid = cal.validate_incumbent_cache(changed)
    assert not invalid.valid and invalid.full_1100_required
    assert len(cal.remaining_l1_physical_manifest(invalid)) == 87
    assert len(cal.l2_physical_manifest(invalid)) == 46
    assert cal.l2_projection_units(invalid)["incumbent_full_1100_case"] == 1100
    with pytest.raises(cal.ContractError, match="validated cache verdict"):
        cal.l2_projection_units(True)  # type: ignore[arg-type]


def test_resource_provenance_dynamic_budget_and_adversarial_rejections() -> None:
    config = cal.load_config()
    cache = cal.validate_incumbent_cache(config)
    observed_manifest = (
        *cal.warmup_l1_physical_manifest(),
        *cal.remaining_l1_physical_manifest(cache),
    )
    projection = cal.l2_physical_manifest(cache)
    observations = []
    for physical in observed_manifest:
        wall_rate = 2.0 if physical.key == "l1-r1-w32-transition-generate" else 1.0
        cpu_rate = 5.0 if physical.key == "l1-r2-w32-transition-generate" else 3.0
        observations.append(
            _observation(
                physical,
                wall_per_unit=wall_rate,
                cpu_per_unit=cpu_rate,
                rss=200,
            )
        )
    observations = tuple(observations)
    budget = cal._resolve_dynamic_budget_from_observations(
        observations,
        observed_manifest,
        projection,
        projection_stage="L2",
        cache_verdict=cache,
    )
    transition_w32 = cal.coverage_class_id(
        cal.CoverageClass(
            "transition_shared_generation_case", 32, "shared", "shared-generator"
        )
    )
    transition_w64 = cal.coverage_class_id(
        cal.CoverageClass(
            "transition_shared_generation_case", 64, "shared", "shared-generator"
        )
    )
    assert budget["maximum_matching_mean_rates"]["monotonic_wall_seconds"][
        transition_w32
    ] == 2.0
    assert budget["maximum_matching_mean_rates"][
        "process_user_plus_system_aggregate_cpu_seconds"
    ][transition_w32] == 5.0
    assert budget["maximum_matching_mean_rates"]["monotonic_wall_seconds"][
        transition_w64
    ] == 1.0
    assert cal.EXPLICIT_COVERAGE_EQUIVALENCES == {}
    assert budget["soft_budget"]["monotonic_wall_seconds"] == 1.25 * budget[
        "expected"
    ]["monotonic_wall_seconds"]
    assert budget["hard_runaway_ceiling"][
        "process_user_plus_system_aggregate_cpu_seconds"
    ] == 1.75 * budget["expected"][
        "process_user_plus_system_aggregate_cpu_seconds"
    ]
    assert budget["peak_simultaneous_tree_rss_guard_bytes"] == 250
    assert budget["wall_cpu_conflated"] is False
    assert budget["summed_serial_peak_rss"] is False
    assert budget["fixed_short_wall_cap_used"] is False
    assert budget["fidelity_reduced"] is False

    with pytest.raises(cal.ContractError, match="provenance binding"):
        cal.validate_resource_observations(
            (replace(observations[0], source="control-model"), observations[1]),
            observed_manifest[:2],
        )
    with pytest.raises(cal.ContractError, match="non-positive"):
        cal.validate_resource_observations(
            (replace(observations[0], monotonic_wall_seconds=float("nan")), observations[1]),
            observed_manifest[:2],
        )
    with pytest.raises(cal.ContractError, match="meter implementation"):
        cal.validate_resource_observations(
            (
                replace(
                    observations[0],
                    provenance=replace(
                        observations[0].provenance, peak_rss_meter="summed-serial-rss"
                    ),
                ),
                observations[1],
            ),
            observed_manifest[:2],
        )
    with pytest.raises(cal.ContractError, match="exactly canonical"):
        cal._resolve_dynamic_budget_from_observations(
            observations,
            observed_manifest,
            projection[:-1],
            projection_stage="L2",
            cache_verdict=cache,
        )

    width32_only_manifest = tuple(
        item for item in observed_manifest if item.width != 64
    )
    width32_only = tuple(
        item for item in observations if item.physical_key in {
            physical.key for physical in width32_only_manifest
        }
    )
    with pytest.raises(cal.ContractError, match=r"coverage: .*width=64"):
        cal._resolve_dynamic_budget_from_observations(
            width32_only,
            width32_only_manifest,
            projection,
            projection_stage="L2",
            cache_verdict=cache,
            require_canonical_stage=False,
        )

    challenger_only_manifest = tuple(
        item for item in observed_manifest if item.arm != "control"
    )
    challenger_only = tuple(
        item for item in observations if item.physical_key in {
            physical.key for physical in challenger_only_manifest
        }
    )
    with pytest.raises(cal.ContractError, match=r"coverage: .*arm=control"):
        cal._resolve_dynamic_budget_from_observations(
            challenger_only,
            challenger_only_manifest,
            projection,
            projection_stage="L2",
            cache_verdict=cache,
            require_canonical_stage=False,
        )

    profile = cal.PhysicalUnit(
        "l1-profile-fresh-r1",
        "width256_profile_case",
        "L1-profile-fresh",
        1,
        "challenger-model",
        8,
        256,
        "challenger",
    )
    profile_observation = _observation(profile)
    with pytest.raises(cal.ContractError, match="inner/outer"):
        cal.validate_resource_observations(
            (replace(profile_observation, inner_monotonic_wall_seconds=1000.0),),
            (profile,),
        )
    with pytest.raises(cal.ContractError, match="overlap"):
        cal._validate_physical_manifest((projection[0], projection[0]))


def test_production_measurement_has_closed_empty_registry_and_no_callback_surface(
    monkeypatch,
) -> None:
    config = cal.load_config()
    monkeypatch.setattr(cal, "live_resource_environment", _live_resource_environment)
    physical = cal.warmup_l1_physical_manifest()[0]
    environment = cal.canonical_authorized_environment(config)
    environment_sha256 = cal.sha256_bytes(cal.canonical_json_bytes(environment))
    manifest_payload = cal._physical_manifest_payload((physical,))
    input_manifest_sha256 = cal.sha256_bytes(
        cal.canonical_json_bytes(manifest_payload)
    )
    registry = _reviewed_registry((physical,))
    descriptor = cal._derive_operation_descriptors(
        (physical,),
        registry=registry,
        input_manifest_sha256=input_manifest_sha256,
        environment_sha256=environment_sha256,
    )[0]
    assert dict(cal.CANONICAL_OPERATION_REGISTRY) == {}
    assert tuple(inspect.signature(cal.authorized_measurement_record).parameters) == (
        "physical_key",
        "operation_id",
        "prior_authorization_path",
    )

    def forbidden_meter() -> cal.MeterSnapshot:
        raise AssertionError("closed registry reached the meter")

    monkeypatch.setattr(cal, "cumulative_resource_meter_snapshot", forbidden_meter)
    with pytest.raises(cal.AuthorizationError, match="empty/unregistered"):
        cal.authorized_measurement_record(
            physical.key,
            descriptor["operation_id"],
            prior_authorization_path=None,
        )

    mutable_dispatch = {"target": lambda: None}

    def global_dispatch() -> None:
        mutable_dispatch["target"]()

    captured = lambda: None

    def closure_dispatch() -> None:
        captured()

    for substituted in (global_dispatch, closure_dispatch, lambda: None):
        with pytest.raises(TypeError, match="unexpected keyword argument 'operation'"):
            cal.authorized_measurement_record(
                physical.key,
                descriptor["operation_id"],
                prior_authorization_path=None,
                operation=substituted,  # type: ignore[call-arg]
            )

    record = cal._measurement_record_from_snapshots_mechanics(
        physical,
        cal.MeterSnapshot(10.0, 20.0, 100),
        cal.MeterSnapshot(12.0, 23.0, 200),
        operation_descriptor=descriptor,
        environment_sha256=environment_sha256,
        prior_authorization_sha256="f" * 64,
    )
    observation = cal._validate_measurement_record_payload(
        record,
        environment_sha256=environment_sha256,
        prior_authorization_sha256="f" * 64,
        authorized_operation_descriptors={physical.key: descriptor},
    )
    assert observation.monotonic_wall_seconds == 2.0
    assert observation.process_user_plus_system_aggregate_cpu_seconds == 3.0
    mutated = copy.deepcopy(record)
    mutated["observation"]["monotonic_wall_seconds"] = 9.0
    unhashed = dict(mutated)
    unhashed.pop("record_sha256")
    mutated["record_sha256"] = cal.sha256_bytes(cal.canonical_json_bytes(unhashed))
    with pytest.raises(cal.ContractError, match="derived observation"):
        cal._validate_measurement_record_payload(
            mutated,
            environment_sha256=environment_sha256,
            prior_authorization_sha256="f" * 64,
            authorized_operation_descriptors={physical.key: descriptor},
        )
    caller_authored = copy.deepcopy(descriptor)
    caller_authored["operation_id"] = "caller-authored-operation"
    with pytest.raises(cal.ContractError, match="caller-authored"):
        cal._validate_operation_descriptors(
            [caller_authored],
            (physical,),
            registry=registry,
            environment_sha256=environment_sha256,
            expected_input_manifest_sha256=input_manifest_sha256,
        )

    cache = cal.validate_incumbent_cache(config)
    full_registry = _reviewed_registry(
        (*cal.warmup_l1_physical_manifest(), *cal.remaining_l1_physical_manifest(cache))
    )
    canonical_prior = cal._prior_measurement_authorization_mechanics(
        config,
        cache,
        registry=full_registry,
    )
    caller_manifest = copy.deepcopy(canonical_prior)
    caller_manifest["physical_manifest"].pop()
    caller_manifest["operation_descriptors"].pop()
    caller_manifest["physical_manifest_sha256"] = cal.sha256_bytes(
        cal.canonical_json_bytes(caller_manifest["physical_manifest"])
    )
    caller_manifest["input_manifest_sha256"] = caller_manifest[
        "physical_manifest_sha256"
    ]
    for item in caller_manifest["operation_descriptors"]:
        item["input_manifest_sha256"] = caller_manifest["input_manifest_sha256"]
    caller_manifest["operation_descriptors_sha256"] = cal.sha256_bytes(
        cal.canonical_json_bytes(caller_manifest["operation_descriptors"])
    )
    with pytest.raises(cal.ContractError, match="prior measurement authorization"):
        cal._validate_prior_measurement_authorization(
            caller_manifest,
            config=config,
            cache_verdict=cache,
            registry=full_registry,
            environment_sha256=environment_sha256,
        )


def test_canonical_remaining_l1_resolver_requires_exact_one_to_one_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    config = cal.load_config()
    cache = cal.validate_incumbent_cache(config)
    rates = cal._load_comparable_resource_rates_mechanics(
        config,
        cache,
        registry=cal.CANONICAL_OPERATION_REGISTRY,
        evidence_root=ROOT,
    )
    assert rates == {}
    missing = cal.missing_exact_remaining_l1_rate_keys(cache)
    assert missing == (
        "carry_arm_forward_item",
        "carry_shared_generation_item",
        "checkpoint_cycle",
        "determinism_extra_rollout_case",
        "q_arm_forward_case",
        "q_shared_generation_case",
        "width256_profile_case",
    )
    with pytest.raises(cal.ContractError) as blocked:
        cal.resolve_remaining_l1_budget()
    assert all(rate_key in str(blocked.value) for rate_key in missing)
    monkeypatch.setattr(cal, "live_resource_environment", _live_resource_environment)
    with pytest.raises(cal.ContractError) as authorization_blocked:
        cal.authorization_expectation(
            scaffold_commit="f" * 40,
            root=987654321,
            output="reserved/exp658/root-987654321",
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )
    assert all(rate_key in str(authorization_blocked.value) for rate_key in missing)

    exact_config, evidence_root, registry = _install_exact_measurement_evidence(
        config, tmp_path, monkeypatch
    )
    rates = cal._load_comparable_resource_rates_mechanics(
        exact_config,
        cache,
        registry=registry,
        evidence_root=evidence_root,
    )
    expected_coverage = {
        cal.physical_coverage_class(item)
        for item in cal.canonical_unresolved_measurement_manifest(cache)
    }
    assert set(rates) == expected_coverage
    flattened_rates = [item for rows in rates.values() for item in rows]
    assert len({rate.record_sha256 for rate in flattened_rates}) == len(
        cal.canonical_unresolved_measurement_manifest(cache)
    )
    assert all(
        rate.comparability
        == "exact-width-arm-source-prior-phase-authorized-component"
        for rate in flattened_rates
    )
    derivation = cal._resolve_remaining_l1_budget_mechanics(
        exact_config,
        cache,
        registry=registry,
        evidence_root=evidence_root,
    )
    assert derivation["stage"] == "remaining-L1-after-warmup"
    assert derivation["projection_manifest_sha256"] == cal.sha256_bytes(
        cal.canonical_json_bytes(
            cal._physical_manifest_payload(cal.remaining_l1_physical_manifest(cache))
        )
    )
    assert set(derivation["selected_rates"]) == {
        cal.coverage_class_id(cal.physical_coverage_class(item))
        for item in cal.remaining_l1_physical_manifest(cache)
    }
    q_control = cal.coverage_class_id(
        cal.CoverageClass("q_arm_forward_case", 32, "control", "control-model")
    )
    q_challenger = cal.coverage_class_id(
        cal.CoverageClass(
            "q_arm_forward_case", 32, "challenger", "challenger-model"
        )
    )
    assert derivation["selected_rates"][q_control]["source"] == (
        "validated-exact-maximum-matching-mean"
    )
    assert q_challenger in derivation["selected_rates"]
    envelope = derivation["numeric_resource_envelope"]
    assert envelope["soft_wall_seconds"] == 1.25 * envelope["expected_wall_seconds"]
    assert envelope["hard_aggregate_cpu_seconds"] == 1.75 * envelope[
        "expected_aggregate_cpu_seconds"
    ]
    assert envelope["rss_guard_bytes"] == math.ceil(
        1.25 * envelope["maximum_observed_rss_bytes"]
    )
    changed = copy.deepcopy(exact_config)
    changed["L1"]["budget"]["exact_component_rate_evidence"][0][
        "record_file_sha256"
    ] = "0" * 64
    with pytest.raises(cal.ContractError, match="hash mismatch"):
        cal._resolve_remaining_l1_budget_mechanics(
            changed,
            cache,
            registry=registry,
            evidence_root=evidence_root,
        )

    invalid_config = copy.deepcopy(exact_config)
    invalid_config["L1"]["deployed_path_profile"]["cached_candidate005"][
        "path_backed_artifacts"
    ]["submission_weights"]["sha256"] = "0" * 64
    invalid = cal.validate_incumbent_cache(invalid_config)
    with pytest.raises(cal.ContractError, match="incumbent_full_1100_case"):
        cal._resolve_remaining_l1_budget_mechanics(
            invalid_config,
            invalid,
            registry=registry,
            evidence_root=evidence_root,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torch_version", "2.5.0+cpu"),
        ("default_dtype", "float64"),
        ("deterministic_algorithms", False),
        ("intra_op_threads", 3),
        ("inter_op_threads", 2),
        ("affinity_cpus", (0, 1, 2)),
        ("affinity_cpus", (0, 1, 2, 3, 4)),
        ("cpu_limit", 3.0),
        ("memory_limit_bytes", 7 * 1024**3),
        ("tmpfs_limit_bytes", 1024**3),
    ],
)
def test_live_environment_rejects_resource_mutation(
    monkeypatch, field: str, value: object
) -> None:
    live = _live_resource_environment()
    live[field] = value
    monkeypatch.setattr(cal, "live_resource_environment", lambda: live)
    with pytest.raises(cal.ContractError, match="four-CPU/resource/meter"):
        cal.canonical_authorized_environment(cal.load_config())


def _copy_cache_artifacts(config: dict, tmp_path: Path) -> dict:
    copied = copy.deepcopy(config)
    artifacts = copied["L1"]["deployed_path_profile"]["cached_candidate005"][
        "path_backed_artifacts"
    ]
    for record in artifacts.values():
        source = ROOT / record["path"]
        target = tmp_path / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return copied


def test_candidate_cache_parsing_all_paths_pinned_command_and_timing_validity(
    tmp_path: Path,
) -> None:
    config = cal.load_config()
    artifact_names = tuple(
        config["L1"]["deployed_path_profile"]["cached_candidate005"][
            "path_backed_artifacts"
        ]
    )
    for name in artifact_names:
        changed = copy.deepcopy(config)
        changed["L1"]["deployed_path_profile"]["cached_candidate005"][
            "path_backed_artifacts"
        ][name]["sha256"] = "0" * 64
        verdict = cal.validate_incumbent_cache(changed)
        assert not verdict.valid and verdict.full_1100_required
        assert verdict.reason == f"{name}-hash-mismatch"

    copied = _copy_cache_artifacts(config, tmp_path)
    cached = copied["L1"]["deployed_path_profile"]["cached_candidate005"]
    sandbox_record = tmp_path / cached["path_backed_artifacts"]["sandbox_record"]["path"]
    sandbox = json.loads(sandbox_record.read_text(encoding="utf-8"))
    sandbox["total"] = 1099
    sandbox_record.write_text(json.dumps(sandbox, indent=2) + "\n", encoding="utf-8")
    cached["path_backed_artifacts"]["sandbox_record"]["sha256"] = cal.frozen_run.sha256(
        sandbox_record
    )
    assert cal.validate_incumbent_cache(copied, root=tmp_path).reason == (
        "sandbox-record-content-mismatch"
    )

    command = cal.candidate005_full_1100_rerun_command(
        Path("reserved/candidate005-full1100")
    )
    assert "--image" in command
    assert command[command.index("--image") + 1] == cal.PINNED_CANDIDATE_IMAGE
    assert command[command.index("--total") + 1] == "1100"
    assert command[command.index("--seed") + 1] == (
        "1c82674f2826d26d7c5de0bf1933457ddc9ce824c30acff4a8a37578b3a566e4"
    )
    assert cal.runtime_gate(156.0, (8.0, 8.0, 8.0))["decision"] == "pass"
    assert cal.runtime_gate(156.0, (11.0, 11.0, 11.0))["decision"] == "fail"
    assert cal.runtime_gate(156.0, (8.5, 8.5, 8.5))["decision"] == (
        "straddle-requires-exact-100-case-profile"
    )
    assert cal.runtime_gate(
        156.0, (8.5, 8.5, 8.5), exact_100_case_wall_seconds=100.0
    )["decision"] == "pass"
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            cal.runtime_gate(156.0, (8.5, bad, 8.5))
        with pytest.raises(ValueError, match="positive and finite"):
            cal.runtime_gate(
                156.0,
                (8.5, 8.5, 8.5),
                exact_100_case_wall_seconds=bad,
            )


def _authorization_fixture(monkeypatch, tmp_path: Path):
    base_config = cal.load_config()
    config, evidence_root, registry = _install_exact_measurement_evidence(
        base_config, tmp_path, monkeypatch
    )
    cache = cal.validate_incumbent_cache(config)
    monkeypatch.setattr(cal, "independently_verify_unused_root_output", _fake_unused_proof)
    commit = "f" * 40
    expected = cal._authorization_expectation_mechanics(
        config=config,
        scaffold_commit=commit,
        root=987654321,
        output="reserved/exp658/root-987654321",
        deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        registry=registry,
        evidence_root=evidence_root,
    )
    return config, cache, evidence_root, registry, commit, expected


def test_closed_authorization_and_independent_lifecycle_rejections(
    tmp_path: Path, monkeypatch
) -> None:
    real_unused_verifier = cal.independently_verify_unused_root_output
    config, cache, evidence_root, registry, commit, expected = _authorization_fixture(
        monkeypatch, tmp_path
    )
    assert expected["deterministic_attempt_allowance"] == ATTEMPT_ALLOWANCE
    assert expected["split_sha256_by_width"] == {
        "32": cal.EXPECTED_SPLIT_SHA256[32],
        "64": cal.EXPECTED_SPLIT_SHA256[64],
    }
    assert expected["width256_stream_sha256"] == cal.EXPECTED_WIDTH256_STREAM_SHA256
    assert expected["cache_verdict"]["evidence_sha256"] == cache.evidence_sha256
    assert expected["numeric_resource_envelope"] == expected["budget_derivation"][
        "numeric_resource_envelope"
    ]
    assert set(expected["hashes"]) == set(cal.scaffold_hashes())
    assert set(expected["comparable_resource_provenance"]) == {"EXP609", "EXP643"}
    assert expected["environment_sha256"] == cal.sha256_bytes(
        cal.canonical_json_bytes(expected["environment"])
    )
    assert expected["authorized_operation_manifest"] == cal._physical_manifest_payload(
        cal.remaining_l1_physical_manifest(cache)
    )
    assert {
        item["input_manifest_sha256"]
        for item in expected["authorized_operation_descriptors"]
    } == {expected["workload_sha256"]}

    manifest = tmp_path / "authorization.json"
    manifest.write_bytes(cal.canonical_json_bytes(expected))
    monkeypatch.setattr(
        cal,
        "git_read_only_state",
        lambda: {"head": commit, "upstream": commit, "dirty": False},
    )
    def validate_mechanics(path: Path) -> dict:
        return cal._validate_authorization_candidate_mechanics(
            path,
            config=config,
            registry=registry,
            evidence_root=evidence_root,
            repository_root=ROOT,
        )

    assert validate_mechanics(manifest) == expected
    probe = cal.SideEffectProbe()
    with pytest.raises(cal.AuthorizationError, match="fixed repository root"):
        cal.inert_entrypoint(manifest, side_effects=probe)
    assert probe.untouched()

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(cal.AuthorizationError, match="malformed"):
        validate_mechanics(malformed)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    with pytest.raises(cal.AuthorizationError, match="canonical"):
        validate_mechanics(noncanonical)
    changed = copy.deepcopy(expected)
    changed["environment"]["accuracy"] = 0.99
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_bytes(cal.canonical_json_bytes(changed))
    with pytest.raises(cal.AuthorizationError, match="canonical recomputation"):
        validate_mechanics(mismatch)
    for path in (
        ("workload", "apportionment", "l1_totals", "total"),
        (
            "budget_derivation",
            "selected_rates",
            next(iter(expected["budget_derivation"]["selected_rates"])),
            "source",
        ),
        (
            "environment",
            "current_live_probe",
            "meter_implementation",
            "cpu_counter_cumulative",
        ),
    ):
        nested = copy.deepcopy(expected)
        target = nested
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = "mutated" if not isinstance(target[path[-1]], bool) else False
        nested_path = tmp_path / ("nested-" + "-".join(path) + ".json")
        nested_path.write_bytes(cal.canonical_json_bytes(nested))
        with pytest.raises(cal.AuthorizationError, match="canonical recomputation"):
            validate_mechanics(nested_path)
    assert probe.untouched()

    co_mutated = copy.deepcopy(expected)
    co_mutated["workload"]["apportionment"]["l1_totals"]["total"] = 9999
    co_mutated["workload_sha256"] = cal.sha256_bytes(
        cal.canonical_json_bytes(co_mutated["workload"])
    )
    first_selected_rate = next(
        iter(co_mutated["budget_derivation"]["selected_rates"])
    )
    co_mutated["budget_derivation"]["selected_rates"][first_selected_rate][
        "source"
    ] = "CO-MUTATED"
    unhashed_derivation = dict(co_mutated["budget_derivation"])
    unhashed_derivation.pop("derivation_sha256")
    co_mutated["budget_derivation"]["derivation_sha256"] = cal.sha256_bytes(
        cal.canonical_json_bytes(unhashed_derivation)
    )
    co_mutated["budget_derivation_sha256"] = co_mutated["budget_derivation"][
        "derivation_sha256"
    ]
    co_mutated_path = tmp_path / "co-mutated.json"
    co_mutated_path.write_bytes(cal.canonical_json_bytes(co_mutated))
    with pytest.raises(cal.AuthorizationError, match="canonical recomputation"):
        validate_mechanics(co_mutated_path)

    monkeypatch.setattr(
        cal,
        "git_read_only_state",
        lambda: {"head": commit, "upstream": commit, "dirty": True},
    )
    with pytest.raises(cal.AuthorizationError, match="dirty"):
        validate_mechanics(manifest)
    monkeypatch.setattr(
        cal,
        "git_read_only_state",
        lambda: {"head": commit, "upstream": "e" * 40, "dirty": False},
    )
    with pytest.raises(cal.AuthorizationError, match="clean pushed HEAD"):
        validate_mechanics(manifest)

    bad_live = _live_resource_environment()
    bad_live["torch_version"] = "2.5.0+cpu"
    monkeypatch.setattr(cal, "live_resource_environment", lambda: bad_live)
    with pytest.raises(cal.ContractError, match="four-CPU/resource/meter"):
        cal._authorization_expectation_mechanics(
            config=config,
            scaffold_commit=commit,
            root=987654321,
            output="reserved/exp658/root-987654321",
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
            registry=registry,
            evidence_root=evidence_root,
            repository_root=ROOT,
        )
    monkeypatch.setattr(cal, "live_resource_environment", _live_resource_environment)
    with pytest.raises(cal.ContractError, match="attempt allowance"):
        cal.workload_binding(
            config,
            cache,
            deterministic_attempt_allowance=0,
        )
    used = tmp_path / "already-used"
    used.mkdir()
    with pytest.raises(cal.AuthorizationError, match="already used"):
        real_unused_verifier(999999937, str(used))


def test_production_apis_reject_injected_trust_roots_and_synthetic_observations(
    tmp_path: Path, monkeypatch
) -> None:
    config = cal.load_config()
    cache = cal.validate_incumbent_cache(config)
    missing = cal.missing_exact_remaining_l1_rate_keys(cache)
    monkeypatch.setattr(cal, "live_resource_environment", _live_resource_environment)

    with pytest.raises(cal.ContractError) as budget_blocked:
        cal.resolve_remaining_l1_budget()
    assert all(name in str(budget_blocked.value) for name in missing)
    assert "width=32" in str(budget_blocked.value)
    assert "width=64" in str(budget_blocked.value)
    assert "arm=control" in str(budget_blocked.value)
    assert "closed_registry_entries=0" in str(budget_blocked.value)
    with pytest.raises(cal.ContractError) as authorization_blocked:
        cal.authorization_expectation(
            scaffold_commit="f" * 40,
            root=987654321,
            output="reserved/exp658/root-987654321",
            deterministic_attempt_allowance=ATTEMPT_ALLOWANCE,
        )
    assert all(name in str(authorization_blocked.value) for name in missing)

    temporary_authorization = tmp_path / "co-mutated-authorization.json"
    temporary_authorization.write_bytes(cal.canonical_json_bytes({"plausible": True}))
    probe = cal.SideEffectProbe()
    with pytest.raises(cal.AuthorizationError, match="fixed repository root"):
        cal.inert_entrypoint(temporary_authorization, side_effects=probe)
    assert probe.untouched()
    with pytest.raises(cal.AuthorizationError, match="fixed repository root"):
        cal.resolve_dynamic_budget(temporary_authorization, [])

    temporary_receipt = tmp_path / "synthetic-receipt.json"
    temporary_receipt.write_bytes(
        cal.canonical_json_bytes(
            {
                "resources": {
                    "observations": [
                        {
                            "environment_sha256": "a" * 64,
                            "prior_authorization_sha256": "b" * 64,
                            "measurement_record_sha256": "c" * 64,
                        }
                    ]
                }
            }
        )
    )
    with pytest.raises(cal.AuthorizationError, match="fixed repository root"):
        cal.validate_resource_only_receipt(
            temporary_receipt,
            authorization_path=temporary_authorization,
        )

    assert tuple(inspect.signature(cal.validate_authorization_candidate).parameters) == (
        "manifest_path",
    )
    assert tuple(inspect.signature(cal.inert_entrypoint).parameters) == (
        "manifest_path",
        "side_effects",
    )
    assert tuple(inspect.signature(cal.resolve_remaining_l1_budget).parameters) == ()
    assert tuple(inspect.signature(cal.authorization_expectation).parameters) == (
        "scaffold_commit",
        "root",
        "output",
        "deterministic_attempt_allowance",
    )
    assert tuple(inspect.signature(cal.validate_resource_only_receipt).parameters) == (
        "receipt_path",
        "authorization_path",
    )


def _resource_receipt(
    authorization: dict,
    config: dict,
    evidence_root: Path,
    registry: dict[str, cal.CanonicalOperationRegistration],
) -> dict:
    cache = cal.validate_incumbent_cache(config)
    physical_manifest = [
        *cal.warmup_l1_physical_manifest(),
        *cal.remaining_l1_physical_manifest(cache),
    ]
    record_descriptors = [
        dict(item)
        for item in authorization["warmup_measurement_evidence"][
            "record_descriptors"
        ]
    ]
    descriptor_by_physical = {
        item["physical_key"]: item
        for item in authorization["authorized_operation_descriptors"]
    }
    for tick, physical in enumerate(
        cal.remaining_l1_physical_manifest(cache), start=100
    ):
        wall = float(physical.units + tick / 1000)
        cpu = float(physical.units * 2 + tick / 1000)
        record = cal._measurement_record_from_snapshots_mechanics(
            physical,
            cal.MeterSnapshot(3000.0 + tick, 4000.0 + tick, 3000 + tick),
            cal.MeterSnapshot(
                3000.0 + tick + wall,
                4000.0 + tick + cpu,
                4000 + tick,
            ),
            operation_descriptor=descriptor_by_physical[physical.key],
            environment_sha256=authorization["environment_sha256"],
            prior_authorization_sha256=cal.sha256_bytes(
                cal.canonical_json_bytes(authorization)
            ),
            inner_monotonic_wall_seconds=(
                wall * 0.5 if physical.phase == "L1-profile-fresh" else None
            ),
        )
        record_path = Path("evidence/l1-records") / f"{physical.key}.json"
        _write_canonical_json(evidence_root / record_path, record)
        record_descriptors.append(
            {
                "physical_key": physical.key,
                "rate_key": physical.rate_key,
                "record_path": record_path.as_posix(),
                "record_file_sha256": cal.frozen_run.sha256(
                    evidence_root / record_path
                ),
                "record_sha256": record["record_sha256"],
            }
        )
    manifest_payload = cal._physical_manifest_payload(physical_manifest)
    receipt = {
        "schema_version": cal.RESOURCE_RECEIPT_SCHEMA,
        "bindings": {
            "authorization_sha256": cal.sha256_bytes(
                cal.canonical_json_bytes(authorization)
            ),
            "scaffold_hashes_sha256": cal.sha256_bytes(
                cal.canonical_json_bytes(authorization["hashes"])
            ),
            "workload_sha256": authorization["workload_sha256"],
            "budget_derivation_sha256": authorization[
                "budget_derivation_sha256"
            ],
            "cache_evidence_sha256": cache.evidence_sha256,
            "meter_implementation_sha256": cal.meter_implementation_binding()[
                "sha256"
            ],
            "environment_sha256": authorization["environment_sha256"],
            "warmup_observations_sha256": authorization[
                "warmup_observations_sha256"
            ],
            "warmup_measurement_evidence_sha256": cal.sha256_bytes(
                cal.canonical_json_bytes(
                    authorization["warmup_measurement_evidence"]
                )
            ),
        },
        "boundary": {
            "stage": "L1",
            "resource_only": True,
            "scientific_values_present": False,
            "fidelity_reduced": False,
        },
        "identities": {
            "split_sha256_by_width": authorization["workload"][
                "identity_binding"
            ]["split_sha256_by_width"],
            "stream_manifest_sha256": authorization["workload"][
                "identity_binding"
            ]["stream_manifest_sha256"],
            "width256_stream_sha256": authorization["workload"][
                "identity_binding"
            ]["width256_stream_sha256"],
        },
        "counts": {
            "physical_manifest_sha256": cal.sha256_bytes(
                cal.canonical_json_bytes(manifest_payload)
            ),
            "physical_manifest": manifest_payload,
            "completed_physical_units": sum(item.units for item in physical_manifest),
            "completed_components": len(physical_manifest),
        },
        "hashes": {
            "scaffold_binding_sha256": cal.sha256_bytes(
                cal.canonical_json_bytes(authorization["hashes"])
            ),
            "control_checkpoint_sha256": "f" * 64,
            "challenger_checkpoint_sha256": "1" * 64,
        },
        "shapes": {
            "control_parameters": 103522,
            "challenger_parameters": 103522,
            "state_key_count": 24,
        },
        "integrity": {
            "frozen_hashes_exact": True,
            "source_parity_exact": True,
            "control_reload_exact": True,
            "challenger_reload_exact": True,
            "profile_uses_exact_reloaded_challenger": True,
        },
        "determinism": {
            "width32_bit_exact": True,
            "width64_bit_exact": True,
        },
        "resources": {
            "record_descriptors": record_descriptors,
            "wall_cpu_distinct": True,
            "rss_scope": "cgroup-memory.peak",
        },
        "cache_verdict": {
            "valid": cache.valid,
            "reason": cache.reason,
            "evidence_sha256": cache.evidence_sha256,
            "full_1100_required": cache.full_1100_required,
        },
        "command": {"candidate005_full_1100": None},
        "budgets": authorization["numeric_resource_envelope"],
    }
    observations = cal._load_authorized_l1_observations(
        record_descriptors,
        authorization=authorization,
        config=config,
        cache_verdict=cache,
        registry=registry,
        evidence_root=evidence_root,
    )
    receipt["post_l1_budget_derivation"] = cal._post_l1_dynamic_budget_derivation(
        observations,
        record_descriptors,
        authorization=authorization,
        cache_verdict=cache,
    )
    return receipt


def test_strict_nested_resource_receipt_blocks_all_science_leakage(
    monkeypatch, tmp_path: Path,
) -> None:
    config, _cache, evidence_root, registry, _commit, expected = _authorization_fixture(
        monkeypatch, tmp_path
    )
    receipt = _resource_receipt(expected, config, evidence_root, registry)

    def validate_mechanics(candidate: dict) -> None:
        cal._validate_resource_only_receipt_mechanics(
            candidate,
            authorization=expected,
            config=config,
            registry=registry,
            root=ROOT,
            evidence_root=evidence_root,
        )

    validate_mechanics(receipt)

    mutations = []
    for path, key in (
        (("boundary",), "accuracy"),
        (("bindings",), "prediction"),
        (("identities",), "prediction"),
        (("counts",), "correct_count"),
        (("hashes",), "loss"),
        (("shapes",), "error_count"),
        (("integrity",), "recall"),
        (("determinism",), "score"),
        (("resources",), "accuracy"),
        (("cache_verdict",), "loss"),
        (("command",), "prediction"),
        (("budgets",), "recall"),
        (("post_l1_budget_derivation",), "accuracy"),
    ):
        changed = copy.deepcopy(receipt)
        target = changed
        for part in path:
            target = target[part]
        target[key] = 0.99
        mutations.append(changed)
    hidden_descriptor = copy.deepcopy(receipt)
    hidden_descriptor["resources"]["record_descriptors"][0]["accuracy"] = 0.99
    mutations.append(hidden_descriptor)
    synthetic_observation = copy.deepcopy(receipt)
    synthetic_observation["resources"]["observations"] = [
        {
            "physical_key": "plausible",
            "environment_sha256": "a" * 64,
            "prior_authorization_sha256": "b" * 64,
            "measurement_record_sha256": "c" * 64,
        }
    ]
    mutations.append(synthetic_observation)
    hidden_manifest = copy.deepcopy(receipt)
    hidden_manifest["counts"]["physical_manifest"][0]["loss"] = 0.1
    mutations.append(hidden_manifest)
    renamed_component = copy.deepcopy(receipt)
    renamed_component["counts"]["physical_manifest"][0]["key"] = "l1-accuracy-059"
    renamed_component["resources"]["record_descriptors"][0]["physical_key"] = (
        "l1-accuracy-059"
    )
    renamed_component["counts"]["physical_manifest_sha256"] = cal.sha256_bytes(
        cal.canonical_json_bytes(
            renamed_component["counts"]["physical_manifest"]
        )
    )
    mutations.append(renamed_component)
    co_updated_warmup = copy.deepcopy(receipt)
    co_updated_warmup["resources"]["record_descriptors"][0][
        "record_sha256"
    ] = "d" * 64
    co_updated_warmup["bindings"]["warmup_observations_sha256"] = (
        "e" * 64
    )
    mutations.append(co_updated_warmup)
    mutated_runtime_binding = copy.deepcopy(receipt)
    mutated_runtime_binding["bindings"]["environment_sha256"] = "a" * 64
    mutations.append(mutated_runtime_binding)
    co_mutated_post_l1 = copy.deepcopy(receipt)
    post_l1 = co_mutated_post_l1["post_l1_budget_derivation"]
    post_l1["expected"]["monotonic_wall_seconds"] += 1.0
    post_l1["soft_budget"]["monotonic_wall_seconds"] = 1.25 * post_l1[
        "expected"
    ]["monotonic_wall_seconds"]
    unhashed_post_l1 = dict(post_l1)
    unhashed_post_l1.pop("derivation_sha256")
    post_l1["derivation_sha256"] = cal.sha256_bytes(
        cal.canonical_json_bytes(unhashed_post_l1)
    )
    mutations.append(co_mutated_post_l1)
    omitted_post_l1 = copy.deepcopy(receipt)
    omitted_post_l1.pop("post_l1_budget_derivation")
    mutations.append(omitted_post_l1)
    for changed in mutations:
        with pytest.raises(cal.ContractError):
            validate_mechanics(changed)

    nonexistent = copy.deepcopy(receipt)
    nonexistent["resources"]["record_descriptors"][-1][
        "record_path"
    ] = "evidence/l1-records/nonexistent.json"
    with pytest.raises(cal.ContractError, match="path/hash mismatch"):
        validate_mechanics(nonexistent)

    altered_descriptor = receipt["resources"]["record_descriptors"][-1]
    altered_path = evidence_root / altered_descriptor["record_path"]
    original_bytes = altered_path.read_bytes()
    altered_path.write_bytes(b"{}\n")
    try:
        with pytest.raises(cal.ContractError, match="path/hash mismatch"):
            validate_mechanics(receipt)
    finally:
        altered_path.write_bytes(original_bytes)

    omitted = copy.deepcopy(receipt)
    omitted["counts"]["physical_manifest"].pop()
    omitted["resources"]["record_descriptors"].pop()
    omitted["counts"]["completed_components"] -= 1
    omitted["counts"]["completed_physical_units"] = sum(
        item["units"] for item in omitted["counts"]["physical_manifest"]
    )
    omitted["counts"]["physical_manifest_sha256"] = cal.sha256_bytes(
        cal.canonical_json_bytes(omitted["counts"]["physical_manifest"])
    )
    with pytest.raises(cal.ContractError, match="exactly canonical"):
        validate_mechanics(omitted)


def test_import_and_cli_are_side_effect_free_and_fail_closed(monkeypatch) -> None:
    paths = (
        EXPERIMENT / "calibration.py",
        EXPERIMENT / "data.py",
        EXPERIMENT / "model.py",
        EXPERIMENT / "run.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not any("mlflow" in name for name in imports)
        top_level = [
            node
            for node in tree.body
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Import,
                    ast.ImportFrom,
                ),
            )
        ]
        forbidden_calls = {
            node.func.attr
            for statement in top_level
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {"mkdir", "write_text", "write_bytes", "open", "run", "Popen", "start_run"}
        }
        assert forbidden_calls == set()

    before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    completed = subprocess.run(
        [sys.executable, str(EXPERIMENT / "calibration.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert completed.returncode != 0
    assert "later Judge authorization is absent" in completed.stderr
    assert before == after

    def forbidden(*_args, **_kwargs):
        raise AssertionError("side-effect helper reached")

    for name in (
        "checkpoint_roundtrip",
        "optimizer_and_scheduler",
        "frozen_matched_pair",
        "candidate005_full_1100_rerun_command",
        "workload_binding",
    ):
        monkeypatch.setattr(cal, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["calibration.py"])
    with pytest.raises(cal.AuthorizationError, match="absent"):
        cal.main()
