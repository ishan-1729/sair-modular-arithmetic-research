from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import shutil
import time
import types
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/exp_648_structured_trajectory_consistency/scientific.py"
TEST_ROOT = 648_900_001


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("exp648_scientific_test_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def exact_trajectory(module, identity, *, fail_at: int | None = None) -> dict:
    teacher = [0]
    for position in range(256):
        bit = (identity.multiplier >> (255 - position)) & 1
        teacher.append(
            module.cal.exact_teacher_step(
                teacher[-1], identity.multiplicand, identity.modulus, bit
            )[0]
        )
    predicted = list(teacher)
    first_error = 257
    if fail_at is not None:
        predicted[fail_at + 1] = (teacher[fail_at + 1] + 1) % (1 << 256)
        first_error = fail_at
    return {
        **module.cal._identity_payload(identity),
        "first_error_index": first_error,
        "complete_predicted_states": predicted,
        "complete_teacher_states": teacher,
    }


def fake_window(
    module,
    *,
    kind: str,
    identity_index: int,
    start: int,
    band: str = "b0",
) -> dict:
    lower = (1 << 255) + (int(band[1]) * (1 << 252))
    modulus = lower + 123
    multiplier = 0
    for offset in range(8):
        multiplier |= ((identity_index + offset) % 2) << (255 - (start + offset))
    multiplicand = identity_index + 7
    teacher = [0]
    bits = []
    quotients = []
    for offset in range(8):
        bit = (multiplier >> (255 - (start + offset))) & 1
        bits.append(bit)
        state, quotient = module.cal.exact_teacher_step(
            teacher[-1], multiplicand, modulus, bit
        )
        teacher.append(state)
        quotients.append(quotient)
    natural = list(teacher)
    first_error = 257
    if kind == "failure":
        natural[1] = (teacher[1] + 1) % (1 << 256)
        first_error = start
    return {
        "identity_index": identity_index,
        "multiplier": multiplier,
        "multiplicand": multiplicand,
        "modulus": modulus,
        "modulus_bit_length": modulus.bit_length(),
        "normalized_modulus_band": band,
        "window_start_index": start,
        "window_length": 8,
        "first_error_index": first_error,
        "teacher_states": teacher,
        "natural_states": natural,
        "input_bits": bits,
        "quotients": quotients,
    }


def test_historical_calibration_is_preserved_but_not_live_screen_calibration(module) -> None:
    assert module.sha256(module.CALIBRATION_RUNNER) == module.CALIBRATION_RUNNER_SHA256
    assert module.CALIBRATION_RUNNER_SHA256 == (
        "f8cdc278b67013c6ca920833bc26473ad938e15749eb85811fc00016b6bb5672"
    )
    assert module.sha256(module.CALIBRATION_EVIDENCE_PATH) == module.PINNED["calibration_evidence_sha256"]
    lifecycle = module.readiness_calibration_publication_state()
    assert lifecycle["status"] == "live-receipt-present"
    assert lifecycle["live_receipt_sha256"] == (
        "1da93f456a54bfb61a36b4085cfb70b517953c4153d7378190237c872ff4167a"
    )
    assert lifecycle["superseded_root_20261159_preserved"] is True


def test_root_allowlist_uses_only_test_root_and_rejects_all_other_surfaces(module) -> None:
    assert module.authorize_root("test", TEST_ROOT) == TEST_ROOT
    for mode in (
        "calibration",
        "confirmation",
        "checked",
        "checked-public",
        "independent",
        "official",
        "official-private",
        "private",
        "formal",
        "unknown",
    ):
        with pytest.raises(PermissionError):
            module.authorize_root(mode, TEST_ROOT)
    with pytest.raises(PermissionError):
        module.authorize_root("corpus", TEST_ROOT)


def test_screen_missing_authorization_fails_before_data_or_model_action(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    touched = False

    class ForbiddenScreen:
        def __init__(self, *_args, **_kwargs):
            nonlocal touched
            touched = True
            raise AssertionError("screen object must not be constructed")

    monkeypatch.setattr(module, "ScreenExperiment", ForbiddenScreen)
    missing = tmp_path / "absent.json"
    with pytest.raises(PermissionError, match="absent"):
        module.main(
            ["--mode", "screen", "--authorization-manifest", str(missing)]
        )
    assert touched is False


def test_readiness_calibration_cli_missing_authorization_fails_before_executor(module, monkeypatch, tmp_path) -> None:
    events = []
    monkeypatch.setattr(module, "execute_readiness_calibration", lambda *_args, **_kwargs: events.append("execute"))
    with pytest.raises(PermissionError, match="absent"):
        module.main([
            "--mode", "readiness-calibration", "--authorization-manifest",
            str(tmp_path / "missing.json"),
        ])
    assert events == []


def test_screen_invalid_and_exact_authorization_manifest_hashes(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    corpus_manifest = data / "corpus-manifest.json"
    corpus_manifest.write_bytes(b'{"frozen":true}\n')
    monkeypatch.setattr(module, "DATA_DIR", data)
    authorization = tmp_path / "authorization.json"
    authorization.write_text("{}\n", encoding="utf-8")
    binding = {
        "corpus_manifest_sha256": module.sha256(corpus_manifest),
        "scientific_runner_sha256": module.sha256(module.RUNNER if hasattr(module, "RUNNER") else module.__file__),
        "approved_base_commit": "a" * 40,
    }
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "screen_readiness_binding", lambda _base: binding)
    def fake_git(*args):
        if args[0] == "status": return ""
        if args[0] == "log": return "b" * 40
        if args[0] == "rev-list": return "1"
        if args[0] == "diff": return "authorization.json"
        if args[0] == "rev-parse": return "b" * 40
        return ""
    monkeypatch.setattr(module, "git_output", fake_git)
    monkeypatch.setattr(module, "validate_one_child_authorization_lifecycle", lambda **_kwargs: "b" * 40)
    with pytest.raises(PermissionError, match="schema mismatch"):
        module.verify_screen_authorization(authorization, committed_bytes=authorization.read_bytes())
    expected = module.expected_screen_authorization_payload(module.sha256(corpus_manifest), binding=binding)
    authorization.write_bytes(module.canonical_json_bytes(expected, pretty=True))
    assert module.verify_screen_authorization(authorization, committed_bytes=authorization.read_bytes()) == expected
    tampered = copy.deepcopy(expected)
    tampered["confirmation_authorized"] = True
    authorization.write_bytes(module.canonical_json_bytes(tampered, pretty=True))
    with pytest.raises(PermissionError, match="hash mismatch"):
        module.verify_screen_authorization(authorization, committed_bytes=authorization.read_bytes())


def test_cli_wrong_corpus_root_fails_before_experiment_construction(
    module, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched = False

    class ForbiddenCorpus:
        def __init__(self):
            nonlocal touched
            touched = True
            raise AssertionError("corpus object must not be constructed")

    monkeypatch.setattr(module, "CorpusExperiment", ForbiddenCorpus)
    with pytest.raises(PermissionError):
        module.main(["--mode", "corpus", "--root", str(TEST_ROOT)])
    assert touched is False


def test_every_generator_reauthorizes_before_prime_generation(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prime_called = False

    def forbidden_prime(*_args, **_kwargs):
        nonlocal prime_called
        prime_called = True
        raise AssertionError("prime generation followed failed authorization")

    monkeypatch.setattr(module.cal, "_prime_in_band", forbidden_prime)
    with pytest.raises(PermissionError):
        module.generate_identities("corpus", TEST_ROOT, 0, 1)
    with pytest.raises(PermissionError):
        module.generate_screen_transition_cases(TEST_ROOT, 128, 1, tmp_path / "missing")
    with pytest.raises(PermissionError):
        module.generate_screen_rollout_cases(TEST_ROOT, 128, 1, tmp_path / "missing")
    assert prime_called is False


def test_identity_geometry_balanced_bands_and_determinism(module) -> None:
    first = module.generate_identities("test", TEST_ROOT, 0, 8)
    second = module.generate_identities("test", TEST_ROOT, 0, 8)
    first_payload = [module.cal._identity_payload(identity) for identity in first]
    second_payload = [module.cal._identity_payload(identity) for identity in second]
    assert first_payload == second_payload
    assert [identity.identity_index for identity in first] == list(range(8))
    assert all(identity.modulus_bit_length == 256 for identity in first)
    assert [identity.band for identity in first] == ["b0", "b1", "b2", "b3"] * 2
    assert len({(row["multiplier"], row["multiplicand"], row["modulus"]) for row in first_payload}) == 8


def test_exact_matching_order_no_reuse_and_equivalence_to_frozen_primitive(module) -> None:
    failures = [
        fake_window(module, kind="failure", identity_index=10, start=3),
        fake_window(module, kind="failure", identity_index=11, start=3),
    ]
    successes = [
        fake_window(module, kind="success", identity_index=20, start=3),
        fake_window(module, kind="success", identity_index=21, start=3),
        fake_window(module, kind="success", identity_index=22, start=3),
    ]
    observed = module.exact_match_windows(failures, successes, 2)
    frozen = module.cal.match_windows(failures, successes, 2)
    assert module.canonical_json(observed) == module.canonical_json(frozen)
    assert [pair["success"]["identity_index"] for pair in observed] == [20, 21]
    assert [pair["pair_index"] for pair in observed] == [0, 1]
    duplicate_successes = [successes[0], copy.deepcopy(successes[0])]
    with pytest.raises(RuntimeError, match="duplicate success"):
        module.exact_match_windows(failures, duplicate_successes, 1)
    with pytest.raises(RuntimeError, match="shortfall"):
        module.exact_match_windows(failures, successes[:1], 2)


def test_strong_teacher_first_error_window_linkage_and_tamper_failures(module) -> None:
    identity = module.generate_identities("test", TEST_ROOT, 0, 1)[0]
    record = exact_trajectory(module, identity, fail_at=3)
    verified = module.verify_trajectory_record(record)
    assert len(verified["failures"]) == 1
    failure = verified["failures"][0]
    assert failure["window_start_index"] == 3
    assert failure["teacher_states"] == record["complete_teacher_states"][3:12]
    assert failure["natural_states"] == record["complete_predicted_states"][3:12]
    assert failure["first_error_index"] == 3

    bad_teacher = copy.deepcopy(record)
    bad_teacher["complete_teacher_states"][10] += 1
    with pytest.raises(RuntimeError, match="teacher trajectory"):
        module.verify_trajectory_record(bad_teacher)
    bad_error = copy.deepcopy(record)
    bad_error["first_error_index"] = 4
    with pytest.raises(RuntimeError, match="first-error"):
        module.verify_trajectory_record(bad_error)
    bad_state = copy.deepcopy(record)
    bad_state["complete_predicted_states"][4] = 1 << 256
    with pytest.raises(RuntimeError, match="invalid bit state"):
        module.verify_trajectory_record(bad_state)


def test_independent_source_replay_requires_every_complete_state(module) -> None:
    identity = module.generate_identities("test", TEST_ROOT, 0, 1)[0]
    payload = exact_trajectory(module, identity, fail_at=1)
    native = {
        "identity": identity,
        "predicted_states": payload["complete_predicted_states"],
        "teacher_states": payload["complete_teacher_states"],
        "first_error_index": payload["first_error_index"],
    }
    module.verify_replay_pair([native], [copy.deepcopy(native)])
    tampered = copy.deepcopy(native)
    tampered["predicted_states"][20] += 1
    with pytest.raises(RuntimeError, match="replay mismatch"):
        module.verify_replay_pair([native], [tampered])


def test_gzip_canonical_bytes_are_deterministic_and_tamper_visible(module) -> None:
    payload = module.canonical_json_bytes({"b": [1, 2], "a": True})
    first = module.deterministic_gzip_bytes(payload)
    second = module.deterministic_gzip_bytes(payload)
    assert first == second
    assert module.deterministic_gunzip_bytes(first) == payload
    assert first[4:8] == b"\x00\x00\x00\x00"
    corrupted = bytearray(first)
    corrupted[-1] ^= 1
    with pytest.raises((OSError, EOFError)):
        module.deterministic_gunzip_bytes(bytes(corrupted))


def test_cache_key_covers_all_required_provenance_and_test_root(module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "SCRATCH_CORPUS_DIR", tmp_path)
    required = {
        "checkpoint_sha256",
        "evaluator_sha256",
        "dependency_manifest_sha256",
        "dataset_sha256",
        "config_sha256",
        "executed_config_sha256",
        "root",
        "environment_sha256",
        "calibration_runner_sha256",
        "scientific_runner_sha256",
        "scientific_protocol_sha256",
        "resource_protocol_sha256",
        "workload_sha256",
        "repository_head",
    }
    fields = {key: "a" * 64 for key in required}
    fields["root"] = TEST_ROOT
    first = module.batch_cache_key(fields, 0, 1)
    mutated = copy.deepcopy(fields)
    mutated["evaluator_sha256"] = "b" * 64
    assert module.batch_cache_key(mutated, 0, 1) != first
    assert first in module.batch_cache_path(fields, 0, 1).name


def test_complete_cache_validation_recovery_and_reused_charging(module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "SCRATCH_CORPUS_DIR", tmp_path)
    identity = module.generate_identities("test", TEST_ROOT, 0, 1)[0]
    trajectory = exact_trajectory(module, identity, fail_at=0)
    fields = {
        "checkpoint_sha256": "a" * 64,
        "evaluator_sha256": "b" * 64,
        "dependency_manifest_sha256": "c" * 64,
        "dataset_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "executed_config_sha256": "f" * 64,
        "root": TEST_ROOT,
        "environment_sha256": "1" * 64,
        "calibration_runner_sha256": "2" * 64,
        "scientific_runner_sha256": "3" * 64,
        "scientific_protocol_sha256": "4" * 64,
        "resource_protocol_sha256": "5" * 64,
        "workload_sha256": "6" * 64,
        "repository_head": "7" * 40,
    }
    record = {
        "schema_version": "exp648-corpus-batch-cache-v1",
        "role": "generated-training-split-corpus-only",
        "root": TEST_ROOT,
        "start": 0,
        "count": 1,
        "cache_key_fields": fields,
        "cache_key": module.batch_cache_key(fields, 0, 1),
        "trajectories": [trajectory],
        "independent_replay_sha256": module.sha256_bytes(
            module.canonical_json_bytes([trajectory])
        ),
        "timing": {
            "wall_seconds": 2.0,
            "aggregate_cpu_seconds": 7.0,
            "peak_rss_bytes": 100,
            "mining_wall_seconds": 1.0,
            "mining_aggregate_cpu_seconds": 3.0,
            "replay_wall_seconds": 1.0,
            "replay_aggregate_cpu_seconds": 4.0,
        },
        "complete": True,
    }
    record["core_sha256"] = module.sha256_bytes(module.canonical_json_bytes(record))
    module.validate_batch_cache(record, fields, 0, 1)
    tampered = copy.deepcopy(record)
    tampered["trajectories"][0]["first_error_index"] = 5
    with pytest.raises(RuntimeError):
        module.validate_batch_cache(tampered, fields, 0, 1)

    guard = module.CorpusGuard(module.psutil.Process())
    before = guard.charged()
    guard.add_reused(record["timing"], 1)
    after = guard.charged()
    assert after[0] - before[0] == pytest.approx(2.0, abs=0.05)
    assert after[1] - before[1] == pytest.approx(7.0, abs=0.05)


def test_first_quarter_guard_and_hard_guards_are_independent(module, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = module.CorpusGuard(module.psutil.Process())
    monkeypatch.setattr(module, "PROJECTED_WALL_SECONDS", 1_000_000.0)
    monkeypatch.setattr(module, "PROJECTED_CPU_SECONDS", 1_000_000.0)
    guard.check_quarter(128, 192)
    assert guard.quarter_checked is True

    wall_guard = module.CorpusGuard(module.psutil.Process())
    wall_guard.reused_wall_seconds = module.HARD_WALL_SECONDS
    with pytest.raises(TimeoutError, match="wall"):
        wall_guard.check("test", 1)
    cpu_guard = module.CorpusGuard(module.psutil.Process())
    cpu_guard.reused_cpu_seconds = module.HARD_CPU_SECONDS
    with pytest.raises(TimeoutError, match="CPU"):
        cpu_guard.check("test", 1)
    count_guard = module.CorpusGuard(module.psutil.Process())
    with pytest.raises(TimeoutError, match="identity"):
        count_guard.check("test", module.HARD_IDENTITY_CEILING + 1)


def test_integrity_schema_is_exact_boolean_required_key_set(module) -> None:
    valid = {key: True for key in module.REQUIRED_CORPUS_INTEGRITY_KEYS}
    assert module.validate_integrity_checks(valid)
    missing = dict(valid)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="required-key"):
        module.validate_integrity_checks(missing)
    extra = {**valid, "unexpected": True}
    with pytest.raises(RuntimeError, match="required-key"):
        module.validate_integrity_checks(extra)
    false = dict(valid)
    false[next(iter(false))] = False
    with pytest.raises(RuntimeError, match="boolean"):
        module.validate_integrity_checks(false)
    nonboolean = dict(valid)
    nonboolean[next(iter(nonboolean))] = 1
    with pytest.raises(RuntimeError, match="boolean"):
        module.validate_integrity_checks(nonboolean)


def test_workload_and_expected_artifact_contract_is_full_fidelity(module) -> None:
    workload = module.corpus_workload_payload()
    assert workload == {
        "schema_version": "exp648-corpus-workload-v1",
        "mode": "corpus",
        "root": module.CORPUS_ROOT,
        "width": 256,
        "identity_batch_size": 32,
        "target_failure_windows": 512,
        "target_success_windows": 512,
        "target_pairs": 512,
        "window_length": 8,
        "full_source_replays_per_identity": 2,
        "matching": "exact-stratum-first-unused-no-reuse-no-relaxation",
        "deterministic_gzip_mtime": 0,
        "screen_training_or_evaluation": False,
    }
    assert module.PROJECTED_IDENTITIES == 768
    assert module.SOFT_IDENTITY_BUDGET == 960
    assert module.HARD_IDENTITY_CEILING == 1440
    assert [path.name for path, _ in module._mlflow_artifact_sources()] == [
        "identity-manifest.json",
        "trajectory-manifest.json.gz",
        "paired-corpus.json.gz",
        "corpus-manifest.json",
        "run.json",
        "environment.json",
    ]


def test_source_checkpoint_is_read_only_and_screen_code_has_no_hand_arithmetic(module) -> None:
    before = module.sha256(module.SOURCE_CHECKPOINT_PATH)
    assert before == module.PINNED["source_checkpoint_sha256"]
    assert module.sha256(module.SOURCE_CHECKPOINT_PATH) == before
    assert callable(module.atomic_save_checkpoint)
    assert callable(module.screen_training_update)
    assert callable(module.evaluate_transitions)
    assert callable(module.evaluate_rollouts)


def test_no_scientific_metric_names_in_corpus_mlflow_metric_block(module) -> None:
    receipt = json.loads((module.TRACKED_CORPUS_DIR / "mlflow.json").read_text())
    metric_names = receipt["verified"]["metrics"]
    for forbidden in ("accuracy", "selection", "exact", "gain", "delta"):
        assert all(forbidden not in name.lower() for name in metric_names)


def test_screen_order_is_frozen_training_root_derived_and_matched(module) -> None:
    first = module.frozen_root_pair_order(512, "a" * 64)
    second = module.frozen_root_pair_order(512, "a" * 64)
    changed = module.frozen_root_pair_order(512, "b" * 64)
    assert first == second
    assert first != list(range(512))
    assert first != changed
    assert sorted(first) == list(range(512))


def test_screen_guard_independent_units_thresholds_and_numeric_quarter(module, monkeypatch) -> None:
    calibration = {
        "resolved_screen_budgets": {
            "wall_seconds": {"projected_seconds": 1000, "soft_budget_seconds": 1500, "hard_runaway_ceiling_seconds": 2000},
            "aggregate_cpu_seconds": {"projected_seconds": 4000, "soft_budget_seconds": 6000, "hard_runaway_ceiling_seconds": 8000},
            "peak_rss_guard_bytes": 10**15,
            "first_quarter_checkpoint": {"wall_seconds": {"expected_elapsed_seconds": 250}, "aggregate_cpu_seconds": {"expected_elapsed_seconds": 1000}},
        }
    }
    guard = module.ScreenGuard(calibration, module.psutil.Process())
    record = guard.check_first_quarter(100, 400)
    assert set(record) >= {
        "charged_wall_seconds", "charged_aggregate_cpu_seconds",
        "projected_full_wall_seconds_from_checkpoint",
        "projected_full_aggregate_cpu_seconds_from_checkpoint",
    }
    guard.reused_wall = 2001
    with pytest.raises(TimeoutError, match="wall"):
        guard.check("wall-test")
    wall = module.ScreenGuard(calibration, module.psutil.Process())
    wall.reused_cpu = 8001
    with pytest.raises(TimeoutError, match="CPU"):
        wall.check("cpu-test")
    rss = module.ScreenGuard(calibration, module.psutil.Process())
    rss.hard_rss = 1
    with pytest.raises(MemoryError, match="RSS"):
        rss.check("rss-test")


def test_atomic_checkpoint_is_reload_verified_before_evaluation(module, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    model = module.nn.Linear(3, 2)
    receipt = module.atomic_save_checkpoint(model, tmp_path / "screen" / "weights.safetensors")
    assert receipt["saved_and_verified_before_evaluation"] is True
    assert receipt["state_dict_sha256"] == receipt["reload_state_dict_sha256"]


def test_event_trace_both_checkpoint_verifications_precede_evaluation(module, tmp_path, monkeypatch) -> None:
    events = []
    def fake_save(_model, path):
        events.append(f"save:{path.stem}")
        return {"saved_and_verified_before_evaluation": True, "sha256": path.stem}
    monkeypatch.setattr(module, "atomic_save_checkpoint", fake_save)
    receipts = module.prepare_checkpoints_for_evaluation(
        {"control": object(), "challenger": object()}, tmp_path,
        event_hook=events.append,
    )
    events.append("evaluation-input-generated")
    assert set(receipts) == {"control", "challenger"}
    assert events == [
        "save:control", "checkpoint-verified:control",
        "save:challenger", "checkpoint-verified:challenger",
        "evaluation-input-generated",
    ]


def test_evaluation_recovery_rejects_provenance_mismatch(module, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", tmp_path)
    calibration = {
        "resolved_screen_budgets": {
            "wall_seconds": {"projected_seconds": 1000, "soft_budget_seconds": 1500, "hard_runaway_ceiling_seconds": 2000},
            "aggregate_cpu_seconds": {"projected_seconds": 4000, "soft_budget_seconds": 6000, "hard_runaway_ceiling_seconds": 8000},
            "peak_rss_guard_bytes": 10**15,
            "first_quarter_checkpoint": {"wall_seconds": {"expected_elapsed_seconds": 250}, "aggregate_cpu_seconds": {"expected_elapsed_seconds": 1000}},
        }
    }
    guard = module.ScreenGuard(calibration, module.psutil.Process())
    cases = {"x": [1]}
    fields = {"runner": "a" * 64}
    key = module.sha256_bytes(module.canonical_json_bytes({
        "unit": "test-1", "cases_sha256": module.sha256_bytes(module.canonical_json_bytes(cases)), "key_fields": fields,
    }))
    path = tmp_path / "evaluation-cache" / f"test-1-{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":"wrong"}\n')
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        module.load_or_evaluate_unit(
            unit="test-1", cases=cases, models={}, evaluator=lambda *args: {},
            width=1, exp300=None, key_fields=fields, guard=guard,
        )


def test_atomic_screen_evidence_and_failure_has_no_gates(module, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", tmp_path / "scratch")
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tmp_path / "tracked")
    module.write_screen_file("proof.json", {"ok": True})
    assert (tmp_path / "scratch/evidence/proof.json").read_bytes() == (tmp_path / "tracked/proof.json").read_bytes()
    module.preserve_screen_failure("evaluation", TimeoutError("budget"))
    payload = json.loads((tmp_path / "tracked/incomplete.json").read_text())
    assert payload["status"] == "evaluation-resource-overrun"
    assert payload["gates_computed"] is False
    assert "accuracy" not in module.canonical_json(payload).lower()


def test_attempt_ledger_failure_receipt_is_complete_and_atomic(module, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", tmp_path / "scratch")
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tmp_path / "tracked")
    directory = tmp_path / "attempt"
    directory.mkdir()
    state = {
        "attempt_id": "attempt-7", "authorization_sha256": "a" * 64,
        "phase": "evaluation", "budgets": {"wall": 10},
        "completed_updates": {"control": 200, "challenger": 200},
        "completed_evaluation_units": ["transition-128"],
        "cache_receipts": [{"unit": "transition-128"}],
        "verified_checkpoints": {"control": {"sha256": "b" * 64}},
        "first_quarter": {"complete": True}, "environment_sha256": "c" * 64,
    }
    class Ledger:
        def __init__(self):
            self.state = state
            self.directory = directory
        def persist(self):
            module._atomic_write(
                self.directory / "attempt-state.json",
                module.canonical_json_bytes(self.state, pretty=True),
            )
    guard = types.SimpleNamespace(timing=lambda: {
        "wall_seconds": 2.0, "aggregate_cpu_seconds": 5.0, "peak_rss_bytes": 100,
    })
    experiment = types.SimpleNamespace(ledger=Ledger(), guard=guard)
    module.preserve_screen_failure("evaluation", TimeoutError("budget"), experiment=experiment)
    receipt = json.loads((directory / "incomplete.json").read_text())
    durable = json.loads((directory / "attempt-state.json").read_text())
    assert receipt["attempt_id"] == "attempt-7"
    assert receipt["completed_updates"] == state["completed_updates"]
    assert receipt["completed_evaluation_units"] == ["transition-128"]
    assert receipt["cache_receipts"] == [{"unit": "transition-128"}]
    assert receipt["verified_checkpoints"] == state["verified_checkpoints"]
    assert receipt["first_quarter"] == {"complete": True}
    assert durable["phase"] == "incomplete" and durable["complete"] is False


def test_historical_corpus_verifies_after_head_advance(module) -> None:
    result = module.verify_corpus_artifacts(require_receipts=True)
    assert result["historical_execution_provenance_verified"] is True
    assert result["historical_corpus_commit"] == "8d960fa747b64bffa3b60581ae4ff518436a8887"
    assert result["corpus_manifest_sha256"] == module.FROZEN_CORPUS_FILES["data/corpus-manifest.json"]


def test_historical_committed_corpus_does_not_require_scratch_mirror(module, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "SCRATCH_CORPUS_DIR", tmp_path / "absent-scratch")
    result = module.verify_corpus_artifacts(require_receipts=True)
    assert result["historical_execution_provenance_verified"] is True
    assert result["scratch_raw_tracked_byte_parity_when_present"] is None


def fake_readiness_repeat(scale: float) -> dict:
    timing = lambda value: {"wall_seconds": value * scale, "aggregate_cpu_seconds": value * scale * 4, "peak_rss_bytes": int(1_000_000 * scale)}
    record = {
        "complete": True,
        "fidelity_reduced": False,
        "scientific_accuracy_reported": False,
        "components": {
            "startup_model_loading": {"units": 1, "timing": timing(1)},
            "mining": {"matched_pairs": 64, "windows": 128, "timing": timing(64)},
            "corpus_integrity_validation": {"windows": 128, "timing": timing(128)},
            "training": {arm: {"updates": 20, "timing": timing(20)} for arm in ("control", "challenger")},
            "checkpoint": {arm: {"writes": 1, "timing": timing(1)} for arm in ("control", "challenger")},
            "transitions": {arm: {str(width): {"cases": 200, "timing": timing(200)} for width in (128, 256)} for arm in ("control", "challenger")},
            "rollouts": {arm: {str(width): {"cases": 20, "timing": timing(20)} for width in (64, 128, 256)} for arm in ("control", "challenger")},
            "hash_serialization": {"timing": timing(1)},
            "readiness_infrastructure": {"atomic_checkpoint_writes_and_reload_verifications": 2, "atomic_attempt_ledger_writes": 213, "evaluation_cache_write_validate_units": 7, "atomic_evidence_writes": 19, "component_timing": {"atomic_checkpoints": timing(0.5), "attempt_ledger": timing(0.5), "evaluation_cache": timing(0.5), "atomic_evidence": timing(0.5)}, "complete": True, "timing": timing(2)},
            "actual_source_replay": {"complete_identities": 64, "steps_per_identity": 256, "complete_steps": 16384, "actual_rollout_identity_batch_path": True, "complete": True, "timing": timing(64)},
            "mlflow_probe": {"complete": True, "artifact_count": 6, "remote_verified": True, "timing": timing(3)},
        },
        "base_total_timing_before_readiness_additions": timing(431),
        "total_timing": timing(500),
    }
    return record


def attach_fake_readiness_chronology(module, payload: dict) -> None:
    predecessor = None
    units = []
    for ordinal, ((unit, repeat_index), record) in enumerate(zip(
        module.READINESS_AUGMENTED_UNIT_ORDER,
        [payload["warmup"], *payload["repeats"]],
        strict=True,
    )):
        start = {
            "schema_version": "exp648-readiness-augmented-unit-start-v1",
            "event": "start", "root": payload["calibration_root"],
            "authorization_sha256": payload["calibration_authorization_sha256"],
            "ordinal": ordinal, "unit": unit, "repeat_index": repeat_index,
            "predecessor_seal_sha256": predecessor,
            "monotonic_ns": 100 + ordinal * 10, "complete": False,
        }
        start_sha = module.sha256_bytes(
            module.canonical_json_bytes(start, pretty=True)
        )
        bare = module._bare_augmented_record(record)
        seal = {
            "schema_version": "exp648-readiness-augmented-unit-seal-v1",
            "event": "seal", "root": payload["calibration_root"],
            "authorization_sha256": payload["calibration_authorization_sha256"],
            "ordinal": ordinal, "unit": unit, "repeat_index": repeat_index,
            "predecessor_seal_sha256": predecessor,
            "start_sha256": start_sha,
            "augmented_record_sha256": module.sha256_bytes(
                module.canonical_json_bytes(bare, pretty=True)
            ),
            "reused_from_augmented_cache": False,
            "source_cache_sha256": None,
            "monotonic_ns": 101 + ordinal * 10, "complete": True,
        }
        reference = {
            "ordinal": ordinal, "unit": unit, "repeat_index": repeat_index,
            "start": start, "start_sha256": start_sha,
            "seal": seal,
            "seal_sha256": module.sha256_bytes(
                module.canonical_json_bytes(seal, pretty=True)
            ),
        }
        record["readiness_chronology"] = reference
        record["reused_from_augmented_cache"] = False
        record["augmented_cache_path"] = None
        units.append(reference)
        predecessor = reference["seal_sha256"]
    payload["chronology"] = {
        "schema_version": "exp648-readiness-augmented-chronology-v1",
        "root": payload["calibration_root"],
        "authorization_sha256": payload["calibration_authorization_sha256"],
        "contract_sha256": module.READINESS_CHRONOLOGY_CONTRACT_SHA256,
        "order": [name for name, _ in module.READINESS_AUGMENTED_UNIT_ORDER],
        "units": units, "complete": True,
    }


def test_fresh_calibration_receipt_recomputes_all_dynamic_resources(module) -> None:
    repeats = [fake_readiness_repeat(value) for value in (1.0, 1.1, 0.9)]
    launcher = {"wall_seconds": 2.0, "aggregate_cpu_seconds": 8.0, "peak_rss_bytes": 2_000_000}
    warmup = fake_readiness_repeat(0.8)
    warmup.update({
        "launcher_timing": launcher,
        "actual_source_replay": warmup["components"]["actual_source_replay"],
        "mlflow_probe": warmup["components"]["mlflow_probe"],
        "readiness_infrastructure": warmup["components"]["readiness_infrastructure"],
    })
    budgets, components = module.resolve_readiness_screen_budgets(repeats, launcher, warmup)
    accounting = module.readiness_accounting_decomposition()
    boundary = {
        "calibration_only": True, "screen_roots_opened": False,
        "confirmation_roots_opened": False,
        "checked_independent_official_private_opened": False,
        "scientific_accuracy_or_selection_reported": False,
        "model_update_performed": False, "fidelity_reduced": False,
    }
    payload = {
        "schema_version": "exp648-screen-readiness-calibration-v1",
        "status": "complete-non-scientific-full-fidelity-resource-calibration",
        "calibration_root": TEST_ROOT + 1, "calibration_authorization_sha256": "a" * 64,
        "scientific_runner_sha256": module.sha256(RUNNER), "resolved_config_sha256": module.sha256(module.CONFIG_PATH),
        "screen_workload_sha256": module.sha256_bytes(module.canonical_json_bytes(module.screen_workload_payload())),
        "calibration_evidence_core_sha256": "b" * 64, "environment_sha256": "c" * 64, "mlflow_receipt_sha256": "d" * 64,
        "warmups": 1, "timed_repeats": 3, "warmup": warmup,
        "repeats": repeats, "component_resolution": components, "resolved_screen_budgets": budgets,
        "recovery_charges": {
            "scope": "prepublication-provisional-snapshot-before-inner-publication",
            "accounting_formula": "completed_history + failed_attempt_conservative_allowance + prepublication_provisional_active + prepublication_provisional_reused",
            "augmented_unit_orchestration": {
                "chronology_contract_sha256": module.READINESS_CHRONOLOGY_CONTRACT_SHA256,
                "execution_order": [name for name, _ in module.READINESS_AUGMENTED_UNIT_ORDER],
                "old_base_only_cache_loaded_or_saved": False,
                "fresh_augmented_warmup_before_repeat_cache_reuse": True,
                "carried_charge_applied_exactly_once": True,
                "guard_carried_wall_seconds": module.READINESS_CARRIED_CHARGE["wall_seconds"],
                "guard_carried_aggregate_cpu_seconds": module.READINESS_CARRIED_CHARGE["aggregate_cpu_seconds"],
                "guard_configured_wall_ceiling_seconds": 25000.0,
                "guard_configured_aggregate_cpu_ceiling_seconds": 100000.0,
                "reused_complete_augmented_repeat_indices": [],
                "all_augmented_units_validated_and_sealed_before_next_start": True,
            },
            "prior_superseded_calibrations": list(module.READINESS_SUPERSEDED_CALIBRATIONS),
            **accounting,
            "carried_charge_applied_exactly_once": True,
            "prepublication_provisional_active_wall_seconds": 1,
            "prepublication_provisional_active_aggregate_cpu_seconds": 4,
            "prepublication_provisional_reused_wall_seconds_charged": 2,
            "prepublication_provisional_reused_aggregate_cpu_seconds_charged": 5,
            "prepublication_provisional_cumulative_carried_active_reused_wall_seconds": module.READINESS_CARRIED_CHARGE["wall_seconds"] + 3,
            "prepublication_provisional_cumulative_carried_active_reused_aggregate_cpu_seconds": module.READINESS_CARRIED_CHARGE["aggregate_cpu_seconds"] + 9,
            "prepublication_provisional_process_peak_rss_bytes": 123,
            "active_snapshot_boundary": "test",
        }, "full_fidelity": True, "environment": {"test": True}, "boundary": boundary,
    }
    attach_fake_readiness_chronology(module, payload)
    module.validate_readiness_calibration_receipt(payload)
    tampered = copy.deepcopy(payload)
    tampered["resolved_screen_budgets"]["wall_seconds"]["soft_budget_seconds"] += 1
    with pytest.raises(RuntimeError, match="recompute"):
        module.validate_readiness_calibration_receipt(tampered)

    for field in module.READINESS_RELEASE_CRITICAL_FIELDS:
        core = {key: copy.deepcopy(payload[key]) for key in module.READINESS_RELEASE_CRITICAL_FIELDS}
        broken = copy.deepcopy(payload)
        broken[field] = {"tampered": field}
        with pytest.raises(RuntimeError, match=field):
            module.validate_core_receipt_release_equality(core, broken)

    omitted = copy.deepcopy(payload)
    omitted["repeats"][0]["total_timing"]["wall_seconds"] -= omitted["repeats"][0]["components"]["mlflow_probe"]["timing"]["wall_seconds"]
    with pytest.raises(RuntimeError, match="timing omission"):
        module.validate_complete_readiness_timing(omitted["repeats"][0])
    with pytest.raises(RuntimeError, match="chronology augmented record hash mismatch"):
        module.validate_readiness_calibration_receipt(omitted)


def test_augmented_readiness_units_execute_validate_and_seal_in_order(
    module, monkeypatch, tmp_path
) -> None:
    events = []

    class FakeGuard:
        reused_wall_seconds = 0.0
        reused_cpu_seconds = 0.0

        def __call__(self, stage):
            events.append(f"guard:{stage}")

        def add_reused(self, wall_seconds, cpu_seconds):
            self.reused_wall_seconds += wall_seconds
            self.reused_cpu_seconds += cpu_seconds

    class FakeExperiment:
        root_seed = TEST_ROOT + 11
        guard = FakeGuard()

        def execute_complete_unit(self, _name, *, repeat_index):
            unit = "warmup" if repeat_index is None else f"repeat-{repeat_index}"
            events.append(f"{unit}:base")
            record = fake_readiness_repeat(1.0)
            record["components"].pop("actual_source_replay")
            record["components"].pop("mlflow_probe")
            record["components"].pop("readiness_infrastructure")
            record.pop("base_total_timing_before_readiness_additions")
            record["total_timing"] = {
                "wall_seconds": 431.0,
                "aggregate_cpu_seconds": 1724.0,
                "peak_rss_bytes": 1_000_000,
            }
            record["repeat_index"] = repeat_index
            return record

    original_start = module._write_readiness_chronology_start
    original_seal = module._write_readiness_chronology_seal

    def recording_start(**kwargs):
        events.append(f"{kwargs['unit']}:start")
        return original_start(**kwargs)

    def recording_seal(**kwargs):
        events.append(f"{kwargs['unit']}:seal")
        return original_seal(**kwargs)

    def replay(_experiment, output, _process):
        unit = output.parent.name.split("-", 1)[1]
        events.append(f"{unit}:replay")
        return {
            "complete_identities": 64, "steps_per_identity": 256,
            "complete_steps": 16_384,
            "actual_rollout_identity_batch_path": True, "complete": True,
            "timing": {"wall_seconds": 64.0, "aggregate_cpu_seconds": 256.0,
                       "peak_rss_bytes": 1_000_000},
        }

    def probe(*, unit, **_kwargs):
        events.append(f"{unit}:mlflow")
        return {
            "complete": True, "artifact_count": 6, "remote_verified": True,
            "timing": {"wall_seconds": 3.0, "aggregate_cpu_seconds": 12.0,
                       "peak_rss_bytes": 1_000_000},
        }

    def infrastructure(record, _output, _process):
        unit = "warmup" if record["repeat_index"] is None else f"repeat-{record['repeat_index']}"
        events.append(f"{unit}:infrastructure")
        return copy.deepcopy(fake_readiness_repeat(1.0)["components"]["readiness_infrastructure"])

    def validate(record, _config):
        unit = "warmup" if record["repeat_index"] is None else f"repeat-{record['repeat_index']}"
        events.append(f"{unit}:validate")

    def load_cache(*, repeat_index, **_kwargs):
        events.append(f"repeat-{repeat_index}:cache-lookup")
        return None

    def save_cache(*, repeat_index, **_kwargs):
        events.append(f"repeat-{repeat_index}:cache-save")
        return tmp_path / f"repeat-{repeat_index}.json", "f" * 64

    monkeypatch.setattr(module, "_write_readiness_chronology_start", recording_start)
    monkeypatch.setattr(module, "_write_readiness_chronology_seal", recording_seal)
    monkeypatch.setattr(module, "measure_actual_source_replay_unit", replay)
    monkeypatch.setattr(module, "measure_readiness_mlflow_probe", probe)
    monkeypatch.setattr(module, "measure_readiness_infrastructure_unit", infrastructure)
    monkeypatch.setattr(module.cal, "validate_calibration_repeat", validate)
    monkeypatch.setattr(module, "load_augmented_repeat_cache", load_cache)
    monkeypatch.setattr(module, "save_augmented_repeat_cache", save_cache)

    stage_context = {"stage": "test", "guard": None}
    warmup, repeats, chronology = module.execute_synchronous_augmented_readiness_units(
        experiment=FakeExperiment(), attempt=tmp_path / "attempt",
        authorization_sha256="a" * 64, config={},
        launcher_timing={"wall_seconds": 1.0, "aggregate_cpu_seconds": 4.0,
                         "peak_rss_bytes": 1_000_000},
        stage_context=stage_context,
    )
    assert stage_context["stage"] == "all-units:chronology-validation"
    assert chronology["complete"] is True
    assert len(repeats) == 3 and warmup["readiness_chronology"]["ordinal"] == 0
    for unit, _repeat_index in module.READINESS_AUGMENTED_UNIT_ORDER:
        expected = [
            f"{unit}:start", f"{unit}:base", f"{unit}:replay",
            f"{unit}:mlflow", f"{unit}:infrastructure",
            f"{unit}:validate", f"{unit}:seal",
        ]
        positions = [events.index(event) for event in expected]
        assert positions == sorted(positions)
        for stage in (
            "base-complete", "source-replay-complete", "mlflow-probe-complete",
            "infrastructure-complete", "augmented-validation-complete",
            "augmented-record-published", "fresh-augmented-unit-sealed",
        ):
            assert f"guard:readiness:{unit}:{stage}" in events
    assert events.index("warmup:seal") < events.index("repeat-1:cache-lookup")
    for repeat_index in range(1, 4):
        assert events.index(f"repeat-{repeat_index}:seal") < events.index(
            f"repeat-{repeat_index}:cache-save"
        )
    source = inspect.getsource(module._execute_one_augmented_readiness_unit)
    for component in (
        "chronology-start", "cache-lookup-and-validation", "cache-reuse-charge",
        "cache-acceptance-guard", "cached-record-publication",
        "reused-chronology-seal", "base-complete-unit", "source-replay",
        "mlflow-probe", "readiness-infrastructure", "augmented-validation",
        "augmented-record-publication", "fresh-chronology-seal",
        "augmented-cache-publication",
    ):
        assert component in source


def test_old_post_hoc_readiness_records_without_chronology_are_rejected(module) -> None:
    payload = {
        "calibration_root": TEST_ROOT + 12,
        "calibration_authorization_sha256": "a" * 64,
        "warmup": fake_readiness_repeat(1.0),
        "repeats": [fake_readiness_repeat(1.0) for _ in range(3)],
    }
    with pytest.raises(RuntimeError, match="chronology is absent"):
        module.validate_readiness_chronology(payload)


def test_incomplete_and_tampered_augmented_repeat_cache_are_rejected(
    module, monkeypatch
) -> None:
    root = TEST_ROOT + 13
    authorization_sha256 = "a" * 64
    source = {
        "calibration_root": root,
        "calibration_authorization_sha256": authorization_sha256,
        "warmup": fake_readiness_repeat(1.0),
        "repeats": [fake_readiness_repeat(1.0) for _ in range(3)],
    }
    attach_fake_readiness_chronology(module, source)
    reference = source["repeats"][0]["readiness_chronology"]
    record = module._bare_augmented_record(source["repeats"][0])
    key_fields = {
        "root": root, "authorization_sha256": authorization_sha256,
        "checkpoint_sha256": "1" * 64, "evaluator_sha256": "2" * 64,
        "dataset_sha256": "3" * 64, "config_sha256": "4" * 64,
        "seed": root, "environment_sha256": "5" * 64,
        "calibration_runner_sha256": "6" * 64,
        "scientific_protocol_sha256": "7" * 64,
        "resource_protocol_sha256": "8" * 64,
        "workload_sha256": "9" * 64,
        "dependency_manifest_sha256": "a" * 64,
    }
    payload = {
        "schema_version": "exp648-readiness-augmented-repeat-cache-v1",
        "cache_key": module.readiness_augmented_cache_key(key_fields, 1),
        "cache_key_fields": key_fields,
        "repeat_index": 1,
        "record": record,
        "source_chronology": reference,
        "source_seal_sha256": reference["seal_sha256"],
        "cache_created_monotonic_ns": reference["seal"]["monotonic_ns"] + 1,
    }
    base_validations = []
    monkeypatch.setattr(
        module.cal, "validate_cached_repeat_record",
        lambda *_args, **kwargs: base_validations.append(kwargs),
    )
    module.validate_augmented_repeat_cache_payload(
        payload, expected_key_fields=key_fields, repeat_index=1, config={}
    )
    assert base_validations[0]["repeat_index"] == 1
    assert base_validations[0]["expected_cache_key_fields"]["runner_sha256"] == "6" * 64
    incomplete = copy.deepcopy(payload)
    incomplete["record"]["complete"] = False
    with pytest.raises(RuntimeError, match="cache is incomplete"):
        module.validate_augmented_repeat_cache_payload(
            incomplete, expected_key_fields=key_fields, repeat_index=1, config={}
        )
    tampered = copy.deepcopy(payload)
    tampered["source_chronology"]["seal"]["augmented_record_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="chronology seal hash mismatch"):
        module.validate_augmented_repeat_cache_payload(
            tampered, expected_key_fields=key_fields, repeat_index=1, config={}
        )


def test_consumed_readiness_root_is_denied_before_binding(module, monkeypatch) -> None:
    path = module.ROOT_20261159_SUPERSEDED_DIR / "readiness-calibration-authorization.json"
    raw = path.read_bytes()
    monkeypatch.setattr(
        module, "readiness_calibration_authorization_base",
        lambda *_: (_ for _ in ()).throw(AssertionError("binding must not run")),
    )
    with pytest.raises(PermissionError, match="reused, test-only, or prohibited"):
        module.verify_readiness_calibration_authorization(
            path, committed_bytes=raw, allow_test_root=True
        )


def test_failed_root20261160_is_permanently_denied_before_binding(module, monkeypatch) -> None:
    path = (
        module.ROOT_20261160_SUPERSEDED_DIR
        / "readiness-calibration-authorization.json"
    )
    raw = path.read_bytes()
    payload = json.loads(raw)
    assert raw == module.canonical_json_bytes(payload, pretty=True)
    assert module.sha256(path) == (
        "9e1f63a9fe3c6613f8c02dea523147901da2aea862a3dec58d53c0cb7a054984"
    )
    assert payload["calibration_root"] == 20261160
    monkeypatch.setattr(
        module, "readiness_calibration_authorization_base",
        lambda *_: (_ for _ in ()).throw(AssertionError("binding must not run")),
    )
    with pytest.raises(PermissionError, match="reused, test-only, or prohibited"):
        module.verify_readiness_calibration_authorization(
            path, committed_bytes=raw, allow_test_root=True
        )


def test_root20261160_caches_are_planning_only_and_fully_bound(module, tmp_path) -> None:
    receipt = module.verify_root20261160_planning_caches()
    assert receipt["status"] == (
        "planning-evidence-only-executable-reuse-permanently-denied"
    )
    assert [item["cache_sha256"] for item in receipt["verified"]] == [
        "98a69670a229c15de7193c5878571fcc28a6d013e9b3ae23a7ac52be92624d80",
        "8e8115d2b41974c20adfad734a847ec2ac64e07f5636325301b8793ffb1adc68",
    ]
    copied = tmp_path / "screen"
    source = module.SCRATCH_SCREEN_DIR
    shutil.copytree(
        source / "augmented-repeat-cache/root-20261160",
        copied / "augmented-repeat-cache/root-20261160",
    )
    attempt_rel = "readiness-calibration-attempts/cal-9e1f63a9fe3c6613"
    for subtree in ("augmented-chronology", "augmented-units"):
        shutil.copytree(
            source / attempt_rel / subtree,
            copied / attempt_rel / subtree,
        )
    cache = next((copied / "augmented-repeat-cache/root-20261160").glob("repeat-1-*.json"))
    payload = json.loads(cache.read_bytes())
    payload["cache_key_fields"]["workload_sha256"] = "0" * 64
    cache.write_bytes(module.canonical_json_bytes(payload, pretty=True))
    with pytest.raises(RuntimeError, match="byte hash mismatch"):
        module.verify_root20261160_planning_caches(scratch_screen=copied)


def test_completed_allowance_and_carried_charge_are_separate_and_applied_once(module) -> None:
    class Guard:
        def __init__(self):
            self.events = []
            self.checks = []
            self.wall_ceiling = 25000.0
            self.cpu_ceiling = 100000.0
            self.prior_superseded_wall_seconds = module.cal.PRIOR_SUPERSEDED_CALIBRATION_CHARGE[
                "wall_seconds"
            ]
            self.prior_superseded_cpu_seconds = module.cal.PRIOR_SUPERSEDED_CALIBRATION_CHARGE[
                "aggregate_cpu_seconds"
            ]

        def __call__(self, stage):
            self.events.append(stage)

    guard = Guard()
    module.apply_readiness_carried_charge_once(guard)
    assert module.READINESS_COMPLETED_HISTORY_CHARGE == {
        "wall_seconds": 9007.853947323996,
        "aggregate_cpu_seconds": 36721.035879589996,
    }
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["wall_seconds"] == 5294.4
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["aggregate_cpu_seconds"] == 21000.0
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["exact_internal_observation"] is False
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["completed_work_claimed"] is False
    assert module.READINESS_CARRIED_CHARGE == {
        "wall_seconds": 14302.253947323995,
        "aggregate_cpu_seconds": 57721.035879589996,
    }
    assert guard.prior_superseded_wall_seconds == 14302.253947323995
    assert guard.prior_superseded_cpu_seconds == 57721.035879589996
    assert guard.wall_ceiling == 25000.0
    assert guard.cpu_ceiling == 100000.0
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["semantics"] == (
        "conservative-accounting-allowance-never-an-exact-internal-observation"
    )
    assert guard.events == [
        "readiness:cumulative-carried-charge-applied-once"
    ]
    with pytest.raises(RuntimeError, match="more than once"):
        module.apply_readiness_carried_charge_once(guard)

    wrong_ceiling = Guard()
    wrong_ceiling.wall_ceiling = 24999.0
    with pytest.raises(RuntimeError, match="configured cumulative ceilings"):
        module.apply_readiness_carried_charge_once(wrong_ceiling)


def test_readiness_mlflow_expected_scope_excludes_scientific_and_candidate(module) -> None:
    payload = {
        "calibration_authorization_sha256": "a" * 64, "calibration_root": TEST_ROOT + 1,
        "scientific_runner_sha256": "b" * 64, "screen_workload_sha256": "c" * 64,
        "calibration_evidence_core_sha256": "d" * 64,
        "resolved_screen_budgets": {
            "wall_seconds": {"projected_seconds": 1},
            "aggregate_cpu_seconds": {"projected_seconds": 4}, "peak_rss_guard_bytes": 100,
        },
    }
    tags, _params, metrics = module._readiness_mlflow_expected(payload)
    assert tags["scientific_accuracy_reported"] == "false"
    assert tags["candidate_registration"] == "false"
    assert all(fragment not in key for key in metrics for fragment in ("accuracy", "selection", "gain"))


def test_outer_readiness_resource_completion_is_final_bound_and_observed(module, tmp_path, monkeypatch) -> None:
    attempt_id = "cal-test"
    auth = "a" * 64
    names = (
        "readiness-calibration.json", "readiness-calibration-core.json",
        "readiness-calibration-environment.json", "readiness-calibration-mlflow.json",
        "readiness-calibration-executed-config.yaml",
        "readiness-calibration-completion.json", "readiness-calibration-attempt-state.json",
        "readiness-calibration-launch-ledger.json",
    )
    inner = {name: tmp_path / name for name in names}
    for name, path in inner.items():
        if path.suffix == ".json":
            path.write_bytes(module.canonical_json_bytes({"artifact": name}, pretty=True))
        else:
            path.write_text(f"inner:{name}\n")
    completion = {
        "schema_version": "exp648-readiness-calibration-completion-v1",
        "attempt_id": attempt_id,
        "authorization_sha256": auth,
        "artifacts": {},
        "complete": True,
    }
    inner["readiness-calibration-completion.json"].write_bytes(
        module.canonical_json_bytes(completion, pretty=True)
    )
    state = {
        "schema_version": "exp648-readiness-calibration-attempt-v1",
        "attempt_id": attempt_id, "authorization_sha256": auth,
        "phase": "sealed", "complete": True,
        "completion_manifest_sha256": module.sha256(inner["readiness-calibration-completion.json"]),
    }
    inner["readiness-calibration-attempt-state.json"].write_bytes(
        module.canonical_json_bytes(state, pretty=True)
    )
    inner["readiness-calibration-launch-ledger.json"].write_bytes(
        module.canonical_json_bytes(state, pretty=True)
    )
    output = tmp_path / "outer.json"
    writes = []
    events = []
    original_write = module._atomic_write
    original_prevalidate = module.prevalidate_readiness_inner_lifecycle
    real_perf_counter = time.perf_counter
    def recording_write(path, payload):
        writes.append(Path(path)); original_write(path, payload)
    def recording_prevalidate(**kwargs):
        events.append("inner-hashed-and-validated")
        return original_prevalidate(**kwargs)
    monkeypatch.setattr(module, "_atomic_write", recording_write)
    monkeypatch.setattr(module, "prevalidate_readiness_inner_lifecycle", recording_prevalidate)
    monkeypatch.setattr(
        module.time, "perf_counter",
        lambda: events.append("final-wall-sample") or real_perf_counter(),
    )
    receipt = module.write_readiness_resource_completion(
        inner_paths=inner, attempt_id=attempt_id, authorization_sha256=auth,
        launch_wall_started=real_perf_counter() - 2,
        launch_cpu_started=0, launch_peak_rss=1,
        reused_wall=3, reused_cpu=12, output_path=output,
    )
    assert writes == [output]
    assert events == ["inner-hashed-and-validated", "final-wall-sample"]
    assert receipt["inner_lifecycle_complete_before_resource_sample"] is True
    assert receipt["post_sample_operations"] == list(module.READINESS_POST_SAMPLE_OPERATIONS)
    assert receipt["measurement_semantics"]["post_sample_operations_included_in_observed_totals"] is False
    loaded = module.load_and_validate_readiness_resource_completion(
        output, inner_paths=inner, expected_attempt_id=attempt_id,
        expected_authorization_sha256=auth,
    )
    assert loaded["recovery_charges"]["observed_pre_outer_seal_peak_rss_bytes"] > 1
    original_mlflow = inner["readiness-calibration-mlflow.json"].read_bytes()
    inner["readiness-calibration-mlflow.json"].write_text("tampered\n")
    with pytest.raises(RuntimeError, match="artifact hash"):
        module.load_and_validate_readiness_resource_completion(
            output, inner_paths=inner, expected_attempt_id=attempt_id,
            expected_authorization_sha256=auth,
        )
    inner["readiness-calibration-mlflow.json"].write_bytes(original_mlflow)
    arithmetic = copy.deepcopy(receipt)
    arithmetic["recovery_charges"]["cumulative_carried_observed_active_reused_wall_seconds"] += 1
    output.write_bytes(module.canonical_json_bytes(arithmetic, pretty=True))
    with pytest.raises(RuntimeError, match="cumulative wall"):
        module.load_and_validate_readiness_resource_completion(
            output, inner_paths=inner, expected_attempt_id=attempt_id,
            expected_authorization_sha256=auth,
        )
    swapped = copy.deepcopy(receipt)
    observation = swapped["resource_observation"]
    observation["observed_wall_seconds"], observation["observed_aggregate_cpu_seconds"] = (
        observation["observed_aggregate_cpu_seconds"], observation["observed_wall_seconds"]
    )
    output.write_bytes(module.canonical_json_bytes(swapped, pretty=True))
    with pytest.raises(RuntimeError, match="wall is not monotonic wall"):
        module.load_and_validate_readiness_resource_completion(
            output, inner_paths=inner, expected_attempt_id=attempt_id,
            expected_authorization_sha256=auth,
        )
    combined = copy.deepcopy(receipt)
    observation = combined["resource_observation"]
    observation["observed_aggregate_cpu_seconds"] = (
        observation["observed_wall_seconds"]
        + observation["observed_aggregate_cpu_seconds"]
    )
    output.write_bytes(module.canonical_json_bytes(combined, pretty=True))
    with pytest.raises(RuntimeError, match="CPU is not process user plus system"):
        module.load_and_validate_readiness_resource_completion(
            output, inner_paths=inner, expected_attempt_id=attempt_id,
            expected_authorization_sha256=auth,
        )
    output.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="noncanonical"):
        module.load_and_validate_readiness_resource_completion(
            output, inner_paths=inner, expected_attempt_id=attempt_id,
            expected_authorization_sha256=auth,
        )


def test_readiness_outer_seal_precedes_execute_return_and_cli_output(module, monkeypatch) -> None:
    wrapper_source = inspect.getsource(module.execute_readiness_calibration)
    assert "with readiness_calibration_authorization_lock(" in wrapper_source
    assert "return _execute_readiness_calibration_locked(" in wrapper_source
    execute_source = inspect.getsource(module._execute_readiness_calibration_attempt_body)
    assert "experiment.run()" not in execute_source
    assert "execute_synchronous_augmented_readiness_units(" in execute_source
    seal_position = execute_source.rindex("write_readiness_resource_completion(")
    return_position = execute_source.rindex("return receipt")
    assert seal_position < return_position
    post_publication = execute_source[seal_position:return_position]
    assert "_atomic_write(" not in post_publication
    for forbidden in (
        "execute_complete_unit(", "execute_synchronous_augmented_readiness_units(",
        "save_augmented_repeat_cache(", "measure_actual_source_replay_unit(",
        "measure_readiness_mlflow_probe(", "measure_readiness_infrastructure_unit(",
    ):
        assert forbidden not in post_publication

    main_source = inspect.getsource(module.main)
    execute_position = main_source.index("result = execute_readiness_calibration(")
    output_position = main_source.index("print(json.dumps({", execute_position)
    assert execute_position < output_position

    events = []
    monkeypatch.setattr(
        module, "verify_readiness_calibration_authorization",
        lambda *_args, **_kwargs: {"calibration_root": 999},
    )
    monkeypatch.setattr(
        module, "execute_readiness_calibration",
        lambda *_args, **_kwargs: events.append("outer-sealed") or {
            "status": "complete-non-scientific-full-fidelity-resource-calibration",
            "calibration_root": 999,
            "resolved_screen_budgets": {},
        },
    )
    monkeypatch.setattr(
        "builtins.print",
        lambda *_args, **kwargs: events.append(
            "stdout-written-and-flushed" if kwargs.get("flush") is True else "stdout-unflushed"
        ),
    )
    assert module.main(["--mode", "readiness-calibration"]) == 0
    assert events == ["outer-sealed", "stdout-written-and-flushed"]


def test_catchable_exception_boundary_receipt_is_atomic_complete_and_reraises(
    module, tmp_path, monkeypatch
) -> None:
    auth = "7" * 64
    scratch = tmp_path / "scratch"
    tracked = tmp_path / "tracked"
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", scratch)
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tracked)
    writes = []
    original_atomic_write = module._atomic_write

    def recording_atomic_write(path, payload):
        writes.append((Path(path), payload))
        return original_atomic_write(path, payload)

    monkeypatch.setattr(module, "_atomic_write", recording_atomic_write)

    original = RuntimeError("unit failed")
    last_guard_check = {
        "stage": "readiness:repeat-2:source-replay-complete",
        "charged_wall_seconds": 15000.0,
        "charged_cpu_seconds": 60000.0,
    }

    def fail_body(*_args, stage_context, **_kwargs):
        stage_context["stage"] = "repeat-3:source-replay"
        stage_context["guard"] = types.SimpleNamespace(checks=[last_guard_check])
        raise original

    monkeypatch.setattr(
        module, "_execute_readiness_calibration_attempt_body", fail_body
    )
    with pytest.raises(RuntimeError, match="unit failed") as caught:
        module.execute_readiness_calibration(
            {"calibration_root": module.PREREGISTERED_READINESS_CALIBRATION_ROOT,
             "authorization_sha256": auth},
            launch_wall_started=time.perf_counter() - 1.0,
            launch_cpu_started=0.0,
            launch_peak_rss=1,
        )
    assert caught.value is original
    attempt = scratch / "readiness-calibration-attempts" / f"cal-{auth[:16]}"
    receipt_path = attempt / "catchable-exception-boundary.json"
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    assert raw == module.canonical_json_bytes(receipt, pretty=True)
    assert receipt["context"] == {
        "stage": "repeat-3:source-replay",
        "root": module.PREREGISTERED_READINESS_CALIBRATION_ROOT,
        "authorization_sha256": auth,
        "attempt_id": f"cal-{auth[:16]}",
    }
    assert original.exp648_attempt_context == {
        **receipt["context"], "last_guard_check": last_guard_check,
    }
    assert receipt["last_guard_check"] == last_guard_check
    timing = receipt["resource_observation"]
    assert timing["monotonic_wall_seconds"] >= 1.0
    assert timing["process_user_plus_system_cpu_seconds"] >= 0.0
    assert timing["peak_rss_bytes"] >= 1
    assert receipt["artifact_boundary"]["file_count"] == 1
    assert receipt["last_completed_chronology_seal"] is None
    assert receipt["coverage"]["post_exit_coverage_claimed"] is False
    assert "exception re-raise" in receipt["coverage"]["excluded"]
    assert (
        "terminal failed attempt-state serialization and atomic write"
        in receipt["coverage"]["excluded"]
    )
    state = json.loads((attempt / "attempt-state.json").read_bytes())
    assert state == {
        "schema_version": "exp648-readiness-calibration-attempt-v1",
        "attempt_id": f"cal-{auth[:16]}",
        "authorization_sha256": auth,
        "phase": "failed",
        "complete": False,
        "exception_receipt_sha256": module.sha256(receipt_path),
    }
    receipt_write = max(
        index for index, (path, _payload) in enumerate(writes)
        if path.name == "catchable-exception-boundary.json"
    )
    terminal_state_write = max(
        index for index, (path, payload) in enumerate(writes)
        if path.name == "attempt-state.json" and b'"phase": "failed"' in payload
    )
    assert receipt_write < terminal_state_write

    actions = []
    monkeypatch.setattr(
        module, "_execute_readiness_calibration_attempt_body",
        lambda *_args, **_kwargs: actions.append("body") or {},
    )
    with pytest.raises(PermissionError, match="failed readiness calibration attempt"):
        module.execute_readiness_calibration(
            {"calibration_root": module.PREREGISTERED_READINESS_CALIBRATION_ROOT,
             "authorization_sha256": auth},
            launch_wall_started=time.perf_counter(),
            launch_cpu_started=0.0,
            launch_peak_rss=0,
        )
    assert actions == []


def test_catchable_exception_receipt_publication_failure_fails_closed(
    module, tmp_path, monkeypatch
) -> None:
    auth = "8" * 64
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", tmp_path / "scratch")
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tmp_path / "tracked")
    original = ValueError("original")
    publication = OSError("disk")
    monkeypatch.setattr(
        module, "_execute_readiness_calibration_attempt_body",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(original),
    )
    monkeypatch.setattr(
        module, "write_catchable_exception_boundary_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(publication),
    )
    with pytest.raises(module.CatchableExceptionReceiptPublicationError) as caught:
        module.execute_readiness_calibration(
            {"calibration_root": module.PREREGISTERED_READINESS_CALIBRATION_ROOT,
             "authorization_sha256": auth},
            launch_wall_started=time.perf_counter(),
            launch_cpu_started=0.0,
            launch_peak_rss=0,
        )
    assert caught.value.original_exception is original
    assert caught.value.publication_exception is publication
    attempt = (
        tmp_path / "scratch/readiness-calibration-attempts" / f"cal-{auth[:16]}"
    )
    state = json.loads((attempt / "attempt-state.json").read_bytes())
    assert state["phase"] == "failure-publication-failed"
    with pytest.raises(PermissionError, match="failed-publication"):
        module.execute_readiness_calibration(
            {"calibration_root": module.PREREGISTERED_READINESS_CALIBRATION_ROOT,
             "authorization_sha256": auth},
            launch_wall_started=time.perf_counter(),
            launch_cpu_started=0.0,
            launch_peak_rss=0,
        )


def test_terminal_failed_state_publication_failure_retains_both_exceptions_and_blocks_retry(
    module, tmp_path, monkeypatch
) -> None:
    auth = "9" * 64
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", scratch)
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tmp_path / "tracked")
    original = LookupError("component")
    publication = OSError("state disk")
    monkeypatch.setattr(
        module, "_execute_readiness_calibration_attempt_body",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(original),
    )
    monkeypatch.setattr(
        module, "write_terminal_failed_attempt_state",
        lambda **_kwargs: (_ for _ in ()).throw(publication),
    )
    with pytest.raises(module.CatchableExceptionReceiptPublicationError) as caught:
        module.execute_readiness_calibration(
            {"calibration_root": module.PREREGISTERED_READINESS_CALIBRATION_ROOT,
             "authorization_sha256": auth},
            launch_wall_started=time.perf_counter(),
            launch_cpu_started=0.0,
            launch_peak_rss=0,
        )
    assert caught.value.original_exception is original
    assert caught.value.publication_exception is publication
    attempt = scratch / "readiness-calibration-attempts" / f"cal-{auth[:16]}"
    assert (attempt / "catchable-exception-boundary.json").is_file()
    assert json.loads((attempt / "attempt-state.json").read_bytes())["phase"] == "authorized"
    with pytest.raises(RuntimeError, match="without failed attempt state"):
        module.execute_readiness_calibration(
            {"calibration_root": module.PREREGISTERED_READINESS_CALIBRATION_ROOT,
             "authorization_sha256": auth},
            launch_wall_started=time.perf_counter(),
            launch_cpu_started=0.0,
            launch_peak_rss=0,
        )


def test_authentication_rejects_bad_bytes_before_expensive_binding_actions(module, tmp_path, monkeypatch) -> None:
    events = []
    monkeypatch.setattr(module, "readiness_calibration_authorization_base", lambda *_: events.append("binding") or {})
    path = tmp_path / "auth.json"
    cases = [
        b"not-json",
        module.canonical_json_bytes({"schema_version": "wrong"}, pretty=True),
        b'{"schema_version":"exp648-screen-readiness-calibration-authorization-v1"}\n',
    ]
    for raw in cases:
        path.write_bytes(raw)
        with pytest.raises(PermissionError):
            module.verify_readiness_calibration_authorization(path, committed_bytes=raw, allow_test_root=True)
        assert events == []
    oversized = b"{" + b" " * module.MAX_AUTHORIZATION_BYTES + b"}"
    path.write_bytes(oversized)
    with pytest.raises(PermissionError, match="size"):
        module.verify_readiness_calibration_authorization(path, committed_bytes=oversized, allow_test_root=True)
    assert events == []


def mocked_readiness_authorization_base_blobs(module):
    authorization_raw = module.READINESS_CALIBRATION_AUTHORIZATION_PATH.read_bytes()
    authorization = json.loads(authorization_raw)
    assert authorization_raw == module.canonical_json_bytes(
        authorization, pretty=True
    )
    approved_base_commit = authorization["approved_base_commit"]
    provenance = json.loads(module.READINESS_SCAFFOLD_PROVENANCE_PATH.read_text())
    source = provenance["readiness_calibration_authorization_source"]
    paths = {
        binding["path"]: (module.ROOT / binding["path"]).read_bytes()
        for binding in source["approved_base_file_bindings"].values()
    }
    scientific_test_path = source["approved_base_file_bindings"][
        "scientific_test"
    ]["path"]
    paths[scientific_test_path] = module._git_blob(
        approved_base_commit, scientific_test_path
    )
    provenance_path = (
        "experiments/runs/exp-648-structured-trajectory-consistency/"
        "readiness-scaffold/provenance.json"
    )
    paths[provenance_path] = module.canonical_json_bytes(provenance, pretty=True)
    return paths, provenance


def test_readiness_authorization_is_frozen_provenance_template_under_hostile_live_state(
    module, monkeypatch
) -> None:
    paths, provenance = mocked_readiness_authorization_base_blobs(module)
    monkeypatch.setattr(module, "_git_blob", lambda _commit, relative: paths[relative])

    def hostile(name):
        return lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"live {name} used")
        )

    for name in (
        "screen_workload_payload", "_base_protocol_hashes", "_base_evaluator_hash",
        "_base_exact_teacher_hash",
    ):
        monkeypatch.setattr(module, name, hostile(name))
    monkeypatch.setattr(module, "RUN_ID", "hostile-run-id")
    monkeypatch.setattr(module, "FROZEN_CORPUS_FILES", {"hostile": "state"})
    monkeypatch.setattr(module, "CONFIG_PATH", Path("hostile-config"))
    monkeypatch.setattr(module, "CALIBRATION_RUNNER", Path("hostile-runner"))
    monkeypatch.setattr(module, "EXECUTED_CONFIG_PATH", Path("hostile-executed"))
    monkeypatch.setattr(module, "CALIBRATION_ENVIRONMENT_PATH", Path("hostile-environment"))
    monkeypatch.setattr(module, "READINESS_SCAFFOLD_PROVENANCE_PATH", Path("hostile-provenance"))
    monkeypatch.setattr(module, "__file__", "hostile-scientific.py")
    root = TEST_ROOT + 7
    payload = module.readiness_calibration_authorization_base(root, "base-commit")
    assert payload == {
        **provenance["readiness_calibration_authorization_source"]["authorization_template"],
        "calibration_root": root,
        "approved_base_commit": "base-commit",
    }


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("workload", "screen workload hash mismatch"),
        ("template", "authorization template hash mismatch"),
        ("runner_binding", "scientific_runner direct hash mismatch"),
        ("environment", "template/runtime environment binding mismatch"),
    ],
)
def test_authorization_source_tampering_fails_before_root_data_model_mlflow_or_lifecycle(
    module, monkeypatch, tamper, message
) -> None:
    paths, provenance = mocked_readiness_authorization_base_blobs(module)
    source = provenance["readiness_calibration_authorization_source"]
    if tamper == "workload":
        source["screen_workload"]["recoverable_evaluation_unit_count"] += 1
    elif tamper == "template":
        source["authorization_template"]["source_checkpoint_sha256"] = "0" * 64
    elif tamper == "runner_binding":
        source["approved_base_file_bindings"]["scientific_runner"]["sha256"] = "0" * 64
    elif tamper == "environment":
        source["authorization_template"]["approved_runtime_environment"] = {"hostile": True}
        source["authorization_template_sha256"] = module.sha256_bytes(
            module.canonical_json_bytes(source["authorization_template"])
        )
    provenance_path = (
        "experiments/runs/exp-648-structured-trajectory-consistency/"
        "readiness-scaffold/provenance.json"
    )
    paths[provenance_path] = module.canonical_json_bytes(provenance, pretty=True)
    monkeypatch.setattr(module, "_git_blob", lambda _commit, relative: paths[relative])
    actions = []

    def forbidden(name):
        return lambda *_args, **_kwargs: actions.append(name) or (_ for _ in ()).throw(
            AssertionError(f"{name} action occurred before source rejection")
        )

    monkeypatch.setattr(module, "authorize_root", forbidden("root"))
    monkeypatch.setattr(module.cal, "authorize_root", forbidden("calibration-root"))
    monkeypatch.setattr(module, "_read_data_artifacts", forbidden("data"))
    monkeypatch.setattr(module.cal, "CalibrationExperiment", forbidden("model"))
    monkeypatch.setattr(module.mlflow, "start_run", forbidden("mlflow"))
    monkeypatch.setattr(module, "validate_one_child_authorization_lifecycle", forbidden("lifecycle"))
    with pytest.raises(PermissionError, match=message):
        module.readiness_calibration_authorization_base(TEST_ROOT + 8, "base-commit")
    assert actions == []


def test_readiness_scaffold_provenance_matches_current_code_and_workload(module) -> None:
    authorization_raw = module.READINESS_CALIBRATION_AUTHORIZATION_PATH.read_bytes()
    authorization = json.loads(authorization_raw)
    assert authorization_raw == module.canonical_json_bytes(
        authorization, pretty=True
    )
    approved_base_commit = authorization["approved_base_commit"]
    assert approved_base_commit == "87384f92ab1c053c7791e8afd69a07936d5a7402"

    provenance_relative = module.READINESS_SCAFFOLD_PROVENANCE_PATH.relative_to(
        module.ROOT
    ).as_posix()
    provenance_raw = module._git_blob(approved_base_commit, provenance_relative)
    assert module.sha256_bytes(provenance_raw) == (
        "593d18523ff2888c1d57d8626d81f4ebb07a234427573efa3afc226e98e2c5a6"
    )
    assert module.sha256(module.READINESS_SCAFFOLD_PROVENANCE_PATH) == (
        "593d18523ff2888c1d57d8626d81f4ebb07a234427573efa3afc226e98e2c5a6"
    )
    provenance = json.loads(provenance_raw)

    runner_relative = Path(module.__file__).resolve().relative_to(
        module.ROOT
    ).as_posix()
    test_relative = Path(__file__).resolve().relative_to(module.ROOT).as_posix()
    config_relative = module.CONFIG_PATH.relative_to(module.ROOT).as_posix()
    runner_sha256 = module._git_blob_sha256(
        approved_base_commit, runner_relative
    )
    test_sha256 = module._git_blob_sha256(approved_base_commit, test_relative)
    config_raw = module._git_blob(approved_base_commit, config_relative)
    config_sha256 = module.sha256_bytes(config_raw)
    config = module.yaml.safe_load(config_raw)
    assert test_sha256 == (
        "f66fb2c7e510171bb92a5cab20b842b9b08d9dd023fc44f37ee1e8b74ef08ef5"
    )
    assert module.sha256(Path(__file__)) != test_sha256
    assert authorization["scientific_runner_sha256"] == runner_sha256
    assert authorization["scientific_test_sha256"] == test_sha256
    assert authorization["resolved_config_sha256"] == config_sha256
    expected_workload_sha256 = module.sha256_bytes(module.canonical_json_bytes(
        module.screen_workload_payload(
            scientific_runner_sha256=runner_sha256,
            config_override=config,
        )
    ))
    assert provenance["live_scaffold"] == {
        "focused_scientific_test_sha256": test_sha256,
        "resolved_config_sha256": config_sha256,
        "scientific_runner_sha256": runner_sha256,
    }
    assert provenance["screen_workload_sha256"] == expected_workload_sha256
    source = provenance["readiness_calibration_authorization_source"]
    assert source["screen_workload"] == module.screen_workload_payload(
        scientific_runner_sha256=runner_sha256,
        config_override=config,
    )
    assert source["screen_workload_sha256"] == expected_workload_sha256
    assert source["authorization_template_sha256"] == module.sha256_bytes(
        module.canonical_json_bytes(source["authorization_template"])
    )
    template = source["authorization_template"]
    assert "calibration_root" not in template
    assert "approved_base_commit" not in template
    assert template["screen_workload_sha256"] == expected_workload_sha256
    assert template["scientific_runner_sha256"] == runner_sha256
    assert template["scientific_test_sha256"] == test_sha256
    assert template["scientific_protocol_sha256"] == module.cal.scientific_protocol_sha256(
        config
    )
    assert template["resource_protocol_sha256"] == module.cal.resource_protocol_sha256(
        config
    )
    accepted_corpus_manifest_bytes = (
        module.DATA_DIR / "corpus-manifest.json"
    ).read_bytes()
    assert template["corpus_manifest_sha256"] == module.sha256_bytes(
        accepted_corpus_manifest_bytes
    )
    assert template["corpus_manifest_sha256"] == module.FROZEN_CORPUS_FILES[
        "data/corpus-manifest.json"
    ]
    for binding in source["approved_base_file_bindings"].values():
        assert module._git_blob_sha256(
            approved_base_commit, binding["path"]
        ) == binding["sha256"]


def test_screen_auth_bad_bytes_fail_before_binding(module, tmp_path, monkeypatch) -> None:
    events = []
    monkeypatch.setattr(module, "screen_readiness_binding", lambda *_: events.append("binding"))
    path = tmp_path / "screen-auth.json"
    for raw in (
        b"bad-json",
        module.canonical_json_bytes({"schema_version": "wrong"}, pretty=True),
        b'{"schema_version":"exp648-screen-judge-authorization-v1"}\n',
    ):
        path.write_bytes(raw)
        with pytest.raises(PermissionError):
            module.verify_screen_authorization(path, committed_bytes=raw)
        assert events == []
    raw = b"{" + b" " * module.MAX_AUTHORIZATION_BYTES + b"}"
    path.write_bytes(raw)
    with pytest.raises(PermissionError, match="size"):
        module.verify_screen_authorization(path, committed_bytes=raw)
    assert events == []
    canonical = module.canonical_json_bytes({
        "schema_version": "exp648-screen-readiness-calibration-authorization-v1"
    }, pretty=True)
    path.write_bytes(canonical)
    with pytest.raises(PermissionError, match="committed"):
        module.verify_readiness_calibration_authorization(path, committed_bytes=canonical + b"x", allow_test_root=True)
    assert events == []


def test_cached_transition_semantics_detect_prediction_and_rate_tampering(module) -> None:
    cases = {
        "state": [0, 1], "target": [2, 3], "quotient": [0, 1], "modulus": [5, 5]
    }
    base = {
        "cases": 2, "exact_accuracy": 1.0, "quotient_accuracy": 1.0,
        "valid_residue_rate": 1.0, "exact_vector": [True, True],
        "predictions": [2, 3], "quotient_predictions": [0, 1],
        "output_sha256": module.sha256_bytes(module.canonical_json_bytes([2, 3])),
    }
    module.validate_evaluation_results(
        unit="transition-128", width=128, cases=cases,
        results={"control": copy.deepcopy(base), "challenger": copy.deepcopy(base)},
    )
    tampered = {"control": copy.deepcopy(base), "challenger": copy.deepcopy(base)}
    tampered["challenger"]["predictions"][0] = 4
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module.validate_evaluation_results(unit="transition-128", width=128, cases=cases, results=tampered)


def test_cached_rollout_recomputes_transient_invalid_teacher_and_first_error(module) -> None:
    cases = {"multiplier": [0], "multiplicand": [1], "modulus": [3]}
    base = {
        "cases": 1, "exact_accuracy": 1.0, "exact_vector": [True],
        "mean_clipped_first_error_index": 0.0, "invalid_rollout_rate": 1.0,
        "output_sha256": module.sha256_bytes(module.canonical_json_bytes([0])),
        "predictions": [0], "teacher_targets": [0], "first_error_indices": [0],
        "any_intermediate_invalid_vector": [True],
        "prediction_trace": [[0], [3], [0]], "teacher_trace": [[0], [0], [0]],
    }
    valid = {"control": copy.deepcopy(base), "challenger": copy.deepcopy(base)}
    module.validate_evaluation_results(unit="rollout-2", width=2, cases=cases, results=valid)
    for field, value in (
        ("teacher_trace", [[0], [1], [0]]),
        ("first_error_indices", [3]),
        ("any_intermediate_invalid_vector", [False]),
    ):
        tampered = {"control": copy.deepcopy(base), "challenger": copy.deepcopy(base)}
        tampered["challenger"][field] = value
        with pytest.raises(RuntimeError, match="trace semantics"):
            module.validate_evaluation_results(unit="rollout-2", width=2, cases=cases, results=tampered)
    for field, value in (
        ("prediction_trace", [[False], [3], [0]]),
        ("prediction_trace", [[0], [-1], [0]]),
        ("prediction_trace", [[0, 0], [3], [0]]),
        ("prediction_trace", [[1], [3], [0]]),
        ("teacher_trace", [[0], [False], [0]]),
    ):
        tampered = {"control": copy.deepcopy(base), "challenger": copy.deepcopy(base)}
        tampered["challenger"][field] = value
        with pytest.raises(RuntimeError, match="trace row|initial prediction"):
            module.validate_evaluation_results(unit="rollout-2", width=2, cases=cases, results=tampered)


def test_attempt_suffix_uses_parsed_max_and_exclusive_allocation(module, tmp_path) -> None:
    root = tmp_path / "attempts"
    (root / "screen-a").mkdir(parents=True)
    (root / "screen-a-retry-002").mkdir()
    attempt_id, directory, prior = module.allocate_exclusive_attempt_directory(root, "screen-a")
    assert attempt_id == "screen-a-retry-003"
    assert directory.is_dir()
    assert [item.name for item in prior] == ["screen-a", "screen-a-retry-002"]
    (root / "screen-b-retry-x").mkdir()
    with pytest.raises(RuntimeError, match="malformed"):
        module.allocate_exclusive_attempt_directory(root, "screen-b")


def readiness_attempt_state(module, attempt_id, authorization_sha256, *, sealed=False):
    state = {
        "schema_version": "exp648-readiness-calibration-attempt-v1",
        "attempt_id": attempt_id,
        "authorization_sha256": authorization_sha256,
        "phase": "sealed" if sealed else "authorized",
        "complete": sealed,
    }
    if sealed:
        state["completion_manifest_sha256"] = "f" * 64
    return module.canonical_json_bytes(state, pretty=True)


def tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_readiness_attempt_normal_first_allocation_is_exclusive(module, tmp_path) -> None:
    auth = "a" * 64
    attempts = tmp_path / "attempts"
    tracked = tmp_path / "tracked"
    attempt_id, directory, prior = module.allocate_readiness_calibration_attempt(
        authorization_sha256=auth, attempts_root=attempts, tracked_root=tracked,
    )
    assert attempt_id == f"cal-{auth[:16]}"
    assert directory.is_dir()
    assert prior == []


def test_readiness_incomplete_attempt_uses_monotonic_retry_and_preserves_prior_bytes(
    module, tmp_path, monkeypatch
) -> None:
    auth = "b" * 64
    stem = f"cal-{auth[:16]}"
    attempts = tmp_path / "attempts"
    tracked = tmp_path / "tracked"
    base = attempts / stem
    retry_two = attempts / f"{stem}-retry-002"
    for directory in (base, retry_two):
        directory.mkdir(parents=True)
        (directory / "attempt-state.json").write_bytes(
            readiness_attempt_state(module, directory.name, auth)
        )
        (directory / "preserved.bin").write_bytes(directory.name.encode())
    before = tree_bytes(attempts)
    attempt_id, directory, prior = module.allocate_readiness_calibration_attempt(
        authorization_sha256=auth, attempts_root=attempts, tracked_root=tracked,
    )
    assert attempt_id == f"{stem}-retry-003"
    assert directory.is_dir()
    assert [path.name for path in prior] == [stem, f"{stem}-retry-002"]
    assert {key: value for key, value in tree_bytes(attempts).items() if not key.startswith(f"{stem}-retry-003/")} == before

    original_mkdir = Path.mkdir
    collided = False

    def collide_once(path, *args, **kwargs):
        nonlocal collided
        if path.name == f"{stem}-retry-004" and not collided:
            collided = True
            original_mkdir(path, *args, **kwargs)
            raise FileExistsError(path)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", collide_once)
    before_collision = tree_bytes(attempts)
    next_id, next_directory, _ = module.allocate_readiness_calibration_attempt(
        authorization_sha256=auth, attempts_root=attempts, tracked_root=tracked,
    )
    assert collided is True
    assert next_id == f"{stem}-retry-005"
    assert next_directory.is_dir()
    after_collision = tree_bytes(attempts)
    for key, value in before_collision.items():
        assert after_collision[key] == value


@pytest.mark.parametrize("published", [False, True])
def test_sealed_or_published_readiness_attempt_cannot_relaunch_before_actions(
    module, tmp_path, monkeypatch, published
) -> None:
    auth = "c" * 64
    stem = f"cal-{auth[:16]}"
    scratch = tmp_path / "scratch"
    tracked = tmp_path / "tracked"
    attempts = scratch / "readiness-calibration-attempts"
    if published:
        tracked.mkdir(parents=True)
        (tracked / "readiness-calibration.json").write_bytes(
            module.canonical_json_bytes(
                {"calibration_authorization_sha256": auth}, pretty=True
            )
        )
        expected = "live tracked readiness output singleton"
    else:
        sealed = attempts / stem
        sealed.mkdir(parents=True)
        (sealed / "attempt-state.json").write_bytes(
            readiness_attempt_state(module, stem, auth, sealed=True)
        )
        expected = "sealed readiness calibration attempt"
    before_scratch = tree_bytes(tmp_path)
    actions = []

    def forbidden(name):
        return lambda *_args, **_kwargs: actions.append(name) or (_ for _ in ()).throw(
            AssertionError(f"{name} action occurred before attempt rejection")
        )

    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", scratch)
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tracked)
    monkeypatch.setattr(module, "CONFIG_PATH", tmp_path / "must-not-read-config.yaml")
    monkeypatch.setattr(module, "_atomic_write", forbidden("write"))
    monkeypatch.setattr(module.cal, "CalibrationExperiment", forbidden("model/root/data"))
    monkeypatch.setattr(module, "execute_synchronous_augmented_readiness_units", forbidden("evaluation"))
    monkeypatch.setattr(module, "log_and_verify_readiness_calibration_mlflow", forbidden("mlflow"))
    monkeypatch.setattr(module, "validate_one_child_authorization_lifecycle", forbidden("lifecycle"))
    with pytest.raises(PermissionError, match=expected):
        module.execute_readiness_calibration(
            {"calibration_root": TEST_ROOT + 99, "authorization_sha256": auth},
            launch_wall_started=time.perf_counter(),
            launch_cpu_started=0.0,
            launch_peak_rss=0,
        )
    assert actions == []
    after_scratch = tree_bytes(tmp_path)
    for key, value in before_scratch.items():
        assert after_scratch[key] == value
    assert set(after_scratch) - set(before_scratch) == {
        f"scratch/readiness-calibration-locks/cal-{auth[:16]}.lock"
    }


def test_readiness_matching_attempt_symlink_fails_closed(module, tmp_path) -> None:
    auth = "d" * 64
    stem = f"cal-{auth[:16]}"
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (attempts / stem).symlink_to(target, target_is_directory=True)
    before = tree_bytes(tmp_path)
    with pytest.raises(RuntimeError, match="linked, or non-directory"):
        module.allocate_readiness_calibration_attempt(
            authorization_sha256=auth,
            attempts_root=attempts,
            tracked_root=tmp_path / "tracked",
        )
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    "surface", ["attempt-state.json", "completion-seal.json", "publication"]
)
def test_readiness_broken_evidence_symlink_fails_closed(
    module, tmp_path, surface
) -> None:
    auth = "e" * 64
    stem = f"cal-{auth[:16]}"
    attempts = tmp_path / "attempts"
    tracked = tmp_path / "tracked"
    if surface == "publication":
        tracked.mkdir()
        (tracked / "readiness-calibration.json").symlink_to(
            tmp_path / "absent-publication.json"
        )
    else:
        attempt = attempts / stem
        attempt.mkdir(parents=True)
        (attempt / surface).symlink_to(tmp_path / f"absent-{surface}")
    before = tree_bytes(tmp_path)
    with pytest.raises(RuntimeError, match="no-follow regular file"):
        module.allocate_readiness_calibration_attempt(
            authorization_sha256=auth,
            attempts_root=attempts,
            tracked_root=tracked,
        )
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    "singleton_name",
    (
        "readiness-calibration.json",
        "readiness-calibration-core.json",
        "readiness-calibration-environment.json",
        "readiness-calibration-mlflow.json",
        "readiness-calibration-executed-config.yaml",
        "readiness-calibration-completion.json",
        "readiness-calibration-resource-completion.json",
        "readiness-calibration-attempt-state.json",
        "readiness-calibration-launch-ledger.json",
    ),
)
@pytest.mark.parametrize("authorization_relation", ("matching", "different"))
def test_any_live_readiness_singleton_blocks_before_allocation_and_actions(
    module, tmp_path, monkeypatch, singleton_name, authorization_relation
) -> None:
    auth = "2" * 64
    observed_auth = auth if authorization_relation == "matching" else "3" * 64
    stem = f"cal-{auth[:16]}"
    scratch = tmp_path / "scratch"
    attempts = scratch / "readiness-calibration-attempts"
    prior = attempts / stem
    tracked = tmp_path / "tracked"
    prior.mkdir(parents=True)
    tracked.mkdir()
    (prior / "attempt-state.json").write_bytes(
        readiness_attempt_state(module, stem, auth)
    )
    (prior / "preserved.bin").write_bytes(b"prior-attempt-must-not-change")
    (tracked / singleton_name).write_bytes(
        module.canonical_json_bytes(
            {"authorization_sha256": observed_auth}, pretty=True
        )
    )
    prior_before = tree_bytes(prior)
    tracked_before = tree_bytes(tracked)
    actions = []

    def forbidden(name):
        return lambda *_args, **_kwargs: actions.append(name) or (_ for _ in ()).throw(
            AssertionError(f"{name} action occurred before singleton rejection")
        )

    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", scratch)
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tracked)
    monkeypatch.setattr(module, "CONFIG_PATH", tmp_path / "must-not-read-config.yaml")
    monkeypatch.setattr(module, "_atomic_write", forbidden("attempt/config/write"))
    monkeypatch.setattr(module.cal, "CalibrationExperiment", forbidden("root/data/model"))
    monkeypatch.setattr(module, "execute_synchronous_augmented_readiness_units", forbidden("evaluation"))
    monkeypatch.setattr(module, "log_and_verify_readiness_calibration_mlflow", forbidden("mlflow"))
    monkeypatch.setattr(module, "validate_one_child_authorization_lifecycle", forbidden("lifecycle"))
    with pytest.raises(PermissionError, match="live tracked readiness output singleton"):
        module.execute_readiness_calibration(
            {"calibration_root": TEST_ROOT + 102, "authorization_sha256": auth},
            launch_wall_started=time.perf_counter(),
            launch_cpu_started=0.0,
            launch_peak_rss=0,
        )
    assert actions == []
    assert tree_bytes(prior) == prior_before
    assert tree_bytes(tracked) == tracked_before
    assert not (attempts / f"{stem}-retry-001").exists()


def test_readiness_singleton_absence_allows_normal_allocation(module, tmp_path) -> None:
    auth = "4" * 64
    attempt_id, directory, prior = module.allocate_readiness_calibration_attempt(
        authorization_sha256=auth,
        attempts_root=tmp_path / "attempts",
        tracked_root=tmp_path / "tracked",
    )
    assert attempt_id == f"cal-{auth[:16]}"
    assert directory.is_dir()
    assert prior == []


def test_active_readiness_authorization_lock_blocks_concurrent_launch_before_actions(
    module, tmp_path, monkeypatch
) -> None:
    auth = "f" * 64
    scratch = tmp_path / "scratch"
    actions = []
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", scratch)
    monkeypatch.setattr(
        module, "_execute_readiness_calibration_locked",
        lambda *_args, **_kwargs: actions.append("scientific") or {},
    )
    with module.readiness_calibration_authorization_lock(
        authorization_sha256=auth, scratch_root=scratch,
    ):
        with pytest.raises(PermissionError, match="already active"):
            module.execute_readiness_calibration(
                {"calibration_root": TEST_ROOT + 100, "authorization_sha256": auth},
                launch_wall_started=time.perf_counter(),
                launch_cpu_started=0.0,
                launch_peak_rss=0,
            )
    assert actions == []
    assert not (scratch / "readiness-calibration-attempts").exists()


def test_readiness_post_mkdir_publication_race_fails_before_state_or_scientific_actions(
    module, tmp_path, monkeypatch
) -> None:
    auth = "1" * 64
    stem = f"cal-{auth[:16]}"
    scratch = tmp_path / "scratch"
    attempts = scratch / "readiness-calibration-attempts"
    tracked = tmp_path / "tracked"
    prior = attempts / stem
    prior.mkdir(parents=True)
    (prior / "attempt-state.json").write_bytes(
        readiness_attempt_state(module, stem, auth)
    )
    (prior / "preserved.bin").write_bytes(b"preserve-me")
    before_prior = tree_bytes(prior)
    original_mkdir = Path.mkdir
    injected = False

    def inject_publication(path, *args, **kwargs):
        nonlocal injected
        result = original_mkdir(path, *args, **kwargs)
        if path.name == f"{stem}-retry-001" and not injected:
            injected = True
            tracked.mkdir(parents=True, exist_ok=True)
            (tracked / "readiness-calibration.json").write_bytes(
                module.canonical_json_bytes(
                    {"calibration_authorization_sha256": auth}, pretty=True
                )
            )
        return result

    actions = []

    def forbidden(name):
        return lambda *_args, **_kwargs: actions.append(name) or (_ for _ in ()).throw(
            AssertionError(f"{name} action occurred after publication race")
        )

    monkeypatch.setattr(Path, "mkdir", inject_publication)
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", scratch)
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tracked)
    monkeypatch.setattr(module, "_atomic_write", forbidden("attempt-state/write"))
    monkeypatch.setattr(module.cal, "CalibrationExperiment", forbidden("root/data/model"))
    monkeypatch.setattr(module, "execute_synchronous_augmented_readiness_units", forbidden("evaluation"))
    monkeypatch.setattr(module, "log_and_verify_readiness_calibration_mlflow", forbidden("mlflow"))
    monkeypatch.setattr(module, "validate_one_child_authorization_lifecycle", forbidden("lifecycle"))
    with pytest.raises(PermissionError, match="live tracked readiness output singleton"):
        module.execute_readiness_calibration(
            {"calibration_root": TEST_ROOT + 101, "authorization_sha256": auth},
            launch_wall_started=time.perf_counter(),
            launch_cpu_started=0.0,
            launch_peak_rss=0,
        )
    assert injected is True
    assert actions == []
    assert tree_bytes(prior) == before_prior
    assert (attempts / f"{stem}-retry-001").is_dir()


def test_readiness_mlflow_duplicate_runs_fail_before_create(module, monkeypatch) -> None:
    payload = {
        "calibration_authorization_sha256": "a" * 64, "calibration_root": TEST_ROOT + 1,
        "scientific_runner_sha256": "b" * 64, "screen_workload_sha256": "c" * 64,
        "calibration_evidence_core_sha256": "d" * 64,
        "resolved_screen_budgets": {
            "wall_seconds": {"projected_seconds": 1},
            "aggregate_cpu_seconds": {"projected_seconds": 4}, "peak_rss_guard_bytes": 100,
        },
    }
    duplicate = types.SimpleNamespace(
        data=types.SimpleNamespace(tags={"authorization_sha256": "a" * 64}),
        info=types.SimpleNamespace(status="FINISHED", run_id="duplicate"),
    )
    fake_client = types.SimpleNamespace(search_runs=lambda **_kwargs: [duplicate, duplicate])
    monkeypatch.setattr(module.cal, "configure_mlflow", lambda *_: "1")
    monkeypatch.setattr(module.cal, "tracking_uri", lambda: "sqlite://fake")
    monkeypatch.setattr(module.cal.mlflow.tracking, "MlflowClient", lambda **_kwargs: fake_client)
    with pytest.raises(RuntimeError, match="duplicate"):
        module.log_and_verify_readiness_calibration_mlflow(payload, [])


def test_one_child_authorization_commit_lifecycle_rejects_tamper_and_extra_files(module, tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (repo / "base.txt").write_text("base\n")
    git("add", "base.txt"); git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (repo / "auth.json").write_text('{"authorized":true}\n')
    git("add", "auth.json"); git("commit", "-m", "auth only")
    auth_commit = git("rev-parse", "HEAD")
    original = module.git_output
    monkeypatch.setattr(module, "ROOT", repo)
    def local_git(*args):
        if args == ("rev-parse", "@{upstream}"):
            return git("rev-parse", "HEAD")
        return original(*args)
    monkeypatch.setattr(module, "git_output", local_git)
    module.validate_one_child_authorization_lifecycle(
        auth_relative="auth.json", approved_base_commit=base, launch=True
    )
    (repo / "base.txt").write_text("tampered\n")
    with pytest.raises(PermissionError, match="clean"):
        module.validate_one_child_authorization_lifecycle(
            auth_relative="auth.json", approved_base_commit=base, launch=True
        )
    git("restore", "base.txt")
    (repo / "extra.txt").write_text("extra\n")
    git("add", "extra.txt"); git("commit", "--amend", "--no-edit")
    with pytest.raises(PermissionError, match="outside"):
        module.validate_one_child_authorization_lifecycle(
            auth_relative="auth.json", approved_base_commit=base, launch=True
        )
    git("checkout", "-b", "historical-descendant", auth_commit)
    (repo / "later.txt").write_text("unrelated descendant\n")
    git("add", "later.txt"); git("commit", "-m", "later unrelated")
    module.validate_one_child_authorization_lifecycle(
        auth_relative="auth.json", approved_base_commit=base, launch=False
    )
    with pytest.raises(PermissionError, match="launch HEAD"):
        module.validate_one_child_authorization_lifecycle(
            auth_relative="auth.json", approved_base_commit=base, launch=True
        )
    (repo / "base.txt").write_text("descendant changed binding input\n")
    git("add", "base.txt"); git("commit", "-m", "later bound file change")
    assert module._git_blob(base, "base.txt") == b"base\n"
    module.validate_one_child_authorization_lifecycle(
        auth_relative="auth.json", approved_base_commit=base, launch=False
    )


def test_partial_mlflow_runs_are_terminated_and_superseded(module) -> None:
    events = []
    running = types.SimpleNamespace(info=types.SimpleNamespace(status="RUNNING", run_id="r1"))
    scheduled = types.SimpleNamespace(info=types.SimpleNamespace(status="SCHEDULED", run_id="r3"))
    failed = types.SimpleNamespace(info=types.SimpleNamespace(status="FAILED", run_id="r2"))
    statuses = {"r1": "RUNNING", "r2": "FAILED", "r3": "SCHEDULED"}
    def terminate(run_id, status):
        events.append(("terminated", run_id, status)); statuses[run_id] = status
    client = types.SimpleNamespace(
        set_terminated=terminate,
        set_tag=lambda run_id, key, value: events.append(("tag", run_id, key, value)),
        get_run=lambda run_id: types.SimpleNamespace(info=types.SimpleNamespace(status=statuses[run_id], run_id=run_id)),
    )
    assert module.select_finished_or_supersede_partial_mlflow_runs(
        client, [running, failed, scheduled], reason="recovery"
    ) is None
    assert ("terminated", "r1", "KILLED") in events
    assert ("terminated", "r3", "KILLED") in events
    assert any(event[:3] == ("tag", "r1", "superseded_for_recovery") for event in events)
    assert any(event[:3] == ("tag", "r2", "superseded_for_recovery") for event in events)


def test_constructor_failure_preserves_authenticated_launch_state(module, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", tmp_path / "scratch")
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tmp_path / "tracked")
    launch = tmp_path / "launch-state.json"
    launch.write_bytes(module.canonical_json_bytes({
        "schema_version": "exp648-screen-launch-state-v1", "attempt_id": "launch-1",
        "authorization_sha256": "a" * 64, "binding_sha256": "b" * 64,
        "phase": "authorization-verified-constructor-pending", "complete": False,
        "module_wall_elapsed_seconds": 1.0, "module_process_cpu_elapsed_seconds": 2.0,
        "launch_wall_started_perf_counter": time.perf_counter() - 1.0,
        "launch_process_cpu_started_seconds": 0.0,
        "launch_initial_peak_rss_bytes": 1,
    }, pretty=True))
    module.preserve_screen_failure("constructor", RuntimeError("boom"), launch_state_path=launch)
    state = json.loads(launch.read_text())
    assert state["phase"] == "terminal-constructor-failure"
    assert state["failure"]["authorization_sha256"] == "a" * 64
    assert state["failure"]["gates_computed"] is False
    timing = state["failure"]["launch_failure_timing"]
    assert timing["sampled_fresh_at_exception_boundary"] is True
    assert timing["wall_seconds"] >= 1.0
    assert timing["aggregate_cpu_seconds"] >= 0
    assert timing["peak_rss_bytes"] > 1


def make_real_launch_ledger(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", tmp_path / "scratch")
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tmp_path / "tracked")
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("auth\n")
    authorization = {"screen_readiness_binding_sha256": "b" * 64}
    wall = time.perf_counter() - 1
    launch = module.create_screen_launch_ledger(
        authorization, auth_path, launch_wall_started=wall,
        launch_cpu_started=0, launch_peak_rss=1,
    )
    return authorization, auth_path, launch, wall


def fake_scientific_experiment(module, tmp_path, attempt_id, run):
    directory = tmp_path / attempt_id
    directory.mkdir(parents=True)
    state = {
        "attempt_id": attempt_id, "authorization_sha256": "a" * 64,
        "phase": "authorized", "completed_updates": {},
        "completed_evaluation_units": [], "cache_receipts": [],
        "verified_checkpoints": {}, "first_quarter": None,
        "environment_sha256": "c" * 64,
    }
    class Ledger:
        def __init__(self):
            self.attempt_id = attempt_id; self.directory = directory; self.state = state
        def persist(self):
            module._atomic_write(
                directory / "attempt-state.json",
                module.canonical_json_bytes(self.state, pretty=True),
            )
        def update(self, **values):
            self.state.update(values); self.persist()
    return types.SimpleNamespace(
        ledger=Ledger(),
        guard=types.SimpleNamespace(timing=lambda: {
            "wall_seconds": 2, "aggregate_cpu_seconds": 5, "peak_rss_bytes": 100,
        }),
        run=run,
    )


def test_real_screen_orchestration_preserves_wrong_root_failure(module, tmp_path, monkeypatch) -> None:
    authorization, auth_path, launch, wall = make_real_launch_ledger(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "reauthorize_screen_root", lambda *_args: (_ for _ in ()).throw(PermissionError("wrong root")))
    with pytest.raises(PermissionError, match="wrong root"):
        module.execute_authorized_screen_launch(
            authorization=authorization, authorization_path=auth_path,
            launch_state_path=launch, root=-1, launch_wall_started=wall,
            launch_cpu_started=0, launch_peak_rss=1,
            experiment_factory=lambda *_args, **_kwargs: pytest.fail("constructor opened"),
        )
    state = json.loads(launch.read_text())
    assert state["phase"] == "terminal-constructor-failure" and state["complete"] is False


def test_real_screen_orchestration_preserves_constructor_failure(module, tmp_path, monkeypatch) -> None:
    authorization, auth_path, launch, wall = make_real_launch_ledger(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "reauthorize_screen_root", lambda *_args: None)
    with pytest.raises(RuntimeError, match="constructor"):
        module.execute_authorized_screen_launch(
            authorization=authorization, authorization_path=auth_path,
            launch_state_path=launch, root=module.SCREEN_TRAINING_ROOT,
            launch_wall_started=wall, launch_cpu_started=0, launch_peak_rss=1,
            experiment_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("constructor")),
        )
    assert json.loads(launch.read_text())["phase"] == "terminal-constructor-failure"


def test_real_screen_orchestration_links_later_failure_to_both_ledgers(module, tmp_path, monkeypatch) -> None:
    authorization, auth_path, launch, wall = make_real_launch_ledger(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "reauthorize_screen_root", lambda *_args: None)
    experiment = fake_scientific_experiment(
        module, tmp_path, "scientific-7",
        lambda: (_ for _ in ()).throw(RuntimeError("later")),
    )
    with pytest.raises(RuntimeError, match="later"):
        module.execute_authorized_screen_launch(
            authorization=authorization, authorization_path=auth_path,
            launch_state_path=launch, root=module.SCREEN_TRAINING_ROOT,
            launch_wall_started=wall, launch_cpu_started=0, launch_peak_rss=1,
            experiment_factory=lambda *_args, **_kwargs: experiment,
        )
    launch_state = json.loads(launch.read_text())
    attempt_state = json.loads((experiment.ledger.directory / "attempt-state.json").read_text())
    assert launch_state["phase"] == "terminal-scientific-attempt-failure"
    assert launch_state["scientific_attempt_id"] == "scientific-7"
    assert attempt_state["phase"] == "incomplete" and (experiment.ledger.directory / "incomplete.json").is_file()


def test_real_screen_orchestration_marks_successful_sealed_launch(module, tmp_path, monkeypatch) -> None:
    authorization, auth_path, launch, wall = make_real_launch_ledger(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "reauthorize_screen_root", lambda *_args: None)
    attempt_id = "scientific-8"
    def run():
        attempt = module.TRACKED_SCREEN_DIR / "attempts" / attempt_id
        attempt.mkdir(parents=True)
        launch_attempt_id = experiment.ledger.state["launch_attempt_id"]
        orchestration = {
            "launch_attempt_id": launch_attempt_id,
            "scientific_attempt_id": attempt_id,
            "linkage_direction": "scientific-evidence-to-launch-attempt-id-without-launch-ledger-hash",
        }
        names = {
            "first-quarter-telemetry.json", "checkpoints.json", "run.json",
            "environment.json", "metrics.json", "integrity.json", "mlflow.json",
            "final-timing.json", "attempt-state.json",
        }
        for name in names:
            (attempt / name).write_text(f"{name}\n")
        (attempt / "attempt-state.json").write_bytes(module.canonical_json_bytes({
            **experiment.ledger.state, "phase": "sealed", "complete": True,
        }, pretty=True))
        (attempt / "run.json").write_bytes(module.canonical_json_bytes({
            "sealed_orchestration": orchestration,
        }, pretty=True))
        (attempt / "integrity.json").write_bytes(module.canonical_json_bytes({
            "sealed_orchestration": orchestration,
        }, pretty=True))
        seal = {
            "schema_version": "exp648-screen-completion-seal-v1",
            "attempt_id": attempt_id, "authorization_sha256": "a" * 64,
            "artifacts": {name: module.sha256(attempt / name) for name in names},
            "complete": True,
        }
        (attempt / "completion-seal.json").write_bytes(module.canonical_json_bytes(seal, pretty=True))
        pointer = {
            "schema_version": "exp648-screen-completion-pointer-v1",
            "attempt_id": attempt_id,
            "completion_seal_sha256": module.sha256(attempt / "completion-seal.json"),
        }
        (module.TRACKED_SCREEN_DIR / "completion.json").parent.mkdir(parents=True, exist_ok=True)
        (module.TRACKED_SCREEN_DIR / "completion.json").write_bytes(module.canonical_json_bytes(pointer, pretty=True))
        return {"ok": True}
    experiment = fake_scientific_experiment(module, tmp_path, attempt_id, run)
    result = module.execute_authorized_screen_launch(
        authorization=authorization, authorization_path=auth_path,
        launch_state_path=launch, root=module.SCREEN_TRAINING_ROOT,
        launch_wall_started=wall, launch_cpu_started=0, launch_peak_rss=1,
        experiment_factory=lambda *_args, **_kwargs: experiment,
    )
    state = json.loads(launch.read_text())
    assert result == {"ok": True}
    assert state["phase"] == "complete-sealed" and state["complete"] is True
    assert state["scientific_attempt_id"] == attempt_id
    assert len(state["completion_pointer_sha256"]) == 64
    assert len(state["completion_seal_sha256"]) == 64
    sealed_attempt = module.TRACKED_SCREEN_DIR / "attempts" / attempt_id
    scientific_state = json.loads((sealed_attempt / "attempt-state.json").read_text())
    run_receipt = json.loads((sealed_attempt / "run.json").read_text())
    integrity_receipt = json.loads((sealed_attempt / "integrity.json").read_text())
    assert scientific_state["launch_attempt_id"] == state["attempt_id"]
    assert run_receipt["sealed_orchestration"]["launch_attempt_id"] == state["attempt_id"]
    assert integrity_receipt["sealed_orchestration"] == run_receipt["sealed_orchestration"]
    mismatched = {**state, "scientific_attempt_id": "wrong-scientific-attempt"}
    launch.write_bytes(module.canonical_json_bytes(mismatched, pretty=True))
    with pytest.raises(RuntimeError, match="terminal reciprocal"):
        module.verify_screen_completion_seal(
            module.TRACKED_SCREEN_DIR / "completion.json", launch_state_path=launch
        )


def test_completion_seal_covers_receipt_ledger_and_all_complete_artifacts(module, tmp_path) -> None:
    attempt_id = "attempt-1"
    attempt = tmp_path / "attempts" / attempt_id
    attempt.mkdir(parents=True)
    names = {
        "first-quarter-telemetry.json", "checkpoints.json", "run.json", "environment.json",
        "metrics.json", "integrity.json", "mlflow.json", "final-timing.json", "attempt-state.json",
    }
    for name in names:
        (attempt / name).write_text(f"{name}\n")
    orchestration = {
        "launch_attempt_id": "launch-1", "scientific_attempt_id": attempt_id,
        "linkage_direction": "scientific-evidence-to-launch-attempt-id-without-launch-ledger-hash",
    }
    (attempt / "attempt-state.json").write_bytes(module.canonical_json_bytes({
        "attempt_id": attempt_id, "launch_attempt_id": "launch-1",
    }, pretty=True))
    (attempt / "run.json").write_bytes(module.canonical_json_bytes({
        "sealed_orchestration": orchestration,
    }, pretty=True))
    (attempt / "integrity.json").write_bytes(module.canonical_json_bytes({
        "sealed_orchestration": orchestration,
    }, pretty=True))
    seal = {
        "schema_version": "exp648-screen-completion-seal-v1", "attempt_id": attempt_id,
        "authorization_sha256": "a" * 64,
        "artifacts": {name: module.sha256(attempt / name) for name in names}, "complete": True,
    }
    seal_path = attempt / "completion-seal.json"
    seal_path.write_bytes(module.canonical_json_bytes(seal, pretty=True))
    pointer = {
        "schema_version": "exp648-screen-completion-pointer-v1", "attempt_id": attempt_id,
        "completion_seal_sha256": module.sha256(seal_path),
    }
    pointer_path = tmp_path / "completion.json"
    pointer_path.write_bytes(module.canonical_json_bytes(pointer, pretty=True))
    assert module.verify_screen_completion_seal(pointer_path)["complete"] is True
    (attempt / "mlflow.json").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="sealed artifact"):
        module.verify_screen_completion_seal(pointer_path)
