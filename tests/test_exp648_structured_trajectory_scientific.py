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
    assert lifecycle["status"] == "superseded-only-no-live-receipt"
    assert lifecycle["live_receipt_sha256"] is None
    assert lifecycle["superseded_root_20261161_preserved"] is True
    archived = module.validate_root20261161_archive()
    assert archived["authorization_sha256"] == (
        "0553c6434beb9528ea4f4fdb5a30c00bd4fe4b84aba1834d418526ea49e1b2fa"
    )
    assert archived["scratch_file_count"] == 218
    assert archived["mlflow_artifact_file_count"] == 27
    assert not module.SCREEN_READINESS_CALIBRATION_PATH.exists()
    assert not module.READINESS_CALIBRATION_AUTHORIZATION_PATH.exists()


def reseal_root20261161_archive(module, archive: Path) -> None:
    lines = [
        f"{module.sha256(path)}  {path.name}\n"
        for path in sorted(archive.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (archive / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


@pytest.mark.parametrize(
    "surface",
    ("classification", "authorization", "completion", "scratch", "mlflow"),
)
def test_root20261161_archive_tamper_and_crosslinks_fail_closed(
    module, tmp_path, surface
) -> None:
    archive = tmp_path / "root-20261161"
    shutil.copytree(module.ROOT_20261161_SUPERSEDED_DIR, archive)
    if surface == "classification":
        path = archive / "classification.json"
        payload = json.loads(path.read_bytes())
        payload["eligible_for_cache_reuse"] = True
        path.write_bytes(module.canonical_json_bytes(payload, pretty=True))
        expected = "classification mismatch"
    elif surface == "authorization":
        path = archive / "readiness-calibration-authorization.json"
        payload = json.loads(path.read_bytes())
        payload["calibration_root"] += 1
        path.write_bytes(module.canonical_json_bytes(payload, pretty=True))
        expected = "authorization bytes changed"
    elif surface == "completion":
        path = archive / "readiness-calibration-completion.json"
        payload = json.loads(path.read_bytes())
        payload["authorization_sha256"] = "0" * 64
        path.write_bytes(module.canonical_json_bytes(payload, pretty=True))
        expected = "authorization/attempt crosslink"
    elif surface == "scratch":
        manifest = archive / "scratch-files.sha256"
        lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[0] = "0" * 64 + lines[0][64:]
        manifest.write_text("".join(lines), encoding="utf-8")
        inventory_path = archive / "scratch-inventory.json"
        inventory = json.loads(inventory_path.read_bytes())
        inventory["manifest_sha256"] = module.sha256(manifest)
        inventory_path.write_bytes(
            module.canonical_json_bytes(inventory, pretty=True)
        )
        expected = "scratch hash mismatch"
    else:
        manifest = archive / "mlflow-artifacts.sha256"
        lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[0] = "0" * 64 + lines[0][64:]
        manifest.write_text("".join(lines), encoding="utf-8")
        runs_path = archive / "mlflow-runs.json"
        runs = json.loads(runs_path.read_bytes())
        runs["artifact_manifest_sha256"] = module.sha256(manifest)
        runs_path.write_bytes(module.canonical_json_bytes(runs, pretty=True))
        expected = "MLflow artifact hash mismatch"
    reseal_root20261161_archive(module, archive)
    with pytest.raises(PermissionError, match=expected):
        module.validate_root20261161_archive(archive_dir=archive)


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


def test_exact_repeat_uses_complete_ordered_prediction_trajectory_digest(module) -> None:
    first_trace = [[0, 0], [1, 0], [3, 2]]
    second_trace = [[0, 0], [2, 1], [3, 2]]
    first = {
        "predictions": [3, 2],
        "output_sha256": module.sha256_bytes(module.canonical_json_bytes([3, 2])),
        "prediction_trace": first_trace,
        "prediction_trace_sha256": module.sha256_bytes(
            module.canonical_json_bytes(first_trace)
        ),
        "exact_accuracy": 1.0,
        "first_error_indices": [2, 2],
    }
    second = {
        **first,
        "prediction_trace": second_trace,
        "prediction_trace_sha256": module.sha256_bytes(
            module.canonical_json_bytes(second_trace)
        ),
    }
    assert first["predictions"] == second["predictions"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["exact_accuracy"] == second["exact_accuracy"]
    assert first["first_error_indices"] == second["first_error_indices"]
    assert not module.prediction_trajectory_repeat_deterministic(first, second)
    tampered = {**first, "prediction_trace_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="trajectory digest"):
        module.prediction_trajectory_repeat_deterministic(first, tampered)


def test_matched_work_is_independent_per_arm_and_distinct_from_data_order(module) -> None:
    accounting = module.create_matched_work_accounting()
    declared = {
        "clean_forward_steps": module.WINDOW_LENGTH,
        "autonomous_forward_steps": module.WINDOW_LENGTH,
        "endpoint_loss_computations": 1,
        "clean_backward_calls": module.WINDOW_LENGTH,
        "endpoint_backward_calls": 1,
        "gradient_clip_calls": 1,
        "optimizer_steps": 1,
        "scheduler_steps": 1,
        "checkpoint_writes": 1,
        "checkpoint_reload_verifications": 1,
        "checkpoint_byte_hashes": 1,
        "checkpoint_receipt_serializations": 1,
        "transition_evaluation_cases": 2,
        "rollout_evaluation_cases": 2,
        "rollout_evaluation_steps": 4,
        "evaluation_result_hashes": 1,
        "evaluation_result_serializations": 1,
    }
    for arm in ("control", "challenger"):
        for category, count in declared.items():
            module.record_matched_work(
                accounting, arm, category, count, unit="declared-work-unit"
            )
        for operation in (
            "training_update_calls", "scheduler_step_calls",
            "checkpoint_save_reload_hash_calls",
            "checkpoint_receipt_persist_calls", "fresh_evaluator_calls",
            "fresh_result_hashes", "fresh_result_serializations",
        ):
            module.record_actual_matched_work(
                accounting, arm, operation, 1, unit="actual-work-unit"
            )
    receipt = module.finalize_matched_work(accounting)
    assert receipt["valid"] is True
    assert receipt["declared_to_actual_cross_binding_valid"] is True
    assert receipt["independent_from_paired_data_order"] is True
    assert set(receipt["per_arm"]) == {"control", "challenger"}
    assert "batch" not in module.canonical_json(receipt["per_arm"])
    changed = copy.deepcopy(accounting)
    module.record_actual_matched_work(
        changed, "challenger", "fresh_evaluator_calls", 1, unit="extra"
    )
    assert module.finalize_matched_work(changed)["valid"] is False

    for category in (
        "clean_forward_steps", "optimizer_steps", "scheduler_steps", "checkpoint_writes",
        "evaluation_result_hashes",
    ):
        identically_undercounted_actual = copy.deepcopy(accounting)
        for arm in ("control", "challenger"):
            module.record_matched_work(
                identically_undercounted_actual, arm, category, 1,
                unit=f"unmatched-{category}",
            )
        invalid = module.finalize_matched_work(identically_undercounted_actual)
        assert invalid["per_arm_effective_counts_and_ordered_digests_equal"] is True
        assert invalid["per_arm_actual_operation_counts_and_schedules_equal"] is True
        assert invalid["declared_to_actual_cross_binding_valid"] is False
        assert invalid["valid"] is False


def test_screen_execution_schedule_has_exact_shared_counts_and_cache_formula(
    module,
) -> None:
    schedule = module.screen_execution_schedule()
    assert schedule["scientific_attempt_ledger"]["atomic_writes"] == 216
    assert schedule["scientific_attempt_ledger"]["write_reasons"] == {
            "constructor": 1,
            "reciprocal_launch_attempt_binding": 1,
            "corpus_verified": 1,
            "training_started": 1,
            "completed_update_pairs": 200,
            "first_quarter_telemetry": 1,
            "checkpoint_receipts": 2,
            "evaluation_started": 1,
            "transition_units": 2,
            "rollout_units": 3,
            "exact_repeat_pair": 1,
            "finalization_started": 1,
            "sealed": 1,
    }
    assert schedule["scientific_attempt_ledger"][
        "fixed_write_reasons_before_first_quarter_guard"
    ] == {
        "constructor": 1,
        "reciprocal_launch_attempt_binding": 1,
        "corpus_verified": 1,
        "training_started": 1,
    }
    assert schedule["scientific_attempt_ledger"][
        "writes_before_first_quarter_guard"
    ] == 54
    assert schedule["scientific_attempt_ledger"][
        "completed_arm_updates_before_first_quarter_guard"
    ] == 100
    assert schedule["scientific_attempt_ledger"]["total_arm_updates"] == 400
    assert schedule["outer_launch_ledger"][
        "writes_before_first_quarter_guard"
    ] == 2
    assert schedule["outer_launch_ledger"]["atomic_writes"] == 3
    assert schedule["total_screen_ledger_atomic_writes"] == 219
    assert schedule["evaluation_cache_write_validate_units"] == 7
    assert schedule["checkpoint_save_reload_verifications"] == 2
    assert schedule["atomic_evidence_writes"] == 29
    assert [
        (unit["unit"], unit["mode"], unit["width"], unit["cases"])
        for unit in schedule["evaluation_units"]
    ] == [
        ("transition-128", "transition", 128, 2000),
        ("transition-256", "transition", 256, 2000),
        ("rollout-64", "rollout", 64, 200),
        ("rollout-128", "rollout", 128, 200),
        ("rollout-256", "rollout", 256, 200),
        ("exact-repeat-first-256", "rollout", 256, 20),
        ("exact-repeat-second-256", "rollout", 256, 20),
    ]
    assert schedule["per_arm_declared_counts"] == {
        "clean_forward_steps": 1600,
        "autonomous_forward_steps": 1600,
        "endpoint_loss_computations": 200,
        "clean_backward_calls": 1600,
        "endpoint_backward_calls": 200,
        "gradient_clip_calls": 200,
        "optimizer_steps": 200,
        "scheduler_steps": 200,
        "checkpoint_writes": 1,
        "checkpoint_reload_verifications": 1,
        "checkpoint_byte_hashes": 1,
        "checkpoint_receipt_serializations": 1,
        "transition_evaluation_cases": 4000,
        "rollout_evaluation_cases": 640,
        "rollout_evaluation_steps": 99840,
        "evaluation_result_hashes": 7,
        "evaluation_result_serializations": 7,
    }
    assert schedule["per_arm_declared_count_total"] == 110_298
    assert schedule["per_arm_declared_ordered_event_count"] == 1_630
    for cached_units in range(8):
        counts = module.screen_actual_operation_counts(cached_units)
        assert sum(counts.values()) == 423 + cached_units
        assert counts["fresh_evaluator_calls"] == 7 - cached_units
        assert counts["fresh_result_hashes"] == 7 - cached_units
        assert counts["fresh_result_serializations"] == 7 - cached_units
        for name in (
            "cache_record_validations", "cache_per_arm_receipt_validations",
            "cached_result_digest_validations", "cache_reuse_operations",
        ):
            assert counts[name] == cached_units
    with pytest.raises(RuntimeError, match="outside the schedule"):
        module.screen_actual_operation_counts(8)
    live_schedule_source = "\n".join(
        inspect.getsource(value)
        for value in (
            module.screen_execution_schedule,
            module.screen_workload_payload,
            module.measure_readiness_infrastructure_unit,
            module.finalize_matched_work,
        )
    )
    assert "atomic_attempt_ledger_writes" not in live_schedule_source
    assert '"atomic_evidence_writes": 19' not in live_schedule_source
    assert '!= 213' not in live_schedule_source
    schedule_source = inspect.getsource(module.screen_execution_schedule)
    for forbidden_literal in (
        '"writes_before_first_quarter_guard": 54',
        '"completed_arm_updates_before_first_quarter_guard": 100',
        '"total_arm_updates": 400',
        '"ordered_event_count_fresh": 423',
        "sum(scientific_ledger_reasons.values()) + 3",
    ):
        assert forbidden_literal not in schedule_source
    assert "WINDOW_LENGTH" not in inspect.getsource(
        module._exercise_calibration_matched_work_schedule
    )


def test_calibration_matched_work_exercises_exact_screen_schedule_and_tamper_fails(
    module, monkeypatch
) -> None:
    config = module.yaml.safe_load(module.CONFIG_PATH.read_bytes())
    schedule = module.screen_execution_schedule(config_override=config)
    receipt = module._exercise_calibration_matched_work_schedule(config)
    assert receipt["valid"] is True
    assert receipt["screen_execution_schedule_valid"] is True
    assert receipt["screen_execution_schedule_sha256"] == module.sha256_bytes(
        module.canonical_json_bytes(schedule)
    )
    for arm in ("control", "challenger"):
        arm_receipt = receipt["per_arm"][arm]
        assert arm_receipt["counts"] == schedule["per_arm_declared_counts"]
        assert arm_receipt["total_declared_work_units"] == 110_298
        assert arm_receipt["ordered_event_count"] == 1_630
        assert arm_receipt["actual_operation_counts"] == (
            schedule["per_arm_actual_operation_contract"]["fresh_zero_cache_reuse"]
        )
        assert arm_receipt["actual_operation_event_count"] == 423
    tampered = copy.deepcopy(schedule)
    tampered["per_arm_declared_count_total"] += 1
    monkeypatch.setattr(
        module, "screen_execution_schedule", lambda **_kwargs: copy.deepcopy(tampered)
    )
    with pytest.raises(RuntimeError, match="schedule exercise failed"):
        module._exercise_calibration_matched_work_schedule(config)


def test_readiness_infrastructure_physically_runs_every_shared_schedule_unit(
    module, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for arm in ("control", "challenger"):
        path = tmp_path / f"{arm}.safetensors"
        module.save_file({"weight": module.torch.tensor([1.0])}, path)
    repeat = {
        "components": {
            "checkpoint": {
                arm: {"path": f"{arm}.safetensors"}
                for arm in ("control", "challenger")
            }
        }
    }
    output = tmp_path / "infrastructure"
    observed_writes = []

    def recording_write(path, raw):
        path = Path(path)
        observed_writes.append(path.relative_to(output).as_posix())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    monkeypatch.setattr(module, "_atomic_write", recording_write)
    result = module.measure_readiness_infrastructure_unit(
        repeat, output, module.psutil.Process()
    )
    schedule = module.screen_execution_schedule()
    assert result["complete"] is True
    assert result["matched_work_schedule_valid"] is True
    assert result["scientific_attempt_ledger_atomic_writes"] == 216
    assert result["outer_launch_ledger_atomic_writes"] == 3
    assert result["total_screen_ledger_atomic_writes"] == 219
    assert result["evaluation_cache_write_validate_units"] == 7
    assert result["atomic_evidence_writes"] == 29
    assert sum(name == "ledger/attempt-state.json" for name in observed_writes) == 216
    assert sum(name == "ledger/launch-state.json" for name in observed_writes) == 3
    assert sum(name.startswith("evaluation-cache/") for name in observed_writes) == 7
    assert sum(name.startswith("evidence/") for name in observed_writes) == 29
    assert result["execution_schedule_sha256"] == module.sha256_bytes(
        module.canonical_json_bytes(schedule)
    )
    profile = result["evaluation_cache_profile"]
    module.validate_shape_matched_evaluation_cache_profile(profile, schedule)
    assert profile["evaluation_unit_count"] == 7
    assert profile["total_cases"] == 4_640
    assert profile["total_prediction_trace_elements"] == 200_960
    assert profile["total_teacher_trace_elements"] == 200_960
    assert profile["total_trace_elements"] == 401_920
    assert profile["total_canonical_payload_bytes"] == sum(
        unit["canonical_payload_bytes"] for unit in profile["evaluation_units"]
    )
    assert profile[
        "total_external_reference_minimum_payload_bytes"
    ] == sum(module.SHAPE_MATCHED_CACHE_REFERENCE_MINIMUM_BYTES.values())
    assert profile[
        "total_external_reference_minimum_payload_bytes"
    ] == 20_147_941
    assert profile["total_canonical_payload_bytes"] >= 20_147_941
    assert profile["total_actual_schema_projected_payload_bytes"] < profile[
        "total_canonical_payload_bytes"
    ]
    assert all(
        unit["canonical_payload_bytes"]
        >= unit["external_reference_minimum_payload_bytes"]
        for unit in profile["evaluation_units"]
    )
    for unit in profile["evaluation_units"]:
        contract = unit["value_width_contract"]
        assert contract["target_width_bits"] == unit["width"]
        for field in contract["fields"].values():
            assert field["bit_length_histogram"]
            assert field["decimal_digit_histogram"]
            if field["role"] == "width-bearing-integer":
                assert field["minimum_nonzero_bit_length"] >= unit["width"] - 1
        assert unit["actual_schema_projected_payload_bytes"] < unit[
            "canonical_payload_bytes"
        ]
        assert unit["envelope_extra_payload_bytes"] >= (
            module.SHAPE_MATCHED_CACHE_METADATA_PADDING_BYTES
        )
    rollout_payload = json.loads(
        (output / "evaluation-cache/04-rollout-256.json").read_bytes()
    )
    assert set(rollout_payload) == set(
        module.REAL_SCREEN_EVALUATION_CACHE_FIELDS
    ) | {"profiling_envelope"}
    assert "cases" not in {
        key for key in rollout_payload if key != "profiling_envelope"
    }
    assert rollout_payload["profiling_envelope"][
        "raw_cases_absent_from_actual_cache_schema"
    ] is True
    assert "cases" in rollout_payload["profiling_envelope"]
    assert profile["real_cache_schema_comparison"][
        "exact_schema_claimed"
    ] is False
    assert profile["real_cache_schema_comparison"][
        "conservative_serialization_parse_envelope_claimed"
    ] is True
    for arm in schedule["arm_names"]:
        assert len(rollout_payload["results"][arm]["prediction_trace"]) == 257
        assert len(rollout_payload["results"][arm]["teacher_trace"]) == 257
        assert all(
            len(row) == 200
            for row in rollout_payload["results"][arm]["prediction_trace"]
        )
        assert all(
            len(row) == 200
            for row in rollout_payload["results"][arm]["teacher_trace"]
        )


def test_shape_matched_cache_width_and_bytes_grow_at_equal_case_count(
    module,
) -> None:
    config = module.yaml.safe_load(module.CONFIG_PATH.read_bytes())

    def record(mode: str, width: int, cases: int) -> dict:
        rows = width + 1 if mode == "rollout" else 0
        specification = {
            "unit": f"{mode}-{width}", "mode": mode, "width": width,
            "cases": cases,
            "prediction_trace_rows_per_arm": rows,
            "prediction_trace_elements_per_arm": rows * cases,
            "teacher_trace_elements_per_arm": rows * cases,
        }
        fixture = module._shape_matched_evaluation_cache_records(
            specification, config
        )
        return {
            "bytes": len(module.canonical_json_bytes(
                fixture["envelope_record"], pretty=True
            )),
            "contract": fixture["value_width_contract"],
        }

    transition_128 = record("transition", 128, 5)
    transition_256 = record("transition", 256, 5)
    rollout_64 = record("rollout", 64, 3)
    rollout_128 = record("rollout", 128, 3)
    rollout_256 = record("rollout", 256, 3)
    assert transition_256["bytes"] > transition_128["bytes"]
    assert rollout_256["bytes"] > rollout_128["bytes"] > rollout_64["bytes"]
    assert transition_256["contract"]["summary"][
        "width_bearing_decimal_digit_count"
    ] > transition_128["contract"]["summary"][
        "width_bearing_decimal_digit_count"
    ]
    assert rollout_256["contract"]["target_width_decimal_digits"] == 78
    assert rollout_128["contract"]["target_width_decimal_digits"] == 39
    assert rollout_64["contract"]["target_width_decimal_digits"] == 20


@pytest.mark.parametrize("shrunk_value", [0, 1 << 124])
def test_shape_matched_cache_rejects_zero_or_shrunk_width_values(
    module, shrunk_value: int,
) -> None:
    specification = {
        "unit": "transition-128", "mode": "transition", "width": 128,
        "cases": 4, "prediction_trace_rows_per_arm": 0,
        "prediction_trace_elements_per_arm": 0,
        "teacher_trace_elements_per_arm": 0,
    }
    cases, results = module._shape_matched_evaluation_cache_payload(
        specification
    )
    cases["state"][0] = shrunk_value
    with pytest.raises(RuntimeError, match="width-bearing value envelope"):
        module._shape_matched_value_width_contract(
            specification, cases, results
        )


def test_evaluation_cache_records_truthful_fresh_and_reuse_work(
    module, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", tmp_path)
    calibration = {
        "resolved_screen_budgets": {
            "wall_seconds": {"projected_seconds": 1000, "soft_budget_seconds": 1500, "hard_runaway_ceiling_seconds": 2000},
            "aggregate_cpu_seconds": {"projected_seconds": 4000, "soft_budget_seconds": 6000, "hard_runaway_ceiling_seconds": 8000},
            "peak_rss_guard_bytes": 10**15,
            "first_quarter_checkpoint": {"wall_seconds": {"expected_elapsed_seconds": 250}, "aggregate_cpu_seconds": {"expected_elapsed_seconds": 1000}},
        }
    }
    cases = {
        "state": [0], "multiplicand": [0], "modulus": [2],
        "input_bit": [0], "target": [0], "quotient": [0],
    }
    calls = []
    def evaluator(model, _width, _cases, _exp300):
        calls.append(model)
        return {
            "cases": 1, "exact_accuracy": 1.0, "quotient_accuracy": 1.0,
            "valid_residue_rate": 1.0, "exact_vector": [True],
            "output_sha256": module.sha256_bytes(module.canonical_json_bytes([0])),
            "predictions": [0], "quotient_predictions": [0],
        }
    fields = {"runner": "a" * 64}
    fresh = module.create_matched_work_accounting()
    result, reused = module.load_or_evaluate_unit(
        unit="transition-1", cases=cases,
        models={"control": "control", "challenger": "challenger"},
        evaluator=evaluator, width=1, exp300=None, key_fields=fields,
        guard=module.ScreenGuard(calibration, module.psutil.Process()),
        matched_work=fresh,
    )
    assert reused is False and calls == ["control", "challenger"]
    for arm in ("control", "challenger"):
        assert fresh["arms"][arm]["actual_counts"]["fresh_evaluator_calls"] == 1
        assert fresh["arms"][arm]["actual_counts"]["cache_reuse_operations"] == 0

    cached = module.create_matched_work_accounting()
    result_again, reused = module.load_or_evaluate_unit(
        unit="transition-1", cases=cases,
        models={"control": object(), "challenger": object()},
        evaluator=lambda *_args: pytest.fail("evaluator called on valid cache reuse"),
        width=1, exp300=None, key_fields=fields,
        guard=module.ScreenGuard(calibration, module.psutil.Process()),
        matched_work=cached,
    )
    assert reused is True and result_again == result and calls == ["control", "challenger"]
    for arm in ("control", "challenger"):
        counts = cached["arms"][arm]["actual_counts"]
        assert counts["fresh_evaluator_calls"] == 0
        assert counts["cache_record_validations"] == 1
        assert counts["cache_per_arm_receipt_validations"] == 1
        assert counts["cached_result_digest_validations"] == 1
        assert counts["cache_reuse_operations"] == 1

    cache_path = next((tmp_path / "evaluation-cache").glob("transition-1-*.json"))
    record = json.loads(cache_path.read_text())
    record["matched_work_receipts"]["arms"]["control"]["result_canonical_sha256"] = "0" * 64
    core = copy.deepcopy(record)
    core.pop("core_sha256")
    record["core_sha256"] = module.sha256_bytes(module.canonical_json_bytes(core))
    cache_path.write_bytes(module.canonical_json_bytes(record, pretty=True))
    with pytest.raises(RuntimeError, match="matched-work receipt/result"):
        module.load_or_evaluate_unit(
            unit="transition-1", cases=cases,
            models={"control": object(), "challenger": object()},
            evaluator=lambda *_args: pytest.fail("tampered cache invoked evaluator"),
            width=1, exp300=None, key_fields=fields,
            guard=module.ScreenGuard(calibration, module.psutil.Process()),
        )


def test_retry_checkpoints_are_attempt_local_immutable_and_byte_bound(
    module, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    models = {
        "control": module.nn.Linear(3, 2),
        "challenger": module.nn.Linear(3, 2),
    }
    first_dir = tmp_path / "attempt-001" / "checkpoints"
    first = module.prepare_checkpoints_for_evaluation(
        models, first_dir, attempt_id="attempt-001"
    )
    before = {
        arm: (tmp_path / receipt["path"]).read_bytes()
        for arm, receipt in first.items()
    }
    retry_dir = tmp_path / "attempt-002" / "checkpoints"
    retry = module.prepare_checkpoints_for_evaluation(
        models, retry_dir, attempt_id="attempt-002"
    )
    assert all(receipt["attempt_id"] == "attempt-002" for receipt in retry.values())
    assert all(
        (tmp_path / first[arm]["path"]).read_bytes() == before[arm]
        for arm in before
    )
    with pytest.raises(RuntimeError, match="immutable once created"):
        module.prepare_checkpoints_for_evaluation(
            models, first_dir, attempt_id="attempt-001"
        )


def test_control_checkpoint_is_persisted_and_failure_bound_before_challenger(
    module, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SCRATCH_SCREEN_DIR", tmp_path / "scratch")
    monkeypatch.setattr(module, "TRACKED_SCREEN_DIR", tmp_path / "tracked")
    attempt = tmp_path / "attempt-003"
    attempt.mkdir()
    state = {
        "attempt_id": "attempt-003", "authorization_sha256": "a" * 64,
        "phase": "evaluation", "verified_checkpoints": {},
        "completed_updates": {}, "completed_evaluation_units": [],
        "cache_receipts": [], "first_quarter": None,
    }
    class Ledger:
        attempt_id = "attempt-003"
        directory = attempt
        def __init__(self): self.state = state
        def persist(self):
            module._atomic_write(
                attempt / "attempt-state.json",
                module.canonical_json_bytes(self.state, pretty=True),
            )
    ledger = Ledger()
    def persist(arm, receipt):
        ledger.state["verified_checkpoints"][arm] = receipt
        ledger.persist()
    with pytest.raises(RuntimeError, match="fault-after-control"):
        module.prepare_checkpoints_for_evaluation(
            {"control": module.nn.Linear(3, 2), "challenger": module.nn.Linear(3, 2)},
            attempt / "checkpoints", attempt_id="attempt-003",
            receipt_hook=persist,
            event_hook=lambda event: (_ for _ in ()).throw(
                RuntimeError("fault-after-control")
            ) if event == "checkpoint-verified:control" else None,
        )
    (attempt / "checkpoints/.challenger.orphan.tmp").write_bytes(
        b"preserved-orphan-partial-checkpoint-bytes"
    )
    experiment = types.SimpleNamespace(
        ledger=ledger,
        guard=types.SimpleNamespace(timing=lambda: {
            "wall_seconds": 1.0, "aggregate_cpu_seconds": 2.0,
            "peak_rss_bytes": 100,
        }),
    )
    module.preserve_screen_failure(
        "checkpoint", RuntimeError("fault-after-control"), experiment=experiment
    )
    failure = json.loads((attempt / "incomplete.json").read_text())
    inventory = failure["attempt_checkpoint_artifact_inventory"]
    by_name = {item["name"]: item for item in inventory["artifacts"]}
    assert set(by_name) == {"control.safetensors", ".challenger.orphan.tmp"}
    assert by_name["control.safetensors"]["matching_receipt_arms"] == ["control"]
    assert by_name["control.safetensors"]["exactly_one_matching_arm_receipt"] is True
    assert by_name["control.safetensors"]["binding_status"] == "receipt-bound"
    assert by_name["control.safetensors"]["receipt_binds_bytes"] is True
    assert by_name[".challenger.orphan.tmp"]["matching_receipt_arms"] == []
    assert by_name[".challenger.orphan.tmp"]["exactly_one_matching_arm_receipt"] is False
    assert by_name[".challenger.orphan.tmp"]["binding_status"] == "unbound-no-receipt"
    assert by_name[".challenger.orphan.tmp"]["receipt_binds_bytes"] is False
    assert len(by_name[".challenger.orphan.tmp"]["sha256"]) == 64
    assert inventory["all_artifacts_receipt_bound"] is False
    assert failure["preserved_checkpoint_bytes"]["control"]["receipt_hash_matches_preserved_bytes"] is True


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


def fake_shape_matched_value_width_contract(
    module, specification: dict,
) -> dict:
    width = int(specification["width"])
    roles = module._shape_matched_value_field_roles(specification)
    fields = {}
    wide_value = 1 << (width - 1)
    for name, role in roles.items():
        count = role["expected_integer_count"]
        if role["role"] == "width-bearing-integer":
            zero_count = role["expected_schema_zero_count"]
            nonzero_count = count - zero_count
            bit_histogram = {str(width): nonzero_count}
            digit_histogram = {str(len(str(wide_value))): nonzero_count}
            if zero_count:
                bit_histogram["0"] = zero_count
                digit_histogram["1"] = (
                    digit_histogram.get("1", 0) + zero_count
                )
            field = {
                **role,
                "count": count,
                "zero_count": zero_count,
                "nonzero_count": nonzero_count,
                "minimum_value": 0 if zero_count else wide_value,
                "maximum_value": wide_value,
                "minimum_nonzero_bit_length": width,
                "maximum_bit_length": width,
                "minimum_decimal_digits": (
                    1 if zero_count else len(str(wide_value))
                ),
                "maximum_decimal_digits": len(str(wide_value)),
                "bit_length_histogram": bit_histogram,
                "decimal_digit_histogram": digit_histogram,
            }
        else:
            value = int(role["maximum_schema_value"])
            field = {
                **role,
                "count": count,
                "zero_count": 0,
                "nonzero_count": count,
                "minimum_value": value,
                "maximum_value": value,
                "minimum_nonzero_bit_length": value.bit_length(),
                "maximum_bit_length": value.bit_length(),
                "minimum_decimal_digits": len(str(value)),
                "maximum_decimal_digits": len(str(value)),
                "bit_length_histogram": {str(value.bit_length()): count},
                "decimal_digit_histogram": {str(len(str(value))): count},
            }
        fields[name] = field
    width_fields = [
        field for field in fields.values()
        if field["role"] == "width-bearing-integer"
    ]
    small_fields = [
        field for field in fields.values()
        if field["role"] == "naturally-small-schema-integer"
    ]
    contract = {
        "schema_version": "exp648-cache-value-width-contract-v1",
        "target_width_bits": width,
        "target_width_decimal_digits": len(str((1 << width) - 1)),
        "minimum_width_bearing_nonzero_bit_length": width - 1,
        "fixture_construction": {
            "modulus": "(1 << width) - 159",
            "base": "1 << (width - 1)",
            "transition": (
                "base-plus-bounded-offset state/multiplicand; input bit one; "
                "target and quotient from frozen exact_teacher_step"
            ),
            "rollout": (
                "multiplier=base; multiplicand=modulus-1; complete teacher "
                "trajectory from frozen exact_teacher_step; prediction trace "
                "is a separately allocated exact copy"
            ),
        },
        "construction_scope": {
            "resource_calibration_only": True,
            "offline_exact_teacher_labels_only": True,
            "model_or_evaluator_called": False,
            "root_rng_or_data_opened": False,
            "submission_inference_path": False,
            "scientific_accuracy_or_selection_evidence": False,
        },
        "fields": fields,
        "summary": {
            "field_count": len(fields),
            "width_bearing_field_count": len(width_fields),
            "naturally_small_field_count": len(small_fields),
            "integer_count": sum(field["count"] for field in fields.values()),
            "width_bearing_integer_count": sum(
                field["count"] for field in width_fields
            ),
            "width_bearing_zero_count": sum(
                field["zero_count"] for field in width_fields
            ),
            "width_bearing_nonzero_count": sum(
                field["nonzero_count"] for field in width_fields
            ),
            "width_bearing_decimal_digit_count": sum(
                int(digits) * occurrences
                for field in width_fields
                for digits, occurrences in field[
                    "decimal_digit_histogram"
                ].items()
            ),
            "minimum_observed_nonzero_width_bearing_bit_length": min(
                field["minimum_nonzero_bit_length"]
                for field in width_fields
            ),
            "maximum_observed_width_bearing_bit_length": max(
                field["maximum_bit_length"] for field in width_fields
            ),
        },
    }
    module._validate_declared_shape_matched_value_width_contract(
        contract, specification
    )
    return contract


def fake_shape_matched_cache_profile(module, schedule: dict, scale: float) -> dict:
    def digest(value) -> str:
        return module.sha256_bytes(module.canonical_json_bytes(value))

    units = []
    for specification in schedule["evaluation_units"]:
        prediction_per_arm = specification[
            "prediction_trace_elements_per_arm"
        ]
        teacher_per_arm = specification["teacher_trace_elements_per_arm"]
        prediction_total = prediction_per_arm * schedule["arm_count"]
        teacher_total = teacher_per_arm * schedule["arm_count"]
        rollout = specification["mode"] == "rollout"
        value_width_contract = fake_shape_matched_value_width_contract(
            module, specification
        )
        reference_minimum = (
            module.SHAPE_MATCHED_CACHE_REFERENCE_MINIMUM_BYTES.get(
                specification["unit"]
            )
        )
        actual_schema_bytes = max(
            1_000,
            (reference_minimum or (
                1_000 + specification["cases"] * 10
                + prediction_total + teacher_total
            )) // 2,
        )
        canonical_bytes = max(
            reference_minimum or 0,
            actual_schema_bytes
            + module.SHAPE_MATCHED_CACHE_METADATA_PADDING_BYTES + 1_000,
        )
        units.append({
            **copy.deepcopy(specification),
            "arm_count": schedule["arm_count"],
            "total_prediction_trace_elements": prediction_total,
            "total_teacher_trace_elements": teacher_total,
            "total_trace_elements": prediction_total + teacher_total,
            "canonical_payload_bytes": canonical_bytes,
            "actual_schema_projected_payload_bytes": actual_schema_bytes,
            "envelope_extra_payload_bytes": (
                canonical_bytes - actual_schema_bytes
            ),
            "metadata_padding_bytes": (
                module.SHAPE_MATCHED_CACHE_METADATA_PADDING_BYTES
            ),
            "external_reference_minimum_payload_bytes": reference_minimum,
            "canonical_payload_sha256": digest([
                "payload", specification["unit"]
            ]),
            "actual_schema_projected_payload_sha256": digest([
                "actual-schema", specification["unit"]
            ]),
            "core_sha256": digest(["core", specification["unit"]]),
            "cases_sha256": digest(["cases", specification["unit"]]),
            "semantic_fixture_sha256": digest([
                "semantic-fixture", specification["unit"]
            ]),
            "value_width_contract": value_width_contract,
            "value_width_contract_sha256": digest(value_width_contract),
            "matched_work_receipts_sha256": digest([
                "receipts", specification["unit"]
            ]),
            "prediction_trace_sha256_by_arm": {
                arm: digest(["prediction", specification["unit"], arm])
                if rollout else None
                for arm in schedule["arm_names"]
            },
            "teacher_trace_sha256_by_arm": {
                arm: digest(["teacher", specification["unit"], arm])
                if rollout else None
                for arm in schedule["arm_names"]
            },
            "actual_cache_schema_fields_covered": True,
            "raw_cases_are_conservative_envelope_only": True,
            "fresh_cache_serialized": True,
            "recovery_payload_parsed": True,
            "canonical_round_trip_verified": True,
            "core_hash_verified": True,
            "work_receipts_verified": True,
            "semantic_results_verified": True,
        })
    timing = lambda value: {
        "wall_seconds": value * scale,
        "aggregate_cpu_seconds": value * scale * 4,
        "peak_rss_bytes": int(1_000_000 * scale),
    }
    return {
        "schema_version": "exp648-shape-matched-evaluation-cache-profile-v2",
        "evaluation_units": units,
        "evaluation_unit_count": len(units),
        "total_cases": sum(unit["cases"] for unit in units),
        "total_canonical_payload_bytes": sum(
            unit["canonical_payload_bytes"] for unit in units
        ),
        "total_actual_schema_projected_payload_bytes": sum(
            unit["actual_schema_projected_payload_bytes"] for unit in units
        ),
        "total_envelope_extra_payload_bytes": sum(
            unit["envelope_extra_payload_bytes"] for unit in units
        ),
        "total_metadata_padding_bytes": sum(
            unit["metadata_padding_bytes"] for unit in units
        ),
        "total_external_reference_minimum_payload_bytes": sum(
            unit["external_reference_minimum_payload_bytes"] or 0
            for unit in units
        ),
        "total_prediction_trace_elements": sum(
            unit["total_prediction_trace_elements"] for unit in units
        ),
        "total_teacher_trace_elements": sum(
            unit["total_teacher_trace_elements"] for unit in units
        ),
        "total_trace_elements": sum(
            unit["total_trace_elements"] for unit in units
        ),
        "complete_set_sha256": digest(units),
        "value_width_contracts_sha256": digest([
            unit["value_width_contract"] for unit in units
        ]),
        "real_cache_schema_comparison": (
            module._shape_matched_real_cache_schema_comparison()
        ),
        "read_only_value_width_evidence": copy.deepcopy(
            module.SHAPE_MATCHED_CACHE_READ_ONLY_VALUE_WIDTH_EVIDENCE
        ),
        "fresh_cache_serialization_complete": True,
        "recovery_parse_digest_receipt_semantic_validation_complete": True,
        "timing_breakdown": {
            "fixture_construction_and_semantic_validation": timing(0.1),
            "fresh_cache_serialization": timing(0.1),
            "recovery_validation": timing(0.1),
            "complete_evaluation_unit_set": timing(0.4),
        },
    }


def fake_readiness_repeat(
    module, scale: float, config: dict | None = None,
) -> dict:
    timing = lambda value: {"wall_seconds": value * scale, "aggregate_cpu_seconds": value * scale * 4, "peak_rss_bytes": int(1_000_000 * scale)}
    schedule = module.screen_execution_schedule(config_override=config)
    cache_profile = fake_shape_matched_cache_profile(module, schedule, scale)
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
            "readiness_infrastructure": {"execution_schedule_sha256": module.sha256_bytes(module.canonical_json_bytes(schedule)), "atomic_checkpoint_writes_and_reload_verifications": schedule["checkpoint_save_reload_verifications"], "scientific_attempt_ledger_atomic_writes": schedule["scientific_attempt_ledger"]["atomic_writes"], "outer_launch_ledger_atomic_writes": schedule["outer_launch_ledger"]["atomic_writes"], "total_screen_ledger_atomic_writes": schedule["total_screen_ledger_atomic_writes"], "evaluation_cache_write_validate_units": schedule["evaluation_cache_write_validate_units"], "evaluation_cache_profile": cache_profile, "atomic_evidence_writes": schedule["atomic_evidence_writes"], "first_quarter_scientific_ledger_writes": schedule["scientific_attempt_ledger"]["writes_before_first_quarter_guard"], "matched_work_schedule_valid": True, "matched_work_receipt_sha256": "a" * 64, "component_timing": {"atomic_checkpoints": timing(0.5), "scientific_attempt_ledger": timing(0.5), "outer_launch_ledger": timing(0.5), "evaluation_cache": timing(0.5), "atomic_evidence": timing(0.5), "matched_work": timing(0.5)}, "complete": True, "timing": timing(2)},
            "actual_source_replay": {"complete_identities": 64, "steps_per_identity": 256, "complete_steps": 16384, "actual_rollout_identity_batch_path": True, "complete": True, "timing": timing(64)},
            "mlflow_probe": {"complete": True, "artifact_count": 6, "remote_verified": True, "timing": timing(3)},
        },
        "base_total_timing_before_readiness_additions": timing(431),
        "total_timing": timing(500),
    }
    return record


def test_schedule_mutation_covaries_counts_workload_shapes_and_projection(
    module,
) -> None:
    baseline_config = module.yaml.safe_load(module.CONFIG_PATH.read_bytes())
    mutated_config = copy.deepcopy(baseline_config)
    mutated_config["screen"]["updates_per_arm"] = 240
    mutated_config["screen"]["evaluation"]["transition_cases"][512] = 100

    baseline = module.screen_execution_schedule(
        config_override=baseline_config
    )
    mutated = module.screen_execution_schedule(config_override=mutated_config)
    assert baseline["scientific_attempt_ledger"][
        "writes_before_first_quarter_guard"
    ] == 54
    assert baseline["scientific_attempt_ledger"][
        "completed_arm_updates_before_first_quarter_guard"
    ] == 100
    assert baseline["scientific_attempt_ledger"]["total_arm_updates"] == 400
    assert baseline["per_arm_actual_operation_contract"][
        "ordered_event_count_fresh"
    ] == 423
    assert mutated["scientific_attempt_ledger"][
        "writes_before_first_quarter_guard"
    ] == 64
    assert mutated["scientific_attempt_ledger"][
        "completed_arm_updates_before_first_quarter_guard"
    ] == 120
    assert mutated["scientific_attempt_ledger"]["total_arm_updates"] == 480
    assert mutated["scientific_attempt_ledger"]["atomic_writes"] == 257
    assert mutated["total_screen_ledger_atomic_writes"] == 260
    assert mutated["per_arm_actual_operation_contract"][
        "ordered_event_count_fresh"
    ] == 506
    assert sum(module.screen_actual_operation_counts(
        8, config_override=mutated_config
    ).values()) == 514
    assert mutated["per_arm_declared_ordered_event_count"] == 1_953
    assert mutated["per_arm_declared_count_total"] == 111_560

    baseline_workload = module.screen_workload_payload(
        config_override=baseline_config
    )
    mutated_workload = module.screen_workload_payload(
        config_override=mutated_config
    )
    assert module.sha256_bytes(module.canonical_json_bytes(baseline_workload)) != (
        module.sha256_bytes(module.canonical_json_bytes(mutated_workload))
    )
    extra_unit = mutated["evaluation_units"][2]
    assert extra_unit == {
        "unit": "transition-512", "mode": "transition", "width": 512,
        "cases": 100, "prediction_trace_rows_per_arm": 0,
        "prediction_trace_elements_per_arm": 0,
        "teacher_trace_elements_per_arm": 0,
    }
    cases, results = module._shape_matched_evaluation_cache_payload(extra_unit)
    assert len(cases["state"]) == 100
    module.validate_evaluation_results(
        unit=extra_unit["unit"], width=extra_unit["width"],
        cases=cases, results=results,
    )
    profile = fake_shape_matched_cache_profile(module, mutated, 1.0)
    module.validate_shape_matched_evaluation_cache_profile(profile, mutated)
    assert profile["evaluation_unit_count"] == 8
    mutated_matched_work = module._exercise_calibration_matched_work_schedule(
        mutated_config
    )
    assert mutated_matched_work["valid"] is True
    assert mutated_matched_work["per_arm"]["control"][
        "actual_operation_event_count"
    ] == 506

    baseline_repeats = [
        fake_readiness_repeat(module, scale, baseline_config)
        for scale in (1.0, 1.1, 0.9)
    ]
    mutated_repeats = [
        fake_readiness_repeat(module, scale, mutated_config)
        for scale in (1.0, 1.1, 0.9)
    ]
    launcher = {
        "wall_seconds": 2.0,
        "aggregate_cpu_seconds": 8.0,
        "peak_rss_bytes": 2_000_000,
    }
    _, baseline_resolution = module.resolve_readiness_screen_budgets(
        baseline_repeats, launcher, config_override=baseline_config
    )
    _, mutated_resolution = module.resolve_readiness_screen_budgets(
        mutated_repeats, launcher, config_override=mutated_config
    )
    baseline_denominators = baseline_resolution["projection_denominators"]
    mutated_denominators = mutated_resolution["projection_denominators"]
    assert baseline_denominators["evaluation_cache_profiled_unit_count"] == 7
    assert mutated_denominators["evaluation_cache_profiled_unit_count"] == 8
    assert mutated_denominators["updates_per_arm"] == 240
    assert mutated_denominators["quarter_updates_per_arm"] == 60
    assert mutated_denominators["total_arm_updates"] == 480
    assert mutated_denominators["first_quarter_completed_arm_updates"] == 120
    assert mutated_denominators["scientific_attempt_ledger_writes"] == 257
    assert mutated_denominators[
        "first_quarter_scientific_attempt_ledger_writes"
    ] == 64
    assert mutated_denominators["evaluation_units"] == mutated["evaluation_units"]
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        assert "evaluation_cache_per_unit" not in mutated_resolution["rates"][
            resource
        ]
        assert mutated_resolution["projected_components"][resource][
            "evaluation_cache_shape_matched_complete_set"
        ] == mutated_resolution["rates"][resource][
            "evaluation_cache_complete_profiled_set"
        ]
        assert mutated_resolution["projected_components"][resource][
            "training_both_arms"
        ] > baseline_resolution["projected_components"][resource][
            "training_both_arms"
        ]
        assert mutated_resolution["projected_components"][resource][
            "transitions_both_arms"
        ] > baseline_resolution["projected_components"][resource][
            "transitions_both_arms"
        ]


def test_shape_matched_cache_profile_rejects_unit_and_total_tampering(
    module,
) -> None:
    schedule = module.screen_execution_schedule()
    profile = fake_shape_matched_cache_profile(module, schedule, 1.0)
    module.validate_shape_matched_evaluation_cache_profile(profile, schedule)
    mutations = []
    wrong_cases = copy.deepcopy(profile)
    wrong_cases["evaluation_units"][0]["cases"] += 1
    mutations.append(wrong_cases)
    zero_bytes = copy.deepcopy(profile)
    zero_bytes["evaluation_units"][0]["canonical_payload_bytes"] = 0
    mutations.append(zero_bytes)
    below_reference = copy.deepcopy(profile)
    below_reference["evaluation_units"][0]["canonical_payload_bytes"] = (
        module.SHAPE_MATCHED_CACHE_REFERENCE_MINIMUM_BYTES["transition-128"]
        - 1
    )
    below_reference["evaluation_units"][0][
        "envelope_extra_payload_bytes"
    ] = (
        below_reference["evaluation_units"][0]["canonical_payload_bytes"]
        - below_reference["evaluation_units"][0][
            "actual_schema_projected_payload_bytes"
        ]
    )
    mutations.append(below_reference)
    shrunk_width_contract = copy.deepcopy(profile)
    shrunk_width_contract["evaluation_units"][0]["value_width_contract"][
        "fields"
    ]["cases.state"]["minimum_nonzero_bit_length"] = 1
    mutations.append(shrunk_width_contract)
    wrong_trace_count = copy.deepcopy(profile)
    wrong_trace_count["evaluation_units"][2][
        "total_prediction_trace_elements"
    ] += 1
    mutations.append(wrong_trace_count)
    wrong_complete_hash = copy.deepcopy(profile)
    wrong_complete_hash["complete_set_sha256"] = "f" * 64
    mutations.append(wrong_complete_hash)
    for mutated in mutations:
        with pytest.raises(RuntimeError, match="shape-matched evaluation cache"):
            module.validate_shape_matched_evaluation_cache_profile(
                mutated, schedule
            )


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
    repeats = [fake_readiness_repeat(module, value) for value in (1.0, 1.1, 0.9)]
    launcher = {"wall_seconds": 2.0, "aggregate_cpu_seconds": 8.0, "peak_rss_bytes": 2_000_000}
    warmup = fake_readiness_repeat(module, 0.8)
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
        "schema_version": "exp648-screen-readiness-calibration-v2",
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
                "pre_r3_or_same_root_cache_loaded_or_saved": False,
                "fresh_warmup_and_three_fresh_representative_repeats": True,
                "carried_charge_applied_exactly_once": True,
                "guard_carried_wall_seconds": module.READINESS_CARRIED_CHARGE["wall_seconds"],
                "guard_carried_aggregate_cpu_seconds": module.READINESS_CARRIED_CHARGE["aggregate_cpu_seconds"],
                "guard_configured_wall_ceiling_seconds": 30000.0,
                "guard_configured_aggregate_cpu_ceiling_seconds": 120000.0,
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
            record = fake_readiness_repeat(module, 1.0)
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
        return copy.deepcopy(fake_readiness_repeat(module, 1.0)["components"]["readiness_infrastructure"])

    def validate(record, _config):
        unit = "warmup" if record["repeat_index"] is None else f"repeat-{record['repeat_index']}"
        events.append(f"{unit}:validate")

    monkeypatch.setattr(module, "_write_readiness_chronology_start", recording_start)
    monkeypatch.setattr(module, "_write_readiness_chronology_seal", recording_seal)
    monkeypatch.setattr(module, "measure_actual_source_replay_unit", replay)
    monkeypatch.setattr(module, "measure_readiness_mlflow_probe", probe)
    monkeypatch.setattr(module, "measure_readiness_infrastructure_unit", infrastructure)
    monkeypatch.setattr(module.cal, "validate_calibration_repeat", validate)

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
    assert events.index("warmup:seal") < events.index("repeat-1:start")
    assert all(record["reused_from_augmented_cache"] is False for record in [warmup, *repeats])
    assert all(record["augmented_cache_path"] is None for record in [warmup, *repeats])
    source = inspect.getsource(module._execute_one_augmented_readiness_unit)
    for component in (
        "chronology-start", "fresh-cache-exclusion", "base-complete-unit", "source-replay",
        "mlflow-probe", "readiness-infrastructure", "augmented-validation",
        "augmented-record-publication", "fresh-chronology-seal",
    ):
        assert component in source
    for forbidden in (
        "load_augmented_repeat_cache(", "save_augmented_repeat_cache(",
        "cache-reuse-charge", "reused-chronology-seal",
    ):
        assert forbidden not in source


def test_old_post_hoc_readiness_records_without_chronology_are_rejected(module) -> None:
    payload = {
        "calibration_root": TEST_ROOT + 12,
        "calibration_authorization_sha256": "a" * 64,
        "warmup": fake_readiness_repeat(module, 1.0),
        "repeats": [fake_readiness_repeat(module, 1.0) for _ in range(3)],
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
        "warmup": fake_readiness_repeat(module, 1.0),
        "repeats": [fake_readiness_repeat(module, 1.0) for _ in range(3)],
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
    with pytest.raises(PermissionError, match="schema mismatch"):
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
    with pytest.raises(PermissionError, match="schema mismatch"):
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
            self.wall_ceiling = 30000.0
            self.cpu_ceiling = 120000.0
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
        "wall_seconds": 13778.295949158995,
        "aggregate_cpu_seconds": 55378.175260172996,
    }
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["wall_seconds"] == 5294.4
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["aggregate_cpu_seconds"] == 21000.0
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["exact_internal_observation"] is False
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["completed_work_claimed"] is False
    assert module.READINESS_CARRIED_CHARGE == {
        "wall_seconds": 19072.695949158995,
        "aggregate_cpu_seconds": 76378.175260173,
    }
    assert guard.prior_superseded_wall_seconds == 19072.695949158995
    assert guard.prior_superseded_cpu_seconds == 76378.175260173
    assert guard.wall_ceiling == 30000.0
    assert guard.cpu_ceiling == 120000.0
    assert module.ROOT_20261160_FAILED_ATTEMPT_ALLOWANCE["semantics"] == (
        "conservative-accounting-allowance-never-an-exact-internal-observation"
    )
    assert guard.events == [
        "readiness:cumulative-carried-charge-applied-once"
    ]
    with pytest.raises(RuntimeError, match="more than once"):
        module.apply_readiness_carried_charge_once(guard)

    wrong_ceiling = Guard()
    wrong_ceiling.wall_ceiling = 29999.0
    with pytest.raises(RuntimeError, match="configured cumulative ceilings"):
        module.apply_readiness_carried_charge_once(wrong_ceiling)


def test_root20261162_budget_formula_uses_new_carried_charge_and_independent_resources(
    module,
) -> None:
    contract = module.readiness_root20261162_cumulative_budget_contract()
    assert contract["formula"] == (
        "new_carried_charge + 1.5 * (1.25 * representative_full_screen + explicit_setup)"
    )
    assert contract["carried_charge_applied_exactly_once"] is True
    assert contract["root20261161_rates_executable"] is False
    assert contract["root20261161_caches_executable"] is False
    assert contract["fidelity_cap"] is False
    assert contract["scientific_gate"] is False
    assert contract["per_attempt_target"] is False
    expected = {
        "wall_seconds": {
            "new_carried_charge_seconds": 19072.695949158995,
            "representative_full_screen_seconds": 5305.469600509008,
            "explicit_setup_seconds": 73.8932728219952,
            "formula_result_seconds": 29131.291359346378,
            "independently_rounded_cumulative_ceiling_seconds": 30000.0,
        },
        "aggregate_cpu_seconds": {
            "new_carried_charge_seconds": 76378.175260173,
            "representative_full_screen_seconds": 20967.89,
            "explicit_setup_seconds": 135.975487648,
            "formula_result_seconds": 115896.932241645,
            "independently_rounded_cumulative_ceiling_seconds": 120000.0,
        },
    }
    for resource, values in expected.items():
        assert contract[resource] == values


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


def test_historical_v1_authorization_is_fixed_path_nonlaunch_history_only(
    module, monkeypatch
) -> None:
    monkeypatch.setattr(
        module, "validate_one_child_authorization_lifecycle",
        lambda *_args, **_kwargs: pytest.fail(
            "generic current authorization lifecycle used for historical v1"
        ),
    )
    payload = module.verify_historical_root20261161_authorization()
    assert payload["calibration_root"] == 20_261_161
    assert payload["approved_base_commit"] == (
        "87384f92ab1c053c7791e8afd69a07936d5a7402"
    )
    assert payload["original_authorization_commit"] == (
        "16d903301f50e54e3f6d9ab8cb6435e61f0d648f"
    )
    assert payload["original_authorization_path"] == (
        "experiments/runs/exp-648-structured-trajectory-consistency/screen/"
        "readiness-calibration-authorization.json"
    )
    assert payload["historical_nonlaunch_only"] is True


def test_historical_v1_schema_cannot_authorize_root20261162_before_binding(
    module, tmp_path, monkeypatch
) -> None:
    raw = (
        module.ROOT_20261161_SUPERSEDED_DIR
        / "readiness-calibration-authorization.json"
    ).read_bytes()
    path = tmp_path / "v1.json"
    path.write_bytes(raw)
    events = []
    monkeypatch.setattr(
        module, "readiness_calibration_authorization_base",
        lambda *_args, **_kwargs: events.append("binding"),
    )
    with pytest.raises(PermissionError, match="schema mismatch"):
        module.verify_readiness_calibration_authorization(
            path, committed_bytes=raw, allow_test_root=True, launch=False
        )
    assert events == []


def mocked_root20261162_base_blobs(module):
    source_raw = module.ROOT_20261162_AUTHORIZATION_SOURCE_PATH.read_bytes()
    provenance_raw = module.ROOT_20261162_SCAFFOLD_PROVENANCE_PATH.read_bytes()
    source = json.loads(source_raw)
    provenance = json.loads(provenance_raw)
    paths = {}
    for binding in source["approved_base_file_bindings"].values():
        paths[binding["path"]] = (module.ROOT / binding["path"]).read_bytes()
    for binding in source["frozen_corpus_file_bindings"].values():
        paths[binding["path"]] = (module.ROOT / binding["path"]).read_bytes()
    paths[module.ROOT_20261162_AUTHORIZATION_SOURCE_PATH.relative_to(module.ROOT).as_posix()] = source_raw
    paths[module.ROOT_20261162_SCAFFOLD_PROVENANCE_PATH.relative_to(module.ROOT).as_posix()] = provenance_raw
    return paths, source, provenance


def test_readiness_authorization_v2_has_only_four_dynamic_fields(
    module, monkeypatch
) -> None:
    paths, source, _provenance = mocked_root20261162_base_blobs(module)
    monkeypatch.setattr(module, "_git_blob", lambda _commit, relative: paths[relative])
    payload = module.readiness_calibration_authorization_base(
        TEST_ROOT, "base-commit"
    )
    dynamic = {
        "calibration_root": TEST_ROOT,
        "approved_base_commit": "base-commit",
        "authorization_source_blob_sha256": module.sha256_bytes(
            paths[module.ROOT_20261162_AUTHORIZATION_SOURCE_PATH.relative_to(module.ROOT).as_posix()]
        ),
        "scaffold_provenance_blob_sha256": module.sha256_bytes(
            paths[module.ROOT_20261162_SCAFFOLD_PROVENANCE_PATH.relative_to(module.ROOT).as_posix()]
        ),
    }
    assert payload == {**source["authorization_template"], **dynamic}
    assert not set(dynamic).intersection(source["authorization_template"])
    assert not module.READINESS_CALIBRATION_AUTHORIZATION_PATH.exists()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("workload", "screen workload hash mismatch"),
        ("template", "authorization template hash mismatch"),
        ("runner_binding", "scientific_runner direct hash mismatch"),
        ("environment", "authorization environment mismatch"),
    ],
)
def test_authorization_source_tampering_fails_before_root_data_model_mlflow_or_lifecycle(
    module, monkeypatch, tamper, message
) -> None:
    paths, source, provenance = mocked_root20261162_base_blobs(module)
    if tamper == "workload":
        source["screen_workload"]["recoverable_evaluation_unit_count"] += 1
    elif tamper == "template":
        source["authorization_template_sha256"] = "0" * 64
    elif tamper == "runner_binding":
        source["approved_base_file_bindings"]["scientific_runner"]["sha256"] = "0" * 64
    elif tamper == "environment":
        source["authorization_template"]["approved_runtime_environment"] = {"hostile": True}
        source["authorization_template_sha256"] = module.sha256_bytes(
            module.canonical_json_bytes(source["authorization_template"])
        )
    source_path = module.ROOT_20261162_AUTHORIZATION_SOURCE_PATH.relative_to(
        module.ROOT
    ).as_posix()
    paths[source_path] = module.canonical_json_bytes(source, pretty=True)
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
        module.readiness_calibration_authorization_base(TEST_ROOT, "base-commit")
    assert actions == []


def test_authorization_provenance_crosslink_tamper_fails_before_actions(
    module, monkeypatch
) -> None:
    paths, _source, provenance = mocked_root20261162_base_blobs(module)
    provenance["screen_workload_sha256"] = "0" * 64
    provenance_path = module.ROOT_20261162_SCAFFOLD_PROVENANCE_PATH.relative_to(
        module.ROOT
    ).as_posix()
    paths[provenance_path] = module.canonical_json_bytes(provenance, pretty=True)
    monkeypatch.setattr(module, "_git_blob", lambda _commit, relative: paths[relative])
    actions = []
    monkeypatch.setattr(
        module, "authorize_root", lambda *_args, **_kwargs: actions.append("root")
    )
    monkeypatch.setattr(
        module.mlflow, "start_run", lambda *_args, **_kwargs: actions.append("mlflow")
    )
    with pytest.raises(PermissionError, match="scaffold crosslink mismatch"):
        module.readiness_calibration_authorization_base(TEST_ROOT, "base-commit")
    assert actions == []


def test_readiness_scaffold_provenance_matches_current_code_and_workload(module) -> None:
    provenance_raw = module.ROOT_20261162_SCAFFOLD_PROVENANCE_PATH.read_bytes()
    source_raw = module.ROOT_20261162_AUTHORIZATION_SOURCE_PATH.read_bytes()
    provenance = json.loads(provenance_raw)
    source = json.loads(source_raw)
    assert provenance_raw == module.canonical_json_bytes(provenance, pretty=True)
    assert source_raw == module.canonical_json_bytes(source, pretty=True)
    assert module.sha256(module.READINESS_SCAFFOLD_PROVENANCE_PATH) == (
        "593d18523ff2888c1d57d8626d81f4ebb07a234427573efa3afc226e98e2c5a6"
    )
    runner_sha256 = module.sha256(Path(module.__file__))
    test_sha256 = module.sha256(Path(__file__))
    config_sha256 = module.sha256(module.CONFIG_PATH)
    config_raw = module.CONFIG_PATH.read_bytes()
    config = module.yaml.safe_load(config_raw)
    expected_workload_sha256 = module.sha256_bytes(module.canonical_json_bytes(
        module.screen_workload_payload(
            scientific_runner_sha256=runner_sha256,
            config_override=config,
        )
    ))
    assert provenance["screen_workload_sha256"] == expected_workload_sha256
    assert source["screen_workload"] == module.screen_workload_payload(
        scientific_runner_sha256=runner_sha256,
        config_override=config,
    )
    assert source["screen_workload_sha256"] == expected_workload_sha256
    assert source["authorization_template_sha256"] == module.sha256_bytes(
        module.canonical_json_bytes(source["authorization_template"])
    )
    template = source["authorization_template"]
    for dynamic in (
        "calibration_root", "approved_base_commit",
        "authorization_source_blob_sha256", "scaffold_provenance_blob_sha256",
    ):
        assert dynamic not in template
    assert template["screen_workload_sha256"] == expected_workload_sha256
    assert template["scientific_runner_sha256"] == runner_sha256
    assert template["scientific_test_sha256"] == test_sha256
    assert template["resolved_config_sha256"] == config_sha256
    scientific_protocol, resource_protocol = module._base_protocol_hashes(
        "unused", config
    )
    assert template["scientific_protocol_sha256"] == scientific_protocol
    assert template["resource_protocol_sha256"] == resource_protocol
    assert template["calibration_segment_evaluator_sha256"] == module.PINNED[
        "calibration_evaluator_sha256"
    ]
    assert template["screen_evaluator_sha256"] != template[
        "calibration_segment_evaluator_sha256"
    ]
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
        assert module.sha256(module.ROOT / binding["path"]) == binding["sha256"]
    for name, binding in source["frozen_corpus_file_bindings"].items():
        assert binding["sha256"] == module.FROZEN_CORPUS_FILES[name]
        assert module.sha256(module.ROOT / binding["path"]) == binding["sha256"]
    source_path = module.ROOT_20261162_AUTHORIZATION_SOURCE_PATH.relative_to(
        module.ROOT
    ).as_posix()
    provenance_path = module.ROOT_20261162_SCAFFOLD_PROVENANCE_PATH.relative_to(
        module.ROOT
    ).as_posix()
    bound_paths = {
        binding["path"]
        for collection in (
            source["approved_base_file_bindings"],
            source["frozen_corpus_file_bindings"],
        )
        for binding in collection.values()
    }
    assert source_path not in bound_paths
    assert provenance_path not in bound_paths
    assert provenance["dynamic_authorization_fields"] == sorted(
        {
            "calibration_root", "approved_base_commit",
            "authorization_source_blob_sha256", "scaffold_provenance_blob_sha256",
        }
    )
    assert provenance["authorization_manifest_present"] is False
    assert not module.READINESS_CALIBRATION_AUTHORIZATION_PATH.exists()


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
        "schema_version": "exp648-screen-readiness-calibration-authorization-v2"
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
    base["prediction_trace_sha256"] = module.sha256_bytes(
        module.canonical_json_bytes(base["prediction_trace"])
    )
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
            "checkpoints/control.safetensors",
            "checkpoints/challenger.safetensors",
        }
        for name in names:
            (attempt / name).parent.mkdir(parents=True, exist_ok=True)
            (attempt / name).write_text(f"{name}\n")
        checkpoint_hashes = {
            arm: {
                "attempt_id": attempt_id,
                "sha256": module.sha256(attempt / f"checkpoints/{arm}.safetensors"),
            }
            for arm in ("control", "challenger")
        }
        (attempt / "checkpoints.json").write_bytes(module.canonical_json_bytes({
            "authorization_sha256": "a" * 64,
            "checkpoint_hashes": checkpoint_hashes,
        }, pretty=True))
        (attempt / "attempt-state.json").write_bytes(module.canonical_json_bytes({
            **experiment.ledger.state, "phase": "sealed", "complete": True,
            "gates_computed": True,
        }, pretty=True))
        (attempt / "run.json").write_bytes(module.canonical_json_bytes({
            "authorization_sha256": "a" * 64,
            "sealed_orchestration": orchestration,
        }, pretty=True))
        (attempt / "integrity.json").write_bytes(module.canonical_json_bytes({
            "authorization_sha256": "a" * 64,
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
        "checkpoints/control.safetensors", "checkpoints/challenger.safetensors",
    }
    for name in names:
        (attempt / name).parent.mkdir(parents=True, exist_ok=True)
        (attempt / name).write_text(f"{name}\n")
    checkpoint_hashes = {
        arm: {
            "attempt_id": attempt_id,
            "sha256": module.sha256(attempt / f"checkpoints/{arm}.safetensors"),
        }
        for arm in ("control", "challenger")
    }
    (attempt / "checkpoints.json").write_bytes(module.canonical_json_bytes({
        "authorization_sha256": "a" * 64,
        "checkpoint_hashes": checkpoint_hashes,
    }, pretty=True))
    orchestration = {
        "launch_attempt_id": "launch-1", "scientific_attempt_id": attempt_id,
        "linkage_direction": "scientific-evidence-to-launch-attempt-id-without-launch-ledger-hash",
    }
    (attempt / "attempt-state.json").write_bytes(module.canonical_json_bytes({
        "attempt_id": attempt_id, "launch_attempt_id": "launch-1",
        "authorization_sha256": "a" * 64, "phase": "sealed",
        "complete": True, "gates_computed": True,
    }, pretty=True))
    (attempt / "run.json").write_bytes(module.canonical_json_bytes({
        "authorization_sha256": "a" * 64,
        "sealed_orchestration": orchestration,
    }, pretty=True))
    (attempt / "integrity.json").write_bytes(module.canonical_json_bytes({
        "authorization_sha256": "a" * 64,
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
    original_state = json.loads((attempt / "attempt-state.json").read_text())
    original_run = json.loads((attempt / "run.json").read_text())
    def reseal() -> None:
        seal["artifacts"] = {name: module.sha256(attempt / name) for name in names}
        seal_path.write_bytes(module.canonical_json_bytes(seal, pretty=True))
        pointer["completion_seal_sha256"] = module.sha256(seal_path)
        pointer_path.write_bytes(module.canonical_json_bytes(pointer, pretty=True))
    for mutation in (
        {**original_state, "phase": "finalization"},
        {**original_state, "complete": False},
        {**original_state, "gates_computed": False},
        {**original_state, "authorization_sha256": "b" * 64},
    ):
        (attempt / "attempt-state.json").write_bytes(
            module.canonical_json_bytes(mutation, pretty=True)
        )
        reseal()
        with pytest.raises(RuntimeError, match="terminal-complete|authorization binding"):
            module.verify_screen_completion_seal(pointer_path)
    (attempt / "attempt-state.json").write_bytes(
        module.canonical_json_bytes(original_state, pretty=True)
    )
    (attempt / "run.json").write_bytes(module.canonical_json_bytes({
        **original_run, "authorization_sha256": "b" * 64,
    }, pretty=True))
    reseal()
    with pytest.raises(RuntimeError, match="authorization binding"):
        module.verify_screen_completion_seal(pointer_path)
    (attempt / "run.json").write_bytes(
        module.canonical_json_bytes(original_run, pretty=True)
    )
    reseal()
    seal["attempt_id"] = "wrong-attempt-even-though-resealed"
    reseal()
    with pytest.raises(RuntimeError, match="seal attempt ID mismatch"):
        module.verify_screen_completion_seal(pointer_path)
    seal["attempt_id"] = attempt_id
    reseal()
    (attempt / "mlflow.json").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="sealed artifact"):
        module.verify_screen_completion_seal(pointer_path)
