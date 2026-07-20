from __future__ import annotations

import copy
import importlib.util
import json
import time
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


def test_frozen_calibration_import_and_all_pins(module) -> None:
    assert module.sha256(module.CALIBRATION_RUNNER) == module.CALIBRATION_RUNNER_SHA256
    assert module.CALIBRATION_RUNNER_SHA256 == (
        "f8cdc278b67013c6ca920833bc26473ad938e15749eb85811fc00016b6bb5672"
    )
    config = module.yaml.safe_load(module.CONFIG_PATH.read_text())
    observed = module.verify_pinned_foundation(config)
    assert observed["repository_head"] == module.PINNED["repository_head"]
    assert observed["repository_upstream"] == module.PINNED["repository_head"]


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
    with pytest.raises(PermissionError, match="hash mismatch"):
        module.verify_screen_authorization(authorization)
    expected = module.expected_screen_authorization_payload(
        module.sha256(corpus_manifest)
    )
    authorization.write_bytes(module.canonical_json_bytes(expected, pretty=True))
    assert module.verify_screen_authorization(authorization) == expected
    tampered = copy.deepcopy(expected)
    tampered["confirmation_authorized"] = True
    authorization.write_bytes(module.canonical_json_bytes(tampered, pretty=True))
    with pytest.raises(PermissionError, match="hash mismatch"):
        module.verify_screen_authorization(authorization)


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
    source = RUNNER.read_text(encoding="utf-8")
    corpus_class = source.split("class CorpusExperiment:", 1)[1].split(
        "def _load_corpus_pairs", 1
    )[0]
    screen_class = source.split("class ScreenExperiment:", 1)[1].split(
        "def _read_data_artifacts", 1
    )[0]
    assert "save_file(" not in corpus_class
    assert "save_file(" in screen_class
    assert "scientific_accuracy_reported\": False" in source
    assert "selection_metric_reported\": False" in source
    assert "screen_training_update" in source
    assert "evaluate_transitions" in source
    assert "evaluate_rollouts" in source


def test_no_scientific_metric_names_in_corpus_mlflow_metric_block(module) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    metric_block = source.split("mlflow.log_metrics(", 1)[1].split("}", 1)[0]
    for forbidden in ("accuracy", "selection", "exact", "gain", "delta"):
        assert forbidden not in metric_block.lower()
