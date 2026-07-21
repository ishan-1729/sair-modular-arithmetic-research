from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/exp_648_structured_trajectory_consistency/run.py"
CONFIG = ROOT / "configs/exp_648_structured_trajectory_consistency.yaml"
TEST_ROOT = 648_900_001


def load_runner():
    spec = importlib.util.spec_from_file_location("exp648_test_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def fake_window(
    *,
    kind: str,
    identity: int,
    start: int,
    bit_length: int = 256,
    band: str = "b0",
) -> dict:
    modulus = (1 << 255) + 123
    input_bits = [(identity + offset) % 2 for offset in range(8)]
    multiplier = sum(
        bit << (256 - 1 - (start + offset))
        for offset, bit in enumerate(input_bits)
    )
    multiplicand = identity + 5
    teacher = [0]
    quotients = []
    for bit in input_bits:
        raw = 2 * teacher[-1] + bit * multiplicand
        teacher.append(raw % modulus)
        quotients.append(raw // modulus)
    natural = list(teacher)
    first_error = 257
    if kind == "failure":
        natural[1] = 999
        natural[2:] = teacher[2:]
        first_error = start
    return {
        "identity_index": identity,
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
        "input_bits": input_bits,
        "quotients": quotients,
    }


def valid_scientific_integrity(module) -> dict[str, bool]:
    return {key: True for key in module.REQUIRED_SCIENTIFIC_INTEGRITY_KEYS}


def exact_corpus_fixture(module) -> tuple[dict, list[dict]]:
    modulus = (1 << 255) + 123
    identities = [
        module.Identity(0, 1 << 255, 7, modulus),
        module.Identity(1, 1 << 255, 9, modulus),
    ]
    records = []
    replayed = []
    for index, identity in enumerate(identities):
        teacher = [0]
        for position in range(256):
            bit = (identity.multiplier >> (255 - position)) & 1
            teacher.append(
                module.exact_teacher_step(
                    teacher[-1], identity.multiplicand, identity.modulus, bit
                )[0]
            )
        predicted = list(teacher)
        first_error = 257
        if index == 0:
            predicted[1] = (teacher[1] + 1) % (1 << 256)
            first_error = 0
        record = {
            "identity": identity,
            "predicted_states": predicted,
            "teacher_states": teacher,
            "first_error_index": first_error,
        }
        replayed.append(copy.deepcopy(record))
        records.append(
            {
                **module._identity_payload(identity),
                "first_error_index": first_error,
                "complete_predicted_states": predicted,
                "complete_teacher_states": teacher,
            }
        )
    failures, successes = [], []
    for record in replayed:
        local_failures, local_successes = module.trajectory_windows(record)
        failures.extend(local_failures)
        successes.extend(local_successes)
    pairs = module.match_windows(failures, successes, 1)
    payload = {
        "schema_version": "exp648-calibration-corpus-v1",
        "role": "resource-calibration-only-non-scientific",
        "root": TEST_ROOT,
        "target_pairs": 1,
        "identity_trajectories": 2,
        "identity_manifest": [module._identity_payload(identity) for identity in identities],
        "trajectory_manifest": records,
        "pairs": pairs,
        "matching": {},
        "synthetic_state_corruption": False,
        "state_bit_flips_or_injections": False,
    }
    return payload, replayed


def test_fail_closed_mode_root_allowlist_rejects_every_locked_surface() -> None:
    module = load_runner()
    config = load_config()
    module.validate_frozen_config(config)
    for role in config["seed_roles"].values():
        if isinstance(role, dict) and "seed" in role:
            assert "status" not in role
            assert "status_at_preregistration" in role or "current_status" in role
    assert "authorization" not in config["screen"]
    assert "authorization" not in config["confirmation"]
    assert "authorization_at_preregistration" in config["screen"]
    assert "authorization_at_preregistration" in config["confirmation"]
    assert module.authorize_root("test", TEST_ROOT) == TEST_ROOT
    assert module.authorize_root("calibration", 20_261_151) == 20_261_151

    forbidden_roots = {
        20_261_149,
        20_261_152,
        20_261_153,
        20_261_154,
        20_261_155,
        20_261_156,
        20_261_157,
        20_261_158,
    }
    for root in forbidden_roots:
        with pytest.raises(PermissionError):
            module.authorize_root("test", root)
        with pytest.raises(PermissionError):
            module.authorize_root("calibration", root)
    with pytest.raises(PermissionError):
        module.authorize_root("test", 20_261_151)
    with pytest.raises(PermissionError):
        module.authorize_root("calibration", TEST_ROOT)
    for mode in (
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
        "unknown",
    ):
        with pytest.raises(PermissionError):
            module.authorize_root(mode, TEST_ROOT)


def test_every_identity_generator_boundary_reauthorizes_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_runner()
    prime_called = False

    def forbidden_prime(*_args, **_kwargs):
        nonlocal prime_called
        prime_called = True
        raise AssertionError("prime generation must not follow failed authorization")

    monkeypatch.setattr(module, "_prime_in_band", forbidden_prime)
    for mode, root in (
        ("test", 20_261_151),
        ("calibration", TEST_ROOT),
        ("test", 20_261_152),
        ("calibration", 20_261_149),
    ):
        with pytest.raises(PermissionError):
            module.generate_identities(mode, root, 0, 1)
    assert prime_called is False


def test_executable_dependency_manifest_is_pinned_and_tamper_detected() -> None:
    module = load_runner()
    config = load_config()
    manifest = module.dependency_source_manifest()
    assert manifest == config["executable_dependency_sources"]
    module.verify_dependency_source_manifest(manifest)
    assert set(manifest["sources"]) == {"cellular_model", "exp300_data_evaluator"}
    tampered = copy.deepcopy(manifest)
    tampered["sources"]["cellular_model"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        module.verify_dependency_source_manifest(tampered)


def test_source_hash_shape_parameters_identical_clones_and_fresh_optimizers() -> None:
    module = load_runner()
    config = load_config()
    first = module.load_verified_source_model(config, mode="test", root_seed=TEST_ROOT)
    second = module.load_verified_source_model(config, mode="test", root_seed=TEST_ROOT)
    source = ROOT / config["source_checkpoint"]
    assert module.sha256(source) == config["source_checkpoint_sha256"]
    assert sum(parameter.numel() for parameter in first.parameters()) == 99_824
    assert first.width == second.width == 256
    assert first.hidden == second.hidden == 92
    assert first.cycles == second.cycles == 2
    assert module._shape_signature(first) == module._shape_signature(second)
    assert module.state_dict_sha256(first) == module.state_dict_sha256(second)
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=5e-6)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=5e-6)
    assert module.optimizer_is_fresh(first_optimizer)
    assert module.optimizer_is_fresh(second_optimizer)
    assert first_optimizer is not second_optimizer


def test_loss_has_one_arm_variable_and_straight_through_is_hard_forward() -> None:
    module = load_runner()
    base = torch.tensor(2.0, requires_grad=True)
    endpoint = torch.tensor(3.0, requires_grad=True)
    control = module.combine_arm_loss(base, endpoint, 0.0)
    challenger = module.combine_arm_loss(base, endpoint, 0.1)
    assert float(control) == pytest.approx(2.0)
    assert float(challenger) == pytest.approx(2.3)
    with pytest.raises(RuntimeError):
        module.combine_arm_loss(base, endpoint, 0.2)

    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    exposed = module.hard_straight_through(logits)
    assert exposed.detach().tolist() == [0.0, 1.0, 1.0]
    exposed.sum().backward()
    assert logits.grad is not None
    assert torch.all(logits.grad > 0)
    assert torch.allclose(logits.grad, torch.sigmoid(logits) * (1 - torch.sigmoid(logits)))


def test_clean_and_endpoint_loss_formulas_and_no_endpoint_quotient_term() -> None:
    module = load_runner()
    logits = torch.tensor([[0.2, -0.4], [0.3, 0.7]], requires_grad=True)
    qlogits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    quotients = torch.tensor([0, 1])
    clean, clean_stats = module.clean_transition_loss(
        logits, qlogits, targets, quotients
    )
    endpoint, endpoint_stats = module.endpoint_residue_loss(logits, targets)
    assert float(clean) == pytest.approx(
        clean_stats["mean_bit_bce"]
        + 0.45 * clean_stats["worst_position_mean_bce"]
        + 0.10 * clean_stats["quotient_cross_entropy"]
    )
    assert float(endpoint) == pytest.approx(
        endpoint_stats["mean_bit_bce"]
        + 0.45 * endpoint_stats["worst_position_mean_bce"]
    )
    endpoint.backward()
    assert qlogits.grad is None


def test_batch_work_is_matched_and_window_inputs_are_not_corrupted() -> None:
    module = load_runner()
    config = load_config()
    pairs = [
        {
            "failure": fake_window(kind="failure", identity=index, start=index),
            "success": fake_window(kind="success", identity=100 + index, start=index),
        }
        for index in range(8)
    ]
    before = copy.deepcopy(pairs)
    windows = module.CalibrationExperiment.ordered_batch_windows(pairs, 0)
    assert len(windows) == 16
    assert [window["identity_index"] for window in windows[:8]] == list(range(8))
    assert [window["identity_index"] for window in windows[8:]] == list(range(100, 108))
    assert pairs == before
    work = module.matched_work_payload(config, pairs)
    assert "endpoint_loss_weight" not in json.dumps(work)
    source = inspect.getsource(module.CalibrationExperiment.training_update)
    assert "hard_straight_through" in source
    assert '"synthetic_state_corruption": False' in source


@pytest.mark.parametrize(
    ("modulus", "expected"),
    [(8, "b0"), (9, "b0"), (10, "b1"), (11, "b1"), (12, "b2"), (13, "b2"), (14, "b3"), (15, "b3")],
)
def test_normalized_modulus_band_exact_boundaries(modulus: int, expected: str) -> None:
    module = load_runner()
    assert module.normalized_modulus_band(modulus) == expected


def test_matching_is_exact_ordered_no_reuse_no_relaxation_and_allows_distinct_starts() -> None:
    module = load_runner()
    failures = [
        fake_window(kind="failure", identity=1, start=0),
        fake_window(kind="failure", identity=2, start=0),
        fake_window(kind="failure", identity=3, start=1),
    ]
    successes = [
        fake_window(kind="success", identity=10, start=0),
        fake_window(kind="success", identity=11, start=0),
        fake_window(kind="success", identity=10, start=1),
        fake_window(kind="success", identity=12, start=1),
    ]
    pairs = module.match_windows(failures, successes, 3)
    assert [pair["success"]["identity_index"] for pair in pairs] == [10, 11, 10]
    assert [pair["success"]["window_start_index"] for pair in pairs] == [0, 0, 1]
    assert len(
        {
            (pair["success"]["identity_index"], pair["success"]["window_start_index"])
            for pair in pairs
        }
    ) == 3
    mismatched = copy.deepcopy(successes[0])
    mismatched["normalized_modulus_band"] = "b1"
    with pytest.raises(RuntimeError, match="inconsistent|shortfall"):
        module.match_windows([failures[0]], [mismatched], 1)


def test_matching_rejects_incomplete_or_nonexact_windows() -> None:
    module = load_runner()
    failure = fake_window(kind="failure", identity=1, start=248)
    success = fake_window(kind="success", identity=2, start=248)
    assert len(module.match_windows([failure], [success], 1)) == 1
    invalid = copy.deepcopy(failure)
    invalid["window_start_index"] = 249
    invalid["first_error_index"] = 249
    with pytest.raises(RuntimeError, match="complete K=8"):
        module.match_windows([invalid], [success], 1)
    nonexact = copy.deepcopy(success)
    nonexact["natural_states"][3] += 1
    with pytest.raises(RuntimeError, match="not exact"):
        module.match_windows([failure], [nonexact], 1)


def test_strong_corpus_verifier_replays_and_recomputes_every_semantic_field() -> None:
    module = load_runner()
    payload, replayed = exact_corpus_fixture(module)
    receipt = module.verify_calibration_corpus_payload(
        payload,
        mode="test",
        root_seed=TEST_ROOT,
        replayed_trajectories=replayed,
        target_pairs=1,
    )
    assert receipt["full_model_replay_verified"] is True
    assert receipt["semantic_labels_verified"] is True
    assert receipt["deterministic_matching_verified"] is True
    assert receipt["identity_trajectories"] == 2
    assert receipt["matched_pairs"] == 1


@pytest.mark.parametrize(
    "tamper",
    (
        "modulus_bit_length",
        "normalized_band",
        "input_bit",
        "quotient",
        "teacher_step",
        "window_natural_slice",
        "global_first_error",
        "complete_natural_trajectory",
    ),
)
def test_strong_corpus_verifier_rejects_semantic_tampering(tamper: str) -> None:
    module = load_runner()
    payload, replayed = exact_corpus_fixture(module)
    failure = payload["pairs"][0]["failure"]
    if tamper == "modulus_bit_length":
        failure["modulus_bit_length"] -= 1
    elif tamper == "normalized_band":
        failure["normalized_modulus_band"] = "b3"
    elif tamper == "input_bit":
        failure["input_bits"][0] ^= 1
    elif tamper == "quotient":
        failure["quotients"][0] += 1
    elif tamper == "teacher_step":
        failure["teacher_states"][1] += 1
    elif tamper == "window_natural_slice":
        failure["natural_states"][2] += 1
    elif tamper == "global_first_error":
        payload["trajectory_manifest"][0]["first_error_index"] = 1
    elif tamper == "complete_natural_trajectory":
        payload["trajectory_manifest"][0]["complete_predicted_states"][2] += 1
    with pytest.raises(RuntimeError):
        module.verify_calibration_corpus_payload(
            payload,
            mode="test",
            root_seed=TEST_ROOT,
            replayed_trajectories=replayed,
            target_pairs=1,
        )


def test_strong_corpus_verifier_requires_independent_replay() -> None:
    module = load_runner()
    payload, _ = exact_corpus_fixture(module)
    with pytest.raises(RuntimeError, match="requires independent model replay"):
        module.verify_calibration_corpus_payload(
            payload,
            mode="test",
            root_seed=TEST_ROOT,
            replayed_trajectories=None,
            target_pairs=1,
        )


@pytest.mark.parametrize("position", [0, 248, 249])
def test_first_error_zero_based_boundaries(position: int) -> None:
    module = load_runner()
    expected = [0] * 256
    predicted = list(expected)
    predicted[position] = 1
    assert module.first_error_index(predicted, expected) == position
    assert module.first_error_index(expected, expected) == 257
    assert position <= 248 if position != 249 else position > 248


def test_recovered_and_introduced_drift_are_separate() -> None:
    module = load_runner()
    result = module.paired_drift_counts(
        [True, False, True, False, False],
        [True, True, False, False, True],
    )
    assert result == {
        "cases": 5,
        "recovered_control_wrong_challenger_exact": 2,
        "introduced_drift_control_exact_challenger_wrong": 1,
    }


def gate_record() -> dict:
    return {
        "width256_rollout_accuracy": 0.50,
        "mean_clipped_first_error_index": 100.0,
        "width64_rollout_accuracy": 0.95,
        "width128_rollout_accuracy": 0.90,
        "width256_transition_accuracy": 0.995,
        "invalid_rollout_rate": 0.01,
        "exact_repeat_and_determinism": True,
    }


def test_every_screen_gate_passes_at_inclusive_equality_and_each_failure_fails() -> None:
    module = load_runner()
    config = load_config()
    control = gate_record()
    challenger = {
        **gate_record(),
        "width256_rollout_accuracy": 0.53,
        "mean_clipped_first_error_index": 100.0,
        "width64_rollout_accuracy": 0.94,
        "width128_rollout_accuracy": 0.89,
    }
    integrity = valid_scientific_integrity(module)
    at_boundary = module.screen_gate_results(
        control, challenger, integrity, config["screen"]["gates"]
    )
    assert at_boundary["all_passed"] is True
    failure_mutations = {
        "width256_rollout_accuracy": 0.529999,
        "mean_clipped_first_error_index": 99.999,
        "width64_rollout_accuracy": 0.939999,
        "width128_rollout_accuracy": 0.889999,
        "width256_transition_accuracy": 0.994999,
        "invalid_rollout_rate": 0.010001,
    }
    for key, value in failure_mutations.items():
        failed = {**challenger, key: value}
        assert not module.screen_gate_results(
            control, failed, integrity, config["screen"]["gates"]
        )["all_passed"]
    assert not module.screen_gate_results(
        control,
        challenger,
        {**integrity, "matched_work_valid": False},
        config["screen"]["gates"],
    )["all_passed"]


def test_every_confirmation_pair_gate_has_same_equality_boundaries() -> None:
    module = load_runner()
    config = load_config()
    control = gate_record()
    challenger = {
        **gate_record(),
        "width256_rollout_accuracy": 0.53,
        "width64_rollout_accuracy": 0.94,
        "width128_rollout_accuracy": 0.89,
    }
    result = module.confirmation_pair_gate_results(
        control,
        challenger,
        valid_scientific_integrity(module),
        config["confirmation"]["gates_per_pair"],
    )
    assert result["all_passed"] is True
    challenger["width256_rollout_accuracy"] -= 1e-6
    assert not module.confirmation_pair_gate_results(
        control,
        challenger,
        valid_scientific_integrity(module),
        config["confirmation"]["gates_per_pair"],
    )["all_passed"]


def test_scientific_integrity_schema_fails_closed_for_missing_extra_or_either_arm() -> None:
    module = load_runner()
    config = load_config()
    control = gate_record()
    challenger = {
        **gate_record(),
        "width256_rollout_accuracy": 0.53,
        "width64_rollout_accuracy": 0.94,
        "width128_rollout_accuracy": 0.89,
    }
    valid = valid_scientific_integrity(module)
    assert module.validate_scientific_integrity(valid)
    mutations = []
    missing = dict(valid)
    missing.pop("evaluator_hash_valid")
    mutations.append(missing)
    mutations.append({**valid, "arbitrary": True})
    mutations.append({**valid, "matched_work_valid": 1})
    mutations.append({**valid, "control_exact_repeat_deterministic": False})
    mutations.append({**valid, "challenger_exact_repeat_deterministic": False})
    for integrity in mutations:
        assert not module.validate_scientific_integrity(integrity)
        result = module.screen_gate_results(
            control, challenger, integrity, config["screen"]["gates"]
        )
        assert result["all_passed"] is False
        assert result["integrity"] is False
    assert set(config["scientific_integrity_protocol"]["required_keys"]) == set(
        module.REQUIRED_SCIENTIFIC_INTEGRITY_KEYS
    )


def test_pooled_clopper_pearson_edges_and_strict_positive_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_runner()
    none = module.pooled_conditional_clopper_pearson(0, 0)
    assert none["net_exact_gain_lower"] == 0
    assert none["strict_positive_gate_passed"] is False
    zero_recovered = module.pooled_conditional_clopper_pearson(0, 5)
    assert zero_recovered["one_sided_clopper_pearson_lower"] == 0
    assert zero_recovered["net_exact_gain_lower"] < 0
    positive = module.pooled_conditional_clopper_pearson(20, 0)
    assert positive["net_exact_gain_lower"] > 0
    assert positive["strict_positive_gate_passed"] is True
    monkeypatch.setattr(module.beta, "ppf", lambda *_args, **_kwargs: 0.5)
    exact_boundary = module.pooled_conditional_clopper_pearson(1, 1)
    assert exact_boundary["net_exact_gain_lower"] == 0
    assert exact_boundary["strict_positive_gate_passed"] is False
    monkeypatch.setattr(module.beta, "ppf", lambda *_args, **_kwargs: 0.500001)
    above = module.pooled_conditional_clopper_pearson(1, 1)
    assert above["strict_positive_gate_passed"] is True


def test_scientific_resource_hashing_and_composite_cache_invalidation() -> None:
    module = load_runner()
    config = load_config()
    science = module.scientific_protocol_sha256(config)
    resource = module.resource_protocol_sha256(config)
    assert science == "f4cc038b205289203569dceed0ee5e833d406a8d0f79aceaf6c5cd393353c98e"
    assert resource == "8d930606bf3ac8d5f1e6d1394a5000f2883bd3a1ff54d92c0d8af88f347ed37b"
    assert resource != "f0800dff74038df7465abff8360fe572f603bdab4af26f547b130286e169b198"
    assert "readiness_recovery" not in config
    assert "historical_readiness_recovery_root_20261160_to_20261161" in config["resource_protocol"]
    resolved = copy.deepcopy(config)
    resolved["status"] = "calibration-complete-awaiting-judge"
    resolved["phase"] = "post-calibration"
    resolved["corpus"]["resolved_corpus_mining_ceiling"] = {"work": 123}
    resolved["resource_protocol"]["status"] = "complete"
    resolved["resource_protocol"]["resolved"] = {"wall": 456}
    resolved["resource_protocol"]["projections"]["resolved"] = {"cpu": 789}
    assert module.scientific_protocol_sha256(resolved) == science
    assert module.resource_protocol_sha256(resolved) == resource
    changed_science = copy.deepcopy(config)
    changed_science["training"]["batch_size"] = 17
    assert module.scientific_protocol_sha256(changed_science) != science
    changed_resource = copy.deepcopy(config)
    changed_resource["resource_protocol"]["projections"]["soft_wall_seconds"] = (
        "1.30 * projected"
    )
    assert module.resource_protocol_sha256(changed_resource) != resource
    fields = {
        "checkpoint_sha256": "a" * 64,
        "evaluator_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "seed": TEST_ROOT,
        "environment_sha256": "e" * 64,
        "runner_sha256": "f" * 64,
        "scientific_protocol_sha256": science,
        "resource_protocol_sha256": resource,
    }
    first = module.composite_cache_key(fields, 1)
    changed = {**fields, "dataset_sha256": "0" * 64}
    assert module.composite_cache_key(changed, 1) != first
    assert module.composite_cache_key(fields, 2) != first
    artifact = {"source": {"executed": "x", "resolved_config_sha256": "pending"}, "data": 1}
    core = module.calibration_evidence_core_sha256(artifact)
    artifact["source"]["resolved_config_sha256"] = "f" * 64
    artifact["calibration_evidence_core_sha256"] = core
    artifact["calibration_evidence_core_hash_scope"] = "documented"
    assert module.calibration_evidence_core_sha256(artifact) == core


def test_current_lifecycle_fields_distinguish_historical_and_unopened_roots() -> None:
    config = load_config()
    roles = config["seed_roles"]
    assert roles["calibration_root"]["current_status"] == "consumed-historical-non-scientific"
    assert roles["corpus_root"]["current_status"] == "consumed-complete-immutable"
    assert all(
        roles[name]["current_status"] == "unopened-prohibited"
        for name in (
            "screen_training_root", "screen_evaluation_root",
            "confirmation_1_training_root", "confirmation_1_evaluation_root",
            "confirmation_2_training_root", "confirmation_2_evaluation_root",
        )
    )
    assert roles["completed_readiness_root"]["current_status"].endswith("-historical")
    assert roles["post_repair_readiness_calibration_root"]["status_at_preregistration"] == "reserved-unopened-unauthorized"
    assert config["corpus"]["lifecycle_status"] == "consumed-complete-immutable-historical"
    assert config["authorization"]["corpus_root"] == "consumed-complete-immutable-historical"
    assert "reconstruction" in config["authorization"]["screen_roots"]
    prereg = (ROOT / "knowledge/autoresearch/ablations/exp-648-structured-trajectory-consistency.md").read_text()
    assert "Every seed below is currently unused and unopened" not in prereg
    registry = (ROOT / "experiments/registry.yaml").read_text()
    assert "historical_implementation: approved-recovery-base-and-authorization-child-pushed" in registry
    assert not any(
        line.strip() == "implementation: approved-recovery-base-and-authorization-child-pushed"
        for line in registry.splitlines()
    )


def synthetic_repeat(module, config: dict, factor: float, identities: int = 96) -> dict:
    def timing(wall: float) -> dict:
        return {
            "wall_seconds": wall * factor,
            "aggregate_cpu_seconds": wall * factor * 4,
            "peak_rss_bytes": 1_000_000,
        }

    manifest = module.dependency_source_manifest()
    cache_fields = {
        "checkpoint_sha256": "a" * 64,
        "evaluator_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "seed": TEST_ROOT,
        "environment_sha256": "e" * 64,
        "runner_sha256": "f" * 64,
        "scientific_protocol_sha256": module.scientific_protocol_sha256(config),
        "resource_protocol_sha256": module.resource_protocol_sha256(config),
        "workload_sha256": module.workload_sha256(config),
        "dependency_manifest_sha256": manifest["manifest_sha256"],
    }
    return {
        "complete": True,
        "repeat_index": 1,
        "cache_key_fields": cache_fields,
        "workload": module.workload_payload(config),
        "components": {
            "startup_model_loading": {"units": 1, "timing": timing(2)},
            "mining": {
                "matched_pairs": 64,
                "windows": 128,
                "identity_trajectories": identities,
                "timing": timing(64),
            },
            "corpus_integrity_validation": {
                "windows": 128,
                "matched_pairs": 64,
                "timing": timing(1),
            },
            "training": {
                "control": {"updates": 20, "batch_size": 16, "k": 8, "timing": timing(20)},
                "challenger": {"updates": 20, "batch_size": 16, "k": 8, "timing": timing(30)},
            },
            "checkpoint": {
                "control": {"writes": 1, "timing": timing(1)},
                "challenger": {"writes": 1, "timing": timing(2)},
            },
            "transitions": {
                arm: {
                    "128": {"cases": 200, "timing": timing(2)},
                    "256": {"cases": 200, "timing": timing(4)},
                }
                for arm in ("control", "challenger")
            },
            "rollouts": {
                arm: {
                    "64": {"cases": 20, "timing": timing(4)},
                    "128": {"cases": 20, "timing": timing(8)},
                    "256": {"cases": 20, "timing": timing(16)},
                }
                for arm in ("control", "challenger")
            },
            "hash_serialization": {"units": 1, "timing": timing(1)},
        },
        "integrity": {
            "matched_arm_work_sha256_equal": True,
            "identical_initial_tensor_sha256": True,
            "dependency_source_manifest": manifest,
        },
        "integrity_checks": {
            key: True for key in module.REQUIRED_CALIBRATION_INTEGRITY_BOOLEAN_KEYS
        },
        "total_timing": timing(200),
        "fidelity_reduced": False,
    }


def test_complete_calibration_workload_and_dynamic_budget_projection() -> None:
    module = load_runner()
    config = load_config()
    repeats = [synthetic_repeat(module, config, factor) for factor in (1.0, 1.5, 2.0)]
    for repeat in repeats:
        module.validate_calibration_repeat(repeat, config)
    launcher = {
        "wall_seconds": 3.0,
        "aggregate_cpu_seconds": 5.0,
        "peak_rss_bytes": 1_000_000,
    }
    resolved = module.resolve_dynamic_budgets(config, repeats, launcher)
    assert resolved["complete_timed_repeats"] == 3
    assert resolved["rate_selection"] == "maximum-complete-repeat-and-slower-arm"
    assert resolved["component_rates"]["wall_seconds"][
        "launcher_import_config_environment_dependency_setup"
    ] == 3.0
    assert resolved["component_rates"]["aggregate_cpu_seconds"][
        "training_per_update_slower_arm"
    ] > resolved["component_rates"]["wall_seconds"]["training_per_update_slower_arm"]
    for resource in ("wall_seconds", "aggregate_cpu_seconds"):
        rates = resolved["component_rates"][resource]
        screen = resolved["screen"][resource]
        components = screen["component_projection"]
        expected_replay = rates["corpus_integrity_per_window"] * 1_024
        assert components["strong_semantic_integrity_replay_1024_windows"] == pytest.approx(
            expected_replay
        )
        assert components["corpus_mining_512_pairs"] == pytest.approx(
            rates["mining_per_matched_pair"] * 512
        )
        assert screen["projected_component_seconds"] == pytest.approx(
            sum(components.values())
        )
    assert resolved["screen"]["wall_seconds"]["component_projection"][
        "strong_semantic_integrity_replay_1024_windows"
    ] == pytest.approx(16.0)
    assert resolved["screen"]["wall_seconds"]["projected_component_seconds"] == pytest.approx(
        3681.0
    )
    assert resolved["screen"]["aggregate_cpu_seconds"]["component_projection"][
        "strong_semantic_integrity_replay_1024_windows"
    ] == pytest.approx(64.0)
    assert resolved["screen"]["aggregate_cpu_seconds"][
        "projected_component_seconds"
    ] == pytest.approx(14717.0)
    for phase in ("screen", "confirmation_pair", "confirmation_total"):
        for resource in ("wall_seconds", "aggregate_cpu_seconds"):
            record = resolved[phase][resource]
            assert record["soft_budget_seconds"] > record["projected_component_seconds"]
            assert record["hard_runaway_ceiling_seconds"] == pytest.approx(
                1.5 * record["soft_budget_seconds"]
            )
    corpus = resolved["corpus_mining_ceiling"]
    assert corpus["declared_work_unit"] == "complete_width256_source_identity_trajectory"
    assert corpus["required_matched_pairs"] == 512
    assert corpus["projected_identity_trajectories"] == 768
    assert corpus["hard_identity_trajectory_ceiling"] > corpus["projected_identity_trajectories"]
    assert corpus["matching_relaxation_allowed"] is False
    assert resolved["fidelity_reduced"] is False


def test_complete_repeat_validation_rejects_reduced_fidelity_or_counts() -> None:
    module = load_runner()
    config = load_config()
    record = synthetic_repeat(module, config, 1.0)
    reduced = copy.deepcopy(record)
    reduced["components"]["rollouts"]["control"]["256"]["cases"] = 19
    with pytest.raises(RuntimeError, match="rollout workload"):
        module.validate_calibration_repeat(reduced, config)
    reduced = copy.deepcopy(record)
    reduced["fidelity_reduced"] = True
    with pytest.raises(RuntimeError, match="fidelity"):
        module.validate_calibration_repeat(reduced, config)


def test_repeat_validation_rejects_peak_rss_below_component_high_water() -> None:
    module = load_runner()
    config = load_config()
    record = synthetic_repeat(module, config, 1.0)
    record["components"]["training"]["control"]["timing"]["peak_rss_bytes"] = 2_000_000
    with pytest.raises(RuntimeError, match="peak RSS"):
        module.validate_calibration_repeat(record, config)


def test_cache_reuse_validation_is_exact_and_hash_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_runner()
    config = load_config()
    dependency_paths = {
        "cellular_model": "deps/model.py",
        "exp300_data_evaluator": "deps/run.py",
    }
    sources = {}
    for name, relative in dependency_paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = {"path": relative, "sha256": module.sha256(path)}
    manifest_payload = {
        "schema_version": "exp648-executable-dependency-manifest-v1",
        "sources": sources,
    }
    manifest = {
        **manifest_payload,
        "manifest_sha256": module.sha256_text(module.canonical_json(manifest_payload)),
    }
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "DEPENDENCY_SOURCE_PATHS", dependency_paths)
    monkeypatch.setattr(module, "dependency_source_manifest", lambda: copy.deepcopy(manifest))
    record = synthetic_repeat(module, config, 1.0)
    record["repeat_index"] = 2
    record["integrity"]["dependency_source_manifest"] = manifest
    record["cache_key_fields"]["dependency_manifest_sha256"] = manifest[
        "manifest_sha256"
    ]
    for relative, content in (
        ("unit.json", "unit"),
        ("corpus.json", "corpus"),
        ("serialization.json", "serialization"),
        ("control.safetensors", "control"),
        ("challenger.safetensors", "challenger"),
    ):
        (tmp_path / relative).write_text(content)
    record["unit_record"] = {
        "path": "unit.json",
        "sha256": module.sha256(tmp_path / "unit.json"),
    }
    record["components"]["mining"].update(
        {"corpus_path": "corpus.json", "corpus_sha256": module.sha256(tmp_path / "corpus.json")}
    )
    record["components"]["hash_serialization"].update(
        {
            "path": "serialization.json",
            "sha256": module.sha256(tmp_path / "serialization.json"),
        }
    )
    for arm in ("control", "challenger"):
        record["components"]["checkpoint"][arm].update(
            {
                "path": f"{arm}.safetensors",
                "sha256": module.sha256(tmp_path / f"{arm}.safetensors"),
            }
        )
    record["cache_key"] = module.composite_cache_key(record["cache_key_fields"], 2)
    module.validate_cached_repeat_record(
        record,
        config,
        repeat_index=2,
        expected_cache_key_fields=record["cache_key_fields"],
    )
    mutations = []
    wrong_index = copy.deepcopy(record)
    wrong_index["repeat_index"] = 1
    mutations.append(wrong_index)
    wrong_key = copy.deepcopy(record)
    wrong_key["cache_key"] = "0" * 64
    mutations.append(wrong_key)
    wrong_unit = copy.deepcopy(record)
    wrong_unit["unit_record"]["sha256"] = "0" * 64
    mutations.append(wrong_unit)
    wrong_integrity = copy.deepcopy(record)
    wrong_integrity["integrity_checks"]["matched_arm_work_valid"] = False
    mutations.append(wrong_integrity)
    extra_integrity = copy.deepcopy(record)
    extra_integrity["integrity_checks"]["extra"] = True
    mutations.append(extra_integrity)
    for mutated in mutations:
        with pytest.raises(RuntimeError):
            module.validate_cached_repeat_record(
                mutated,
                config,
                repeat_index=2,
                expected_cache_key_fields=record["cache_key_fields"],
            )


def test_launcher_setup_and_environment_hash_scopes_are_explicit() -> None:
    module = load_runner()
    config = load_config()
    source = RUNNER.read_text()
    assert source.index("_MODULE_IMPORT_WALL_STARTED") < source.index("import mlflow")
    assert config["resource_protocol"]["calibration"][
        "launcher_import_config_environment_dynamic_dependency_setup"
    ] == "measured-from-top-of-module"
    environment = {"a": 1, "b": [2, 3]}
    canonical = module.sha256_text(module.canonical_json(environment))
    pretty = module.sha256_bytes(module.canonical_json_bytes(environment, pretty=True))
    assert canonical != pretty
    measurement = module.Measurement(module.psutil.Process())
    observed = measurement.stop().payload()
    assert observed["peak_rss_bytes"] >= module.psutil.Process().memory_info().rss


def test_reused_complete_time_is_charged_separately_from_active_wall_and_cpu() -> None:
    module = load_runner()
    process = module.psutil.Process()
    guard = module.CalibrationRunawayGuard(
        process, wall_ceiling=10_000, cpu_ceiling=40_000
    )
    guard.add_reused(123.0, 456.0)
    wall, cpu = guard.charged()
    assert wall >= 123.0
    assert cpu >= 456.0
    assert guard.reused_wall_seconds == 123.0
    assert guard.reused_cpu_seconds == 456.0
    too_small = module.CalibrationRunawayGuard(
        process, wall_ceiling=100.0, cpu_ceiling=40_000
    )
    with pytest.raises(TimeoutError, match="wall planning ceiling"):
        too_small.add_reused(100.0, 0.0)


def test_prior_superseded_calibration_is_charged_but_not_classified_as_reuse() -> None:
    module = load_runner()
    process = module.psutil.Process()
    priors = module.PRIOR_SUPERSEDED_CALIBRATIONS
    prior = module.PRIOR_SUPERSEDED_CALIBRATION_CHARGE
    assert [record["evidence_sha256"] for record in priors] == [
        "f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4",
        "d965d944330fe66f4c333a09479e12b2db893ad19c18492f60ea24dbf68a8ad6",
    ]
    guard = module.CalibrationRunawayGuard(
        process,
        wall_ceiling=14_400,
        cpu_ceiling=57_600,
        prior_superseded_wall_seconds=prior["wall_seconds"],
        prior_superseded_cpu_seconds=prior["aggregate_cpu_seconds"],
    )
    wall, cpu = guard.charged()
    assert wall >= 3812.769163506
    assert cpu >= 15762.009662108001
    assert guard.reused_wall_seconds == guard.reused_cpu_seconds == 0
    assert all(record["eligible_for_projection_rates"] is False for record in priors)
    assert all(record["classified_as_reused_repeat"] is False for record in priors)


def test_superseded_calibration_evidence_is_preserved_byte_for_byte() -> None:
    module = load_runner()
    scratch_root = ROOT / ".scratch/exp-648-structured-trajectory-consistency/superseded"
    tracked_root = ROOT / "experiments/runs/exp-648-structured-trajectory-consistency/superseded"
    f522 = "f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4"
    d965 = "d965d944330fe66f4c333a09479e12b2db893ad19c18492f60ea24dbf68a8ad6"
    f522_manifest = json.loads((scratch_root / f522 / "provenance.json").read_text())
    d965_manifest = json.loads((scratch_root / d965 / "provenance.json").read_text())
    assert module.sha256(scratch_root / f522 / "calibration.json") == (
        "f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4"
    )
    assert module.sha256(scratch_root / d965 / "calibration.json") == d965
    assert module.sha256(scratch_root / f522 / "environment.json") == (
        "d5754d46d762643e9488a85924208ccd83a959206a14c7b7041cea7d332819c9"
    )
    assert f522_manifest["prior_superseded_calibration_charge"] == {
        "wall_seconds": 1478.023299573,
        "aggregate_cpu_seconds": 6085.320000000001,
        "eligible_for_new_projection_rates": False,
        "classified_as_reused_repeat": False,
        "charged_to_cumulative_planning_ceiling": True,
    }
    assert d965_manifest["completed_calibration_charge"] == {
        "wall_seconds": 2334.745863933,
        "aggregate_cpu_seconds": 9676.689662108,
        "eligible_for_new_projection_rates": False,
        "classified_as_reused_repeat": False,
        "charged_to_cumulative_planning_ceiling": True,
    }
    assert d965_manifest["executed_pre_resolution_config"] == {
        "recorded_sha256": "cbc50fd22333126db992c74de307b9f344be8e6769a7d8f46d0f954678c3f491",
        "bytes_preserved": False,
        "independently_reconstructable": False,
        "limitation": "The exact executed pre-resolution config bytes were not preserved before post-calibration resolution and cannot be independently reconstructed. The hash remains a source claim inside the preserved calibration artifact, not a byte-verifiable file receipt.",
    }
    for evidence in (f522, d965):
        for name in (
            "calibration.json",
            "environment.json",
            "resolved-config.yaml",
            "run.py",
            "provenance.json",
        ):
            assert (scratch_root / evidence / name).read_bytes() == (
                tracked_root / evidence / name
            ).read_bytes()


def test_executed_config_is_preserved_before_root_authorization_with_raw_tracked_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_runner()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_bytes = b"run_id: exact-pre-resolution\nstatus: pending\n"
    config_path.write_bytes(config_bytes)
    receipt = module.preserve_executed_config(
        config_path,
        tmp_path / "raw",
        tmp_path / "tracked",
    )
    assert (tmp_path / receipt["raw_path"]).read_bytes() == config_bytes
    assert (tmp_path / receipt["tracked_path"]).read_bytes() == config_bytes
    assert receipt["sha256"] == module.sha256_bytes(config_bytes)
    assert receipt["raw_sha256"] == receipt["tracked_sha256"] == receipt["sha256"]
    assert receipt["raw_tracked_byte_identical"] is True
    assert receipt["preserved_before_root_authorization"] is True
    source = inspect.getsource(module.CalibrationExperiment.__init__)
    assert source.index("preserve_executed_config") < source.index("authorize_root")


def test_run_recovery_order_requires_fresh_warmup_before_reuse() -> None:
    module = load_runner()
    source = inspect.getsource(module.CalibrationExperiment.run)
    warmup = source.index("execute_complete_unit(warmup_name")
    warmup_finished = source.index("fresh-complete-warmup-finished")
    reuse_charge = source.index("guard.add_reused")
    missing_repeat = source.index("timed-repeat-{repeat_index}")
    assert warmup < warmup_finished < reuse_charge < missing_repeat
    assert "cached_record is not None" in source


def test_wall_and_cpu_planning_ceilings_are_distinct_and_fidelity_is_binding() -> None:
    config = load_config()
    calibration = config["resource_protocol"]["calibration"]
    assert calibration["initial_planning_hard_ceiling_wall_seconds"] == 25_000
    assert calibration["initial_planning_hard_ceiling_aggregate_cpu_seconds"] == 100_000
    source = inspect.getsource(load_runner().CalibrationExperiment.__init__)
    assert 'wall_ceiling=float(calibration["initial_planning_hard_ceiling_wall_seconds"])' in source
    assert 'calibration["initial_planning_hard_ceiling_aggregate_cpu_seconds"]' in source
    assert config["resource_protocol"]["measurement"][
        "wall_and_cpu_are_not_interchangeable"
    ] is True
    assert config["resource_protocol"]["fidelity_may_be_reduced_to_fit_budget"] is False
    assert config["resource_protocol"]["counts_or_work_may_be_reduced_to_fit_budget"] is False


def test_raw_and_tracked_calibration_are_byte_identical_when_present() -> None:
    raw = ROOT / ".scratch/exp-648-structured-trajectory-consistency/calibration.json"
    tracked = ROOT / "experiments/runs/exp-648-structured-trajectory-consistency/calibration.json"
    if not raw.exists() and not tracked.exists():
        pytest.skip("calibration has not completed")
    assert raw.is_file() and tracked.is_file()
    assert raw.read_bytes() == tracked.read_bytes()
