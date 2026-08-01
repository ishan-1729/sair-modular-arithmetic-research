from __future__ import annotations

import copy
import builtins
import importlib.util
import inspect
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/exp_649_stochastic_trajectory_breadth/run.py"
CONFIG = ROOT / "configs/exp_649_stochastic_trajectory_breadth.yaml"
SEED_RETIREMENT_MANIFEST = (
    ROOT / "experiments/exp_649_stochastic_trajectory_breadth/seed_retirement_manifest.json"
)
TEST_ONLY_PARITY_ROOT_SEED = 649_900_001


def load_runner():
    spec = importlib.util.spec_from_file_location("exp649_test_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


class TinyCell(nn.Module):
    def __init__(self, dilations: list[int]) -> None:
        super().__init__()
        self.dilations = dilations
        self.stem = nn.Conv1d(6, 1, 1, bias=False)
        self.output_norm = nn.Identity()
        self.local_head = nn.Conv1d(1, 1, 1, bias=False)
        nn.init.constant_(self.stem.weight, 0.125)
        nn.init.constant_(self.local_head.weight, 1.0)
        self.encoded_state_planes: list[torch.Tensor] = []

    def recurrent_update(
        self, state: torch.Tensor, inputs: torch.Tensor, dilation: int
    ) -> torch.Tensor:
        self.encoded_state_planes.append(inputs[:, 0].detach().clone())
        return state + inputs[:, :1] * 0.01

    def forward(self, encoded: torch.Tensor):
        hidden = self.stem(encoded)
        for dilation in self.dilations:
            hidden = self.recurrent_update(hidden, encoded, dilation)
        return self.local_head(F.silu(self.output_norm(hidden))).squeeze(1), None


def test_frozen_k_mapping_and_correct_tier0_width256_geometry() -> None:
    module = load_runner()
    config = load_config()
    module.validate_frozen_config(config)
    specs = module.path_specs(config)
    assert len(specs) == 8
    assert [(p.noise_seed, p.orientation) for p in specs[2:]] == [
        (649001, "original"),
        (649002, "swapped"),
        (649003, "original"),
        (649004, "swapped"),
        (649005, "original"),
        (649006, "swapped"),
    ]
    geometry = config["candidate_geometry"]
    assert geometry["total_rows"] == 1100
    assert geometry["future_width256_rows"]["total"] == 110
    tier0, tier7 = geometry["future_width256_rows"]["batches"]
    assert (tier0["count"], tier0["operand_bits_min"], tier0["operand_bits_max"]) == (
        10,
        65,
        128,
    )
    assert (tier7["count"], tier7["operand_bits"]) == (100, 512)
    assert module.sha256(ROOT / config["candidate_model"]) == config["candidate_model_sha256"]
    assert module.sha256(ROOT / config["candidate_manifest"]) == config["candidate_manifest_sha256"]
    assert module.sha256(ROOT / config["candidate_artifact"]) == config["candidate_artifact_sha256"]
    assert module.sha256(ROOT / config["source_model"]) == config["source_model_sha256"]
    portable_fixture = (
        ROOT
        / ".scratch/exp-660-symbol-equivariant-formal/fixtures/exp609-weights.safetensors"
    )
    assert portable_fixture.is_file() and not portable_fixture.is_symlink()
    assert module.sha256(portable_fixture) == config["source_checkpoint_sha256"]
    config["source_checkpoint"] = str(portable_fixture.relative_to(ROOT))
    hashes = module.current_source_hashes(config)
    assert hashes["official_evaluator_commit_sha"] == config["official_evaluator"]["commit_sha"]
    for name, record in config["official_evaluator"]["source_files"].items():
        assert hashes[f"official_{name}_git_blob_oid"] == record["git_blob_oid"]
        assert (
            hashes[f"official_{name}_canonical_blob_sha256"]
            == record["canonical_blob_sha256"]
        )
        assert hashes[f"official_{name}_checkout_sha256"] in record[
            "known_checkout_sha256"
        ]
    assert (
        hashes["seed_retirement_manifest_sha256"]
        == config["seed_retirement_manifest_sha256"]
    )


def test_seed_retirement_manifest_is_pinned_static_and_semantically_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_runner()
    config = load_config()
    payload = json.loads(SEED_RETIREMENT_MANIFEST.read_text())
    module.validate_seed_retirement_manifest_payload(payload)
    assert module.verify_seed_retirement_manifest(config) == module.sha256(
        SEED_RETIREMENT_MANIFEST
    )
    assert payload["schema_version"] == "exp649-seed-retirement-v1"
    assert payload["incident"]["status"] == "contained-root-retired"
    assert payload["seed_roles"] == {
        "retired_calibration_root": 20261148,
        "replacement_calibration_input_only_root": 20261150,
        "formal_unopened_root": 20261149,
        "test_only_parity_root": 649900001,
    }
    rendered = module.canonical_json(payload)
    assert "config_sha256" not in rendered
    assert "runner_sha256" not in rendered

    wrong_path = copy.deepcopy(config)
    wrong_path["seed_retirement_manifest"] = "experiments/wrong.json"
    with pytest.raises(RuntimeError, match="manifest path changed"):
        module.verify_seed_retirement_manifest(wrong_path)
    wrong_pin = copy.deepcopy(config)
    wrong_pin["seed_retirement_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="manifest pin changed"):
        module.verify_seed_retirement_manifest(wrong_pin)
    wrong_schema = copy.deepcopy(payload)
    wrong_schema["schema_version"] = "exp649-seed-retirement-v0"
    with pytest.raises(RuntimeError, match="schema or roles changed"):
        module.validate_seed_retirement_manifest_payload(wrong_schema)
    wrong_role = copy.deepcopy(payload)
    wrong_role["seed_roles"]["replacement_calibration_input_only_root"] = 20261148
    with pytest.raises(RuntimeError, match="schema or roles changed"):
        module.validate_seed_retirement_manifest_payload(wrong_role)
    monkeypatch.setattr(module, "sha256", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        module.verify_seed_retirement_manifest(config)


def test_official_evaluator_verification_precedes_import_and_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_runner()
    config = load_config()
    invalid_receipt = module._expected_official_source_hashes(config)
    invalid_receipt["official_decoder_canonical_blob_sha256"] = "0" * 64
    invalid_receipt["official_decoder_checkout_sha256"] = config[
        "official_evaluator"
    ]["source_files"]["decoder"]["canonical_blob_sha256"]
    imported_decoder = False
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        nonlocal imported_decoder
        if name == "modchallenge.evaluation.decoder":
            imported_decoder = True
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="verification receipt"):
        module.load_verified_official_decoder(config, invalid_receipt)
    assert not imported_decoder

    source = inspect.getsource(module.run_calibration)
    verification = source.index("verify_official_evaluator_sources(config)")
    decoder = source.index("load_verified_official_decoder")
    inputs = source.index("build_calibration_inputs")
    models = source.index("load_frozen_models")
    assert verification < decoder < inputs < models


def test_official_source_identity_is_canonical_git_blob_and_confined() -> None:
    module = load_runner()
    config = load_config()
    receipt = module.verify_official_evaluator_sources(config)
    module.validate_official_source_receipt(config, receipt)
    for name, record in config["official_evaluator"]["source_files"].items():
        assert record["sha256"] == record["known_checkout_sha256"][0]
        assert record["canonical_blob_sha256"] == record["known_checkout_sha256"][1]
        assert receipt[f"official_{name}_git_blob_oid"] == record["git_blob_oid"]
        assert (
            receipt[f"official_{name}_canonical_blob_sha256"]
            == record["canonical_blob_sha256"]
        )

    escaped = copy.deepcopy(config)
    escaped["official_evaluator"]["source_files"]["config"]["path"] = "AGENTS.md"
    with pytest.raises(RuntimeError, match="escaped its repository"):
        module.verify_official_evaluator_sources(escaped)


def test_frozen_runtime_is_configured_and_verified_before_formal_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_runner()
    config = load_config()
    calls: list[tuple[str, object]] = []
    interop = [2]
    monkeypatch.setattr(
        module.torch, "set_num_threads", lambda value: calls.append(("threads", value))
    )
    monkeypatch.setattr(module.torch, "get_num_interop_threads", lambda: interop[0])
    monkeypatch.setattr(
        module.torch,
        "set_num_interop_threads",
        lambda value: (calls.append(("interop", value)), interop.__setitem__(0, value)),
    )
    monkeypatch.setattr(
        module.torch,
        "use_deterministic_algorithms",
        lambda value: calls.append(("deterministic", value)),
    )
    environment = {
        "torch_threads": 4,
        "torch_interop_threads": 1,
        "deterministic_algorithms": True,
    }
    monkeypatch.setattr(module, "runtime_environment", lambda: dict(environment))
    assert module.configure_frozen_runtime(config) == environment
    assert calls == [("threads", 4), ("interop", 1), ("deterministic", True)]


def test_label_free_test_seed_inputs_exactly_clone_pinned_official_private_inputs() -> None:
    module = load_runner()
    config = load_config()
    assert TEST_ONLY_PARITY_ROOT_SEED == 649_900_001
    protected = {
        config["data"]["calibration_root_seed"],
        config["data"]["formal_root_seed"],
        *config["data"]["retired_root_seed_denylist"],
    }
    assert TEST_ONLY_PARITY_ROOT_SEED not in protected
    observed = module._build_label_free_official_inputs(
        config, TEST_ONLY_PARITY_ROOT_SEED
    )

    from modchallenge.config import EvalConfig
    from modchallenge.testgen.generator import generate_private_test_set

    official = generate_private_test_set(
        master_seed=module._master_seed_bytes(TEST_ONLY_PARITY_ROOT_SEED),
        config=EvalConfig(
            total_problems=1100,
            primes_per_tier=5,
            edge_cases_per_tier=4,
        ),
    )
    for observed_batch, official_tier in zip(
        observed["candidate_batches"], official.tiers, strict=True
    ):
        assert observed_batch["tier_id"] == official_tier.tier_id
        assert [
            {
                "a": record["a"],
                "b": record["b"],
                "p": record["p"],
                "tier_id": observed_batch["tier_id"],
            }
            for record in observed_batch["records"]
        ] == [case.to_input_dict() for case in official_tier.cases]
        assert all("expected" not in record for record in observed_batch["records"])
    assert observed["portfolio_batches"][0]["records"] == observed["candidate_batches"][0][
        "records"
    ][40:50]
    assert observed["portfolio_batches"][1]["records"] == observed["candidate_batches"][7][
        "records"
    ]


def test_noise_is_reproducible_and_continuously_injected_at_all_17_sites() -> None:
    module = load_runner()
    specs = module.path_specs(load_config())
    model = TinyCell([1] * 16)
    encoded = torch.zeros((len(specs) * 2, 6, 1), dtype=torch.float32)

    streams = module.CanonicalNoiseStreams(specs, "calibration", "d" * 64, [4, 9], 1, 17)
    first_noise: list[torch.Tensor] = []
    first = module.noisy_transition(
        model,
        encoded,
        streams,
        [4, 9],
        0,
        [1] * 16,
        trace=lambda _site, _hidden, noise: first_noise.append(noise.clone()),
    )
    streams.reset()
    second_noise: list[torch.Tensor] = []
    second = module.noisy_transition(
        model,
        encoded,
        streams,
        [4, 9],
        0,
        [1] * 16,
        trace=lambda _site, _hidden, noise: second_noise.append(noise.clone()),
    )
    assert len(first_noise) == len(second_noise) == 17
    assert torch.equal(first, second)
    for left, right in zip(first_noise, second_noise, strict=True):
        assert torch.equal(left, right)
        assert torch.count_nonzero(left[:4]) == 0  # two controls x two cases
        assert torch.count_nonzero(left[4:]) > 0


def test_sigma_zero_paths_match_legacy_and_only_hard_state_crosses_bits() -> None:
    module = load_runner()
    specs = module.path_specs(load_config())
    model = TinyCell([1] * 16)
    cases = {"a": [1, 2], "b": [2, 1], "p": [3, 3], "case_index": [0, 1]}
    before = module.state_dict_sha256(model)
    runaway_stages: list[str] = []
    result = module.vectorized_rollout(
        model,
        cases,
        specs,
        "calibration",
        "a" * 64,
        2,
        [1] * 16,
        runaway_guard=runaway_stages.append,
        runaway_check_interval=1,
    )
    original = module.legacy_rollout(
        model, cases, "original", 2, runaway_stages.append, 1
    )
    swapped = module.legacy_rollout(
        model, cases, "swapped", 2, runaway_stages.append, 1
    )
    module.assert_sigma_zero_legacy_equivalence(result, original, swapped)
    assert result["paths"]["control_original"]["prefix_outputs"] == original[
        "prefix_outputs"
    ]
    assert result["flattened_path_case_batch"] == 16
    assert result["state_dict_sha256_before"] == result["state_dict_sha256_after"] == before
    assert all(
        set(plane.unique().tolist()) <= {-1.0, 1.0}
        for plane in model.encoded_state_planes
    )
    assert result["paths"]["control_original"]["orientation"] == "original"
    assert result["paths"]["control_swapped"]["orientation"] == "swapped"
    assert "before vectorized width-256 rollout" in runaway_stages
    assert any("within vectorized rollout" in stage for stage in runaway_stages)
    assert "after vectorized width-256 rollout" in runaway_stages
    assert any("within legacy original rollout" in stage for stage in runaway_stages)
    assert any("within legacy swapped rollout" in stage for stage in runaway_stages)
    multipliers, multiplicands, _ = module._oriented_inputs(cases, specs[:2])
    assert multipliers == cases["a"] + cases["b"]
    assert multiplicands == cases["b"] + cases["a"]
    rerun = module.vectorized_rollout(
        model, cases, specs, "calibration", "a" * 64, 2, [1] * 16
    )
    module.assert_exact_rerun(result, rerun)
    changed_coordinate = copy.deepcopy(rerun)
    changed_coordinate["paths"]["noise_649001_original"]["noise_coordinates"][0][
        "case_index"
    ] = 999
    with pytest.raises(RuntimeError, match="exact rerun"):
        module.assert_exact_rerun(result, changed_coordinate)
    changed_prefix = copy.deepcopy(result)
    changed_prefix["paths"]["control_original"]["prefix_outputs"][0][0] ^= 1
    with pytest.raises(RuntimeError, match="does not bit-match"):
        module.assert_sigma_zero_legacy_equivalence(changed_prefix, original, swapped)


def _path(outputs: list[int], valid: list[bool]) -> dict:
    return {
        "outputs": outputs,
        "valid": valid,
        "invalid_completed_cases": valid.count(False),
    }


def test_strict_oracle_gate_per_path_validity_interval_and_futility() -> None:
    module = load_runner()
    expected = [0] * 200
    passing = _path([0] * 181 + [1] * 19, [True] * 200)
    excluded = _path([0] * 200, [False] * 5 + [True] * 195)
    summary = module.oracle_summary({"passing": passing}, expected)
    assert summary["valid_paths"] == ["passing"]
    assert summary["validity_gate_passed"]
    assert summary["oracle_correct"] == 181
    assert summary["passed_strict_gate"]
    assert summary["exact_95_percent_binomial_interval"][0] < 181 / 200
    invalid_path = module.oracle_summary({"passing": passing, "excluded": excluded}, expected)
    assert invalid_path["oracle_correct"] == 181
    assert not invalid_path["validity_gate_passed"]
    assert not invalid_path["passed_strict_gate"]
    failing = module.oracle_summary(
        {"only": _path([0] * 180 + [1] * 20, [True] * 200)}, expected
    )
    assert not failing["passed_strict_gate"]
    assert module.futility_reason(19, {"only": 4}) is None
    assert module.futility_reason(20, {"only": 4}) == "twentieth_oracle_failure"
    assert module.futility_reason(0, {"only": 5}) == "fifth_invalid_case_in_a_path"


def test_scientific_and_resource_hashes_exclude_only_resolved_values() -> None:
    module = load_runner()
    config = load_config()
    scientific = module.scientific_config_sha256(config)
    resource = module.resource_protocol_sha256(config)
    changed = copy.deepcopy(config)
    changed["resource_protocol"]["resolved"] = {
        "calibration_status": "complete",
        "measured_wall_seconds": 12.3,
    }
    assert module.scientific_config_sha256(changed) == scientific
    assert module.resource_protocol_sha256(changed) == resource
    changed = copy.deepcopy(config)
    changed["noise"]["sigma"] = 0.3
    assert module.scientific_config_sha256(changed) != scientific
    assert module.resource_protocol_sha256(changed) != resource
    changed = copy.deepcopy(config)
    changed["source_checkpoint_sha256"] = "0" * 64
    assert module.resource_protocol_sha256(changed) != resource
    changed = copy.deepcopy(config)
    changed["resource_protocol"]["formal_soft_multiplier_over_measured_projection"] = 1.6
    assert module.scientific_config_sha256(changed) != scientific
    assert module.resource_protocol_sha256(changed) != resource


def test_frozen_config_rejects_unknown_resource_and_changed_planning_literals() -> None:
    module = load_runner()
    config = load_config()
    module.validate_frozen_config(config)

    unknown = copy.deepcopy(config)
    unknown["resource_protocol"]["unregistered_workload_knob"] = 1
    with pytest.raises(ValueError, match="canonical configuration changed"):
        module.validate_frozen_config(unknown)

    changed_basis = copy.deepcopy(config)
    changed_basis["resource_protocol"]["planning_runaway_basis"][
        "candidate005_isolated_wall_seconds"
    ] = 155.0
    with pytest.raises(ValueError, match="projection changed"):
        module.validate_frozen_config(changed_basis)

    resolved_only = copy.deepcopy(config)
    resolved_only["resource_protocol"]["resolved"] = {
        "calibration_status": "complete",
        "arbitrary_measured_value": 12.0,
    }
    module.validate_frozen_config(resolved_only)


def test_projection_uses_complete_candidate_total_and_separate_process_cpu() -> None:
    module = load_runner()
    config = load_config()
    replicates = [
        {
            "total_wall_seconds": wall,
            "total_process_cpu_seconds": cpu,
            "portfolio_wall_seconds": portfolio,
            "portfolio_process_cpu_seconds": portfolio_cpu,
        }
        for wall, cpu, portfolio, portfolio_cpu in (
            (200.0, 700.0, 40.0, 150.0),
            (210.0, 730.0, 42.0, 160.0),
            (205.0, 720.0, 41.0, 155.0),
        )
    ]
    legacy = {
        "passed": True,
        "prefix_residue_bitmatch_passed": True,
        "cases_per_orientation": 110,
        "legacy_output_sha256": "d" * 64,
        "observed_control_output_sha256": "d" * 64,
        "wall_seconds": 11.0,
        "process_cpu_seconds": 40.0,
    }
    resolved = module.resolve_resource_projection(
        config,
        replicates,
        {"wall_seconds": 2.0, "process_cpu_seconds": 7.0},
        legacy,
    )
    assert resolved["candidate_total_wall_upper_seconds"] == pytest.approx(257.0)
    assert resolved["candidate_total_process_cpu_upper_seconds_diagnostic"] == pytest.approx(881.0)
    assert resolved["candidate_wall_gate_passed"]
    assert resolved["formal_path_case_equivalents"] == 3600
    assert resolved["formal_vectorized_path_case_equivalents"] == 3200
    assert resolved["formal_legacy_path_case_equivalents"] == 400
    assert resolved["projected_formal_wall_seconds"] == pytest.approx(
        42.0 * (200 * 2 / 110) + 11.0 * (200 / 110) + 2.0
    )
    assert resolved["projected_formal_process_cpu_seconds"] == pytest.approx(
        160.0 * (200 * 2 / 110) + 40.0 * (200 / 110) + 7.0
    )
    assert resolved["formal_soft_wall_seconds"] == pytest.approx(
        resolved["projected_formal_wall_seconds"] * 1.5
    )
    assert resolved["formal_hard_wall_seconds"] == pytest.approx(
        resolved["projected_formal_wall_seconds"] * 2.0
    )
    assert (
        config["resource_protocol"]["initial_calibration_hard_ceiling_process_cpu_seconds"]
        > config["resource_protocol"]["initial_calibration_hard_ceiling_wall_seconds"]
    )
    basis = config["resource_protocol"]["planning_runaway_basis"]
    assert basis["raw_cross_environment_projection_wall_seconds"] == pytest.approx(
        156.0 * 4 + 131.313 * (8 * 110 / 100 * 4 + 2 * 110 / 100)
    )
    assert 5760 * 2.5 == config["resource_protocol"][
        "initial_calibration_hard_ceiling_wall_seconds"
    ]
    assert 14400 * 4 == config["resource_protocol"][
        "initial_calibration_hard_ceiling_process_cpu_seconds"
    ]
    assert basis["eligible_for_candidate_runtime_verdict"] is False


def test_runaway_guard_charges_reused_wall_and_process_cpu_and_is_wired_in_stage() -> None:
    module = load_runner()
    process = module.psutil.Process()
    now = module.time.perf_counter()
    cpu_now = module._process_cpu(process)
    with pytest.raises(TimeoutError, match="wall ceiling charged wall"):
        module._check_calibration_runaway(
            stage="charged wall",
            run_started=now,
            run_cpu_started=cpu_now,
            process=process,
            reused_wall_charge=6.0,
            reused_cpu_charge=0.0,
            wall_ceiling=5.0,
            cpu_ceiling=math.inf,
        )
    with pytest.raises(TimeoutError, match="process-CPU ceiling charged cpu"):
        module._check_calibration_runaway(
            stage="charged cpu",
            run_started=now,
            run_cpu_started=cpu_now,
            process=process,
            reused_wall_charge=0.0,
            reused_cpu_charge=6.0,
            wall_ceiling=math.inf,
            cpu_ceiling=5.0,
        )
    source = inspect.getsource(module.run_calibration)
    assert "decoder_bundle, runaway_guard" in source
    assert "replicates[0], runaway_guard" in source
    assert "_measure_bookkeeping(config, output, runaway_guard)" in source
    assert 'runaway_guard("after bookkeeping")' in source


def test_bookkeeping_projection_is_target_free_schema_equivalent_and_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_runner()
    config = load_config()
    payload = module._target_free_formal_projection_payload(config)
    assert len(payload["cases"]) == 200
    assert len(payload["paths"]) == 200 * 8
    assert len(payload["paths"][0]["prefix_outputs"]) == 256
    assert len(payload["paths"][0]["prefix_validity"]) == 256
    assert payload["contains_targets_or_labels"] is False
    assert "expected" not in module.canonical_json(payload)

    monkeypatch.setattr(module, "_ensure_scratch_output", lambda output: output.resolve())
    runaway_stages: list[str] = []
    measured = module._measure_bookkeeping(config, tmp_path, runaway_stages.append)
    artifact = tmp_path / "formal-bookkeeping-projection.json"
    assert artifact.is_file()
    assert module.sha256(artifact) == measured["artifact_sha256"]
    assert measured["artifact_bytes"] == artifact.stat().st_size
    assert measured["cases"] == 200
    assert measured["paths"] == 8
    assert measured["prefixes_per_path_case"] == 256
    assert measured["contains_targets_or_labels"] is False
    assert runaway_stages == [
        "before target-free formal bookkeeping projection",
        "after target-free formal bookkeeping projection",
    ]


def test_calibration_replicate_executes_all_1100_candidate_and_10_100_portfolio_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_runner()
    config = load_config()
    specs = module.path_specs(config)
    candidate_batches = [
        {
            "tier_id": tier,
            "records": [
                {"a": str(index + 1), "b": str(index + 2), "p": "257"}
                for index in range(100)
            ],
        }
        for tier in range(11)
    ]
    portfolio_batches = [
        {
            "batch_id": name,
            "dataset_sha256": str(size) * 64,
            "records": [{}] * size,
            "cases": {
                "a": [300] * size,
                "b": [500] * size,
                "p": [257] * size,
                "case_index": list(range(offset, offset + size)),
            },
        }
        for name, size, offset in (("tier0_width256_band", 10, 0), ("tier7", 100, 10))
    ]
    candidate_call_sizes: list[int] = []
    preprocess_counts = {"a": 0, "b": 0, "p": 0}
    portfolio_call_sizes: list[int] = []
    reduced_first_rows: list[tuple[int, int]] = []
    decode_calls: list[tuple[str, int, bool]] = []
    runaway_stages: list[str] = []

    class FakeCandidate:
        def preprocess_a(self, value):
            preprocess_counts["a"] += 1
            return value

        def preprocess_b(self, value):
            preprocess_counts["b"] += 1
            return value

        def preprocess_p(self, value):
            preprocess_counts["p"] += 1
            return value

        def predict_digits_batch(self, rows):
            candidate_call_sizes.append(len(rows))
            return [[0] for _ in rows]

    def fake_rollout(_source, cases, _specs, *_args, **_kwargs):
        size = len(cases["p"])
        portfolio_call_sizes.append(size)
        reduced_first_rows.append((cases["a"][0], cases["b"][0]))
        paths = {
            spec.path_id: {
                "orientation": spec.orientation,
                "noise_seed": spec.noise_seed,
                "outputs": [0] * size,
                "prefix_outputs": [[0] * 256 for _ in range(size)],
                "valid": [True] * size,
                "prefix_validity": [[True] * 256 for _ in range(size)],
                "invalid_completed_cases": 0,
                "noise_coordinates": [{"derived_generator_seed": spec.noise_seed}],
                "timing": {},
            }
            for spec in specs
        }
        return {
            "paths": paths,
            "coordinate_log": (
                [{"multiplier_bit": 255, "injection_index": 0}]
                + [{}] * (256 * 17 - 2)
                + [{"multiplier_bit": 0, "injection_index": 16}]
            ),
            "state_dict_sha256_before": "same",
            "state_dict_sha256_after": "same",
        }

    monkeypatch.setattr(module, "vectorized_rollout", fake_rollout)
    real_decode, malformed_output = module.load_verified_official_decoder(config)

    def recording_decode(digits, *, base, prime, is_tier_zero=False):
        decode_calls.append((base, prime, is_tier_zero))
        if len(decode_calls) == 1:
            raise malformed_output("intentional malformed-output timing check")
        return real_decode(digits, base=base, prime=prime, is_tier_zero=is_tier_zero)

    result = module.execute_calibration_replicate(
        config,
        nn.Linear(1, 1),
        FakeCandidate(),
        {"candidate_batches": candidate_batches, "portfolio_batches": portfolio_batches},
        (recording_decode, malformed_output),
        runaway_stages.append,
    )
    assert candidate_call_sizes == [100] * 11
    assert preprocess_counts == {"a": 1100, "b": 1100, "p": 1100}
    assert len(decode_calls) == 1100
    assert all(call[0] == "p" for call in decode_calls)
    assert [call[2] for call in decode_calls] == [True] * 100 + [False] * 1000
    assert portfolio_call_sizes == [10, 100]
    assert reduced_first_rows == [(43, 243), (43, 243)]
    for tier_id in range(11):
        assert f"before candidate tier {tier_id} batch" in runaway_stages
        assert f"after candidate tier {tier_id} batch" in runaway_stages
    for batch_id in ("tier0_width256_band", "tier7"):
        assert f"before portfolio batch {batch_id}" in runaway_stages
        assert f"after portfolio batch {batch_id}" in runaway_stages
    assert result["candidate_output_sha256"]
    assert result["portfolio_output_sha256"]
    assert len(result["control_evidence"]) == 2
    material = {"environment_sha256": "a" * 64, "source_hashes": {"runner": "b" * 64}}
    cache_key = module.sha256_text(module.canonical_json(material))
    partial_path = tmp_path / "partial.json"
    module._write_partial_checkpoint(partial_path, cache_key, material, result, [result])
    partial = json.loads(partial_path.read_text())
    cached = module.validate_partial_cache(partial, cache_key, material, 3)
    assert cached == [result]

    stale_key = copy.deepcopy(partial)
    stale_key["cache_key"] = "0" * 64
    stale_key["replicate_chain_sha256"] = module._replicate_chain_sha256(
        stale_key["cache_key"], stale_key["warmup"], stale_key["replicates"]
    )
    stale_key["content_sha256"] = module._partial_content_sha256(stale_key)
    with pytest.raises(RuntimeError, match="cache key mismatch"):
        module.validate_partial_cache(stale_key, cache_key, material, 3)

    other_material = {"environment_sha256": "c" * 64, "source_hashes": {"runner": "b" * 64}}
    other_key = module.sha256_text(module.canonical_json(other_material))
    with pytest.raises(RuntimeError, match="cache material mismatch"):
        module.validate_partial_cache(partial, other_key, other_material, 3)

    incomplete = copy.deepcopy(partial)
    incomplete["replicates"][0].pop("portfolio_output_sha256")
    incomplete["replicate_chain_sha256"] = module._replicate_chain_sha256(
        cache_key, incomplete["warmup"], incomplete["replicates"]
    )
    incomplete["content_sha256"] = module._partial_content_sha256(incomplete)
    with pytest.raises(RuntimeError, match="replicate is incomplete"):
        module.validate_partial_cache(incomplete, cache_key, material, 3)

    for invalid_timing in (-1.0, math.nan, math.inf):
        invalid = copy.deepcopy(partial)
        invalid["replicates"][0]["total_wall_seconds"] = invalid_timing
        invalid["replicate_chain_sha256"] = module._replicate_chain_sha256(
            cache_key, invalid["warmup"], invalid["replicates"]
        )
        invalid["content_sha256"] = module._partial_content_sha256(invalid)
        with pytest.raises(RuntimeError, match="finite and nonnegative"):
            module.validate_partial_cache(invalid, cache_key, material, 3)

    invalid_hash = copy.deepcopy(partial)
    invalid_hash["replicates"][0]["candidate_output_sha256"] = "not-a-hash"
    invalid_hash["replicate_chain_sha256"] = module._replicate_chain_sha256(
        cache_key, invalid_hash["warmup"], invalid_hash["replicates"]
    )
    invalid_hash["content_sha256"] = module._partial_content_sha256(invalid_hash)
    with pytest.raises(RuntimeError, match="not a SHA-256"):
        module.validate_partial_cache(invalid_hash, cache_key, material, 3)

    missing_prefix_residue = copy.deepcopy(partial)
    missing_prefix_residue["replicates"][0]["control_evidence"][0][
        "control_original"
    ]["prefix_outputs"][0].pop()
    missing_prefix_residue["replicate_chain_sha256"] = module._replicate_chain_sha256(
        cache_key,
        missing_prefix_residue["warmup"],
        missing_prefix_residue["replicates"],
    )
    missing_prefix_residue["content_sha256"] = module._partial_content_sha256(
        missing_prefix_residue
    )
    with pytest.raises(RuntimeError, match="control evidence changed shape"):
        module.validate_partial_cache(missing_prefix_residue, cache_key, material, 3)

    broken_integrity = copy.deepcopy(partial)
    broken_integrity["replicates"][0]["portfolio_wall_seconds"] += 1.0
    with pytest.raises(RuntimeError, match="content integrity mismatch"):
        module.validate_partial_cache(broken_integrity, cache_key, material, 3)

    def fake_legacy(_source, cases, _orientation, width, *_args):
        return {
            "outputs": [0] * len(cases["p"]),
            "prefix_outputs": [[0] * width for _ in cases["p"]],
            "prefix_validity": [[True] * width for _ in cases["p"]],
        }

    monkeypatch.setattr(module, "legacy_rollout", fake_legacy)
    parity = module.verify_actual_sigma_zero_controls(
        config,
        nn.Linear(1, 1),
        {"portfolio_batches": portfolio_batches},
        result,
    )
    assert parity["passed"]
    assert parity["prefix_residue_bitmatch_passed"]
    assert parity["cases_per_orientation"] == 110
    assert parity["legacy_output_sha256"] == parity["observed_control_output_sha256"]
    module._write_partial_checkpoint(
        partial_path, cache_key, material, result, [result], parity
    )
    assert module.validate_partial_cache(
        json.loads(partial_path.read_text()), cache_key, material, 3
    ) == [result]


def _passing_formal_inputs(module, config):
    artifact_sha256 = "f" * 64
    current = {
        "source_checkpoint_sha256": "source",
        "candidate_artifact_sha256": "candidate",
        "scientific_config_sha256": "science",
        "resource_protocol_sha256": "resource",
        "runner_sha256": "runner",
        "environment_sha256": "environment",
        "preregistration_sha256": "preregistration",
        "source_note_sha256": "source-note",
        "seed_retirement_manifest_sha256": "seed-manifest",
    }
    calibration = {
        "status": "complete",
        "source": dict(current),
        "workload": {
            "candidate_tier_batch_sizes": [100] * 11,
            "candidate_total_rows": 1100,
            "portfolio_batch_sizes": [10, 100],
            "portfolio_total_rows": 110,
        },
        "replicates": [{}, {}, {}],
        "resolved_resource": {
            "candidate_wall_gate_passed": True,
            "candidate_total_wall_upper_seconds": 250.0,
        },
        "legacy_sigma_zero_control_verification": {
            "passed": True,
            "prefix_residue_bitmatch_passed": True,
            "cases_per_orientation": 110,
            "legacy_output_sha256": "d" * 64,
            "observed_control_output_sha256": "d" * 64,
        },
        "weights": {
            "unchanged": True,
            "state_dict_sha256_before": "a" * 64,
            "state_dict_sha256_after": "a" * 64,
        },
    }
    config = copy.deepcopy(config)
    config["resource_protocol"]["resolved"] = {
        "calibration_status": "complete",
        "formal_authorization_status": "authorized",
        "calibration_output_sha256": artifact_sha256,
        "formal_collision_audit_status": "passed",
    }
    return config, calibration, current, artifact_sha256


def test_formal_refuses_pending_missing_runtime_hash_dirty_and_authorization() -> None:
    module = load_runner()
    original = load_config()
    config, calibration, current, artifact_sha256 = _passing_formal_inputs(module, original)
    authorization = config["formal"]["required_authorization"]
    assert module.formal_refusal_reasons(
        config, calibration, current, False, authorization, artifact_sha256
    ) == []

    pending = copy.deepcopy(config)
    pending["resource_protocol"]["resolved"]["calibration_status"] = "pending"
    assert "calibration_pending" in module.formal_refusal_reasons(
        pending, calibration, current, False, authorization, artifact_sha256
    )
    assert "calibration_missing" in module.formal_refusal_reasons(
        config, None, current, False, authorization, artifact_sha256
    )
    runtime = copy.deepcopy(calibration)
    runtime["resolved_resource"]["candidate_total_wall_upper_seconds"] = 285.01
    assert "candidate_runtime_gate_miss" in module.formal_refusal_reasons(
        config, runtime, current, False, authorization, artifact_sha256
    )
    wrong_hash = dict(current, runner_sha256="changed")
    assert "hash_mismatch:runner_sha256" in module.formal_refusal_reasons(
        config, calibration, wrong_hash, False, authorization, artifact_sha256
    )
    for key in (
        "environment_sha256",
        "preregistration_sha256",
        "source_note_sha256",
    ):
        mismatched = dict(current)
        mismatched[key] = "changed"
        assert f"hash_mismatch:{key}" in module.formal_refusal_reasons(
            config,
            calibration,
            mismatched,
            False,
            authorization,
            artifact_sha256,
        )
    missing_environment = dict(current)
    missing_environment.pop("environment_sha256")
    assert "current_context_missing:environment_sha256" in module.formal_refusal_reasons(
        config,
        calibration,
        missing_environment,
        False,
        authorization,
        artifact_sha256,
    )
    assert "calibration_artifact_hash_mismatch" in module.formal_refusal_reasons(
        config, calibration, current, False, authorization, "e" * 64
    )
    no_artifact = copy.deepcopy(config)
    no_artifact["resource_protocol"]["resolved"].pop("calibration_output_sha256")
    assert "calibration_artifact_hash_missing" in module.formal_refusal_reasons(
        no_artifact, calibration, current, False, authorization, None
    )
    no_collision_audit = copy.deepcopy(config)
    no_collision_audit["resource_protocol"]["resolved"][
        "formal_collision_audit_status"
    ] = "pending"
    assert "formal_collision_audit_missing" in module.formal_refusal_reasons(
        no_collision_audit, calibration, current, False, authorization, artifact_sha256
    )
    incomplete_replicates = copy.deepcopy(calibration)
    incomplete_replicates["replicates"] = [{}, {}]
    assert "calibration_replicates_incomplete" in module.formal_refusal_reasons(
        config, incomplete_replicates, current, False, authorization, artifact_sha256
    )
    changed_weights = copy.deepcopy(calibration)
    changed_weights["weights"]["state_dict_sha256_after"] = "b" * 64
    assert "source_weights_not_proven_unchanged" in module.formal_refusal_reasons(
        config, changed_weights, current, False, authorization, artifact_sha256
    )
    assert "repository_dirty" in module.formal_refusal_reasons(
        config, calibration, current, True, authorization, artifact_sha256
    )
    assert "later_judge_authorization_missing" in module.formal_refusal_reasons(
        config, calibration, current, False, None, artifact_sha256
    )
    formal_source = inspect.getsource(module._formal_refusal_from_disk)
    assert formal_source.index("configure_frozen_runtime") < formal_source.index(
        "require_formal_preconditions"
    )


def test_calibration_formal_roots_schema_and_import_are_side_effect_free() -> None:
    module = load_runner()
    config = load_config()
    assert config["data"]["calibration_root_seed"] == 20261150
    assert config["data"]["formal_root_seed"] == 20261149
    assert config["data"]["retired_root_seed_denylist"] == [20261148]
    assert config["data"]["calibration_root_seed"] != config["data"]["formal_root_seed"]
    for protected_root in (20261148, 20261149, 20261150):
        with pytest.raises(RuntimeError, match="active or retired experiment root"):
            module._build_label_free_official_inputs(config, protected_root)
    retired_as_active = copy.deepcopy(config)
    retired_as_active["data"]["calibration_root_seed"] = 20261148
    with pytest.raises(RuntimeError, match="retired or collides"):
        module.build_calibration_inputs(retired_as_active)
    formal_as_active = copy.deepcopy(config)
    formal_as_active["data"]["calibration_root_seed"] = 20261149
    with pytest.raises(RuntimeError, match="retired or collides"):
        module.build_calibration_inputs(formal_as_active)
    schema = module.formal_schema()
    assert schema["case"] == [
        "case_index",
        "case_id",
        "identity_id",
        "a",
        "b",
        "p",
        "expected",
    ]
    assert {"noise_coordinate", "prefix_outputs", "prefix_validity", "timing"} <= set(
        schema["path"]
    )
    assert "seed_retirement_manifest_sha256" in schema["provenance"]
    source = RUNNER.read_text()
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "formal identities" in module.__doc__
