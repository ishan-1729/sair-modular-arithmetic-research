"""Calibration-only scaffold for EXP-649 stochastic trajectory breadth.

Calibration consumes generated input-only training identities. Formal mode is
fail-closed until a committed calibration and a later Judge authorization exist;
this scaffold deliberately does not construct formal identities or evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import importlib.util
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

import psutil
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SRC = ROOT / "sources/github/modular-arithmetic-challenge/src"
FROZEN_SCIENTIFIC_CONFIG_SHA256 = "ea70a40c01e1c407be05ce7d5e7740862e670a0dd36715635614c521cae3415a"
FROZEN_RESOURCE_PROTOCOL_SHA256 = "8d13c0f5d6a859a36d9360ad4209b492bc10dae42a6e6be80c0d9bdc5b3fb7cd"
for import_root in (ROOT, UPSTREAM_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _without_resolved(config: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(config)
    frozen.setdefault("resource_protocol", {}).pop("resolved", None)
    return frozen


def scientific_config_sha256(config: dict[str, Any]) -> str:
    """Hash every frozen field; only measured/derived ``resolved`` may change."""

    return sha256_text(canonical_json(_without_resolved(config)))


def resource_protocol_sha256(config: dict[str, Any]) -> str:
    """Hash every field that can change the measured or projected workload."""

    frozen = _without_resolved(config)
    workload_keys = (
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_model",
        "source_model_sha256",
        "candidate_artifact",
        "candidate_artifact_sha256",
        "candidate_model",
        "candidate_model_sha256",
        "candidate_manifest",
        "candidate_manifest_sha256",
        "candidate_output_base",
        "seed_retirement_manifest",
        "seed_retirement_manifest_sha256",
        "official_evaluator",
        "model",
        "paths",
        "noise",
        "data",
        "candidate_geometry",
        "batching",
        "gates",
        "futility",
        "formal",
        "resource_protocol",
    )
    fields = {
        key: frozen[key]
        for key in workload_keys
    }
    return sha256_text(canonical_json(fields))


EXPECTED_SEED_RETIREMENT_MANIFEST: dict[str, Any] = {
    "schema_version": "exp649-seed-retirement-v1",
    "incident": {
        "incident_id": "exp649-calibration-root-label-contamination-2026-07-20",
        "schema": "calibration-root-label-generation",
        "status": "contained-root-retired",
    },
    "seed_roles": {
        "retired_calibration_root": 20261148,
        "replacement_calibration_input_only_root": 20261150,
        "formal_unopened_root": 20261149,
        "test_only_parity_root": 649900001,
    },
    "contamination": {
        "scope_cases": 1100,
        "expected_values": "computed-in-memory",
        "artifact_written": False,
        "calibration_mode_executed": False,
        "formal_mode_executed": False,
    },
    "upstream": {
        "commit_sha": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
        "generator_path": "sources/github/modular-arithmetic-challenge/src/modchallenge/testgen/generator.py",
        "generator_sha256": "77f4142fe097ef66f2bd4452547b465ef3f75646191ed8258c508f5e3caa85c0",
    },
    "human_receipt_path": "knowledge/autoresearch/analysis/exp-649-calibration-seed-retirement.md",
    "containment_requirements": [
        "never_generate_inputs_or_labels_with_retired_root_20261148",
        "replacement_root_20261150_is_input_only_and_requires_clean_committed_calibration",
        "formal_root_20261149_remains_unopened_until_collision_audit_and_judge_authorization",
        "official_reference_parity_tests_use_only_test_root_649900001",
        "no_generated_identity_or_label_is_stored_in_this_manifest",
    ],
}


def validate_seed_retirement_manifest_payload(payload: dict[str, Any]) -> None:
    if payload != EXPECTED_SEED_RETIREMENT_MANIFEST:
        raise RuntimeError("EXP-649 seed-retirement manifest schema or roles changed")
    rendered = canonical_json(payload)
    if "config_sha256" in rendered or "runner_sha256" in rendered:
        raise RuntimeError("seed-retirement manifest contains a circular config/runner hash")


def verify_seed_retirement_manifest(config: dict[str, Any]) -> str:
    expected_path = "experiments/exp_649_stochastic_trajectory_breadth/seed_retirement_manifest.json"
    expected_sha256 = "b00eaca8a12a397851103d570ca5b00cb94f925d4f05d96a827f77eb752b712f"
    if config.get("seed_retirement_manifest") != expected_path:
        raise RuntimeError("EXP-649 seed-retirement manifest path changed")
    if config.get("seed_retirement_manifest_sha256") != expected_sha256:
        raise RuntimeError("EXP-649 seed-retirement manifest pin changed")
    path = (ROOT / expected_path).resolve()
    observed = sha256(path)
    if observed != expected_sha256:
        raise RuntimeError("EXP-649 seed-retirement manifest hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_seed_retirement_manifest_payload(payload)
    receipt = ROOT / payload["human_receipt_path"]
    if not receipt.is_file():
        raise RuntimeError("EXP-649 seed-retirement human receipt is missing")
    return observed


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class PathSpec(NamedTuple):
    path_id: str
    orientation: str
    noise_seed: int | None
    sigma: float


def path_specs(config: dict[str, Any]) -> list[PathSpec]:
    return [
        PathSpec(
            path_id=str(item["id"]),
            orientation=str(item["orientation"]),
            noise_seed=None if item["noise_seed"] is None else int(item["noise_seed"]),
            sigma=float(item["sigma"]),
        )
        for item in config["paths"]["order"]
    ]


def validate_frozen_config(config: dict[str, Any]) -> None:
    source_pins = {
        "source_checkpoint": ".scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors",
        "source_checkpoint_sha256": "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd",
        "source_model": "experiments/exp_220_width_independent_cellular/model.py",
        "source_model_sha256": "9b271fd13084011e1631125040e4080ea4397a0356dab1f1ce81bfee3f799edc",
        "candidate_artifact": "submission/weights.safetensors",
        "candidate_artifact_sha256": "2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca",
        "candidate_model": "submission/model.py",
        "candidate_model_sha256": "e3d4aa9ff3e22dde516d261b099826622fa37d0688fae47c07ff95fe876e9477",
        "candidate_manifest": "submission/manifest.json",
        "candidate_manifest_sha256": "ee49f91b8a1a4aa2cde418ebec59ea1890d93fda6dc72338989c0aa3308e7c83",
        "candidate_output_base": "p",
        "seed_retirement_manifest": "experiments/exp_649_stochastic_trajectory_breadth/seed_retirement_manifest.json",
        "seed_retirement_manifest_sha256": "b00eaca8a12a397851103d570ca5b00cb94f925d4f05d96a827f77eb752b712f",
    }
    if any(config.get(key) != value for key, value in source_pins.items()):
        raise ValueError("EXP-649's frozen model/evaluator source pins changed")
    verify_seed_retirement_manifest(config)
    specs = path_specs(config)
    expected_specs = [
        PathSpec("control_original", "original", None, 0.0),
        PathSpec("control_swapped", "swapped", None, 0.0),
        PathSpec("noise_649001_original", "original", 649001, 0.2),
        PathSpec("noise_649002_swapped", "swapped", 649002, 0.2),
        PathSpec("noise_649003_original", "original", 649003, 0.2),
        PathSpec("noise_649004_swapped", "swapped", 649004, 0.2),
        PathSpec("noise_649005_original", "original", 649005, 0.2),
        PathSpec("noise_649006_swapped", "swapped", 649006, 0.2),
    ]
    if (
        specs != expected_specs
        or int(config["paths"]["stochastic_k"]) != 6
        or int(config["paths"]["total"]) != 8
        or set(config["paths"]) != {"stochastic_k", "total", "order"}
    ):
        raise ValueError("EXP-649 requires two controls and the frozen six-path mapping")
    model = config["model"]
    expected_model = {
        "width": 256,
        "hidden": 92,
        "cycles": 2,
        "dilation_schedule": [1, 2, 4, 8, 16, 32, 64, 128, 128, 64, 32, 16, 8, 4, 2, 1],
        "recurrent_visits_per_multiplier_bit": 16,
        "state_between_multiplier_bits": "hard_decoded_integer_residue_only",
        "continuous_hidden_persists_between_multiplier_bits": False,
        "weights_change": False,
    }
    if model != expected_model:
        raise ValueError("the frozen model geometry is width 256 with 16 visits")
    noise = config["noise"]
    required_noise: dict[str, Any] = {
        "distribution": "iid_cpu_float32_gaussian",
        "sigma": 0.2,
        "clipping": "none",
        "rescaling": "none",
        "injection_sites_per_multiplier_bit": 17,
        "injection_order": "stem_then_before_each_recurrent_visit",
        "generator_scope": "one_cpu_torch_generator_per_stochastic_path",
        "root_seed_domain_fields": ["phase", "dataset_sha256", "path_noise_seed"],
        "logical_coordinate_fields": ["case_index", "multiplier_bit", "injection_index"],
        "coordinate_traversal": "multiplier_bit_msb_to_lsb_then_injection_0_to_16_then_case_channel_position",
        "reset_points": [
            "before_warmup",
            "after_warmup",
            "before_each_replicate",
            "before_exact_rerun",
        ],
        "exact_equations": {
            "stem": "h = stem(x) + epsilon_0",
            "recurrent": "h = recurrent_update(h + epsilon_j, x, dilation_j)",
        },
    }
    if noise != required_noise:
        raise ValueError("EXP-649's stochastic-noise protocol changed")
    data = config["data"]
    required_data = {
        "width": 256,
        "calibration_root_seed": 20261150,
        "formal_root_seed": 20261149,
        "retired_root_seed_denylist": [20261148],
        "master_seed_encoding": "unsigned_big_endian_32_bytes",
        "calibration_identity_role": "generated_training_identities_only",
        "formal_identity_role": "fresh_held_out_identities_pending_collision_audit",
        "official_private_generator_input_clone": "exact_hmac_rng_order_without_expected_values",
        "primes_per_tier": 5,
        "edge_cases_per_tier": 4,
        "formal_cases": 200,
        "maximum_formal_cases": 200,
        "no_calibration_labels_or_oracle": True,
        "require_identity_and_case_disjointness": True,
    }
    if data != required_data:
        raise ValueError("EXP-649's frozen calibration/formal data contract changed")
    official = config.get("official_evaluator", {})
    expected_official = {
        "repository": "sources/github/modular-arithmetic-challenge",
        "commit_sha": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
        "source_files": {
            "config": {
                "path": "sources/github/modular-arithmetic-challenge/src/modchallenge/config.py",
                "sha256": "1c6dae879d0b3c183fce5f93bdbe8e06faaa17b74e49ab696e4165518adcdca8",
            },
            "generator": {
                "path": "sources/github/modular-arithmetic-challenge/src/modchallenge/testgen/generator.py",
                "sha256": "77f4142fe097ef66f2bd4452547b465ef3f75646191ed8258c508f5e3caa85c0",
            },
            "primes": {
                "path": "sources/github/modular-arithmetic-challenge/src/modchallenge/testgen/primes.py",
                "sha256": "5d7f494a1560efbfeb74ed02ddd67a17001bc9ebeb6345e1567bca1bf7dc8bac",
            },
            "pipeline": {
                "path": "sources/github/modular-arithmetic-challenge/src/modchallenge/evaluation/pipeline.py",
                "sha256": "b903f7bed3cfe658e32d58a64b30ddc83adc7b7f1b50be9f9f45dd6a0b3aadc9",
            },
            "decoder": {
                "path": "sources/github/modular-arithmetic-challenge/src/modchallenge/evaluation/decoder.py",
                "sha256": "e785476ef1f2cd148c1a429248fe7eb413a029c1223b106b0ebb87bdfc6880e6",
            },
            "loader": {
                "path": "sources/github/modular-arithmetic-challenge/src/modchallenge/evaluation/loader.py",
                "sha256": "c4d12b876698b9651392b52a84edadbcfd485a5a1d1e79c5b183382d65bc215f",
            },
        },
    }
    if official != expected_official or config.get("candidate_output_base") != "p":
        raise ValueError("the pinned official evaluator provenance changed")
    geometry = config["candidate_geometry"]
    expected_scored_tiers = [
        {"tier_id": 1, "prime_bits_min": 1, "prime_bits_max": 3, "operand_bits": 32},
        {"tier_id": 2, "prime_bits_min": 4, "prime_bits_max": 8, "operand_bits": 48},
        {"tier_id": 3, "prime_bits_min": 9, "prime_bits_max": 16, "operand_bits": 64},
        {"tier_id": 4, "prime_bits_min": 17, "prime_bits_max": 32, "operand_bits": 96},
        {"tier_id": 5, "prime_bits_min": 33, "prime_bits_max": 64, "operand_bits": 128},
        {"tier_id": 6, "prime_bits_min": 65, "prime_bits_max": 128, "operand_bits": 256},
        {"tier_id": 7, "prime_bits_min": 129, "prime_bits_max": 256, "operand_bits": 512},
        {"tier_id": 8, "prime_bits_min": 257, "prime_bits_max": 512, "operand_bits": 1024},
        {"tier_id": 9, "prime_bits_min": 513, "prime_bits_max": 1024, "operand_bits": 2048},
        {"tier_id": 10, "prime_bits_min": 1025, "prime_bits_max": 2048, "operand_bits": 4096},
    ]
    expected_subtiers = [
        [1, 4], [5, 16], [17, 32], [33, 64], [65, 128],
        [129, 256], [257, 512], [513, 1024], [1025, 2048], [2049, 4096],
    ]
    batches = geometry["future_width256_rows"]["batches"]
    if (
        int(geometry["total_rows"]) != 1100
        or int(geometry["rows_per_tier"]) != 100
        or geometry["tier_ids"] != list(range(11))
        or int(geometry["edge_rows_per_scored_tier"]) != 4
        or geometry["scored_tiers"] != expected_scored_tiers
        or geometry["tier0_operand_subtiers"] != expected_subtiers
        or [int(x["count"]) for x in batches] != [10, 100]
    ):
        raise ValueError("candidate projection must contain 1,100 plus 10/100 rows")
    tier0 = batches[0]
    if (int(tier0["operand_bits_min"]), int(tier0["operand_bits_max"])) != (65, 128):
        raise ValueError("Tier-0's width-256 dispatch is the 65--128 operand band")
    if int(geometry["future_width256_rows"]["total"]) != 110:
        raise ValueError("width-256 dispatch must include all 110 rows")
    if batches != [
        {
            "id": "tier0_width256_band",
            "tier_id": 0,
            "count": 10,
            "operand_bits_min": 65,
            "operand_bits_max": 128,
        },
        {"id": "tier7", "tier_id": 7, "count": 100, "operand_bits": 512},
    ]:
        raise ValueError("EXP-649's future width-256 dispatch changed")
    if not bool(config["batching"]["all_eight_paths_vectorized"]):
        raise ValueError("all eight paths must be vectorized")
    if config["batching"] != {
        "candidate_workload": "tier_order_0_to_10_one_exact_100_row_batch_per_tier",
        "portfolio_batch_order": ["tier0_width256_band", "tier7"],
        "portfolio_exact_batch_sizes": [10, 100],
        "path_case_layout": "flatten_path_then_case_for_one_model_call_per_multiplier_bit",
        "case_order": "canonical_input_generation_order_no_reordering",
        "all_eight_paths_vectorized": True,
    }:
        raise ValueError("EXP-649's batching contract changed")
    gates = config["gates"]
    required_gates = {
        "oracle_minimum_correct_cases": 181,
        "oracle_strictly_greater_than_accuracy": 0.90,
        "maximum_invalid_completed_cases_per_path": 4,
        "maximum_invalid_rate_per_path": 0.02,
        "invalid_definition": "any_decoded_prefix_or_final_outside_half_open_interval_zero_p",
        "require_exact_binomial_interval": True,
        "require_exact_rerun": True,
        "require_sigma_zero_legacy_bitmatch": True,
        "require_state_dict_hash_unchanged": True,
        "candidate_wall_upper_formula": "1.20 * max(total_wall_seconds) + 5",
        "candidate_wall_upper_seconds": 285,
        "candidate_cpu_upper_formula_diagnostic_only": "1.20 * max(process_user_plus_system_cpu_seconds) + 5",
    }
    if gates != required_gates:
        raise ValueError("EXP-649's decision gates changed")
    if config["futility"] != {
        "oracle_failures": 20,
        "invalid_completed_cases_in_any_path": 5,
        "no_resampling_replacement_filtering_or_tuning": True,
    }:
        raise ValueError("EXP-649's futility rule changed")
    formal = config["formal"]
    schema = formal_schema()
    expected_records = {
        "case": schema["case"],
        "path": schema["path"],
        "provenance": schema["provenance"],
    }
    if (
        not bool(formal["authorization_required"])
        or formal["required_authorization"] != "judge-approved-exp-649-formal"
        or formal["raw_schema_version"] != "exp649-formal-v1"
        or int(formal["complete_exact_reruns"]) != 1
        or int(formal["legacy_control_verification_paths"]) != 2
        or any(bool(formal[key]) for key in (
            "selector_allowed",
            "voting_allowed",
            "margin_or_convergence_selection_allowed",
        ))
        or formal.get("record_fields") != expected_records
    ):
        raise ValueError("EXP-649's formal schema changed")
    resource = config["resource_protocol"]
    required_resource = {
        "torch_threads": 4,
        "torch_interop_threads": 1,
        "deterministic_algorithms": True,
        "warmups": 1,
        "warmup_workload": "complete_candidate_1100_plus_all_eight_width256_paths_batches_10_and_100",
        "timed_replicates": 3,
        "replicate_workload": "complete_same_process_end_to_end_inference",
        "process_cpu_measurement": "psutil_user_plus_system_seconds",
        "portfolio_only_timing_reported_separately": True,
        "no_mismatched_cached_or_cross_environment_verdict_timing": True,
        "cache_completed_calibration_replicates": True,
        "resume_requires_fresh_complete_warmup": True,
        "partial_schema_version": "exp649-calibration-partial-v3",
        "partial_checkpoint_filename": "calibration-partial.json",
        "partial_cache_key_fields": [
            "environment_sha256",
            "source_hashes",
            "scientific_config_sha256",
            "resource_protocol_sha256",
            "dataset_sha256",
            "preregistration_sha256",
            "source_note_sha256",
        ],
        "initial_calibration_ceiling_role": "generous_planning_runaway_protection_only",
        "fidelity_may_be_reduced_to_meet_ceiling": False,
        "in_stage_runaway_guard": {
            "charged_time": "active_session_plus_reused_complete_replicates",
            "candidate_tier_batches": "before_and_after_each",
            "portfolio_batches": "before_and_after_each",
            "vectorized_multiplier_bit_interval": 16,
            "legacy_multiplier_bit_interval": 16,
            "bookkeeping": "before_and_after",
        },
        "formal_projection_formula": "measured_vectorized_8x110_scaled_to_8x200x2_plus_measured_legacy_2x110_scaled_to_2x200_plus_measured_target_free_schema_200x8_serialization_hash_and_scratch_write",
        "formal_soft_multiplier_over_measured_projection": 1.5,
        "formal_hard_multiplier_over_measured_projection": 2.0,
        "calibration_output": ".scratch/exp-649-stochastic-trajectory-breadth/calibration.json",
    }
    if any(resource.get(key) != value for key, value in required_resource.items()):
        raise ValueError("EXP-649's resource protocol changed")
    basis = resource["planning_runaway_basis"]
    raw_projection = float(basis["candidate005_isolated_wall_seconds"]) * int(
        basis["candidate_complete_runs_including_warmup"]
    ) + float(basis["exp647_width256_100_case_wall_seconds"]) * (
        int(basis["portfolio_paths"])
        * int(basis["portfolio_rows"])
        / 100
        * int(basis["portfolio_complete_runs_including_warmup"])
        + int(basis["legacy_control_paths"]) * int(basis["legacy_control_rows"]) / 100
    )
    wall_ceiling = float(basis["rounded_planning_projection_wall_seconds"]) * float(
        basis["generous_runaway_multiplier"]
    )
    cpu_ceiling = wall_ceiling * int(basis["process_cpu_thread_multiplier"])
    if not math.isclose(raw_projection, float(basis["raw_cross_environment_projection_wall_seconds"])):
        raise ValueError("planning-only cross-environment projection changed")
    if not math.isclose(wall_ceiling, float(resource["initial_calibration_hard_ceiling_wall_seconds"])):
        raise ValueError("wall runaway ceiling no longer follows its planning derivation")
    if not math.isclose(
        cpu_ceiling, float(resource["initial_calibration_hard_ceiling_process_cpu_seconds"])
    ):
        raise ValueError("process-CPU runaway ceiling no longer follows its planning derivation")
    if bool(basis["eligible_for_candidate_runtime_verdict"]):
        raise ValueError("cross-environment planning observations cannot enter the verdict")
    # This canonical gate makes ``resource_protocol.resolved`` the only
    # mutable subtree. It also rejects unknown fields and any otherwise
    # unchecked planning/workload literal.
    if scientific_config_sha256(config) != FROZEN_SCIENTIFIC_CONFIG_SHA256:
        raise ValueError("EXP-649's frozen canonical configuration changed")
    if resource_protocol_sha256(config) != FROZEN_RESOURCE_PROTOCOL_SHA256:
        raise ValueError("EXP-649's frozen canonical resource contract changed")


def bigint_bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def features(
    state: list[int], multiplicand: list[int], modulus: list[int], bits: list[int], width: int
) -> torch.Tensor:
    batch = len(state)
    bit_plane = torch.tensor(bits, dtype=torch.float32)[:, None].expand(batch, width)
    lsb = torch.zeros((batch, width), dtype=torch.float32)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (
            bigint_bits(state, width),
            bigint_bits(multiplicand, width),
            bigint_bits(modulus, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul_(2.0).sub_(1.0)


def decode_logits(logits: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in logits.ge(0).tolist()
    ]


class CanonicalNoiseStreams:
    """One resettable CPU generator per stochastic path and a frozen traversal."""

    def __init__(
        self,
        specs: list[PathSpec],
        phase: str,
        dataset_sha256: str,
        case_indices: list[int],
        multiplier_width: int,
        injection_sites: int,
    ) -> None:
        self.specs = specs
        self.phase = phase
        self.dataset_sha256 = dataset_sha256
        self.case_indices = tuple(case_indices)
        self.multiplier_width = multiplier_width
        self.injection_sites = injection_sites
        self.generators: dict[str, torch.Generator] = {}
        self.root_seeds: dict[str, int] = {}
        self.coordinate_log: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.generators = {}
        self.root_seeds = {}
        self.coordinate_log = []
        self._next_bit = self.multiplier_width - 1
        self._next_injection = 0
        for spec in self.specs:
            if spec.noise_seed is None:
                continue
            material = canonical_json(
                {
                    "phase": self.phase,
                    "dataset_sha256": self.dataset_sha256,
                    "path_noise_seed": spec.noise_seed,
                }
            )
            seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
            seed &= (1 << 63) - 1
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            self.generators[spec.path_id] = generator
            self.root_seeds[spec.path_id] = seed

    def draw(
        self,
        case_indices: list[int],
        multiplier_bit: int,
        injection_index: int,
        hidden: int,
        width: int,
    ) -> torch.Tensor:
        if tuple(case_indices) != self.case_indices:
            raise RuntimeError("noise case order changed")
        if (multiplier_bit, injection_index) != (self._next_bit, self._next_injection):
            raise RuntimeError("noise coordinate traversal changed")
        self.coordinate_log.append(
            {
                "multiplier_bit": multiplier_bit,
                "injection_index": injection_index,
                "case_indices": list(self.case_indices),
            }
        )
        self._next_injection += 1
        if self._next_injection == self.injection_sites:
            self._next_injection = 0
            self._next_bit -= 1
        rows: list[torch.Tensor] = []
        shape = (len(case_indices), hidden, width)
        for spec in self.specs:
            if spec.noise_seed is None:
                rows.append(torch.zeros(shape, dtype=torch.float32))
            else:
                rows.append(
                    torch.randn(shape, generator=self.generators[spec.path_id], dtype=torch.float32)
                    * spec.sigma
                )
        return torch.stack(rows, dim=0)

    def descriptor(self, spec: PathSpec, case_index: int) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "dataset_sha256": self.dataset_sha256,
            "path_noise_seed": spec.noise_seed,
            "derived_generator_seed": self.root_seeds.get(spec.path_id),
            "case_index": case_index,
            "multiplier_bits": [self.multiplier_width - 1, 0, -1],
            "injection_indices": [0, self.injection_sites - 1],
            "layout": "case-channel-position within each bit/injection draw",
        }


def noisy_transition(
    model: torch.nn.Module,
    encoded: torch.Tensor,
    streams: CanonicalNoiseStreams,
    case_indices: list[int],
    multiplier_bit: int,
    dilations: list[int],
    trace: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    path_count = len(streams.specs)
    cases = len(case_indices)
    hidden = model.stem(encoded)
    noise = streams.draw(case_indices, multiplier_bit, 0, hidden.shape[1], hidden.shape[2])
    noise = noise.reshape(path_count * cases, hidden.shape[1], hidden.shape[2])
    hidden = hidden + noise
    if trace is not None:
        trace(0, hidden, noise)
    for injection_index, dilation in enumerate(dilations, start=1):
        noise = streams.draw(
            case_indices, multiplier_bit, injection_index, hidden.shape[1], hidden.shape[2]
        ).reshape_as(hidden)
        hidden = model.recurrent_update(hidden + noise, encoded, int(dilation))
        if trace is not None:
            trace(injection_index, hidden, noise)
    normalized = F.silu(model.output_norm(hidden))
    return model.local_head(normalized).squeeze(1)


def _oriented_inputs(
    cases: dict[str, list[int]], specs: list[PathSpec]
) -> tuple[list[int], list[int], list[int]]:
    multipliers: list[int] = []
    multiplicands: list[int] = []
    moduli: list[int] = []
    for spec in specs:
        if spec.orientation == "original":
            multipliers.extend(cases["a"])
            multiplicands.extend(cases["b"])
        elif spec.orientation == "swapped":
            multipliers.extend(cases["b"])
            multiplicands.extend(cases["a"])
        else:
            raise ValueError(spec.orientation)
        moduli.extend(cases["p"])
    return multipliers, multiplicands, moduli


@torch.inference_mode()
def vectorized_rollout(
    model: torch.nn.Module,
    cases: dict[str, list[int]],
    specs: list[PathSpec],
    phase: str,
    dataset_sha256: str,
    width: int,
    dilations: list[int],
    trace: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    runaway_guard: Callable[[str], None] | None = None,
    runaway_check_interval: int = 16,
) -> dict[str, Any]:
    count = len(cases["p"])
    if not count or not (len(cases["a"]) == len(cases["b"]) == count):
        raise ValueError("paired cases must be non-empty and equal length")
    case_indices = list(cases.get("case_index", range(count)))
    multipliers, multiplicands, moduli = _oriented_inputs(cases, specs)
    states = [0] * (len(specs) * count)
    prefix_outputs = [[[] for _ in range(count)] for _ in specs]
    prefix_validity = [[[] for _ in range(count)] for _ in specs]
    streams = CanonicalNoiseStreams(
        specs, phase, dataset_sha256, case_indices, width, len(dilations) + 1
    )
    before = state_dict_sha256(model)
    process = psutil.Process()
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    if runaway_check_interval <= 0:
        raise ValueError("runaway check interval must be positive")
    if runaway_guard is not None:
        runaway_guard("before vectorized width-256 rollout")
    for bit_index in range(width - 1, -1, -1):
        if runaway_guard is not None and (
            bit_index == width - 1 or bit_index % runaway_check_interval == 0
        ):
            runaway_guard(f"within vectorized rollout before multiplier bit {bit_index}")
        input_bits = [(value >> bit_index) & 1 for value in multipliers]
        encoded = features(states, multiplicands, moduli, input_bits, width)
        logits = noisy_transition(
            model, encoded, streams, case_indices, bit_index, dilations, trace=trace
        )
        states = decode_logits(logits)
        for path_index in range(len(specs)):
            offset = path_index * count
            for case_offset in range(count):
                value = states[offset + case_offset]
                prefix_outputs[path_index][case_offset].append(value)
                prefix_validity[path_index][case_offset].append(
                    0 <= value < cases["p"][case_offset]
                )
    if runaway_guard is not None:
        runaway_guard("after vectorized width-256 rollout")
    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    cpu_elapsed = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("model state_dict changed during frozen rollout")
    paths: dict[str, Any] = {}
    for path_index, spec in enumerate(specs):
        offset = path_index * count
        predictions = states[offset : offset + count]
        valid = [all(values) and 0 <= predictions[i] < cases["p"][i] for i, values in enumerate(prefix_validity[path_index])]
        paths[spec.path_id] = {
            "orientation": spec.orientation,
            "noise_seed": spec.noise_seed,
            "outputs": predictions,
            "prefix_outputs": prefix_outputs[path_index],
            "valid": valid,
            "prefix_validity": prefix_validity[path_index],
            "invalid_completed_cases": valid.count(False),
            "noise_coordinates": [
                streams.descriptor(spec, case_index) for case_index in case_indices
            ],
            "timing": {
                "vectorized_shared_wall_seconds": elapsed,
                "vectorized_shared_process_cpu_seconds": cpu_elapsed,
            },
        }
    return {
        "paths": paths,
        "timing": {"wall_seconds": elapsed, "process_cpu_seconds": cpu_elapsed},
        "flattened_path_case_batch": len(specs) * count,
        "coordinate_log": streams.coordinate_log,
        "state_dict_sha256_before": before,
        "state_dict_sha256_after": after,
    }


@torch.inference_mode()
def legacy_rollout(
    model: torch.nn.Module,
    cases: dict[str, list[int]],
    orientation: str,
    width: int,
    runaway_guard: Callable[[str], None] | None = None,
    runaway_check_interval: int = 16,
) -> dict[str, Any]:
    multiplier = cases["a"] if orientation == "original" else cases["b"]
    multiplicand = cases["b"] if orientation == "original" else cases["a"]
    state = [0] * len(cases["p"])
    prefix_outputs: list[list[int]] = [[] for _ in state]
    prefix: list[list[bool]] = [[] for _ in state]
    if runaway_check_interval <= 0:
        raise ValueError("runaway check interval must be positive")
    if runaway_guard is not None:
        runaway_guard(f"before legacy {orientation} rollout")
    for bit_index in range(width - 1, -1, -1):
        if runaway_guard is not None and (
            bit_index == width - 1 or bit_index % runaway_check_interval == 0
        ):
            runaway_guard(
                f"within legacy {orientation} rollout before multiplier bit {bit_index}"
            )
        bits = [(value >> bit_index) & 1 for value in multiplier]
        logits, _ = model(features(state, multiplicand, cases["p"], bits, width))
        state = decode_logits(logits)
        for index, value in enumerate(state):
            prefix_outputs[index].append(value)
            prefix[index].append(0 <= value < cases["p"][index])
    if runaway_guard is not None:
        runaway_guard(f"after legacy {orientation} rollout")
    return {
        "outputs": state,
        "prefix_outputs": prefix_outputs,
        "prefix_validity": prefix,
    }


def assert_sigma_zero_legacy_equivalence(
    vectorized: dict[str, Any], original: dict[str, Any], swapped: dict[str, Any]
) -> None:
    for path_id, legacy in (("control_original", original), ("control_swapped", swapped)):
        observed = vectorized["paths"][path_id]
        if (
            observed["outputs"] != legacy["outputs"]
            or observed["prefix_outputs"] != legacy["prefix_outputs"]
            or observed["prefix_validity"] != legacy["prefix_validity"]
        ):
            raise RuntimeError(f"{path_id} does not bit-match untouched legacy rollout")


def assert_exact_rerun(first: dict[str, Any], second: dict[str, Any]) -> None:
    """Require every final/prefix output, validity flag, and coordinate to repeat."""

    if _deterministic_rollout_payload(first) != _deterministic_rollout_payload(second):
        raise RuntimeError("exact rerun did not reproduce every path output")


def exact_binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> list[float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial counts")
    from scipy.stats import beta

    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [lower, upper]


def oracle_summary(
    paths: dict[str, Any], expected: list[int], maximum_invalid: int = 4
) -> dict[str, Any]:
    valid_paths = [
        path_id
        for path_id, record in paths.items()
        if int(record["invalid_completed_cases"]) <= maximum_invalid
    ]
    successes = sum(
        any(
            paths[path_id]["valid"][index]
            and paths[path_id]["outputs"][index] == target
            for path_id in valid_paths
        )
        for index, target in enumerate(expected)
    )
    validity_gate_passed = len(valid_paths) == len(paths)
    return {
        "cases": len(expected),
        "valid_paths": valid_paths,
        "validity_gate_passed": validity_gate_passed,
        "oracle_correct": successes,
        "oracle_accuracy": successes / len(expected),
        "exact_95_percent_binomial_interval": exact_binomial_interval(successes, len(expected)),
        "passed_strict_gate": (
            len(expected) == 200 and successes >= 181 and validity_gate_passed
        ),
    }


def futility_reason(oracle_failures: int, invalid_counts: dict[str, int]) -> str | None:
    if oracle_failures >= 20:
        return "twentieth_oracle_failure"
    if invalid_counts and max(invalid_counts.values()) >= 5:
        return "fifth_invalid_case_in_a_path"
    return None


def resolve_resource_projection(
    config: dict[str, Any],
    replicates: list[dict[str, float]],
    bookkeeping: dict[str, Any],
    legacy_control_verification: dict[str, Any],
) -> dict[str, Any]:
    if len(replicates) != int(config["resource_protocol"]["timed_replicates"]):
        raise ValueError("all three complete timing replicates are required")
    max_total_wall = max(float(item["total_wall_seconds"]) for item in replicates)
    max_total_cpu = max(float(item["total_process_cpu_seconds"]) for item in replicates)
    max_portfolio_wall = max(float(item["portfolio_wall_seconds"]) for item in replicates)
    max_portfolio_cpu = max(float(item["portfolio_process_cpu_seconds"]) for item in replicates)
    wall_upper = 1.20 * max_total_wall + 5.0
    cpu_upper = 1.20 * max_total_cpu + 5.0
    if (
        not bool(legacy_control_verification.get("passed"))
        or not bool(legacy_control_verification.get("prefix_residue_bitmatch_passed"))
        or not isinstance(legacy_control_verification.get("legacy_output_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", legacy_control_verification["legacy_output_sha256"]
        )
        is None
        or legacy_control_verification.get("legacy_output_sha256")
        != legacy_control_verification.get("observed_control_output_sha256")
    ):
        raise ValueError("measured legacy-control verification must pass")
    if int(legacy_control_verification.get("cases_per_orientation", -1)) != 110:
        raise ValueError("legacy-control timing must cover two orientations of all 110 rows")
    # The vectorized observation covers 8x110 once. Formal needs 8x200 plus
    # one exact rerun. Legacy is measured independently at 2x110 and scaled
    # to 2x200. Schema-equivalent bookkeeping is measured independently.
    vectorized_scale = (8 * 200 * 2) / (8 * 110)
    legacy_scale = (2 * 200) / (2 * 110)
    projected_wall = (
        max_portfolio_wall * vectorized_scale
        + float(legacy_control_verification["wall_seconds"]) * legacy_scale
        + float(bookkeeping["wall_seconds"])
    )
    projected_cpu = (
        max_portfolio_cpu * vectorized_scale
        + float(legacy_control_verification["process_cpu_seconds"]) * legacy_scale
        + float(bookkeeping["process_cpu_seconds"])
    )
    resource = config["resource_protocol"]
    return {
        "candidate_total_wall_upper_seconds": wall_upper,
        "candidate_total_process_cpu_upper_seconds_diagnostic": cpu_upper,
        "candidate_wall_gate_passed": wall_upper <= float(config["gates"]["candidate_wall_upper_seconds"]),
        "projected_formal_wall_seconds": projected_wall,
        "projected_formal_process_cpu_seconds": projected_cpu,
        "formal_soft_wall_seconds": projected_wall * float(resource["formal_soft_multiplier_over_measured_projection"]),
        "formal_soft_process_cpu_seconds": projected_cpu * float(resource["formal_soft_multiplier_over_measured_projection"]),
        "formal_hard_wall_seconds": projected_wall * float(resource["formal_hard_multiplier_over_measured_projection"]),
        "formal_hard_process_cpu_seconds": projected_cpu * float(resource["formal_hard_multiplier_over_measured_projection"]),
        "formal_path_case_equivalents": 3600,
        "formal_vectorized_path_case_equivalents": 3200,
        "formal_legacy_path_case_equivalents": 400,
        "formal_vectorized_scale_from_8x110": vectorized_scale,
        "formal_legacy_scale_from_2x110": legacy_scale,
    }


def formal_refusal_reasons(
    config: dict[str, Any],
    calibration: dict[str, Any] | None,
    current_hashes: dict[str, str],
    repository_dirty: bool,
    authorization: str | None,
    calibration_artifact_sha256: str | None = None,
) -> list[str]:
    resolved = config["resource_protocol"].get("resolved", {})
    reasons: list[str] = []
    for key in (
        "environment_sha256",
        "preregistration_sha256",
        "source_note_sha256",
        "seed_retirement_manifest_sha256",
    ):
        if key not in current_hashes:
            reasons.append(f"current_context_missing:{key}")
    if resolved.get("calibration_status") != "complete":
        reasons.append("calibration_pending")
    if calibration is None:
        reasons.append("calibration_missing")
    else:
        if calibration.get("status") != "complete":
            reasons.append("calibration_artifact_not_complete")
        source = calibration.get("source", {})
        for key, value in current_hashes.items():
            if source.get(key) != value:
                reasons.append(f"hash_mismatch:{key}")
        expected_workload = {
            "candidate_tier_batch_sizes": [100] * 11,
            "candidate_total_rows": 1100,
            "portfolio_batch_sizes": [10, 100],
            "portfolio_total_rows": 110,
        }
        if calibration.get("workload") != expected_workload:
            reasons.append("calibration_workload_mismatch")
        replicates = calibration.get("replicates")
        if not isinstance(replicates, list) or len(replicates) != int(
            config["resource_protocol"]["timed_replicates"]
        ):
            reasons.append("calibration_replicates_incomplete")
        measured = calibration.get("resolved_resource", {})
        if not bool(measured.get("candidate_wall_gate_passed")) or float(
            measured.get("candidate_total_wall_upper_seconds", math.inf)
        ) > float(config["gates"]["candidate_wall_upper_seconds"]):
            reasons.append("candidate_runtime_gate_miss")
        legacy = calibration.get("legacy_sigma_zero_control_verification", {})
        if (
            not bool(legacy.get("passed"))
            or not bool(legacy.get("prefix_residue_bitmatch_passed"))
            or int(legacy.get("cases_per_orientation", -1)) != 110
            or not isinstance(legacy.get("legacy_output_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", legacy["legacy_output_sha256"]) is None
            or legacy.get("legacy_output_sha256")
            != legacy.get("observed_control_output_sha256")
        ):
            reasons.append("legacy_sigma_zero_verification_missing")
        weights = calibration.get("weights", {})
        if (
            not bool(weights.get("unchanged"))
            or not weights.get("state_dict_sha256_before")
            or weights.get("state_dict_sha256_before") != weights.get("state_dict_sha256_after")
        ):
            reasons.append("source_weights_not_proven_unchanged")
    expected_artifact = resolved.get("calibration_output_sha256")
    if (
        not isinstance(expected_artifact, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_artifact) is None
        or not isinstance(calibration_artifact_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", calibration_artifact_sha256) is None
    ):
        reasons.append("calibration_artifact_hash_missing")
    elif expected_artifact != calibration_artifact_sha256:
        reasons.append("calibration_artifact_hash_mismatch")
    if resolved.get("formal_collision_audit_status") != "passed":
        reasons.append("formal_collision_audit_missing")
    if repository_dirty:
        reasons.append("repository_dirty")
    if (
        resolved.get("formal_authorization_status") != "authorized"
        or authorization != config["formal"]["required_authorization"]
    ):
        reasons.append("later_judge_authorization_missing")
    return reasons


def require_formal_preconditions(
    config: dict[str, Any],
    calibration: dict[str, Any] | None,
    current_hashes: dict[str, str],
    repository_dirty: bool,
    authorization: str | None,
    calibration_artifact_sha256: str | None = None,
) -> None:
    reasons = formal_refusal_reasons(
        config,
        calibration,
        current_hashes,
        repository_dirty,
        authorization,
        calibration_artifact_sha256,
    )
    if reasons:
        raise RuntimeError("formal execution refused: " + ", ".join(reasons))


def formal_schema() -> dict[str, Any]:
    return {
        "schema_version": "exp649-formal-v1",
        "case": ["case_index", "case_id", "identity_id", "a", "b", "p", "expected"],
        "path": [
            "path_id",
            "output",
            "prefix_outputs",
            "orientation",
            "noise_seed",
            "noise_coordinate",
            "valid",
            "prefix_validity",
            "timing",
        ],
        "provenance": [
            "repository_commit",
            "source_checkpoint_sha256",
            "source_model_sha256",
            "candidate_artifact_sha256",
            "candidate_model_sha256",
            "candidate_manifest_sha256",
            "seed_retirement_manifest_sha256",
            "config_sha256",
            "scientific_config_sha256",
            "resource_protocol_sha256",
            "runner_sha256",
            "preregistration_sha256",
            "source_note_sha256",
            "dataset_sha256",
            "environment_sha256",
        ],
    }


def verify_official_evaluator_sources(config: dict[str, Any]) -> dict[str, str]:
    """Verify pinned evaluator bytes before importing or executing any of them."""

    official = config["official_evaluator"]
    repository = (ROOT / official["repository"]).resolve()
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()
    if commit != official["commit_sha"]:
        raise RuntimeError("pinned official evaluator commit mismatch")
    official_hashes: dict[str, str] = {"official_evaluator_commit_sha": commit}
    for name, record in official["source_files"].items():
        observed = sha256((ROOT / record["path"]).resolve())
        if observed != record["sha256"]:
            raise RuntimeError(f"pinned official evaluator {name} hash mismatch")
        official_hashes[f"official_{name}_sha256"] = observed
    return official_hashes


def _expected_official_source_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {
        "official_evaluator_commit_sha": config["official_evaluator"]["commit_sha"],
        **{
            f"official_{name}_sha256": record["sha256"]
            for name, record in config["official_evaluator"]["source_files"].items()
        },
    }


def current_source_hashes(
    config: dict[str, Any], verified_official_hashes: dict[str, str] | None = None
) -> dict[str, str]:
    retirement_manifest_hash = verify_seed_retirement_manifest(config)
    official_hashes = (
        verify_official_evaluator_sources(config)
        if verified_official_hashes is None
        else copy.deepcopy(verified_official_hashes)
    )
    if official_hashes != _expected_official_source_hashes(config):
        raise RuntimeError("official evaluator verification receipt is not the frozen receipt")
    return {
        "source_checkpoint_sha256": sha256((ROOT / config["source_checkpoint"]).resolve()),
        "source_model_sha256": sha256((ROOT / config["source_model"]).resolve()),
        "candidate_artifact_sha256": sha256((ROOT / config["candidate_artifact"]).resolve()),
        "candidate_model_sha256": sha256((ROOT / config["candidate_model"]).resolve()),
        "candidate_manifest_sha256": sha256((ROOT / config["candidate_manifest"]).resolve()),
        "seed_retirement_manifest_sha256": retirement_manifest_hash,
        "scientific_config_sha256": scientific_config_sha256(config),
        "resource_protocol_sha256": resource_protocol_sha256(config),
        "runner_sha256": sha256(Path(__file__)),
        **official_hashes,
    }


def _master_seed_bytes(root_seed: int) -> bytes:
    if not 0 <= root_seed < 1 << 256:
        raise ValueError("root seed must fit the frozen unsigned 32-byte encoding")
    return root_seed.to_bytes(32, "big", signed=False)


def _input_record(
    tier_id: int, index: int, a: int, b: int, p: int
) -> dict[str, Any]:
    payload = {"tier_id": tier_id, "index": index, "a": str(a), "b": str(b), "p": str(p)}
    payload["case_id"] = sha256_text(canonical_json(payload))
    payload["identity_id"] = sha256_text(str(p))
    return payload


def _build_label_free_official_inputs(
    config: dict[str, Any],
    root_seed: int,
    verified_official_hashes: dict[str, str] | None = None,
    *,
    allow_active_calibration_root: bool = False,
) -> dict[str, Any]:
    """Clone official input generation for a supplied root without labels."""

    validate_frozen_config(config)
    protected_roots = {
        int(config["data"]["calibration_root_seed"]),
        int(config["data"]["formal_root_seed"]),
        *(int(value) for value in config["data"]["retired_root_seed_denylist"]),
    }
    if root_seed in protected_roots and not (
        allow_active_calibration_root
        and root_seed == int(config["data"]["calibration_root_seed"])
    ):
        raise RuntimeError("test/helper generation refused for an active or retired experiment root")
    verified = (
        verify_official_evaluator_sources(config)
        if verified_official_hashes is None
        else verified_official_hashes
    )
    if verified != _expected_official_source_hashes(config):
        raise RuntimeError("official evaluator verification receipt is not the frozen receipt")
    master_seed = _master_seed_bytes(root_seed)
    rows_per_tier = int(config["candidate_geometry"]["rows_per_tier"])
    primes_per_tier = int(config["data"]["primes_per_tier"])
    edge_cases = int(config["data"]["edge_cases_per_tier"])
    from modchallenge.config import MULT_SUB_TIERS, TIERS
    from modchallenge.testgen.generator import _tier0_prime_for_sub
    from modchallenge.testgen.primes import generate_primes_for_tier
    from sympy import nextprime

    candidate_batches: list[dict[str, Any]] = []

    # This is the input-only equivalent of pinned
    # ``_generate_multiplication_cases``: same one HMAC-derived RNG, prime
    # construction, sub-tier order, numeric intervals, and first-four edges.
    tier0_rng = random.Random(
        hmac.new(master_seed, b"cases-tier-0-mult", hashlib.sha256).digest()
    )
    tier0_records: list[dict[str, Any]] = []
    per_sub, remainder = divmod(rows_per_tier, len(MULT_SUB_TIERS))
    sub_primes = [
        int(nextprime((2 ** max_bits - 1) ** 2))
        if max_bits <= 128
        else int(_tier0_prime_for_sub(max_bits))
        for _, max_bits in MULT_SUB_TIERS
    ]
    edge_ops: tuple[tuple[int | None, int | None], ...] = (
        (0, None),
        (None, 0),
        (1, None),
        (None, 1),
    )
    for sub_index, (minimum_bits, maximum_bits) in enumerate(MULT_SUB_TIERS):
        count = per_sub + (1 if sub_index < remainder else 0)
        lo = 2 ** (minimum_bits - 1) if minimum_bits > 1 else 0
        hi = 2 ** maximum_bits
        modulus = sub_primes[sub_index]
        for local_index in range(count):
            if sub_index == 0 and local_index < edge_cases:
                left_fixed, right_fixed = edge_ops[local_index % len(edge_ops)]
                left = left_fixed if left_fixed is not None else tier0_rng.randrange(lo, hi)
                right = right_fixed if right_fixed is not None else tier0_rng.randrange(lo, hi)
            else:
                left = tier0_rng.randrange(lo, hi)
                right = tier0_rng.randrange(lo, hi)
            tier0_records.append(
                _input_record(0, len(tier0_records), left, right, modulus)
            )
    candidate_batches.append({"tier_id": 0, "records": tier0_records})

    # Input-only equivalent of pinned ``_generate_tier_cases``. In
    # particular, random cases select among five primes with the cases RNG;
    # they do not cycle through a reduced prime set.
    for tier in TIERS[1:]:
        tier_id = int(tier.tier_id)
        rng = random.Random(
            hmac.new(
                master_seed, f"cases-tier-{tier_id}".encode(), hashlib.sha256
            ).digest()
        )
        primes = generate_primes_for_tier(tier, primes_per_tier, master_seed)
        records: list[dict[str, Any]] = []
        operand_bits = int(tier.operand_bits if tier.operand_bits > 0 else tier.max_bits)
        max_value = 2 ** operand_bits
        edge_count = min(edge_cases, rows_per_tier)
        for index in range(edge_count):
            modulus = primes[index % len(primes)]
            left_fixed, right_fixed = edge_ops[index % len(edge_ops)]
            left = left_fixed if left_fixed is not None else rng.randrange(0, max_value)
            right = right_fixed if right_fixed is not None else rng.randrange(0, max_value)
            records.append(_input_record(tier_id, index, left, right, modulus))
        for index in range(edge_count, rows_per_tier):
            modulus = primes[rng.randrange(len(primes))]
            left = rng.randrange(0, max_value)
            right = rng.randrange(0, max_value)
            records.append(_input_record(tier_id, index, left, right, modulus))
        candidate_batches.append({"tier_id": tier_id, "records": records})

    # The fifth Tier-0 sub-tier is 65--128 operand bits; its product-bound p
    # occupies the width-256 dispatch. Tier 7 contributes the other 100 rows.
    tier0_width256 = tier0_records[40:50]
    tier7 = candidate_batches[7]["records"]
    selected = [
        ("tier0_width256_band", tier0_width256),
        ("tier7", tier7),
    ]
    portfolio_batches: list[dict[str, Any]] = []
    case_index = 0
    for batch_id, records in selected:
        cases = {"a": [], "b": [], "p": [], "case_index": []}
        for record in records:
            cases["a"].append(int(record["a"]))
            cases["b"].append(int(record["b"]))
            cases["p"].append(int(record["p"]))
            cases["case_index"].append(case_index)
            case_index += 1
        dataset_hash = sha256_text(canonical_json(cases))
        portfolio_batches.append(
            {"batch_id": batch_id, "records": records, "cases": cases, "dataset_sha256": dataset_hash}
        )

    all_records = [record for batch in candidate_batches for record in batch["records"]]
    if len(all_records) != 1100 or [len(batch["records"]) for batch in portfolio_batches] != [10, 100]:
        raise RuntimeError("generated calibration workload does not match frozen geometry")
    if any("expected" in record for record in all_records):
        raise RuntimeError("calibration input generation produced forbidden labels")
    return {
        "candidate_batches": candidate_batches,
        "portfolio_batches": portfolio_batches,
        "dataset_sha256": sha256_text(canonical_json(all_records)),
        "identity_ids": sorted({record["identity_id"] for record in all_records}),
        "case_ids": [record["case_id"] for record in all_records],
    }


def build_calibration_inputs(
    config: dict[str, Any], verified_official_hashes: dict[str, str] | None = None
) -> dict[str, Any]:
    """Build only input records from the frozen calibration root; never labels."""

    root_seed = int(config["data"]["calibration_root_seed"])
    retired = {int(value) for value in config["data"]["retired_root_seed_denylist"]}
    if root_seed in retired or root_seed == int(config["data"]["formal_root_seed"]):
        raise RuntimeError("production calibration root is retired or collides with formal root")
    return _build_label_free_official_inputs(
        config,
        root_seed,
        verified_official_hashes,
        allow_active_calibration_root=True,
    )


def calibration_workload_shape(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_tier_batch_sizes": [len(batch["records"]) for batch in inputs["candidate_batches"]],
        "candidate_total_rows": sum(len(batch["records"]) for batch in inputs["candidate_batches"]),
        "portfolio_batch_sizes": [len(batch["records"]) for batch in inputs["portfolio_batches"]],
        "portfolio_total_rows": sum(len(batch["records"]) for batch in inputs["portfolio_batches"]),
    }


def load_verified_official_decoder(
    config: dict[str, Any],
    verified_official_hashes: dict[str, str] | None = None,
) -> tuple[Callable[..., int], type[Exception]]:
    verified = (
        verify_official_evaluator_sources(config)
        if verified_official_hashes is None
        else verified_official_hashes
    )
    if verified != _expected_official_source_hashes(config):
        raise RuntimeError("official evaluator verification receipt is not the frozen receipt")
    from modchallenge.evaluation.decoder import MalformedOutput, decode_answer

    return decode_answer, MalformedOutput


def load_frozen_models(
    config: dict[str, Any], verified_official_hashes: dict[str, str] | None = None
) -> tuple[torch.nn.Module, Any]:
    verified = (
        verify_official_evaluator_sources(config)
        if verified_official_hashes is None
        else verified_official_hashes
    )
    if verified != _expected_official_source_hashes(config):
        raise RuntimeError("official evaluator verification receipt is not the frozen receipt")
    from safetensors.torch import load_file

    source_path = (ROOT / config["source_checkpoint"]).resolve()
    candidate_path = (ROOT / config["candidate_artifact"]).resolve()
    if sha256(source_path) != config["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")
    if sha256(candidate_path) != config["candidate_artifact_sha256"]:
        raise RuntimeError("Candidate005 artifact hash mismatch")
    for path_key, hash_key in (
        ("source_model", "source_model_sha256"),
        ("candidate_model", "candidate_model_sha256"),
        ("candidate_manifest", "candidate_manifest_sha256"),
    ):
        if sha256((ROOT / config[path_key]).resolve()) != config[hash_key]:
            raise RuntimeError(f"{path_key} hash mismatch")
    cellular = load_module(
        "cellular_for_exp649", ROOT / "experiments/exp_220_width_independent_cellular/model.py"
    )
    source = cellular.SharedDilatedConvGRU(
        width=int(config["model"]["width"]),
        hidden=int(config["model"]["hidden"]),
        cycles=int(config["model"]["cycles"]),
    )
    source.load_state_dict(load_file(source_path))
    source.eval().requires_grad_(False)
    from modchallenge.evaluation.loader import load_model, validate_manifest

    submission_dir = ROOT / "submission"
    manifest = validate_manifest(submission_dir)
    if manifest.output_base != config["candidate_output_base"]:
        raise RuntimeError("Candidate005 output-base contract changed")
    candidate = load_model(submission_dir, manifest)
    return source, candidate


def _process_cpu(process: psutil.Process) -> float:
    times = process.cpu_times()
    return float(times.user + times.system)


def _check_calibration_runaway(
    *,
    stage: str,
    run_started: float,
    run_cpu_started: float,
    process: psutil.Process,
    reused_wall_charge: float,
    reused_cpu_charge: float,
    wall_ceiling: float,
    cpu_ceiling: float,
) -> None:
    if reused_wall_charge + time.perf_counter() - run_started > wall_ceiling:
        raise TimeoutError(f"EXP-649 calibration exceeded its wall ceiling {stage}")
    if reused_cpu_charge + _process_cpu(process) - run_cpu_started > cpu_ceiling:
        raise TimeoutError(f"EXP-649 calibration exceeded its process-CPU ceiling {stage}")


def _deterministic_rollout_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "paths": {
            path_id: {
                key: value
                for key, value in record.items()
                if key != "timing"
            }
            for path_id, record in result["paths"].items()
        },
        "coordinate_log": result["coordinate_log"],
        "state_dict_sha256_before": result["state_dict_sha256_before"],
        "state_dict_sha256_after": result["state_dict_sha256_after"],
    }


def _reduced_portfolio_cases(batch: dict[str, Any]) -> dict[str, list[int]]:
    raw = batch["cases"]
    return {
        "a": [value % modulus for value, modulus in zip(raw["a"], raw["p"], strict=True)],
        "b": [value % modulus for value, modulus in zip(raw["b"], raw["p"], strict=True)],
        "p": list(raw["p"]),
        "case_index": list(raw["case_index"]),
    }


@torch.inference_mode()
def execute_calibration_replicate(
    config: dict[str, Any],
    source: torch.nn.Module,
    candidate: Any,
    inputs: dict[str, Any],
    decoder_bundle: tuple[Callable[..., int], type[Exception]] | None = None,
    runaway_guard: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    decode_answer, malformed_output = (
        load_verified_official_decoder(config)
        if decoder_bundle is None
        else decoder_bundle
    )
    process = psutil.Process()
    cpu_start = _process_cpu(process)
    wall_start = time.perf_counter()
    candidate_outputs: list[Any] = []
    for batch in inputs["candidate_batches"]:
        tier_id = int(batch["tier_id"])
        if runaway_guard is not None:
            runaway_guard(f"before candidate tier {tier_id} batch")
        encoded = [
            (
                candidate.preprocess_a(record["a"]),
                candidate.preprocess_b(record["b"]),
                candidate.preprocess_p(record["p"]),
            )
            for record in batch["records"]
        ]
        if len(encoded) != 100:
            raise RuntimeError("candidate tier batching changed")
        digits_batch = candidate.predict_digits_batch(encoded)
        if len(digits_batch) != len(batch["records"]):
            raise RuntimeError("candidate output count changed")
        decoded_batch: list[str] = []
        for digits, record in zip(digits_batch, batch["records"], strict=True):
            try:
                decoded_batch.append(
                    str(
                        decode_answer(
                            digits,
                            base=config["candidate_output_base"],
                            prime=int(record["p"]),
                            is_tier_zero=int(batch["tier_id"]) == 0,
                        )
                    )
                )
            except malformed_output:
                decoded_batch.append("")
        candidate_outputs.append(decoded_batch)
        if runaway_guard is not None:
            runaway_guard(f"after candidate tier {tier_id} batch")

    portfolio_cpu_start = _process_cpu(process)
    portfolio_wall_start = time.perf_counter()
    portfolio_results: list[dict[str, Any]] = []
    specs = path_specs(config)
    for batch in inputs["portfolio_batches"]:
        if runaway_guard is not None:
            runaway_guard(f"before portfolio batch {batch['batch_id']}")
        # Pairwise sanctioned preprocessing is part of the timed future dispatch.
        cases = _reduced_portfolio_cases(batch)
        result = vectorized_rollout(
            source,
            cases,
            specs,
            "calibration",
            batch["dataset_sha256"],
            int(config["model"]["width"]),
            [int(value) for value in config["model"]["dilation_schedule"]],
            runaway_guard=runaway_guard,
            runaway_check_interval=int(
                config["resource_protocol"]["in_stage_runaway_guard"][
                    "vectorized_multiplier_bit_interval"
                ]
            ),
        )
        portfolio_results.append(result)
        if runaway_guard is not None:
            runaway_guard(f"after portfolio batch {batch['batch_id']}")
    portfolio_wall = time.perf_counter() - portfolio_wall_start
    portfolio_cpu = _process_cpu(process) - portfolio_cpu_start
    total_wall = time.perf_counter() - wall_start
    total_cpu = _process_cpu(process) - cpu_start
    deterministic = [_deterministic_rollout_payload(result) for result in portfolio_results]
    control_evidence = [
        {
            path_id: {
                "outputs": result["paths"][path_id]["outputs"],
                "prefix_outputs": result["paths"][path_id]["prefix_outputs"],
                "prefix_validity": result["paths"][path_id]["prefix_validity"],
            }
            for path_id in ("control_original", "control_swapped")
        }
        for result in portfolio_results
    ]
    coordinate_summary = [
        {
            "batch_id": batch["batch_id"],
            "case_indices": batch["cases"]["case_index"],
            "draw_sites": len(result["coordinate_log"]),
            "first": result["coordinate_log"][0],
            "last": result["coordinate_log"][-1],
            "path_generator_roots": {
                path_id: record["noise_coordinates"][0]["derived_generator_seed"]
                for path_id, record in result["paths"].items()
            },
        }
        for batch, result in zip(inputs["portfolio_batches"], portfolio_results, strict=True)
    ]
    return {
        "total_wall_seconds": total_wall,
        "total_process_cpu_seconds": total_cpu,
        "portfolio_wall_seconds": portfolio_wall,
        "portfolio_process_cpu_seconds": portfolio_cpu,
        "candidate_output_sha256": sha256_text(canonical_json(candidate_outputs)),
        "portfolio_output_sha256": sha256_text(canonical_json(deterministic)),
        "coordinate_summary": coordinate_summary,
        "control_evidence": control_evidence,
    }


def verify_actual_sigma_zero_controls(
    config: dict[str, Any],
    source: torch.nn.Module,
    inputs: dict[str, Any],
    observed_replicate: dict[str, Any],
    runaway_guard: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compare timed cbc-checkpoint controls with untouched legacy rollouts."""

    process = psutil.Process()
    cpu_start = _process_cpu(process)
    wall_start = time.perf_counter()
    state_before = state_dict_sha256(source)
    checked = 0
    legacy_evidence: list[dict[str, Any]] = []
    observed_evidence: list[dict[str, Any]] = []
    for batch, observed in zip(
        inputs["portfolio_batches"], observed_replicate["control_evidence"], strict=True
    ):
        cases = _reduced_portfolio_cases(batch)
        interval = int(
            config["resource_protocol"]["in_stage_runaway_guard"][
                "legacy_multiplier_bit_interval"
            ]
        )
        original = legacy_rollout(
            source,
            cases,
            "original",
            int(config["model"]["width"]),
            runaway_guard,
            interval,
        )
        swapped = legacy_rollout(
            source,
            cases,
            "swapped",
            int(config["model"]["width"]),
            runaway_guard,
            interval,
        )
        assert_sigma_zero_legacy_equivalence({"paths": observed}, original, swapped)
        legacy_evidence.append(
            {"control_original": original, "control_swapped": swapped}
        )
        observed_evidence.append(copy.deepcopy(observed))
        checked += len(cases["p"])
    state_after = state_dict_sha256(source)
    if state_before != state_after:
        raise RuntimeError("source weights changed during legacy control verification")
    legacy_output_sha256 = sha256_text(canonical_json(legacy_evidence))
    observed_output_sha256 = sha256_text(canonical_json(observed_evidence))
    if legacy_output_sha256 != observed_output_sha256:
        raise RuntimeError("legacy/control prefix-residue evidence hash mismatch")
    return {
        "passed": True,
        "prefix_residue_bitmatch_passed": True,
        "cases_per_orientation": checked,
        "legacy_output_sha256": legacy_output_sha256,
        "observed_control_output_sha256": observed_output_sha256,
        "wall_seconds": time.perf_counter() - wall_start,
        "process_cpu_seconds": _process_cpu(process) - cpu_start,
        "state_dict_sha256_before": state_before,
        "state_dict_sha256_after": state_after,
    }


def _target_free_formal_projection_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Build the 200x8 decoded-prefix evidence shape without any targets."""

    cases = int(config["data"]["formal_cases"])
    width = int(config["model"]["width"])
    specs = path_specs(config)
    case_records = [
        {
            "case_index": index,
            "case_id": sha256_text(f"target-free-projection-case-{index}"),
            "identity_id": sha256_text(f"target-free-projection-identity-{index}"),
            "a": "0",
            "b": "0",
            "p": "257",
        }
        for index in range(cases)
    ]
    path_records: list[dict[str, Any]] = []
    for spec in specs:
        for index in range(cases):
            path_records.append(
                {
                    "case_index": index,
                    "path_id": spec.path_id,
                    "output": 0,
                    "prefix_outputs": [0] * width,
                    "orientation": spec.orientation,
                    "noise_seed": spec.noise_seed,
                    "noise_coordinate": {
                        "case_index": index,
                        "multiplier_bits": [width - 1, 0, -1],
                        "injection_indices": [0, 16],
                    },
                    "valid": True,
                    "prefix_validity": [True] * width,
                    "timing": {
                        "vectorized_shared_wall_seconds": 0.0,
                        "vectorized_shared_process_cpu_seconds": 0.0,
                    },
                }
            )
    return {
        "schema_version": config["formal"]["raw_schema_version"],
        "role": "target-free-schema-equivalent-resource-projection",
        "contains_targets_or_labels": False,
        "cases": case_records,
        "paths": path_records,
        "provenance": {
            field: "0" * 64
            for field in formal_schema()["provenance"]
        },
    }


def _measure_bookkeeping(
    config: dict[str, Any],
    output: Path,
    runaway_guard: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output = _ensure_scratch_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if runaway_guard is not None:
        runaway_guard("before target-free formal bookkeeping projection")
    process = psutil.Process()
    cpu_start = _process_cpu(process)
    wall_start = time.perf_counter()
    payload = _target_free_formal_projection_payload(config)
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    artifact = output / "formal-bookkeeping-projection.json"
    temporary = artifact.with_suffix(artifact.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(artifact)
    artifact_sha256 = sha256(artifact)
    if runaway_guard is not None:
        runaway_guard("after target-free formal bookkeeping projection")
    elapsed = time.perf_counter() - wall_start
    cpu_elapsed = _process_cpu(process) - cpu_start
    return {
        "wall_seconds": elapsed,
        "process_cpu_seconds": cpu_elapsed,
        "artifact_path": (
            str(artifact.relative_to(ROOT)) if artifact.is_relative_to(ROOT) else str(artifact)
        ),
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact.stat().st_size,
        "cases": int(config["data"]["formal_cases"]),
        "paths": len(path_specs(config)),
        "prefixes_per_path_case": int(config["model"]["width"]),
        "contains_targets_or_labels": False,
    }


def _ensure_scratch_output(output: Path) -> Path:
    resolved = output.resolve()
    scratch = (ROOT / ".scratch").resolve()
    if not resolved.is_relative_to(scratch):
        raise RuntimeError("calibration may write only below repository .scratch")
    return resolved


def runtime_environment() -> dict[str, Any]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[-1].strip()
                break
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "torch_config_sha256": sha256_text(torch.__config__.show()),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def configure_frozen_runtime(config: dict[str, Any]) -> dict[str, Any]:
    """Configure and verify the exact thread/determinism contract."""

    resource = config["resource_protocol"]
    desired_threads = int(resource["torch_threads"])
    desired_interop = int(resource["torch_interop_threads"])
    desired_determinism = bool(resource["deterministic_algorithms"])
    torch.set_num_threads(desired_threads)
    if torch.get_num_interop_threads() != desired_interop:
        try:
            torch.set_num_interop_threads(desired_interop)
        except RuntimeError as exc:
            if torch.get_num_interop_threads() != desired_interop:
                raise RuntimeError(
                    "cannot establish frozen torch interop-thread environment"
                ) from exc
    torch.use_deterministic_algorithms(desired_determinism)
    environment = runtime_environment()
    if (
        environment["torch_threads"] != desired_threads
        or environment["torch_interop_threads"] != desired_interop
        or environment["deterministic_algorithms"] != desired_determinism
    ):
        raise RuntimeError("frozen runtime environment configuration did not take effect")
    return environment


def calibration_cache_material(
    config: dict[str, Any],
    inputs: dict[str, Any],
    environment: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "environment_sha256": sha256_text(canonical_json(environment)),
        "source_hashes": copy.deepcopy(source_hashes),
        "scientific_config_sha256": scientific_config_sha256(config),
        "resource_protocol_sha256": resource_protocol_sha256(config),
        "dataset_sha256": inputs["dataset_sha256"],
        "preregistration_sha256": sha256((ROOT / config["preregistration"]).resolve()),
        "source_note_sha256": sha256((ROOT / config["source_note"]).resolve()),
    }


def current_formal_context_hashes(
    config: dict[str, Any],
    environment: dict[str, Any],
    verified_official_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    return {
        **current_source_hashes(config, verified_official_hashes),
        "environment_sha256": sha256_text(canonical_json(environment)),
        "preregistration_sha256": sha256((ROOT / config["preregistration"]).resolve()),
        "source_note_sha256": sha256((ROOT / config["source_note"]).resolve()),
    }


def _replicate_chain_sha256(
    cache_key: str, warmup: dict[str, Any], replicates: list[dict[str, Any]]
) -> str:
    chain = sha256_text(cache_key + canonical_json(warmup))
    for replicate in replicates:
        chain = sha256_text(chain + canonical_json(replicate))
    return chain


def _partial_content_sha256(payload: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("content_sha256", None)
    return sha256_text(canonical_json(unsigned))


def _validate_nonnegative_finite(record: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"calibration partial {field} is not numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise RuntimeError(f"calibration partial {field} is not finite and nonnegative")


def _validate_replicate_structure(replicate: dict[str, Any]) -> None:
    required = {
        "total_wall_seconds",
        "total_process_cpu_seconds",
        "portfolio_wall_seconds",
        "portfolio_process_cpu_seconds",
        "candidate_output_sha256",
        "portfolio_output_sha256",
        "coordinate_summary",
        "control_evidence",
    }
    if not isinstance(replicate, dict) or set(replicate) != required:
        raise RuntimeError("calibration partial replicate is incomplete")
    _validate_nonnegative_finite(
        replicate,
        (
            "total_wall_seconds",
            "total_process_cpu_seconds",
            "portfolio_wall_seconds",
            "portfolio_process_cpu_seconds",
        ),
    )
    for field in ("candidate_output_sha256", "portfolio_output_sha256"):
        if not isinstance(replicate[field], str) or re.fullmatch(
            r"[0-9a-f]{64}", replicate[field]
        ) is None:
            raise RuntimeError(f"calibration partial {field} is not a SHA-256")
    summaries = replicate["coordinate_summary"]
    if not isinstance(summaries, list) or len(summaries) != 2:
        raise RuntimeError("calibration partial coordinate summary is malformed")
    expected_batches = (("tier0_width256_band", 10), ("tier7", 100))
    for summary, (batch_id, cases) in zip(summaries, expected_batches, strict=True):
        required_summary = {
            "batch_id",
            "case_indices",
            "draw_sites",
            "first",
            "last",
            "path_generator_roots",
        }
        if not isinstance(summary, dict) or set(summary) != required_summary:
            raise RuntimeError("calibration partial coordinate summary is malformed")
        if (
            summary["batch_id"] != batch_id
            or not isinstance(summary["case_indices"], list)
            or len(summary["case_indices"]) != cases
            or int(summary["draw_sites"]) != 256 * 17
            or summary["first"].get("multiplier_bit") != 255
            or summary["first"].get("injection_index") != 0
            or summary["last"].get("multiplier_bit") != 0
            or summary["last"].get("injection_index") != 16
            or not isinstance(summary["path_generator_roots"], dict)
            or len(summary["path_generator_roots"]) != 8
        ):
            raise RuntimeError("calibration partial coordinate evidence changed shape")
    controls = replicate["control_evidence"]
    if not isinstance(controls, list) or len(controls) != 2:
        raise RuntimeError("calibration partial control evidence is malformed")
    for observed, (_, cases) in zip(controls, expected_batches, strict=True):
        if not isinstance(observed, dict) or set(observed) != {
            "control_original",
            "control_swapped",
        }:
            raise RuntimeError("calibration partial control evidence is malformed")
        for path_id in ("control_original", "control_swapped"):
            path = observed[path_id]
            if (
                not isinstance(path, dict)
                or set(path) != {"outputs", "prefix_outputs", "prefix_validity"}
                or not isinstance(path["outputs"], list)
                or len(path["outputs"]) != cases
                or not isinstance(path["prefix_outputs"], list)
                or len(path["prefix_outputs"]) != cases
                or any(
                    not isinstance(prefix, list)
                    or len(prefix) != 256
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in prefix)
                    for prefix in path["prefix_outputs"]
                )
                or not isinstance(path["prefix_validity"], list)
                or len(path["prefix_validity"]) != cases
                or any(
                    not isinstance(prefix, list) or len(prefix) != 256
                    for prefix in path["prefix_validity"]
                )
            ):
                raise RuntimeError("calibration partial control evidence changed shape")


def validate_partial_cache(
    partial: dict[str, Any],
    expected_cache_key: str,
    expected_cache_material: dict[str, Any],
    maximum_replicates: int,
    expected_schema_version: str = "exp649-calibration-partial-v3",
) -> list[dict[str, Any]]:
    required_top_level = {
        "schema_version",
        "cache_key",
        "cache_material",
        "warmup",
        "replicates",
        "legacy_control_verification",
        "replicate_chain_sha256",
        "content_sha256",
    }
    if not isinstance(partial, dict) or set(partial) != required_top_level:
        raise RuntimeError("calibration partial schema is incomplete")
    if partial.get("schema_version") != expected_schema_version:
        raise RuntimeError("calibration partial schema version mismatch")
    if partial.get("cache_material") != expected_cache_material:
        raise RuntimeError("calibration partial cache material mismatch; refusing stale evidence")
    material_key = sha256_text(canonical_json(expected_cache_material))
    if material_key != expected_cache_key:
        raise RuntimeError("expected calibration cache key does not match its material")
    if partial.get("cache_key") != expected_cache_key:
        raise RuntimeError("calibration partial cache key mismatch; refusing stale evidence")
    if partial.get("content_sha256") != _partial_content_sha256(partial):
        raise RuntimeError("calibration partial content integrity mismatch")
    replicates = partial.get("replicates")
    if not isinstance(replicates, list) or len(replicates) > maximum_replicates:
        raise RuntimeError("calibration partial has invalid completed-replicate evidence")
    warmup = partial.get("warmup")
    _validate_replicate_structure(warmup)
    for replicate in replicates:
        _validate_replicate_structure(replicate)
    if partial.get("replicate_chain_sha256") != _replicate_chain_sha256(
        expected_cache_key, warmup, replicates
    ):
        raise RuntimeError("calibration partial replicate chain mismatch")
    legacy = partial.get("legacy_control_verification")
    if legacy is not None:
        if (
            not isinstance(legacy, dict)
            or not bool(legacy.get("passed"))
            or not bool(legacy.get("prefix_residue_bitmatch_passed"))
            or int(legacy.get("cases_per_orientation", -1)) != 110
            or not isinstance(legacy.get("legacy_output_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", legacy["legacy_output_sha256"]) is None
            or legacy.get("legacy_output_sha256")
            != legacy.get("observed_control_output_sha256")
            or not isinstance(legacy.get("state_dict_sha256_before"), str)
            or re.fullmatch(r"[0-9a-f]{64}", legacy["state_dict_sha256_before"]) is None
            or legacy.get("state_dict_sha256_before")
            != legacy.get("state_dict_sha256_after")
        ):
            raise RuntimeError("calibration partial legacy evidence is malformed")
        _validate_nonnegative_finite(legacy, ("wall_seconds", "process_cpu_seconds"))
    return copy.deepcopy(replicates)


def _write_partial_checkpoint(
    path: Path,
    cache_key: str,
    cache_material: dict[str, Any],
    warmup: dict[str, Any],
    replicates: list[dict[str, Any]],
    legacy_control_verification: dict[str, Any] | None = None,
    schema_version: str = "exp649-calibration-partial-v3",
) -> None:
    payload = {
        "schema_version": schema_version,
        "cache_key": cache_key,
        "cache_material": cache_material,
        "warmup": warmup,
        "replicates": replicates,
        "legacy_control_verification": legacy_control_verification,
        "replicate_chain_sha256": _replicate_chain_sha256(cache_key, warmup, replicates),
    }
    payload["content_sha256"] = _partial_content_sha256(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_calibration(config_path: Path, output: Path) -> dict[str, Any]:
    """Run the frozen no-label calibration; never called by import or tests."""

    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    validate_frozen_config(config)
    if git_output("status", "--porcelain"):
        raise RuntimeError("calibration requires a clean repository")
    # Verify every pinned evaluator byte before importing or executing its
    # generator, decoder, loader, or any submission/model code.
    verified_official_hashes = verify_official_evaluator_sources(config)
    source_hashes = current_source_hashes(config, verified_official_hashes)
    environment = configure_frozen_runtime(config)
    decoder_bundle = load_verified_official_decoder(config, verified_official_hashes)
    output = _ensure_scratch_output(output)
    inputs = build_calibration_inputs(config, verified_official_hashes)
    source, candidate = load_frozen_models(config, verified_official_hashes)
    cache_material = calibration_cache_material(config, inputs, environment, source_hashes)
    cache_key = sha256_text(canonical_json(cache_material))
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / config["resource_protocol"]["partial_checkpoint_filename"]
    maximum_replicates = int(config["resource_protocol"]["timed_replicates"])
    if partial_path.exists():
        replicates = validate_partial_cache(
            json.loads(partial_path.read_text()),
            cache_key,
            cache_material,
            maximum_replicates,
            config["resource_protocol"]["partial_schema_version"],
        )
    else:
        replicates = []
    reused_replicates = len(replicates)
    reused_wall_charge = sum(float(item["total_wall_seconds"]) for item in replicates)
    reused_cpu_charge = sum(float(item["total_process_cpu_seconds"]) for item in replicates)
    source_state_before = state_dict_sha256(source)
    run_started = time.perf_counter()
    process = psutil.Process()
    run_cpu_started = _process_cpu(process)
    wall_ceiling = float(config["resource_protocol"]["initial_calibration_hard_ceiling_wall_seconds"])
    cpu_ceiling = float(
        config["resource_protocol"]["initial_calibration_hard_ceiling_process_cpu_seconds"]
    )

    def runaway_guard(stage: str) -> None:
        _check_calibration_runaway(
            stage=stage,
            run_started=run_started,
            run_cpu_started=run_cpu_started,
            process=process,
            reused_wall_charge=reused_wall_charge,
            reused_cpu_charge=reused_cpu_charge,
            wall_ceiling=wall_ceiling,
            cpu_ceiling=cpu_ceiling,
        )

    warmup = execute_calibration_replicate(
        config, source, candidate, inputs, decoder_bundle, runaway_guard
    )
    _write_partial_checkpoint(
        partial_path,
        cache_key,
        cache_material,
        warmup,
        replicates,
        schema_version=config["resource_protocol"]["partial_schema_version"],
    )
    runaway_guard("after warmup")
    for _ in range(len(replicates), maximum_replicates):
        replicates.append(
            execute_calibration_replicate(
                config, source, candidate, inputs, decoder_bundle, runaway_guard
            )
        )
        _write_partial_checkpoint(
            partial_path,
            cache_key,
            cache_material,
            warmup,
            replicates,
            schema_version=config["resource_protocol"]["partial_schema_version"],
        )
        runaway_guard("after a timed replicate")
    output_pairs = {
        (item["candidate_output_sha256"], item["portfolio_output_sha256"])
        for item in [warmup, *replicates]
    }
    if len(output_pairs) != 1:
        raise RuntimeError("calibration outputs changed after RNG reset")
    source_state_after = state_dict_sha256(source)
    if source_state_before != source_state_after:
        raise RuntimeError("source weights changed during calibration")
    legacy_control_verification = verify_actual_sigma_zero_controls(
        config, source, inputs, replicates[0], runaway_guard
    )
    _write_partial_checkpoint(
        partial_path,
        cache_key,
        cache_material,
        warmup,
        replicates,
        legacy_control_verification,
        config["resource_protocol"]["partial_schema_version"],
    )
    runaway_guard("after legacy verification")
    bookkeeping = _measure_bookkeeping(config, output, runaway_guard)
    runaway_guard("after bookkeeping")
    resolved = resolve_resource_projection(
        config, replicates, bookkeeping, legacy_control_verification
    )
    payload = {
        "run_id": config["run_id"],
        "status": "complete" if resolved["candidate_wall_gate_passed"] else "runtime-gate-miss",
        "role": "training-identity-input-only-calibration-not-model-evidence",
        "source": {
            **source_hashes,
            "config_sha256": sha256(config_path),
            "repository_commit": git_output("rev-parse", "HEAD"),
            "repository_dirty": False,
            "preregistration_sha256": sha256((ROOT / config["preregistration"]).resolve()),
            "source_note_sha256": sha256((ROOT / config["source_note"]).resolve()),
            "dataset_sha256": inputs["dataset_sha256"],
            "environment_sha256": sha256_text(canonical_json(environment)),
        },
        "workload": calibration_workload_shape(inputs),
        "identity_audit_material": {
            "phase": "calibration",
            "root_seed": int(config["data"]["calibration_root_seed"]),
            "identity_ids": inputs["identity_ids"],
            "case_ids": inputs["case_ids"],
            "contains_labels_or_oracle": False,
        },
        "rng": {
            "root_seed_domain_fields": config["noise"]["root_seed_domain_fields"],
            "logical_coordinate_fields": config["noise"]["logical_coordinate_fields"],
            "coordinate_traversal": config["noise"]["coordinate_traversal"],
            "warmup_coordinate_summary": warmup["coordinate_summary"],
            "replicate_coordinate_summaries": [item["coordinate_summary"] for item in replicates],
        },
        "warmup": {
            key: value
            for key, value in warmup.items()
            if key not in {"coordinate_summary", "control_evidence"}
        },
        "replicates": [
            {
                key: value
                for key, value in item.items()
                if key not in {"coordinate_summary", "control_evidence"}
            }
            for item in replicates
        ],
        "legacy_sigma_zero_control_verification": legacy_control_verification,
        "calibration_partial": {
            "path": str(partial_path.relative_to(ROOT)),
            "sha256": sha256(partial_path),
            "cache_key": cache_key,
            "reused_completed_replicates": reused_replicates,
            "reused_wall_seconds_charged_to_runaway_ceiling": reused_wall_charge,
            "reused_process_cpu_seconds_charged_to_runaway_ceiling": reused_cpu_charge,
            "active_session_wall_seconds": time.perf_counter() - run_started,
            "active_session_process_cpu_seconds": _process_cpu(process) - run_cpu_started,
            "repeated_interrupted_warmups_reconstructed": False,
            "fresh_warmup_completed_on_this_process": True,
        },
        "bookkeeping_timing": bookkeeping,
        "resolved_resource": resolved,
        "weights": {
            "state_dict_sha256_before": source_state_before,
            "state_dict_sha256_after": source_state_after,
            "unchanged": True,
        },
        "environment": environment,
        "limitations": [
            "Calibration uses generated training-identity inputs and no labels or oracle.",
            "Each timing is a complete end-to-end replicate; resumed timings require the exact cache key and a fresh local warmup.",
            "Reused completed replicates are charged to both runaway ceilings; prior interrupted warmups are disclosed but not reconstructed.",
            "CPU upper-bound timing is diagnostic; the wall upper bound is the pass gate.",
            "No formal identity, target, score, selector, or benchmark answer is constructed.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / "calibration.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "resolved_resource": resolved}, indent=2))
    return payload


def _formal_refusal_from_disk(
    config: dict[str, Any], authorization: str | None
) -> None:
    verified_official_hashes = verify_official_evaluator_sources(config)
    environment = configure_frozen_runtime(config)
    current_hashes = current_formal_context_hashes(
        config, environment, verified_official_hashes
    )
    resolved = config["resource_protocol"].get("resolved", {})
    calibration_path = (ROOT / config["resource_protocol"]["calibration_output"]).resolve()
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    require_formal_preconditions(
        config,
        calibration,
        current_hashes,
        bool(git_output("status", "--porcelain")),
        authorization,
        sha256(calibration_path) if calibration_path.exists() else None,
    )
    # Passing the gate authorizes a later implementation task, not this scaffold.
    raise RuntimeError("formal execution refused: scaffold has no authorized formal executor")


def describe(config: dict[str, Any]) -> dict[str, Any]:
    validate_frozen_config(config)
    return {
        "run_id": config["run_id"],
        "status": config["status"],
        "scientific_config_sha256": scientific_config_sha256(config),
        "resource_protocol_sha256": resource_protocol_sha256(config),
        "formal_schema": formal_schema(),
        "formal_refusal": "pending calibration and later Judge authorization",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_649_stochastic_trajectory_breadth.yaml",
    )
    parser.add_argument("--mode", choices=("describe", "calibration", "formal"), default="describe")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-649-stochastic-trajectory-breadth",
    )
    parser.add_argument("--judge-authorization")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_frozen_config(config)
    if args.mode == "formal":
        _formal_refusal_from_disk(config, args.judge_authorization)
    if args.mode == "calibration":
        run_calibration(args.config, args.output)
        return 0
    print(json.dumps(describe(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
