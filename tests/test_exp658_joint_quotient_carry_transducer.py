"""Exhaustive L0 verification for EXP-658."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/exp_658_joint_quotient_carry_transducer"
CONFIG_PATH = ROOT / "configs/exp_658_joint_quotient_carry_transducer.yaml"
SOURCE_MODEL_PATH = ROOT / "experiments/exp_220_width_independent_cellular/model.py"
EXPECTED_SOURCE_SHA256 = (
    "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
)
EXPECTED_PARAMETER_COUNT = 103522
SOURCE_PARAMETER_COUNT = 99824
ALLOWED_PATHS = {
    "configs/exp_658_joint_quotient_carry_transducer.yaml",
    "experiments/exp_658_joint_quotient_carry_transducer/__init__.py",
    "experiments/exp_658_joint_quotient_carry_transducer/model.py",
    "experiments/exp_658_joint_quotient_carry_transducer/data.py",
    "experiments/exp_658_joint_quotient_carry_transducer/run.py",
    "knowledge/autoresearch/ablations/exp-658-joint-quotient-carry-transducer.md",
    "tests/test_exp658_joint_quotient_carry_transducer.py",
}
HISTORICAL_L0_TEST_SHA256 = (
    "8c6a4b3352e2229d3fded622637f350b56f6a99ec42771f664cb5a98f33ebbd6"
)
T045_RECORDS = {
    "experiments/runs/exp-658-joint-quotient-carry-transducer/run.json": (
        "3fc7788268ef54d097edf3db198c19782507f023843acd1f9c7ee17e3f6a5e22"
    ),
    "experiments/runs/exp-658-joint-quotient-carry-transducer/environment.json": (
        "3b4886ab656c1c64f9add7248aa95fec0d99f0492a71e43ca64148ce61a2216f"
    ),
    "experiments/runs/exp-658-joint-quotient-carry-transducer/failure.json": (
        "11aff543b63ae8e4cf7144d612d64d8d8c63146fc734e900db30d9e0d46f6e88"
    ),
    "evals/results/exp-658-joint-quotient-carry-transducer/decision.json": (
        "ebf0e3d997d52100c68b3d6d838306b6e1d291852f3d9afe4a1721487dfc36e9"
    ),
}
POST_T045_AMENDMENT = (
    ROOT
    / "experiments/runs/exp-658-joint-quotient-carry-transducer/provenance-amendment-post-t045.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model_module = load_module("exp658_model_for_tests", EXPERIMENT / "model.py")
data_module = load_module("exp658_data_for_tests", EXPERIMENT / "data.py")

_path_before_run_import = list(sys.path)
_had_model_before_run_import = "model" in sys.modules
_model_before_run_import = sys.modules.get("model")
try:
    sys.path.insert(0, str(EXPERIMENT))
    sys.modules["model"] = model_module
    run_module = load_module("exp658_run_for_tests", EXPERIMENT / "run.py")
finally:
    sys.path[:] = _path_before_run_import
    if _had_model_before_run_import:
        sys.modules["model"] = _model_before_run_import
    else:
        sys.modules.pop("model", None)

assert sys.path == _path_before_run_import
assert ("model" in sys.modules) is _had_model_before_run_import
if _had_model_before_run_import:
    assert sys.modules["model"] is _model_before_run_import

source_module = load_module("exp658_source_model_for_tests", SOURCE_MODEL_PATH)


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_freezes_l0_source_graph_arms_stages_and_future_gates() -> None:
    payload = config()
    assert payload["status"] == "preregistered-l0-only"
    assert payload["source"]["checkpoint_sha256"] == EXPECTED_SOURCE_SHA256
    assert payload["source"]["parameters"] == SOURCE_PARAMETER_COUNT
    assert payload["source"]["recurrent_input_channels"] == 98
    assert payload["model"] == {
        "hidden": 92,
        "cycles": 2,
        "quotient_classes": [0, 1, 2],
        "signed_carry_classes": [-2, -1, 0, 1],
        "quotient_message_channels": 2,
        "carry_message_channels": 2,
        "message_channels": 4,
        "message_gate_delta_initialization": "exact-zero",
        "message_candidate_delta_initialization": "exact-zero",
        "parameters": EXPECTED_PARAMETER_COUNT,
        "maximum_parameter_ratio": 1.05,
    }
    assert payload["arms"]["control"]["quotient_auxiliary_loss_weight"] == 0.0
    assert payload["arms"]["control"]["signed_carry_auxiliary_loss_weight"] == 0.0
    assert payload["arms"]["challenger"]["quotient_auxiliary_loss_weight"] == 0.10
    assert payload["arms"]["challenger"]["signed_carry_auxiliary_loss_weight"] == 0.10
    assert payload["arms"]["matched"]["sole_future_difference"] == "auxiliary-loss-weights"
    assert list(payload["stages"]) == ["L0", "L1", "L2", "L3"]
    assert payload["stages"]["L0"]["optimizer_steps"] == 0
    assert payload["run_control"]["calibration_root"] is None
    assert payload["run_control"]["scientific_roots"] == []
    assert payload["run_control"]["authorization_file"] is None
    assert payload["L0"]["exhaustive_prime_domain"]["expected_legal_tuples"] == 1991546
    assert payload["L0"]["fresh_random_transitions_per_width"] == 100000
    assert payload["L0"]["source_parity_transitions_per_width"] == 4096


def test_config_restores_exact_t017_data_and_l1_calibration_contract() -> None:
    payload = config()
    assert payload["future_data"]["training_mix"] == {
        "unbiased_natural_fraction": 0.50,
        "quotient2_fraction": 0.20,
        "reduction_boundary_fraction": 0.15,
        "reduction_boundary_relative_margin": 0.05,
        "rare_signed_carry_fraction": 0.15,
        "rare_signed_carry_classes": [-2, 1],
        "precedence": [
            "unbiased-natural",
            "quotient2",
            "reduction-boundary",
            "rare-signed-carry",
        ],
        "duplicate_rows": "prohibited",
        "deficit_fill": "unbiased-natural",
    }
    l1 = payload["L1"]
    assert l1["warmup_repeats"] == 1
    assert l1["fresh_matched_repeats"] == 3
    assert l1["reinitialize_before_each_fresh_repeat"] is True
    assert l1["matched_training_component"] == {
        "optimizer_updates_per_arm_per_repeat": 16,
        "width32_updates_per_arm_per_repeat": 8,
        "width64_updates_per_arm_per_repeat": 8,
        "width_order": "alternating",
        "batch_size": 128,
        "exact_formal_forward_loss_backward_clipping_optimizer_path": True,
    }
    assert l1["evaluation_per_arm_width_repeat"] == {
        "natural_transitions": 512,
        "natural_rollouts": 32,
    }
    assert l1["challenger_causal_rollouts_per_width_repeat"] == {
        "zero-quotient-message": 32,
        "zero-signed-carry-message": 32,
    }
    assert l1["rare_diagnostics_per_width"] == {
        "examples_per_quotient_class": 256,
        "labeled_positions_per_signed_carry_class": 256,
    }
    assert l1["checkpoint_integrity_per_arm"] == {
        "saves": 1,
        "sha256": "required",
        "reload_equivalence": "exact",
    }
    profile = l1["deployed_path_profile"]
    assert profile["environment"] == "same-pinned-four-cpu-candidate-environment"
    assert profile["profiled_model"] == "exp658-selected-reloaded-challenger-only"
    assert profile["candidate005_role"] == (
        "path-validated-cache-or-pinned-full1100-incumbent-measurement-only"
    )
    assert profile["width"] == 256
    assert profile["warmup_batch_cases"] == 4
    assert profile["fresh_repeat_batches"] == 3
    assert profile["fresh_cases_per_batch"] == 8
    assert profile["cross_environment_addition"] == "prohibited"
    cached = profile["cached_candidate005"]
    assert cached["incumbent_wall_seconds"] == 156.0
    assert cached["incremental_allowance_seconds"] == 129.0
    assert cached["submission_weights"] == {
        "path": "submission/weights.safetensors",
        "sha256": "2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca",
    }
    assert cached["sandbox_record"] == {
        "path": "experiments/runs/candidate-005-tier6-conditional-commutative-integration/sandbox-run.json",
        "sha256": "bcf82f8bf3135a6857b0e30f01a60fbec62beeaa1e45e1f029aee39c9041a6fa",
    }
    assert cached["exact_environment_identity"] == {
        "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
        "dockerfile_sha256": "1e04249cf1b8410336f11b9509e918aaf8e9dbf5117d0a972ddd5e16d68d0b41",
        "manifest_sha256": "ee49f91b8a1a4aa2cde418ebec59ea1890d93fda6dc72338989c0aa3308e7c83",
        "model_sha256": "e3d4aa9ff3e22dde516d261b099826622fa37d0688fae47c07ff95fe876e9477",
        "engine_version_sha256": "3931f1f1919b7f3e27ca7a7c8c3bfeb89ee7f9e65a85d4a6b79b2ee8f003aa92",
        "image_id": "0ff410bfb61c61d6a07b17b11337694a69910874cb59ab109b1128b3224a8423",
        "cpus": "4",
        "memory": "8g",
        "tmpfs_size": "2g",
        "timeout_seconds": 300,
    }
    assert cached["reuse_requires"] == (
        "exact-artifact-sandbox-record-and-environment-identity-match"
    )
    assert cached["invalidation"] == (
        "cached-I-and-A-unavailable-require-fresh-same-environment-incumbent-measurement"
    )
    budget = l1["budget"]
    assert budget["soft_wall_and_aggregate_cpu_multiplier"] == 1.25
    assert budget["hard_runaway_wall_and_aggregate_cpu_multiplier"] == 1.75
    assert budget["peak_rss_guard_multiplier"] == 1.25
    runtime = l1["runtime_feasibility"]
    assert runtime["incumbent_symbol"] == "I_same_environment"
    assert runtime["candidate_promotion_ceiling_wall_seconds"] == 285.0
    assert runtime["incremental_allowance_symbol"] == "A"
    assert runtime["incremental_allowance_formula"] == "A=285.0-I_same_environment"
    assert runtime["incumbent_source"] == (
        "fresh-same-environment-measurement-or-exact-identity-validated-cache"
    )
    assert runtime["cached_identity_matched_values"] == {
        "I_same_environment": 156.0,
        "A": 129.0,
    }
    assert runtime["cache_invalid_behavior"] == (
        "I_same_environment-and-A-unavailable-L1-runtime-gate-fails-closed"
    )
    assert runtime["pass_if"] == "I_same_environment-available-and-U100<=A"
    assert runtime["reject_if"] == (
        "I_same_environment-unavailable-or-100*min(fresh_challenger_profile_wall_seconds/8)>A"
    )
    assert runtime["straddle_action"] == "profile-one-exact-100-case-deployed-batch"
    assert runtime["straddle_pass_if"] == (
        "I_same_environment-available-and-1.10*measured_100_case_challenger_wall_seconds<=A"
    )
    assert runtime["aggregate_cpu_substituted_for_wall"] == "prohibited"


def test_config_restores_exact_t017_l2_denominators_gates_and_futility() -> None:
    l2 = config()["L2"]
    assert l2["training"] == {
        "arms": ["control", "challenger"],
        "updates_per_arm": 400,
        "batch_size": 128,
        "widths": [32, 64],
        "width_order": "alternating",
        "train_test_identities_and_batches": "exact-match",
        "checkpoint_and_sha256_before_evaluation": "required",
    }
    assert l2["denominators_per_arm"] == {
        "natural_transitions_per_width": 20000,
        "natural_rollouts_per_width": 1000,
        "examples_per_quotient_class_per_width": 2000,
        "labeled_positions_per_signed_carry_class_per_width": 2000,
        "challenger_zero-quotient-message_cases_per_width": 500,
        "challenger_zero-signed-carry-message_cases_per_width": 500,
        "causal_cases_paired_to": "first-500-normal-rollouts",
        "deterministic_normal_rollouts_per_width": 100,
        "deterministic_repeats": 2,
    }
    assert l2["rare_recall_contract"] == {
        "binding_arm": "challenger",
        "control_arm": "same-denominator-nonbinding-diagnostic",
        "same_denominators": "required",
    }
    gates = l2["gates"]
    assert gates["quotient_recall_each_class_width"] == {
        "binding_arm": "challenger",
        "control_reporting": "same-denominator-nonbinding-diagnostic",
        "denominator": 2000,
        "maximum_errors": 2,
        "minimum_fraction": 0.999,
    }
    assert gates["signed_carry_recall_each_class_width"] == {
        "binding_arm": "challenger",
        "control_reporting": "same-denominator-nonbinding-diagnostic",
        "denominator": 2000,
        "maximum_errors": 2,
        "minimum_fraction": 0.999,
    }
    assert gates["width64_natural_transition"] == {
        "denominator": 20000,
        "maximum_errors": 8,
        "minimum_fraction": 0.9996,
    }
    assert gates["width64_paired_rollout_gain"] == {
        "denominator": 1000,
        "minimum_net_correct": 30,
    }
    assert gates["width32_paired_rollout_retention"] == {
        "denominator": 1000,
        "minimum_net_correct": -10,
    }
    for key in (
        "zero-quotient-message-causality-each-width",
        "zero-signed-carry-message-causality-each-width",
    ):
        assert gates[key] == {
            "denominator": 500,
            "minimum_net_lost_correct": 10,
            "maximum_one_sided_exact_mcnemar_p": 0.01,
        }
    assert gates["validity_each_width"] == {
        "denominator": 1000,
        "maximum_invalid_rollouts": 10,
    }
    assert gates["determinism"] == "bit-exact-both-100-case-repeats"
    assert gates["integrity"] == {
        "artifact_hashes": "exact",
        "initialization_parameter_values": "exact",
        "post_update_parameter_structure": "exact-keys-shapes-dtypes-devices-trainable-masks-topology-LR-scheduler-and-step-counts",
        "post_update_parameter_values": "expected-to-diverge-under-auxiliary-supervision",
        "checkpoint_reload": "exact",
    }
    assert gates["resources"] == {
        "rss_wall_and_aggregate_cpu": "within-L1-calibrated-gates"
    }
    assert l2["futility"] == {
        "class_diagnostic": "stop-immediately-on-third-error",
        "width64_transition": "stop-immediately-on-ninth-error",
        "paired_rollout_block_size": 100,
        "width64_gain_rule": "stop-if-(current_net_gain+1000-cases_seen)<30",
        "width32_retention_rule": "stop-if-(current_net_gain+1000-cases_seen)<-10",
        "causal_rule": "stop-if-(current_net_damage+500-cases_seen)<10",
        "validity_rule": "stop-immediately-on-eleventh-invalid-rollout",
        "training_loss_early_stopping": "prohibited",
    }


def test_config_restores_exact_t017_l3_serial_roots_gates_and_futility() -> None:
    l3 = config()["L3"]
    assert l3["roots"] == {
        "count": 2,
        "identities": "unassigned",
        "ordering": "serial",
        "root_two_launch": "prohibited-if-root-one-fails-any-conjunctive-gate",
    }
    assert l3["training_per_root"] == {
        "arms": ["control", "challenger"],
        "updates_per_arm": 400,
        "batch_size": 128,
        "widths": [128, 256],
        "architecture_and_loss_contract": "same-resolved-L2-contract",
    }
    assert l3["denominators_per_root"] == {
        "width256_transitions_per_arm": 50000,
        "width256_rollouts_per_arm": 1000,
        "width128_retention_rollouts_per_arm": 500,
        "examples_per_quotient_class": 2000,
        "labeled_positions_per_signed_carry_class": 2000,
        "challenger_zero-quotient-message_cases": 500,
        "challenger_zero-signed-carry-message_cases": 500,
        "causal_cases": "paired",
        "deterministic_challenger_rollouts": 100,
        "deterministic_repeats": 2,
        "deployed_runtime_tier7_cases": 100,
    }
    assert l3["rare_recall_contract"] == {
        "binding_arm": "challenger",
        "control_arm": "same-denominator-nonbinding-diagnostic",
        "same_denominators": "required",
    }
    gates = l3["gates_per_root"]
    assert gates["width256_transition"] == {
        "denominator": 50000,
        "maximum_errors": 20,
        "minimum_fraction": 0.9996,
    }
    assert gates["width256_paired_rollout_gain"] == {
        "denominator": 1000,
        "minimum_net_correct": 30,
    }
    assert gates["width128_paired_rollout_retention"] == {
        "denominator": 500,
        "minimum_net_correct": -5,
    }
    for key in ("quotient_recall_each_class", "signed_carry_recall_each_class"):
        assert gates[key] == {
            "binding_arm": "challenger",
            "control_reporting": "same-denominator-nonbinding-diagnostic",
            "denominator": 2000,
            "maximum_errors": 2,
            "minimum_fraction": 0.999,
        }
    for key in (
        "zero-quotient-message-causality",
        "zero-signed-carry-message-causality",
    ):
        assert gates[key] == {
            "denominator": 500,
            "minimum_net_lost_correct": 10,
            "maximum_one_sided_exact_mcnemar_p": 0.01,
        }
    assert gates["validity"] == {
        "denominator": 1000,
        "maximum_invalid_width256_rollouts": 10,
    }
    assert gates["determinism"] == "exact-100-case-repeat"
    assert gates["runtime"] == {
        "environment": "same-pinned-candidate-environment",
        "incumbent_wall_seconds": "I_same_environment",
        "incremental_allowance_formula": "A=285.0-I_same_environment",
        "combined_wall_ceiling_seconds": 285.0,
        "declared_headroom_required": True,
        "aggregate_cpu_and_rss": "within-L1-calibrated-gates",
        "cached_values_reuse": (
            "exact-L1-artifact-sandbox-record-and-environment-identity-match-only"
        ),
        "unavailable_incumbent_behavior": "fail-closed-no-root-promotion",
    }
    assert l3["aggregate_gate"] == {
        "width256_rollouts_across_roots": 2000,
        "minimum_challenger_control_net_correct_gain": 60,
        "every_per_root_gate": "required",
    }
    assert l3["futility"] == {
        "class_diagnostic": "stop-immediately-on-third-error",
        "width256_transition": "stop-immediately-on-twenty-first-error",
        "paired_rollouts": "same-exact-attainable-count-bounds-as-L2",
        "causal_interventions": "same-exact-attainable-count-bounds-as-L2",
        "validity": "stop-immediately-on-eleventh-invalid-rollout",
        "root_one_failure": "stop-before-root-two",
        "training_loss_early_stopping": "prohibited",
    }


def test_future_protocol_uses_only_implemented_zero_message_interventions() -> None:
    assert model_module.JointQuotientCarryTransducer.VALID_MESSAGE_ABLATIONS == {
        "none",
        "zero-quotient-message",
        "zero-signed-carry-message",
    }
    payload = config()
    assert set(payload["L1"]["challenger_causal_rollouts_per_width_repeat"]) == {
        "zero-quotient-message",
        "zero-signed-carry-message",
    }
    assert {
        key for key in payload["L2"]["gates"] if "causality" in key
    } == {
        "zero-quotient-message-causality-each-width",
        "zero-signed-carry-message-causality-each-width",
    }
    assert {
        key for key in payload["L3"]["gates_per_root"] if "causality" in key
    } == {
        "zero-quotient-message-causality",
        "zero-signed-carry-message-causality",
    }
    forbidden_fragments = (
        "posterior" + "-cycle",
        "posterior" + "_cycle",
        "quotient" + "-cycle",
        "quotient" + "_cycle",
        "carry" + "-cycle",
        "carry" + "_cycle",
    )
    for relative_path in ALLOWED_PATHS:
        source = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert not any(fragment in source for fragment in forbidden_fragments)


def test_exp609_checkpoint_exists_and_has_exact_source_geometry() -> None:
    payload = config()
    checkpoint = ROOT / payload["source"]["checkpoint"]
    assert checkpoint.is_file()
    assert sha256(checkpoint) == EXPECTED_SOURCE_SHA256
    state = load_file(str(checkpoint))
    assert sum(value.numel() for value in state.values()) == SOURCE_PARAMETER_COUNT
    assert state["gate_weight"].shape == (184, 98, 3)
    assert state["candidate_weight"].shape == (92, 98, 3)
    assert state["quotient_head.2.weight"].shape == (3, 92)


def test_exhaustive_prime_domain_proves_exact_quotient_and_signed_carry_recurrence() -> None:
    quotient_vocabulary: set[int] = set()
    carry_vocabulary: set[int] = set()
    count = 0
    for transition in data_module.iter_exhaustive_small_prime_transitions():
        trace = data_module.quotient_carry_trace(
            transition, transition.modulus.bit_length()
        )
        count += 1
        quotient_vocabulary.add(trace.quotient)
        carry_vocabulary.update(trace.carry_in)
        carry_vocabulary.update(trace.carry_out)
        assert trace.target == (
            2 * transition.state
            + transition.operand_bit * transition.multiplicand
        ) % transition.modulus
        assert trace.terminal_carry == 0
        assert all(
            (numerator - target_bit) % 2 == 0
            for numerator, target_bit in zip(
                trace.recurrence_numerators, trace.target_bits, strict=True
            )
        )
        assert all(value in data_module.SIGNED_CARRY_CLASSES for value in trace.carry_out)
    assert count == 1991546
    assert quotient_vocabulary == {0, 1, 2}
    assert carry_vocabulary == {-2, -1, 0, 1}


@pytest.mark.parametrize("width,seed", [(32, 2026118032), (64, 2026118064)])
def test_one_hundred_thousand_fresh_legal_transitions_per_width(
    width: int, seed: int
) -> None:
    quotient_vocabulary: set[int] = set()
    carry_vocabulary: set[int] = set()
    count = 0
    for transition in data_module.iter_random_legal_transitions(
        width=width, count=100000, seed=seed
    ):
        assert transition.modulus.bit_length() == width
        assert data_module.is_prime(transition.modulus)
        trace = data_module.quotient_carry_trace(transition, width)
        count += 1
        quotient_vocabulary.add(trace.quotient)
        carry_vocabulary.update(trace.carry_in)
        carry_vocabulary.update(trace.carry_out)
        assert trace.quotient in data_module.QUOTIENT_CLASSES
        assert trace.terminal_carry == 0
        assert all(
            (numerator - target_bit) % 2 == 0
            for numerator, target_bit in zip(
                trace.recurrence_numerators, trace.target_bits, strict=True
            )
        )
    assert count == 100000
    assert quotient_vocabulary == {0, 1, 2}
    assert carry_vocabulary == {-2, -1, 0, 1}


def test_split_convolution_geometry_parameter_count_and_zero_initialization() -> None:
    torch.manual_seed(658)
    model = model_module.JointQuotientCarryTransducer()
    assert model.gate_weight.shape == (184, 98, 3)
    assert model.candidate_weight.shape == (92, 98, 3)
    assert model.message_gate_weight.shape == (184, 4, 3)
    assert model.message_candidate_weight.shape == (92, 4, 3)
    assert model.quotient_message_embedding.shape == (3, 2)
    assert model.signed_carry_message_embedding.shape == (4, 2)
    assert model.signed_carry_head.weight.shape == (4, 92, 1)
    assert torch.count_nonzero(model.message_gate_weight) == 0
    assert torch.count_nonzero(model.message_candidate_weight) == 0
    assert model.parameter_count() == EXPECTED_PARAMETER_COUNT
    assert model.parameter_count() / SOURCE_PARAMETER_COUNT <= 1.05


def test_control_and_challenger_have_exactly_identical_state_dictionaries() -> None:
    control, challenger = run_module.build_matched_arms(config())
    assert control.parameter_count() == challenger.parameter_count() == EXPECTED_PARAMETER_COUNT
    assert tuple(control.state_dict()) == tuple(challenger.state_dict())
    for name in control.state_dict():
        left = control.state_dict()[name]
        right = challenger.state_dict()[name]
        assert left.shape == right.shape, name
        assert torch.equal(left, right), name


@pytest.mark.parametrize("width,seed", [(32, 2026118132), (64, 2026118164)])
def test_exact_exp609_source_logit_parity_on_4096_legal_transitions(
    width: int, seed: int
) -> None:
    payload = config()
    checkpoint = ROOT / payload["source"]["checkpoint"]
    source = source_module.SharedDilatedConvGRU(width=128, hidden=92, cycles=2)
    source.load_state_dict(load_file(str(checkpoint)), strict=True)
    challenger = model_module.JointQuotientCarryTransducer()
    challenger.load_exp609_source(checkpoint)
    source.eval()
    challenger.eval()
    transitions = tuple(
        data_module.iter_random_legal_transitions(
            width=width, count=4096, seed=seed
        )
    )
    maximum_residue_difference = 0.0
    maximum_quotient_difference = 0.0
    compared = 0
    with torch.inference_mode():
        for start in range(0, len(transitions), 64):
            features = data_module.feature_tensor(transitions[start : start + 64], width)
            source_residue, source_quotient = source(features)
            output = challenger(features)
            residue_difference = float(
                (source_residue - output.residue_logits).abs().max()
            )
            quotient_difference = float(
                (source_quotient - output.quotient_logits).abs().max()
            )
            maximum_residue_difference = max(
                maximum_residue_difference, residue_difference
            )
            maximum_quotient_difference = max(
                maximum_quotient_difference, quotient_difference
            )
            assert torch.equal(
                source_residue.ge(0), output.residue_logits.ge(0)
            )
            compared += len(features)
    assert compared == 4096
    assert maximum_residue_difference == 0.0
    assert maximum_quotient_difference == 0.0


def _inference_tree() -> ast.Module:
    methods = (
        model_module.JointQuotientCarryTransducer.dilation_schedule,
        model_module.JointQuotientCarryTransducer._message_posteriors,
        model_module.JointQuotientCarryTransducer._soft_messages,
        model_module.JointQuotientCarryTransducer.recurrent_update,
        model_module.JointQuotientCarryTransducer.forward,
    )
    source = "\n".join(inspect.getsource(method) for method in methods)
    return ast.parse(inspect.cleandoc(source))


def test_inference_scoped_ast_rejects_exact_or_selected_message_execution() -> None:
    tree = _inference_tree()
    forbidden_calls = {
        "argmax",
        "argmin",
        "gather",
        "topk",
        "where",
        "round",
        "int",
        "long",
        "item",
        "tolist",
        "remainder",
        "fmod",
        "div",
        "floor_divide",
        "decode",
        "decode_bits",
        "bits_to_int",
        "bits_to_integer",
    }
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert not calls.intersection(forbidden_calls)
    forbidden_binary = (
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.LShift,
        ast.RShift,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
    )
    assert not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, forbidden_binary)
        for node in ast.walk(tree)
    )
    source = ast.unparse(tree).lower()
    for forbidden_text in (
        "teacher",
        "target",
        "expert",
        "candidate selection",
        "quotient * modulus",
        "carry propagation",
    ):
        assert forbidden_text not in source
    for branch in (node for node in ast.walk(tree) if isinstance(node, ast.If)):
        assert isinstance(branch.test, (ast.Compare, ast.UnaryOp))
        assert "posterior" not in ast.unparse(branch.test).lower()
        assert "logit" not in ast.unparse(branch.test).lower()


def test_soft_messages_are_posterior_weighted_learned_channels_only() -> None:
    torch.manual_seed(658)
    model = model_module.JointQuotientCarryTransducer().eval()
    state = torch.randn(3, 92, 32)
    quotient_posterior, carry_posterior = model._message_posteriors(state)
    quotient_message, carry_message = model._soft_messages(state, "none")
    assert quotient_posterior.shape == (3, 3)
    assert carry_posterior.shape == (3, 4, 32)
    assert torch.allclose(quotient_posterior.sum(1), torch.ones(3))
    assert torch.allclose(carry_posterior.sum(1), torch.ones(3, 32))
    assert quotient_message.shape == (3, 2, 32)
    assert carry_message.shape == (3, 2, 32)
    assert torch.equal(
        model._soft_messages(state, "zero-quotient-message")[0],
        torch.zeros_like(quotient_message),
    )
    assert torch.equal(
        model._soft_messages(state, "zero-signed-carry-message")[1],
        torch.zeros_like(carry_message),
    )


def test_future_runner_is_fail_closed_and_post_t045_park_boundary_is_intact() -> None:
    payload = config()
    assert run_module.validate_l0_contract(payload).is_file()
    with pytest.raises(run_module.L0OnlyError, match="L0 only"):
        run_module.require_future_authorization()

    run_root = ROOT / "experiments/runs/exp-658-joint-quotient-carry-transducer"
    eval_root = ROOT / "evals/results/exp-658-joint-quotient-carry-transducer"
    actual = {
        str(path.relative_to(ROOT))
        for parent in (run_root, eval_root)
        for path in parent.rglob("*")
        if path.is_file()
    }
    assert actual == set(T045_RECORDS) | {str(POST_T045_AMENDMENT.relative_to(ROOT))}
    for relative, expected in T045_RECORDS.items():
        assert sha256(ROOT / relative) == expected

    run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    failure = json.loads((run_root / "failure.json").read_text(encoding="utf-8"))
    decision = json.loads((eval_root / "decision.json").read_text(encoding="utf-8"))
    assert run["decision"] == "park"
    assert run["metrics"] is None and run["artifacts"]["metrics"] == ""
    assert run["evidence"]["l1_scientific_evidence_persisted"] is False
    assert run["evidence"]["l2_executed"] is False
    assert run["evidence"]["mlflow_run_id"] is None
    assert run["evidence"]["candidate_005_changed"] is False
    assert failure["boundary"] == {
        "candidate_005_changed": False,
        "checked_evaluation_executed": False,
        "l1_calibration_persisted": False,
        "l1_scientific_values_persisted": False,
        "l2_executed": False,
        "mlflow_run_id": None,
        "official_evaluation_executed": False,
        "private_evaluation_executed": False,
        "scientific_metrics": None,
    }
    assert decision["scientific_metrics"] is None
    assert decision["goal_completion"] is False
    assert decision["candidate_005"]["changed"] is False
    assert not (run_root / "authorization.json").exists()
    assert not (run_root / "l1-checkpoints").exists()

    amendment = json.loads(POST_T045_AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["schema_version"] == "exp658-post-t045-provenance-amendment-v1"
    assert amendment["historical_l0_test_sha256"] == HISTORICAL_L0_TEST_SHA256
    assert amendment["accepted_t045_records"] == T045_RECORDS
    assert amendment["boundary"] == {
        "live_authorization": False,
        "scientific_metrics": False,
        "mlflow_run": False,
        "promotion": False,
        "candidate_change": False,
    }
    cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not set(cached).intersection(ALLOWED_PATHS)
