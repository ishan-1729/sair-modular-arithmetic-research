from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_659_symbol_equivariant_recurrence.model import (  # noqa: E402
    SymbolRecurrentReasoner,
)
from experiments.exp_659_symbol_equivariant_recurrence.run import (  # noqa: E402
    Experiment,
    stage_sequence,
)


CONFIG = ROOT / "configs/exp_660_symbol_equivariant_formal.yaml"
ENTRYPOINT = ROOT / "experiments/exp_660_symbol_equivariant_formal/run.py"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_fresh_formal_identity_and_branch_are_frozen() -> None:
    config = load_config()
    assert config["run_id"] == "exp-660-symbol-equivariant-formal"
    assert config["data"]["formal_seed"] == config["seed"] == 20261177
    assert config["data"]["formal_seed_status"] == "unused-unopened-for-training"
    assert config["source_contract"] == {
        "predecessor": "exp-659-symbol-equivariant-recurrence",
        "predecessor_status": "parked-before-science",
        "required_branch": "codex/exp660-symbol-equivariant-formal",
        "entrypoint": "experiments/exp_660_symbol_equivariant_formal/run.py",
        "formal_initialization": "ordinary-pytorch-initialization-only",
        "calibration_checkpoint_reuse": "forbidden",
    }


def test_full_curriculum_and_formal_denominators_are_not_reduced() -> None:
    config = load_config()
    assert stage_sequence(config, "formal") == [(8, 800), (12, 1200), (16, 2000)]
    assert sum(steps for _, steps in stage_sequence(config, "formal")) == 4000
    assert config["training"]["total_steps_per_arm"] == 4000
    assert config["evaluation"]["rollout_cases"] == {16: 300, 24: 200, 32: 200}
    assert config["evaluation"]["transition_cases"] == {16: 300, 24: 300, 32: 300}
    assert config["resource_budget"]["fidelity_reduced"] is False


def test_calibration_is_resource_only_and_forbids_checkpoint_reuse() -> None:
    config = load_config()
    assert config["resource_budget"]["calibration_status"] == "pending"
    assert config["source_contract"]["calibration_checkpoint_reuse"] == "forbidden"
    assert config["resource_budget"]["reuse_completed_arms"] is True
    assert stage_sequence(config, "calibration") == [(8, 20), (12, 30), (16, 50)]
    for key in (
        "expected_formal_wall_seconds",
        "expected_formal_aggregate_cpu_seconds",
        "expected_formal_cgroup_cpu_seconds",
        "soft_budget_wall_seconds",
        "soft_budget_aggregate_cpu_seconds",
        "soft_budget_cgroup_cpu_seconds",
        "hard_budget_wall_seconds",
        "hard_budget_aggregate_cpu_seconds",
        "hard_budget_cgroup_cpu_seconds",
        "hard_peak_rss_bytes",
    ):
        assert config["resource_budget"][key] is None


def test_headroom_contract_distinguishes_wall_process_cgroup_and_rss() -> None:
    budget = load_config()["resource_budget"]
    assert budget["soft_multiplier"] == pytest.approx(1.25)
    assert budget["hard_multiplier"] == pytest.approx(1.75)
    assert budget["rss_multiplier"] == pytest.approx(1.25)
    prior = budget["prior_design_evidence"]
    assert prior["wall_seconds"] == pytest.approx(18.397680725996906)
    assert prior["aggregate_process_cpu_seconds"] == pytest.approx(47.2)
    assert prior["cgroup_cpu_seconds"] == pytest.approx(66.292)
    assert prior["peak_rss_bytes"] == 437198848


def test_exact_architecture_and_parameter_matching_are_preserved() -> None:
    config = load_config()
    values = config["model"]
    models = {
        arm: SymbolRecurrentReasoner(
            arm,
            hidden=values["hidden"],
            position_hidden=values["position_hidden"],
            heads=values["heads"],
            recurrent_steps=values["recurrent_steps"],
            expansion=values["expansion"],
        )
        for arm in config["arms"]
    }
    faithful = models["symbol_equivariant"].parameter_count()
    assert config["arms"] == ["position_only", "symbol_equivariant", "broken_tying"]
    assert 0.75 <= models["position_only"].parameter_count() / faithful <= 1.25
    assert 0.75 <= models["broken_tying"].parameter_count() / faithful <= 1.25


def test_both_binary_core_permutations_and_broken_tying_contrast() -> None:
    torch.manual_seed(20261177)
    faithful = SymbolRecurrentReasoner(
        "symbol_equivariant", hidden=16, position_hidden=20, heads=4
    ).eval()
    broken = SymbolRecurrentReasoner(
        "broken_tying", hidden=16, position_hidden=20, heads=4
    ).eval()
    state = torch.randn(3, 7, 2, 16)
    base = faithful.core_operator(state)
    for permutation in (torch.tensor([0, 1]), torch.tensor([1, 0])):
        assert torch.allclose(
            faithful.core_operator(state[:, :, permutation]),
            base[:, :, permutation],
            atol=1e-6,
            rtol=0.0,
        )
    swap = torch.tensor([1, 0])
    broken_residual = (
        broken.core_operator(state[:, :, swap])
        - broken.core_operator(state)[:, :, swap]
    ).abs().max()
    assert float(broken_residual) >= 1e-4


def test_primary_gate_is_conjunctive_and_uses_best_matched_control() -> None:
    experiment = object.__new__(Experiment)
    experiment.config = load_config()
    rollout = lambda accuracy, invalid=0.0: {  # noqa: E731
        "exact_accuracy": accuracy,
        "any_invalid_residue_rate": invalid,
    }
    arms = {
        "symbol_equivariant": {
            "parameters": 100,
            "evaluation": {
                "16": {"rollout": rollout(0.80)},
                "24": {"rollout": rollout(0.20)},
                "32": {"rollout": rollout(0.10)},
            },
        },
        "position_only": {
            "parameters": 100,
            "evaluation": {
                "16": {"rollout": rollout(0.10)},
                "24": {"rollout": rollout(0.05)},
                "32": {"rollout": rollout(0.04)},
            },
        },
        "broken_tying": {
            "parameters": 100,
            "evaluation": {
                "16": {"rollout": rollout(0.10)},
                "24": {"rollout": rollout(0.05)},
                "32": {"rollout": rollout(0.05)},
            },
        },
    }
    controls = {
        "core_equivariance": {
            "symbol_equivariant": {"permutations": {"swap": {"maximum_absolute_residual": 0.0}}},
            "broken_tying": {"permutations": {"swap": {"maximum_absolute_residual": 0.01}}},
        },
        "task_symmetry_applicability": {"verdict": "non-applicable-domain-not-closed"},
        "deterministic_replay": {"passed": True},
        "randomized_weight_dependence": {"exact_drop": 0.20},
        "static_leakage": {"passed": True},
    }
    resource = {
        "wall_seconds": 1.0,
        "aggregate_process_cpu_seconds": 1.0,
        "cgroup_cpu_seconds": 1.0,
        "peak_rss_bytes": 1,
    }
    budget = experiment.config["resource_budget"]
    budget.update(
        {
            "hard_budget_wall_seconds": 2,
            "hard_budget_aggregate_cpu_seconds": 2,
            "hard_budget_cgroup_cpu_seconds": 2,
            "hard_peak_rss_bytes": 2,
        }
    )
    gates, comparison = experiment.gate_results(arms, controls, resource)
    assert comparison["best_control_width32"] == pytest.approx(0.05)
    assert comparison["width32_gain"] == pytest.approx(0.05)
    assert all(gates.values())
    arms["symbol_equivariant"]["evaluation"]["32"]["rollout"]["exact_accuracy"] = 0.099
    gates, _ = experiment.gate_results(arms, controls, resource)
    assert not gates["width32_gain"]


def test_entrypoint_and_inference_source_have_no_forbidden_solver() -> None:
    source = (
        ENTRYPOINT.read_text(encoding="utf-8")
        + inspect.getsource(SymbolRecurrentReasoner)
    ).lower()
    for forbidden in (
        "expected_answer",
        "barrett",
        "montgomery",
        "crt_recomb",
        "torch.remainder",
        "fmod(",
        "data/checked",
        "data/private",
    ):
        assert forbidden not in source
