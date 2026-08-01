from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_655_layer_loop_compute_matched_recurrence.fixtures import (  # noqa: E402
    FIXTURES,
    verified_copy,
)
from experiments.exp_655_layer_loop_compute_matched_recurrence.model import (  # noqa: E402
    ARM_ORDER,
    PRIMARY_ARMS,
    LayerLoopReasoner,
    state_bytes,
)
from experiments.exp_655_layer_loop_compute_matched_recurrence.run import (  # noqa: E402
    Experiment,
    bits,
    features,
    hierarchical_paired_bootstrap,
)


CONFIG = ROOT / "configs/exp_655_layer_loop_compute_matched_recurrence.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def model(arm: str) -> LayerLoopReasoner:
    torch.manual_seed(6557)
    return LayerLoopReasoner(
        arm, hidden=16, heads=4, stored_blocks=2, expansion=2
    ).eval()


def test_frozen_roots_curriculum_denominators_and_scope() -> None:
    payload = config()
    assert payload["arms"] == list(ARM_ORDER)
    assert payload["primary_arms"] == list(PRIMARY_ARMS)
    assert payload["training"]["stages"] == [
        {"width": 8, "steps": 800},
        {"width": 12, "steps": 1200},
        {"width": 16, "steps": 2000},
    ]
    assert payload["training"]["total_steps_per_arm"] == 4000
    assert payload["evaluation"]["rollout_cases"] == {16: 300, 32: 300}
    assert payload["evaluation"]["bootstrap_resamples"] == 100000
    assert payload["evaluation"]["bootstrap_seeds"] == {
        16: 65595002,
        32: 65595001,
    }
    assert payload["data"]["calibration_root"] == 20261178
    assert payload["data"]["formal_roots"] == [
        {
            "root": 20261179,
            "init_seed": 65570001,
            "train_seed": 65573001,
            "eval_seed": 65574001,
        },
        {
            "root": 20261180,
            "init_seed": 65570002,
            "train_seed": 65573002,
            "eval_seed": 65574002,
        },
        {
            "root": 20261181,
            "init_seed": 65570003,
            "train_seed": 65573003,
            "eval_seed": 65574003,
        },
    ]
    assert payload["resource_budget"]["fidelity_reduced"] is False
    assert payload["source_contract"]["checked_prior_independent_official_private"] == "prohibited"


def test_exact_schedule_geometry_and_complete_block_calls() -> None:
    layer = model("layer_loop_r2")
    whole = model("model_loop_r2")
    r1 = model("r1")
    untied = model("untied_equal_compute")
    assert layer.stored_blocks == 2
    assert layer.schedule() == (0, 0, 1, 1)
    assert whole.schedule() == (0, 1, 0, 1)
    assert r1.schedule() == (0, 1)
    assert untied.schedule() == (0, 1, 2, 3)
    assert layer.complete_block_calls() == whole.complete_block_calls() == 4
    assert untied.complete_block_calls() == 4
    assert r1.complete_block_calls() == 2
    assert untied.parameter_count() > layer.parameter_count()


def test_primary_initial_state_is_byte_identical_and_only_schedule_differs() -> None:
    torch.manual_seed(65570001)
    layer = LayerLoopReasoner(
        "layer_loop_r2", hidden=16, heads=4, stored_blocks=2, expansion=2
    )
    state = {key: value.detach().clone() for key, value in layer.state_dict().items()}
    whole = LayerLoopReasoner(
        "model_loop_r2", hidden=16, heads=4, stored_blocks=2, expansion=2
    )
    whole.load_state_dict(state)
    assert state_bytes(layer) == state_bytes(whole)
    assert layer.schedule() != whole.schedule()
    assert set(layer.state_dict()) == set(whole.state_dict())


def test_complete_blocks_noncommute_and_schedule_swap_is_causal() -> None:
    primary = model("layer_loop_r2")
    generator = torch.Generator().manual_seed(17)
    state = torch.randn(3, 9, primary.hidden, generator=generator)
    b1_b2 = primary.blocks[1](primary.blocks[0](state))
    b2_b1 = primary.blocks[0](primary.blocks[1](state))
    assert float((b1_b2 - b2_b1).abs().max()) > 1e-7
    tensor = features([0, 1, 2], [3, 5, 7], [251, 241, 239], [0, 1, 1], 8)
    layer_logits, _ = primary(tensor)
    whole_logits, _ = primary(tensor, schedule="model_loop_r2")
    assert not torch.equal(layer_logits, whole_logits)


@pytest.mark.parametrize("arm", ARM_ORDER)
@pytest.mark.parametrize("width", [8, 12])
def test_width_flexible_finite_forward(arm: str, width: int) -> None:
    candidate = model(arm)
    tensor = features([0, 1], [3, 5], [251, 241], [0, 1], width)
    logits, quotient = candidate(tensor)
    assert logits.shape == (2, width)
    assert quotient.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_malformed_inputs_and_intervention_bounds_fail_closed() -> None:
    candidate = model("layer_loop_r2")
    with pytest.raises(ValueError, match="shape"):
        candidate(torch.zeros(2, 8))
    with pytest.raises(ValueError, match="6 feature roles"):
        candidate(torch.zeros(2, 5, 8))
    malformed = torch.zeros(2, 6, 8)
    malformed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        candidate(malformed)
    with pytest.raises(ValueError, match="unavailable block"):
        candidate(torch.zeros(2, 6, 8), block_order=(0, 9))
    with pytest.raises(ValueError, match="nonnegative"):
        candidate(torch.zeros(2, 6, 8), visits=-1)


def test_tiny_training_is_deterministic() -> None:
    torch.use_deterministic_algorithms(True)
    inputs = features([0, 1, 2, 3], [3, 3, 5, 5], [251, 251, 241, 241], [0, 1, 0, 1], 8)
    targets = bits([0, 4, 4, 11], 8)
    snapshots = []
    predictions = []
    for _ in range(2):
        torch.manual_seed(65570001)
        candidate = LayerLoopReasoner(
            "layer_loop_r2", hidden=16, heads=4, stored_blocks=2, expansion=2
        )
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
        for _step in range(2):
            logits, _ = candidate(inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        snapshots.append(state_bytes(candidate))
        predictions.append(candidate(inputs)[0].detach())
    assert snapshots[0] == snapshots[1]
    assert torch.equal(predictions[0], predictions[1])


def test_exact_hierarchical_bootstrap_contract() -> None:
    deltas = np.zeros((3, 300), dtype=np.float64)
    deltas[:, :30] = 1.0
    result = hierarchical_paired_bootstrap(
        deltas, seed=65595001, resamples=100000
    )
    assert result["clusters"] == 3
    assert result["paired_identities_per_cluster"] == 300
    assert result["resamples"] == 100000
    assert result["mean"] == pytest.approx(0.1, abs=2e-3)
    assert len(result["distribution_sha256"]) == 64
    with pytest.raises(ValueError, match="100,000"):
        hierarchical_paired_bootstrap(deltas, seed=1, resamples=1000)


def test_regular_copy_materializer_is_idempotent_non_aliasing_and_hash_bound(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "nested/destination.bin"
    source.write_bytes(b"exp655-fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    first = verified_copy(source, destination, digest)
    second = verified_copy(source, destination, digest)
    assert first["reused"] is False
    assert second["reused"] is True
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino
    assert not destination.is_symlink()
    destination.unlink()
    destination.symlink_to(source)
    with pytest.raises(RuntimeError, match="symlink"):
        verified_copy(source, destination, digest)
    assert {fixture.name for fixture in FIXTURES} == {
        "exp609_checkpoint",
        "dockerfile",
        "generator",
        "primes",
    }


def test_gate_is_conjunctive_across_all_three_roots() -> None:
    experiment = object.__new__(Experiment)
    experiment.config = config()

    def arm(accuracy16: float, accuracy32: float) -> dict:
        return {
            "16": {
                "cases": 300,
                "exact_accuracy": accuracy16,
                "any_invalid_residue_rate": 0.0,
            },
            "32": {
                "cases": 300,
                "exact_accuracy": accuracy32,
                "any_invalid_residue_rate": 0.0,
            },
        }

    roots = []
    for _ in range(3):
        roots.append(
            {
                "evaluation": {
                    "layer_loop_r2": arm(0.82, 0.10),
                    "model_loop_r2": arm(0.82, 0.06),
                    "r1": arm(0.50, 0.01),
                    "untied_equal_compute": arm(0.70, 0.03),
                },
                "randomized_weight_dependence": {
                    value: {"trained_minus_randomized": 0.25}
                    for value in PRIMARY_ARMS
                },
                "deterministic_repeat": {
                    value: {"passed": True} for value in PRIMARY_ARMS
                },
                "structural_controls": {
                    "primary_initial_state_identical": True,
                    "noncommutation": {
                        value: {"noncommuting": True} for value in PRIMARY_ARMS
                    },
                    "malformed_and_leakage": {
                        "malformed_passed": True,
                        "leakage_passed": True,
                    },
                },
            }
        )
    bootstrap = {
        "16": {"one_sided_95_lower_bound": 0.0},
        "32": {"one_sided_95_lower_bound": 0.02},
    }
    resource_result = {
        "wall_seconds": 10.0,
        "process_cpu_seconds": 20.0,
        "cgroup_cpu_seconds": 21.0,
        "peak_rss_bytes": 100,
    }
    admission = {
        "hard_budget": {
            "wall_seconds": 20,
            "process_cpu_seconds": 40,
            "cgroup_cpu_seconds": 42,
            "peak_rss_bytes": 200,
        }
    }
    gates, comparison = experiment.gate_results(
        roots, bootstrap, resource_result, admission
    )
    assert all(gates.values())
    assert comparison["width32_unweighted_mean_delta"] == pytest.approx(0.04)
    roots[2]["evaluation"]["layer_loop_r2"]["16"]["exact_accuracy"] = 0.79
    gates, _ = experiment.gate_results(roots, bootstrap, resource_result, admission)
    assert not gates["primary_width16_each_root"]


def test_inference_module_has_no_solver_or_benchmark_access() -> None:
    source = inspect.getsource(LayerLoopReasoner).lower()
    for forbidden in (
        "expected_answer",
        "barrett",
        "montgomery",
        "crt_recomb",
        "torch.remainder",
        "fmod(",
        "data/checked",
        "data/private",
        "submission/",
    ):
        assert forbidden not in source
