from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT / "experiments/exp_647_operand_swap_consistency/run.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("exp647_test_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scientific_hash_ignores_only_resolved_resource_budget() -> None:
    module = load_runner()
    config = yaml.safe_load(
        (ROOT / "configs/exp_647_operand_swap_consistency.yaml").read_text()
    )
    first = module.scientific_config_sha256(config)
    config["resource_budget"]["calibration_status"] = "complete"
    config["resource_budget"]["resolved"] = {"soft_budget_wall_seconds": 123}
    assert module.scientific_config_sha256(config) == first
    config["training"]["steps"] += 1
    assert module.scientific_config_sha256(config) != first


def test_confidence_teacher_routes_gradient_only_to_lower_margin_endpoint() -> None:
    module = load_runner()
    original = torch.tensor([[0.2, -0.4], [0.3, -0.6]], requires_grad=True)
    swapped = torch.tensor([[1.0, -1.2], [0.1, -0.2]], requires_grad=True)
    loss, swapped_is_teacher = module.confidence_teacher_loss(
        original,
        swapped,
        torch.tensor([0.1, 0.9]),
        torch.tensor([0.8, 0.2]),
    )
    loss.backward()
    assert swapped_is_teacher.tolist() == [True, False]
    assert original.grad is not None and swapped.grad is not None
    assert float(original.grad[0].abs().sum()) > 0
    assert float(swapped.grad[0].abs().sum()) == 0
    assert float(original.grad[1].abs().sum()) == 0
    assert float(swapped.grad[1].abs().sum()) > 0


def test_paired_rollout_summary_uses_frozen_margin_tie_break() -> None:
    module = load_runner()
    result = module.paired_rollout_summary(
        original_predictions=[7, 2, 5, 8],
        swapped_predictions=[7, 3, 4, 9],
        original_margins=[0.4, 0.2, 0.8, 0.5],
        swapped_margins=[0.4, 0.7, 0.1, 0.6],
        targets=[7, 3, 5, 8],
    )
    assert result["use_swapped"] == [False, True, False, True]
    assert result["intersection_correct"] == 1
    assert result["original_only_correct"] == 2
    assert result["swapped_only_correct"] == 1
    assert result["oracle_union_exact_accuracy"] == 1.0
    assert result["selected_correct"] == 3


def test_budget_projection_distinguishes_wall_and_aggregate_cpu() -> None:
    module = load_runner()
    config = yaml.safe_load(
        (ROOT / "configs/exp_647_operand_swap_consistency.yaml").read_text()
    )

    def arm_record():
        return {
            "training": {
                "per_width_timing": {
                    str(width): {
                        "updates": 1,
                        "wall_seconds": width / 64,
                        "aggregate_cpu_seconds": width / 16,
                    }
                    for width in (64, 128, 256)
                }
            },
            "evaluation": {
                str(width): {
                    "transition": {
                        "timing": {
                            "wall_seconds": 1.0,
                            "aggregate_cpu_seconds": 4.0,
                        }
                    },
                    "original": {
                        "timing": {
                            "wall_seconds": 2.0,
                            "aggregate_cpu_seconds": 8.0,
                        }
                    },
                    "swapped": {
                        "timing": {
                            "wall_seconds": 2.0,
                            "aggregate_cpu_seconds": 8.0,
                        }
                    },
                }
                for width in (64, 128, 256)
            },
        }

    resolved = module.resolve_budget(
        config,
        {
            "paired_rollout_control": arm_record(),
            "confidence_consistency": arm_record(),
        },
    )
    assert resolved["formal_update_counts_by_width_per_arm"] == {
        "64": 60,
        "128": 120,
        "256": 60,
    }
    assert (
        resolved["projected_formal_aggregate_cpu_seconds"]
        > resolved["projected_formal_wall_seconds"]
    )
    assert (
        resolved["hard_runaway_ceiling_wall_seconds"]
        > resolved["soft_budget_wall_seconds"]
    )
    assert resolved["fidelity_reduced"] is False


def test_paired_arm_comparison_reports_direction_and_exact_p_value() -> None:
    module = load_runner()
    comparison = module.paired_arm_comparison(
        control_predictions=[1, 0, 3, 0],
        challenger_predictions=[1, 2, 0, 4],
        targets=[1, 2, 3, 4],
    )
    assert comparison["both_correct"] == 1
    assert comparison["control_only_correct"] == 1
    assert comparison["challenger_only_correct"] == 2
    assert comparison["neither_correct"] == 0
    assert 0 <= comparison["exact_two_sided_mcnemar_p"] <= 1
