from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continuous_state_plane_is_bounded_and_sign_preserving() -> None:
    runner = load_module(
        "exp638_state_plane_test",
        ROOT / "experiments/exp_638_full_depth_soft_boundary_diagnostic/run.py",
    )
    logits = torch.tensor([[-100.0, -1.0, 0.0, 1.0, 100.0]])
    plane = runner.continuous_state_plane(logits, 1.0)
    assert plane.shape == logits.shape
    assert torch.all(plane >= -1.0) and torch.all(plane <= 1.0)
    assert plane[0, 0] < plane[0, 1] < plane[0, 2] < plane[0, 3] < plane[0, 4]
    assert plane[0, 2] == 0.0


def test_replace_state_plane_preserves_all_context_planes() -> None:
    runner = load_module(
        "exp638_replace_plane_test",
        ROOT / "experiments/exp_638_full_depth_soft_boundary_diagnostic/run.py",
    )
    features = torch.randn(3, 6, 8)
    state = torch.randn(3, 8).clamp(-1.0, 1.0)
    replaced = runner.replace_state_plane(features, state)
    assert torch.equal(replaced[:, 0], state)
    assert torch.equal(replaced[:, 1:], features[:, 1:])


def test_resource_budget_uses_distinct_wall_and_cpu_fields() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/exp_638_full_depth_soft_boundary_diagnostic.yaml").read_text()
    )
    budget = config["resource_budget"]
    assert budget["limit_unit"] == "wall_seconds"
    assert budget["expected_total_wall_seconds"] < budget["soft_budget_wall_seconds"]
    assert budget["soft_budget_wall_seconds"] < budget["hard_runaway_ceiling_wall_seconds"]
    assert budget["expected_total_aggregate_cpu_minutes"] > 0
    assert budget["hard_ceiling_aggregate_cpu_minutes_upper_bound"] > budget[
        "expected_total_aggregate_cpu_minutes"
    ]
