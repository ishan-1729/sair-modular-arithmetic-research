from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_routing():
    path = ROOT / "experiments/exp_625_tier6_confidence_routed_width4_search/routing.py"
    spec = importlib.util.spec_from_file_location("confidence_routing_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ranked_indices_break_ties_by_stable_row_index() -> None:
    routing = load_routing()
    assert routing.ranked_indices([0.5, 0.7, 0.7, 0.1], 2) == [1, 2]


def test_oracle_allocation_prefers_recoveries_and_reports_losses() -> None:
    routing = load_routing()
    greedy = [False, True, True, False]
    beam = [True, False, True, False]
    selected = routing.oracle_allocation(greedy, beam, 2)
    assert selected == [0, 2]
    result = routing.routed_coverage(greedy, beam, selected)
    assert result == {
        "correct": 3,
        "coverage": 0.75,
        "routed": 2,
        "routed_greedy_errors": 1,
        "routed_beam_recoveries": 1,
        "routed_greedy_losses": 0,
    }
