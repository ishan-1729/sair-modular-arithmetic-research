from experiments.exp_635_independent_cellular_diversity_feasibility.run import (
    paired_comparison,
    wilson_interval,
)


def test_paired_comparison_uses_only_frozen_margin_rule():
    control = {
        "expected": [1, 4, 7],
        "predictions": [1, 2, 7],
        "correctness": [True, False, True],
        "minimum_margin": [0.8, 0.1, 0.6],
        "exact_accuracy": 2 / 3,
    }
    challenger = {
        "expected": [1, 4, 7],
        "predictions": [3, 4, 0],
        "correctness": [False, True, False],
        "minimum_margin": [0.2, 0.7, 0.5],
        "exact_accuracy": 1 / 3,
    }
    result = paired_comparison(control, challenger)
    assert result["oracle_union_correct"] == 3
    assert result["routed_correct"] == 3
    assert result["challenger_route_count"] == 1
    assert result["control_only"] == 2
    assert result["challenger_only"] == 1


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(86, 100)
    assert low < 0.86 < high
