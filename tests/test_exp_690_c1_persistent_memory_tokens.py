from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.exp_690_c1_persistent_memory_tokens import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_690_c1_persistent_memory_tokens.yaml").read_text())


def model() -> run.PersistentMemoryStudent:
    state, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    return run.new_student(CONFIG, state, 690900).eval()


def test_frozen_protocol_and_all_seven_tournament_roots() -> None:
    assert CONFIG["roots"] == {"calibration": 20261314, "formal": 20261315,
                               "confirmations": [20261316, 20261317],
                               "independent_seeds": [20261318, 20261319, 20261320, 20261321, 20261322]}
    assert CONFIG["model"]["hidden"] == 48 and CONFIG["model"]["cycles"] == 1
    assert CONFIG["model"]["outward_dilations"] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert CONFIG["training"]["updates"] == 2400 and CONFIG["training"]["horizon"] == 8
    assert CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"]["w256_sequential_blocks"] == [200, 300, 450, 600]


def test_four_persistent_tokens_have_exact_shape_and_case_local_reset() -> None:
    cell = model(); first = cell.initial_tokens(3); second = cell.initial_tokens(3)
    assert first.shape == (3, 4, 16)
    assert torch.equal(first, second)
    first[0, 0, 0] += 1
    assert not torch.equal(first, second)
    assert torch.equal(second[0], second[1])


def test_access_is_exact_zero_initialized_and_core_is_candidate_derived() -> None:
    cell = model(); pruned, manifest = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    assert torch.count_nonzero(cell.access_out.weight) == 0
    assert len(manifest["selected_source_channels_ranked"]) == 48
    assert all(torch.equal(cell.core.state_dict()[name], value) for name, value in pruned.items())


@pytest.mark.parametrize("width", [64, 128, 256])
def test_zero_access_primary_and_masked_control_are_bit_exact(width: int) -> None:
    cell = model(); inputs = torch.randn(2, 6, width); tokens = cell.initial_tokens(2)
    primary = cell.step(inputs, tokens, True); control = cell.step(inputs, tokens, False)
    assert torch.equal(primary[0], control[0]) and torch.equal(primary[1], control[1])
    assert torch.equal(primary[2], control[2])


def test_token_computation_persists_and_changes_after_a_step() -> None:
    cell = model(); tokens = cell.initial_tokens(2); inputs = torch.randn(2, 6, 64)
    _, _, updated = cell.step(inputs, tokens, True)
    assert updated.shape == tokens.shape and not torch.equal(updated, tokens)


def test_spatial_token_computation_is_identical_and_only_access_is_masked() -> None:
    source = inspect.getsource(run.PersistentMemoryStudent.step)
    assert source.count("scaled_dot_product_attention") == 2
    assert source.count("self.token_update") == 1
    assert "residual * float(inject)" in source
    assert source.index("residual * float(inject)") < source.index("for dilation")


def test_one_cycle_is_exact_outward_dilation_schedule() -> None:
    cell = model(); cell.core.width = 256
    assert list(cell.core.dilation_schedule(256)) == CONFIG["model"]["outward_dilations"]


def test_primary_access_has_gradient_while_masked_access_is_zero() -> None:
    primary = model().train(); control = model().train(); inputs = torch.randn(2, 6, 64)
    p = primary.step(inputs, primary.initial_tokens(2), True)[0].sum(); p.backward()
    c = control.step(inputs, control.initial_tokens(2), False)[0].sum(); c.backward()
    assert float(primary.access_out.weight.grad.abs().sum()) > 0
    assert float(control.access_out.weight.grad.abs().sum()) == 0


def test_rollout_call_reset_is_deterministic_and_cross_call_isolated() -> None:
    cell = model(); args = ([17, 19], [23, 29], [61, 59], 64)
    first = run.rollout(cell, *args); _ = run.rollout(cell, [7], [11], [53], 64); second = run.rollout(cell, *args)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert first["token_diagnostics"] == second["token_diagnostics"]


@pytest.mark.parametrize("intervention", ["reset", "freeze", "permute", "randomize"])
def test_declared_token_interventions_execute(intervention: str) -> None:
    result = run.rollout(model(), [17, 19], [23, 29], [61, 59], 64, True, intervention)
    assert result["cases"] == 2 and len(result["correct_vector"]) == 2


def test_paired_statistics_bind_wins_losses_ties_and_time_uniform_interval() -> None:
    pair = run.paired_counts([True, True, False, False], [False, True, True, False])
    assert pair == {"cases": 4, "wins": 1, "losses": 1, "ties": 2}
    interval = run.anytime_hoeffding_interval([1.0, 0.0, -1.0, 0.0], 0.0025)
    assert interval["lower"] <= interval["mean"] <= interval["upper"]


def test_candidate_lower_tier_route_is_unchanged_and_no_operand_swap() -> None:
    source = inspect.getsource(run.deployed_route)
    assert "incumbent.predict_digits_batch(lower)" in source
    assert '"operand_swap_route": False' in source
    assert "labels_available_to_timed_route" in source
    assert not any(token in source.lower() for token in ("expected_answer", "output_repair", "answer_selector"))


def test_exact_arithmetic_is_absent_from_learned_model_inference() -> None:
    source = inspect.getsource(run.PersistentMemoryStudent).lower()
    forbidden = ("barrett", "montgomery", "long_division", "crt", "expected", "target", "modulus")
    assert not any(token in source for token in forbidden)


def test_runtime_admission_is_exact_complete_route_rule() -> None:
    source = inspect.getsource(run.Experiment.calibration)
    assert "2.0 * maximum" in source
    assert "complete_1100_case_route" in source
    assert "variability_trigger" in source and "extra_cold_repetitions" in source
    assert CONFIG["gates"]["maximum_candidate_guarded_wall_seconds"] == 285.0
    assert CONFIG["resource_policy"]["maximum_peak_rss_bytes"] == 6 * 1024**3


@pytest.mark.parametrize("mode,root", [("calibration", 20261315), ("formal", 20261314),
                                        ("confirmation", 20261315), ("independent", 20261317)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_science_requires_complete_training_full_denominators_and_causal_gate() -> None:
    source = inspect.getsource(run.Experiment.science)
    for token in ('"complete_updates"', '"healthy_control"', '"paired_gain"', '"causal_erasure"',
                  '"confirmation_threshold"', '"w128_retention"', '"primary_w256_transition"'):
        assert token in source
    assert "max(self.config[\"evaluation\"][\"w256_sequential_blocks\"])" in source


def test_canonical_records_absent_before_or_schema_valid_after_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-690-c1-persistent-memory-tokens/run.json"
    metrics_path = ROOT / "evals/results/exp-690-c1-persistent-memory-tokens/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if run_path.exists():
        record = json.loads(run_path.read_text()); metrics = json.loads(metrics_path.read_text())
        assert record["schema_version"] == metrics["schema_version"] == "exp690-terminal-v1"
        assert record["completion_marker"] == metrics["completion_marker"] == "terminal-persisted"
        assert set(record["roots"]) == {str(x) for x in range(20261314, 20261323)}


def test_tournament_contract_cannot_cancel_other_entrants() -> None:
    limitations = " ".join(run.Experiment.limitations(False))
    assert "C1 is first by council ID, not a selected winner" in limitations
    assert "cannot cancel or weaken C2-C7" in limitations
