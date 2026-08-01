from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.exp_692_c3_quotient_carry_conditioning import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_692_c3_quotient_carry_conditioning.yaml").read_text())


def model(seed: int = 692900) -> run.QuotientCarryFilmStudent:
    state, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    return run.new_student(CONFIG, state, seed).eval()


def test_frozen_protocol_and_all_nine_roots() -> None:
    assert CONFIG["roots"] == {"calibration": 20261332, "formal": 20261333,
                               "confirmations": [20261334, 20261335],
                               "independent_seeds": [20261336, 20261337, 20261338, 20261339, 20261340]}
    assert CONFIG["model"]["hidden"] == 48 and CONFIG["model"]["cycles"] == 1
    assert CONFIG["model"]["quotient_classes"] == 3 and CONFIG["model"]["signed_carry_classes"] == 4
    assert CONFIG["training"]["updates"] == 2400 and CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"]["w256_sequential_blocks"] == [200, 300, 450, 600]


def test_offline_quotient_and_signed_carry_semantics() -> None:
    states = [0, 37, 58]; multiplicands = [17, 23, 59]; moduli = [61, 61, 61]; flags = [0, 1, 1]
    labels = run.offline_carry_targets(states, multiplicands, moduli, flags, 64)
    assert labels.shape == (3, 64) and labels.min() >= 0 and labels.max() < 4
    source = inspect.getsource(run.offline_carry_targets)
    assert "quotient_carry_trace" in source and "LegalTransition" in source


def test_film_is_exact_zero_initialized_and_core_is_candidate_derived() -> None:
    cell = model(); pruned, manifest = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    assert torch.count_nonzero(cell.film.weight) == 0
    assert len(manifest["selected_source_channels_ranked"]) == 48
    assert all(torch.equal(cell.core.state_dict()[name], value) for name, value in pruned.items())


@pytest.mark.parametrize("width", [64, 128, 256])
def test_zero_film_primary_control_and_candidate_core_are_bit_exact(width: int) -> None:
    cell = model(); inputs = torch.randn(2, 6, width)
    primary = cell.step(inputs, True); control = cell.step(inputs, False); core = cell.core(inputs)
    assert torch.equal(primary[0], control[0]) and torch.equal(primary[1], control[1])
    assert torch.equal(primary[0], core[0]) and torch.equal(primary[1], core[1])
    assert primary[2].shape == control[2].shape == (2, 4, width)


def test_only_conditioning_access_is_masked_while_posterior_film_work_matches() -> None:
    cell = model(); cell.film.weight.data.normal_(); inputs = torch.randn(3, 6, 64)
    primary = cell.step(inputs, True); control = cell.step(inputs, False)
    assert not torch.equal(primary[0], control[0])
    assert primary[1].shape == control[1].shape and primary[2].shape == control[2].shape
    source = inspect.getsource(run.QuotientCarryFilmStudent.step)
    assert "film_delta * float(conditioning_access)" in source
    assert source.index("self._posteriors") < source.index("self.film") < source.index("for dilation")


def test_primary_film_gradient_and_control_mask() -> None:
    primary = model().train(); control = model().train(); inputs = torch.randn(2, 6, 64)
    primary.step(inputs, True)[0].sum().backward(); control.step(inputs, False)[0].sum().backward()
    assert float(primary.film.weight.grad.abs().sum()) > 0
    assert float(control.film.weight.grad.abs().sum()) == 0


def test_auxiliary_heads_receive_offline_losses_in_both_arms(tmp_path: Path) -> None:
    exp = run.Experiment(run.CONFIG_PATH, tmp_path, "smoke", None)
    exp.root = 692000; exp.splits = {w: run.make_split(exp.config, exp.exp300, w, exp.root) for w in (64, 128, 256)}
    _, records = exp.train_routes(tmp_path, 1)
    for arm in run.TRAINED_ARMS:
        row = records[arm]["training"]["history"][0]
        assert row["quotient_head_gradient"] > 0 and row["carry_head_gradient"] > 0
        assert row["posterior_calls"] == row["film_calls"] == row["auxiliary_loss_calls"] == 8
    assert records["primary"]["initial_state_sha256"] == records["access_masked_control"]["initial_state_sha256"]


def test_one_cycle_is_exact_outward_dilation_schedule() -> None:
    assert list(model().core.dilation_schedule(256)) == CONFIG["model"]["outward_dilations"]


def test_rollout_is_deterministic_and_cross_call_isolated() -> None:
    cell = model(); args = ([17, 19], [23, 29], [61, 59], 64)
    first = run.rollout(cell, *args); _ = run.rollout(cell, [7], [11], [53], 64); second = run.rollout(cell, *args)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert first["posterior_diagnostics"] == second["posterior_diagnostics"]
    assert first["maximum_posterior_shapes"] == [[2, 3], [2, 4, 64]]


@pytest.mark.parametrize("intervention", ["zero_quotient", "zero_carry", "zero_joint", "permute", "randomize_heads"])
def test_declared_interventions_execute(intervention: str) -> None:
    result = run.rollout(model(), [17, 19], [23, 29], [61, 59], 64, True, intervention)
    assert result["cases"] == 2 and len(result["correct_vector"]) == 2


def test_posterior_shapes_and_entropy_diagnostics() -> None:
    result = run.rollout(model(), [17, 19], [23, 29], [61, 59], 64)
    assert set(result["posterior_diagnostics"]) == {"quotient_entropy", "carry_entropy", "film_delta_norm"}
    assert result["posterior_diagnostics"]["quotient_entropy"] > 0
    assert result["posterior_diagnostics"]["carry_entropy"] > 0


def test_randomized_extension_preserves_same_pruned_core() -> None:
    first = model(1); second = model(2)
    assert all(torch.equal(first.core.state_dict()[name], second.core.state_dict()[name]) for name in first.core.state_dict())
    assert not torch.equal(first.carry_head.weight, second.carry_head.weight)


def test_candidate_lower_tiers_are_unchanged_and_deployed_route_has_no_labels() -> None:
    source = inspect.getsource(run.deployed_route)
    assert "incumbent.predict_digits_batch(lower)" in source
    assert '"operand_swap_route": False' in source and '"labels_available_to_timed_route": False' in source
    assert "offline_carry_targets" not in source and "quotient_carry_trace" not in source


def test_programmed_arithmetic_absent_from_learned_inference_class() -> None:
    source = inspect.getsource(run.QuotientCarryFilmStudent).lower()
    forbidden = ("barrett", "montgomery", "long_division", "crt", "divmod", "modulus", "target", "expected", "opcode", "comparator")
    assert not any(token in source for token in forbidden)


def test_runtime_admission_is_exact_complete_route_rule() -> None:
    source = inspect.getsource(run.Experiment.calibration)
    assert "2.0 * maximum" in source and "complete_1100_case_route" in source
    assert "variability_trigger" in source and "extra_cold_repetitions" in source
    assert CONFIG["gates"]["maximum_candidate_guarded_wall_seconds"] == 285.0
    assert CONFIG["resource_policy"]["maximum_peak_rss_bytes"] == int(6.4 * 1024**3)


@pytest.mark.parametrize("mode,root", [("calibration", 20261333), ("formal", 20261332),
                                        ("confirmation", 20261333), ("independent", 20261335)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_science_requires_full_denominators_confidence_and_separate_causality() -> None:
    source = inspect.getsource(run.Experiment.science)
    for token in ('"complete_updates"', '"healthy_control"', '"paired_gain"', '"causal_erasure"',
                  '"posterior_noncollapse"', '"confirmation_threshold"', '"w128_retention"', '"primary_w256_transition"'):
        assert token in source
    for intervention in ("zero_quotient", "zero_carry", "zero_joint", "permute", "randomize_heads"):
        assert intervention in source
    assert "randomized_full_weights" in source and "anytime_hoeffding_interval" in source


def test_canonical_records_absent_before_or_schema_valid_after_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-692-c3-quotient-carry-conditioning/run.json"
    metrics_path = ROOT / "evals/results/exp-692-c3-quotient-carry-conditioning/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if run_path.exists():
        record = json.loads(run_path.read_text()); metrics = json.loads(metrics_path.read_text())
        assert record["completion_marker"] == metrics["completion_marker"] == "terminal-persisted"


def test_tournament_contract_cannot_cancel_other_entrants() -> None:
    limitations = " ".join(run.Experiment.limitations(False))
    assert "C3 is independently required" in limitations and "cannot cancel or weaken C4-C7" in limitations
