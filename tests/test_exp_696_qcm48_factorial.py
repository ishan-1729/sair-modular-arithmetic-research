from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.exp_696_qcm48_factorial import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_696_qcm48_factorial.yaml").read_text())


def model(seed: int = 696900) -> run.QCM48Student:
    state, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    return run.new_student(CONFIG, state, seed).eval()


def test_frozen_protocol_roots_and_complete_first_rung() -> None:
    assert CONFIG["roots"] == {"calibration": 20261368, "formal": 20261369,
                               "confirmations": [20261370, 20261371],
                               "independent_seeds": [20261372, 20261373, 20261374, 20261375, 20261376]}
    assert CONFIG["model"]["hidden"] == 48 and CONFIG["model"]["codec_rank"] == 4
    assert CONFIG["model"]["memory_tokens"] == 4 and CONFIG["model"]["token_dimension"] == 16
    assert CONFIG["training"]["updates"] == 2400 and CONFIG["training"]["horizon"] == 8
    assert CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"]["transition_cases_per_width"] == 3000
    assert CONFIG["evaluation"]["w128_rollout_cases"] == 300
    assert CONFIG["evaluation"]["w256_sequential_blocks"] == [200, 300, 450, 600]


def test_source_provenance_is_exact_c1_c3_composition() -> None:
    contract = CONFIG["source_contract"]
    assert contract["c1_runner"]["sha256"] == "aeaed1afc32e02e076185343b0a2ca6c33ce13c52ce5845181fbd08ab72316a8"
    assert contract["c3_runner"]["sha256"] == "cedcabc2545e50822466c7eb363a623340af55a771c35fc44616350b282cf6fe"
    assert all(run.sha256(ROOT / value["path"]) == value["sha256"] for value in contract.values())


def test_four_routes_are_exact_two_mask_factorial() -> None:
    assert CONFIG["routes"] == {
        "neither": {"trained": True, "qc_access": False, "token_access": False},
        "qc_only": {"trained": True, "qc_access": True, "token_access": False},
        "token_only": {"trained": True, "qc_access": False, "token_access": True},
        "both": {"trained": True, "qc_access": True, "token_access": True},
    }
    source = inspect.getsource(run.QCM48Student.step)
    assert "qc_message * float(qc_access)" in source
    assert "token_message * float(token_access)" in source
    assert "cross_message * float(cross_access)" in source


def test_four_case_local_tokens_reset_and_do_not_alias() -> None:
    cell = model(); first = cell.initial_tokens(3); second = cell.initial_tokens(3)
    assert first.shape == (3, 4, 16) and torch.equal(first, second)
    first[0, 0, 0] += 1
    assert not torch.equal(first, second) and torch.equal(second[0], second[1])


def test_core_is_candidate_derived_and_all_output_projections_start_zero() -> None:
    cell = model(); pruned, manifest = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    assert len(manifest["selected_source_channels_ranked"]) == 48
    assert all(torch.equal(cell.core.state_dict()[name], value) for name, value in pruned.items())
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in (cell.qc_up, cell.token_up, cell.cross_up))


@pytest.mark.parametrize("width", [64, 128, 256])
def test_zero_initialized_four_arms_share_logits_heads_tokens_and_work(width: int) -> None:
    cell = model(); inputs = torch.randn(2, 6, width); tokens = cell.initial_tokens(2)
    rows = [cell.step(inputs, tokens.clone(), qc, token) for qc, token in ((False, False), (True, False), (False, True), (True, True))]
    for index in range(1, 4):
        assert torch.equal(rows[0][0], rows[index][0]); assert torch.equal(rows[0][1], rows[index][1])
        assert torch.equal(rows[0][2], rows[index][2]); assert torch.equal(rows[0][3], rows[index][3])
        assert rows[0][4].keys() == rows[index][4].keys()


def test_rank_four_codec_and_explicit_cross_term_precede_recurrence() -> None:
    cell = model(); assert cell.qc_down.out_channels == cell.token_down.out_channels == 4
    assert cell.qc_up.in_channels == cell.token_up.in_channels == cell.cross_up.in_channels == 4
    source = inspect.getsource(run.QCM48Student.step)
    assert "qc_rank * token_rank" in source
    assert source.index("qc_rank * token_rank") < source.index("for dilation")
    assert source.count("scaled_dot_product_attention") == 2


def test_masks_are_the_only_causal_access_and_gradients_follow_factorial() -> None:
    gradients = {}
    for name, (qc, token) in {"neither": (False, False), "qc_only": (True, False), "token_only": (False, True), "both": (True, True)}.items():
        cell = model().train(); inputs = torch.randn(2, 6, 64); tokens = cell.initial_tokens(2)
        cell.step(inputs, tokens, qc, token)[0].sum().backward()
        gradients[name] = tuple(0.0 if layer.weight.grad is None else float(layer.weight.grad.abs().sum()) for layer in (cell.qc_up, cell.token_up, cell.cross_up))
    assert gradients["neither"] == (0.0, 0.0, 0.0)
    assert gradients["qc_only"][0] > 0 and gradients["qc_only"][1:] == (0.0, 0.0)
    assert gradients["token_only"][1] > 0 and gradients["token_only"][0] == gradients["token_only"][2] == 0.0
    assert all(value > 0 for value in gradients["both"])


def test_offline_quotient_and_signed_carry_labels_never_enter_inference_class() -> None:
    labels = run.offline_carry_targets([0, 37], [17, 23], [61, 61], [0, 1], 64)
    assert labels.shape == (2, 64) and labels.min() >= 0 and labels.max() < 4
    source = inspect.getsource(run.QCM48Student).lower()
    forbidden = ("barrett", "montgomery", "long_division", "crt", "divmod", "modulus", "target", "expected", "verifier")
    assert not any(token in source for token in forbidden)


def test_auxiliary_heads_receive_losses_in_all_four_arms(tmp_path: Path) -> None:
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "smoke", None)
    experiment.root = 696000; experiment.splits = {width: run.make_split(experiment.config, experiment.exp300, width, experiment.root) for width in (64, 128, 256)}
    _, records = experiment.train_routes(tmp_path, 1)
    for arm in run.TRAINED_ARMS:
        row = records[arm]["training"]["history"][0]
        assert row["quotient_head_gradient"] > 0 and row["carry_head_gradient"] > 0
        assert row["posterior_calls"] == row["token_reads"] == row["token_writes"] == row["codec_calls"] == row["fusion_calls"] == row["auxiliary_loss_calls"] == 8
    assert len({records[arm]["initial_state_sha256"] for arm in run.TRAINED_ARMS}) == 1
    assert len({records[arm]["parameters"] for arm in run.TRAINED_ARMS}) == 1


def test_rollout_reset_is_deterministic_and_cross_call_isolated() -> None:
    cell = model(); args = ([17, 19], [23, 29], [61, 59], 64)
    first = run.rollout(cell, *args); _ = run.rollout(cell, [7], [11], [53], 64); second = run.rollout(cell, *args)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert first["diagnostics"] == second["diagnostics"]
    assert first["maximum_shapes"] == {"quotient": [2, 3], "carry": [2, 4, 64], "tokens": [2, 4, 16], "rank": [2, 4, 64]}


@pytest.mark.parametrize("intervention", run.INTERVENTIONS)
def test_all_frozen_causal_interventions_execute(intervention: str) -> None:
    result = run.rollout(model(), [17, 19], [23, 29], [61, 59], 64, True, True, intervention)
    assert result["cases"] == 2 and len(result["correct_vector"]) == 2


def test_factorial_statistic_uses_common_paired_denominator_and_bound_two() -> None:
    interval = run.scaled_anytime_interval([1.0, -2.0, 0.0, 2.0], 0.001, 2.0)
    assert interval["absolute_bound"] == 2.0 and interval["lower"] <= interval["mean"] <= interval["upper"]
    source = inspect.getsource(run.common_rollouts)
    assert "float(b) - float(q) - float(t) + float(n)" in source
    assert "scaled_anytime_interval(interaction_values, alpha, 2.0)" in source


def test_arm_view_exposes_width_and_fixed_masks_to_transition_evaluator() -> None:
    cell = model(); view = run.ArmView(cell, False, True); view.width = 128
    assert view.width == cell.width == cell.core.width == 128
    logits, quotient = view(torch.randn(2, 6, 128)); assert logits.shape == (2, 128) and quotient.shape == (2, 3)


def test_deployed_route_is_both_only_lower_tier_preserving_and_label_free() -> None:
    source = inspect.getsource(run.deployed_route)
    assert "incumbent.predict_digits_batch(lower)" in source
    assert "lower_smoke" in source and "wide_smoke" in source
    assert '"deployed_arm": "both"' in source and '"labels_available_to_timed_route": False' in source
    assert "offline_carry_targets" not in source and "quotient_carry_trace" not in source
    assert '"operand_swap_route": False' in source


def test_runtime_admission_is_measured_complete_route_not_projection() -> None:
    source = inspect.getsource(run.Experiment.calibration)
    assert "2.0 * maximum" in source and "complete_1100_case_route" in source
    assert "variability_trigger" in source and "extra_cold_repetitions" in source
    assert CONFIG["gates"]["maximum_candidate_guarded_wall_seconds"] == 285.0
    assert CONFIG["resource_policy"]["maximum_peak_rss_bytes"] == int(6.4 * 1024**3)


@pytest.mark.parametrize("mode,root", [("calibration", 20261369), ("formal", 20261368),
                                        ("confirmation", 20261369), ("independent", 20261371)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_science_requires_full_work_common_factorial_confidence_and_causality() -> None:
    source = inspect.getsource(run.Experiment.science)
    for token in ('"healthy_neither"', '"complete_updates"', '"both_w256_transition"', '"both_w256_rollout"',
                  '"paired_gain"', '"positive_interaction"', '"interaction_confidence"', '"causal_erasure"',
                  '"cross_term_dependence"', '"confirmation_threshold"'):
        assert token in source
    for intervention in run.INTERVENTIONS: assert intervention in source or intervention in inspect.getsource(run.rollout)
    assert "both-qc_only-token_only+neither" in source


def test_root_lifecycle_is_strictly_sequential() -> None:
    confirmation = inspect.getsource(run.Experiment.confirmation); independent = inspect.getsource(run.Experiment.independent)
    assert "confirmation predecessor gate failed" in confirmation
    assert "both confirmations required" in independent and "independent predecessor gate failed" in independent


def test_canonical_records_absent_before_or_schema_valid_after_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-696-qcm48-factorial/run.json"
    metrics_path = ROOT / "evals/results/exp-696-qcm48-factorial/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if run_path.exists():
        record = json.loads(run_path.read_text()); metrics = json.loads(metrics_path.read_text())
        assert record["schema_version"] == metrics["schema_version"] == "exp696-terminal-v1"
        assert record["completion_marker"] == metrics["completion_marker"] == "terminal-persisted"
        assert set(record["roots"]) == {str(root) for root in range(20261368, 20261377)}


def test_library_decision_and_tournament_limitations_are_explicit() -> None:
    assert "Native PyTorch" in CONFIG["library_decision"]
    limitations = " ".join(run.Experiment.limitations(False))
    assert "C5 remains unresolved" in limitations and "seventh-member adversarial chair is mandatory" in limitations
    assert "no entrant cancels another" in limitations
