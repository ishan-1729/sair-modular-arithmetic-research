from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.exp_694_routed_antithetic_quorum import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_694_routed_antithetic_quorum.yaml").read_text())


def model(seed: int = 694900) -> run.RoutedAntitheticQuorumStudent:
    state, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    return run.new_student(CONFIG, state, seed).eval()


def test_frozen_protocol_roots_model_and_complete_rung() -> None:
    assert CONFIG["roots"] == {"calibration": 20261350, "formal": 20261351,
                               "confirmations": [20261352, 20261353],
                               "independent_seeds": [20261354, 20261355, 20261356, 20261357, 20261358]}
    assert CONFIG["model"]["hidden"] == 56 and CONFIG["model"]["branch_fraction"] == 0.25
    assert CONFIG["model"]["proposals"] == 3 and CONFIG["model"]["quota_group_size"] == 4
    assert CONFIG["training"]["updates"] == 2400 and CONFIG["training"]["horizon"] == 8
    assert CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"]["transition_cases_per_width"] == 3000
    assert CONFIG["evaluation"]["w256_sequential_blocks"] == [200, 300, 450, 600]


@pytest.mark.parametrize("batch,routed,eligible", [(1, 0, 0), (4, 1, 4), (16, 4, 16), (18, 4, 16)])
@pytest.mark.parametrize("route_mode", ["learned-survival-q", "deterministic-target-independent-random"])
def test_exact_group_quota_avoids_batch_global_topk_collapse(batch: int, routed: int, eligible: int, route_mode: str) -> None:
    cell = model(); scores = torch.linspace(-1, 1, batch)
    indices, actual_eligible = cell.route_indices(scores, 7, route_mode, 20261350)
    assert len(indices) == routed and actual_eligible == eligible
    if eligible:
        assert len(set((indices // 4).tolist())) == eligible // 4
        assert routed * 4 == eligible


def test_random_route_is_target_independent_deterministic_and_seeded() -> None:
    cell = model(); a, _ = cell.route_indices(torch.randn(16), 9, "deterministic-target-independent-random", 11)
    b, _ = cell.route_indices(torch.randn(16), 9, "deterministic-target-independent-random", 11)
    c, _ = cell.route_indices(torch.randn(16), 10, "deterministic-target-independent-random", 11)
    assert torch.equal(a, b) and not torch.equal(a, c)
    source = inspect.getsource(run.RoutedAntitheticQuorumStudent.route_indices).lower()
    assert not any(token in source for token in ("target_bits", "answer", "modulus", "expected"))


def test_learned_route_uses_survival_q_and_differs_from_random_control() -> None:
    cell = model(); scores = torch.tensor([0.0, 4.0, 1.0, 2.0, 9.0, 0.0, 1.0, 2.0])
    learned, _ = cell.route_indices(scores, 0, "learned-survival-q", 0)
    random, _ = cell.route_indices(scores, 0, "deterministic-target-independent-random", 0)
    assert learned.tolist() == [1, 4] and not torch.equal(learned, random)


@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("route_mode", ["learned-survival-q", "deterministic-target-independent-random"])
def test_actual_sparse_step_has_exact_two_extra_proposals(width: int, route_mode: str) -> None:
    cell = model(); logits, quotient, diagnostic, survival = cell.step(torch.randn(16, 6, width), 3, route_mode, 7)
    assert logits.shape == (16, width) and quotient.shape == (16, 3) and survival.shape == (16,)
    assert diagnostic["eligible"] == 16 and diagnostic["routed"] == 4
    assert diagnostic["proposal_calls"] == 24
    assert diagnostic["route_fraction"] == 0.25


def test_antithetic_branches_are_plus_and_minus_same_learned_delta() -> None:
    source = inspect.getsource(run.RoutedAntitheticQuorumStudent.step)
    assert "plus = selected + delta" in source
    assert "else -delta" in source
    assert "torch.stack((selected, plus, minus), dim=1)" in source
    assert "index_copy" in source and "index_select" in source


def test_energy_and_quorum_produce_one_deterministic_next_state() -> None:
    cell = model(); cell.antithetic_delta.weight.data.normal_(std=0.01)
    inputs = torch.randn(8, 6, 64)
    first = cell.step(inputs, 5, "learned-survival-q", 1)
    second = cell.step(inputs, 5, "learned-survival-q", 1)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    source = inspect.getsource(run.RoutedAntitheticQuorumStudent.step)
    assert "self.energy" in source and "self.quorum_scale * agreement" in source
    assert "weights = selection.softmax(1)" in source


@pytest.mark.parametrize("intervention", ["route_randomization", "energy_erasure", "energy_permutation", "sign_collapse", "branch_shuffle"])
def test_all_preregistered_interventions_execute(intervention: str) -> None:
    cell = model(); result = run.rollout(cell, [17, 19, 23, 29], [31, 37, 41, 43], [61, 59, 53, 47], 64,
                                        "learned-survival-q", intervention, 5)
    assert result["cases"] == 4 and len(result["correct_vector"]) == 4
    assert result["routing"]["routed"] * 4 == result["routing"]["eligible"]


def test_primary_and_control_match_initialization_parameterization_and_work(tmp_path: Path) -> None:
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "smoke", None)
    experiment.root = 694000
    experiment.splits = {width: run.make_split(CONFIG, experiment.exp300, width, experiment.root) for width in (64, 128, 256)}
    models, records = experiment.train_routes(tmp_path, 1)
    assert records["primary"]["initial_state_sha256"] == records["random_routing_control"]["initial_state_sha256"]
    assert records["primary"]["parameters"] == records["random_routing_control"]["parameters"]
    for key in ("updates", "student_calls", "teacher_calls", "routed_examples", "eligible_examples", "proposal_calls", "exact_loss_calls"):
        assert records["primary"]["training"][key] == records["random_routing_control"]["training"][key]
    assert not any(a.data_ptr() == b.data_ptr() for a, b in zip(models["primary"].parameters(), models["random_routing_control"].parameters(), strict=True))


def test_router_energy_delta_and_core_all_receive_gradient() -> None:
    cell = model().train(); exp300 = run.exp687.load_exp300(); teacher = run.new_teacher(CONFIG, ROOT / CONFIG["source_checkpoint"])
    split = run.make_split(CONFIG, exp300, 64, 20261350); sampler = exp300.BigIntSampler(split["train"], 7)
    left, right, modulus = sampler.pairs(16); starts = [0] * 16
    optimizer = torch.optim.AdamW(cell.parameters(), lr=1e-5)
    run.train_update(CONFIG, exp300, cell, teacher, optimizer, (left, right, modulus, starts), 64, "learned-survival-q", 3)
    groups = (cell.survival_q.parameters(), cell.energy.parameters(), cell.antithetic_delta.parameters(), cell.core.parameters())
    assert all(sum(float(parameter.grad.abs().sum()) for parameter in group if parameter.grad is not None) > 0 for group in groups)


def test_rollout_is_cross_call_isolated_and_bit_exact() -> None:
    cell = model(); args = ([17, 19, 23, 29], [31, 37, 41, 43], [61, 59, 53, 47], 64)
    first = run.rollout(cell, *args); _ = run.rollout(cell, [7, 11, 13, 17], [19, 23, 29, 31], [47, 43, 41, 37], 64)
    second = run.rollout(cell, *args)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert first["routing"] == second["routing"]


def test_no_prohibited_exact_arithmetic_or_target_selection_in_model() -> None:
    source = inspect.getsource(run.RoutedAntitheticQuorumStudent).lower()
    forbidden = ("barrett", "montgomery", "long_division", "crt", "expected_answer", "answer_selector", "output_repair")
    assert not any(token in source for token in forbidden)
    deployed = inspect.getsource(run.deployed_route)
    assert '"labels_available_to_timed_route": False' in deployed and '"operand_swap_route": False' in deployed


def test_runtime_admission_is_exact_cold_complete_route_rule() -> None:
    source = inspect.getsource(run.Experiment.calibration)
    assert "2.0 * maximum" in source and '"complete_1100_case_route"' in source
    assert "variability_trigger" in source and "extra_cold_repetitions" in source
    assert '"exact_quota"' in source and '"deterministic_outputs"' in source
    assert CONFIG["gates"]["maximum_candidate_guarded_wall_seconds"] == 285.0
    assert CONFIG["resource_policy"]["maximum_peak_rss_bytes"] == int(6.4 * 1024**3)


@pytest.mark.parametrize("mode,root", [("calibration", 20261351), ("formal", 20261350),
                                        ("confirmation", 20261351), ("independent", 20261353)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_science_requires_full_rung_healthy_control_full_denominators_and_causality() -> None:
    source = inspect.getsource(run.Experiment.science)
    for token in ('"complete_updates"', '"healthy_control"', '"paired_gain"', '"causal_erasure"',
                  '"confirmation_threshold"', '"w128_retention"', '"primary_w256_transition"',
                  '"comparison_interpretable"', '"non-comparative-unhealthy-control"'):
        assert token in source
    assert "max(self.config[\"evaluation\"][\"w256_sequential_blocks\"])" in source
    assert "anytime_hoeffding_interval" in source


def test_native_pytorch_library_decision_is_explicit() -> None:
    assert "Native PyTorch" in CONFIG["library_decision"]
    assert "index_select/index_copy" in CONFIG["library_decision"]


def test_canonical_records_absent_before_or_schema_valid_after_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-694-routed-antithetic-quorum/run.json"
    metrics_path = ROOT / "evals/results/exp-694-routed-antithetic-quorum/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if run_path.exists():
        record = json.loads(run_path.read_text()); metrics = json.loads(metrics_path.read_text())
        assert record["schema_version"] == metrics["schema_version"] == "exp694-terminal-v1"
        assert record["completion_marker"] == metrics["completion_marker"] == "terminal-persisted"
        assert set(record["roots"]) == {str(value) for value in range(20261350, 20261359)}


def test_tournament_contract_cannot_cancel_remaining_entrants() -> None:
    limitations = " ".join(run.Experiment.limitations(False))
    assert "C5 is independently required" in limitations and "cannot cancel or weaken C6-C7" in limitations
