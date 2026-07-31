from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.exp_691_c2_hard_indexed_trace import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_691_c2_hard_indexed_trace.yaml").read_text())


def model(seed: int = 691900) -> run.HardIndexedTraceStudent:
    state, _ = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    return run.new_student(CONFIG, state, seed).eval()


def test_frozen_protocol_and_all_nine_tournament_roots() -> None:
    assert CONFIG["roots"] == {"calibration": 20261323, "formal": 20261324,
                               "confirmations": [20261325, 20261326],
                               "independent_seeds": [20261327, 20261328, 20261329, 20261330, 20261331]}
    assert CONFIG["model"] == {"source_hidden": 92, "hidden": 48, "width": 256, "cycles": 1,
                               "outward_dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                               "trace_block_size": 16, "trace_record_dimension": 32}
    assert CONFIG["training"]["updates"] == 2400 and CONFIG["training"]["horizon"] == 8
    assert CONFIG["data"]["curriculum"] == [64, 128, 256, 256]
    assert CONFIG["evaluation"]["w256_sequential_blocks"] == [200, 300, 450, 600]


@pytest.mark.parametrize("width,blocks", [(64, 4), (128, 8), (256, 16)])
def test_trace_has_exact_block_slot_record_shape(width: int, blocks: int) -> None:
    trace = model().initial_trace(3, width)
    assert trace.shape == (3, blocks, 16, 32)
    assert torch.count_nonzero(trace) == 0


def test_append_is_chronological_immutable_and_never_overwrites() -> None:
    cell = model(); trace = cell.initial_trace(2, 64)
    snapshots = []
    for position in range(18):
        record = torch.full((2, 32), float(position + 1)); before = trace.clone()
        trace = cell.append_record(trace, record, position)
        assert torch.equal(trace.flatten(1, 2)[:, position], record)
        assert torch.equal(trace.flatten(1, 2)[:, :position], before.flatten(1, 2)[:, :position])
        snapshots.append(record)
    assert torch.equal(trace[:, 0, 15], snapshots[15]) and torch.equal(trace[:, 1, 0], snapshots[16])
    with pytest.raises(IndexError, match="capacity"):
        cell.append_record(trace, torch.zeros(2, 32), 64)


def test_two_stage_pointer_hard_forward_equals_selected_gather() -> None:
    cell = model(); trace = torch.randn(2, 4, 16, 32); pooled = torch.randn(2, 96)
    read, diagnostics = cell.read_trace(pooled, trace, 31)
    expected = torch.stack([trace[row, block, slot] for row, (block, slot) in enumerate(zip(diagnostics["block_index"], diagnostics["slot_index"], strict=True))])
    assert torch.equal(read, expected)
    assert all(16 * block + slot < 31 for block, slot in zip(diagnostics["block_index"], diagnostics["slot_index"], strict=True))


def test_two_stage_straight_through_preserves_query_and_memory_gradients() -> None:
    cell = model().train(); trace = torch.randn(2, 4, 16, 32, requires_grad=True); pooled = torch.randn(2, 96, requires_grad=True)
    read, _ = cell.read_trace(pooled, trace, 47); read.square().mean().backward()
    assert float(trace.grad.abs().sum()) > 0 and float(pooled.grad.abs().sum()) > 0
    assert sum(float(parameter.grad.abs().sum()) for name, parameter in cell.named_parameters() if name.startswith(("block_", "slot_")) and parameter.grad is not None) > 0


def test_empty_trace_read_is_exact_zero() -> None:
    cell = model(); trace = cell.initial_trace(2, 64); read, pointer = cell.read_trace(torch.randn(2, 96), trace, 0)
    assert torch.count_nonzero(read) == 0 and pointer["read_age"] is None


def test_correction_is_exact_zero_initialized_and_core_is_candidate_derived() -> None:
    cell = model(); pruned, manifest = run.prepare_pruned(CONFIG, ROOT / CONFIG["source_checkpoint"])
    assert torch.count_nonzero(cell.correction_out.weight) == 0
    assert len(manifest["selected_source_channels_ranked"]) == 48
    assert all(torch.equal(cell.core.state_dict()[name], value) for name, value in pruned.items())


@pytest.mark.parametrize("width", [64, 128, 256])
def test_zero_correction_primary_control_and_candidate_core_are_bit_exact(width: int) -> None:
    cell = model(); inputs = torch.randn(2, 6, width); trace = cell.initial_trace(2, width)
    primary = cell.step(inputs, trace, 0, True); control = cell.step(inputs, trace, 0, False); core = cell.core(inputs)
    assert torch.equal(primary[0], control[0]) and torch.equal(primary[1], control[1])
    assert torch.equal(primary[0], core[0]) and torch.equal(primary[1], core[1])
    assert torch.equal(primary[2], control[2])


def test_only_correction_access_is_masked_while_trace_pointer_work_matches() -> None:
    cell = model(); cell.correction_out.weight.data.normal_(); inputs = torch.randn(2, 6, 64)
    trace = cell.initial_trace(2, 64); trace = cell.append_record(trace, torch.randn(2, 32), 0)
    primary = cell.step(inputs, trace, 1, True); control = cell.step(inputs, trace, 1, False)
    assert not torch.equal(primary[0], control[0])
    assert primary[3]["block_index"] == control[3]["block_index"] and primary[3]["slot_index"] == control[3]["slot_index"]
    source = inspect.getsource(run.HardIndexedTraceStudent.step)
    assert "correction * float(correction_access)" in source
    assert source.index("read_trace") < source.index("for dilation") < source.index("append_record")


def test_primary_correction_has_gradient_while_masked_control_is_zero() -> None:
    primary = model().train(); control = model().train(); inputs = torch.randn(2, 6, 64)
    p_trace = primary.append_record(primary.initial_trace(2, 64), torch.randn(2, 32), 0)
    c_trace = control.append_record(control.initial_trace(2, 64), torch.randn(2, 32), 0)
    primary.step(inputs, p_trace, 1, True)[0].sum().backward(); control.step(inputs, c_trace, 1, False)[0].sum().backward()
    assert float(primary.correction_out.weight.grad.abs().sum()) > 0
    assert float(control.correction_out.weight.grad.abs().sum()) == 0


def test_one_cycle_is_exact_outward_dilation_schedule() -> None:
    cell = model(); cell.core.width = 256
    assert list(cell.core.dilation_schedule(256)) == CONFIG["model"]["outward_dilations"]


def test_rollout_appends_exactly_one_nonzero_record_per_horner_step() -> None:
    result = run.rollout(model(), [17, 19], [23, 29], [61, 59], 64)
    assert result["trace_shape"] == [2, 4, 16, 32]
    assert result["trace_nonzero_records"] == 2 * 64


def test_rollout_reset_is_deterministic_and_cross_call_isolated() -> None:
    cell = model(); args = ([17, 19], [23, 29], [61, 59], 64)
    first = run.rollout(cell, *args); _ = run.rollout(cell, [7], [11], [53], 64); second = run.rollout(cell, *args)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert first["pointer_diagnostics"] == second["pointer_diagnostics"]


@pytest.mark.parametrize("intervention", ["erase", "permute", "uniform", "random_pointer"])
def test_declared_trace_pointer_interventions_execute(intervention: str) -> None:
    result = run.rollout(model(), [17, 19], [23, 29], [61, 59], 64, True, intervention)
    assert result["cases"] == 2 and len(result["correct_vector"]) == 2
    assert result["trace_shape"] == [2, 4, 16, 32]


def test_uniform_and_random_pointer_modes_change_address_policy() -> None:
    cell = model(); trace = torch.randn(4, 4, 16, 32); pooled = torch.randn(4, 96); generator = torch.Generator().manual_seed(7)
    _, uniform = cell.read_trace(pooled, trace, 47, "uniform", generator)
    _, random_pointer = cell.read_trace(pooled, trace, 47, "random_pointer", generator)
    assert uniform["block_index"] == [0, 0, 0, 0] and uniform["slot_index"] == [0, 0, 0, 0]
    assert random_pointer["block_index"] != uniform["block_index"] or random_pointer["slot_index"] != uniform["slot_index"]


def test_randomized_trace_parameters_preserve_same_pruned_core() -> None:
    first = model(1); second = model(2)
    assert all(torch.equal(first.core.state_dict()[name], second.core.state_dict()[name]) for name in first.core.state_dict())
    assert not torch.equal(first.record_encoder.weight, second.record_encoder.weight)


def test_pointer_diagnostics_cover_entropy_utilization_collapse_and_age() -> None:
    result = run.rollout(model(), [17, 19], [23, 29], [61, 59], 64)
    diagnostics = result["pointer_diagnostics"]
    assert set(diagnostics) == {"block_entropy", "slot_entropy", "unique_addresses", "last_address_fraction", "mean_read_age", "mean_record_norm"}
    assert diagnostics["unique_addresses"] >= 1 and diagnostics["mean_read_age"] >= 0


def test_paired_statistics_bind_wins_losses_ties_and_time_uniform_interval() -> None:
    pair = run.paired_counts([True, True, False, False], [False, True, True, False])
    assert pair == {"cases": 4, "wins": 1, "losses": 1, "ties": 2}
    interval = run.anytime_hoeffding_interval([1.0, 0.0, -1.0, 0.0], 0.0025)
    assert interval["lower"] <= interval["mean"] <= interval["upper"]


def test_candidate_lower_tier_route_is_unchanged_and_no_operand_swap() -> None:
    source = inspect.getsource(run.deployed_route)
    assert "incumbent.predict_digits_batch(lower)" in source
    assert '"operand_swap_route": False' in source and "labels_available_to_timed_route" in source
    assert not any(token in source.lower() for token in ("expected_answer", "output_repair", "answer_selector"))


def test_exact_arithmetic_and_programmed_semantics_absent_from_learned_inference() -> None:
    source = inspect.getsource(run.HardIndexedTraceStudent).lower()
    forbidden = ("barrett", "montgomery", "long_division", "crt", "expected", "target", "modulus", "program_counter", "opcode")
    assert not any(token in source for token in forbidden)


def test_runtime_admission_is_exact_complete_route_rule() -> None:
    source = inspect.getsource(run.Experiment.calibration)
    assert "2.0 * maximum" in source and "complete_1100_case_route" in source
    assert "variability_trigger" in source and "extra_cold_repetitions" in source
    assert CONFIG["gates"]["maximum_candidate_guarded_wall_seconds"] == 285.0
    assert CONFIG["resource_policy"]["maximum_peak_rss_bytes"] == int(6.4 * 1024**3)


@pytest.mark.parametrize("mode,root", [("calibration", 20261324), ("formal", 20261323),
                                        ("confirmation", 20261324), ("independent", 20261326)])
def test_root_allowlist_fails_closed(mode: str, root: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not frozen"):
        run.Experiment(run.CONFIG_PATH, tmp_path, mode, root)


def test_science_requires_complete_training_full_denominators_confidence_and_causality() -> None:
    source = inspect.getsource(run.Experiment.science)
    for token in ('"complete_updates"', '"healthy_control"', '"paired_gain"', '"causal_erasure"',
                  '"pointer_noncollapse"', '"confirmation_threshold"', '"w128_retention"', '"primary_w256_transition"'):
        assert token in source
    assert "max(self.config[\"evaluation\"][\"w256_sequential_blocks\"])" in source
    assert "anytime_hoeffding_interval" in source


def test_canonical_records_absent_before_or_schema_valid_after_publication() -> None:
    run_path = ROOT / "experiments/runs/exp-691-c2-hard-indexed-trace/run.json"
    metrics_path = ROOT / "evals/results/exp-691-c2-hard-indexed-trace/metrics.json"
    assert run_path.exists() == metrics_path.exists()
    if run_path.exists():
        record = json.loads(run_path.read_text()); metrics = json.loads(metrics_path.read_text())
        assert record["schema_version"] == metrics["schema_version"] == "exp691-terminal-v1"
        assert record["completion_marker"] == metrics["completion_marker"] == "terminal-persisted"
        assert set(record["roots"]) == {str(value) for value in range(20261323, 20261332)}


def test_tournament_contract_cannot_cancel_other_entrants() -> None:
    limitations = " ".join(run.Experiment.limitations(False))
    assert "C2 is independently required" in limitations
    assert "cannot cancel or weaken C3-C7" in limitations
