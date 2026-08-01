from __future__ import annotations

import ast
import copy
import inspect
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_670_clipped_poisson_loop_curriculum.model import (
    ROUTES, TRAINED_ROUTES, LoopCurriculumCell, frozen_count_schedules,
)
from experiments.exp_670_clipped_poisson_loop_curriculum.run import (
    Experiment, build_model, features, load_exp300, sha256, train_route,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp_670_clipped_poisson_loop_curriculum.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SOURCE = ROOT / CONFIG["source_checkpoint"]


def model():
    torch.manual_seed(670)
    return build_model(CONFIG, SOURCE)


def sample(width=16, batch=3):
    return features(list(range(batch)), [7] * batch, [65521] * batch,
                    [0, 1, 0][:batch], width)


def test_frozen_source_paper_fixtures_roots_routes():
    assert sha256(SOURCE) == CONFIG["source_checkpoint_sha256"]
    assert sha256(ROOT / CONFIG["paper"]["path"]) == CONFIG["paper"]["sha256"]
    for name, digest in CONFIG["fixture_hashes"].items():
        fixture = ROOT / ".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources" / name
        assert sha256(fixture) == digest
    assert tuple(CONFIG["routes"]) == ROUTES
    assert tuple(CONFIG["trained_routes"]) == TRAINED_ROUTES
    assert CONFIG["roots"] == {"calibration": 20261230, "viability": 20261231,
                               "confirmations": [20261232, 20261233]}


def test_source_mapped_poisson_bounds_histogram_replay_and_mean_work():
    first = frozen_count_schedules(CONFIG, 400)
    second = frozen_count_schedules(CONFIG, 400)
    assert first == second
    assert set(first) == set(TRAINED_ROUTES)
    assert all(len(values) == 400 and sum(values) == 1600 for values in first.values())
    assert all(2 <= value <= 8 for values in first.values() for value in values)
    assert Counter(first["clipped_poisson"]) == Counter(first["matched_histogram"])
    assert Counter(first["clipped_poisson"]) == Counter(first["shuffled_count_independent"])
    assert set(first["fixed_mean"]) == {4}
    assert first["clipped_poisson"] != first["shuffled_count_independent"]
    assert 0.8 < torch.tensor(first["clipped_poisson"], dtype=torch.float32).var().item() < 4.5


def test_count_is_sampled_once_per_batch_not_per_example():
    schedules = frozen_count_schedules(CONFIG, 2)
    assert schedules["clipped_poisson"] == [5, 3]
    assert all(isinstance(count, int) for values in schedules.values() for count in values)


def test_zero_initialization_is_exact_proposal_identity_at_every_count():
    current = model()
    batch = sample()
    proposal = current(batch, route="proposal_only")
    for route in TRAINED_ROUTES:
        for count in (2, 4, 8):
            refined = current._run(batch, route, count)
            assert torch.equal(refined.state, proposal.state)
            assert torch.equal(refined.logits, proposal.logits)
            assert refined.iterations.unique().item() == count
    assert torch.count_nonzero(current.operator.output.weight) == 0
    assert torch.count_nonzero(current.operator.output.bias) == 0


def test_input_and_proposal_are_reinjected_without_position_parameters():
    source = inspect.getsource(type(model().operator).forward)
    assert "proposal" in source and "features" in source
    assert not any("position" in name or "embedding" in name for name, _ in model().named_parameters())
    assert model().operator.norm.num_channels == 2 * 92 + 6


def test_mandatory_outer_scan_is_invariant_to_refinement_count():
    current = model()
    batch = sample(width=32)
    observed = []
    original = current.base.recurrent_update

    def record(state, input_features, dilation):
        observed.append(dilation)
        return original(state, input_features, dilation)

    current.base.recurrent_update = record
    expected = current.base.dilation_schedule(32)
    for count in (0, 2, 8):
        observed.clear()
        current._run(batch, "proposal_only" if count == 0 else "clipped_poisson", count)
        assert observed == expected


def test_inference_count_is_fixed_target_free_and_route_counts_cannot_override_it():
    current = model()
    batch = sample()
    assert current(batch, route="clipped_poisson").iterations.unique().item() == 4
    assert "count" not in inspect.signature(LoopCurriculumCell.forward).parameters
    assert current(batch, route="proposal_only").iterations.unique().item() == 0


def test_all_nonrandomized_routes_have_identical_initial_bytes():
    baseline = model().state_dict()
    for _route in ROUTES[:-1]:
        assert all(torch.equal(value, model().state_dict()[name]) for name, value in baseline.items())


def test_randomized_weight_intervention_changes_refinement():
    current = model()
    batch = sample()
    before = current(batch, route="randomized").logits
    current.randomized_intervention(11)
    after = current(batch, route="randomized").logits
    assert not torch.equal(before, after)


def test_one_update_is_causal_and_roundtrips(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["batch_size"] = 2
    config["training"]["schedule"] = [16] * 4
    config["data"]["training_widths"] = [16]
    config["data"]["widths"] = [16]
    exp300 = load_exp300()
    raw = exp300.bigint_prime_split({"data": {"prime_bits_min": 13, "prime_bits_max": 16,
                                              "train_prime_identities": 8,
                                              "test_prime_identities": 4}}, 67001)
    splits = {16: {"train": raw["train"], "validation": raw["test"][:2],
                   "test": raw["test"][2:]}}
    trained, result = train_route(config, exp300, SOURCE, splits, 20261230,
                                  "clipped_poisson", 1, tmp_path, "calibration")
    assert result["checkpoint_reload_exact"]
    assert result["training"]["refinement_calls"] == 10
    assert torch.count_nonzero(trained.operator.output.weight) > 0


def test_roots_splits_and_formal_lock(tmp_path):
    experiment = Experiment(CONFIG_PATH, tmp_path, "preregister", 20261230)
    for split in experiment.splits.values():
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["validation"]).isdisjoint(split["test"])
    with pytest.raises(ValueError):
        Experiment(CONFIG_PATH, tmp_path, "formal", 20261230)


def test_no_prohibited_inference_arithmetic_or_answer_routing():
    forward = inspect.getsource(LoopCurriculumCell.forward)
    forward_tree = ast.parse(forward.removeprefix("    "))
    names = {node.id for node in ast.walk(forward_tree) if isinstance(node, ast.Name)}
    assert not names.intersection({"target", "answer", "correct", "modulus"})
    assert all(token not in forward for token in ("%", "//", "while"))
    module = __import__("experiments.exp_670_clipped_poisson_loop_curriculum.model", fromlist=["*"])
    tree = ast.parse(inspect.getsource(module))
    prohibited = {"pow", "divmod", "crt", "mod_inverse"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in prohibited for node in ast.walk(tree))
