from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_671_fixed_point_damping_halting.model import (
    ROUTES,
    TRAINED_ROUTES,
    FixedPointReasonerCell,
    PreNormSharedRefinement,
)
from experiments.exp_671_fixed_point_damping_halting.run import (
    Experiment,
    build_model,
    calibrate_threshold,
    features,
    load_exp300,
    reliability,
    sha256,
    train_route,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp_671_fixed_point_damping_halting.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
SOURCE = ROOT / CONFIG["source_checkpoint"]


@pytest.fixture
def model():
    torch.manual_seed(671)
    return build_model(CONFIG, SOURCE)


def sample(width=16, batch=3):
    return features(list(range(batch)), [7] * batch, [65521] * batch,
                    [0, 1, 0][:batch], width)


def test_frozen_source_paper_fixtures_roots_and_routes():
    assert sha256(SOURCE) == CONFIG["source_checkpoint_sha256"]
    assert sha256(ROOT / CONFIG["paper"]["path"]) == CONFIG["paper"]["sha256"]
    assert CONFIG["paper"]["pages"] == list(range(5, 15)) + [25, 27]
    for name, digest in CONFIG["fixture_hashes"].items():
        fixture = ROOT / ".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources" / name
        assert sha256(fixture) == digest
    assert tuple(CONFIG["routes"]) == ROUTES
    assert tuple(CONFIG["trained_routes"]) == TRAINED_ROUTES
    assert CONFIG["roots"] == {"calibration": 20261234, "viability": 20261235,
                               "confirmations": [20261236, 20261237]}


def test_exact_zero_scale_preserves_proposal_for_every_route(model):
    batch = sample()
    proposal = model(batch, route="proposal_only")
    for route in ROUTES:
        current = model(batch, route=route, threshold=0.0)
        assert torch.equal(current.state, proposal.state)
        assert torch.equal(current.logits, proposal.logits)
    assert torch.count_nonzero(model.refinement.residual_scale) == 0


def test_prenorm_shared_scale_and_iterationwise_reinjection_are_source_mapped(model):
    assert isinstance(model.refinement, PreNormSharedRefinement)
    assert isinstance(model.refinement.norm, torch.nn.GroupNorm)
    source = inspect.getsource(PreNormSharedRefinement.forward)
    assert "self.norm" in source and "proposal" in source and "features" in source
    assert "residual_scale" in source and "state + scale * delta" in source
    assert model.refinement.norm.num_channels == 2 * 92 + 6
    assert len({id(model.refinement) for _ in range(6)}) == 1


def test_training_uses_six_calls_three_deep_losses_and_tbptt_detach(model):
    result = model.forward_train(sample(), route="full_fpr")
    assert result.iterations.unique().item() == 6
    assert len(result.deep_logits) == len(result.deep_quotient_logits) == 3
    source = inspect.getsource(FixedPointReasonerCell._fixed)
    assert "state = state.detach()" in source
    assert model.tbptt_window == 2


def test_normalized_infinity_residual_is_exact(model):
    state = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    proposed = torch.tensor([[[2.0, -2.0], [3.0, 8.0]]])
    assert torch.equal(model.residual(state, proposed), torch.tensor([0.5]))


def test_damping_patience_geometric_decay_and_hard_max(model):
    result = model(sample(), route="damped_fixed", threshold=1.0)
    assert result.iterations.unique().item() == CONFIG["model"]["hard_max_iterations"]
    assert result.hard_max.all()
    assert torch.all(result.final_damping < CONFIG["model"]["initial_damping"])
    assert CONFIG["model"]["patience"] == 2
    assert CONFIG["model"]["damping_decay"] == 0.5


def test_target_free_adaptive_halting_respects_minimum_and_hard_bound(model):
    result = model(sample(), route="full_fpr", threshold=1.0, minimum_iterations=2)
    assert result.iterations.unique().item() == 2
    assert not result.hard_max.any()
    signature = inspect.signature(FixedPointReasonerCell.forward)
    assert not {"target", "answer", "correct", "validity"}.intersection(signature.parameters)


def test_plain_and_component_controls_are_structurally_distinct(model):
    source = inspect.getsource(PreNormSharedRefinement.forward)
    assert "if plain" in source
    assert "torch.zeros_like(proposal)" in source
    assert "reset_scale" in source
    assert set(("damped_fixed", "halt_without_damping", "plain_fixed", "reset", "randomized")).issubset(ROUTES)


def test_randomized_and_reset_interventions_are_causal(model):
    batch = sample()
    proposal = model(batch, route="proposal_only")
    model.randomized_intervention(11)
    randomized = model(batch, route="randomized", threshold=0.0)
    assert not torch.equal(randomized.state, proposal.state)
    model.reset_intervention()
    reset = model(batch, route="reset", threshold=0.0)
    assert torch.equal(reset.state, proposal.state)


def test_threshold_calibration_and_reliability_use_frozen_validation_mapping():
    calibration = {"selected_threshold": 0.1,
                   "probabilities": {"below_threshold": 0.9,
                                     "at_or_above_threshold": 0.1,
                                     "constant": 0.5}}
    records = {"correct": [True, True, False, False],
               "residual": [0.01, 0.02, 0.2, 0.3]}
    measured = reliability(records, calibration)
    assert measured["better_than_constant"]
    assert measured["brier_residual"] < measured["brier_constant"]
    assert CONFIG["threshold_calibration"]["test_labels_forbidden"] is True


def test_split_separation_and_formal_root_locks(tmp_path):
    experiment = Experiment(CONFIG_PATH, tmp_path, "preregister", 20261234)
    for split in experiment.splits.values():
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["train"]).isdisjoint(split["test"])
        assert set(split["validation"]).isdisjoint(split["test"])
    with pytest.raises(ValueError):
        Experiment(CONFIG_PATH, tmp_path, "formal", 20261234)
    with pytest.raises(RuntimeError, match="calibration"):
        Experiment(CONFIG_PATH, tmp_path, "formal", 20261235).run()


def test_one_update_changes_learned_scale_and_checkpoint_roundtrips(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["batch_size"] = 2
    config["training"]["schedule"] = [16]
    config["data"]["training_widths"] = [16]
    config["data"]["widths"] = [16]
    exp300 = load_exp300()
    raw = exp300.bigint_prime_split({"data": {"prime_bits_min": 13, "prime_bits_max": 16,
                                              "train_prime_identities": 8,
                                              "test_prime_identities": 4}}, 67101)
    splits = {16: {"train": raw["train"], "validation": raw["test"][:2],
                   "test": raw["test"][2:]}}
    trained, result = train_route(config, exp300, SOURCE, splits, 20261234,
                                  "full_fpr", 1, tmp_path, "calibration")
    assert result["checkpoint_reload_exact"]
    assert result["training"]["operator_calls"] == 12
    assert torch.count_nonzero(trained.refinement.residual_scale) > 0


def test_deterministic_manual_checkpoint_reload(model, tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, path)
    reloaded = build_model(CONFIG, SOURCE)
    reloaded.load_state_dict(load_file(path))
    left = model(sample(), route="full_fpr", threshold=0.01)
    right = reloaded(sample(), route="full_fpr", threshold=0.01)
    assert torch.equal(left.logits, right.logits)
    assert torch.equal(left.iterations, right.iterations)


def test_no_prohibited_arithmetic_answer_routing_or_unbounded_loop():
    module = __import__("experiments.exp_671_fixed_point_damping_halting.model", fromlist=["*"])
    tree = ast.parse(inspect.getsource(module))
    prohibited_calls = {"pow", "divmod", "crt", "mod_inverse"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in prohibited_calls for node in ast.walk(tree))
    forward = inspect.getsource(FixedPointReasonerCell.forward)
    assert all(token not in forward for token in ("target", "answer", "correct", "modulus", "%", "//", "while"))
    adaptive = inspect.getsource(FixedPointReasonerCell._adaptive)
    assert "range(self.hard_max_iterations)" in adaptive and "while" not in adaptive


def test_calibration_function_has_no_test_surface():
    source = inspect.getsource(calibrate_threshold)
    assert "test" not in source
    assert "correct" in source and "residual" in source
