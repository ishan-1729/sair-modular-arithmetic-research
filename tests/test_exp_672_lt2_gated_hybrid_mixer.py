from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_672_lt2_gated_hybrid_mixer.model import (
    ROUTES, TRAINED_ROUTES, LT2GatedHybridCell, LT2HybridMixer, MatchedSharedConv,
)
from experiments.exp_672_lt2_gated_hybrid_mixer.run import (
    Experiment, build_model, features, load_exp300, sha256, train_route,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp_672_lt2_gated_hybrid_mixer.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
SOURCE = ROOT / CONFIG["source_checkpoint"]


@pytest.fixture
def model():
    torch.manual_seed(672)
    return build_model(CONFIG, SOURCE)


def sample(width=16, batch=3):
    return features(list(range(batch)), [7] * batch, [65521] * batch,
                    [0, 1, 0][:batch], width)


def test_frozen_source_paper_official_pin_fixtures_roots_and_routes():
    assert sha256(SOURCE) == CONFIG["source_checkpoint_sha256"]
    assert sha256(ROOT / CONFIG["paper"]["path"]) == CONFIG["paper"]["sha256"]
    assert CONFIG["paper"]["pages"] == [3, 4, 5, 8, 9, 10, 11]
    assert CONFIG["official_source"]["commit"] == "adc2ba2f2704ff82744700679d1d0a090314df6c"
    assert CONFIG["official_source"]["license"] == "BSD-3-Clause"
    browser = ROOT / "tmp/playwright/exp672/official-pinned-parity.json"
    assert browser.is_file()
    assert CONFIG["official_source"]["transformer_response_sha256"] in browser.read_text()
    for name, digest in CONFIG["fixture_hashes"].items():
        fixture = ROOT / ".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources" / name
        assert sha256(fixture) == digest
    assert tuple(CONFIG["routes"]) == ROUTES
    assert tuple(CONFIG["trained_routes"]) == TRAINED_ROUTES
    assert CONFIG["roots"] == {"calibration": 20261238, "viability": 20261239,
                               "confirmations": [20261240, 20261241]}


def test_zero_gates_make_all_gated_routes_exactly_proposal_identical(model):
    batch = sample()
    proposal = model(batch, route="proposal_only")
    for route in ("full_hybrid", "gate_only", "shared_conv_control", "reset", "randomized"):
        result = model(batch, route=route)
        assert torch.equal(result.state, proposal.state)
        assert torch.equal(result.logits, proposal.logits)
        assert torch.equal(result.quotient_logits, proposal.quotient_logits)
    assert torch.count_nonzero(model.loop_gate) == 0
    assert torch.count_nonzero(model.output_gate) == 0


def test_genuinely_shared_hybrid_and_official_shaped_gate(model):
    assert isinstance(model.hybrid, LT2HybridMixer)
    assert isinstance(model.control, MatchedSharedConv)
    assert model.loop_count == 4
    assert len({id(model.hybrid) for _ in range(model.loop_count)}) == 1
    assert torch.count_nonzero(model.hybrid.sparse_gate.weight) == 0
    x = torch.zeros(2, 190, 8)
    assert torch.equal(torch.sigmoid(model.hybrid.sparse_gate(x)), torch.full((2, 24, 8), 0.5))
    source = inspect.getsource(LT2HybridMixer)
    assert "cumsum(-1)" in source and "unfold(-1, self.sparse_window, 1)" in source
    assert "softmax(-1)" in source and "sigmoid(self.sparse_gate(x))" in source


def test_loop_and_output_gate_math_matches_frozen_adaptation(model):
    source = inspect.getsource(LT2GatedHybridCell._run)
    assert "mixed + rho * state" in source
    assert "proposal + torch.tanh(self.output_gate) * (state - proposal)" in source
    assert "self.loop_gate[index]" in source
    assert model.hybrid.sparse_window == 5


def test_parameter_and_compute_matched_control_are_disclosed_and_close(model):
    full = model.trainable_parameter_count("full_hybrid")
    control = model.trainable_parameter_count("shared_conv_control")
    assert abs(full - control) / full < 0.35
    full_flops = model.analytical_mixer_flops(128, 16, "full_hybrid")
    control_flops = model.analytical_mixer_flops(128, 16, "shared_conv_control")
    assert full_flops > 0 and control_flops > 0 and full_flops != control_flops


def test_fixed_gate_control_is_structurally_distinct(model):
    result = model(sample(), route="mixer_fixed_gate")
    proposal = model(sample(), route="proposal_only")
    assert not torch.equal(result.state, proposal.state)
    assert CONFIG["model"]["fixed_mixer_scale"] == 0.05
    assert set(("gate_only", "mixer_fixed_gate", "shared_conv_control",
                "proposal_only", "reset", "randomized")).issubset(ROUTES)


def test_depth_curve_is_bounded_and_state_shapes_are_stable(model):
    batch = sample(width=24, batch=2)
    for count in (1, 2, 3, 4):
        result = model(batch, route="full_hybrid", loops=count)
        assert len(result.loop_states) == count
        assert result.state.shape == (2, 92, 24)
        assert result.iterations.unique().item() == count
    with pytest.raises(ValueError):
        model(batch, route="full_hybrid", loops=5)


def test_split_separation_and_root_locks(tmp_path):
    experiment = Experiment(CONFIG_PATH, tmp_path, "preregister", 20261238)
    for split in experiment.splits.values():
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["train"]).isdisjoint(split["test"])
        assert set(split["validation"]).isdisjoint(split["test"])
    with pytest.raises(ValueError):
        Experiment(CONFIG_PATH, tmp_path, "formal", 20261238)
    with pytest.raises(RuntimeError, match="calibration"):
        Experiment(CONFIG_PATH, tmp_path, "formal", 20261239).run()


def test_two_updates_open_output_then_mixer_learning_and_reload(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["batch_size"] = 2
    config["training"]["schedule"] = [16]
    config["data"]["training_widths"] = [16]
    config["data"]["widths"] = [16]
    exp300 = load_exp300()
    raw = exp300.bigint_prime_split({"data": {"prime_bits_min": 13, "prime_bits_max": 16,
                                              "train_prime_identities": 8,
                                              "test_prime_identities": 4}}, 67201)
    splits = {16: {"train": raw["train"], "validation": raw["test"][:2],
                   "test": raw["test"][2:]}}
    initial = build_model(config, SOURCE)
    before = initial.hybrid.linear_q.weight.detach().clone()
    trained, result = train_route(config, exp300, SOURCE, splits, 20261238,
                                  "full_hybrid", 2, tmp_path, "calibration")
    assert result["checkpoint_reload_exact"]
    assert result["training"]["shared_loop_calls"] == 16
    assert torch.count_nonzero(trained.output_gate) > 0
    assert not torch.equal(before, trained.hybrid.linear_q.weight)


def test_gate_reset_and_mixer_randomization_are_separate_causal_interventions(model):
    with torch.no_grad():
        model.output_gate.fill_(0.2)
        model.loop_gate.fill_(0.1)
    before = {name: value.detach().clone() for name, value in model.hybrid.state_dict().items()}
    model.reset_gate_intervention()
    assert torch.count_nonzero(model.output_gate) == torch.count_nonzero(model.loop_gate) == 0
    assert all(torch.equal(before[name], value) for name, value in model.hybrid.state_dict().items())
    model.randomized_mixer_intervention(11)
    assert any(not torch.equal(before[name], value) for name, value in model.hybrid.state_dict().items())


def test_deterministic_manual_checkpoint_reload(model, tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, path)
    reloaded = build_model(CONFIG, SOURCE)
    reloaded.load_state_dict(load_file(path))
    left = model(sample(), route="full_hybrid")
    right = reloaded(sample(), route="full_hybrid")
    assert torch.equal(left.logits, right.logits)
    assert torch.equal(left.state, right.state)


def test_no_prohibited_arithmetic_answer_routing_external_calls_or_unbounded_loop():
    module = __import__("experiments.exp_672_lt2_gated_hybrid_mixer.model", fromlist=["*"])
    tree = ast.parse(inspect.getsource(module))
    prohibited_calls = {"pow", "divmod", "crt", "mod_inverse", "subprocess", "requests"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in prohibited_calls for node in ast.walk(tree))
    forward = inspect.getsource(LT2GatedHybridCell.forward)
    assert all(token not in forward for token in
               ("target", "answer", "correct", "validity", "modulus", "%", "//", "while"))
    run = inspect.getsource(LT2GatedHybridCell._run)
    assert "for index in range(count)" in run and "while" not in run
