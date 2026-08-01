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
from experiments.exp_684_stars_random_depth_factorial.model import (
    JVP_ROUTES, ROUTES, TRAINED_ROUTES, FactorialCell, compact_hash,
    frozen_count_schedules,
)
from experiments.exp_684_stars_random_depth_factorial.run import (
    Experiment, InferenceView, build_model, features, load_exp300, sha256, train_route,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp_684_stars_random_depth_factorial.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SOURCE = ROOT / CONFIG["source"]["checkpoint"]


def model():
    torch.manual_seed(684)
    return build_model(CONFIG, SOURCE)


def sample(width=16, batch=3):
    return features(list(range(batch)), [7] * batch, [65521] * batch,
                    [0, 1, 0][:batch], width)


def test_authenticated_sources_roots_and_factorial_routes():
    assert sha256(SOURCE) == CONFIG["source"]["checkpoint_sha256"]
    for paper in ("stars_paper", "count_paper"):
        assert sha256(ROOT / CONFIG["source"][paper]["path"]) == CONFIG["source"][paper]["sha256"]
    for name, digest in CONFIG["source"]["fixtures"].items():
        assert sha256(ROOT / ".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources" / name) == digest
    assert tuple(CONFIG["routes"]) == ROUTES == TRAINED_ROUTES
    assert set(JVP_ROUTES) == {"fixed_stars", "random_stars", "shuffled_random_stars",
                              "random_stop_gradient"}
    assert CONFIG["roots"] == {"calibration": 20261286, "viability": 20261287,
                               "confirmations": [20261288, 20261289]}


def test_frozen_balanced_random_and_shuffled_schedules_exactly_match_work():
    schedules = frozen_count_schedules(CONFIG, 400)
    assert schedules == frozen_count_schedules(CONFIG, 400)
    random_counts = schedules["random_stars"]
    shuffled = schedules["shuffled_random_stars"]
    assert compact_hash(random_counts) == CONFIG["count_schedule"]["random_sha256"]
    assert compact_hash(shuffled) == CONFIG["count_schedule"]["shuffled_sha256"]
    assert random_counts != shuffled
    assert Counter(random_counts) == Counter(shuffled)
    assert all(sum(values[stratum::4]) == 200 for values in (random_counts, shuffled)
               for stratum in range(4))
    assert all(1 <= count <= 4 for values in schedules.values() for count in values)
    levels = [6, 7, 8, 7]
    work = {route: sum(count * levels[index % 4] for index, count in enumerate(values))
            for route, values in schedules.items()}
    assert set(work.values()) == {5600}


def test_count_is_one_integer_per_batch_and_fixed_routes_use_two_cycles():
    schedules = frozen_count_schedules(CONFIG, 4)
    assert schedules["fixed_no_jvp"] == schedules["fixed_stars"] == [2, 2, 2, 2]
    assert all(isinstance(count, int) for values in schedules.values() for count in values)


def test_inference_is_inherited_bit_exact_candidate_graph_parameters_and_calls():
    current = model()
    source = SharedDilatedConvGRU(**CONFIG["model"])
    source.load_state_dict(load_file(SOURCE))
    batch = sample(width=64)
    assert FactorialCell.forward is SharedDilatedConvGRU.forward
    assert torch.equal(current(batch)[0], source(batch)[0])
    assert torch.equal(current(batch)[1], source(batch)[1])
    assert current.parameter_count() == source.parameter_count()
    assert set(current.state_dict()) == set(source.state_dict())
    assert len(current.dilation_schedule(64)) == 12
    assert InferenceView(current)(batch, route="random_stars").iterations.unique().item() == 12


def test_two_cycle_training_graph_is_bit_exact_inference():
    current = model()
    batch = sample(width=64)
    training = current.forward_train(batch, cycles=2)
    inference = current(batch)
    assert torch.equal(training.logits, inference[0])
    assert torch.equal(training.quotient_logits, inference[1])
    assert training.recurrent_visits == len(current.dilation_schedule(64))


def test_random_cycles_only_repeat_existing_dilation_visits():
    current = model()
    assert current.cycle_schedule(32, 1) == [1, 2, 4, 8, 16]
    assert current.cycle_schedule(32, 2) == current.dilation_schedule(32)
    assert current.cycle_schedule(32, 3)[-5:] == [1, 2, 4, 8, 16]
    with pytest.raises(ValueError):
        current.cycle_schedule(32, 0)


def test_native_power_jvp_has_higher_order_parameter_gradient():
    current = model()
    batch = sample(batch=2)
    prediction = current.forward_train(batch, cycles=2)
    record = current.jvp_proxy(prediction.boundary, batch, prediction.final_dilation, seed=7)
    gradients = torch.autograd.grad(record.proxy, tuple(current.parameters()), allow_unused=True)
    present = [gradient for gradient in gradients if gradient is not None]
    assert record.jvp_calls == 2 and record.parameter_gradient_enabled
    assert abs(float(record.vector_norm) - 1.0) < 1e-5 and float(record.proxy) > 0
    assert present and sum(float(gradient.abs().sum()) for gradient in present) > 0


def test_stop_gradient_computes_same_proxy_without_parameter_gradient():
    current = model()
    batch = sample(batch=2)
    prediction = current.forward_train(batch, cycles=2)
    active = current.jvp_proxy(prediction.boundary, batch, prediction.final_dilation, seed=8)
    stopped = current.jvp_proxy(prediction.boundary, batch, prediction.final_dilation,
                                seed=8, stop_gradient=True)
    assert torch.equal(active.proxy.detach(), stopped.proxy)
    assert not stopped.proxy.requires_grad and not stopped.parameter_gradient_enabled


def test_randomized_weight_intervention_changes_inference():
    current = model()
    batch = sample()
    before = current(batch)[0]
    current.randomized_intervention(9)
    assert not torch.equal(before, current(batch)[0])


def test_one_joint_update_roundtrips_and_discloses_jvp_work(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["batch_size"] = 2
    config["training"]["schedule"] = [16] * 4
    config["data"]["training_widths"] = [16]
    config["data"]["widths"] = [16]
    exp300 = load_exp300()
    raw = exp300.bigint_prime_split({"data": {"prime_bits_min": 13,
                                              "prime_bits_max": 16,
                                              "train_prime_identities": 8,
                                              "test_prime_identities": 4}}, 68401)
    splits = {16: {"train": raw["train"], "validation": raw["test"][:2],
                   "test": raw["test"][2:]}}
    _, result = train_route(config, exp300, SOURCE, splits, 20261286,
                            "random_stars", 1, tmp_path, "calibration")
    assert result["checkpoint_reload_exact"]
    assert result["training"]["jvp_calls"] == 4
    assert result["training"]["recurrent_visits"] == 8
    assert result["training"]["jvp_gradient_compute_matched"] is False


def test_no_jvp_route_performs_no_jvp_calls(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["batch_size"] = 1
    config["training"]["schedule"] = [16] * 4
    config["data"]["training_widths"] = [16]
    config["data"]["widths"] = [16]
    exp300 = load_exp300()
    raw = exp300.bigint_prime_split({"data": {"prime_bits_min": 13,
                                              "prime_bits_max": 16,
                                              "train_prime_identities": 8,
                                              "test_prime_identities": 4}}, 68402)
    splits = {16: {"train": raw["train"], "validation": raw["test"][:2],
                   "test": raw["test"][2:]}}
    _, result = train_route(config, exp300, SOURCE, splits, 20261286,
                            "random_no_jvp", 1, tmp_path, "calibration")
    assert result["training"]["jvp_calls"] == 0


def test_freeze_preflight_splits_and_formal_lock(tmp_path):
    experiment = Experiment(CONFIG_PATH, tmp_path, "freeze", 20261286)
    experiment.freeze()
    assert Experiment(CONFIG_PATH, tmp_path, "preflight", 20261286).preflight()["status"] == "passed"
    for split in experiment.splits.values():
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["validation"]).isdisjoint(split["test"])
    with pytest.raises(ValueError):
        Experiment(CONFIG_PATH, tmp_path, "formal", 20261286)


def test_no_prohibited_inference_arithmetic_jvp_count_or_routing():
    forward = inspect.getsource(FactorialCell.forward)
    assert "jvp" not in forward.lower() and "cycle" not in inspect.signature(FactorialCell.forward).parameters
    assert all(token not in forward for token in ("%", "//", "answer", "correct", "valid"))
    module = __import__("experiments.exp_684_stars_random_depth_factorial.model", fromlist=["*"])
    tree = ast.parse(inspect.getsource(module))
    prohibited = {"pow", "divmod", "crt", "mod_inverse"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in prohibited for node in ast.walk(tree))
