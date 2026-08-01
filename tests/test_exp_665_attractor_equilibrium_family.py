from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from experiments.exp_665_attractor_equilibrium_family.model import ROUTES, TRAINED_ROUTES, AttractorCell
from experiments.exp_665_attractor_equilibrium_family.run import (
    Experiment,
    basin_diagnostic,
    build_model,
    correctness_auc,
    features,
    load_exp300,
    train_route,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp_665_attractor_equilibrium_family.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SOURCE = ROOT / CONFIG["source_checkpoint"]


def model() -> AttractorCell:
    torch.manual_seed(665)
    return build_model(CONFIG, SOURCE)


def sample(width: int = 16, batch: int = 3) -> torch.Tensor:
    return features(list(range(batch)), [7] * batch, [65521] * batch, [0, 1, 0][:batch], width)


def test_frozen_dependency_and_route_contract() -> None:
    assert CONFIG["dependency"] == {
        "name": "torchdeq", "version": "0.1.0", "license": "MIT",
        "source": "https://github.com/locuslab/torchdeq",
        "visible_latest_commit_date": "2023-10-31",
        "compatibility": "cpu-pytorch-2.5.1-deterministic-probe-pass",
    }
    assert tuple(CONFIG["routes"]) == ROUTES
    assert tuple(CONFIG["trained_routes"]) == TRAINED_ROUTES
    assert CONFIG["primary_route"] == "anderson_phantom"


def test_proposal_is_bit_identical_to_frozen_source() -> None:
    subject, batch = model(), sample()
    direct = subject.base(batch); proposal = subject(batch, route="proposal_only")
    assert torch.equal(direct[0], proposal.logits)
    assert torch.equal(direct[1], proposal.quotient_logits)
    assert all(not parameter.requires_grad for parameter in subject.base.parameters())


def test_exact_initialization_preserves_proposal_for_deterministic_routes() -> None:
    subject, batch = model(), sample()
    proposal = subject(batch, route="proposal_only")
    for route in ("fixed_depth_shared", "plain_picard_phantom", "anderson_phantom", "anderson_implicit"):
        result = subject(batch, route=route)
        assert torch.equal(proposal.logits, result.logits), route
        assert torch.equal(proposal.quotient_logits, result.quotient_logits), route
        assert torch.equal(result.relative_residual, torch.zeros_like(result.relative_residual)), route


def test_zero_initialized_operator_maps_any_state_to_proposal() -> None:
    subject, batch = model(), sample(); proposal = subject.proposal_state(batch)
    arbitrary = torch.randn_like(proposal)
    assert torch.equal(subject.operator(arbitrary, proposal, batch), proposal)


@pytest.mark.parametrize("route", TRAINED_ROUTES)
def test_each_trainable_route_has_finite_nonzero_gradient(route: str) -> None:
    subject, batch = model(), sample(); subject.train(); generator = torch.Generator().manual_seed(41)
    output = subject(batch, route=route, generator=generator); output.logits.square().mean().backward()
    gradients = [parameter.grad for parameter in subject.correction.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients) > 0


def test_anderson_phantom_and_implicit_are_deterministic_on_cpu() -> None:
    for route in ("anderson_phantom", "anderson_implicit"):
        first, second = model(), model(); batch = sample()
        left = first(batch, route=route); right = second(batch, route=route)
        assert torch.equal(left.logits, right.logits)
        assert torch.equal(left.relative_residual, right.relative_residual)
        left.logits.square().mean().backward(); right.logits.square().mean().backward()
        assert all(torch.equal(a.grad, b.grad) for a, b in zip(first.correction.parameters(), second.correction.parameters(), strict=True))


def test_eqr_noise_breadth_is_deterministic_for_logical_seed() -> None:
    subject, batch = model(), sample(); subject.eval()
    left = subject(batch, route="eqr_finite", generator=torch.Generator().manual_seed(9))
    right = subject(batch, route="eqr_finite", generator=torch.Generator().manual_seed(9))
    assert torch.equal(left.logits, right.logits)
    assert torch.equal(left.relative_residual, right.relative_residual)


def test_correctness_auc_direction_and_constant_baseline() -> None:
    assert correctness_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 1.0
    assert correctness_auc([0.1, 0.2], [True, True]) == 0.5


def test_basin_perturbation_is_target_free_and_bounded() -> None:
    exp300 = load_exp300(); cases = exp300.BigIntSampler([65521], 4).transitions(4)
    payload = {"state": cases[0], "multiplicand": cases[1], "modulus": cases[2], "input_bit": cases[3], "target": cases[4], "quotient": cases[5].tolist(), "width": 16}
    result = basin_diagnostic(model(), "anderson_phantom", payload, 8)
    assert result["cases"] == 4
    assert 0 <= result["decoded_return_fraction"] <= 1
    assert result["distance_contraction_ratio"] >= 0


def test_roots_splits_and_confirmation_authority_are_frozen(tmp_path: Path) -> None:
    calibration = Experiment(CONFIG_PATH, tmp_path, "preregister", 20261210)
    for groups in calibration.splits.values():
        assert set(groups["train"]).isdisjoint(groups["validation"])
        assert set(groups["train"]).isdisjoint(groups["test"])
        assert set(groups["validation"]).isdisjoint(groups["test"])
    with pytest.raises(ValueError): Experiment(CONFIG_PATH, tmp_path, "formal", 20261210)
    assert CONFIG["roots"] == {"calibration": 20261210, "viability": 20261211, "confirmations": [20261212, 20261213]}


def test_one_update_checkpoint_roundtrip(tmp_path: Path) -> None:
    local = copy.deepcopy(CONFIG); local["training"]["batch_size"] = 2; local["training"]["schedule"] = [16]; local["data"]["training_widths"] = [16]
    local["data"]["widths"] = [16]; local["data"]["prime_bit_span"] = 4; local["data"]["train_prime_identities"] = 8; local["data"]["validation_prime_identities"] = 2; local["data"]["test_prime_identities"] = 2
    exp300 = load_exp300(); raw = exp300.bigint_prime_split({"data": {"prime_bits_min": 13, "prime_bits_max": 16, "train_prime_identities": 8, "test_prime_identities": 4}}, 66501)
    split = {16: {"train": raw["train"], "validation": raw["test"][:2], "test": raw["test"][2:]}}
    _, record = train_route(local, exp300, SOURCE, split, 20261210, "anderson_phantom", 1, tmp_path, "calibration")
    assert record["training"]["updates"] == 1
    assert record["checkpoint_reload_exact"]
    assert record["checkpoint"]["bytes"] > 0


def test_no_prohibited_arithmetic_or_answer_dependent_inference() -> None:
    source = inspect.getsource(__import__("experiments.exp_665_attractor_equilibrium_family.model", fromlist=["*"]))
    tree = ast.parse(source)
    forbidden_names = {"pow", "divmod", "crt", "mod_inverse"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_names for node in ast.walk(tree))
    forward_source = inspect.getsource(AttractorCell.forward)
    assert "target" not in forward_source and "modulus" not in forward_source and "answer" not in forward_source
    assert "%" not in forward_source and "//" not in forward_source


def test_checkpoint_geometry_and_parameter_matching() -> None:
    subject = model(); source = load_file(SOURCE)
    assert set(source) == set(subject.base.state_dict())
    assert subject.trainable_parameter_count() > 0
    assert len({model().trainable_parameter_count() for _ in TRAINED_ROUTES}) == 1
