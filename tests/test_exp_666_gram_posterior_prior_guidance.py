from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file
from torch.distributions import Independent, Normal, kl_divergence

from experiments.exp_666_gram_posterior_prior_guidance.model import ROUTES, TRAINED_ROUTES, GRAMCell
from experiments.exp_666_gram_posterior_prior_guidance.run import Experiment, bits, build_model, features, load_exp300, train_route, variational_diagnostic

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp_666_gram_posterior_prior_guidance.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SOURCE = ROOT / CONFIG["source_checkpoint"]


def model() -> GRAMCell:
    torch.manual_seed(666)
    return build_model(CONFIG, SOURCE)


def sample(width: int = 16, batch: int = 3) -> torch.Tensor:
    return features(list(range(batch)), [7] * batch, [65521] * batch, [0, 1, 0][:batch], width)


def targets(width: int = 16, batch: int = 3) -> torch.Tensor:
    return bits([3, 5, 9][:batch], width)


def test_frozen_source_paper_roots_and_route_contract() -> None:
    assert CONFIG["paper"]["sha256"] == "c53be1192d945358fc0fa31cab4c9d8ffc09fbd686340161de504ebeb24de0f7"
    assert CONFIG["source_checkpoint_sha256"] == "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
    assert tuple(CONFIG["routes"]) == ROUTES
    assert tuple(CONFIG["trained_routes"]) == TRAINED_ROUTES
    assert CONFIG["roots"] == {"calibration": 20261214, "viability": 20261215, "confirmations": [20261216, 20261217]}
    assert CONFIG["inference"] == {"trajectories": 1, "fixed_seed": 66624001, "target_free_prior_only": True, "adaptive_stopping": False, "lprm_or_majority": False}


def test_proposal_is_bit_identical_to_frozen_source() -> None:
    subject, batch = model(), sample()
    direct = subject.base(batch)
    proposal = subject(batch, route="proposal_only")
    assert torch.equal(direct[0], proposal.logits)
    assert torch.equal(direct[1], proposal.quotient_logits)
    assert all(not parameter.requires_grad for parameter in subject.base.parameters())


@pytest.mark.parametrize("route", ROUTES[:-1])
def test_exact_initialization_preserves_proposal(route: str) -> None:
    subject, batch = model(), sample()
    proposal = subject(batch, route="proposal_only")
    result = subject(batch, route=route, generator=torch.Generator().manual_seed(41))
    assert torch.equal(result.logits, proposal.logits)
    assert torch.equal(result.quotient_logits, proposal.quotient_logits)


def test_inference_signature_is_structurally_target_free() -> None:
    signature = inspect.signature(GRAMCell.forward)
    assert "target" not in signature.parameters and "answer" not in signature.parameters
    source = inspect.getsource(GRAMCell.forward)
    assert "forward_train" not in source and "target_bits" not in source
    assert "target_bits" in inspect.signature(GRAMCell.forward_train).parameters


def test_native_reparameterization_and_kl_are_finite_differentiable() -> None:
    mean = torch.tensor([[0.1, -0.2]], requires_grad=True)
    scale = torch.tensor([[0.4, 0.7]], requires_grad=True)
    posterior = Independent(Normal(mean, scale), 1)
    prior = Independent(Normal(torch.zeros_like(mean), torch.ones_like(scale)), 1)
    sample_value = posterior.rsample()
    loss = sample_value.square().mean() + kl_divergence(posterior, prior).mean()
    loss.backward()
    assert mean.grad is not None and scale.grad is not None
    assert torch.isfinite(mean.grad).all() and torch.isfinite(scale.grad).all()


def test_posterior_uses_targets_only_on_training_route() -> None:
    subject, batch = model(), sample()
    with torch.no_grad():
        subject.posterior_mean_delta.weight.fill_(0.01)
        subject.guidance_output.weight.fill_(0.01)
    left = subject.forward_train(batch, targets(), route="gram_full", generator=torch.Generator().manual_seed(7))
    right = subject.forward_train(batch, targets().flip(0), route="gram_full", generator=torch.Generator().manual_seed(7))
    prior = subject(batch, route="gram_full", generator=torch.Generator().manual_seed(7))
    assert left.posterior_mean is not None
    assert not torch.equal(left.posterior_mean, right.posterior_mean)
    assert prior.posterior_mean is None and prior.posterior_scale is None


def test_fixed_seed_prior_replay_and_single_trajectory() -> None:
    subject, batch = model(), sample()
    with torch.no_grad(): subject.guidance_output.weight.fill_(0.01)
    left = subject(batch, route="gram_full", generator=torch.Generator().manual_seed(CONFIG["inference"]["fixed_seed"]))
    right = subject(batch, route="gram_full", generator=torch.Generator().manual_seed(CONFIG["inference"]["fixed_seed"]))
    assert torch.equal(left.state, right.state)
    assert torch.equal(left.logits, right.logits)
    assert left.iterations.unique().tolist() == [CONFIG["model"]["supervision_steps"] * CONFIG["model"]["transitions_per_supervision"]]


def test_deep_supervision_and_truncation_are_explicit() -> None:
    output = model().forward_train(sample(), targets(), route="gram_full", generator=torch.Generator().manual_seed(5))
    assert len(output.supervision_logits) == CONFIG["model"]["supervision_steps"]
    assert all(item.shape == (3, 16) for item in output.supervision_logits)
    source = inspect.getsource(GRAMCell._run)
    assert source.count(".detach()") >= 2


@pytest.mark.parametrize("route", TRAINED_ROUTES)
def test_each_trained_route_has_finite_nonzero_gradient(route: str) -> None:
    subject, batch = model(), sample(); subject.train()
    output = subject.forward_train(batch, targets(), route=route, generator=torch.Generator().manual_seed(41))
    loss = sum(item.square().mean() for item in output.supervision_logits) + output.kl.mean() * 0.001
    loss.backward()
    gradients = [parameter.grad for parameter in subject.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients) > 0


def test_randomized_intervention_changes_guidance_state() -> None:
    subject, batch = model(), sample(); baseline = subject(batch, route="gram_full", generator=torch.Generator().manual_seed(3))
    subject.randomized_guidance(66691)
    changed = subject(batch, route="randomized_guidance", generator=torch.Generator().manual_seed(3))
    assert not torch.equal(changed.state, baseline.state)


def test_variational_diagnostic_records_target_free_prior_and_shuffle() -> None:
    subject = model()
    with torch.no_grad():
        subject.posterior_mean_delta.weight.fill_(0.01)
        subject.guidance_output.weight.fill_(0.01)
    raw = load_exp300().BigIntSampler([65521], 4).transitions(4)
    cases = {"state": raw[0], "multiplicand": raw[1], "modulus": raw[2], "input_bit": raw[3], "target": raw[4], "quotient": raw[5].tolist(), "width": 16}
    record = variational_diagnostic(subject, "gram_full", cases, 8)
    assert record["target_free_inference"] and record["single_prior_trajectory"]
    assert record["target_shuffle_state_distance"] >= 0
    assert record["prior_scale_min"] > 0


def test_roots_and_fresh_splits_are_frozen(tmp_path: Path) -> None:
    experiment = Experiment(CONFIG_PATH, tmp_path, "preregister", 20261214)
    for groups in experiment.splits.values():
        assert set(groups["train"]).isdisjoint(groups["validation"])
        assert set(groups["train"]).isdisjoint(groups["test"])
        assert set(groups["validation"]).isdisjoint(groups["test"])
    with pytest.raises(ValueError): Experiment(CONFIG_PATH, tmp_path, "formal", 20261214)


def test_one_update_checkpoint_roundtrip(tmp_path: Path) -> None:
    local = copy.deepcopy(CONFIG); local["training"]["batch_size"] = 2; local["training"]["schedule"] = [16]; local["data"]["training_widths"] = [16]
    local["data"]["widths"] = [16]; local["data"]["prime_bit_span"] = 4; local["data"]["train_prime_identities"] = 8; local["data"]["validation_prime_identities"] = 2; local["data"]["test_prime_identities"] = 2
    exp300 = load_exp300(); raw = exp300.bigint_prime_split({"data": {"prime_bits_min": 13, "prime_bits_max": 16, "train_prime_identities": 8, "test_prime_identities": 4}}, 66601)
    split = {16: {"train": raw["train"], "validation": raw["test"][:2], "test": raw["test"][2:]}}
    _, record = train_route(local, exp300, SOURCE, split, 20261214, "gram_full", 1, tmp_path, "calibration")
    assert record["training"]["updates"] == 1 and record["checkpoint_reload_exact"]


def test_no_prohibited_arithmetic_or_answer_dependent_inference() -> None:
    source = inspect.getsource(__import__("experiments.exp_666_gram_posterior_prior_guidance.model", fromlist=["*"]))
    tree = ast.parse(source)
    forbidden_names = {"pow", "divmod", "crt", "mod_inverse"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_names for node in ast.walk(tree))
    inference = inspect.getsource(GRAMCell.forward)
    inference_tree = ast.parse(textwrap.dedent(inference))
    names = {node.id for node in ast.walk(inference_tree) if isinstance(node, ast.Name)}
    assert names.isdisjoint({"target", "target_bits", "answer", "modulus"})
    assert "%" not in inference and "//" not in inference


def test_checkpoint_geometry_and_parameter_matching() -> None:
    subject = model(); source = load_file(SOURCE)
    assert set(source) == set(subject.base.state_dict())
    assert subject.trainable_parameter_count() > 0
    assert len({model().trainable_parameter_count() for _ in ROUTES}) == 1
