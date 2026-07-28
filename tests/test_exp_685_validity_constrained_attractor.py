from __future__ import annotations

import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from experiments.exp_665_attractor_equilibrium_family.model import AttractorCell
from experiments.exp_685_validity_constrained_attractor.model import (
    build_model,
    deterministic_modulus_permutation,
    lexicographic_invalid_probability,
    supervised_objective,
)
from experiments.exp_685_validity_constrained_attractor.run import ARM_ORDER, Experiment, TRAINED_ARMS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp_685_validity_constrained_attractor.yaml"
CHECKPOINT = ROOT / ".scratch/exp-685-validity-constrained-attractor/fixtures/exp665-anderson-phantom.safetensors"


def config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def bits_lsb(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor([[(value >> index) & 1 for index in range(width)] for value in values], dtype=torch.float32)


def brute_invalid(probabilities: list[float], modulus: int) -> float:
    total = 0.0
    for row in itertools.product((0, 1), repeat=len(probabilities)):
        value = sum(bit << index for index, bit in enumerate(row))
        mass = 1.0
        for bit, probability in zip(row, probabilities, strict=True):
            mass *= probability if bit else 1.0 - probability
        if value >= modulus:
            total += mass
    return total


def test_frozen_manifest_and_routes():
    data = config()
    assert data["source_checkpoint_sha256"] == "a06103dbe216bd9ae2570cf78f7bdb017a10398bc1e3fa4f9a667101eb1d25e4"
    assert data["training"]["validity_weight"] == 0.05
    assert data["training"]["steps"] == 200
    assert data["training"]["schedule"] == [64, 128, 256, 128]
    assert tuple(data["routes"]) == ARM_ORDER
    assert TRAINED_ARMS == ("validity_constrained", "ce_only", "shuffled_modulus")
    assert data["roots"] == {"calibration": 20261290, "formal": 20261291, "confirmations": [20261292, 20261293]}


def test_lexicographic_probability_matches_bruteforce():
    probabilities = [0.2, 0.7, 0.4]
    logits = torch.logit(torch.tensor([probabilities], dtype=torch.float64))
    for modulus in range(1, 8):
        observed = float(lexicographic_invalid_probability(logits, bits_lsb([modulus], 3)))
        assert abs(observed - brute_invalid(probabilities, modulus)) < 1e-12


def test_lexicographic_boundary_includes_equality():
    logits = torch.tensor([[-30.0, 30.0, -30.0]])  # binary 010 = 2
    equal = float(lexicographic_invalid_probability(logits, bits_lsb([2], 3)))
    greater = float(lexicographic_invalid_probability(logits, bits_lsb([3], 3)))
    assert equal > 0.999999
    assert greater < 1e-10


def test_validity_probability_has_finite_nonzero_gradient():
    logits = torch.tensor([[0.2, -0.3, 0.4, -0.5]], requires_grad=True)
    loss = lexicographic_invalid_probability(logits, bits_lsb([9], 4)).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.norm()) > 0.0


def test_shuffled_modulus_is_deterministic_and_nonidentity():
    first = deterministic_modulus_permutation(32, 68591, torch.device("cpu"))
    second = deterministic_modulus_permutation(32, 68591, torch.device("cpu"))
    assert torch.equal(first, second)
    assert not torch.equal(first, torch.arange(32))
    assert sorted(first.tolist()) == list(range(32))


def test_objective_changes_only_by_frozen_validity_term():
    data = config(); logits = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    output = SimpleNamespace(logits=logits, quotient_logits=torch.tensor([[0.1, 0.7, -0.2]], requires_grad=True), relative_residual=torch.tensor([0.01]))
    target = bits_lsb([3], 3); modulus = bits_lsb([5], 3); quotient = torch.tensor([1])
    primary, primary_parts = supervised_objective(output, target, quotient, modulus, data, kind="validity", permutation_seed=7)
    ce, ce_parts = supervised_objective(output, target, quotient, modulus, data, kind="ce_only", permutation_seed=7)
    assert torch.allclose(primary - ce, 0.05 * primary_parts["validity_loss"])
    assert torch.equal(primary_parts["validity_loss"], ce_parts["validity_loss"])


def test_formal_model_is_exact_exp665_class_with_no_new_parameters():
    data = config(); model = build_model(data, CHECKPOINT)
    assert type(model) is AttractorCell
    assert model.parameter_count() == 142728
    assert model.trainable_parameter_count() == 42904
    forward = inspect.getsource(AttractorCell.forward)
    assert "validity" not in forward and "modulus" not in forward and "repair" not in forward


def test_randomized_control_preserves_architecture_but_changes_weights():
    data = config(); frozen = build_model(data, CHECKPOINT); randomized = build_model(data, CHECKPOINT, randomized_seed=68591)
    assert type(frozen) is type(randomized) is AttractorCell
    assert frozen.state_dict().keys() == randomized.state_dict().keys()
    assert any(not torch.equal(frozen.state_dict()[name], randomized.state_dict()[name]) for name in frozen.state_dict() if name.startswith("correction."))
    assert all(torch.equal(frozen.state_dict()[name], randomized.state_dict()[name]) for name in frozen.state_dict() if name.startswith("base."))


def test_rootless_all_mode_smoke(tmp_path):
    experiment = Experiment(CONFIG_PATH, tmp_path, "smoke", None)
    result = experiment.smoke()
    assert result["root"] is None
    assert result["one_update_each"]
    assert result["one_case_transition_rollout_each"]
    assert result["reload_exact"]
    assert result["admission_math_exercised"]
    assert result["terminal_serializations"] == ["calibration-terminal.json", "formal-terminal.json", "confirmation-terminal.json"]


def test_safety_boundary_keeps_exact_arithmetic_out_of_model_module():
    source = (ROOT / "experiments/exp_685_validity_constrained_attractor/model.py").read_text(encoding="utf-8")
    assert "%" not in source
    assert "decode(" not in source
    assert "modular" not in source.lower()
    assert "lexicographic_invalid_probability" in source
