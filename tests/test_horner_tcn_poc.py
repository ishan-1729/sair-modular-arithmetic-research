from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import torch

from pocs.learned_horner_tcn.config import load_config
from pocs.learned_horner_tcn.data import split_primes, training_moduli
from pocs.learned_horner_tcn.evaluate import load_checkpoint
from pocs.learned_horner_tcn.evaluation import evaluate_rollout
from pocs.learned_horner_tcn.model import make_features


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "poc_horner_tcn_smoke.yaml"
CHECKPOINT = ROOT / "pocs" / "learned_horner_tcn" / "artifacts" / "poc_weights.pt"


def test_prime_identity_holdout_has_no_training_leakage() -> None:
    config = load_config(CONFIG)
    train_primes, heldout = split_primes(config)
    values = training_moduli(config, train_primes, heldout)
    assert heldout == [11, 23, 29, 37, 61]
    assert set(values).isdisjoint(heldout)
    assert any(value not in train_primes for value in values)


def test_inference_model_source_contains_no_modulo_operator() -> None:
    source = (ROOT / "pocs" / "learned_horner_tcn" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    inference_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"make_features", "forward", "predict_state"}
    ]
    assert inference_functions
    assert not any(
        isinstance(child, ast.Mod)
        for function in inference_functions
        for child in ast.walk(function)
    )


def test_reference_checkpoint_shape_hash_and_forward_contract() -> None:
    config = load_config(CONFIG)
    model, loaded_config, _, _ = load_checkpoint(CHECKPOINT, config)
    assert loaded_config.width == 8
    assert model.parameter_count() == 21_441
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == (
        "1ebee3ae0042bb76995221f582ee200685c10d18ec7612928dfc2e23126f09fc"
    )
    state = torch.tensor([0, 1], dtype=torch.int64)
    multiplicand = torch.tensor([7, 9], dtype=torch.int64)
    modulus = torch.tensor([11, 23], dtype=torch.int64)
    bit = torch.tensor([0, 1], dtype=torch.int64)
    logits = model(make_features(state, multiplicand, modulus, bit, loaded_config.width))
    assert logits.shape == (2, 8)


def test_reference_checkpoint_has_nontrivial_heldout_rollout() -> None:
    config = load_config(CONFIG)
    model, loaded_config, _, heldout = load_checkpoint(CHECKPOINT, config)
    result = evaluate_rollout(
        model,
        heldout,
        loaded_config,
        operand_bits=8,
        seed=loaded_config.seed + 108,
        cases=256,
    )
    assert result["exact_accuracy"] >= 0.80
