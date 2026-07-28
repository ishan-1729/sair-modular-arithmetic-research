from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch
import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/exp_674_latent_thought_regularization.yaml"
FEASIBILITY = ROOT / "experiments/exp_674_latent_thought_regularization/feasibility.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_frozen_source_hashes_and_roots_are_unopened() -> None:
    cfg = config()
    source = cfg["source"]
    for key, expected in (
        ("paper", source["paper_sha256"]),
        ("checkpoint", source["checkpoint_sha256"]),
        ("model", source["model_sha256"]),
    ):
        observed = hashlib.sha256((ROOT / source[key]).read_bytes()).hexdigest()
        assert observed == expected
    assert set(cfg["frozen_counterfactual_package"]["roots"]) == {
        20261246,
        20261247,
        20261248,
        20261249,
    }
    assert all(
        value == "unopened-topology-feasibility-blocked"
        for value in cfg["frozen_counterfactual_package"]["roots"].values()
    )


def test_visual_source_pages_define_weight_not_hidden_regularization() -> None:
    cfg = config()
    reader = PdfReader(ROOT / cfg["source"]["paper"])
    page_ten = reader.pages[9].extract_text() or ""
    assert "regularize the weights" in page_ten
    assert "parameter group" in page_ten
    assert "successive blocks" in page_ten
    assert cfg["paper_mechanism"]["hidden_trajectory_target"] == "absent"
    assert cfg["paper_mechanism"]["external_reference_model"] == "absent"


def test_cosine_math_and_self_gradient_are_zero() -> None:
    module = load_module("exp674_feasibility_math", FEASIBILITY)
    left = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
    right = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
    distance = module.cosine_distance(left, right)
    assert distance.item() == 0.0
    distance.backward()
    assert torch.equal(left.grad, torch.zeros_like(left))
    assert torch.equal(right.grad, torch.zeros_like(right))


def test_literal_tied_parameter_objective_has_no_update_causality() -> None:
    module = load_module("exp674_feasibility_gradient", FEASIBILITY)
    cfg = config()
    model_module = module.load_module("exp674_test_source", ROOT / cfg["source"]["model"])
    model = module._new_model(model_module, ROOT / cfg["source"]["checkpoint"], 128)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.zero_grad(set_to_none=True)
    objective = module.shared_self_regularizer(model)
    objective.backward()
    optimizer.step()
    assert abs(objective.item()) <= module.NUMERICAL_ZERO_TOLERANCE
    assert max(
        float((before[name] - value).abs().max()) for name, value in model.state_dict().items()
    ) <= module.NUMERICAL_ZERO_TOLERANCE


def test_source_formal_reload_architecture_state_calls_and_outputs_are_exact() -> None:
    module = load_module("exp674_feasibility_identity", FEASIBILITY)
    cfg = config()
    model_module = module.load_module("exp674_identity_source", ROOT / cfg["source"]["model"])
    receipt = module.formal_identity(model_module, ROOT / cfg["source"]["checkpoint"])
    assert receipt["architecture"] == "SharedDilatedConvGRU"
    assert receipt["parameter_count"] == 99824
    assert receipt["state_dict_tensor_identity"] is True
    assert len(receipt["state_dict_keys"]) == 16
    for width, row in receipt["widths"].items():
        assert row["left_recurrent_calls"] == row["right_recurrent_calls"]
        assert row["left_recurrent_calls"] == row["expected_recurrent_calls"]
        assert row["logits_bit_identical"] is True, width
        assert row["quotient_logits_bit_identical"] is True, width


def test_feasibility_oracle_is_deterministic_and_opens_no_science() -> None:
    module = load_module("exp674_feasibility_oracle", FEASIBILITY)
    first = module.run(CONFIG)
    second = module.run(CONFIG)
    assert first == second
    assert abs(first["literal_self_cosine_distance"]) <= first["numerical_zero_tolerance"]
    assert first["literal_self_cosine_max_parameter_gradient"] <= first["numerical_zero_tolerance"]
    assert first["training_updates"] == 0
    assert first["formal_cases"] == 0
    assert first["scientific_metrics"] is None
    assert first["scientific_verdict"] is None


def test_scope_excludes_huginn_and_prohibited_surfaces() -> None:
    cfg = config()
    assert cfg["safety"]["M006a_huginn"] == "excluded"
    assert cfg["safety"]["external_execution"] == "forbidden"
    assert cfg["safety"]["checked_independent_official_private_candidate_submission_promotion"] == "forbidden"
    source = FEASIBILITY.read_text(encoding="utf-8").lower()
    for prohibited in ("barrett", "montgomery", "chinese remainder", "private benchmark"):
        assert prohibited not in source
