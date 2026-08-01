"""Authenticate and falsify the EXP674 paper-faithful topology mapping.

The source regularizer acts on corresponding weight tensors in successive
untied blocks. Candidate005 has one recurrent parameter set shared across all
visits. Applying the source objective literally therefore compares every
tensor with itself and has no training signal. This module measures that fact;
it does not implement a hidden-state teacher or alter formal inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
NUMERICAL_ZERO_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return one minus flattened cosine similarity."""

    # Evaluate the topology identity in float64 so float32 reduction noise is
    # not mistaken for a scientific training signal. Gradients still flow to
    # the original tensors through the dtype conversion.
    return 1.0 - F.cosine_similarity(
        left.reshape(-1).double(), right.reshape(-1).double(), dim=0
    )


def shared_self_regularizer(model: torch.nn.Module) -> torch.Tensor:
    """Literal Candidate005 image of the paper's adjacent-weight objective."""

    terms = [cosine_distance(parameter, parameter) for parameter in model.parameters()]
    return torch.stack(terms).mean()


def _new_model(model_module, checkpoint: Path, width: int):
    model = model_module.SharedDilatedConvGRU(width=width, hidden=92, cycles=2)
    model.load_state_dict(load_file(str(checkpoint)))
    return model


def formal_identity(model_module, checkpoint: Path) -> dict:
    """Check exact source reload/output/schema/call identity at all research widths."""

    records = {}
    for width in (64, 128, 256):
        left = _new_model(model_module, checkpoint, width).eval()
        right = _new_model(model_module, checkpoint, width).eval()
        generator = torch.Generator().manual_seed(674_000 + width)
        features = torch.randn((2, 6, width), generator=generator)
        left_calls = 0
        right_calls = 0
        left_update = left.recurrent_update
        right_update = right.recurrent_update

        def count_left(*args, **kwargs):
            nonlocal left_calls
            left_calls += 1
            return left_update(*args, **kwargs)

        def count_right(*args, **kwargs):
            nonlocal right_calls
            right_calls += 1
            return right_update(*args, **kwargs)

        with torch.inference_mode(), patch.object(left, "recurrent_update", count_left), patch.object(
            right, "recurrent_update", count_right
        ):
            left_outputs = left(features)
            right_outputs = right(features)
        expected_calls = len(left.dilation_schedule(width))
        records[str(width)] = {
            "expected_recurrent_calls": expected_calls,
            "left_recurrent_calls": left_calls,
            "right_recurrent_calls": right_calls,
            "logits_bit_identical": torch.equal(left_outputs[0], right_outputs[0]),
            "quotient_logits_bit_identical": torch.equal(left_outputs[1], right_outputs[1]),
        }

    first = _new_model(model_module, checkpoint, 128)
    second = _new_model(model_module, checkpoint, 128)
    keys = list(first.state_dict())
    return {
        "architecture": type(first).__name__,
        "parameter_count": sum(parameter.numel() for parameter in first.parameters()),
        "state_dict_keys": keys,
        "state_dict_tensor_identity": all(
            torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in keys
        ),
        "widths": records,
    }


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    paper = ROOT / source["paper"]
    checkpoint = ROOT / source["checkpoint"]
    model_path = ROOT / source["model"]
    observed_hashes = {
        "paper_sha256": sha256(paper),
        "checkpoint_sha256": sha256(checkpoint),
        "model_sha256": sha256(model_path),
    }
    expected_hashes = {key: source[key] for key in observed_hashes}
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"source hash mismatch: {observed_hashes!r}")

    model_module = load_module("exp674_candidate_source", model_path)
    model = _new_model(model_module, checkpoint, 128)
    model.zero_grad(set_to_none=True)
    objective = shared_self_regularizer(model)
    objective.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    maximum_gradient = max(float(gradient.abs().max()) for gradient in gradients if gradient is not None)
    paper_text = " ".join(
        (page.extract_text() or "")
        for page in __import__("pypdf").PdfReader(paper).pages[8:10]
    )
    source_terms_present = all(
        term in paper_text
        for term in ("regularize the weights", "parameter group", "successive blocks")
    )
    result = {
        "run_id": config["run_id"],
        "status": "completed-parked-topology-feasibility",
        "decision": config["decision"],
        "source_hashes": observed_hashes,
        "source_terms_present": source_terms_present,
        "paper_object": config["paper_mechanism"]["object"],
        "candidate_sharing": config["candidate_topology"]["sharing"],
        "literal_self_cosine_distance": float(objective.detach()),
        "literal_self_cosine_max_parameter_gradient": maximum_gradient,
        "numerical_zero_tolerance": NUMERICAL_ZERO_TOLERANCE,
        "formal_source_identity": formal_identity(model_module, checkpoint),
        "training_updates": 0,
        "formal_cases": 0,
        "roots": config["frozen_counterfactual_package"]["roots"],
        "scientific_metrics": None,
        "scientific_verdict": None,
        "limitations": [
            "The source regularizer acts on weights, not hidden trajectories.",
            "A training-only untied hidden-state reference would be a different distillation mechanism.",
            "No calibration, training, W128/W256 evaluation, or confirmation was opened.",
        ],
    }
    if (
        not source_terms_present
        or objective.detach().abs().item() > NUMERICAL_ZERO_TOLERANCE
        or maximum_gradient > NUMERICAL_ZERO_TOLERANCE
    ):
        raise RuntimeError("topology feasibility oracle did not reproduce the frozen finding")
    if not all(
        row["left_recurrent_calls"] == row["right_recurrent_calls"] == row["expected_recurrent_calls"]
        and row["logits_bit_identical"]
        and row["quotient_logits_bit_identical"]
        for row in result["formal_source_identity"]["widths"].values()
    ):
        raise RuntimeError("source formal identity check failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/exp_674_latent_thought_regularization.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.config.resolve())
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
