"""Fail-closed future-run scaffold for EXP-658.

T020 authorizes L0 construction and verification only.  This module can verify
the frozen source and instantiate matched arms, but it deliberately has no
optimizer, tracker, root allocation, artifact writer, or scientific evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from .model import JointQuotientCarryTransducer
except ImportError:  # pragma: no cover - direct-script import path
    from model import JointQuotientCarryTransducer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/exp_658_joint_quotient_carry_transducer.yaml"
EXPECTED_SOURCE_SHA256 = (
    "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
)


class L0OnlyError(RuntimeError):
    """Raised whenever code attempts to cross the authorized L0 boundary."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("EXP-658 config must be a mapping")
    return payload


def validate_l0_contract(config: dict[str, Any]) -> Path:
    if config["status"] != "preregistered-l0-only":
        raise L0OnlyError("EXP-658 is not frozen at L0")
    if config["run_control"] != {
        "active_stage": "L0",
        "calibration_root": None,
        "scientific_roots": [],
        "authorization_file": None,
        "mlflow": "prohibited",
        "evidence_registration": "prohibited",
        "registry_edit": "prohibited",
        "governance_edit": "prohibited",
        "submission_edit": "prohibited",
    }:
        raise L0OnlyError("EXP-658 run-control boundary changed")
    source = ROOT / config["source"]["checkpoint"]
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = config["source"]["checkpoint_sha256"]
    if expected != EXPECTED_SOURCE_SHA256 or sha256(source) != expected:
        raise RuntimeError("EXP-609 checkpoint SHA-256 mismatch")
    control = config["arms"]["control"]
    challenger = config["arms"]["challenger"]
    if (
        control["quotient_auxiliary_loss_weight"],
        control["signed_carry_auxiliary_loss_weight"],
        challenger["quotient_auxiliary_loss_weight"],
        challenger["signed_carry_auxiliary_loss_weight"],
    ) != (0.0, 0.0, 0.10, 0.10):
        raise RuntimeError("matched-arm auxiliary weights changed")
    return source


def build_matched_arms(
    config: dict[str, Any],
) -> tuple[JointQuotientCarryTransducer, JointQuotientCarryTransducer]:
    source = validate_l0_contract(config)
    torch.manual_seed(config["seed"])
    control = JointQuotientCarryTransducer(
        hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]
    )
    control.load_exp609_source(source)
    challenger = JointQuotientCarryTransducer(
        hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]
    )
    challenger.load_state_dict(control.state_dict(), strict=True)
    return control, challenger


def require_future_authorization() -> None:
    raise L0OnlyError(
        "EXP-658 T020 authorizes L0 only; calibration, training, roots, MLflow, "
        "evaluation, and evidence publication require a separate Judge package"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config = load_config(arguments.config.resolve())
    validate_l0_contract(config)
    require_future_authorization()
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except L0OnlyError as error:
        raise SystemExit(str(error)) from error
