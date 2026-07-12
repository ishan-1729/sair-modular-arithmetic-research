"""Local MLflow setup and durable incumbent bootstrap for autoresearch."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "sair-modular-arithmetic"
REGISTERED_MODEL = "sair-modular-arithmetic-candidate"
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracking_uri() -> str:
    database = (ROOT / ".scratch" / "mlflow" / "mlflow.db").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.as_posix()}"


def configure_mlflow(experiment_name: str = DEFAULT_EXPERIMENT) -> str:
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_root = (ROOT / ".scratch" / "mlflow" / "artifacts").resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.as_uri(),
            tags={
                "project": "sair-modular-arithmetic-research",
                "visibility": "local-private",
            },
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def find_canonical_run(client: MlflowClient, experiment_id: str, run_id: str):
    matches = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.canonical_run_id = '{run_id}'",
        max_results=10,
    )
    return matches[0] if matches else None


def bootstrap_incumbent() -> dict[str, Any]:
    experiment_id = configure_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri())
    canonical_run_id = "candidate-001-horner-tcn9"
    existing = find_canonical_run(client, experiment_id, canonical_run_id)
    if existing is not None:
        return {
            "status": "existing",
            "experiment_id": experiment_id,
            "mlflow_run_id": existing.info.run_id,
            "tracking_uri": tracking_uri(),
        }

    checked = json.loads(
        (ROOT / "evals/results/candidate-001-horner-tcn9/checked-public-1100.json")
        .read_text(encoding="utf-8")
    )
    independent = json.loads(
        (ROOT / "evals/results/candidate-001-horner-tcn9/generated-independent-1100.json")
        .read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (ROOT / "evals/results/candidate-001-horner-tcn9/metrics.json")
        .read_text(encoding="utf-8")
    )
    run_record = json.loads(
        (ROOT / "experiments/runs/candidate-001-horner-tcn9/run.json")
        .read_text(encoding="utf-8")
    )
    git_commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))

    with mlflow.start_run(run_name=canonical_run_id) as active:
        run_id = active.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": canonical_run_id,
                "hypothesis": run_record["hypothesis"],
                "status": "completed",
                "role": "incumbent",
                "ab_group": "historical-incumbent",
                "promotion_gate": "validated-highest-tier-2",
                "challenge_commit": run_record["source"]["challenge_commit"],
                "repository_commit": git_commit,
                "repository_dirty": str(dirty).lower(),
                "canonical_run_record": "experiments/runs/candidate-001-horner-tcn9/run.json",
                "canonical_metrics": "evals/results/candidate-001-horner-tcn9/metrics.json",
            }
        )
        mlflow.log_params(
            {
                "architecture": metrics["model"]["architecture"],
                "parameters": metrics["model"]["parameters"],
                "output_base": metrics["model"]["output_base"],
                "seed": run_record["seed"],
                "training_steps": run_record["training"]["steps"],
                "batch_size": run_record["training"]["batch_size"],
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "artifact_bytes": metrics["runtime"]["artifact_bytes_clean"],
                "dataset_kind": checked["benchmark"]["kind"],
            }
        )
        mlflow.log_metrics(
            {
                "transition_exact": metrics["evaluation"]["transition"]["synthetic_heldout"]["exact_accuracy"],
                "rollout_tier2_synthetic_exact": metrics["evaluation"]["rollout"]["synthetic_tier2_8step"]["exact_accuracy"],
                "checked_overall_accuracy": checked["result"]["overall_accuracy"],
                "checked_highest_tier_above_90": checked["result"]["highest_tier_above_90"],
                "checked_tier1_accuracy": checked["result"]["tiers"][1]["accuracy"],
                "checked_tier2_accuracy": checked["result"]["tiers"][2]["accuracy"],
                "independent_overall_accuracy": independent["overall_accuracy"],
                "independent_highest_tier_above_90": independent["highest_tier_above_90"],
                "independent_tier1_accuracy": independent["tiers"][1]["accuracy"],
                "independent_tier2_accuracy": independent["tiers"][2]["accuracy"],
                "sandbox_inference_seconds": metrics["runtime"]["sandbox_inference_seconds"],
                "randomized_tier2_accuracy": metrics["controls"]["randomized_weights"]["checked_public_tier_2_accuracy"],
            }
        )
        for filename, digest in checked["benchmark"]["sha256"].items():
            mlflow.set_tag(f"dataset_sha256.{filename}", digest)
        mlflow.log_artifacts(str(ROOT / "submission"), artifact_path="submission")
        for path in (
            ROOT / "configs/candidate_001_horner_tcn9.yaml",
            ROOT / "experiments/runs/candidate-001-horner-tcn9/run.json",
            ROOT / "experiments/runs/candidate-001-horner-tcn9/environment.json",
            ROOT / "evals/results/candidate-001-horner-tcn9/metrics.json",
            ROOT / "evals/results/candidate-001-horner-tcn9/checked-public-1100.json",
            ROOT / "evals/results/candidate-001-horner-tcn9/generated-independent-1100.json",
        ):
            mlflow.log_artifact(str(path), artifact_path="canonical")
        mlflow.log_dict(
            {
                "git_commit": git_commit,
                "dirty": dirty,
                "checkpoint_sha256": sha256(ROOT / "submission/weights.safetensors"),
                "limitations": metrics["limitations"],
            },
            "diagnostics/bootstrap.json",
        )

    try:
        client.create_registered_model(
            REGISTERED_MODEL,
            description="Promoted SAIR modular-arithmetic submission candidates.",
            tags={"project": "sair-modular-arithmetic-research"},
        )
    except MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise
    version = client.create_model_version(
        name=REGISTERED_MODEL,
        source=f"runs:/{run_id}/submission",
        run_id=run_id,
        description=(
            "candidate-001-horner-tcn9; weights sha256 "
            "22c78b5d1db0136a8cb93d6c010cce025c251d70b62baef79bebc50e016af585"
        ),
        tags={"canonical_run_id": canonical_run_id},
    )
    client.set_registered_model_alias(REGISTERED_MODEL, "incumbent", version.version)
    client.set_registered_model_alias(REGISTERED_MODEL, "validated", version.version)
    return {
        "status": "created",
        "experiment_id": experiment_id,
        "mlflow_run_id": run_id,
        "registered_model": REGISTERED_MODEL,
        "model_version": version.version,
        "tracking_uri": tracking_uri(),
    }


def register_candidate_002() -> dict[str, Any]:
    """Track and register the independently promoted Tier-3 hybrid."""
    experiment_id = configure_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri())
    canonical_run_id = "candidate-002-tier3-hybrid"
    existing = find_canonical_run(client, experiment_id, canonical_run_id)
    if existing is None:
        metrics = json.loads(
            (ROOT / "evals/results/candidate-002-tier3-hybrid/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        with mlflow.start_run(run_name=canonical_run_id) as active:
            run_id = active.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": canonical_run_id,
                    "status": "completed",
                    "role": "incumbent",
                    "promotion_gate": "validated-highest-tier-3",
                    "challenge_commit": CHALLENGE_COMMIT,
                    "repository_commit": git_value("rev-parse", "HEAD"),
                    "repository_dirty": str(bool(git_value("status", "--porcelain"))).lower(),
                    "checked_public_selection_surface": "false",
                }
            )
            mlflow.log_params(
                {
                    "architecture": "Tier2Tier3HornerEnsemble",
                    "parameters": metrics["runtime"]["parameters"],
                    "checkpoint_sha256": metrics["checkpoint_sha256"],
                    "artifact_bytes_clean": metrics["runtime"]["artifact_bytes_clean"],
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                }
            )
            independent = metrics["evaluation"]["rollout"]["independent_official_scored_tiers"]
            checked = metrics["evaluation"]["rollout"]["checked_public_scored_tiers"]
            mlflow.log_metrics(
                {
                    "independent_overall_accuracy": independent["exact_accuracy"],
                    "independent_highest_tier_above_90": independent["highest_tier_above_90"],
                    "independent_tier1_accuracy": independent["tier_1_accuracy"],
                    "independent_tier2_accuracy": independent["tier_2_accuracy"],
                    "independent_tier3_accuracy": independent["tier_3_accuracy"],
                    "checked_overall_accuracy": checked["exact_accuracy"],
                    "checked_tier3_accuracy": checked["tier_3_accuracy"],
                    "randomized_tier3_accuracy": metrics["controls"]["randomized_weights_tier_3_accuracy"],
                    "inference_seconds": metrics["runtime"]["inference_seconds"],
                }
            )
            mlflow.log_artifacts(str(ROOT / "submission"), artifact_path="submission")
            mlflow.log_artifacts(
                str(ROOT / "experiments/runs/candidate-002-tier3-hybrid"),
                artifact_path="canonical/run",
            )
            mlflow.log_artifact(
                str(ROOT / "evals/results/candidate-002-tier3-hybrid/metrics.json"),
                artifact_path="canonical/eval",
            )
    else:
        run_id = existing.info.run_id

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    matching = next(
        (
            item
            for item in versions
            if item.tags.get("canonical_run_id") == canonical_run_id
        ),
        None,
    )
    if matching is None:
        matching = client.create_model_version(
            name=REGISTERED_MODEL,
            source=f"runs:/{run_id}/submission",
            run_id=run_id,
            description=(
                "candidate-002-tier3-hybrid; weights sha256 "
                "69a53615904386bfeb6d07c0ab79c612297e7c18a706ca2058545b0cc2271157"
            ),
            tags={"canonical_run_id": canonical_run_id},
        )
    client.set_registered_model_alias(REGISTERED_MODEL, "incumbent", matching.version)
    client.set_registered_model_alias(REGISTERED_MODEL, "validated", matching.version)
    return {
        "status": "registered",
        "experiment_id": experiment_id,
        "mlflow_run_id": run_id,
        "registered_model": REGISTERED_MODEL,
        "model_version": matching.version,
        "tracking_uri": tracking_uri(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("bootstrap-incumbent", "register-candidate-002", "show")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "bootstrap-incumbent":
        payload = bootstrap_incumbent()
    elif args.command == "register-candidate-002":
        payload = register_candidate_002()
    else:
        experiment_id = configure_mlflow()
        client = MlflowClient(tracking_uri=tracking_uri())
        runs = client.search_runs([experiment_id], order_by=["attributes.start_time DESC"])
        payload = {
            "tracking_uri": tracking_uri(),
            "experiment_id": experiment_id,
            "runs": [
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "canonical_run_id": run.data.tags.get("canonical_run_id"),
                    "role": run.data.tags.get("role"),
                }
                for run in runs
            ],
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
