"""Run the matched 1,200-step random-versus-transfer EXP-106 comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXP100_DIR = ROOT / "experiments" / "exp_100_width_scaling"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP100_DIR))

import model as exp100_model  # noqa: E402
import run as exp100  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_exp105():
    path = ROOT / "experiments" / "exp_105_width_transfer" / "run.py"
    spec = importlib.util.spec_from_file_location("exp105_transfer_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load EXP-105 transfer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def compare_results(
    random_result: dict[str, Any],
    transfer_result: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[str, str, dict[str, float | bool]]:
    random_transition = random_result["evaluation"]["heldout_transition"][
        "exact_accuracy"
    ]
    transfer_transition = transfer_result["evaluation"]["heldout_transition"][
        "exact_accuracy"
    ]
    random_rollout = random_result["evaluation"]["heldout_rollout"][
        "exact_accuracy"
    ]
    transfer_rollout = transfer_result["evaluation"]["heldout_rollout"][
        "exact_accuracy"
    ]
    random_invalid = random_result["evaluation"]["heldout_rollout"][
        "any_invalid_residue_rate"
    ]
    transfer_invalid = transfer_result["evaluation"]["heldout_rollout"][
        "any_invalid_residue_rate"
    ]
    comparison: dict[str, float | bool] = {
        "transition_gain": transfer_transition - random_transition,
        "rollout_gain": transfer_rollout - random_rollout,
        "invalid_rate_change": transfer_invalid - random_invalid,
    }
    transfer_selected = (
        comparison["rollout_gain"]
        >= float(gates["minimum_transfer_rollout_gain"])
        and comparison["transition_gain"]
        >= -float(gates["maximum_transfer_transition_loss"])
        and comparison["invalid_rate_change"]
        <= float(gates["maximum_transfer_invalid_rate_change"])
    )
    winner = "width16-transfer" if transfer_selected else "width16-random"
    winner_result = transfer_result if transfer_selected else random_result
    winner_transition = winner_result["evaluation"]["heldout_transition"][
        "exact_accuracy"
    ]
    winner_rollout = winner_result["evaluation"]["heldout_rollout"][
        "exact_accuracy"
    ]
    comparison["transfer_selected"] = transfer_selected
    comparison["winner_transition"] = winner_transition
    comparison["winner_rollout"] = winner_rollout
    if winner_rollout >= float(gates["independent_gate_rollout"]):
        decision = "advance-independent-tier3-gate"
    elif (
        winner_transition >= float(gates["full_budget_transition"])
        and winner_rollout >= float(gates["full_budget_rollout"])
    ):
        decision = "advance-winner-to-2400"
    else:
        decision = "revise-curriculum-before-full-budget"
    return decision, winner, comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_106_width_transfer_1200.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-106-width-transfer-1200",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_config_path = (ROOT / config["base_config"]).resolve()
    matched = copy.deepcopy(
        yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    )
    matched["run_id"] = config["run_id"]
    matched["training"]["steps"] = int(config["training"]["steps"])
    matched["evaluation"]["heldout_transition_cases"] = int(
        config["evaluation"]["heldout_transition_cases"]
    )
    matched["evaluation"]["rollout_cases"] = int(
        config["evaluation"]["rollout_cases"]
    )
    source_path = (ROOT / config["transfer_checkpoint"]).resolve()
    if sha256(source_path) != config["transfer_checkpoint_sha256"]:
        raise RuntimeError("transfer checkpoint hash mismatch")
    exp105 = load_exp105()
    torch.set_num_threads(int(matched["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    configure_mlflow(matched["tracking"]["experiment_name"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_arm = next(
        item
        for item in matched["comparison"]["arms"]
        if item["width"] == 16
    )
    random_arm = {**base_arm, "name": "width16-random"}
    transfer_arm = {**base_arm, "name": "width16-transfer"}
    original_constructor = exp100.SharedContextHornerTCN
    transfer_report: dict[str, Any] = {}

    def transferred_constructor(
        width: int, hidden: int, blocks: int
    ) -> exp100_model.SharedContextHornerTCN:
        nonlocal transfer_report
        model = original_constructor(width=width, hidden=hidden, blocks=blocks)
        transfer_report = exp105.transfer_width9_to_width16(model, source_path)
        return model

    git_commit = git_output("rev-parse", "HEAD")
    git_dirty = bool(git_output("status", "--porcelain"))
    started = time.perf_counter()
    arm_run_ids: dict[str, str] = {}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "hypothesis": config["hypothesis"],
                "status": "running",
                "role": "challenger-intermediate",
                "baseline_run": "candidate-001-horner-tcn9",
                "ab_group": "EXP-106",
                "promotion_gate": "preregistered-matched-1200",
                "repository_commit": git_commit,
                "repository_dirty": str(git_dirty).lower(),
                "challenge_commit": exp100.CHALLENGE_COMMIT,
            }
        )
        mlflow.log_params(
            {
                "seed": config["seed"],
                "steps": matched["training"]["steps"],
                "batch_size": matched["training"]["batch_size"],
                "transition_cases": matched["evaluation"]["heldout_transition_cases"],
                "rollout_cases": matched["evaluation"]["rollout_cases"],
                "transfer_checkpoint_sha256": sha256(source_path),
                "config_sha256": sha256(config_path),
                "runner_sha256": sha256(Path(__file__)),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(
            str(ROOT / "knowledge/autoresearch/ablations/exp-106-width-transfer-1200.md"),
            artifact_path="preregistration",
        )
        with mlflow.start_run(run_name=random_arm["name"], nested=True) as run:
            arm_run_ids[random_arm["name"]] = run.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": config["run_id"],
                    "parent_run_id": parent_run_id,
                    "ab_group": random_arm["name"],
                    "initialization": "random",
                    "status": "running",
                }
            )
            random_result = exp100.train_arm(
                random_arm, matched, output / random_arm["name"]
            )
            mlflow.set_tag("status", "completed")
        with mlflow.start_run(run_name=transfer_arm["name"], nested=True) as run:
            arm_run_ids[transfer_arm["name"]] = run.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": config["run_id"],
                    "parent_run_id": parent_run_id,
                    "ab_group": transfer_arm["name"],
                    "initialization": "learned-width9-overlap",
                    "status": "running",
                }
            )
            exp100.SharedContextHornerTCN = transferred_constructor
            try:
                transfer_result = exp100.train_arm(
                    transfer_arm, matched, output / transfer_arm["name"]
                )
            finally:
                exp100.SharedContextHornerTCN = original_constructor
            transfer_result["controls"]["control_semantics"] = (
                "The keys named untrained_* are pre-finetune transferred-model metrics."
            )
            mlflow.set_tag("status", "completed")
        decision, winner, comparison = compare_results(
            random_result, transfer_result, config["gates"]
        )
        elapsed = time.perf_counter() - started
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": matched["tracking"]["experiment_name"],
                "parent_run_id": parent_run_id,
                "arm_run_ids": arm_run_ids,
            },
            "source": {
                "repository_commit": git_commit,
                "dirty": git_dirty,
                "challenge_commit": exp100.CHALLENGE_COMMIT,
                "runner_sha256": sha256(Path(__file__)),
                "config_sha256": sha256(config_path),
                "base_config_sha256": sha256(base_config_path),
            },
            "transfer": transfer_report,
            "results": {
                random_arm["name"]: random_result,
                transfer_arm["name"]: transfer_result,
            },
            "comparison": comparison,
            "winner": winner,
            "decision": decision,
            "resource": {
                "wall_seconds": elapsed,
                "cap_cpu_minutes": config["resource_cap"]["aggregate_cpu_minutes"],
            },
            "limitations": [
                "Intermediate heldout-prime A/B; no official tier score was produced.",
                "No checked public expected answers were read or evaluated.",
                "A full-budget decision is not candidate promotion."
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        mlflow.log_metrics(
            {
                "random_transition_exact": random_result["evaluation"]["heldout_transition"]["exact_accuracy"],
                "random_rollout_exact": random_result["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "transfer_transition_exact": transfer_result["evaluation"]["heldout_transition"]["exact_accuracy"],
                "transfer_rollout_exact": transfer_result["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "transition_gain": float(comparison["transition_gain"]),
                "rollout_gain": float(comparison["rollout_gain"]),
                "invalid_rate_change": float(comparison["invalid_rate_change"]),
                "wall_seconds": elapsed,
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags(
            {"status": "completed", "decision": decision, "winner": winner}
        )
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "decision": decision,
                "winner": winner,
                "comparison": comparison,
                "random_transition": random_result["evaluation"]["heldout_transition"]["exact_accuracy"],
                "random_rollout": random_result["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "transfer_transition": transfer_result["evaluation"]["heldout_transition"]["exact_accuracy"],
                "transfer_rollout": transfer_result["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
