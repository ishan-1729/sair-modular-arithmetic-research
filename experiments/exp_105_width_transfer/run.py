"""Run EXP-105: learned width-9 initialization versus width-16 random init."""

from __future__ import annotations

import argparse
import hashlib
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
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
EXP100_DIR = ROOT / "experiments" / "exp_100_width_scaling"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP100_DIR))

import model as exp100_model  # noqa: E402
import run as exp100  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


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


def transfer_width9_to_width16(
    model: exp100_model.SharedContextHornerTCN,
    source_path: Path,
) -> dict[str, Any]:
    source = load_file(str(source_path), device="cpu")
    target = model.state_dict()
    exact: list[str] = []
    special: list[str] = []
    for key, source_tensor in source.items():
        if key in target and target[key].shape == source_tensor.shape:
            target[key].copy_(source_tensor)
            exact.append(key)

    source_width = int(source["position"].shape[-1])
    target_width = model.width
    if source_width >= target_width:
        raise ValueError("source must be narrower than target")

    target["position"][:, :, :source_width].copy_(source["position"])
    special.append("position")
    target["modulus_context.0.weight"][:, :source_width].copy_(
        source["modulus_context.0.weight"]
    )
    special.append("modulus_context.0.weight")

    source_global = source["global_head.weight"].reshape(
        source_width, model.hidden, source_width
    )
    target_global = target["global_head.weight"].reshape(
        target_width, model.hidden, target_width
    )
    target_global[:source_width, :, :source_width].copy_(source_global)
    special.append("global_head.weight")
    target["global_head.bias"][:source_width].copy_(
        source["global_head.bias"]
    )
    special.append("global_head.bias")

    source_quotient = source["quotient_head.weight"].reshape(
        3, model.hidden, source_width
    )
    target_quotient = target["quotient_head.weight"].reshape(
        3, model.hidden, target_width
    )
    target_quotient[:, :, :source_width].copy_(source_quotient)
    special.append("quotient_head.weight")
    model.load_state_dict(target)
    return {
        "source_width": source_width,
        "target_width": target_width,
        "exact_tensor_count": len(exact),
        "exact_tensors": sorted(exact),
        "overlap_tensors": special,
        "source_sha256": sha256(source_path),
    }


def decide(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[str, dict[str, float | bool]]:
    transition = challenger["evaluation"]["heldout_transition"][
        "exact_accuracy"
    ]
    rollout = challenger["evaluation"]["heldout_rollout"]["exact_accuracy"]
    invalid = challenger["evaluation"]["heldout_rollout"][
        "any_invalid_residue_rate"
    ]
    slope = challenger["training"]["final_100_step_batch_exact_delta"]
    diagnostics: dict[str, float | bool] = {
        "transition_gain": transition
        - float(baseline["transition_accuracy"]),
        "rollout_gain": rollout - float(baseline["rollout_accuracy"]),
        "invalid_rate_change": invalid
        - float(baseline["invalid_rollout_rate"]),
        "positive_final_slope": slope > 0,
    }
    passed = (
        diagnostics["transition_gain"]
        >= float(gates["minimum_transition_gain"])
        and rollout >= float(gates["minimum_rollout_accuracy"])
        and invalid <= float(gates["maximum_invalid_rollout_rate"])
        and (not gates["require_positive_final_slope"] or slope > 0)
    )
    return ("pass-continue-matched-1200" if passed else "reject-transfer"), diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_105_width_transfer_smoke.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-105-width-transfer-smoke",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_config_path = (ROOT / config["matched_training_config"]).resolve()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    source_path = (ROOT / config["transfer_source"]["checkpoint"]).resolve()
    if sha256(source_path) != config["transfer_source"]["checkpoint_sha256"]:
        raise RuntimeError("width-9 transfer checkpoint hash mismatch")
    baseline_metrics_path = (
        ROOT
        / "experiments/runs/exp-100-width16-shared-context-smoke/width16-challenger-metrics.json"
    )
    baseline_metrics = json.loads(
        baseline_metrics_path.read_text(encoding="utf-8")
    )
    torch.set_num_threads(int(base_config["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    configure_mlflow(base_config["tracking"]["experiment_name"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    transfer_report: dict[str, Any] = {}
    original_constructor = exp100.SharedContextHornerTCN

    def transferred_constructor(
        width: int, hidden: int, blocks: int
    ) -> exp100_model.SharedContextHornerTCN:
        nonlocal transfer_report
        model = original_constructor(width=width, hidden=hidden, blocks=blocks)
        if width != 16:
            raise ValueError("EXP-105 expects the width-16 arm")
        transfer_report = transfer_width9_to_width16(model, source_path)
        return model

    git_commit = git_output("rev-parse", "HEAD")
    git_dirty = bool(git_output("status", "--porcelain"))
    arm = next(
        item
        for item in base_config["comparison"]["arms"]
        if item["name"] == "width16-challenger"
    )
    started = time.perf_counter()
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "hypothesis": config["hypothesis"],
                "status": "running",
                "role": "challenger-smoke",
                "baseline_run": config["baseline"]["run_id"],
                "baseline_mlflow_run": config["baseline"]["mlflow_run_id"],
                "ab_group": "EXP-105",
                "promotion_gate": "preregistered-width-transfer-smoke",
                "repository_commit": git_commit,
                "repository_dirty": str(git_dirty).lower(),
                "challenge_commit": exp100.CHALLENGE_COMMIT,
            }
        )
        mlflow.log_params(
            {
                "seed": config["seed"],
                "steps": base_config["training"]["steps"],
                "batch_size": base_config["training"]["batch_size"],
                "width": 16,
                "initialization": "learned-width9-overlap",
                "transfer_checkpoint_sha256": sha256(source_path),
                "baseline_checkpoint_sha256": config["baseline"]["checkpoint_sha256"],
                "config_sha256": sha256(config_path),
                "base_config_sha256": sha256(base_config_path),
                "runner_sha256": sha256(Path(__file__)),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(
            str(ROOT / "knowledge/autoresearch/ablations/exp-105-width-transfer.md"),
            artifact_path="preregistration",
        )
        with mlflow.start_run(
            run_name="width16-transfer", nested=True
        ) as nested:
            nested_run_id = nested.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": config["run_id"],
                    "parent_run_id": parent_run_id,
                    "ab_group": "width16-transfer",
                    "status": "running",
                }
            )
            exp100.SharedContextHornerTCN = transferred_constructor
            try:
                challenger = exp100.train_arm(
                    arm, base_config, output / "width16-transfer"
                )
            finally:
                exp100.SharedContextHornerTCN = original_constructor
            challenger["controls"]["control_semantics"] = (
                "The keys named untrained_* are pre-finetune transferred-model metrics in EXP-105."
            )
            mlflow.log_metrics(
                {
                    "prefinetune_transition_exact": challenger["controls"]["untrained_transition"]["exact_accuracy"],
                    "prefinetune_rollout_exact": challenger["controls"]["untrained_rollout"]["exact_accuracy"],
                }
            )
            mlflow.set_tag("status", "completed")
        decision, comparison = decide(
            config["baseline"], challenger, config["gates"]
        )
        elapsed = time.perf_counter() - started
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": base_config["tracking"]["experiment_name"],
                "parent_run_id": parent_run_id,
                "arm_run_id": nested_run_id,
            },
            "source": {
                "repository_commit": git_commit,
                "dirty": git_dirty,
                "challenge_commit": exp100.CHALLENGE_COMMIT,
                "runner_sha256": sha256(Path(__file__)),
                "config_sha256": sha256(config_path),
                "base_config_sha256": sha256(base_config_path),
            },
            "baseline": {
                "config": config["baseline"],
                "metrics_path": str(baseline_metrics_path.relative_to(ROOT)),
                "metrics": baseline_metrics,
            },
            "transfer": transfer_report,
            "challenger": challenger,
            "comparison": comparison,
            "decision": decision,
            "resource": {
                "wall_seconds": elapsed,
                "cap_cpu_minutes": config["resource_cap"]["cpu_minutes"],
            },
            "limitations": [
                "Smoke-scale initialization A/B only; not a candidate result.",
                "The random baseline is reused from deterministic EXP-100 rather than recomputed.",
                "No checked public expected answers were read or evaluated."
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        mlflow.log_metrics(
            {
                "transition_gain": float(comparison["transition_gain"]),
                "rollout_gain": float(comparison["rollout_gain"]),
                "invalid_rate_change": float(comparison["invalid_rate_change"]),
                "challenger_transition_exact": challenger["evaluation"]["heldout_transition"]["exact_accuracy"],
                "challenger_rollout_exact": challenger["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "wall_seconds": elapsed,
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "decision": decision,
                "comparison": comparison,
                "transition": challenger["evaluation"]["heldout_transition"]["exact_accuracy"],
                "rollout": challenger["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "invalid_rollout": challenger["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"],
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
