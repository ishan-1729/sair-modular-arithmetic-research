"""Run EXP-201 full-budget transferred hard mining at width 32."""

from __future__ import annotations

import argparse
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
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
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/exp_201_width32_full_budget.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".scratch/exp-201-width32-full-budget"
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source_path) != config["source_checkpoint_sha256"]:
        raise RuntimeError("frozen width-16 checkpoint hash mismatch")
    exp105 = load_module(
        "exp105_for_exp201", ROOT / "experiments/exp_105_width_transfer/run.py"
    )
    exp200 = load_module(
        "exp200_for_exp201", ROOT / "experiments/exp_200_width32_transfer_ab/run.py"
    )
    matched = exp200.matched_config(config)
    arm = next(
        item
        for item in matched["comparison"]["arms"]
        if item["name"] == "width32-transfer-hard-mining"
    )
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original_constructor = exp100.SharedContextHornerTCN
    original_sampler = exp100.TransitionSampler
    original_split = exp100.split_primes
    transfer_report: dict[str, Any] = {}
    active_sampler = None

    def transferred_constructor(
        width: int, hidden: int, blocks: int
    ) -> exp100_model.SharedContextHornerTCN:
        nonlocal transfer_report
        model = original_constructor(width=width, hidden=hidden, blocks=blocks)
        transfer_report = exp105.transfer_width9_to_width16(model, source_path)
        return model

    def sampler_factory(
        width: int,
        minimum_bits: int,
        maximum_bits: int,
        primes: list[int],
        seed: int,
        composite_fraction: float,
        random_state_fraction: float,
    ):
        nonlocal active_sampler
        if composite_fraction > 0:
            active_sampler = exp200.ScalableHardMiningSampler(
                width,
                minimum_bits,
                maximum_bits,
                primes,
                seed,
                composite_fraction,
                random_state_fraction,
                pool_factor=int(config["hard_mining"]["candidate_pool_factor"]),
                quotient2_fraction=float(config["hard_mining"]["quotient2_fraction"]),
                boundary_fraction=float(config["hard_mining"]["boundary_fraction"]),
                boundary_relative_margin=float(
                    config["hard_mining"]["boundary_relative_margin"]
                ),
            )
            return active_sampler
        return exp200.ScalableTransitionSampler(
            width,
            minimum_bits,
            maximum_bits,
            primes,
            seed,
            composite_fraction,
            random_state_fraction,
        )

    git_commit = git_output("rev-parse", "HEAD")
    git_dirty = bool(git_output("status", "--porcelain"))
    started = time.perf_counter()
    exp100.SharedContextHornerTCN = transferred_constructor
    exp100.TransitionSampler = sampler_factory
    exp100.split_primes = exp200.sampled_prime_split
    try:
        with mlflow.start_run(run_name=config["run_id"]) as parent:
            parent_run_id = parent.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": config["run_id"],
                    "hypothesis": config["hypothesis"],
                    "status": "running",
                    "role": "tier4-full-budget-gate",
                    "baseline_run": "exp-200-width32-transfer-ab",
                    "ab_group": "EXP-201",
                    "promotion_gate": "preregistered-width32-full-budget",
                    "repository_commit": git_commit,
                    "repository_dirty": str(git_dirty).lower(),
                    "challenge_commit": exp100.CHALLENGE_COMMIT,
                }
            )
            mlflow.log_params(
                {
                    "seed": config["seed"],
                    "steps": config["training"]["steps"],
                    "batch_size": config["training"]["batch_size"],
                    "source_checkpoint_sha256": sha256(source_path),
                    "config_sha256": sha256(config_path),
                    "runner_sha256": sha256(Path(__file__)),
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                }
            )
            mlflow.log_artifact(str(config_path), artifact_path="preregistration")
            mlflow.log_artifact(
                str(ROOT / "knowledge/autoresearch/ablations/exp-201-width32-full-budget.md"),
                artifact_path="preregistration",
            )
            with mlflow.start_run(run_name=arm["name"], nested=True) as nested:
                nested_run_id = nested.info.run_id
                mlflow.set_tags(
                    {
                        "canonical_run_id": config["run_id"],
                        "parent_run_id": parent_run_id,
                        "ab_group": arm["name"],
                        "initialization": "width16-overlap",
                        "status": "running",
                    }
                )
                result = exp100.train_arm(arm, matched, output / arm["name"])
                mlflow.set_tag("status", "completed")
            if active_sampler is None:
                raise RuntimeError("hard-mining sampler was not constructed")
            mining_stats = active_sampler.mining_stats()
            result["hard_mining"] = mining_stats
            transition = result["evaluation"]["heldout_transition"]["exact_accuracy"]
            rollout = result["evaluation"]["heldout_rollout"]["exact_accuracy"]
            invalid = result["evaluation"]["heldout_rollout"][
                "any_invalid_residue_rate"
            ]
            gates = config["gates"]
            passed = (
                transition >= gates["minimum_transition_accuracy"]
                and rollout >= gates["minimum_rollout_accuracy"]
                and invalid <= gates["maximum_invalid_rollout_rate"]
            )
            decision = (
                "advance-five-independent-tier4-seeds"
                if passed
                else "revise-width32-before-independent"
            )
            summary = {
                "run_id": config["run_id"],
                "status": "completed",
                "hypothesis": config["hypothesis"],
                "mlflow": {
                    "tracking_uri": tracking_uri(),
                    "experiment_name": config["tracking"]["experiment_name"],
                    "experiment_id": experiment_id,
                    "parent_run_id": parent_run_id,
                    "arm_run_id": nested_run_id,
                },
                "source": {
                    "repository_commit": git_commit,
                    "dirty": git_dirty,
                    "challenge_commit": exp100.CHALLENGE_COMMIT,
                    "config_sha256": sha256(config_path),
                    "runner_sha256": sha256(Path(__file__)),
                    "source_checkpoint_sha256": sha256(source_path),
                },
                "transfer": transfer_report,
                "result": result,
                "hard_mining": mining_stats,
                "gates": {**gates, "passed": passed},
                "decision": decision,
                "resource": {
                    "wall_seconds": time.perf_counter() - started,
                    "cpu_minute_cap": config["resource_cap"]["cpu_minutes"],
                },
                "limitations": [
                    "Heldout-prime full-budget gate; no official Tier-4 score was produced.",
                    "Passing authorizes independent seeds, not candidate promotion.",
                    "No checked public expected answers were read or evaluated.",
                ],
            }
            summary_path = output / "summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            mlflow.log_metrics(
                {
                    "transition_exact": transition,
                    "rollout_exact": rollout,
                    "invalid_rollout": invalid,
                    "wall_seconds": summary["resource"]["wall_seconds"],
                }
            )
            mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
            mlflow.set_tags({"status": "completed", "decision": decision})
    finally:
        exp100.SharedContextHornerTCN = original_constructor
        exp100.TransitionSampler = original_sampler
        exp100.split_primes = original_split
    print(json.dumps({"decision": decision, "transition": transition, "rollout": rollout, "invalid_rollout": invalid, "checkpoint": result["checkpoint"], "mlflow": summary["mlflow"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
