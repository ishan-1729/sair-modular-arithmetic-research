"""Run the authorized EXP-111 2,400-step hard-mined width-16 model."""

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
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_111_hard_mining_2400.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-111-hard-mining-2400",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_config_path = (ROOT / config["base_config"]).resolve()
    matched = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    matched["run_id"] = config["run_id"]
    matched["training"]["steps"] = int(config["training"]["steps"])
    matched["resource_cap"]["continuation_steps"] = int(
        config["training"]["schedule_steps"]
    )
    matched["evaluation"]["heldout_transition_cases"] = int(
        config["evaluation"]["heldout_transition_cases"]
    )
    matched["evaluation"]["rollout_cases"] = int(
        config["evaluation"]["rollout_cases"]
    )
    source_path = (ROOT / config["transfer_checkpoint"]).resolve()
    if sha256(source_path) != config["transfer_checkpoint_sha256"]:
        raise RuntimeError("transfer checkpoint hash mismatch")
    exp105 = load_module(
        "exp105_for_exp111",
        ROOT / "experiments" / "exp_105_width_transfer" / "run.py",
    )
    exp110 = load_module(
        "exp110_for_exp111",
        ROOT / "experiments" / "exp_110_quotient2_hard_mining" / "run.py",
    )
    torch.set_num_threads(int(matched["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    configure_mlflow(matched["tracking"]["experiment_name"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arm = next(item for item in matched["comparison"]["arms"] if item["width"] == 16)
    arm = {**arm, "name": "width16-transfer-hard-mining-2400"}
    original_constructor = exp100.SharedContextHornerTCN
    original_sampler = exp100.TransitionSampler
    transfer_report: dict[str, Any] = {}
    hard_sampler = None

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
        nonlocal hard_sampler
        if composite_fraction > 0:
            hard_sampler = exp110.HardMiningSampler(
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
                boundary_relative_margin=float(config["hard_mining"]["boundary_relative_margin"]),
            )
            return hard_sampler
        return original_sampler(
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
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "hypothesis": config["hypothesis"],
                "status": "running",
                "role": "challenger-full-budget",
                "baseline_run": "exp-110-quotient2-hard-mining",
                "ab_group": "EXP-111",
                "promotion_gate": "preregistered-heldout-tier3-gate",
                "repository_commit": git_commit,
                "repository_dirty": str(git_dirty).lower(),
                "challenge_commit": exp100.CHALLENGE_COMMIT,
            }
        )
        mlflow.log_params(
            {
                "seed": config["seed"],
                "steps": matched["training"]["steps"],
                "schedule_steps": matched["resource_cap"]["continuation_steps"],
                "batch_size": matched["training"]["batch_size"],
                "transition_cases": matched["evaluation"]["heldout_transition_cases"],
                "rollout_cases": matched["evaluation"]["rollout_cases"],
                "candidate_pool_factor": config["hard_mining"]["candidate_pool_factor"],
                "quotient2_fraction": config["hard_mining"]["quotient2_fraction"],
                "boundary_fraction": config["hard_mining"]["boundary_fraction"],
                "transfer_checkpoint_sha256": sha256(source_path),
                "config_sha256": sha256(config_path),
                "runner_sha256": sha256(Path(__file__)),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(
            str(ROOT / "knowledge/autoresearch/ablations/exp-111-hard-mining-2400.md"),
            artifact_path="preregistration",
        )
        with mlflow.start_run(run_name=arm["name"], nested=True) as nested:
            nested_run_id = nested.info.run_id
            mlflow.set_tags(
                {
                    "canonical_run_id": config["run_id"],
                    "parent_run_id": parent_run_id,
                    "ab_group": arm["name"],
                    "initialization": "learned-width9-overlap",
                    "curriculum": "quotient2-boundary-hard-mining",
                    "status": "running",
                }
            )
            exp100.SharedContextHornerTCN = transferred_constructor
            exp100.TransitionSampler = sampler_factory
            try:
                result = exp100.train_arm(arm, matched, output / arm["name"])
            finally:
                exp100.SharedContextHornerTCN = original_constructor
                exp100.TransitionSampler = original_sampler
            if hard_sampler is None:
                raise RuntimeError("hard-mining sampler was not constructed")
            mining_stats = hard_sampler.mining_stats()
            result["hard_mining"] = mining_stats
            mlflow.log_metrics(
                {
                    "selected_quotient2_fraction": float(mining_stats["selected_quotient2_fraction"]),
                    "selected_boundary_fraction": float(mining_stats["selected_boundary_fraction"]),
                }
            )
            mlflow.set_tag("status", "completed")
        transition = result["evaluation"]["heldout_transition"]["exact_accuracy"]
        rollout = result["evaluation"]["heldout_rollout"]["exact_accuracy"]
        invalid = result["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"]
        passed = (
            transition >= float(config["gates"]["minimum_transition_accuracy"])
            and rollout >= float(config["gates"]["minimum_rollout_accuracy"])
            and invalid <= float(config["gates"]["maximum_invalid_rollout_rate"])
        )
        decision = (
            "advance-five-independent-tier3-seeds"
            if passed
            else "revise-before-independent-evaluation"
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
            "transfer": transfer_report,
            "result": result,
            "hard_mining": mining_stats,
            "gates": {**config["gates"], "passed": passed},
            "decision": decision,
            "resource": {
                "wall_seconds": elapsed,
                "cap_cpu_minutes": config["resource_cap"]["cpu_minutes"],
                "network_training_examples": matched["training"]["steps"] * matched["training"]["batch_size"],
                "offline_candidate_examples": matched["training"]["steps"] * matched["training"]["batch_size"] * config["hard_mining"]["candidate_pool_factor"],
            },
            "limitations": [
                "Heldout-prime full-budget gate; no official tier score was produced.",
                "No checked public expected answers were read or evaluated.",
                "Passing authorizes independent seeds, not candidate promotion."
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
                "transition": transition,
                "rollout": rollout,
                "invalid_rollout": invalid,
                "checkpoint": result["checkpoint"],
                "hard_mining": mining_stats,
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
