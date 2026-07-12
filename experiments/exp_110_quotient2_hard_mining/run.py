"""Run EXP-110 quotient-2 and reduction-boundary hard-example mining."""

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


def load_exp105():
    path = ROOT / "experiments" / "exp_105_width_transfer" / "run.py"
    spec = importlib.util.spec_from_file_location("exp105_for_exp110", path)
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


class HardMiningSampler(exp100.TransitionSampler):
    def __init__(
        self,
        *args,
        pool_factor: int,
        quotient2_fraction: float,
        boundary_fraction: float,
        boundary_relative_margin: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pool_factor = pool_factor
        self.quotient2_fraction = quotient2_fraction
        self.boundary_fraction = boundary_fraction
        self.boundary_relative_margin = boundary_relative_margin
        self.selected_total = 0
        self.selected_quotient2 = 0
        self.selected_boundary = 0

    def _draw(
        self,
        mask: torch.Tensor,
        available: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        indices = torch.nonzero(mask & available, as_tuple=False).flatten()
        if not len(indices) or count <= 0:
            return torch.empty(0, dtype=torch.long)
        order = torch.randperm(len(indices), generator=self.generator)
        chosen = indices[order[: min(count, len(indices))]]
        available[chosen] = False
        return chosen

    def transitions(self, size: int):
        pool_size = size * self.pool_factor
        values = super().transitions(pool_size)
        state, multiplicand, modulus, bit, target, quotient = values
        raw = 2 * state + bit * multiplicand
        margin = torch.minimum((raw - modulus).abs(), (raw - 2 * modulus).abs())
        boundary = (
            margin.double() / modulus.double()
            <= self.boundary_relative_margin
        )
        available = torch.ones(pool_size, dtype=torch.bool)
        q2 = self._draw(
            quotient.eq(2),
            available,
            round(size * self.quotient2_fraction),
        )
        near = self._draw(
            boundary,
            available,
            round(size * self.boundary_fraction),
        )
        remaining = self._draw(
            available,
            available,
            size - len(q2) - len(near),
        )
        selected = torch.cat((q2, near, remaining))
        if len(selected) != size:
            raise RuntimeError(f"hard-mining selection produced {len(selected)} rows")
        selected = selected[
            torch.randperm(len(selected), generator=self.generator)
        ]
        self.selected_total += size
        self.selected_quotient2 += int(quotient[selected].eq(2).sum())
        self.selected_boundary += int(boundary[selected].sum())
        return tuple(value[selected] for value in values)

    def mining_stats(self) -> dict[str, float | int]:
        return {
            "selected_total": self.selected_total,
            "selected_quotient2": self.selected_quotient2,
            "selected_boundary": self.selected_boundary,
            "selected_quotient2_fraction": (
                self.selected_quotient2 / self.selected_total
            ),
            "selected_boundary_fraction": (
                self.selected_boundary / self.selected_total
            ),
            "candidate_pool_factor": self.pool_factor,
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
    comparison: dict[str, float | bool] = {
        "transition_gain": transition
        - float(baseline["transition_accuracy"]),
        "rollout_gain": rollout - float(baseline["rollout_accuracy"]),
        "invalid_rate_change": invalid
        - float(baseline["invalid_rollout_rate"]),
    }
    passed = (
        comparison["transition_gain"]
        >= float(gates["minimum_transition_gain"])
        and comparison["rollout_gain"]
        >= float(gates["minimum_rollout_gain"])
        and comparison["invalid_rate_change"]
        <= float(gates["maximum_invalid_rate_change"])
    )
    comparison["ablation_passed"] = passed
    if not passed:
        return "reject-hard-mining-mixture", comparison
    if rollout >= float(gates["independent_gate_rollout"]):
        return "advance-independent-tier3-gate", comparison
    if (
        transition >= float(gates["full_budget_transition"])
        and rollout >= float(gates["full_budget_rollout"])
    ):
        return "advance-hard-mining-to-2400", comparison
    return "pass-but-refine-before-full-budget", comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_110_quotient2_hard_mining.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-110-quotient2-hard-mining",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_config_path = (ROOT / config["base_config"]).resolve()
    matched = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
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
    arm = next(item for item in matched["comparison"]["arms"] if item["width"] == 16)
    arm = {**arm, "name": "width16-transfer-hard-mining"}
    original_constructor = exp100.SharedContextHornerTCN
    original_sampler = exp100.TransitionSampler
    transfer_report: dict[str, Any] = {}
    hard_sampler: HardMiningSampler | None = None

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
            hard_sampler = HardMiningSampler(
                width,
                minimum_bits,
                maximum_bits,
                primes,
                seed,
                composite_fraction,
                random_state_fraction,
                pool_factor=int(config["training"]["candidate_pool_factor"]),
                quotient2_fraction=float(config["training"]["quotient2_fraction"]),
                boundary_fraction=float(config["training"]["boundary_fraction"]),
                boundary_relative_margin=float(config["training"]["boundary_relative_margin"]),
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
                "role": "challenger-curriculum",
                "baseline_run": config["baseline"]["run_id"],
                "baseline_mlflow_run": config["baseline"]["mlflow_run_id"],
                "ab_group": "EXP-110",
                "promotion_gate": "preregistered-quotient2-hard-mining",
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
                "candidate_pool_factor": config["training"]["candidate_pool_factor"],
                "quotient2_fraction": config["training"]["quotient2_fraction"],
                "boundary_fraction": config["training"]["boundary_fraction"],
                "boundary_relative_margin": config["training"]["boundary_relative_margin"],
                "transfer_checkpoint_sha256": sha256(source_path),
                "config_sha256": sha256(config_path),
                "runner_sha256": sha256(Path(__file__)),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(
            str(ROOT / "knowledge/autoresearch/ablations/exp-110-quotient2-hard-mining.md"),
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
                challenger = exp100.train_arm(arm, matched, output / arm["name"])
            finally:
                exp100.SharedContextHornerTCN = original_constructor
                exp100.TransitionSampler = original_sampler
            if hard_sampler is None:
                raise RuntimeError("hard-mining sampler was not constructed")
            mining_stats = hard_sampler.mining_stats()
            challenger["hard_mining"] = mining_stats
            mlflow.log_metrics(
                {
                    "selected_quotient2_fraction": float(mining_stats["selected_quotient2_fraction"]),
                    "selected_boundary_fraction": float(mining_stats["selected_boundary_fraction"]),
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
            "baseline": config["baseline"],
            "transfer": transfer_report,
            "challenger": challenger,
            "hard_mining": mining_stats,
            "comparison": comparison,
            "decision": decision,
            "resource": {
                "wall_seconds": elapsed,
                "cap_cpu_minutes": config["resource_cap"]["cpu_minutes"],
                "network_training_examples": matched["training"]["steps"] * matched["training"]["batch_size"],
                "offline_candidate_examples": matched["training"]["steps"] * matched["training"]["batch_size"] * config["training"]["candidate_pool_factor"],
            },
            "limitations": [
                "Matched training-step ablation with a 4x offline candidate-generation pool.",
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
                "challenger_transition_exact": challenger["evaluation"]["heldout_transition"]["exact_accuracy"],
                "challenger_rollout_exact": challenger["evaluation"]["heldout_rollout"]["exact_accuracy"],
                "challenger_invalid_rollout": challenger["evaluation"]["heldout_rollout"]["any_invalid_residue_rate"],
                "transition_gain": float(comparison["transition_gain"]),
                "rollout_gain": float(comparison["rollout_gain"]),
                "invalid_rate_change": float(comparison["invalid_rate_change"]),
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
                "hard_mining": mining_stats,
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
