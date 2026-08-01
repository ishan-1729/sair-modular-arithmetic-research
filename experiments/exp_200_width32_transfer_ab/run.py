"""Run EXP-200: matched random versus width-16 transfer at width 32."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from sympy import isprime, nextprime


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


def sampled_prime_split(
    minimum_bits: int,
    maximum_bits: int,
    seed: int,
) -> dict[str, list[int]]:
    """Bounded deterministic identity split without an exhaustive 2^width sieve."""
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    counts = {"train": 128, "validation": 32, "test": 32}
    for bits in range(minimum_bits, maximum_bits + 1):
        low = max(2, 1 << (bits - 1))
        high = 1 << bits
        seen: set[int] = set()
        group: list[int] = []
        target = sum(counts.values())
        while len(group) < target:
            prime = int(nextprime(rng.randrange(low, high)))
            if prime < high and prime not in seen:
                seen.add(prime)
                group.append(prime)
        cursor = 0
        for split_name, count in counts.items():
            splits[split_name].extend(group[cursor : cursor + count])
            cursor += count
    return splits


class ScalableTransitionSampler(exp100.TransitionSampler):
    """EXP-100 sampler with bounded composite storage for 32-bit moduli."""

    def __init__(
        self,
        width: int,
        minimum_bits: int,
        maximum_bits: int,
        primes: list[int],
        seed: int,
        composite_fraction: float,
        random_state_fraction: float,
    ) -> None:
        self.width = width
        self.minimum_bits = minimum_bits
        self.maximum_bits = maximum_bits
        self.primes = torch.tensor(primes, dtype=torch.int64)
        rng = random.Random(seed + 991)
        low = max(4, 1 << (minimum_bits - 1))
        high = 1 << maximum_bits
        composites: set[int] = set()
        while len(composites) < 4096:
            candidate = rng.randrange(low, high)
            if not isprime(candidate):
                composites.add(candidate)
        self.composites = torch.tensor(sorted(composites), dtype=torch.int64)
        self.generator = torch.Generator().manual_seed(seed)
        self.composite_fraction = composite_fraction
        self.random_state_fraction = random_state_fraction


class ScalableHardMiningSampler(ScalableTransitionSampler):
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
        self, mask: torch.Tensor, available: torch.Tensor, count: int
    ) -> torch.Tensor:
        indices = torch.nonzero(mask & available, as_tuple=False).flatten()
        if not len(indices) or count <= 0:
            return torch.empty(0, dtype=torch.long)
        order = torch.randperm(len(indices), generator=self.generator)
        chosen = indices[order[: min(count, len(indices))]]
        available[chosen] = False
        return chosen

    def transitions(self, size: int):
        values = super().transitions(size * self.pool_factor)
        state, multiplicand, modulus, bit, target, quotient = values
        raw = 2 * state + bit * multiplicand
        margin = torch.minimum((raw - modulus).abs(), (raw - 2 * modulus).abs())
        boundary = margin.double() / modulus.double() <= self.boundary_relative_margin
        available = torch.ones(len(state), dtype=torch.bool)
        q2 = self._draw(
            quotient.eq(2), available, round(size * self.quotient2_fraction)
        )
        near = self._draw(
            boundary, available, round(size * self.boundary_fraction)
        )
        remaining = self._draw(available, available, size - len(q2) - len(near))
        selected = torch.cat((q2, near, remaining))
        if len(selected) != size:
            raise RuntimeError(f"hard-mining selection produced {len(selected)} rows")
        selected = selected[torch.randperm(len(selected), generator=self.generator)]
        self.selected_total += size
        self.selected_quotient2 += int(quotient[selected].eq(2).sum())
        self.selected_boundary += int(boundary[selected].sum())
        return tuple(value[selected] for value in values)

    def mining_stats(self) -> dict[str, float | int]:
        return {
            "selected_total": self.selected_total,
            "selected_quotient2": self.selected_quotient2,
            "selected_boundary": self.selected_boundary,
            "selected_quotient2_fraction": self.selected_quotient2
            / self.selected_total,
            "selected_boundary_fraction": self.selected_boundary
            / self.selected_total,
            "candidate_pool_factor": self.pool_factor,
        }


def matched_config(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    arm_base = {
        "width": 32,
        "prime_bits_min": evaluation["prime_bits_min"],
        "prime_bits_max": evaluation["prime_bits_max"],
    }
    return {
        "run_id": config["run_id"],
        "seed": config["seed"],
        "tracking": {"experiment_name": config["tracking"]["experiment_name"]},
        "comparison": {
            "arms": [
                {**arm_base, "name": "width32-random-hard-mining"},
                {**arm_base, "name": "width32-transfer-hard-mining"},
            ]
        },
        "model": config["model"],
        "training": config["training"],
        "evaluation": {
            "heldout_transition_cases": evaluation["heldout_transition_cases"],
            "rollout_cases": evaluation["rollout_cases"],
            "prime_split": evaluation["prime_split"],
        },
        "resource_cap": {
            "continuation_steps": config["resource_cap"]["continuation_steps"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/exp_200_width32_transfer_ab.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".scratch/exp-200-width32-transfer-ab"
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source_path) != config["source_checkpoint_sha256"]:
        raise RuntimeError("frozen width-16 checkpoint hash mismatch")
    exp105 = load_module(
        "exp105_for_exp200", ROOT / "experiments/exp_105_width_transfer/run.py"
    )
    matched = matched_config(config)
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original_constructor = exp100.SharedContextHornerTCN
    original_sampler = exp100.TransitionSampler
    original_split_primes = exp100.split_primes
    results: dict[str, Any] = {}
    transfer_report: dict[str, Any] = {}
    mining_stats: dict[str, Any] = {}

    def sampler_factory(arm_name: str):
        active = None

        def factory(
            width: int,
            minimum_bits: int,
            maximum_bits: int,
            primes: list[int],
            seed: int,
            composite_fraction: float,
            random_state_fraction: float,
        ):
            nonlocal active
            if composite_fraction > 0:
                active = ScalableHardMiningSampler(
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
                return active
            return ScalableTransitionSampler(
                width,
                minimum_bits,
                maximum_bits,
                primes,
                seed,
                composite_fraction,
                random_state_fraction,
            )

        def finish() -> None:
            if active is None:
                raise RuntimeError("hard-mining sampler was not constructed")
            mining_stats[arm_name] = active.mining_stats()

        return factory, finish

    git_commit = git_output("rev-parse", "HEAD")
    git_dirty = bool(git_output("status", "--porcelain"))
    started = time.perf_counter()
    exp100.split_primes = sampled_prime_split
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "hypothesis": config["hypothesis"],
                "status": "running",
                "role": "tier4-matched-smoke",
                "ab_group": "EXP-200",
                "promotion_gate": "preregistered-width32-transfer-ab",
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
            str(ROOT / "knowledge/autoresearch/ablations/exp-200-width32-transfer-ab.md"),
            artifact_path="preregistration",
        )
        for arm in matched["comparison"]["arms"]:
            with mlflow.start_run(run_name=arm["name"], nested=True) as nested:
                mlflow.set_tags(
                    {
                        "canonical_run_id": config["run_id"],
                        "parent_run_id": parent_run_id,
                        "ab_group": arm["name"],
                        "initialization": "width16-overlap"
                        if "transfer" in arm["name"]
                        else "random",
                        "status": "running",
                    }
                )
                factory, finish = sampler_factory(arm["name"])

                def transferred_constructor(
                    width: int, hidden: int, blocks: int
                ) -> exp100_model.SharedContextHornerTCN:
                    nonlocal transfer_report
                    model = original_constructor(width=width, hidden=hidden, blocks=blocks)
                    transfer_report = exp105.transfer_width9_to_width16(model, source_path)
                    return model

                exp100.TransitionSampler = factory
                if "transfer" in arm["name"]:
                    exp100.SharedContextHornerTCN = transferred_constructor
                try:
                    results[arm["name"]] = exp100.train_arm(
                        arm, matched, output / arm["name"]
                    )
                    finish()
                finally:
                    exp100.SharedContextHornerTCN = original_constructor
                    exp100.TransitionSampler = original_sampler
                results[arm["name"]]["hard_mining"] = mining_stats[arm["name"]]
                mlflow.set_tag("status", "completed")

        random_result = results["width32-random-hard-mining"]
        transfer_result = results["width32-transfer-hard-mining"]
        random_transition = random_result["evaluation"]["heldout_transition"][
            "exact_accuracy"
        ]
        random_rollout = random_result["evaluation"]["heldout_rollout"][
            "exact_accuracy"
        ]
        transfer_transition = transfer_result["evaluation"]["heldout_transition"][
            "exact_accuracy"
        ]
        transfer_rollout = transfer_result["evaluation"]["heldout_rollout"][
            "exact_accuracy"
        ]
        transfer_invalid = transfer_result["evaluation"]["heldout_rollout"][
            "any_invalid_residue_rate"
        ]
        comparison = {
            "transition_gain": transfer_transition - random_transition,
            "rollout_gain": transfer_rollout - random_rollout,
            "transfer_transition": transfer_transition,
            "transfer_rollout": transfer_rollout,
            "transfer_invalid_rollout": transfer_invalid,
        }
        gates = config["gates"]
        passed = (
            comparison["transition_gain"] >= gates["minimum_transition_gain"]
            and comparison["rollout_gain"] >= gates["minimum_rollout_gain"]
            and transfer_transition >= gates["minimum_transfer_transition"]
            and transfer_rollout >= gates["minimum_transfer_rollout"]
            and transfer_invalid <= gates["maximum_transfer_invalid_rollout"]
        )
        decision = (
            "advance-width32-full-budget" if passed else "reject-direct-width32-transfer"
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
            "arms": results,
            "comparison": comparison,
            "gates": {**gates, "passed": passed},
            "decision": decision,
            "resource": {
                "wall_seconds": time.perf_counter() - started,
                "cpu_minute_cap": config["resource_cap"]["cpu_minutes"],
            },
            "limitations": [
                "Heldout-prime smoke test; no official Tier-4 score was produced.",
                "Width 32 remains an int64-compatible stepping stone and does not solve long-width scaling.",
                "No checked public expected answers were read or evaluated.",
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({**comparison, "wall_seconds": summary["resource"]["wall_seconds"]})
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": decision})
    exp100.split_primes = original_split_primes
    print(json.dumps({"decision": decision, "comparison": comparison, "mlflow": summary["mlflow"], "checkpoints": {name: result["checkpoint"] for name, result in results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
