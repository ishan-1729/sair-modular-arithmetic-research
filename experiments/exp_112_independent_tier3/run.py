"""Evaluate the frozen EXP-111 checkpoint on five official-generator Tier 3 seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
import yaml
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SRC = ROOT / "sources/github/modular-arithmetic-challenge/src"
EXP100_DIR = ROOT / "experiments/exp_100_width_scaling"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP100_DIR))
sys.path.insert(0, str(UPSTREAM_SRC))

from model import SharedContextHornerTCN, predict_state  # noqa: E402
from modchallenge.config import TIERS  # noqa: E402
from modchallenge.testgen.generator import _generate_tier_cases  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def predict_cases(model: SharedContextHornerTCN, cases) -> dict:
    modulus = torch.tensor([int(case.p) for case in cases], dtype=torch.int64)
    multiplier = torch.tensor(
        [int(case.a) % int(case.p) for case in cases], dtype=torch.int64
    )
    multiplicand = torch.tensor(
        [int(case.b) % int(case.p) for case in cases], dtype=torch.int64
    )
    expected = torch.tensor([int(case.expected) for case in cases], dtype=torch.int64)
    state = torch.zeros(len(cases), dtype=torch.int64)
    started = time.perf_counter()
    for step in range(model.width):
        bit = (multiplier >> (model.width - 1 - step)) & 1
        state, _ = predict_state(model, state, multiplicand, modulus, bit)
    elapsed = time.perf_counter() - started
    correct = state.eq(expected)
    invalid = state.ge(modulus)
    per_prime = defaultdict(lambda: [0, 0])
    for prime, ok in zip(modulus.tolist(), correct.tolist(), strict=True):
        per_prime[prime][0] += int(ok)
        per_prime[prime][1] += 1
    return {
        "cases": len(cases),
        "correct": int(correct.sum()),
        "exact_accuracy": float(correct.float().mean()),
        "invalid_residues": int(invalid.sum()),
        "inference_seconds": elapsed,
        "per_prime": {
            str(prime): {
                "cases": values[1],
                "correct": values[0],
                "exact_accuracy": values[0] / values[1],
            }
            for prime, values in sorted(per_prime.items())
        },
        "errors": [
            {
                "index": index,
                "p": int(modulus[index]),
                "predicted": int(state[index]),
                "expected": int(expected[index]),
            }
            for index in range(len(cases))
            if not bool(correct[index])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/exp_112_independent_tier3.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".scratch/exp-112-independent-tier3.json"
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = (ROOT / config["checkpoint"]).resolve()
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint hash mismatch")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = SharedContextHornerTCN(**config["architecture"])
    model.load_state_dict(load_file(checkpoint))
    model.eval()
    configure_mlflow("sair-modular-arithmetic")
    results = {}
    started = time.perf_counter()
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({
            "canonical_run_id": config["run_id"],
            "status": "running",
            "role": "independent-generalization-gate",
            "tier": "3",
            "challenge_commit": CHALLENGE_COMMIT,
            "checked_public_used": "false",
        })
        mlflow.log_params({
            "checkpoint_sha256": sha256(checkpoint),
            "config_sha256": sha256(config_path),
            "seeds": len(config["evaluation"]["seeds"]),
            "cases_per_seed": config["evaluation"]["cases_per_seed"],
            "primes_per_seed": config["evaluation"]["primes_per_seed"],
        })
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        prereg = ROOT / "knowledge/autoresearch/ablations/exp-112-independent-tier3.md"
        mlflow.log_artifact(str(prereg), artifact_path="preregistration")
        for seed_text in config["evaluation"]["seeds"]:
            seed = seed_text.encode("utf-8")
            seed_hash = hashlib.sha256(seed).hexdigest()
            cases = _generate_tier_cases(
                tier=TIERS[config["evaluation"]["tier_id"]],
                num_cases=config["evaluation"]["cases_per_seed"],
                num_primes=config["evaluation"]["primes_per_seed"],
                edge_cases=config["evaluation"]["edge_cases_per_seed"],
                seed=seed,
            )
            results[seed_hash[:16]] = {
                "seed_sha256": seed_hash,
                **predict_cases(model, cases),
            }
        total_cases = sum(item["cases"] for item in results.values())
        total_correct = sum(item["correct"] for item in results.values())
        aggregate_accuracy = total_correct / total_cases
        passing_seeds = sum(
            item["exact_accuracy"] >= config["gates"]["per_seed_passing_accuracy"]
            for item in results.values()
        )
        minimum_seed_accuracy = min(item["exact_accuracy"] for item in results.values())
        passed = (
            aggregate_accuracy >= config["gates"]["minimum_aggregate_accuracy"]
            and passing_seeds >= config["gates"]["minimum_passing_seeds"]
            and minimum_seed_accuracy >= config["gates"]["minimum_single_seed_accuracy"]
        )
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "checkpoint": {"path": config["checkpoint"], "sha256": sha256(checkpoint)},
            "generator": {"implementation": "modchallenge.testgen.generator._generate_tier_cases", "challenge_commit": CHALLENGE_COMMIT},
            "checked_public_used": False,
            "tier": 3,
            "seeds": results,
            "aggregate": {
                "cases": total_cases,
                "correct": total_correct,
                "exact_accuracy": aggregate_accuracy,
                "passing_seeds": passing_seeds,
                "minimum_seed_accuracy": minimum_seed_accuracy,
            },
            "gates": {**config["gates"], "passed": passed},
            "decision": "promote-to-submission-integration" if passed else "retain-incumbent",
            "runtime": {"wall_seconds": time.perf_counter() - started, "device": "cpu", "torch_threads": 4},
            "mlflow": {"tracking_uri": tracking_uri(), "experiment_name": "sair-modular-arithmetic", "run_id": run.info.run_id},
            "limitations": ["Five independent generator seeds; the official private seed is unknowable.", "Tier 3 only; full-tier integration evaluation remains required.", "Checked public Tier 3 answers were not read."],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({"tier3_exact": aggregate_accuracy, "passing_seeds": passing_seeds, "minimum_seed_accuracy": minimum_seed_accuracy, "wall_seconds": summary["runtime"]["wall_seconds"]})
        mlflow.log_artifact(str(args.output), artifact_path="canonical-preview")
        mlflow.set_tags({"status": "completed", "decision": summary["decision"]})
    print(json.dumps({"decision": summary["decision"], "aggregate": summary["aggregate"], "per_seed": {key: value["exact_accuracy"] for key, value in results.items()}, "mlflow_run_id": summary["mlflow"]["run_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
