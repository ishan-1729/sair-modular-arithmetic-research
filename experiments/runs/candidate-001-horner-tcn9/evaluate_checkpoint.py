"""Post-training checks on checked public rows and independent generator seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from model import WIDTH, HornerTCN9, predict_state


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_SRC = ROOT / "sources/github/modular-arithmetic-challenge/src"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(checkpoint_path: Path) -> tuple[HornerTCN9, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = HornerTCN9(hidden=config["hidden"], blocks=config["blocks"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_rows(model: HornerTCN9, rows: list[dict[str, str | int]]) -> dict[str, object]:
    """Apply sanctioned pairwise residues followed by the learned recurrence."""
    modulus = torch.tensor([int(row["p"]) for row in rows], dtype=torch.int64)
    multiplier = torch.tensor(
        [int(str(row["a"])) % int(row["p"]) for row in rows], dtype=torch.int64
    )
    multiplicand = torch.tensor(
        [int(str(row["b"])) % int(row["p"]) for row in rows], dtype=torch.int64
    )
    learned = torch.zeros(len(rows), dtype=torch.int64)
    started = time.perf_counter()
    for step in range(WIDTH):
        bit = (multiplier >> (WIDTH - 1 - step)) & 1
        learned, _ = predict_state(model, learned, multiplicand, modulus, bit)
    elapsed = time.perf_counter() - started
    expected = torch.tensor([int(row["expected"]) for row in rows], dtype=torch.int64)
    correct = learned.eq(expected)
    by_prime: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for p, ok in zip(modulus.tolist(), correct.tolist(), strict=True):
        by_prime[p][0] += int(ok)
        by_prime[p][1] += 1
    errors = [
        {
            "index": index,
            "p": int(modulus[index]),
            "predicted": int(learned[index]),
            "expected": int(expected[index]),
        }
        for index in range(len(rows))
        if not bool(correct[index])
    ]
    return {
        "cases": len(rows),
        "correct": int(correct.sum()),
        "exact_accuracy": float(correct.float().mean()),
        "inference_seconds": elapsed,
        "problems_per_second": len(rows) / elapsed,
        "per_prime": {
            str(p): {
                "correct": values[0],
                "cases": values[1],
                "exact_accuracy": values[0] / values[1],
            }
            for p, values in sorted(by_prime.items())
        },
        "errors": errors,
    }


def read_jsonl(path: Path) -> list[dict[str, str | int]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def independent_rows(seed: bytes, tier_id: int) -> list[dict[str, str | int]]:
    sys.path.insert(0, str(UPSTREAM_SRC))
    from modchallenge.config import TIERS
    from modchallenge.testgen.generator import _generate_tier_cases

    cases = _generate_tier_cases(
        tier=TIERS[tier_id],
        num_cases=100,
        num_primes=5,
        edge_cases=4,
        seed=seed,
    )
    return [case.to_full_dict() for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint)

    public = {}
    for tier_id in (1, 2):
        path = ROOT / f"data/external/public_benchmark/tier_{tier_id}.jsonl"
        public[f"tier_{tier_id}"] = {
            "source": str(path.relative_to(ROOT)),
            "source_sha256": digest(path),
            **predict_rows(model, read_jsonl(path)),
        }

    independent = {}
    seeds = [
        b"tier2-independent-20260712-a",
        b"tier2-independent-20260712-b",
        b"tier2-independent-20260712-c",
        b"tier2-independent-20260712-d",
        b"tier2-independent-20260712-e",
    ]
    for seed in seeds:
        key = hashlib.sha256(seed).hexdigest()[:16]
        independent[key] = {
            "seed_sha256": hashlib.sha256(seed).hexdigest(),
            "tier_1": predict_rows(model, independent_rows(seed, 1)),
            "tier_2": predict_rows(model, independent_rows(seed, 2)),
        }

    safe_path = args.checkpoint.with_suffix(".safetensors")
    save_file(
        {key: value.detach().contiguous() for key, value in checkpoint["state_dict"].items()},
        safe_path,
        metadata={
            "architecture": "HornerTCN9",
            "width": str(WIDTH),
            "source_checkpoint_sha256": digest(args.checkpoint),
        },
    )
    tier2_correct = sum(run["tier_2"]["correct"] for run in independent.values())
    tier2_cases = sum(run["tier_2"]["cases"] for run in independent.values())
    tier1_correct = sum(run["tier_1"]["correct"] for run in independent.values())
    tier1_cases = sum(run["tier_1"]["cases"] for run in independent.values())
    results = {
        "checkpoint": {
            "pytorch_path": str(args.checkpoint),
            "pytorch_sha256": digest(args.checkpoint),
            "safetensors_path": str(safe_path),
            "safetensors_sha256": digest(safe_path),
            "safetensors_bytes": safe_path.stat().st_size,
        },
        "public_checked": public,
        "independent_generator_seeds": independent,
        "independent_aggregate": {
            "tier_1": {
                "cases": tier1_cases,
                "correct": tier1_correct,
                "exact_accuracy": tier1_correct / tier1_cases,
            },
            "tier_2": {
                "cases": tier2_cases,
                "correct": tier2_correct,
                "exact_accuracy": tier2_correct / tier2_cases,
            },
        },
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "public_tier1": public["tier_1"]["exact_accuracy"],
        "public_tier2": public["tier_2"]["exact_accuracy"],
        "independent_tier1": results["independent_aggregate"]["tier_1"]["exact_accuracy"],
        "independent_tier2": results["independent_aggregate"]["tier_2"]["exact_accuracy"],
        "safetensors_sha256": results["checkpoint"]["safetensors_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
