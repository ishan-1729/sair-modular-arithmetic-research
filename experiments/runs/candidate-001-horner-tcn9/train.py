"""Train and evaluate the nine-bit learned Horner transition on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model import WIDTH, HornerTCN9, ints_to_bits, make_features, predict_state


ROOT = Path(__file__).resolve().parents[3]
PUBLIC_FILES = (
    ROOT / "data/external/public_benchmark/tier_1.jsonl",
    ROOT / "data/external/public_benchmark/tier_2.jsonl",
)


@dataclass(frozen=True)
class Config:
    seed: int = 20260712
    hidden: int = 48
    blocks: int = 7
    steps: int = 2400
    batch_size: int = 768
    learning_rate: float = 0.0025
    min_learning_rate: float = 0.00012
    weight_decay: float = 0.00002
    worst_bit_weight: float = 0.45
    quotient_weight: float = 0.10
    random_state_fraction: float = 0.15
    composite_fraction: float = 0.15
    torch_threads: int = 4
    eval_cases: int = 100_000
    rollout_cases: int = 50_000


def primes_upto(limit: int) -> list[int]:
    sieve = [True] * (limit + 1)
    sieve[0:2] = [False, False]
    for value in range(2, math.isqrt(limit) + 1):
        if sieve[value]:
            sieve[value * value : limit + 1 : value] = [False] * (
                ((limit - value * value) // value) + 1
            )
    return [value for value, flag in enumerate(sieve) if flag]


def load_public_exclusions() -> tuple[set[tuple[int, int, int]], dict[str, str]]:
    """Read inputs only and create a reduced-pair denylist; never read expected."""
    excluded: set[tuple[int, int, int]] = set()
    checksums: dict[str, str] = {}
    for path in PUBLIC_FILES:
        payload = path.read_bytes()
        checksums[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
        for line in payload.decode("utf-8").splitlines():
            row = json.loads(line)
            p = int(row["p"])
            # The Tier-1 residue domains are saturated by the public set (for
            # p=2 there are only four possible pairs), so reduced-pair
            # exclusion is mathematically incompatible with learning Tier 1.
            # Those original 32-bit examples are still never training rows.
            # Tier-2 reduced pairs can and are withheld exactly.
            if p > 7:
                excluded.add((int(row["a"]) % p, int(row["b"]) % p, p))
    return excluded, checksums


class Sampler:
    def __init__(self, config: Config, exclusions: set[tuple[int, int, int]], seed: int) -> None:
        self.config = config
        self.exclusions = exclusions
        self.primes = torch.tensor(primes_upto(257), dtype=torch.int64)
        self.composites = torch.tensor(
            [value for value in range(2, 258) if value not in set(self.primes.tolist())],
            dtype=torch.int64,
        )
        self.generator = torch.Generator().manual_seed(seed)

    def _draw_pairs(self, size: int, allow_composites: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        use_composite = (
            torch.rand(size, generator=self.generator) < self.config.composite_fraction
            if allow_composites
            else torch.zeros(size, dtype=torch.bool)
        )
        modulus = self.primes[
            torch.randint(len(self.primes), (size,), generator=self.generator)
        ]
        if use_composite.any():
            modulus[use_composite] = self.composites[
                torch.randint(len(self.composites), (int(use_composite.sum()),), generator=self.generator)
            ]
        multiplier = (torch.rand(size, generator=self.generator) * modulus.double()).long()
        multiplicand = (torch.rand(size, generator=self.generator) * modulus.double()).long()

        # Resample complete reduced pairs that occur in the official public set.
        # This denylist affects inputs only; expected benchmark answers are unused.
        for index in range(size):
            while (int(multiplier[index]), int(multiplicand[index]), int(modulus[index])) in self.exclusions:
                multiplier[index] = torch.randint(
                    0, int(modulus[index]), (1,), generator=self.generator
                )
                multiplicand[index] = torch.randint(
                    0, int(modulus[index]), (1,), generator=self.generator
                )
        return multiplier, multiplicand, modulus

    def transitions(self, size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        multiplier, multiplicand, modulus = self._draw_pairs(size)
        step = torch.randint(0, WIDTH, (size,), generator=self.generator)
        prefix = multiplier >> (WIDTH - step)
        state = (prefix * multiplicand) % modulus
        bit = (multiplier >> (WIDTH - 1 - step)) & 1

        random_mask = torch.rand(size, generator=self.generator) < self.config.random_state_fraction
        if random_mask.any():
            state[random_mask] = (
                torch.rand(int(random_mask.sum()), generator=self.generator)
                * modulus[random_mask].double()
            ).long()
            bit[random_mask] = torch.randint(
                0, 2, (int(random_mask.sum()),), generator=self.generator
            )
        raw = 2 * state + bit * multiplicand
        target = raw % modulus
        quotient = raw // modulus
        return state, multiplicand, modulus, bit, target, quotient

    def rollouts(self, size: int, allow_composites: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        return self._draw_pairs(size, allow_composites=allow_composites)


def loss_fn(
    logits: Tensor,
    quotient_logits: Tensor,
    target: Tensor,
    quotient: Tensor,
    config: Config,
) -> tuple[Tensor, dict[str, float]]:
    target_bits = ints_to_bits(target)
    element = F.binary_cross_entropy_with_logits(logits, target_bits, reduction="none")
    mean_bce = element.mean()
    worst = element.max(1).values.mean()
    quotient_loss = F.cross_entropy(quotient_logits, quotient)
    loss = mean_bce + config.worst_bit_weight * worst + config.quotient_weight * quotient_loss
    with torch.no_grad():
        exact = (logits >= 0).eq(target_bits.bool()).all(1).float().mean()
    return loss, {
        "loss": float(loss),
        "mean_bce": float(mean_bce),
        "worst_bit_bce": float(worst),
        "quotient_ce": float(quotient_loss),
        "batch_exact": float(exact),
    }


@torch.inference_mode()
def evaluate_transitions(model: HornerTCN9, sampler: Sampler, cases: int) -> dict[str, float | int]:
    model.eval()
    correct = bit_correct = valid = processed = 0
    while processed < cases:
        size = min(4096, cases - processed)
        state, multiplicand, modulus, bit, target, _ = sampler.transitions(size)
        predicted, logits = predict_state(model, state, multiplicand, modulus, bit)
        correct += int(predicted.eq(target).sum())
        bit_correct += int((logits >= 0).eq(ints_to_bits(target).bool()).sum())
        valid += int((predicted < modulus).sum())
        processed += size
    return {
        "cases": cases,
        "correct": correct,
        "exact_accuracy": correct / cases,
        "bit_accuracy": bit_correct / (cases * WIDTH),
        "valid_residue_rate": valid / cases,
    }


@torch.inference_mode()
def evaluate_rollouts(
    model: HornerTCN9,
    sampler: Sampler,
    cases: int,
    steps: int,
    modulus_filter: set[int] | None = None,
) -> dict[str, object]:
    model.eval()
    correct = chain_correct = invalid_count = processed = 0
    per_modulus: dict[int, list[int]] = {}
    first_error = [0] * steps
    while processed < cases:
        size = min(4096, cases - processed)
        multiplier, multiplicand, modulus = sampler.rollouts(size)
        if modulus_filter is not None:
            allowed = torch.tensor(sorted(modulus_filter), dtype=torch.int64)
            modulus = allowed[torch.randint(len(allowed), (size,), generator=sampler.generator)]
            multiplier = (torch.rand(size, generator=sampler.generator) * modulus.double()).long()
            multiplicand = (torch.rand(size, generator=sampler.generator) * modulus.double()).long()
        learned = torch.zeros(size, dtype=torch.int64)
        oracle = torch.zeros_like(learned)
        whole_chain = torch.ones(size, dtype=torch.bool)
        seen_error = torch.zeros(size, dtype=torch.bool)
        invalid = torch.zeros(size, dtype=torch.bool)
        start = WIDTH - steps
        for local_step, bit_index in enumerate(range(start, WIDTH)):
            bit = (multiplier >> (WIDTH - 1 - bit_index)) & 1
            learned, _ = predict_state(model, learned, multiplicand, modulus, bit)
            oracle = (2 * oracle + bit * multiplicand) % modulus
            step_ok = learned.eq(oracle)
            newly_wrong = (~step_ok) & (~seen_error)
            first_error[local_step] += int(newly_wrong.sum())
            seen_error |= ~step_ok
            whole_chain &= step_ok
            invalid |= learned >= modulus
        batch_correct = learned.eq((multiplier * multiplicand) % modulus)
        correct += int(batch_correct.sum())
        chain_correct += int(whole_chain.sum())
        invalid_count += int(invalid.sum())
        for p in modulus.unique().tolist():
            mask = modulus.eq(p)
            stats = per_modulus.setdefault(int(p), [0, 0])
            stats[0] += int(batch_correct[mask].sum())
            stats[1] += int(mask.sum())
        processed += size
    return {
        "cases": cases,
        "correct": correct,
        "exact_accuracy": correct / cases,
        "all_steps_correct": chain_correct,
        "all_steps_exact_accuracy": chain_correct / cases,
        "any_invalid_residue_rate": invalid_count / cases,
        "steps": steps,
        "first_error_by_step": first_error,
        "per_modulus": {
            str(p): {"correct": values[0], "cases": values[1], "exact_accuracy": values[0] / values[1]}
            for p, values in sorted(per_modulus.items())
        },
    }


def train(config: Config, output_dir: Path) -> tuple[HornerTCN9, dict[str, object]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    exclusions, public_hashes = load_public_exclusions()
    sampler = Sampler(config, exclusions, config.seed + 1)
    model = HornerTCN9(config.hidden, config.blocks)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.steps, eta_min=config.min_learning_rate
    )
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, config.steps + 1):
        state, multiplicand, modulus, bit, target, quotient = sampler.transitions(config.batch_size)
        logits, quotient_logits = model(make_features(state, multiplicand, modulus, bit), modulus)
        loss, stats = loss_fn(logits, quotient_logits, target, quotient, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            record = {
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.perf_counter() - started,
                **stats,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    training_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "horner_tcn9.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "parameters": model.parameter_count(),
            "training_seconds": training_seconds,
            "history": history,
            "excluded_public_reduced_pairs": len(exclusions),
            "public_input_source_sha256": public_hashes,
        },
        checkpoint,
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "training_seconds": training_seconds,
        "history": history,
        "public_exclusion": {
            "reduced_pairs": len(exclusions),
            "source_sha256": public_hashes,
            "expected_field_used": False,
        },
    }
    return model, metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "run")
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--blocks", type=int, default=Config.blocks)
    parser.add_argument("--eval-cases", type=int, default=Config.eval_cases)
    parser.add_argument("--rollout-cases", type=int, default=Config.rollout_cases)
    args = parser.parse_args()
    config = Config(
        steps=args.steps,
        batch_size=args.batch_size,
        hidden=args.hidden,
        blocks=args.blocks,
        eval_cases=args.eval_cases,
        rollout_cases=args.rollout_cases,
    )
    model, metadata = train(config, args.output)
    eval_started = time.perf_counter()
    exclusions, _ = load_public_exclusions()
    eval_sampler = Sampler(config, exclusions, config.seed + 100_000)
    tier1 = {2, 3, 5, 7}
    tier2 = set(primes_upto(257)) - tier1
    metrics = {
        "config": asdict(config),
        "architecture": {
            "name": "learned_width9_horner_tcn",
            "parameters": model.parameter_count(),
            "dilations": model.dilations,
            "modulus_embedding": model.modulus_embedding_enabled,
            "symbolic_arithmetic_in_forward": False,
        },
        "training": metadata,
        "evaluation": {
            "heldout_transitions": evaluate_transitions(model, eval_sampler, config.eval_cases),
            "tier1_rollout_9step": evaluate_rollouts(
                model, eval_sampler, config.rollout_cases // 5, 9, tier1
            ),
            "tier2_rollout_8step": evaluate_rollouts(
                model, eval_sampler, config.rollout_cases, 8, tier2 - {257}
            ),
            "tier2_rollout_9step": evaluate_rollouts(
                model, eval_sampler, config.rollout_cases, 9, tier2
            ),
            "p257_rollout_9step": evaluate_rollouts(
                model, eval_sampler, config.rollout_cases // 5, 9, {257}
            ),
        },
    }
    metrics["evaluation_seconds"] = time.perf_counter() - eval_started
    checkpoint = Path(metadata["checkpoint"])
    metrics["checkpoint_sha256"] = sha256(checkpoint)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "metrics": str(metrics_path),
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "heldout_transition": metrics["evaluation"]["heldout_transitions"]["exact_accuracy"],
        "tier1_9step": metrics["evaluation"]["tier1_rollout_9step"]["exact_accuracy"],
        "tier2_8step": metrics["evaluation"]["tier2_rollout_8step"]["exact_accuracy"],
        "tier2_9step": metrics["evaluation"]["tier2_rollout_9step"]["exact_accuracy"],
        "p257_9step": metrics["evaluation"]["p257_rollout_9step"]["exact_accuracy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
