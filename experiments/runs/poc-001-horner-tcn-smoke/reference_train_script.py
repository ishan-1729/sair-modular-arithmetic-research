#!/usr/bin/env python3
"""CPU-scale POC for a learned Horner modular transition.

The neural cell receives binary channels for (t, b, p, multiplier_bit) and
predicts all bits of t' = (2*t + multiplier_bit*b) mod p.  The equation is
used only to create labels and metrics; the model forward path contains only
learned convolutions and pointwise nonlinearities.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class Config:
    seed: int = 20260711
    width: int = 8
    min_prime_bits: int = 4
    max_prime_bits: int | None = None
    holdout_fraction: float = 0.25
    augment_composite_moduli: bool = False
    hidden: int = 32
    blocks: int = 5
    steps: int = 1800
    batch_size: int = 256
    lr: float = 2.5e-3
    weight_decay: float = 2e-5
    worst_bit_weight: float = 0.35
    one_bit_probability: float = 0.70
    train_a_bits: int = 16
    eval_transitions: int = 20000
    eval_rollouts: int = 4000
    threads: int = 4


def primes_upto(n: int) -> list[int]:
    sieve = [True] * (n + 1)
    sieve[:2] = [False, False]
    for x in range(2, int(math.isqrt(n)) + 1):
        if sieve[x]:
            sieve[x * x : n + 1 : x] = [False] * (((n - x * x) // x) + 1)
    return [x for x, yes in enumerate(sieve) if yes]


def split_primes(cfg: Config) -> tuple[list[int], list[int]]:
    """Seeded, width-stratified 80/20 split; no held-out p appears in train."""
    rng = random.Random(cfg.seed)
    groups: dict[int, list[int]] = {}
    for p in primes_upto((1 << cfg.width) - 1):
        if p.bit_length() >= cfg.min_prime_bits and (
            cfg.max_prime_bits is None or p.bit_length() <= cfg.max_prime_bits
        ):
            groups.setdefault(p.bit_length(), []).append(p)
    train: list[int] = []
    heldout: list[int] = []
    for bits in sorted(groups):
        group = groups[bits][:]
        rng.shuffle(group)
        n_holdout = max(1, math.ceil(cfg.holdout_fraction * len(group)))
        heldout.extend(sorted(group[:n_holdout]))
        train.extend(sorted(group[n_holdout:]))
    heldout = sorted(heldout)
    if cfg.augment_composite_moduli:
        max_bits = cfg.max_prime_bits or cfg.width
        all_moduli = range(1 << (cfg.min_prime_bits - 1), 1 << max_bits)
        # Prime holdouts remain strictly unseen; composites add boundary coverage.
        train = [p for p in all_moduli if p not in heldout]
    return sorted(train), heldout


def ints_to_bits(x: Tensor, width: int) -> Tensor:
    """Convert integer [batch] tensor to float LSB-first bits [batch,width]."""
    shifts = torch.arange(width, device=x.device, dtype=torch.int64)
    return ((x[:, None] >> shifts[None, :]) & 1).float()


def make_features(t: Tensor, b: Tensor, p: Tensor, bit: Tensor, width: int) -> Tensor:
    """Raw factor channels only: no bit*b or other arithmetic feature."""
    batch = t.shape[0]
    t_bits = ints_to_bits(t, width)
    b_bits = ints_to_bits(b, width)
    p_bits = ints_to_bits(p, width)
    bit_plane = bit.float()[:, None].expand(batch, width)
    lsb = torch.zeros((batch, width), device=t.device)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    # Center binary inputs; retain explicit boundary markers for zero padding.
    x = torch.stack((t_bits, b_bits, p_bits, bit_plane, lsb, msb), dim=1)
    return x.mul(2.0).sub(1.0)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(F.silu(self.norm(x)))
        y = self.conv2(F.silu(y))
        return x + y


class HornerTCN(nn.Module):
    """Non-causal local convolutional transition cell with hard-bit rollout."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.width = cfg.width
        self.stem = nn.Conv1d(6, cfg.hidden, 1)
        dilations = [1 << (i % max(1, math.ceil(math.log2(cfg.width)))) for i in range(cfg.blocks)]
        self.dilations = dilations
        self.body = nn.ModuleList(
            ResidualDilatedBlock(cfg.hidden, dilation) for dilation in dilations
        )
        self.head_norm = nn.GroupNorm(1, cfg.hidden)
        self.head = nn.Conv1d(cfg.hidden, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        for block in self.body:
            h = block(h)
        return self.head(F.silu(self.head_norm(h))).squeeze(1)


class TrajectorySampler:
    """Vectorized sampler of teacher-forced states actually visited by Horner."""

    def __init__(self, primes: list[int], cfg: Config, device: torch.device, seed: int):
        self.primes = torch.tensor(primes, dtype=torch.int64, device=device)
        self.cfg = cfg
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def sample(self, batch_size: int, a_bits: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        a_bits = a_bits or self.cfg.train_a_bits
        p_idx = torch.randint(len(self.primes), (batch_size,), generator=self.gen, device=self.device)
        p = self.primes[p_idx]
        # Uniform residues for b, but t is always generated from a real prefix.
        b = (torch.rand(batch_size, generator=self.gen, device=self.device) * p.float()).long()
        a = torch.randint(0, 1 << a_bits, (batch_size,), generator=self.gen, device=self.device)
        step = torch.randint(0, a_bits, (batch_size,), generator=self.gen, device=self.device)
        # Before `step`, Horner has consumed the `step` most-significant bits.
        prefix_shift = a_bits - step
        prefix = a >> prefix_shift
        t = (prefix * b) % p
        # Conditional next-bit sampling keeps (prefix, bit) on a realizable true
        # trajectory while spending more updates on the harder add-and-reduce arm.
        bit = (
            torch.rand(batch_size, generator=self.gen, device=self.device)
            < self.cfg.one_bit_probability
        ).long()
        target = (2 * t + bit * b) % p
        return t, b, p, bit, target


def loss_fn(logits: Tensor, targets: Tensor, cfg: Config) -> tuple[Tensor, dict[str, float]]:
    target_bits = ints_to_bits(targets, cfg.width)
    element = F.binary_cross_entropy_with_logits(logits, target_bits, reduction="none")
    mean_bce = element.mean()
    worst_bit = element.max(dim=1).values.mean()
    loss = mean_bce + cfg.worst_bit_weight * worst_bit
    with torch.no_grad():
        pred = (logits >= 0).to(target_bits.dtype)
        exact = pred.eq(target_bits).all(dim=1).float().mean()
        bit_acc = pred.eq(target_bits).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "mean_bce": float(mean_bce.detach()),
        "worst_bit_bce": float(worst_bit.detach()),
        "batch_exact": float(exact),
        "batch_bit_acc": float(bit_acc),
    }


@torch.inference_mode()
def predict_int(model: nn.Module, t: Tensor, b: Tensor, p: Tensor, bit: Tensor, cfg: Config) -> tuple[Tensor, Tensor]:
    logits = model(make_features(t, b, p, bit, cfg.width))
    pred_bits = (logits >= 0).long()
    weights = 1 << torch.arange(cfg.width, device=t.device, dtype=torch.int64)
    pred_int = (pred_bits * weights[None, :]).sum(dim=1)
    return pred_int, logits


@torch.inference_mode()
def evaluate_transitions(model: nn.Module, sampler: TrajectorySampler, cfg: Config) -> dict[str, float]:
    total = cfg.eval_transitions
    batch_size = min(2048, total)
    n = exact = bit_correct = bit_total = valid = 0
    exact_by_bit = {0: [0, 0], 1: [0, 0]}
    margins: list[Tensor] = []
    while n < total:
        size = min(batch_size, total - n)
        t, b, p, bit, target = sampler.sample(size)
        pred, logits = predict_int(model, t, b, p, bit, cfg)
        target_bits = ints_to_bits(target, cfg.width).bool()
        pred_bits = logits >= 0
        is_exact = pred.eq(target)
        exact += int(is_exact.sum())
        bit_correct += int(pred_bits.eq(target_bits).sum())
        bit_total += size * cfg.width
        valid += int((pred < p).sum())
        signed_margin = torch.where(target_bits, logits, -logits)
        margins.append(signed_margin.min(dim=1).values.cpu())
        for bit_value in (0, 1):
            mask = bit == bit_value
            exact_by_bit[bit_value][0] += int(is_exact[mask].sum())
            exact_by_bit[bit_value][1] += int(mask.sum())
        n += size
    all_margins = torch.cat(margins)
    return {
        "samples": total,
        "exact_accuracy": exact / total,
        "bit_accuracy": bit_correct / bit_total,
        "valid_residue_rate": valid / total,
        "exact_bit_0": exact_by_bit[0][0] / exact_by_bit[0][1],
        "exact_bit_1": exact_by_bit[1][0] / exact_by_bit[1][1],
        "worst_bit_margin_mean": float(all_margins.mean()),
        "worst_bit_margin_p01": float(torch.quantile(all_margins, 0.01)),
    }


@torch.inference_mode()
def evaluate_rollouts(
    model: nn.Module,
    primes: list[int],
    cfg: Config,
    a_bits: int,
    seed: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    total = cfg.eval_rollouts
    prime_tensor = torch.tensor(primes, dtype=torch.int64, device=device)
    final_exact = all_steps_exact = invalid_any = 0
    first_error_counts = [0] * a_bits
    # Keep memory bounded while preserving fully autonomous hard-bit feedback.
    batch_size = min(1024, total)
    n = 0
    while n < total:
        size = min(batch_size, total - n)
        p = prime_tensor[torch.randint(len(primes), (size,), generator=gen, device=device)]
        b = (torch.rand(size, generator=gen, device=device) * p.float()).long()
        if a_bits <= 62:
            a = torch.randint(0, 1 << a_bits, (size,), generator=gen, device=device)
        else:
            raise ValueError("This small-width POC only supports <=62 rollout bits")
        learned_t = torch.zeros(size, dtype=torch.int64, device=device)
        oracle_t = torch.zeros_like(learned_t)
        chain_good = torch.ones(size, dtype=torch.bool, device=device)
        ever_invalid = torch.zeros_like(chain_good)
        first_seen = torch.zeros_like(chain_good)
        for step in range(a_bits):
            bit = (a >> (a_bits - 1 - step)) & 1
            learned_t, _ = predict_int(model, learned_t, b, p, bit, cfg)
            oracle_t = (2 * oracle_t + bit * b) % p
            step_good = learned_t.eq(oracle_t)
            newly_bad = (~step_good) & (~first_seen)
            first_error_counts[step] += int(newly_bad.sum())
            first_seen |= ~step_good
            chain_good &= step_good
            ever_invalid |= learned_t >= p
        target = (a * b) % p
        final_exact += int(learned_t.eq(target).sum())
        all_steps_exact += int(chain_good.sum())
        invalid_any += int(ever_invalid.sum())
        n += size
    return {
        "samples": total,
        "a_bits": a_bits,
        "final_exact_accuracy": final_exact / total,
        "all_steps_exact_accuracy": all_steps_exact / total,
        "any_invalid_residue_rate": invalid_any / total,
        "first_error_by_step": first_error_counts,
    }


def train(cfg: Config, output_dir: Path) -> dict:
    started = time.perf_counter()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")

    if cfg.max_prime_bits is not None and cfg.max_prime_bits > cfg.width:
        raise ValueError("max_prime_bits cannot exceed the binary canvas width")
    train_primes, heldout_primes = split_primes(cfg)
    model = HornerTCN(cfg).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.08
    )
    train_sampler = TrajectorySampler(train_primes, cfg, device, cfg.seed + 1)
    history: list[dict] = []
    train_started = time.perf_counter()
    model.train()
    for step in range(1, cfg.steps + 1):
        # First 15% is a width curriculum, while still excluding held-out primes.
        if step <= int(cfg.steps * 0.15):
            max_bits = min(cfg.width, cfg.min_prime_bits + step * 2 // max(1, int(cfg.steps * 0.15)))
            eligible = [p for p in train_primes if p.bit_length() <= max_bits]
            sampler = TrajectorySampler(eligible, cfg, device, cfg.seed + 1000 + step)
        else:
            sampler = train_sampler
        t, b, p, bit, target = sampler.sample(cfg.batch_size)
        logits = model(make_features(t, b, p, bit, cfg.width))
        loss, stats = loss_fn(logits, target, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == cfg.steps:
            record = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.perf_counter() - train_started,
                **stats,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    train_seconds = time.perf_counter() - train_started

    model.eval()
    eval_started = time.perf_counter()
    heldout_sampler = TrajectorySampler(heldout_primes, cfg, device, cfg.seed + 2)
    seen_sampler = TrajectorySampler(train_primes, cfg, device, cfg.seed + 3)
    transition_heldout = evaluate_transitions(model, heldout_sampler, cfg)
    transition_seen = evaluate_transitions(model, seen_sampler, cfg)
    rollouts_heldout = {
        str(bits): evaluate_rollouts(
            model, heldout_primes, cfg, bits, cfg.seed + 100 + bits
        )
        for bits in (8, 16, 32)
    }
    rollouts_seen = {
        "16": evaluate_rollouts(model, train_primes, cfg, 16, cfg.seed + 200)
    }
    eval_seconds = time.perf_counter() - eval_started

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "train_moduli": train_primes,
            "heldout_primes": heldout_primes,
        },
        output_dir / "horner_tcn.pt",
    )
    results = {
        "status": "research_poc_not_competition_ready",
        "config": asdict(cfg),
        "architecture": {
            "name": "noncausal_dilated_residual_tcn",
            "input_channels": ["t_bits", "b_bits", "p_bits", "multiplier_bit", "lsb_marker", "msb_marker"],
            "dilations": model.dilations,
            "parameters": parameter_count,
            "hard_binary_feedback": True,
            "forward_path_uses_symbolic_arithmetic": False,
        },
        "split": {
            "strategy": "seeded_width_stratified_holdout",
            "train_moduli": train_primes,
            "composite_augmentation": cfg.augment_composite_moduli,
            "heldout_primes": heldout_primes,
        },
        "timing_seconds": {
            "train": train_seconds,
            "eval": eval_seconds,
            "total": time.perf_counter() - started,
        },
        "transition": {
            "seen_training_moduli": transition_seen,
            "heldout_primes": transition_heldout,
        },
        "rollout": {
            "seen_training_moduli": rollouts_seen,
            "heldout_primes": rollouts_heldout,
        },
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    return results


def parse_args() -> tuple[Config, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "run")
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--blocks", type=int, default=Config.blocks)
    parser.add_argument("--width", type=int, default=Config.width)
    parser.add_argument("--min-prime-bits", type=int, default=Config.min_prime_bits)
    parser.add_argument("--max-prime-bits", type=int, default=Config.max_prime_bits)
    parser.add_argument("--holdout-fraction", type=float, default=Config.holdout_fraction)
    parser.add_argument("--augment-composite-moduli", action="store_true")
    parser.add_argument("--one-bit-probability", type=float, default=Config.one_bit_probability)
    parser.add_argument("--eval-transitions", type=int, default=Config.eval_transitions)
    parser.add_argument("--eval-rollouts", type=int, default=Config.eval_rollouts)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--threads", type=int, default=Config.threads)
    args = parser.parse_args()
    cfg = Config(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        hidden=args.hidden,
        blocks=args.blocks,
        width=args.width,
        min_prime_bits=args.min_prime_bits,
        max_prime_bits=args.max_prime_bits,
        holdout_fraction=args.holdout_fraction,
        augment_composite_moduli=args.augment_composite_moduli,
        one_bit_probability=args.one_bit_probability,
        eval_transitions=args.eval_transitions,
        eval_rollouts=args.eval_rollouts,
        threads=args.threads,
    )
    return cfg, args.output_dir


if __name__ == "__main__":
    config, out = parse_args()
    train(config, out)
