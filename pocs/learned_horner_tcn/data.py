"""Offline trajectory generation and prime-disjoint split helpers."""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor

from .config import POCConfig


def primes_upto(limit: int) -> list[int]:
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for value in range(2, math.isqrt(limit) + 1):
        if sieve[value]:
            sieve[value * value : limit + 1 : value] = [False] * (
                ((limit - value * value) // value) + 1
            )
    return [value for value, is_prime in enumerate(sieve) if is_prime]


def split_primes(config: POCConfig) -> tuple[list[int], list[int]]:
    """Create a deterministic width-stratified prime identity split."""
    rng = random.Random(config.seed)
    groups: dict[int, list[int]] = {}
    for prime in primes_upto((1 << config.width) - 1):
        bits = prime.bit_length()
        if bits < config.min_prime_bits:
            continue
        if config.max_prime_bits is not None and bits > config.max_prime_bits:
            continue
        groups.setdefault(bits, []).append(prime)

    training: list[int] = []
    heldout: list[int] = []
    for bits in sorted(groups):
        group = groups[bits][:]
        rng.shuffle(group)
        holdout_count = max(1, math.ceil(config.holdout_fraction * len(group)))
        heldout.extend(sorted(group[:holdout_count]))
        training.extend(sorted(group[holdout_count:]))
    if not training or not heldout:
        raise ValueError("prime split must contain both training and held-out values")
    return sorted(training), sorted(heldout)


def training_moduli(
    config: POCConfig,
    training_primes: list[int],
    heldout_primes: list[int],
) -> list[int]:
    """Optionally densify training with composites while preserving prime holdout."""
    if not config.augment_composite_moduli:
        return training_primes
    minimum = 1 << (config.min_prime_bits - 1)
    maximum_bits = config.max_prime_bits or config.width
    maximum = min((1 << maximum_bits) - 1, (1 << config.width) - 1)
    heldout = set(heldout_primes)
    return [modulus for modulus in range(minimum, maximum + 1) if modulus not in heldout]


class TrajectorySampler:
    """Sample legal local states from prefixes of true Horner trajectories."""

    def __init__(self, primes: list[int], config: POCConfig, device: torch.device, seed: int):
        if not primes:
            raise ValueError("at least one prime is required")
        self.primes = torch.tensor(primes, dtype=torch.int64, device=device)
        self.config = config
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def sample(self, batch_size: int, operand_bits: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return `(state, b, p, bit, target)`; arithmetic is label-only."""
        operand_bits = operand_bits or self.config.train_operand_bits
        if operand_bits > 62:
            raise ValueError("the small POC sampler is limited to at most 62 operand bits")
        indices = torch.randint(
            len(self.primes),
            (batch_size,),
            generator=self.generator,
            device=self.device,
        )
        modulus = self.primes[indices]
        multiplicand = (
            torch.rand(batch_size, generator=self.generator, device=self.device)
            * modulus.float()
        ).long()
        multiplier = torch.randint(
            0,
            1 << operand_bits,
            (batch_size,),
            generator=self.generator,
            device=self.device,
        )
        step = torch.randint(
            0,
            operand_bits,
            (batch_size,),
            generator=self.generator,
            device=self.device,
        )
        prefix = multiplier >> (operand_bits - step)
        state = (prefix * multiplicand) % modulus
        # Oversample the harder add-and-reduce branch while keeping every
        # `(state, bit)` pair realizable as part of some true trajectory.
        bit = (
            torch.rand(batch_size, generator=self.generator, device=self.device)
            < self.config.one_bit_probability
        ).long()
        target = (2 * state + bit * multiplicand) % modulus
        return state, multiplicand, modulus, bit, target
