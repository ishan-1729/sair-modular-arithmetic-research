"""Offline exact quotient and signed-carry labels for EXP-658.

All Python-integer arithmetic in this module is restricted to generated labels,
tests, and future evaluation truth.  Nothing here is imported by the learned
inference module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator

import torch
from torch import Tensor


QUOTIENT_CLASSES = (0, 1, 2)
SIGNED_CARRY_CLASSES = (-2, -1, 0, 1)
SIGNED_CARRY_TO_INDEX = {value: index for index, value in enumerate(SIGNED_CARRY_CLASSES)}


@dataclass(frozen=True, slots=True)
class LegalTransition:
    state: int
    multiplicand: int
    modulus: int
    operand_bit: int


@dataclass(frozen=True, slots=True)
class QuotientCarryTrace:
    inputs: LegalTransition
    width: int
    target: int
    quotient: int
    target_bits: tuple[int, ...]
    carry_in: tuple[int, ...]
    carry_out: tuple[int, ...]
    recurrence_numerators: tuple[int, ...]
    terminal_carry: int

    @property
    def signed_carry_class_indices(self) -> tuple[int, ...]:
        return tuple(SIGNED_CARRY_TO_INDEX[value] for value in self.carry_out)


def is_prime(value: int) -> bool:
    """Deterministic primality predicate for the unsigned 64-bit domain."""

    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime
    odd_part = value - 1
    power = 0
    while odd_part % 2 == 0:
        power += 1
        odd_part //= 2
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def small_prime_domain() -> tuple[int, ...]:
    """Return the exact exhaustive L0 prime domain, 3 through 251."""

    return tuple(value for value in range(3, 252) if is_prime(value))


def quotient_carry_trace(
    transition: LegalTransition, width: int
) -> QuotientCarryTrace:
    """Construct the exact offline Horner quotient/signed-carry trace.

    For local bit position ``i``, the verified recurrence is

    ``n_i = s_{i-1} + b*a_i - q*p_i + kappa_i`` and
    ``kappa_{i+1} = (n_i - target_i) // 2``.

    The scalar ``q`` is an offline label only.  One high tail position consumes
    the shifted state MSB and must end with signed carry zero.
    """

    state = transition.state
    multiplicand = transition.multiplicand
    modulus = transition.modulus
    operand_bit = transition.operand_bit
    if width < 2:
        raise ValueError("width must be at least two")
    if not 0 <= state < modulus or not 0 <= multiplicand < modulus:
        raise ValueError("state and multiplicand must be legal residues")
    if operand_bit not in (0, 1):
        raise ValueError("operand bit must be binary")
    if modulus.bit_length() > width:
        raise ValueError("modulus does not fit the declared width")

    unreduced = 2 * state + operand_bit * multiplicand
    quotient, target = divmod(unreduced, modulus)
    carry = 0
    target_bits: list[int] = []
    carry_in: list[int] = []
    carry_out: list[int] = []
    recurrence_numerators: list[int] = []
    for position in range(width):
        shifted_state_bit = 0 if position == 0 else (state >> (position - 1)) & 1
        multiplicand_bit = (multiplicand >> position) & 1
        modulus_bit = (modulus >> position) & 1
        target_bit = (target >> position) & 1
        numerator = (
            shifted_state_bit
            + operand_bit * multiplicand_bit
            - quotient * modulus_bit
            + carry
        )
        difference = numerator - target_bit
        if difference % 2:
            raise AssertionError("signed-carry recurrence numerator is not even")
        next_carry = difference // 2
        target_bits.append(target_bit)
        carry_in.append(carry)
        carry_out.append(next_carry)
        recurrence_numerators.append(numerator)
        carry = next_carry

    tail_numerator = ((state >> (width - 1)) & 1) + carry
    if tail_numerator % 2:
        raise AssertionError("terminal signed-carry numerator is not even")
    terminal_carry = tail_numerator // 2
    return QuotientCarryTrace(
        inputs=transition,
        width=width,
        target=target,
        quotient=quotient,
        target_bits=tuple(target_bits),
        carry_in=tuple(carry_in),
        carry_out=tuple(carry_out),
        recurrence_numerators=tuple(recurrence_numerators),
        terminal_carry=terminal_carry,
    )


def iter_exhaustive_small_prime_transitions() -> Iterator[LegalTransition]:
    for modulus in small_prime_domain():
        for state in range(modulus):
            for multiplicand in range(modulus):
                yield LegalTransition(state, multiplicand, modulus, 0)
                yield LegalTransition(state, multiplicand, modulus, 1)


@lru_cache(maxsize=None)
def prime_pool(width: int, count: int, seed: int) -> tuple[int, ...]:
    """Build a deterministic pool of exact-width primes for offline sampling."""

    if width not in (32, 64):
        raise ValueError("the L0 random domain is restricted to widths 32 and 64")
    if count < 1:
        raise ValueError("prime-pool count must be positive")
    rng = random.Random(seed)
    lower = 1 << (width - 1)
    upper = 1 << width
    primes: list[int] = []
    seen: set[int] = set()
    while len(primes) < count:
        candidate = rng.randrange(lower, upper) | 1
        if candidate not in seen and is_prime(candidate):
            primes.append(candidate)
            seen.add(candidate)
    return tuple(primes)


def iter_random_legal_transitions(
    *, width: int, count: int, seed: int, prime_count: int = 257
) -> Iterator[LegalTransition]:
    primes = prime_pool(width, prime_count, seed + 1)
    rng = random.Random(seed)
    for _ in range(count):
        modulus = primes[rng.randrange(len(primes))]
        yield LegalTransition(
            state=rng.randrange(modulus),
            multiplicand=rng.randrange(modulus),
            modulus=modulus,
            operand_bit=rng.randrange(2),
        )


def _bits(values: Iterable[int], width: int) -> Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def feature_tensor(transitions: Iterable[LegalTransition], width: int) -> Tensor:
    """Encode legal transitions as the six generic source feature planes."""

    rows = tuple(transitions)
    if not rows:
        raise ValueError("at least one transition is required")
    state = _bits((row.state for row in rows), width)
    multiplicand = _bits((row.multiplicand for row in rows), width)
    modulus = _bits((row.modulus for row in rows), width)
    operand_bit = torch.tensor(
        [row.operand_bit for row in rows], dtype=torch.float32
    )[:, None].expand(len(rows), width)
    lsb = torch.zeros((len(rows), width), dtype=torch.float32)
    msb = torch.zeros_like(lsb)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (state, multiplicand, modulus, operand_bit, lsb, msb), dim=1
    ).mul(2.0).sub(1.0)
