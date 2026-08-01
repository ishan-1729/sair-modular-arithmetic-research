"""Offline radix-4 generation and tensor feature construction."""

from __future__ import annotations

import random

import torch


def bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def features(
    states: list[int],
    rights: list[int],
    moduli: list[int],
    digits: list[int],
    width: int,
) -> torch.Tensor:
    """Build legal bit planes; digit planes are low bit then high bit."""
    if not all(0 <= digit < 4 for digit in digits):
        raise ValueError("radix-4 digits must be between zero and three")
    batch = len(states)
    low = torch.tensor([digit & 1 for digit in digits], dtype=torch.float32)
    high = torch.tensor([digit >> 1 for digit in digits], dtype=torch.float32)
    low_plane = low[:, None].expand(batch, width)
    high_plane = high[:, None].expand(batch, width)
    lsb = torch.zeros(batch, width)
    msb = torch.zeros(batch, width)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (
            bits(states, width),
            bits(rights, width),
            bits(moduli, width),
            low_plane,
            high_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul(2.0).sub(1.0)


def exact_transition(
    state: int, right: int, modulus: int, digit: int
) -> tuple[int, int]:
    """Offline label only: return target residue and quotient class."""
    if not 0 <= state < modulus or not 0 <= right < modulus:
        raise ValueError("state and right operand must be valid residues")
    if not 0 <= digit < 4:
        raise ValueError("radix-4 digit must be between zero and three")
    raw = 4 * state + digit * right
    return raw % modulus, raw // modulus


class Radix4Sampler:
    def __init__(self, primes: list[int], seed: int) -> None:
        self.primes = primes
        self.rng = random.Random(seed)

    def transitions(self, size: int):
        states = []
        rights = []
        moduli = []
        digits = []
        targets = []
        quotients = []
        for _ in range(size):
            modulus = self.rng.choice(self.primes)
            state = self.rng.randrange(modulus)
            right = self.rng.randrange(modulus)
            digit = self.rng.randrange(4)
            target, quotient = exact_transition(state, right, modulus, digit)
            states.append(state)
            rights.append(right)
            moduli.append(modulus)
            digits.append(digit)
            targets.append(target)
            quotients.append(quotient)
        return (
            states,
            rights,
            moduli,
            digits,
            targets,
            torch.tensor(quotients, dtype=torch.long),
        )

    def pairs(self, size: int) -> tuple[list[int], list[int], list[int]]:
        moduli = [self.rng.choice(self.primes) for _ in range(size)]
        return (
            [self.rng.randrange(prime) for prime in moduli],
            [self.rng.randrange(prime) for prime in moduli],
            moduli,
        )


def teacher_rollout(left: int, right: int, modulus: int, width: int) -> int:
    """Offline radix-4 Horner label path for tests and evaluation truth."""
    if width % 2:
        raise ValueError("radix-4 rollout width must be even")
    state = 0
    for position in range(width // 2):
        shift = width - 2 * (position + 1)
        digit = (left >> shift) & 3
        state, _ = exact_transition(state, right, modulus, digit)
    return state

