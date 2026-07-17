"""Offline labels and legal tensor features for EXP-637."""

from __future__ import annotations

import random

import torch
from torch import Tensor


def bits(values: list[int], width: int) -> Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def features(
    states: list[int],
    rights: list[int],
    moduli: list[int],
    input_bits: list[int],
    width: int,
) -> Tensor:
    """Build the same six legal feature planes used by the radix-2 teacher."""
    if not all(bit in (0, 1) for bit in input_bits):
        raise ValueError("input bits must be zero or one")
    batch = len(states)
    bit_plane = torch.tensor(input_bits, dtype=torch.float32)[:, None].expand(
        batch, width
    )
    lsb = torch.zeros(batch, width)
    msb = torch.zeros(batch, width)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    return torch.stack(
        (
            bits(states, width),
            bits(rights, width),
            bits(moduli, width),
            bit_plane,
            lsb,
            msb,
        ),
        dim=1,
    ).mul(2.0).sub(1.0)


def macro_features(
    states: list[int],
    rights: list[int],
    moduli: list[int],
    digits: list[int],
    width: int,
) -> tuple[Tensor, Tensor]:
    """Return phase-1 high-bit features and phase-2 low-bit context."""
    if not all(0 <= digit < 4 for digit in digits):
        raise ValueError("radix-4 digits must be between zero and three")
    high = [digit >> 1 for digit in digits]
    low = [digit & 1 for digit in digits]
    return (
        features(states, rights, moduli, high, width),
        features([0] * len(states), rights, moduli, low, width),
    )


def exact_radix2_transition(
    state: int, right: int, modulus: int, input_bit: int
) -> tuple[int, int]:
    """Offline label only."""
    if not 0 <= state < modulus or not 0 <= right < modulus:
        raise ValueError("state and right operand must be valid residues")
    if input_bit not in (0, 1):
        raise ValueError("input bit must be zero or one")
    raw = 2 * state + input_bit * right
    return raw % modulus, raw // modulus


def exact_macro_transition(
    state: int, right: int, modulus: int, digit: int
) -> tuple[int, int, int, int]:
    """Offline labels `(intermediate, q1, final, q2)`."""
    if not 0 <= digit < 4:
        raise ValueError("radix-4 digit must be between zero and three")
    intermediate, q1 = exact_radix2_transition(
        state, right, modulus, digit >> 1
    )
    final, q2 = exact_radix2_transition(
        intermediate, right, modulus, digit & 1
    )
    return intermediate, q1, final, q2


class MacroSampler:
    def __init__(self, primes: list[int], seed: int) -> None:
        self.primes = primes
        self.rng = random.Random(seed)

    def transitions(self, size: int):
        states: list[int] = []
        rights: list[int] = []
        moduli: list[int] = []
        digits: list[int] = []
        intermediates: list[int] = []
        finals: list[int] = []
        q1s: list[int] = []
        q2s: list[int] = []
        for _ in range(size):
            modulus = self.rng.choice(self.primes)
            state = self.rng.randrange(modulus)
            right = self.rng.randrange(modulus)
            digit = self.rng.randrange(4)
            intermediate, q1, final, q2 = exact_macro_transition(
                state, right, modulus, digit
            )
            states.append(state)
            rights.append(right)
            moduli.append(modulus)
            digits.append(digit)
            intermediates.append(intermediate)
            finals.append(final)
            q1s.append(q1)
            q2s.append(q2)
        return (
            states,
            rights,
            moduli,
            digits,
            intermediates,
            finals,
            torch.tensor(q1s, dtype=torch.long),
            torch.tensor(q2s, dtype=torch.long),
        )

    def pairs(self, size: int) -> tuple[list[int], list[int], list[int]]:
        moduli = [self.rng.choice(self.primes) for _ in range(size)]
        return (
            [self.rng.randrange(prime) for prime in moduli],
            [self.rng.randrange(prime) for prime in moduli],
            moduli,
        )


def teacher_rollout(left: int, right: int, modulus: int, width: int) -> int:
    """Offline exact macro path used only by tests and evaluation truth."""
    if width % 2:
        raise ValueError("macro rollout width must be even")
    state = 0
    for position in range(width // 2):
        shift = width - 2 * (position + 1)
        digit = (left >> shift) & 3
        _, _, state, _ = exact_macro_transition(state, right, modulus, digit)
    return state
