"""Exact transition, autonomous rollout, and compliance-control metrics."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

from .config import POCConfig
from .data import TrajectorySampler
from .model import HornerTCN, ints_to_bits, predict_state


@torch.inference_mode()
def evaluate_transitions(
    model: HornerTCN,
    sampler: TrajectorySampler,
    config: POCConfig,
    cases: int | None = None,
) -> dict[str, Any]:
    total = cases or config.transition_cases
    batch_size = min(2048, total)
    processed = exact = bit_correct = valid = 0
    while processed < total:
        size = min(batch_size, total - processed)
        state, multiplicand, modulus, bit, target = sampler.sample(size)
        predicted, logits = predict_state(model, state, multiplicand, modulus, bit)
        target_bits = ints_to_bits(target, config.width).bool()
        exact += int(predicted.eq(target).sum())
        bit_correct += int((logits >= 0).eq(target_bits).sum())
        valid += int((predicted < modulus).sum())
        processed += size
    return {
        "cases": total,
        "correct": exact,
        "exact_accuracy": exact / total,
        "bit_accuracy": bit_correct / (total * config.width),
        "valid_residue_rate": valid / total,
    }


@torch.inference_mode()
def evaluate_rollout(
    model: HornerTCN,
    primes: list[int],
    config: POCConfig,
    operand_bits: int,
    seed: int,
    cases: int | None = None,
) -> dict[str, Any]:
    if operand_bits > 62:
        raise ValueError("the small POC evaluator is limited to at most 62 operand bits")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    total = cases or config.rollout_cases
    prime_tensor = torch.tensor(primes, dtype=torch.int64, device=device)
    final_correct = chain_correct = any_invalid = processed = 0
    first_error_by_step = [0] * operand_bits
    while processed < total:
        size = min(1024, total - processed)
        modulus = prime_tensor[
            torch.randint(len(primes), (size,), generator=generator, device=device)
        ]
        multiplicand = (
            torch.rand(size, generator=generator, device=device) * modulus.float()
        ).long()
        multiplier = torch.randint(
            0,
            1 << operand_bits,
            (size,),
            generator=generator,
            device=device,
        )
        learned_state = torch.zeros(size, dtype=torch.int64, device=device)
        oracle_state = torch.zeros_like(learned_state)
        whole_chain = torch.ones(size, dtype=torch.bool, device=device)
        first_seen = torch.zeros_like(whole_chain)
        invalid = torch.zeros_like(whole_chain)
        for step in range(operand_bits):
            bit = (multiplier >> (operand_bits - 1 - step)) & 1
            learned_state, _ = predict_state(
                model, learned_state, multiplicand, modulus, bit
            )
            oracle_state = (2 * oracle_state + bit * multiplicand) % modulus
            step_correct = learned_state.eq(oracle_state)
            newly_wrong = (~step_correct) & (~first_seen)
            first_error_by_step[step] += int(newly_wrong.sum())
            first_seen |= ~step_correct
            whole_chain &= step_correct
            invalid |= learned_state >= modulus
        target = (multiplier * multiplicand) % modulus
        final_correct += int(learned_state.eq(target).sum())
        chain_correct += int(whole_chain.sum())
        any_invalid += int(invalid.sum())
        processed += size
    return {
        "cases": total,
        "correct": final_correct,
        "exact_accuracy": final_correct / total,
        "all_steps_correct": chain_correct,
        "all_steps_exact_accuracy": chain_correct / total,
        "any_invalid_residue_rate": any_invalid / total,
        "operand_bits": operand_bits,
        "first_error_by_step": first_error_by_step,
    }


def randomized_copy(model: HornerTCN, seed: int) -> HornerTCN:
    """Return a deterministically reinitialized control with identical shape."""
    control = copy.deepcopy(model)
    torch.manual_seed(seed)
    for module in control.modules():
        if isinstance(module, (nn.Conv1d, nn.GroupNorm)):
            module.reset_parameters()
    control.eval()
    return control


def zero_baseline_rollout(
    primes: list[int],
    config: POCConfig,
    operand_bits: int,
    seed: int,
    cases: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prime_tensor = torch.tensor(primes, dtype=torch.int64)
    modulus = prime_tensor[torch.randint(len(primes), (cases,), generator=generator)]
    multiplicand = (torch.rand(cases, generator=generator) * modulus.float()).long()
    multiplier = torch.randint(0, 1 << operand_bits, (cases,), generator=generator)
    target = (multiplier * multiplicand) % modulus
    correct = int(target.eq(0).sum())
    return {
        "cases": cases,
        "correct": correct,
        "exact_accuracy": correct / cases,
        "operand_bits": operand_bits,
    }
