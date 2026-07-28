"""Training-only lexicographic residue-validity loss for EXP685."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor

from experiments.exp_665_attractor_equilibrium_family.model import AttractorCell


def build_model(config: dict[str, Any], checkpoint, *, randomized_seed: int | None = None) -> AttractorCell:
    """Construct the exact EXP665 inference model and load its full checkpoint."""
    model = AttractorCell(**config["model"])
    model.load_state_dict(load_file(checkpoint), strict=True)
    if randomized_seed is not None:
        model.randomized_correction(int(randomized_seed))
    return model


def lexicographic_invalid_probability(logits: Tensor, modulus_bits_lsb: Tensor) -> Tensor:
    """Return P(Bernoulli(logits) interpreted as binary >= fixed modulus).

    The fixed modulus bits are an offline training/evaluation input.  The
    log-domain MSB-first recurrence sums mutually exclusive first-greater
    events plus exact equality and remains differentiable in every logit.
    """
    if logits.ndim != 2 or modulus_bits_lsb.shape != logits.shape:
        raise ValueError("logits and modulus bits must have identical [batch,width] shape")
    if not torch.all((modulus_bits_lsb == 0) | (modulus_bits_lsb == 1)):
        raise ValueError("modulus bits must be binary")
    work = logits.to(torch.float64)
    modulus = modulus_bits_lsb.to(device=logits.device, dtype=torch.bool)
    log_one = F.logsigmoid(work)
    log_zero = F.logsigmoid(-work)
    prefix_equal = torch.zeros(work.shape[0], device=work.device, dtype=work.dtype)
    greater_terms: list[Tensor] = []
    for index in range(work.shape[1] - 1, -1, -1):
        bit_is_one = modulus[:, index]
        greater_terms.append(torch.where(bit_is_one, torch.full_like(prefix_equal, -torch.inf), prefix_equal + log_one[:, index]))
        prefix_equal = prefix_equal + torch.where(bit_is_one, log_one[:, index], log_zero[:, index])
    return torch.exp(torch.logsumexp(torch.stack([*greater_terms, prefix_equal], dim=0), dim=0))


def deterministic_modulus_permutation(batch: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(batch, generator=generator)
    if batch > 1 and torch.equal(permutation, torch.arange(batch)):
        permutation = permutation.roll(1)
    return permutation.to(device=device)


def supervised_objective(
    output,
    target_bits: Tensor,
    quotient: Tensor,
    modulus_bits: Tensor,
    config: dict[str, Any],
    *,
    kind: str,
    permutation_seed: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Matched EXP665 objective with one training-only validity intervention."""
    per_bit = F.binary_cross_entropy_with_logits(output.logits, target_bits, reduction="none")
    ce = per_bit.mean()
    ce = ce + float(config["training"]["worst_bit_weight"]) * per_bit.mean(0).max()
    ce = ce + float(config["training"]["quotient_weight"]) * F.cross_entropy(output.quotient_logits, quotient)
    ce = ce + float(config["training"]["residual_weight"]) * output.relative_residual.mean()
    selected_modulus = modulus_bits
    if kind == "shuffled_modulus":
        selected_modulus = modulus_bits[deterministic_modulus_permutation(len(modulus_bits), permutation_seed, modulus_bits.device)]
    invalid_probability = lexicographic_invalid_probability(output.logits, selected_modulus)
    validity = invalid_probability.mean()
    weight = 0.0 if kind == "ce_only" else float(config["training"]["validity_weight"])
    total = ce + weight * validity
    return total, {
        "ce_loss": ce,
        "validity_loss": validity,
        "invalid_probability_mean": invalid_probability.mean(),
        "invalid_probability_max": invalid_probability.max(),
        "weight": torch.tensor(weight, device=output.logits.device, dtype=output.logits.dtype),
    }
