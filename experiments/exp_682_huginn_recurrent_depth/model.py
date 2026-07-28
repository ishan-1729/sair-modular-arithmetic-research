"""Executable Huginn topology probe around Candidate005.

This module deliberately is not an inference candidate.  It makes the source
components and the exact-transfer obstruction executable before any root is
opened.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal, Poisson


@dataclass
class HuginnTrace:
    depth: int
    prefix_without_credit: int
    suffix_with_credit: int
    latent_mode: str
    reinjection: bool
    concat: bool
    rmsnorm: bool
    core_calls: int


def sample_bounded_lognormal_poisson_depths(
    count: int, *, targeted_mean: int = 5, sigma: float = 0.5,
    minimum: int = 2, maximum: int = 12, seed: int = 0,
) -> Tensor:
    """Eq. 1-2 with a preregistered safety bound and replayable RNG."""
    if count < 1 or targeted_mean <= 1 or not 0 < sigma <= 1 or minimum < 1 or maximum < minimum:
        raise ValueError("invalid depth distribution")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        bar_r = torch.tensor(float(targeted_mean - 1))
        tau = Normal(torch.log(bar_r) - 0.5 * sigma**2, sigma).sample((count,))
        depths = Poisson(tau.exp()).sample().to(torch.long).add_(1)
    return depths.clamp_(minimum, maximum)


class HuginnTopologyProbe(nn.Module):
    """Source-complete topology probe using Candidate005's learned boundaries."""

    def __init__(self, source: nn.Module, *, latent_std: float = 0.02, release_seed: int = 68205171) -> None:
        super().__init__()
        self.hidden = int(source.hidden)
        self.cycles = int(source.cycles)
        self.latent_std = float(latent_std)
        self.release_seed = int(release_seed)
        self.prelude = copy.deepcopy(source.stem)
        self.source_state_norm = copy.deepcopy(source.state_norm)
        self.gate_weight = nn.Parameter(source.gate_weight.detach().clone())
        self.gate_bias = nn.Parameter(source.gate_bias.detach().clone())
        self.candidate_weight = nn.Parameter(source.candidate_weight.detach().clone())
        self.candidate_bias = nn.Parameter(source.candidate_bias.detach().clone())
        self.concat_adapter = nn.Conv1d(2 * self.hidden, self.hidden, 1)
        self.input_projection = nn.Conv1d(6, self.hidden, 1)
        self.pre_norm = nn.RMSNorm(self.hidden)
        self.post_norm = nn.RMSNorm(self.hidden)
        self.coda_norm = copy.deepcopy(source.output_norm)
        self.coda_local = copy.deepcopy(source.local_head)
        self.coda_quotient = copy.deepcopy(source.quotient_head)
        with torch.no_grad():
            self.concat_adapter.weight.zero_(); self.concat_adapter.bias.zero_()
            self.concat_adapter.weight[:, :self.hidden, 0].copy_(0.5 * torch.eye(self.hidden))
            self.concat_adapter.weight[:, self.hidden:, 0].copy_(torch.eye(self.hidden))
            self.input_projection.weight.copy_(source.stem.weight); self.input_projection.bias.copy_(source.stem.bias)

    def dilation_schedule(self, width: int) -> list[int]:
        maximum = max(1, 1 << max(0, math.ceil(math.log2(width)) - 1))
        ascending = []
        dilation = 1
        while dilation <= maximum:
            ascending.append(dilation); dilation *= 2
        schedule: list[int] = []
        for cycle in range(self.cycles):
            schedule.extend(ascending if cycle % 2 == 0 else reversed(ascending))
        return schedule

    def latent(self, features: Tensor, mode: str, *, seed: int | None = None) -> Tensor:
        shape = (features.shape[0], self.hidden, features.shape[-1])
        if mode == "zero":
            return features.new_zeros(shape)
        if mode not in {"random", "deterministic"}:
            raise ValueError("latent mode must be random, deterministic, or zero")
        chosen = self.release_seed if mode == "deterministic" else seed
        if chosen is None:
            return torch.randn(shape, dtype=features.dtype, device=features.device) * self.latent_std
        generator = torch.Generator(device=features.device).manual_seed(int(chosen))
        return torch.randn(shape, generator=generator, dtype=features.dtype, device=features.device) * self.latent_std

    def _candidate_update(self, state: Tensor, inputs: Tensor, dilation: int) -> Tensor:
        normalized = self.source_state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        gates = torch.sigmoid(F.conv1d(combined, self.gate_weight, self.gate_bias, padding=dilation, dilation=dilation))
        reset, update = gates.chunk(2, dim=1)
        candidate_inputs = torch.cat((reset * normalized, inputs), dim=1)
        candidate = torch.tanh(F.conv1d(candidate_inputs, self.candidate_weight, self.candidate_bias, padding=dilation, dilation=dilation))
        return (1.0 - update) * state + update * candidate

    @staticmethod
    def _rms(norm: nn.RMSNorm, value: Tensor) -> Tensor:
        return norm(value.transpose(1, 2)).transpose(1, 2)

    def recurrent_block(self, state: Tensor, embedded: Tensor, features: Tensor, *, reinjection: bool, concat: bool, rmsnorm: bool) -> Tensor:
        injected = embedded if reinjection else torch.zeros_like(embedded)
        if concat:
            state = self.concat_adapter(torch.cat((state, injected), dim=1))
        else:
            state = state + injected
        if rmsnorm:
            state = self._rms(self.pre_norm, state)
        raw = features if reinjection else torch.zeros_like(features)
        for dilation in self.dilation_schedule(features.shape[-1]):
            state = self._candidate_update(state, raw, dilation)
        return self._rms(self.post_norm, state) if rmsnorm else state

    def decode(self, state: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.silu(self.coda_norm(state))
        logits = self.coda_local(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), dim=1)
        return logits, self.coda_quotient(pooled)

    def forward(
        self, features: Tensor, *, depth: int = 5, latent_mode: str = "deterministic",
        latent_seed: int | None = None, reinjection: bool = True, concat: bool = True,
        rmsnorm: bool = True, tbptt_last: int = 8, full_bptt: bool = False,
        reset: bool = False, shuffled: bool = False, return_trace: bool = False,
    ):
        if features.ndim != 3 or features.shape[1] != 6 or not torch.isfinite(features).all():
            raise ValueError("features must be finite [batch, 6, width]")
        if not 2 <= int(depth) <= 12 or tbptt_last < 1:
            raise ValueError("depth outside frozen bounds")
        embedded = self.prelude(features)
        initial = self.latent(features, latent_mode, seed=latent_seed)
        state = initial
        prefix = 0 if full_bptt else max(0, int(depth) - int(tbptt_last))
        for step in range(int(depth)):
            if step == prefix and prefix:
                state = state.detach()
            if reset and step:
                state = initial
            if shuffled and step:
                state = state.roll(1, dims=0)
            state = self.recurrent_block(state, embedded, features, reinjection=reinjection, concat=concat, rmsnorm=rmsnorm)
        outputs = self.decode(state)
        trace = HuginnTrace(int(depth), prefix, int(depth) - prefix, latent_mode, reinjection, concat, rmsnorm, int(depth) * len(self.dilation_schedule(features.shape[-1])))
        return (*outputs, trace) if return_trace else outputs


def structural_transfer_receipt(source: nn.Module, probe: HuginnTopologyProbe, features: Tensor) -> dict[str, object]:
    """Prove why a causal source-complete transfer is not function preserving."""
    source.eval(); probe.eval()
    with torch.no_grad():
        original = source(features)
        deterministic = probe(features, depth=2, latent_mode="deterministic")
        alternate = probe(features, depth=2, latent_mode="random", latent_seed=probe.release_seed + 1)
        no_norm = probe(features, depth=2, latent_mode="deterministic", rmsnorm=False)
    def delta(left, right):
        return max(float((left[0] - right[0]).abs().max()), float((left[1] - right[1]).abs().max()))
    return {
        "source_norm_type": type(source.state_norm).__name__,
        "required_norm_type": type(probe.pre_norm).__name__,
        "source_has_concat_adapter": hasattr(source, "concat_adapter"),
        "source_has_random_latent": hasattr(source, "latent"),
        "source_vs_required_max_delta": delta(original, deterministic),
        "latent_seed_intervention_max_delta": delta(deterministic, alternate),
        "rmsnorm_intervention_max_delta": delta(deterministic, no_norm),
        "exact_checkpoint_transfer": torch.equal(original[0], deterministic[0]) and torch.equal(original[1], deterministic[1]),
        "causal_required_components": delta(deterministic, alternate) > 0 and delta(deterministic, no_norm) > 0,
    }


__all__ = ["HuginnTopologyProbe", "HuginnTrace", "sample_bounded_lognormal_poisson_depths", "structural_transfer_receipt"]
