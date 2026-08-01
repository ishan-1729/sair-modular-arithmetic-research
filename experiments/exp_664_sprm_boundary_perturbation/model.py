"""Training-only latent TBPTT perturbation with exact source inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU


ROUTE_MODES = {"sprm_boundary", "clean_tbptt", "wrong_boundary", "unscaled_boundary"}


def linear_beta(tau: Tensor, *, start: float, end: float, steps: int) -> Tensor:
    """Linearly interpolate the DDPM beta schedule for integer tau in [1, steps]."""
    if steps < 2 or torch.any(tau < 1) or torch.any(tau > steps):
        raise ValueError("tau must be in [1, steps] and steps must exceed one")
    return start + (end - start) * (tau.to(torch.float32) - 1.0) / float(steps - 1)


def state_scaled_perturbation(
    state: Tensor,
    beta: Tensor,
    *,
    generator: torch.Generator,
    state_scaled: bool,
) -> tuple[Tensor, dict[str, float]]:
    """Equation 9 using per-example hidden-state moments."""
    if state.ndim != 3 or beta.shape != (state.shape[0], 1, 1):
        raise ValueError("state must be [batch,channels,width] and beta [batch,1,1]")
    detached = state.detach()
    if state_scaled:
        mean = detached.mean(dim=(1, 2), keepdim=True)
        variance = detached.var(dim=(1, 2), unbiased=False, keepdim=True)
        epsilon = torch.randn(detached.shape, generator=generator, device=detached.device, dtype=detached.dtype)
        epsilon = mean + variance.clamp_min(1e-12).sqrt() * epsilon
    else:
        epsilon = torch.randn(detached.shape, generator=generator, device=detached.device, dtype=detached.dtype)
    perturbed = (1.0 - beta).sqrt() * detached + beta.sqrt() * epsilon
    stats = {
        "pre_mean": float(detached.mean()),
        "pre_variance": float(detached.var(unbiased=False)),
        "beta_mean": float(beta.mean()),
        "epsilon_mean": float(epsilon.mean()),
        "epsilon_variance": float(epsilon.var(unbiased=False)),
        "post_mean": float(perturbed.mean()),
        "post_variance": float(perturbed.var(unbiased=False)),
        "noise_rms": float((perturbed - detached).square().mean().sqrt()),
    }
    return perturbed, stats


@dataclass
class TrainingOutput:
    logits: Tensor
    quotient_logits: Tensor
    window_logits: list[Tensor]
    window_quotient_logits: list[Tensor]
    boundary_events: list[dict[str, float | int | str]]
    block_passes: int
    analytical_flops: int


class SPRMCell(nn.Module):
    """EXP-609 cell whose public forward is exactly the frozen source call graph."""

    def __init__(
        self,
        *,
        hidden: int = 92,
        cycles: int = 2,
        boundary_passes: int = 4,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.base = SharedDilatedConvGRU(width=256, hidden=hidden, cycles=cycles)
        self.boundary_passes = boundary_passes
        self.beta_start, self.beta_end, self.beta_steps = beta_start, beta_end, beta_steps

    @property
    def width(self) -> int:
        return self.base.width

    @width.setter
    def width(self, value: int) -> None:
        self.base.width = value

    def parameter_count(self) -> int:
        return self.base.parameter_count()

    def load_source(self, state: dict[str, Tensor]) -> None:
        self.base.load_state_dict(state, strict=True)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Noise-free inference: direct delegation, with no extra call or RNG use."""
        self.base.width = features.shape[-1]
        return self.base(features)

    def _readout(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)

    def _flops(self, batch: int, width: int, passes: int) -> int:
        hidden, channels = self.base.hidden, self.base.hidden + 6
        per_pass = width * (2 * hidden * channels * 3 + hidden * channels * 3)
        return int(batch * passes * per_pass)

    def training_forward(self, features: Tensor, *, route: str, generator: torch.Generator) -> TrainingOutput:
        """Run the exact dilation schedule once, detaching at frozen TBPTT boundaries."""
        if route not in ROUTE_MODES:
            raise ValueError(f"unknown training route: {route}")
        width, batch = features.shape[-1], features.shape[0]
        self.base.width = width
        dilations = self.base.dilation_schedule(width)
        hidden = self.base.stem(features)
        events: list[dict[str, float | int | str]] = []
        window_logits: list[Tensor] = []
        window_quotient: list[Tensor] = []
        canonical = {index for index in range(self.boundary_passes, len(dilations), self.boundary_passes)}
        wrong = {index + 1 for index in canonical if index + 1 < len(dilations)}
        for index, dilation in enumerate(dilations, start=1):
            hidden = self.base.recurrent_update(hidden, features, dilation)
            if index in canonical:
                local_logits, local_quotient = self._readout(hidden)
                window_logits.append(local_logits)
                window_quotient.append(local_quotient)
                hidden = hidden.detach()
                if route in {"sprm_boundary", "unscaled_boundary"}:
                    tau = torch.randint(1, self.beta_steps + 1, (batch,), generator=generator, device=hidden.device)
                    beta = linear_beta(tau, start=self.beta_start, end=self.beta_end, steps=self.beta_steps)[:, None, None]
                    hidden, stats = state_scaled_perturbation(hidden, beta, generator=generator, state_scaled=route == "sprm_boundary")
                    events.append({"pass": index, "kind": "canonical", **stats})
                else:
                    observed = hidden.detach()
                    events.append({"pass": index, "kind": "detach", "pre_mean": float(observed.mean()), "pre_variance": float(observed.var(unbiased=False)), "beta_mean": 0.0, "epsilon_mean": 0.0, "epsilon_variance": 0.0, "post_mean": float(observed.mean()), "post_variance": float(observed.var(unbiased=False)), "noise_rms": 0.0})
            if route == "wrong_boundary" and index in wrong:
                tau = torch.randint(1, self.beta_steps + 1, (batch,), generator=generator, device=hidden.device)
                beta = linear_beta(tau, start=self.beta_start, end=self.beta_end, steps=self.beta_steps)[:, None, None]
                perturbed, stats = state_scaled_perturbation(hidden, beta, generator=generator, state_scaled=True)
                hidden = hidden + (perturbed - hidden.detach())
                events.append({"pass": index, "kind": "wrong", **stats})
        logits, quotient = self._readout(hidden)
        window_logits.append(logits)
        window_quotient.append(quotient)
        return TrainingOutput(logits, quotient, window_logits, window_quotient, events, len(dilations), self._flops(batch, width, len(dilations)))
