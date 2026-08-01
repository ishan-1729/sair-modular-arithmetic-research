"""Function-preserving EXP609 cellular extension for discrete DRM canvases."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU


MASK_TOKEN = 2


def cosine_mask_ratio(tau: Tensor) -> Tensor:
    """Equation 8's masked fraction: 1 - cos²(pi*tau/2)."""
    return 1.0 - torch.cos(math.pi * tau / 2.0).square()


def schedule_mask_ratio(tau: Tensor, schedule: str) -> Tensor:
    if schedule == "cosine":
        return cosine_mask_ratio(tau)
    if schedule == "linear":
        return tau
    if schedule == "all_mask":
        return torch.ones_like(tau)
    raise ValueError(f"unknown mask schedule: {schedule}")


def corrupt_target(
    target_bits: Tensor,
    tau: Tensor,
    *,
    schedule: str,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Uniform salt-and-pepper target masking used only by offline training."""
    if target_bits.ndim != 2 or tau.shape != (target_bits.shape[0],):
        raise ValueError("target_bits must be [batch,width] and tau [batch]")
    batch, width = target_bits.shape
    counts = torch.floor(schedule_mask_ratio(tau, schedule) * width).to(torch.long)
    canvas = target_bits.to(torch.long).clone()
    masked = torch.zeros_like(target_bits, dtype=torch.bool)
    random_values = torch.rand((batch, width), generator=generator, device=target_bits.device)
    order = random_values.argsort(dim=1)
    positions = torch.arange(width, device=target_bits.device)[None]
    selected = positions < counts[:, None]
    masked.scatter_(1, order, selected)
    canvas[masked] = MASK_TOKEN
    return canvas, masked


@dataclass
class DRMOutput:
    logits: Tensor
    quotient_logits: Tensor
    canvases: list[Tensor]
    block_passes: int
    analytical_flops: int


class DRMCell(nn.Module):
    """Parameter-superset cell shared by DRM and every matched control."""

    def __init__(self, *, hidden: int = 92, cycles: int = 2, canvas_embedding: int = 16) -> None:
        super().__init__()
        self.base = SharedDilatedConvGRU(width=256, hidden=hidden, cycles=cycles)
        self.canvas_embedding = nn.Embedding(3, canvas_embedding)
        self.canvas_projection = nn.Conv1d(canvas_embedding, hidden, 1)
        nn.init.normal_(self.canvas_embedding.weight, std=0.02)
        nn.init.zeros_(self.canvas_projection.weight)
        nn.init.zeros_(self.canvas_projection.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def load_source(self, state: dict[str, Tensor]) -> None:
        self.base.load_state_dict(state, strict=True)

    def source_forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        self.base.width = features.shape[-1]
        return self.base(features)

    def _readout(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)

    def _phi(self, hidden: Tensor, features: Tensor) -> Tensor:
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            hidden = self.base.recurrent_update(hidden, features, dilation)
        return hidden

    def _flops(self, batch: int, width: int, block_passes: int) -> int:
        dilations = len(self.base.dilation_schedule(width))
        channels, hidden = self.base.hidden + 6, self.base.hidden
        recurrent = width * dilations * 3 * (2 * hidden * channels + hidden * channels)
        return int(batch * block_passes * recurrent)

    def denoise_window(
        self,
        features: Tensor,
        canvas: Tensor,
        *,
        warmup_steps: int,
        gradient_steps: int,
    ) -> DRMOutput:
        if features.ndim != 3 or features.shape[1] != 6:
            raise ValueError("features must have shape [batch,6,width]")
        if canvas.shape != (features.shape[0], features.shape[-1]):
            raise ValueError("canvas must have shape [batch,width]")
        if canvas.min() < 0 or canvas.max() > MASK_TOKEN:
            raise ValueError("canvas tokens must be 0, 1, or MASK")
        hidden = self.base.stem(features)
        canvas_signal = self.canvas_projection(self.canvas_embedding(canvas).transpose(1, 2))
        if warmup_steps:
            with torch.no_grad():
                for _ in range(warmup_steps):
                    hidden = self._phi(hidden + canvas_signal, features)
            hidden = hidden.detach()
        for step in range(gradient_steps):
            hidden = self._phi(hidden + (canvas_signal if step == 0 else 0.0), features)
        logits, quotient = self._readout(hidden)
        passes = warmup_steps + gradient_steps
        return DRMOutput(logits, quotient, [], passes, self._flops(features.shape[0], features.shape[-1], passes))

    @torch.no_grad()
    def generate_and_remask(
        self,
        features: Tensor,
        *,
        warmup_steps: int,
        gradient_steps: int,
        inference_steps: int,
        schedule: str = "cosine",
        remask: bool = True,
        seed: int,
    ) -> DRMOutput:
        batch, width = features.shape[0], features.shape[-1]
        canvas = torch.full((batch, width), MASK_TOKEN, dtype=torch.long, device=features.device)
        generator = torch.Generator(device=features.device).manual_seed(int(seed))
        if inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        interior = torch.rand(max(0, inference_steps - 1), generator=generator, device=features.device)
        times = torch.cat((torch.ones(1, device=features.device), interior.sort(descending=True).values, torch.zeros(1, device=features.device)))
        canvases = [canvas.clone()]
        total_passes = total_flops = 0
        final: DRMOutput | None = None
        for index in range(inference_steps):
            final = self.denoise_window(features, canvas, warmup_steps=warmup_steps, gradient_steps=gradient_steps)
            predicted = final.logits.ge(0).to(torch.long)
            total_passes += final.block_passes
            total_flops += final.analytical_flops
            next_tau = times[index + 1]
            count = int(torch.floor(schedule_mask_ratio(next_tau, schedule) * width).item()) if remask else 0
            if count:
                random_values = torch.rand((batch, width), generator=generator, device=features.device)
                selected = random_values.argsort(dim=1)[:, :count]
                canvas = predicted.clone()
                canvas.scatter_(1, selected, MASK_TOKEN)
            else:
                canvas = predicted
            canvases.append(canvas.clone())
        if final is None:
            raise AssertionError("unreachable")
        logits = torch.where(canvas.bool(), torch.ones_like(final.logits), -torch.ones_like(final.logits))
        return DRMOutput(logits, final.quotient_logits, canvases, total_passes, total_flops)
