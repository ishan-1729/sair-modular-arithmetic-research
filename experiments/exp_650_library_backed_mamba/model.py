"""Library-backed Mamba-2 hybrid and matched transition controls for EXP-650."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Mamba2Config
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer


Direction = Literal["forward", "reverse", "bidirectional"]


class PostNormResidual(nn.Module):
    """Paper-specified residual followed by RMSNorm (norm&add/postnorm)."""

    def __init__(self, hidden: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps)

    def forward(self, residual: Tensor, update: Tensor) -> Tensor:
        return self.norm(residual + update)


class TokenMLP(nn.Module):
    def __init__(self, hidden: int, expansion: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden, hidden * expansion),
            nn.GELU(),
            nn.Linear(hidden * expansion, hidden),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.layers(hidden)


class TransposedSequenceMLP(nn.Module):
    """MLP-t alternative: dense all-position mixing over the transposed sequence."""

    def __init__(self, width: int, expansion: int = 2) -> None:
        super().__init__()
        self.width = int(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width * expansion),
            nn.GELU(),
            nn.Linear(width * expansion, width),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.shape[1] != self.width:
            raise ValueError(f"MLP-t is frozen for width {self.width}, got {hidden.shape[1]}")
        return self.layers(hidden.transpose(1, 2)).transpose(1, 2)


def mamba2_config(
    hidden: int,
    state_size: int,
    conv_kernel: int,
    expand: int,
    head_dim: int,
    n_groups: int,
) -> Mamba2Config:
    intermediate = hidden * expand
    if intermediate % head_dim:
        raise ValueError("hidden * expand must be divisible by head_dim")
    heads = intermediate // head_dim
    if heads % n_groups:
        raise ValueError("Mamba-2 heads must be divisible by n_groups")
    return Mamba2Config(
        hidden_size=hidden,
        state_size=state_size,
        num_hidden_layers=2,
        expand=expand,
        head_dim=head_dim,
        num_heads=heads,
        n_groups=n_groups,
        conv_kernel=conv_kernel,
        vocab_size=8,
        use_cache=False,
        chunk_size=256,
        rms_norm=True,
    )


class LibraryMamba2Hybrid(nn.Module):
    """Mamba2 -> Mamba2 -> Attention -> MLP with postnorm residuals.

    The Mamba operators are the maintained Transformers implementation.  Exact
    arithmetic is absent from this learned forward path.
    """

    input_features = 6

    def __init__(
        self,
        width: int,
        hidden: int = 32,
        state_size: int = 8,
        conv_kernel: int = 4,
        expand: int = 2,
        head_dim: int = 8,
        n_groups: int = 1,
        attention_heads: int = 4,
        mlp_expansion: int = 2,
        cross_mixer: Literal["attention", "mlp_t"] = "attention",
    ) -> None:
        super().__init__()
        if hidden % attention_heads:
            raise ValueError("hidden must be divisible by attention_heads")
        self.width = int(width)
        self.hidden = int(hidden)
        self.cross_mixer = cross_mixer
        self.input_projection = nn.Linear(self.input_features, hidden)
        config = mamba2_config(
            hidden, state_size, conv_kernel, expand, head_dim, n_groups
        )
        self.mamba_blocks = nn.ModuleList(
            [Mamba2Mixer(config, layer_idx=index) for index in range(2)]
        )
        self.mamba_postnorm = nn.ModuleList(
            [PostNormResidual(hidden) for _ in range(2)]
        )
        if cross_mixer == "attention":
            self.cross = nn.MultiheadAttention(
                hidden, attention_heads, dropout=0.0, batch_first=True
            )
        elif cross_mixer == "mlp_t":
            self.cross = TransposedSequenceMLP(width)
        else:
            raise ValueError(f"unknown cross mixer: {cross_mixer}")
        self.cross_postnorm = PostNormResidual(hidden)
        self.mlp = TokenMLP(hidden, mlp_expansion)
        self.mlp_postnorm = PostNormResidual(hidden)
        self.output = nn.Linear(hidden, 1)

    def _mamba_flow(self, hidden: Tensor, reverse: bool) -> Tensor:
        if reverse:
            hidden = hidden.flip(1)
        for block, postnorm in zip(
            self.mamba_blocks, self.mamba_postnorm, strict=True
        ):
            hidden = postnorm(hidden, block(hidden, cache_params=None))
        return hidden.flip(1) if reverse else hidden

    def forward(self, features: Tensor, direction: Direction = "forward") -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_features:
            raise ValueError("features must have shape [batch, width, 6]")
        if self.cross_mixer == "mlp_t" and features.shape[1] != self.width:
            raise ValueError("MLP-t cannot transfer to a different width")
        embedded = self.input_projection(features)
        if direction == "forward":
            hidden = self._mamba_flow(embedded, reverse=False)
        elif direction == "reverse":
            hidden = self._mamba_flow(embedded, reverse=True)
        elif direction == "bidirectional":
            hidden = 0.5 * (
                self._mamba_flow(embedded, reverse=False)
                + self._mamba_flow(embedded, reverse=True)
            )
        else:
            raise ValueError(f"unknown direction: {direction}")
        if self.cross_mixer == "attention":
            mixed, _ = self.cross(hidden, hidden, hidden, need_weights=False)
        else:
            mixed = self.cross(hidden)
        hidden = self.cross_postnorm(hidden, mixed)
        hidden = self.mlp_postnorm(hidden, self.mlp(hidden))
        return self.output(hidden).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MatchedConvGRU(nn.Module):
    """Width-shared dilated ConvGRU control on the same legal feature planes."""

    input_features = 6

    def __init__(self, width: int, hidden: int = 30, cycles: int = 2) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden = int(hidden)
        self.cycles = int(cycles)
        channels = hidden + self.input_features
        self.stem = nn.Conv1d(self.input_features, hidden, 1)
        self.state_norm = nn.GroupNorm(1, hidden)
        self.gate_weight = nn.Parameter(torch.empty(2 * hidden, channels, 3))
        self.gate_bias = nn.Parameter(torch.zeros(2 * hidden))
        self.candidate_weight = nn.Parameter(torch.empty(hidden, channels, 3))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden))
        self.output_norm = nn.GroupNorm(1, hidden)
        self.output = nn.Conv1d(hidden, 1, 1)
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate_weight, a=math.sqrt(5))

    def dilation_schedule(self, width: int) -> list[int]:
        maximum = 1 << max(0, math.ceil(math.log2(width)) - 1)
        base = []
        dilation = 1
        while dilation <= maximum:
            base.append(dilation)
            dilation *= 2
        result = []
        for cycle in range(self.cycles):
            result.extend(base if cycle % 2 == 0 else reversed(base))
        return result

    def recurrent_update(self, state: Tensor, inputs: Tensor, dilation: int) -> Tensor:
        normalized = self.state_norm(state)
        combined = torch.cat((normalized, inputs), dim=1)
        reset, update = torch.sigmoid(
            F.conv1d(
                combined,
                self.gate_weight,
                self.gate_bias,
                padding=dilation,
                dilation=dilation,
            )
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            F.conv1d(
                torch.cat((reset * normalized, inputs), dim=1),
                self.candidate_weight,
                self.candidate_bias,
                padding=dilation,
                dilation=dilation,
            )
        )
        return (1.0 - update) * state + update * candidate

    def forward(self, features: Tensor) -> Tensor:
        inputs = features.transpose(1, 2)
        hidden = self.stem(inputs)
        for dilation in self.dilation_schedule(features.shape[1]):
            hidden = self.recurrent_update(hidden, inputs, dilation)
        return self.output(F.silu(self.output_norm(hidden))).squeeze(1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MatchedRecurrentAttention(nn.Module):
    """Recurrently shared global-attention control without a state-space mixer."""

    input_features = 6

    def __init__(
        self,
        width: int,
        hidden: int = 32,
        heads: int = 4,
        feedforward: int = 64,
        iterations: int = 3,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.iterations = int(iterations)
        self.input_projection = nn.Linear(self.input_features, hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, batch_first=True
        )
        self.attention_postnorm = PostNormResidual(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, feedforward), nn.GELU(), nn.Linear(feedforward, hidden)
        )
        self.mlp_postnorm = PostNormResidual(hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.input_projection(features)
        for _ in range(self.iterations):
            attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
            hidden = self.attention_postnorm(hidden, attended)
            hidden = self.mlp_postnorm(hidden, self.mlp(hidden))
        return self.output(hidden).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(arm: str, width: int, config: dict) -> nn.Module:
    if arm == "mamba2_hybrid":
        return LibraryMamba2Hybrid(width=width, **config["mamba2_hybrid"])
    if arm == "convgru_control":
        return MatchedConvGRU(width=width, **config["convgru_control"])
    if arm == "attention_control":
        return MatchedRecurrentAttention(width=width, **config["attention_control"])
    raise ValueError(f"unknown arm: {arm}")
