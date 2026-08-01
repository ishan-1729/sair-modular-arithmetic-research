"""Width-independent chunk-shared recurrent modular transition model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ChunkSharedBiGRU(nn.Module):
    """Encode local chunks, refine shared global context, and decode locally."""

    def __init__(self, width: int, chunk_size: int = 4, hidden: int = 64, cycles: int = 3) -> None:
        super().__init__()
        if width % chunk_size:
            raise ValueError("width must be divisible by chunk_size")
        self.width = width
        self.chunk_size = chunk_size
        self.hidden = hidden
        self.cycles = cycles
        self.encoder = nn.Sequential(nn.Linear(6 * chunk_size, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.recurrent = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.merge = nn.Linear(2 * hidden, hidden)
        self.refine_norm = nn.LayerNorm(hidden)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, chunk_size))
        self.quotient_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, width = features.shape
        if channels != 6 or width % self.chunk_size:
            raise ValueError(f"expected six channels and width divisible by {self.chunk_size}")
        chunks = features.reshape(batch, channels, width // self.chunk_size, self.chunk_size).permute(0, 2, 1, 3).reshape(batch, width // self.chunk_size, channels * self.chunk_size)
        hidden = self.encoder(chunks)
        for _ in range(self.cycles):
            recurrent, _ = self.recurrent(hidden)
            hidden = self.refine_norm(hidden + self.merge(recurrent))
        logits = self.decoder(hidden).reshape(batch, width)
        pooled = torch.cat((hidden.mean(dim=1), hidden.amax(dim=1)), dim=1)
        return logits, self.quotient_head(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

