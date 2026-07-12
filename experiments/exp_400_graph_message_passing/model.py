"""Width-shared graph models over legal bit-position nodes and local relations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def dilation_schedule(width: int, cycles: int) -> list[int]:
    maximum = max(1, 1 << max(0, math.ceil(math.log2(width)) - 1))
    values = []
    dilation = 1
    while dilation <= maximum:
        values.append(dilation)
        dilation *= 2
    schedule = []
    for cycle in range(cycles):
        schedule.extend(values if cycle % 2 == 0 else reversed(values))
    return schedule


def shifted(nodes: Tensor, offset: int) -> tuple[Tensor, Tensor]:
    batch, width, hidden = nodes.shape
    result = torch.zeros_like(nodes)
    valid = torch.zeros((batch, width, 1), dtype=torch.bool, device=nodes.device)
    if offset > 0:
        result[:, offset:] = nodes[:, :-offset]
        valid[:, offset:] = True
    elif offset < 0:
        amount = -offset
        result[:, :-amount] = nodes[:, amount:]
        valid[:, :-amount] = True
    else:
        result = nodes
        valid[:] = True
    return result, valid


class GraphBase(nn.Module):
    def __init__(self, width: int, hidden: int, cycles: int) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.cycles = cycles
        self.stem = nn.Linear(6, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.local_head = nn.Linear(hidden, 1)
        self.quotient_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def finish(self, nodes: Tensor) -> tuple[Tensor, Tensor]:
        nodes = torch.nn.functional.silu(self.norm(nodes))
        logits = self.local_head(nodes).squeeze(-1)
        pooled = torch.cat((nodes.mean(dim=1), nodes.amax(dim=1)), dim=1)
        return logits, self.quotient_head(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class RecurrentGCN(GraphBase):
    """Fixed mean aggregation over self/left/right relations with a shared GRU update."""

    def __init__(self, width: int, hidden: int = 92, cycles: int = 2) -> None:
        super().__init__(width, hidden, cycles)
        self.self_linear = nn.Linear(hidden, hidden)
        self.neighbor_linear = nn.Linear(hidden, hidden)
        self.update = nn.GRUCell(hidden, hidden)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.stem(features.transpose(1, 2))
        for dilation in dilation_schedule(nodes.shape[1], self.cycles):
            left, left_valid = shifted(nodes, dilation)
            right, right_valid = shifted(nodes, -dilation)
            count = 1.0 + left_valid.float() + right_valid.float()
            aggregate = (nodes + left + right) / count
            message = torch.tanh(self.self_linear(nodes) + self.neighbor_linear(aggregate))
            nodes = self.update(message.flatten(0, 1), nodes.flatten(0, 1)).view_as(nodes)
        return self.finish(nodes)


class RecurrentGAT(GraphBase):
    """Masked learned attention over self/left/right relations with a shared GRU update."""

    def __init__(self, width: int, hidden: int = 92, cycles: int = 2) -> None:
        super().__init__(width, hidden, cycles)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.direction_bias = nn.Parameter(torch.zeros(3))
        self.update = nn.GRUCell(hidden, hidden)
        self.scale = hidden ** -0.5

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.stem(features.transpose(1, 2))
        for dilation in dilation_schedule(nodes.shape[1], self.cycles):
            left, left_valid = shifted(nodes, dilation)
            right, right_valid = shifted(nodes, -dilation)
            neighbors = torch.stack((left, nodes, right), dim=2)
            valid = torch.cat((left_valid, torch.ones_like(left_valid), right_valid), dim=2)
            query = self.query(nodes).unsqueeze(2)
            scores = (query * self.key(neighbors)).sum(-1) * self.scale + self.direction_bias
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=2)
            message = (weights.unsqueeze(-1) * self.value(neighbors)).sum(2)
            nodes = self.update(message.flatten(0, 1), nodes.flatten(0, 1)).view_as(nodes)
        return self.finish(nodes)


class RecurrentTypedMPNN(GraphBase):
    """Typed learned messages with max aggregation and a shared recurrent node update."""

    def __init__(self, width: int, hidden: int = 80, cycles: int = 2, edge_hidden: int = 8) -> None:
        super().__init__(width, hidden, cycles)
        self.edge_embedding = nn.Embedding(3, edge_hidden)
        self.message = nn.Sequential(nn.Linear(2 * hidden + edge_hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update = nn.GRUCell(hidden, hidden)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.stem(features.transpose(1, 2))
        for dilation in dilation_schedule(nodes.shape[1], self.cycles):
            left, left_valid = shifted(nodes, dilation)
            right, right_valid = shifted(nodes, -dilation)
            sources = (left, nodes, right)
            valid = (left_valid, torch.ones_like(left_valid), right_valid)
            messages = []
            for edge_type, (source, mask) in enumerate(zip(sources, valid)):
                edge = self.edge_embedding.weight[edge_type].view(1, 1, -1).expand(nodes.shape[0], nodes.shape[1], -1)
                value = self.message(torch.cat((source, nodes, edge), dim=-1))
                messages.append(value.masked_fill(~mask, torch.finfo(value.dtype).min))
            aggregate = torch.stack(messages, dim=2).amax(dim=2)
            nodes = self.update(aggregate.flatten(0, 1), nodes.flatten(0, 1)).view_as(nodes)
        return self.finish(nodes)
