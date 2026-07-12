"""PyTorch Geometric graph models for width-shared modular state transitions."""

from __future__ import annotations

import functools
import math

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, GCNConv, MessagePassing


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


@functools.lru_cache(maxsize=128)
def cpu_edges(batch_size: int, width: int, dilation: int) -> tuple[Tensor, Tensor]:
    sources = []
    targets = []
    attributes = []
    scale = math.log2(dilation) / max(1.0, math.log2(width))
    for batch in range(batch_size):
        offset = batch * width
        for target in range(width):
            sources.append(offset + target); targets.append(offset + target); attributes.append((0.0, 1.0, 0.0, scale))
            if target >= dilation:
                sources.append(offset + target - dilation); targets.append(offset + target); attributes.append((1.0, 0.0, 0.0, scale))
            if target + dilation < width:
                sources.append(offset + target + dilation); targets.append(offset + target); attributes.append((0.0, 0.0, 1.0, scale))
    return torch.tensor((sources, targets), dtype=torch.long), torch.tensor(attributes, dtype=torch.float32)


def graph_edges(batch_size: int, width: int, dilation: int, device: torch.device) -> tuple[Tensor, Tensor]:
    edge_index, edge_attr = cpu_edges(batch_size, width, dilation)
    return edge_index.to(device), edge_attr.to(device)


class PyGGraphBase(nn.Module):
    def __init__(self, width: int, hidden: int, cycles: int) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.cycles = cycles
        self.stem = nn.Linear(6, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.local_head = nn.Linear(hidden, 1)
        self.quotient_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def finish(self, nodes: Tensor, batch_size: int, width: int) -> tuple[Tensor, Tensor]:
        nodes = torch.nn.functional.silu(self.norm(nodes)).view(batch_size, width, self.hidden)
        logits = self.local_head(nodes).squeeze(-1)
        pooled = torch.cat((nodes.mean(dim=1), nodes.amax(dim=1)), dim=1)
        return logits, self.quotient_head(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class PyGRecurrentGCN(PyGGraphBase):
    def __init__(self, width: int, hidden: int = 92, cycles: int = 2) -> None:
        super().__init__(width, hidden, cycles)
        self.conv = GCNConv(hidden, hidden, add_self_loops=False, normalize=True)
        self.update = nn.GRUCell(hidden, hidden)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, _, width = features.shape
        nodes = self.stem(features.transpose(1, 2)).reshape(batch_size * width, self.hidden)
        for dilation in dilation_schedule(width, self.cycles):
            edge_index, _ = graph_edges(batch_size, width, dilation, nodes.device)
            message = torch.tanh(self.conv(nodes, edge_index))
            nodes = self.update(message, nodes)
        return self.finish(nodes, batch_size, width)


class PyGRecurrentGATv2(PyGGraphBase):
    def __init__(self, width: int, hidden: int = 92, cycles: int = 2, heads: int = 4) -> None:
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        super().__init__(width, hidden, cycles)
        self.conv = GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, edge_dim=4, add_self_loops=False, share_weights=False, residual=True)
        self.update = nn.GRUCell(hidden, hidden)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, _, width = features.shape
        nodes = self.stem(features.transpose(1, 2)).reshape(batch_size * width, self.hidden)
        for dilation in dilation_schedule(width, self.cycles):
            edge_index, edge_attr = graph_edges(batch_size, width, dilation, nodes.device)
            message = torch.tanh(self.conv(nodes, edge_index, edge_attr=edge_attr))
            nodes = self.update(message, nodes)
        return self.finish(nodes, batch_size, width)


class TypedMaxLayer(MessagePassing):
    def __init__(self, hidden: int, edge_dim: int = 4) -> None:
        super().__init__(aggr="max", flow="source_to_target", node_dim=0)
        self.message_mlp = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, nodes, edge_index, edge_attr):
        return self.propagate(edge_index, x=nodes, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))


class PyGRecurrentTypedMaxMPNN(PyGGraphBase):
    def __init__(self, width: int, hidden: int = 80, cycles: int = 2) -> None:
        super().__init__(width, hidden, cycles)
        self.conv = TypedMaxLayer(hidden, edge_dim=4)
        self.update = nn.GRUCell(hidden, hidden)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, _, width = features.shape
        nodes = self.stem(features.transpose(1, 2)).reshape(batch_size * width, self.hidden)
        for dilation in dilation_schedule(width, self.cycles):
            edge_index, edge_attr = graph_edges(batch_size, width, dilation, nodes.device)
            message = self.conv(nodes, edge_index, edge_attr)
            nodes = self.update(message, nodes)
        return self.finish(nodes, batch_size, width)
