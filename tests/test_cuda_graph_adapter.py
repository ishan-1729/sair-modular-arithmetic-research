from __future__ import annotations

import ast
import copy
import inspect

import torch
from safetensors.torch import save_file
from torch_geometric.data import Batch, Data

from experiments.cuda_portability import graph as adapter
from experiments.cuda_portability.runtime import load_safetensors_on_device
from experiments.exp_401_pyg_graph_message_passing import model as frozen


def test_exp510_typed_max_cpu_forward_is_exact_and_edges_are_cached(
    monkeypatch,
) -> None:
    torch.manual_seed(510)
    reference = frozen.PyGRecurrentTypedMaxMPNN(
        width=9, hidden=16, cycles=1
    ).eval()
    portable_model = copy.deepcopy(reference)
    portable = adapter.Exp510TypedMaxAdapter(portable_model, frozen).eval()
    features = torch.randn(3, 6, 9)
    expected = reference(features)
    calls = 0
    original = frozen.cpu_edges

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(frozen, "cpu_edges", counted)
    actual = portable(features)
    first_calls = calls
    repeated = portable(features)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(repeated[0], actual[0])
    assert torch.equal(repeated[1], actual[1])
    assert first_calls == len(frozen.dilation_schedule(9, 1))
    assert calls == first_calls


def test_pyg_batch_moves_once_to_the_model_device() -> None:
    model = frozen.PyGRecurrentTypedMaxMPNN(width=4, hidden=8, cycles=1)
    batch = Batch.from_data_list(
        [
            Data(x=torch.randn(4, 3), edge_index=torch.tensor([[0, 1], [1, 2]])),
            Data(x=torch.randn(4, 3), edge_index=torch.tensor([[2, 3], [3, 0]])),
        ]
    )
    moved = adapter.move_graph_batch_to_model(batch, model)
    assert moved.x.device == next(model.parameters()).device
    assert moved.edge_index.device == next(model.parameters()).device


def test_safetensors_checkpoint_loads_on_explicit_model_device(tmp_path) -> None:
    torch.manual_seed(511)
    source = frozen.PyGRecurrentTypedMaxMPNN(width=5, hidden=8, cycles=1)
    checkpoint = tmp_path / "graph.safetensors"
    save_file(
        {name: value.detach().contiguous() for name, value in source.state_dict().items()},
        checkpoint,
    )
    target = frozen.PyGRecurrentTypedMaxMPNN(width=5, hidden=8, cycles=1)
    load_safetensors_on_device(target, checkpoint, torch.device("cpu"))
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            source.state_dict().values(), target.state_dict().values(), strict=True
        )
    )


def test_prepared_graph_recurrence_contains_no_transfer_boundary() -> None:
    tree = ast.parse(inspect.getsource(adapter.typed_max_forward_prepared))
    forbidden = {"cpu", "cuda", "item", "numpy", "to", "tolist"}
    loop_calls = []
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        loop_calls.extend(
            node.func.attr
            for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        )
    assert forbidden.isdisjoint(loop_calls)
