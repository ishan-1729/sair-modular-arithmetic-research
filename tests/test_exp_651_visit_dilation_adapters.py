from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_651_visit_dilation_adapters.model import VisitDilationAdapterCell
from experiments.exp_651_visit_dilation_adapters.run import Experiment, RootSpec, aggregate_resources, paired_bootstrap, sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/exp_651_visit_dilation_adapters.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def adapter(kind: str, rank: int) -> VisitDilationAdapterCell:
    torch.manual_seed(651)
    base = SharedDilatedConvGRU(width=128, hidden=12, cycles=2)
    return VisitDilationAdapterCell(base, kind=kind, rank=rank, alpha=rank, max_dilation_levels=8)


def test_frozen_roots_denominators_and_closed_surfaces() -> None:
    payload = config()
    assert payload["data"]["calibration"]["root"] == 20261182
    assert [item["root"] for item in payload["data"]["formal_roots"]] == [20261183, 20261184, 20261185]
    assert payload["evaluation"]["widths"] == [64, 128, 256]
    assert payload["evaluation"]["transition_cases"] == {64: 1000, 128: 1000, 256: 1000}
    assert payload["evaluation"]["rollout_cases"] == {64: 200, 128: 200, 256: 200}
    assert payload["training"]["steps"] == 200
    assert payload["resource_budget"]["fidelity_reduced"] is False
    assert payload["source_contract"]["checked_prior_independent_official_private"] == "prohibited"


def test_exact_initialization_indexing_and_parameter_matching() -> None:
    visit = adapter("visit_dilation", 8).eval()
    global_control = adapter("global", 128).eval()
    reference = SharedDilatedConvGRU(width=128, hidden=12, cycles=2).eval()
    reference.load_state_dict(visit.base.state_dict())
    assert visit.adapter_parameter_count() == global_control.adapter_parameter_count()
    assert visit.slot_count == 16
    assert global_control.slot_count == 1
    assert visit.active_slot_count(64) == 12
    assert visit.active_slot_count(128) == 14
    assert visit.active_slot_count(256) == 16
    assert visit.slot_for(0, 1) == 0
    assert visit.slot_for(0, 128) == 7
    assert visit.slot_for(1, 1) == 8
    assert visit.slot_for(1, 128) == 15
    for width in (64, 128, 256):
        features = torch.randn(3, 6, width)
        visit.width = reference.width = width
        expected = reference(features)
        observed = visit(features)
        assert torch.equal(expected[0], observed[0])
        assert torch.equal(expected[1], observed[1])


def test_activation_zeroing_permutation_export_reload_and_gradients(tmp_path: Path) -> None:
    model = adapter("visit_dilation", 8)
    features = torch.randn(2, 6, 64)
    initial = model(features)
    loss = initial[0].square().mean() + initial[1].square().mean()
    loss.backward()
    assert model.gate_b.grad is not None and model.gate_b.grad.abs().sum() > 0
    assert model.candidate_b.grad is not None and model.candidate_b.grad.abs().sum() > 0
    assert model.gate_a.grad is not None and model.gate_a.grad.abs().sum() == 0
    optimizer = torch.optim.SGD(model.adapter_parameters(), lr=0.1)
    optimizer.step()
    changed = model(features)
    zeroed = model(features, adapter_enabled=False)
    permuted = model(features, slot_permutation=tuple(reversed(range(16))))
    assert not torch.equal(changed[0], zeroed[0])
    assert not torch.equal(changed[0], permuted[0])
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = adapter("visit_dilation", 8)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(model(features)[0], reloaded(features)[0])
    assert all(parameter.grad is None for parameter in model.base.parameters())


def test_malformed_and_leakage_fail_closed() -> None:
    model = adapter("visit_dilation", 8)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 5, 64))
    malformed = torch.zeros(2, 6, 64)
    malformed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(malformed)
    with pytest.raises(ValueError, match="power of two"):
        model.slot_for(0, 3)
    with pytest.raises(ValueError, match="bijection"):
        model(torch.zeros(2, 6, 64), slot_permutation=(0,) * 16)
    source = inspect.getsource(VisitDilationAdapterCell).lower()
    for forbidden in ("expected_answer", "barrett", "montgomery", "crt_recomb", "torch.remainder", "fmod(", "data/checked", "data/private", "submission/"):
        assert forbidden not in source


def test_deterministic_initialization_and_forward() -> None:
    torch.use_deterministic_algorithms(True)
    first = adapter("visit_dilation", 8).eval()
    second = adapter("visit_dilation", 8).eval()
    features = torch.randn(2, 6, 64, generator=torch.Generator().manual_seed(7))
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))
    assert torch.equal(first(features)[0], second(features)[0])


def test_hierarchical_bootstrap_and_conjunctive_gate() -> None:
    deltas = np.zeros((3, 200), dtype=np.float64)
    deltas[:, :20] = 1.0
    result = paired_bootstrap(deltas, seed=65195002, resamples=100_000)
    assert result["mean"] == pytest.approx(0.1)
    assert result["clusters"] == 3
    assert len(result["distribution_sha256"]) == 64
    with pytest.raises(ValueError, match="100,000"):
        paired_bootstrap(deltas, seed=1, resamples=100)

    experiment = object.__new__(Experiment)
    experiment.config = config()

    def arm(w64: float, w128: float, w256: float, adapter_arm: bool = False) -> dict:
        value = {
            "evaluation": {str(width): {"rollout": {"exact_accuracy": score, "any_invalid_residue_rate": 0.0}} for width, score in ((64, w64), (128, w128), (256, w256))},
            "deterministic_reload": True,
        }
        if adapter_arm:
            value["activation_and_interventions"] = {"adapter_delta_rms": 0.001, "zeroing_changed_predictions": 2, "permutation_changed_predictions": 2, "trained_minus_randomized_w128": 0.2}
        return value

    roots = [{"arms": {"visit_dilation_adapter": arm(0.96, 0.90, 0.30, True), "global_adapter_control": arm(0.96, 0.87, 0.28, True), "shared_core_continuation": arm(0.96, 0.86, 0.27)}} for _ in range(3)]
    bootstrap = {"64": {"one_sided_95_lower_bound": 0.0}, "128": {"one_sided_95_lower_bound": 0.01}, "256": {"one_sided_95_lower_bound": 0.005}}
    resource = {"wall_seconds": 10, "process_cpu_seconds": 20, "cgroup_cpu_seconds": 21, "peak_rss_bytes": 100}
    admission = {"hard_budget": {"wall_seconds": 20, "process_cpu_seconds": 40, "cgroup_cpu_seconds": 42, "peak_rss_bytes": 200}}
    checks, _ = experiment.gates(roots, bootstrap, resource, admission)
    assert all(checks.values())
    roots[2]["arms"]["visit_dilation_adapter"]["evaluation"]["128"]["rollout"]["exact_accuracy"] = 0.86
    checks, _ = experiment.gates(roots, bootstrap, resource, admission)
    assert not checks["w128_each_root"]


def test_phase_resource_accounting_and_hash_valid_partial_recovery(tmp_path: Path) -> None:
    combined = aggregate_resources([
        {"wall_seconds": 2.0, "process_cpu_seconds": 7.0, "cgroup_cpu_seconds": 7.5, "peak_rss_bytes": 100},
        {"wall_seconds": 3.0, "process_cpu_seconds": 10.0, "cgroup_cpu_seconds": 10.5, "peak_rss_bytes": 120},
    ])
    assert combined == {"wall_seconds": 5.0, "process_cpu_seconds": 17.0, "cgroup_cpu_seconds": 18.0, "peak_rss_bytes": 120}

    fixture = tmp_path / "completed-arm.safetensors"
    fixture.write_bytes(b"hash-valid-test-checkpoint")
    experiment = object.__new__(Experiment)
    experiment.mode = "formal"
    experiment.output = tmp_path
    experiment.config = {"arms": {"visit_dilation_adapter": {}}}
    experiment.source_identity = lambda: {"source": "frozen"}
    spec = RootSpec(root=20261183, train_seed=1, eval_seed=2)
    progress = {
        "source": experiment.source_identity(),
        "root": spec.root,
        "steps": 200,
        "arms": {"visit_dilation_adapter": {"checkpoint": {"path": str(fixture), "sha256": sha256(fixture)}}},
    }
    path = tmp_path / "formal/root-20261183-progress.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(progress), encoding="utf-8")
    assert set(experiment._partial_root_arms(spec, 200)) == {"visit_dilation_adapter"}
    progress["arms"]["visit_dilation_adapter"]["checkpoint"]["sha256"] = "0" * 64
    path.write_text(json.dumps(progress), encoding="utf-8")
    assert experiment._partial_root_arms(spec, 200) == {}
