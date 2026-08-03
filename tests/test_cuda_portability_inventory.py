from __future__ import annotations

import json

import pytest

from experiments.cuda_portability import inventory


@pytest.fixture(scope="module")
def manifest() -> dict:
    return inventory.generate()


def test_generated_inventory_is_fresh_and_closed(manifest: dict) -> None:
    config = inventory.yaml.safe_load(
        inventory.CONFIG_PATH.read_text(encoding="utf-8")
    )
    committed = json.loads(
        (inventory.ROOT / config["scan"]["manifest"]).read_text(encoding="utf-8")
    )
    assert inventory.encode(manifest) == inventory.encode(committed)
    assert manifest["readiness"] == "cuda-prepared-hardware-unvalidated"
    assert all(manifest["closure"].values())
    assert manifest["closure"]["no_adapter_required"]
    assert manifest["source"]["registry"] == "experiments/registry.yaml"
    assert len(manifest["source"]["registry_sha256"]) == 64
    assert manifest["source"]["python_file_count"] == len(
        inventory.discover_python(config)
    )
    assert manifest["source"]["root_count"] == len(inventory.discover_roots(config))
    assert manifest["source"]["python_file_count"] >= 229
    assert manifest["source"]["root_count"] >= 159


def test_every_module_has_evidence_and_every_hotspot_inherits_classification(
    manifest: dict,
) -> None:
    allowed = {
        "device-native-unvalidated",
        "intentionally-cpu-boundary",
        "archived-frozen-route",
    }
    for row in manifest["modules"]:
        assert row["classification"] in allowed
        assert row["evidence"]
        assert row["reachability_roots"]
        assert "local_import_parents" in row
        for hotspot in row["hotspots"]:
            assert hotspot["symbol"] and hotspot["line"] > 0
            assert hotspot["kind"] in {
                "tensor-factory",
                "host-boundary",
                "device-boundary",
                "checkpoint",
                "optimizer",
            }


def test_archived_routes_have_exact_terminal_registry_evidence(manifest: dict) -> None:
    archived = [
        row for row in manifest["modules"]
        if row["classification"] == "archived-frozen-route"
    ]
    assert archived
    for row in archived:
        lifecycle = row["durable_lifecycle"]
        assert lifecycle["all_terminal"]
        assert lifecycle["all_artifact_identities_complete"]
        assert not lifecycle["active_consumers"]
        assert lifecycle["routes"]
        for route in lifecycle["routes"]:
            assert route["run_id"]
            assert route["status"]
            assert route["durable_artifacts"]
            assert not route["identity_mismatches"]
            for artifact in route["durable_artifacts"]:
                assert artifact["path"]
                assert len(artifact["sha256"]) == 64


def test_live_and_superseded_route_dispositions_are_consumer_aware(
    manifest: dict,
) -> None:
    rows = {row["path"]: row for row in manifest["modules"]}

    exp649 = rows["experiments/exp_649_stochastic_trajectory_breadth/run.py"]
    assert exp649["classification"] == "intentionally-cpu-boundary"
    boundary = exp649["exp649_cpu_boundary"]
    assert boundary["authenticated"]
    assert boundary["noise_distribution"] == "iid_cpu_float32_gaussian"
    assert boundary["coordinate_traversal"] == (
        "multiplier_bit_msb_to_lsb_then_injection_0_to_16_then_case_channel_position"
    )
    assert boundary["reset_points"] == [
        "before_warmup",
        "after_warmup",
        "before_each_replicate",
        "before_exact_rerun",
    ]
    assert not boundary["runtime_gate_passed"]
    assert not boundary["formal_authorized"]
    assert not boundary["formal_root_opened"]
    assert all(len(value) == 64 for value in boundary["hashes"].values())

    shared_model = rows["experiments/exp_220_width_independent_cellular/model.py"]
    assert shared_model["classification"] == "intentionally-cpu-boundary"
    assert {
        route["run_id"]
        for route in shared_model["durable_lifecycle"]["active_consumers"]
    } == {"exp-649-stochastic-trajectory-breadth"}
    assert shared_model["exp649_cpu_boundary"] == boundary

    exp699 = rows[
        "experiments/exp_699_h92_w256_backbone_factorial_complete/run.py"
    ]
    assert exp699["classification"] == "device-native-unvalidated"
    assert exp699["adapter"]["coverage"] == "full-active-route"

    superseded = [
        row
        for path, row in rows.items()
        if path.startswith(
            "experiments/runs/exp-648-structured-trajectory-consistency/superseded/"
        )
    ]
    assert len(superseded) == 2
    assert all(
        row["classification"] == "archived-frozen-route" for row in superseded
    )


def test_representative_recurrent_loops_have_no_transfer_calls(
    manifest: dict,
) -> None:
    audits = {row["path"]: row for row in manifest["hot_loop_audit"]}
    assert set(audits) == {
        "experiments/cuda_portability/graph.py",
        "experiments/cuda_portability/recurrent.py",
        "experiments/cloud_benchmarks/exp699_cuda_replay/run.py",
    }
    assert all(row["passed"] and not row["violations"] for row in audits.values())
    assert audits[
        "experiments/cloud_benchmarks/exp699_cuda_replay/run.py"
    ]["functions"] == ["rollout", "train_update"]
    assert audits["experiments/cuda_portability/graph.py"]["functions"] == [
        "typed_max_forward_prepared"
    ]
    assert audits["experiments/cuda_portability/recurrent.py"]["functions"] == [
        "neural_ram_forward_prepared"
    ]
    assert all(
        row["forbidden_factories"] == ["as_tensor", "from_numpy", "tensor"]
        for row in audits.values()
    )
