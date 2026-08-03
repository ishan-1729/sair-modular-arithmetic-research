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
    assert manifest["readiness"] == "cuda-inventory-open"
    assert all(manifest["closure"].values())
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
        "adapter-required",
        "intentionally-cpu-boundary",
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


def test_exp699_recurrent_loops_have_no_host_or_device_transfer_calls(
    manifest: dict,
) -> None:
    assert manifest["hot_loop_audit"] == [
        {
            "path": "experiments/cloud_benchmarks/exp699_cuda_replay/run.py",
            "functions": ["rollout", "train_update"],
            "forbidden_calls": ["cpu", "cuda", "item", "numpy", "to", "tolist"],
            "violations": [],
            "passed": True,
        }
    ]
