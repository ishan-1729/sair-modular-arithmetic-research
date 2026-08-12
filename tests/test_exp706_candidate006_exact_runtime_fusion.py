from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from tools import build_candidate006_exact_runtime_artifact as builder


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_preserves_manifest_and_weights(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    receipt = builder.build(output)
    assert receipt["manifest_unchanged"] is True
    assert receipt["weights_unchanged"] is True
    assert receipt["training_executed"] is False
    assert receipt["files"]["model.py"] == builder.OPTIMIZED_MODEL_SHA256


def test_optimized_wide_kernel_preserves_output_signs(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    builder.build(output)
    baseline_module = load_module(
        "exp706_baseline",
        builder.SOURCE / "model.py",
    )
    optimized_module = load_module("exp706_optimized", output / "model.py")
    baseline = baseline_module.Tier2HornerModel()
    optimized = optimized_module.Tier2HornerModel()
    baseline.load(str(builder.SOURCE))
    optimized.load(str(output))
    generator = torch.Generator().manual_seed(706)
    features = torch.randn((8, 6, 256), generator=generator)
    with torch.inference_mode():
        expected, _ = baseline.tier7_network(features)
        actual, auxiliary = optimized.tier7_network(features)
    assert torch.equal(expected.ge(0), actual.ge(0))
    assert auxiliary.shape == (8, 0)


def test_private_seed1_tier7_probe_hash_is_frozen() -> None:
    import yaml

    config = yaml.safe_load(
        (ROOT / "configs/exp_706_candidate006_exact_runtime_fusion.yaml").read_text()
    )
    assert config["gates"]["maximum_inference_seconds"] == 300
    assert config["gates"]["require_overall_accuracy"] == 0.641
    assert config["gates"]["require_tier7_accuracy"] == 0.54
    assert config["ruleset_scope"]["purpose"] == "pinned-runtime-equivalence-only"
    assert (
        config["ruleset_scope"]["current_upstream_submission_compliance"]
        == "blocked-deterministic-full-width-operand-prereduction"
    )
    assert (
        config["evaluation"]["seed1_private_tier7_prediction_sha256"]
        == "d7678ff8a32f58b4417cceb829a3cac47834999ec85fdffa7532a4cd49029f9f"
    )


def test_durable_official_sandbox_clears_written_runtime_gate() -> None:
    run_directory = (
        ROOT / "experiments/runs/exp-706-candidate006-exact-runtime-fusion"
    )
    official_directory = run_directory / "official-sandbox"
    result = json.loads((official_directory / "result.json").read_text())
    sandbox = json.loads((official_directory / "sandbox-run.json").read_text())
    metrics = json.loads(
        (
            ROOT
            / "evals/results/exp-706-candidate006-exact-runtime-fusion/metrics.json"
        ).read_text()
    )
    compliance = json.loads(
        (run_directory / "current-rules-compliance.json").read_text()
    )
    log_path = official_directory / "container.stderr.log"
    log = log_path.read_text()

    assert result["status"] == "completed"
    assert all(tier["completed"] for tier in result["tiers"])
    assert [tier["accuracy"] for tier in result["tiers"][1:8]] == [
        1.0,
        1.0,
        0.98,
        0.96,
        0.96,
        0.97,
        0.54,
    ]
    assert metrics["official_sandbox"]["inference_seconds"] == 281.6
    assert metrics["official_sandbox"]["modeled_tiers1_through7_accuracy_unchanged"]
    assert metrics["gates"]["no_accuracy_sacrifice"]
    assert metrics["gates"]["pinned_rules_runtime_gate_passed"]
    assert metrics["gates"]["current_upstream_compliance_passed"] is False
    assert metrics["gates"]["promotion_passed"] is False
    assert compliance["status"] == "blocked"
    assert compliance["runtime_ruleset"]["runtime_result_valid"] is True
    assert sandbox["timeout_seconds"] == 300
    assert sandbox["cpus"] == "4"
    assert sandbox["memory"] == "8g"
    assert "Tier 7 complete: 100 predictions in 281.6s" in log
    assert "Tier 10 complete: 100 predictions in 281.6s" in log
    assert (
        hashlib.sha256(log_path.read_bytes()).hexdigest()
        == sandbox["container_stderr_sha256"]
    )
