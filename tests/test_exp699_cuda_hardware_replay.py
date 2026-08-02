from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import pytest
import torch
import yaml

from experiments.cloud_benchmarks.exp699_cuda_replay import run as replay
from experiments.exp_699_h92_w256_backbone_factorial_complete import run as source

ROOT = Path(__file__).resolve().parents[1]
REPLAY_CONFIG = yaml.safe_load(replay.REPLAY_CONFIG_PATH.read_text(encoding="utf-8"))
SCIENTIFIC_CONFIG = yaml.safe_load(
    (ROOT / REPLAY_CONFIG["source"]["config"]).read_text(encoding="utf-8")
)
CHECKPOINT = Path(SCIENTIFIC_CONFIG["source_checkpoint"])


def test_replay_preregistration_freezes_completed_arm_and_cost_ceiling() -> None:
    assert REPLAY_CONFIG["classification"] == "non-scientific-hardware-replay"
    assert REPLAY_CONFIG["source"]["formal_root"] == 20261396
    assert REPLAY_CONFIG["replay"] == {
        "arm": "teacher_cosine",
        "device": "cuda",
        "torch_threads": 4,
        "updates": 4800,
        "transition_cases_per_width": 3000,
        "w128_rollout_cases": 300,
        "w256_rollout_cases": 600,
        "checkpoint_updates": [400, 800, 1200, 2400, 3600, 4800],
        "curve_every_updates": 100,
    }
    assert REPLAY_CONFIG["guardrails"]["maximum_wall_seconds"] == 7200
    assert REPLAY_CONFIG["guardrails"]["maximum_compute_usd"] == 5.0
    assert REPLAY_CONFIG["guardrails"]["minimum_required_speedup"] == 3.0
    assert REPLAY_CONFIG["guardrails"]["allow_full_cpu_replay"] is False


def test_local_reference_is_scoped_observed_completed_arm() -> None:
    reference = json.loads(replay.LOCAL_REFERENCE_PATH.read_text(encoding="utf-8"))
    assert reference["classification"] == "observed-completed-cpu-arm"
    assert reference["root"] == 20261396 and reference["arm"] == "teacher_cosine"
    assert reference["resource"]["route_wall_seconds"] == 11452.156460124002
    assert reference["work"] == {
        "updates": 4800,
        "examples": 614400,
        "model_calls": 38400,
        "transition_cases_per_width": 3000,
        "w128_rollout_cases": 300,
        "w256_rollout_cases": 600,
    }
    assert reference["evaluations"]["128"]["rollout"]["correct"] == 260
    assert reference["evaluations"]["256"]["rollout"]["correct"] == 373


def test_requirements_are_index_isolated_and_exactly_pinned() -> None:
    torch_manifest = (
        replay.BENCHMARK_DIR / "requirements-torch-cu124.txt"
    ).read_text(encoding="utf-8")
    pypi_manifest = (replay.BENCHMARK_DIR / "requirements.txt").read_text(
        encoding="utf-8"
    )
    bootstrap = (replay.BENCHMARK_DIR / "bootstrap.sh").read_text(
        encoding="utf-8"
    )
    guard = (replay.BENCHMARK_DIR / "run_guarded.sh").read_text(
        encoding="utf-8"
    )
    assert "--index-url https://download.pytorch.org/whl/cu124" in torch_manifest
    assert "torch==2.5.1+cu124" in torch_manifest
    for requirement in (
        "numpy==2.5.1",
        "mlflow==3.14.0",
        "psutil==7.0.0",
        "sympy==1.13.1",
        "PyYAML==6.0.3",
        "safetensors==0.8.0",
    ):
        assert requirement in pypi_manifest
    assert 'expected_uv="uv 0.11.28"' in bootstrap
    assert "uv pip install" in bootstrap and "uv pip check" in bootstrap
    assert "timeout --foreground --signal=TERM" in guard
    assert 'EXP699_MAX_WALL_SECONDS:-7200' in guard


def test_device_resolution_fails_closed_without_cuda() -> None:
    assert replay.resolve_device("cpu") == torch.device("cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA replay requested"):
            replay.resolve_device("cuda")


def test_cost_admission_uses_lower_wall_or_cash_ceiling() -> None:
    assert replay.admitted_wall_seconds(7200, 0.69, 5.0) == 7200
    assert replay.admitted_wall_seconds(7200, 3.0, 5.0) == 6000
    assert replay.admitted_wall_seconds(900, None, 5.0) == 900
    with pytest.raises(ValueError, match="hourly rate"):
        replay.admitted_wall_seconds(7200, 0.0, 5.0)


def test_dry_run_persists_preflight_receipt(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=replay.REPLAY_CONFIG_PATH,
        source_checkpoint=CHECKPOINT,
        output=tmp_path,
        profile="smoke",
        device="cpu",
        hourly_rate_usd=None,
        max_wall_seconds=30.0,
        max_compute_usd=0.0,
        allow_cpu_full_replay=False,
        dry_run=True,
    )
    result = replay.execute(args)
    persisted = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert result["status"] == "dry-run"
    assert persisted["status"] == "admitted"
    assert persisted["guardrails"]["maximum_compute_usd"] == 0.0


def test_deadline_is_checked_at_work_boundaries() -> None:
    deadline = replay.Deadline(0.001)
    time.sleep(0.003)
    with pytest.raises(TimeoutError, match="unit-test"):
        deadline.check("unit-test")


def test_device_adapter_cpu_features_and_loss_match_frozen_source() -> None:
    parent = source.base.load_exp300()
    adapter = replay.DeviceExp300(parent, torch.device("cpu"), replay.Deadline(30))
    state = [3, 7]
    multiplicand = [5, 11]
    modulus = [13, 17]
    input_bit = [0, 1]
    assert torch.equal(
        adapter.features(state, multiplicand, modulus, input_bit, 16),
        parent.features(state, multiplicand, modulus, input_bit, 16),
    )
    torch.manual_seed(7)
    logits = torch.randn(2, 16)
    quotient_logits = torch.randn(2, 3)
    targets = [9, 4]
    quotients = torch.tensor([0, 1], dtype=torch.long)
    expected_loss, expected_stats = parent.transition_loss(
        logits, quotient_logits, targets, quotients, 16, 0.45, 0.10
    )
    actual_loss, actual_stats = adapter.transition_loss(
        logits, quotient_logits, targets, quotients, 16, 0.45, 0.10
    )
    assert torch.equal(actual_loss, expected_loss)
    assert actual_stats == expected_stats


def test_one_update_cpu_replay_is_byte_identical_to_frozen_route(
    tmp_path: Path,
) -> None:
    assert CHECKPOINT.is_file()
    config = copy.deepcopy(SCIENTIFIC_CONFIG)
    config["training"]["profile_width"] = 128
    parent = source.base.load_exp300()
    source.base.configure_torch(config)
    root = int(REPLAY_CONFIG["source"]["formal_root"])
    splits = {
        width: source.make_split(config, parent, width, root)
        for width in replay.WIDTHS
    }
    _, expected = source.train_arm(
        config, parent, CHECKPOINT, splits, root, replay.ARM, 1, tmp_path / "source"
    )
    adapter = replay.DeviceExp300(
        parent, torch.device("cpu"), replay.Deadline(120)
    )
    _, actual = replay.train_arm(
        config,
        adapter,
        CHECKPOINT,
        splits,
        root,
        replay.ARM,
        1,
        tmp_path / "replay",
        {1},
        1,
    )
    assert actual["initial_state_sha256"] == expected["initial_state_sha256"]
    assert actual["final_state_sha256"] == expected["final_state_sha256"]
    assert actual["final_checkpoint"]["sha256"] == expected["final_checkpoint"]["sha256"]
    assert actual["training"]["updates"] == expected["training"]["updates"]
    assert actual["training"]["examples"] == expected["training"]["examples"]
    assert actual["training"]["model_calls"] == expected["training"]["model_calls"]
    assert actual["training"]["width_order_sha256"] == expected["training"]["width_order_sha256"]
    assert actual["training"]["data_order_sha256"] == expected["training"]["data_order_sha256"]
    assert actual["training"]["curves"] == expected["training"]["curves"]
    assert actual["reload_exact"] is True


def test_full_cpu_replay_requires_explicit_opt_in(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=replay.REPLAY_CONFIG_PATH,
        source_checkpoint=CHECKPOINT,
        output=tmp_path,
        profile="full",
        device="cpu",
        hourly_rate_usd=0.1,
        max_wall_seconds=1.0,
        max_compute_usd=0.01,
        allow_cpu_full_replay=False,
        dry_run=True,
    )
    with pytest.raises(RuntimeError, match="full CPU replay is disabled"):
        replay.execute(args)


def test_source_identity_is_frozen() -> None:
    identity = replay.source_identity(REPLAY_CONFIG, CHECKPOINT)
    assert identity["source_sha256"] == {
        "config": REPLAY_CONFIG["source"]["config_sha256"],
        "runner": REPLAY_CONFIG["source"]["runner_sha256"],
    }
    assert (
        identity["source_checkpoint_sha256"]
        == REPLAY_CONFIG["source"]["checkpoint_sha256"]
    )
