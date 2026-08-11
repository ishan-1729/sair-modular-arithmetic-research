from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from experiments.exp_703_exp701_terminal_evaluation_cpu_continuation import run


def config() -> dict:
    return yaml.safe_load(run.CONFIG_PATH.read_text(encoding="utf-8"))


def fake_evaluation(width: int, cfg: dict) -> dict:
    return {
        "width": width,
        "transition": {
            "cases": cfg["evaluation"]["transition_cases_per_width"],
            "correct": 0,
            "exact_accuracy": 0.0,
        },
        "rollout": {
            "cases": cfg["evaluation"]["rollout_cases"][str(width)],
            "correct": 0,
            "exact_accuracy": 0.0,
        },
        "resource": {"wall_seconds": 1.0},
        "paired_evidence_resource": {"wall_seconds": 1.0},
    }


def test_cpu_contract_is_distinct_and_freezes_science() -> None:
    cfg, exp702_cfg = run.load_contract(run.CONFIG_PATH)
    assert cfg["run_id"] == run.RUN_ID
    assert cfg["run_id"] != exp702_cfg["run_id"]
    assert cfg["evaluation"] == exp702_cfg["evaluation"]
    assert cfg["terminal_policy"] == exp702_cfg["terminal_policy"]
    assert cfg["evaluation"]["training_allowed"] is False
    assert cfg["cpu_execution"]["device"] == "cpu"


def test_cpu_runtime_is_explicit_and_deterministic() -> None:
    cfg = config()
    device, receipt = run.configure_cpu(cfg)
    assert device == torch.device("cpu")
    assert receipt["device"] == "cpu"
    assert receipt["deterministic_algorithms"] is True
    assert receipt["cpu"]["torch_threads"] == cfg["cpu_execution"]["torch_threads"]


def test_phase_is_atomic_full_denominator_and_resumable(tmp_path: Path) -> None:
    cfg = config()
    calls = 0

    def compute() -> dict:
        nonlocal calls
        calls += 1
        return fake_evaluation(128, cfg)

    first, reused = run.load_or_compute_phase(
        output=tmp_path,
        width=128,
        identity="a" * 64,
        config=cfg,
        compute=compute,
        run_started=0.0,
    )
    assert reused is False
    assert calls == 1
    assert first["transition"]["cases"] == 3000
    assert first["rollout"]["cases"] == 300
    manifest = json.loads(
        (tmp_path / "phases/evaluation__straight_through_restarts__128.manifest.json").read_text()
    )
    assert manifest["status"] == "completed"

    second, reused = run.load_or_compute_phase(
        output=tmp_path,
        width=128,
        identity="a" * 64,
        config=cfg,
        compute=lambda: pytest.fail("completed phase was recomputed"),
        run_started=0.0,
    )
    assert reused is True
    assert second == first


def test_phase_rejects_wrong_denominator(tmp_path: Path) -> None:
    cfg = config()
    bad = fake_evaluation(256, cfg)
    bad["rollout"]["cases"] = 599
    with pytest.raises(run.SemanticFailure, match="denominator"):
        run.load_or_compute_phase(
            output=tmp_path,
            width=256,
            identity="b" * 64,
            config=cfg,
            compute=lambda: bad,
            run_started=0.0,
        )


def test_runner_contains_no_training_route() -> None:
    source = Path(run.__file__).read_text(encoding="utf-8")
    assert "train_arm_resumable(" not in source
    assert "train_arm(" not in source
    assert 'get("training_allowed") is not False' in source


def test_durable_terminal_result_matches_phase_hashes_and_denominators() -> None:
    run_dir = (
        run.ROOT
        / "experiments/runs/exp-703-exp701-terminal-evaluation-cpu-continuation"
    )
    terminal_path = run_dir / "raw/terminal-summary.json"
    publication = json.loads((run_dir / "raw/publication-receipt.json").read_text())
    terminal = json.loads(terminal_path.read_text())
    assert terminal["status"] == "completed"
    assert terminal["decision"] == "complete-negative-full-denominator"
    assert terminal["training_executed"] is False
    assert hashlib.sha256(terminal_path.read_bytes()).hexdigest() == publication["terminal_summary_sha256"]
    for width, transition_cases, rollout_cases in (("128", 3000, 300), ("256", 3000, 600)):
        phase = run_dir / f"raw/phases/evaluation__straight_through_restarts__{width}.json"
        manifest = json.loads(
            phase.with_name(phase.stem + ".manifest.json").read_text()
        )
        result = json.loads(phase.read_text())
        assert manifest["status"] == "completed"
        assert hashlib.sha256(phase.read_bytes()).hexdigest() == manifest["result_sha256"]
        assert result["transition"]["cases"] == transition_cases
        assert result["rollout"]["cases"] == rollout_cases
        assert terminal["evaluations"]["straight_through_restarts"][width] == result
