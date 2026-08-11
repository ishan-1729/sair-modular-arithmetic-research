from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from experiments.exp_702_exp701_terminal_evaluation_continuation import run


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(run.CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def verified(config: dict) -> dict:
    return run.verify_predecessor(config, run.ROOT / config["predecessor"]["archive"])


def test_exact_s3t_predecessor_is_terminal_incomplete(config: dict, verified: dict) -> None:
    assert verified["opened"]["identity_sha256"] == config["predecessor"]["identity_sha256"]
    assert verified["lifecycle"]["recoveries"] == 3
    assert verified["terminal"] == {
        "mode": "formal", "reason": "recoverable restart bound exceeded", "retry_allowed": False,
        "root": 20261414, "run_id": config["predecessor"]["run_id"],
        "schema_version": "exp701-terminal-failure-v1", "status": "terminal-failure",
    }
    assert verified["training"]["training"]["updates"] == 4800


def test_successor_source_contains_no_training_execution() -> None:
    tree = ast.parse(Path(run.__file__).read_text())
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert "train_arm_resumable" not in called
    assert "train_update" not in called
    assert "train" not in called


def _evaluation(accuracy: float) -> dict:
    return {"transition": {"exact_accuracy": 1.0}, "rollout": {"exact_accuracy": accuracy, "any_invalid_residue_rate": 0.0}}


def test_integrated_negative_decision_is_successor_only(config: dict, verified: dict) -> None:
    missing = {"128": _evaluation(0.9), "256": _evaluation(0.7)}
    result = run.integrate_result(config, verified, missing, {"status": "terminal-incomplete"})
    assert result["run_id"] == run.RUN_ID
    assert result["relationship"] == "terminal-evaluation-successor-not-exp701-completion"
    assert result["decision"] == "complete-negative-full-denominator"
    assert result["training_executed"] is False
    assert len(result["reused_phases"]) == 6
    assert len(result["executed_phases"]) == 2


def test_positive_outcome_terminates_without_unplanned_diagnostics(config: dict, verified: dict) -> None:
    missing = {"128": _evaluation(1.0), "256": _evaluation(0.9)}
    result = run.integrate_result(config, verified, missing, {"status": "terminal-incomplete"})
    assert result["status"] == "completed"
    assert result["confirmation_trigger"] is True
    assert result["decision"] == "complete-positive-preregistered-gate-reached-no-additional-science"
    assert result["additional_diagnostics_executed"] is False


def test_phase_publication_is_retry_safe_and_identity_bound(tmp_path, config: dict) -> None:
    calls = []
    first, reused = run.load_or_publish_phase(
        output=tmp_path, width=128, identity="a" * 64, config=config,
        compute=lambda: calls.append("computed") or _evaluation(0.5),
    )
    second, reused_second = run.load_or_publish_phase(
        output=tmp_path, width=128, identity="a" * 64, config=config,
        compute=lambda: calls.append("recomputed") or _evaluation(0.6),
    )
    assert first == second and calls == ["computed"]
    assert reused is False and reused_second is True
    with pytest.raises(run.SemanticFailure, match="identity mismatch"):
        run.load_or_publish_phase(
            output=tmp_path, width=128, identity="b" * 64, config=config,
            compute=lambda: _evaluation(0.7),
        )


def test_durable_publication_keeps_identities_separate(config: dict) -> None:
    exp701 = json.loads((run.ROOT / "experiments/runs/exp-701-cuda-resumable-h92-w256-backbone-factorial/terminal-incomplete.json").read_text())
    exp702 = json.loads((run.ROOT / "experiments/runs/exp-702-exp701-terminal-evaluation-continuation/preregistration.json").read_text())
    assert exp701["status"] == "terminal-incomplete"
    assert exp701["scientific_verdict"] is None
    assert exp702["status"] == "preregistered-launch-review-pending"
    assert exp702["predecessor_archive_sha256"] == config["predecessor"]["archive_sha256"]
    assert exp702["paid_provider_launch_performed"] is False
    assert exp702["terminal_policy"]["all_valid_result_pairs_publish"] is True
    assert set(exp702["bindings"]) >= {
        "runner", "config", "test", "orchestrator", "guest_supervisor",
        "windows_watchdog", "canonical_cu124_lock", "source_checkpoint",
    }
