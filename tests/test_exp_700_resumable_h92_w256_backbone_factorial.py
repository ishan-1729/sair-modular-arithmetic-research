from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import yaml

from experiments.exp_699_h92_w256_backbone_factorial_complete import run as exp699
from experiments.exp_700_resumable_h92_w256_backbone_factorial import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_700_resumable_h92_w256_backbone_factorial.yaml").read_text())
OLD = yaml.safe_load((ROOT / "configs/exp_699_h92_w256_backbone_factorial_complete.yaml").read_text())


def test_fresh_identity_roots_and_unchanged_scientific_contract() -> None:
    assert CONFIG["run_id"] == "exp-700-resumable-h92-w256-backbone-factorial"
    assert CONFIG["roots"] == {"calibration": 20261404, "formal": 20261405, "confirmations": [20261406, 20261407], "independent": [20261408, 20261409, 20261410, 20261411, 20261412]}
    for key in ("model", "arms", "data", "training", "evaluation", "gates", "resource_policy"):
        assert CONFIG[key] == OLD[key]
    assert run.sha256(Path(CONFIG["source_checkpoint"])) == CONFIG["source_checkpoint_sha256"]


def test_primary_route_functions_are_exact_exp699_aliases() -> None:
    for name in ("new_model", "balanced_width", "make_scheduler", "rollout", "interval", "evaluate", "paired_evidence", "evaluate_with_evidence", "lower_tier_parity"):
        assert getattr(run, name) is getattr(exp699, name)
    assert run.base.train_update is exp699.base.train_update
    assert run.base.tensor_native_rollout is exp699.base.tensor_native_rollout


def test_recovery_snapshot_contains_every_required_state() -> None:
    source = inspect.getsource(run.train_arm_resumable)
    for token in ("model", "optimizer", "scheduler", "python_rng", "numpy_rng", "torch_rng", "sampler_rng", "start_rng", "curves", "width_order", "data_order_digests", "work"):
        assert f'"{token}"' in source
    assert CONFIG["recovery"]["snapshot_every_updates"] <= 100
    assert "os.replace" in inspect.getsource(run.atomic_bytes)
    assert "os.fsync" in inspect.getsource(run.atomic_bytes)


def test_resume_hash_gates_root_snapshot_phase_and_work() -> None:
    root_resume = inspect.getsource(run.Experiment.open_or_resume_root)
    snapshot = inspect.getsource(run.load_snapshot)
    phase = inspect.getsource(run.phase_result)
    assert "resume root/source identity mismatch" in root_resume
    assert "resume abnormal-exit marker mismatch" in root_resume
    assert "snapshot hash mismatch" in snapshot and "logical ledger mismatch" in snapshot
    assert "completed phase integrity mismatch" in phase
    assert "fresh invocation would duplicate an opened root" in root_resume


def test_interrupted_and_uninterrupted_training_eval_diagnostic_are_exact(tmp_path: Path) -> None:
    result = run.smoke(run.CONFIG_PATH, tmp_path)
    assert result["status"] == "smoke-passed"
    assert result["train_eval_diagnostic_recovery"] is True
    assert result["uninterrupted_signature"] == result["interrupted_resumed_signature"]
    assert set(result["all_modes_exercised"]) == {"calibration", "formal", "confirmations", "independent"}


def test_resume_does_not_write_second_root_marker(tmp_path: Path) -> None:
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "formal", 20261405)
    run.atomic_json(tmp_path / "preflight-root-audit.json", {"fresh": True})
    root_dir = experiment.open_or_resume_root(); marker = root_dir / "root-opened.json"; before = marker.read_bytes()
    lifecycle = json.loads((root_dir / "run-active.json").read_text()); run.atomic_json(root_dir / "run-active.json", {**lifecycle, "status": "recoverable-interruption"})
    resumed = run.Experiment(run.CONFIG_PATH, tmp_path, "formal", 20261405, resume=True)
    assert resumed.open_or_resume_root() == root_dir
    assert marker.read_bytes() == before
    assert json.loads((root_dir / "run-active.json").read_text())["recoveries"] == 1


def test_semantic_identity_failure_is_terminal_not_recoverable(tmp_path: Path) -> None:
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "formal", 20261405)
    run.atomic_json(tmp_path / "preflight-root-audit.json", {"fresh": True})
    root_dir = experiment.open_or_resume_root(); active = json.loads((root_dir / "run-active.json").read_text())
    run.atomic_json(root_dir / "run-active.json", {**active, "identity_sha256": "0" * 64, "status": "recoverable-interruption"})
    with pytest.raises(run.SemanticFailure, match="abnormal-exit marker mismatch"):
        run.Experiment(run.CONFIG_PATH, tmp_path, "formal", 20261405, resume=True).open_or_resume_root()
    assert not isinstance(run.SemanticFailure("x"), run.RecoverableInterruption)


def test_all_denominators_conditional_roots_and_runtime_policy_remain_frozen() -> None:
    assert CONFIG["training"]["updates"] == 4800 and CONFIG["training"]["batch_size"] == 16 and CONFIG["training"]["horizon"] == 8
    evaluation = CONFIG["evaluation"]
    assert (evaluation["transition_cases_per_arm_width"], evaluation["w128_rollout_cases_per_arm"], evaluation["w256_rollout_cases_per_arm"]) == (3000, 300, 600)
    source = inspect.getsource(run.Experiment.science)
    assert "for arm in ARM_ORDER" in source and "for width in WIDTHS" in source
    assert "candidate_runtime" not in source
    predecessor = inspect.getsource(run.Experiment.require_predecessor_gate)
    assert "both passing confirmations" in predecessor


def test_service_contract_is_bounded_persistent_and_training_only() -> None:
    assert CONFIG["service"]["start_limit_burst"] == 3
    assert CONFIG["service"]["runtime_hard_multiplier"] == 1.75
    assert CONFIG["recovery"]["maximum_recoverable_restarts"] == 3
    assert "submission/" in CONFIG["library_decision"]
    audit = inspect.getsource(run.Experiment.audit_fresh_roots)
    assert "systemctl" in audit and 'ROOT.parent.name == ".scratch"' in audit


def test_service_proof_distinguishes_recoverable_and_terminal(tmp_path: Path) -> None:
    first, first_status = run.service_proof(tmp_path, "recoverable")
    second, second_status = run.service_proof(tmp_path, "recoverable")
    terminal, terminal_status = run.service_proof(tmp_path, "terminal")
    assert (first["status"], first_status) == ("recoverable-interruption", 75)
    assert (second["status"], second_status, second["attempts"]) == ("completed-after-restart", 0, 2)
    assert (terminal["status"], terminal_status, terminal["attempts"]) == ("terminal-failure", 0, 1)


def test_deployed_inference_remains_hard_learned_residue_only() -> None:
    source = inspect.getsource(run.rollout).lower()
    forbidden = ("barrett", "montgomery", "crt", "repair", "expected", "quotient", "carry", "token", "adapter", "routing")
    assert not any(token in source for token in forbidden)
    assert "diagnostic" not in source and "lower_tier_parity" not in source
