from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from experiments.exp_699_h92_w256_backbone_factorial_complete import run as exp699
from experiments.exp_700_resumable_h92_w256_backbone_factorial import run as exp700
from experiments.cloud_benchmarks.exp699_cuda_replay import run as cuda_replay
from experiments.cloud_benchmarks.exp701_runpod import offpod_sync, recovery_transport
from experiments.exp_701_cuda_resumable_h92_w256_backbone_factorial import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/exp_701_cuda_resumable_h92_w256_backbone_factorial.yaml").read_text())
OLD = yaml.safe_load((ROOT / "configs/exp_700_resumable_h92_w256_backbone_factorial.yaml").read_text())


def test_fresh_identity_roots_and_unchanged_scientific_contract() -> None:
    assert CONFIG["run_id"] == "exp-701-cuda-resumable-h92-w256-backbone-factorial"
    assert CONFIG["roots"] == {"calibration": 20261413, "formal": 20261414, "confirmations": [20261415, 20261416], "independent": [20261417, 20261418, 20261419, 20261420, 20261421]}
    for key in ("model", "arms", "data", "training", "evaluation", "gates", "resource_policy"):
        assert CONFIG[key] == OLD[key]
    assert run.sha256(ROOT / CONFIG["source_checkpoint"]) == CONFIG["source_checkpoint_sha256"]


def test_primary_route_is_exp700_science_with_cuda_only_execution_delta() -> None:
    for key in ("model", "arms", "data", "training", "evaluation", "gates", "resource_policy"):
        assert CONFIG[key] == OLD[key]
    assert run.rollout is cuda_replay.rollout
    assert run.evaluate_with_evidence is cuda_replay.evaluate_with_evidence
    assert run.balanced_width is exp699.balanced_width
    assert run.make_scheduler is exp699.make_scheduler
    assert CONFIG["gpu_execution"]["fidelity_reduced"] is False
    assert CONFIG["gpu_execution"]["accepted_gate_run"] == "cuda-hardware-acceptance-runpod-20260805g"


def test_recovery_snapshot_contains_every_required_state() -> None:
    source = inspect.getsource(run.train_arm_resumable)
    for token in ("model", "optimizer", "scheduler", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "sampler_rng", "start_rng", "curves", "width_order", "data_order_digests", "work"):
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
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "formal", 20261414, device="cpu")
    run.atomic_json(tmp_path / "preflight-root-audit.json", {"fresh": True})
    root_dir = experiment.open_or_resume_root(); marker = root_dir / "root-opened.json"; before = marker.read_bytes()
    lifecycle = json.loads((root_dir / "run-active.json").read_text()); run.atomic_json(root_dir / "run-active.json", {**lifecycle, "status": "recoverable-interruption"})
    resumed = run.Experiment(run.CONFIG_PATH, tmp_path, "formal", 20261414, resume=True, device="cpu")
    assert resumed.open_or_resume_root() == root_dir
    assert marker.read_bytes() == before
    assert json.loads((root_dir / "run-active.json").read_text())["recoveries"] == 1


def test_semantic_identity_failure_is_terminal_not_recoverable(tmp_path: Path) -> None:
    experiment = run.Experiment(run.CONFIG_PATH, tmp_path, "formal", 20261414, device="cpu")
    run.atomic_json(tmp_path / "preflight-root-audit.json", {"fresh": True})
    root_dir = experiment.open_or_resume_root(); active = json.loads((root_dir / "run-active.json").read_text())
    run.atomic_json(root_dir / "run-active.json", {**active, "identity_sha256": "0" * 64, "status": "recoverable-interruption"})
    with pytest.raises(run.SemanticFailure, match="abnormal-exit marker mismatch"):
        run.Experiment(run.CONFIG_PATH, tmp_path, "formal", 20261414, resume=True, device="cpu").open_or_resume_root()
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
    assert CONFIG["gpu_execution"]["authorization_max_total_cost_usd"] == 9.0
    assert CONFIG["gpu_execution"]["authorization_max_lifetime_hours"] == 10
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
    assert "state_bits = logits.ge(0)" in source
    assert "prediction = source.base.decode_boolean(state_bits.cpu())" in source
    assert "return (" in source
    assert "barrett" not in source and "montgomery" not in source and "crt" not in source


def test_runpod_slice_is_admitted_resumable_and_not_a_scientific_stop() -> None:
    card = yaml.safe_load((ROOT / "experiments/cloud_benchmarks/exp701_runpod/decision-card.yaml").read_text())
    provider = yaml.safe_load((ROOT / "configs/exp_701_runpod_control.yaml").read_text())
    assert card["resource_preflight"]["exact_work"]["training_updates"] == 4 * 4800
    assert card["provider_slice"]["lifecycle_cost_upper_usd"] == 7.50795
    assert card["provider_slice"]["scientific_stop"] is False
    assert provider["authorization"]["required_value"] == card["required_authorization"]["exact_value"]
    assert "MAX_TOTAL_USD_9.00" in provider["authorization"]["required_value"]
    assert "MAX_PROVIDER_HOURS_10" in provider["authorization"]["required_value"]


def test_guest_package_is_shell_valid_and_never_imports_cpu_partial_work() -> None:
    guest = ROOT / "experiments/cloud_benchmarks/exp701_runpod/guest_supervisor.sh"
    builder = ROOT / "experiments/cloud_benchmarks/exp701_runpod/prepare_source.sh"
    host_sync = ROOT / "experiments/cloud_benchmarks/exp701_runpod/host_sync.sh"
    subprocess.run(["bash", "-n", guest], check=True)
    subprocess.run(["bash", "-n", builder], check=True)
    subprocess.run(["bash", "-n", host_sync], check=True)
    source = guest.read_text()
    assert "20261404" not in source and "20261405" not in source
    for root in range(20261413, 20261422):
        assert str(root) in source
    assert "--resume" in source and "checkpoint-export" in source
    assert "/workspace/exp701-durable" in source
    assert "wait_for_current_ack" in source and "ack_valid no" in source


def test_external_copy_survives_pod_delete_and_resumes_same_root_exactly(
    tmp_path: Path,
) -> None:
    pod_mount = tmp_path / "disposable-pod-network-mount"
    output = (
        pod_mount
        / "experiment"
        / "exp-701-cuda-resumable-h92-w256-backbone-factorial"
    )
    experiment = run.Experiment(
        run.CONFIG_PATH, output, "formal", 20261414, device="cpu"
    )
    run.atomic_json(output / "preflight-root-audit.json", {"fresh": True})
    root_dir = experiment.open_or_resume_root()
    marker_before = (root_dir / "root-opened.json").read_bytes()
    identity = experiment.identity_sha256()
    splits = experiment.splits()

    _, reference_record = run.train_arm_resumable(
        experiment.config,
        experiment.exp300,
        experiment.checkpoint,
        splits,
        20261414,
        run.ARM_ORDER[0],
        3,
        tmp_path / "uninterrupted-reference",
        identity,
        False,
        snapshot_every=1,
    )
    with pytest.raises(run.RecoverableInterruption):
        run.train_arm_resumable(
            experiment.config,
            experiment.exp300,
            experiment.checkpoint,
            splits,
            20261414,
            run.ARM_ORDER[0],
            3,
            root_dir,
            identity,
            False,
            interrupt_after_update=2,
            snapshot_every=1,
        )
    experiment.mark_lifecycle(
        "recoverable-interruption", reason="simulated-provider-stop"
    )

    network_archive = pod_mount / "durable" / "exp701-recovery.tar.gz"
    export_receipt = recovery_transport.create_export(output, network_archive)
    offpod_directory = tmp_path / "external-host-copy"
    sync_receipt = offpod_sync.sync_local(network_archive, offpod_directory)
    assert sync_receipt["archive_sha256"] == export_receipt["archive_sha256"]

    shutil.rmtree(pod_mount)
    assert not pod_mount.exists()
    host_archive = offpod_directory / network_archive.name
    restored = recovery_transport.restore_export(
        host_archive,
        pod_mount,
        expected_sha256=sync_receipt["archive_sha256"],
    )
    restored_output = Path(restored["restored_output"])
    assert restored_output == output
    resumed = run.Experiment(
        run.CONFIG_PATH,
        restored_output,
        "formal",
        20261414,
        resume=True,
        device="cpu",
    )
    restored_root = resumed.open_or_resume_root()
    assert (restored_root / "root-opened.json").read_bytes() == marker_before
    _, resumed_record = run.train_arm_resumable(
        resumed.config,
        resumed.exp300,
        resumed.checkpoint,
        resumed.splits(),
        20261414,
        run.ARM_ORDER[0],
        3,
        restored_root,
        resumed.identity_sha256(),
        True,
        snapshot_every=1,
    )
    assert run.recovery_signature(resumed_record) == run.recovery_signature(
        reference_record
    )

    corrupted = tmp_path / "corrupted-export.tar.gz"
    shutil.copy2(host_archive, corrupted)
    with corrupted.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        recovery_transport.restore_export(
            corrupted,
            tmp_path / "rejected-restore",
            expected_sha256=sync_receipt["archive_sha256"],
        )
