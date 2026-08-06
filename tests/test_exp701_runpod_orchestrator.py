from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile

import pytest

from experiments.cloud_benchmarks.exp701_runpod import orchestrate


def arguments(tmp_path: Path) -> argparse.Namespace:
    private = tmp_path / "id_ed25519"
    public = tmp_path / "id_ed25519.pub"
    private.write_text("private", encoding="utf-8")
    public.write_text("ssh-ed25519 public", encoding="utf-8")
    return argparse.Namespace(
        data_center_id="EU-RO-1",
        offpod_destination=tmp_path / "offpod",
        provider_receipt=tmp_path / "provider.json",
    )


def test_controller_command_is_exact_bounded_and_persistent(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    public = tmp_path / "id_ed25519.pub"
    command = orchestrate.controller_command(args, public)
    rendered = " ".join(command)
    assert "--max-total-usd 9.00" in rendered
    assert "--max-provider-seconds 36000" in rendered
    assert "--network-volume-size-gb 50" in rendered
    assert "--network-volume-data-center-id EU-RO-1" in rendered
    assert "--label sair-exp701-full-fidelity" in rendered
    assert "--offpod-receipt" in command
    assert "--use-image-command" in command
    assert f"PUBLIC_KEY={public}" in command
    assert "RUNPOD_API_KEY" not in rendered
    assert orchestrate.APPROVAL not in rendered


def test_runpod_endpoint_requires_public_ip_and_ssh_mapping() -> None:
    assert orchestrate.ssh_endpoint(None) is None
    assert orchestrate.ssh_endpoint({"publicIp": "1.2.3.4"}) is None
    assert orchestrate.ssh_endpoint(
        {"publicIp": "1.2.3.4", "portMappings": {"22": 23456}}
    ) == ("1.2.3.4", 23456)
    with pytest.raises(RuntimeError, match="invalid SSH port"):
        orchestrate.ssh_endpoint(
            {"publicIp": "1.2.3.4", "portMappings": {"22": 70000}}
        )


def test_ssh_transport_is_strict_and_identity_scoped(tmp_path: Path) -> None:
    identity = tmp_path / "id"
    known = tmp_path / "known_hosts"
    options = orchestrate.ssh_options(identity, known, "1.2.3.4", 2222)
    rendered = " ".join(options)
    assert "StrictHostKeyChecking=yes" in rendered
    assert f"UserKnownHostsFile={known}" in rendered
    assert "IdentitiesOnly=yes" in rendered
    assert str(identity) in options


def test_remote_guest_is_detached_and_resume_restores_same_output() -> None:
    fresh = orchestrate.remote_start_command("a" * 64, "b" * 40, "c" * 48, None)
    assert "nohup bash -lc" in fresh
    assert "guest-exit" in fresh and "guest.pid" in fresh
    assert "SAIR_EXP701_SYNC_NONCE=" + "c" * 48 in fresh
    resume = orchestrate.remote_start_command(
        "a" * 64,
        "b" * 40,
        "c" * 48,
        {"archive_sha256": "d" * 64},
    )
    assert "recovery_transport restore" in resume
    assert f".scratch/{orchestrate.RUN_ID}" in resume
    assert "sair-exp701-resume.tar.gz" in resume


def test_preflight_binds_archive_commit_and_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"archive")
    receipt = tmp_path / "preflight.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "exp701-runpod-preflight-receipt-v1",
                "status": "passed-awaiting-paid-authorization",
                "required_authorization": orchestrate.APPROVAL,
                "source_archive": {
                    "sha256": orchestrate.sha256(archive),
                    "bytes": archive.stat().st_size,
                    "repository_commit": "e" * 40,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        orchestrate,
        "source_identity",
        lambda _archive: {"repository_commit": "e" * 40},
    )
    assert orchestrate.verify_preflight(receipt, archive)["source_archive"][
        "repository_commit"
    ] == "e" * 40
    archive.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="no longer matches"):
        orchestrate.verify_preflight(receipt, archive)


def test_guest_establishes_external_baseline_before_cuda_bootstrap() -> None:
    guest = (
        orchestrate.ROOT
        / "experiments/cloud_benchmarks/exp701_runpod/guest_supervisor.sh"
    ).read_text(encoding="utf-8")
    assert "export SAIR_SOURCE_DIRTY=0" in guest
    assert guest.index("export_once") < guest.index("environments/cuda/bootstrap.sh")
    assert guest.index("wait_for_current_ack") < guest.index(
        "environments/cuda/bootstrap.sh"
    )
    assert "offpod-pre-science-sync-failed" in guest


def test_complete_dry_run_is_nonpaid_and_receipt_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "source.tar.gz"
    identity = json.dumps({"repository_commit": "f" * 40}).encode()
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("./.sair-source.json")
        member.size = len(identity)
        bundle.addfile(member, io.BytesIO(identity))
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "exp701-runpod-preflight-receipt-v1",
                "status": "passed-awaiting-paid-authorization",
                "required_authorization": orchestrate.APPROVAL,
                "source_archive": {
                    "sha256": orchestrate.sha256(archive),
                    "bytes": archive.stat().st_size,
                    "repository_commit": "f" * 40,
                },
            }
        ),
        encoding="utf-8",
    )
    private = tmp_path / "id_ed25519"
    private.write_text("private", encoding="utf-8")
    private.with_suffix(".pub").write_text("ssh-ed25519 public", encoding="utf-8")
    assert orchestrate.main(
        [
            "--data-center-id",
            "EU-RO-1",
            "--source",
            str(archive),
            "--preflight",
            str(preflight),
            "--ssh-private-key",
            str(private),
            "--session",
            str(tmp_path / "session"),
            "--offpod-destination",
            str(tmp_path / "offpod"),
            "--provider-receipt",
            str(tmp_path / "provider.json"),
            "--orchestrator-receipt",
            str(tmp_path / "orchestrator.json"),
            "--dry-run",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry-run"
    assert result["maximum_total_usd"] == 9.0
    assert result["maximum_provider_seconds"] == 36000
    assert result["required_authorization"] == orchestrate.APPROVAL
    assert not (tmp_path / "session").exists()
