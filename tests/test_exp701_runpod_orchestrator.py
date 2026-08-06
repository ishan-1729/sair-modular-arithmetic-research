from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile

import pytest

from experiments.cloud_benchmarks.exp701_runpod import (
    offpod_sync,
    orchestrate,
    recover_retained_volume,
    recovery_transport,
)


def arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_center_id="EU-RO-1",
        offpod_destination=tmp_path / "offpod",
        provider_receipt=tmp_path / "provider.json",
    )


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}

    def upload_file(self, source, bucket, key, ExtraArgs=None) -> None:
        self.objects[(bucket, key)] = (
            Path(source).read_bytes(),
            dict((ExtraArgs or {}).get("Metadata") or {}),
        )

    def head_object(self, *, Bucket, Key):
        data, metadata = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "Metadata": metadata, "ETag": '"etag"'}

    def download_file(self, bucket, key, destination) -> None:
        Path(destination).write_bytes(self.objects[(bucket, key)][0])

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)][0])}

    def put_object(self, *, Bucket, Key, Body, **_values) -> None:
        self.objects[(Bucket, Key)] = (bytes(Body), {})


def test_controller_command_is_direct_bounded_and_preload_gated(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    ready = tmp_path / "ready.json"
    guest = orchestrate.direct_start_command("a" * 64, "b" * 40, "c" * 48, None)
    command = orchestrate.controller_command(
        args,
        guest_command=guest,
        readiness_receipt=ready,
        source_sha256="a" * 64,
    )
    rendered = " ".join(command)
    assert "--max-total-usd 8.27" in rendered
    assert "--max-provider-seconds 32400" in rendered
    assert "--network-volume-size-gb 50" in rendered
    assert "--network-volume-data-center-id EU-RO-1" in rendered
    assert f"--network-volume-ready-receipt {ready}" in rendered
    assert "--network-volume-ready-source-sha256 " + "a" * 64 in rendered
    assert "guest_supervisor.sh" in rendered
    assert "--template-id" not in command
    assert "--use-image-command" not in command
    assert "--port" not in command
    assert "PUBLIC_KEY" not in rendered
    assert "RUNPOD_API_KEY" not in rendered
    assert orchestrate.APPROVAL not in rendered


def test_direct_guest_runs_in_foreground_and_resume_restores_same_output() -> None:
    fresh = orchestrate.direct_start_command("a" * 64, "b" * 40, "c" * 48, None)
    assert "nohup" not in fresh and "guest.pid" not in fresh
    assert "bash experiments/cloud_benchmarks/exp701_runpod/guest_supervisor.sh" in fresh
    assert "SAIR_EXP701_SYNC_NONCE=" + "c" * 48 in fresh
    assert "SAIR_EXP701_SLICE_SECONDS=30600" in fresh
    assert "guest-exit" in fresh
    resume = orchestrate.direct_start_command(
        "a" * 64,
        "b" * 40,
        "c" * 48,
        {"archive_sha256": "d" * 64},
    )
    assert "recovery_transport restore" in resume
    assert f".scratch/{orchestrate.RUN_ID}" in resume
    assert orchestrate.RESUME_KEY in resume


def test_s3_upload_and_offpod_sync_are_sha_bound_and_acknowledged(tmp_path: Path) -> None:
    client = FakeS3()
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"authenticated source")
    uploaded = offpod_sync.upload_file_verified(
        client, source, bucket="volume-1", key=orchestrate.SOURCE_KEY
    )
    assert uploaded["status"] == "verified"
    assert uploaded["sha256"] == recovery_transport.sha256(source)
    assert uploaded["verification"] == "authenticated-download-sha256"

    output = tmp_path / "experiment" / orchestrate.RUN_ID
    output.mkdir(parents=True)
    (output / "guest-terminal.json").write_text(
        json.dumps({"status": "completed", "resume_required": False}),
        encoding="utf-8",
    )
    remote = tmp_path / "remote" / "sair-exp701-checkpoint-export.tar.gz"
    receipt = recovery_transport.create_export(output, remote)
    for suffix in ("", ".sha256", ".receipt.json"):
        path = Path(str(remote) + suffix)
        key = orchestrate.REMOTE_ARCHIVE_KEY + suffix
        client.objects[("volume-1", key)] = (path.read_bytes(), {})
    result = offpod_sync.sync_s3(
        client,
        "volume-1",
        orchestrate.REMOTE_ARCHIVE_KEY,
        tmp_path / "offpod",
        remote_ack=orchestrate.REMOTE_ACK_KEY,
        nonce="n" * 48,
        endpoint_url="https://s3api-eu-ro-1.runpod.io/",
    )
    assert result["transport"] == "runpod-s3"
    assert result["archive_sha256"] == receipt["archive_sha256"]
    ack = json.loads(client.objects[("volume-1", orchestrate.REMOTE_ACK_KEY)][0])
    assert ack["nonce"] == "n" * 48
    assert ack["archive_sha256"] == receipt["archive_sha256"]
    assert orchestrate.terminal_from_export(result)["status"] == "completed"


def test_s3_endpoint_is_allowlisted() -> None:
    assert orchestrate.s3_endpoint("EU-RO-1") == "https://s3api-eu-ro-1.runpod.io/"
    assert offpod_sync.runpod_s3_region(
        "https://s3api-eu-ro-1.runpod.io/"
    ) == "eu-ro-1"
    assert offpod_sync.runpod_s3_region(
        "https://s3api-eur-is-1.runpod.io/"
    ) == "eur-is-1"
    with pytest.raises(ValueError, match="not configured"):
        orchestrate.s3_endpoint("US-UNKNOWN-1")
    with pytest.raises(ValueError, match="hostname"):
        offpod_sync.runpod_s3_region("https://s3.amazonaws.com/")


def test_retained_volume_recovery_is_short_bounded_and_never_starts_science(
    tmp_path: Path,
) -> None:
    private = tmp_path / "id"
    public = private.with_suffix(".pub")
    private.write_text("private", encoding="utf-8")
    public.write_text("ssh-ed25519 public", encoding="utf-8")
    args = argparse.Namespace(
        volume_id="volume-1",
        provider_receipt=tmp_path / "provider.json",
    )
    command = recover_retained_volume.recovery_controller_command(args, public)
    rendered = " ".join(command)
    assert "--network-volume-id volume-1" in rendered
    assert "--max-total-usd 0.50" in rendered
    assert "--max-provider-seconds 1800" in rendered
    assert "--label sair-exp701-recovery-export" in rendered
    assert "exp_701_cuda_resumable" not in rendered
    assert "RUNPOD_API_KEY" not in rendered
    assert orchestrate.APPROVAL not in rendered


def test_retained_volume_requires_pod_delete_before_volume_cleanup() -> None:
    assert recover_retained_volume.pod_deletion_confirmed(
        {"status": "resource-destroyed", "events": [{"event": "delete-confirmed"}]}
    )
    assert not recover_retained_volume.pod_deletion_confirmed(
        {"status": "controller-started", "events": [{"event": "delete-requested"}]}
    )


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
        orchestrate, "source_identity", lambda _archive: {"repository_commit": "e" * 40}
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
    assert "bash environments/cuda/bootstrap.sh" in guest
    assert "bash environments/cuda/run-acceptance.sh" in guest
    assert 'tee -a "$output/guest-supervisor.log"' in guest


def test_exported_guest_failure_is_visible_while_provider_keeps_running(
    tmp_path: Path,
) -> None:
    output = tmp_path / "experiment" / orchestrate.RUN_ID
    output.mkdir(parents=True)
    (output / "guest-terminal.json").write_text(
        json.dumps({"status": "failed", "resume_required": False, "exit_code": 1}),
        encoding="utf-8",
    )
    archive = tmp_path / "sair-exp701-checkpoint-export.tar.gz"
    recovery_transport.create_export(output, archive)
    receipt_path = tmp_path / "offpod-sync-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "verified",
                "archive": str(archive),
                "archive_sha256": recovery_transport.sha256(archive),
                "restore_verification": {"status": "verified"},
            }
        ),
        encoding="utf-8",
    )

    class RunningController:
        def poll(self) -> None:
            return None

    status, terminal = orchestrate.wait_for_controller_or_guest_terminal(
        RunningController(), receipt_path, timeout=1
    )
    assert status is None
    assert terminal == {"status": "failed", "resume_required": False, "exit_code": 1}


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
    assert orchestrate.main(
        [
            "--data-center-id",
            "EU-RO-1",
            "--source",
            str(archive),
            "--preflight",
            str(preflight),
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
    assert result["transport"] == "runpod-s3-direct-start"
    assert result["maximum_total_usd"] == 8.27
    assert result["maximum_provider_seconds"] == 32400
    assert result["required_authorization"] == orchestrate.APPROVAL
    assert not (tmp_path / "session").exists()
