"""Atomically copy and verify EXP701 recovery exports outside a disposable Pod."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .recovery_transport import atomic_json, restore_export, sha256


def _declared_sha(path: Path) -> str:
    value = path.read_text(encoding="utf-8").split()[0]
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("invalid recovery checksum sidecar")
    return value


def verify_triplet(archive: Path, checksum: Path, receipt: Path) -> dict[str, Any]:
    declared = _declared_sha(checksum)
    actual = sha256(archive)
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    if actual != declared or metadata.get("archive_sha256") != actual:
        raise RuntimeError("recovery export triplet failed SHA-256 verification")
    if metadata.get("status") != "verified":
        raise RuntimeError("recovery export receipt is not verified")
    return {"archive_sha256": actual, "archive_bytes": archive.stat().st_size}


def verify_restorable(archive: Path, digest: str, destination: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(dir=destination) as temporary_raw:
        restored = restore_export(
            archive, Path(temporary_raw), expected_sha256=digest
        )
    tree = restored["tree_verification"]
    return {
        "status": restored["status"],
        "files": tree["files"],
        "bytes": tree["bytes"],
        "root_markers": tree["root_markers"],
        "snapshots": len(tree["snapshots"]),
        "completed_phases": len(tree["completed_phases"]),
    }


def sync_local(remote_archive: Path, destination: Path) -> dict[str, Any]:
    remote_archive = remote_archive.resolve()
    remote_checksum = remote_archive.with_suffix(remote_archive.suffix + ".sha256")
    remote_receipt = remote_archive.with_suffix(remote_archive.suffix + ".receipt.json")
    verified = verify_triplet(remote_archive, remote_checksum, remote_receipt)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temporary_raw:
        temporary = Path(temporary_raw)
        copied_archive = temporary / remote_archive.name
        copied_checksum = temporary / remote_checksum.name
        copied_receipt = temporary / remote_receipt.name
        shutil.copy2(remote_archive, copied_archive)
        shutil.copy2(remote_checksum, copied_checksum)
        shutil.copy2(remote_receipt, copied_receipt)
        verify_triplet(copied_archive, copied_checksum, copied_receipt)
        for source in (copied_archive, copied_checksum, copied_receipt):
            os.replace(source, destination / source.name)
    result = {
        "schema_version": "exp701-offpod-sync-v1",
        "status": "verified",
        "transport": "local-drill",
        "destination": str(destination.resolve()),
        "archive": str((destination / remote_archive.name).resolve()),
        **verified,
        "restore_verification": verify_restorable(
            destination / remote_archive.name,
            verified["archive_sha256"],
            destination,
        ),
    }
    atomic_json(destination / "offpod-sync-receipt.json", result)
    return result


def sync_ssh(
    target: str,
    remote_archive: str,
    destination: Path,
    *,
    identity: Path,
    port: int,
    remote_ack: str,
    nonce: str,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    remote_path = PurePosixPath(remote_archive)
    ack_path = PurePosixPath(remote_ack)
    if not remote_path.is_absolute() or not ack_path.is_absolute():
        raise ValueError("remote archive and acknowledgement paths must be absolute")
    if ".." in remote_path.parts or ".." in ack_path.parts:
        raise ValueError("remote paths may not contain parent traversal")
    if len(nonce) < 16:
        raise ValueError("sync nonce must contain at least 16 characters")
    names = [remote_path.name, remote_path.name + ".sha256", remote_path.name + ".receipt.json"]
    ssh_options = [
        "-i", str(identity), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
    ]
    with tempfile.TemporaryDirectory(dir=destination) as temporary_raw:
        temporary = Path(temporary_raw)
        # Pull the small sidecars first. If the guest rotates the triplet during
        # transfer, end-to-end verification fails and the monitor retries.
        for name in names[1:]:
            subprocess.run(
                [
                    "scp", "-q", "-P", str(port), *ssh_options,
                    f"{target}:{remote_path.with_name(name).as_posix()}",
                    str(temporary / name),
                ],
                check=True,
                timeout=120,
                capture_output=True,
            )
        declared = _declared_sha(temporary / names[1])
        retained = destination / names[0]
        if retained.is_file() and sha256(retained) == declared:
            shutil.copy2(retained, temporary / names[0])
        else:
            subprocess.run(
                [
                    "scp", "-q", "-P", str(port), *ssh_options,
                    f"{target}:{remote_path.as_posix()}", str(temporary / names[0]),
                ],
                check=True,
                timeout=300,
                capture_output=True,
            )
        verified = verify_triplet(
            temporary / names[0], temporary / names[1], temporary / names[2]
        )
        for name in names:
            os.replace(temporary / name, destination / name)
    result = {
        "schema_version": "exp701-offpod-sync-v1",
        "status": "verified",
        "transport": "scp",
        "target": target,
        "remote_archive": remote_archive,
        "destination": str(destination.resolve()),
        "archive": str((destination / names[0]).resolve()),
        **verified,
        "restore_verification": verify_restorable(
            destination / names[0], verified["archive_sha256"], destination
        ),
    }
    atomic_json(destination / "offpod-sync-receipt.json", result)
    acknowledgement = {
        "schema_version": "exp701-offpod-ack-v1",
        "status": "verified-offpod-copy",
        "nonce": nonce,
        "archive_sha256": verified["archive_sha256"],
        "archive_bytes": verified["archive_bytes"],
        "verified_at_epoch": time.time(),
    }
    parent = shlex.quote(ack_path.parent.as_posix())
    final = shlex.quote(ack_path.as_posix())
    remote_command = (
        f"umask 077 && mkdir -p {parent} && "
        f"temporary=$(mktemp {parent}/.offpod-ack.XXXXXX) && "
        f"cat > \"$temporary\" && mv \"$temporary\" {final}"
    )
    subprocess.run(
        ["ssh", "-p", str(port), *ssh_options, target, remote_command],
        input=(json.dumps(acknowledgement, sort_keys=True) + "\n").encode(),
        check=True,
        timeout=120,
        capture_output=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--remote-archive", required=True)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--local", action="store_true")
    transport.add_argument("--ssh-target")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--remote-ack")
    parser.add_argument("--nonce")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-seconds", type=int, default=36000)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 30 or args.max_seconds <= 0:
        parser.error("poll/max seconds are outside the safe range")
    if args.ssh_target and (
        args.identity is None
        or not args.identity.is_file()
        or not args.remote_ack
        or not args.nonce
    ):
        parser.error("SSH sync requires identity, remote acknowledgement, and nonce")
    started = time.monotonic()
    while True:
        try:
            if args.local:
                result = sync_local(Path(args.remote_archive), args.destination)
            else:
                result = sync_ssh(
                    args.ssh_target, args.remote_archive, args.destination,
                    identity=args.identity, port=args.port,
                    remote_ack=args.remote_ack, nonce=args.nonce,
                )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            print(
                json.dumps(
                    {
                        "schema_version": "exp701-offpod-sync-event-v1",
                        "status": "retrying",
                        "error_type": type(error).__name__,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.once:
                raise
            delay = min(30, max(1, args.poll_seconds))
            if time.monotonic() - started + delay >= args.max_seconds:
                return 75
            time.sleep(delay)
            continue
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.once or time.monotonic() - started + args.poll_seconds >= args.max_seconds:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
