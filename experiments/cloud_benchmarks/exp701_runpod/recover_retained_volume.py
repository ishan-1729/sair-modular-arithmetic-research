"""Recover a verified EXP701 export from a retained RunPod network volume.

This host-only recovery attaches the retained volume to one short-lived Pod,
copies and restore-verifies the export outside RunPod, delete-confirms the Pod,
and only then delete-confirms the retained volume. It never starts science.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any
from urllib import error, request

from . import offpod_sync, orchestrate


ROOT = orchestrate.ROOT
FULL_APPROVAL = orchestrate.APPROVAL
GENERIC_CONTROLLER_APPROVAL = "I_AUTHORIZE_PAID_GPU_LAUNCH"
REMOTE_ARCHIVE = "/workspace/exp701-durable/sair-exp701-checkpoint-export.tar.gz"
REMOTE_ACK = "/workspace/exp701-durable/recovery-cleanup-ack.json"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def provider_request(token: str, method: str, path: str) -> tuple[int, Any]:
    operation = request.Request(
        "https://rest.runpod.io/v1" + path,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with request.urlopen(operation, timeout=30) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except error.HTTPError as failure:
        if failure.code == 404:
            return 404, None
        raise


def recovery_controller_command(
    args: argparse.Namespace, public_key: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "environments.providers.controller",
        "--config",
        str(ROOT / "configs/cuda_provider_control.yaml"),
        "run",
        "--provider",
        "runpod",
        "--resource-selector",
        "NVIDIA GeForce RTX 4090",
        "--image",
        "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--label",
        "sair-exp701-recovery-export",
        "--hourly-rate-usd",
        "0.74",
        "--storage-hourly-usd",
        "0.010795",
        "--max-total-usd",
        "0.50",
        "--max-provider-seconds",
        "1800",
        "--cuda-runtime",
        "any",
        "--disk-gb",
        "50",
        "--template-id",
        "runpod-torch-v280",
        "--use-image-command",
        "--port",
        "22/tcp",
        "--environment-value-file",
        f"PUBLIC_KEY={public_key}",
        "--command",
        "true",
        "--network-volume-id",
        args.volume_id,
        "--network-volume-size-gb",
        "50",
        "--network-volume-mount-path",
        "/workspace",
        "--receipt",
        str(args.provider_receipt),
    ]


def pod_deletion_confirmed(receipt: dict[str, Any]) -> bool:
    return receipt.get("status") == "resource-destroyed" and any(
        event.get("event") == "delete-confirmed"
        for event in receipt.get("events", [])
    )


def delete_volume(token: str, volume_id: str, timeout: int = 180) -> dict[str, Any]:
    delete_status, _ = provider_request(
        token, "DELETE", f"/networkvolumes/{volume_id}"
    )
    deadline = time.monotonic() + timeout
    probes = 0
    while True:
        probes += 1
        status, _ = provider_request(token, "GET", f"/networkvolumes/{volume_id}")
        if status == 404:
            return {
                "status": "delete-confirmed",
                "delete_http_status": delete_status,
                "absence_http_status": status,
                "probes": probes,
            }
        if time.monotonic() >= deadline:
            raise TimeoutError("retained network-volume deletion was not confirmed")
        time.sleep(5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--ssh-private-key", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--provider-receipt", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.ssh_private_key = args.ssh_private_key.resolve()
    args.session = args.session.resolve()
    args.destination = args.destination.resolve()
    args.provider_receipt = args.provider_receipt.resolve()
    args.recovery_receipt = args.recovery_receipt.resolve()
    public_key = args.ssh_private_key.with_suffix(args.ssh_private_key.suffix + ".pub")
    if os.environ.get("SAIR_CLOUD_LAUNCH_APPROVED") != FULL_APPROVAL:
        raise RuntimeError("exact EXP701 paid-recovery authorization is absent")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise RuntimeError("RUNPOD_API_KEY is absent")
    if not args.ssh_private_key.is_file() or not public_key.is_file():
        raise RuntimeError("recovery SSH keypair is missing")
    if any(path.exists() for path in (args.session, args.provider_receipt, args.recovery_receipt)):
        raise RuntimeError("recovery session or receipt already exists")
    if args.destination.exists() and any(args.destination.iterdir()):
        raise RuntimeError("recovery destination must be fresh")
    pods = orchestrate.api_get(token, "/pods")
    volumes = orchestrate.api_get(token, "/networkvolumes")
    if not isinstance(pods, list) or pods:
        raise RuntimeError("RunPod Pod inventory is not empty")
    if not isinstance(volumes, list) or [str(v.get("id")) for v in volumes] != [args.volume_id]:
        raise RuntimeError("retained network-volume inventory is not exact")

    args.session.mkdir(parents=True, exist_ok=False)
    args.destination.mkdir(parents=True, exist_ok=True)
    known_hosts = args.session / "known_hosts"
    controller_log = args.session / "controller.log"
    command = recovery_controller_command(args, public_key)
    child_environment = {**os.environ, "SAIR_CLOUD_LAUNCH_APPROVED": GENERIC_CONTROLLER_APPROVAL}
    controller_handle = controller_log.open("w", encoding="utf-8")
    controller: subprocess.Popen[Any] | None = None
    resource_id: str | None = None
    sync_result: dict[str, Any] | None = None
    failure: str | None = None
    controller_status: int | None = None
    volume_cleanup: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        controller = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_environment,
            stdout=controller_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        resource_id = orchestrate.wait_for_resource(args.provider_receipt, controller)
        host, port = orchestrate.wait_for_ssh(
            token, resource_id, controller, args.ssh_private_key, known_hosts
        )
        sync_result = offpod_sync.sync_ssh(
            f"root@{host}",
            REMOTE_ARCHIVE,
            args.destination,
            identity=args.ssh_private_key,
            known_hosts=known_hosts,
            port=port,
            remote_ack=REMOTE_ACK,
            nonce=secrets.token_hex(24),
        )
        if (
            sync_result.get("status") != "verified"
            or (sync_result.get("restore_verification") or {}).get("status")
            != "verified"
        ):
            raise RuntimeError("retained export did not restore-verify")
    except Exception as caught:
        failure = f"{type(caught).__name__}: {caught}"
    finally:
        try:
            controller_status = orchestrate.terminate(controller, 300)
        except Exception as cleanup_error:
            failure = (f"{failure}; " if failure else "") + (
                f"controller cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        controller_handle.close()
        provider = read_json(args.provider_receipt) or {}
        if sync_result is not None and pod_deletion_confirmed(provider):
            try:
                volume_cleanup = delete_volume(token, args.volume_id)
            except Exception as cleanup_error:
                failure = (f"{failure}; " if failure else "") + (
                    f"volume cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        result = {
            "schema_version": "exp701-retained-volume-recovery-v1",
            "status": (
                "recovered-and-cleaned"
                if failure is None
                and sync_result is not None
                and pod_deletion_confirmed(provider)
                and (volume_cleanup or {}).get("status") == "delete-confirmed"
                else "failed-volume-retained"
            ),
            "retained_volume_id": args.volume_id,
            "recovery_resource_id": resource_id,
            "provider_controller_status": controller_status,
            "provider_pod_delete_confirmed": pod_deletion_confirmed(provider),
            "offpod_sync": sync_result,
            "network_volume_cleanup": volume_cleanup,
            "failure": failure,
            "elapsed_seconds": time.monotonic() - started,
        }
        orchestrate.atomic_json(args.recovery_receipt, result)
    if result["status"] != "recovered-and-cleaned":
        raise RuntimeError(f"retained-volume recovery failed; inspect {args.recovery_receipt}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
