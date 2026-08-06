"""Run EXP701 on RunPod through S3-preloaded storage and direct Pod startup.

The provider controller remains the sole paid-resource owner. It creates a
network volume, pauses before Pod creation, and waits for this orchestrator to
upload and verify the authenticated source archive through RunPod's S3 API.
The Pod starts the guest supervisor directly, while the host continuously
copies and restore-verifies checkpoint exports through the same S3 endpoint.
No SSH service, public port, or provider credential enters the guest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
from urllib import parse, request

from .offpod_sync import runpod_s3_client, upload_file_verified
from .recovery_transport import restore_export


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "exp-701-cuda-resumable-h92-w256-backbone-factorial"
APPROVAL = (
    "I_AUTHORIZE_PAID_GPU_LAUNCH RUNPOD_SECURE_RTX4090 "
    "NETWORK_VOLUME_GB_50 OFFPOD_SYNC_REQUIRED "
    "MAX_TOTAL_USD_9.00 MAX_PROVIDER_HOURS_10"
)
# Attempts A-F consumed conservative upper bounds of 2,869.913 provider seconds
# and $0.598537. This ceiling rounds the remaining provider allowance down.
FRESH_MAX_TOTAL_USD = "8.27"
FRESH_MAX_PROVIDER_SECONDS = 32_400
GUEST_SLICE_SECONDS = 30_600
CONTROLLER_CONFIG = ROOT / "configs/exp_701_runpod_control.yaml"
DEFAULT_SOURCE = ROOT / ".scratch/exp701-launch/source.tar.gz"
DEFAULT_PREFLIGHT = ROOT / "experiments/runs" / RUN_ID / "preflight.json"
DEFAULT_SESSION = ROOT / ".scratch/exp701-runpod-session"
DEFAULT_OFFPOD = ROOT / ".scratch/exp701-offpod"
DEFAULT_PROVIDER_RECEIPT = (
    ROOT / "experiments/runs" / RUN_ID / "raw/provider-lifecycle.json"
)
DEFAULT_ORCHESTRATOR_RECEIPT = (
    ROOT / "experiments/runs" / RUN_ID / "raw/orchestrator-lifecycle.json"
)
SOURCE_KEY = "sair-exp701-source.tar.gz"
RESUME_KEY = "sair-exp701-resume.tar.gz"
REMOTE_ARCHIVE_KEY = "exp701-durable/sair-exp701-checkpoint-export.tar.gz"
REMOTE_ACK_KEY = "exp701-durable/offpod-sync-ack.json"
S3_ENDPOINTS = {
    "EU-RO-1": "https://s3api-eu-ro-1.runpod.io/",
    "EUR-IS-1": "https://s3api-eur-is-1.runpod.io/",
}
REQUIRED_EXECUTABLE_ARCHIVE_PATHS = (
    "environments/cuda/bootstrap.sh",
    "environments/cuda/run-acceptance.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_identity(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as bundle:
        member = next(
            (
                item
                for item in bundle.getmembers()
                if (item.name[2:] if item.name.startswith("./") else item.name)
                == ".sair-source.json"
            ),
            None,
        )
        if member is None or not member.isfile():
            raise RuntimeError("source archive omits .sair-source.json")
        handle = bundle.extractfile(member)
        if handle is None:
            raise RuntimeError("source identity cannot be extracted")
        return json.loads(handle.read())


def verify_source_archive_runtime(archive: Path) -> dict[str, Any]:
    """Reject a payload whose Git/archive modes cannot pass CUDA verification."""
    with tarfile.open(archive, "r:gz") as bundle:
        members = {
            item.name[2:] if item.name.startswith("./") else item.name: item
            for item in bundle.getmembers()
        }
    missing = [
        path for path in REQUIRED_EXECUTABLE_ARCHIVE_PATHS if path not in members
    ]
    non_files = [
        path
        for path in REQUIRED_EXECUTABLE_ARCHIVE_PATHS
        if path in members and not members[path].isfile()
    ]
    non_executable = [
        path
        for path in REQUIRED_EXECUTABLE_ARCHIVE_PATHS
        if path in members and members[path].isfile() and not members[path].mode & 0o111
    ]
    if missing or non_files or non_executable:
        raise RuntimeError(
            "source archive violates executable launcher contract: "
            f"missing={missing}, non_files={non_files}, "
            f"non_executable={non_executable}"
        )
    return {
        "status": "source-runtime-contract-verified",
        "executable_paths": list(REQUIRED_EXECUTABLE_ARCHIVE_PATHS),
    }


def verify_preflight(path: Path, archive: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "exp701-runpod-preflight-receipt-v1"
        or receipt.get("status") != "passed-awaiting-paid-authorization"
    ):
        raise RuntimeError("EXP701 preflight receipt is not launch-admitted")
    frozen = receipt["source_archive"]
    actual_sha = sha256(archive)
    if actual_sha != frozen["sha256"] or archive.stat().st_size != frozen["bytes"]:
        raise RuntimeError("source archive no longer matches the admitted preflight")
    verify_source_archive_runtime(archive)
    identity = source_identity(archive)
    if identity.get("repository_commit") != frozen["repository_commit"]:
        raise RuntimeError("source archive commit does not match the admitted preflight")
    if receipt["required_authorization"] != APPROVAL:
        raise RuntimeError("preflight authorization phrase changed")
    return receipt


def verify_resume_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    restore = receipt.get("restore_verification") or {}
    archive = Path(str(receipt.get("archive", ""))).resolve()
    if receipt.get("status") != "verified" or restore.get("status") != "verified":
        raise RuntimeError("resume receipt lacks restore verification")
    if not archive.is_file() or sha256(archive) != receipt.get("archive_sha256"):
        raise RuntimeError("resume archive no longer matches its verified receipt")
    return {**receipt, "archive": str(archive)}


def api_get(token: str, path: str, query: dict[str, str] | None = None) -> Any:
    url = "https://rest.runpod.io/v1" + path
    if query:
        url += "?" + parse.urlencode(query)
    operation = request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with request.urlopen(operation, timeout=30) as response:
        return json.load(response)


def s3_endpoint(data_center_id: str) -> str:
    try:
        return S3_ENDPOINTS[data_center_id]
    except KeyError as error:
        raise ValueError(
            f"RunPod S3 is not configured for data center {data_center_id}"
        ) from error


def direct_start_command(
    source_sha: str,
    source_commit: str,
    nonce: str,
    resume: dict[str, Any] | None,
) -> str:
    restore = ""
    if resume is not None:
        digest = shlex.quote(str(resume["archive_sha256"]))
        restore = f"""
test "$(sha256sum /workspace/{RESUME_KEY} | cut -d' ' -f1)" = {digest}
test ! -e /workspace/sair-exp701-restore
python3 -m experiments.cloud_benchmarks.exp701_runpod.recovery_transport restore \\
  --archive /workspace/{RESUME_KEY} \\
  --destination /workspace/sair-exp701-restore --sha256 {digest}
mkdir -p .scratch
test ! -e .scratch/{RUN_ID}
mv /workspace/sair-exp701-restore/experiment/{RUN_ID} .scratch/{RUN_ID}
if test -d /workspace/sair-exp701-restore/mlflow; then
  test ! -e .scratch/mlflow
  mv /workspace/sair-exp701-restore/mlflow .scratch/mlflow
fi
"""
    return f"""set -euo pipefail
umask 077
test "$(sha256sum /workspace/{SOURCE_KEY} | cut -d' ' -f1)" = {shlex.quote(source_sha)}
test ! -e /workspace/sair-exp701/repository
mkdir -p /workspace/sair-exp701/repository /workspace/exp701-durable
tar -xzf /workspace/{SOURCE_KEY} -C /workspace/sair-exp701/repository
cd /workspace/sair-exp701/repository
test "$(python3 -c 'import json; print(json.load(open(".sair-source.json"))["repository_commit"])')" = {shlex.quote(source_commit)}
{restore}
set +e
env \\
  SAIR_EXP701_SOURCE_SHA256={shlex.quote(source_sha)} \\
  SAIR_EXP701_SYNC_NONCE={shlex.quote(nonce)} \\
  SAIR_EXP701_SLICE_SECONDS={GUEST_SLICE_SECONDS} \\
  bash experiments/cloud_benchmarks/exp701_runpod/guest_supervisor.sh
status=$?
set -e
printf '%s\n' "$status" > /workspace/exp701-durable/guest-exit
if test "$status" -ne 0 && test "$status" -ne 75; then
  output=.scratch/{RUN_ID}
  mkdir -p "$output"
  printf '{{"status":"failed","resume_required":false,"exit_code":%s}}\n' "$status" > "$output/guest-terminal.json"
  python3 -m experiments.cloud_benchmarks.exp701_runpod.recovery_transport export \\
    --output "$output" \\
    --archive /workspace/exp701-durable/sair-exp701-checkpoint-export.tar.gz \\
    --mlflow .scratch/mlflow
  for attempt in $(seq 1 60); do
    if python3 - /workspace/exp701-durable/sair-exp701-checkpoint-export.tar.gz \\
      /workspace/exp701-durable/offpod-sync-ack.json {shlex.quote(nonce)} <<'PY'
import hashlib, json, pathlib, sys
archive, ack_path = map(pathlib.Path, sys.argv[1:3])
ack = json.loads(ack_path.read_text())
digest = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
digest = digest.hexdigest()
raise SystemExit(0 if ack.get("nonce") == sys.argv[3] and ack.get("archive_sha256") == digest else 1)
PY
    then break; fi
    sleep 15
  done
fi
exit "$status"
"""


def controller_command(
    args: argparse.Namespace,
    *,
    guest_command: str,
    readiness_receipt: Path,
    source_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "environments.providers.controller",
        "--config",
        str(CONTROLLER_CONFIG),
        "run",
        "--provider",
        "runpod",
        "--resource-selector",
        "NVIDIA GeForce RTX 4090",
        "--image",
        "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--label",
        "sair-exp701-full-fidelity",
        "--hourly-rate-usd",
        "0.74",
        "--storage-hourly-usd",
        "0.010795",
        "--max-total-usd",
        FRESH_MAX_TOTAL_USD,
        "--max-provider-seconds",
        str(FRESH_MAX_PROVIDER_SECONDS),
        "--cuda-runtime",
        "any",
        "--disk-gb",
        "50",
        "--command",
        guest_command,
        "--network-volume-size-gb",
        "50",
        "--network-volume-data-center-id",
        args.data_center_id,
        "--network-volume-name",
        "sair-exp701-recovery",
        "--network-volume-ready-receipt",
        str(readiness_receipt),
        "--network-volume-ready-timeout-seconds",
        "600",
        "--network-volume-ready-source-sha256",
        source_sha256,
        "--offpod-receipt",
        str(args.offpod_destination / "offpod-sync-receipt.json"),
        "--receipt",
        str(args.provider_receipt),
    ]


def wait_for_network_volume(
    receipt: Path, process: subprocess.Popen[Any]
) -> str:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        value = read_json(receipt) or {}
        if value.get("network_volume_id"):
            return str(value["network_volume_id"])
        if process.poll() is not None:
            raise RuntimeError("provider controller exited before volume creation")
        time.sleep(2)
    raise TimeoutError("provider did not return a network volume within 120 seconds")


def terminal_from_export(receipt: dict[str, Any]) -> dict[str, Any]:
    archive = Path(str(receipt.get("archive", ""))).resolve()
    digest = str(receipt.get("archive_sha256", ""))
    with tempfile.TemporaryDirectory(dir=archive.parent) as temporary:
        restored = restore_export(
            archive, Path(temporary), expected_sha256=digest
        )
        terminal_path = Path(restored["restored_output"]) / "guest-terminal.json"
        if not terminal_path.is_file():
            raise RuntimeError("recovery export lacks guest-terminal.json")
        return json.loads(terminal_path.read_text(encoding="utf-8"))


def exported_guest_terminal(receipt_path: Path) -> dict[str, Any] | None:
    receipt = read_json(receipt_path) or {}
    if receipt.get("status") != "verified" or (
        receipt.get("restore_verification") or {}
    ).get("status") != "verified":
        return None
    try:
        return terminal_from_export(receipt)
    except RuntimeError as error:
        if "lacks guest-terminal.json" in str(error):
            return None
        raise


def wait_for_controller_or_guest_terminal(
    controller: subprocess.Popen[Any],
    receipt_path: Path,
    *,
    timeout: float,
) -> tuple[int | None, dict[str, Any] | None]:
    deadline = time.monotonic() + timeout
    while True:
        status = controller.poll()
        if status is not None:
            return status, None
        terminal = exported_guest_terminal(receipt_path)
        if terminal is not None:
            return None, terminal
        if time.monotonic() >= deadline:
            raise TimeoutError("provider controller and guest terminal both exceeded timeout")
        time.sleep(5)


def terminate(process: subprocess.Popen[Any] | None, timeout: int) -> int | None:
    if process is None:
        return None
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"process {process.pid} did not stop within {timeout}s") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-center-id", required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--offpod-destination", type=Path, default=DEFAULT_OFFPOD)
    parser.add_argument("--provider-receipt", type=Path, default=DEFAULT_PROVIDER_RECEIPT)
    parser.add_argument(
        "--orchestrator-receipt", type=Path, default=DEFAULT_ORCHESTRATOR_RECEIPT
    )
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "source",
        "preflight",
        "session",
        "offpod_destination",
        "provider_receipt",
        "orchestrator_receipt",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if "REQUIRED_AT_LAUNCH" in args.data_center_id:
        raise RuntimeError("bind a live Secure RTX4090-compatible data center")
    endpoint = s3_endpoint(args.data_center_id)
    preflight = verify_preflight(args.preflight, args.source)
    resume = (
        None
        if args.resume_receipt is None
        else verify_resume_receipt(args.resume_receipt.resolve())
    )
    nonce = secrets.token_hex(24)
    readiness_receipt = args.session / "network-volume-ready.json"
    guest_command = direct_start_command(
        preflight["source_archive"]["sha256"],
        preflight["source_archive"]["repository_commit"],
        nonce,
        resume,
    )
    command = controller_command(
        args,
        guest_command=guest_command,
        readiness_receipt=readiness_receipt,
        source_sha256=preflight["source_archive"]["sha256"],
    )
    plan = {
        "schema_version": "exp701-runpod-orchestration-plan-v2",
        "status": "dry-run" if args.dry_run else "launching",
        "transport": "runpod-s3-direct-start",
        "source_sha256": preflight["source_archive"]["sha256"],
        "source_bytes": preflight["source_archive"]["bytes"],
        "source_commit": preflight["source_archive"]["repository_commit"],
        "data_center_id": args.data_center_id,
        "s3_endpoint": endpoint,
        "provider": "RunPod Secure",
        "gpu": "NVIDIA GeForce RTX 4090",
        "maximum_total_usd": float(FRESH_MAX_TOTAL_USD),
        "maximum_provider_seconds": FRESH_MAX_PROVIDER_SECONDS,
        "network_volume_gb": 50,
        "offpod_destination": str(args.offpod_destination),
        "resume": resume is not None,
        "controller_command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "required_authorization": APPROVAL,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if os.environ.get("SAIR_CLOUD_LAUNCH_APPROVED") != APPROVAL:
        raise RuntimeError(f"live launch requires SAIR_CLOUD_LAUNCH_APPROVED={APPROVAL}")
    token = os.environ.get("RUNPOD_API_KEY")
    access_key = os.environ.get("RUNPOD_S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY")
    if not token:
        raise RuntimeError("RUNPOD_API_KEY is absent")
    if not access_key or not secret_key:
        raise RuntimeError("RunPod S3 credentials are absent")
    existing = api_get(token, "/pods", {"includeMachine": "true"})
    if not isinstance(existing, list) or existing:
        raise RuntimeError("RunPod Pod inventory is not empty; refusing duplication")
    if any(
        path.exists()
        for path in (args.session, args.provider_receipt, args.orchestrator_receipt)
    ):
        raise RuntimeError("session or durable launch receipt already exists; refusing duplication")
    if args.offpod_destination.exists() and any(args.offpod_destination.iterdir()):
        raise RuntimeError("off-Pod destination is not fresh for this provider session")

    args.session.mkdir(parents=True, exist_ok=False)
    args.offpod_destination.mkdir(parents=True, exist_ok=True)
    controller_log = args.session / "controller.log"
    host_sync_log = args.session / "host-sync.log"
    controller: subprocess.Popen[Any] | None = None
    host_sync: subprocess.Popen[Any] | None = None
    volume_id: str | None = None
    terminal: dict[str, Any] | None = None
    failure: str | None = None
    started = time.monotonic()
    controller_handle = controller_log.open("w", encoding="utf-8")
    host_sync_handle = host_sync_log.open("w", encoding="utf-8")
    try:
        controller = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=controller_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        volume_id = wait_for_network_volume(args.provider_receipt, controller)
        client = runpod_s3_client(
            endpoint,
            access_key_id=access_key,
            secret_access_key=secret_key,
        )
        source_upload = upload_file_verified(
            client, args.source, bucket=volume_id, key=SOURCE_KEY
        )
        resume_upload = None
        if resume is not None:
            resume_upload = upload_file_verified(
                client, Path(resume["archive"]), bucket=volume_id, key=RESUME_KEY
            )
        host_sync = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.cloud_benchmarks.exp701_runpod.offpod_sync",
                "--s3-bucket",
                volume_id,
                "--s3-endpoint",
                endpoint,
                "--remote-archive",
                REMOTE_ARCHIVE_KEY,
                "--remote-ack",
                REMOTE_ACK_KEY,
                "--nonce",
                nonce,
                "--destination",
                str(args.offpod_destination),
                "--poll-seconds",
                "30",
                "--max-seconds",
                str(FRESH_MAX_PROVIDER_SECONDS + 600),
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=host_sync_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        atomic_json(
            readiness_receipt,
            {
                "schema_version": "exp701-network-volume-ready-v1",
                "status": "verified",
                "network_volume_id": volume_id,
                "source": {
                    "key": SOURCE_KEY,
                    "sha256": source_upload["sha256"],
                    "bytes": source_upload["bytes"],
                },
                "resume": resume_upload,
                "upload_verification": source_upload,
            },
        )
        offpod_receipt = args.offpod_destination / "offpod-sync-receipt.json"
        controller_status, exported_terminal = wait_for_controller_or_guest_terminal(
            controller,
            offpod_receipt,
            timeout=FRESH_MAX_PROVIDER_SECONDS + 600,
        )
        if exported_terminal is not None:
            terminal = exported_terminal
            if terminal.get("status") not in {"completed", "slice-checkpointed"}:
                raise RuntimeError(f"EXP701 guest ended unsuccessfully: {terminal}")
            controller_status = terminate(controller, 480)
        elif controller_status != 0:
            raise RuntimeError(f"provider controller exited with status {controller_status}")
        offpod = read_json(args.offpod_destination / "offpod-sync-receipt.json") or {}
        if offpod.get("status") != "verified" or (
            offpod.get("restore_verification") or {}
        ).get("status") != "verified":
            raise RuntimeError("controller ended without a restore-verified off-Pod receipt")
        terminal = terminal_from_export(offpod)
        if terminal.get("status") not in {"completed", "slice-checkpointed"}:
            raise RuntimeError(f"EXP701 guest ended unsuccessfully: {terminal}")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        host_sync_status = None
        controller_status = None if controller is None else controller.poll()
        try:
            host_sync_status = terminate(host_sync, 60)
        except Exception as cleanup_error:
            failure = (f"{failure}; " if failure else "") + (
                f"host-sync cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        try:
            controller_status = terminate(controller, 480)
        except Exception as cleanup_error:
            failure = (f"{failure}; " if failure else "") + (
                f"controller cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        controller_handle.close()
        host_sync_handle.close()
        provider = read_json(args.provider_receipt) or {}
        pod_deleted = any(
            event.get("event") == "delete-confirmed"
            for event in provider.get("events", [])
        )
        volume_deleted = any(
            event.get("event") == "network-volume-delete-confirmed"
            for event in provider.get("events", [])
        )
        result = {
            **plan,
            "schema_version": "exp701-runpod-orchestration-receipt-v2",
            "status": (
                "completed-and-cleaned"
                if failure is None
                and terminal is not None
                and terminal.get("status") == "completed"
                and pod_deleted
                and volume_deleted
                else "slice-checkpointed-and-cleaned"
                if failure is None
                and terminal is not None
                and terminal.get("status") == "slice-checkpointed"
                and pod_deleted
                and volume_deleted
                else "failed"
            ),
            "network_volume_id": volume_id,
            "guest_terminal": terminal,
            "controller_exit_status": controller_status,
            "host_sync_exit_status": host_sync_status,
            "provider_pod_delete_confirmed": pod_deleted,
            "network_volume_delete_confirmed": volume_deleted,
            "failure": failure,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.orchestrator_receipt, result)
    if result["status"] not in {
        "completed-and-cleaned",
        "slice-checkpointed-and-cleaned",
    }:
        raise RuntimeError(f"EXP701 orchestration failed; inspect {args.orchestrator_receipt}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
