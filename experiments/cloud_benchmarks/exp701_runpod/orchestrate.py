"""Run the bounded EXP701 RunPod slice from the host without exposing credentials.

The provider controller remains the sole resource owner. This companion uploads
the authenticated source over ephemeral-key SSH, starts the guest supervisor
detached, runs restore-verified off-Pod synchronization, waits on one persistent
completion channel, and then asks the controller to delete-confirm both resources.
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
import time
from typing import Any
from urllib import parse, request


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "exp-701-cuda-resumable-h92-w256-backbone-factorial"
APPROVAL = (
    "I_AUTHORIZE_PAID_GPU_LAUNCH RUNPOD_SECURE_RTX4090 "
    "NETWORK_VOLUME_GB_50 OFFPOD_SYNC_REQUIRED "
    "MAX_TOTAL_USD_9.00 MAX_PROVIDER_HOURS_10"
)
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


def pod_status(token: str, resource_id: str) -> dict[str, Any] | None:
    result = api_get(token, "/pods", {"id": resource_id, "includeMachine": "true"})
    return result[0] if isinstance(result, list) and result else None


def ssh_endpoint(status: dict[str, Any] | None) -> tuple[str, int] | None:
    if not status:
        return None
    host = status.get("publicIp")
    mappings = status.get("portMappings") or {}
    raw_port = mappings.get("22") or mappings.get(22)
    if not host or not raw_port:
        return None
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise RuntimeError("RunPod returned an invalid SSH port")
    return str(host), port


def ssh_options(identity: Path, known_hosts: Path, host: str, port: int) -> list[str]:
    return [
        "ssh",
        "-i",
        str(identity),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        f"root@{host}",
    ]


def scp_options(identity: Path, known_hosts: Path, port: int) -> list[str]:
    return [
        "scp",
        "-q",
        "-i",
        str(identity),
        "-P",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]


def controller_command(args: argparse.Namespace, public_key: Path) -> list[str]:
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
        "--hourly-rate-usd",
        "0.74",
        "--storage-hourly-usd",
        "0.010795",
        "--max-total-usd",
        "9.00",
        "--max-provider-seconds",
        "36000",
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
        "--network-volume-size-gb",
        "50",
        "--network-volume-data-center-id",
        args.data_center_id,
        "--network-volume-name",
        "sair-exp701-recovery",
        "--offpod-receipt",
        str(args.offpod_destination / "offpod-sync-receipt.json"),
        "--receipt",
        str(args.provider_receipt),
    ]


def wait_for_resource(receipt: Path, process: subprocess.Popen[Any]) -> str:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("provider controller exited before resource creation")
        value = read_json(receipt) or {}
        if value.get("resource_id"):
            return str(value["resource_id"])
        time.sleep(2)
    raise TimeoutError("provider did not return a resource id within 120 seconds")


def wait_for_ssh(
    token: str,
    resource_id: str,
    process: subprocess.Popen[Any],
    identity: Path,
    known_hosts: Path,
) -> tuple[str, int]:
    deadline = time.monotonic() + 600
    endpoint: tuple[str, int] | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("provider controller exited before SSH readiness")
        endpoint = ssh_endpoint(pod_status(token, resource_id))
        if endpoint:
            host, port = endpoint
            keyscan = subprocess.run(
                ["ssh-keyscan", "-T", "8", "-p", str(port), host],
                capture_output=True,
                timeout=15,
                check=False,
            )
            if keyscan.returncode == 0 and keyscan.stdout:
                known_hosts.write_bytes(keyscan.stdout)
                probe = subprocess.run(
                    ssh_options(identity, known_hosts, host, port)
                    + ["printf SAIR_SSH_READY"],
                    capture_output=True,
                    timeout=20,
                    check=False,
                )
                if probe.returncode == 0 and probe.stdout == b"SAIR_SSH_READY":
                    return endpoint
        time.sleep(5)
    raise TimeoutError("RunPod SSH did not become ready within 600 seconds")


def remote_start_command(
    source_sha: str,
    source_commit: str,
    nonce: str,
    resume: dict[str, Any] | None,
) -> str:
    restore = ""
    if resume is not None:
        digest = shlex.quote(str(resume["archive_sha256"]))
        restore = f"""
test ! -e /workspace/sair-exp701-restore
python3 -m experiments.cloud_benchmarks.exp701_runpod.recovery_transport restore \\
  --archive /workspace/sair-exp701-resume.tar.gz \\
  --destination /workspace/sair-exp701-restore --sha256 {digest}
mkdir -p /workspace/sair-exp701/repository/.scratch
test ! -e /workspace/sair-exp701/repository/.scratch/{RUN_ID}
mv /workspace/sair-exp701-restore/experiment/{RUN_ID} \\
  /workspace/sair-exp701/repository/.scratch/{RUN_ID}
if test -d /workspace/sair-exp701-restore/mlflow; then
  test ! -e /workspace/sair-exp701/repository/.scratch/mlflow
  mv /workspace/sair-exp701-restore/mlflow \\
    /workspace/sair-exp701/repository/.scratch/mlflow
fi
"""
    guest_wrapper = f"""set +e
env \\
  SAIR_EXP701_SOURCE_SHA256={shlex.quote(source_sha)} \\
  SAIR_EXP701_SYNC_NONCE={shlex.quote(nonce)} \\
  SAIR_EXP701_SLICE_SECONDS=32400 \\
  bash experiments/cloud_benchmarks/exp701_runpod/guest_supervisor.sh
status=$?
printf '%s\\n' \"$status\" > /workspace/exp701-durable/guest-exit
exit \"$status\"
"""
    return f"""set -euo pipefail
umask 077
test \"$(sha256sum /workspace/sair-exp701-source.tar.gz | cut -d' ' -f1)\" = {shlex.quote(source_sha)}
mkdir -p /workspace/sair-exp701/repository /workspace/exp701-durable
tar -xzf /workspace/sair-exp701-source.tar.gz -C /workspace/sair-exp701/repository
cd /workspace/sair-exp701/repository
test \"$(python3 -c 'import json; print(json.load(open(\".sair-source.json\"))[\"repository_commit\"])')\" = {shlex.quote(source_commit)}
{restore}
rm -f /workspace/exp701-durable/guest-exit /workspace/exp701-durable/guest.pid
nohup bash -lc {shlex.quote(guest_wrapper)} \\
  > /workspace/exp701-durable/guest-supervisor.log 2>&1 < /dev/null &
guest_pid=$!
printf '%s\\n' \"$guest_pid\" > /workspace/exp701-durable/guest.pid
printf '%s' \"$guest_pid\"
"""


def wait_for_guest(command: list[str], maximum_seconds: int = 34500) -> int:
    wait_command = """
set -euo pipefail
pid="$(cat /workspace/exp701-durable/guest.pid)"
while ! test -f /workspace/exp701-durable/guest-exit; do
  kill -0 "$pid" 2>/dev/null || { printf 'guest-exit-marker-missing' >&2; exit 97; }
  sleep 60
done
cat /workspace/exp701-durable/guest-exit
"""
    try:
        completed = subprocess.run(
            command + [wait_command],
            capture_output=True,
            text=True,
            timeout=maximum_seconds,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as first_error:
        probe = subprocess.run(
            command
            + [
                "if test -f /workspace/exp701-durable/guest-exit; then "
                "cat /workspace/exp701-durable/guest-exit; "
                "elif kill -0 $(cat /workspace/exp701-durable/guest.pid) 2>/dev/null; "
                "then printf running; else printf missing; fi"
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip().isdigit():
            return int(probe.stdout.strip())
        raise RuntimeError(
            f"guest completion channel failed after one fresh probe: {probe.stdout.strip()}"
        ) from first_error
    value = completed.stdout.strip()
    if not value.isdigit():
        raise RuntimeError(f"guest returned an invalid exit marker: {value!r}")
    return int(value)


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
    parser.add_argument("--ssh-private-key", type=Path, required=True)
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
    args.source = args.source.resolve()
    args.preflight = args.preflight.resolve()
    args.ssh_private_key = args.ssh_private_key.resolve()
    args.session = args.session.resolve()
    args.offpod_destination = args.offpod_destination.resolve()
    args.provider_receipt = args.provider_receipt.resolve()
    args.orchestrator_receipt = args.orchestrator_receipt.resolve()
    if "REQUIRED_AT_LAUNCH" in args.data_center_id:
        raise RuntimeError("bind a live Secure RTX4090-compatible data center")
    preflight = verify_preflight(args.preflight, args.source)
    public_key = args.ssh_private_key.with_suffix(args.ssh_private_key.suffix + ".pub")
    if not args.ssh_private_key.is_file() or not public_key.is_file():
        raise RuntimeError("fresh EXP701 SSH keypair is missing")
    resume = (
        None
        if args.resume_receipt is None
        else verify_resume_receipt(args.resume_receipt.resolve())
    )
    if any(path.exists() for path in (args.session, args.provider_receipt, args.orchestrator_receipt)):
        raise RuntimeError("session or durable launch receipt already exists; refusing duplication")
    if args.offpod_destination.exists() and any(args.offpod_destination.iterdir()):
        raise RuntimeError("off-Pod destination is not fresh for this provider session")
    command = controller_command(args, public_key)
    plan = {
        "schema_version": "exp701-runpod-orchestration-plan-v1",
        "status": "dry-run" if args.dry_run else "launching",
        "source_sha256": preflight["source_archive"]["sha256"],
        "source_bytes": preflight["source_archive"]["bytes"],
        "source_commit": preflight["source_archive"]["repository_commit"],
        "data_center_id": args.data_center_id,
        "provider": "RunPod Secure",
        "gpu": "NVIDIA GeForce RTX 4090",
        "maximum_total_usd": 9.0,
        "maximum_provider_seconds": 36000,
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
    if not token:
        raise RuntimeError("RUNPOD_API_KEY is absent")
    existing = api_get(token, "/pods", {"includeMachine": "true"})
    if not isinstance(existing, list) or existing:
        raise RuntimeError("RunPod inventory is not empty; refusing a duplicate paid launch")

    args.session.mkdir(parents=True, exist_ok=False)
    args.offpod_destination.mkdir(parents=True, exist_ok=True)
    known_hosts = args.session / "known_hosts"
    controller_log = args.session / "controller.log"
    host_sync_log = args.session / "host-sync.log"
    nonce = secrets.token_hex(24)
    controller: subprocess.Popen[Any] | None = None
    host_sync: subprocess.Popen[Any] | None = None
    resource_id: str | None = None
    guest_status: int | None = None
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
        resource_id = wait_for_resource(args.provider_receipt, controller)
        host, port = wait_for_ssh(
            token, resource_id, controller, args.ssh_private_key, known_hosts
        )
        remote = f"root@{host}"
        scp = scp_options(args.ssh_private_key, known_hosts, port)
        subprocess.run(
            scp + [str(args.source), f"{remote}:/workspace/sair-exp701-source.tar.gz"],
            check=True,
            timeout=180,
        )
        if resume is not None:
            subprocess.run(
                scp
                + [str(resume["archive"]), f"{remote}:/workspace/sair-exp701-resume.tar.gz"],
                check=True,
                timeout=600,
            )
        sync_environment = {
            **os.environ,
            "SAIR_EXP701_SSH_TARGET": remote,
            "SAIR_EXP701_SSH_PORT": str(port),
            "SAIR_EXP701_SSH_IDENTITY": str(args.ssh_private_key),
            "SAIR_EXP701_SYNC_NONCE": nonce,
            "SAIR_EXP701_OFFPOD_DESTINATION": str(args.offpod_destination),
        }
        host_sync = subprocess.Popen(
            ["bash", str(ROOT / "experiments/cloud_benchmarks/exp701_runpod/host_sync.sh")],
            cwd=ROOT,
            env=sync_environment,
            stdout=host_sync_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        ssh = ssh_options(args.ssh_private_key, known_hosts, host, port)
        start = remote_start_command(
            preflight["source_archive"]["sha256"],
            preflight["source_archive"]["repository_commit"],
            nonce,
            resume,
        )
        subprocess.run(ssh + [start], check=True, timeout=180, capture_output=True)
        guest_status = wait_for_guest(ssh)
        if guest_status not in {0, 75}:
            raise RuntimeError(f"EXP701 guest exited terminally with status {guest_status}")
        offpod = read_json(args.offpod_destination / "offpod-sync-receipt.json") or {}
        if offpod.get("status") != "verified" or (offpod.get("restore_verification") or {}).get("status") != "verified":
            raise RuntimeError("guest ended without a restore-verified off-Pod receipt")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        host_sync_status = None
        controller_status = None
        try:
            host_sync_status = terminate(host_sync, 60)
        except Exception as cleanup_error:
            failure = (
                f"{failure}; " if failure else ""
            ) + f"host-sync cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
        try:
            controller_status = terminate(controller, 480)
        except Exception as cleanup_error:
            failure = (
                f"{failure}; " if failure else ""
            ) + f"controller cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
        controller_handle.close()
        host_sync_handle.close()
        provider = read_json(args.provider_receipt) or {}
        pod_deleted = any(
            event.get("event") == "delete-confirmed" for event in provider.get("events", [])
        )
        volume_deleted = any(
            event.get("event") == "network-volume-delete-confirmed"
            for event in provider.get("events", [])
        )
        result = {
            **plan,
            "schema_version": "exp701-runpod-orchestration-receipt-v1",
            "status": (
                "completed-and-cleaned"
                if failure is None and guest_status == 0 and pod_deleted and volume_deleted
                else "slice-checkpointed-and-cleaned"
                if failure is None and guest_status == 75 and pod_deleted and volume_deleted
                else "failed"
            ),
            "resource_id": resource_id,
            "guest_exit_status": guest_status,
            "controller_exit_status": controller_status,
            "host_sync_exit_status": host_sync_status,
            "provider_pod_delete_confirmed": pod_deleted,
            "network_volume_delete_confirmed": volume_deleted,
            "failure": failure,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.orchestrator_receipt, result)
    if result["status"] not in {"completed-and-cleaned", "slice-checkpointed-and-cleaned"}:
        raise RuntimeError(f"EXP701 orchestration failed; inspect {args.orchestrator_receipt}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
