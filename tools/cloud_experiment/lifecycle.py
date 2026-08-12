"""Fail-closed lifecycle orchestration independent of experiment semantics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import tarfile
import time
from typing import Any, Iterator, Mapping

from .checkpoints import (
    latest_checkpoint,
    publish_checkpoint,
    restore_latest_checkpoint,
)
from .errors import CloudExperimentError
from .io import (
    atomic_json,
    canonical_json_bytes,
    file_identity,
    read_json,
    resolve_within,
    sha256_bytes,
    sha256_file,
)
from .manifest import LoadedManifest
from .package import build_package, verify_package
from .providers import CommandResult, Lease, Provider, lease_from_dict, load_provider


LIFECYCLE_SCHEMA = "sair-cloud-lifecycle-v1"
AUTHORIZATION_SCHEMA = "sair-cloud-authorization-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@contextmanager
def lifecycle_lock(run_directory: Path, *, blocking: bool = False) -> Iterator[None]:
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / ".lifecycle.lock"
    with path.open("a+b") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise CloudExperimentError("another controller owns this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _decimal(value: Any, *, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CloudExperimentError(f"{label} must be a finite decimal") from exc
    if not result.is_finite() or result < 0:
        raise CloudExperimentError(f"{label} must be a finite non-negative decimal")
    return result


def _load_authorization(
    manifest: LoadedManifest, authorization_path: Path | None
) -> dict[str, Any] | None:
    provider_config = manifest.data["provider"]
    if not provider_config["paid"]:
        if authorization_path is not None:
            raise CloudExperimentError("authorization is not accepted for an unpaid provider")
        return None
    if authorization_path is None:
        raise CloudExperimentError("paid provider launch requires an authorization receipt")
    authorization = read_json(authorization_path.resolve())
    required = {
        "schema_version",
        "run_id",
        "manifest_sha256",
        "provider_driver",
        "maximum_total_cost_usd",
        "maximum_provider_lifetime_seconds",
        "expires_at_utc",
    }
    if not isinstance(authorization, dict) or set(authorization) != required:
        raise CloudExperimentError("authorization must contain exactly the seven public fields")
    expected = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "provider_driver": provider_config["driver"],
        "maximum_total_cost_usd": str(manifest.data["budget"]["max_total_usd"]),
        "maximum_provider_lifetime_seconds": manifest.data["budget"]["max_provider_seconds"],
    }
    for key, value in expected.items():
        actual = authorization.get(key)
        if key == "maximum_total_cost_usd":
            if _decimal(actual, label=key) != _decimal(value, label=key):
                raise CloudExperimentError(f"authorization field mismatch: {key}")
        elif actual != value:
            raise CloudExperimentError(f"authorization field mismatch: {key}")
    try:
        expiry = datetime.fromisoformat(
            str(authorization["expires_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise CloudExperimentError("authorization expiry is invalid") from exc
    if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
        raise CloudExperimentError("authorization is expired")
    return authorization


def _load_provider(
    manifest: LoadedManifest, *, credential_access_allowed: bool = False
) -> Provider:
    config = manifest.data["provider"]
    provider = load_provider(
        config["driver"],
        options=config["options"],
        control_directory=manifest.run_directory,
        credential_access_allowed=credential_access_allowed,
    )
    if bool(provider.paid) != bool(config["paid"]):
        raise CloudExperimentError("provider paid classification disagrees with manifest")
    return provider


def _pre_admit(manifest: LoadedManifest, doctor: Mapping[str, Any]) -> dict[str, Any]:
    budget = manifest.data["budget"]
    if doctor.get("credential_read_performed") is not False:
        raise CloudExperimentError("provider doctor must prove credential-free preflight")
    if doctor.get("provider_mutation_performed") is not False:
        raise CloudExperimentError("provider doctor must prove mutation-free preflight")
    rate = _decimal(doctor.get("hourly_rate_usd"), label="provider hourly rate")
    fixed = _decimal(doctor.get("fixed_cost_usd", "0"), label="provider fixed cost")
    upper = rate * Decimal(int(budget["max_provider_seconds"])) / Decimal(3600) + fixed
    maximum = _decimal(budget["max_total_usd"], label="budget.max_total_usd")
    if upper > maximum:
        raise CloudExperimentError(
            f"provider cost upper {upper} USD exceeds authorization {maximum} USD"
        )
    return {
        "hourly_rate_usd": str(rate),
        "fixed_cost_usd": str(fixed),
        "maximum_total_usd": str(maximum),
        "maximum_provider_seconds": int(budget["max_provider_seconds"]),
        "cleanup_reserve_seconds": int(budget["cleanup_reserve_seconds"]),
        "admitted_upper_usd": str(upper),
    }


def doctor(manifest: LoadedManifest) -> dict[str, Any]:
    provider = _load_provider(manifest)
    result = provider.doctor()
    admission = _pre_admit(manifest, result)
    return {
        "schema_version": "sair-cloud-doctor-v1",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "provider": result,
        "admission": admission,
    }


def _safe_extract_source(archive_path: Path, workspace: Path) -> None:
    temporary = workspace.with_name(f".{workspace.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                pure = PurePosixPath(member.name)
                if member.issym() or member.islnk() or pure.is_absolute() or ".." in pure.parts:
                    raise CloudExperimentError(f"unsafe source member: {member.name}")
                resolve_within(temporary, member.name, label="source member")
            archive.extractall(temporary, members=members, filter="data")
        if workspace.exists():
            raise CloudExperimentError("workspace already exists for a new launch")
        os.replace(temporary, workspace)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _package_identity(manifest: LoadedManifest, package_directory: Path | None) -> tuple[Path, dict[str, Any]]:
    directory = (package_directory or manifest.run_directory / "package").resolve()
    receipt = verify_package(manifest, directory)
    return directory, receipt


def _manifest_lock(manifest: LoadedManifest, package: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "sair-cloud-run-lock-v1",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "package_sha256": package["source_archive"]["sha256"],
        "repository_commit": package["repository_commit"],
        "provider_driver": manifest.data["provider"]["driver"],
    }


def _write_lifecycle(manifest: LoadedManifest, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_json(manifest.run_directory / "lifecycle.json", state)
    if state.get("cleanup") is not None and state.get("status") in {
        "completed",
        "failed",
        "cleanup-failed",
    }:
        atomic_json(
            manifest.run_directory
            / "attempts"
            / f"attempt-{int(state['attempt']):04d}.json",
            state,
        )


def _command_environment(
    manifest: LoadedManifest,
    *,
    phase_id: str,
    command: Mapping[str, Any],
    workspace: Path,
    latest: Mapping[str, Any] | None,
    attempt: int,
) -> dict[str, str]:
    environment = {
        **manifest.data["execution"].get("environment", {}),
        **command.get("environment", {}),
        "SAIR_CLOUD_RUN_ID": manifest.run_id,
        "SAIR_CLOUD_PHASE_ID": phase_id,
        "SAIR_CLOUD_CONTROL_DIR": str(manifest.run_directory),
        "SAIR_CLOUD_WORKSPACE": str(workspace),
        "SAIR_CLOUD_CHECKPOINT_DIR": str(manifest.run_directory / "checkpoints"),
        "SAIR_CLOUD_MANIFEST_SHA256": manifest.sha256,
        "SAIR_CLOUD_HEARTBEAT_PATH": str(
            manifest.run_directory
            / "heartbeats"
            / f"attempt-{attempt:04d}-{phase_id}.heartbeat"
        ),
    }
    if latest is not None:
        environment["SAIR_CLOUD_RESUME_ARCHIVE"] = str(
            manifest.run_directory / "checkpoints" / latest["archive"]["path"]
        )
    return {str(key): str(value) for key, value in environment.items()}


def _write_command_logs(
    run_directory: Path, *, attempt: int, phase_id: str, result: CommandResult
) -> dict[str, Any]:
    directory = run_directory / "logs" / f"attempt-{attempt:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path = directory / f"{phase_id}.stdout.log"
    stderr_path = directory / f"{phase_id}.stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "stdout": file_identity(stdout_path, relative_to=run_directory),
        "stderr": file_identity(stderr_path, relative_to=run_directory),
    }


def _output_identities(workspace: Path, command: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for relative in command.get("required_outputs", []):
        path = resolve_within(workspace, relative, label="required output")
        if not path.is_file():
            raise CloudExperimentError(f"required output is absent: {relative}")
        values.append(file_identity(path, relative_to=workspace))
    return values


def _phase_receipt_path(run_directory: Path, phase_id: str) -> Path:
    return run_directory / "phases" / f"{phase_id}.json"


def _completed_phase(
    manifest: LoadedManifest, package_sha256: str, phase_id: str
) -> dict[str, Any] | None:
    path = _phase_receipt_path(manifest.run_directory, phase_id)
    if not path.is_file():
        return None
    value = read_json(path)
    if value.get("status") != "completed":
        return None
    if value.get("manifest_sha256") != manifest.sha256 or value.get(
        "package_sha256"
    ) != package_sha256:
        raise CloudExperimentError(f"completed phase identity mismatch: {phase_id}")
    return value


def _verify_hardware(manifest: LoadedManifest, lease: Lease) -> None:
    expected = manifest.data["provider"]["expected_hardware"]
    actual = lease.hardware
    if actual.get("accelerator") != expected["accelerator"] or int(
        actual.get("accelerator_count", -1)
    ) != int(expected["accelerator_count"]):
        raise CloudExperimentError(
            f"allocated hardware mismatch: expected {expected}, observed {actual}"
        )


def _verify_lease_identity(
    manifest: LoadedManifest, lease: Lease, *, owner_nonce: str
) -> None:
    if lease.owner_nonce != owner_nonce:
        raise CloudExperimentError("allocated lease owner nonce mismatch")
    if not lease.resource_id or any(character.isspace() for character in lease.resource_id):
        raise CloudExperimentError("allocated lease resource id is invalid")
    rate = _decimal(lease.hourly_rate_usd, label="allocated lease hourly rate")
    budget = manifest.data["budget"]
    upper = rate * Decimal(int(budget["max_provider_seconds"])) / Decimal(3600)
    if upper > _decimal(budget["max_total_usd"], label="budget.max_total_usd"):
        raise CloudExperimentError("allocated lease rate exceeds admitted cost ceiling")


def _run_command(
    provider: Provider,
    lease: Lease,
    manifest: LoadedManifest,
    *,
    attempt: int,
    phase_id: str,
    command: Mapping[str, Any],
    workspace: Path,
    latest: Mapping[str, Any] | None,
    compute_deadline_epoch_seconds: int,
) -> tuple[CommandResult, dict[str, Any]]:
    working_directory = resolve_within(
        workspace,
        manifest.data["execution"]["working_directory"],
        label="execution.working_directory",
    )
    if not working_directory.is_dir():
        raise CloudExperimentError(f"working directory is absent: {working_directory}")
    declared_timeout = int(command["timeout_seconds"])
    remaining_seconds = compute_deadline_epoch_seconds - int(time.time())
    if remaining_seconds < declared_timeout:
        raise CloudExperimentError(
            f"insufficient budgeted compute window for {phase_id}: "
            f"needs {declared_timeout}s, has {max(0, remaining_seconds)}s"
        )
    result = provider.run(
        lease,
        argv=[str(value) for value in command["argv"]],
        cwd=working_directory,
        environment=_command_environment(
            manifest,
            phase_id=phase_id,
            command=command,
            workspace=workspace,
            latest=latest,
            attempt=attempt,
        ),
        timeout_seconds=declared_timeout,
        idle_timeout_seconds=int(manifest.data["budget"]["max_idle_seconds"]),
    )
    logs = _write_command_logs(
        manifest.run_directory, attempt=attempt, phase_id=phase_id, result=result
    )
    return result, logs


def _checkpoint_if_available(
    manifest: LoadedManifest,
    package_sha256: str,
    workspace: Path,
    phase_id: str,
    command: Mapping[str, Any],
    retained_paths: list[str],
) -> dict[str, Any] | None:
    declared = list(
        dict.fromkeys([*retained_paths, *command.get("checkpoint_paths", [])])
    )
    if not declared:
        return None
    existing = [
        value
        for value in declared
        if resolve_within(workspace, value, label="checkpoint path").exists()
    ]
    if not existing:
        return None
    return publish_checkpoint(
        workspace=workspace,
        checkpoint_directory=manifest.run_directory / "checkpoints",
        declared_paths=declared,
        manifest_sha256=manifest.sha256,
        package_sha256=package_sha256,
        phase_id=phase_id,
    )


def _new_state(
    manifest: LoadedManifest,
    package: Mapping[str, Any],
    *,
    owner_nonce: str,
    attempt: int,
) -> dict[str, Any]:
    return {
        "schema_version": LIFECYCLE_SCHEMA,
        "status": "starting",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "package_sha256": package["source_archive"]["sha256"],
        "repository_commit": package["repository_commit"],
        "owner_nonce": owner_nonce,
        "attempt": attempt,
        "provider_driver": manifest.data["provider"]["driver"],
        "started_at_utc": utc_now(),
        "canary": None,
        "phases": [],
        "latest_checkpoint": None,
        "lease": None,
        "cleanup": None,
        "errors": [],
    }


def _existing_state(manifest: LoadedManifest) -> dict[str, Any]:
    path = manifest.run_directory / "lifecycle.json"
    if not path.is_file():
        raise CloudExperimentError("resume requires an existing lifecycle receipt")
    state = read_json(path)
    if state.get("schema_version") != LIFECYCLE_SCHEMA:
        raise CloudExperimentError("lifecycle receipt schema mismatch")
    return state


def _prepare_workspace(
    manifest: LoadedManifest,
    package_directory: Path,
    package: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[Path, dict[str, Any] | None]:
    workspace = manifest.run_directory / "workspace"
    lock_value = _manifest_lock(manifest, package)
    lock_path = manifest.run_directory / "manifest.lock.json"
    if resume:
        if read_json(lock_path) != lock_value:
            raise CloudExperimentError("resume source/config/provider identity mismatch")
        fresh = workspace.with_name(
            f".{workspace.name}.resume-{os.getpid()}-{secrets.token_hex(4)}"
        )
        archive_path = package_directory / package["source_archive"]["path"]
        _safe_extract_source(archive_path, fresh)
        try:
            restored = restore_latest_checkpoint(
                checkpoint_directory=manifest.run_directory / "checkpoints",
                workspace=fresh,
                manifest_sha256=manifest.sha256,
                package_sha256=package["source_archive"]["sha256"],
            )
            completed_receipts = [
                path
                for path in (manifest.run_directory / "phases").glob("*.json")
                if read_json(path).get("status") == "completed"
            ]
            if restored is None and completed_receipts:
                raise CloudExperimentError(
                    "resume has completed phases but no verified checkpoint"
                )
            backup = workspace.with_name(
                f".{workspace.name}.previous-{os.getpid()}-{secrets.token_hex(4)}"
            )
            if workspace.exists():
                os.replace(workspace, backup)
            try:
                os.replace(fresh, workspace)
            except Exception:
                if backup.exists() and not workspace.exists():
                    os.replace(backup, workspace)
                raise
            shutil.rmtree(backup, ignore_errors=True)
            return workspace, restored
        except Exception:
            shutil.rmtree(fresh, ignore_errors=True)
            raise
    if lock_path.exists() or (manifest.run_directory / "lifecycle.json").exists():
        raise CloudExperimentError("launch refuses an existing run; use resume")
    archive_path = package_directory / package["source_archive"]["path"]
    _safe_extract_source(archive_path, workspace)
    atomic_json(lock_path, lock_value)
    return workspace, None


def _run_lifecycle(
    manifest: LoadedManifest,
    *,
    package_directory: Path | None,
    authorization_path: Path | None,
    resume: bool,
) -> dict[str, Any]:
    authorization = _load_authorization(manifest, authorization_path)
    package_dir, package = _package_identity(manifest, package_directory)
    with lifecycle_lock(manifest.run_directory):
        if resume:
            previous = _existing_state(manifest)
            if previous.get("manifest_sha256") != manifest.sha256 or previous.get(
                "package_sha256"
            ) != package["source_archive"]["sha256"]:
                raise CloudExperimentError("resume lifecycle identity mismatch")
            if previous.get("status") == "completed":
                raise CloudExperimentError("lifecycle is already complete; run finalize")
            if not previous.get("cleanup", {}).get("exact_absence_confirmed"):
                raise CloudExperimentError("resume requires deletion-confirmed prior resources")
            owner_nonce = str(previous["owner_nonce"])
            attempt = int(previous.get("attempt", 0)) + 1
        else:
            owner_nonce = secrets.token_hex(16)
            attempt = 1
        workspace, restored = _prepare_workspace(
            manifest, package_dir, package, resume=resume
        )
        state = _new_state(
            manifest, package, owner_nonce=owner_nonce, attempt=attempt
        )
        if restored is not None:
            state["latest_checkpoint"] = restored
        if authorization is not None:
            state["authorization_sha256"] = sha256_bytes(
                canonical_json_bytes(authorization)
            )
        _write_lifecycle(manifest, state)

        provider: Provider | None = None
        lease: Lease | None = None
        failure: BaseException | None = None
        budget = manifest.data["budget"]
        started_epoch = int(time.time())
        deadline = started_epoch + int(budget["max_provider_seconds"])
        try:
            provider = _load_provider(
                manifest,
                credential_access_allowed=bool(manifest.data["provider"]["paid"]),
            )
            provider_doctor = provider.doctor()
            state["admission"] = _pre_admit(manifest, provider_doctor)
            state["provider_doctor"] = provider_doctor
            if provider.inventory(owner_nonce=None):
                raise CloudExperimentError("provider inventory is not empty before allocation")
            candidate = provider.allocate(
                owner_nonce=owner_nonce,
                run_id=manifest.run_id,
                deadline_epoch_seconds=deadline,
            )
            _verify_lease_identity(manifest, candidate, owner_nonce=owner_nonce)
            lease = candidate
            state["lease"] = asdict(lease)
            _verify_hardware(manifest, lease)
            state["status"] = "running-canary"
            _write_lifecycle(manifest, state)

            canary = manifest.data["execution"]["canary"]
            result, logs = _run_command(
                provider,
                lease,
                manifest,
                attempt=attempt,
                phase_id="canary",
                command=canary,
                workspace=workspace,
                latest=latest_checkpoint(manifest.run_directory / "checkpoints"),
                compute_deadline_epoch_seconds=deadline
                - int(budget["cleanup_reserve_seconds"]),
            )
            state["canary"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "logs": logs,
                "started_at_epoch_seconds": result.started_at_epoch_seconds,
                "finished_at_epoch_seconds": result.finished_at_epoch_seconds,
            }
            _write_lifecycle(manifest, state)
            if result.returncode != 0:
                raise CloudExperimentError("canary command failed")

            package_sha256 = package["source_archive"]["sha256"]
            retained_paths: list[str] = []
            for phase in manifest.data["execution"]["phases"]:
                phase_id = str(phase["id"])
                completed = _completed_phase(manifest, package_sha256, phase_id)
                if completed is not None:
                    state["phases"].append(
                        {"id": phase_id, "status": "reused", "receipt": completed}
                    )
                    retained_paths.extend(
                        str(value) for value in phase.get("required_outputs", [])
                    )
                    retained_paths = list(dict.fromkeys(retained_paths))
                    _write_lifecycle(manifest, state)
                    continue
                if int(time.time()) >= deadline - int(budget["cleanup_reserve_seconds"]):
                    raise CloudExperimentError("cleanup reserve reached before next phase")
                state["status"] = f"running-phase:{phase_id}"
                _write_lifecycle(manifest, state)
                current_latest = latest_checkpoint(manifest.run_directory / "checkpoints")
                result, logs = _run_command(
                    provider,
                    lease,
                    manifest,
                    attempt=attempt,
                    phase_id=phase_id,
                    command=phase,
                    workspace=workspace,
                    latest=current_latest,
                    compute_deadline_epoch_seconds=deadline
                    - int(budget["cleanup_reserve_seconds"]),
                )
                checkpoint: dict[str, Any] | None = None
                if result.returncode == 0 or phase.get("checkpoint_on_failure", False):
                    checkpoint_retained = list(retained_paths)
                    if result.returncode == 0:
                        checkpoint_retained.extend(
                            str(value) for value in phase.get("required_outputs", [])
                        )
                    checkpoint = _checkpoint_if_available(
                        manifest,
                        package_sha256,
                        workspace,
                        phase_id,
                        phase,
                        list(dict.fromkeys(checkpoint_retained)),
                    )
                receipt = {
                    "schema_version": "sair-cloud-phase-v1",
                    "id": phase_id,
                    "status": "completed" if result.returncode == 0 else "failed",
                    "attempt": attempt,
                    "manifest_sha256": manifest.sha256,
                    "package_sha256": package_sha256,
                    "returncode": result.returncode,
                    "logs": logs,
                    "outputs": _output_identities(workspace, phase)
                    if result.returncode == 0
                    else [],
                    "checkpoint": checkpoint,
                    "started_at_epoch_seconds": result.started_at_epoch_seconds,
                    "finished_at_epoch_seconds": result.finished_at_epoch_seconds,
                }
                atomic_json(_phase_receipt_path(manifest.run_directory, phase_id), receipt)
                atomic_json(
                    manifest.run_directory
                    / "phases"
                    / phase_id
                    / f"attempt-{attempt:04d}.json",
                    receipt,
                )
                state["phases"].append(
                    {"id": phase_id, "status": receipt["status"], "receipt": receipt}
                )
                state["latest_checkpoint"] = latest_checkpoint(
                    manifest.run_directory / "checkpoints"
                )
                _write_lifecycle(manifest, state)
                if result.returncode != 0:
                    raise CloudExperimentError(
                        f"phase {phase_id} failed with exit code {result.returncode}"
                    )
                retained_paths.extend(
                    str(value) for value in phase.get("required_outputs", [])
                )
                retained_paths = list(dict.fromkeys(retained_paths))
            state["status"] = "compute-complete"
            _write_lifecycle(manifest, state)
        except BaseException as exc:  # cleanup must also run for cancellation/interrupt
            failure = exc
            state["status"] = "failed"
            state["errors"].append(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "at_utc": utc_now(),
                }
            )
            _write_lifecycle(manifest, state)
        finally:
            cleanup: dict[str, Any]
            if provider is not None and lease is not None:
                try:
                    cleanup = provider.cleanup(lease)
                except BaseException as cleanup_exc:
                    cleanup = {
                        "resource_id": lease.resource_id,
                        "owner_nonce": lease.owner_nonce,
                        "deleted": False,
                        "exact_absence_confirmed": False,
                        "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    }
                    if failure is None:
                        failure = cleanup_exc
            elif provider is not None:
                try:
                    cleanup = provider.cleanup_owner(owner_nonce=owner_nonce)
                except BaseException as cleanup_exc:
                    cleanup = {
                        "resource_id": None,
                        "owner_nonce": owner_nonce,
                        "deleted": False,
                        "exact_absence_confirmed": False,
                        "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    }
                    if failure is None:
                        failure = cleanup_exc
            else:
                cleanup = {
                    "resource_id": None,
                    "owner_nonce": owner_nonce,
                    "deleted": False,
                    "exact_absence_confirmed": True,
                    "provider_resource_created": False,
                }
            state["cleanup"] = cleanup
            state["provider_elapsed_seconds"] = max(0, int(time.time()) - started_epoch)
            if failure is None and cleanup.get("exact_absence_confirmed"):
                state["status"] = "completed"
                state["completed_at_utc"] = utc_now()
            elif not cleanup.get("exact_absence_confirmed"):
                state["status"] = "cleanup-failed"
            _write_lifecycle(manifest, state)
        if failure is not None:
            if isinstance(failure, KeyboardInterrupt):
                raise failure
            if isinstance(failure, CloudExperimentError):
                raise failure
            raise CloudExperimentError(f"lifecycle failed: {failure}") from failure
        return state


def launch(
    manifest: LoadedManifest,
    *,
    package_directory: Path | None = None,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    return _run_lifecycle(
        manifest,
        package_directory=package_directory,
        authorization_path=authorization_path,
        resume=False,
    )


def resume(
    manifest: LoadedManifest,
    *,
    package_directory: Path | None = None,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    return _run_lifecycle(
        manifest,
        package_directory=package_directory,
        authorization_path=authorization_path,
        resume=True,
    )


def canary(
    manifest: LoadedManifest,
    *,
    package_directory: Path | None = None,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    _load_authorization(manifest, authorization_path)
    package_dir, package = _package_identity(manifest, package_directory)
    canary_root = manifest.run_directory / "canary"
    with lifecycle_lock(manifest.run_directory):
        workspace = canary_root / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)
        _safe_extract_source(
            package_dir / package["source_archive"]["path"], workspace
        )
        provider = _load_provider(
            manifest,
            credential_access_allowed=bool(manifest.data["provider"]["paid"]),
        )
        provider_doctor = provider.doctor()
        admission = _pre_admit(manifest, provider_doctor)
        if provider.inventory(owner_nonce=None):
            raise CloudExperimentError("provider inventory is not empty before canary")
        owner_nonce = secrets.token_hex(16)
        lease: Lease | None = None
        receipt: dict[str, Any] = {
            "schema_version": "sair-cloud-canary-v1",
            "run_id": manifest.run_id,
            "manifest_sha256": manifest.sha256,
            "package_sha256": package["source_archive"]["sha256"],
            "status": "starting",
            "admission": admission,
            "cleanup": None,
        }
        try:
            candidate = provider.allocate(
                owner_nonce=owner_nonce,
                run_id=manifest.run_id,
                deadline_epoch_seconds=int(time.time())
                + int(manifest.data["budget"]["max_provider_seconds"]),
            )
            _verify_lease_identity(manifest, candidate, owner_nonce=owner_nonce)
            lease = candidate
            _verify_hardware(manifest, lease)
            result, logs = _run_command(
                provider,
                lease,
                manifest,
                attempt=0,
                phase_id="standalone-canary",
                command=manifest.data["execution"]["canary"],
                workspace=workspace,
                latest=None,
                compute_deadline_epoch_seconds=int(time.time())
                + int(manifest.data["budget"]["max_provider_seconds"])
                - int(manifest.data["budget"]["cleanup_reserve_seconds"]),
            )
            receipt.update(
                {
                    "status": "passed" if result.returncode == 0 else "failed",
                    "returncode": result.returncode,
                    "logs": logs,
                    "hardware": lease.hardware,
                }
            )
            if result.returncode != 0:
                raise CloudExperimentError("standalone canary failed")
        finally:
            try:
                receipt["cleanup"] = (
                    provider.cleanup(lease)
                    if lease is not None
                    else provider.cleanup_owner(owner_nonce=owner_nonce)
                )
            except BaseException as cleanup_exc:
                receipt["status"] = "cleanup-failed"
                receipt["cleanup"] = {
                    "resource_id": None if lease is None else lease.resource_id,
                    "owner_nonce": owner_nonce,
                    "deleted": False,
                    "exact_absence_confirmed": False,
                    "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                }
                atomic_json(canary_root / "receipt.json", receipt)
                raise CloudExperimentError(
                    f"standalone canary cleanup failed: {cleanup_exc}"
                ) from cleanup_exc
            atomic_json(canary_root / "receipt.json", receipt)
        return receipt


def status(manifest: LoadedManifest) -> dict[str, Any]:
    path = manifest.run_directory / "lifecycle.json"
    if not path.is_file():
        return {
            "schema_version": LIFECYCLE_SCHEMA,
            "run_id": manifest.run_id,
            "manifest_sha256": manifest.sha256,
            "status": "not-started",
        }
    state = read_json(path)
    if state.get("manifest_sha256") != manifest.sha256:
        raise CloudExperimentError("status manifest identity mismatch")
    return state


def cleanup(manifest: LoadedManifest) -> dict[str, Any]:
    with lifecycle_lock(manifest.run_directory):
        state = _existing_state(manifest)
        provider = _load_provider(manifest, credential_access_allowed=True)
        prior = state.get("cleanup")
        if prior and prior.get("exact_absence_confirmed") and not provider.inventory(
            owner_nonce=str(state.get("owner_nonce"))
        ):
            return prior
        lease_value = state.get("lease")
        if lease_value is None:
            result = provider.cleanup_owner(owner_nonce=str(state.get("owner_nonce")))
        else:
            result = provider.cleanup(lease_from_dict(lease_value))
        state["cleanup"] = result
        if result.get("exact_absence_confirmed") and state.get("status") == "cleanup-failed":
            state["status"] = "failed"
        _write_lifecycle(manifest, state)
        return result


def finalize(manifest: LoadedManifest) -> dict[str, Any]:
    with lifecycle_lock(manifest.run_directory):
        state = _existing_state(manifest)
        if state.get("status") != "completed":
            raise CloudExperimentError("finalize requires a completed lifecycle")
        if not state.get("cleanup", {}).get("exact_absence_confirmed"):
            raise CloudExperimentError("finalize requires exact resource absence")
        package = verify_package(manifest)
        workspace = manifest.run_directory / "workspace"
        output_map: dict[str, dict[str, Any]] = {}
        for phase in manifest.data["execution"]["phases"]:
            receipt = _completed_phase(
                manifest, package["source_archive"]["sha256"], str(phase["id"])
            )
            if receipt is None:
                raise CloudExperimentError(f"finalize lacks completed phase {phase['id']}")
            for item in _output_identities(workspace, phase):
                prior = output_map.get(item["path"])
                if prior is not None and prior != item:
                    raise CloudExperimentError(
                        f"two phases disagree on final output identity: {item['path']}"
                    )
                output_map[item["path"]] = item
        outputs = [output_map[key] for key in sorted(output_map)]
        publication = manifest.publication_directory
        final = {
            "schema_version": "sair-cloud-finalization-v1",
            "run_id": manifest.run_id,
            "manifest_sha256": manifest.sha256,
            "package_sha256": package["source_archive"]["sha256"],
            "lifecycle_sha256": sha256_file(manifest.run_directory / "lifecycle.json"),
            "repository_commit": package["repository_commit"],
            "outputs": outputs,
            "latest_checkpoint": latest_checkpoint(
                manifest.run_directory / "checkpoints"
            ),
            "cleanup": state["cleanup"],
            "finalized_at_utc": utc_now(),
        }
        if publication.exists():
            existing_path = publication / "finalized.json"
            if existing_path.is_file():
                existing = read_json(existing_path)
                comparable = {key: value for key, value in final.items() if key != "finalized_at_utc"}
                old = {key: value for key, value in existing.items() if key != "finalized_at_utc"}
                if old == comparable:
                    return existing
            raise CloudExperimentError("publication directory already exists with different bytes")
        temporary = publication.with_name(
            f".{publication.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
        )
        temporary.mkdir(parents=True)
        try:
            (temporary / "outputs").mkdir()
            for item in outputs:
                source = resolve_within(workspace, item["path"], label="final output")
                destination = resolve_within(
                    temporary / "outputs", item["path"], label="published output"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if file_identity(
                    destination, relative_to=temporary / "outputs"
                ) != item:
                    raise CloudExperimentError(
                        f"published output verification failed: {item['path']}"
                    )
            shutil.copy2(manifest.run_directory / "lifecycle.json", temporary / "lifecycle.json")
            shutil.copy2(manifest.run_directory / "manifest.lock.json", temporary / "manifest.lock.json")
            atomic_json(temporary / "finalized.json", final)
            publication.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, publication)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final


def ensure_package(manifest: LoadedManifest, output_directory: Path | None = None) -> dict[str, Any]:
    directory = output_directory or manifest.run_directory / "package"
    if (directory / "package.json").is_file():
        return verify_package(manifest, directory)
    return build_package(manifest, directory)
