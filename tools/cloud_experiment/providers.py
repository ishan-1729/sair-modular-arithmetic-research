"""Provider protocol and deterministic local provider implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Protocol, runtime_checkable

from .errors import CloudExperimentError
from .io import atomic_json, read_json


@dataclass(frozen=True)
class Lease:
    resource_id: str
    owner_nonce: str
    hourly_rate_usd: str
    created_at_epoch_seconds: int
    hardware: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    started_at_epoch_seconds: int
    finished_at_epoch_seconds: int


@runtime_checkable
class Provider(Protocol):
    provider_name: str
    paid: bool

    def doctor(self) -> dict[str, Any]: ...

    def inventory(self, *, owner_nonce: str | None) -> list[dict[str, Any]]: ...

    def allocate(
        self,
        *,
        owner_nonce: str,
        run_id: str,
        deadline_epoch_seconds: int,
    ) -> Lease: ...

    def run(
        self,
        lease: Lease,
        *,
        argv: list[str],
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        idle_timeout_seconds: int,
    ) -> CommandResult: ...

    def cleanup(self, lease: Lease) -> dict[str, Any]: ...

    def cleanup_owner(self, *, owner_nonce: str) -> dict[str, Any]: ...


def _decimal(value: Any, *, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CloudExperimentError(f"{label} must be a finite decimal") from exc
    if not result.is_finite() or result < 0:
        raise CloudExperimentError(f"{label} must be a finite non-negative decimal")
    return result


def load_provider(
    driver: str,
    *,
    options: Mapping[str, Any],
    control_directory: Path,
    credential_access_allowed: bool,
) -> Provider:
    module_name, separator, attribute = driver.partition(":")
    if not separator:
        raise CloudExperimentError("provider driver must use module:Class syntax")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
        provider = factory(
            options=dict(options),
            control_directory=control_directory,
            credential_access_allowed=credential_access_allowed,
        )
    except (ImportError, AttributeError, TypeError) as exc:
        raise CloudExperimentError(f"cannot load provider driver {driver}: {exc}") from exc
    if not isinstance(provider, Provider):
        raise CloudExperimentError(f"provider driver does not satisfy the protocol: {driver}")
    return provider


class LocalProvider:
    """Execute commands locally while exercising the complete provider lifecycle."""

    provider_name = "local"
    paid = False

    def __init__(
        self,
        *,
        options: Mapping[str, Any],
        control_directory: Path,
        credential_access_allowed: bool = False,
    ) -> None:
        self.options = dict(options)
        self.control_directory = control_directory.resolve()
        self.credential_access_allowed = credential_access_allowed
        self.registry_path = self.control_directory / "provider" / "local-resources.json"
        self.hourly_rate = _decimal(
            self.options.get("hourly_rate_usd", "0"), label="provider hourly rate"
        )
        hardware = self.options.get(
            "hardware", {"accelerator": "CPU", "accelerator_count": 0}
        )
        if not isinstance(hardware, dict):
            raise CloudExperimentError("local provider hardware must be an object")
        self.hardware = {
            "accelerator": str(hardware.get("accelerator", "CPU")),
            "accelerator_count": int(hardware.get("accelerator_count", 0)),
        }

    def _registry(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            return {"schema_version": "sair-local-provider-v1", "resources": []}
        value = read_json(self.registry_path)
        if value.get("schema_version") != "sair-local-provider-v1" or not isinstance(
            value.get("resources"), list
        ):
            raise CloudExperimentError("local provider registry is invalid")
        return value

    def _write_registry(self, value: dict[str, Any]) -> None:
        atomic_json(self.registry_path, value)

    def doctor(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "available": True,
            "paid": self.paid,
            "hourly_rate_usd": str(self.hourly_rate),
            "hardware": self.hardware,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "credential_read_performed": False,
            "provider_mutation_performed": False,
        }

    def inventory(self, *, owner_nonce: str | None) -> list[dict[str, Any]]:
        resources = [dict(item) for item in self._registry()["resources"]]
        if owner_nonce is None:
            return resources
        return [item for item in resources if item.get("owner_nonce") == owner_nonce]

    def allocate(
        self,
        *,
        owner_nonce: str,
        run_id: str,
        deadline_epoch_seconds: int,
    ) -> Lease:
        registry = self._registry()
        if registry["resources"]:
            raise CloudExperimentError("provider inventory is not empty before allocation")
        lease = Lease(
            resource_id=f"local-{owner_nonce[:16]}",
            owner_nonce=owner_nonce,
            hourly_rate_usd=str(self.hourly_rate),
            created_at_epoch_seconds=int(time.time()),
            hardware=dict(self.hardware),
            metadata={
                "run_id": run_id,
                "deadline_epoch_seconds": deadline_epoch_seconds,
                "pid": os.getpid(),
            },
        )
        registry["resources"].append(asdict(lease))
        self._write_registry(registry)
        return lease

    def run(
        self,
        lease: Lease,
        *,
        argv: list[str],
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        idle_timeout_seconds: int,
    ) -> CommandResult:
        if not self.inventory(owner_nonce=lease.owner_nonce):
            raise CloudExperimentError("local provider lease is not live")
        started = int(time.time())
        heartbeat_value = environment.get("SAIR_CLOUD_HEARTBEAT_PATH")
        if not heartbeat_value:
            raise CloudExperimentError("command environment omits heartbeat path")
        heartbeat = Path(heartbeat_value).resolve()
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        heartbeat.touch()
        process_environment = {**os.environ, **dict(environment)}
        temporary_root = self.control_directory / "provider" / "command-output"
        temporary_root.mkdir(parents=True, exist_ok=True)
        returncode = 127
        stdout = ""
        stderr = ""
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            stdout_path = Path(directory) / "stdout"
            stderr_path = Path(directory) / "stderr"
            try:
                with stdout_path.open("wb") as stdout_handle, stderr_path.open(
                    "wb"
                ) as stderr_handle:
                    process = subprocess.Popen(
                        argv,
                        cwd=cwd,
                        env=process_environment,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                    )
                    began = time.monotonic()
                    last_heartbeat = began
                    heartbeat_mtime = heartbeat.stat().st_mtime_ns
                    termination: str | None = None
                    while process.poll() is None:
                        now = time.monotonic()
                        try:
                            current_mtime = heartbeat.stat().st_mtime_ns
                        except FileNotFoundError:
                            current_mtime = heartbeat_mtime
                        if current_mtime != heartbeat_mtime:
                            heartbeat_mtime = current_mtime
                            last_heartbeat = now
                        if now - began >= timeout_seconds:
                            termination = f"command timed out after {timeout_seconds}s"
                            returncode = 124
                            break
                        if now - last_heartbeat >= idle_timeout_seconds:
                            termination = (
                                f"command heartbeat was idle for {idle_timeout_seconds}s"
                            )
                            returncode = 125
                            break
                        time.sleep(0.05)
                    if termination is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                    else:
                        returncode = int(process.returncode)
                stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
                stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                if termination is not None:
                    stderr += f"\n{termination}\n"
            except OSError as exc:
                returncode = 127
                stderr = f"cannot execute command: {exc}\n"
        return CommandResult(returncode, stdout, stderr, started, int(time.time()))

    def cleanup(self, lease: Lease) -> dict[str, Any]:
        registry = self._registry()
        before = list(registry["resources"])
        for item in before:
            if item.get("resource_id") == lease.resource_id and item.get("owner_nonce") != lease.owner_nonce:
                raise CloudExperimentError("refusing to delete a resource owned by another lifecycle")
        registry["resources"] = [
            item
            for item in before
            if not (
                item.get("resource_id") == lease.resource_id
                and item.get("owner_nonce") == lease.owner_nonce
            )
        ]
        self._write_registry(registry)
        remaining = self.inventory(owner_nonce=lease.owner_nonce)
        return {
            "resource_id": lease.resource_id,
            "owner_nonce": lease.owner_nonce,
            "deleted": len(registry["resources"]) < len(before),
            "exact_absence_confirmed": not remaining,
        }

    def cleanup_owner(self, *, owner_nonce: str) -> dict[str, Any]:
        registry = self._registry()
        owned = [
            item for item in registry["resources"] if item.get("owner_nonce") == owner_nonce
        ]
        registry["resources"] = [
            item for item in registry["resources"] if item.get("owner_nonce") != owner_nonce
        ]
        self._write_registry(registry)
        return {
            "resource_id": None,
            "owner_nonce": owner_nonce,
            "deleted": bool(owned),
            "reconciled_resource_ids": sorted(
                str(item.get("resource_id")) for item in owned
            ),
            "exact_absence_confirmed": not self.inventory(owner_nonce=owner_nonce),
        }


def lease_from_dict(value: Mapping[str, Any]) -> Lease:
    try:
        return Lease(
            resource_id=str(value["resource_id"]),
            owner_nonce=str(value["owner_nonce"]),
            hourly_rate_usd=str(value["hourly_rate_usd"]),
            created_at_epoch_seconds=int(value["created_at_epoch_seconds"]),
            hardware=dict(value["hardware"]),
            metadata=dict(value["metadata"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CloudExperimentError("saved provider lease is invalid") from exc
