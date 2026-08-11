"""Fail-closed EXP702 lifecycle with injected REST/S3 transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import tarfile
from typing import Any, Callable, Protocol
from datetime import datetime, timezone


class LifecycleFailure(RuntimeError):
    """A lifecycle invariant failed closed."""


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class AdmissionSnapshot:
    observed_at_seconds: int
    pods: tuple[str, ...]
    volumes: tuple[str, ...]


@dataclass(frozen=True)
class PodFacts:
    pod_id: str
    name: str
    owner_nonce: str
    volume_id: str
    gpu: str | None
    gpu_count: int | None
    cloud_type: str
    data_center_id: str
    interruptible: bool | None
    cost_per_hour_usd: Decimal
    quarantined: bool


class Transport(Protocol):
    now_seconds: int
    def watchdog_ready(self, *, suffix: str, deadline_seconds: int) -> None: ...
    def admission_snapshot(self) -> AdmissionSnapshot: ...
    def create_volume(self, *, name: str, size_gb: int, owner_nonce: str) -> str: ...
    def preload_volume(self, *, volume_id: str, artifacts: dict[str, Path], owner_nonce: str) -> None: ...
    def create_pod(self, *, name: str, volume_id: str, owner_nonce: str) -> str: ...
    def verify_quarantined_pod(self, *, pod_id: str, name: str, volume_id: str,
                               owner_nonce: str,
                               readiness_deadline_seconds: int,
                               fact_recorder: Callable[[dict[str, Any]], None]) -> PodFacts: ...
    def release_pod(self, *, pod_id: str, volume_id: str, owner_nonce: str) -> None: ...
    def guest_state(self, *, pod_id: str, volume_id: str, owner_nonce: str) -> dict[str, Any]: ...
    def download_export(self, *, pod_id: str, volume_id: str, owner_nonce: str, sequence: int) -> bytes: ...
    def publish_export(self, *, data: bytes, identity: dict[str, Any]) -> dict[str, Any]: ...
    def acknowledge_export(self, *, pod_id: str, volume_id: str, owner_nonce: str,
                           sequence: int, archive_sha256: str) -> None: ...
    def delete_pod(self, *, pod_id: str, owner_nonce: str) -> None: ...
    def delete_volume(self, *, volume_id: str, owner_nonce: str) -> None: ...
    def inventories(self) -> tuple[tuple[str, ...], tuple[str, ...]]: ...
    def reconcile_resources(self, *, pod_name: str, volume_name: str,
                            owner_nonce: str) -> tuple[str | None, str | None]: ...
    def wait(self, seconds: int) -> None: ...


@dataclass
class FakeTransport:
    """Deterministic no-I/O lifecycle transport."""

    scenario: str = "positive"
    now_seconds: int = 0
    secure_rate: Decimal = Decimal("0.74")
    pods: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    publications: list[dict[str, Any]] = field(default_factory=list)
    acknowledgements: list[dict[str, Any]] = field(default_factory=list)
    guest_polls: int = 0
    publication_attempts: int = 0

    def event(self, operation: str, **fields: Any) -> None:
        self.events.append({"operation": operation, "at_seconds": self.now_seconds, **fields})

    def watchdog_ready(self, *, suffix: str, deadline_seconds: int) -> None:
        self.event("watchdog-ready", suffix=suffix, deadline_seconds=deadline_seconds)

    def admission_snapshot(self) -> AdmissionSnapshot:
        self.event("admission-snapshot")
        return AdmissionSnapshot(self.now_seconds, tuple(sorted(self.pods)), tuple(sorted(self.volumes)))

    def create_volume(self, *, name: str, size_gb: int, owner_nonce: str) -> str:
        if self.volumes:
            raise LifecycleFailure("exactly one volume may be created")
        value = f"vol-{owner_nonce[:12]}"
        self.volumes[value] = owner_nonce
        self.event("create-volume", volume_id=value, name=name, size_gb=size_gb)
        return value

    def preload_volume(self, *, volume_id: str, artifacts: dict[str, Path], owner_nonce: str) -> None:
        if self.volumes.get(volume_id) != owner_nonce or not all(path.is_file() for path in artifacts.values()):
            raise LifecycleFailure("S3 preload identity mismatch")
        self.event("s3-preload", volume_id=volume_id, artifacts=sorted(artifacts))

    def create_pod(self, *, name: str, volume_id: str, owner_nonce: str) -> str:
        if self.pods or self.volumes.get(volume_id) != owner_nonce:
            raise LifecycleFailure("exactly one owned Pod may be created")
        if self.scenario == "partial-create":
            self.event("create-pod-failed", volume_id=volume_id, name=name)
            raise LifecycleFailure("simulated Pod create failure")
        value = f"pod-{owner_nonce[:12]}"
        self.pods[value] = owner_nonce
        self.event("create-pod", pod_id=value, volume_id=volume_id, name=name)
        return value

    def verify_quarantined_pod(self, *, pod_id: str, name: str, volume_id: str,
                               owner_nonce: str,
                               readiness_deadline_seconds: int,
                               fact_recorder: Callable[[dict[str, Any]], None]) -> PodFacts:
        if self.pods.get(pod_id) != owner_nonce:
            raise LifecycleFailure("quarantined Pod identity mismatch")
        self.event("rest-get-pod", pod_id=pod_id)
        gpu = "unexpected GPU" if self.scenario == "pod-mismatch" else "NVIDIA GeForce RTX 4090"
        fact_recorder({
            "schema_version": "exp702-sanitized-returned-pod-facts-v2",
            "pod_id": pod_id, "name": name, "owner_nonce_matches": True,
            "network_volume_id": volume_id, "gpu_display_name": gpu, "gpu_count": 1,
            "secure_cloud": True, "data_center_id": "US-CA-2", "interruptible": False,
            "cost_per_hour_usd": str(self.secure_rate), "environment_subset": True,
            "provider_added_environment_key_count": 0,
            "docker_start_cmd_exact": True, "allocation_verified": True,
            "projection_errors": [],
        })
        return PodFacts(pod_id, name, owner_nonce, volume_id, gpu, 1, "SECURE",
                        "US-CA-2", False, self.secure_rate, True)

    def release_pod(self, *, pod_id: str, volume_id: str, owner_nonce: str) -> None:
        if self.pods.get(pod_id) != owner_nonce or self.volumes.get(volume_id) != owner_nonce:
            raise LifecycleFailure("Pod release identity mismatch")
        self.event("release-pod", pod_id=pod_id, volume_id=volume_id)
        self.event("guest-hardware-acceptance", gpu_count=1,
                   gpu_display_name="NVIDIA GeForce RTX 4090", cuda_acceptance="passed")

    def _archive(self, pod_id: str, volume_id: str, owner_nonce: str, sequence: int) -> bytes:
        terminal = sequence >= 2
        outcome = "positive" if self.scenario == "positive" else "negative"
        summary = json.dumps({"scientific_outcome": outcome}, sort_keys=True).encode() + b"\n"
        record_value: dict[str, Any] = {
            "schema_version": "exp702-guest-export-v2", "pod_id": pod_id,
            "volume_id": volume_id, "owner_nonce": owner_nonce, "sequence": sequence,
            "export_kind": "final" if terminal else "periodic",
        }
        if terminal:
            record_value.update({"scientific_outcome": outcome,
                                 "terminal_summary_sha256": hashlib.sha256(summary).hexdigest()})
        record = json.dumps(record_value, sort_keys=True).encode() + b"\n"
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            info = tarfile.TarInfo("exp702/export.json")
            info.size = len(record); info.mtime = 0
            archive.addfile(info, io.BytesIO(record))
            if terminal:
                info = tarfile.TarInfo(
                    ".scratch/exp-702-exp701-terminal-evaluation-continuation/terminal-summary.json")
                info.size = len(summary); info.mtime = 0
                archive.addfile(info, io.BytesIO(summary))
        return output.getvalue()

    def download_export(self, *, pod_id: str, volume_id: str, owner_nonce: str, sequence: int) -> bytes:
        if self.pods.get(pod_id) != owner_nonce or self.volumes.get(volume_id) != owner_nonce:
            raise LifecycleFailure("export identity mismatch")
        if self.scenario == "export-failure" and sequence > 0:
            raise LifecycleFailure("simulated export download failure")
        self.event("s3-download-export", sequence=sequence)
        return self._archive(pod_id, volume_id, owner_nonce, sequence)

    def publish_export(self, *, data: bytes, identity: dict[str, Any]) -> dict[str, Any]:
        self.publication_attempts += 1
        if ((self.scenario == "publication-failure" and self.publication_attempts <= 3)
                or (self.scenario == "publication-retry" and self.publication_attempts == 1)):
            self.event("publish-export-failed", sequence=identity["sequence"])
            raise LifecycleFailure("simulated durable publication failure")
        value = {**identity, "bytes": len(data), "durable": True}
        self.publications.append(value); self.event("publish-export", **value)
        return value

    def acknowledge_export(self, *, pod_id: str, volume_id: str, owner_nonce: str,
                           sequence: int, archive_sha256: str) -> None:
        if self.scenario == "stale-ack" and sequence > 0:
            self.now_seconds += 721
            raise LifecycleFailure("simulated stale acknowledgement")
        value = {"pod_id": pod_id, "volume_id": volume_id, "owner_nonce": owner_nonce,
                 "sequence": sequence, "archive_sha256": archive_sha256}
        self.acknowledgements.append(value); self.event("s3-acknowledge-export", **value)

    def guest_state(self, *, pod_id: str, volume_id: str, owner_nonce: str) -> dict[str, Any]:
        if self.pods.get(pod_id) != owner_nonce:
            raise LifecycleFailure("guest state identity mismatch")
        self.guest_polls += 1; self.event("s3-guest-state", poll=self.guest_polls)
        if self.scenario == "guest-failure" and self.guest_polls >= 2:
            return {"state": "failed", "latest_sequence": 1}
        if self.scenario == "interruption" and self.guest_polls >= 2:
            raise KeyboardInterrupt("simulated controller death")
        if self.guest_polls >= 3:
            outcome = "positive" if self.scenario == "positive" else "negative"
            data = self._archive(pod_id, volume_id, owner_nonce, 2)
            return {"state": f"completed-{outcome}", "latest_sequence": 2,
                    "final_export": {"pod_id": pod_id, "volume_id": volume_id,
                                     "owner_nonce": owner_nonce, "sequence": 2,
                                     "archive_sha256": hashlib.sha256(data).hexdigest(),
                                     "scientific_outcome": outcome}}
        return {"state": "running", "latest_sequence": self.guest_polls - 1}

    def delete_pod(self, *, pod_id: str, owner_nonce: str) -> None:
        self.event("delete-pod", pod_id=pod_id)
        if self.scenario == "deletion-failure":
            raise LifecycleFailure("simulated Pod deletion failure")
        if self.pods.get(pod_id) != owner_nonce:
            raise LifecycleFailure("Pod deletion identity mismatch")
        del self.pods[pod_id]

    def delete_volume(self, *, volume_id: str, owner_nonce: str) -> None:
        self.event("delete-volume", volume_id=volume_id)
        if self.volumes.get(volume_id) != owner_nonce:
            raise LifecycleFailure("volume deletion identity mismatch")
        del self.volumes[volume_id]

    def inventories(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        self.event("inventory-confirmation", pods=sorted(self.pods), volumes=sorted(self.volumes))
        return tuple(sorted(self.pods)), tuple(sorted(self.volumes))

    def reconcile_resources(self, *, pod_name: str, volume_name: str,
                            owner_nonce: str) -> tuple[str | None, str | None]:
        pod = next((key for key, value in self.pods.items() if value == owner_nonce), None)
        volume = next((key for key, value in self.volumes.items() if value == owner_nonce), None)
        self.event("reconcile-uncertain-create", pod_id=pod, volume_id=volume,
                   pod_name=pod_name, volume_name=volume_name)
        return pod, volume

    def wait(self, seconds: int) -> None:
        self.event("wait", seconds=seconds)
        self.now_seconds += seconds


def acquire_suffix_lock(lock_root: Path, suffix: str) -> tuple[Path, int]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", suffix):
        raise LifecycleFailure("single lifecycle suffix is not canonical")
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / f"{suffix}.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise LifecycleFailure("single lifecycle suffix is already locked") from error
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    return path, descriptor


def restore_verify(data: bytes, destination: Path, identity: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(data).hexdigest()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or any(member.issym() or member.islnk() or member.isdev()
                              or member.name.startswith("/") or ".." in Path(member.name).parts
                              for member in members):
            raise LifecycleFailure("downloaded export is not independently restorable")
        archive.extractall(destination, filter="data")
    record = json.loads((destination / "exp702/export.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != "exp702-guest-export-v2":
        raise LifecycleFailure("restored export schema mismatch")
    for key, expected in identity.items():
        if record.get(key) != expected:
            raise LifecycleFailure(f"restored export identity mismatch: {key}")
    restored = {"status": "restore-verified", "archive_sha256": digest, **identity,
                "export_kind": record.get("export_kind", "periodic")}
    if record.get("export_kind") == "final":
        summary_path = (destination / ".scratch/exp-702-exp701-terminal-evaluation-continuation"
                        / "terminal-summary.json")
        if not summary_path.is_file():
            raise LifecycleFailure("final export omits terminal-summary.json")
        summary_data = summary_path.read_bytes()
        summary = json.loads(summary_data)
        summary_sha256 = hashlib.sha256(summary_data).hexdigest()
        outcome = record.get("scientific_outcome")
        if (record.get("terminal_summary_sha256") != summary_sha256
                or summary.get("scientific_outcome") != outcome
                or outcome not in {"positive", "negative"}):
            raise LifecycleFailure("final export terminal summary mismatch")
        restored.update({"scientific_outcome": outcome,
                         "terminal_summary_sha256": summary_sha256,
                         "terminal_summary_verified": True})
    return restored


def _publish(transport: Transport, data: bytes, restored: dict[str, Any]) -> dict[str, Any]:
    identity = {key: restored[key] for key in
                ("pod_id", "volume_id", "owner_nonce", "sequence", "archive_sha256")}
    value = transport.publish_export(data=data, identity=identity)
    if value.get("durable") is not True:
        raise LifecycleFailure("off-Pod publication was not durably confirmed")
    return value


def run_lifecycle(*, transport: Transport, launch_dir: Path, receipt_path: Path,
                  suffix: str, maximum_total_usd: Decimal, maximum_provider_seconds: int,
                  cleanup_start_seconds: int, sync_interval_seconds: int, stale_seconds: int,
                  expected_gpu: str, maximum_rate: Decimal, volume_size_gb: int,
                  expected_data_center: str = "US-CA-2",
                  user_attested_balance_usd: Decimal = Decimal("10.00")) -> dict[str, Any]:
    """Execute exactly one admission/create/monitor/restore/cleanup lifecycle."""
    if (maximum_total_usd != Decimal("2.90") or maximum_rate > Decimal("1.39")
            or cleanup_start_seconds != 6_600 or maximum_provider_seconds != 7_200
            or sync_interval_seconds != 300 or stale_seconds != 720
            or user_attested_balance_usd < maximum_total_usd):
        raise LifecycleFailure("cleanup/deadline/monitor contract was weakened")
    package = json.loads((launch_dir / "package.json").read_text(encoding="utf-8"))
    lock_path, lock_fd = acquire_suffix_lock(launch_dir / ".locks", suffix)
    owner_nonce = hashlib.sha256(f"{suffix}:{secrets.token_hex(32)}".encode()).hexdigest()
    volume_name = f"sair-exp702-{suffix}-{owner_nonce[:12]}"
    pod_name = volume_name
    receipt: dict[str, Any] = {
        "schema_version": "exp702-lifecycle-v2", "suffix": suffix, "owner_nonce": owner_nonce,
        "intended_volume_name": volume_name, "intended_pod_name": pod_name,
        "status": "running", "transport": "fake" if isinstance(transport, FakeTransport) else "runpod-rest-s3",
        "provider_mutation_performed": not isinstance(transport, FakeTransport),
        "pod_id": None, "volume_id": None, "exports": [], "errors": [],
        "pod_delete_confirmed": False, "volume_delete_confirmed": False,
        "exact_absence_confirmed": False, "monitor_interval_seconds": sync_interval_seconds,
        "final_export_confirmed": False, "final_acknowledged": False,
        "terminal_status_observed": False,
        "controller_pid": os.getpid(),
        "controller_process_identity": f"{os.getpid()}:" + Path("/proc/self/stat").read_text().split()[21],
        "controller_started_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(receipt_path, receipt)
    restore_verified = False
    durable_publication = False
    pending: tuple[bytes, dict[str, Any], Path] | None = None
    try:
        transport.watchdog_ready(suffix=suffix, deadline_seconds=maximum_provider_seconds)
        receipt["watchdog_ready_before_create"] = True
        atomic_json(receipt_path, receipt)
        snapshot = transport.admission_snapshot()  # the one immediate live recheck
        if (snapshot.observed_at_seconds < 0 or snapshot.observed_at_seconds > 60
                or snapshot.pods or snapshot.volumes):
            raise LifecycleFailure("fresh REST inventory admission failed before creation")
        receipt["admission"] = {"observed_at_seconds": snapshot.observed_at_seconds,
                                "funding_basis": "user-attested",
                                "attested_balance_usd": str(user_attested_balance_usd),
                                "pods": [], "volumes": []}
        artifacts = {
            "exp702/inputs/exp702-source.tar.gz": launch_dir / package["source_archive"]["path"],
            "exp702/inputs/exp701-terminal-predecessor.tar.gz": launch_dir / package["predecessor_archive"]["path"],
            "exp702/inputs/torch-2.5.1+cu124-cp312-cp312-linux_x86_64.whl": launch_dir / package["torch_wheel"]["path"],
        }
        receipt["volume_id"] = transport.create_volume(name=volume_name, size_gb=volume_size_gb,
                                                        owner_nonce=owner_nonce)
        atomic_json(receipt_path, receipt)
        try:
            receipt["pod_id"] = transport.create_pod(name=pod_name, volume_id=receipt["volume_id"],
                                                     owner_nonce=owner_nonce)
        except (LifecycleFailure, OSError):
            baseline = {"status": "restore-verified", "kind": "empty-volume-before-pod",
                        "pod_id": None, "volume_id": receipt["volume_id"], "owner_nonce": owner_nonce,
                        "artifacts": {}}
            atomic_json(launch_dir / "offpod-restores" / suffix / "latest.json", baseline)
            receipt["exports"].append({**baseline, "downloaded_off_pod": True,
                                       "durably_published": True, "acknowledged": False})
            restore_verified = durable_publication = True
            raise
        atomic_json(receipt_path, receipt)
        empty_baseline = {"status": "restore-verified", "kind": "empty-quarantined-volume",
                          "pod_id": receipt["pod_id"], "volume_id": receipt["volume_id"],
                          "owner_nonce": owner_nonce, "artifacts": {}}
        atomic_json(launch_dir / "offpod-restores" / suffix / "latest.json", empty_baseline)
        receipt["exports"].append({**empty_baseline, "downloaded_off_pod": True,
                                   "durably_published": True, "acknowledged": False})
        restore_verified = durable_publication = True
        def record_returned_facts(value: dict[str, Any]) -> None:
            receipt.setdefault("returned_pod_fact_attempts", []).append(value)
            receipt["returned_pod_facts"] = value
            atomic_json(receipt_path, receipt)

        facts = transport.verify_quarantined_pod(
            pod_id=receipt["pod_id"], name=pod_name, volume_id=receipt["volume_id"],
            owner_nonce=owner_nonce,
            readiness_deadline_seconds=min(
                cleanup_start_seconds, transport.now_seconds + stale_seconds),
            fact_recorder=record_returned_facts)
        if (facts.pod_id != receipt["pod_id"] or facts.name != pod_name
                or facts.owner_nonce != owner_nonce or facts.volume_id != receipt["volume_id"]
                or (facts.gpu is not None and facts.gpu != expected_gpu)
                or (facts.gpu_count is not None and facts.gpu_count != 1)
                or facts.cloud_type not in {"SECURE", "COMMUNITY"} or facts.data_center_id != expected_data_center
                or (facts.interruptible is not None and facts.interruptible is not False)
                or facts.cost_per_hour_usd > maximum_rate
                or not facts.cost_per_hour_usd.is_finite() or facts.cost_per_hour_usd < 0
                or facts.quarantined is not True):
            raise LifecycleFailure("returned quarantined Pod facts failed exact verification")
        receipt["post_create_verification_passed"] = True
        atomic_json(receipt_path, receipt)
        transport.preload_volume(volume_id=receipt["volume_id"], artifacts=artifacts, owner_nonce=owner_nonce)
        baseline = {"status": "restore-verified", "kind": "pre-release-artifact-set",
                    "pod_id": receipt["pod_id"], "volume_id": receipt["volume_id"],
                    "owner_nonce": owner_nonce,
                    "artifacts": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for key, path in artifacts.items()}}
        atomic_json(launch_dir / "offpod-restores" / suffix / "latest.json", baseline)
        receipt["exports"].append({**baseline, "downloaded_off_pod": True,
                                   "durably_published": True, "acknowledged": False})
        restore_verified = durable_publication = True
        transport.release_pod(pod_id=receipt["pod_id"], volume_id=receipt["volume_id"],
                              owner_nonce=owner_nonce)
        receipt["pod_released_after_verification"] = True
        atomic_json(receipt_path, receipt)
        downloaded_sequences: set[int] = set()
        terminal = "running"
        while terminal == "running" and transport.now_seconds < cleanup_start_seconds:
            try:
                guest = transport.guest_state(pod_id=receipt["pod_id"], volume_id=receipt["volume_id"],
                                              owner_nonce=owner_nonce)
            except (LifecycleFailure, OSError, KeyError, ValueError) as error:
                if transport.now_seconds >= stale_seconds:
                    raise LifecycleFailure("guest readiness exceeded bounded retry window") from error
                transport.wait(15)
                continue
            terminal = str(guest.get("state"))
            if terminal.startswith("completed-"):
                receipt["terminal_status_observed"] = True
            if terminal == "failed":
                raise LifecycleFailure("guest reported failure")
            sequence_value = (guest.get("final_export") or {}).get("sequence") if terminal.startswith("completed-") else guest.get("latest_sequence")
            if sequence_value is None:
                if transport.now_seconds >= stale_seconds:
                    raise LifecycleFailure("export readiness exceeded bounded retry window")
                transport.wait(15)
                terminal = "running"
                continue
            sequence = int(sequence_value)
            if sequence in downloaded_sequences:
                if terminal == "running":
                    transport.wait(sync_interval_seconds)
                    continue
                raise LifecycleFailure("terminal status did not name a distinct final export")
            data = transport.download_export(pod_id=receipt["pod_id"], volume_id=receipt["volume_id"],
                                             owner_nonce=owner_nonce, sequence=sequence)
            retained = launch_dir / "offpod-downloads" / suffix / f"export-{sequence:04d}.tar.gz"
            retained.parent.mkdir(parents=True, exist_ok=True)
            temporary = retained.with_name(f".{retained.name}.tmp-{os.getpid()}")
            temporary.write_bytes(data); os.replace(temporary, retained)
            restore_dir = launch_dir / "offpod-restores" / suffix / f"export-{sequence:04d}"
            if restore_dir.exists():
                shutil.rmtree(restore_dir)
            restored = restore_verify(data, restore_dir, {"pod_id": receipt["pod_id"],
                                      "volume_id": receipt["volume_id"], "owner_nonce": owner_nonce,
                                      "sequence": sequence})
            downloaded_sequences.add(sequence)
            final_binding = guest.get("final_export") if terminal.startswith("completed-") else None
            if terminal.startswith("completed-") and not isinstance(final_binding, dict):
                raise LifecycleFailure("terminal status omits exact final export binding")
            if final_binding is not None:
                expected_final = {"pod_id": receipt["pod_id"], "volume_id": receipt["volume_id"],
                                  "owner_nonce": owner_nonce, "sequence": sequence,
                                  "archive_sha256": restored["archive_sha256"],
                                  "scientific_outcome": terminal.removeprefix("completed-")}
                if any(final_binding.get(key) != value for key, value in expected_final.items()):
                    raise LifecycleFailure("terminal status final export binding mismatch")
                if (restored.get("export_kind") != "final"
                        or restored.get("scientific_outcome") != expected_final["scientific_outcome"]
                        or restored.get("terminal_summary_verified") is not True):
                    raise LifecycleFailure("terminal final archive verification failed")
            restored["download_path"] = str(retained.resolve())
            atomic_json(launch_dir / "offpod-restores" / suffix / "latest.json", restored)
            restore_verified = True
            publication = None; failures: list[str] = []
            for _ in range(3):
                try:
                    publication = _publish(transport, data, restored); break
                except (LifecycleFailure, OSError) as error:
                    failures.append(str(error))
            if publication is None:
                pending = (data, restored, retained)
                raise LifecycleFailure("durable publication failed after three attempts: " + "; ".join(failures))
            durable_publication = True
            transport.acknowledge_export(pod_id=receipt["pod_id"], volume_id=receipt["volume_id"],
                                         owner_nonce=owner_nonce, sequence=sequence,
                                         archive_sha256=restored["archive_sha256"])
            restored.update({"acknowledged": True, "downloaded_off_pod": True,
                             "durably_published": True, "publication": publication})
            atomic_json(launch_dir / "offpod-restores" / suffix / "latest.json", restored)
            receipt["exports"].append(restored)
            if terminal.startswith("completed-"):
                receipt["final_export"] = expected_final
                receipt["final_export_confirmed"] = True
                receipt["final_acknowledged"] = True
            if terminal == "running":
                transport.wait(sync_interval_seconds)
        receipt["scientific_outcome"] = terminal.removeprefix("completed-") if terminal.startswith("completed-") else None
        receipt["status"] = ("completed" if terminal.startswith("completed-")
                             and receipt["final_export_confirmed"] and receipt["final_acknowledged"]
                             else "cleanup-deadline")
    except (LifecycleFailure, KeyboardInterrupt, OSError, KeyError, ValueError, tarfile.TarError) as error:
        receipt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        receipt["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        receipt["cleanup_started_at_seconds"] = min(transport.now_seconds, cleanup_start_seconds)
        pod_id = receipt.get("pod_id"); volume_id = receipt.get("volume_id")
        try:
            reconciled_pod, reconciled_volume = transport.reconcile_resources(
                pod_name=pod_name, volume_name=volume_name, owner_nonce=owner_nonce)
            if pod_id is not None and reconciled_pod not in (None, pod_id):
                raise LifecycleFailure("cleanup Pod reconciliation contradicted known identity")
            if volume_id is not None and reconciled_volume not in (None, volume_id):
                raise LifecycleFailure("cleanup volume reconciliation contradicted known identity")
            pod_id = pod_id or reconciled_pod
            volume_id = volume_id or reconciled_volume
            receipt["pod_id"] = pod_id
            receipt["volume_id"] = volume_id
            receipt["uncertain_create_reconciliation"] = {
                "performed": True, "pod_id_recovered": reconciled_pod,
                "volume_id_recovered": reconciled_volume,
            }
        except (LifecycleFailure, OSError, KeyError, ValueError) as error:
            receipt["errors"].append({"type": type(error).__name__, "message": str(error),
                                      "cleanup": "uncertain-create-reconciliation"})
        if not pod_id:
            receipt["pod_delete_confirmed"] = True
        while pod_id and transport.now_seconds <= maximum_provider_seconds:
            try:
                transport.delete_pod(pod_id=pod_id, owner_nonce=owner_nonce)
            except (LifecycleFailure, OSError) as error:
                receipt["errors"].append({"type": type(error).__name__, "message": str(error), "cleanup": "pod"})
            receipt["pod_delete_confirmed"] = pod_id not in transport.inventories()[0]
            if receipt["pod_delete_confirmed"]:
                break
            transport.wait(15)
        if pending is not None and receipt["pod_delete_confirmed"]:
            data, restored, retained = pending
            while transport.now_seconds <= maximum_provider_seconds:
                try:
                    publication = _publish(transport, retained.read_bytes(), restored)
                    if restored.get("export_kind") == "final":
                        transport.acknowledge_export(
                            pod_id=restored["pod_id"], volume_id=restored["volume_id"],
                            owner_nonce=restored["owner_nonce"], sequence=restored["sequence"],
                            archive_sha256=restored["archive_sha256"])
                        receipt["final_export_confirmed"] = True
                        receipt["final_acknowledged"] = True
                    restored.update({"acknowledged": restored.get("export_kind") == "final",
                                     "downloaded_off_pod": True,
                                     "durably_published": True, "publication": publication,
                                     "publication_recovered_after_pod_delete": True})
                    atomic_json(launch_dir / "offpod-restores" / suffix / "latest.json", restored)
                    receipt["exports"].append(restored); durable_publication = True; pending = None
                    break
                except (LifecycleFailure, OSError) as error:
                    receipt["errors"].append({"type": type(error).__name__, "message": str(error),
                                              "cleanup": "publication-recovery"})
                    transport.wait(15)
        completion_gate = not receipt["terminal_status_observed"] or (
            receipt["final_export_confirmed"] and receipt["final_acknowledged"])
        while (volume_id and restore_verified and durable_publication and completion_gate
               and receipt["pod_delete_confirmed"]
               and transport.now_seconds <= maximum_provider_seconds):
            try:
                transport.delete_volume(volume_id=volume_id, owner_nonce=owner_nonce)
            except (LifecycleFailure, OSError) as error:
                receipt["errors"].append({"type": type(error).__name__, "message": str(error), "cleanup": "volume"})
            receipt["volume_delete_confirmed"] = volume_id not in transport.inventories()[1]
            if receipt["volume_delete_confirmed"]:
                break
            transport.wait(15)
        receipt["volume_delete_restore_gated"] = True
        pods, volumes = transport.inventories()
        receipt["final_inventory"] = {"pods": list(pods), "volumes": list(volumes)}
        receipt["exact_absence_confirmed"] = not pods and not volumes
        receipt["finished_by_seconds"] = min(transport.now_seconds, maximum_provider_seconds)
        if not receipt["exact_absence_confirmed"]:
            receipt["status"] = "failed-cleanup"
        atomic_json(receipt_path, receipt)
        os.close(lock_fd); lock_path.unlink(missing_ok=True)
    return receipt


def provider_delete(*, transport: Transport, pod_id: str | None, volume_id: str | None,
                    owner_nonce: str, delete_pod: bool = False, delete_volume: bool = False,
                    restore_receipt: Path | None = None, require_pod_absence: bool = False,
                    require_volume_absence: bool = False, confirm_exact_absence: bool = False,
                    reconcile_uncertain_create: bool = False, pod_name: str | None = None,
                    volume_name: str | None = None) -> dict[str, Any]:
    """Watchdog-compatible identity-bound cleanup and exact absence check."""
    if reconcile_uncertain_create:
        if not pod_name or not volume_name:
            raise LifecycleFailure("uncertain-create reconciliation requires intended names")
        reconciled_pod, reconciled_volume = transport.reconcile_resources(
            pod_name=pod_name, volume_name=volume_name, owner_nonce=owner_nonce)
        pod_id = pod_id or reconciled_pod
        volume_id = volume_id or reconciled_volume
    if delete_pod:
        if not pod_id:
            raise LifecycleFailure("Pod deletion requires exact Pod ID")
        transport.delete_pod(pod_id=pod_id, owner_nonce=owner_nonce)
    if delete_volume:
        if not volume_id or restore_receipt is None:
            raise LifecycleFailure("volume deletion requires exact ID and restore receipt")
        restored = json.loads(restore_receipt.read_text(encoding="utf-8"))
        for key, expected in (("status", "restore-verified"), ("owner_nonce", owner_nonce),
                              ("pod_id", pod_id), ("volume_id", volume_id)):
            if restored.get(key) != expected:
                raise LifecycleFailure(f"restore-gated volume deletion mismatch: {key}")
        transport.delete_volume(volume_id=volume_id, owner_nonce=owner_nonce)
    pods, volumes = transport.inventories()
    if require_pod_absence and pod_id in pods:
        raise LifecycleFailure("Pod absence confirmation failed")
    if require_volume_absence and volume_id in volumes:
        raise LifecycleFailure("volume absence confirmation failed")
    if confirm_exact_absence and (pods or volumes):
        raise LifecycleFailure("exact zero-resource confirmation failed")
    return {"status": "absence-confirmed", "pod_id": pod_id, "volume_id": volume_id,
            "owner_nonce": owner_nonce, "pods": list(pods), "volumes": list(volumes),
            "exact_absence_confirmed": not pods and not volumes}
