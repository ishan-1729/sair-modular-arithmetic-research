"""Package and control EXP702 through dependency-injected provider transports."""

from __future__ import annotations

import argparse
from decimal import Decimal
import gzip
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any

import yaml

from experiments.exp_702_exp701_terminal_evaluation_continuation import run
from experiments.cloud_benchmarks.exp702_runpod import controller


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTROL = ROOT / "configs/exp_702_runpod_control.yaml"
DURABLE = ROOT / "experiments/runs" / run.RUN_ID
WHEEL_SHA256 = "bf6484bfe5bc4f92a4a1a1bf553041505e19a911f717065330eb061afe0e14d7"
WHEEL_BYTES = 908_202_405
OVERLAY_PATHS = (
    "configs/exp_702_exp701_terminal_evaluation_continuation.yaml",
    "configs/exp_702_runpod_control.yaml",
    "experiments/exp_702_exp701_terminal_evaluation_continuation/__init__.py",
    "experiments/exp_702_exp701_terminal_evaluation_continuation/run.py",
    "experiments/cloud_benchmarks/exp702_runpod/__init__.py",
    "experiments/cloud_benchmarks/exp702_runpod/orchestrate.py",
    "experiments/cloud_benchmarks/exp702_runpod/controller.py",
    "experiments/cloud_benchmarks/exp702_runpod/runpod_transport.py",
    "experiments/cloud_benchmarks/exp702_runpod/guest_supervisor.sh",
    "experiments/cloud_benchmarks/exp702_runpod/windows_deadline_watchdog.ps1",
    "experiments/cloud_benchmarks/exp702_runpod/windows_paid_lifecycle.ps1",
    "environments/cuda/bootstrap.sh",
    "environments/cuda/run-acceptance.sh",
    "environments/cuda/verify_environment.py",
    "experiments/cuda_portability/runtime.py",
    "environments/cuda/profiles.yaml",
    "environments/cuda/requirements-cu124.txt",
    "experiments/runs/exp-702-exp701-terminal-evaluation-continuation/preregistration.json",
    "tests/test_exp_702_exp701_terminal_evaluation_continuation.py",
    "tests/test_exp702_runpod_orchestrator.py",
)
CHECKPOINT_PATH = ".scratch/exp701-input/exp609-weights.safetensors"
RUNTIME_BINDING_LABELS = (
    "runner", "config", "control_config", "test", "orchestrator",
    "orchestrator_test", "controller", "runpod_transport", "guest_supervisor", "windows_watchdog",
    "windows_paid_launcher",
    "cuda_bootstrap", "cuda_acceptance", "cuda_environment_verifier",
    "cuda_runtime", "cuda_profiles", "canonical_cu124_lock",
    "source_checkpoint", "torch_cu124_wheel",
)


class PlanFailure(RuntimeError):
    """A local launch-review condition failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, *, display: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise PlanFailure(f"required launch artifact is absent: {path}")
    return {
        "path": display or str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_no_provider_mutation(flag: bool, control: dict[str, Any]) -> None:
    if not flag:
        raise PlanFailure("--no-provider-mutation is mandatory")
    if control.get("provider_mutation_allowed") is not False or control.get("credential_access_allowed") is not False:
        raise PlanFailure("EXP702 no-paid control policy was weakened")


def verify_wheel(path: Path) -> dict[str, Any]:
    value = identity(path)
    if value["sha256"] != WHEEL_SHA256 or value["bytes"] != WHEEL_BYTES:
        raise PlanFailure("preloaded cu124 wheel identity mismatch")
    return value


def _materialize(source: Path, destination: Path, expected_sha256: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) != expected_sha256:
        raise PlanFailure(f"existing launch artifact conflicts: {destination.name}")
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    result = identity(destination, display=destination.name)
    if result["sha256"] != expected_sha256:
        raise PlanFailure(f"published launch artifact identity mismatch: {destination.name}")
    return result


def _add_bytes(bundle: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    bundle.addfile(info, io.BytesIO(data))


def build_source_archive(destination: Path, preregistration: dict[str, Any]) -> dict[str, Any]:
    """Archive committed HEAD plus the exact reviewed EXP702 overlay.

    Dirty unrelated worktree bytes never enter the payload.  This is especially
    important for the immutable EXP701 boundary and concurrent EXP689 work.
    """
    head_tar = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    overlays = set(OVERLAY_PATHS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as output:
                with tarfile.open(fileobj=io.BytesIO(head_tar), mode="r:") as committed:
                    for member in sorted(committed.getmembers(), key=lambda item: item.name):
                        if not member.isfile() or member.name in overlays:
                            continue
                        if "exp_689" in member.name or "exp-689" in member.name:
                            raise PlanFailure("EXP689 is forbidden from the EXP702 payload")
                        handle = committed.extractfile(member)
                        if handle is not None:
                            _add_bytes(output, member.name, handle.read(), member.mode & 0o777)
                for relative in sorted(overlays):
                    path = ROOT / relative
                    if not path.is_file():
                        raise PlanFailure(f"EXP702 payload overlay is absent: {relative}")
                    mode = 0o755 if path.suffix in {".sh", ".ps1"} else 0o644
                    _add_bytes(output, relative, path.read_bytes(), mode)
                checkpoint = ROOT / CHECKPOINT_PATH
                checkpoint_id = identity(checkpoint, display=CHECKPOINT_PATH)
                expected = preregistration["bindings"]["source_checkpoint"]
                if checkpoint_id["sha256"] != expected["sha256"]:
                    raise PlanFailure("EXP702 source checkpoint binding mismatch")
                _add_bytes(output, CHECKPOINT_PATH, checkpoint.read_bytes())
                source_record = {
                    "schema_version": "exp702-source-v1",
                    "run_id": run.RUN_ID,
                    "repository_commit": head,
                    "challenge_commit": preregistration["challenge_source"]["commit"],
                    "bindings": preregistration["bindings"],
                }
                _add_bytes(output, ".sair-exp702-source.json", json.dumps(source_record, sort_keys=True).encode() + b"\n")
    return {**identity(destination, display=destination.name), "repository_commit": head}


def package_launch(control_path: Path, archive: Path, wheel: Path, output: Path, *, no_provider_mutation: bool) -> dict[str, Any]:
    control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    require_no_provider_mutation(no_provider_mutation, control)
    scientific = yaml.safe_load(run.CONFIG_PATH.read_text(encoding="utf-8"))
    verified = run.verify_predecessor(scientific, archive)
    preregistration = run.verify_bound_sources(scientific)
    wheel_id = verify_wheel(wheel)
    output.mkdir(parents=True, exist_ok=True)
    predecessor_copy = output / "exp701-terminal-predecessor.tar.gz"
    predecessor_id = _materialize(archive, predecessor_copy, scientific["predecessor"]["archive_sha256"])
    wheel_copy = output / "torch-2.5.1+cu124-cp312-cp312-linux_x86_64.whl"
    wheel_id = _materialize(wheel, wheel_copy, wheel_id["sha256"])
    source_id = build_source_archive(output / "exp702-source.tar.gz", preregistration)
    payload = {
        "schema_version": "exp702-dry-run-package-v1",
        "status": "packaged-no-provider-mutation",
        "run_id": run.RUN_ID,
        "provider_mutation_performed": False,
        "credential_read_performed": False,
        "source_archive": source_id,
        "predecessor_archive": predecessor_id,
        "predecessor_identity_sha256": verified["opened"]["identity_sha256"],
        "source_checkpoint": preregistration["bindings"]["source_checkpoint"],
        "canonical_environment_lock": preregistration["bindings"]["canonical_cu124_lock"],
        "torch_wheel": wheel_id,
        "challenge_source": preregistration["challenge_source"],
        "code_bindings": preregistration["bindings"],
        "authenticated_admission_schema": control["authenticated_admission_schema"],
        "required_runtime_binding_labels": list(RUNTIME_BINDING_LABELS),
        "guest_entrypoint": "experiments/cloud_benchmarks/exp702_runpod/guest_supervisor.sh",
        "controller_entrypoint": "experiments/cloud_benchmarks/exp702_runpod/controller.py",
        "transport_factory": "experiments.cloud_benchmarks.exp702_runpod.runpod_transport:create_transport",
        "offpod_publisher": "experiments.cloud_benchmarks.exp702_runpod.runpod_transport:AtomicDirectoryPublisher",
        "watchdog_entrypoint": "experiments/cloud_benchmarks/exp702_runpod/windows_deadline_watchdog.ps1",
        "windows_launcher_entrypoint": "experiments/cloud_benchmarks/exp702_runpod/windows_paid_lifecycle.ps1",
        "provider_delete_entrypoint": "python -m experiments.cloud_benchmarks.exp702_runpod.orchestrate provider-delete",
        "live_contract": {
            "cloudType": "COMMUNITY", "gpuTypeIds": [control["resource"]["gpu"]], "gpuCount": 1,
            "gpuTypePriority": control["authenticated_admission_schema"]["quarantined_pod"]["gpu_type_priority"],
            "interruptible": False,
            "dataCenterPriority": control["authenticated_admission_schema"]["quarantined_pod"]["data_center_priority"],
            "dataCenterIds": list(control["resource"]["pod_data_center_ids"]),
            "networkVolumeMountPath": control["authenticated_admission_schema"]["quarantined_pod"]["network_volume_mount_path"],
            "s3Endpoint": control["resource"]["s3_endpoint"],
            "maximum_pairs": 1, "image": control["runtime"]["image"],
            "guest_command": ["mkdir -p /workspace/sair-exp702/repository && tar -xzf /workspace/exp702/inputs/exp702-source.tar.gz -C /workspace/sair-exp702/repository && cd /workspace/sair-exp702/repository && bash experiments/cloud_benchmarks/exp702_runpod/guest_supervisor.sh"],
            "environment": {
                "SAIR_EXP702_SOURCE_SHA256": source_id["sha256"],
                "SAIR_EXP702_SYNC_SECONDS": str(control["persistence"]["offpod_sync_interval_seconds"]),
            },
            "cpu_fallback_allowed": False, "unbound_torch_download_allowed": False,
        },
    }
    atomic_json(output / "package.json", payload)
    atomic_json(DURABLE / "dry-run-package.json", payload)
    return payload


def audit_package(control_path: Path, launch_dir: Path, *, require_preloaded_wheel: bool,
                  require_all_runtime_bindings: bool, no_provider_mutation: bool) -> dict[str, Any]:
    control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    require_no_provider_mutation(no_provider_mutation, control)
    package_path = launch_dir / "package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get("provider_mutation_performed") is not False or package.get("credential_read_performed") is not False:
        raise PlanFailure("package does not prove the no-provider boundary")
    if package.get("authenticated_admission_schema") != control.get("authenticated_admission_schema"):
        raise PlanFailure("package authenticated admission schema differs from reviewed control")
    bindings = package.get("code_bindings", {})
    if require_all_runtime_bindings and set(RUNTIME_BINDING_LABELS) - set(bindings):
        raise PlanFailure("package omits required runtime bindings")
    source_path = launch_dir / package["source_archive"]["path"]
    if identity(source_path, display=source_path.name)["sha256"] != package["source_archive"]["sha256"]:
        raise PlanFailure("source archive changed after packaging")
    with tarfile.open(source_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        if any("exp_689" in name or "exp-689" in name for name in members):
            raise PlanFailure("EXP689 is present in the launch archive")
        for label in RUNTIME_BINDING_LABELS:
            spec = bindings[label]
            if label == "torch_cu124_wheel":
                continue
            member = members.get(spec["path"])
            if member is None or not member.isfile():
                raise PlanFailure(f"bound runtime member is absent: {label}")
            handle = archive.extractfile(member)
            data = b"" if handle is None else handle.read()
            if len(data) != int(spec["bytes"]) or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise PlanFailure(f"bound runtime member mismatch: {label}")
    wheel_path = launch_dir / package["torch_wheel"]["path"]
    if require_preloaded_wheel and verify_wheel(wheel_path)["sha256"] != package["torch_wheel"]["sha256"]:
        raise PlanFailure("preloaded cu124 wheel is not launch-consumable")
    predecessor_path = launch_dir / package["predecessor_archive"]["path"]
    if sha256(predecessor_path) != package["predecessor_archive"]["sha256"]:
        raise PlanFailure("predecessor launch copy changed")
    result = {
        "schema_version": "exp702-package-audit-v1", "status": "audited-no-provider-mutation",
        "run_id": run.RUN_ID, "provider_mutation_performed": False,
        "credential_read_performed": False, "package_sha256": sha256(package_path),
        "source_archive_sha256": package["source_archive"]["sha256"],
        "runtime_bindings_verified": list(RUNTIME_BINDING_LABELS),
        "preloaded_wheel_verified": require_preloaded_wheel,
    }
    atomic_json(launch_dir / "audit.json", result)
    return result


def plan_launch(control_path: Path, launch_dir: Path, hourly_rate: Decimal, maximum: Decimal,
                seconds: int, cleanup_start_seconds: int = 12_600, *, no_provider_mutation: bool) -> dict[str, Any]:
    control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    require_no_provider_mutation(no_provider_mutation, control)
    package = json.loads((launch_dir / "package.json").read_text(encoding="utf-8"))
    if package.get("status") != "packaged-no-provider-mutation" or package.get("provider_mutation_performed") is not False:
        raise PlanFailure("launch package is not a no-provider artifact")
    budget = control["budget"]
    if maximum != Decimal(str(budget["maximum_total_usd"])) or seconds != int(budget["maximum_provider_seconds"]):
        raise PlanFailure("requested lifecycle ceiling differs from the reviewed control")
    if hourly_rate != Decimal(str(budget["compute_usd_per_hour"])):
        raise PlanFailure("requested compute rate differs from the reviewed control")
    if cleanup_start_seconds != int(budget["cleanup_start_seconds"]) or cleanup_start_seconds > 12_600:
        raise PlanFailure("requested proactive cleanup reserve differs from the reviewed control")
    audit_package(control_path, launch_dir, require_preloaded_wheel=True,
                  require_all_runtime_bindings=True, no_provider_mutation=True)
    phases = {key: int(value) for key, value in budget["phases_seconds"].items()}
    if sum(phases.values()) != seconds:
        raise PlanFailure("lifecycle phase admission does not cover the full provider lifetime")
    combined_rate = hourly_rate + Decimal(str(budget["storage_usd_per_hour_upper"]))
    admitted = combined_rate * Decimal(seconds) / Decimal(3600)
    if admitted > maximum:
        raise PlanFailure("rate-bound compute plus storage exceeds the total ceiling")
    plan = {
        "schema_version": "exp702-dry-run-plan-v1",
        "status": "executable-fake-proven-no-provider-mutation",
        "run_id": run.RUN_ID,
        "provider_mutation_performed": False,
        "credential_read_performed": False,
        "maximum_total_usd": f"{maximum:.2f}",
        "maximum_provider_seconds": seconds,
        "combined_hourly_rate_upper_usd": f"{combined_rate:.6f}",
        "rate_bound_total_upper_usd": f"{admitted:.6f}",
        "admitted_phases_seconds": phases,
        "single_resource_lock": {"required": True, "maximum_pairs": 1, "owner": "windows-host-controller"},
        "watchdog": {"independent": True, "id_bound": True, "ready_handshake_before_first_create": True,
                     "uncertain_create_reconciliation": "suffix-name-owner-nonce",
                     "cleanup_start_seconds": cleanup_start_seconds, "deadline_seconds": seconds,
                     "script": control["lifecycle"]["independent_watchdog"]},
        "launch_contract": {
            "review_only": False,
            "transport": package["transport_factory"],
            "offpod_publisher": package["offpod_publisher"],
            "credential_access_after_exact_authorization_only": True,
            "resource_selector": control["resource"]["gpu"],
            "cloud_type": control["resource"]["cloud_type"],
            "network_volume_data_center_id": control["resource"]["data_center_id"],
            "data_center_ids": package["live_contract"].get("dataCenterIds", []),
            "data_center_priority": package["live_contract"]["dataCenterPriority"],
            "gpu_type_priority": package["live_contract"]["gpuTypePriority"],
            "gpu_count": package["live_contract"]["gpuCount"],
            "interruptible": package["live_contract"]["interruptible"],
            "network_volume_mount_path": package["live_contract"]["networkVolumeMountPath"],
            "s3_endpoint": control["resource"]["s3_endpoint"],
            "image": control["runtime"]["image"],
            "artifacts": {
                "/workspace/exp702/inputs/exp702-source.tar.gz": package["source_archive"]["sha256"],
                "/workspace/exp702/inputs/exp701-terminal-predecessor.tar.gz": package["predecessor_archive"]["sha256"],
                "/workspace/exp702/inputs/torch-2.5.1+cu124-cp312-cp312-linux_x86_64.whl": package["torch_wheel"]["sha256"],
            },
            "guest_command": package["live_contract"]["guest_command"],
            "environment": package["live_contract"]["environment"],
            "provider_credential_in_guest": False,
        },
        "authenticated_admission": {
            **package["authenticated_admission_schema"],
            "one_immediate_observation_before_create": True,
            "configuration_substitution_allowed": False,
            "post_create_rest_fact_verification_before_preload_or_release": True,
        },
        "persistence": {
            "sync_interval_seconds": int(control["persistence"]["offpod_sync_interval_seconds"]),
            "initial_restore_verified_export_required": True,
            "periodic_restore_verified_offpod_exports_required": True,
            "final_restore_verified_export_required": True,
            "publication_retry_safe": True,
        },
        "cleanup": {
            "start_seconds": cleanup_start_seconds,
            "pod_delete_confirmation_required": True,
            "volume_delete_confirmation_required": True,
            "volume_delete_blocked_until_offpod_restore_verified": True,
            "billing_ends_only_after_exact_absence_confirmation": True,
            "retry_window_seconds": [cleanup_start_seconds, seconds],
            "volume_only_partial_create_restore_gated": True,
        },
        "package_sha256": sha256(launch_dir / "package.json"),
        "package_audit_sha256": sha256(launch_dir / "audit.json"),
        "source_archive_sha256": package["source_archive"]["sha256"],
        "predecessor_archive_sha256": package["predecessor_archive"]["sha256"],
        "torch_wheel_sha256": package["torch_wheel"]["sha256"],
    }
    atomic_json(launch_dir / "plan.json", plan)
    atomic_json(DURABLE / "dry-run-plan.json", plan)
    return plan


def simulate_lifecycle(control_path: Path, launch_dir: Path, suffix: str, maximum: Decimal,
                       seconds: int, cleanup_start_seconds: int, *, fake_provider: bool,
                       no_provider_mutation: bool) -> dict[str, Any]:
    control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    require_no_provider_mutation(no_provider_mutation, control)
    if not fake_provider:
        raise PlanFailure("--fake-provider is mandatory for no-provider fixture simulation")
    budget = control["budget"]
    if maximum != Decimal(str(budget["maximum_total_usd"])) or seconds != int(budget["maximum_provider_seconds"]):
        raise PlanFailure("simulation ceiling differs from reviewed control")
    if cleanup_start_seconds != int(budget["cleanup_start_seconds"]) or cleanup_start_seconds > 12_600:
        raise PlanFailure("simulation cleanup start differs from reviewed control")
    audit_package(control_path, launch_dir, require_preloaded_wheel=True,
                  require_all_runtime_bindings=True, no_provider_mutation=True)
    scenarios = ("positive", "negative", "interruption", "stale-ack", "export-failure",
                 "guest-failure", "deletion-failure", "publication-failure", "partial-create",
                 "publication-retry", "retry")
    receipts: dict[str, Any] = {}
    for scenario in scenarios:
        transport = controller.FakeTransport(scenario="negative" if scenario == "retry" else scenario)
        scenario_suffix = f"{suffix}-{scenario}"
        path = launch_dir / "lifecycle-receipts" / f"{scenario_suffix}.json"
        receipt = controller.run_lifecycle(
            transport=transport, launch_dir=launch_dir, receipt_path=path,
            suffix=scenario_suffix, maximum_total_usd=maximum,
            maximum_provider_seconds=seconds,
            cleanup_start_seconds=cleanup_start_seconds,
            sync_interval_seconds=int(control["persistence"]["offpod_sync_interval_seconds"]),
            stale_seconds=int(control["persistence"]["offpod_ack_stale_seconds"]),
            expected_gpu=control["resource"]["gpu"],
            maximum_rate=Decimal(str(budget["compute_usd_per_hour"])),
            volume_size_gb=int(control["resource"]["network_volume_gb"]),
        )
        if scenario == "retry":
            before = path.read_bytes()
            atomic_json(path, receipt)
            receipt["retry_safe_publication_verified"] = json.loads(before) == json.loads(path.read_bytes())
        receipt["fake_events"] = transport.events
        receipts[scenario] = receipt
    successful = all(receipts[name]["status"] == "completed" for name in
                     ("positive", "negative", "publication-retry", "retry"))
    honest_failures = all(receipts[name]["errors"] for name in
                          ("interruption", "stale-ack", "export-failure", "guest-failure",
                           "deletion-failure", "publication-failure"))
    honest_failures = honest_failures and bool(receipts["partial-create"]["errors"])
    result = {
        "schema_version": "exp702-fake-lifecycle-suite-v1",
        "status": "fake-lifecycle-proven" if successful and honest_failures else "fake-lifecycle-failed",
        "run_id": run.RUN_ID, "provider_mutation_performed": False,
        "credential_read_performed": False, "suffix": suffix, "scenarios": receipts,
    }
    atomic_json(launch_dir / "lifecycle.json", result)
    atomic_json(DURABLE / "dry-run-lifecycle.json", result)
    atomic_json(DURABLE / "dry-run-live-controller.json", result)
    if result["status"] != "fake-lifecycle-proven":
        raise PlanFailure("fake lifecycle suite did not prove every required outcome")
    return result


def load_transport(factory_spec: str, *, control: dict[str, Any], launch_dir: Path) -> controller.Transport:
    """Load an operator-supplied factory without endpoint or credential discovery."""
    if ":" not in factory_spec:
        raise PlanFailure("transport factory must be module:function")
    module_name, attribute = factory_spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    package = json.loads((launch_dir / "package.json").read_text(encoding="utf-8"))
    transport = factory(control=control, launch_dir=launch_dir, package=package)
    if isinstance(transport, controller.FakeTransport):
        raise PlanFailure("execute-live rejects FakeTransport")
    return transport


def verify_live_authorization(path: Path, *, suffix: str, launch_dir: Path,
                              control: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "approved": True, "run_id": run.RUN_ID, "single_lifecycle_suffix": suffix,
        "package_sha256": sha256(launch_dir / "package.json"),
        "maximum_total_usd": f"{Decimal(str(control['budget']['maximum_total_usd'])):.2f}",
        "maximum_provider_seconds": int(control["budget"]["maximum_provider_seconds"]),
        "cleanup_start_seconds": int(control["budget"]["cleanup_start_seconds"]),
    }
    for key, required in expected.items():
        if value.get(key) != required:
            raise PlanFailure(f"live authorization mismatch: {key}")
    return value


def execute_live(control_path: Path, launch_dir: Path, suffix: str, factory_spec: str,
                 receipt_path: Path, authorization_receipt: Path) -> dict[str, Any]:
    control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    audit_package(control_path, launch_dir, require_preloaded_wheel=True,
                  require_all_runtime_bindings=True, no_provider_mutation=True)
    verify_live_authorization(authorization_receipt, suffix=suffix, launch_dir=launch_dir, control=control)
    budget = control["budget"]
    transport = load_transport(factory_spec, control=control, launch_dir=launch_dir)
    return controller.run_lifecycle(
        transport=transport, launch_dir=launch_dir, receipt_path=receipt_path, suffix=suffix,
        maximum_total_usd=Decimal(str(budget["maximum_total_usd"])),
        maximum_provider_seconds=int(budget["maximum_provider_seconds"]),
        cleanup_start_seconds=int(budget["cleanup_start_seconds"]),
        sync_interval_seconds=int(control["persistence"]["offpod_sync_interval_seconds"]),
        stale_seconds=int(control["persistence"]["offpod_ack_stale_seconds"]),
        expected_gpu=control["resource"]["gpu"],
        maximum_rate=Decimal(str(budget["compute_usd_per_hour"])),
        volume_size_gb=int(control["resource"]["network_volume_gb"]),
        expected_data_center=control["resource"]["data_center_id"],
        user_attested_balance_usd=Decimal(str(
            control["authenticated_admission_schema"]["funding"]["attested_balance_usd"])),
    )


def delete_live(control_path: Path, launch_dir: Path, factory_spec: str,
                authorization_receipt: Path, lifecycle_suffix: str,
                controller_receipt: Path, **operation: Any) -> dict[str, Any]:
    control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    verify_live_authorization(authorization_receipt, suffix=lifecycle_suffix,
                              launch_dir=launch_dir, control=control)
    ownership = json.loads(controller_receipt.read_text(encoding="utf-8"))
    expected = {"suffix": lifecycle_suffix, "owner_nonce": operation.get("owner_nonce")}
    for key, value in expected.items():
        if ownership.get(key) != value:
            raise PlanFailure(f"provider-delete controller receipt mismatch: {key}")
    for key in ("pod_id", "volume_id"):
        supplied = operation.get(key)
        if supplied is not None and ownership.get(key) not in (None, supplied):
            raise PlanFailure(f"provider-delete controller receipt mismatch: {key}")
    operation["pod_name"] = ownership.get("intended_pod_name")
    operation["volume_name"] = ownership.get("intended_volume_name")
    transport = load_transport(factory_spec, control=control, launch_dir=launch_dir)
    return controller.provider_delete(transport=transport, **operation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    package.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL)
    package.add_argument("--archive", type=Path, required=True)
    package.add_argument("--wheel", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--no-provider-mutation", action="store_true")
    plan = sub.add_parser("plan")
    plan.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL)
    plan.add_argument("--launch-dir", type=Path, required=True)
    plan.add_argument("--hourly-rate-usd", type=Decimal, required=True)
    plan.add_argument("--max-total-usd", type=Decimal, required=True)
    plan.add_argument("--max-provider-seconds", type=int, required=True)
    plan.add_argument("--cleanup-start-seconds", type=int, required=True)
    plan.add_argument("--no-provider-mutation", action="store_true")
    audit = sub.add_parser("audit-package")
    audit.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL)
    audit.add_argument("--launch-dir", type=Path, required=True)
    audit.add_argument("--require-preloaded-wheel", action="store_true")
    audit.add_argument("--require-all-runtime-bindings", action="store_true")
    audit.add_argument("--no-provider-mutation", action="store_true")
    simulate = sub.add_parser("simulate-lifecycle", aliases=["simulate-live-controller"])
    simulate.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL)
    simulate.add_argument("--launch-dir", type=Path, required=True)
    simulate.add_argument("--single-lifecycle-suffix", required=True)
    simulate.add_argument("--max-total-usd", type=Decimal, required=True)
    simulate.add_argument("--max-provider-seconds", type=int, required=True)
    simulate.add_argument("--cleanup-start-seconds", type=int, required=True)
    simulate.add_argument("--fake-provider", action="store_true")
    simulate.add_argument("--no-provider-mutation", action="store_true")
    live = sub.add_parser("execute-live")
    live.add_argument("--control-config", type=Path, required=True)
    live.add_argument("--launch-dir", type=Path, required=True)
    live.add_argument("--single-lifecycle-suffix", required=True)
    live.add_argument("--transport-factory", required=True)
    live.add_argument("--receipt", type=Path, required=True)
    live.add_argument("--authorization-receipt", type=Path, required=True)
    delete = sub.add_parser("provider-delete")
    delete.add_argument("--control-config", type=Path, required=True)
    delete.add_argument("--launch-dir", type=Path, required=True)
    delete.add_argument("--transport-factory", required=True)
    delete.add_argument("--authorization-receipt", type=Path, required=True)
    delete.add_argument("--single-lifecycle-suffix", required=True)
    delete.add_argument("--controller-receipt", type=Path, required=True)
    delete.add_argument("--pod-id")
    delete.add_argument("--volume-id")
    delete.add_argument("--owner-nonce", required=True)
    delete.add_argument("--restore-receipt", type=Path)
    delete.add_argument("--delete-pod", action="store_true")
    delete.add_argument("--delete-volume", action="store_true")
    delete.add_argument("--require-pod-absence", action="store_true")
    delete.add_argument("--require-volume-absence", action="store_true")
    delete.add_argument("--confirm-exact-absence", action="store_true")
    delete.add_argument("--reconcile-uncertain-create", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "package":
            result = package_launch(args.control_config, args.archive, args.wheel, args.output, no_provider_mutation=args.no_provider_mutation)
        elif args.command == "audit-package":
            result = audit_package(args.control_config, args.launch_dir,
                                   require_preloaded_wheel=args.require_preloaded_wheel,
                                   require_all_runtime_bindings=args.require_all_runtime_bindings,
                                   no_provider_mutation=args.no_provider_mutation)
        elif args.command == "plan":
            result = plan_launch(args.control_config, args.launch_dir, args.hourly_rate_usd,
                                 args.max_total_usd, args.max_provider_seconds,
                                 args.cleanup_start_seconds, no_provider_mutation=args.no_provider_mutation)
        elif args.command in {"simulate-lifecycle", "simulate-live-controller"}:
            result = simulate_lifecycle(args.control_config, args.launch_dir,
                                        args.single_lifecycle_suffix, args.max_total_usd,
                                        args.max_provider_seconds, args.cleanup_start_seconds,
                                        fake_provider=args.fake_provider,
                                        no_provider_mutation=args.no_provider_mutation)
        elif args.command == "execute-live":
            result = execute_live(args.control_config, args.launch_dir, args.single_lifecycle_suffix,
                                  args.transport_factory, args.receipt, args.authorization_receipt)
        else:
            result = delete_live(
                args.control_config, args.launch_dir, args.transport_factory, args.authorization_receipt,
                args.single_lifecycle_suffix, args.controller_receipt,
                pod_id=args.pod_id, volume_id=args.volume_id, owner_nonce=args.owner_nonce,
                delete_pod=args.delete_pod, delete_volume=args.delete_volume,
                restore_receipt=args.restore_receipt,
                require_pod_absence=args.require_pod_absence,
                require_volume_absence=args.require_volume_absence,
                confirm_exact_absence=args.confirm_exact_absence,
                reconcile_uncertain_create=args.reconcile_uncertain_create,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, PlanFailure, controller.LifecycleFailure,
            run.SemanticFailure, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "blocked-no-provider-mutation", "reason": str(error), "provider_mutation_performed": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
