"""RunPod REST resource and datacenter S3 transport for EXP702.

The factory is the only credential boundary.  ``orchestrate.execute_live`` calls
it only after package audit and exact authorization verification.  HTTP and S3
clients are injected in tests, so no verification command needs network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from typing import Any, Callable, Mapping
from urllib import error as urlerror, parse, request

from .controller import AdmissionSnapshot, LifecycleFailure, PodFacts


RUNPOD_REST_API = "https://rest.runpod.io/v1"
MAX_HTTP_ERROR_BODY_BYTES = 4096
MAX_HTTP_ERROR_TEXT_CHARS = 512
REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-amzn-requestid", "cf-ray")
class RunPodTransportError(LifecycleFailure):
    """A concrete provider or S3 operation failed closed."""


class RunPodHTTPError(RunPodTransportError):
    """A bounded, sanitized provider HTTP failure safe for durable receipts."""

    def __init__(self, method: str, path: str, status: int, diagnostic: dict[str, Any]):
        self.method = method
        self.path = path
        self.status = status
        self.diagnostic = diagnostic
        super().__init__(
            f"RunPod {method} {path} failed: "
            + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )


ExchangeResult = tuple[int, bytes] | tuple[int, bytes, Mapping[str, str]]
Exchange = Callable[[str, str, dict[str, str], bytes | None], ExchangeResult]


def _bounded_http_diagnostic(status: int, raw: bytes,
                             headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    bounded = raw[:MAX_HTTP_ERROR_BODY_BYTES]
    result: dict[str, Any] = {
        "schema_version": "exp702-sanitized-runpod-http-error-v1",
        "http_status": int(status),
        "body_bytes_observed": len(raw),
        "body_truncated": len(raw) > MAX_HTTP_ERROR_BODY_BYTES,
        "bounded_body_sha256": hashlib.sha256(bounded).hexdigest(),
    }
    try:
        value = json.loads(bounded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key in ("error", "message", "status", "code"):
            item = value.get(key)
            if type(item) is str:
                sanitized[key] = item[:MAX_HTTP_ERROR_TEXT_CHARS]
            elif type(item) in (int, float) and type(item) is not bool:
                sanitized[key] = item
        if sanitized:
            result["body"] = sanitized
    request_ids: dict[str, str] = {}
    for key, value in (headers or {}).items():
        normalized = str(key).lower()
        candidate = str(value)
        if (normalized in REQUEST_ID_HEADERS and len(candidate) <= 128
                and re.fullmatch(r"[A-Za-z0-9._:/-]+", candidate)):
            request_ids[normalized] = candidate
    if request_ids:
        result["request_ids"] = dict(sorted(request_ids.items()))
    return result


def canonical_pod_create_payload(*, name: str, gpu: str, data_center_id: str | None,
                                 image: str, command: str, volume_id: str,
                                 environment: dict[str, str]) -> dict[str, Any]:
    """Build the single source-backed RunPod REST Pod-create request."""
    strings = {"name": name, "gpu": gpu, "image": image,
               "command": command, "volume_id": volume_id}
    if any(type(value) is not str or not value for value in strings.values()):
        raise RunPodTransportError("canonical Pod-create inputs must be non-empty strings")
    if (type(environment) is not dict
            or any(type(key) is not str or not key or type(value) is not str
                   for key, value in environment.items())):
        raise RunPodTransportError("canonical Pod-create environment must be string-to-string")
    payload = {
        "name": name, "imageName": image, "computeType": "GPU",
        "gpuTypeIds": [gpu], "gpuCount": 1, "gpuTypePriority": "availability",
        "cloudType": "COMMUNITY", "dataCenterPriority": "availability", "interruptible": False,
        "containerDiskInGb": 20, "networkVolumeId": volume_id,
        "volumeMountPath": "/workspace", "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [command], "env": dict(sorted(environment.items())),
        "locked": False,
    }
    if data_center_id is not None:
        payload["dataCenterIds"] = [data_center_id]
    return payload


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def urllib_exchange(method: str, url: str, headers: dict[str, str], body: bytes | None) -> ExchangeResult:
    outbound = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(outbound, timeout=30) as response:  # nosec B310: fixed RunPod endpoint
            return int(response.status), response.read(), dict(response.headers.items())
    except urlerror.HTTPError as failure:
        return int(failure.code), failure.read(MAX_HTTP_ERROR_BODY_BYTES + 1), dict(failure.headers.items())


@dataclass
class RunPodRESTAPI:
    """Small adapter over RunPod's supported REST v1 Pod/volume schemas."""

    api_key: str
    exchange: Exchange = urllib_exchange
    endpoint: str = RUNPOD_REST_API

    def _json(self, method: str, path: str, payload: dict[str, Any] | None = None,
              *, allow_not_found: bool = False) -> Any:
        body = None if payload is None else _canonical_json_bytes(payload)
        exchanged = self.exchange(method, self.endpoint.rstrip("/") + path, {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        }, body)
        status, raw = exchanged[:2]
        response_headers = exchanged[2] if len(exchanged) == 3 else {}
        if allow_not_found and status == 404:
            return None
        if status < 200 or status >= 300:
            raise RunPodHTTPError(method, path, status,
                                  _bounded_http_diagnostic(status, raw, response_headers))
        return {} if not raw else json.loads(raw)

    def list_pods(self) -> list[dict[str, Any]]:
        value = self._json("GET", "/pods?" + parse.urlencode({"includeMachine": "true"}))
        if not isinstance(value, list):
            raise RunPodTransportError("RunPod Pod inventory response is not a list")
        return value

    def list_network_volumes(self) -> list[dict[str, Any]]:
        value = self._json("GET", "/networkvolumes")
        if not isinstance(value, list):
            raise RunPodTransportError("RunPod network-volume inventory response is not a list")
        return value

    def create_network_volume(self, *, data_center_id: str, name: str, size_gb: int) -> dict[str, Any]:
        return self._json("POST", "/networkvolumes", {
            "dataCenterId": data_center_id, "name": name, "size": size_gb,
        })

    def create_pod(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/pods", payload)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        query = parse.urlencode({"includeMachine": "true", "includeNetworkVolume": "true"})
        value = self._json("GET", "/pods/" + parse.quote(pod_id, safe="") + "?" + query)
        if not isinstance(value, dict):
            raise RunPodTransportError("RunPod returned Pod response is not an object")
        return value

    def delete_pod(self, pod_id: str) -> None:
        self._json("DELETE", "/pods/" + parse.quote(pod_id, safe=""), allow_not_found=True)

    def delete_network_volume(self, volume_id: str) -> None:
        self._json("DELETE", "/networkvolumes/" + parse.quote(volume_id, safe=""), allow_not_found=True)


def _safe_key(value: str) -> str:
    key = PurePosixPath(value)
    if key.is_absolute() or ".." in key.parts or not key.parts:
        raise RunPodTransportError("S3 object key is not relative and canonical")
    return str(key)


@dataclass
class RunPodS3:
    """Path-style S3 object contract for a RunPod network-volume bucket."""

    client: Any
    endpoint: str

    def put_file_verified(self, source: Path, *, bucket: str, key: str) -> dict[str, Any]:
        key = _safe_key(key)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.client.upload_file(str(source), bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
        response = self.client.get_object(Bucket=bucket, Key=key)
        downloaded = response["Body"].read()
        if len(downloaded) != source.stat().st_size or hashlib.sha256(downloaded).hexdigest() != digest:
            raise RunPodTransportError("RunPod S3 upload verification failed")
        return {"key": key, "sha256": digest, "bytes": len(downloaded), "status": "verified"}

    def get_bytes(self, *, bucket: str, key: str) -> bytes:
        return bytes(self.client.get_object(Bucket=bucket, Key=_safe_key(key))["Body"].read())

    def put_json(self, *, bucket: str, key: str, value: dict[str, Any]) -> None:
        self.client.put_object(Bucket=bucket, Key=_safe_key(key),
                               Body=(json.dumps(value, sort_keys=True) + "\n").encode(),
                               ContentType="application/json")

    def get_json(self, *, bucket: str, key: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(bucket=bucket, key=key))


@dataclass
class AtomicDirectoryPublisher:
    root: Path

    def publish(self, *, data: bytes, identity: dict[str, Any]) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        if digest != identity.get("archive_sha256"):
            raise RunPodTransportError("off-Pod publication digest mismatch")
        destination = self.root / str(identity["owner_nonce"]) / f"export-{int(identity['sequence']):04d}.tar.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        temporary.write_bytes(data)
        os.replace(temporary, destination)
        receipt = {**identity, "durable": True, "bytes": len(data), "path": str(destination.resolve())}
        receipt_path = destination.with_suffix(destination.suffix + ".json")
        temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.tmp-{os.getpid()}")
        temporary_receipt.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_receipt, receipt_path)
        return receipt


@dataclass
class RunPodSecureTransport:
    api: RunPodRESTAPI
    s3: RunPodS3
    publisher: AtomicDirectoryPublisher
    package: dict[str, Any]
    launch_dir: Path
    data_center_id: str
    expected_gpu: str
    maximum_rate: Decimal
    monotonic_seconds: Callable[[], int]
    sleep: Callable[[float], None]

    @property
    def now_seconds(self) -> int:
        return int(self.monotonic_seconds())

    @staticmethod
    def _nonce(item: dict[str, Any]) -> str | None:
        env = item.get("env") or {}
        if isinstance(env, dict):
            value = env.get("SAIR_EXP702_OWNER_NONCE")
            return None if value is None else str(value)
        return None

    def watchdog_ready(self, *, suffix: str, deadline_seconds: int) -> None:
        ready_path = self.launch_dir / "watchdog" / f"{suffix}.ready.json"
        # Windows PowerShell 5.1's historical ``-Encoding UTF8`` writer emits a
        # UTF-8 BOM.  New receipts are BOM-free, but accept exactly one optional
        # UTF-8 BOM so an otherwise valid ready receipt remains interoperable.
        value = json.loads(ready_path.read_bytes().decode("utf-8-sig"))
        if value != {"schema_version": "exp702-watchdog-ready-v1", "status": "ready",
                     "suffix": suffix, "deadline_seconds": deadline_seconds}:
            raise RunPodTransportError("watchdog ready receipt mismatch")

    def admission_snapshot(self) -> AdmissionSnapshot:
        pods = tuple(sorted(str(item["id"]) for item in self.api.list_pods()))
        volumes = tuple(sorted(str(item["id"]) for item in self.api.list_network_volumes()))
        if pods or volumes:
            raise RunPodTransportError("authenticated REST resource inventory is not exactly zero")
        return AdmissionSnapshot(self.now_seconds, pods, volumes)

    def _find_volume(self, name: str) -> str | None:
        matches = [str(item["id"]) for item in self.api.list_network_volumes()
                   if item.get("name") == name and item.get("dataCenterId") == self.data_center_id]
        if len(matches) > 1:
            raise RunPodTransportError("ambiguous volume reconciliation")
        return matches[0] if matches else None

    def _find_pod(self, name: str, owner_nonce: str) -> str | None:
        matches = [str(item["id"]) for item in self.api.list_pods()
                   if item.get("name") == name and self._nonce(item) == owner_nonce]
        if len(matches) > 1:
            raise RunPodTransportError("ambiguous Pod reconciliation")
        return matches[0] if matches else None

    def create_volume(self, *, name: str, size_gb: int, owner_nonce: str) -> str:
        try:
            result = self.api.create_network_volume(data_center_id=self.data_center_id, name=name, size_gb=size_gb)
            return str(result["id"])
        except (OSError, KeyError, RunPodTransportError):
            reconciled = self._find_volume(name)
            if reconciled is None:
                raise
            return reconciled

    def preload_volume(self, *, volume_id: str, artifacts: dict[str, Path], owner_nonce: str) -> None:
        for key, source in sorted(artifacts.items()):
            self.s3.put_file_verified(source, bucket=volume_id, key=key)

    def create_pod(self, *, name: str, volume_id: str, owner_nonce: str) -> str:
        contract = self.package["live_contract"]
        expected_shape = {
            "gpuTypePriority": "availability",
            "dataCenterPriority": "availability",
            "networkVolumeMountPath": "/workspace",
        }
        if any(contract.get(key) != value for key, value in expected_shape.items()):
            raise RunPodTransportError("package Pod-create priority or attached-volume binding mismatch")
        guest_command = " ".join(contract["guest_command"])
        release_path = f"/workspace/exp702/control/{owner_nonce}.release.json"
        command = ("set -eu; release=" + release_path
                   + "; while [ ! -f \"$release\" ]; do sleep 1; done; "
                   + "exec bash -lc \"$SAIR_EXP702_GUEST_COMMAND\"")
        environment = {**contract["environment"], "SAIR_EXP702_OWNER_NONCE": owner_nonce,
                       "SAIR_EXP702_VOLUME_ID": volume_id,
                       "SAIR_EXP702_GUEST_COMMAND": guest_command,
                       "SAIR_EXP702_QUARANTINE": "release-marker-v1"}
        if contract.get("dataCenterIds") != [self.data_center_id]:
            raise RunPodTransportError("package Pod datacenter binding differs from the volume datacenter")
        payload = canonical_pod_create_payload(
            name=name, gpu=self.expected_gpu, data_center_id=self.data_center_id,
            image=contract["image"], command=command, volume_id=volume_id,
            environment=environment)
        canonical_body = _canonical_json_bytes(payload)
        projection = {
            "schema_version": "exp702-sanitized-pod-create-request-v1",
            "method": "POST", "path": "/pods",
            "canonical_body_bytes": len(canonical_body),
            "canonical_body_sha256": hashlib.sha256(canonical_body).hexdigest(),
            "name": name, "compute_type": payload["computeType"],
            "gpu_type_ids": payload["gpuTypeIds"], "gpu_count": payload["gpuCount"],
            "gpu_type_priority": payload["gpuTypePriority"],
            "cloud_type": payload["cloudType"],
            "data_center_ids": payload.get("dataCenterIds", []),
            "data_center_priority": payload["dataCenterPriority"],
            "interruptible": payload["interruptible"],
            "network_volume_id": volume_id,
            "volume_mount_path": payload["volumeMountPath"],
            "environment_keys": sorted(environment),
            "environment_sha256": hashlib.sha256(
                _canonical_json_bytes(environment)).hexdigest(),
            "docker_start_cmd_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "local_volume_field_present": "volumeInGb" in payload,
        }
        evidence_path = self.launch_dir / "pod-create-requests" / f"{name}.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = evidence_path.with_name(f".{evidence_path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(projection, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, evidence_path)
        try:
            result = self.api.create_pod(payload=payload)
            return str(result["id"])
        except (OSError, KeyError, RunPodTransportError):
            reconciled = self._find_pod(name, owner_nonce)
            if reconciled is None:
                raise
            return reconciled

    def verify_quarantined_pod(self, *, pod_id: str, name: str, volume_id: str,
                               owner_nonce: str,
                               readiness_deadline_seconds: int,
                               fact_recorder: Callable[[dict[str, Any]], None]) -> PodFacts:
        contract = self.package["live_contract"]
        guest_command = " ".join(contract["guest_command"])
        release_path = f"/workspace/exp702/control/{owner_nonce}.release.json"
        expected_command = ("set -eu; release=" + release_path
                            + "; while [ ! -f \"$release\" ]; do sleep 1; done; "
                            + "exec bash -lc \"$SAIR_EXP702_GUEST_COMMAND\"")
        expected_env = {**contract["environment"], "SAIR_EXP702_OWNER_NONCE": owner_nonce,
                        "SAIR_EXP702_VOLUME_ID": volume_id,
                        "SAIR_EXP702_GUEST_COMMAND": guest_command,
                        "SAIR_EXP702_QUARANTINE": "release-marker-v1"}

        while True:
            item = self.api.get_pod(pod_id)
            incomplete: list[str] = []
            contradictions: list[str] = []

            def nested_object(key: str, label: str) -> dict[str, Any]:
                value = item.get(key)
                if value is None:
                    incomplete.append(label + "-absent-or-null")
                    return {}
                if type(value) is not dict:
                    contradictions.append(label + "-not-object")
                    return {}
                return value

            raw_gpu = item.get("gpu")
            if raw_gpu is None:
                gpu: dict[str, Any] = {}
            elif type(raw_gpu) is not dict:
                contradictions.append("gpu-not-object")
                gpu = {}
            else:
                gpu = raw_gpu
            machine = nested_object("machine", "machine")
            network_volume = nested_object("networkVolume", "network-volume")
            env = item.get("env")
            command = item.get("dockerStartCmd")

            def exact_string(value: Any, expected: str, label: str) -> str | None:
                if value is None:
                    incomplete.append(label + "-absent-or-null")
                    return None
                if (type(value) is not str or not value or len(value) > 256
                        or any(ord(character) < 32 for character in value)):
                    contradictions.append(label + "-invalid")
                    return None
                if value != expected:
                    contradictions.append(label + "-contradiction")
                return value

            parsed_id = exact_string(item.get("id"), pod_id, "pod-id")
            parsed_name = exact_string(item.get("name"), name, "name")
            raw_gpu_name = gpu.get("displayName")
            if raw_gpu_name is None:
                parsed_gpu_name = None
            else:
                parsed_gpu_name = exact_string(raw_gpu_name, self.expected_gpu,
                                               "gpu-display-name")
            parsed_data_center = exact_string(machine.get("dataCenterId"), self.data_center_id,
                                              "data-center-id")
            parsed_volume_id = exact_string(network_volume.get("id"), volume_id,
                                            "network-volume-id")

            raw_count = gpu.get("count")
            if raw_count is None:
                parsed_count = None
            elif type(raw_count) is not int:
                contradictions.append("gpu-count-invalid"); parsed_count = None
            else:
                parsed_count = raw_count
                if parsed_count != 1:
                    contradictions.append("gpu-count-contradiction")

            raw_secure = machine.get("secureCloud")
            if raw_secure is None:
                incomplete.append("secure-cloud-absent-or-null"); parsed_secure = None
            elif type(raw_secure) is not bool:
                contradictions.append("secure-cloud-invalid"); parsed_secure = None
            else:
                parsed_secure = raw_secure

            raw_interruptible = item.get("interruptible")
            if raw_interruptible is None:
                parsed_interruptible = None
            elif type(raw_interruptible) is not bool:
                contradictions.append("interruptible-invalid"); parsed_interruptible = None
            else:
                parsed_interruptible = raw_interruptible
                if parsed_interruptible is not False:
                    contradictions.append("interruptible-contradiction")

            raw_rate = item.get("costPerHr")
            if raw_rate is None:
                incomplete.append("cost-per-hour-absent-or-null"); rate = None
            else:
                try:
                    if type(raw_rate) not in (str, int, float) or type(raw_rate) is bool:
                        raise ValueError
                    rate = Decimal(str(raw_rate))
                except Exception:
                    rate = None; contradictions.append("cost-per-hour-invalid")
                if rate is not None and (not rate.is_finite() or rate < 0 or rate > self.maximum_rate):
                    contradictions.append("cost-per-hour-contradiction")

            if env is None:
                incomplete.append("environment-absent-or-null")
                owner_matches = False
            elif type(env) is not dict:
                contradictions.append("environment-invalid")
                owner_matches = False
            else:
                environment_subset = True
                owner_value = env.get("SAIR_EXP702_OWNER_NONCE")
                owner_matches = owner_value == owner_nonce
                for key, expected in expected_env.items():
                    actual = env.get(key)
                    if actual is None:
                        incomplete.append("environment-value-absent-or-null")
                        environment_subset = False
                    elif actual != expected:
                        contradictions.append("environment-value-contradiction")
                        environment_subset = False
                if owner_value is None:
                    incomplete.append("owner-nonce-absent-or-null")
                elif not owner_matches:
                    contradictions.append("owner-nonce-contradiction")
                provider_added_environment_key_count = len(set(env) - set(expected_env))

            if env is None or type(env) is not dict:
                environment_subset = False
                provider_added_environment_key_count = 0

            legacy_expected = {
                "gpuCount": 1,
                "gpuDisplayName": self.expected_gpu,
                "cloudType": "COMMUNITY",
                "networkVolumeId": volume_id,
                "dataCenterId": self.data_center_id,
            }
            for key, expected in legacy_expected.items():
                if key in item and item[key] != expected:
                    contradictions.append("legacy-" + key + "-contradiction")

            if command is None:
                incomplete.append("docker-start-cmd-absent-or-null"); command_exact = False
            else:
                command_exact = (type(command) is list and len(command) == 1
                                 and type(command[0]) is str and command[0] == expected_command)
                if not command_exact:
                    contradictions.append("docker-start-cmd-not-exact")

            errors = sorted(set(incomplete + contradictions))
            projection = {
                "schema_version": "exp702-sanitized-returned-pod-facts-v2",
                "pod_id": parsed_id, "name": parsed_name,
                "owner_nonce_matches": owner_matches,
                "network_volume_id": parsed_volume_id, "gpu_display_name": parsed_gpu_name,
                "gpu_count": parsed_count, "secure_cloud": parsed_secure,
                "data_center_id": parsed_data_center, "interruptible": parsed_interruptible,
                "cost_per_hour_usd": None if rate is None else str(rate),
                "environment_subset": environment_subset,
                "provider_added_environment_key_count": provider_added_environment_key_count,
                "docker_start_cmd_exact": command_exact,
                "allocation_verified": not incomplete and not contradictions,
                "projection_errors": errors,
            }
            fact_recorder(projection)
            if contradictions:
                raise RunPodTransportError("returned Pod facts contradict the exact quarantine contract")
            if not incomplete:
                assert (parsed_id is not None and parsed_name is not None
                        and parsed_secure is not None and parsed_data_center is not None
                        and parsed_volume_id is not None
                        and rate is not None)
                return PodFacts(
                    parsed_id, parsed_name, owner_nonce, parsed_volume_id, parsed_gpu_name,
                    parsed_count, "SECURE" if parsed_secure else "COMMUNITY",
                    parsed_data_center, parsed_interruptible, rate, True,
                )
            if self.now_seconds + 15 > readiness_deadline_seconds:
                raise RunPodTransportError("quarantined Pod facts exceeded bounded readiness deadline")
            self.sleep(15)

    def release_pod(self, *, pod_id: str, volume_id: str, owner_nonce: str) -> None:
        self.s3.put_json(bucket=volume_id, key=f"exp702/control/{owner_nonce}.release.json", value={
            "schema_version": "exp702-pod-release-v1", "status": "released-after-rest-verification",
            "pod_id": pod_id, "volume_id": volume_id, "owner_nonce": owner_nonce,
        })

    def guest_state(self, *, pod_id: str, volume_id: str, owner_nonce: str) -> str:
        value = self.s3.get_json(bucket=volume_id, key=f"exp702/status/{owner_nonce}.json")
        if (value.get("schema_version") != "exp702-guest-state-v2"
                or value.get("pod_id") != pod_id or value.get("volume_id") != volume_id
                or value.get("owner_nonce") != owner_nonce):
            raise RunPodTransportError("guest status identity mismatch")
        state = str(value.get("state"))
        if state not in {"running", "failed", "completed-positive", "completed-negative"}:
            raise RunPodTransportError("guest status state is invalid")
        if state.startswith("completed-"):
            final = value.get("final_export")
            if not isinstance(final, dict) or any(final.get(key) != expected for key, expected in {
                    "pod_id": pod_id, "volume_id": volume_id, "owner_nonce": owner_nonce,
                    "scientific_outcome": state.removeprefix("completed-")}.items()):
                raise RunPodTransportError("terminal status final export identity mismatch")
        return value

    def download_export(self, *, pod_id: str, volume_id: str, owner_nonce: str, sequence: int) -> bytes:
        prefix = f"exp702/exports/{owner_nonce}/export-{sequence:04d}.tar.gz"
        receipt = self.s3.get_json(bucket=volume_id, key=prefix + ".receipt.json")
        checksum = self.s3.get_bytes(bucket=volume_id, key=prefix + ".sha256").decode().split()[0]
        data = self.s3.get_bytes(bucket=volume_id, key=prefix)
        digest = hashlib.sha256(data).hexdigest()
        expected = {"pod_id": pod_id, "volume_id": volume_id, "owner_nonce": owner_nonce,
                    "sequence": sequence, "archive_sha256": digest}
        if checksum != digest or any(receipt.get(key) != value for key, value in expected.items()):
            raise RunPodTransportError("S3 export triplet identity mismatch")
        return data

    def publish_export(self, *, data: bytes, identity: dict[str, Any]) -> dict[str, Any]:
        return self.publisher.publish(data=data, identity=identity)

    def acknowledge_export(self, *, pod_id: str, volume_id: str, owner_nonce: str,
                           sequence: int, archive_sha256: str) -> None:
        self.s3.put_json(bucket=volume_id, key=f"exp702/acks/{owner_nonce}.json", value={
            "schema_version": "exp702-offpod-ack-v1", "status": "restore-verified-offpod",
            "pod_id": pod_id, "volume_id": volume_id, "owner_nonce": owner_nonce,
            "sequence": sequence, "archive_sha256": archive_sha256,
        })

    def delete_pod(self, *, pod_id: str, owner_nonce: str) -> None:
        self.api.delete_pod(pod_id)

    def delete_volume(self, *, volume_id: str, owner_nonce: str) -> None:
        self.api.delete_network_volume(volume_id)

    def inventories(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (tuple(sorted(str(item["id"]) for item in self.api.list_pods())),
                tuple(sorted(str(item["id"]) for item in self.api.list_network_volumes())))

    def reconcile_resources(self, *, pod_name: str, volume_name: str,
                            owner_nonce: str) -> tuple[str | None, str | None]:
        return self._find_pod(pod_name, owner_nonce), self._find_volume(volume_name)

    def wait(self, seconds: int) -> None:
        self.sleep(seconds)


def read_credentials() -> tuple[str, str, str]:
    values = tuple(os.environ.get(name, "") for name in
                   ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"))
    if not all(values):
        raise RunPodTransportError("RunPod REST/S3 credentials are absent")
    return values  # type: ignore[return-value]


def build_s3_client(endpoint: str, access_key: str, secret_key: str) -> Any:
    from boto3 import client
    from botocore.config import Config
    host = parse.urlparse(endpoint).hostname or ""
    if not endpoint.startswith("https://") or not host.startswith("s3api-") or not host.endswith(".runpod.io"):
        raise RunPodTransportError("RunPod S3 endpoint is not allowlisted")
    region = host[len("s3api-"):-len(".runpod.io")]
    return client("s3", endpoint_url=endpoint.rstrip("/"), aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key, region_name=region,
                  config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                connect_timeout=15, read_timeout=30,
                                retries={"max_attempts": 3, "mode": "standard"}))


def create_transport(*, control: dict[str, Any], launch_dir: Path, package: dict[str, Any],
                     credential_reader: Callable[[], tuple[str, str, str]] = read_credentials,
                     exchange: Exchange = urllib_exchange, s3_factory: Callable[[str, str, str], Any] = build_s3_client,
                     clock: Callable[[], float] = time.monotonic,
                     sleeper: Callable[[float], None] = time.sleep) -> RunPodSecureTransport:
    """Construct authenticated clients; caller must first verify authorization."""
    api_key, access_key, secret_key = credential_reader()
    endpoint = str(control["resource"]["s3_endpoint"])
    started = clock()
    return RunPodSecureTransport(
        api=RunPodRESTAPI(api_key=api_key, exchange=exchange),
        s3=RunPodS3(s3_factory(endpoint, access_key, secret_key), endpoint),
        publisher=AtomicDirectoryPublisher(launch_dir / "offpod-publications"),
        package=package, launch_dir=launch_dir,
        data_center_id=str(control["resource"]["data_center_id"]),
        expected_gpu=str(control["resource"]["gpu"]),
        maximum_rate=Decimal(str(control["budget"]["compute_usd_per_hour"])),
        monotonic_seconds=lambda: int(clock() - started), sleep=sleeper,
    )
