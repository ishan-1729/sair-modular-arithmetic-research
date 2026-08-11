from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from experiments.cloud_benchmarks.exp702_runpod import controller, orchestrate, runpod_transport
from experiments.exp_702_exp701_terminal_evaluation_continuation import run


def _fake_launch(root: Path) -> None:
    artifacts = {}
    for name, key in (("source.tar.gz", "source_archive"),
                      ("predecessor.tar.gz", "predecessor_archive"),
                      ("torch.whl", "torch_wheel")):
        path = root / name; path.write_bytes(name.encode())
        artifacts[key] = {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "package.json").write_text(json.dumps({
        **artifacts,
        "live_contract": {
            "cloudType": "COMMUNITY", "gpuTypeIds": ["NVIDIA RTX A5000"],
            "gpuCount": 1, "gpuTypePriority": "availability",
            "dataCenterPriority": "availability", "networkVolumeMountPath": "/workspace",
            "s3Endpoint": "https://s3api-us-il-1.runpod.io/",
            "dataCenterIds": ["US-IL-1"],
            "image": "bound-image", "guest_command": ["true"],
            "environment": {"SAIR_EXP702_SYNC_SECONDS": "300"},
        },
    }), encoding="utf-8")


def _lifecycle(tmp_path: Path, transport: controller.Transport, suffix: str) -> dict:
    _fake_launch(tmp_path)
    return controller.run_lifecycle(
        transport=transport, launch_dir=tmp_path, receipt_path=tmp_path / f"{suffix}.json",
        suffix=suffix, maximum_total_usd=Decimal("2.90"), maximum_provider_seconds=7200,
        cleanup_start_seconds=6600, sync_interval_seconds=300, stale_seconds=720,
        expected_gpu="NVIDIA GeForce RTX 4090", maximum_rate=Decimal("0.74"), volume_size_gb=50,
    )


def test_no_paid_flag_is_mandatory() -> None:
    with pytest.raises(orchestrate.PlanFailure, match="mandatory"):
        orchestrate.require_no_provider_mutation(False, {
            "provider_mutation_allowed": False, "credential_access_allowed": False,
        })


@pytest.mark.parametrize("scenario,outcome", [("positive", "positive"), ("negative", "negative")])
def test_fixture_lifecycle_is_ready_first_300_second_monitored_and_exact_zero(tmp_path, scenario, outcome) -> None:
    transport = controller.FakeTransport(scenario=scenario)
    receipt = _lifecycle(tmp_path, transport, f"unit-{scenario}")
    assert receipt["status"] == "completed"
    assert receipt["scientific_outcome"] == outcome
    assert receipt["final_export_confirmed"] is True
    assert receipt["final_acknowledged"] is True
    final = receipt["final_export"]
    assert final["scientific_outcome"] == outcome and final["sequence"] == 2
    assert final["pod_id"] == receipt["pod_id"] and final["volume_id"] == receipt["volume_id"]
    assert final["owner_nonce"] == receipt["owner_nonce"]
    assert receipt["pod_delete_confirmed"] and receipt["volume_delete_confirmed"]
    assert receipt["exact_absence_confirmed"] is True
    operations = [event["operation"] for event in transport.events]
    assert operations.index("watchdog-ready") < operations.index("admission-snapshot") < operations.index("create-volume")
    assert operations.count("admission-snapshot") == 1
    create_index = operations.index("create-volume")
    assert not any(name.startswith("create-") for name in operations[:create_index])
    assert (operations.index("create-volume") < operations.index("create-pod")
            < operations.index("rest-get-pod") < operations.index("s3-preload")
            < operations.index("release-pod") < operations.index("guest-hardware-acceptance")
            < operations.index("s3-guest-state"))
    monitor_waits = [event for event in transport.events
                     if event["operation"] == "wait" and event["at_seconds"] < receipt["cleanup_started_at_seconds"]]
    assert monitor_waits and {event["seconds"] for event in monitor_waits} == {300}
    final_download = max(index for index, event in enumerate(transport.events)
                         if event["operation"] == "s3-download-export")
    final_publish = max(index for index, event in enumerate(transport.events)
                        if event["operation"] == "publish-export")
    final_ack = max(index for index, event in enumerate(transport.events)
                    if event["operation"] == "s3-acknowledge-export")
    assert final_download < final_publish < final_ack < operations.index("delete-pod") < operations.index("delete-volume")


@pytest.mark.parametrize("scenario", ["interruption", "stale-ack", "export-failure", "guest-failure"])
def test_failure_paths_delete_pod_and_restore_gate_volume(tmp_path, scenario) -> None:
    receipt = _lifecycle(tmp_path, controller.FakeTransport(scenario=scenario), f"unit-{scenario}")
    assert receipt["errors"]
    assert receipt["pod_delete_confirmed"] is True
    assert receipt["volume_delete_restore_gated"] is True
    assert receipt["exact_absence_confirmed"] is True


def test_publication_failure_recovers_download_after_pod_delete_then_cleans_volume(tmp_path) -> None:
    transport = controller.FakeTransport(scenario="publication-failure")
    receipt = _lifecycle(tmp_path, transport, "unit-publication-failure")
    recovered = [item for item in receipt["exports"] if item.get("publication_recovered_after_pod_delete")]
    assert recovered and Path(recovered[0]["download_path"]).is_file()
    operations = [event["operation"] for event in transport.events]
    assert operations.index("delete-pod") < max(index for index, value in enumerate(operations) if value == "publish-export")
    assert operations.index("delete-pod") < operations.index("delete-volume")
    assert receipt["exact_absence_confirmed"] is True


def test_partial_create_has_empty_volume_restore_proof_and_exact_zero(tmp_path) -> None:
    receipt = _lifecycle(tmp_path, controller.FakeTransport(scenario="partial-create"), "unit-partial-create")
    assert receipt["pod_id"] is None and receipt["volume_id"]
    assert receipt["exports"][0]["kind"] == "empty-volume-before-pod"
    assert receipt["exact_absence_confirmed"] is True


def test_returned_fact_mismatch_never_preloads_or_releases_and_cleans_promptly(tmp_path) -> None:
    transport = controller.FakeTransport(scenario="pod-mismatch")
    receipt = _lifecycle(tmp_path, transport, "unit-pod-mismatch")
    operations = [event["operation"] for event in transport.events]
    assert receipt["status"] == "failed" and receipt["exact_absence_confirmed"] is True
    assert receipt["returned_pod_facts"]["gpu_display_name"] == "unexpected GPU"
    assert set(receipt["returned_pod_facts"]) == {
        "schema_version", "pod_id", "name", "owner_nonce_matches", "network_volume_id",
        "gpu_display_name", "gpu_count", "secure_cloud", "data_center_id",
        "interruptible", "cost_per_hour_usd", "environment_subset",
        "provider_added_environment_key_count", "docker_start_cmd_exact",
        "allocation_verified", "projection_errors",
    }
    assert "rest-get-pod" in operations and "s3-preload" not in operations and "release-pod" not in operations
    assert operations.index("rest-get-pod") < operations.index("delete-pod") < operations.index("delete-volume")


def test_user_attested_funding_must_cover_hard_cap_before_create(tmp_path) -> None:
    _fake_launch(tmp_path)
    transport = controller.FakeTransport()
    with pytest.raises(controller.LifecycleFailure, match="weakened"):
        controller.run_lifecycle(
            transport=transport, launch_dir=tmp_path, receipt_path=tmp_path / "low.json",
            suffix="unit-low-attestation", maximum_total_usd=Decimal("2.90"),
            maximum_provider_seconds=7200, cleanup_start_seconds=6600,
            sync_interval_seconds=300, stale_seconds=720,
            expected_gpu="NVIDIA GeForce RTX 4090", maximum_rate=Decimal("0.74"),
            volume_size_gb=50, user_attested_balance_usd=Decimal("2.89"),
        )
    assert transport.events == []


def test_cleanup_retries_to_deadline_and_never_claims_false_absence(tmp_path) -> None:
    transport = controller.FakeTransport(scenario="deletion-failure")
    receipt = _lifecycle(tmp_path, transport, "unit-delete-failure")
    assert receipt["status"] == "failed-cleanup"
    assert receipt["finished_by_seconds"] == 7200
    assert receipt["pod_delete_confirmed"] is False
    assert receipt["volume_delete_confirmed"] is False
    assert receipt["exact_absence_confirmed"] is False


class FinalAckFailure(controller.FakeTransport):
    def acknowledge_export(self, **values) -> None:
        if values["sequence"] == 2:
            raise controller.LifecycleFailure("fixture rejects final acknowledgement")
        super().acknowledge_export(**values)


def test_terminal_final_ack_failure_deletes_pod_but_retains_restore_gated_volume(tmp_path) -> None:
    receipt = _lifecycle(tmp_path, FinalAckFailure(), "unit-final-ack-failure")
    assert receipt["terminal_status_observed"] is True
    assert receipt["pod_delete_confirmed"] is True
    assert receipt["final_acknowledged"] is False
    assert receipt["volume_delete_confirmed"] is False
    assert receipt["exact_absence_confirmed"] is False


class DelayedReadiness(controller.FakeTransport):
    readiness_attempts: int = 0

    def guest_state(self, **values):
        self.readiness_attempts += 1
        if self.readiness_attempts <= 2:
            self.event("guest-not-ready")
            raise controller.LifecycleFailure("fixture guest status absent")
        return super().guest_state(**values)


def test_guest_readiness_is_bounded_then_monitoring_uses_300_seconds(tmp_path) -> None:
    transport = DelayedReadiness()
    receipt = _lifecycle(tmp_path, transport, "unit-delayed-readiness")
    assert receipt["status"] == "completed"
    waits = [event["seconds"] for event in transport.events if event["operation"] == "wait"]
    assert waits[:2] == [15, 15]
    assert 300 in waits


@dataclass
class HTTPFixture:
    calls: list[dict] = field(default_factory=list)
    pods: list[dict] = field(default_factory=list)
    volumes: list[dict] = field(default_factory=list)
    lose_volume_response: bool = False
    lose_pod_response: bool = False
    secure_rate: str = "0.27"
    pod_get_overrides: list[dict] = field(default_factory=list)

    def __call__(self, method, url, headers, body):
        payload = None if body is None else json.loads(body)
        self.calls.append({"method": method, "url": url, "headers": headers, "payload": payload})
        path = url.removeprefix(runpod_transport.RUNPOD_REST_API)
        if method == "GET" and path == "/pods?includeMachine=true":
            return 200, json.dumps(self.pods).encode()
        if method == "GET" and path == "/networkvolumes":
            return 200, json.dumps(self.volumes).encode()
        if (method == "GET"
                and path.startswith("/pods/")
                and path.endswith("?includeMachine=true&includeNetworkVolume=true")):
            pod_id = path.split("?", 1)[0].rsplit("/", 1)[1]
            value = next(item for item in self.pods if item["id"] == pod_id)
            if self.pod_get_overrides:
                value = {**value, **self.pod_get_overrides.pop(0)}
            return 200, json.dumps(value).encode()
        if method == "POST" and path == "/networkvolumes":
            value = {"id": "volume-fixture", **payload}; self.volumes.append(value)
            if self.lose_volume_response:
                self.lose_volume_response = False; raise OSError("fixture response loss")
            return 201, json.dumps(value).encode()
        if method == "POST" and path == "/pods":
            value = {"id": "pod-fixture", **payload, "costPerHr": self.secure_rate,
                     "gpu": {"count": payload["gpuCount"],
                             "displayName": "NVIDIA RTX A5000"},
                     "machine": {"dataCenterId": "US-IL-1", "secureCloud": False},
                     "networkVolume": {"id": payload["networkVolumeId"]}}
            for obsolete in ("gpuCount", "cloudType", "networkVolumeId"):
                value.pop(obsolete)
            self.pods.append(value)
            if self.lose_pod_response:
                self.lose_pod_response = False; raise OSError("fixture response loss")
            return 201, json.dumps(value).encode()
        if method == "DELETE" and path.startswith("/pods/"):
            self.pods = [item for item in self.pods if item["id"] != path.rsplit("/", 1)[1]]
            return 204, b""
        if method == "DELETE" and path.startswith("/networkvolumes/"):
            self.volumes = [item for item in self.volumes if item["id"] != path.rsplit("/", 1)[1]]
            return 204, b""
        raise AssertionError((method, path, payload))


class S3Fixture:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, source, bucket, key, ExtraArgs=None):
        self.calls.append(("upload_file", bucket, key)); self.objects[(bucket, key)] = Path(source).read_bytes()

    def get_object(self, *, Bucket, Key):
        self.calls.append(("get_object", Bucket, Key)); return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, *, Bucket, Key, Body, **_values):
        self.calls.append(("put_object", Bucket, Key)); self.objects[(Bucket, Key)] = bytes(Body)


def _concrete(tmp_path: Path, http: HTTPFixture, s3: S3Fixture) -> runpod_transport.RunPodSecureTransport:
    _fake_launch(tmp_path)
    package = json.loads((tmp_path / "package.json").read_text())
    return runpod_transport.RunPodSecureTransport(
        api=runpod_transport.RunPodRESTAPI("fixture-token", exchange=http),
        s3=runpod_transport.RunPodS3(s3, "https://s3api-us-il-1.runpod.io/"),
        publisher=runpod_transport.AtomicDirectoryPublisher(tmp_path / "published"),
        package=package, launch_dir=tmp_path, data_center_id="US-IL-1",
        expected_gpu="NVIDIA RTX A5000", maximum_rate=Decimal("0.27"),
        monotonic_seconds=lambda: 0, sleep=lambda _seconds: None,
    )


def test_fixture_backed_rest_uses_current_routes_and_exact_create_schemas(tmp_path) -> None:
    http, s3 = HTTPFixture(), S3Fixture()
    transport = _concrete(tmp_path, http, s3)
    nonce = "n" * 64; name = "sair-exp702-fixture-" + nonce[:12]
    volume = transport.create_volume(name=name, size_gb=50, owner_nonce=nonce)
    pod = transport.create_pod(name=name, volume_id=volume, owner_nonce=nonce)
    captured = []
    facts = transport.verify_quarantined_pod(
        pod_id=pod, name=name, volume_id=volume, owner_nonce=nonce,
        readiness_deadline_seconds=720,
        fact_recorder=captured.append)
    transport.preload_volume(volume_id=volume, artifacts={"exp702/inputs/source.tar.gz": tmp_path / "source.tar.gz"},
                             owner_nonce=nonce)
    transport.release_pod(pod_id=pod, volume_id=volume, owner_nonce=nonce)
    volume_call = next(call for call in http.calls if call["method"] == "POST" and call["url"].endswith("/networkvolumes"))
    assert volume_call["payload"] == {"dataCenterId": "US-IL-1", "name": name, "size": 50}
    pod_call = next(call for call in http.calls if call["method"] == "POST" and call["url"].endswith("/pods"))
    assert pod_call["payload"]["cloudType"] == "COMMUNITY"
    assert pod_call["payload"]["interruptible"] is False
    assert pod_call["payload"]["dataCenterIds"] == ["US-IL-1"]
    assert pod_call["payload"]["dataCenterPriority"] == "availability"
    assert pod_call["payload"]["gpuTypeIds"] == ["NVIDIA RTX A5000"]
    assert pod_call["payload"]["gpuTypePriority"] == "availability"
    assert pod_call["payload"]["imageName"] == "bound-image"
    assert "release.json" in pod_call["payload"]["dockerStartCmd"][0]
    assert pod_call["payload"]["networkVolumeId"] == volume
    assert pod_call["payload"]["volumeMountPath"] == "/workspace"
    assert "volumeInGb" not in pod_call["payload"]
    assert pod_call["payload"]["env"]["SAIR_EXP702_OWNER_NONCE"] == nonce
    assert all("/v1/network-volumes" not in call["url"] for call in http.calls)
    assert ("upload_file", volume, "exp702/inputs/source.tar.gz") in s3.calls
    assert facts.gpu == "NVIDIA RTX A5000" and facts.cost_per_hour_usd == Decimal("0.27")
    assert facts.quarantined is True
    assert next(call for call in http.calls if call["method"] == "GET" and "/pods/pod-fixture?" in call["url"])["url"].endswith(
        "/pods/pod-fixture?includeMachine=true&includeNetworkVolume=true")
    assert captured[0]["schema_version"] == "exp702-sanitized-returned-pod-facts-v2"
    assert captured[0]["network_volume_id"] == volume
    assert captured[0]["owner_nonce_matches"] is True
    assert captured[0]["environment_subset"] is True
    assert captured[0]["allocation_verified"] is True
    assert captured[0]["docker_start_cmd_exact"] is True
    assert captured[0]["projection_errors"] == []
    assert ("put_object", volume, f"exp702/control/{nonce}.release.json") in s3.calls
    assert pod == "pod-fixture"
    evidence = json.loads(next((tmp_path / "pod-create-requests").glob("*.json")).read_text())
    canonical = json.dumps(pod_call["payload"], sort_keys=True, separators=(",", ":")).encode()
    assert evidence["schema_version"] == "exp702-sanitized-pod-create-request-v1"
    assert evidence["canonical_body_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert evidence["canonical_body_bytes"] == len(canonical)
    assert evidence["gpu_type_priority"] == evidence["data_center_priority"] == "availability"
    assert evidence["local_volume_field_present"] is False
    serialized_evidence = json.dumps(evidence)
    assert "fixture-token" not in serialized_evidence and nonce not in serialized_evidence


def test_pod_request_evidence_exists_before_post_and_http_error_is_bounded_and_sanitized(tmp_path) -> None:
    observed = {}
    nonce = "h" * 64
    name = "sair-exp702-http-error-" + nonce[:12]

    def failing_exchange(method, url, headers, body):
        if method == "GET" and url.endswith("/pods?includeMachine=true"):
            return 200, b"[]"
        observed["evidence_exists_before_post"] = any(
            (tmp_path / "pod-create-requests").glob("*.json"))
        observed["body"] = body
        raw = json.dumps({
            "error": "create pod: " + ("no capacity " * 100), "status": 500,
            "secret": "must-never-be-retained",
        }).encode()
        return 500, raw, {"X-Request-Id": "req-safe-123", "Authorization": "secret"}

    transport = _concrete(tmp_path, HTTPFixture(), S3Fixture())
    transport.api.exchange = failing_exchange
    with pytest.raises(runpod_transport.RunPodHTTPError) as raised:
        transport.create_pod(name=name, volume_id="volume-fixture", owner_nonce=nonce)
    diagnostic = raised.value.diagnostic
    assert observed["evidence_exists_before_post"] is True
    assert diagnostic["http_status"] == 500 and diagnostic["body_truncated"] is False
    assert diagnostic["body"]["error"].startswith("create pod: no capacity")
    assert len(diagnostic["body"]["error"]) == runpod_transport.MAX_HTTP_ERROR_TEXT_CHARS
    assert diagnostic["body"]["status"] == 500
    assert diagnostic["request_ids"] == {"x-request-id": "req-safe-123"}
    serialized = json.dumps(diagnostic)
    assert "must-never-be-retained" not in serialized and "Authorization" not in serialized
    evidence = json.loads(next((tmp_path / "pod-create-requests").glob("*.json")).read_text())
    assert evidence["canonical_body_sha256"] == hashlib.sha256(observed["body"]).hexdigest()


class UncertainCreateCleanup(controller.FakeTransport):
    def create_pod(self, *, name: str, volume_id: str, owner_nonce: str) -> str:
        pod_id = f"pod-{owner_nonce[:12]}"
        self.pods[pod_id] = owner_nonce
        self.event("create-pod-response-lost", pod_id=pod_id, volume_id=volume_id, name=name)
        raise controller.LifecycleFailure("uncertain Pod-create response")


def test_uncertain_pod_create_is_reconciled_and_deleted_without_preload_or_release(tmp_path) -> None:
    transport = UncertainCreateCleanup()
    receipt = _lifecycle(tmp_path, transport, "unit-uncertain-create-cleanup")
    operations = [event["operation"] for event in transport.events]
    assert receipt["status"] == "failed"
    assert receipt["uncertain_create_reconciliation"]["performed"] is True
    assert receipt["uncertain_create_reconciliation"]["pod_id_recovered"] == receipt["pod_id"]
    assert receipt["pod_delete_confirmed"] is True
    assert receipt["volume_delete_confirmed"] is True
    assert receipt["exact_absence_confirmed"] is True
    assert "s3-preload" not in operations and "release-pod" not in operations
    assert operations.index("reconcile-uncertain-create") < operations.index("delete-pod")


@pytest.mark.parametrize("case", [
    "legacy-contradiction", "wrong-type", "malformed", "nonfinite",
])
def test_returned_pod_parser_persists_sanitized_projection_then_rejects(case, tmp_path) -> None:
    http = HTTPFixture()
    transport = _concrete(tmp_path, http, S3Fixture())
    nonce = "c" * 64
    name = "sair-exp702-invalid-" + nonce[:12]
    volume = transport.create_volume(name=name, size_gb=50, owner_nonce=nonce)
    pod = transport.create_pod(name=name, volume_id=volume, owner_nonce=nonce)
    item = http.pods[0]
    if case == "legacy-contradiction":
        item["gpuCount"] = 2
    elif case == "wrong-type":
        item["gpu"]["count"] = "1"
    elif case == "malformed":
        item["dockerStartCmd"] = {"command": item["dockerStartCmd"][0]}
    elif case == "nonfinite":
        item["costPerHr"] = "NaN"
    captured = []
    with pytest.raises(runpod_transport.RunPodTransportError, match="contradict"):
        transport.verify_quarantined_pod(
            pod_id=pod, name=name, volume_id=volume, owner_nonce=nonce,
            readiness_deadline_seconds=720,
            fact_recorder=captured.append)
    assert len(captured) == 1 and captured[0]["projection_errors"]
    serialized = json.dumps(captured[0], allow_nan=False)
    assert "dockerStartCmd" not in serialized and nonce not in serialized


def test_provider_added_environment_and_unavailable_convenience_fields_are_tolerated(tmp_path) -> None:
    http = HTTPFixture()
    transport = _concrete(tmp_path, http, S3Fixture())
    nonce = "u" * 64; name = "sair-exp702-provider-normalized-" + nonce[:12]
    volume = transport.create_volume(name=name, size_gb=50, owner_nonce=nonce)
    pod = transport.create_pod(name=name, volume_id=volume, owner_nonce=nonce)
    item = http.pods[0]
    item["gpu"] = None
    item.pop("interruptible", None)
    item["env"]["RUNPOD_PROVIDER_ADDED"] = "opaque"
    item["gpuCount"] = 1
    captured = []

    facts = transport.verify_quarantined_pod(
        pod_id=pod, name=name, volume_id=volume, owner_nonce=nonce,
        readiness_deadline_seconds=720, fact_recorder=captured.append)

    assert facts.gpu is None and facts.gpu_count is None and facts.interruptible is None
    assert captured == [{**captured[0]}]
    assert captured[0]["environment_subset"] is True
    assert captured[0]["provider_added_environment_key_count"] == 1
    assert captured[0]["allocation_verified"] is True
    assert captured[0]["projection_errors"] == []
    assert "RUNPOD_PROVIDER_ADDED" not in json.dumps(captured[0])


def test_quarantine_poll_retries_only_null_initialization_at_exact_15_second_cadence(tmp_path) -> None:
    http, s3 = HTTPFixture(), S3Fixture()
    transport = _concrete(tmp_path, http, s3)
    nonce = "d" * 64; name = "sair-exp702-delayed-" + nonce[:12]
    volume = transport.create_volume(name=name, size_gb=50, owner_nonce=nonce)
    pod = transport.create_pod(name=name, volume_id=volume, owner_nonce=nonce)
    http.pod_get_overrides = [{"machine": None}, {"networkVolume": None}, {"env": {}}]
    clock = [0]
    sleeps = []
    transport.monotonic_seconds = lambda: clock[0]
    def advance(seconds):
        sleeps.append(seconds); clock[0] += int(seconds)
    transport.sleep = advance
    captured = []

    facts = transport.verify_quarantined_pod(
        pod_id=pod, name=name, volume_id=volume, owner_nonce=nonce,
        readiness_deadline_seconds=720, fact_recorder=captured.append)

    assert facts.pod_id == pod
    assert sleeps == [15, 15, 15] and clock[0] == 45
    assert len(captured) == 4
    assert all(item["projection_errors"] for item in captured[:3])
    assert captured[3]["projection_errors"] == []
    assert s3.calls == []
    assert all(set(item) == {
        "schema_version", "pod_id", "name", "owner_nonce_matches", "network_volume_id",
        "gpu_display_name", "gpu_count", "secure_cloud", "data_center_id",
        "interruptible", "cost_per_hour_usd", "environment_subset",
        "provider_added_environment_key_count", "docker_start_cmd_exact",
        "allocation_verified", "projection_errors",
    } for item in captured)


def test_quarantine_poll_fails_present_contradiction_without_sleep(tmp_path) -> None:
    http = HTTPFixture()
    transport = _concrete(tmp_path, http, S3Fixture())
    nonce = "e" * 64; name = "sair-exp702-contradiction-" + nonce[:12]
    volume = transport.create_volume(name=name, size_gb=50, owner_nonce=nonce)
    pod = transport.create_pod(name=name, volume_id=volume, owner_nonce=nonce)
    http.pod_get_overrides = [{"interruptible": True}]
    sleeps = []
    transport.sleep = sleeps.append
    captured = []

    with pytest.raises(runpod_transport.RunPodTransportError, match="contradict"):
        transport.verify_quarantined_pod(
            pod_id=pod, name=name, volume_id=volume, owner_nonce=nonce,
            readiness_deadline_seconds=720, fact_recorder=captured.append)

    assert sleeps == [] and len(captured) == 1
    assert "interruptible-contradiction" in captured[0]["projection_errors"]


def test_quarantine_poll_stops_at_720_seconds_without_oversleep(tmp_path) -> None:
    http = HTTPFixture()
    transport = _concrete(tmp_path, http, S3Fixture())
    nonce = "f" * 64; name = "sair-exp702-deadline-" + nonce[:12]
    volume = transport.create_volume(name=name, size_gb=50, owner_nonce=nonce)
    pod = transport.create_pod(name=name, volume_id=volume, owner_nonce=nonce)
    http.pod_get_overrides = [{"machine": None} for _ in range(49)]
    clock = [0]
    sleeps = []
    transport.monotonic_seconds = lambda: clock[0]
    def advance(seconds):
        sleeps.append(seconds); clock[0] += int(seconds)
    transport.sleep = advance
    captured = []

    with pytest.raises(runpod_transport.RunPodTransportError, match="bounded readiness deadline"):
        transport.verify_quarantined_pod(
            pod_id=pod, name=name, volume_id=volume, owner_nonce=nonce,
            readiness_deadline_seconds=720, fact_recorder=captured.append)

    assert clock[0] == 720 and sleeps == [15] * 48 and len(captured) == 49


def test_authenticated_admission_uses_only_fresh_rest_inventories(tmp_path) -> None:
    http = HTTPFixture(secure_rate="0.71")
    transport = _concrete(tmp_path, http, S3Fixture())
    observed = transport.admission_snapshot()
    assert observed.pods == () and observed.volumes == ()
    paths = {call["url"].removeprefix(runpod_transport.RUNPOD_REST_API) for call in http.calls}
    assert {"/pods?includeMachine=true", "/networkvolumes"} <= paths
    assert all(call["headers"]["Authorization"] == "Bearer fixture-token" for call in http.calls)
    assert not any("graphql" in call["url"].lower() for call in http.calls)


def test_reviewed_admission_schema_names_match_authenticated_responses() -> None:
    control = orchestrate.yaml.safe_load(orchestrate.DEFAULT_CONTROL.read_text(encoding="utf-8"))
    schema = control["authenticated_admission_schema"]
    assert schema["observation_count_before_create"] == 1
    assert schema["funding"] == {"basis": "user-attested", "attested_balance_usd": 10.0,
                                 "minimum_required_usd": 2.9}
    assert schema["rest_inventory"] == {
        "pods": "/pods?includeMachine=true", "volumes": "/networkvolumes",
    }
    assert schema["quarantined_pod"]["verify_returned_facts"] == (
        "/pods/{pod_id}?includeMachine=true&includeNetworkVolume=true")
    assert schema["quarantined_pod"]["readiness_poll_interval_seconds"] == 15
    assert schema["quarantined_pod"]["readiness_deadline_seconds"] == 720
    assert schema["quarantined_pod"]["maximum_cost_per_hour_usd"] == 0.27
    assert schema["quarantined_pod"]["gpu_type_priority"] == "availability"
    assert schema["quarantined_pod"]["data_center_priority"] == "availability"
    assert schema["quarantined_pod"]["network_volume_mount_path"] == "/workspace"
    assert schema["quarantined_pod"]["forbid_local_volume_field_when_network_volume_attached"] is True


@pytest.mark.parametrize("lost", ["volume", "pod"])
def test_response_loss_is_reconciled_by_exact_name_and_nonce(tmp_path, lost) -> None:
    http = HTTPFixture(lose_volume_response=lost == "volume", lose_pod_response=lost == "pod")
    transport = _concrete(tmp_path, http, S3Fixture())
    nonce = "a" * 64; name = "sair-exp702-loss-" + nonce[:12]
    volume = transport.create_volume(name=name, size_gb=50, owner_nonce=nonce)
    assert volume == "volume-fixture"
    pod = transport.create_pod(name=name, volume_id=volume, owner_nonce=nonce)
    assert pod == "pod-fixture"
    assert len(http.volumes) == 1 and len(http.pods) == 1


def test_s3_export_triplet_download_restore_and_ack_share_one_protocol(tmp_path) -> None:
    http, s3 = HTTPFixture(), S3Fixture()
    transport = _concrete(tmp_path, http, s3)
    nonce, pod, volume, sequence = "b" * 64, "pod-fixture", "volume-fixture", 7
    fake = controller.FakeTransport()
    archive = fake._archive(pod, volume, nonce, sequence)
    digest = hashlib.sha256(archive).hexdigest()
    key = f"exp702/exports/{nonce}/export-{sequence:04d}.tar.gz"
    s3.objects[(volume, key)] = archive
    s3.objects[(volume, key + ".sha256")] = f"{digest}  export.tar.gz\n".encode()
    s3.objects[(volume, key + ".receipt.json")] = json.dumps({
        "schema_version": "exp702-guest-export-v2", "pod_id": pod, "volume_id": volume,
        "owner_nonce": nonce, "sequence": sequence, "archive_sha256": digest,
    }).encode()
    downloaded = transport.download_export(pod_id=pod, volume_id=volume, owner_nonce=nonce, sequence=sequence)
    restored = controller.restore_verify(downloaded, tmp_path / "restore", {
        "pod_id": pod, "volume_id": volume, "owner_nonce": nonce, "sequence": sequence,
    })
    transport.acknowledge_export(pod_id=pod, volume_id=volume, owner_nonce=nonce,
                                 sequence=sequence, archive_sha256=restored["archive_sha256"])
    ack = json.loads(s3.objects[(volume, f"exp702/acks/{nonce}.json")])
    assert ack["status"] == "restore-verified-offpod" and ack["archive_sha256"] == digest


def test_factory_reads_all_credentials_only_when_called_and_binds_s3_endpoint(tmp_path) -> None:
    events = []
    transport = runpod_transport.create_transport(
        control={"resource": {"gpu": "NVIDIA RTX A5000", "data_center_id": "US-IL-1",
                              "s3_endpoint": "https://s3api-us-il-1.runpod.io/"},
                 "budget": {"compute_usd_per_hour": 0.27}},
        launch_dir=tmp_path, package={"live_contract": {}},
        credential_reader=lambda: events.append("credentials") or ("api", "access", "secret"),
        exchange=lambda *_: (_ for _ in ()).throw(AssertionError("no HTTP expected")),
        s3_factory=lambda endpoint, *_: events.append(endpoint) or S3Fixture(),
        clock=lambda: 10.0, sleeper=lambda _seconds: None,
    )
    assert events == ["credentials", "https://s3api-us-il-1.runpod.io/"]
    assert transport.data_center_id == "US-IL-1"


def test_exact_authorization_precedes_rest_and_s3_factory(tmp_path, monkeypatch) -> None:
    _fake_launch(tmp_path)
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({
        "approved": True, "run_id": run.RUN_ID, "single_lifecycle_suffix": "unit-authorized",
        "package_sha256": orchestrate.sha256(tmp_path / "package.json"),
        "maximum_total_usd": "2.90", "maximum_provider_seconds": 7200,
        "cleanup_start_seconds": 6600,
    }))
    events = []
    monkeypatch.setattr(orchestrate, "audit_package", lambda *args, **kwargs: events.append("audit"))
    original = orchestrate.verify_live_authorization
    monkeypatch.setattr(orchestrate, "verify_live_authorization",
                        lambda *args, **kwargs: events.append("authorization") or original(*args, **kwargs))
    monkeypatch.setattr(orchestrate, "load_transport",
                        lambda *args, **kwargs: events.append("rest+s3-credentials") or controller.FakeTransport())
    monkeypatch.setattr(controller, "run_lifecycle", lambda **kwargs: {"status": "fixture"})
    orchestrate.execute_live(orchestrate.DEFAULT_CONTROL, tmp_path, "unit-authorized",
                             "unused:factory", tmp_path / "receipt.json", authorization)
    assert events == ["audit", "authorization", "rest+s3-credentials"]


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_watchdog_ready_accepts_utf8_with_only_an_optional_bom(tmp_path, prefix) -> None:
    suffix = "unit-watchdog-reader"
    ready = tmp_path / "watchdog" / f"{suffix}.ready.json"
    ready.parent.mkdir()
    ready.write_bytes(prefix + json.dumps({
        "schema_version": "exp702-watchdog-ready-v1", "status": "ready",
        "suffix": suffix, "deadline_seconds": 7200,
    }).encode("utf-8"))
    _concrete(tmp_path, HTTPFixture(), S3Fixture()).watchdog_ready(
        suffix=suffix, deadline_seconds=7200,
    )


@pytest.mark.parametrize("prefix", [b"\x00", b"\xff\xfe", b"garbage", b"\xef\xbb"])
def test_watchdog_ready_rejects_malformed_non_bom_prefixes(tmp_path, prefix) -> None:
    suffix = "unit-watchdog-reader"
    ready = tmp_path / "watchdog" / f"{suffix}.ready.json"
    ready.parent.mkdir()
    ready.write_bytes(prefix + json.dumps({
        "schema_version": "exp702-watchdog-ready-v1", "status": "ready",
        "suffix": suffix, "deadline_seconds": 7200,
    }).encode("utf-8"))
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        _concrete(tmp_path, HTTPFixture(), S3Fixture()).watchdog_ready(
            suffix=suffix, deadline_seconds=7200,
        )


def test_native_windows_watchdog_writer_emits_bom_free_strict_json(tmp_path) -> None:
    root = orchestrate.ROOT / "experiments/cloud_benchmarks/exp702_runpod"
    suffix = "unit-native-writer"
    controller_receipt = tmp_path / "controller.json"
    ready_receipt = tmp_path / "watchdog" / f"{suffix}.ready.json"
    restore_receipt = tmp_path / "restore.json"
    controller_receipt.write_text(json.dumps({
        "suffix": suffix, "owner_nonce": "a" * 64, "exact_absence_confirmed": True,
    }), encoding="utf-8")

    def windows_path(path: Path) -> str:
        return subprocess.run(
            ["wslpath", "-w", str(path)], check=True, text=True,
            capture_output=True,
        ).stdout.strip()

    start = datetime.now(timezone.utc).replace(microsecond=0)
    deletion = [
        "wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc",
        "source /home/ishan/.venvs/sair-mod-challenge/bin/activate && true",
    ]
    completed = subprocess.run([
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        windows_path(root / "windows_deadline_watchdog.ps1"),
        "-LifecycleSuffix", suffix,
        "-LifecycleStartUtc", start.isoformat().replace("+00:00", "Z"),
        "-CleanupDeadlineUtc", (start + timedelta(seconds=6600)).isoformat().replace("+00:00", "Z"),
        "-HardDeadlineUtc", (start + timedelta(seconds=7200)).isoformat().replace("+00:00", "Z"),
        "-ControllerReceipt", windows_path(controller_receipt),
        "-OffPodRestoreReceipt", windows_path(restore_receipt),
        "-WatchdogReadyReceipt", windows_path(ready_receipt),
        "-DeletionCommandBase64", base64.b64encode(json.dumps(deletion).encode()).decode(),
    ], check=False, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    raw = ready_receipt.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8").startswith("{")
    assert json.loads(raw.decode("utf-8")) == {
        "schema_version": "exp702-watchdog-ready-v1", "status": "ready",
        "suffix": suffix, "deadline_seconds": 7200,
    }


def test_guest_and_windows_scripts_bind_ready_reconcile_export_and_deadlines() -> None:
    root = orchestrate.ROOT / "experiments/cloud_benchmarks/exp702_runpod"
    guest = (root / "guest_supervisor.sh").read_text(encoding="utf-8")
    launcher = (root / "windows_paid_lifecycle.ps1").read_text(encoding="utf-8")
    watchdog = (root / "windows_deadline_watchdog.ps1").read_text(encoding="utf-8")
    for token in ("exp702/export.json", "exp702-guest-export-v2", "acks/$owner_nonce", "write_state",
                  "export_kind", "terminal_summary_sha256", "flock -x 8"):
        assert token in guest
    assert guest.index("flock -x 8") < guest.index('sequence="$(cat "$durable/export-sequence"')
    assert guest.index('export_once final "$scientific_outcome"') < guest.index('write_state "completed-$scientific_outcome"')
    assert launcher.index("Start-Process -FilePath 'powershell.exe'") < launcher.index("'execute-live'")
    assert "watchdog did not acknowledge readiness before create" in launcher
    for token in ("wsl.exe", "Ubuntu-24.04", "bash -lc", "source", "VenvLinuxPath/bin/activate",
                  "RepositoryLinuxPath"):
        assert token in launcher
    assert "DeletionCommandBase64" in launcher and "DeletionCommandBase64" in watchdog
    for token in ("exp702-watchdog-ready-v1", "--reconcile-uncertain-create", "--delete-pod",
                  "--restore-receipt", "--delete-volume", "--confirm-exact-absence",
                  "6600", "7200", "Start-Sleep -Seconds 15"):
        assert token in watchdog


def test_restore_gated_provider_delete_rejects_missing_receipt(tmp_path) -> None:
    transport = controller.FakeTransport(pods={"pod": "nonce"}, volumes={"volume": "nonce"})
    with pytest.raises(controller.LifecycleFailure, match="restore receipt"):
        controller.provider_delete(transport=transport, pod_id="pod", volume_id="volume",
                                   owner_nonce="nonce", delete_volume=True)
