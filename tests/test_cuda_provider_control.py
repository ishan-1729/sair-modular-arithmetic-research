from __future__ import annotations

from collections import deque
from decimal import Decimal
import json
from pathlib import Path

import pytest
import yaml

from environments.providers import controller


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cuda_provider_control.yaml"


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += float(seconds)


class FakeTransport:
    def __init__(self, responses) -> None:
        self.responses = deque(responses)
        self.calls = []

    def request(self, method, url, token, payload=None):
        self.calls.append(
            {"method": method, "url": url, "token": token, "payload": payload}
        )
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def budget(
    *, rate: str = "0.69", seconds: int = 1800, maximum: str = "1.00"
) -> controller.BudgetPlan:
    return controller.admit_budget(
        hourly_rate_usd=Decimal(rate),
        storage_hourly_usd=Decimal("0.01"),
        max_total_usd=Decimal(maximum),
        max_provider_seconds=seconds,
        reserve_fraction=Decimal("0.15"),
        minimum_total_usd=Decimal("0.10"),
    )


def spec(provider: str) -> controller.LaunchSpec:
    return controller.LaunchSpec(
        provider=provider,
        resource_selector=("NVIDIA GeForce RTX 4090" if provider == "runpod" else "123456"),
        image="example/pytorch:cuda",
        label="sair-test",
        command="environments/cuda/run-acceptance.sh --profile cu124",
        cuda_runtime="12.4",
        disk_gb=50,
    )


def test_budget_admission_reserves_headroom_and_includes_storage() -> None:
    admitted = budget()
    assert admitted.admitted_upper_usd == "0.350000"
    assert admitted.budget_after_reserve_usd == "0.850000"
    assert admitted.budget_limited_seconds == 4371
    with pytest.raises(ValueError, match="exceeds reserved budget"):
        budget(rate="3.00", seconds=1800, maximum="1.00")


def test_plan_mode_is_redacted_and_never_requires_a_credential(capsys) -> None:
    assert controller.main(
        [
            "--config",
            str(CONFIG_PATH),
            "plan",
            "--provider",
            "runpod",
            "--resource-selector",
            "NVIDIA GeForce RTX 4090",
            "--image",
            "example/pytorch:cuda",
            "--hourly-rate-usd",
            "0.69",
            "--storage-hourly-usd",
            "0.006945",
            "--max-total-usd",
            "1.00",
            "--max-provider-seconds",
            "1800",
            "--command",
            "true",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "admitted-dry-run"
    assert result["credential_environment"] == "RUNPOD_API_KEY"
    assert "command" not in result and len(result["command_sha256"]) == 64


def test_command_file_and_result_port_are_redacted(tmp_path, capsys) -> None:
    command_path = tmp_path / "guest-command.sh"
    command_path.write_text("echo acceptance\n", encoding="utf-8")
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"source archive")
    assert controller.main(
        [
            "--config",
            str(CONFIG_PATH),
            "plan",
            "--provider",
            "runpod",
            "--resource-selector",
            "NVIDIA GeForce RTX 4090",
            "--image",
            "example/pytorch:cuda",
            "--hourly-rate-usd",
            "0.69",
            "--storage-hourly-usd",
            "0.006945",
            "--max-total-usd",
            "1.00",
            "--max-provider-seconds",
            "1800",
            "--command-file",
            str(command_path),
            "--port",
            "8000/http",
            "--environment-file",
            f"SAIR_SOURCE_B64={payload_path}",
        ]
    ) == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["ports"] == ["8000/http"]
    assert result["command_sha256"] == controller.sha256_text("echo acceptance\n")
    assert "command" not in result
    assert result["environment"] == [
        {
            "name": "SAIR_SOURCE_B64_000",
            "value_chars": 20,
            "value_sha256": controller.sha256_text("c291cmNlIGFyY2hpdmU="),
        }
    ]
    assert "c291cmNlIGFyY2hpdmU=" not in output


def test_runpod_guest_exit_still_requires_confirmed_delete(tmp_path) -> None:
    clock = FakeClock()
    transport = FakeTransport(
        [
            (201, {"id": "pod-1", "adjustedCostPerHr": 0.69}),
            (200, [{"id": "pod-1", "desiredStatus": "RUNNING", "adjustedCostPerHr": 0.69}]),
            (200, [{"id": "pod-1", "desiredStatus": "EXITED", "adjustedCostPerHr": 0.69}]),
            (204, {}),
            (200, []),
        ]
    )
    receipt_path = tmp_path / "runpod.json"
    result = controller.execute(
        config=config(),
        config_path=CONFIG_PATH,
        spec=spec("runpod"),
        budget=budget(),
        receipt_path=receipt_path,
        token="runpod-secret-test-token",
        transport=transport,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert result["status"] == "resource-destroyed"
    assert result["guest_terminal_status"] == "EXITED"
    assert result["timing"] == {
        "provider_billed_lifetime_seconds": 10.0,
        "provider_billed_lifetime_source": "controller-create-request-to-delete-confirmation-upper-bound",
        "provider_invoice_amount_usd": None,
        "guest_runtime_seconds": 10.0,
        "deletion_confirmed_elapsed_seconds": 10.0,
    }
    assert [call["method"] for call in transport.calls] == [
        "POST",
        "GET",
        "GET",
        "DELETE",
        "GET",
    ]
    create = transport.calls[0]["payload"]
    assert create["computeType"] == "GPU" and create["volumeInGb"] == 0
    assert create["env"] == {}
    assert "runpod-secret-test-token" not in receipt_path.read_text(encoding="utf-8")


def test_runpod_result_port_is_declared_only_when_requested() -> None:
    provider = controller.Provider(
        "runpod", config()["providers"]["runpod"], "secret", FakeTransport([])
    )
    assert "ports" not in provider.create_payload(spec("runpod"))
    with_port = controller.LaunchSpec(
        **{**spec("runpod").__dict__, "ports": ("8000/http",)}
    )
    assert provider.create_payload(with_port)["ports"] == ["8000/http"]


def test_runpod_template_launch_keeps_create_payload_small() -> None:
    provider = controller.Provider(
        "runpod", config()["providers"]["runpod"], "secret", FakeTransport([])
    )
    templated = controller.LaunchSpec(
        **{
            **spec("runpod").__dict__,
            "ports": ("8000/http",),
            "environment": (("SAIR_SOURCE_B64_000", "large-value"),),
            "template_id": "template-1",
        }
    )
    payload = provider.create_payload(templated)
    assert payload["templateId"] == "template-1"
    assert "imageName" not in payload
    assert payload["env"] == {"SAIR_SOURCE_B64_000": "large-value"}
    assert payload["ports"] == ["8000/http"]


def test_runpod_can_preserve_official_image_startup() -> None:
    provider = controller.Provider(
        "runpod", config()["providers"]["runpod"], "secret", FakeTransport([])
    )
    official = controller.LaunchSpec(
        **{**spec("runpod").__dict__, "use_image_command": True}
    )
    payload = provider.create_payload(official)
    assert payload["dockerEntrypoint"] == []
    assert payload["dockerStartCmd"] == []
    any_cuda = controller.LaunchSpec(
        **{**official.__dict__, "cuda_runtime": "any"}
    )
    assert "allowedCudaVersions" not in provider.create_payload(any_cuda)


def test_environment_file_is_chunked_below_linux_argument_limit(tmp_path, capsys) -> None:
    payload_path = tmp_path / "large.bin"
    payload_path.write_bytes(b"x" * 100_000)
    arguments = [
        "--config", str(CONFIG_PATH), "plan", "--provider", "runpod",
        "--resource-selector", "NVIDIA GeForce RTX 4090",
        "--image", "example/pytorch:cuda", "--hourly-rate-usd", "0.69",
        "--storage-hourly-usd", "0.006945", "--max-total-usd", "1.00",
        "--max-provider-seconds", "1800", "--command", "true",
        "--environment-file", f"SAIR_SOURCE_B64={payload_path}",
    ]
    assert controller.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in result["environment"]] == [
        "SAIR_SOURCE_B64_000", "SAIR_SOURCE_B64_001", "SAIR_SOURCE_B64_002"
    ]
    assert all(row["value_chars"] <= 60_000 for row in result["environment"])


def test_environment_value_file_is_hashed_not_printed(tmp_path, capsys) -> None:
    public_key = "ssh-ed25519 AAAAC3Nza-test sair-ephemeral"
    key_path = tmp_path / "acceptance.pub"
    key_path.write_text(public_key + "\n", encoding="utf-8")
    arguments = [
        "--config", str(CONFIG_PATH), "plan", "--provider", "runpod",
        "--resource-selector", "NVIDIA GeForce RTX 4090",
        "--image", "example/pytorch:cuda", "--hourly-rate-usd", "0.69",
        "--storage-hourly-usd", "0.006945", "--max-total-usd", "1.00",
        "--max-provider-seconds", "1800", "--command", "true",
        "--environment-value-file", f"SSH_PUBLIC_KEY={key_path}",
    ]
    assert controller.main(arguments) == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["environment"] == [{
        "name": "SSH_PUBLIC_KEY", "value_chars": len(public_key),
        "value_sha256": controller.sha256_text(public_key),
    }]
    assert public_key not in output


def test_rate_change_triggers_immediate_finally_delete(tmp_path) -> None:
    clock = FakeClock()
    transport = FakeTransport(
        [
            (201, {"id": "pod-2"}),
            (200, [{"id": "pod-2", "desiredStatus": "RUNNING", "costPerHr": 0.75}]),
            (204, {}),
            (200, []),
        ]
    )
    with pytest.raises(RuntimeError, match="exceeds admitted"):
        controller.execute(
            config=config(),
            config_path=CONFIG_PATH,
            spec=spec("runpod"),
            budget=budget(),
            receipt_path=tmp_path / "rate.json",
            token="secret",
            transport=transport,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    receipt = json.loads((tmp_path / "rate.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "resource-destroyed"
    assert any(row["event"] == "delete-confirmed" for row in receipt["events"])


def test_delete_retries_after_guest_failure(tmp_path) -> None:
    clock = FakeClock()
    transport = FakeTransport(
        [
            (201, {"id": "pod-3"}),
            (200, [{"id": "pod-3", "desiredStatus": "EXITED", "costPerHr": 0.69}]),
            RuntimeError("temporary delete failure"),
            (204, {}),
            (200, []),
        ]
    )
    result = controller.execute(
        config=config(),
        config_path=CONFIG_PATH,
        spec=spec("runpod"),
        budget=budget(),
        receipt_path=tmp_path / "retry.json",
        token="secret",
        transport=transport,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert result["status"] == "resource-destroyed"
    assert [row["event"] for row in result["events"]].count("delete-retry") == 1
    assert result["timing"]["provider_billed_lifetime_seconds"] == 5.0


def test_vast_uses_offer_contract_and_destroy_endpoints(tmp_path) -> None:
    clock = FakeClock()
    transport = FakeTransport(
        [
            (200, {"success": True, "new_contract": 789}),
            (200, {"instances": {"id": 789, "actual_status": "running", "dph_total": 0.4}}),
            (200, {"instances": {"id": 789, "actual_status": "stopped", "dph_total": 0.4}}),
            (200, {"success": True}),
            RuntimeError("provider HTTP 404 for GET instance"),
        ]
    )
    result = controller.execute(
        config=config(),
        config_path=CONFIG_PATH,
        spec=spec("vast"),
        budget=budget(rate="0.40"),
        receipt_path=tmp_path / "vast.json",
        token="vast-secret-test-token",
        transport=transport,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert result["status"] == "resource-destroyed"
    assert transport.calls[0]["method"] == "PUT"
    assert transport.calls[0]["url"].endswith("/asks/123456/")
    assert transport.calls[-2]["method"] == "DELETE"
    assert transport.calls[-2]["url"].endswith("/instances/789/")
    assert transport.calls[0]["payload"]["env"] == {}
    assert "vast-secret-test-token" not in (tmp_path / "vast.json").read_text(encoding="utf-8")


def test_guest_credentials_and_unauthorized_live_mode_are_rejected(monkeypatch) -> None:
    unsafe = controller.LaunchSpec(
        **{**spec("runpod").__dict__, "command": "echo $RUNPOD_API_KEY"}
    )
    with pytest.raises(ValueError, match="forbidden credential marker"):
        controller.validate_guest(unsafe, config(), None)
    monkeypatch.delenv("SAIR_CLOUD_LAUNCH_APPROVED", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-be-read-without-approval")
    with pytest.raises(RuntimeError, match="requires SAIR_CLOUD_LAUNCH_APPROVED"):
        controller.main(
            [
                "--config",
                str(CONFIG_PATH),
                "run",
                "--provider",
                "runpod",
                "--resource-selector",
                "NVIDIA GeForce RTX 4090",
                "--image",
                "example/pytorch:cuda",
                "--hourly-rate-usd",
                "0.69",
                "--storage-hourly-usd",
                "0.006945",
                "--max-total-usd",
                "1.00",
                "--max-provider-seconds",
                "1800",
                "--command",
                "true",
            ]
        )
