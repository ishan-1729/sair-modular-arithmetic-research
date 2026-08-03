"""Budget-admitted external RunPod/Vast lifecycle controller.

Plan mode is side-effect free and never reads credentials. Execute mode is
explicitly authorization-gated, keeps provider credentials in the controller,
and confirms DELETE from a finally path before reporting cost closure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import FrameType
from typing import Any, Callable, Protocol
from urllib import error, parse, request

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/cuda_provider_control.yaml"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def money(value: str | float | Decimal) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("money values must be finite")
    return result


@dataclass(frozen=True)
class BudgetPlan:
    hourly_rate_usd: str
    storage_hourly_usd: str
    max_total_usd: str
    max_provider_seconds: int
    reserve_fraction: str
    admitted_upper_usd: str
    budget_after_reserve_usd: str
    budget_limited_seconds: int


def admit_budget(
    *,
    hourly_rate_usd: Decimal,
    storage_hourly_usd: Decimal,
    max_total_usd: Decimal,
    max_provider_seconds: int,
    reserve_fraction: Decimal,
    minimum_total_usd: Decimal,
) -> BudgetPlan:
    if hourly_rate_usd <= 0 or storage_hourly_usd < 0:
        raise ValueError("provider and storage rates must be positive/non-negative")
    if max_total_usd < minimum_total_usd or max_provider_seconds <= 0:
        raise ValueError("budget and provider lifetime must be positive")
    if reserve_fraction < 0 or reserve_fraction >= 1:
        raise ValueError("budget reserve must be in [0, 1)")
    combined_rate = hourly_rate_usd + storage_hourly_usd
    available = max_total_usd * (Decimal(1) - reserve_fraction)
    upper = combined_rate * Decimal(max_provider_seconds) / Decimal(3600)
    budget_seconds = int(
        (available * Decimal(3600) / combined_rate).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )
    if upper > available:
        raise ValueError(
            f"projected provider upper ${upper:.6f} exceeds reserved budget "
            f"${available:.6f}; reduce lifetime/rate or raise the explicit cap"
        )
    return BudgetPlan(
        hourly_rate_usd=str(hourly_rate_usd),
        storage_hourly_usd=str(storage_hourly_usd),
        max_total_usd=str(max_total_usd),
        max_provider_seconds=int(max_provider_seconds),
        reserve_fraction=str(reserve_fraction),
        admitted_upper_usd=str(upper.quantize(Decimal("0.000001"))),
        budget_after_reserve_usd=str(available.quantize(Decimal("0.000001"))),
        budget_limited_seconds=budget_seconds,
    )


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        token: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, Any]: ...


class HttpTransport:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = float(timeout_seconds)

    def request(
        self,
        method: str,
        url: str,
        token: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        data = None if payload is None else canonical_json(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        operation = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(operation, timeout=self.timeout_seconds) as response:
                raw = response.read()
                return response.status, {} if not raw else json.loads(raw)
        except error.HTTPError as failure:
            raw = failure.read()
            detail = raw.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"provider HTTP {failure.code} for {method} {url}: {detail[:500]}"
            ) from failure


class Provider:
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        token: str,
        transport: Transport,
    ) -> None:
        self.name = name
        self.config = config
        self.token = token
        self.transport = transport

    def url(self, field: str, **values: str) -> str:
        path = str(self.config[field]).format(**values)
        return f"{self.config['base_url'].rstrip('/')}/{path.lstrip('/')}"

    def request(
        self,
        operation: str,
        payload: dict[str, Any] | None = None,
        **values: str,
    ) -> tuple[int, Any]:
        return self.transport.request(
            str(self.config[f"{operation}_method"]),
            self.url(f"{operation}_path", **values),
            self.token,
            payload,
        )

    def create_payload(self, spec: "LaunchSpec") -> dict[str, Any]:
        if self.name == "runpod":
            return {
                "name": spec.label,
                "imageName": spec.image,
                "computeType": "GPU",
                "gpuTypeIds": [spec.resource_selector],
                "gpuCount": 1,
                "cloudType": "SECURE",
                "containerDiskInGb": spec.disk_gb,
                "volumeInGb": 0,
                "allowedCudaVersions": [spec.cuda_runtime],
                "dockerEntrypoint": ["bash", "-lc"],
                "dockerStartCmd": [spec.command],
                "env": {},
                "locked": False,
            }
        return {
            "image": spec.image,
            "label": spec.label,
            "disk": spec.disk_gb,
            "runtype": "args",
            "target_state": "running",
            "args": ["bash", "-lc", spec.command],
            "cancel_unavail": True,
            "env": {},
        }

    def create(self, spec: "LaunchSpec") -> tuple[str, dict[str, Any]]:
        payload = self.create_payload(spec)
        _, response = self.request(
            "create", payload, resource_selector=spec.resource_selector
        )
        raw_id = response.get("id") if self.name == "runpod" else response.get(
            "new_contract"
        )
        if raw_id is None:
            raise RuntimeError("provider create response omitted the resource id")
        return str(raw_id), response

    def status(self, resource_id: str) -> dict[str, Any] | None:
        _, response = self.request("status", resource_id=resource_id)
        if self.name == "runpod":
            if isinstance(response, list):
                return response[0] if response else None
            return response
        return response.get("instances") if isinstance(response, dict) else None

    def status_name(self, status: dict[str, Any] | None) -> str | None:
        if status is None:
            return None
        field = "desiredStatus" if self.name == "runpod" else "actual_status"
        value = status.get(field)
        return None if value is None else str(value)

    def observed_hourly_rate(self, status: dict[str, Any] | None) -> Decimal | None:
        if not status:
            return None
        candidates: list[Any]
        if self.name == "runpod":
            candidates = [status.get("adjustedCostPerHr"), status.get("costPerHr")]
        else:
            nested = status.get("instance") or {}
            candidates = [
                status.get("dph_total"),
                nested.get("discountedTotalPerHour"),
                nested.get("totalHour"),
            ]
        for candidate in candidates:
            if candidate is not None:
                return money(candidate)
        return None

    def delete(self, resource_id: str) -> Any:
        _, response = self.request("delete", resource_id=resource_id)
        return response

    def deletion_confirmed(self, resource_id: str) -> bool:
        try:
            status = self.status(resource_id)
        except RuntimeError as failure:
            if "HTTP 404" in str(failure):
                return True
            raise
        if status is None:
            return True
        return self.status_name(status) in {"TERMINATED", "terminated", "deleted"}


@dataclass(frozen=True)
class LaunchSpec:
    provider: str
    resource_selector: str
    image: str
    label: str
    command: str
    cuda_runtime: str
    disk_gb: int


class TerminationRequested(RuntimeError):
    pass


def termination_handler(signum: int, _frame: FrameType | None) -> None:
    raise TerminationRequested(f"received signal {signum}")


def validate_guest(spec: LaunchSpec, config: dict[str, Any], token: str | None) -> None:
    rendered = canonical_json(asdict(spec))
    for fragment in config["guest"]["forbidden_fragments"]:
        if str(fragment).lower() in rendered.lower():
            raise ValueError(f"guest payload contains forbidden credential marker: {fragment}")
    if token and token in rendered:
        raise ValueError("provider credential appears in guest payload")


def redacted_plan(
    spec: LaunchSpec,
    budget: BudgetPlan,
    config_path: Path,
    provider_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "cuda-provider-plan-v1",
        "status": "admitted-dry-run",
        "provider": spec.provider,
        "resource_selector": spec.resource_selector,
        "image": spec.image,
        "label": spec.label,
        "command_sha256": sha256_text(spec.command),
        "cuda_runtime": spec.cuda_runtime,
        "disk_gb": spec.disk_gb,
        "credential_environment": provider_config["credential_environment"],
        "budget": asdict(budget),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "limitations": [
            "No provider request was made and no current offer price was verified.",
            "Client power/network/process failure can outlive this controller; use prepaid balance and disable auto-recharge before launch.",
            "Guest completion is not cost closure; only confirmed DELETE is cost closure.",
        ],
    }


class Receipt:
    def __init__(self, path: Path, initial: dict[str, Any]) -> None:
        self.path = path
        self.value = initial
        self.write()

    def event(self, name: str, elapsed: float, **evidence: Any) -> None:
        self.value.setdefault("events", []).append(
            {"event": name, "elapsed_seconds": elapsed, **evidence}
        )
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def execute(
    *,
    config: dict[str, Any],
    config_path: Path,
    spec: LaunchSpec,
    budget: BudgetPlan,
    receipt_path: Path,
    token: str,
    transport: Transport,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    provider_config = config["providers"][spec.provider]
    validate_guest(spec, config, token)
    started = monotonic()
    resource_id: str | None = None
    running_at: float | None = None
    terminal_at: float | None = None
    destroyed_at: float | None = None
    lifecycle = config["lifecycle"]
    receipt = Receipt(
        receipt_path,
        {
            **redacted_plan(spec, budget, config_path, provider_config),
            "schema_version": lifecycle["receipt_schema"],
            "status": "controller-started",
            "events": [],
        },
    )
    provider = Provider(spec.provider, provider_config, token, transport)
    prior_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        prior_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, termination_handler)
    failure: Exception | None = None
    try:
        receipt.event("create-requested", monotonic() - started)
        resource_id, create_response = provider.create(spec)
        receipt.value["resource_id"] = resource_id
        receipt.event(
            "resource-created",
            monotonic() - started,
            response_sha256=sha256_text(canonical_json(create_response)),
        )
        while True:
            elapsed = monotonic() - started
            if elapsed >= budget.max_provider_seconds:
                raise TimeoutError("provider lifetime ceiling reached")
            status = provider.status(resource_id)
            name = provider.status_name(status)
            observed_rate = provider.observed_hourly_rate(status)
            receipt.event(
                "provider-status",
                elapsed,
                provider_status=name,
                observed_hourly_rate_usd=(
                    None if observed_rate is None else str(observed_rate)
                ),
            )
            if observed_rate is not None and observed_rate > money(
                budget.hourly_rate_usd
            ):
                raise RuntimeError(
                    f"provider rate {observed_rate} exceeds admitted "
                    f"{budget.hourly_rate_usd}"
                )
            if name in set(provider_config["running_statuses"]) and running_at is None:
                running_at = monotonic()
            if name in set(provider_config["terminal_statuses"]):
                terminal_at = monotonic()
                break
            sleep(float(lifecycle["poll_seconds"]))
        receipt.value["guest_terminal_status"] = name
    except Exception as error_value:
        failure = error_value
        receipt.value["failure"] = {
            "type": type(error_value).__name__,
            "message": str(error_value),
        }
    finally:
        if resource_id is not None:
            destroy_started = monotonic()
            attempts = 0
            while monotonic() - destroy_started < float(
                lifecycle["destroy_timeout_seconds"]
            ):
                attempts += 1
                try:
                    response = provider.delete(resource_id)
                    receipt.event(
                        "delete-requested",
                        monotonic() - started,
                        attempt=attempts,
                        response_sha256=sha256_text(canonical_json(response)),
                    )
                    if provider.deletion_confirmed(resource_id):
                        destroyed_at = monotonic()
                        receipt.event(
                            "delete-confirmed",
                            destroyed_at - started,
                            attempt=attempts,
                        )
                        break
                except Exception as delete_error:
                    if "HTTP 404" in str(delete_error):
                        destroyed_at = monotonic()
                        receipt.event(
                            "delete-confirmed",
                            destroyed_at - started,
                            attempt=attempts,
                            already_absent=True,
                        )
                        break
                    receipt.event(
                        "delete-retry",
                        monotonic() - started,
                        attempt=attempts,
                        error_type=type(delete_error).__name__,
                        error=str(delete_error),
                    )
                sleep(float(lifecycle["destroy_retry_seconds"]))
        for signum, prior in prior_handlers.items():
            signal.signal(signum, prior)

    finished = monotonic()
    billed_lifetime = None if resource_id is None else finished - started
    guest_runtime = (
        None
        if running_at is None or terminal_at is None
        else terminal_at - running_at
    )
    receipt.value["timing"] = {
        "provider_billed_lifetime_seconds": billed_lifetime,
        "provider_billed_lifetime_source": (
            "controller-create-request-to-delete-confirmation-upper-bound"
        ),
        "provider_invoice_amount_usd": None,
        "guest_runtime_seconds": guest_runtime,
        "deletion_confirmed_elapsed_seconds": (
            None if destroyed_at is None else destroyed_at - started
        ),
    }
    if billed_lifetime is not None:
        combined = money(budget.hourly_rate_usd) + money(
            budget.storage_hourly_usd
        )
        receipt.value["cost_upper_at_confirmation_usd"] = str(
            (combined * Decimal(str(billed_lifetime)) / Decimal(3600)).quantize(
                Decimal("0.000001")
            )
        )
    if resource_id is not None and destroyed_at is None:
        receipt.value["status"] = "destruction-unconfirmed"
        receipt.write()
        raise RuntimeError(f"provider destruction is unconfirmed: {resource_id}")
    receipt.value["status"] = "resource-destroyed" if resource_id else "create-failed"
    receipt.write()
    if failure is not None:
        raise failure
    return receipt.value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "run"):
        command = subparsers.add_parser(mode)
        command.add_argument("--provider", choices=("runpod", "vast"), required=True)
        command.add_argument("--resource-selector", required=True)
        command.add_argument("--image", required=True)
        command.add_argument("--hourly-rate-usd", required=True)
        command.add_argument("--storage-hourly-usd", required=True)
        command.add_argument("--max-total-usd", required=True)
        command.add_argument("--max-provider-seconds", required=True, type=int)
        command.add_argument("--command", required=True)
        command.add_argument("--label", default="sair-cuda-acceptance")
        command.add_argument("--cuda-runtime", default="12.4")
        command.add_argument("--disk-gb", type=int, default=50)
        command.add_argument("--receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != "cuda-provider-control-v1":
        raise ValueError("unsupported provider control schema")
    if args.disk_gb <= 0:
        raise ValueError("disk size must be positive")
    spec = LaunchSpec(
        provider=args.provider,
        resource_selector=str(args.resource_selector),
        image=args.image,
        label=args.label,
        command=args.command,
        cuda_runtime=args.cuda_runtime,
        disk_gb=args.disk_gb,
    )
    provider_config = config["providers"][args.provider]
    validate_guest(spec, config, None)
    budget = admit_budget(
        hourly_rate_usd=money(args.hourly_rate_usd),
        storage_hourly_usd=money(args.storage_hourly_usd),
        max_total_usd=money(args.max_total_usd),
        max_provider_seconds=args.max_provider_seconds,
        reserve_fraction=money(config["budget"]["reserve_fraction"]),
        minimum_total_usd=money(config["budget"]["minimum_max_total_usd"]),
    )
    if args.mode == "plan":
        result = redacted_plan(spec, budget, config_path, provider_config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    authorization = config["authorization"]
    if os.environ.get(authorization["environment"]) != authorization["required_value"]:
        raise RuntimeError(
            f"live provider execution requires {authorization['environment']}="
            f"{authorization['required_value']}"
        )
    credential_name = provider_config["credential_environment"]
    token = os.environ.get(credential_name)
    if not token:
        raise RuntimeError(f"missing external provider credential: {credential_name}")
    receipt_path = (
        args.receipt
        or ROOT / ".scratch" / f"cuda-provider-{args.provider}-receipt.json"
    ).resolve()
    result = execute(
        config=config,
        config_path=config_path,
        spec=spec,
        budget=budget,
        receipt_path=receipt_path,
        token=token,
        transport=HttpTransport(config["lifecycle"]["request_timeout_seconds"]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
