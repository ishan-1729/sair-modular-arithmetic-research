"""Public CLI for the experiment-agnostic cloud lifecycle."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

from .checkpoints import publish_checkpoint
from .errors import CloudExperimentError
from .io import canonical_json_bytes, read_json
from .lifecycle import (
    AUTHORIZATION_SCHEMA,
    canary,
    cleanup,
    doctor,
    ensure_package,
    finalize,
    launch,
    resume,
    status,
)
from .manifest import LoadedManifest, load_manifest
from .package import build_package, verify_package
from .schema import MANIFEST_SCHEMA


def _print(value: Any) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def _manifest(args: argparse.Namespace) -> LoadedManifest:
    return load_manifest(args.manifest)


def _path(value: str | None) -> Path | None:
    return None if value is None else Path(value).resolve()


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _manifest(args)
    return {
        "schema_version": "sair-cloud-manifest-validation-v1",
        "status": "valid",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "repo_root": str(manifest.repo_root),
    }


def command_schema(_args: argparse.Namespace) -> dict[str, Any]:
    return MANIFEST_SCHEMA


def command_doctor(args: argparse.Namespace) -> dict[str, Any]:
    return doctor(_manifest(args))


def command_package(args: argparse.Namespace) -> dict[str, Any]:
    return build_package(_manifest(args), _path(args.output_directory))


def command_verify_package(args: argparse.Namespace) -> dict[str, Any]:
    return verify_package(_manifest(args), _path(args.output_directory))


def command_canary(args: argparse.Namespace) -> dict[str, Any]:
    return canary(
        _manifest(args),
        package_directory=_path(args.package_directory),
        authorization_path=_path(args.authorization),
    )


def command_launch(args: argparse.Namespace) -> dict[str, Any]:
    return launch(
        _manifest(args),
        package_directory=_path(args.package_directory),
        authorization_path=_path(args.authorization),
    )


def command_resume(args: argparse.Namespace) -> dict[str, Any]:
    return resume(
        _manifest(args),
        package_directory=_path(args.package_directory),
        authorization_path=_path(args.authorization),
    )


def command_status(args: argparse.Namespace) -> dict[str, Any]:
    return status(_manifest(args))


def command_cleanup(args: argparse.Namespace) -> dict[str, Any]:
    return cleanup(_manifest(args))


def command_finalize(args: argparse.Namespace) -> dict[str, Any]:
    return finalize(_manifest(args))


def command_authorization_template(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _manifest(args)
    if not manifest.data["provider"]["paid"]:
        raise CloudExperimentError("unpaid providers do not require authorization")
    if args.expires_at_utc:
        expires = args.expires_at_utc
    else:
        expires = (
            datetime.now(timezone.utc) + timedelta(minutes=args.valid_for_minutes)
        ).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "provider_driver": manifest.data["provider"]["driver"],
        "maximum_total_cost_usd": str(manifest.data["budget"]["max_total_usd"]),
        "maximum_provider_lifetime_seconds": manifest.data["budget"][
            "max_provider_seconds"
        ],
        "expires_at_utc": expires,
    }


def command_checkpoint_publish(args: argparse.Namespace) -> dict[str, Any]:
    control_value = args.control_directory or os.environ.get("SAIR_CLOUD_CONTROL_DIR")
    workspace_value = args.workspace or os.environ.get("SAIR_CLOUD_WORKSPACE")
    phase_id = args.phase or os.environ.get("SAIR_CLOUD_PHASE_ID")
    if not control_value or not workspace_value or not phase_id:
        raise CloudExperimentError(
            "checkpoint publish needs control directory, workspace, and phase"
        )
    control = Path(control_value).resolve()
    lock = read_json(control / "manifest.lock.json")
    return publish_checkpoint(
        workspace=Path(workspace_value).resolve(),
        checkpoint_directory=control / "checkpoints",
        declared_paths=args.path,
        manifest_sha256=str(lock["manifest_sha256"]),
        package_sha256=str(lock["package_sha256"]),
        phase_id=str(phase_id),
    )


def _add_manifest(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)


def _add_package_directory(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package-directory", type=Path)


def _add_authorization(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authorization", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.cloud_experiment",
        description=(
            "Validate, package, run, resume, and finalize an arbitrary experiment "
            "through a manifest-driven provider lifecycle."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a manifest")
    _add_manifest(validate_parser)
    validate_parser.set_defaults(handler=command_validate)

    schema_parser = subparsers.add_parser("schema", help="print the manifest JSON schema")
    schema_parser.set_defaults(handler=command_schema)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check provider importability and finite budget admission"
    )
    _add_manifest(doctor_parser)
    doctor_parser.set_defaults(handler=command_doctor)

    package_parser = subparsers.add_parser(
        "package", help="build a deterministic source package"
    )
    _add_manifest(package_parser)
    package_parser.add_argument("--output-directory", type=Path)
    package_parser.set_defaults(handler=command_package)

    verify_parser = subparsers.add_parser(
        "verify-package", help="independently verify an existing source package"
    )
    _add_manifest(verify_parser)
    verify_parser.add_argument("--output-directory", type=Path)
    verify_parser.set_defaults(handler=command_verify_package)

    canary_parser = subparsers.add_parser(
        "canary", help="allocate, run only the environment canary, and clean up"
    )
    _add_manifest(canary_parser)
    _add_package_directory(canary_parser)
    _add_authorization(canary_parser)
    canary_parser.set_defaults(handler=command_canary)

    for name, handler, help_text in (
        ("launch", command_launch, "start a new full lifecycle"),
        ("resume", command_resume, "resume an interrupted lifecycle"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        _add_manifest(child)
        _add_package_directory(child)
        _add_authorization(child)
        child.set_defaults(handler=handler)

    for name, handler, help_text in (
        ("status", command_status, "read durable lifecycle status without provider access"),
        ("cleanup", command_cleanup, "idempotently delete resources owned by the lifecycle"),
        ("finalize", command_finalize, "publish verified outputs after exact cleanup"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        _add_manifest(child)
        child.set_defaults(handler=handler)

    auth_parser = subparsers.add_parser(
        "authorization-template",
        help="print the exact seven-field paid-run authorization template",
    )
    _add_manifest(auth_parser)
    auth_parser.add_argument("--valid-for-minutes", type=int, default=30)
    auth_parser.add_argument("--expires-at-utc")
    auth_parser.set_defaults(handler=command_authorization_template)

    checkpoint_parser = subparsers.add_parser(
        "checkpoint", help="manage immutable experiment checkpoints"
    )
    checkpoint_subparsers = checkpoint_parser.add_subparsers(
        dest="checkpoint_command", required=True
    )
    publish_parser = checkpoint_subparsers.add_parser(
        "publish", help="publish an immutable checkpoint generation"
    )
    publish_parser.add_argument("--control-directory")
    publish_parser.add_argument("--workspace")
    publish_parser.add_argument("--phase")
    publish_parser.add_argument("--path", action="append", required=True)
    publish_parser.set_defaults(handler=command_checkpoint_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except CloudExperimentError as exc:
        sys.stderr.write(
            json.dumps(
                {
                    "schema_version": "sair-cloud-error-v1",
                    "status": "failed-closed",
                    "error": str(exc),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    _print(result)
    return 0
