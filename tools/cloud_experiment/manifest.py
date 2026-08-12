"""Load and semantically validate an experiment-agnostic cloud manifest."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml

from .errors import CloudExperimentError
from .io import canonical_json_bytes, resolve_within, safe_relative_path, sha256_bytes
from .schema import MANIFEST_SCHEMA


SECRET_NAME = re.compile(r"(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE)", re.I)


@dataclass(frozen=True)
class LoadedManifest:
    path: Path
    repo_root: Path
    data: dict[str, Any]
    sha256: str

    @property
    def run_id(self) -> str:
        return str(self.data["experiment"]["id"])

    @property
    def run_directory(self) -> Path:
        return resolve_within(
            self.repo_root,
            self.data["artifacts"]["run_directory"],
            label="artifacts.run_directory",
        )

    @property
    def publication_directory(self) -> Path:
        return resolve_within(
            self.repo_root,
            self.data["artifacts"]["publication_directory"],
            label="artifacts.publication_directory",
        )


def _repo_root(path: Path) -> Path:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CloudExperimentError("manifest must live inside a Git worktree") from exc
    return Path(output).resolve()


def _validate_paths(data: Mapping[str, Any]) -> None:
    path_fields: list[tuple[str, str]] = [
        ("execution.working_directory", data["execution"]["working_directory"]),
        ("artifacts.run_directory", data["artifacts"]["run_directory"]),
        ("artifacts.publication_directory", data["artifacts"]["publication_directory"]),
    ]
    for index, pattern in enumerate(data["package"]["include"]):
        path_fields.append((f"package.include[{index}]", pattern))
    for index, pattern in enumerate(data["package"]["exclude"]):
        path_fields.append((f"package.exclude[{index}]", pattern))
    commands = [("execution.canary", data["execution"]["canary"])] + [
        (f"execution.phases[{index}]", phase)
        for index, phase in enumerate(data["execution"]["phases"])
    ]
    for label, command in commands:
        for key in ("required_outputs", "checkpoint_paths"):
            for index, value in enumerate(command.get(key, [])):
                path_fields.append((f"{label}.{key}[{index}]", value))
    for label, value in path_fields:
        safe_relative_path(str(value), label=label)


def _validate_environment(data: Mapping[str, Any]) -> None:
    environments: list[tuple[str, Mapping[str, str]]] = [
        ("execution.environment", data["execution"].get("environment", {})),
        (
            "execution.canary.environment",
            data["execution"]["canary"].get("environment", {}),
        ),
    ]
    environments.extend(
        (f"execution.phases[{index}].environment", phase.get("environment", {}))
        for index, phase in enumerate(data["execution"]["phases"])
    )
    for label, environment in environments:
        for name in environment:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise CloudExperimentError(f"invalid environment name at {label}: {name!r}")
            if SECRET_NAME.search(name):
                raise CloudExperimentError(
                    f"secret-like environment name is forbidden in manifests: {name}"
                )


def _reject_secret_option_keys(value: Any, *, path: str = "provider.options") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if SECRET_NAME.search(name):
                raise CloudExperimentError(
                    f"secret-like provider option key is forbidden in manifests: {path}.{name}"
                )
            _reject_secret_option_keys(child, path=f"{path}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_option_keys(child, path=f"{path}[{index}]")


def _validate_semantics(data: Mapping[str, Any]) -> None:
    _validate_paths(data)
    _validate_environment(data)
    _reject_secret_option_keys(data["provider"]["options"])
    phase_ids = [str(phase["id"]) for phase in data["execution"]["phases"]]
    if len(set(phase_ids)) != len(phase_ids):
        raise CloudExperimentError("execution phase ids must be unique")
    budget = data["budget"]
    if budget["cleanup_reserve_seconds"] >= budget["max_provider_seconds"]:
        raise CloudExperimentError("cleanup reserve must be below provider lifetime")
    planned = int(data["execution"]["canary"]["timeout_seconds"]) + sum(
        int(phase["timeout_seconds"]) for phase in data["execution"]["phases"]
    )
    compute_window = int(budget["max_provider_seconds"]) - int(
        budget["cleanup_reserve_seconds"]
    )
    if planned > compute_window:
        raise CloudExperimentError(
            f"declared command timeouts ({planned}s) exceed compute window ({compute_window}s)"
        )
    try:
        amount = Decimal(str(budget["max_total_usd"]))
    except InvalidOperation as exc:
        raise CloudExperimentError("budget.max_total_usd must be finite") from exc
    if not amount.is_finite():
        raise CloudExperimentError("budget.max_total_usd must be finite")
    run_path = safe_relative_path(
        data["artifacts"]["run_directory"], label="artifacts.run_directory"
    )
    publication_path = safe_relative_path(
        data["artifacts"]["publication_directory"],
        label="artifacts.publication_directory",
    )
    if run_path == publication_path:
        raise CloudExperimentError("run and publication directories must differ")


def load_manifest(path: Path | str) -> LoadedManifest:
    manifest_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CloudExperimentError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CloudExperimentError("manifest root must be an object")
    errors = sorted(
        Draft202012Validator(MANIFEST_SCHEMA).iter_errors(raw),
        key=lambda error: tuple(str(part) for part in error.path),
    )
    if errors:
        rendered = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise CloudExperimentError(f"manifest schema validation failed: {rendered}")
    _validate_semantics(raw)
    root = _repo_root(manifest_path)
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise CloudExperimentError("manifest resolves outside its Git worktree") from exc
    digest = sha256_bytes(canonical_json_bytes(raw))
    return LoadedManifest(manifest_path, root, dict(raw), digest)


def sanitized_manifest(manifest: LoadedManifest) -> dict[str, Any]:
    """Return the manifest for receipts after a defensive secret-name check."""

    _validate_environment(manifest.data)
    return manifest.data
