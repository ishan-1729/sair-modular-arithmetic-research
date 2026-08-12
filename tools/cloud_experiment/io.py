"""Canonical serialization, hashing, paths, and atomic filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
from typing import Any, Iterable

from .errors import CloudExperimentError


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise CloudExperimentError(f"required file is absent: {path}")
    display = path.relative_to(relative_to).as_posix() if relative_to else str(path)
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudExperimentError(f"invalid JSON receipt {path}: {exc}") from exc


def safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not value or "\x00" in value or "\\" in value:
        raise CloudExperimentError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CloudExperimentError(f"{label} escapes its declared root: {value!r}")
    return path


def resolve_within(root: Path, value: str, *, label: str) -> Path:
    relative = safe_relative_path(value, label=label)
    root_resolved = root.resolve()
    candidate = (root_resolved / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise CloudExperimentError(f"{label} escapes its declared root: {value!r}") from exc
    return candidate


def identities(paths: Iterable[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    return [file_identity(path, relative_to=relative_to) for path in sorted(paths)]
