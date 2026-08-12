"""Immutable checkpoint generations and independently verified restore."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gzip
import io
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Iterable, Iterator

from .errors import CloudExperimentError
from .io import (
    atomic_bytes,
    atomic_json,
    file_identity,
    read_json,
    resolve_within,
    sha256_bytes,
    sha256_file,
)


CHECKPOINT_SCHEMA = "sair-cloud-checkpoint-v1"


@contextmanager
def checkpoint_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".checkpoint.lock"
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _selected_files(workspace: Path, declared: Iterable[str]) -> list[Path]:
    selected: dict[str, Path] = {}
    for value in declared:
        path = resolve_within(workspace, value, label="checkpoint path")
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        for candidate in candidates:
            if candidate.is_symlink():
                raise CloudExperimentError(f"checkpoint inputs may not be symlinks: {candidate}")
            selected[candidate.relative_to(workspace).as_posix()] = candidate
    return [selected[key] for key in sorted(selected)]


def _archive_bytes(workspace: Path, files: list[Path]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                data = path.read_bytes()
                relative = path.relative_to(workspace).as_posix()
                info = tarfile.TarInfo(relative)
                info.size = len(data)
                info.mode = 0o755 if data.startswith(b"#!") else 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def _next_generation(directory: Path) -> int:
    values: list[int] = []
    for path in directory.glob("generation-*.json"):
        try:
            values.append(int(path.stem.rsplit("-", 1)[-1]))
        except ValueError:
            continue
    return max(values, default=0) + 1


def publish_checkpoint(
    *,
    workspace: Path,
    checkpoint_directory: Path,
    declared_paths: Iterable[str],
    manifest_sha256: str,
    package_sha256: str,
    phase_id: str,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    checkpoint_directory = checkpoint_directory.resolve()
    with checkpoint_lock(checkpoint_directory):
        files = _selected_files(workspace, declared_paths)
        if not files:
            raise CloudExperimentError("no declared checkpoint files exist")
        generation = _next_generation(checkpoint_directory)
        stem = f"generation-{generation:06d}"
        archive_path = checkpoint_directory / f"{stem}.tar.gz"
        metadata_path = checkpoint_directory / f"{stem}.json"
        if archive_path.exists() or metadata_path.exists():
            raise CloudExperimentError(f"checkpoint generation already exists: {stem}")
        data = _archive_bytes(workspace, files)
        metadata = {
            "schema_version": CHECKPOINT_SCHEMA,
            "generation": generation,
            "phase_id": phase_id,
            "manifest_sha256": manifest_sha256,
            "package_sha256": package_sha256,
            "archive": {
                "path": archive_path.name,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            },
            "files": [file_identity(path, relative_to=workspace) for path in files],
        }
        atomic_bytes(archive_path, data)
        atomic_json(metadata_path, metadata)
        atomic_json(
            checkpoint_directory / "latest.json",
            {
                "schema_version": "sair-cloud-checkpoint-pointer-v1",
                "generation": generation,
                "metadata": metadata_path.name,
                "metadata_sha256": sha256_file(metadata_path),
                "archive": archive_path.name,
                "archive_sha256": metadata["archive"]["sha256"],
            },
        )
        return metadata


def latest_checkpoint(checkpoint_directory: Path) -> dict[str, Any] | None:
    pointer_path = checkpoint_directory / "latest.json"
    if not pointer_path.is_file():
        return None
    pointer = read_json(pointer_path)
    metadata_path = checkpoint_directory / str(pointer.get("metadata", ""))
    if sha256_file(metadata_path) != pointer.get("metadata_sha256"):
        raise CloudExperimentError("checkpoint metadata pointer hash mismatch")
    metadata = read_json(metadata_path)
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA:
        raise CloudExperimentError("checkpoint metadata schema mismatch")
    archive_path = checkpoint_directory / str(metadata.get("archive", {}).get("path", ""))
    if sha256_file(archive_path) != metadata.get("archive", {}).get("sha256"):
        raise CloudExperimentError("checkpoint archive hash mismatch")
    if pointer.get("archive_sha256") != metadata.get("archive", {}).get("sha256"):
        raise CloudExperimentError("checkpoint pointer archive hash mismatch")
    return metadata


def restore_latest_checkpoint(
    *,
    checkpoint_directory: Path,
    workspace: Path,
    manifest_sha256: str,
    package_sha256: str,
) -> dict[str, Any] | None:
    metadata = latest_checkpoint(checkpoint_directory)
    if metadata is None:
        return None
    if metadata.get("manifest_sha256") != manifest_sha256:
        raise CloudExperimentError("checkpoint manifest identity mismatch")
    if metadata.get("package_sha256") != package_sha256:
        raise CloudExperimentError("checkpoint package identity mismatch")
    archive_path = checkpoint_directory / metadata["archive"]["path"]
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                pure = PurePosixPath(member.name)
                if member.issym() or member.islnk() or pure.is_absolute() or ".." in pure.parts:
                    raise CloudExperimentError(f"unsafe checkpoint member: {member.name}")
                resolve_within(workspace, member.name, label="checkpoint member")
            archive.extractall(workspace, members=members, filter="data")
    except tarfile.TarError as exc:
        raise CloudExperimentError(f"checkpoint archive cannot be restored: {exc}") from exc
    actual = [
        file_identity(resolve_within(workspace, item["path"], label="checkpoint file"), relative_to=workspace)
        for item in metadata["files"]
    ]
    if actual != metadata["files"]:
        raise CloudExperimentError("restored checkpoint tree identity mismatch")
    return metadata
