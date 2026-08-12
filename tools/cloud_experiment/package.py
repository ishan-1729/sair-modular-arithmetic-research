"""Build and verify deterministic source packages from generic manifests."""

from __future__ import annotations

from fnmatch import fnmatchcase
import gzip
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from typing import Any, Iterable

from .errors import CloudExperimentError
from .io import (
    atomic_bytes,
    atomic_json,
    canonical_json_bytes,
    file_identity,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .manifest import LoadedManifest, sanitized_manifest


PACKAGE_SCHEMA = "sair-cloud-package-v1"


def _matches(path: str, patterns: Iterable[str]) -> bool:
    pure = PurePosixPath(path)
    return any(fnmatchcase(path, pattern) or pure.match(pattern) for pattern in patterns)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _dirty_paths(root: Path) -> list[str]:
    try:
        tracked = subprocess.check_output(
            ["git", "diff", "--name-only", "-z", "HEAD"], cwd=root
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CloudExperimentError("cannot inspect Git worktree state") from exc
    try:
        paths = [
            item.decode("utf-8")
            for item in [*tracked.split(b"\0"), *untracked.split(b"\0")]
            if item
        ]
    except UnicodeDecodeError as exc:
        raise CloudExperimentError("Git worktree paths must be UTF-8") from exc
    return sorted(set(paths))


def _repository_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CloudExperimentError("cannot read repository commit") from exc


def collect_package_files(manifest: LoadedManifest) -> tuple[list[Path], list[str]]:
    config = manifest.data["package"]
    include = tuple(str(value) for value in config["include"])
    exclude = tuple(str(value) for value in config["exclude"])
    automatic_excludes = {
        manifest.run_directory.resolve(),
        manifest.publication_directory.resolve(),
    }
    selected: dict[str, Path] = {}
    for pattern in include:
        for path in manifest.repo_root.glob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if path.is_symlink():
                raise CloudExperimentError(f"package inputs may not be symlinks: {path}")
            if any(root == resolved or root in resolved.parents for root in automatic_excludes):
                continue
            relative = _relative(path, manifest.repo_root)
            if relative.startswith(".git/") or _matches(relative, exclude):
                continue
            selected[relative] = path
    if not selected:
        raise CloudExperimentError("package.include selected no files")
    dirty = _dirty_paths(manifest.repo_root)
    relevant_dirty = sorted(
        value
        for value in dirty
        if value in selected or _matches(value, include)
    )
    if config["require_clean"] and relevant_dirty:
        raise CloudExperimentError(
            "package requires clean included inputs: " + ", ".join(relevant_dirty)
        )
    return [selected[key] for key in sorted(selected)], relevant_dirty


def _tar_member(name: str, data: bytes, *, mode: int) -> tuple[tarfile.TarInfo, io.BytesIO]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info, io.BytesIO(data)


def _archive_bytes(manifest: LoadedManifest, files: list[Path], source_record: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                relative = _relative(path, manifest.repo_root)
                data = path.read_bytes()
                mode = 0o755 if data.startswith(b"#!") else 0o644
                info, handle = _tar_member(relative, data, mode=mode)
                archive.addfile(info, handle)
            manifest_bytes = canonical_json_bytes(sanitized_manifest(manifest))
            info, handle = _tar_member(
                ".sair-cloud/manifest.json", manifest_bytes, mode=0o644
            )
            archive.addfile(info, handle)
            info, handle = _tar_member(
                ".sair-cloud/source.json", canonical_json_bytes(source_record), mode=0o644
            )
            archive.addfile(info, handle)
    return output.getvalue()


def build_package(manifest: LoadedManifest, output_directory: Path | None = None) -> dict[str, Any]:
    output = (output_directory or (manifest.run_directory / "package")).resolve()
    files, dirty = collect_package_files(manifest)
    source_files = [file_identity(path, relative_to=manifest.repo_root) for path in files]
    source_record = {
        "schema_version": "sair-cloud-source-v1",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "repository_commit": _repository_commit(manifest.repo_root),
        "source_files": source_files,
        "dirty_included_paths": dirty,
    }
    archive_data = _archive_bytes(manifest, files, source_record)
    archive_path = output / "source.tar.gz"
    atomic_bytes(archive_path, archive_data)
    receipt = {
        "schema_version": PACKAGE_SCHEMA,
        "run_id": manifest.run_id,
        "manifest": {
            "path": _relative(manifest.path, manifest.repo_root),
            "sha256": manifest.sha256,
        },
        "repository_commit": source_record["repository_commit"],
        "dirty_included_paths": dirty,
        "source_files": source_files,
        "source_archive": {
            "path": archive_path.name,
            "bytes": len(archive_data),
            "sha256": sha256_bytes(archive_data),
        },
        "provider_mutation_performed": False,
        "credential_read_performed": False,
    }
    atomic_json(output / "package.json", receipt)
    return receipt


def _safe_archive_names(archive: tarfile.TarFile) -> list[str]:
    names: list[str] = []
    for member in archive.getmembers():
        pure = PurePosixPath(member.name)
        if member.issym() or member.islnk() or pure.is_absolute() or ".." in pure.parts:
            raise CloudExperimentError(f"unsafe source archive member: {member.name}")
        if member.isfile():
            names.append(member.name)
    return names


def verify_package(manifest: LoadedManifest, output_directory: Path | None = None) -> dict[str, Any]:
    output = (output_directory or (manifest.run_directory / "package")).resolve()
    receipt = read_json(output / "package.json")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != PACKAGE_SCHEMA:
        raise CloudExperimentError("package receipt schema mismatch")
    if receipt.get("run_id") != manifest.run_id:
        raise CloudExperimentError("package run identity mismatch")
    if receipt.get("manifest", {}).get("sha256") != manifest.sha256:
        raise CloudExperimentError("package manifest identity mismatch")
    archive_path = output / str(receipt.get("source_archive", {}).get("path", ""))
    expected = receipt.get("source_archive", {})
    actual = file_identity(archive_path, relative_to=output)
    if actual["sha256"] != expected.get("sha256") or actual["bytes"] != expected.get("bytes"):
        raise CloudExperimentError("source package bytes changed after packaging")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            names = _safe_archive_names(archive)
            embedded = archive.extractfile(".sair-cloud/manifest.json")
            if embedded is None:
                raise CloudExperimentError("source package omits embedded manifest")
            embedded_manifest = json.loads(embedded.read().decode("utf-8"))
    except (tarfile.TarError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudExperimentError(f"source package is not independently readable: {exc}") from exc
    if sha256_bytes(canonical_json_bytes(embedded_manifest)) != manifest.sha256:
        raise CloudExperimentError("embedded manifest identity mismatch")
    if sorted(item["path"] for item in receipt["source_files"]) != sorted(
        name for name in names if not name.startswith(".sair-cloud/")
    ):
        raise CloudExperimentError("source archive inventory mismatch")
    return receipt
