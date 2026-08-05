"""Create, verify, and restore authenticated EXP701 recovery exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_tree(output: Path) -> dict[str, Any]:
    if not output.is_dir():
        raise FileNotFoundError(output)
    snapshots: list[dict[str, Any]] = []
    for manifest_path in sorted(output.rglob("recovery/*/latest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = manifest_path.with_name("latest.pt")
        actual = sha256(snapshot)
        if actual != manifest.get("snapshot_sha256"):
            raise RuntimeError(f"recovery snapshot mismatch: {manifest_path}")
        snapshots.append(
            {
                "manifest": manifest_path.relative_to(output).as_posix(),
                "snapshot_sha256": actual,
                "identity_sha256": manifest.get("identity_sha256"),
                "arm": manifest.get("arm"),
                "update": manifest.get("update"),
            }
        )
    phases: list[dict[str, Any]] = []
    for manifest_path in sorted(output.rglob("phases/*.manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        result_path = manifest_path.with_name(
            manifest_path.name.removesuffix(".manifest.json") + ".json"
        )
        actual = sha256(result_path)
        if actual != manifest.get("result_sha256"):
            raise RuntimeError(f"completed phase mismatch: {manifest_path}")
        phases.append(
            {
                "manifest": manifest_path.relative_to(output).as_posix(),
                "result_sha256": actual,
                "phase_identity_sha256": manifest.get("phase_identity_sha256"),
            }
        )
    roots = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("root-opened.json")
    )
    files = [path for path in output.rglob("*") if path.is_file()]
    return {
        "schema_version": "exp701-recovery-tree-verification-v1",
        "output": str(output),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "root_markers": roots,
        "snapshots": snapshots,
        "completed_phases": phases,
        "passed": True,
    }


def create_export(
    output: Path,
    archive: Path,
    *,
    mlflow: Path | None = None,
) -> dict[str, Any]:
    output = output.resolve()
    archive = archive.resolve()
    verification = verify_tree(output)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
        bundle.add(output, arcname=f"experiment/{output.name}", recursive=True)
        if mlflow is not None and mlflow.is_dir():
            bundle.add(mlflow.resolve(), arcname="mlflow", recursive=True)
    os.replace(temporary, archive)
    digest = sha256(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum_temporary = checksum.with_name(f".{checksum.name}.tmp-{os.getpid()}")
    checksum_temporary.write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    os.replace(checksum_temporary, checksum)
    receipt = {
        "schema_version": "exp701-recovery-export-v1",
        "status": "verified",
        "archive": str(archive),
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "experiment_directory": output.name,
        "tree_verification": verification,
        "contains_mlflow": bool(mlflow is not None and mlflow.is_dir()),
    }
    atomic_json(archive.with_suffix(archive.suffix + ".receipt.json"), receipt)
    return receipt


def _safe_members(bundle: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    root = destination.resolve()
    members = bundle.getmembers()
    for member in members:
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(f"unsafe recovery archive member: {member.name}")
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"recovery archive escapes destination: {member.name}")
    return members


def restore_export(
    archive: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    archive = archive.resolve()
    if sha256(archive) != expected_sha256:
        raise RuntimeError("off-Pod recovery export SHA-256 mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError("recovery destination must be empty")
    with tarfile.open(archive, "r:gz") as bundle:
        members = _safe_members(bundle, destination)
        bundle.extractall(destination, members=members, filter="data")
    experiment_parent = destination / "experiment"
    candidates = [path for path in experiment_parent.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError("recovery export must contain exactly one experiment tree")
    verification = verify_tree(candidates[0])
    return {
        "schema_version": "exp701-recovery-restore-v1",
        "status": "verified",
        "archive_sha256": expected_sha256,
        "restored_output": str(candidates[0]),
        "tree_verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--archive", type=Path, required=True)
    export.add_argument("--mlflow", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    restore = sub.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = create_export(args.output, args.archive, mlflow=args.mlflow)
    elif args.command == "verify":
        result = verify_tree(args.output)
    else:
        result = restore_export(args.archive, args.destination, expected_sha256=args.sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
