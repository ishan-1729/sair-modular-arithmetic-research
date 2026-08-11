"""Prepare an audited EXP701 resume export after a pre-update control failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from . import recovery_transport


RUN_ID = "exp-701-cuda-resumable-h92-w256-backbone-factorial"
ROOT = "20261414"
ARM = "straight_through_cosine"
REMOVED_PATHS = (
    "guest-terminal.json",
    f"formal/root-{ROOT}/terminal-failure.json",
    f"formal/root-{ROOT}/phases/train__{ARM}.manifest.json",
)


def file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": recovery_transport.sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def validate_failed_boundary(output: Path) -> dict[str, Any]:
    terminal = json.loads((output / REMOVED_PATHS[0]).read_text(encoding="utf-8"))
    failure = json.loads((output / REMOVED_PATHS[1]).read_text(encoding="utf-8"))
    phase = json.loads((output / REMOVED_PATHS[2]).read_text(encoding="utf-8"))
    if terminal != {"status": "failed", "resume_required": False, "exit_code": 1}:
        raise RuntimeError(f"unexpected guest terminal: {terminal}")
    expected_failure = {
        "mode": "formal",
        "reason": f"resume state missing for {ARM}",
        "retry_allowed": False,
        "root": int(ROOT),
        "run_id": RUN_ID,
        "schema_version": "exp701-terminal-failure-v1",
        "status": "terminal-failure",
    }
    if failure != expected_failure:
        raise RuntimeError(f"unexpected formal terminal failure: {failure}")
    if phase.get("phase") != f"train/{ARM}" or phase.get("status") != "started":
        raise RuntimeError(f"unexpected failed phase marker: {phase}")
    if set(phase) != {"schema_version", "phase", "phase_identity_sha256", "status"}:
        raise RuntimeError(f"failed phase marker contains result state: {phase}")
    arm_root = output / "formal" / f"root-{ROOT}"
    unexpected = [
        path
        for path in (
            arm_root / "recovery" / ARM,
            arm_root / "arms" / ARM,
            arm_root / "phases" / f"train__{ARM}.json",
        )
        if path.exists()
    ]
    if unexpected:
        raise RuntimeError(f"failed arm contains scientific state: {unexpected}")
    return {"guest_terminal": terminal, "terminal_failure": failure, "phase": phase}


def prepare(
    source_archive: Path,
    output_archive: Path,
    receipt_path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if recovery_transport.sha256(source_archive) != expected_sha256:
        raise RuntimeError("input recovery archive SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="exp701-rollback-") as temporary:
        restored = Path(temporary) / "input"
        recovery_transport.restore_export(
            source_archive, restored, expected_sha256=expected_sha256
        )
        output = restored / "experiment" / RUN_ID
        boundary = validate_failed_boundary(output)
        before = file_manifest(output)
        removed = {
            relative: before[relative]
            for relative in REMOVED_PATHS
        }
        for relative in REMOVED_PATHS:
            (output / relative).unlink()
        retained = file_manifest(output)
        expected_retained = {
            path: identity for path, identity in before.items() if path not in REMOVED_PATHS
        }
        if retained != expected_retained:
            raise RuntimeError("scientific tree changed while removing control markers")
        mlflow = restored / "mlflow"
        export = recovery_transport.create_export(
            output, output_archive, mlflow=mlflow if mlflow.is_dir() else None
        )
        verified = Path(temporary) / "verified"
        verification = recovery_transport.restore_export(
            output_archive,
            verified,
            expected_sha256=export["archive_sha256"],
        )
        verified_manifest = file_manifest(verified / "experiment" / RUN_ID)
        if verified_manifest != retained:
            raise RuntimeError("prepared recovery export changed retained files")
    receipt = {
        "schema_version": "exp701-control-failure-rollback-v1",
        "status": "verified",
        "input_archive": str(source_archive.resolve()),
        "input_archive_sha256": expected_sha256,
        "archive": str(output_archive.resolve()),
        "archive_sha256": export["archive_sha256"],
        "archive_bytes": export["archive_bytes"],
        "output_archive": str(output_archive.resolve()),
        "output_archive_sha256": export["archive_sha256"],
        "output_archive_bytes": export["archive_bytes"],
        "validated_failure_boundary": boundary,
        "removed_control_markers": removed,
        "retained_files": len(retained),
        "retained_bytes": sum(item["bytes"] for item in retained.values()),
        "retained_tree_exact_after_restore": True,
        "restore_verification": verification,
        "scientific_state_changed": False,
        "justification": (
            "The failed arm had no snapshot, checkpoint, result, or update. Only the "
            "failed provider-lifecycle terminal and its pre-compute started marker were "
            "removed; every retained file is byte-identical."
        ),
    }
    recovery_transport.atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.source_archive,
        args.output_archive,
        args.receipt,
        expected_sha256=args.source_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
