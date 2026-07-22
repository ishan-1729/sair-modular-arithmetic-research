"""Checksum-bound regular-copy fixture materialization for EXP-655/T048."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"


@dataclass(frozen=True)
class Fixture:
    name: str
    source_relative: str | None
    filename: str
    sha256: str


FIXTURES = (
    Fixture(
        "exp609_checkpoint",
        None,
        "exp609-weights.safetensors",
        "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd",
    ),
    Fixture(
        "dockerfile",
        "docker/Dockerfile",
        "Dockerfile",
        "1e04249cf1b8410336f11b9509e918aaf8e9dbf5117d0a972ddd5e16d68d0b41",
    ),
    Fixture(
        "generator",
        "src/modchallenge/testgen/generator.py",
        "generator.py",
        "77f4142fe097ef66f2bd4452547b465ef3f75646191ed8258c508f5e3caa85c0",
    ),
    Fixture(
        "primes",
        "src/modchallenge/testgen/primes.py",
        "primes.py",
        "5d7f494a1560efbfeb74ed02ddd67a17001bc9ebeb6345e1567bca1bf7dc8bac",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_non_symlink(path: Path, label: str) -> os.stat_result:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is absent: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")
    return metadata


def verified_copy(source: Path, destination: Path, expected_sha256: str) -> dict[str, object]:
    source = source.resolve(strict=True)
    source_stat = _regular_non_symlink(source, "fixture source")
    if sha256(source) != expected_sha256:
        raise RuntimeError(f"fixture source SHA-256 mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination_stat = _regular_non_symlink(destination, "fixture destination")
        if sha256(destination) == expected_sha256:
            if (source_stat.st_dev, source_stat.st_ino) == (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                raise RuntimeError("fixture destination aliases source bytes")
            return {
                "path": str(destination),
                "sha256": expected_sha256,
                "bytes": destination_stat.st_size,
                "reused": True,
                "regular_copy": True,
            }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        temporary_stat = _regular_non_symlink(temporary, "temporary fixture")
        if sha256(temporary) != expected_sha256:
            raise RuntimeError("copied fixture SHA-256 mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    destination_stat = _regular_non_symlink(destination, "fixture destination")
    if sha256(destination) != expected_sha256:
        raise RuntimeError("materialized fixture SHA-256 mismatch")
    if (source_stat.st_dev, source_stat.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise RuntimeError("fixture destination aliases source bytes")
    return {
        "path": str(destination),
        "sha256": expected_sha256,
        "bytes": destination_stat.st_size,
        "reused": False,
        "regular_copy": True,
    }


def materialize(
    *,
    checkpoint_source: Path,
    upstream_root: Path,
    destination_root: Path,
    exp658_root: Path | None = None,
) -> dict[str, object]:
    upstream_root = upstream_root.resolve(strict=True)
    git_dir = upstream_root / ".git"
    if not git_dir.exists():
        raise RuntimeError("upstream root is not the immutable challenge checkout")
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if commit != CHALLENGE_COMMIT:
        raise RuntimeError("upstream challenge commit mismatch")
    outputs: dict[str, object] = {}
    for fixture in FIXTURES:
        source = (
            checkpoint_source
            if fixture.source_relative is None
            else upstream_root / fixture.source_relative
        )
        outputs[fixture.name] = verified_copy(
            source, destination_root / fixture.filename, fixture.sha256
        )
    if exp658_root is not None:
        outputs["exp658_source_checkpoint"] = verified_copy(
            checkpoint_source,
            exp658_root / "source_checkpoint.safetensors",
            FIXTURES[0].sha256,
        )
        for fixture in FIXTURES[1:]:
            outputs[f"exp658_{fixture.name}"] = verified_copy(
                upstream_root / str(fixture.source_relative),
                exp658_root / "pinned_sources" / fixture.filename,
                fixture.sha256,
            )
    return {
        "challenge_commit": commit,
        "destination_root": str(destination_root),
        "fixtures": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-source", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=ROOT / ".scratch/exp-655-layer-loop-compute-matched-recurrence/fixtures",
    )
    parser.add_argument(
        "--exp658-root",
        type=Path,
        default=ROOT / ".scratch/exp-658-joint-quotient-carry-transducer",
    )
    args = parser.parse_args()
    import json

    print(
        json.dumps(
            materialize(
                checkpoint_source=args.checkpoint_source,
                upstream_root=args.upstream_root,
                destination_root=args.destination_root,
                exp658_root=args.exp658_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
