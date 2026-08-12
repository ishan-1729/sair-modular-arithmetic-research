"""Build Candidate006 with an inference-only exact-score runtime rewrite."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/runs/exp-704-candidate005-tier7-exp701-integration/candidate006"
OPTIMIZED_MODEL = (
    ROOT
    / "experiments/exp_706_candidate006_exact_runtime_fusion/optimized_model.py"
)
SOURCE_HASHES = {
    "model.py": "b7c255af7b619e6f689a84e5307ce445432e4285af0cdd9f514d982ebc8c0315",
    "manifest.json": "dbeb95ed3b085aa2b338ec57041d343ec924ca72f4b3c7fd3094d6cd215a0c91",
    "weights.safetensors": "a7d25ee4eed49c5f625bc8369447be9f51ff1be54766631d8b96ab725860553e",
}
OPTIMIZED_MODEL_SHA256 = "082ca41d86c758254dd0e75841c1b1d8df94bbf3280d539389782deaf4857d9d"
REQUIRED_FILES = frozenset(SOURCE_HASHES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def verify_sources() -> None:
    actual = {path.name for path in SOURCE.iterdir() if path.is_file()}
    if actual != REQUIRED_FILES:
        raise RuntimeError(f"Candidate006 source shape changed: {sorted(actual)}")
    for name, expected in SOURCE_HASHES.items():
        if sha256(SOURCE / name) != expected:
            raise RuntimeError(f"Candidate006 source hash mismatch: {name}")
    if sha256(OPTIMIZED_MODEL) != OPTIMIZED_MODEL_SHA256:
        raise RuntimeError("optimized model source hash mismatch")


def build(output: Path) -> dict[str, object]:
    verify_sources()
    output = output.resolve()
    if output in {SOURCE.resolve(), OPTIMIZED_MODEL.parent.resolve()}:
        raise RuntimeError("refusing to overwrite a source directory")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    shutil.copyfile(OPTIMIZED_MODEL, output / "model.py")
    shutil.copyfile(SOURCE / "manifest.json", output / "manifest.json")
    shutil.copyfile(SOURCE / "weights.safetensors", output / "weights.safetensors")
    hashes = {name: sha256(output / name) for name in sorted(REQUIRED_FILES)}
    return {
        "schema_version": "exp706-build-v1",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "output": display_path(output),
        "files": hashes,
        "weights_unchanged": hashes["weights.safetensors"] == SOURCE_HASHES["weights.safetensors"],
        "manifest_unchanged": hashes["manifest.json"] == SOURCE_HASHES["manifest.json"],
        "training_executed": False,
        "optimization": [
            "cached-invariant-feature-planes",
            "tensor-native-state-carry",
            "channels-last-conv2d-equivalent-kernel",
            "four-independent-case-shards",
            "unused-quotient-head-elision",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
