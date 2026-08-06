"""Validate a restored EXP701 CUDA smoke before scientific roots open."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODES = ("calibration", "formal", "confirmations", "independent")
RUN_ID = "exp-701-cuda-resumable-h92-w256-backbone-factorial"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def verify_restored_smoke(output: Path) -> dict[str, Any]:
    """Fail closed unless the restored smoke is complete and internally exact."""

    failure = output / "terminal-failure.json"
    if failure.exists():
        raise RuntimeError("restored smoke contains a terminal failure")

    smoke_path = output / "smoke.json"
    smoke = read_json(smoke_path)
    if smoke.get("status") != "smoke-passed":
        raise RuntimeError("restored smoke did not pass")
    if smoke.get("all_modes_exercised") != list(MODES):
        raise RuntimeError("restored smoke mode coverage mismatch")
    if smoke.get("train_eval_diagnostic_recovery") is not True:
        raise RuntimeError("restored smoke recovery proof is absent")
    if smoke.get("uninterrupted_signature") != smoke.get(
        "interrupted_resumed_signature"
    ):
        raise RuntimeError("restored smoke signatures differ")

    terminal_hashes: dict[str, str] = {}
    for mode in MODES:
        terminal_path = output / "smoke" / f"{mode}-terminal.json"
        terminal = read_json(terminal_path)
        recovery = terminal.get("recovery") or {}
        if (
            terminal.get("status") != "smoke-passed"
            or terminal.get("run_id") != RUN_ID
            or terminal.get("mode") != mode
            or terminal.get("root") is not None
            or not str(terminal.get("device", "")).startswith("cuda")
            or set(recovery) != {
                "training_exact",
                "evaluation_reused",
                "diagnostic_reused",
                "predictions_exact",
            }
            or not all(value is True for value in recovery.values())
        ):
            raise RuntimeError(f"restored smoke terminal mismatch: {mode}")
        terminal_hashes[mode] = sha256(terminal_path)

    return {
        "schema_version": "exp701-restored-smoke-verification-v1",
        "status": "verified-restored-smoke",
        "smoke_sha256": sha256(smoke_path),
        "terminal_sha256": terminal_hashes,
        "modes": list(MODES),
        "recovery_exact": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify_restored_smoke(args.output.resolve())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
