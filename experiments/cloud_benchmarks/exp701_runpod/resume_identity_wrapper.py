"""Bind a resumed EXP701 root to its frozen compatible CUDA identity.

Physical RTX 4090 memory reporting varies slightly across compatible hosts and
driver builds. This control-plane wrapper verifies the live runtime, records it
separately, and uses the opened root's execution receipt for identity comparison.
It also corrects a recovery-control ambiguity in the frozen runner: root-level
``--resume`` means "reuse completed root work", not "every later arm already
has a snapshot".  Existing arm snapshots remain mandatory for arm recovery;
an untouched arm starts from its preregistered initialization exactly as it
would in an uninterrupted run.  No config, seed, update budget, or scientific
artifact is changed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

# The wrapper is installed outside the source archive so its bytes can be bound
# by the controller without changing scientific source.  When Python executes
# that absolute external path, it does not put the verified repository root on
# sys.path even though the guest invokes the wrapper from that directory.
repository = Path.cwd().resolve()
sys.path.insert(0, str(repository))

from experiments.exp_701_cuda_resumable_h92_w256_backbone_factorial import (
    run as exp701,
)


def option(name: str, default: str | None = None) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def compatible_runtime(
    current: dict[str, Any], frozen: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    current_normalized = copy.deepcopy(current)
    frozen_normalized = copy.deepcopy(frozen)
    current_memory = int(current_normalized["gpu"].pop("total_memory_bytes"))
    frozen_memory = int(frozen_normalized["gpu"].pop("total_memory_bytes"))
    difference_fraction = abs(current_memory - frozen_memory) / frozen_memory
    compatible = (
        current_normalized == frozen_normalized
        and current_memory >= 24_000_000_000
        and difference_fraction <= 0.01
    )
    return compatible, {
        "current_total_memory_bytes": current_memory,
        "frozen_total_memory_bytes": frozen_memory,
        "difference_fraction": difference_fraction,
    }


def arm_resume_decision(
    output: Path, arm: str, requested: bool
) -> tuple[bool, dict[str, Any]]:
    """Resolve root recovery into the only valid per-arm recovery mode."""
    snapshot, manifest = exp701.snapshot_paths(output, arm)
    final_record, final_manifest = exp701.phase_paths(output, f"train/{arm}")
    arm_directory = output / "arms" / arm
    recovery_directory = output / "recovery" / arm
    if not requested:
        return False, {
            "arm": arm,
            "requested_resume": False,
            "effective_resume": False,
            "reason": "fresh-root",
        }
    if manifest.is_file():
        if not snapshot.is_file():
            raise RuntimeError(f"resume manifest exists without snapshot for {arm}")
        return True, {
            "arm": arm,
            "requested_resume": True,
            "effective_resume": True,
            "reason": "existing-arm-snapshot",
            "snapshot_manifest": str(manifest),
            "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        }
    unexpected = [
        str(path)
        for path in (snapshot, final_record, final_manifest, arm_directory, recovery_directory)
        if path.exists()
    ]
    if unexpected:
        raise RuntimeError(
            f"unstarted resumed arm has unexpected state for {arm}: {unexpected}"
        )
    return False, {
        "arm": arm,
        "requested_resume": True,
        "effective_resume": False,
        "reason": "unstarted-arm-starts-fresh",
    }


def install_resume_identity_adapter() -> None:
    if "--resume" not in sys.argv:
        return
    mode = option("--mode")
    root = option("--root")
    output = Path(option("--output", str(exp701.OUTPUT)) or exp701.OUTPUT).resolve()
    if mode is None or root is None:
        raise RuntimeError("resume identity wrapper requires explicit mode and root")
    opened_path = output / mode / f"root-{root}" / "root-opened.json"
    opened = json.loads(opened_path.read_text(encoding="utf-8"))
    frozen = opened["source"]["execution"]
    original = exp701.runtime_receipt

    def normalized_runtime(device: Any) -> dict[str, Any]:
        current = original(device)
        compatible, memory = compatible_runtime(current, frozen)
        receipt = {
            "schema_version": "exp701-resume-runtime-compatibility-v1",
            "status": "compatible" if compatible else "rejected",
            "normalized_identity_fields": ["gpu.total_memory_bytes"],
            "current_execution": current,
            "frozen_execution": frozen,
            "memory": memory,
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scientific_state_changed": False,
        }
        exp701.atomic_json(output / "resume-runtime-compatibility.json", receipt)
        if not compatible:
            raise RuntimeError(f"resumed CUDA runtime is incompatible: {receipt}")
        return copy.deepcopy(frozen)

    exp701.runtime_receipt = normalized_runtime


def install_resume_arm_adapter(output: Path) -> None:
    """Keep root recovery from demanding snapshots for untouched later arms."""
    if "--resume" not in sys.argv:
        return
    original = exp701.train_arm_resumable
    decisions: list[dict[str, Any]] = []

    def recovery_aware_train_arm(
        config: dict[str, Any],
        exp300: Any,
        checkpoint: Path,
        splits: dict[int, Any],
        root: int,
        arm: str,
        updates: int,
        arm_output: Path,
        identity_sha256: str,
        resume: bool,
        interrupt_after_update: int | None = None,
        snapshot_every: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        effective, decision = arm_resume_decision(arm_output, arm, resume)
        decisions.append(decision)
        exp701.atomic_json(
            output / "resume-arm-selection-compatibility.json",
            {
                "schema_version": "exp701-resume-arm-selection-v1",
                "status": "compatible",
                "decisions": decisions,
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scientific_state_changed": False,
            },
        )
        return original(
            config,
            exp300,
            checkpoint,
            splits,
            root,
            arm,
            updates,
            arm_output,
            identity_sha256,
            effective,
            interrupt_after_update,
            snapshot_every,
        )

    exp701.train_arm_resumable = recovery_aware_train_arm


def main() -> int:
    install_resume_identity_adapter()
    output = Path(option("--output", str(exp701.OUTPUT)) or exp701.OUTPUT).resolve()
    install_resume_arm_adapter(output)
    return exp701.main()


if __name__ == "__main__":
    raise SystemExit(main())
