"""Build Candidate007 from frozen Candidate006 bytes plus the compliant front end."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.exp_708_candidate007_current_rules.compliant_patch import compose


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/runs/exp-706-candidate006-exact-runtime-fusion/artifact"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_model = SOURCE / "model.py"
    model_source = compose(source_model.read_text())
    (output / "model.py").write_text(model_source)
    import shutil

    shutil.copy2(SOURCE / "weights.safetensors", output / "weights.safetensors")
    manifest = {
        "entry_class": "model.Tier2HornerModel",
        "output_base": "p",
        "framework": "pytorch",
        "model_description": (
            "Candidate007 retains Candidate006's frozen learned experts through "
            "64-bit moduli. Raw operand bits are fed MSB-first on a fixed feedback-free "
            "schedule; the trained transitions learn both reductions and the "
            "product. Unsupported inputs use a constant fallback."
        ),
        "training_description": (
            "No parameters were changed. Candidate006's experts are copied "
            "byte-for-byte. A separately evaluated 25-update reducer continuation "
            "was rejected on runtime and is not included in this artifact."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "candidate": "Candidate007",
        "source_candidate": "Candidate006",
        "training_executed": False,
        "weights_unchanged": sha256(SOURCE / "weights.safetensors")
        == sha256(output / "weights.safetensors"),
        "files": {
            name: sha256(output / name)
            for name in ("manifest.json", "model.py", "weights.safetensors")
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
