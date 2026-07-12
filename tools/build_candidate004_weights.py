"""Compose Candidate 004 from two frozen, hash-verified safetensors artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = "a991c389aec5f8c4e8153846b07d7d1207bdd0eaf192ff11ca036663f056a514"
TIER5_SHA256 = "050daf0a2d3b964ed9a58522c18d896f94a0d950722d0c34296b0718c03676f7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("tier5", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    tier5 = args.tier5.resolve()
    output = args.output.resolve()
    if sha256(base) != BASE_SHA256:
        raise RuntimeError("Candidate 003 source hash mismatch")
    if sha256(tier5) != TIER5_SHA256:
        raise RuntimeError("Tier-5 source hash mismatch")
    tensors = dict(load_file(str(base), device="cpu"))
    for name, tensor in load_file(str(tier5), device="cpu").items():
        tensors[f"tier5.{name}"] = tensor.contiguous()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            "candidate": "004",
            "base_sha256": BASE_SHA256,
            "tier5_sha256": TIER5_SHA256,
        },
    )
    print(
        f"wrote {output.relative_to(ROOT)} parameters="
        f"{sum(tensor.numel() for tensor in tensors.values())} sha256={sha256(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
