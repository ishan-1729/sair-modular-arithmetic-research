"""Create a deterministic randomized-weight control outside the submission.

The output preserves tensor names, shapes, dtypes, and per-tensor scale while
replacing every floating-point learned value. It is an evaluation control, not
a candidate-building utility.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def randomize_weights(source: Path, output: Path, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    randomized: dict[str, torch.Tensor] = {}
    for name, tensor in load_file(str(source), device="cpu").items():
        if not tensor.is_floating_point():
            randomized[name] = tensor
            continue
        float_tensor = tensor.float()
        scale = float(float_tensor.std(unbiased=False).item())
        if not scale:
            scale = 0.02
        values = torch.randn(
            tensor.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).mul_(scale)
        randomized[name] = values.to(dtype=tensor.dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        randomized,
        str(output),
        metadata={"control": "randomized-weights", "seed": str(seed)},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    randomize_weights(args.source.resolve(), args.output.resolve(), args.seed)
    print(
        f"wrote {args.output} sha256={sha256(args.output.resolve())} "
        f"seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
