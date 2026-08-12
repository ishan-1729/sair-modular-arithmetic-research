"""Build Candidate006 by adding EXP701's frozen Tier-7 specialist to Candidate005."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
INCUMBENT = (
    ROOT
    / "experiments/runs/candidate-005-tier6-conditional-commutative-integration/artifact"
)
TIER7_SOURCE = (
    ROOT
    / "experiments/runs/exp-704-candidate005-tier7-exp701-integration/source/teacher-cosine.safetensors"
)
INCUMBENT_HASHES = {
    "model.py": "e3d4aa9ff3e22dde516d261b099826622fa37d0688fae47c07ff95fe876e9477",
    "manifest.json": "ee49f91b8a1a4aa2cde418ebec59ea1890d93fda6dc72338989c0aa3308e7c83",
    "weights.safetensors": "2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca",
}
TIER7_SHA256 = "b2ef6691ce98195a857401692cbf94335ebeebd22d0e4077a2ee30d6669f715e"
REQUIRED_FILES = frozenset(INCUMBENT_HASHES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one model template anchor, found {count}: {old!r}")
    return text.replace(old, new)


def candidate_model(incumbent: str) -> str:
    text = replace_once(
        incumbent,
        '"""Submission adapter for the learned Horner/cellular ensemble."""',
        '"""Submission adapter with Candidate005 through Tier 6 and EXP701 at Tier 7."""',
    )
    text = replace_once(
        text,
        "TIER6_WIDTH = 128\nTIER6_ROUTE_FRACTION = 0.40",
        "TIER6_WIDTH = 128\nTIER7_WIDTH = 256\nTIER6_ROUTE_FRACTION = 0.40",
    )
    text = replace_once(
        text,
        "        self.tier6_network: _SharedDilatedConvGRU32 | None = None\n",
        "        self.tier6_network: _SharedDilatedConvGRU32 | None = None\n"
        "        self.tier7_network: _SharedDilatedConvGRU32 | None = None\n",
    )
    text = replace_once(
        text,
        "        self.tier6_network = tier6_network\n",
        "        self.tier6_network = tier6_network\n"
        "        tier7_network = _SharedDilatedConvGRU32()\n"
        "        tier7_network.load_state_dict(\n"
        "            {\n"
        "                key.removeprefix(\"tier7.\"): value\n"
        "                for key, value in weights.items()\n"
        "                if key.startswith(\"tier7.\")\n"
        "            }\n"
        "        )\n"
        "        tier7_network.eval()\n"
        "        self.tier7_network = tier7_network\n",
    )
    text = replace_once(
        text,
        "    @torch.inference_mode()\n"
        "    def predict_digits_batch(\n",
        "    def _tier7_rollout(\n"
        "        self,\n"
        "        multiplier: list[int],\n"
        "        multiplicand: list[int],\n"
        "        modulus: list[int],\n"
        "    ) -> list[int]:\n"
        "        if self.tier7_network is None:\n"
        "            raise RuntimeError(\"model has not been loaded\")\n"
        "        state = [0] * len(multiplier)\n"
        "        for bit_index in range(TIER7_WIDTH - 1, -1, -1):\n"
        "            bit = [(value >> bit_index) & 1 for value in multiplier]\n"
        "            logits, _ = self.tier7_network(\n"
        "                _bigint_features(\n"
        "                    state, multiplicand, modulus, bit, TIER7_WIDTH\n"
        "                )\n"
        "            )\n"
        "            state = _decode_bigint_logits(logits)\n"
        "        return state\n\n"
        "    @torch.inference_mode()\n"
        "    def predict_digits_batch(\n",
    )
    text = replace_once(
        text,
        "            or self.tier6_network is None\n",
        "            or self.tier6_network is None\n"
        "            or self.tier7_network is None\n",
    )
    text = replace_once(
        text,
        "        tier6_group: list[tuple[int, int, int, int]] = []\n",
        "        tier6_group: list[tuple[int, int, int, int]] = []\n"
        "        tier7_group: list[tuple[int, int, int, int]] = []\n",
    )
    text = replace_once(
        text,
        "            if modulus_value > (1 << TIER6_WIDTH) - 1:\n",
        "            if modulus_value > (1 << TIER7_WIDTH) - 1:\n",
    )
    text = replace_once(
        text,
        "            else:\n"
        "                tier6_group.append(item)\n\n"
        "        weights = 1 << torch.arange(WIDTH, dtype=torch.int64)\n",
        "            elif modulus_value <= (1 << TIER6_WIDTH) - 1:\n"
        "                tier6_group.append(item)\n"
        "            else:\n"
        "                tier7_group.append(item)\n\n"
        "        weights = 1 << torch.arange(WIDTH, dtype=torch.int64)\n",
    )
    text = replace_once(
        text,
        "            for item, predicted in zip(tier6_group, selected, strict=True):\n"
        "                outputs[item[0]] = [predicted if predicted < item[3] else 0]\n"
        "        return outputs\n",
        "            for item, predicted in zip(tier6_group, selected, strict=True):\n"
        "                outputs[item[0]] = [predicted if predicted < item[3] else 0]\n\n"
        "        if tier7_group:\n"
        "            modulus = [item[3] for item in tier7_group]\n"
        "            multiplier = [item[1] for item in tier7_group]\n"
        "            multiplicand = [item[2] for item in tier7_group]\n"
        "            predicted = self._tier7_rollout(multiplier, multiplicand, modulus)\n"
        "            for item, value in zip(tier7_group, predicted, strict=True):\n"
        "                outputs[item[0]] = [value if value < item[3] else 0]\n"
        "        return outputs\n",
    )
    return text


def candidate_manifest(incumbent: str) -> str:
    manifest = json.loads(incumbent)
    manifest["model_description"] = (
        "Candidate005's frozen learned ensemble handles Tiers 1-6 unchanged. "
        "Moduli from 129 through 256 bits route to the frozen EXP701 teacher-cosine "
        "H92 two-cycle width-independent cellular specialist. Inputs above the learned "
        "256-bit boundary retain the deterministic zero fallback."
    )
    manifest["training_description"] = (
        "Candidate005 is byte-frozen through Tier 6. The added Tier-7 checkpoint is "
        "EXP701 teacher_cosine from formal root 20261414 (4,800 updates), selected by "
        "the preregistered four-arm full-denominator factorial completed by EXP703. "
        "Candidate006 integration performs no additional training or answer-based tuning."
    )
    return json.dumps(manifest, indent=2) + "\n"


def randomized_tensor(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if not tensor.is_floating_point() or tensor.numel() < 2:
        return tensor.clone().contiguous()
    source = tensor.float()
    deviation = source.std(unbiased=False)
    if float(deviation) == 0.0:
        return tensor.clone().contiguous()
    return (
        torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
        .mul_(deviation)
        .add_(source.mean())
        .to(dtype=tensor.dtype)
        .contiguous()
    )


def verify_sources() -> None:
    actual = {path.name for path in INCUMBENT.iterdir() if path.is_file()}
    if actual != REQUIRED_FILES:
        raise RuntimeError(f"incumbent is not the exact three-file artifact: {actual}")
    for name, expected in INCUMBENT_HASHES.items():
        if sha256(INCUMBENT / name) != expected:
            raise RuntimeError(f"incumbent {name} hash mismatch")
    if not TIER7_SOURCE.is_file() or sha256(TIER7_SOURCE) != TIER7_SHA256:
        raise RuntimeError("Tier-7 source checkpoint missing or hash-mismatched")


def build(output: Path, randomize_tier7_seed: int | None = None) -> dict[str, object]:
    verify_sources()
    output = output.resolve()
    if output == INCUMBENT.resolve():
        raise RuntimeError("refusing to overwrite the incumbent submission")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output must be empty before build: {output}")

    (output / "model.py").write_text(
        candidate_model((INCUMBENT / "model.py").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        candidate_manifest((INCUMBENT / "manifest.json").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    tensors = dict(load_file(str(INCUMBENT / "weights.safetensors"), device="cpu"))
    specialist = load_file(str(TIER7_SOURCE), device="cpu")
    generator = None
    if randomize_tier7_seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(randomize_tier7_seed)
    for name, tensor in specialist.items():
        value = (
            randomized_tensor(tensor, generator)
            if generator is not None
            else tensor.clone().contiguous()
        )
        tensors[f"tier7.{name}"] = value
    save_file(tensors, str(output / "weights.safetensors"))
    actual = {path.name for path in output.iterdir() if path.is_file()}
    if actual != REQUIRED_FILES:
        raise RuntimeError(f"built artifact is not clean: {actual}")
    return {
        "directory": str(output),
        "randomized_tier7_seed": randomize_tier7_seed,
        "model_sha256": sha256(output / "model.py"),
        "manifest_sha256": sha256(output / "manifest.json"),
        "weights_sha256": sha256(output / "weights.safetensors"),
        "parameters": sum(tensor.numel() for tensor in tensors.values()),
        "bytes": sum(path.stat().st_size for path in output.iterdir()),
        "source_candidate005_weights_sha256": INCUMBENT_HASHES["weights.safetensors"],
        "source_tier7_weights_sha256": TIER7_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--randomize-tier7-seed", type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.randomize_tier7_seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
