"""Build the scratch Candidate 005 artifact from two frozen checkpoints.

The builder intentionally derives Candidate 005 from the exact Candidate 004
three-file artifact.  It adds one hash-verified Tier-6 cellular checkpoint and
the frozen 40% conditional commutative route without mutating the incumbent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
INCUMBENT = ROOT / "submission"
TIER6_SOURCE = (
    ROOT
    / ".scratch"
    / "exp-609-tier6-straight-through-confirmation"
    / "straight_through_continuation"
    / "weights.safetensors"
)
INCUMBENT_HASHES = {
    "model.py": "14b55ccb6443918d3b11000f8cc90e44ff249253bb72e0fd40ddfdbf0e0a70ea",
    "manifest.json": "fa293827cfe815a9325800b91c9ff8eac41257169af1efac0fcb73a2e4e16dcd",
    "weights.safetensors": "fb938e208e96945b4d192b9bbddf976c8c04a077b6f39df9e9196d36e76fefa4",
}
TIER6_SHA256 = "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
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
        '"""Submission adapter for the promoted learned Horner/cellular ensemble."""',
        '"""Submission adapter for the learned Horner/cellular ensemble."""',
    )
    text = replace_once(
        text,
        "TIER5_WIDTH = 64\nHIDDEN = 48",
        "TIER5_WIDTH = 64\nTIER6_WIDTH = 128\nTIER6_ROUTE_FRACTION = 0.40\nHIDDEN = 48",
    )
    text = replace_once(
        text,
        '    """Width-independent cellular transition specialized through width 32."""',
        '    """Width-independent cellular transition used by the wider specialists."""',
    )
    text = replace_once(
        text,
        "def _bigint_features(\n"
        "    state: list[int],\n"
        "    multiplicand: list[int],\n"
        "    modulus: list[int],\n"
        "    bit: list[int],\n"
        "    width: int,\n"
        ") -> Tensor:\n"
        "    \"\"\"Encode Python big integers without routing them through signed int64.\"\"\"\n"
        "    batch = len(state)\n"
        "    planes = torch.empty((batch, 6, width), dtype=torch.float32)\n"
        "    for row, (state_value, multiplicand_value, modulus_value, input_bit) in enumerate(\n"
        "        zip(state, multiplicand, modulus, bit, strict=True)\n"
        "    ):\n"
        "        for position in range(width):\n"
        "            planes[row, 0, position] = (state_value >> position) & 1\n"
        "            planes[row, 1, position] = (multiplicand_value >> position) & 1\n"
        "            planes[row, 2, position] = (modulus_value >> position) & 1\n"
        "        planes[row, 3, :] = input_bit\n"
        "        planes[row, 4, :] = 0\n"
        "        planes[row, 5, :] = 0\n"
        "        planes[row, 4, 0] = 1\n"
        "        planes[row, 5, -1] = 1\n"
        "    return planes.mul_(2.0).sub_(1.0)\n",
        "def _bigint_bits(values: list[int], width: int) -> Tensor:\n"
        "    return torch.tensor(\n"
        "        [\n"
        "            [(value >> position) & 1 for position in range(width)]\n"
        "            for value in values\n"
        "        ],\n"
        "        dtype=torch.float32,\n"
        "    )\n\n\n"
        "def _bigint_features(\n"
        "    state: list[int],\n"
        "    multiplicand: list[int],\n"
        "    modulus: list[int],\n"
        "    bit: list[int],\n"
        "    width: int,\n"
        ") -> Tensor:\n"
        "    \"\"\"Encode Python big integers without signed-int64 conversion.\"\"\"\n"
        "    batch = len(state)\n"
        "    bit_plane = torch.tensor(bit, dtype=torch.float32)[:, None].expand(\n"
        "        batch, width\n"
        "    )\n"
        "    lsb = torch.zeros((batch, width), dtype=torch.float32)\n"
        "    msb = torch.zeros_like(lsb)\n"
        "    lsb[:, 0] = 1.0\n"
        "    msb[:, -1] = 1.0\n"
        "    return torch.stack(\n"
        "        (\n"
        "            _bigint_bits(state, width),\n"
        "            _bigint_bits(multiplicand, width),\n"
        "            _bigint_bits(modulus, width),\n"
        "            bit_plane,\n"
        "            lsb,\n"
        "            msb,\n"
        "        ),\n"
        "        dim=1,\n"
        "    ).mul_(2.0).sub_(1.0)\n",
    )
    text = replace_once(
        text,
        "        self.tier5_network: _SharedDilatedConvGRU32 | None = None\n",
        "        self.tier5_network: _SharedDilatedConvGRU32 | None = None\n"
        "        self.tier6_network: _SharedDilatedConvGRU32 | None = None\n",
    )
    text = replace_once(
        text,
        "        self.tier5_network = tier5_network\n",
        "        self.tier5_network = tier5_network\n"
        "        tier6_network = _SharedDilatedConvGRU32()\n"
        "        tier6_network.load_state_dict(\n"
        "            {\n"
        "                key.removeprefix(\"tier6.\"): value\n"
        "                for key, value in weights.items()\n"
        "                if key.startswith(\"tier6.\")\n"
        "            }\n"
        "        )\n"
        "        tier6_network.eval()\n"
        "        self.tier6_network = tier6_network\n",
    )
    text = replace_once(
        text,
        "    @torch.inference_mode()\n"
        "    def predict_digits_batch(\n",
        "    def _tier6_rollout(\n"
        "        self,\n"
        "        multiplier: list[int],\n"
        "        multiplicand: list[int],\n"
        "        modulus: list[int],\n"
        "    ) -> tuple[list[int], list[float]]:\n"
        "        if self.tier6_network is None:\n"
        "            raise RuntimeError(\"model has not been loaded\")\n"
        "        state = [0] * len(multiplier)\n"
        "        minimum_margin = torch.full((len(multiplier),), math.inf)\n"
        "        for bit_index in range(TIER6_WIDTH - 1, -1, -1):\n"
        "            bit = [(value >> bit_index) & 1 for value in multiplier]\n"
        "            logits, _ = self.tier6_network(\n"
        "                _bigint_features(\n"
        "                    state, multiplicand, modulus, bit, TIER6_WIDTH\n"
        "                )\n"
        "            )\n"
        "            minimum_margin = torch.minimum(\n"
        "                minimum_margin, logits.abs().amin(dim=1)\n"
        "            )\n"
        "            state = _decode_bigint_logits(logits)\n"
        "        return state, minimum_margin.tolist()\n\n"
        "    @torch.inference_mode()\n"
        "    def predict_digits_batch(\n",
    )
    text = replace_once(
        text,
        "            or self.tier5_network is None\n",
        "            or self.tier5_network is None\n"
        "            or self.tier6_network is None\n",
    )
    text = replace_once(
        text,
        "        tier5_group: list[tuple[int, int, int, int]] = []\n",
        "        tier5_group: list[tuple[int, int, int, int]] = []\n"
        "        tier6_group: list[tuple[int, int, int, int]] = []\n",
    )
    text = replace_once(
        text,
        "            if modulus_value > (1 << TIER5_WIDTH) - 1:\n",
        "            if modulus_value > (1 << TIER6_WIDTH) - 1:\n",
    )
    text = replace_once(
        text,
        "            else:\n"
        "                tier5_group.append(item)\n\n"
        "        weights = 1 << torch.arange(WIDTH, dtype=torch.int64)\n",
        "            elif modulus_value <= (1 << TIER5_WIDTH) - 1:\n"
        "                tier5_group.append(item)\n"
        "            else:\n"
        "                tier6_group.append(item)\n\n"
        "        weights = 1 << torch.arange(WIDTH, dtype=torch.int64)\n",
    )
    text = replace_once(
        text,
        "            for item, predicted in zip(tier5_group, state, strict=True):\n"
        "                outputs[item[0]] = [predicted if predicted < item[3] else 0]\n"
        "        return outputs\n",
        "            for item, predicted in zip(tier5_group, state, strict=True):\n"
        "                outputs[item[0]] = [predicted if predicted < item[3] else 0]\n\n"
        "        if tier6_group:\n"
        "            multiplier = [item[1] for item in tier6_group]\n"
        "            multiplicand = [item[2] for item in tier6_group]\n"
        "            modulus = [item[3] for item in tier6_group]\n"
        "            selected, original_margin = self._tier6_rollout(\n"
        "                multiplier, multiplicand, modulus\n"
        "            )\n"
        "            route_count = round(len(tier6_group) * TIER6_ROUTE_FRACTION)\n"
        "            routed_indices = sorted(\n"
        "                range(len(tier6_group)),\n"
        "                key=lambda index: (original_margin[index], index),\n"
        "            )[:route_count]\n"
        "            if routed_indices:\n"
        "                swapped, swapped_margin = self._tier6_rollout(\n"
        "                    [multiplicand[index] for index in routed_indices],\n"
        "                    [multiplier[index] for index in routed_indices],\n"
        "                    [modulus[index] for index in routed_indices],\n"
        "                )\n"
        "                for local, index in enumerate(routed_indices):\n"
        "                    if swapped_margin[local] > original_margin[index]:\n"
        "                        selected[index] = swapped[local]\n"
        "            for item, predicted in zip(tier6_group, selected, strict=True):\n"
        "                outputs[item[0]] = [predicted if predicted < item[3] else 0]\n"
        "        return outputs\n",
    )
    return text


def candidate_manifest(incumbent: str) -> str:
    manifest = json.loads(incumbent)
    manifest["model_description"] = (
        "Learned Horner ensemble with width-9 and width-16 TCN specialists for "
        "Tiers 2-3 and separate 99,824-parameter width-independent cellular "
        "specialists through Tiers 4-6. Tier 6 executes the original orientation "
        "for every row, conditionally executes the commuted orientation for the "
        "lowest-margin 40%, and selects the higher learned-logit-margin result. "
        "Inputs above the learned 128-bit boundary receive a deterministic zero "
        "fallback."
    )
    manifest["training_description"] = (
        "Candidate 004's frozen Tier-2 through Tier-5 specialists are unchanged. "
        "The Tier-6 cellular specialist is the frozen EXP-609 straight-through "
        "continuation endpoint; the target-free 40% commutative route was frozen "
        "by EXP-641 and confirmed at 486/500 exact on fresh width-128 cases in "
        "EXP-642. Checked answers were not used for Candidate 005 integration or "
        "selection."
    )
    return json.dumps(manifest, indent=2) + "\n"


def randomized_tensor(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if not tensor.is_floating_point() or tensor.numel() < 2:
        return tensor.clone().contiguous()
    source = tensor.float()
    deviation = source.std(unbiased=False)
    if float(deviation) == 0.0:
        return tensor.clone().contiguous()
    randomized = torch.randn(
        tensor.shape, generator=generator, dtype=torch.float32
    ).mul_(deviation).add_(source.mean())
    return randomized.to(dtype=tensor.dtype).contiguous()


def verify_sources() -> None:
    actual = {path.name for path in INCUMBENT.iterdir() if path.is_file()}
    if actual != REQUIRED_FILES:
        raise RuntimeError(f"incumbent is not the exact three-file artifact: {actual}")
    for name, expected in INCUMBENT_HASHES.items():
        if sha256(INCUMBENT / name) != expected:
            raise RuntimeError(f"incumbent {name} hash mismatch")
    if not TIER6_SOURCE.is_file() or sha256(TIER6_SOURCE) != TIER6_SHA256:
        raise RuntimeError("Tier-6 source checkpoint missing or hash-mismatched")


def build(output: Path, randomize_tier6_seed: int | None) -> dict[str, object]:
    verify_sources()
    output = output.resolve()
    if output == INCUMBENT.resolve():
        raise RuntimeError("refusing to overwrite the incumbent submission")
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir():
        if child.is_dir() or child.name not in REQUIRED_FILES:
            raise RuntimeError(f"unexpected pre-existing artifact path: {child}")

    model_text = candidate_model((INCUMBENT / "model.py").read_text(encoding="utf-8"))
    manifest_text = candidate_manifest(
        (INCUMBENT / "manifest.json").read_text(encoding="utf-8")
    )
    (output / "model.py").write_text(model_text, encoding="utf-8")
    (output / "manifest.json").write_text(manifest_text, encoding="utf-8")

    tensors = dict(load_file(str(INCUMBENT / "weights.safetensors"), device="cpu"))
    tier6 = load_file(str(TIER6_SOURCE), device="cpu")
    generator = None
    if randomize_tier6_seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(randomize_tier6_seed)
    for name, tensor in tier6.items():
        value = (
            randomized_tensor(tensor, generator)
            if generator is not None
            else tensor.clone().contiguous()
        )
        tensors[f"tier6.{name}"] = value
    # Safetensors metadata uses a map whose serialized key order is not stable
    # across processes in the pinned build.  Keep provenance in the resolved
    # config and omit metadata so repeated builds are byte-for-byte identical.
    save_file(tensors, str(output / "weights.safetensors"))
    actual = {path.name for path in output.iterdir() if path.is_file()}
    if actual != REQUIRED_FILES:
        raise RuntimeError(f"built artifact is not clean: {actual}")
    result: dict[str, object] = {
        "directory": str(output),
        "randomized_tier6_seed": randomize_tier6_seed,
        "model_sha256": sha256(output / "model.py"),
        "manifest_sha256": sha256(output / "manifest.json"),
        "weights_sha256": sha256(output / "weights.safetensors"),
        "parameters": sum(tensor.numel() for tensor in tensors.values()),
        "bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
        "source_candidate004_weights_sha256": INCUMBENT_HASHES["weights.safetensors"],
        "source_tier6_weights_sha256": TIER6_SHA256,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--randomize-tier6-seed", type=int)
    args = parser.parse_args()
    result = build(args.output, args.randomize_tier6_seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
