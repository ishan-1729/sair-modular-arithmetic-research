"""Build the prediction-equivalent tensor-native runtime form of Candidate006."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools import build_candidate006_tier7_artifact as base


LEGACY = (
    "        state = [0] * len(multiplier)\n"
    "        for bit_index in range(TIER7_WIDTH - 1, -1, -1):\n"
    "            bit = [(value >> bit_index) & 1 for value in multiplier]\n"
    "            logits, _ = self.tier7_network(\n"
    "                _bigint_features(\n"
    "                    state, multiplicand, modulus, bit, TIER7_WIDTH\n"
    "                )\n"
    "            )\n"
    "            state = _decode_bigint_logits(logits)\n"
    "        return state\n"
)
TENSOR_NATIVE = (
    "        batch = len(multiplier)\n"
    "        multiplier_bits = _bigint_bits(multiplier, TIER7_WIDTH)\n"
    "        invariant = torch.empty((batch, 6, TIER7_WIDTH), dtype=torch.float32)\n"
    "        invariant[:, 1, :] = _bigint_bits(multiplicand, TIER7_WIDTH).mul(2).sub(1)\n"
    "        invariant[:, 2, :] = _bigint_bits(modulus, TIER7_WIDTH).mul(2).sub(1)\n"
    "        invariant[:, 4, :].fill_(-1)\n"
    "        invariant[:, 4, 0] = 1\n"
    "        invariant[:, 5, :].fill_(-1)\n"
    "        invariant[:, 5, -1] = 1\n"
    "        state_bits = torch.zeros((batch, TIER7_WIDTH), dtype=torch.bool)\n"
    "        for position in range(TIER7_WIDTH):\n"
    "            invariant[:, 0, :] = state_bits.float().mul(2).sub(1)\n"
    "            invariant[:, 3, :] = multiplier_bits[\n"
    "                :, TIER7_WIDTH - 1 - position, None\n"
    "            ].mul(2).sub(1)\n"
    "            logits, _ = self.tier7_network(invariant)\n"
    "            state_bits = logits.ge(0)\n"
    "        return _decode_bigint_logits(state_bits.float().mul(2).sub(1))\n"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path) -> dict[str, object]:
    receipt = base.build(output)
    model_path = output.resolve() / "model.py"
    text = model_path.read_text(encoding="utf-8")
    if text.count(LEGACY) != 1:
        raise RuntimeError("EXP704 legacy rollout anchor is not unique")
    model_path.write_text(text.replace(LEGACY, TENSOR_NATIVE), encoding="utf-8")
    receipt["model_sha256"] = sha256(model_path)
    receipt["runtime_optimization"] = "cached-invariant-tensor-native-tier7-rollout"
    receipt["weights_unchanged"] = True
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
