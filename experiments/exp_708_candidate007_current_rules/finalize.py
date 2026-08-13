"""Canary and finalize the measured Candidate007 submission artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from modchallenge.security.static_check import check_submission


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/runs/exp-708-candidate007-current-rules"
ARTIFACT = RUN / "artifact"
RESULT = RUN / "checked-public-current-rules.json"
SANDBOX_RESULT = RUN / "official-sandbox/result.json"
SANDBOX_RUN = RUN / "official-sandbox/sandbox-run.json"
EXPECTED = {
    "manifest.json": "95707215a920ab2cab09fd16ad1347d5b8055626776999af71e54cc2867364d8",
    "model.py": "bf4bdbdda90ef148ebc9e0b4db3c5c73d7b73e2156c868ce0c05b5c1471d8833",
    "weights.safetensors": "a7d25ee4eed49c5f625bc8369447be9f51ff1be54766631d8b96ab725860553e",
}
EXPECTED_RESULT = "3724f8b21b4ac2e2f22ab517273c4fabe09f8daac59cfefdabf724f9482c0169"
EXPECTED_SANDBOX_RESULT = "94902f92549327a0df1e5031915decf201195ee746719e7198e9cccc1176806f"
EXPECTED_SANDBOX_RUN = "1eb2fa04b5a84b6f8624758b007d0092c4325e62add4f63c59d2e824f601ab89"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model():
    spec = importlib.util.spec_from_file_location("candidate007", ARTIFACT / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Candidate007 model cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Tier2HornerModel()
    model.load(str(ARTIFACT))
    return model


def verify() -> dict:
    actual = {name: sha256(ARTIFACT / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(f"Candidate007 artifact hash mismatch: {actual}")
    findings = check_submission(ARTIFACT)
    if findings:
        raise RuntimeError("; ".join(finding.format() for finding in findings))
    result = json.loads(RESULT.read_text())
    if sha256(RESULT) != EXPECTED_RESULT:
        raise RuntimeError("Candidate007 checked result hash mismatch")
    checks = result["checks"]
    summary = result["result"]
    timing = result["timing_seconds"]
    if checks != {
        "static_analysis": "clean",
        "preprocess_isolated": True,
        "deterministic": True,
    }:
        raise RuntimeError(f"Candidate007 check gate failed: {checks}")
    if timing["inference"] > 300:
        raise RuntimeError("Candidate007 exceeded 300 inference seconds")
    if summary["overall_accuracy"] < 0.48:
        raise RuntimeError("Candidate007 missed the accuracy floor")
    if summary["highest_tier_above_90"] < 5:
        raise RuntimeError("Candidate007 missed the retained-tier gate")
    if not all(tier["completed"] for tier in summary["tiers"]):
        raise RuntimeError("Candidate007 left a tier incomplete")
    if sha256(SANDBOX_RESULT) != EXPECTED_SANDBOX_RESULT:
        raise RuntimeError("Candidate007 sandbox result hash mismatch")
    if sha256(SANDBOX_RUN) != EXPECTED_SANDBOX_RUN:
        raise RuntimeError("Candidate007 sandbox metadata hash mismatch")
    sandbox_result = json.loads(SANDBOX_RESULT.read_text())
    sandbox_run = json.loads(SANDBOX_RUN.read_text())
    if sandbox_result.get("status") != "completed":
        raise RuntimeError("Candidate007 sandbox did not complete")
    if not all(tier["completed"] for tier in sandbox_result["tiers"]):
        raise RuntimeError("Candidate007 sandbox left a tier incomplete")
    if sandbox_run["container_wall_seconds"] > 300:
        raise RuntimeError("Candidate007 full sandbox wall exceeded 300 seconds")
    if sandbox_run["cpus"] != "4" or sandbox_run["memory"] != "8g":
        raise RuntimeError("Candidate007 sandbox resource identity mismatch")
    return {
        "status": "submission-valid",
        "artifact_sha256": actual,
        "result_sha256": EXPECTED_RESULT,
        "sandbox_result_sha256": EXPECTED_SANDBOX_RESULT,
        "sandbox_run_sha256": EXPECTED_SANDBOX_RUN,
        "current_rules_commit": "99cac6e",
        "checks": checks,
        "inference_seconds": timing["inference"],
        "overall_accuracy": summary["overall_accuracy"],
        "highest_tier_above_90": summary["highest_tier_above_90"],
        "tier_accuracy": {
            str(tier["tier_id"]): tier["accuracy"] for tier in summary["tiers"]
        },
        "sandbox_container_wall_seconds": sandbox_run["container_wall_seconds"],
        "sandbox_overall_accuracy": sandbox_result["overall_accuracy"],
        "sandbox_highest_tier_above_90": sandbox_result["highest_tier_above_90"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("canary", "finalize"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "canary":
        model = load_model()
        rows = [(str((1 << 31) + 123), str((1 << 30) + 77), "7")]
        prediction = model.predict_digits_batch(rows)
        payload = {
            "status": "passed",
            "prediction": prediction,
            "expected": [[((1 << 31) + 123) * ((1 << 30) + 77) % 7]],
        }
        if payload["prediction"] != payload["expected"]:
            raise RuntimeError("Candidate007 canary prediction mismatch")
        path = args.output / "canary.json"
    else:
        payload = verify()
        path = args.output / "terminal-summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
