"""Run EXP-660 using the frozen EXP-659 architecture and measurement engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.exp_659_symbol_equivariant_recurrence.run import Experiment


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_660_symbol_equivariant_formal.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-660-symbol-equivariant-formal",
    )
    parser.add_argument("--mode", choices=("calibration", "formal"), required=True)
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode)
    result = (
        experiment.calibration() if args.mode == "calibration" else experiment.formal()
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "mode": args.mode,
                "status": result["status"],
                "decision": result.get(
                    "decision", result.get("admission", {}).get("decision")
                ),
                "resource": result.get("resource", result.get("measured_total")),
                "recommended_resource_budget": result.get(
                    "recommended_resource_budget"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
