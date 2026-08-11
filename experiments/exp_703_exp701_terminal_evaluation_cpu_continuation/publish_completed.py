"""Publish EXP703 from two already-completed, authenticated phase results."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from experiments.exp_702_exp701_terminal_evaluation_continuation import run as exp702
from experiments.exp_703_exp701_terminal_evaluation_cpu_continuation import run


def publish(config_path: Path, archive: Path, output: Path) -> dict:
    config, exp702_config = run.load_contract(config_path)
    with run.single_writer(output):
        verified = run.authenticate_predecessor(config, exp702_config, archive)
        preflight_path = output / "preflight.json"
        if not preflight_path.is_file():
            raise run.SemanticFailure("completed CPU phases lack their preflight")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        identity = preflight.get("identity_sha256")
        if not isinstance(identity, str) or len(identity) != 64:
            raise run.SemanticFailure("completed CPU phases lack a valid identity")
        results = {}
        for width in (128, 256):
            result_path = output / "phases" / f"evaluation__{run.ARM}__{width}.json"
            manifest_path = result_path.with_name(result_path.stem + ".manifest.json")
            if not result_path.is_file() or not manifest_path.is_file():
                raise run.SemanticFailure(f"evaluation/{run.ARM}/{width} is incomplete")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") != "completed"
                or manifest.get("identity_sha256") != identity
                or manifest.get("result_sha256") != run.sha256(result_path)
            ):
                raise run.SemanticFailure(f"evaluation/{run.ARM}/{width} publication identity mismatch")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            run.validate_evaluation(result, width, config)
            results[str(width)] = result

        publication_config = copy.deepcopy(config)
        publication_config["predecessor"]["relationship"] = exp702_config["predecessor"]["relationship"]
        predecessor_source = {
            "run_id": config["predecessor"]["run_id"],
            "status": "terminal-incomplete",
            "archive_sha256": config["predecessor"]["archive_sha256"],
            "identity_sha256": verified["opened"]["identity_sha256"],
            "source_commit": exp702_config["predecessor"]["source_commit"],
            "recovery_count": verified["lifecycle"]["recoveries"],
            "terminal_reason": verified["terminal"]["reason"],
        }
        terminal = exp702.integrate_result(
            publication_config, verified, results, predecessor_source
        )
        terminal.update(
            {
                "schema_version": "exp703-terminal-v1",
                "run_id": run.RUN_ID,
                "relationship": "cpu-terminal-evaluation-successor-not-exp701-or-exp702-completion",
                "source": {**preflight["source"], "identity_sha256": identity},
                "recovered_completed_phases": [
                    f"evaluation/{run.ARM}/128",
                    f"evaluation/{run.ARM}/256",
                ],
                "resource": {
                    "phase_resources": {
                        width: results[width].get("resource") for width in ("128", "256")
                    },
                    "paired_evidence_resources": {
                        width: results[width].get("paired_evidence_resource")
                        for width in ("128", "256")
                    },
                },
                "limitations": config["limitations"],
            }
        )
        terminal_path = output / "terminal-summary.json"
        run.atomic_json(terminal_path, terminal)
        publication_receipt = {
            "schema_version": "exp703-publication-v1",
            "run_id": run.RUN_ID,
            "status": "completed",
            "identity_sha256": identity,
            "terminal_summary": str(terminal_path),
            "terminal_summary_sha256": run.sha256(terminal_path),
            "phase_result_sha256": {
                width: run.sha256(
                    output / "phases" / f"evaluation__{run.ARM}__{width}.json"
                )
                for width in ("128", "256")
            },
            "published_at": run.utc_now(),
        }
        run.atomic_json(output / "publication-receipt.json", publication_receipt)
        run.atomic_json(
            output / "status.json",
            {
                "schema_version": "exp703-cpu-status-v1",
                "run_id": run.RUN_ID,
                "status": "completed",
                "phase": "terminal-publication",
                "identity_sha256": identity,
                "terminal_summary": str(terminal_path),
                "terminal_summary_sha256": publication_receipt["terminal_summary_sha256"],
                "completed_at": publication_receipt["published_at"],
            },
        )
        return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=run.CONFIG_PATH)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=run.OUTPUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    archive = args.archive or run.ROOT / config["predecessor"]["archive"]
    result = publish(args.config, archive, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
