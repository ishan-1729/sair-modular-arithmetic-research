"""Authenticate EXP701 and compute only its two missing evaluations on CPU.

This successor deliberately has a new run identity.  It reuses EXP702's exact
predecessor authentication and frozen evaluation functions, but it does not
claim to satisfy EXP702's CUDA runtime preregistration.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import yaml
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.cloud_benchmarks.exp699_cuda_replay import run as cuda_replay  # noqa: E402
from experiments.cuda_portability.runtime import (  # noqa: E402
    configure_determinism,
    module_device,
    resolve_device,
    runtime_receipt,
)
from experiments.exp_686_w256_straight_through_specialist import run as base  # noqa: E402
from experiments.exp_701_cuda_resumable_h92_w256_backbone_factorial import run as exp701  # noqa: E402
from experiments.exp_702_exp701_terminal_evaluation_continuation import run as exp702  # noqa: E402

RUN_ID = "exp-703-exp701-terminal-evaluation-cpu-continuation"
CONFIG_PATH = ROOT / "configs/exp_703_exp701_terminal_evaluation_cpu_continuation.yaml"
OUTPUT = ROOT / ".scratch/exp703-cpu-overnight"
ARM = "straight_through_restarts"


class SemanticFailure(RuntimeError):
    """A provenance, execution, or completion invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return exp702.sha256(path)


def cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def load_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("run_id") != RUN_ID:
        raise SemanticFailure("CPU successor run identity mismatch")
    if config.get("evaluation", {}).get("training_allowed") is not False:
        raise SemanticFailure("CPU successor must forbid training")
    predecessor = config["predecessor"]
    exp702_config_path = ROOT / predecessor["exp702_config"]
    if sha256(exp702_config_path) != predecessor["exp702_config_sha256"]:
        raise SemanticFailure("frozen EXP702 config mismatch")
    if sha256(Path(exp702.__file__)) != predecessor["exp702_runner_sha256"]:
        raise SemanticFailure("frozen EXP702 authentication runner mismatch")
    exp702_config = yaml.safe_load(exp702_config_path.read_text(encoding="utf-8"))
    if config["evaluation"] != exp702_config["evaluation"]:
        raise SemanticFailure("CPU successor changed frozen evaluation contract")
    if config["terminal_policy"] != exp702_config["terminal_policy"]:
        raise SemanticFailure("CPU successor changed frozen terminal policy")
    if predecessor["archive_sha256"] != exp702_config["predecessor"]["archive_sha256"]:
        raise SemanticFailure("CPU successor changed predecessor archive")
    if predecessor["checkpoint_sha256"] != exp702_config["predecessor"]["members"]["final_checkpoint"]["sha256"]:
        raise SemanticFailure("CPU successor changed predecessor checkpoint")
    return config, exp702_config


def authenticate_predecessor(
    config: dict[str, Any], exp702_config: dict[str, Any], archive: Path
) -> dict[str, Any]:
    if sha256(archive) != config["predecessor"]["archive_sha256"]:
        raise SemanticFailure("CPU successor predecessor archive mismatch")
    try:
        verified = exp702.verify_predecessor(exp702_config, archive)
    except (exp702.SemanticFailure, OSError) as error:
        raise SemanticFailure(str(error)) from error
    if exp702.sha256_bytes(verified["checkpoint"]) != config["predecessor"]["checkpoint_sha256"]:
        raise SemanticFailure("authenticated checkpoint bytes changed")
    return verified


def configure_cpu(config: dict[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    expected = config["cpu_execution"]
    device = resolve_device(expected["device"], require_accelerator=False)
    if device.type != "cpu":
        raise SemanticFailure("EXP703 requires an explicit CPU device")
    if torch.__version__ != expected["torch"]:
        raise SemanticFailure(
            f"CPU torch runtime mismatch: {torch.__version__!r} != {expected['torch']!r}"
        )
    configure_determinism(
        device=device,
        seed=int(config["predecessor"]["root"]) + 698,
        torch_threads=int(expected["torch_threads"]),
        interop_threads=int(expected["interop_threads"]),
    )
    if not torch.are_deterministic_algorithms_enabled():
        raise SemanticFailure("deterministic algorithms are not enabled")
    receipt = runtime_receipt(device)
    receipt["cpu"] = {
        "model": cpu_model_name(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "platform": platform.platform(),
    }
    return device, receipt


@contextlib.contextmanager
def single_writer(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".exp703-writer.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SemanticFailure("another EXP703 writer owns this output") from error
        yield


def rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


@contextlib.contextmanager
def phase_heartbeat(
    output: Path, phase: str, interval_seconds: int, run_started: float
) -> Iterator[None]:
    stop = threading.Event()
    phase_started = time.monotonic()

    def publish() -> None:
        atomic_json(
            output / "status.json",
            {
                "schema_version": "exp703-cpu-status-v1",
                "run_id": RUN_ID,
                "status": "running",
                "phase": phase,
                "pid": os.getpid(),
                "phase_elapsed_seconds": time.monotonic() - phase_started,
                "run_elapsed_seconds": time.monotonic() - run_started,
                "process_cpu_seconds": time.process_time(),
                "rss_bytes": rss_bytes(),
                "updated_at": utc_now(),
            },
        )

    def loop() -> None:
        while not stop.wait(interval_seconds):
            publish()

    publish()
    thread = threading.Thread(target=loop, name="exp703-phase-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=max(1, interval_seconds))
        publish()


def validate_evaluation(result: dict[str, Any], width: int, config: dict[str, Any]) -> None:
    try:
        exp702.validate_evaluation(result, width, config)
    except exp702.SemanticFailure as error:
        raise SemanticFailure(str(error)) from error


def load_or_compute_phase(
    *,
    output: Path,
    width: int,
    identity: str,
    config: dict[str, Any],
    compute: Callable[[], dict[str, Any]],
    run_started: float,
) -> tuple[dict[str, Any], bool]:
    phase = f"evaluation/{ARM}/{width}"
    path = output / "phases" / f"evaluation__{ARM}__{width}.json"
    manifest_path = path.with_name(path.stem + ".manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("phase") != phase or manifest.get("identity_sha256") != identity:
            raise SemanticFailure(f"existing {phase} identity mismatch")
        if manifest.get("status") == "completed":
            if not path.is_file() or manifest.get("result_sha256") != sha256(path):
                raise SemanticFailure(f"completed {phase} result mismatch")
            result = json.loads(path.read_text(encoding="utf-8"))
            validate_evaluation(result, width, config)
            return result, True
        if manifest.get("status") != "started":
            raise SemanticFailure(f"existing {phase} manifest has invalid status")
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_evaluation(result, width, config)
    else:
        atomic_json(
            manifest_path,
            {
                "schema_version": "exp703-phase-v1",
                "run_id": RUN_ID,
                "phase": phase,
                "identity_sha256": identity,
                "status": "started",
                "started_at": utc_now(),
            },
        )
        with phase_heartbeat(
            output,
            phase,
            int(config["cpu_execution"]["phase_heartbeat_seconds"]),
            run_started,
        ):
            result = compute()
        validate_evaluation(result, width, config)
        atomic_json(path, result)
    atomic_json(
        manifest_path,
        {
            "schema_version": "exp703-phase-v1",
            "run_id": RUN_ID,
            "phase": phase,
            "identity_sha256": identity,
            "status": "completed",
            "result": str(path),
            "result_sha256": sha256(path),
            "result_bytes": path.stat().st_size,
            "completed_at": utc_now(),
        },
    )
    return result, False


def prepare_model(
    config: dict[str, Any], verified: dict[str, Any], output: Path, device: torch.device
) -> tuple[dict[str, Any], cuda_replay.DeviceExp300, Any, dict[int, Any]]:
    predecessor_config = yaml.safe_load(exp701.CONFIG_PATH.read_text(encoding="utf-8"))
    predecessor_config["_runtime_device"] = str(device)
    exp300 = cuda_replay.DeviceExp300(base.load_exp300(), device, exp701.NoDeadline())
    checkpoint = output / "predecessor" / "straight_through_restarts-model.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and sha256(checkpoint) != config["predecessor"]["checkpoint_sha256"]:
        raise SemanticFailure("existing CPU checkpoint copy mismatch")
    if not checkpoint.exists():
        checkpoint.write_bytes(verified["checkpoint"])
    if sha256(checkpoint) != config["predecessor"]["checkpoint_sha256"]:
        raise SemanticFailure("CPU checkpoint copy mismatch")
    model = exp701.new_model(
        predecessor_config, exp701.ROOT / predecessor_config["source_checkpoint"]
    )
    model.load_state_dict(load_file(checkpoint, device="cpu"))
    model.to(device)
    model.eval()
    if module_device(model) != device:
        raise SemanticFailure("model did not remain on CPU")
    root = int(config["predecessor"]["root"])
    splits = {
        width: exp701.make_split(predecessor_config, exp300.parent, width, root)
        for width in (128, 256)
    }
    return predecessor_config, exp300, model, splits


def verify_only(config_path: Path, archive: Path) -> dict[str, Any]:
    config, exp702_config = load_contract(config_path)
    verified = authenticate_predecessor(config, exp702_config, archive)
    return {
        "schema_version": "exp703-preflight-v1",
        "run_id": RUN_ID,
        "status": "verified",
        "predecessor_identity_sha256": verified["opened"]["identity_sha256"],
        "archive_sha256": sha256(archive),
        "checkpoint_sha256": config["predecessor"]["checkpoint_sha256"],
        "training_allowed": False,
        "evaluation_denominators": config["evaluation"],
    }


def smoke(config_path: Path, archive: Path, output: Path) -> dict[str, Any]:
    config, exp702_config = load_contract(config_path)
    verified = authenticate_predecessor(config, exp702_config, archive)
    device, runtime = configure_cpu(config)
    predecessor_config, exp300, model, splits = prepare_model(
        config, verified, output, device
    )
    root = int(config["predecessor"]["root"])
    started = time.monotonic()
    results = {}
    for width in (128, 256):
        results[str(width)] = exp701.evaluate_with_evidence(
            predecessor_config, exp300, model, splits[width], root, width, 2, 1
        )
    return {
        "schema_version": "exp703-cpu-smoke-v1",
        "run_id": RUN_ID,
        "status": "smoke-passed",
        "runtime": runtime,
        "wall_seconds": time.monotonic() - started,
        "widths": {
            width: {
                "transition_cases": result["transition"]["cases"],
                "rollout_cases": result["rollout"]["cases"],
                "prediction_sha256": result["rollout"]["prediction_sha256"],
            }
            for width, result in results.items()
        },
    }


def execute(config_path: Path, archive: Path, output: Path) -> dict[str, Any]:
    run_started = time.monotonic()
    config, exp702_config = load_contract(config_path)
    with single_writer(output):
        verified = authenticate_predecessor(config, exp702_config, archive)
        device, runtime = configure_cpu(config)
        runner_path = Path(__file__)
        source = {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "runner": str(runner_path),
            "runner_sha256": sha256(runner_path),
            "exp702_config_sha256": sha256(ROOT / config["predecessor"]["exp702_config"]),
            "exp702_runner_sha256": sha256(Path(exp702.__file__)),
            "archive_sha256": sha256(archive),
            "checkpoint_sha256": config["predecessor"]["checkpoint_sha256"],
            "runtime": runtime,
        }
        identity = exp701.canonical_hash({"run_id": RUN_ID, **source})
        atomic_json(
            output / "preflight.json",
            {
                "schema_version": "exp703-preflight-v1",
                "run_id": RUN_ID,
                "status": "verified",
                "identity_sha256": identity,
                "source": source,
                "training_allowed": False,
                "verified_at": utc_now(),
            },
        )
        predecessor_config, exp300, model, splits = prepare_model(
            config, verified, output, device
        )
        root = int(config["predecessor"]["root"])
        results: dict[str, Any] = {}
        reused: list[str] = []
        for width in (128, 256):
            transitions = int(config["evaluation"]["transition_cases_per_width"])
            rollouts = int(config["evaluation"]["rollout_cases"][str(width)])
            result, was_reused = load_or_compute_phase(
                output=output,
                width=width,
                identity=identity,
                config=config,
                compute=lambda width=width, transitions=transitions, rollouts=rollouts: exp701.evaluate_with_evidence(
                    predecessor_config,
                    exp300,
                    model,
                    splits[width],
                    root,
                    width,
                    transitions,
                    rollouts,
                ),
                run_started=run_started,
            )
            results[str(width)] = result
            if was_reused:
                reused.append(f"evaluation/{ARM}/{width}")
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
            config, verified, results, predecessor_source
        )
        terminal.update(
            {
                "schema_version": "exp703-terminal-v1",
                "run_id": RUN_ID,
                "relationship": "cpu-terminal-evaluation-successor-not-exp701-or-exp702-completion",
                "source": {**source, "identity_sha256": identity},
                "recovered_completed_phases": reused,
                "resource": {
                    "wall_seconds": time.monotonic() - run_started,
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
        if terminal_path.exists():
            existing = json.loads(terminal_path.read_text(encoding="utf-8"))
            if existing != terminal:
                raise SemanticFailure("existing EXP703 terminal publication conflicts")
        else:
            atomic_json(terminal_path, terminal)
        atomic_json(
            output / "status.json",
            {
                "schema_version": "exp703-cpu-status-v1",
                "run_id": RUN_ID,
                "status": "completed",
                "phase": "terminal-publication",
                "identity_sha256": identity,
                "terminal_summary": str(terminal_path),
                "terminal_summary_sha256": sha256(terminal_path),
                "run_elapsed_seconds": time.monotonic() - run_started,
                "completed_at": utc_now(),
            },
        )
        return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    archive = args.archive or ROOT / config["predecessor"]["archive"]
    try:
        if args.verify_only:
            result = verify_only(args.config, archive)
        elif args.smoke:
            result = smoke(args.config, archive, args.output)
        else:
            result = execute(args.config, archive, args.output)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, exp702.SemanticFailure) as error:
        failure = {
            "schema_version": "exp703-terminal-failure-v1",
            "run_id": RUN_ID,
            "status": "terminal-failure",
            "reason": str(error),
            "retry_allowed": True,
            "failed_at": utc_now(),
        }
        atomic_json(args.output / "failure.json", failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
