"""Authenticate EXP701 s3t and compute only its two missing evaluations as EXP702."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import inspect
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.cloud_benchmarks.exp699_cuda_replay import run as cuda_replay  # noqa: E402
from experiments.cuda_portability.runtime import configure_determinism, resolve_device, runtime_receipt  # noqa: E402
from experiments.exp_686_w256_straight_through_specialist import run as base  # noqa: E402
from experiments.exp_701_cuda_resumable_h92_w256_backbone_factorial import run as exp701  # noqa: E402

RUN_ID = "exp-702-exp701-terminal-evaluation-continuation"
CONFIG_PATH = ROOT / "configs/exp_702_exp701_terminal_evaluation_continuation.yaml"
OUTPUT = ROOT / ".scratch/exp-702-exp701-terminal-evaluation-continuation"
PREREGISTRATION_PATH = ROOT / "experiments/runs" / RUN_ID / "preregistration.json"
PREFIX = "experiment/exp-701-cuda-resumable-h92-w256-backbone-factorial/"
ARM = "straight_through_restarts"


class SemanticFailure(RuntimeError):
    """A provenance or scope violation that must fail closed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_hash(value: Any) -> str:
    return sha256_bytes(inspect.getsource(value).encode())


def verify_bound_sources(config: dict[str, Any]) -> dict[str, Any]:
    """Authenticate every local byte that defines the successor or its runtime."""
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    if preregistration.get("run_id") != RUN_ID:
        raise SemanticFailure("EXP702 preregistration identity mismatch")
    bindings = preregistration.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise SemanticFailure("EXP702 preregistration lacks exact source bindings")
    for label, spec in bindings.items():
        path = ROOT / str(spec.get("path", ""))
        if not path.is_file():
            raise SemanticFailure(f"bound EXP702 source is absent: {label}")
        actual = sha256(path)
        if actual != spec.get("sha256"):
            raise SemanticFailure(f"bound EXP702 source mismatch: {label}")
        expected_bytes = spec.get("bytes")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise SemanticFailure(f"bound EXP702 source size mismatch: {label}")
    challenge = preregistration.get("challenge_source", {})
    if challenge.get("commit") != config["predecessor"]["source"]["challenge_commit"]:
        raise SemanticFailure("bound challenge source commit mismatch")
    return preregistration


def enforce_cuda_runtime(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    expected = config["gpu_execution"]
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SemanticFailure("EXP702 scientific evaluation requires CUDA; CPU fallback is forbidden")
    if torch.__version__ != expected["torch"]:
        raise SemanticFailure(f"torch runtime mismatch: {torch.__version__!r}")
    if torch.version.cuda != expected["cuda_runtime"]:
        raise SemanticFailure(f"CUDA runtime mismatch: {torch.version.cuda!r}")
    name = torch.cuda.get_device_name(device)
    if name != expected["gpu"]:
        raise SemanticFailure(f"GPU device class mismatch: {name!r}")
    if not torch.are_deterministic_algorithms_enabled():
        raise SemanticFailure("deterministic algorithms are not enabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise SemanticFailure("TF32 must remain disabled")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SemanticFailure("CUBLAS deterministic workspace is not pinned")
    return runtime_receipt(device)


@contextlib.contextmanager
def single_writer(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".exp702-writer.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SemanticFailure("another EXP702 writer owns this output") from error
        yield


def validate_evaluation(result: dict[str, Any], width: int, config: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        raise SemanticFailure(f"evaluation/{ARM}/{width} is not an object")
    for section in ("transition", "rollout"):
        value = result.get(section)
        if not isinstance(value, dict):
            raise SemanticFailure(f"evaluation/{ARM}/{width} lacks {section}")
        accuracy = value.get("exact_accuracy")
        if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool) or not 0.0 <= float(accuracy) <= 1.0:
            raise SemanticFailure(f"evaluation/{ARM}/{width} has invalid {section} exact accuracy")
    expected_transition = int(config["evaluation"]["transition_cases_per_width"])
    expected_rollout = int(config["evaluation"]["rollout_cases"][str(width)])
    for section, expected in (("transition", expected_transition), ("rollout", expected_rollout)):
        observed = result[section].get("cases", result[section].get("total", expected))
        if int(observed) != expected:
            raise SemanticFailure(f"evaluation/{ARM}/{width} changed the preregistered {section} denominator")


def load_or_publish_phase(
    *, output: Path, width: int, identity: str, config: dict[str, Any], compute: Any
) -> tuple[dict[str, Any], bool]:
    path = output / "phases" / f"evaluation__{ARM}__{width}.json"
    manifest_path = path.with_name(path.stem + ".manifest.json")
    phase = f"evaluation/{ARM}/{width}"
    if manifest_path.exists():
        if not path.is_file():
            raise SemanticFailure(f"completed {phase} manifest has no result")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("phase") != phase or manifest.get("identity_sha256") != identity or manifest.get("result_sha256") != sha256(path):
            raise SemanticFailure(f"completed {phase} identity mismatch")
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_evaluation(result, width, config)
        return result, True
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_evaluation(result, width, config)
    else:
        result = compute()
        validate_evaluation(result, width, config)
        atomic_json(path, result)
    manifest = {
        "schema_version": "exp702-phase-v1", "run_id": RUN_ID,
        "phase": phase, "identity_sha256": identity, "status": "completed",
        "result": str(path), "result_sha256": sha256(path),
    }
    atomic_json(manifest_path, manifest)
    return result, False


def _member_bytes(archive: tarfile.TarFile, relative: str) -> bytes:
    name = PREFIX + relative
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise SemanticFailure(f"predecessor member missing: {relative}") from error
    if not member.isfile() or member.issym() or member.islnk():
        raise SemanticFailure(f"predecessor member is not a regular file: {relative}")
    handle = archive.extractfile(member)
    if handle is None:
        raise SemanticFailure(f"predecessor member unreadable: {relative}")
    return handle.read()


def _json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticFailure(f"invalid predecessor JSON: {label}") from error
    if not isinstance(value, dict):
        raise SemanticFailure(f"invalid predecessor object: {label}")
    return value


def verify_predecessor(config: dict[str, Any], archive_path: Path) -> dict[str, Any]:
    verify_bound_sources(config)
    predecessor = config["predecessor"]
    if sha256(archive_path) != predecessor["archive_sha256"] or archive_path.stat().st_size != int(predecessor["archive_bytes"]):
        raise SemanticFailure("predecessor archive byte identity mismatch")
    source_paths = {
        "config_sha256": exp701.CONFIG_PATH,
        "runner_sha256": Path(exp701.__file__),
        "test_sha256": exp701.TEST_PATH,
        "source_checkpoint_sha256": ROOT / exp701.Experiment.__init__.__globals__["yaml"].safe_load(exp701.CONFIG_PATH.read_text())["source_checkpoint"],
    }
    for key, path in source_paths.items():
        if not path.is_file() or sha256(path) != predecessor["source"][key]:
            raise SemanticFailure(f"frozen EXP701 source mismatch: {key}")
    semantic_objects = {
        "evaluate_with_evidence_sha256": cuda_replay.evaluate_with_evidence,
        "evaluate_sha256": cuda_replay.evaluate,
        "device_exp300_sha256": cuda_replay.DeviceExp300,
        "make_split_sha256": exp701.make_split,
    }
    for key, value in semantic_objects.items():
        if source_hash(value) != predecessor["semantics"][key]:
            raise SemanticFailure(f"frozen evaluation semantic mismatch: {key}")
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        if len(names) != len(set(names)):
            raise SemanticFailure("duplicate archive member names")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise SemanticFailure("unsafe predecessor archive path")
        values: dict[str, Any] = {}
        raw_members: dict[str, bytes] = {}
        for key, spec in predecessor["members"].items():
            data = _member_bytes(archive, spec["path"])
            if sha256_bytes(data) != spec["sha256"]:
                raise SemanticFailure(f"predecessor member hash mismatch: {key}")
            raw_members[key] = data
            if spec["path"].endswith(".json"):
                values[key] = _json(data, key)
        opened = values["root_opened"]
        lifecycle = values["lifecycle"]
        terminal = values["terminal_failure"]
        missing = values["missing_128_manifest"]
        if opened.get("identity_sha256") != predecessor["identity_sha256"] or opened.get("root") != predecessor["root"]:
            raise SemanticFailure("predecessor root identity mismatch")
        if opened.get("source", {}).get("repository_commit") != predecessor["source_commit"] or opened.get("source", {}).get("scientific_contract_sha256") != predecessor["source"]["scientific_contract_sha256"]:
            raise SemanticFailure("predecessor scientific source mismatch")
        if lifecycle.get("recoveries") != 3 or lifecycle.get("identity_sha256") != predecessor["identity_sha256"]:
            raise SemanticFailure("predecessor recovery ledger mismatch")
        if terminal.get("status") != "terminal-failure" or terminal.get("retry_allowed") is not False or terminal.get("reason") != "recoverable restart bound exceeded":
            raise SemanticFailure("EXP701 is not the frozen terminal-incomplete predecessor")
        if missing.get("status") != "started" or missing.get("phase") != "evaluation/straight_through_restarts/128":
            raise SemanticFailure("missing evaluation evidence is ambiguous")
        progress = values["progress"]
        if set(progress.get("evaluations", {})) != {"teacher_cosine", "teacher_restarts", "straight_through_cosine"}:
            raise SemanticFailure("unexpected completed evaluation set")
        for arm, widths in predecessor["reused_evaluation_result_sha256"].items():
            for width, expected in widths.items():
                relative = f"formal/root-{predecessor['root']}/phases/evaluation__{arm}__{width}.json"
                if sha256_bytes(_member_bytes(archive, relative)) != expected:
                    raise SemanticFailure(f"reused evaluation hash mismatch: {arm}/{width}")
        training = values["final_training"]
        if training.get("arm") != ARM or training.get("training", {}).get("updates") != 4800:
            raise SemanticFailure("final arm training is incomplete")
        if training.get("final_checkpoint", {}).get("sha256") != predecessor["members"]["final_checkpoint"]["sha256"]:
            raise SemanticFailure("final checkpoint record mismatch")
    return {"opened": opened, "lifecycle": lifecycle, "terminal": terminal, "progress": progress, "training": training, "checkpoint": raw_members["final_checkpoint"]}


def integrate_result(config: dict[str, Any], verified: dict[str, Any], missing: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for width in (128, 256):
        validate_evaluation(missing.get(str(width)), width, config)
    evaluations = json.loads(json.dumps(verified["progress"]["evaluations"]))
    evaluations[ARM] = missing
    if set(evaluations[ARM]) != {"128", "256"}:
        raise SemanticFailure("both successor evaluations are required")
    winner = max(evaluations, key=lambda arm: evaluations[arm]["256"]["rollout"]["exact_accuracy"])
    best_w256 = float(evaluations[winner]["256"]["rollout"]["exact_accuracy"])
    gate = float(config["terminal_policy"]["advancement_gate_w256_rollout"])
    positive = best_w256 >= gate
    return {
        "schema_version": "exp702-terminal-v1",
        "run_id": RUN_ID,
        "status": "completed",
        "relationship": config["predecessor"]["relationship"],
        "predecessor": source,
        "root": config["predecessor"]["root"],
        "evaluations": evaluations,
        "winner": winner,
        "confirmation_trigger": positive,
        "decision": (
            config["terminal_policy"]["if_gate_reached"]
            if positive else config["terminal_policy"]["otherwise_decision"]
        ),
        "additional_diagnostics_executed": False,
        "executed_phases": [f"evaluation/{ARM}/128", f"evaluation/{ARM}/256"],
        "reused_phases": [f"evaluation/{arm}/{width}" for arm in sorted(evaluations) if arm != ARM for width in ("128", "256")],
        "training_executed": False,
        "limitations": config["limitations"],
    }


def execute(config_path: Path, archive_path: Path, output: Path, device_name: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("run_id") != RUN_ID or config.get("evaluation", {}).get("training_allowed") is not False:
        raise SemanticFailure("unexpected successor contract")
    with single_writer(output):
        verified = verify_predecessor(config, archive_path)
        device = resolve_device(device_name, require_accelerator=True)
        predecessor_config = yaml.safe_load(exp701.CONFIG_PATH.read_text())
        predecessor_config["_runtime_device"] = str(device)
        root = int(config["predecessor"]["root"])
        configure_determinism(device=device, seed=root + 698, torch_threads=int(predecessor_config["training"]["torch_threads"]))
        runtime = enforce_cuda_runtime(config, device)
        exp300 = cuda_replay.DeviceExp300(base.load_exp300(), device, exp701.NoDeadline())
        checkpoint_copy = output / "predecessor" / "straight_through_restarts-model.safetensors"
        checkpoint_copy.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_copy.exists() and sha256(checkpoint_copy) != config["predecessor"]["members"]["final_checkpoint"]["sha256"]:
            raise SemanticFailure("existing successor checkpoint copy mismatch")
        if not checkpoint_copy.exists():
            checkpoint_copy.write_bytes(verified["checkpoint"])
        if sha256(checkpoint_copy) != config["predecessor"]["members"]["final_checkpoint"]["sha256"]:
            raise SemanticFailure("successor checkpoint copy mismatch")
        model = exp701.new_model(predecessor_config, exp701.ROOT / predecessor_config["source_checkpoint"])
        model.load_state_dict(load_file(checkpoint_copy)); model.eval()
        splits = {width: exp701.make_split(predecessor_config, exp300.parent, width, root) for width in (128, 256)}
        identity = exp701.canonical_hash({"run_id": RUN_ID, "config_sha256": sha256(config_path), "archive_sha256": sha256(archive_path), "checkpoint_sha256": sha256(checkpoint_copy), "runtime": runtime})
        results: dict[str, Any] = {}
        reused: list[str] = []
        for width in (128, 256):
            transitions = int(config["evaluation"]["transition_cases_per_width"])
            rollouts = int(config["evaluation"]["rollout_cases"][str(width)])
            result, was_reused = load_or_publish_phase(
                output=output, width=width, identity=identity, config=config,
                compute=lambda width=width, transitions=transitions, rollouts=rollouts: exp701.evaluate_with_evidence(predecessor_config, exp300, model, splits[width], root, width, transitions, rollouts),
            )
            results[str(width)] = result
            if was_reused:
                reused.append(f"evaluation/{ARM}/{width}")
        predecessor_source = {"run_id": config["predecessor"]["run_id"], "status": "terminal-incomplete", "archive_sha256": config["predecessor"]["archive_sha256"], "identity_sha256": config["predecessor"]["identity_sha256"], "source_commit": config["predecessor"]["source_commit"], "recovery_count": verified["lifecycle"]["recoveries"], "terminal_reason": verified["terminal"]["reason"]}
        terminal = integrate_result(config, verified, results, predecessor_source)
        terminal["source"] = {"config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "runtime": runtime, "identity_sha256": identity}
        terminal["recovered_completed_phases"] = reused
        terminal_path = output / "terminal-summary.json"
        if terminal_path.exists():
            if json.loads(terminal_path.read_text(encoding="utf-8")) != terminal:
                raise SemanticFailure("existing EXP702 terminal publication conflicts with authenticated phases")
        else:
            atomic_json(terminal_path, terminal)
        return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    archive = args.archive or ROOT / config["predecessor"]["archive"]
    try:
        verified = verify_predecessor(config, archive)
        if args.verify_only:
            print(json.dumps({"status": "verified", "run_id": RUN_ID, "predecessor_identity_sha256": verified["opened"]["identity_sha256"]}, sort_keys=True))
            return 0
        print(json.dumps(execute(args.config, archive, args.output, args.device), sort_keys=True))
        return 0
    except (OSError, SemanticFailure, tarfile.TarError) as error:
        print(json.dumps({"status": "terminal-failure", "run_id": RUN_ID, "reason": str(error), "retry_allowed": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
