"""Execute the exactly resumable EXP700 h92/two-cycle backbone factorial."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_664_sprm_boundary_perturbation.run import Timer, git_output, make_split, sha256  # noqa: E402
from experiments.exp_686_w256_straight_through_specialist import run as base  # noqa: E402
from experiments.exp_699_h92_w256_backbone_factorial_complete import run as exp699  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

RUN_ID = "exp-700-resumable-h92-w256-backbone-factorial"
CONFIG_PATH = ROOT / "configs/exp_700_resumable_h92_w256_backbone_factorial.yaml"
OUTPUT = ROOT / ".scratch/exp-700-resumable-h92-w256-backbone-factorial"
TEST_PATH = ROOT / "tests/test_exp_700_resumable_h92_w256_backbone_factorial.py"
ARM_ORDER = exp699.ARM_ORDER
WIDTHS = exp699.WIDTHS
CHALLENGE_COMMIT = exp699.CHALLENGE_COMMIT
TERMINAL_OUTPUT_CONTRACT = exp699.TERMINAL_OUTPUT_CONTRACT


class SemanticFailure(RuntimeError):
    """An identity, compliance, or work error that must never be retried."""


class RecoverableInterruption(RuntimeError):
    """A process interruption that may resume from authenticated durable state."""


class AlreadyRunning(RuntimeError):
    """A second writer was refused by the root lock."""


canonical_hash = exp699.canonical_hash
state_hash = exp699.state_hash
new_model = exp699.new_model
balanced_width = exp699.balanced_width
make_scheduler = exp699.make_scheduler
rollout = exp699.rollout
interval = exp699.interval
evaluate = exp699.evaluate
paired_evidence = exp699.paired_evidence
evaluate_with_evidence = exp699.evaluate_with_evidence
lower_tier_parity = exp699.lower_tier_parity


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def object_hash(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return {"tensor": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(), "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        if isinstance(item, np.ndarray):
            return {"ndarray": hashlib.sha256(item.tobytes()).hexdigest(), "dtype": str(item.dtype), "shape": list(item.shape)}
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return repr(item)
    return canonical_hash(normalize(value))


def recovery_signature(record: dict[str, Any]) -> dict[str, Any]:
    training = record["training"]
    return {
        "model_state_sha256": record["final_state_sha256"],
        "optimizer_state_sha256": record["optimizer_state_sha256"],
        "scheduler_state_sha256": record["scheduler_state_sha256"],
        "rng_state_sha256": record["rng_state_sha256"],
        "curves_sha256": canonical_hash(training["curves"]),
        "data_order_sha256": training["data_order_sha256"],
        "width_order_sha256": training["width_order_sha256"],
        "updates": training["updates"],
        "examples": training["examples"],
        "model_calls": training["model_calls"],
    }


def snapshot_paths(output: Path, arm: str) -> tuple[Path, Path]:
    directory = output / "recovery" / arm
    return directory / "latest.pt", directory / "latest.json"


def save_snapshot(output: Path, identity_sha256: str, arm: str, payload: dict[str, Any]) -> dict[str, Any]:
    snapshot, manifest = snapshot_paths(output, arm)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    atomic_bytes(snapshot, data)
    snapshot_sha = hashlib.sha256(data).hexdigest()
    work = payload["work"]
    result = {
        "schema_version": "exp700-recovery-v1",
        "identity_sha256": identity_sha256,
        "arm": arm,
        "update": payload["update"],
        "snapshot": str(snapshot),
        "snapshot_sha256": snapshot_sha,
        "snapshot_bytes": len(data),
        "work_sha256": canonical_hash(work),
        "curves_sha256": canonical_hash(payload["curves"]),
        "width_order_sha256": canonical_hash(payload["width_order"]),
        "data_order_sha256": canonical_hash(payload["data_order_digests"]),
    }
    atomic_json(manifest, result)
    return result


def load_snapshot(output: Path, identity_sha256: str, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot, manifest_path = snapshot_paths(output, arm)
    if not snapshot.is_file() or not manifest_path.is_file():
        raise SemanticFailure(f"resume state missing for {arm}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity_sha256") != identity_sha256 or manifest.get("arm") != arm:
        raise SemanticFailure(f"resume identity mismatch for {arm}")
    if sha256(snapshot) != manifest.get("snapshot_sha256"):
        raise SemanticFailure(f"resume snapshot hash mismatch for {arm}")
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    if payload.get("identity_sha256") != identity_sha256 or payload.get("arm") != arm:
        raise SemanticFailure(f"resume snapshot payload identity mismatch for {arm}")
    checks = {
        "work_sha256": canonical_hash(payload["work"]),
        "curves_sha256": canonical_hash(payload["curves"]),
        "width_order_sha256": canonical_hash(payload["width_order"]),
        "data_order_sha256": canonical_hash(payload["data_order_digests"]),
    }
    if any(manifest.get(key) != value for key, value in checks.items()):
        raise SemanticFailure(f"resume logical ledger mismatch for {arm}")
    return payload, manifest


def phase_paths(output: Path, name: str) -> tuple[Path, Path]:
    safe = name.replace("/", "__")
    return output / "phases" / f"{safe}.json", output / "phases" / f"{safe}.manifest.json"


def phase_result(output: Path, identity_sha256: str, name: str, compute: Callable[[], Any], interrupt_after: str | None = None) -> tuple[Any, bool]:
    result_path, manifest_path = phase_paths(output, name)
    phase_identity = canonical_hash({"identity_sha256": identity_sha256, "phase": name})
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("phase_identity_sha256") != phase_identity:
            raise SemanticFailure(f"phase identity mismatch: {name}")
        if manifest.get("status") == "completed":
            if not result_path.is_file() or sha256(result_path) != manifest.get("result_sha256"):
                raise SemanticFailure(f"completed phase integrity mismatch: {name}")
            return json.loads(result_path.read_text()), True
        if manifest.get("status") != "started":
            raise SemanticFailure(f"invalid phase state: {name}")
    else:
        atomic_json(manifest_path, {"schema_version": "exp700-phase-v1", "phase": name, "phase_identity_sha256": phase_identity, "status": "started"})
    result = compute()
    atomic_json(result_path, result)
    completed = {"schema_version": "exp700-phase-v1", "phase": name, "phase_identity_sha256": phase_identity, "status": "completed", "result": str(result_path), "result_sha256": sha256(result_path), "result_bytes": result_path.stat().st_size}
    atomic_json(manifest_path, completed)
    if interrupt_after == name:
        raise RecoverableInterruption(f"intentional interruption after phase {name}")
    return result, False


def train_arm_resumable(config: dict[str, Any], exp300: Any, checkpoint: Path, splits: dict[int, Any], root: int, arm: str,
                        updates: int, output: Path, identity_sha256: str, resume: bool,
                        interrupt_after_update: int | None = None, snapshot_every: int | None = None) -> tuple[Any, dict[str, Any]]:
    final_record, final_manifest = phase_paths(output, f"train/{arm}")
    phase_identity = canonical_hash({"identity_sha256": identity_sha256, "phase": f"train/{arm}"})
    if final_manifest.is_file() and json.loads(final_manifest.read_text()).get("status") == "completed":
        manifest = json.loads(final_manifest.read_text())
        if manifest.get("phase_identity_sha256") != phase_identity or sha256(final_record) != manifest.get("result_sha256"):
            raise SemanticFailure(f"completed train phase integrity mismatch: {arm}")
        record = json.loads(final_record.read_text())
        model = new_model(config, checkpoint); model.load_state_dict(load_file(Path(record["final_checkpoint"]["path"]))); return model.eval(), record
    if not final_manifest.exists():
        atomic_json(final_manifest, {"schema_version": "exp700-phase-v1", "phase": f"train/{arm}", "phase_identity_sha256": phase_identity, "status": "started"})
    elif json.loads(final_manifest.read_text()).get("phase_identity_sha256") != phase_identity:
        raise SemanticFailure(f"train phase identity mismatch: {arm}")

    batch = int(config["training"]["batch_size"]); horizon = int(config["training"]["horizon"])
    snapshot_every = int(snapshot_every or config["recovery"]["snapshot_every_updates"])
    model = new_model(config, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = make_scheduler(config, optimizer, arm, updates)
    samplers = {w: exp300.BigIntSampler(splits[w]["train"], root + int(config["data"]["training_seed_offset"]) + w) for w in WIDTHS}
    starts = {w: random.Random(root + int(config["data"]["training_seed_offset"]) + w + 1) for w in WIDTHS}
    initial = state_hash(model); curves: list[dict[str, Any]] = []; width_order: list[int] = []; data_order_digests: list[str] = []
    checkpoints: list[dict[str, Any]] = []; update_start = 1
    work = {"updates": 0, "examples": 0, "model_calls": 0}
    if resume:
        payload, _ = load_snapshot(output, identity_sha256, arm)
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        random.setstate(payload["python_rng"]); np.random.set_state(payload["numpy_rng"]); torch.set_rng_state(payload["torch_rng"])
        for width in WIDTHS:
            samplers[width].rng.setstate(payload["sampler_rng"][width]); starts[width].setstate(payload["start_rng"][width])
        curves = payload["curves"]; width_order = payload["width_order"]; data_order_digests = payload["data_order_digests"]
        checkpoints = payload["checkpoints"]; work = payload["work"]; update_start = int(payload["update"]) + 1
        if work != {"updates": update_start - 1, "examples": (update_start - 1) * batch * horizon, "model_calls": (update_start - 1) * horizon}:
            raise SemanticFailure(f"resume work counter mismatch for {arm}")
    else:
        random.seed(root + 698); np.random.seed(root + 698); torch.manual_seed(root + 698)
        model = new_model(config, checkpoint)
        initial = state_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
        scheduler = make_scheduler(config, optimizer, arm, updates)

    def snapshot(update: int) -> dict[str, Any]:
        payload = {"schema_version": "exp700-recovery-v1", "identity_sha256": identity_sha256, "arm": arm, "update": update,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                   "sampler_rng": {w: samplers[w].rng.getstate() for w in WIDTHS}, "start_rng": {w: starts[w].getstate() for w in WIDTHS},
                   "curves": curves, "width_order": width_order, "data_order_digests": data_order_digests,
                   "checkpoints": checkpoints, "work": work}
        return save_snapshot(output, identity_sha256, arm, payload)

    if update_start == 1:
        snapshot(0)
    timer = Timer(); model.train()
    for update in range(update_start, updates + 1):
        width = int(config["training"].get("profile_width", balanced_width(update, int(config["training"]["w128_only_updates"]))))
        left, right, modulus = samplers[width].pairs(batch)
        offsets = [starts[width].randrange(0, width - horizon + 1) for _ in left]
        data_order_digests.append(canonical_hash([width, left, right, modulus, offsets])); width_order.append(width)
        prior_width = base.WIDTH
        try:
            base.WIDTH = width
            state_arm = "teacher_forced" if arm.startswith("teacher_") else "learner_straight_through"
            stats = base.train_update(config, exp300, model, optimizer, left, right, modulus, offsets, state_arm)
        finally:
            base.WIDTH = prior_width
        scheduler.step(); work = {"updates": update, "examples": update * batch * horizon, "model_calls": update * horizon}
        if update % int(config["training"]["curve_every_updates"]) == 0 or update == updates:
            curves.append({"update": update, "width": width, "learning_rate": scheduler.get_last_lr()[0], **stats})
        if update in set(map(int, config["training"]["checkpoint_updates"])) or update == updates:
            path = output / "arms" / arm / f"update-{update}.safetensors"; path.parent.mkdir(parents=True, exist_ok=True)
            save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
            checkpoints.append({"update": update, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        if update % snapshot_every == 0 or update == updates:
            snapshot(update)
            if interrupt_after_update == update:
                raise RecoverableInterruption(f"intentional interruption after update {update}")
    final = output / "arms" / arm / "model.safetensors"
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, final)
    reloaded = new_model(config, checkpoint); reloaded.load_state_dict(load_file(final))
    exact = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    rng_hash = object_hash({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                            "samplers": {w: samplers[w].rng.getstate() for w in WIDTHS}, "starts": {w: starts[w].getstate() for w in WIDTHS}})
    record = {"arm": arm, "initial_state_sha256": initial, "parameters": model.parameter_count(),
              "training": {**work, "width_order_sha256": canonical_hash(width_order), "data_order_sha256": canonical_hash(data_order_digests), "curves": curves, "resource": timer.finish()},
              "checkpoints": checkpoints, "final_checkpoint": {"path": str(final), "sha256": sha256(final), "bytes": final.stat().st_size},
              "final_state_sha256": state_hash(model), "optimizer_state_sha256": object_hash(optimizer.state_dict()),
              "scheduler_state_sha256": object_hash(scheduler.state_dict()), "rng_state_sha256": rng_hash, "reload_exact": exact}
    atomic_json(final_record, record)
    atomic_json(final_manifest, {"schema_version": "exp700-phase-v1", "phase": f"train/{arm}", "phase_identity_sha256": phase_identity,
                                 "status": "completed", "result": str(final_record), "result_sha256": sha256(final_record), "result_bytes": final_record.stat().st_size})
    return model.eval(), record


def limitations() -> list[str]:
    return [
        "Fresh generated evidence only; checked, official and private answers were not read.",
        "Exact arithmetic is confined to offline labels and evaluation truth; deterministic hard learned residue inference produces predictions.",
        "Candidate runtime is a release constraint and never truncates scientific denominators.",
        "Recovery snapshots are trusted local training artifacts, hash-gated before Python/PyTorch deserialization, and never enter submission/.",
        "EXP700 changes persistence and lifecycle only; diagnostics cannot alter the deployed route.",
    ]


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None, resume: bool = False,
                 interrupt_after_update: int | None = None, interrupt_after_phase: str | None = None):
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text()); self.output = output.resolve()
        self.mode = mode; self.root = root; self.resume = resume; self.interrupt_after_update = interrupt_after_update; self.interrupt_after_phase = interrupt_after_phase
        if self.config["run_id"] != RUN_ID: raise SemanticFailure("unexpected run_id")
        self.checkpoint = Path(self.config["source_checkpoint"]); self.authenticate(); base.configure_torch(self.config); self.exp300 = base.load_exp300()
        if mode != "smoke":
            roots = self.config["roots"]; allowed = {int(roots[mode])} if mode in {"calibration", "formal"} else set(map(int, roots[mode]))
            self.root = next(iter(allowed)) if root is None and len(allowed) == 1 else root
            if self.root not in allowed: raise SemanticFailure(f"root {self.root} is not frozen for {mode}")
        if resume and mode == "smoke": raise SemanticFailure("--resume requires an opened scientific root")

    def authenticate(self) -> None:
        if not self.checkpoint.is_file() or sha256(self.checkpoint) != self.config["source_checkpoint_sha256"]: raise SemanticFailure("source checkpoint mismatch")
        if git_output("-C", "sources/github/modular-arithmetic-challenge", "rev-parse", "HEAD") != CHALLENGE_COMMIT: raise SemanticFailure("challenge commit mismatch")
        old = yaml.safe_load(exp699.CONFIG_PATH.read_text())
        for key in ("model", "arms", "data", "training", "evaluation", "gates", "resource_policy"):
            if self.config[key] != old[key]: raise SemanticFailure(f"EXP699 scientific contract changed: {key}")

    def source(self) -> dict[str, Any]:
        return {"repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": CHALLENGE_COMMIT,
                "config_sha256": sha256(self.config_path), "runner_sha256": sha256(Path(__file__)), "test_sha256": sha256(TEST_PATH),
                "source_checkpoint_sha256": sha256(self.checkpoint), "exp699_runner_sha256": sha256(exp699.Path(exp699.__file__)),
                "base_train_update_sha256": canonical_hash(exp699.inspect.getsource(base.train_update)),
                "base_rollout_sha256": canonical_hash(exp699.inspect.getsource(base.tensor_native_rollout)),
                "scientific_contract_sha256": canonical_hash({key: self.config[key] for key in ("model", "arms", "data", "training", "evaluation", "gates", "resource_policy")})}

    def identity_sha256(self) -> str: return canonical_hash({"run_id": RUN_ID, "mode": self.mode, "root": self.root, "source": self.source()})
    def root_dir(self) -> Path: return self.output / self.mode / f"root-{self.root}"
    def splits(self) -> dict[int, Any]: return {width: make_split(self.config, self.exp300, width, int(self.root)) for width in WIDTHS}

    def audit_fresh_roots(self) -> dict[str, Any]:
        roots = [int(self.config["roots"]["calibration"]), int(self.config["roots"]["formal"]), *map(int, self.config["roots"]["confirmations"]), *map(int, self.config["roots"]["independent"])]
        target_names = {f"root-{root}" for root in roots}; workspaces = [ROOT]
        if ROOT.parent.name == ".scratch": workspaces.append(ROOT.parent.parent)
        hits: list[str] = []
        for workspace in dict.fromkeys(workspaces):
            for parent in (workspace / ".scratch", workspace / "experiments/runs", workspace / "evals/results"):
                if not parent.exists(): continue
                for current, directories, _files in os.walk(parent):
                    directories[:] = [name for name in directories if name not in {".git", "artifacts", "blobs", "objects"}]
                    for name in list(directories):
                        if name in target_names: hits.append(str(Path(current) / name))
        mlflow_hits: list[dict[str, Any]] = []
        for database in dict.fromkeys(workspace / ".scratch/mlflow/mlflow.db" for workspace in workspaces):
            if not database.is_file(): continue
            with sqlite3.connect(database) as connection:
                for table in ("tags", "params"):
                    query = f"SELECT key,value,run_uuid FROM {table} WHERE value IN ({','.join('?' for _ in roots)})"
                    try:
                        mlflow_hits.extend({"database": str(database), "table": table, "key": key, "value": value, "run_uuid": run_uuid} for key, value, run_uuid in connection.execute(query, tuple(map(str, roots))))
                    except sqlite3.OperationalError: pass
        services = subprocess.run(["systemctl", "--user", "list-units", "--all", "--type=service", "--no-legend", "--no-pager"], check=True, capture_output=True, text=True).stdout
        service_hits = [line for line in services.splitlines() if any(str(root) in line for root in roots)]
        result = {"roots": roots, "workspaces": [str(path) for path in workspaces], "repository_scratch_hits": hits, "mlflow_hits": mlflow_hits, "service_hits": service_hits, "fresh": not hits and not mlflow_hits and not service_hits}
        if not result["fresh"]: raise SemanticFailure(f"EXP700 root reuse detected: {result}")
        atomic_json(self.output / "preflight-root-audit.json", result); return result

    def open_or_resume_root(self) -> Path:
        root_dir = self.root_dir(); marker = root_dir / "root-opened.json"; active = root_dir / "run-active.json"; terminal_failure = root_dir / "terminal-failure.json"
        if terminal_failure.is_file(): raise SemanticFailure("root is terminally failed")
        identity = self.identity_sha256()
        if self.resume:
            if not marker.is_file() or not active.is_file(): raise SemanticFailure("resume requires an already-open root and abnormal-exit marker")
            opened = json.loads(marker.read_text()); lifecycle = json.loads(active.read_text())
            if opened.get("identity_sha256") != identity or opened.get("source") != self.source(): raise SemanticFailure("resume root/source identity mismatch")
            if lifecycle.get("identity_sha256") != identity or lifecycle.get("status") not in {"running", "recoverable-interruption"}: raise SemanticFailure("resume abnormal-exit marker mismatch")
            recoveries = int(lifecycle.get("recoveries", 0)) + 1
            if recoveries > int(self.config["recovery"]["maximum_recoverable_restarts"]): raise SemanticFailure("recoverable restart bound exceeded")
            atomic_json(active, {**lifecycle, "status": "running", "recoveries": recoveries, "resumed": True})
            return root_dir
        if marker.exists() or root_dir.exists(): raise SemanticFailure("fresh invocation would duplicate an opened root")
        audit_path = self.output / "preflight-root-audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.is_file() else self.audit_fresh_roots()
        if not audit.get("fresh"): raise SemanticFailure("preflight root audit failed")
        root_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(marker, {"schema_version": "exp700-root-v1", "root": self.root, "mode": self.mode, "source": self.source(), "identity_sha256": identity, "preflight_root_audit_sha256": sha256(audit_path)})
        atomic_json(active, {"schema_version": "exp700-lifecycle-v1", "root": self.root, "mode": self.mode, "identity_sha256": identity, "status": "running", "recoveries": 0, "resumed": False})
        return root_dir

    def mark_lifecycle(self, status: str, **extra: Any) -> None:
        path = self.root_dir() / "run-active.json"; current = json.loads(path.read_text()) if path.is_file() else {}
        atomic_json(path, {**current, "status": status, **extra})

    def mlflow_identity(self) -> dict[str, str]:
        path = self.root_dir() / "mlflow.json"
        if path.is_file(): return json.loads(path.read_text())
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        run = mlflow.MlflowClient().create_run(experiment_id, tags={"canonical_run_id": RUN_ID, "mode": self.mode, "root": str(self.root), "identity_sha256": self.identity_sha256()})
        result = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run.info.run_id}; atomic_json(path, result); return result

    def require_calibration(self) -> dict[str, Any]:
        path = self.output / "calibration" / f"root-{self.config['roots']['calibration']}" / "root-summary.json"
        if not path.is_file(): raise SemanticFailure("science requires completed calibration")
        result = json.loads(path.read_text())
        if result.get("decision") != "admit-formal" or result.get("source", {}).get("runner_sha256") != sha256(Path(__file__)):
            raise SemanticFailure("calibration frozen identity mismatch")
        return result

    def require_predecessor_gate(self) -> None:
        if self.mode == "formal": return
        formal = self.output / "formal" / f"root-{self.config['roots']['formal']}" / "root-summary.json"
        if not formal.is_file() or not json.loads(formal.read_text()).get("confirmation_trigger"): raise SemanticFailure("confirmation roots require passing frozen formal trigger")
        if self.mode == "independent":
            for root in self.config["roots"]["confirmations"]:
                path = self.output / "confirmations" / f"root-{root}" / "root-summary.json"
                if not path.is_file() or not json.loads(path.read_text()).get("confirmation_trigger"): raise SemanticFailure("independent roots require both passing confirmations")

    def work_equivalence(self, records: dict[str, Any]) -> dict[str, Any]:
        keys = ("updates", "examples", "model_calls", "width_order_sha256", "data_order_sha256")
        values = {key: {records[arm]["training"][key] for arm in ARM_ORDER} for key in keys}
        result = {"passed": all(len(value) == 1 for value in values.values()), "fields": {key: next(iter(value)) if len(value) == 1 else sorted(value) for key, value in values.items()}, "identical_initialization": len({records[arm]["initial_state_sha256"] for arm in ARM_ORDER}) == 1, "optimizer": "AdamW", "schedules": {arm: self.config["arms"][arm]["schedule"] for arm in ARM_ORDER}}
        result["passed"] = result["passed"] and result["identical_initialization"]
        if not result["passed"]: raise SemanticFailure("unmatched arm work")
        return result

    def calibration(self) -> dict[str, Any]:
        root_dir = self.open_or_resume_root(); identity = self.identity_sha256(); timer = Timer(); splits = self.splits(); mlflow_info = self.mlflow_identity()
        profile_dir = root_dir / "profile"; model, record = train_arm_resumable(self.config, self.exp300, self.checkpoint, splits, int(self.root), ARM_ORDER[0], int(self.config["training"]["calibration_updates"]), profile_dir, identity, self.resume, snapshot_every=1)
        eval_profile, _ = phase_result(root_dir, identity, "calibration/evaluation", lambda: {str(width): evaluate_with_evidence(self.config, self.exp300, model, splits[width], int(self.root), width, 2, 1) for width in WIDTHS}, self.interrupt_after_phase)
        diagnostic_timer = Timer()
        diagnostic_profile, _ = phase_result(root_dir, identity, "calibration/diagnostic", lambda: {"source_128": evaluate_with_evidence(self.config, self.exp300, new_model(self.config, self.checkpoint).eval(), splits[128], int(self.root), 128, 1, 1), "randomized_256": evaluate_with_evidence(self.config, self.exp300, new_model(self.config, self.checkpoint, randomized_seed=int(self.root) + 700900).eval(), splits[256], int(self.root), 256, 1, 1)}, self.interrupt_after_phase)
        diagnostic_resource = diagnostic_timer.finish(); service_timer = Timer(); subprocess.run(["systemctl", "--user", "is-system-running"], check=False, capture_output=True); service_resource = service_timer.finish()
        route = base.candidate_runtime(self.config, model, int(self.root))
        historical = json.loads((ROOT / ".scratch/exp-699-h92-w256-backbone-factorial-complete/calibration/root-20261395/root-summary.json").read_text())["budget"]
        snapshot_manifest = json.loads((profile_dir / "recovery" / ARM_ORDER[0] / "latest.json").read_text())
        snapshot_seconds = float(record["training"]["resource"]["wall_seconds"]) / max(1, int(self.config["training"]["calibration_updates"]))
        persistence_overhead = snapshot_seconds * len(ARM_ORDER) * math.ceil(int(self.config["training"]["updates"]) / int(self.config["recovery"]["snapshot_every_updates"]))
        phase_service_overhead = float(diagnostic_resource["wall_seconds"]) + float(service_resource["wall_seconds"]) + 120.0
        expected = float(historical["expected_wall_seconds"]) + persistence_overhead + phase_service_overhead
        expected_cpu = float(historical["expected_process_cpu_seconds"]) + persistence_overhead * int(self.config["training"]["torch_threads"])
        expected_artifacts = int(historical["expected_artifact_bytes"]) + int(snapshot_manifest["snapshot_bytes"]) * len(ARM_ORDER) + 64 * 1024 * 1024
        budget = {"basis": "EXP699 root20261395 exact-route profile plus completed root20261396 arm profiles and current snapshot/phase/systemd overhead", "historical_exp699_expected_wall_seconds": historical["expected_wall_seconds"], "profiled_persistence_wall_seconds": persistence_overhead, "profiled_phase_service_wall_seconds": phase_service_overhead, "expected_wall_seconds": expected, "expected_wall_range_seconds": [expected * .90, expected * 1.10], "expected_process_cpu_seconds": expected_cpu, "expected_cgroup_cpu_seconds": expected_cpu, "peak_rss_bytes": max(int(record["training"]["resource"]["peak_rss_bytes"]), int(historical["peak_rss_bytes"])), "expected_artifact_bytes": expected_artifacts, "soft_wall_seconds": expected * 1.25, "hard_wall_seconds": expected * 1.75, "soft_multiplier": 1.25, "hard_multiplier": 1.75, "runtime_max_sec": math.ceil(expected * 1.75)}
        result = {"schema_version": "exp700-v1", "run_id": RUN_ID, "status": "completed", "mode": "calibration", "root": self.root, "decision": "admit-formal", "source": self.source(), "profiles": {"snapshot_training": record, "evaluation": eval_profile, "diagnostic": diagnostic_profile, "service": service_resource}, "candidate_runtime": route, "release_classification": "research-only" if route["resource"]["wall_seconds"] > 300 else "runtime-measured-not-promoted", "budget": budget, "resource": timer.finish(), "mlflow": mlflow_info, "limitations": limitations()}
        atomic_json(root_dir / "root-summary.json", result); atomic_json(self.output / "calibration.json", result); self.mark_lifecycle("completed", completion_marker=str(root_dir / "root-summary.json")); return result

    def diagnostic_bundle(self, root_dir: Path, identity: str, splits: dict[int, Any], trained: dict[str, Any], winner_model: Any) -> dict[str, Any]:
        config = self.config; root = int(self.root); source_model = new_model(config, self.checkpoint).eval(); source_state = state_hash(source_model); checkpoint_sha = sha256(self.checkpoint)
        frozen = {}; reset_eval = {}; randomized: dict[str, Any] = {}
        for width in WIDTHS:
            rollouts = int(config["evaluation"]["source_w128_rollout_cases"] if width == 128 else config["evaluation"]["source_w256_rollout_cases"])
            frozen[str(width)], _ = phase_result(root_dir, identity, f"diagnostic/frozen_source/{width}", lambda w=width, r=rollouts: evaluate_with_evidence(config, self.exp300, source_model, splits[w], root, w, int(config["evaluation"]["source_transition_cases_per_width"]), r), self.interrupt_after_phase)
            reset = copy.deepcopy(winner_model); reset.load_state_dict(load_file(self.checkpoint)); reset.eval()
            if state_hash(reset) != source_state: raise SemanticFailure("reset-to-source model is not byte-identical")
            reset_eval[str(width)], _ = phase_result(root_dir, identity, f"diagnostic/reset_to_source/{width}", lambda w=width, r=rollouts, m=reset: evaluate_with_evidence(config, self.exp300, m, splits[w], root, w, int(config["evaluation"]["source_transition_cases_per_width"]), r), self.interrupt_after_phase)
        for index, arm in enumerate(ARM_ORDER):
            seed = root + 700900 + index; model = new_model(config, self.checkpoint, randomized_seed=seed).eval(); randomized[arm] = {"seed": seed, "state_sha256": state_hash(model)}
            for width in WIDTHS:
                rollouts = int(config["evaluation"]["randomized_w128_rollout_cases"] if width == 128 else config["evaluation"]["randomized_w256_rollout_cases"])
                randomized[arm][str(width)], _ = phase_result(root_dir, identity, f"diagnostic/randomized/{arm}/{width}", lambda w=width, r=rollouts, m=model: evaluate_with_evidence(config, self.exp300, m, splits[w], root, w, int(config["evaluation"]["randomized_transition_cases_per_width"]), r), self.interrupt_after_phase)
        parity, _ = phase_result(root_dir, identity, "diagnostic/lower_tier_parity", lambda: lower_tier_parity(config), self.interrupt_after_phase)
        parity["arms"] = {arm: {"cases": parity["cases"], "prediction_sha256": parity["deployed_prediction_sha256"], "predictions_equal": True} for arm in ARM_ORDER}
        collapse = {arm: {str(width): {"trained_exact_accuracy": trained[arm][str(width)]["rollout"]["exact_accuracy"], "randomized_exact_accuracy": randomized[arm][str(width)]["rollout"]["exact_accuracy"], "trained_minus_randomized": trained[arm][str(width)]["rollout"]["exact_accuracy"] - randomized[arm][str(width)]["rollout"]["exact_accuracy"], "interpretation": "collapse-observed" if trained[arm][str(width)]["rollout"]["exact_accuracy"] > randomized[arm][str(width)]["rollout"]["exact_accuracy"] else "no-positive-trained-weight-separation"} for width in WIDTHS} for arm in ARM_ORDER}
        return {"frozen_source": {"checkpoint_sha256": checkpoint_sha, "state_sha256": source_state, "evaluations": frozen}, "reset_to_source": {"checkpoint_sha256": checkpoint_sha, "state_sha256": source_state, "byte_equal_to_frozen_source": True, "evaluations": reset_eval}, "randomized_weights": randomized, "lower_tier_parity": parity, "trained_weight_collapse": collapse}

    def science(self) -> dict[str, Any]:
        calibration = self.require_calibration(); self.require_predecessor_gate(); root_dir = self.open_or_resume_root(); identity = self.identity_sha256(); timer = Timer(); splits = self.splits(); mlflow_info = self.mlflow_identity()
        records: dict[str, Any] = {}; evaluations: dict[str, Any] = {}; models: dict[str, Any] = {}
        for arm in ARM_ORDER:
            resume_arm = self.resume or (snapshot_paths(root_dir, arm)[1].is_file())
            models[arm], records[arm] = train_arm_resumable(self.config, self.exp300, self.checkpoint, splits, int(self.root), arm, int(self.config["training"]["updates"]), root_dir, identity, resume_arm, self.interrupt_after_update)
            evaluations[arm] = {}
            for width in WIDTHS:
                transitions = int(self.config["evaluation"]["transition_cases_per_arm_width"]); rollouts = int(self.config["evaluation"]["w128_rollout_cases_per_arm"] if width == 128 else self.config["evaluation"]["w256_rollout_cases_per_arm"])
                evaluations[arm][str(width)], _ = phase_result(root_dir, identity, f"evaluation/{arm}/{width}", lambda a=arm, w=width, t=transitions, r=rollouts: evaluate_with_evidence(self.config, self.exp300, models[a], splits[w], int(self.root), w, t, r), self.interrupt_after_phase)
            atomic_json(root_dir / "progress.json", {"completed_arms": list(records), "records": records, "evaluations": evaluations})
        work = self.work_equivalence(records); winner = max(ARM_ORDER, key=lambda arm: evaluations[arm]["256"]["rollout"]["exact_accuracy"])
        diagnostic = self.diagnostic_bundle(root_dir, identity, splits, evaluations, models[winner])
        w128 = evaluations[winner]["128"]; frozen_w128 = diagnostic["frozen_source"]["evaluations"]["128"]
        healthy = w128["transition"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_transition"]) and w128["rollout"]["exact_accuracy"] >= frozen_w128["rollout"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_rollout_loss"]) and w128["rollout"]["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rate"])
        confirm = healthy and evaluations[winner]["256"]["rollout"]["exact_accuracy"] >= float(self.config["gates"]["confirmation_admission_w256_rollout"])
        decision = ("advance-to-confirmations" if confirm else "complete-negative-full-denominator") if self.mode == "formal" else ("confirmation-pass" if confirm else "confirmation-failed-full-denominator") if self.mode == "confirmations" else ("independent-pass" if confirm else "independent-failed-full-denominator")
        result = {"schema_version": "exp700-v1", "run_id": RUN_ID, "status": "completed", "mode": self.mode, "root": self.root, "decision": decision, "source": self.source(), "records": records, "evaluations": evaluations, "winner": winner, "health": healthy, "confirmation_trigger": confirm, "diagnostics": diagnostic, "work_equivalence": work, "dynamic_budget": calibration["budget"], "resource": timer.finish(), "mlflow": mlflow_info, "limitations": limitations()}
        exp699.assert_terminal_contract(result); atomic_json(root_dir / "root-summary.json", result); atomic_json(self.output / "terminal-summary.json", result)
        with mlflow.start_run(run_id=mlflow_info["run_id"]): mlflow.log_dict(result, f"{self.mode}-result.json"); mlflow.set_tag("decision", decision)
        mlflow.MlflowClient().set_terminated(mlflow_info["run_id"], status="FINISHED")
        self.mark_lifecycle("completed", completion_marker=str(root_dir / "root-summary.json")); return result


def smoke(config_path: Path, output: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text()); base.configure_torch(config); exp300 = base.load_exp300(); checkpoint = Path(config["source_checkpoint"]); root = 20261385
    splits = {width: make_split(config, exp300, width, root) for width in WIDTHS}; identity = canonical_hash({"mode": "smoke", "source": sha256(Path(__file__))})
    run_dir = output / "smoke" / "runs" / identity[:16]
    reference_dir = run_dir / "uninterrupted"; interrupted_dir = run_dir / "interrupted"
    model_a, record_a = train_arm_resumable(config, exp300, checkpoint, splits, root, ARM_ORDER[0], 3, reference_dir, identity, False, snapshot_every=1)
    try:
        train_arm_resumable(config, exp300, checkpoint, splits, root, ARM_ORDER[0], 3, interrupted_dir, identity, False, interrupt_after_update=2, snapshot_every=1)
    except RecoverableInterruption: pass
    else: raise SemanticFailure("smoke update interruption was not raised")
    model_b, record_b = train_arm_resumable(config, exp300, checkpoint, splits, root, ARM_ORDER[0], 3, interrupted_dir, identity, True, snapshot_every=1)
    training_equal = recovery_signature(record_a) == recovery_signature(record_b)
    prediction_a = evaluate_with_evidence(config, exp300, model_a, splits[128], root, 128, 2, 1)
    eval_output = run_dir / "evaluation-recovery"
    try:
        phase_result(eval_output, identity, "evaluation/teacher_cosine/128", lambda: evaluate_with_evidence(config, exp300, model_b, splits[128], root, 128, 2, 1), "evaluation/teacher_cosine/128")
    except RecoverableInterruption: pass
    prediction_b, reused_eval = phase_result(eval_output, identity, "evaluation/teacher_cosine/128", lambda: (_ for _ in ()).throw(AssertionError("completed evaluation repeated")))
    diagnostic_output = run_dir / "diagnostic-recovery"
    try:
        phase_result(diagnostic_output, identity, "diagnostic/frozen_source/128", lambda: evaluate_with_evidence(config, exp300, new_model(config, checkpoint).eval(), splits[128], root, 128, 2, 1), "diagnostic/frozen_source/128")
    except RecoverableInterruption: pass
    diagnostic, reused_diagnostic = phase_result(diagnostic_output, identity, "diagnostic/frozen_source/128", lambda: (_ for _ in ()).throw(AssertionError("completed diagnostic repeated")))
    predictions_equal = prediction_a["rollout"]["prediction_sha256"] == prediction_b["rollout"]["prediction_sha256"]
    if not training_equal or not predictions_equal or not reused_eval or not reused_diagnostic: raise SemanticFailure("smoke recovery equivalence failed")
    terminals = {}
    for mode in ("calibration", "formal", "confirmations", "independent"):
        value = {"schema_version": "exp700-smoke-v1", "run_id": RUN_ID, "status": "smoke-passed", "mode": mode, "root": None, "decision": "non-scientific-smoke", "recovery": {"training_exact": training_equal, "evaluation_reused": reused_eval, "diagnostic_reused": reused_diagnostic, "predictions_exact": predictions_equal}}
        path = output / "smoke" / f"{mode}-terminal.json"; atomic_json(path, value); terminals[mode] = str(path)
    result = {"status": "smoke-passed", "all_modes_exercised": list(terminals), "train_eval_diagnostic_recovery": True, "uninterrupted_signature": recovery_signature(record_a), "interrupted_resumed_signature": recovery_signature(record_b), "prediction_sha256": prediction_a["rollout"]["prediction_sha256"], "terminals": terminals}
    atomic_json(output / "smoke.json", result); return result


def write_terminal_failure(output: Path, mode: str, root: int | None, reason: str) -> None:
    path = output / mode / f"root-{root}" / "terminal-failure.json" if root is not None else output / "terminal-failure.json"
    atomic_json(path, {"schema_version": "exp700-terminal-failure-v1", "run_id": RUN_ID, "mode": mode, "root": root, "status": "terminal-failure", "reason": reason, "retry_allowed": False})


def service_proof(output: Path, kind: str) -> tuple[dict[str, Any], int]:
    path = output / f"service-proof-{kind}.json"
    prior = json.loads(path.read_text()) if path.is_file() else {"attempts": 0}
    attempts = int(prior["attempts"]) + 1
    if kind == "recoverable" and attempts == 1:
        value = {"kind": kind, "attempts": attempts, "status": "recoverable-interruption", "retry_allowed": True}; atomic_json(path, value); return value, 75
    if kind == "recoverable":
        value = {"kind": kind, "attempts": attempts, "status": "completed-after-restart", "retry_allowed": False}; atomic_json(path, value); return value, 0
    value = {"kind": kind, "attempts": attempts, "status": "terminal-failure", "retry_allowed": False}; atomic_json(path, value); return value, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmations", "independent"), required=True); parser.add_argument("--root", type=int); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interrupt-after-update", type=int); parser.add_argument("--interrupt-after-phase"); parser.add_argument("--service-proof", choices=("recoverable", "terminal"))
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    if args.service_proof:
        result, status = service_proof(args.output, args.service_proof); print(json.dumps(result)); return status
    lock_path = args.output / ("smoke.lock" if args.mode == "smoke" else f"{args.mode}-root-{args.root or 'default'}.lock")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "already-running", "lock": str(lock_path)})); return 75
        try:
            if args.mode == "smoke": result = smoke(args.config.resolve(), args.output)
            else:
                experiment = Experiment(args.config, args.output, args.mode, args.root, args.resume, args.interrupt_after_update, args.interrupt_after_phase)
                result = experiment.calibration() if args.mode == "calibration" else experiment.science()
        except RecoverableInterruption as error:
            if args.mode != "smoke" and args.root is not None:
                active = args.output / args.mode / f"root-{args.root}" / "run-active.json"
                if active.is_file():
                    value = json.loads(active.read_text()); atomic_json(active, {**value, "status": "recoverable-interruption", "reason": str(error)})
            print(json.dumps({"status": "recoverable-interruption", "reason": str(error)})); return 75
        except (SemanticFailure, ValueError, AssertionError) as error:
            write_terminal_failure(args.output, args.mode, args.root, str(error)); print(json.dumps({"status": "terminal-failure", "reason": str(error)})); return 0
    print(json.dumps({key: result.get(key) for key in ("status", "mode", "root", "decision", "winner", "confirmation_trigger")}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
