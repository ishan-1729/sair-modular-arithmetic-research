"""Generate and check the closed CUDA-portability experiment inventory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import defaultdict, deque
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/cuda_portability.yaml"

TENSOR_FACTORIES = frozenset(
    {
        "arange",
        "as_tensor",
        "empty",
        "empty_like",
        "from_numpy",
        "full",
        "full_like",
        "linspace",
        "ones",
        "ones_like",
        "rand",
        "randint",
        "randn",
        "randperm",
        "tensor",
        "zeros",
        "zeros_like",
    }
)
HOST_BOUNDARIES = frozenset({"cpu", "item", "numpy", "tolist"})
DEVICE_BOUNDARIES = frozenset({"cuda", "to"})
CHECKPOINT_CALLS = frozenset(
    {"load", "load_file", "load_state_dict", "save", "save_file", "state_dict"}
)
OPTIMIZER_CALLS = frozenset(
    {"Adadelta", "Adagrad", "Adam", "AdamW", "RMSprop", "SGD", "load_state_dict"}
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class ModuleScanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.imports: set[str] = set()
        self.hotspots: list[dict[str, Any]] = []
        self.references_torch = False

    @property
    def symbol(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)
            if alias.name == "torch" or alias.name.startswith("torch."):
                self.references_torch = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        if module:
            self.imports.add(module)
        for alias in node.names:
            if alias.name != "*":
                separator = "" if module.endswith(".") else "."
                self.imports.add(f"{module}{separator}{alias.name}")
        if node.module:
            if node.module == "torch" or node.module.startswith("torch."):
                self.references_torch = True
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in {"torch", "Tensor"}:
            self.references_torch = True

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        leaf = None if name is None else name.rsplit(".", 1)[-1]
        kind: str | None = None
        if name and (name.startswith("torch.") or name.startswith("Tensor.")):
            self.references_torch = True
        if name and name.startswith("torch.") and leaf in TENSOR_FACTORIES:
            kind = "tensor-factory"
        elif leaf in HOST_BOUNDARIES:
            kind = "host-boundary"
        elif leaf in DEVICE_BOUNDARIES:
            kind = "device-boundary"
        elif leaf in CHECKPOINT_CALLS:
            kind = "checkpoint"
        elif leaf in OPTIMIZER_CALLS or (name and ".optim." in name):
            kind = "optimizer"
        if kind:
            self.hotspots.append(
                {
                    "line": int(node.lineno),
                    "symbol": self.symbol,
                    "operation": name or leaf,
                    "kind": kind,
                }
            )
        self.generic_visit(node)


def scan_exclusion_patterns(config: dict[str, Any]) -> tuple[str, ...]:
    patterns: list[str] = []
    for exclusion in config["scan"].get("exclude", []):
        pattern = exclusion if isinstance(exclusion, str) else exclusion["pattern"]
        patterns.append(str(pattern))
    return tuple(patterns)


def scan_path_is_excluded(path: Path, config: dict[str, Any]) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    return any(
        fnmatchcase(relative, pattern)
        for pattern in scan_exclusion_patterns(config)
    )


def discover_python(config: dict[str, Any]) -> list[Path]:
    included: set[Path] = set()
    for pattern in config["scan"]["include"]:
        included.update(
            path
            for path in ROOT.glob(pattern)
            if path.is_file() and not scan_path_is_excluded(path, config)
        )
    return sorted(included, key=lambda path: path.relative_to(ROOT).as_posix())


def discover_roots(config: dict[str, Any]) -> list[Path]:
    roots: set[Path] = set()
    for pattern in config["scan"]["roots"]:
        roots.update(
            path
            for path in ROOT.glob(pattern)
            if path.is_file() and not scan_path_is_excluded(path, config)
        )
    return sorted(roots, key=lambda path: path.relative_to(ROOT).as_posix())


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_import(
    imported: str, importer: Path, modules: dict[str, Path]
) -> Path | None:
    candidate = imported
    if imported.startswith("."):
        level = len(imported) - len(imported.lstrip("."))
        suffix = imported[level:]
        importer_parts = module_name(importer).split(".")
        package_parts = (
            importer_parts if importer.name == "__init__.py" else importer_parts[:-1]
        )
        if level > len(package_parts) + 1:
            return None
        keep = len(package_parts) - (level - 1)
        candidate = ".".join(package_parts[:keep] + ([suffix] if suffix else []))
    while candidate:
        if candidate in modules:
            return modules[candidate]
        candidate = candidate.rpartition(".")[0]
    return None


def scan_module(path: Path) -> ModuleScanner:
    scanner = ModuleScanner()
    scanner.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    scanner.hotspots.sort(
        key=lambda row: (row["line"], row["kind"], row["operation"])
    )
    return scanner


def reverse_reachability(
    paths: list[Path], roots: list[Path], scans: dict[Path, ModuleScanner]
) -> tuple[dict[Path, list[str]], dict[Path, list[str]]]:
    modules = {module_name(path): path for path in paths}
    edges: dict[Path, set[Path]] = defaultdict(set)
    parents: dict[Path, set[str]] = defaultdict(set)
    for importer, scan in scans.items():
        for imported in scan.imports:
            resolved = resolve_import(imported, importer, modules)
            if resolved is not None and resolved != importer:
                edges[importer].add(resolved)
                parents[resolved].add(importer.relative_to(ROOT).as_posix())
    reached_by: dict[Path, set[str]] = defaultdict(set)
    for root in roots:
        root_name = root.relative_to(ROOT).as_posix()
        queue = deque([root])
        seen = {root}
        while queue:
            current = queue.popleft()
            reached_by[current].add(root_name)
            for child in edges.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
    return (
        {path: sorted(values) for path, values in reached_by.items()},
        {path: sorted(values) for path, values in parents.items()},
    )


def family_for(relative: str) -> str:
    parts = Path(relative).parts
    if len(parts) > 1 and parts[0] == "experiments":
        return parts[1]
    return "experiments"


def durable_registry_index(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load hash-bound lifecycle evidence from the durable experiment registry."""
    registry_path = ROOT / config["lifecycle_policy"]["registry"]
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for raw in registry["runs"]:
        route: dict[str, Any] = {
            "run_id": str(raw["run_id"]),
            "status": str(raw["status"]),
        }
        artifacts: dict[str, dict[str, str]] = {}
        identity_mismatches: list[dict[str, str]] = []

        def record_artifact(field: str, relative: str, expected: str | None) -> None:
            path = ROOT / relative
            if not path.is_file():
                return
            actual = sha256(path)
            if expected is not None and actual != expected:
                identity_mismatches.append(
                    {
                        "field": field,
                        "path": relative,
                        "registry_sha256": expected,
                        "current_sha256": actual,
                    }
                )
                return
            artifacts[relative] = {"field": field, "sha256": actual}

        for field in (
            "config",
            "run_record",
            "calibration_artifact",
            "environment",
            "executed_config",
            "mlflow_receipt",
        ):
            relative = raw.get(field)
            if isinstance(relative, str):
                record_artifact(field, relative, None)

        def find_declared_artifacts(value: Any, field: str = "evidence") -> None:
            if isinstance(value, dict):
                path_value = value.get("path")
                hash_value = value.get("sha256")
                if (
                    isinstance(path_value, str)
                    and isinstance(hash_value, str)
                    and path_value.startswith(("experiments/runs/", "evals/results/"))
                ):
                    record_artifact(field, path_value, hash_value)
                for key, child in value.items():
                    find_declared_artifacts(child, str(key))
            elif isinstance(value, list):
                for child in value:
                    find_declared_artifacts(child, field)

        find_declared_artifacts(raw)
        route["durable_artifacts"] = [
            {"path": path, **identity}
            for path, identity in sorted(artifacts.items())
        ]
        route["identity_mismatches"] = identity_mismatches
        route["artifact_identity_complete"] = bool(route["durable_artifacts"])
        index[route["run_id"]] = route
    return index


def route_ids_for_path(
    relative: str, registry: dict[str, dict[str, Any]]
) -> list[str]:
    """Resolve one source path to exact durable routes without a blanket default."""
    run_match = re.match(r"^experiments/runs/([^/]+)/", relative)
    if run_match and run_match.group(1) in registry:
        return [run_match.group(1)]

    prefixes: set[str] = set()
    for match in re.finditer(r"(?:^|/)exp_(\d{3})(?:_|/)", relative):
        prefixes.add(f"exp-{int(match.group(1)):03d}-")
    for match in re.finditer(r"(?:^|/)candidate_(\d{3})(?:_|/)", relative):
        prefixes.add(f"candidate-{int(match.group(1)):03d}-")
    return sorted(
        run_id
        for run_id in registry
        if any(run_id.startswith(prefix) for prefix in prefixes)
    )


def lifecycle_for_path(
    relative: str,
    reachability_roots: list[str],
    registry: dict[str, dict[str, Any]],
    config: dict[str, Any],
    dynamic_consumer_ids: list[str] | None = None,
) -> dict[str, Any]:
    direct_ids = route_ids_for_path(relative, registry)
    consumer_ids = sorted(
        set(dynamic_consumer_ids or [])
        | {
            run_id
            for root in reachability_roots
            for run_id in route_ids_for_path(root, registry)
        }
    )
    terminal = set(config["lifecycle_policy"]["terminal_statuses"])
    active = set(config["lifecycle_policy"]["active_statuses"])
    all_ids = sorted(set(direct_ids) | set(consumer_ids))
    routes = [registry[run_id] for run_id in all_ids]
    active_consumers = [
        registry[run_id]
        for run_id in consumer_ids
        if registry[run_id]["status"] in active
    ]
    return {
        "direct_route_ids": direct_ids,
        "consumer_route_ids": consumer_ids,
        "routes": routes,
        "active_consumers": active_consumers,
        "all_terminal": bool(routes)
        and all(route["status"] in terminal for route in routes),
        "all_artifact_identities_complete": bool(routes)
        and all(route["artifact_identity_complete"] for route in routes),
    }


def authenticate_exp649_boundary(config: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the frozen CPU stochastic semantics that CUDA would change."""
    policy = config["lifecycle_policy"]["exp649_cpu_boundary"]
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("runner", "runner_sha256"),
        ("seed_retirement_manifest", "seed_retirement_sha256"),
        ("run_record", "run_record_sha256"),
    ):
        actual = sha256(ROOT / policy[path_key])
        if actual != policy[hash_key]:
            raise ValueError(
                f"EXP-649 {path_key} identity changed: expected "
                f"{policy[hash_key]}, got {actual}"
            )

    dynamic_sources: list[dict[str, str]] = []
    for source in policy.get("dynamic_sources", []):
        actual = sha256(ROOT / source["path"])
        if actual != source["sha256"]:
            raise ValueError(
                f"EXP-649 dynamic source identity changed for {source['path']}: "
                f"expected {source['sha256']}, got {actual}"
            )
        dynamic_sources.append(dict(source))

    frozen = yaml.safe_load((ROOT / policy["config"]).read_text(encoding="utf-8"))
    record = json.loads((ROOT / policy["run_record"]).read_text(encoding="utf-8"))
    noise = frozen["noise"]
    semantic = record["semantic_state"]
    decision = record["decision"]
    expected_reset_points = [
        "before_warmup",
        "after_warmup",
        "before_each_replicate",
        "before_exact_rerun",
    ]
    expected_traversal = (
        "multiplier_bit_msb_to_lsb_then_injection_0_to_16_then_case_channel_position"
    )
    checks = {
        "distribution": noise.get("distribution") == "iid_cpu_float32_gaussian",
        "coordinate_traversal": noise.get("coordinate_traversal")
        == expected_traversal,
        "reset_points": noise.get("reset_points") == expected_reset_points,
        "registry_lifecycle": semantic.get("registry_lifecycle") == "planned",
        "formal_authorized": semantic.get("formal_authorized") is False,
        "formal_root_opened": semantic.get("formal_root_opened") is False,
        "runtime_gate_passed": decision.get("runtime_gate_passed") is False,
        "outcome": decision.get("outcome") == "blocked-from-exp-649-formal",
    }
    if not all(checks.values()):
        raise ValueError(f"EXP-649 CPU boundary contract changed: {checks}")
    return {
        "run_id": "exp-649-stochastic-trajectory-breadth",
        "authenticated": True,
        "hashes": {
            key: policy[key]
            for key in (
                "config_sha256",
                "runner_sha256",
                "seed_retirement_sha256",
                "run_record_sha256",
            )
        },
        "noise_distribution": noise["distribution"],
        "coordinate_traversal": noise["coordinate_traversal"],
        "reset_points": noise["reset_points"],
        "runtime_gate_passed": False,
        "formal_authorized": False,
        "formal_root_opened": False,
        "dynamic_sources": dynamic_sources,
    }


def classify(
    relative: str,
    scan: ModuleScanner,
    config: dict[str, Any],
    lifecycle: dict[str, Any],
    exp649_boundary: dict[str, Any],
) -> tuple[str, str, str]:
    hotspot_counts = {
        kind: sum(row["kind"] == kind for row in scan.hotspots)
        for kind in sorted({row["kind"] for row in scan.hotspots})
    }
    imports = sorted(scan.imports)
    static_evidence = (
        f"imports={imports[:8]!r}; hotspot_counts={hotspot_counts!r}"
    )
    adaptation = config.get("adapted_sources", {}).get(relative)
    cpu_boundary = config.get("active_cpu_boundaries", {}).get(relative)
    if relative in set(config["ported_paths"]):
        return (
            "device-native-unvalidated",
            "Explicit device/runtime adapter is present; physical CUDA execution "
            f"is still unvalidated; {static_evidence}.",
            "model tensors follow the selected device; exact labels and evidence serialization remain CPU boundaries",
        )
    if adaptation and adaptation["coverage"] == "full-active-route":
        return (
            "device-native-unvalidated",
            f"Frozen source is covered by {adaptation['adapter']} for symbols "
            f"{adaptation['symbols']!r}; physical CUDA remains unvalidated; "
            f"{static_evidence}.",
            "frozen source stays immutable; adapter owns input/model device placement",
        )
    if cpu_boundary:
        return (
            "intentionally-cpu-boundary",
            f"{cpu_boundary['reason']} Active symbols are "
            f"{cpu_boundary['symbols']!r}; {static_evidence}.",
            str(cpu_boundary["responsibility"]),
        )
    active_route_ids = {
        route["run_id"] for route in lifecycle["active_consumers"]
    }
    route_ids = {route["run_id"] for route in lifecycle["routes"]}
    exp649_run_id = exp649_boundary["run_id"]
    if exp649_run_id in active_route_ids or exp649_run_id in route_ids:
        return (
            "intentionally-cpu-boundary",
            "The authenticated EXP-649 route requires iid CPU float32 Gaussian "
            "noise with frozen coordinate traversal and generator reset points; "
            "moving this source or its shared dependency to CUDA would change the "
            f"experiment contract; hashes={exp649_boundary['hashes']!r}; "
            f"{static_evidence}.",
            "retain the EXP-649 route and shared dependencies on CPU unless a new experiment is preregistered",
        )
    if (
        lifecycle["all_terminal"]
        and lifecycle["all_artifact_identities_complete"]
        and not lifecycle["active_consumers"]
    ):
        identities = [
            {
                "run_id": route["run_id"],
                "status": route["status"],
                "artifact": route["durable_artifacts"][0],
            }
            for route in lifecycle["routes"]
        ]
        return (
            "archived-frozen-route",
            "All exact registry routes consuming this source are terminal and have "
            f"authenticated durable run records: {identities!r}; {static_evidence}.",
            "preserve historical source and use its durable result; do not port or rerun it as part of the active CUDA route",
        )
    if adaptation:
        return (
            "adapter-required",
            f"Partial adapter {adaptation['adapter']} covers only "
            f"{adaptation['symbols']!r} for {adaptation['consumer']}; remaining "
            f"symbols still require classification; {static_evidence}.",
            "selected representative route is adapted; remaining module routes are pending",
        )
    if scan.references_torch:
        return (
            "adapter-required",
            "Static AST scan finds torch use outside the approved explicit-device "
            f"adapter set; {static_evidence}.",
            "requires representative family adapter/parity evidence before CUDA execution",
        )
    return (
        "intentionally-cpu-boundary",
        f"{config['classification_policy']['no_torch_reason']} {static_evidence}.",
        "CPU orchestration/exact-label/evidence responsibility only",
    )


def audit_hot_loop(
    path: Path,
    function_names: set[str],
    forbidden_calls: set[str],
    forbidden_factories: set[str],
) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[dict[str, Any]] = []
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = defaultdict(list)
    for definition in ast.walk(tree):
        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[definition.name].append(definition)

    def boundary(call: ast.Call) -> tuple[str, str] | None:
        name = dotted_name(call.func)
        if not name:
            return None
        leaf = name.rsplit(".", 1)[-1]
        if leaf in forbidden_calls:
            return name, "transfer-call"
        if leaf in forbidden_factories:
            return name, "python-backed-tensor-factory"
        return None

    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if function.name not in function_names:
            continue
        for loop in ast.walk(function):
            if not isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
                continue
            for node in ast.walk(loop):
                if isinstance(node, ast.Call):
                    name = dotted_name(node.func)
                    direct = boundary(node)
                    if direct:
                        violations.append(
                            {
                                "function": function.name,
                                "line": int(node.lineno),
                                "call": direct[0],
                                "kind": direct[1],
                                "via": None,
                            }
                        )
                    if not name:
                        continue
                    helper_name = name.rsplit(".", 1)[-1]
                    for helper in definitions.get(helper_name, []):
                        if helper is function:
                            continue
                        for helper_node in ast.walk(helper):
                            if not isinstance(helper_node, ast.Call):
                                continue
                            nested = boundary(helper_node)
                            if nested:
                                violations.append(
                                    {
                                        "function": function.name,
                                        "line": int(node.lineno),
                                        "call": nested[0],
                                        "kind": nested[1],
                                        "via": name,
                                        "helper_line": int(helper_node.lineno),
                                    }
                                )
    return violations


def generate(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registry = durable_registry_index(config)
    exp649_boundary = authenticate_exp649_boundary(config)
    paths = discover_python(config)
    roots = discover_roots(config)
    scans = {path: scan_module(path) for path in paths}
    reachability, import_parents = reverse_reachability(paths, roots, scans)
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        scan = scans[path]
        path_reachability = reachability.get(
            path, ["mechanical-all-experiment-python"]
        )
        lifecycle = lifecycle_for_path(
            relative,
            path_reachability,
            registry,
            config,
            (
                [exp649_boundary["run_id"]]
                if relative
                in {
                    source["path"]
                    for source in exp649_boundary["dynamic_sources"]
                }
                else []
            ),
        )
        classification, evidence, responsibility = classify(
            relative,
            scan,
            config,
            lifecycle,
            exp649_boundary,
        )
        frozen = relative.startswith("experiments/exp_")
        rows.append(
            {
                "path": relative,
                "module": module_name(path),
                "family": family_for(relative),
                "reachability_roots": path_reachability,
                "local_import_parents": import_parents.get(path, []),
                "classification": classification,
                "adapter": config.get("adapted_sources", {}).get(relative),
                "active_cpu_boundary": config.get("active_cpu_boundaries", {}).get(
                    relative
                ),
                "durable_lifecycle": lifecycle,
                "exp649_cpu_boundary": (
                    exp649_boundary
                    if exp649_boundary["run_id"]
                    in {
                        route["run_id"]
                        for route in lifecycle["routes"]
                        + lifecycle["active_consumers"]
                    }
                    else None
                ),
                "evidence": evidence,
                "responsibility": responsibility,
                "frozen_identity_constraint": (
                    str(config["classification_policy"]["frozen_reason"])
                    if frozen or classification == "archived-frozen-route"
                    else None
                ),
                "smoke_command": (
                    config.get("adapted_sources", {})
                    .get(relative, {})
                    .get("smoke_command")
                    or (
                        "python -m pytest -q tests/test_exp699_cuda_hardware_replay.py"
                        if relative
                        == "experiments/cloud_benchmarks/exp699_cuda_replay/run.py"
                        else None
                    )
                ),
                "source_sha256": sha256(path),
                "torch_referenced": scan.references_torch,
                "hotspots": scan.hotspots,
            }
        )

    hot_loop_audit: list[dict[str, Any]] = []
    for spec in config["recurrent_hot_loops"]:
        path = ROOT / spec["path"]
        forbidden = set(spec["forbidden_calls"])
        forbidden_factories = set(spec.get("forbidden_factories", []))
        violations = audit_hot_loop(
            path,
            set(spec["functions"]),
            forbidden,
            forbidden_factories,
        )
        hot_loop_audit.append(
            {
                "path": spec["path"],
                "functions": sorted(spec["functions"]),
                "forbidden_calls": sorted(forbidden),
                "forbidden_factories": sorted(forbidden_factories),
                "violations": violations,
                "passed": not violations,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "cuda-portability-inventory-v1",
        "readiness": str(config["readiness"]),
        "source": {
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": sha256(config_path),
            "registry": config["lifecycle_policy"]["registry"],
            "registry_sha256": sha256(
                ROOT / config["lifecycle_policy"]["registry"]
            ),
            "python_file_count": len(paths),
            "root_count": len(roots),
            "tree_sha256": sha256_bytes(
                "\n".join(f"{row['path']} {row['source_sha256']}" for row in rows).encode()
            ),
        },
        "closure": {
            "all_discovered_files_inventoried": len(rows) == len(paths),
            "all_rows_classified": all(row["classification"] for row in rows),
            "all_classifications_evidenced": all(row["evidence"] for row in rows),
            "no_adapter_required": all(
                row["classification"] != "adapter-required" for row in rows
            ),
            "recurrent_hot_loops_pass": all(row["passed"] for row in hot_loop_audit),
        },
        "summary": {
            "classifications": dict(
                sorted(
                    {
                        name: sum(row["classification"] == name for row in rows)
                        for name in {row["classification"] for row in rows}
                    }.items()
                )
            ),
            "hotspots": dict(
                sorted(
                    {
                        name: sum(
                            hotspot["kind"] == name
                            for row in rows
                            for hotspot in row["hotspots"]
                        )
                        for name in {
                            hotspot["kind"]
                            for row in rows
                            for hotspot in row["hotspots"]
                        }
                    }.items()
                )
            ),
        },
        "hot_loop_audit": hot_loop_audit,
        "modules": rows,
    }
    return payload


def encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["scan"]["manifest"]
    rendered = encode(generate(config_path))
    if args.write:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(rendered, encoding="utf-8")
        print(manifest_path.relative_to(ROOT).as_posix())
        return 0
    if not manifest_path.is_file():
        raise SystemExit(f"missing CUDA portability manifest: {manifest_path}")
    current = manifest_path.read_text(encoding="utf-8")
    if current != rendered:
        raise SystemExit(
            "CUDA portability inventory is stale; run "
            "python -m experiments.cuda_portability.inventory --write"
        )
    payload = json.loads(current)
    if not all(payload["closure"].values()):
        raise SystemExit(f"CUDA portability inventory is open: {payload['closure']}")
    print(json.dumps({"status": "closed", **payload["source"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
