"""Generate and check the closed CUDA-portability experiment inventory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict, deque
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


def discover_python(config: dict[str, Any]) -> list[Path]:
    included: set[Path] = set()
    for pattern in config["scan"]["include"]:
        included.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(included, key=lambda path: path.relative_to(ROOT).as_posix())


def discover_roots(config: dict[str, Any]) -> list[Path]:
    roots: set[Path] = set()
    for pattern in config["scan"]["roots"]:
        roots.update(path for path in ROOT.glob(pattern) if path.is_file())
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


def classify(
    relative: str,
    scan: ModuleScanner,
    config: dict[str, Any],
) -> tuple[str, str, str]:
    hotspot_counts = {
        kind: sum(row["kind"] == kind for row in scan.hotspots)
        for kind in sorted({row["kind"] for row in scan.hotspots})
    }
    imports = sorted(scan.imports)
    static_evidence = (
        f"imports={imports[:8]!r}; hotspot_counts={hotspot_counts!r}"
    )
    if relative in set(config["ported_paths"]):
        return (
            "device-native-unvalidated",
            "Explicit device/runtime adapter is present; physical CUDA execution "
            f"is still unvalidated; {static_evidence}.",
            "model tensors follow the selected device; exact labels and evidence serialization remain CPU boundaries",
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


def calls_in_for_loops(path: Path, function_names: set[str]) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[dict[str, Any]] = []
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
                    if name:
                        violations.append(
                            {
                                "function": function.name,
                                "line": int(node.lineno),
                                "call": name,
                            }
                        )
    return violations


def generate(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = discover_python(config)
    roots = discover_roots(config)
    scans = {path: scan_module(path) for path in paths}
    reachability, import_parents = reverse_reachability(paths, roots, scans)
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        scan = scans[path]
        classification, evidence, responsibility = classify(relative, scan, config)
        frozen = relative.startswith("experiments/exp_")
        rows.append(
            {
                "path": relative,
                "module": module_name(path),
                "family": family_for(relative),
                "reachability_roots": reachability.get(
                    path, ["mechanical-all-experiment-python"]
                ),
                "local_import_parents": import_parents.get(path, []),
                "classification": classification,
                "evidence": evidence,
                "responsibility": responsibility,
                "frozen_identity_constraint": (
                    str(config["classification_policy"]["frozen_reason"])
                    if frozen
                    else None
                ),
                "smoke_command": (
                    "python -m pytest -q tests/test_exp699_cuda_hardware_replay.py"
                    if relative
                    == "experiments/cloud_benchmarks/exp699_cuda_replay/run.py"
                    else None
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
        loop_calls = calls_in_for_loops(path, set(spec["functions"]))
        violations = [
            row for row in loop_calls if row["call"].rsplit(".", 1)[-1] in forbidden
        ]
        hot_loop_audit.append(
            {
                "path": spec["path"],
                "functions": sorted(spec["functions"]),
                "forbidden_calls": sorted(forbidden),
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
