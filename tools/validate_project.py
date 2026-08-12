"""Validate the repository's knowledge, dataset, provenance, and run contracts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import yaml
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
CANDIDATE_SHA256 = {
    "manifest.json": "dbeb95ed3b085aa2b338ec57041d343ec924ca72f4b3c7fd3094d6cd215a0c91",
    "model.py": "082ca41d86c758254dd0e75841c1b1d8df94bbf3280d539389782deaf4857d9d",
    "weights.safetensors": "a7d25ee4eed49c5f625bc8369447be9f51ff1be54766631d8b96ab725860553e",
}
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
ARXIV_PATTERNS = (
    re.compile(
        r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/"
        r"((?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?)"
        r"(?:\.pdf)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\barXiv\s*:\s*((?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?)",
        re.IGNORECASE,
    ),
)
REQUIRED_DIRS = (
    "knowledge",
    "configs",
    "data",
    "evals",
    "experiments",
    "pocs",
    "research",
    "submission",
    "sources",
    "tests",
    "tools",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_research_paper_ids(catalog: dict[str, Any]) -> set[str]:
    """Derive the selected paper IDs directly from the source catalog and citations."""

    if catalog.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    collections = catalog.get("collections")
    resolved = catalog.get("resolved_arxiv_papers")
    web_papers = catalog.get("web_papers")
    if not isinstance(collections, list) or not isinstance(resolved, list) or not isinstance(web_papers, list):
        raise ValueError("collection arrays are malformed")

    def base(identifier: str) -> str:
        return re.sub(r"v\d+$", "", identifier, flags=re.IGNORECASE)

    identifiers: set[str] = set()
    for collection in collections:
        if not isinstance(collection, dict):
            raise ValueError("collection entry is not an object")
        for identifier in collection.get("explicit_arxiv_ids", []):
            identifiers.add(f"arxiv:{base(str(identifier))}")
        if not collection.get("include_arxiv_citations"):
            continue
        for relative in collection.get("source_files", []):
            path = (ROOT / str(relative)).resolve()
            try:
                path.relative_to(ROOT)
            except ValueError as exc:
                raise ValueError(f"catalog source escapes repository: {relative}") from exc
            text = path.read_text(encoding="utf-8")
            for pattern in ARXIV_PATTERNS:
                identifiers.update(
                    f"arxiv:{base(match.group(1))}"
                    for match in pattern.finditer(text)
                )
    for paper in resolved:
        if not isinstance(paper, dict) or not paper.get("arxiv_id"):
            raise ValueError("resolved_arxiv_papers entry is malformed")
        identifiers.add(f"arxiv:{base(str(paper['arxiv_id']))}")
    for paper in web_papers:
        if not isinstance(paper, dict) or not paper.get("id"):
            raise ValueError("web_papers entry is malformed")
        identifiers.add(f"web:{paper['id']}")
    return identifiers


def parse_frontmatter(path: Path) -> tuple[dict[str, Any] | None, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("frontmatter starts but has no closing delimiter") from exc
    metadata = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must parse to a mapping")
    return metadata, "\n".join(lines[end + 1 :])


def validate_knowledge() -> list[str]:
    errors: list[str] = []
    for path in sorted(KNOWLEDGE.rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            metadata, body = parse_frontmatter(path)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"{rel}: invalid frontmatter: {exc}")
            continue

        if path.name in {"index.md", "log.md"}:
            if path == KNOWLEDGE / "index.md":
                if not metadata or str(metadata.get("okf_version")) != "0.1":
                    errors.append(f"{rel}: root index must declare okf_version 0.1")
            elif metadata is not None:
                errors.append(f"{rel}: reserved index/log files must not have frontmatter")
            if path.name == "log.md" and not re.search(r"^## \d{4}-\d{2}-\d{2}$", body, re.MULTILINE):
                errors.append(f"{rel}: log must contain an ISO YYYY-MM-DD heading")
        else:
            required = ("type", "title", "description", "status", "timestamp")
            if metadata is None:
                errors.append(f"{rel}: concept document is missing YAML frontmatter")
            else:
                for key in required:
                    if metadata.get(key) in (None, "", []):
                        errors.append(f"{rel}: frontmatter field {key!r} is required")

        for raw_target in LINK_RE.findall(body):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            resolved = (KNOWLEDGE / target.lstrip("/")) if target.startswith("/") else (path.parent / target)
            if not resolved.resolve().exists():
                errors.append(f"{rel}: broken local link {raw_target!r}")
    return errors


def validate_dataset() -> list[str]:
    errors: list[str] = []
    descriptor_path = ROOT / "data" / "croissant.json"
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"data/croissant.json: cannot parse: {exc}"]

    if descriptor.get("conformsTo") != "http://mlcommons.org/croissant/1.1":
        errors.append("data/croissant.json: expected Croissant 1.1 conformance")
    if descriptor.get("@type") != "sc:Dataset" or not descriptor.get("name"):
        errors.append("data/croissant.json: missing Dataset type or name")
    distributions = descriptor.get("distribution")
    if not isinstance(distributions, list) or not distributions:
        return errors + ["data/croissant.json: distribution must be a non-empty list"]

    file_objects = [item for item in distributions if item.get("@type") == "cr:FileObject"]
    if len(file_objects) != 11:
        errors.append(f"data/croissant.json: expected 11 tier FileObjects, found {len(file_objects)}")
    for item in file_objects:
        rel_path = item.get("contentUrl")
        if not isinstance(rel_path, str):
            errors.append(f"data/croissant.json: FileObject {item.get('@id')} lacks contentUrl")
            continue
        path = ROOT / "data" / rel_path
        if not path.is_file():
            errors.append(f"data/croissant.json: missing distribution file data/{rel_path}")
            continue
        actual_hash = sha256(path)
        if actual_hash != item.get("sha256"):
            errors.append(f"data/{rel_path}: SHA-256 differs from Croissant metadata")
        if item.get("contentSize") != f"{path.stat().st_size} B":
            errors.append(f"data/{rel_path}: contentSize differs from Croissant metadata")

        tier = int(path.stem.split("_")[1])
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 100:
            errors.append(f"data/{rel_path}: expected 100 records, found {len(lines)}")
        for line_number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"data/{rel_path}:{line_number}: invalid JSON: {exc}")
                break
            if set(record) != {"a", "b", "p", "expected", "tier_id"}:
                errors.append(f"data/{rel_path}:{line_number}: unexpected record fields")
                break
            if record["tier_id"] != tier:
                errors.append(f"data/{rel_path}:{line_number}: tier_id mismatch")
                break
            for key in ("a", "b", "p", "expected"):
                value = record[key]
                if not isinstance(value, str) or not value.isdigit() or (len(value) > 1 and value[0] == "0"):
                    errors.append(f"data/{rel_path}:{line_number}: {key} is not canonical decimal")
                    break

    record_sets = descriptor.get("recordSet")
    if not isinstance(record_sets, list) or not record_sets:
        errors.append("data/croissant.json: recordSet must be a non-empty list")
    return errors


def validate_checksum_file() -> list[str]:
    errors: list[str] = []
    checksum_path = ROOT / "data" / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"data/checksums.sha256: cannot read: {exc}"]
    if len(lines) != 11:
        errors.append(f"data/checksums.sha256: expected 11 entries, found {len(lines)}")
    for line in lines:
        try:
            expected, rel = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"data/checksums.sha256: malformed line {line!r}")
            continue
        path = ROOT / "data" / rel.strip()
        if not path.is_file():
            errors.append(f"data/checksums.sha256: missing data/{rel.strip()}")
        elif sha256(path) != expected:
            errors.append(f"data/checksums.sha256: hash mismatch for data/{rel.strip()}")
    return errors


def validate_research_library() -> list[str]:
    """Validate the frozen PDF inventory without accessing the network."""

    errors: list[str] = []
    research = ROOT / "research"
    manifest_path = research / "manifest.json"
    sources_path = research / "sources.json"
    checksums_path = research / "checksums.sha256"
    verification_path = research / "verification.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"research/manifest.json: cannot parse: {exc}"]
    try:
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
        expected_ids = expected_research_paper_ids(sources)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"research/sources.json: cannot derive paper IDs: {exc}"]
    try:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"research/verification.json: cannot parse: {exc}"]
    try:
        checksum_lines = checksums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"research/checksums.sha256: cannot read: {exc}"]

    if manifest.get("schema_version") != 1:
        errors.append("research/manifest.json: schema_version must be 1")
    if not sources_path.is_file():
        errors.append("research/sources.json: missing source catalog")
    elif manifest.get("source_catalog_sha256") != sha256(sources_path):
        errors.append("research/manifest.json: source catalog hash is stale")
    papers = manifest.get("papers")
    if not isinstance(papers, list) or not papers:
        return errors + ["research/manifest.json: papers must be a non-empty list"]
    if manifest.get("sync_complete") is not True:
        errors.append("research/manifest.json: sync_complete must be true")
    selected_ids = manifest.get("selected_paper_ids")
    if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
        errors.append("research/manifest.json: selected_paper_ids must be a list of strings")
    elif set(selected_ids) != expected_ids:
        errors.append("research/manifest.json: selected IDs differ from source catalog")

    checksums: dict[str, str] = {}
    for line_number, line in enumerate(checksum_lines, start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"research/checksums.sha256:{line_number}: malformed line")
            continue
        relative = relative.strip()
        if relative in checksums:
            errors.append(f"research/checksums.sha256:{line_number}: duplicate path {relative}")
        checksums[relative] = expected

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            errors.append(f"research/manifest.json: entry {index} is not an object")
            continue
        identifier = paper.get("id")
        label = str(identifier or f"entry {index}")
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"research/manifest.json: entry {index} has no id")
        elif identifier in seen_ids:
            errors.append(f"research/manifest.json: duplicate id {identifier}")
        seen_ids.add(label)
        if paper.get("status") not in {"downloaded", "cached"}:
            errors.append(f"research/manifest.json: {label} is not successfully acquired")
        relative = paper.get("local_path")
        if not isinstance(relative, str) or not relative.startswith("research/papers/"):
            errors.append(f"research/manifest.json: {label} has unsafe local_path")
            continue
        path = (ROOT / relative).resolve()
        try:
            path.relative_to((research / "papers").resolve())
        except ValueError:
            errors.append(f"research/manifest.json: {label} escapes research/papers")
            continue
        if relative in seen_paths:
            errors.append(f"research/manifest.json: duplicate local_path {relative}")
        seen_paths.add(relative)
        if not path.is_file():
            errors.append(f"research/manifest.json: missing {relative}")
            continue
        with path.open("rb") as handle:
            if not handle.read(8).startswith(b"%PDF-"):
                errors.append(f"{relative}: missing PDF signature")
        actual_hash = sha256(path)
        if paper.get("sha256") != actual_hash:
            errors.append(f"{relative}: hash differs from manifest")
        if paper.get("bytes") != path.stat().st_size:
            errors.append(f"{relative}: byte count differs from manifest")
        if checksums.get(relative) != actual_hash:
            errors.append(f"{relative}: hash differs from research/checksums.sha256")

    if set(checksums) != seen_paths:
        errors.append("research/checksums.sha256: path coverage differs from manifest")
    if seen_ids != expected_ids:
        errors.append("research/manifest.json: paper IDs differ from source catalog")
    disk_paths = {
        path.relative_to(ROOT).as_posix()
        for path in (research / "papers").rglob("*.pdf")
    }
    if disk_paths != seen_paths:
        errors.append("research/papers: PDF set differs from manifest")

    if verification.get("schema_version") != 1:
        errors.append("research/verification.json: schema_version must be 1")
    if verification.get("manifest_sha256") != sha256(manifest_path):
        errors.append("research/verification.json: manifest hash is stale")
    if verification.get("checksums_sha256") != sha256(checksums_path):
        errors.append("research/verification.json: checksum-file hash is stale")
    verified = verification.get("papers")
    if not isinstance(verified, list):
        errors.append("research/verification.json: papers must be a list")
    else:
        verified_ids = {item.get("id") for item in verified if isinstance(item, dict)}
        if verified_ids != seen_ids:
            errors.append("research/verification.json: paper IDs differ from manifest")
        for item in verified:
            if not isinstance(item, dict) or not isinstance(item.get("page_count"), int) or item["page_count"] < 1:
                errors.append("research/verification.json: every paper needs a positive page_count")
                break
    return errors


def validate_json_instances(instance_glob: str, schema_path: Path) -> list[str]:
    errors: list[str] = []
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for path in sorted(ROOT.glob(instance_glob)):
        instance = json.loads(path.read_text(encoding="utf-8"))
        for issue in sorted(validator.iter_errors(instance), key=lambda error: list(error.path)):
            location = "/".join(str(part) for part in issue.path) or "<root>"
            errors.append(f"{path.relative_to(ROOT).as_posix()}:{location}: {issue.message}")
    return errors


def validate_upstream_snapshot() -> list[str]:
    path = ROOT / "sources" / "github" / "modular-arithmetic-challenge"
    if not path.is_dir():
        return ["missing pinned official challenge repository"]
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"cannot resolve official challenge commit: {exc}"]
    actual = result.stdout.strip()
    return [] if actual == CHALLENGE_COMMIT else [f"official challenge commit is {actual}, expected {CHALLENGE_COMMIT}"]


def validate_release_boundary() -> list[str]:
    errors: list[str] = []
    submission = ROOT / "submission"
    forbidden_names = {"tier_0.jsonl", "tier_1.jsonl", "tier_2.jsonl", "HF_TOKEN", ".env"}
    for path in submission.rglob("*"):
        if path.is_file() and (path.name in forbidden_names or "public_benchmark" in path.as_posix()):
            errors.append(f"submission boundary contains forbidden file: {path.relative_to(ROOT).as_posix()}")
    return errors


def validate_submission_artifact() -> list[str]:
    errors: list[str] = []
    submission = ROOT / "submission"
    required = {"manifest.json", "model.py", "weights.safetensors"}
    actual = {path.name for path in submission.iterdir() if path.is_file()}
    if actual != required:
        errors.append(
            "submission: expected exactly manifest.json, model.py, and "
            f"weights.safetensors; found {sorted(actual)}"
        )
    unexpected_dirs = [
        path.name
        for path in submission.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    ]
    if unexpected_dirs:
        errors.append(f"submission: unexpected directories {sorted(unexpected_dirs)}")

    manifest_path = submission / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"submission/manifest.json: cannot parse: {exc}")
        else:
            if manifest.get("entry_class") != "model.Tier2HornerModel":
                errors.append("submission/manifest.json: unexpected entry_class")
            if manifest.get("output_base") != "p":
                errors.append("submission/manifest.json: expected output_base 'p'")

    for name, expected_hash in CANDIDATE_SHA256.items():
        path = submission / name
        if path.is_file() and sha256(path) != expected_hash:
            errors.append(f"submission/{name}: staged research-candidate hash mismatch")
    clean_bytes = sum((submission / name).stat().st_size for name in required if (submission / name).is_file())
    if clean_bytes != 2_323_261:
        errors.append(f"submission: clean staged research artifact is {clean_bytes} bytes, expected 2323261")
    return errors


def collect_errors() -> list[str]:
    errors = [f"missing required directory: {name}" for name in REQUIRED_DIRS if not (ROOT / name).is_dir()]
    errors.extend(validate_knowledge())
    errors.extend(validate_dataset())
    errors.extend(validate_checksum_file())
    errors.extend(validate_research_library())
    errors.extend(validate_upstream_snapshot())
    errors.extend(validate_release_boundary())
    errors.extend(validate_submission_artifact())
    errors.extend(
        validate_json_instances(
            "evals/results/*/metrics.json",
            ROOT / "evals" / "schemas" / "eval-result.schema.json",
        )
    )
    errors.extend(
        validate_json_instances(
            "experiments/runs/*/run.json",
            ROOT / "experiments" / "schemas" / "run.schema.json",
        )
    )
    crate = ROOT / "ro-crate-metadata.json"
    if not crate.is_file():
        errors.append("missing ro-crate-metadata.json")
    else:
        try:
            graph = json.loads(crate.read_text(encoding="utf-8")).get("@graph")
            if not isinstance(graph, list) or not graph:
                errors.append("ro-crate-metadata.json: @graph must be non-empty")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"ro-crate-metadata.json: cannot parse: {exc}")
    return errors


def main() -> int:
    errors = collect_errors()
    if errors:
        print(f"Project validation failed with {len(errors)} issue(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    concept_count = sum(1 for path in KNOWLEDGE.rglob("*.md") if path.name not in {"index.md", "log.md"})
    print(f"Project validation passed: {concept_count} OKF concepts, 1,100 benchmark records, pinned upstream {CHALLENGE_COMMIT[:12]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
