"""Verify the downloaded research-paper library without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
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


def arxiv_base(identifier: str) -> str:
    return re.sub(r"v\d+$", "", identifier, flags=re.IGNORECASE)


def expected_paper_ids(root: Path, catalog: dict[str, Any]) -> set[str]:
    """Derive the exact selected ID set independently of the sync manifest."""

    if catalog.get("schema_version") != 1:
        raise ValueError("research/sources.json schema_version must be 1")
    collections = catalog.get("collections")
    resolved = catalog.get("resolved_arxiv_papers")
    web_papers = catalog.get("web_papers")
    if not isinstance(collections, list) or not isinstance(resolved, list) or not isinstance(web_papers, list):
        raise ValueError("research/sources.json collection arrays are malformed")
    identifiers: set[str] = set()
    for collection in collections:
        if not isinstance(collection, dict):
            raise ValueError("research/sources.json collection entry is not an object")
        for identifier in collection.get("explicit_arxiv_ids", []):
            identifiers.add(f"arxiv:{arxiv_base(str(identifier))}")
        if not collection.get("include_arxiv_citations"):
            continue
        for relative in collection.get("source_files", []):
            path = (root / str(relative)).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"catalog source escapes repository: {relative}") from exc
            text = path.read_text(encoding="utf-8")
            for pattern in ARXIV_PATTERNS:
                identifiers.update(
                    f"arxiv:{arxiv_base(match.group(1))}"
                    for match in pattern.finditer(text)
                )
    for paper in resolved:
        if not isinstance(paper, dict) or not paper.get("arxiv_id"):
            raise ValueError("resolved_arxiv_papers entry is malformed")
        identifiers.add(f"arxiv:{arxiv_base(str(paper['arxiv_id']))}")
    for paper in web_papers:
        if not isinstance(paper, dict) or not paper.get("id"):
            raise ValueError("web_papers entry is malformed")
        identifiers.add(f"web:{paper['id']}")
    return identifiers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Repository root (defaults to the root containing this script).",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Atomically write research/verification.json after a successful check.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_checksums(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"{path.name}:{line_number}: malformed checksum line")
            continue
        relative = relative.strip()
        if relative in entries:
            errors.append(f"{path.name}:{line_number}: duplicate path {relative}")
        entries[relative] = expected
    return entries, errors


def safe_repo_path(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve()
    papers = (root / "research" / "papers").resolve()
    try:
        candidate.relative_to(papers)
    except ValueError:
        return None
    return candidate


def verify(root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = root / "research" / "manifest.json"
    sources_path = root / "research" / "sources.json"
    checksum_path = root / "research" / "checksums.sha256"
    try:
        manifest = load_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, [f"cannot read {manifest_path}: {exc}"]
    try:
        sources = load_json(sources_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, [f"cannot read {sources_path}: {exc}"]
    try:
        checksum_entries, checksum_errors = parse_checksums(checksum_path)
    except (OSError, UnicodeError) as exc:
        return {}, [f"cannot read {checksum_path}: {exc}"]
    errors.extend(checksum_errors)

    if manifest.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")
    if manifest.get("source_catalog_sha256") != sha256(sources_path):
        errors.append("manifest source_catalog_sha256 differs from research/sources.json")
    papers = manifest.get("papers")
    if not isinstance(papers, list) or not papers:
        return {}, errors + ["manifest papers must be a non-empty list"]
    try:
        expected_ids = expected_paper_ids(root, sources)
    except (OSError, UnicodeError, ValueError) as exc:
        return {}, errors + [f"cannot derive expected paper IDs: {exc}"]
    if manifest.get("sync_complete") is not True:
        errors.append("manifest sync_complete must be true")
    selected_ids = manifest.get("selected_paper_ids")
    if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
        errors.append("manifest selected_paper_ids must be a list of strings")
    elif set(selected_ids) != expected_ids:
        errors.append("manifest selected_paper_ids differs from research/sources.json")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    results: list[dict[str, Any]] = []
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            errors.append(f"paper[{index}] is not an object")
            continue
        label = str(paper.get("id") or f"paper[{index}]")
        if label in seen_ids:
            errors.append(f"duplicate paper id: {label}")
        seen_ids.add(label)
        status = paper.get("status")
        if status not in {"downloaded", "cached"}:
            errors.append(f"{label}: acquisition status is {status!r}")
            continue

        relative = paper.get("local_path")
        if not isinstance(relative, str) or not relative:
            errors.append(f"{label}: missing local_path")
            continue
        if relative in seen_paths:
            errors.append(f"duplicate local_path: {relative}")
        seen_paths.add(relative)
        path = safe_repo_path(root, relative)
        if path is None:
            errors.append(f"{label}: local_path escapes research/papers: {relative}")
            continue
        if not path.is_file():
            errors.append(f"{label}: missing file {relative}")
            continue

        size = path.stat().st_size
        actual_hash = sha256(path)
        with path.open("rb") as handle:
            header = handle.read(8)
            handle.seek(max(0, size - 4096))
            trailer = handle.read()
        if not header.startswith(b"%PDF-"):
            errors.append(f"{label}: missing PDF signature")
        if b"%%EOF" not in trailer:
            errors.append(f"{label}: missing PDF EOF marker")
        if paper.get("bytes") != size:
            errors.append(f"{label}: byte count differs from manifest")
        if paper.get("sha256") != actual_hash:
            errors.append(f"{label}: SHA-256 differs from manifest")
        if checksum_entries.get(relative) != actual_hash:
            errors.append(f"{label}: checksum file differs for {relative}")

        try:
            reader = PdfReader(path, strict=False)
            if reader.is_encrypted:
                errors.append(f"{label}: PDF is encrypted")
                continue
            page_count = len(reader.pages)
            if page_count < 1:
                errors.append(f"{label}: PDF contains no pages")
                continue
            first_page_text = (reader.pages[0].extract_text() or "").strip()
            if len(first_page_text) < 20:
                errors.append(f"{label}: first page has less than 20 extractable characters")
            results.append(
                {
                    "id": label,
                    "local_path": relative,
                    "sha256": actual_hash,
                    "bytes": size,
                    "page_count": page_count,
                    "first_page_text_characters": len(first_page_text),
                    "pdf_header": header.decode("ascii", errors="replace").strip(),
                }
            )
        except Exception as exc:  # pypdf intentionally exposes varied parse exceptions.
            errors.append(f"{label}: pypdf parse failed: {exc}")

    extra_checksums = sorted(set(checksum_entries) - seen_paths)
    if extra_checksums:
        errors.append(f"checksum file has unregistered paths: {extra_checksums}")
    missing_checksums = sorted(seen_paths - set(checksum_entries))
    if missing_checksums:
        errors.append(f"manifest paths missing from checksum file: {missing_checksums}")
    if seen_ids != expected_ids:
        errors.append("manifest paper IDs differ from research/sources.json")
    disk_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "research" / "papers").rglob("*.pdf")
    }
    extra_pdfs = sorted(disk_paths - seen_paths)
    if extra_pdfs:
        errors.append(f"unregistered PDFs on disk: {extra_pdfs}")

    report = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "verifier": "tools/research/verify_paper_library.py",
        "manifest_sha256": sha256(manifest_path),
        "checksums_sha256": sha256(checksum_path),
        "paper_count": len(results),
        "total_bytes": sum(item["bytes"] for item in results),
        "total_pages": sum(item["page_count"] for item in results),
        "papers": sorted(results, key=lambda item: item["id"]),
    }
    return report, errors


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    report, errors = verify(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"research paper verification failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    if args.write_report:
        path = root / "research" / "verification.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
        print(f"wrote {path.relative_to(root).as_posix()}")
    print(
        f"verified {report['paper_count']} PDFs, {report['total_pages']} pages, "
        f"{report['total_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
