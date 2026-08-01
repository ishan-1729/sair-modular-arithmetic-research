from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stale_pdf_pruning_is_hash_guarded_and_path_confined(tmp_path: Path) -> None:
    papers = tmp_path / "research" / "papers"
    papers.mkdir(parents=True)

    removable = papers / "removable.pdf"
    removable_bytes = b"%PDF-1.4\ncontrolled fixture\n%%EOF\n"
    removable.write_bytes(removable_bytes)

    modified = papers / "modified.pdf"
    modified.write_bytes(b"%PDF-1.4\nlocally modified fixture\n%%EOF\n")

    selected = papers / "selected.pdf"
    selected_bytes = b"%PDF-1.4\nstill selected fixture\n%%EOF\n"
    selected.write_bytes(selected_bytes)

    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\noutside fixture\n%%EOF\n")

    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    escaped = outside_directory / "escaped.pdf"
    escaped.write_bytes(b"%PDF-1.4\nescaped fixture\n%%EOF\n")
    (papers / "linked-directory").symlink_to(
        outside_directory,
        target_is_directory=True,
    )

    prior_path = papers / "prior-location.pdf"
    prior_path.write_bytes(b"%PDF-1.4\nprior fixture\n%%EOF\n")

    module_url = (ROOT / "tools" / "research" / "sync-paper-library.mjs").as_uri()
    payload = {
        "root": str(tmp_path),
        "removable_sha": hashlib.sha256(removable_bytes).hexdigest(),
        "selected_sha": hashlib.sha256(selected_bytes).hexdigest(),
    }
    script = f"""
import assert from 'node:assert/strict';
import {{
  assertManagedPaperPathsStable,
  pruneStalePaperFiles,
}} from {json.dumps(module_url)};

const fixture = {json.dumps(payload)};
const selectedItems = [{{ localPath: 'research/papers/SELECTED.pdf' }}];

const pruned = pruneStalePaperFiles(fixture.root, [
  {{ local_path: 'research/papers/removable.pdf', sha256: fixture.removable_sha }},
  {{ local_path: 'research/papers/selected.pdf', sha256: fixture.selected_sha }},
], selectedItems);
assert.deepEqual(pruned, ['research/papers/removable.pdf']);

assert.throws(
  () => pruneStalePaperFiles(fixture.root, [
    {{ local_path: 'research/papers/modified.pdf', sha256: '0'.repeat(64) }},
  ], []),
  /locally modified stale paper/,
);
assert.throws(
  () => pruneStalePaperFiles(fixture.root, [
    {{ local_path: 'outside.pdf', sha256: '0'.repeat(64) }},
  ], []),
  /outside research.papers/,
);
assert.throws(
  () => pruneStalePaperFiles(fixture.root, [
    {{ local_path: 'research/papers/linked-directory/escaped.pdf', sha256: '0'.repeat(64) }},
  ], []),
  /symbolic link/,
);
assert.throws(
  () => pruneStalePaperFiles(fixture.root, [{{ sha256: '0'.repeat(64) }}], []),
  /missing local_path/,
);
assert.throws(
  () => assertManagedPaperPathsStable(
    fixture.root,
    new Map([['web:relocated', {{
      id: 'web:relocated',
      local_path: 'research/papers/prior-location.pdf',
    }}]]),
    [{{
      key: 'web:relocated',
      localPath: 'research/papers/new-location.pdf',
    }}],
  ),
  /changes local_path/,
);
"""

    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not removable.exists()
    assert modified.exists()
    assert selected.exists()
    assert outside.exists()
    assert escaped.exists()
    assert prior_path.exists()
