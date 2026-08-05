#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output="${1:-$repository_root/.scratch/exp701-launch/source.tar.gz}"
checkpoint="$repository_root/.scratch/exp701-input/exp609-weights.safetensors"
challenge="$repository_root/sources/github/modular-arithmetic-challenge"

git -C "$repository_root" diff --quiet
git -C "$repository_root" diff --cached --quiet
test -f "$checkpoint"
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
test "$(git -C "$challenge" rev-parse HEAD)" = "82510bba00a1126649bd76dd1a451f14d0b3eb60"

temporary="$(mktemp -d)"
trap 'rm -rf -- "$temporary"' EXIT
git -C "$repository_root" archive HEAD | tar -x -C "$temporary"
mkdir -p "$temporary/sources/github/modular-arithmetic-challenge"
git -C "$challenge" archive HEAD | tar -x -C "$temporary/sources/github/modular-arithmetic-challenge"
mkdir -p "$temporary/.scratch/exp701-input"
cp "$checkpoint" "$temporary/.scratch/exp701-input/exp609-weights.safetensors"

source_commit="$(git -C "$repository_root" rev-parse HEAD)"
python - "$temporary/.sair-source.json" "$source_commit" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "repository_commit": sys.argv[2],
    "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
    "source_checkpoint_sha256": "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd",
}, indent=2, sort_keys=True) + "\n")
PY

mkdir -p "$(dirname "$output")"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf "$output" -C "$temporary" .
sha256sum "$output" > "$output.sha256"
printf 'source=%s\nsha256=%s\n' "$output" "$(cut -d' ' -f1 "$output.sha256")"
