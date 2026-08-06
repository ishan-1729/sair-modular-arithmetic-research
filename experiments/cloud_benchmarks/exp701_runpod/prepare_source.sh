#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output="${1:-$repository_root/.scratch/exp701-launch/source.tar.gz}"
checkpoint="$repository_root/.scratch/exp701-input/exp609-weights.safetensors"
challenge="$repository_root/sources/github/modular-arithmetic-challenge"

# This is the complete, reviewable EXP701 runtime payload. Never replace it with
# a repository-wide archive: the development tree contains benchmark snapshots
# and historical evaluation evidence that a training Pod must not receive.
payload_paths=(
  configs/cuda_portability.yaml
  configs/exp_664_sprm_boundary_perturbation.yaml
  configs/exp_686_w256_straight_through_specialist.yaml
  configs/exp_698_h92_w256_backbone_factorial.yaml
  configs/exp_699_h92_w256_backbone_factorial_complete.yaml
  configs/exp_700_resumable_h92_w256_backbone_factorial.yaml
  configs/exp_701_cuda_resumable_h92_w256_backbone_factorial.yaml
  configs/exp_701_runpod_control.yaml
  environments/cuda/bootstrap.sh
  environments/cuda/profiles.yaml
  environments/cuda/requirements-cu121.txt
  environments/cuda/requirements-cu124.txt
  environments/cuda/run-acceptance.sh
  environments/cuda/verify_environment.py
  experiments/cloud_benchmarks/exp699_cuda_replay/__init__.py
  experiments/cloud_benchmarks/exp699_cuda_replay/run.py
  experiments/cloud_benchmarks/exp701_runpod/decision-card.yaml
  experiments/cloud_benchmarks/exp701_runpod/guest_supervisor.sh
  experiments/cloud_benchmarks/exp701_runpod/host_sync.sh
  experiments/cloud_benchmarks/exp701_runpod/offpod_sync.py
  experiments/cloud_benchmarks/exp701_runpod/prepare_source.sh
  experiments/cloud_benchmarks/exp701_runpod/recovery_transport.py
  experiments/council_tournament_common/__init__.py
  experiments/council_tournament_common/statistics.py
  experiments/cuda_portability/__init__.py
  experiments/cuda_portability/acceptance.py
  experiments/cuda_portability/exp300.py
  experiments/cuda_portability/graph.py
  experiments/cuda_portability/inventory.py
  experiments/cuda_portability/recurrent.py
  experiments/cuda_portability/runtime.py
  experiments/exp_100_width_scaling/model.py
  experiments/exp_220_width_independent_cellular/model.py
  experiments/exp_300_width64_bit_tensor_transfer/run.py
  experiments/exp_401_pyg_graph_message_passing/model.py
  experiments/exp_654_neural_ram_executor/model.py
  experiments/exp_664_sprm_boundary_perturbation/__init__.py
  experiments/exp_664_sprm_boundary_perturbation/model.py
  experiments/exp_664_sprm_boundary_perturbation/run.py
  experiments/exp_686_w256_straight_through_specialist/__init__.py
  experiments/exp_686_w256_straight_through_specialist/run.py
  experiments/exp_698_h92_w256_backbone_factorial/__init__.py
  experiments/exp_698_h92_w256_backbone_factorial/run.py
  experiments/exp_699_h92_w256_backbone_factorial_complete/__init__.py
  experiments/exp_699_h92_w256_backbone_factorial_complete/run.py
  experiments/exp_700_resumable_h92_w256_backbone_factorial/__init__.py
  experiments/exp_700_resumable_h92_w256_backbone_factorial/run.py
  experiments/exp_701_cuda_resumable_h92_w256_backbone_factorial/__init__.py
  experiments/exp_701_cuda_resumable_h92_w256_backbone_factorial/run.py
  submission/manifest.json
  submission/model.py
  submission/weights.safetensors
  tests/test_cuda_hardware_acceptance.py
  tests/test_exp_701_cuda_resumable_h92_w256_backbone_factorial.py
  tools/__init__.py
  tools/autoresearch_mlflow.py
)

git -C "$repository_root" diff --quiet
git -C "$repository_root" diff --cached --quiet
test -f "$checkpoint"
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
test "$(git -C "$challenge" rev-parse HEAD)" = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
for path in "${payload_paths[@]}"; do
  git -C "$repository_root" cat-file -e "HEAD:$path"
done

temporary="$(mktemp -d)"
archive_temporary=""
cleanup() {
  if [[ -n "$archive_temporary" ]]; then
    rm -f -- "$archive_temporary"
  fi
  if [[ -n "$temporary" && "$temporary" == /tmp/tmp.* ]]; then
    rm -rf -- "$temporary"
  fi
}
trap cleanup EXIT
git -C "$repository_root" archive --format=tar HEAD "${payload_paths[@]}" | tar -x -C "$temporary"
mkdir -p "$temporary/sources/github/modular-arithmetic-challenge"
git -C "$challenge" archive --format=tar HEAD src | tar -x -C "$temporary/sources/github/modular-arithmetic-challenge"
mkdir -p "$temporary/.scratch/exp701-input"
cp "$checkpoint" "$temporary/.scratch/exp701-input/exp609-weights.safetensors"

source_commit="$(git -C "$repository_root" rev-parse HEAD)"
python - "$temporary" "$source_commit" "${payload_paths[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
repository_paths = sys.argv[3:]
source = {
    "schema_version": "sair-exp701-source-v1",
    "repository_commit": sys.argv[2],
    "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
    "source_checkpoint_sha256": "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd",
    "payload_policy": "explicit-runtime-allowlist-v1",
    "allowlisted_repository_paths": repository_paths,
    "allowlisted_challenge_paths": ["src"],
    "required_training_artifacts": [
        ".scratch/exp701-input/exp609-weights.safetensors",
        "submission/weights.safetensors",
    ],
}
(root / ".sair-source.json").write_text(
    json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

forbidden_prefixes = (
    ".git/",
    ".github/",
    "data/",
    "docs/",
    "evals/",
    "experiments/runs/",
    "knowledge/",
    "research/",
    "sources/github/modular-arithmetic-challenge/public_benchmark/",
    "sources/github/modular-arithmetic-challenge/tests/",
)
forbidden_names = {".env", ".netrc", "credentials", "id_rsa", "id_ed25519", "secrets"}
files = []
for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
    relative = path.relative_to(root).as_posix()
    if relative == ".sair-payload-inventory.json":
        continue
    if relative.startswith(forbidden_prefixes):
        raise SystemExit(f"forbidden payload prefix: {relative}")
    if any(part.lower() in forbidden_names for part in Path(relative).parts):
        raise SystemExit(f"forbidden credential-like payload name: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append({
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })

inventory = {
    "schema_version": "sair-exp701-payload-inventory-v1",
    "policy": "explicit-runtime-allowlist-v1",
    "repository_commit": source["repository_commit"],
    "challenge_commit": source["challenge_commit"],
    "forbidden_prefixes": list(forbidden_prefixes),
    "file_count": len(files),
    "payload_bytes_before_inventory": sum(item["bytes"] for item in files),
    "files": files,
}
(root / ".sair-payload-inventory.json").write_text(
    json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

mkdir -p "$(dirname "$output")"
archive_temporary="$(mktemp "${output}.tmp.XXXXXX")"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -cf - -C "$temporary" . | gzip -n > "$archive_temporary"
mv -f -- "$archive_temporary" "$output"
archive_temporary=""
sha256sum "$output" > "$output.sha256"
printf 'source=%s\nsha256=%s\n' "$output" "$(cut -d' ' -f1 "$output.sha256")"
