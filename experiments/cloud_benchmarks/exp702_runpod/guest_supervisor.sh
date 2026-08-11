#!/usr/bin/env bash
set -euo pipefail

workspace="${SAIR_EXP702_WORKSPACE:-/workspace/sair-exp702}"
source_archive="${SAIR_EXP702_SOURCE_ARCHIVE:-/workspace/exp702/inputs/exp702-source.tar.gz}"
predecessor_archive="${SAIR_EXP702_PREDECESSOR_ARCHIVE:-/workspace/exp702/inputs/exp701-terminal-predecessor.tar.gz}"
wheel="${SAIR_EXP702_TORCH_WHEEL:-/workspace/exp702/inputs/torch-2.5.1+cu124-cp312-cp312-linux_x86_64.whl}"
durable="${SAIR_EXP702_DURABLE_DIRECTORY:-/workspace/exp702}"
output="$workspace/repository/.scratch/exp-702-exp701-terminal-evaluation-continuation"
sync_seconds="${SAIR_EXP702_SYNC_SECONDS:-300}"
pod_id="${SAIR_EXP702_POD_ID:-${RUNPOD_POD_ID:?required}}"
volume_id="${SAIR_EXP702_VOLUME_ID:?required}"
owner_nonce="${SAIR_EXP702_OWNER_NONCE:?required}"

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed '/^[[:space:]]*$/d')
test "${#gpu_names[@]}" -eq 1
case "${gpu_names[0]}" in
  "NVIDIA A100 80GB PCIe"|"NVIDIA A100-SXM4-80GB"|"NVIDIA H100 80GB HBM3"|"NVIDIA H100 PCIe"|"NVIDIA L40S"|"NVIDIA RTX 6000 Ada Generation"|"NVIDIA A40"|"NVIDIA RTX A6000"|"NVIDIA GeForce RTX 5090"|"NVIDIA GeForce RTX 4090"|"NVIDIA GeForce RTX 3090"|"NVIDIA RTX A5000"|"NVIDIA L4") ;;
  *) echo "unapproved GPU: ${gpu_names[0]}" >&2; exit 74 ;;
esac
test "${SAIR_CUDA_ALLOW_CPU_FALLBACK:-0}" = "0"

mkdir -p "$workspace/repository" "$durable/exports/$owner_nonce" "$durable/status" "$durable/acks"
exec 9>"$durable/single-resource-guest.lock"
flock -n 9 || { echo 'another EXP702 guest owns this resource' >&2; exit 73; }

test "$(sha256sum "$source_archive" | cut -d' ' -f1)" = "${SAIR_EXP702_SOURCE_SHA256:?required}"
test "$(sha256sum "$predecessor_archive" | cut -d' ' -f1)" = "dcd500925e9c57ce7daf4af78e068747c91fc71c84df04a795d1e50e456a42fc"
test "$(stat -c '%s' "$wheel")" = "908202405"
test "$(sha256sum "$wheel" | cut -d' ' -f1)" = "bf6484bfe5bc4f92a4a1a1bf553041505e19a911f717065330eb061afe0e14d7"
if ! test -f "$workspace/repository/.sair-exp702-source.json"; then
  tar -xzf "$source_archive" -C "$workspace/repository"
fi
cd "$workspace/repository"

python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
record = json.loads(Path('.sair-exp702-source.json').read_text())
for label, spec in record['bindings'].items():
    if label == 'torch_cu124_wheel':
        continue
    path = Path(spec['path'])
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        raise SystemExit(f'bound runtime file absent or wrong-sized: {label}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise SystemExit(f'bound runtime file mismatch: {label}')
PY

export_once() {
  local export_kind="${1:-periodic}"
  local scientific_outcome="${2:-}"
  local temporary="$durable/.exp702-results.tar.gz.tmp-$$"
  local manifest_root="$durable/.manifest-$$"
  (
    flock -x 8
    local sequence
    sequence="$(cat "$durable/export-sequence" 2>/dev/null || printf '0')"
    local prefix="$durable/exports/$owner_nonce/export-$(printf '%04d' "$sequence").tar.gz"
    local terminal_summary_sha256=""
    if test "$export_kind" = final; then
      test "$scientific_outcome" = positive -o "$scientific_outcome" = negative
      test -f "$output/terminal-summary.json"
      terminal_summary_sha256="$(sha256sum "$output/terminal-summary.json" | cut -d' ' -f1)"
    fi
    rm -rf "$manifest_root"
    mkdir -p "$manifest_root/exp702"
    python3 - "$manifest_root/exp702/export.json" "$pod_id" "$volume_id" "$owner_nonce" \
      "$sequence" "$export_kind" "$scientific_outcome" "$terminal_summary_sha256" <<'PY'
import json, pathlib, sys
value = {"schema_version": "exp702-guest-export-v2", "pod_id": sys.argv[2],
         "volume_id": sys.argv[3], "owner_nonce": sys.argv[4], "sequence": int(sys.argv[5]),
         "export_kind": sys.argv[6]}
if sys.argv[6] == "final":
    value.update(scientific_outcome=sys.argv[7], terminal_summary_sha256=sys.argv[8])
pathlib.Path(sys.argv[1]).write_text(json.dumps(value, sort_keys=True) + "\n")
PY
    tar -czf "$temporary" -C "$manifest_root" exp702 \
      -C "$workspace/repository" ".scratch/exp-702-exp701-terminal-evaluation-continuation"
    tar -tzf "$temporary" >/dev/null
    mv "$temporary" "$prefix"
    sha256sum "$prefix" > "$prefix.sha256"
    printf '{"schema_version":"exp702-guest-export-v2","pod_id":"%s","volume_id":"%s","owner_nonce":"%s","sequence":%s,"archive_sha256":"%s"}\n' \
      "$pod_id" "$volume_id" "$owner_nonce" "$sequence" \
      "$(cut -d' ' -f1 "$prefix.sha256")" > "$prefix.receipt.json"
    rm -rf "$manifest_root"
    printf '%s\n' "$((sequence + 1))" > "$durable/export-sequence"
    python3 - "$durable/last-export.json.tmp-$$" "$pod_id" "$volume_id" "$owner_nonce" \
      "$sequence" "$(cut -d' ' -f1 "$prefix.sha256")" "$export_kind" "$scientific_outcome" <<'PY'
import json, pathlib, sys
value = {"pod_id": sys.argv[2], "volume_id": sys.argv[3], "owner_nonce": sys.argv[4],
         "sequence": int(sys.argv[5]), "archive_sha256": sys.argv[6], "export_kind": sys.argv[7]}
if sys.argv[7] == "final": value["scientific_outcome"] = sys.argv[8]
pathlib.Path(sys.argv[1]).write_text(json.dumps(value, sort_keys=True) + "\n")
PY
    mv "$durable/last-export.json.tmp-$$" "$durable/last-export.json"
  ) 8>"$durable/export.lock"
}

write_state() {
  local state="$1"
  python3 - "$durable/status/$owner_nonce.json.tmp-$$" "$pod_id" "$volume_id" "$owner_nonce" \
    "$state" "$durable/last-export.json" <<'PY'
import json, pathlib, sys
last = json.loads(pathlib.Path(sys.argv[6]).read_text())
value = {"schema_version": "exp702-guest-state-v2", "pod_id": sys.argv[2],
         "volume_id": sys.argv[3], "owner_nonce": sys.argv[4], "state": sys.argv[5],
         "latest_sequence": last["sequence"]}
if sys.argv[5].startswith("completed-"):
    if last.get("export_kind") != "final": raise SystemExit("terminal state lacks final export")
    value["final_export"] = last
pathlib.Path(sys.argv[1]).write_text(json.dumps(value, sort_keys=True) + "\n")
PY
  mv "$durable/status/$owner_nonce.json.tmp-$$" "$durable/status/$owner_nonce.json"
}

mkdir -p "$output"
export_once
write_state running
SAIR_CUDA_PRELOADED_TORCH_WHEEL="$wheel" bash environments/cuda/bootstrap.sh \
  --profile cu124 --venv "$workspace/venv-cu124" --receipt "$output/environment.json"
python="$workspace/venv-cu124/bin/python"
"$python" - "$output/environment.json" "$wheel" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text())
wheel = Path(sys.argv[2])
bound = receipt.get('preloaded_torch_wheel', {})
if bound.get('sha256') != hashlib.sha256(wheel.read_bytes()).hexdigest():
    raise SystemExit('bootstrap did not consume the preloaded cu124 wheel')
if bound.get('hash_enforcement') != 'uv --require-hashes' or bound.get('canonical_lock_unchanged') is not True:
    raise SystemExit('preloaded wheel consumption was not canonical-lock bound')
PY
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAIR_CUDA_PROFILE=cu124
bash environments/cuda/run-acceptance.sh --profile cu124 --venv "$workspace/venv-cu124" --output "$output/hardware-acceptance"
"$python" - "$output/hardware-acceptance/acceptance.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
if receipt.get("status") != "passed" or receipt.get("matrix", {}).get("passed") is not True:
    raise SystemExit("exact CUDA acceptance gate did not pass before evaluation")
PY

periodic_export() {
  while sleep "$sync_seconds"; do export_once; done
}
periodic_export &
export_pid=$!
on_exit() {
  status=$?
  kill "$export_pid" 2>/dev/null || true
  wait "$export_pid" 2>/dev/null || true
  if test "$status" -ne 0 && test "$status" -ne 75; then
    export_once || true
    write_state failed || true
  fi
}
trap on_exit EXIT

"$python" -m experiments.exp_702_exp701_terminal_evaluation_continuation.run \
  --config configs/exp_702_exp701_terminal_evaluation_continuation.yaml \
  --archive "$predecessor_archive" --output "$output" --device cuda
test -f "$output/terminal-summary.json"
scientific_outcome="$(python - "$output/terminal-summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print('positive' if value.get('scientific_outcome') == 'positive' else 'negative')
PY
)"
export_once final "$scientific_outcome"
final_sequence="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sequence"])' "$durable/last-export.json")"
final_digest="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive_sha256"])' "$durable/last-export.json")"
write_state "completed-$scientific_outcome"
for _attempt in $(seq 1 60); do
  if python3 - "$durable/acks/$owner_nonce.json" "$owner_nonce" "$final_sequence" "$final_digest" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
value = json.loads(path.read_text())
ok = (value.get('status') == 'restore-verified-offpod' and value.get('owner_nonce') == sys.argv[2]
      and value.get('sequence') == int(sys.argv[3]) and value.get('archive_sha256') == sys.argv[4])
raise SystemExit(0 if ok else 1)
PY
  then exit 0; fi
  sleep 15
done
exit 75
