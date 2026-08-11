#!/usr/bin/env bash
set -euo pipefail

archive="${SAIR_EXP701_SOURCE_ARCHIVE:-/workspace/sair-exp701-source.tar.gz}"
expected_sha="${SAIR_EXP701_SOURCE_SHA256:?SAIR_EXP701_SOURCE_SHA256 is required}"
workspace="${SAIR_EXP701_WORKSPACE:-/workspace/sair-exp701}"
slice_seconds="${SAIR_EXP701_SLICE_SECONDS:-32400}"
sync_nonce="${SAIR_EXP701_SYNC_NONCE:?SAIR_EXP701_SYNC_NONCE is required}"
runner_wrapper="${SAIR_EXP701_RUNNER_WRAPPER:?SAIR_EXP701_RUNNER_WRAPPER is required}"
repository="$workspace/repository"
output="$repository/.scratch/exp-701-cuda-resumable-h92-w256-backbone-factorial"
durable_directory="${SAIR_EXP701_DURABLE_DIRECTORY:-/workspace/exp701-durable}"
export_path="$durable_directory/sair-exp701-checkpoint-export.tar.gz"
sync_ack_path="$durable_directory/offpod-sync-ack.json"
export_lock="$durable_directory/export.lock"
sync_stale_seconds="${SAIR_EXP701_SYNC_STALE_SECONDS:-720}"
sync_initial_wait_seconds="${SAIR_EXP701_SYNC_INITIAL_WAIT_SECONDS:-900}"

mkdir -p "$output" "$durable_directory"
exec > >(tee -a "$output/guest-supervisor.log") 2>&1

test "$(sha256sum "$archive" | cut -d' ' -f1)" = "$expected_sha"
mkdir -p "$repository"
tar -xzf "$archive" -C "$repository"
cd "$repository"

export SAIR_SOURCE_COMMIT="$(python -c 'import json; print(json.load(open(".sair-source.json"))["repository_commit"])')"
export SAIR_CHALLENGE_COMMIT="$(python -c 'import json; print(json.load(open(".sair-source.json"))["challenge_commit"])')"
export SAIR_SOURCE_DIRTY=0
test "$SAIR_CHALLENGE_COMMIT" = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
test "$(sha256sum .scratch/exp701-input/exp609-weights.safetensors | cut -d' ' -f1)" = "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"

python=python3

export_once() {
  mkdir -p "$durable_directory"
  flock "$export_lock" "$python" -m \
    experiments.cloud_benchmarks.exp701_runpod.recovery_transport export \
    --output "$output" --archive "$export_path" \
    --mlflow "$repository/.scratch/mlflow"
}

ack_valid() {
  local require_current="$1"
  "$python" - "$export_path" "$sync_ack_path" "$sync_nonce" \
    "$sync_stale_seconds" "$require_current" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import time

archive, ack_path = map(Path, sys.argv[1:3])
nonce, maximum_age, require_current = sys.argv[3], float(sys.argv[4]), sys.argv[5] == "yes"
ack = json.loads(ack_path.read_text(encoding="utf-8"))
if ack.get("status") != "verified-offpod-copy" or ack.get("nonce") != nonce:
    raise SystemExit(1)
age = time.time() - float(ack["verified_at_epoch"])
if age < -60 or age > maximum_age:
    raise SystemExit(1)
if require_current:
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ack.get("archive_sha256"):
        raise SystemExit(1)
PY
}

wait_for_current_ack() {
  local waited=0
  while ! ack_valid yes; do
    if ((waited >= sync_initial_wait_seconds)); then
      return 1
    fi
    sleep 15
    waited=$((waited + 15))
  done
}

# Establish a restore-verified external baseline before environment bootstrap.
# If bootstrap fails, the controller can still delete both paid resources without
# risking the loss of any scientific root.
export_once
if ! wait_for_current_ack; then
  printf '{"status":"offpod-initial-sync-failed","resume_required":true}\n' > \
    "$output/guest-terminal.json"
  export_once
  exit 75
fi

bash environments/cuda/bootstrap.sh --profile cu124 --venv "$workspace/venv-cu124" --receipt "$output/environment.json"
python="$workspace/venv-cu124/bin/python"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAIR_CUDA_PROFILE=cu124

bash environments/cuda/run-acceptance.sh --profile cu124 --venv "$workspace/venv-cu124" --output "$output/hardware-acceptance"
"$python" -m pytest -q tests/test_exp_701_cuda_resumable_h92_w256_backbone_factorial.py \
  --junitxml="$output/focused-junit.xml"
smoke_output="$output/pre-root-smoke"
if ! test -f "$smoke_output/smoke.json"; then
  "$python" "$runner_wrapper" \
    --mode smoke --device cuda --output "$smoke_output"
fi
"$python" -m experiments.cloud_benchmarks.exp701_runpod.pre_science \
  --output "$smoke_output" \
  --receipt "$output/restored-smoke-verification.json"

resume_arg() {
  local mode="$1" root="$2"
  if test -f "$output/$mode/root-$root/root-opened.json" && \
     ! test -f "$output/$mode/root-$root/root-summary.json"; then
    printf '%s' '--resume'
  fi
}

run_root() {
  local mode="$1" root="$2" resume
  resume="$(resume_arg "$mode" "$root")"
  if test -n "$resume"; then
    "$python" "$runner_wrapper" \
      --mode "$mode" --root "$root" --device cuda --output "$output" "$resume"
  elif ! test -f "$output/$mode/root-$root/root-summary.json"; then
    "$python" "$runner_wrapper" \
      --mode "$mode" --root "$root" --device cuda --output "$output"
  fi
  if ! test -f "$output/$mode/root-$root/root-summary.json"; then
    printf 'EXP701 root returned without a terminal summary: mode=%s root=%s\n' \
      "$mode" "$root" >&2
    return 1
  fi
  export_once
}

triggered() {
  "$python" - "$1" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("confirmation_trigger") else 1)
PY
}

science() {
  run_root calibration 20261413
  run_root formal 20261414
  if triggered "$output/formal/root-20261414/root-summary.json"; then
    run_root confirmations 20261415
    run_root confirmations 20261416
    if triggered "$output/confirmations/root-20261415/root-summary.json" && \
       triggered "$output/confirmations/root-20261416/root-summary.json"; then
      run_root independent 20261417
      run_root independent 20261418
      run_root independent 20261419
      run_root independent 20261420
      run_root independent 20261421
    fi
  fi
}

export_once
if ! wait_for_current_ack; then
  printf '{"status":"offpod-pre-science-sync-failed","resume_required":true}\n' > \
    "$output/guest-terminal.json"
  export_once
  exit 75
fi

science &
science_pid=$!
supervisor_pid=$$
(
  while kill -0 "$science_pid" 2>/dev/null; do
    sleep 300
    export_once
  done
) &
exporter_pid=$!
(
  while kill -0 "$science_pid" 2>/dev/null; do
    sleep 60
    if ! ack_valid no; then
      kill -TERM "$supervisor_pid" 2>/dev/null || true
      exit 0
    fi
  done
) &
sync_watchdog_pid=$!
(
  sleep "$slice_seconds"
  kill -TERM "$supervisor_pid" 2>/dev/null || true
) &
slice_pid=$!

terminate() {
  trap - TERM INT
  kill -TERM "$science_pid" 2>/dev/null || true
  wait "$science_pid" 2>/dev/null || true
  printf '{"status":"slice-checkpointed","resume_required":true}\n' > "$output/guest-terminal.json"
  export_once
  if ! wait_for_current_ack; then
    printf '{"status":"final-offpod-sync-failed","resume_required":true}\n' > \
      "$output/guest-terminal.json"
  fi
  exit 75
}
trap terminate TERM INT

set +e
wait "$science_pid"
status=$?
set -e
kill "$exporter_pid" "$sync_watchdog_pid" "$slice_pid" 2>/dev/null || true
wait "$exporter_pid" "$sync_watchdog_pid" "$slice_pid" 2>/dev/null || true
if ((status == 0)); then
  printf '{"status":"completed","resume_required":false}\n' > "$output/guest-terminal.json"
else
  printf '{"status":"failed","resume_required":false,"exit_code":%d}\n' "$status" > "$output/guest-terminal.json"
fi
export_once
if ! wait_for_current_ack; then
  printf '{"status":"final-offpod-sync-failed","resume_required":true}\n' > \
    "$output/guest-terminal.json"
  status=75
fi
exit "$status"
