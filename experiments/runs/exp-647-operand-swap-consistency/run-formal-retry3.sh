#!/usr/bin/env bash
# shellcheck disable=SC2317
set -Eeuo pipefail

readonly ROOT="/mnt/d/Projects/SAIR_mod_challenge"
readonly SCRATCH="$ROOT/.scratch/exp-647-operand-swap-consistency"
readonly FORMAL="$SCRATCH/formal"
readonly CONTROL="$FORMAL/paired_rollout_control"
readonly CHALLENGER="$FORMAL/confidence_consistency"
readonly RUNNER="$ROOT/experiments/exp_647_operand_swap_consistency/run.py"
readonly CONFIG="$ROOT/configs/exp_647_operand_swap_consistency.yaml"
readonly CALIBRATION="$ROOT/experiments/runs/exp-647-operand-swap-consistency/calibration.json"
readonly SOURCE="$ROOT/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"
readonly CONTROL_CHECKPOINT="$CONTROL/weights.safetensors"
readonly CONTROL_TRAINING_CACHE="$CONTROL/training-9145dbe16314ed9e09d00145dbbb8ef874211cf93b3828daf69c5ab87c1878df.json"
readonly CONTROL_EVALUATION_CACHE="$CONTROL/evaluation-09ddf4ca0d562425a4c32002281eb0ee4dcd547c22ac23ed413e65561e63f5ee.json"

readonly SERVICE_LOG="$SCRATCH/formal-windows-retry3-service.log"
readonly RUNNER_LOG="$SCRATCH/formal-windows-retry3-runner.log"
readonly START_MARKER="$SCRATCH/formal-windows-retry3-start.json"
readonly ALERT_MARKER="$SCRATCH/formal-windows-retry3-alert.json"
readonly EXIT_MARKER="$SCRATCH/formal-windows-retry3-exit.json"
readonly PROBE_START_MARKER="$SCRATCH/formal-windows-retry3-probe-start.json"
readonly PROBE_HEARTBEAT_MARKER="$SCRATCH/formal-windows-retry3-probe-heartbeat.json"
readonly PROBE_COMPLETION_MARKER="$SCRATCH/formal-windows-retry3-probe-completion.json"
readonly RETRY2_EXIT_MARKER="$SCRATCH/formal-systemd-retry2-exit.json"

readonly EXPECTED_RUNNER_SHA="308248442faf5066cb5a8572fb80d88d7cedd77af035d1eb7322041c1987e326"
readonly EXPECTED_CONFIG_SHA="d216ceaaedf2be11fbb4e3e82a3b841d1922913e87114beec77901dc052ac91f"
readonly EXPECTED_SCIENTIFIC_SHA="903e770466bcfd131cff4b6639bf2e0649957c6895b348751b09a871f9fb24fa"
readonly EXPECTED_CALIBRATION_SHA="a76feb8f06b5f08d13a310b32b8db08e14ce5d3b528183a5f3fdd36cbf8acfde"
readonly EXPECTED_SOURCE_SHA="cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
readonly EXPECTED_CONTROL_SHA="b3c8c3f247f42dfe82d28732d3474737ac621ddffc116573fbe0145d3c292ac0"
readonly EXPECTED_TRAINING_CACHE_SHA="bb973255165e79e153e6d7540e60be16be1ec69847ee5691b76411355af955ba"
readonly EXPECTED_EVALUATION_CACHE_SHA="eaecdc2b168abb49e8db175f6d1845835e46bf02efec0572447b479d94216357"
readonly EXPECTED_TRAINING_KEY="9145dbe16314ed9e09d00145dbbb8ef874211cf93b3828daf69c5ab87c1878df"
readonly EXPECTED_EVALUATION_KEY="09ddf4ca0d562425a4c32002281eb0ee4dcd547c22ac23ed413e65561e63f5ee"

readonly ALERT_WALL_SECONDS=11000
readonly ALERT_CPU_SECONDS=36000
readonly HARD_WALL_SECONDS=22000
readonly HARD_CPU_SECONDS=88000
readonly PROBE_SECONDS=180

mode="formal"
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [--preflight-only|--persistence-probe]\n' "$0" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --preflight-only) mode="preflight" ;;
    --persistence-probe) mode="probe" ;;
    *)
      printf 'usage: %s [--preflight-only|--persistence-probe]\n' "$0" >&2
      exit 2
      ;;
  esac
fi

export PATH="/home/ishan/.local/bin:/home/ishan/.venvs/sair-mod-challenge/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONDONTWRITEBYTECODE=1

cd "$ROOT"
# shellcheck source=/dev/null
source /home/ishan/.venvs/sair-mod-challenge/bin/activate

if [[ "$mode" != "preflight" ]]; then
  exec >> "$SERVICE_LOG" 2>&1
fi

fail() {
  printf 'EXP-647 retry3 %s failed: %s\n' "$mode" "$*" >&2
  exit 1
}

expect_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  [[ -f "$path" ]] || fail "missing file: $path"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail "SHA-256 mismatch for $path: $actual"
}

write_marker() {
  local path="$1"
  local payload="$2"
  local temporary="${path}.tmp.$$"
  printf '%s\n' "$payload" > "$temporary"
  mv -f "$temporary" "$path"
}

scientific_sha256() {
  python - "$CONFIG" <<'PY'
import copy
import hashlib
import json
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
scientific = copy.deepcopy(config)
scientific.pop("resource_budget", None)
payload = json.dumps(scientific, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(payload.encode()).hexdigest())
PY
}

run_preflight() {
  [[ -z "$(git status --porcelain)" ]] || fail "repository is not clean"
  [[ -z "$(pgrep -af '[e]xperiments/exp_647_operand_swap_consistency/run.py.*--mode formal' || true)" ]] \
    || fail "another EXP-647 formal runner is already live"

  expect_sha256 "$RUNNER" "$EXPECTED_RUNNER_SHA"
  expect_sha256 "$CONFIG" "$EXPECTED_CONFIG_SHA"
  expect_sha256 "$CALIBRATION" "$EXPECTED_CALIBRATION_SHA"
  expect_sha256 "$SOURCE" "$EXPECTED_SOURCE_SHA"
  expect_sha256 "$CONTROL_CHECKPOINT" "$EXPECTED_CONTROL_SHA"
  expect_sha256 "$CONTROL_TRAINING_CACHE" "$EXPECTED_TRAINING_CACHE_SHA"
  expect_sha256 "$CONTROL_EVALUATION_CACHE" "$EXPECTED_EVALUATION_CACHE_SHA"

  local scientific_sha
  scientific_sha="$(scientific_sha256)"
  [[ "$scientific_sha" == "$EXPECTED_SCIENTIFIC_SHA" ]] \
    || fail "scientific config SHA-256 mismatch: $scientific_sha"

  [[ "$(jq -r '.cache_key' "$CONTROL_TRAINING_CACHE")" == "$EXPECTED_TRAINING_KEY" ]] \
    || fail "control training cache key mismatch"
  [[ "$(jq -r '.result.training.steps' "$CONTROL_TRAINING_CACHE")" == "240" ]] \
    || fail "control training cache does not contain 240 steps"
  [[ "$(jq -r '.result.checkpoint.sha256' "$CONTROL_TRAINING_CACHE")" == "$EXPECTED_CONTROL_SHA" ]] \
    || fail "control training cache checkpoint hash mismatch"
  [[ "$(jq -r '.cache_key' "$CONTROL_EVALUATION_CACHE")" == "$EXPECTED_EVALUATION_KEY" ]] \
    || fail "control evaluation cache key mismatch"
  jq -e '
    .summary["64"].transition.cases == 1000 and
    .summary["128"].transition.cases == 3000 and
    .summary["256"].transition.cases == 3000 and
    .summary["64"].original.cases == 300 and
    .summary["64"].swapped.cases == 300 and
    .summary["128"].original.cases == 300 and
    .summary["128"].swapped.cases == 300 and
    .summary["256"].original.cases == 200 and
    .summary["256"].swapped.cases == 200
  ' "$CONTROL_EVALUATION_CACHE" >/dev/null \
    || fail "control evaluation cache case counts are incomplete"

  [[ -f "$CHALLENGER/initial.safetensors" ]] \
    || fail "challenger frozen initial checkpoint is missing"
  expect_sha256 "$CHALLENGER/initial.safetensors" "$EXPECTED_SOURCE_SHA"
  [[ ! -e "$CHALLENGER/weights.safetensors" ]] \
    || fail "challenger final checkpoint already exists"
  if compgen -G "$CHALLENGER/training-*.json" >/dev/null; then
    fail "challenger training cache already exists"
  fi
  if compgen -G "$CHALLENGER/evaluation-*.json" >/dev/null; then
    fail "challenger evaluation cache already exists"
  fi
  [[ ! -e "$SCRATCH/summary.json" ]] || fail "formal summary already exists"
  [[ ! -e "$RETRY2_EXIT_MARKER" ]] || fail "retry2 exit marker appeared after the audited interruption"
}

launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
repository_commit="$(git rev-parse HEAD)"

if [[ "$mode" == "preflight" ]]; then
  run_preflight
  printf 'formal_retry3_preflight=passed\n'
  printf 'repository_commit=%s\n' "$repository_commit"
  printf 'launcher_sha256=%s\n' "$launcher_sha"
  printf 'runner_sha256=%s\n' "$EXPECTED_RUNNER_SHA"
  printf 'scientific_config_sha256=%s\n' "$EXPECTED_SCIENTIFIC_SHA"
  exit 0
fi

if [[ "$mode" == "probe" ]]; then
  rm -f -- "$PROBE_START_MARKER" "$PROBE_HEARTBEAT_MARKER" "$PROBE_COMPLETION_MARKER"
  run_preflight
  probe_nonce="$(tr -d '-' </proc/sys/kernel/random/uuid)"
  probe_started_at="$(date --iso-8601=seconds)"
  write_marker "$PROBE_START_MARKER" "{\"status\":\"running\",\"started_at\":\"$probe_started_at\",\"probe_nonce\":\"$probe_nonce\",\"launcher_sha256\":\"$launcher_sha\",\"repository_commit\":\"$repository_commit\",\"probe_seconds\":$PROBE_SECONDS,\"pid\":$$}"
  for elapsed in 30 60 90 120 150 180; do
    sleep 30
    heartbeat_at="$(date --iso-8601=seconds)"
    write_marker "$PROBE_HEARTBEAT_MARKER" "{\"status\":\"running\",\"observed_at\":\"$heartbeat_at\",\"probe_nonce\":\"$probe_nonce\",\"launcher_sha256\":\"$launcher_sha\",\"repository_commit\":\"$repository_commit\",\"elapsed_seconds\":$elapsed,\"pid\":$$}"
  done
  probe_completed_at="$(date --iso-8601=seconds)"
  write_marker "$PROBE_COMPLETION_MARKER" "{\"status\":\"completed\",\"started_at\":\"$probe_started_at\",\"completed_at\":\"$probe_completed_at\",\"probe_nonce\":\"$probe_nonce\",\"launcher_sha256\":\"$launcher_sha\",\"repository_commit\":\"$repository_commit\",\"elapsed_seconds\":$PROBE_SECONDS,\"exit_code\":0}"
  exit 0
fi

run_preflight
[[ -f "$PROBE_START_MARKER" && -f "$PROBE_COMPLETION_MARKER" ]] \
  || fail "persistence-probe start/completion evidence is incomplete"
probe_nonce="$(jq -er '.probe_nonce' "$PROBE_START_MARKER")" \
  || fail "persistence-probe start nonce is missing"
jq -e \
  --arg nonce "$probe_nonce" \
  --arg launcher "$launcher_sha" \
  --arg commit "$repository_commit" \
  --argjson elapsed "$PROBE_SECONDS" '
    .status == "completed" and
    .exit_code == 0 and
    .probe_nonce == $nonce and
    .launcher_sha256 == $launcher and
    .repository_commit == $commit and
    .elapsed_seconds == $elapsed and
    (.started_at | type == "string") and
    (.completed_at | type == "string")
  ' "$PROBE_COMPLETION_MARKER" >/dev/null \
  || fail "persistence-probe completion is not bound to this launcher, commit, nonce, and duration"
jq -e \
  --arg nonce "$probe_nonce" \
  --arg launcher "$launcher_sha" \
  --arg commit "$repository_commit" \
  --argjson elapsed "$PROBE_SECONDS" '
    .status == "running" and
    .probe_nonce == $nonce and
    .launcher_sha256 == $launcher and
    .repository_commit == $commit and
    .probe_seconds == $elapsed and
    (.started_at | type == "string")
  ' "$PROBE_START_MARKER" >/dev/null \
  || fail "persistence-probe start is not bound to this launcher, commit, nonce, and duration"
probe_started_epoch="$(date --date="$(jq -r '.started_at' "$PROBE_START_MARKER")" +%s)" \
  || fail "persistence-probe start timestamp is invalid"
probe_completed_epoch="$(date --date="$(jq -r '.completed_at' "$PROBE_COMPLETION_MARKER")" +%s)" \
  || fail "persistence-probe completion timestamp is invalid"
probe_elapsed=$((probe_completed_epoch - probe_started_epoch))
[[ "$probe_elapsed" -ge 170 && "$probe_elapsed" -le 600 ]] \
  || fail "persistence-probe observed duration is outside [170, 600] seconds: $probe_elapsed"
[[ ! -e "$START_MARKER" && ! -e "$ALERT_MARKER" && ! -e "$EXIT_MARKER" ]] \
  || fail "retry3 formal marker already exists; refusing an unaudited restart"

formal_phase="preflight-complete"
monitor_pid=""
exit_marker_written=0

write_exit_marker() {
  local status="$1"
  [[ "$exit_marker_written" -eq 0 ]] || return 0
  exit_marker_written=1
  local ended_at
  ended_at="$(date --iso-8601=seconds)"
  write_marker "$EXIT_MARKER" "{\"status\":\"exited\",\"ended_at\":\"$ended_at\",\"exit_code\":$status,\"phase\":\"$formal_phase\",\"probe_nonce\":\"$probe_nonce\",\"launcher_sha256\":\"$launcher_sha\",\"repository_commit\":\"$repository_commit\",\"runner_log\":\".scratch/exp-647-operand-swap-consistency/formal-windows-retry3-runner.log\"}"
}

on_exit() {
  local status="$?"
  trap - EXIT
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  write_exit_marker "$status"
  exit "$status"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

started_at="$(date --iso-8601=seconds)"
started_epoch="$(date +%s)"
write_marker "$START_MARKER" "{\"status\":\"running\",\"started_at\":\"$started_at\",\"probe_nonce\":\"$probe_nonce\",\"repository_commit\":\"$repository_commit\",\"launcher_sha256\":\"$launcher_sha\",\"full_fidelity\":true,\"control_cache_reused\":true,\"training_updates\":240,\"retry_wall_alert_seconds\":$ALERT_WALL_SECONDS,\"retry_cpu_alert_seconds\":$ALERT_CPU_SECONDS,\"retry_wall_hard_seconds\":$HARD_WALL_SECONDS,\"retry_cpu_hard_seconds\":$HARD_CPU_SECONDS}"

printf 'EXP-647 retry3 preflight passed at %s; starting full-fidelity formal runner.\n' "$started_at"
formal_phase="formal-runner"
ulimit -t "$HARD_CPU_SECONDS"
set +e
timeout --signal=TERM --kill-after=90s "${HARD_WALL_SECONDS}s" \
  python -u "$RUNNER" \
    --config "$CONFIG" \
    --output "$SCRATCH" \
    --mode formal \
  >> "$RUNNER_LOG" 2>&1 &
timeout_pid=$!
set -e

monitor_resources() {
  local wall_alerted=0
  local cpu_alerted=0
  while kill -0 "$timeout_pid" 2>/dev/null; do
    local now_epoch wall_seconds python_pid cpu_seconds observed_at alert_changed
    now_epoch="$(date +%s)"
    wall_seconds=$((now_epoch - started_epoch))
    python_pid="$(pgrep -P "$timeout_pid" -f 'exp_647_operand_swap_consistency/run.py.*--mode formal' | head -1 || true)"
    cpu_seconds=0
    if [[ -n "$python_pid" ]]; then
      cpu_seconds="$(ps -p "$python_pid" -o cputimes= 2>/dev/null | tr -d ' ' || true)"
      [[ "$cpu_seconds" =~ ^[0-9]+$ ]] || cpu_seconds=0
    fi
    alert_changed=0
    if [[ "$wall_alerted" -eq 0 && "$wall_seconds" -ge "$ALERT_WALL_SECONDS" ]]; then
      wall_alerted=1
      alert_changed=1
    fi
    if [[ "$cpu_alerted" -eq 0 && "$cpu_seconds" -ge "$ALERT_CPU_SECONDS" ]]; then
      cpu_alerted=1
      alert_changed=1
    fi
    if [[ "$alert_changed" -eq 1 ]]; then
      observed_at="$(date --iso-8601=seconds)"
      write_marker "$ALERT_MARKER" "{\"status\":\"soft-budget-alert\",\"observed_at\":\"$observed_at\",\"probe_nonce\":\"$probe_nonce\",\"wall_seconds\":$wall_seconds,\"aggregate_process_cpu_seconds\":$cpu_seconds,\"wall_alerted\":$wall_alerted,\"cpu_alerted\":$cpu_alerted,\"wall_alert_seconds\":$ALERT_WALL_SECONDS,\"aggregate_process_cpu_alert_seconds\":$ALERT_CPU_SECONDS,\"launcher_sha256\":\"$launcher_sha\"}"
    fi
    if [[ "$cpu_seconds" -ge "$HARD_CPU_SECONDS" ]]; then
      observed_at="$(date --iso-8601=seconds)"
      write_marker "$ALERT_MARKER" "{\"status\":\"hard-cpu-ceiling\",\"observed_at\":\"$observed_at\",\"probe_nonce\":\"$probe_nonce\",\"wall_seconds\":$wall_seconds,\"aggregate_process_cpu_seconds\":$cpu_seconds,\"hard_cpu_seconds\":$HARD_CPU_SECONDS,\"launcher_sha256\":\"$launcher_sha\"}"
      kill -TERM "$timeout_pid" 2>/dev/null || true
      return 0
    fi
    sleep 30
  done
}

monitor_resources &
monitor_pid=$!
set +e
wait "$timeout_pid"
status=$?
set -e
formal_phase="formal-runner-exited"
exit "$status"
