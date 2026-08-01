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
readonly RUNNER_LOG="$SCRATCH/formal-systemd-retry2-runner.log"
readonly START_MARKER="$SCRATCH/formal-systemd-retry2-start.json"
readonly EXIT_MARKER="$SCRATCH/formal-systemd-retry2-exit.json"

readonly EXPECTED_RUNNER_SHA="308248442faf5066cb5a8572fb80d88d7cedd77af035d1eb7322041c1987e326"
readonly EXPECTED_CONFIG_SHA="d216ceaaedf2be11fbb4e3e82a3b841d1922913e87114beec77901dc052ac91f"
readonly EXPECTED_SCIENTIFIC_SHA="903e770466bcfd131cff4b6639bf2e0649957c6895b348751b09a871f9fb24fa"
readonly EXPECTED_CALIBRATION_SHA="a76feb8f06b5f08d13a310b32b8db08e14ce5d3b528183a5f3fdd36cbf8acfde"
readonly EXPECTED_SOURCE_SHA="cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
readonly EXPECTED_CONTROL_SHA="b3c8c3f247f42dfe82d28732d3474737ac621ddffc116573fbe0145d3c292ac0"
readonly EXPECTED_TRAINING_KEY="9145dbe16314ed9e09d00145dbbb8ef874211cf93b3828daf69c5ab87c1878df"
readonly EXPECTED_EVALUATION_KEY="09ddf4ca0d562425a4c32002281eb0ee4dcd547c22ac23ed413e65561e63f5ee"

export PATH="/home/ishan/.local/bin:/home/ishan/.venvs/sair-mod-challenge/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONDONTWRITEBYTECODE=1

cd "$ROOT"
# shellcheck source=/dev/null
source /home/ishan/.venvs/sair-mod-challenge/bin/activate

fail() {
  printf 'EXP-647 retry2 preflight failed: %s\n' "$*" >&2
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

exit_marker_written=0
write_exit_marker() {
  local status="$1"
  [[ "$exit_marker_written" -eq 0 ]] || return 0
  exit_marker_written=1
  local ended_at
  ended_at="$(date --iso-8601=seconds)"
  write_marker "$EXIT_MARKER" "{\"ended_at\":\"$ended_at\",\"exit_code\":$status,\"service_unit\":\"sair-exp647-formal-retry2.service\",\"runner_log\":\".scratch/exp-647-operand-swap-consistency/formal-systemd-retry2-runner.log\"}"
}

on_exit() {
  local status="$?"
  trap - EXIT
  write_exit_marker "$status"
  exit "$status"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -z "$(git status --porcelain)" ]] || fail "repository is not clean"
[[ -z "$(pgrep -af '^python(3)? .*experiments/exp_647_operand_swap_consistency/run.py.*--mode formal' || true)" ]] \
  || fail "another EXP-647 formal runner is already live"

expect_sha256 "$RUNNER" "$EXPECTED_RUNNER_SHA"
expect_sha256 "$CONFIG" "$EXPECTED_CONFIG_SHA"
expect_sha256 "$CALIBRATION" "$EXPECTED_CALIBRATION_SHA"
expect_sha256 "$SOURCE" "$EXPECTED_SOURCE_SHA"
expect_sha256 "$CONTROL_CHECKPOINT" "$EXPECTED_CONTROL_SHA"

scientific_sha="$(python - "$CONFIG" <<'PY'
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
)"
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

[[ ! -e "$CHALLENGER/weights.safetensors" ]] \
  || fail "challenger final checkpoint already exists"
if compgen -G "$CHALLENGER/training-*.json" >/dev/null; then
  fail "challenger training cache already exists"
fi
if compgen -G "$CHALLENGER/evaluation-*.json" >/dev/null; then
  fail "challenger evaluation cache already exists"
fi
[[ ! -e "$SCRATCH/summary.json" ]] || fail "formal summary already exists"

started_at="$(date --iso-8601=seconds)"
repository_commit="$(git rev-parse HEAD)"
write_marker "$START_MARKER" "{\"started_at\":\"$started_at\",\"repository_commit\":\"$repository_commit\",\"service_unit\":\"sair-exp647-formal-retry2.service\",\"full_fidelity\":true,\"control_cache_reused\":true,\"retry_wall_hard_seconds\":24000,\"retry_aggregate_cpu_hard_seconds\":95000}"

printf 'EXP-647 retry2 preflight passed at %s; starting full-fidelity formal runner.\n' "$started_at"
set +e
timeout --signal=TERM --kill-after=90s 24000s \
  python -u "$RUNNER" \
    --config "$CONFIG" \
    --output "$SCRATCH" \
    --mode formal \
  >> "$RUNNER_LOG" 2>&1
status=$?
set -e
exit "$status"
