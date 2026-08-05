#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: EXP699_MAX_WALL_SECONDS=7200 run_guarded.sh COMMAND [ARG ...]" >&2
  exit 2
fi

maximum_wall_seconds="${EXP699_MAX_WALL_SECONDS:-7200}"
kill_grace_seconds="${EXP699_KILL_GRACE_SECONDS:-120}"

if ! [[ "$maximum_wall_seconds" =~ ^[0-9]+$ ]] || (( maximum_wall_seconds < 1 )); then
  echo "EXP699_MAX_WALL_SECONDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$kill_grace_seconds" =~ ^[0-9]+$ ]] || (( kill_grace_seconds < 1 )); then
  echo "EXP699_KILL_GRACE_SECONDS must be a positive integer" >&2
  exit 2
fi

# The Python runner checks its own deadline at every work boundary. GNU timeout
# is a second containment layer for a blocked CUDA kernel or unresponsive child.
shell_ceiling_seconds=$((maximum_wall_seconds + kill_grace_seconds))
exec timeout --foreground --signal=TERM \
  --kill-after="${kill_grace_seconds}s" \
  "${shell_ceiling_seconds}s" \
  "$@"
