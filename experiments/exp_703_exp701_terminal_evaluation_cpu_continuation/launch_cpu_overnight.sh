#!/usr/bin/env bash
set -uo pipefail

repo=/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-651-visit-dilation-adapters
output="$repo/.scratch/exp703-cpu-overnight"
mkdir -p "$output"
exec > >(tee -a "$output/stdout.log") 2> >(tee -a "$output/stderr.log" >&2)

cd "$repo"
source /home/ishan/.venvs/sair-mod-challenge/bin/activate
export PYTHONUNBUFFERED=1

for attempt in $(seq 1 12); do
  printf '{"attempt":%s,"pid":%s,"started_at":"%s"}\n' \
    "$attempt" "$$" "$(date --utc --iso-8601=seconds)" > "$output/launcher-status.json"
  python -u -m experiments.exp_703_exp701_terminal_evaluation_cpu_continuation.run \
    --config configs/exp_703_exp701_terminal_evaluation_cpu_continuation.yaml \
    --archive .scratch/exp701-offpod-s3t/sair-exp701-checkpoint-export.tar.gz \
    --output .scratch/exp703-cpu-overnight
  code=$?
  if [[ -f "$output/terminal-summary.json" ]]; then
    exit 0
  fi
  printf '{"attempt":%s,"exit_code":%s,"retrying":true,"updated_at":"%s"}\n' \
    "$attempt" "$code" "$(date --utc --iso-8601=seconds)" > "$output/launcher-status.json"
  sleep 60
done

printf '{"status":"retry-budget-exhausted","attempts":12,"updated_at":"%s"}\n' \
  "$(date --utc --iso-8601=seconds)" > "$output/launcher-status.json"
exit 2
