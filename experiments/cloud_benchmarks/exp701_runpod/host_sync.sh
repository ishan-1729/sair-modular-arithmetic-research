#!/usr/bin/env bash
set -euo pipefail
umask 077

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
target="${SAIR_EXP701_SSH_TARGET:?SAIR_EXP701_SSH_TARGET is required}"
port="${SAIR_EXP701_SSH_PORT:?SAIR_EXP701_SSH_PORT is required}"
identity="${SAIR_EXP701_SSH_IDENTITY:?SAIR_EXP701_SSH_IDENTITY is required}"
nonce="${SAIR_EXP701_SYNC_NONCE:?SAIR_EXP701_SYNC_NONCE is required}"
destination="${SAIR_EXP701_OFFPOD_DESTINATION:-$repository_root/.scratch/exp701-offpod}"
remote_archive="${SAIR_EXP701_REMOTE_ARCHIVE:-/workspace/exp701-durable/sair-exp701-checkpoint-export.tar.gz}"
remote_ack="${SAIR_EXP701_REMOTE_ACK:-/workspace/exp701-durable/offpod-sync-ack.json}"
python="${SAIR_EXP701_HOST_PYTHON:-python}"

test -f "$identity"
mkdir -p "$destination"
cd "$repository_root"
exec "$python" -m experiments.cloud_benchmarks.exp701_runpod.offpod_sync \
  --ssh-target "$target" \
  --port "$port" \
  --identity "$identity" \
  --remote-archive "$remote_archive" \
  --remote-ack "$remote_ack" \
  --nonce "$nonce" \
  --destination "$destination" \
  --poll-seconds 300 \
  --max-seconds 39600
