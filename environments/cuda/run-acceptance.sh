#!/usr/bin/env bash
set -euo pipefail

profile="${SAIR_CUDA_PROFILE:-cu124}"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
environment_dir=""
output_dir=""

while (($#)); do
  case "$1" in
    --profile) profile="$2"; shift 2 ;;
    --venv) environment_dir="$2"; shift 2 ;;
    --output) output_dir="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

environment_dir="${environment_dir:-$repository_root/.venv-cuda-$profile}"
output_dir="${output_dir:-$repository_root/.scratch/cuda-acceptance-$profile}"
python="$environment_dir/bin/python"
[[ -x "$python" ]] || {
  echo "missing CUDA environment; run environments/cuda/bootstrap.sh first" >&2
  exit 3
}
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export SAIR_CUDA_PROFILE="$profile"
"$python" "$repository_root/environments/cuda/verify_environment.py" \
  --profile "$profile" --receipt "$output_dir/environment.json"
exec "$python" -m experiments.cuda_portability.acceptance \
  --require-cuda --profile "$profile" --output "$output_dir"
