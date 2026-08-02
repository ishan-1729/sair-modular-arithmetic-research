#!/usr/bin/env bash
set -euo pipefail

expected_uv="uv 0.11.28"
actual_uv="$(uv --version)"
if [[ "$actual_uv" != "$expected_uv" ]]; then
  echo "expected $expected_uv, found $actual_uv" >&2
  exit 2
fi

benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$benchmark_dir/../../.." && pwd)"
environment_dir="${1:-$repository_root/.venv-exp699-cuda}"

uv venv --python 3.12 "$environment_dir"
uv pip install --python "$environment_dir/bin/python" \
  --requirement "$benchmark_dir/requirements-torch-cu124.txt"
uv pip install --python "$environment_dir/bin/python" \
  --requirement "$benchmark_dir/requirements.txt"
uv pip check --python "$environment_dir/bin/python"
"$environment_dir/bin/python" -c 'import torch; assert torch.__version__ == "2.5.1+cu124"; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
