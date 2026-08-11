#!/usr/bin/env bash
set -euo pipefail

profile="cu124"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
environment_dir=""
receipt_path=""
dry_run=0
preloaded_torch_wheel="${SAIR_CUDA_PRELOADED_TORCH_WHEEL:-}"

while (($#)); do
  case "$1" in
    --profile) profile="$2"; shift 2 ;;
    --venv) environment_dir="$2"; shift 2 ;;
    --receipt) receipt_path="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$profile" in
  cu121)
    minimum_driver="525.60.13"
    minimum_compute="7.0"
    lock_sha256="817f01d626ed840d4a8912968229d87450ba60da7d2cb42552fccccfca9c4f70"
    ;;
  cu124)
    minimum_driver="550.54.14"
    minimum_compute="7.0"
    lock_sha256="2ef0ecd02324eb9d840d06000dfa3f11d7b6f75223796db95188edbf71278355"
    ;;
  *) echo "unknown CUDA profile: $profile" >&2; exit 2 ;;
esac

environment_dir="${environment_dir:-$repository_root/.venv-cuda-$profile}"
receipt_path="${receipt_path:-$repository_root/.scratch/cuda-environment-$profile.json}"
lock_path="$repository_root/environments/cuda/requirements-$profile.txt"
uv_version="0.11.28"
uv_installer_url="https://astral.sh/uv/0.11.28/install.sh"
uv_installer_sha256="b7b3fe80cad1142a2a5794050b7db7b3291d1bac1423b0732571dd9366e8ca8b"
cu124_torch_url="https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-linux_x86_64.whl"
cu124_torch_sha256="bf6484bfe5bc4f92a4a1a1bf553041505e19a911f717065330eb061afe0e14d7"

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
  echo "CUDA bootstrap supports Linux x86_64 only" >&2
  exit 3
}
echo "$lock_sha256  $lock_path" | sha256sum --check --status || {
  echo "CUDA lock hash mismatch: $lock_path" >&2
  exit 3
}

if [[ -n "$preloaded_torch_wheel" ]]; then
  [[ "$profile" == "cu124" ]] || {
    echo "preloaded PyTorch wheel is supported only for the cu124 profile" >&2
    exit 3
  }
  [[ -f "$preloaded_torch_wheel" ]] || {
    echo "preloaded PyTorch wheel is absent: $preloaded_torch_wheel" >&2
    exit 3
  }
  [[ "$(stat -c '%s' "$preloaded_torch_wheel")" == "908202405" ]] || {
    echo "preloaded PyTorch wheel size mismatch" >&2
    exit 3
  }
  echo "$cu124_torch_sha256  $preloaded_torch_wheel" | \
    sha256sum --check --status || {
      echo "preloaded PyTorch wheel hash mismatch" >&2
      exit 3
    }
fi

if ((dry_run)); then
  printf '{"status":"dry-run","profile":"%s","venv":"%s","lock_sha256":"%s","minimum_driver":"%s","minimum_compute_capability":"%s"}\n' \
    "$profile" "$environment_dir" "$lock_sha256" "$minimum_driver" "$minimum_compute"
  exit 0
fi

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is required; CUDA bootstrap never falls back to CPU" >&2
  exit 4
}
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0 | head -1 | tr -d '[:space:]')"
compute="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0 | head -1 | tr -d '[:space:]')"
[[ -n "$driver" && -n "$compute" ]] || {
  echo "no NVIDIA GPU 0 is available" >&2
  exit 4
}
[[ "$(printf '%s\n%s\n' "$minimum_driver" "$driver" | sort -V | head -1)" == "$minimum_driver" ]] || {
  echo "NVIDIA driver $driver is below required $minimum_driver" >&2
  exit 4
}
[[ "$(printf '%s\n%s\n' "$minimum_compute" "$compute" | sort -V | head -1)" == "$minimum_compute" ]] || {
  echo "compute capability $compute is below required $minimum_compute" >&2
  exit 4
}

temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "$temporary_dir"' EXIT
uv_binary=""
if command -v uv >/dev/null && [[ "$(uv --version)" == "uv $uv_version (x86_64-unknown-linux-gnu)" ]]; then
  uv_binary="$(command -v uv)"
else
  command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 5; }
  download=""
  if command -v curl >/dev/null; then
    download="curl"
  elif command -v wget >/dev/null; then
    download="wget"
  else
    echo "curl or wget is required to bootstrap pinned uv" >&2
    exit 5
  fi
  installer="$temporary_dir/uv-installer.sh"
  if [[ "$download" == "curl" ]]; then
    curl --proto '=https' --tlsv1.2 -LsSf "$uv_installer_url" -o "$installer"
  else
    wget --https-only -qO "$installer" "$uv_installer_url"
  fi
  echo "$uv_installer_sha256  $installer" | sha256sum --check --status || {
    echo "uv installer hash mismatch" >&2
    exit 5
  }
  uv_home="$repository_root/.scratch/tools/uv-$uv_version"
  env UV_UNMANAGED_INSTALL="$uv_home" sh "$installer"
  uv_binary="$uv_home/uv"
fi

[[ "$($uv_binary --version)" == "uv $uv_version (x86_64-unknown-linux-gnu)" ]] || {
  echo "pinned uv runtime verification failed" >&2
  exit 5
}

temporary_lock=""
if [[ -n "$preloaded_torch_wheel" ]]; then
  temporary_lock="$temporary_dir/requirements-$profile-preloaded.txt"
  python3 - "$lock_path" "$temporary_lock" "$cu124_torch_url" \
    "$preloaded_torch_wheel" "$cu124_torch_sha256" <<'PY'
from pathlib import Path
import sys

source, destination, expected_url, wheel, expected_sha256 = sys.argv[1:]
text = Path(source).read_text(encoding="utf-8")
needle = f"torch @ {expected_url} \\\n    --hash=sha256:{expected_sha256}"
replacement = f"torch @ {Path(wheel).resolve().as_uri()} \\\n    --hash=sha256:{expected_sha256}"
if text.count(needle) != 1:
    raise SystemExit("canonical PyTorch lock stanza is absent or ambiguous")
Path(destination).write_text(text.replace(needle, replacement), encoding="utf-8")
PY
fi

if [[ -e "$environment_dir" && ! -f "$environment_dir/.sair-cuda-profile" ]]; then
  echo "refusing to reuse unmanaged environment directory: $environment_dir" >&2
  exit 6
fi
"$uv_binary" --no-config python install 3.12
if [[ ! -d "$environment_dir" ]]; then
  "$uv_binary" --no-config venv --managed-python --python 3.12 "$environment_dir"
  printf '%s\n' "$profile" > "$environment_dir/.sair-cuda-profile"
elif [[ "$(<"$environment_dir/.sair-cuda-profile")" != "$profile" ]]; then
  echo "environment profile marker does not match $profile" >&2
  exit 6
fi
env -u UV_INDEX -u UV_EXTRA_INDEX_URL -u UV_INDEX_URL \
  -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
  "$uv_binary" --no-config pip sync --python "$environment_dir/bin/python" \
  --require-hashes --default-index https://pypi.org/simple \
  "${temporary_lock:-$lock_path}"
"$uv_binary" --no-config pip check --python "$environment_dir/bin/python"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export SAIR_UV_BIN="$uv_binary"
printf '%s\n' "$SAIR_UV_BIN" > "$environment_dir/.sair-uv-bin"
"$environment_dir/bin/python" "$repository_root/environments/cuda/verify_environment.py" \
  --profile "$profile" --receipt "$receipt_path"
if [[ -n "$preloaded_torch_wheel" ]]; then
  "$environment_dir/bin/python" - "$receipt_path" "$preloaded_torch_wheel" \
    "$cu124_torch_sha256" <<'PY'
import json
from pathlib import Path
import sys

receipt_path = Path(sys.argv[1])
wheel = Path(sys.argv[2])
expected_sha256 = sys.argv[3]
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
receipt["preloaded_torch_wheel"] = {
    "path": str(wheel),
    "bytes": wheel.stat().st_size,
    "sha256": expected_sha256,
    "canonical_lock_unchanged": True,
    "hash_enforcement": "uv --require-hashes",
}
temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(receipt_path)
PY
fi
printf 'CUDA environment ready: %s\nReceipt: %s\n' "$environment_dir" "$receipt_path"
