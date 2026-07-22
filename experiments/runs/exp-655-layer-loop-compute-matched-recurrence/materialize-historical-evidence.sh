#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PRIMARY_WORKSPACE [DESTINATION_WORKTREE]" >&2
  exit 2
fi

primary=$(realpath -e -- "$1")
destination=$(realpath -e -- "${2:-$(git rev-parse --show-toplevel)}")
expected_destination=/mnt/d/Projects/SAIR_mod_challenge_exp655_t048_wt

if [[ "$primary" != /mnt/d/Projects/SAIR_mod_challenge ]]; then
  echo "refusing unexpected primary workspace: $primary" >&2
  exit 1
fi
if [[ "$destination" != "$expected_destination" ]]; then
  echo "refusing unexpected destination worktree: $destination" >&2
  exit 1
fi
if [[ "$primary" == "$destination" ]]; then
  echo "source and destination roots alias" >&2
  exit 1
fi

inventory=$(mktemp /tmp/exp655-t049-inventory.XXXXXX)
sorted_inventory=$(mktemp /tmp/exp655-t049-inventory-sorted.XXXXXX)
trap 'rm -f -- "$inventory" "$sorted_inventory"' EXIT
copied=0
reused=0

sha256() {
  sha256sum -- "$1" | awk '{print $1}'
}

require_within() {
  local path=$1
  local root=$2
  local label=$3
  case "$path" in
    "$root"|"$root"/*) ;;
    *)
      echo "$label escapes intended root: $path" >&2
      exit 1
      ;;
  esac
}

copy_one() {
  local relative=$1
  local expected=$2
  local source_path source_real destination_path destination_real parent parent_real

  if [[ "$relative" == /* || "/$relative/" == *"/../"* ]]; then
    echo "unsafe relative path: $relative" >&2
    exit 1
  fi
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid expected SHA-256 for $relative" >&2
    exit 1
  fi

  source_path="$primary/$relative"
  if [[ ! -f "$source_path" || -L "$source_path" ]]; then
    echo "source is absent, non-regular, or linked: $source_path" >&2
    exit 1
  fi
  source_real=$(realpath -e -- "$source_path")
  require_within "$source_real" "$primary" source
  if [[ "$(sha256 "$source_real")" != "$expected" ]]; then
    echo "source SHA-256 mismatch: $relative" >&2
    exit 1
  fi

  destination_path="$destination/$relative"
  destination_real=$(realpath -m -- "$destination_path")
  require_within "$destination_real" "$destination" destination
  parent=$(dirname -- "$destination_path")
  mkdir -p -- "$parent"
  parent_real=$(realpath -e -- "$parent")
  require_within "$parent_real" "$destination" destination-parent

  if [[ -e "$destination_path" || -L "$destination_path" ]]; then
    if [[ ! -f "$destination_path" || -L "$destination_path" ]]; then
      echo "destination is non-regular or linked: $relative" >&2
      exit 1
    fi
    if [[ "$(sha256 "$destination_path")" != "$expected" ]]; then
      echo "existing destination SHA-256 mismatch: $relative" >&2
      exit 1
    fi
    reused=$((reused + 1))
  else
    local temporary="$destination_path.t049.$$"
    cp --reflink=never -- "$source_real" "$temporary"
    if [[ ! -f "$temporary" || -L "$temporary" || "$(sha256 "$temporary")" != "$expected" ]]; then
      rm -f -- "$temporary"
      echo "copied destination verification failed: $relative" >&2
      exit 1
    fi
    mv -- "$temporary" "$destination_path"
    copied=$((copied + 1))
  fi

  if [[ "$(stat -c '%d:%i' -- "$source_real")" == "$(stat -c '%d:%i' -- "$destination_path")" ]]; then
    echo "destination hardlink aliases source: $relative" >&2
    exit 1
  fi
  if [[ "$(stat -c '%h' -- "$destination_path")" != 1 ]]; then
    echo "destination has unexpected hardlink count: $relative" >&2
    exit 1
  fi
  if [[ "$(sha256 "$destination_path")" != "$expected" ]]; then
    echo "destination SHA-256 mismatch: $relative" >&2
    exit 1
  fi
  printf '%s  %s\n' "$expected" "$relative" >>"$inventory"
}

copy_manifest() {
  local manifest_relative=$1
  local manifest_sha=$2
  local expected_count=$3
  local expected_bytes=$4
  local manifest="$destination/$manifest_relative"
  local count=0
  local bytes=0
  local line expected relative

  if [[ ! -f "$manifest" || -L "$manifest" ]]; then
    echo "tracked manifest absent or linked: $manifest_relative" >&2
    exit 1
  fi
  if ! git -C "$destination" cat-file -e "HEAD:$manifest_relative"; then
    echo "manifest is not tracked at HEAD: $manifest_relative" >&2
    exit 1
  fi
  if ! git -C "$destination" diff --quiet HEAD -- "$manifest_relative"; then
    echo "tracked manifest differs from HEAD: $manifest_relative" >&2
    exit 1
  fi
  if [[ "$(sha256 "$manifest")" != "$manifest_sha" ]]; then
    echo "tracked manifest SHA-256 mismatch: $manifest_relative" >&2
    exit 1
  fi

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    expected=${line:0:64}
    if [[ "${line:64:2}" != "  " ]]; then
      echo "noncanonical manifest line: $manifest_relative" >&2
      exit 1
    fi
    relative=${line:66}
    copy_one "$relative" "$expected"
    count=$((count + 1))
    bytes=$((bytes + $(stat -c '%s' -- "$destination/$relative")))
  done <"$manifest"

  if [[ "$count" -ne "$expected_count" || "$bytes" -ne "$expected_bytes" ]]; then
    echo "manifest count/byte mismatch: $manifest_relative ($count/$bytes)" >&2
    exit 1
  fi
}

copy_one ".scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors" "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
copy_one ".scratch/exp-660-symbol-equivariant-formal/fixtures/exp609-weights.safetensors" "cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
copy_one ".scratch/exp-648-structured-trajectory-consistency/calibration.json" "716852e1fea2ad21401016cdeb3e5e9ef5e529ca263b91a6079904e6075770db"

copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4/calibration.json" "f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4"
copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4/environment.json" "d5754d46d762643e9488a85924208ccd83a959206a14c7b7041cea7d332819c9"
copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4/provenance.json" "429508f0f2f16105ab2e3d06bc50adfa6d8b26b95c932088a12f03bd9cc1b86c"
copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4/resolved-config.yaml" "5d1e30bdb7dbf67c1a448b725976e9f83daf144079c7d5bcde43e74e6948a0e7"
copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/f522a8a5e645d99e575d56cd0cfcefdb300965606cd9947d4eb4ec74aac01df4/run.py" "8dee04eaf6f45de2efe2a31d1a3d22031c11a35450d9930ce609a67f241c709b"

copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/d965d944330fe66f4c333a09479e12b2db893ad19c18492f60ea24dbf68a8ad6/calibration.json" "d965d944330fe66f4c333a09479e12b2db893ad19c18492f60ea24dbf68a8ad6"
copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/d965d944330fe66f4c333a09479e12b2db893ad19c18492f60ea24dbf68a8ad6/environment.json" "d5754d46d762643e9488a85924208ccd83a959206a14c7b7041cea7d332819c9"
copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/d965d944330fe66f4c333a09479e12b2db893ad19c18492f60ea24dbf68a8ad6/provenance.json" "4403531091863f1918cf87dc8b25be851414ee211b207ffce9b96b5f84f65641"
copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/d965d944330fe66f4c333a09479e12b2db893ad19c18492f60ea24dbf68a8ad6/resolved-config.yaml" "97877cec756125ec77e535290196eed015af77c470e442523fb0bbd049f12057"
copy_one ".scratch/exp-648-structured-trajectory-consistency/superseded/d965d944330fe66f4c333a09479e12b2db893ad19c18492f60ea24dbf68a8ad6/run.py" "3b7848227d21f46b9f3f67c40ccb7c29dfee57fbfa05e6312f18dca2aa871af8"

copy_manifest \
  "experiments/runs/exp-648-structured-trajectory-consistency/screen/superseded/root-20261161/scratch-files.sha256" \
  "c74a07dec1e0e9331917dea8d2f25cfcd6a2a7c17380fe35e24ada125d3c0c5e" \
  218 \
  24573993
copy_manifest \
  "experiments/runs/exp-648-structured-trajectory-consistency/screen/superseded/root-20261161/mlflow-artifacts.sha256" \
  "7743b0fc1e77349378b06863c0506c3fa34cf98c66819a8f2db5dcbee19b70b8" \
  27 \
  252562

copy_one ".scratch/exp-648-structured-trajectory-consistency/screen/augmented-repeat-cache/root-20261160/repeat-1-3207e01b0318af83234079e8488a9431c66fa69e174dd9ad449bf798f1525788.json" "98a69670a229c15de7193c5878571fcc28a6d013e9b3ae23a7ac52be92624d80"
copy_one ".scratch/exp-648-structured-trajectory-consistency/screen/augmented-repeat-cache/root-20261160/repeat-2-0676d6cbc44c56f65daeb35d5d26d757047bbb9f937c12805a387bf68b16d59e.json" "8e8115d2b41974c20adfad734a847ec2ac64e07f5636325301b8793ffb1adc68"
copy_one ".scratch/exp-648-structured-trajectory-consistency/screen/readiness-calibration-attempts/cal-9e1f63a9fe3c6613/augmented-chronology/01-repeat-1-seal.json" "d0f48daff1ceb20adca98d255d3350a9384aa448f92e732f282d9243db36bbac"
copy_one ".scratch/exp-648-structured-trajectory-consistency/screen/readiness-calibration-attempts/cal-9e1f63a9fe3c6613/augmented-chronology/02-repeat-2-seal.json" "4911baa37cc3c16a67533effd23f228dc1b62077ce9a1958ed33a73093aabd20"
copy_one ".scratch/exp-648-structured-trajectory-consistency/screen/readiness-calibration-attempts/cal-9e1f63a9fe3c6613/augmented-units/01-repeat-1/augmented-unit.json" "eb107e8ba481738dfc4ea87c722ef1b3e31eb67ebb36416da7a947802dddc80a"
copy_one ".scratch/exp-648-structured-trajectory-consistency/screen/readiness-calibration-attempts/cal-9e1f63a9fe3c6613/augmented-units/02-repeat-2/augmented-unit.json" "c00bcc33473c0cd641910a4ccaa06d0e77fa1a34388117d55c0977ad52fb9b29"

sort -- "$inventory" >"$sorted_inventory"
if [[ "$(wc -l <"$inventory")" -ne 264 || "$(uniq "$sorted_inventory" | wc -l)" -ne 264 ]]; then
  echo "unexpected materialized inventory cardinality" >&2
  exit 1
fi

total_bytes=0
while IFS= read -r line; do
  relative=${line:66}
  total_bytes=$((total_bytes + $(stat -c '%s' -- "$destination/$relative")))
done <"$sorted_inventory"

printf 'schema_version=exp655-t049-portability-execution-v1\n'
printf 'source_root=%s\n' "$primary"
printf 'destination_root=%s\n' "$destination"
printf 'materialized_files=264\n'
printf 'materialized_bytes=%s\n' "$total_bytes"
printf 'copied_files=%s\n' "$copied"
printf 'reused_files=%s\n' "$reused"
printf 'inventory_sha256=%s\n' "$(sha256 "$sorted_inventory")"
printf 'regular_files=true\nsource_destination_inode_distinct=true\nreflink_disabled=true\n'
