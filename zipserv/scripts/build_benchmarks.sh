#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG=${ZIP_CONFIG:-$EXP_ROOT/configs/experiments.json}

readarray -t SETTINGS < <(
  python3 - "$CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
print(config["paths"]["zipserv_source"])
print(config["paths"]["cuda_path"])
print(config["gpu"]["sm"])
print(config["phases"]["tune"]["warmup"])
print(config["phases"]["tune"]["repeat"])
print(config["phases"]["final"]["warmup"])
print(config["phases"]["final"]["repeat"])
PY
)

ZIP_ROOT=${ZIP_ROOT:-${SETTINGS[0]}}
CUDA_PATH=${CUDA_PATH:-${SETTINGS[1]}}
SMS=${SMS:-${SETTINGS[2]}}
TUNE_WARMUP=${SETTINGS[3]}
TUNE_REPEAT=${SETTINGS[4]}
FINAL_WARMUP=${SETTINGS[5]}
FINAL_REPEAT=${SETTINGS[6]}

BENCH_DIR=$ZIP_ROOT/kernel_benchmark
UTILS_HEADER=$BENCH_DIR/utils.h
BIN_DIR=$EXP_ROOT/bin

for required in "$CUDA_PATH/bin/nvcc" "$ZIP_ROOT/build/Makefile" \
  "$BENCH_DIR/Makefile" "$UTILS_HEADER"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$BIN_DIR"
HEADER_BACKUP=$(mktemp "$EXP_ROOT/.utils.h.backup.XXXXXX")
cp -p -- "$UTILS_HEADER" "$HEADER_BACKUP"

restore_header() {
  if [[ -f "$HEADER_BACKUP" ]]; then
    cp -p -- "$HEADER_BACKUP" "$UTILS_HEADER"
    rm -f -- "$HEADER_BACKUP"
  fi
}
trap restore_header EXIT INT TERM

set_iterations() {
  local warmup=$1
  local repeat=$2
  sed -i -E \
    -e "s/^#define WARM_UP_ITERATION .*/#define WARM_UP_ITERATION $warmup/" \
    -e "s/^#define BENCHMARK_ITERATION .*/#define BENCHMARK_ITERATION $repeat/" \
    "$UTILS_HEADER"

  rg -q "^#define WARM_UP_ITERATION $warmup$" "$UTILS_HEADER"
  rg -q "^#define BENCHMARK_ITERATION $repeat$" "$UTILS_HEADER"
}

build_variant() {
  local name=$1
  local warmup=$2
  local repeat=$3

  echo "Building $name benchmark (warmup=$warmup, repeat=$repeat, sm=$SMS)"
  set_iterations "$warmup" "$repeat"
  make -C "$BENCH_DIR" clean
  make -C "$BENCH_DIR" \
    CUDA_PATH="$CUDA_PATH" \
    MY_PATH="$ZIP_ROOT" \
    SMS="$SMS" \
    test_mm
  cp -- "$BENCH_DIR/test_mm" "$BIN_DIR/test_mm_$name"
}

echo "Building ZipServ shared library (CUDA=$CUDA_PATH, sm=$SMS)"
make -C "$ZIP_ROOT/build" clean
make -C "$ZIP_ROOT/build" \
  CUDA_PATH="$CUDA_PATH" \
  SMS="$SMS"
cp -- "$ZIP_ROOT/build/libL_API.so" "$BIN_DIR/libL_API.so"

if [[ "$TUNE_WARMUP" == "$FINAL_WARMUP" && "$TUNE_REPEAT" == "$FINAL_REPEAT" ]]; then
  build_variant tune "$TUNE_WARMUP" "$TUNE_REPEAT"
  cp -- "$BIN_DIR/test_mm_tune" "$BIN_DIR/test_mm_final"
  echo "Tune and final settings match; reused the same benchmark binary."
else
  build_variant tune "$TUNE_WARMUP" "$TUNE_REPEAT"
  build_variant final "$FINAL_WARMUP" "$FINAL_REPEAT"
fi

restore_header
trap - EXIT INT TERM

{
  printf 'config=%s\n' "$CONFIG"
  printf 'zipserv_source=%s\n' "$ZIP_ROOT"
  printf 'cuda_path=%s\n' "$CUDA_PATH"
  printf 'sm=%s\n' "$SMS"
  printf 'tune_warmup=%s\n' "$TUNE_WARMUP"
  printf 'tune_repeat=%s\n' "$TUNE_REPEAT"
  printf 'final_warmup=%s\n' "$FINAL_WARMUP"
  printf 'final_repeat=%s\n' "$FINAL_REPEAT"
  printf 'zipserv_commit=%s\n' "$(git -C "$ZIP_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf 'built_at=%s\n' "$(date --iso-8601=seconds)"
} > "$BIN_DIR/build_manifest.txt"

echo "Built:"
ls -lh "$BIN_DIR/test_mm_tune" "$BIN_DIR/test_mm_final" "$BIN_DIR/libL_API.so"
echo "The original $UTILS_HEADER was restored."
