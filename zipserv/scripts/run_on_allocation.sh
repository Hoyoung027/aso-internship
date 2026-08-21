#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup_runtime.sh"

MODE=${1:-run}
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Warning: SLURM_JOB_ID is unset. Run this on a GPU allocation." >&2
fi

exec python3 "$SCRIPT_DIR/run_experiments.py" \
  --config "$ZIP_CONFIG" \
  --mode "$MODE" \
  "$@"
