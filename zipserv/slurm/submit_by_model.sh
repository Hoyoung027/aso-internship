#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
RESULT_ROOT=${ZIP_MODEL_RESULT_ROOT:-$EXP_ROOT/results/by-model}

ALL_MODELS=(
  llama3.1-8b
  llama3.1-70b
  llama3.1-405b
  qwen2.5-7b
  qwen2.5-14b
  qwen2.5-32b
  qwen2.5-72b
  gemma3-12b
  gemma3-27b
  mistral-24b
  mistral-123b
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${ALL_MODELS[@]}")
fi

mkdir -p "$RESULT_ROOT"

for model in "${MODELS[@]}"; do
  known=false
  for candidate in "${ALL_MODELS[@]}"; do
    if [[ "$model" == "$candidate" ]]; then
      known=true
      break
    fi
  done
  if [[ "$known" != true ]]; then
    echo "Unknown model: $model" >&2
    exit 1
  fi

  output_dir=$RESULT_ROOT/$model
  echo "Submitting $model -> $output_dir"
  sbatch "$SCRIPT_DIR/run_zipserv.sbatch" all \
    --models "$model" \
    --output-dir "$output_dir"
done
