#!/bin/bash
#SBATCH -J gptoss-moe-kernels
#SBATCH -p asus_pro6000
#SBATCH -q pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --time=0-14:00:00
#SBATCH --output=/home/hybyun0207/aso-internship/vLLM/MoE/results/slurm-%j.out

set -euo pipefail

PROJECT_DIR=/home/hybyun0207/aso-internship/vLLM/MoE
EXP_ROOT=${MOE_EXP_ROOT:-/lustre/hybyun0207/vllm023-moe}
VLLM_ENV=$EXP_ROOT/env
VLLM_SRC=$EXP_ROOT/src/vllm
MODEL_PATH=$EXP_ROOT/models/gpt-oss-20b
FLASHINFER_TOOLKIT_ROOT=$EXP_ROOT/cuda/13.0-full
FLASHINFER_CUDA_HOME=$FLASHINFER_TOOLKIT_ROOT/nvidia/cu13
FLASHINFER_WORKSPACE_BASE=$EXP_ROOT/flashinfer-cuda130-full
RUN_ROOT=${MOE_RUN_ROOT:-$EXP_ROOT/runs/job-$SLURM_JOB_ID}
CONFIG=${MOE_CONFIG:-$PROJECT_DIR/configs/experiments.yaml}

source "$PROJECT_DIR/setup_runtime_env.sh"

PYTHON_BIN="$VLLM_ENV/bin/python"
VLLM_BIN="$VLLM_ENV/bin/vllm"

if [[ ! -x "$PYTHON_BIN" || ! -x "$VLLM_BIN" ]]; then
  echo "Invalid vLLM environment: $VLLM_ENV" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Missing model config: $MODEL_PATH/config.json" >&2
  exit 1
fi

echo "START_DATE=$(date --iso-8601=seconds)"
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RUN_ROOT=$RUN_ROOT"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_NVCC_RELEASE=$CUDA_NVCC_RELEASE"
echo "CUDA_PTXAS_RELEASE=$CUDA_PTXAS_RELEASE"
echo "CUDA_HEADER_RELEASE=$CUDA_HEADER_RELEASE"
echo "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"

nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,power.limit \
  --format=csv
"$CUDA_HOME/bin/nvcc" --version
"$PYTHON_BIN" - <<'PY'
from flashinfer.jit.cpp_ext import get_cuda_path, get_cuda_version

cuda_version = get_cuda_version()
print("FlashInfer CUDA path:", get_cuda_path())
print("FlashInfer CUDA version:", cuda_version)
PY

export MODEL_PATH RUN_ROOT

mkdir -p \
  "$RUN_ROOT" \
  "$RUN_ROOT/metadata" \
  "$RUN_ROOT/manifests" \
  "$TRITON_CACHE_DIR" \
  "$PROJECT_DIR/results"

RUN_ARGS=(
  --config "$CONFIG"
  --model-path "$MODEL_PATH"
  --run-root "$RUN_ROOT"
  --vllm-source "$VLLM_SRC"
  --vllm-bin "$VLLM_BIN"
)

if [[ -n "${MOE_BACKENDS:-}" ]]; then
  RUN_ARGS+=(--backends "$MOE_BACKENDS")
fi
if [[ -n "${NUM_TOKENS:-}" ]]; then
  RUN_ARGS+=(--num-tokens "$NUM_TOKENS")
fi
if [[ -n "${WARMUP:-}" ]]; then
  RUN_ARGS+=(--warmup "$WARMUP")
fi
if [[ -n "${REPEAT:-}" ]]; then
  RUN_ARGS+=(--repeat "$REPEAT")
fi

RUN_STATUS=0
"$PYTHON_BIN" "$PROJECT_DIR/run_backend_experiments.py" "${RUN_ARGS[@]}" \
  || RUN_STATUS=$?

ANALYZE_STATUS=0
"$PYTHON_BIN" "$PROJECT_DIR/analyze_moe_traces.py" \
  --metadata-dir "$RUN_ROOT/metadata" \
  --output "$RUN_ROOT/moe_kernel_results.csv" \
  || ANALYZE_STATUS=$?

if [[ -f "$RUN_ROOT/moe_kernel_results.csv" ]]; then
  cp \
    "$RUN_ROOT/moe_kernel_results.csv" \
    "$PROJECT_DIR/results/moe_kernel_results-${SLURM_JOB_ID}.csv"
fi

PLOT_STATUS=0
PLOT_DIR=${MOE_PLOT_DIR:-$PROJECT_DIR/results/plots/job-$SLURM_JOB_ID}
if (( ANALYZE_STATUS == 0 )); then
  "$PYTHON_BIN" "$PROJECT_DIR/visualize_moe_results.py" \
    --input "$RUN_ROOT/moe_kernel_results.csv" \
    --output-dir "$PLOT_DIR" \
    || PLOT_STATUS=$?
else
  echo "Skipping plots because trace analysis failed."
fi

echo "RUN_STATUS=$RUN_STATUS"
echo "ANALYZE_STATUS=$ANALYZE_STATUS"
echo "PLOT_STATUS=$PLOT_STATUS"
echo "PLOT_DIR=$PLOT_DIR"
echo "END_DATE=$(date --iso-8601=seconds)"

if (( RUN_STATUS != 0 )); then
  exit "$RUN_STATUS"
fi
if (( ANALYZE_STATUS != 0 )); then
  exit "$ANALYZE_STATUS"
fi
exit "$PLOT_STATUS"
