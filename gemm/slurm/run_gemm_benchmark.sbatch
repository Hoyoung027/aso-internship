#!/bin/bash
#SBATCH -J gptoss-gemm
#SBATCH -p asus_pro6000
#SBATCH -q pro6000_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTXPRO6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/hybyun0207/aso-internship/gemm/result/slurm-%j.out

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/hybyun0207/aso-internship/gemm}
CONFIG=${CONFIG:-$PROJECT_DIR/configs/experiments.yaml}
GEMM_ROOT=${GEMM_ROOT:-/lustre/hybyun0207/gptoss-gemm}
GEMM_ENV=${GEMM_ENV:-/lustre/hybyun0207/envs/gemm}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_DIR/result}
TRIALS=${TRIALS:-3}
RESULT_DIR=${RESULT_DIR:-$RESULT_ROOT/job-${SLURM_JOB_ID}}
AUTOTUNE_CACHE_DIR=${AUTOTUNE_CACHE_DIR:-$GEMM_ROOT/cache/autotune/job-${SLURM_JOB_ID}}
FORCE_RETUNE=${FORCE_RETUNE:-1}

if [[ ! "$TRIALS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIALS must be a positive integer, got: $TRIALS" >&2
  exit 2
fi
if [[ ! "$FORCE_RETUNE" =~ ^[01]$ ]]; then
  echo "FORCE_RETUNE must be 0 or 1, got: $FORCE_RETUNE" >&2
  exit 2
fi

module purge
export GEMM_ROOT GEMM_ENV GEMM_RESULT_ROOT="$RESULT_ROOT"
source "$PROJECT_DIR/setup_runtime_env.sh"

mkdir -p "$RESULT_DIR" "$AUTOTUNE_CACHE_DIR"
cp "$CONFIG" "$RESULT_DIR/experiments.yaml"

echo "START_DATE=$(date --iso-8601=seconds)"
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG=$CONFIG"
echo "GEMM_ENV=$GEMM_ENV"
echo "CUDA_HOME=$CUDA_HOME"
echo "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"
echo "AUTOTUNE_CACHE_DIR=$AUTOTUNE_CACHE_DIR"
echo "RESULT_DIR=$RESULT_DIR"
echo "TRIALS=$TRIALS"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv
"$CUDA_HOME/bin/nvcc" --version

"$GEMM_ENV/bin/python" - <<'PY' | tee "$RESULT_DIR/environment.txt"
from importlib.metadata import version
import torch
import flashinfer

print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("FlashInfer:", version("flashinfer-python"))
print("FlashInfer cubin:", version("flashinfer-cubin"))
print("cuDNN:", torch.backends.cudnn.version())
print("GPU:", torch.cuda.get_device_name())
print("Capability:", torch.cuda.get_device_capability())

assert torch.__version__.startswith("2.11.0")
assert torch.version.cuda == "13.0"
assert version("flashinfer-python") == "0.6.12"
assert torch.cuda.get_device_capability() == (12, 0)
assert (torch.backends.cudnn.version() or 0) >= 91400
PY

{
  echo "benchmark_sha256=$(sha256sum "$PROJECT_DIR/benchmark_gemm.py" | awk '{print $1}')"
  echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
  echo "cuda_home=$CUDA_HOME"
  echo "flashinfer_workspace_base=$FLASHINFER_WORKSPACE_BASE"
} >> "$RESULT_DIR/environment.txt"

COMMON=(
  --config "$CONFIG"
  --output-dir "$RESULT_DIR"
  --cache-dir "$AUTOTUNE_CACHE_DIR"
)

echo "Running BF16 smoke test; first FlashInfer JIT may take a long time"
"$GEMM_ENV/bin/python" "$PROJECT_DIR/benchmark_gemm.py" \
  "${COMMON[@]}" \
  --mode smoke \
  --precision bf16 \
  --m-values 128 \
  --warmup 2 \
  --repeat 3

echo "Running PyTorch MXFP8 compatibility smoke test"
"$GEMM_ENV/bin/python" "$PROJECT_DIR/benchmark_gemm.py" \
  --config "$CONFIG" \
  --output-dir "$RESULT_DIR/smoke" \
  --cache-dir "$AUTOTUNE_CACHE_DIR" \
  --mode torch \
  --precision mxfp8 \
  --m-values 128 \
  --warmup 2 \
  --repeat 3 \
  --fail-fast

echo "Running FlashInfer MXFP8 compatibility smoke test"
"$GEMM_ENV/bin/python" "$PROJECT_DIR/benchmark_gemm.py" \
  --config "$CONFIG" \
  --output-dir "$RESULT_DIR/smoke" \
  --cache-dir "$AUTOTUNE_CACHE_DIR" \
  --mode default \
  --precision mxfp8 \
  --m-values 128 \
  --warmup 2 \
  --repeat 3 \
  --fail-fast

for precision in bf16 mxfp8 mxfp4; do
  tune_args=()
  if (( FORCE_RETUNE == 1 )); then
    tune_args+=(--force-retune)
  fi
  echo "Tuning FlashInfer precision=$precision"
  "$GEMM_ENV/bin/python" "$PROJECT_DIR/benchmark_gemm.py" \
    "${COMMON[@]}" \
    --mode tune \
    --precision "$precision" \
    "${tune_args[@]}"
done

for ((trial = 1; trial <= TRIALS; trial++)); do
  for precision in bf16 mxfp8; do
    echo "Benchmark PyTorch precision=$precision trial=$trial/$TRIALS"
    "$GEMM_ENV/bin/python" "$PROJECT_DIR/benchmark_gemm.py" \
      "${COMMON[@]}" \
      --mode torch \
      --precision "$precision" \
      --trial "$trial"
  done

  for precision in bf16 mxfp8 mxfp4; do
    for mode in default tuned; do
      echo "Benchmark FlashInfer precision=$precision mode=$mode trial=$trial/$TRIALS"
      "$GEMM_ENV/bin/python" "$PROJECT_DIR/benchmark_gemm.py" \
        "${COMMON[@]}" \
        --mode "$mode" \
        --precision "$precision" \
        --trial "$trial"
    done
  done
done

"$GEMM_ENV/bin/python" "$PROJECT_DIR/benchmark_gemm.py" \
  "${COMMON[@]}" \
  --mode aggregate

echo "SUMMARY_CSV=$RESULT_DIR/summary.csv"
echo "END_DATE=$(date --iso-8601=seconds)"
