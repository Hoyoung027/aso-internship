#!/bin/bash
#SBATCH -J torch-mxfp4-smoke
#SBATCH -p asus_pro6000
#SBATCH -q pro6000_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTXPRO6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/hybyun0207/aso-internship/gemm/result/torch-mxfp4-smoke-%j.out

set -euo pipefail

PROJECT_DIR=/home/hybyun0207/aso-internship/gemm
CONFIG=$PROJECT_DIR/configs/experiments.yaml
GEMM_ROOT=/lustre/hybyun0207/gptoss-gemm
GEMM_ENV=/lustre/hybyun0207/envs/gemm
RESULT_DIR=$PROJECT_DIR/result/torch-mxfp4-smoke-${SLURM_JOB_ID}
AUTOTUNE_CACHE_DIR=$GEMM_ROOT/cache/autotune/torch-mxfp4-smoke-${SLURM_JOB_ID}

export GEMM_ROOT GEMM_ENV GEMM_RESULT_ROOT=$PROJECT_DIR/result

module purge
source "$PROJECT_DIR/setup_runtime_env.sh"

mkdir -p "$RESULT_DIR" "$AUTOTUNE_CACHE_DIR"

echo "START_DATE=$(date --iso-8601=seconds)"
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv

"$GEMM_ENV/bin/python" "$PROJECT_DIR/benchmark_gemm.py" \
  --config "$CONFIG" \
  --output-dir "$RESULT_DIR" \
  --cache-dir "$AUTOTUNE_CACHE_DIR" \
  --mode torch \
  --precision mxfp4 \
  --m-values 128 \
  --warmup 2 \
  --repeat 3 \
  --fail-fast

echo "RESULT_DIR=$RESULT_DIR"
echo "END_DATE=$(date --iso-8601=seconds)"
