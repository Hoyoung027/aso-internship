#!/bin/bash

set -euo pipefail

PAPER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_DIR=$(cd "$PAPER_DIR/.." && pwd)
FLASHMOE_SOURCE=${FLASHMOE_SOURCE:-/home/hybyun0207/FlashMoE}
NUM_GPUS=${NUM_GPUS:-${SLURM_NTASKS:-}}
BUILD_DIR=${BUILD_DIR:-$PAPER_DIR/build-sm120-fp32}
RESULT_DIR=${RESULT_DIR:-$PROJECT_DIR/results/paper-reproduction/job-${SLURM_JOB_ID:-manual}-${NUM_GPUS}gpu}

if [[ ! "$NUM_GPUS" =~ ^(2|4|8)$ ]]; then
  echo "NUM_GPUS must be 2, 4, or 8; got: ${NUM_GPUS:-<empty>}" >&2
  exit 2
fi
if [[ -n "${SLURM_NTASKS:-}" ]] && (( SLURM_NTASKS != NUM_GPUS )); then
  echo "One MPI/NVSHMEM rank is required per GPU." >&2
  echo "SLURM_NTASKS=$SLURM_NTASKS, NUM_GPUS=$NUM_GPUS" >&2
  exit 2
fi
if [[ -n "${SLURM_NNODES:-}" ]] && (( SLURM_NNODES != 1 )); then
  echo "The paper protocol requires one node; SLURM_NNODES=$SLURM_NNODES" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"

finish() {
  status=$?
  trap - EXIT
  {
    echo "END_DATE=$(date --iso-8601=seconds)"
    echo "EXIT_CODE=$status"
  } | tee -a "$RESULT_DIR/job_status.txt"
  exit "$status"
}
trap finish EXIT

module purge
module load gnu12/12.2.0
module load openmpi4/4.1.5
module load cuda/12.8

export OMP_NUM_THREADS=1
export FLASHMOE_PROJECT_DIR=$PROJECT_DIR
export FLASHMOE_SOURCE
source "$PROJECT_DIR/setup_runtime_env.sh"

for executable in "$FLASHMOE_REAL_CMAKE" "$PYTHON_BIN" mpirun ninja flock; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable not found: $executable" >&2
    exit 1
  fi
done

visible_gpu_count=$(nvidia-smi -L | wc -l)
if (( visible_gpu_count != NUM_GPUS )); then
  echo "Visible GPU count must match the MPI rank count." >&2
  echo "nvidia-smi reports $visible_gpu_count GPUs, NUM_GPUS=$NUM_GPUS" >&2
  exit 2
fi

PAPER_SOURCE=$("$PAPER_DIR/prepare_source.sh")
SOURCE_MANIFEST=$PAPER_SOURCE/.paper_source_manifest
if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  echo "Prepared-source manifest is missing: $SOURCE_MANIFEST" >&2
  exit 1
fi
export FLASHMOE_PREPARED_SOURCE=$PAPER_SOURCE
export FLASHMOE_COMMIT
export FLASHMOE_PATCH_SHA256
FLASHMOE_COMMIT=$(awk -F= '$1 == "flashmoe_commit" {print $2}' "$SOURCE_MANIFEST")
FLASHMOE_PATCH_SHA256=$(awk -F= '$1 == "patch_sha256" {print $2}' "$SOURCE_MANIFEST")
if [[ -z "$FLASHMOE_COMMIT" || -z "$FLASHMOE_PATCH_SHA256" ]]; then
  echo "Prepared-source manifest is incomplete: $SOURCE_MANIFEST" >&2
  exit 1
fi

{
  echo "START_DATE=$(date --iso-8601=seconds)"
  echo "JOB_ID=${SLURM_JOB_ID:-manual}"
  echo "NODE=$(hostname)"
  echo "PARTITION=${SLURM_JOB_PARTITION:-manual}"
  echo "NUM_GPUS=$NUM_GPUS"
  echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
  echo "SLURM_NNODES=${SLURM_NNODES:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "PAPER_SOURCE=$PAPER_SOURCE"
  echo "BUILD_DIR=$BUILD_DIR"
  echo "RESULT_DIR=$RESULT_DIR"
  echo "DTYPE=fp32"
  echo "HIDDEN_SIZE=2048"
  echo "FFN_SIZE=2048"
  echo "TOP_K=2"
  echo "CAPACITY_FACTOR=1.0"
  echo "CUDA_GRAPH=off"
  echo "WARMUP=32"
  echo "RUNS=32"
  echo "MEASUREMENT_SCOPE=router_excluded_fused_distributed_moe_kernel"
  echo "SM120_CUBLASDX_MODIFIER=generic"
  echo "SM120_PIPELINE_STAGES=1"
} | tee "$RESULT_DIR/job_status.txt"

{
  echo "[allocation]"
  env | LC_ALL=C sort | grep -E '^(SLURM_|CUDA_VISIBLE_DEVICES=|NVSHMEM|OMPI_|CUDA_HOME=|MATHDX_ROOT=)' || true
  echo
  echo "[prepared source]"
  cat "$SOURCE_MANIFEST"
  echo
  echo "[software]"
  "$CUDA_HOME/bin/nvcc" --version
  mpirun --version | head -n 3
  "$FLASHMOE_REAL_CMAKE" --version | head -n 1
  "$PYTHON_BIN" --version
  echo
  echo "[gpu]"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv
} > "$RESULT_DIR/environment.txt"

nvidia-smi topo -m > "$RESULT_DIR/topology.txt"

exec 9>"$BUILD_DIR.lock"
flock 9
echo "Configuring isolated FP32 paper build: $BUILD_DIR"
"$FLASHMOE_REAL_CMAKE" \
  -S "$PAPER_SOURCE/csrc" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
echo "Building testFlashMoE"
BUILD_JOBS=${BUILD_JOBS:-$((NUM_GPUS * ${SLURM_CPUS_PER_TASK:-2}))}
"$FLASHMOE_REAL_CMAKE" \
  --build "$BUILD_DIR" \
  --target testFlashMoE \
  --parallel "$BUILD_JOBS"
flock -u 9

BENCHMARK_BIN=$BUILD_DIR/testFlashMoE
if [[ ! -x "$BENCHMARK_BIN" ]]; then
  echo "Benchmark binary was not created: $BENCHMARK_BIN" >&2
  exit 1
fi

export NVSHMEM_BOOTSTRAP=MPI
"$PYTHON_BIN" "$PAPER_DIR/run_paper_benchmark.py" \
  --binary "$BENCHMARK_BIN" \
  --world-size "$NUM_GPUS" \
  --result-dir "$RESULT_DIR"

echo "RESULT_DIR=$RESULT_DIR"
