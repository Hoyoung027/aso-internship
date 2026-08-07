#!/bin/bash
#SBATCH -J flashmoe-bench
#SBATCH -p gigabyte_pro6000
#SBATCH -q pro6000_qos
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:RTXPRO6000:2
#SBATCH --cpus-per-task=8
#SBATCH --time=0-05:00:00
#SBATCH --output=/home/hybyun0207/aso-internship/flashmoe/results/slurm-%j.out

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/hybyun0207/aso-internship/flashmoe}
FLASHMOE_SOURCE=${FLASHMOE_SOURCE:-/home/hybyun0207/FlashMoE}
BUILD_DIR=${BUILD_DIR:-$PROJECT_DIR/build-sm120-generic}
RESULT_DIR=${RESULT_DIR:-$PROJECT_DIR/results/job-${SLURM_JOB_ID}}

# FlashMoE launches one MPI/NVSHMEM PE per GPU.  SLURM_NTASKS is the
# allocation-wide rank count, while SLURM_GPUS_ON_NODE is only per-node and
# therefore cannot represent a multi-node 4-GPU allocation.
NUM_GPUS=${NUM_GPUS:-${SLURM_NTASKS:-2}}
SMOKE_ONLY=${SMOKE_ONLY:-0}
TOKENS_PER_RANKS=${TOKENS_PER_RANKS:-"1024 2048 4096 8192"}
TOKEN_DIM=${TOKEN_DIM:-5120}
FFN_DIM=${FFN_DIM:-8192}
NUM_EXPERTS=${NUM_EXPERTS:-16}
TOP_K=${TOP_K:-1}
GRAPH_LAUNCHES=${GRAPH_LAUNCHES:-0}
WARMUP=${WARMUP:-20}
RUNS=${RUNS:-50}
OUTER_REPEATS=${OUTER_REPEATS:-5}
RTOL=${RTOL:-8e-2}
ATOL=${ATOL:-8e-3}

SMOKE_TOKENS=${SMOKE_TOKENS:-128}
SMOKE_WARMUP=${SMOKE_WARMUP:-2}
SMOKE_RUNS=${SMOKE_RUNS:-3}

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  fi
}

require_nonnegative_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a nonnegative integer, got: $value" >&2
    exit 2
  fi
}

require_positive_integer NUM_GPUS "$NUM_GPUS"
require_nonnegative_integer SMOKE_ONLY "$SMOKE_ONLY"
if (( SMOKE_ONLY > 1 )); then
  echo "SMOKE_ONLY must be 0 or 1, got: $SMOKE_ONLY" >&2
  exit 2
fi

if [[ -n "${SLURM_NTASKS:-}" ]] && (( NUM_GPUS != SLURM_NTASKS )); then
  echo "FlashMoE requires one Slurm task per GPU." >&2
  echo "NUM_GPUS=$NUM_GPUS but SLURM_NTASKS=$SLURM_NTASKS" >&2
  echo "Submit with --ntasks=$NUM_GPUS (and matching GPU resources)." >&2
  exit 2
fi

require_positive_integer TOKEN_DIM "$TOKEN_DIM"
require_positive_integer FFN_DIM "$FFN_DIM"
require_positive_integer NUM_EXPERTS "$NUM_EXPERTS"
require_positive_integer TOP_K "$TOP_K"
require_nonnegative_integer GRAPH_LAUNCHES "$GRAPH_LAUNCHES"
require_nonnegative_integer WARMUP "$WARMUP"
require_positive_integer RUNS "$RUNS"
require_positive_integer OUTER_REPEATS "$OUTER_REPEATS"
require_positive_integer SMOKE_TOKENS "$SMOKE_TOKENS"
require_nonnegative_integer SMOKE_WARMUP "$SMOKE_WARMUP"
require_positive_integer SMOKE_RUNS "$SMOKE_RUNS"

if (( NUM_EXPERTS % NUM_GPUS != 0 )); then
  echo "NUM_EXPERTS=$NUM_EXPERTS must be divisible by NUM_GPUS=$NUM_GPUS" >&2
  exit 2
fi
if (( TOP_K > NUM_EXPERTS )); then
  echo "TOP_K=$TOP_K must not exceed NUM_EXPERTS=$NUM_EXPERTS" >&2
  exit 2
fi

read -r -a TOKEN_LIST <<< "$TOKENS_PER_RANKS"
if (( ${#TOKEN_LIST[@]} == 0 )); then
  echo "TOKENS_PER_RANKS must contain at least one value" >&2
  exit 2
fi
for tokens in "${TOKEN_LIST[@]}"; do
  require_positive_integer TOKENS_PER_RANK "$tokens"
done

mkdir -p "$RESULT_DIR/raw"

module purge
module load gnu12/12.2.0
module load openmpi4/4.1.5
module load cuda/12.8
source "$PROJECT_DIR/setup_runtime_env.sh"

for executable in "$FLASHMOE_REAL_CMAKE" mpirun ninja; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable not found: $executable" >&2
    exit 1
  fi
done

TILE_HEADER=$FLASHMOE_SOURCE/csrc/include/flashmoe/tile.cuh
FLASHMOE_TEST_SOURCE=$FLASHMOE_SOURCE/csrc/tests/flashmoe.cu
if [[ ! -f "$TILE_HEADER" ]]; then
  echo "Missing FlashMoE tile header: $TILE_HEADER" >&2
  exit 1
fi
generic_patch_count=$(grep -c 'Arch != 1200' "$TILE_HEADER" || true)
if (( generic_patch_count < 3 )); then
  echo "SM120 generic patch was not found in all three cuBLASDx selections." >&2
  echo "Expected at least 3 occurrences of 'Arch != 1200' in $TILE_HEADER" >&2
  exit 1
fi
if [[ ! -f "$FLASHMOE_TEST_SOURCE" ]]; then
  echo "Missing FlashMoE test source: $FLASHMOE_TEST_SOURCE" >&2
  exit 1
fi
sm120_ps2_patch_count=$(grep -c 'arch == 1200 ? 2' "$FLASHMOE_TEST_SOURCE" || true)
if (( sm120_ps2_patch_count < 1 )); then
  echo "SM120 pS=2 shared-memory patch was not found in $FLASHMOE_TEST_SOURCE" >&2
  exit 1
fi

echo "START_DATE=$(date --iso-8601=seconds)"
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "NUM_GPUS=$NUM_GPUS"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_TASKS_PER_NODE=${SLURM_TASKS_PER_NODE:-}"
echo "SMOKE_ONLY=$SMOKE_ONLY"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "FLASHMOE_SOURCE=$FLASHMOE_SOURCE"
echo "BUILD_DIR=$BUILD_DIR"
echo "RESULT_DIR=$RESULT_DIR"
echo "TOKENS_PER_RANKS=$TOKENS_PER_RANKS"
echo "TOKEN_DIM=$TOKEN_DIM"
echo "FFN_DIM=$FFN_DIM"
echo "NUM_EXPERTS=$NUM_EXPERTS"
echo "TOP_K=$TOP_K"
echo "GRAPH_LAUNCHES=$GRAPH_LAUNCHES"
echo "WARMUP=$WARMUP"
echo "RUNS=$RUNS"
echo "OUTER_REPEATS=$OUTER_REPEATS"
echo "SM120_GENERIC_PATCH_COUNT=$generic_patch_count"
echo "SM120_PIPELINE_STAGES=2"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv
nvidia-smi topo -m
"$CUDA_HOME/bin/nvcc" --version
mpirun --version | head -n 3
"$FLASHMOE_REAL_CMAKE" --version | head -n 1

{
  echo "start_date=$(date --iso-8601=seconds)"
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "partition=$SLURM_JOB_PARTITION"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "num_gpus=$NUM_GPUS"
  echo "slurm_ntasks=${SLURM_NTASKS:-}"
  echo "slurm_nnodes=${SLURM_NNODES:-}"
  echo "slurm_tasks_per_node=${SLURM_TASKS_PER_NODE:-}"
  echo "smoke_only=$SMOKE_ONLY"
  echo "tokens_per_rank=$TOKENS_PER_RANKS"
  echo "token_dim=$TOKEN_DIM"
  echo "ffn_dim=$FFN_DIM"
  echo "num_experts=$NUM_EXPERTS"
  echo "top_k=$TOP_K"
  echo "graph_launches=$GRAPH_LAUNCHES"
  echo "warmup=$WARMUP"
  echo "runs=$RUNS"
  echo "outer_repeats=$OUTER_REPEATS"
  echo "cublasdx_modifier=generic"
  echo "sm120_pipeline_stages=2"
  echo "flashmoe_commit=$(git -C "$FLASHMOE_SOURCE" rev-parse HEAD)"
  echo "tile_header_sha256=$(sha256sum "$TILE_HEADER" | awk '{print $1}')"
  echo "flashmoe_test_sha256=$(sha256sum "$FLASHMOE_TEST_SOURCE" | awk '{print $1}')"
  echo "cuda_version=$("$CUDA_HOME/bin/nvcc" --version | tail -n 1)"
  echo "mpi_version=$(mpirun --version | head -n 1)"
} > "$RESULT_DIR/environment.txt"

echo "Configuring local patched FlashMoE source"
"$FLASHMOE_REAL_CMAKE" \
  -S "$FLASHMOE_SOURCE/csrc" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

echo "Building testFlashMoE"
"$FLASHMOE_REAL_CMAKE" \
  --build "$BUILD_DIR" \
  --target testFlashMoE \
  --parallel "$SLURM_CPUS_PER_TASK"

BENCHMARK_BIN=$BUILD_DIR/testFlashMoE
if [[ ! -x "$BENCHMARK_BIN" ]]; then
  echo "Benchmark binary was not created: $BENCHMARK_BIN" >&2
  exit 1
fi

export NVSHMEM_BOOTSTRAP=MPI

echo "Running smoke test"
mpirun -np "$NUM_GPUS" --bind-to none --map-by slot \
  "$BENCHMARK_BIN" \
  "$SMOKE_TOKENS" "$TOKEN_DIM" "$FFN_DIM" "$NUM_EXPERTS" "$TOP_K" \
  0 "$SMOKE_WARMUP" "$SMOKE_RUNS" "$RTOL" "$ATOL" \
  2>&1 | tee "$RESULT_DIR/smoke.log"

if (( SMOKE_ONLY == 1 )); then
  echo "SMOKE_TEST=PASS"
  echo "END_DATE=$(date --iso-8601=seconds)"
  exit 0
fi

CSV_PATH=$RESULT_DIR/benchmark.csv
echo 'outer_repeat,rank,dtype,S,H,I,E,k,FlashMoE_Time(ms),error(%),EC,GPU,SM,Topology,MLPType,bM,bN0,bK0,bN1,bK1,threads,blocks/SM,SMs,blocks,rtol,atol,graph_launches,warmup,runs,workspace(MiB)' > "$CSV_PATH"

for tokens in "${TOKEN_LIST[@]}"; do
  for ((outer_repeat = 1; outer_repeat <= OUTER_REPEATS; outer_repeat++)); do
    raw_log=$RESULT_DIR/raw/m${tokens}-repeat${outer_repeat}.log
    echo "Benchmark tokens_per_rank=$tokens outer_repeat=$outer_repeat/$OUTER_REPEATS"

    mpirun -np "$NUM_GPUS" --bind-to none --map-by slot \
      "$BENCHMARK_BIN" \
      "$tokens" "$TOKEN_DIM" "$FFN_DIM" "$NUM_EXPERTS" "$TOP_K" \
      "$GRAPH_LAUNCHES" "$WARMUP" "$RUNS" "$RTOL" "$ATOL" \
      2>&1 | tee "$raw_log"

    awk -v outer_repeat="$outer_repeat" \
      '/^[0-9]+, / { print outer_repeat "," $0 }' \
      "$raw_log" >> "$CSV_PATH"
  done
done

echo "BENCHMARK_CSV=$CSV_PATH"
echo "END_DATE=$(date --iso-8601=seconds)"
