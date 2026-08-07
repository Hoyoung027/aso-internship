#!/bin/bash

# Submit short FlashMoE correctness/runtime checks at 1, 2, and 4 GPUs.
# The 4-GPU case uses two gigabyte_pro6000 nodes with two GPUs per node.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/hybyun0207/aso-internship/flashmoe}
SBATCH_SCRIPT=$PROJECT_DIR/slurm/run_benchmark.sbatch
PARTITION=${PARTITION:-gigabyte_pro6000}
QOS=${QOS:-pro6000_qos}
GPU_GRES=${GPU_GRES:-RTXPRO6000}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
TIME_LIMIT=${TIME_LIMIT:-0-01:00:00}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/results"

submit_smoke() {
  local gpu_count=$1
  local nodes=$2
  local tasks_per_node=$3
  local dependency=${4:-}
  local -a dependency_arg=()

  if [[ -n "$dependency" ]]; then
    # Serialize jobs so they do not configure/build in the shared build
    # directory concurrently. afterany still tests larger sizes if an earlier
    # runtime smoke check fails.
    dependency_arg=(--dependency="afterany:$dependency")
  fi

  sbatch --parsable \
    --job-name="flashmoe-smoke-${gpu_count}g" \
    --output="$PROJECT_DIR/results/slurm-${RUN_STAMP}-${gpu_count}gpu-%j.out" \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --nodes="$nodes" \
    --ntasks="$gpu_count" \
    --ntasks-per-node="$tasks_per_node" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --gres="gpu:${GPU_GRES}:${tasks_per_node}" \
    --time="$TIME_LIMIT" \
    "${dependency_arg[@]}" \
    --export="ALL,NUM_GPUS=$gpu_count,SMOKE_ONLY=1" \
    "$SBATCH_SCRIPT"
}

job_1=$(submit_smoke 1 1 1)
job_1=${job_1%%;*}
job_2=$(submit_smoke 2 1 2 "$job_1")
job_2=${job_2%%;*}
job_4=$(submit_smoke 4 2 2 "$job_2")
job_4=${job_4%%;*}

echo "Submitted FlashMoE scaling smoke tests:"
echo "  1 GPU: job $job_1"
echo "    log: $PROJECT_DIR/results/slurm-${RUN_STAMP}-1gpu-${job_1}.out"
echo "  2 GPU: job $job_2 (after job $job_1)"
echo "    log: $PROJECT_DIR/results/slurm-${RUN_STAMP}-2gpu-${job_2}.out"
echo "  4 GPU: job $job_4 (2 nodes, after job $job_2)"
echo "    log: $PROJECT_DIR/results/slurm-${RUN_STAMP}-4gpu-${job_4}.out"
echo "Run data:  $PROJECT_DIR/results/job-<job-id>/"
