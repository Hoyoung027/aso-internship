#!/bin/bash

set -euo pipefail

PAPER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_DIR=$(cd "$PAPER_DIR/.." && pwd)
LOG_DIR=$PROJECT_DIR/results/paper-reproduction/slurm
PARTITION=${PARTITION:-asus_pro6000}
QOS=${QOS:-pro6000_qos}
stamp=$(date +%Y%m%d-%H%M%S)

mkdir -p "$LOG_DIR"

for gpu_count in 2 4 8; do
  log_path=$LOG_DIR/slurm-${stamp}-${PARTITION}-${gpu_count}gpu-%j.out
  submission=$(sbatch --parsable \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --output="$log_path" \
    "$PAPER_DIR/slurm/run_${gpu_count}gpu.sbatch")
  job_id=${submission%%;*}
  echo "partition=$PARTITION ${gpu_count}gpu job=$job_id log=${log_path//%j/$job_id}"
done
