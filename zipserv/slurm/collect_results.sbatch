#!/usr/bin/env bash
#SBATCH --job-name=zip-collect
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

set -euo pipefail

EXP_ROOT=/home/hybyun0207/aso-internship/zipserv
MODE=$1
RUN_DIR=$2

python3 "$EXP_ROOT/scripts/collect_results.py" \
  --mode "$MODE" \
  --run-dir "$RUN_DIR" \
  --config "$EXP_ROOT/configs/experiments.json"
