#!/usr/bin/env bash
#SBATCH --job-name=zipserv
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=/home/hybyun0207/aso-internship/zipserv/results/slurm-%j.out

set -euo pipefail

EXP_ROOT=/home/hybyun0207/aso-internship/zipserv
MODE=${1:-all}
if [[ $# -gt 0 ]]; then
  shift
fi

source "$EXP_ROOT/scripts/setup_runtime.sh"

if [[ ! -x "$EXP_ROOT/bin/test_mm_tune" || ! -x "$EXP_ROOT/bin/test_mm_final" ]]; then
  echo "Benchmark binaries are missing." >&2
  echo "Run $EXP_ROOT/scripts/build_benchmarks.sh once before submitting model jobs." >&2
  exit 1
fi

python3 "$EXP_ROOT/scripts/run_experiments.py" \
  --config "$EXP_ROOT/configs/experiments.json" \
  --mode "$MODE" \
  "$@"
