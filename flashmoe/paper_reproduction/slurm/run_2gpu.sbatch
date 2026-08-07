#!/bin/bash
#SBATCH -J flashmoe-paper-2gpu
#SBATCH -p asus_pro6000
#SBATCH -q pro6000_qos
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:RTXPRO6000:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00

set -euo pipefail
export NUM_GPUS=2
PAPER_DIR=${PAPER_DIR:-/home/hybyun0207/aso-internship/flashmoe/paper_reproduction}
exec "$PAPER_DIR/slurm/run_job.sh"
