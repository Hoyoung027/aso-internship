#!/usr/bin/env bash

_ZIP_SETUP_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ZIP_EXP_ROOT=$(cd -- "$_ZIP_SETUP_SCRIPT_DIR/.." && pwd)
export ZIP_CONFIG=${ZIP_CONFIG:-$ZIP_EXP_ROOT/configs/experiments.json}
export ZIP_ROOT=${ZIP_ROOT:-/home/hybyun0207/ZipServ_ASPLOS26}
export CUDA_PATH=${CUDA_PATH:-/opt/ohpc/pub/apps/cuda/12.8}
export CUDA_HOME=$CUDA_PATH
export LInfer_HOME=$ZIP_ROOT
export MY_PATH=$ZIP_ROOT

export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ZIP_EXP_ROOT/bin:$CUDA_PATH/lib64"
unset LD_PRELOAD

echo "ZIP_EXP_ROOT=$ZIP_EXP_ROOT"
echo "ZIP_ROOT=$ZIP_ROOT"
echo "CUDA_PATH=$CUDA_PATH"
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-UNSET}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"

unset _ZIP_SETUP_SCRIPT_DIR
