#!/bin/bash
# Source this file on a GPU node before running the FlashMoE experiment.

FLASHMOE_PROJECT_DIR=${FLASHMOE_PROJECT_DIR:-/home/hybyun0207/aso-internship/flashmoe}
FLASHMOE_SOURCE=${FLASHMOE_SOURCE:-/home/hybyun0207/FlashMoE}
FLASHMOE_CONDA_ENV=${FLASHMOE_CONDA_ENV:-/home/hybyun0207/miniconda3/envs/flashmoe}

CUDA_HOME=${CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}
MATHDX_ROOT=${MATHDX_ROOT:-/home/hybyun0207/opt/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12}
NVSHMEM_HOME=${NVSHMEM_HOME:-/home/hybyun0207/opt/libnvshmem-linux-x86_64-3.7.2_cuda12-archive}
NVSHMEM_LIB_HOME=${NVSHMEM_LIB_HOME:-$NVSHMEM_HOME/lib}

PYTHON_BIN=$FLASHMOE_CONDA_ENV/bin/python
TORCHRUN_BIN=$FLASHMOE_CONDA_ENV/bin/torchrun
FLASHMOE_REAL_CMAKE=$FLASHMOE_CONDA_ENV/bin/cmake
CMAKE_WRAPPER=$FLASHMOE_PROJECT_DIR/bin/cmake

for executable in \
  "$PYTHON_BIN" \
  "$TORCHRUN_BIN" \
  "$FLASHMOE_REAL_CMAKE" \
  "$CMAKE_WRAPPER" \
  "$CUDA_HOME/bin/nvcc"; do
  if [[ ! -x "$executable" ]]; then
    echo "Missing executable: $executable" >&2
    return 1 2>/dev/null || exit 1
  fi
done

for required in \
  "$FLASHMOE_SOURCE/quickstart.py" \
  "$MATHDX_ROOT/lib/cmake/mathdx/mathdx-config.cmake" \
  "$NVSHMEM_HOME/lib/cmake/nvshmem/NVSHMEMConfig.cmake" \
  "$NVSHMEM_LIB_HOME/libnvshmem_host.so" \
  "$NVSHMEM_LIB_HOME/libnvshmem_device.a"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing FlashMoE dependency: $required" >&2
    return 1 2>/dev/null || exit 1
  fi
done

export FLASHMOE_PROJECT_DIR FLASHMOE_SOURCE FLASHMOE_CONDA_ENV
export CUDA_HOME MATHDX_ROOT NVSHMEM_HOME NVSHMEM_LIB_HOME
export PYTHON_BIN TORCHRUN_BIN FLASHMOE_REAL_CMAKE CMAKE_WRAPPER
export PATH="$FLASHMOE_PROJECT_DIR/bin:$FLASHMOE_CONDA_ENV/bin:$CUDA_HOME/bin:${PATH:-}"
export PYTHONPATH="$FLASHMOE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export CMAKE_PREFIX_PATH="$NVSHMEM_HOME:$NVSHMEM_LIB_HOME:$MATHDX_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$NVSHMEM_LIB_HOME:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export FLASHMOE_CACHE_DIR=${FLASHMOE_CACHE_DIR:-/home/hybyun0207/.cache/flashmoe_jit}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/home/hybyun0207/.cache/nv/ComputeCache}
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-/home/hybyun0207/.cache/cpm}

mkdir -p \
  "$FLASHMOE_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$CPM_SOURCE_CACHE" \
  "$FLASHMOE_PROJECT_DIR/results"
