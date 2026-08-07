#!/bin/bash
# Source this file on an RTX PRO 6000 compute node.

GEMM_PROJECT_DIR=${GEMM_PROJECT_DIR:-/home/hybyun0207/aso-internship/gemm}
GEMM_ROOT=${GEMM_ROOT:-/lustre/hybyun0207/gptoss-gemm}
GEMM_ENV=${GEMM_ENV:-/lustre/hybyun0207/envs/gemm}
GEMM_RESULT_ROOT=${GEMM_RESULT_ROOT:-$GEMM_PROJECT_DIR/result}
FLASHINFER_TOOLKIT_ROOT=${FLASHINFER_TOOLKIT_ROOT:-/lustre/hybyun0207/vllm023-moe/cuda/13.0-full}
CUDA_HOME=${CUDA_HOME:-$FLASHINFER_TOOLKIT_ROOT/nvidia/cu13}
CUDA_DRIVER_STUB_DIR=${CUDA_DRIVER_STUB_DIR:-/opt/ohpc/pub/apps/cuda/12.8/lib64/stubs}

for executable in \
  "$GEMM_ENV/bin/python" \
  "$CUDA_HOME/bin/nvcc" \
  "$CUDA_HOME/bin/ptxas"; do
  if [[ ! -x "$executable" ]]; then
    echo "Missing executable: $executable" >&2
    return 1 2>/dev/null || exit 1
  fi
done

for required in \
  "$CUDA_HOME/include/cuda.h" \
  "$CUDA_HOME/lib/libcudart.so" \
  "$CUDA_DRIVER_STUB_DIR/libcuda.so"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing CUDA file: $required" >&2
    return 1 2>/dev/null || exit 1
  fi
done

export GEMM_PROJECT_DIR GEMM_ROOT GEMM_ENV GEMM_RESULT_ROOT
export FLASHINFER_TOOLKIT_ROOT CUDA_HOME CUDA_DRIVER_STUB_DIR
export PATH="$GEMM_ENV/bin:$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_DRIVER_STUB_DIR:${LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export FLASHINFER_EXTRA_LDFLAGS="-L$CUDA_HOME/lib -L$CUDA_DRIVER_STUB_DIR"

export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$GEMM_ROOT/cache/flashinfer}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$GEMM_ROOT/cache/torch-extensions}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$GEMM_ROOT/cache/cuda}
export AUTOTUNE_CACHE_DIR=${AUTOTUNE_CACHE_DIR:-$GEMM_ROOT/cache/autotune}
export FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0
export MAX_JOBS=${MAX_JOBS:-16}

mkdir -p \
  "$FLASHINFER_WORKSPACE_BASE" \
  "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" \
  "$AUTOTUNE_CACHE_DIR" \
  "$GEMM_RESULT_ROOT"
