#!/bin/bash
# Source this file before running the MoE benchmark on a GPU node.

# Project and storage defaults. Callers may override any of these before
# sourcing this file.
PROJECT_DIR=${PROJECT_DIR:-/home/hybyun0207/aso-internship/vLLM/MoE}
EXP_ROOT=${EXP_ROOT:-/lustre/hybyun0207/vllm023-moe}
VLLM_ENV=${VLLM_ENV:-$EXP_ROOT/env}
VLLM_SRC=${VLLM_SRC:-$EXP_ROOT/src/vllm}
MODEL_PATH=${MODEL_PATH:-$EXP_ROOT/models/gpt-oss-20b}

# RTX PRO 6000 is SM120. FlashInfer 0.6.12 requires CUDA >= 12.9 to
# normalize and JIT-compile an SM12.x target. The cluster-wide CUDA 12.8
# toolkit is therefore insufficient even though the PyTorch wheel is cu130.
# The vLLM environment can contain a mixed CUDA compiler/header set because
# FlashInfer's cuda-tile dependency and PyTorch resolve different CUDA minors.
# Use an isolated CUDA 13.0 toolkit prepared by setup_flashinfer_cuda.sh.
if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  echo "Missing vLLM Python environment: $VLLM_ENV/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi
FLASHINFER_TOOLKIT_ROOT=${FLASHINFER_TOOLKIT_ROOT:-$EXP_ROOT/cuda/13.0-full}
FLASHINFER_CUDA_HOME=${FLASHINFER_CUDA_HOME:-$FLASHINFER_TOOLKIT_ROOT/nvidia/cu13}
CUDA_DRIVER_STUB_DIR=${CUDA_DRIVER_STUB_DIR:-/opt/ohpc/pub/apps/cuda/12.8/lib64/stubs}

if [[ ! -x "$FLASHINFER_CUDA_HOME/bin/nvcc" ]]; then
  echo "Missing CUDA >= 12.9 nvcc: $FLASHINFER_CUDA_HOME/bin/nvcc" >&2
  echo "Run setup_flashinfer_cuda.sh on the login node." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -d "$FLASHINFER_CUDA_HOME/lib" ]]; then
  echo "Missing CUDA library directory: $FLASHINFER_CUDA_HOME/lib" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -e "$FLASHINFER_CUDA_HOME/lib64" ]]; then
  echo "Missing CUDA lib64 compatibility path: $FLASHINFER_CUDA_HOME/lib64" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -d "$CUDA_DRIVER_STUB_DIR" ]]; then
  echo "Missing CUDA driver stub directory: $CUDA_DRIVER_STUB_DIR" >&2
  return 1 2>/dev/null || exit 1
fi

CUDA_RUNTIME_HEADER="$FLASHINFER_CUDA_HOME/include/cuda_runtime_api.h"
if [[ ! -f "$CUDA_RUNTIME_HEADER" ]]; then
  echo "Missing CUDA runtime header: $CUDA_RUNTIME_HEADER" >&2
  return 1 2>/dev/null || exit 1
fi
for required in \
  include/cublasLt.h \
  include/curand_kernel.h \
  lib/libcudart.so \
  lib/libcublas.so \
  lib/libcublasLt.so \
  lib/libcurand.so; do
  if [[ ! -e "$FLASHINFER_CUDA_HOME/$required" ]]; then
    echo "Missing FlashInfer CUDA dependency: $FLASHINFER_CUDA_HOME/$required" >&2
    return 1 2>/dev/null || exit 1
  fi
done

CUDA_NVCC_RELEASE=$("$FLASHINFER_CUDA_HOME/bin/nvcc" --version | sed -n \
  's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' | head -n 1)
CUDA_PTXAS_RELEASE=$("$FLASHINFER_CUDA_HOME/bin/ptxas" --version | sed -n \
  's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' | head -n 1)
CUDA_CUDART_VERSION=$(awk \
  '$1 == "#define" && $2 == "CUDART_VERSION" { print $3; exit }' \
  "$CUDA_RUNTIME_HEADER")
CUDA_HEADER_RELEASE="$((CUDA_CUDART_VERSION / 1000)).$(((CUDA_CUDART_VERSION % 1000) / 10))"

if [[ -z "$CUDA_NVCC_RELEASE" || -z "$CUDA_PTXAS_RELEASE" || \
      "$CUDA_NVCC_RELEASE" != "$CUDA_HEADER_RELEASE" || \
      "$CUDA_PTXAS_RELEASE" != "$CUDA_HEADER_RELEASE" ]]; then
  echo "CUDA compiler/assembler/header mismatch: " \
    "nvcc=${CUDA_NVCC_RELEASE:-unknown} " \
    "ptxas=${CUDA_PTXAS_RELEASE:-unknown} headers=$CUDA_HEADER_RELEASE" >&2
  echo "Run setup_flashinfer_cuda.sh on the login node." >&2
  return 1 2>/dev/null || exit 1
fi
case "$CUDA_NVCC_RELEASE" in
  12.9|12.[1-9][0-9]|1[3-9].*) ;;
  *)
    echo "SM120 FlashInfer requires CUDA >= 12.9, got $CUDA_NVCC_RELEASE" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

NVVM_METADATA=$(find "$FLASHINFER_TOOLKIT_ROOT" -maxdepth 1 -type d \
  -name 'nvidia_nvvm-*.dist-info' -print -quit)
if [[ -z "$NVVM_METADATA" || $(basename "$NVVM_METADATA") != nvidia_nvvm-13.0.88.dist-info ]]; then
  echo "Expected nvidia-nvvm 13.0.88, found: ${NVVM_METADATA:-missing}" >&2
  return 1 2>/dev/null || exit 1
fi

export PROJECT_DIR MODEL_PATH EXP_ROOT
export FLASHINFER_TOOLKIT_ROOT FLASHINFER_CUDA_HOME CUDA_DRIVER_STUB_DIR
export CUDA_NVCC_RELEASE CUDA_PTXAS_RELEASE CUDA_HEADER_RELEASE
export CUDA_HOME="$FLASHINFER_CUDA_HOME"
export PATH="$VLLM_ENV/bin:$CUDA_HOME/bin:${PATH:-}"
if ! command -v ninja >/dev/null 2>&1; then
  echo "Missing ninja executable in the vLLM environment: $VLLM_ENV/bin/ninja" >&2
  echo "Run setup_vllm023_editable.sh on the login node." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$CUDA_HOME/lib"
fi

# FlashInfer's generated ninja file searches CUDA_HOME/lib64, while the cu13
# Python package stores libraries in CUDA_HOME/lib. Add the real directory and
# the cluster driver stub directory explicitly without changing runtime library
# resolution for the rest of vLLM.
export FLASHINFER_EXTRA_LDFLAGS="-L$CUDA_HOME/lib -L$CUDA_DRIVER_STUB_DIR"
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$EXP_ROOT/flashinfer-cuda130-full}

export HF_HOME=${HF_HOME:-$EXP_ROOT/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$EXP_ROOT/cache/torch}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$EXP_ROOT/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/${USER}/triton-${SLURM_JOB_ID:-interactive}}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export GPTOSS_MOE_PROFILE_LAYER0=1

mkdir -p \
  "$FLASHINFER_WORKSPACE_BASE" \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR" \
  "$EXP_ROOT/runs"

unset PYTHONPYCACHEPREFIX
