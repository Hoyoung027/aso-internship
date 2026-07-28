#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_ROOT=${EXP_ROOT:-/lustre/hybyun0207/vllm023-moe}
VLLM_ENV=${VLLM_ENV:-$EXP_ROOT/env}
FLASHINFER_TOOLKIT_ROOT=${FLASHINFER_TOOLKIT_ROOT:-$EXP_ROOT/cuda/13.0-full}
CUDA_DRIVER_STUB_DIR=${CUDA_DRIVER_STUB_DIR:-/opt/ohpc/pub/apps/cuda/12.8/lib64/stubs}

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  echo "Missing vLLM Python environment: $VLLM_ENV/bin/python" >&2
  exit 1
fi

UV_BIN=${UV_BIN:-}
if [[ -z "$UV_BIN" ]] && command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
fi
if [[ -z "$UV_BIN" ]] && [[ -x "$EXP_ROOT/tools/bin/uv" ]]; then
  UV_BIN="$EXP_ROOT/tools/bin/uv"
fi
if [[ -z "$UV_BIN" ]] && [[ -x "$VLLM_ENV/bin/uv" ]]; then
  UV_BIN="$VLLM_ENV/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv is required to install the isolated CUDA toolkit." >&2
  exit 1
fi

mkdir -p "$FLASHINFER_TOOLKIT_ROOT"

# Install every CUDA component from the 13.0.2 metapackage. Installing only the
# nvcc extra lets nvidia-cuda-nvcc pull an unpinned, newer nvidia-nvvm package;
# that produced PTX 9.3 while the CUDA 13.0 ptxas only accepted PTX 9.0.
"$UV_BIN" pip install \
  --python "$VLLM_ENV/bin/python" \
  --target "$FLASHINFER_TOOLKIT_ROOT" \
  --reinstall \
  --link-mode copy \
  "cuda-toolkit[all]==13.0.2"

FLASHINFER_CUDA_HOME="$FLASHINFER_TOOLKIT_ROOT/nvidia/cu13"
NVCC="$FLASHINFER_CUDA_HOME/bin/nvcc"
PTXAS="$FLASHINFER_CUDA_HOME/bin/ptxas"
CUDA_RUNTIME_HEADER="$FLASHINFER_CUDA_HOME/include/cuda_runtime_api.h"

if [[ ! -x "$NVCC" || ! -x "$PTXAS" || ! -f "$CUDA_RUNTIME_HEADER" ]]; then
  echo "Incomplete isolated CUDA toolkit: $FLASHINFER_CUDA_HOME" >&2
  exit 1
fi
for header in cublasLt.h curand_kernel.h; do
  if [[ ! -f "$FLASHINFER_CUDA_HOME/include/$header" ]]; then
    echo "Missing CUDA development header: $header" >&2
    exit 1
  fi
done

# CUDA Python wheels install only versioned shared objects under lib/. Create
# the traditional toolkit names expected by FlashInfer's generated ninja file.
CUDA_LIB_DIR="$FLASHINFER_CUDA_HOME/lib"
ensure_library_link() {
  local target=$1
  local link=$2
  if [[ ! -e "$CUDA_LIB_DIR/$target" ]]; then
    echo "Missing CUDA library: $CUDA_LIB_DIR/$target" >&2
    exit 1
  fi
  if [[ ! -e "$CUDA_LIB_DIR/$link" && ! -L "$CUDA_LIB_DIR/$link" ]]; then
    ln -s "$target" "$CUDA_LIB_DIR/$link"
  fi
}

ensure_library_link libcudart.so.13 libcudart.so
ensure_library_link libcublas.so.13 libcublas.so
ensure_library_link libcublasLt.so.13 libcublasLt.so
ensure_library_link libcurand.so.10 libcurand.so

if [[ ! -e "$FLASHINFER_CUDA_HOME/lib64" && ! -L "$FLASHINFER_CUDA_HOME/lib64" ]]; then
  ln -s lib "$FLASHINFER_CUDA_HOME/lib64"
fi

NVCC_RELEASE=$("$NVCC" --version | sed -n \
  's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' | head -n 1)
PTXAS_RELEASE=$("$PTXAS" --version | sed -n \
  's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' | head -n 1)
CUDART_VERSION=$(awk \
  '$1 == "#define" && $2 == "CUDART_VERSION" { print $3; exit }' \
  "$CUDA_RUNTIME_HEADER")
CUDA_HEADER_RELEASE="$((CUDART_VERSION / 1000)).$(((CUDART_VERSION % 1000) / 10))"

if [[ "$NVCC_RELEASE" != "$CUDA_HEADER_RELEASE" || \
      "$PTXAS_RELEASE" != "$CUDA_HEADER_RELEASE" ]]; then
  echo "CUDA compiler/assembler/header mismatch after installation: " \
    "nvcc=$NVCC_RELEASE ptxas=$PTXAS_RELEASE headers=$CUDA_HEADER_RELEASE" >&2
  exit 1
fi

NVVM_METADATA=$(find "$FLASHINFER_TOOLKIT_ROOT" -maxdepth 1 -type d \
  -name 'nvidia_nvvm-*.dist-info' -print -quit)
if [[ -z "$NVVM_METADATA" || $(basename "$NVVM_METADATA") != nvidia_nvvm-13.0.88.dist-info ]]; then
  echo "Expected nvidia-nvvm 13.0.88, found: ${NVVM_METADATA:-missing}" >&2
  exit 1
fi

if [[ ! -d "$CUDA_DRIVER_STUB_DIR" ]]; then
  echo "Missing CUDA driver stub directory: $CUDA_DRIVER_STUB_DIR" >&2
  exit 1
fi

# Catch compiler/NVVM/ptxas mismatches and missing linker aliases before a GPU
# job spends time loading the model.
PROBE_DIR=$(mktemp -d)
trap 'rm -rf "$PROBE_DIR"' EXIT
printf '%s\n' \
  '__global__ void kernel() {}' \
  'int main() { kernel<<<1, 1>>>(); return 0; }' \
  > "$PROBE_DIR/sm120.cu"
"$NVCC" \
  -gencode=arch=compute_120f,code=sm_120f \
  -c "$PROBE_DIR/sm120.cu" \
  -o "$PROBE_DIR/sm120.o"
c++ \
  -shared \
  -x c++ /dev/null \
  -L"$CUDA_LIB_DIR" \
  -L"$CUDA_DRIVER_STUB_DIR" \
  -lcudart \
  -lcuda \
  -o "$PROBE_DIR/cuda-link-probe.so"

echo "FlashInfer isolated CUDA toolkit: $FLASHINFER_CUDA_HOME"
echo "CUDA compiler/header version: $NVCC_RELEASE"
echo "PTX assembler version: $PTXAS_RELEASE"
echo "NVVM: 13.0.88"
echo "SM120 compile/link probe: OK"
echo "Run: source $PROJECT_DIR/setup_runtime_env.sh"
