#!/bin/bash
# Create an isolated CUDA 13 / PyTorch / FlashInfer environment for this benchmark.

set -euo pipefail

GEMM_ROOT=${GEMM_ROOT:-/lustre/hybyun0207/gptoss-gemm}
GEMM_ENV=${GEMM_ENV:-/lustre/hybyun0207/envs/gemm}
UV_BIN=${UV_BIN:-/lustre/hybyun0207/vllm023-moe/tools/bin/uv}
UV_CACHE_DIR=${UV_CACHE_DIR:-$GEMM_ROOT/cache/uv}
UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$GEMM_ROOT/python}
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv executable not found: $UV_BIN" >&2
  exit 1
fi

mkdir -p \
  "$GEMM_ROOT/cache/uv" \
  "$GEMM_ROOT/cache/flashinfer" \
  "$GEMM_ROOT/cache/autotune" \
  "$GEMM_ROOT/cache/torch-extensions" \
  "$GEMM_ROOT/cache/cuda"

export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_HTTP_TIMEOUT
export UV_LINK_MODE=copy

"$UV_BIN" python install --no-bin 3.11

if [[ ! -x "$GEMM_ENV/bin/python" ]]; then
  "$UV_BIN" venv --python 3.11 "$GEMM_ENV"
else
  echo "Reusing existing environment: $GEMM_ENV"
fi

"$UV_BIN" pip install \
  --python "$GEMM_ENV/bin/python" \
  --torch-backend=cu130 \
  "torch==2.11.0" \
  "torchao==0.17.0" \
  "flashinfer-python[cu13]==0.6.12" \
  "flashinfer-cubin==0.6.12" \
  "numpy==2.3.5" \
  "pyyaml==6.0.3" \
  "pandas>=2.3,<3" \
  "matplotlib==3.11.1"

"$GEMM_ENV/bin/python" - <<'PY'
from importlib.metadata import version
import torch

print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("TorchAO:", version("torchao"))
print("FlashInfer:", version("flashinfer-python"))
print("FlashInfer cubin:", version("flashinfer-cubin"))

assert torch.__version__.startswith("2.11.0")
assert torch.version.cuda == "13.0"
assert version("torchao") == "0.17.0"
assert version("flashinfer-python") == "0.6.12"
assert version("flashinfer-cubin") == "0.6.12"
PY

echo "Environment ready: $GEMM_ENV"
echo "Run 'source /home/hybyun0207/aso-internship/gemm/setup_runtime_env.sh' on a GPU node."
