#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_ROOT=${EXP_ROOT:-/lustre/hybyun0207/vllm023-moe}
VLLM_ENV=${VLLM_ENV:-$EXP_ROOT/env}
VLLM_SRC=${VLLM_SRC:-$EXP_ROOT/src/vllm}
FLASHINFER_TOOLKIT_ROOT=${FLASHINFER_TOOLKIT_ROOT:-$EXP_ROOT/cuda/13.0-full}
PATCH_FILE="$PROJECT_DIR/patches/gpt_oss_layer0_moe_profile.patch"
export PATH="$VLLM_ENV/bin:${PATH:-}"

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  echo "Missing Python environment: $VLLM_ENV" >&2
  exit 1
fi
if [[ ! -d "$VLLM_SRC/.git" ]]; then
  echo "Missing vLLM source checkout: $VLLM_SRC" >&2
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
  echo "uv is required. Install it on the login node first." >&2
  exit 1
fi

SOURCE_TAG=$(git -C "$VLLM_SRC" describe --tags --exact-match)
if [[ "$SOURCE_TAG" != "v0.23.0" ]]; then
  echo "Expected vLLM source tag v0.23.0, got $SOURCE_TAG" >&2
  exit 1
fi

if git -C "$VLLM_SRC" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "Source reference patch is already applied."
else
  git -C "$VLLM_SRC" apply --check "$PATCH_FILE"
  git -C "$VLLM_SRC" apply "$PATCH_FILE"
  echo "Applied layer-0 MoE patch to the source reference."
fi

# Keep the official wheel installation so its native CUDA extensions remain
# available. Editable installation only linked Python files and lost vllm._C.
"$UV_BIN" pip install \
  --python "$VLLM_ENV/bin/python" \
  --reinstall-package vllm \
  --no-deps \
  --link-mode copy \
  "vllm==0.23.0" \
  --torch-backend=cu130

"$UV_BIN" pip install \
  --python "$VLLM_ENV/bin/python" \
  ijson \
  pyyaml \
  ninja \
  "matplotlib==3.11.1" \
  "scipy==1.17.1" \
  "onnx==1.22.0" \
  "onnxslim==0.1.94"

# vLLM's emulation backend only imports quark.torch.kernel.mx. Installing the
# full Quark dependency set also pulls dataset/evaluation packages that are not
# used by this benchmark, so keep the runtime dependency set minimal and pinned.
"$UV_BIN" pip install \
  --python "$VLLM_ENV/bin/python" \
  --no-deps \
  "amd-quark==0.12.post1"

EXP_ROOT="$EXP_ROOT" \
VLLM_ENV="$VLLM_ENV" \
FLASHINFER_TOOLKIT_ROOT="$FLASHINFER_TOOLKIT_ROOT" \
UV_BIN="$UV_BIN" \
  bash "$PROJECT_DIR/setup_flashinfer_cuda.sh"

SITE_PACKAGES=$("$VLLM_ENV/bin/python" -c \
  "import sysconfig; print(sysconfig.get_paths()['purelib'])")
INSTALLED_GPT_OSS="$SITE_PACKAGES/vllm/model_executor/models/gpt_oss.py"

if [[ ! -f "$INSTALLED_GPT_OSS" ]]; then
  echo "Missing installed GPT-OSS source: $INSTALLED_GPT_OSS" >&2
  exit 1
fi

if grep -q "GPTOSS_FUSED_MOE_L0_M" "$INSTALLED_GPT_OSS"; then
  echo "Installed-package instrumentation is already applied."
else
  cp -p "$INSTALLED_GPT_OSS" "$INSTALLED_GPT_OSS.vllm023-original"
  patch \
    --batch \
    --forward \
    --strip=1 \
    --directory="$SITE_PACKAGES" \
    < "$PATCH_FILE"
  echo "Patched installed GPT-OSS Python source."
fi

"$VLLM_ENV/bin/python" -c "
from importlib.metadata import version
import inspect
import shutil
import vllm._C
import vllm._moe_C
import vllm.model_executor.models.gpt_oss as gpt_oss
from quark.torch.kernel import mx

source = inspect.getfile(gpt_oss)
assert version('vllm') == '0.23.0', version('vllm')
assert 'GPTOSS_FUSED_MOE_L0_M' in open(source, encoding='utf-8').read()
assert shutil.which('ninja'), 'ninja executable is not on PATH'
print('vLLM:', version('vllm'))
print('AMD Quark:', version('amd-quark'))
print('SciPy:', version('scipy'))
print('ONNX:', version('onnx'))
print('ONNXSlim:', version('onnxslim'))
print('Matplotlib:', version('matplotlib'))
print('Ninja:', shutil.which('ninja'))
print('GPT-OSS source:', source)
print('Native extensions: OK')
print('Instrumentation: OK')
"
