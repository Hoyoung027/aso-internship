#!/usr/bin/env bash
set -euo pipefail

# --- 1. 독립 환경 생성: 기존 실험의 site-packages를 공유하지 않는다 ---
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PYTHON=${KV_BASE_PYTHON:-/lustre/hybyun0207/gptoss-gemm/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11}
if [[ ! -x "$PROJECT/.venv/bin/python" ]]; then
    "$BASE_PYTHON" -m venv "$PROJECT/.venv"
fi

# --- 2. 패키지 설치: GPU용 Torch를 명시하고 나머지 분석 의존성 설치 ---
"$PROJECT/.venv/bin/python" -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu130
"$PROJECT/.venv/bin/python" -m pip install -r "$PROJECT/requirements.txt"
"$PROJECT/.venv/bin/python" -m pip check

# --- 3. 재현 정보: 실제 설치된 전이 의존성까지 기록 ---
"$PROJECT/.venv/bin/python" -m pip freeze > "$PROJECT/requirements.lock.txt"
echo "Environment ready: $PROJECT/.venv"
