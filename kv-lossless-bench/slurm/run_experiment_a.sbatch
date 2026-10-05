#!/usr/bin/env bash
#SBATCH --job-name=kv-exp-a
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-bootstrap-%j.out
set -euo pipefail

# --- 1. 실행 환경: 경로·로컬 tokenizer·결과 위치와 CPU 스레드 수 설정 ---
PROJECT=/home/hybyun0207/aso-internship/kv-lossless-bench
PYTHON=$PROJECT/.venv/bin/python
export KV_RUN_TIMESTAMP=$(TZ=Asia/Seoul date +%y%m%d_%H%M%S)
export KV_EXPERIMENT=exp-a
RUN_ID=$("$PYTHON" "$PROJECT/src/run_metadata.py")
RUN=$PROJECT/results/$RUN_ID
mkdir -p "$PROJECT/results"
mkdir "$RUN"
# Slurm 지시문은 동적 날짜를 확장하지 않으므로 작업 시작 후 식별자 포함 로그로 전환한다.
exec > "$RUN/${RUN_ID}.log" 2>&1
echo "RUN_ID=$RUN_ID"
"$PYTHON" -m pip freeze > "$RUN/${RUN_ID}_requirements.txt"
: "${KV_TOKENIZER:?Set KV_TOKENIZER to the matching local Llama-3.1 tokenizer directory}"
PROMPTS=${KV_PROMPTS:-$PROJECT/data/prompts_smoke.jsonl}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --- 2. 예제 입력 처리: 기본 smoke 입력에만 짧은 텍스트 반복 허용 ---
EXTRA=()
if [[ "$PROMPTS" == "$PROJECT/data/prompts_smoke.jsonl" ]]; then
    EXTRA+=(--repeat-short-prompts)
fi

# --- 3. GPU 수집: prefill/decode 종료 후 최종 KV를 CPU 파일로 저장 ---
"$PYTHON" "$PROJECT/src/collect_kv.py" \
    --tokenizer "$KV_TOKENIZER" --prompts "$PROMPTS" \
    --prefill-tokens "${KV_PREFILL_TOKENS:-512}" --decode-tokens "${KV_DECODE_TOKENS:-128}" \
    --output "$PROJECT/results" "${EXTRA[@]}"

# --- 4. CPU 분석: 지수 분포와 메타데이터 포함 저장량 계산 ---
"$PYTHON" "$PROJECT/src/analyze_kv.py" --input "$RUN/kv" --output "$PROJECT/results"

# --- 5. 시각화: matplotlib가 있으면 CSV에서 그래프 생성 ---
if "$PYTHON" -c 'import matplotlib' >/dev/null 2>&1; then
    "$PYTHON" "$PROJECT/src/plot_analysis.py" --input "$RUN/analysis"
else
    echo "matplotlib unavailable; CSV/report complete, plots skipped"
fi
