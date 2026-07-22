# GPT-OSS 20B vLLM 서빙 및 MoE 프로파일링 준비

이 문서는 연세대학교 HPC 클러스터의 NVIDIA RTX PRO 6000 Blackwell
Server Edition에서 GPT-OSS 20B를 vLLM으로 서빙한 과정을 기록한다.

현재 완료한 범위는 다음과 같다.

- vLLM `0.23.0` 환경 구성
- GPT-OSS 20B MXFP4 모델 로드
- CUDA Graph를 끈 eager 실행
- OpenAI 호환 API 서버 실행
- `/health`, `/v1/models`, `/v1/chat/completions` 검증

향후에는 동일 환경에서 Prefill/Decode를 분리하고 Attention, MoE, Other의
실행시간을 측정한다.

## 1. 검증된 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM | 97,887 MiB |
| Compute capability | SM 12.0 |
| NVIDIA driver | 580.126.09 |
| Driver CUDA | 13.0 |
| Python | 3.11 |
| PyTorch | 2.11.0+cu130 |
| vLLM | 0.23.0 |
| 모델 | GPT-OSS 20B |
| 모델 양자화 | GPT-OSS MXFP4 |
| Attention backend | `TRITON_ATTN` |
| MoE backend | `MARLIN` |
| CUDA Graph | 비활성화 (`--enforce-eager`) |

vLLM 소스 기준 버전은 공식 저장소의
[`v0.23.0`](https://github.com/vllm-project/vllm/tree/v0.23.0) 태그다.

## 2. 경로 구성

검증에 사용한 기본 경로는 다음과 같다.

```text
/lustre/hybyun0207/
├── envs/vllm023/
├── models/gpt-oss-20b/
├── .cache/
└── gptoss-profile/
    ├── logs/
    └── profiles/
```

모델 디렉터리는 `config.json`, tokenizer 파일 및 safetensors shard를
포함해야 한다.

```bash
find /lustre/hybyun0207 -maxdepth 5 -name config.json -print
```

## 3. Python 환경 구성

처음에는 `/home/hybyun0207/miniconda3/envs/vllm023`에 환경을 만들었으나,
공유 파일시스템의 작은 파일 I/O로 인해 `import torch`와 `import vllm`이
수 분 이상 지연됐다. 환경을 `/lustre`에 새로 만든 뒤 문제가 완화됐다.

패키지 다운로드와 환경 설치는 클러스터 정책에 따라 로그인 서버에서
수행한다.

```bash
mkdir -p \
  /lustre/hybyun0207/envs \
  /lustre/hybyun0207/.cache/uv

export UV_CACHE_DIR=/lustre/hybyun0207/.cache/uv

conda create \
  --prefix /lustre/hybyun0207/envs/vllm023 \
  python=3.11 \
  -y

conda activate /lustre/hybyun0207/envs/vllm023

python -m pip install --upgrade pip uv

uv pip install \
  --python /lustre/hybyun0207/envs/vllm023/bin/python \
  vllm==0.23.0 \
  --torch-backend=cu130

uv pip install \
  --python /lustre/hybyun0207/envs/vllm023/bin/python \
  openai requests
```

버전 확인은 vLLM을 직접 import하지 않고도 가능하다.

```bash
python -c "from importlib.metadata import version; print('vLLM:', version('vllm')); print('PyTorch:', version('torch'))"
```

로그인 서버에서 `Can't initialize NVML` 경고가 발생할 수 있다. 로그인
서버에 GPU가 노출되지 않아서 발생하는 경고이므로 import가 완료된다면
문제가 아니다.

## 4. GPU 할당 후 환경 확인

GPU 자원은 반드시 SLURM으로 할당받는다. 다음은 할당이 완료된 GPU
노드 내부에서 실행한다.

```bash
conda activate /lustre/hybyun0207/envs/vllm023

echo "JOB_ID=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

`CUDA_VISIBLE_DEVICES`는 SLURM이 설정한 값을 사용하며 직접 변경하지 않는다.

Python bytecode cache에 아래와 같이 작업별 `/tmp` 경로를 지정하면 새 작업을
시작할 때마다 torch와 vLLM 모듈을 다시 컴파일해 첫 import가 매우 느려질 수
있다.

```bash
# 사용하지 않는다.
# export PYTHONPYCACHEPREFIX=/tmp/${USER}/pycache-${SLURM_JOB_ID}

unset PYTHONPYCACHEPREFIX
```

GPU와 패키지를 확인한다.

```bash
python -u -c "
import torch
import vllm
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('imports OK')
"
```

검증 당시 첫 import는 약 22초, 같은 노드에서 이어진 두 번째 import는 약
7초가 소요됐다.

## 5. 서버 실행 전 환경변수

```bash
export MODEL_PATH=/lustre/hybyun0207/models/gpt-oss-20b
export EXP_ROOT=/lustre/hybyun0207/gptoss-profile
export LOG_ROOT="$EXP_ROOT/logs"
export PROFILE_ROOT="$EXP_ROOT/profiles"

export HF_HOME=/lustre/hybyun0207/.cache/huggingface
export TORCH_HOME=/lustre/hybyun0207/.cache/torch
export VLLM_CACHE_ROOT=/lustre/hybyun0207/.cache/vllm
export TRITON_CACHE_DIR=/tmp/${USER}/triton-${SLURM_JOB_ID}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p \
  "$LOG_ROOT" \
  "$PROFILE_ROOT" \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR"
```

### FlashInfer sampler 비활성화

초기 실행에서는 모델 가중치까지 정상적으로 로드된 뒤 FlashInfer의 top-k /
top-p sampler가 SM 12.0을 판별하지 못해 다음 오류가 발생했다.

```text
Failed to get device capability: SM 12.x requires CUDA >= 12.9.
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

PyTorch는 `cu130`이었고 GPU도 SM 12.0으로 정상 인식됐다. 실패 지점은
Attention 또는 MoE가 아니라 dummy sampling 단계였다. 다음 환경변수로
FlashInfer sampler만 비활성화해 해결했다.

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

이 설정은 Attention과 MoE backend를 바꾸지 않는다. 검증된 실행에서는
Attention이 `TRITON_ATTN`, MXFP4 MoE가 `MARLIN`을 사용했다.

## 6. vLLM 서버 실행

첫 검증에서는 최대 문맥을 8K, 동시 sequence를 4개로 제한했다. 향후
프로파일링 조건과 일치시키기 위해 CUDA Graph와 torch.compile을
`--enforce-eager`로 비활성화했다.

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  > "$LOG_ROOT/server.log" 2>&1 &

export VLLM_PID=$!
echo "vLLM PID=$VLLM_PID"
```

첫 실행에서는 약 12.82 GiB의 checkpoint를 읽고 모델 가중치 적재에 약
50초가 소요됐다. 전체 모델은 GPU 메모리 약 13.8 GiB를 사용했다.

로그 확인:

```bash
tail -f "$LOG_ROOT/server.log"
```

`Ctrl+C`는 `tail`만 종료하며 백그라운드 서버는 유지된다.

## 7. 서버 검증

### Health check

```bash
curl -i http://127.0.0.1:8000/health
```

정상 결과:

```text
HTTP/1.1 200 OK
```

### 모델 목록

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

응답에 `gpt-oss-20b`와 `max_model_len: 8192`가 표시되는지 확인한다.

### 생성 요청

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {
        "role": "user",
        "content": "한국어로 세 문장 이내로 자기소개해 줘."
      }
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }' \
  | python -m json.tool
```

검증 요청에서는 다음 token 수가 기록됐다.

```text
prompt_tokens:     82
completion_tokens: 171
total_tokens:      253
```

GPT-OSS는 reasoning 모델이므로 JSON 응답의 `reasoning`과 최종 사용자 응답인
`content`가 분리된다. `completion_tokens`에는 reasoning token도 포함되므로
겉으로 보이는 답변 길이보다 생성 시간이 길 수 있다. 첫 요청은 kernel
초기화와 cache warming의 영향도 받을 수 있다.

최종 응답만 한글로 출력하려면 다음처럼 JSON의 `content`를 선택한다.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "한국어로 인사해 줘."}],
    "temperature": 0,
    "max_tokens": 64
  }' \
  | python -c "import json, sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## 8. 서버 종료

테스트가 끝난 서버는 GPU를 계속 점유하므로 반드시 종료한다.

```bash
kill "$VLLM_PID"
wait "$VLLM_PID" 2>/dev/null
nvidia-smi
```

SLURM interactive allocation도 끝낼 경우 `exit`하거나 해당 job을 취소한다.

```bash
scancel "$SLURM_JOB_ID"
```

## 9. 현재 알려진 경고

다음 Transformers v4 deprecation 경고는 서버 실패 원인이 아니었다.

```text
Support for Transformers v4 is deprecated.
```

재현성을 위해 vLLM `0.23.0` 환경에서 Transformers만 임의로 업그레이드하지
않는다.

다음 reasoning token ID 자동 초기화 경고도 서버 기동을 막지는 않았다.

```text
Auto-initialization of reasoning token IDs failed.
```

## 10. 다음 실험

첫 프로파일링에서는 CUDA Graph를 계속 끄고 다음 세 범주만 분리한다.

- Attention: QKV projection, RoPE, attention kernel, output projection
- MoE: router, top-k expert 선택, token dispatch, expert FFN, 결과 결합
- Other: norm, residual, LM head, sampling, KV cache 및 runtime overhead

최소 조건은 다음과 같다.

| Sweep | Batch | Input tokens | Output tokens |
|---|---:|---:|---:|
| Prefill length | 1 | 128, 256, 512, 1,024, 2,048, 4,096, 8,192 | 1 |
| Decode context length | 1 | 128, 256, 512, 1,024, 2,048, 4,096, 8,192 | 128 |
| Decode batch | 2, 4, 8, 16, 32 | 1,024 | 128 |

먼저 Prefill 1,024 조건 하나의 Torch Profiler trace를 수집해 추가한
`GPTOSS_ATTENTION_*`, `GPTOSS_MOE_*` annotation을 확인한 뒤 자동 집계한다.

## 11. 고정 길이 Prefill/Decode 요청

[`run_profile_request.py`](run_profile_request.py)는 vLLM의
`/v1/completions`에 token ID prompt를 직접 전송한다. 따라서 tokenizer나
chat template의 영향 없이 sequence당 입력 token 수를 정확히 통제할 수
있다. 또한 선택적으로 `/start_profile`과 `/stop_profile`을 호출한다.

전체 실험 매트릭스는 [`configs/experiments.yaml`](configs/experiments.yaml)에
정의하며 [`run_experiments.py`](run_experiments.py)가 조건을 순회한다.

```bash
# 실행할 명령만 확인
python run_experiments.py --phase all --dry-run

# Prefill 조건 실행
python run_experiments.py --phase prefill

# Decode 조건 실행
python run_experiments.py --phase decode
```

최초 검증에서는 profiler endpoint를 호출하지 않고 요청 규격만 확인한다.

```bash
python run_experiments.py --phase prefill --no-profile
```

서버는 Torch Profiler 옵션을 포함해 실행되어 있어야 하며, 요청 전에
`/health`가 HTTP 200을 반환해야 한다.

8,192-token Decode는 128개 출력 token까지 포함하므로 profiler 서버의
`--max-model-len`은 최소 8,320보다 커야 한다. 이 실험에서는 16,384로
고정한다. 또한 batch 32의 1,024-token prompt 전체가 한 Prefill step에서
처리되도록 다음 서버 옵션을 사용한다.

```bash
--max-model-len 16384 \
--max-num-seqs 32 \
--max-num-batched-tokens 32768 \
--no-enable-prefix-caching
```

`max-num-batched-tokens`가 이보다 작으면 batch Prefill이 여러 chunk로 나뉘어
조건 간 phase 비교가 어려워진다.

vLLM 0.23.0은 지원되는 모델에서 prefix caching을 기본 활성화한다. warmup과
측정 요청이 같은 prompt를 사용하므로 이를 끄지 않으면 측정 Prefill이 대부분
cache hit가 되어, 예를 들어 1,024-token 요청에서도 trace에는 마지막 16개
token만 계산된 것으로 나타날 수 있다. 이 실험에서는 전체 input length를 매번
실제로 계산하도록 `--no-enable-prefix-caching`을 반드시 사용한다.

### Prefill 1,024 tokens

```bash
python run_profile_request.py \
  --phase prefill \
  --input-len 1024 \
  --output-len 1 \
  --batch-size 1 \
  --profile
```

### Prefill 2,048 tokens

```bash
python run_profile_request.py \
  --phase prefill \
  --input-len 2048 \
  --output-len 1 \
  --batch-size 1 \
  --profile
```

### Decode context-length 및 batch scaling

Decode는 먼저 batch 1에서 입력 길이를 변경한다. 그다음 입력 길이 1,024와
출력 길이 128을 고정하고 batch를 변경한다. 전체 조건은 YAML에 정의되어
있으므로 다음 명령으로 실행한다.

```bash
for batch in 1 2 4 8 16 32; do
  python run_profile_request.py \
    --phase decode \
    --input-len 1024 \
    --output-len 128 \
    --batch-size "$batch" \
    --profile
done
```

처음에는 전체 조건을 실행하지 말고 `batch=1` 하나로 trace 생성과 annotation
수집을 검증한다. 스크립트는 기본적으로 profiler를 켜기 전에 같은 조건의
warmup 요청을 한 번 보낸다.

작은 요청 응답과 metadata는 Git 저장소 내부에 저장한다.

```text
MoE/results/requests/
```

대용량 Torch trace는 `/home`과 Git 저장소에 기록하지 않고 YAML의
`paths.torch_traces`가 지정하는 Lustre 경로에 저장한다.

```text
/lustre/hybyun0207/gptoss-profile/torch-traces/
```

## 12. Trace 집계와 experiment.csv

[`analyze_traces.py`](analyze_traces.py)는 각 요청의 JSON metadata와 Torch
Profiler trace를 결합해 `results/experiment.csv`를 만든다. 전체 매트릭스를
`run_experiments.py`로 실행하면 모든 요청이 끝난 뒤 이 분석기가 자동으로
실행된다. 이미 만들어진 trace만 다시 분석하려면 다음 명령을 사용한다.

```bash
cd /home/hybyun0207/aso-internship/MoE

python analyze_traces.py \
  --request-dir results/requests \
  --trace-dir /lustre/hybyun0207/gptoss-profile/torch-traces \
  --output results/experiment.csv
```

CSV의 주요 열은 다음과 같다.

| 열 | 의미 |
|---|---|
| `phase` | `prefill` 또는 `decode` |
| `batch_size` | 한 API 요청에 포함된 sequence 수 |
| `input_tokens_per_sequence` | sequence당 입력 token 수 |
| `output_tokens_per_sequence` | sequence당 요청한 생성 token 수 |
| `end_to_end_latency_ms` | client가 관측한 전체 HTTP 요청 시간 |
| `phase_gpu_total_ms` | 선택한 phase의 GPU annotation 총시간 |
| `attention_gpu_total_ms` | 모든 layer의 Attention GPU 시간 합 |
| `moe_gpu_total_ms` | 모든 layer의 MoE GPU 시간 합 |
| `other_gpu_total_ms` | phase GPU 시간에서 Attention과 MoE를 뺀 값 |
| `layer_count` | trace에서 발견한 GPT-OSS layer 수 |
| `model_forward_count` | trace에 포함된 전체 model forward 수 |
| `step_count` | 실제 집계에 사용한 Prefill 또는 Decode step 수 |
| `profiled_context_tokens` | trace에서 실제로 계산된 Prefill token 총수 |
| `profiled_generation_tokens` | trace에서 실제로 계산된 Decode token 총수 |
| `analyzed_context_tokens` | 순수 Prefill GPU latency 집계에 포함된 token 수 |
| `analyzed_generation_tokens` | 순수 Decode GPU latency 집계에 포함된 token 수 |
| `mixed_step_count` | Prefill과 Decode가 한 forward에 섞인 scheduler step 수 |
| `excluded_mixed_context_tokens` | mixed step과 함께 GPU 집계에서 제외한 Prefill token 수 |
| `excluded_mixed_generation_tokens` | mixed step과 함께 GPU 집계에서 제외한 Decode token 수 |

`output_len=N`인 decode 요청에는 Prefill forward 1회와 Decode forward
`N-1`회에 해당하는 generation token 처리가 함께 들어간다. 분석기는
`execute_context_X(...)_generation_Y(...)` annotation을 이용해 순수 Prefill과
순수 Decode scheduler range 안에 포함된 Attention/MoE event만 선택한다. 따라서
현재 `output_len=128` 조건에서 순수 Decode `step_count`는 127이다.

OpenAI batch 요청의 각 sequence가 scheduler에 도착하는 시점이 다르면 첫
단계에서 일부 sequence의 Prefill과 먼저 도착한 sequence의 Decode가 하나의
forward에 섞일 수 있다. 이 mixed forward의 Attention/MoE 시간은 두 phase로
정확히 분할할 수 없으므로 GPU latency 집계에서 제외하고, 관련 step 및 token
수를 위의 `mixed_*`, `excluded_mixed_*` 열에 명시한다. 전체 token이 실제로
계산됐는지는 `profiled_context_tokens`, `profiled_generation_tokens`로 별도
검증한다.

`attention_percent`, `moe_percent`, `other_percent`는
`phase_gpu_total_ms = 100%`를 기준으로 계산된다. 반면
`end_to_end_latency_ms`에는 HTTP frontend와 CPU scheduling 등도 포함되므로
두 시간의 기준을 혼합해 비율을 계산하지 않는다.

파일 이름은 실험 조건과 UTC timestamp를 포함한다.

```text
prefill-b1-i1024-o1-YYYYMMDDTHHMMSSffffffZ.json
decode-b16-i1024-o128-YYYYMMDDTHHMMSSffffffZ.json
```

Profiler 없이 요청 형식과 token 수만 검증하려면 `--profile`을 생략한다.

```bash
python run_profile_request.py --phase prefill --warmup 0
```

스크립트는 `add_special_tokens=false`를 설정하고, 기본적으로 seed 42로 만든
다양한 token ID를 사용한다. 동일 token 반복으로 특정 expert routing이
편향되는 것을 줄이면서 모든 조건에서 재현 가능한 prompt를 만든다. 응답의
`usage.prompt_tokens`가 `batch_size * input_len`과 다르면 실패 처리한다.
`ignore_eos=true`와 `min_tokens=output_len`을 사용하므로 각 sequence는 지정한
수만큼 decode한다.

`results/`는 `.gitignore`로 제외되어 로컬 실험 결과가 실수로 commit되지
않는다. 최종 CSV와 그래프 중 공유할 산출물만 검토 후 별도 위치에 추가한다.

## 13. 프로파일링 대상과 측정 범위

이번 실험은 GPT-OSS의 개별 CUDA 커널 구현을 수정한 것이 아니다. editable로
설치한 vLLM의 GPT-OSS Python 구현에서 Attention block과 MoE block 호출을
`torch.profiler.record_function`으로 감싸고, PyTorch Profiler가 해당 범위의
CPU/CUDA activity를 trace에 기록하도록 구성했다.

전체 측정 흐름은 다음과 같다.

```text
수정한 vLLM GPT-OSS Python 소스
        ↓
torch.profiler.record_function annotation
        ↓
Attention/MoE 내부의 여러 CUDA kernel 실행
        ↓
PyTorch Profiler가 CPU/CUDA trace 수집
        ↓
analyze_traces.py가 GPU annotation 시간 집계
        ↓
results/experiment.csv
```

### Attention 측정 범위

`TransformerBlock.forward()`에서 `self.attn(...)` 호출 전체를 layer별
annotation으로 감쌌다.

```python
with torch.profiler.record_function(
    f"GPTOSS_ATTENTION_L{self.layer_idx}"
):
    hidden_states = self.attn(hidden_states, positions)
```

여기서 `self.attn`은 단일 Attention CUDA kernel이 아니라
`OAIAttention.forward()` 전체다. 따라서 `attention_gpu_total_ms`에는 다음
연산이 포함된다.

```text
QKV projection
→ Q/K/V 분리
→ RoPE
→ Attention backend kernel
→ Output projection
```

즉 이 실험의 Attention 시간은 순수 attention kernel만의 시간이 아니라 각
Transformer layer의 전체 Attention block 실행시간이다.

### MoE 측정 범위

동일한 `TransformerBlock.forward()`에서 `self.mlp(...)` 호출 전체를 layer별
annotation으로 감쌌다.

```python
with torch.profiler.record_function(f"GPTOSS_MOE_L{self.layer_idx}"):
    output = self.mlp(hidden_states)
```

GPT-OSS의 `self.mlp`는 `MLPBlock.forward()` 전체이므로
`moe_gpu_total_ms`에는 다음 연산이 포함된다.

```text
Router linear 연산
→ Expert routing 정보 계산
→ FusedMoE 호출
→ 선택된 expert FFN 실행
→ 결과 결합
```

따라서 현재 MoE 시간은 expert FFN만의 시간이 아니라 router와 FusedMoE를
포함한 전체 MoE block 시간이다.

### 수정하지 않은 코드

다음과 같은 vLLM 네이티브 CUDA extension이나 kernel 구현은 수정하지 않았다.

```text
vllm._C
vllm._moe_C
MARLIN MoE kernel
Triton Attention kernel
```

Python 레벨의 호출 범위에 annotation만 추가했으므로 이 변경을 위해 네이티브
CUDA extension을 다시 빌드하지 않았다.

### Trace 수집 방법

vLLM 서버는 `--enforce-eager`와 Torch Profiler 설정을 사용한다. 각 실험은
다음 순서로 하나의 측정 요청만 trace에 포함한다.

```text
Profiler 밖에서 warmup 요청
→ POST /start_profile
→ 측정 대상 요청 1개
→ POST /stop_profile
→ compressed Torch trace 저장
```

`--enforce-eager`를 사용하므로 CUDA Graph replay가 아니라 각 layer와 step의
Attention/MoE 호출이 trace에 반복해서 나타난다.

분석기는 다음 이름의 annotation을 찾는다.

```text
GPTOSS_ATTENTION_L0 ... GPTOSS_ATTENTION_L23
GPTOSS_MOE_L0       ... GPTOSS_MOE_L23
```

동일한 이름의 CPU annotation과 GPU annotation 중 latency 계산에는
`cat == "gpu_user_annotation"`인 GPU event의 `dur`만 사용한다. 따라서 CSV의
Attention/MoE latency는 Python 함수의 CPU 실행시간이 아니라 annotation
범위에 포함된 GPU 작업시간이다.

### Other와 E2E 시간

LayerNorm, residual, embedding, 최종 norm, LM head, sampling, KV cache 및 기타
runtime 작업은 Attention/MoE annotation 바깥에 있다. `Other`는 다음과 같이
계산한다.

```text
Other GPU 시간
= phase 전체 GPU 시간 - Attention GPU 시간 - MoE GPU 시간
```

`end_to_end_latency_ms`는 Torch Profiler에서 가져온 값이 아니다.
`run_profile_request.py`가 `time.perf_counter()`로 HTTP 요청 전후를 측정한
client-side wall-clock 시간이며 다음 항목을 모두 포함한다.

```text
HTTP 요청 및 응답
+ vLLM scheduling
+ Prefill
+ Decode
+ Sampling
+ CPU/runtime overhead
```

따라서 실험에는 서로 다른 두 시간 기준이 함께 저장된다.

```text
end_to_end_latency_ms
  └─ 사용자 관점의 전체 요청 wall-clock latency

attention/moe/other GPU latency
  └─ GPT-OSS annotation과 PyTorch GPU trace 기반 latency
```

Decode 행에서는 `end_to_end_latency_ms`에 Prefill 1회와 Decode 전체가 모두
포함된다. 반면 `phase_gpu_total_ms`, Attention, MoE, Other 및 step당 시간은
vLLM scheduler annotation으로 확인한 순수 Decode step만 집계한다. Prefill과
Decode가 섞인 forward가 있으면 이를 두 phase로 임의 분할하지 않고 제외하며,
제외한 step과 token 수를 CSV에 함께 기록한다.
