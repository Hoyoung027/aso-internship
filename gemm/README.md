# GPT-OSS-20B representative GEMM microbenchmark

RTX PRO 6000 Blackwell(SM120)에서 GPT-OSS-20B의 대표 GEMM shape를 가져와
PyTorch와 FlashInfer GEMM 성능을 비교한다. 모델을 로드하거나 vLLM을 서빙하지
않으며 MoE layer 전체를 재현하는 실험도 아니다.

측정하는 연산은 다음 다섯 GEMM case의 `A[M,K] @ B[K,N]`이다.

| GEMM case | K | N | 의미 |
|---|---:|---:|---|
| QKV | 2880 | 5120 | Q, K, V projection을 합친 출력 |
| O | 4096 | 2880 | attention output projection |
| Router Gate | 2880 | 32 | 32개 local expert의 router logit |
| Expert W13 | 2880 | 5760 | 2880차원 gate/up projection을 결합한 출력 |
| Expert W2 | 2880 | 2880 | expert down projection |

QKV의 `N=5120`은 `64*64 + 8*64 + 8*64`다. Router Gate는 expert 내부의
W1 gate가 아니라 MoE router projection이다. W13의 `N=5760`은 gate와 up
projection의 `2880+2880`이며, 실제 모델에서는 SwiGLUOAI 뒤에 W2가 이어진다.
이 benchmark의 W13/W2는 expert 하나의 2차원 dense GEMM shape로 실행하며,
32개 expert의 grouped GEMM이나 token dispatch를 재현하지 않는다.

## 실험 범위

포함하지 않는 연산:

- Attention의 `QK^T`, softmax, `P@V`
- MoE routing softmax와 top-k
- token dispatch/permutation
- W13과 W2 사이의 SwiGLUOAI
- 여러 expert를 묶는 grouped/fused MoE 실행
- expert 결과 reduction
- quantization 실행시간
- AutoTuner 탐색시간
- CUDA Graph

quantization과 AutoTuner는 timed region 밖에서 수행한다. 본 측정은 CUDA
Event로 GEMM 호출 하나만 측정한다.

## 비교 대상

| experiment ID | API | 입력/weight | 출력 | AutoTuner |
|---|---|---|---|---|
| `torch_bf16` | `torch.nn.functional.linear` | BF16 | BF16 | N/A |
| `torch_mxfp8` | `aten._scaled_mm.out` | MXFP8 E4M3 + E8M0 block-32 | BF16 | N/A |
| `flashinfer_bf16_default` | `flashinfer.mm_bf16` | BF16 | BF16 | fallback tactic |
| `flashinfer_bf16_tuned` | `flashinfer.mm_bf16` | BF16 | BF16 | cache 사용 |
| `flashinfer_mxfp8_default` | `flashinfer.mm_mxfp8` | MXFP8 block-32 | BF16 | fallback tactic |
| `flashinfer_mxfp8_tuned` | `flashinfer.mm_mxfp8` | MXFP8 block-32 | BF16 | cache 사용 |
| `flashinfer_mxfp4_default` | `flashinfer.mm_fp4` | MXFP4 block-32 | BF16 | fallback tactic |
| `flashinfer_mxfp4_tuned` | `flashinfer.mm_fp4` | MXFP4 block-32 | BF16 | cache 사용 |

일반 FP8 per-tensor scale 실험은 제거했다. PyTorch MXFP8과 FlashInfer MXFP8은
동일한 BF16 원본을 FlashInfer `mxfp8_quantize`로 변환해 얻은 E4M3 값과 E8M0
block-32 scale을 사용한다. 128x4 swizzled scale storage는 FlashInfer에는
`uint8`, PyTorch에는 동일 bit를 `float8_e8m0fnu` view로 전달한다. quantization은
양쪽 모두 timed region 밖이다. 출력도 미리 할당하고 GEMM 호출만 측정한다.
MXFP4에는 PyTorch 기준선을 두지 않고 FlashInfer default와 tuned의 차이를 본다.

실제 GPT-OSS-20B checkpoint의 Attention과 Router projection은 양자화 제외
대상이므로 이 세 case는 BF16 결과가 실제 모델에 가장 가깝다. Expert W13/W2
weight는 MXFP4이지만, 이 benchmark의 `flashinfer.mm_fp4`는 activation도
MXFP4로 quantize하므로 vLLM의 W4A16/W4A8 MoE 경로와 동일하지 않다. MXFP4
결과는 약속한 대로 FlashInfer default/tuned tactic 효과를 확인하는 용도다.
나머지 저정밀 조합도 동일 shape의 GEMM 특성을 보기 위한 가상 실험이다.

## SM120 지원 제약

FlashInfer 0.6.12의 실제 API 제약을 코드에 반영했다.

- BF16 `auto`: SM120에서는 cuDNN과 TinyGEMM 후보를 사용할 수 있다.
- MXFP8 `auto`: SM120 CUTLASS 경로와 128x4 swizzled scale을 사용한다.
- MXFP8은 CUTLASS common check에서 `N>=128`을 요구한다.
- 따라서 Router Gate `(N=32)`의 FlashInfer MXFP8은 `unsupported`로 기록한다.
- PyTorch MXFP8 Router는 별도로 실행을 시도하며, FlashInfer가 미지원이므로 해당
  shape에 대한 두 구현의 speedup은 계산하지 않는다.
- SM120 MXFP4에서 `use_nvfp4=False`이면 b12x/CUTLASS가 아니라 cuDNN
  경로를 사용한다.
- SM120 MXFP4는 cuDNN 9.14 이상이 필요하며 현재 환경의 cuDNN 9.19를
  전제로 한다.

지원되지 않는 case나 runtime error는 CSV에서 `status`, `error`로 남는다.
다른 API로 몰래 대체하지 않는다.

## 기본 조건

`configs/experiments.yaml`의 기본값:

```text
M = 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
    1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768
seed = 42
warmup = 100
repeat = 100
독립 trial = 3
GPU = RTX PRO 6000 Blackwell 1장
CUDA Graph = OFF
bias = 없음
```

BF16 master activation과 weight는 projection/M으로부터 유도한 고정 seed로
생성한다. 서로 다른 프로세스의 default/tuned 실험에서도 동일한 데이터가
재현된다. weight와 quantized weight는 timed region 밖에서 준비한다.

QKV/O/Router에서 `M`은 입력 token 수로 직접 해석할 수 있다. Expert W13/W2의
`M`은 expert 하나가 처리하는 token 수 `M_e`다. 32 experts, top-k 4가 균등하게
분배된다고 가정하면 전체 token 수와의 관계는 `M_e = M_total / 8`이다.

## 파일

```text
gemm/
├── README.md
├── setup_env.sh
├── setup_runtime_env.sh
├── benchmark_gemm.py
├── configs/
│   └── experiments.yaml
├── slurm/
│   └── run_gemm_benchmark.sbatch
└── result/
    └── .gitignore
```

## 1. 저장 경로 준비

코드와 최종 실험 결과는 home에 두고, 용량이 큰 Python 환경, JIT cache와
AutoTuner cache는 Lustre에 둔다.

```bash
export GEMM_ROOT=/lustre/hybyun0207/gptoss-gemm
export GEMM_ENV=/lustre/hybyun0207/envs/gemm
export GEMM_RESULT_ROOT=/home/hybyun0207/aso-internship/gemm/result

mkdir -p \
  "$GEMM_ROOT/cache" \
  "$GEMM_RESULT_ROOT"
```

기존 vLLM 환경을 수정하지 않는다. GEMM 실험용 Python virtual environment를
별도로 만든다.

## 2. CUDA 13 full toolkit 확인

FlashInfer SM120 JIT에는 PyTorch wheel의 CUDA runtime만으로 부족하고 `nvcc`,
`ptxas`, headers와 link library가 필요하다. 이미 검증한 다음 toolkit을
재사용한다.

```bash
export FLASHINFER_TOOLKIT_ROOT=/lustre/hybyun0207/vllm023-moe/cuda/13.0-full
export CUDA_HOME="$FLASHINFER_TOOLKIT_ROOT/nvidia/cu13"

test -x "$CUDA_HOME/bin/nvcc"
test -x "$CUDA_HOME/bin/ptxas"
test -f "$CUDA_HOME/include/cuda.h"
test -e "$CUDA_HOME/lib/libcudart.so"

"$CUDA_HOME/bin/nvcc" --version
```

기대되는 major version은 CUDA 13.0이다. 이 디렉터리가 없다면 GEMM 환경을
설치하기 전에 CUDA 13 full toolkit부터 복원해야 한다. CUDA 12.8 module의
`nvcc`로 대체하면 PyTorch cu130/FlashInfer JIT 조건과 달라진다.

## 3. Python 환경 설치

패키지 설치는 로그인 노드에서 수행한다. 기존 vLLM 실험에서 설치한 `uv`를
재사용한다.

```bash
cd /home/hybyun0207/aso-internship/gemm

export GEMM_ROOT=/lustre/hybyun0207/gptoss-gemm
export GEMM_ENV=/lustre/hybyun0207/envs/gemm
export UV_BIN=/lustre/hybyun0207/vllm023-moe/tools/bin/uv

bash setup_env.sh
```

`setup_env.sh`는 다음을 수행한다.

1. uv Python 3.11 설치 또는 재사용
2. `/lustre/hybyun0207/envs/gemm` virtual environment 생성
3. PyTorch 2.11.0 cu130 설치
4. FlashInfer Python/cubin 0.6.12와 cu13 의존성 설치
5. NumPy, PyYAML, pandas, matplotlib 설치
6. 버전 assertion 실행

기존 환경이 있으면 삭제하지 않고 패키지를 확인/갱신한다. 완전히 새 환경이
필요하면 기존 디렉터리를 직접 백업한 뒤 새 경로를 `GEMM_ROOT`로 지정한다.

수동으로 설치 내용을 확인하려면:

```bash
"$GEMM_ENV/bin/python" - <<'PY'
from importlib.metadata import version
import torch

print(torch.__version__)
print(torch.version.cuda)
print(version("flashinfer-python"))
print(version("flashinfer-cubin"))
PY
```

기대값:

```text
torch = 2.11.0+cu130
torch.version.cuda = 13.0
flashinfer-python = 0.6.12
flashinfer-cubin = 0.6.12
```

## 4. GPU 노드 runtime 환경

대화형 GPU에서 점검하거나 sbatch가 실행될 때 다음 파일을 source한다.

```bash
source /home/hybyun0207/aso-internship/gemm/setup_runtime_env.sh
```

이 스크립트가 설정하는 주요 값:

```text
CUDA_HOME              = CUDA 13 full toolkit
FLASHINFER_WORKSPACE_BASE = FlashInfer JIT cache
TORCH_EXTENSIONS_DIR   = PyTorch extension cache
CUDA_CACHE_PATH        = CUDA driver cache
AUTOTUNE_CACHE_DIR     = FlashInfer tactic JSON cache
FLASHINFER_AUTOTUNER_LOAD_FROM_FILE = 0
```

CUDA driver stub은 link 단계에서만 `LIBRARY_PATH`와
`FLASHINFER_EXTRA_LDFLAGS`에 넣고 runtime `LD_LIBRARY_PATH`에는 넣지 않는다.

GPU 노드 검증:

```bash
source /home/hybyun0207/aso-internship/gemm/setup_runtime_env.sh

"$GEMM_ENV/bin/python" - <<'PY'
from importlib.metadata import version
import torch
import flashinfer

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("FlashInfer:", version("flashinfer-python"))
print("cuDNN:", torch.backends.cudnn.version())
print("GPU:", torch.cuda.get_device_name())
print("Capability:", torch.cuda.get_device_capability())

assert torch.version.cuda == "13.0"
assert version("flashinfer-python") == "0.6.12"
assert torch.cuda.get_device_capability() == (12, 0)
assert torch.backends.cudnn.version() >= 91400
PY
```

## 5. 대화형 스모크 테스트

GPU를 할당받은 상태에서 BF16 FlashInfer 하나만 확인한다.

```bash
cd /home/hybyun0207/aso-internship/gemm
source setup_runtime_env.sh

SMOKE_DIR="$GEMM_PROJECT_DIR/result/smoke-$(date +%Y%m%d-%H%M%S)"

"$GEMM_ENV/bin/python" benchmark_gemm.py \
  --config configs/experiments.yaml \
  --mode smoke \
  --precision bf16 \
  --m-values 128 \
  --warmup 2 \
  --repeat 3 \
  --output-dir "$SMOKE_DIR"
```

첫 FlashInfer 호출은 SM120 JIT 때문에 오래 걸릴 수 있다. JIT 시간은 warmup
전에 발생하므로 CSV의 CUDA Event latency에는 들어가지 않는다.

## 6. 전체 sbatch 실행

```bash
cd /home/hybyun0207/aso-internship/gemm

JOB_ID=$(sbatch --parsable slurm/run_gemm_benchmark.sbatch)

echo "Job: $JOB_ID"
echo "Slurm log: $PWD/result/slurm-${JOB_ID}.out"
echo "Run data: $PWD/result/job-${JOB_ID}"
```

진행 상황:

```bash
squeue -j "$JOB_ID"
tail -f "result/slurm-${JOB_ID}.out"
```

sbatch 실행 순서:

1. CUDA/PyTorch/FlashInfer/cuDNN/SM120 검증
2. FlashInfer BF16 스모크/JIT
3. 동일 quantized input을 사용하는 PyTorch/FlashInfer MXFP8 호환성 스모크
4. BF16, MXFP8, MXFP4 AutoTuner cache 생성
5. trial마다 PyTorch BF16/MXFP8 측정
6. trial마다 FlashInfer default/tuned 측정
7. 전체 raw CSV 집계

Default와 tuned는 별도 Python 프로세스로 실행한다. AutoTuner의 process-local
cache가 default 결과에 섞이지 않는다.

FlashInfer BF16 AutoTuner는 workspace tensor의 shape도 cache key에 포함한다.
기본 32 MiB workspace가 cuDNN tactic 실행 중 소폭 커지면 tuning 때 저장한 key와
tuned 측정 때의 key가 달라져 fallback으로 돌아갈 수 있다. 이를 방지하기 위해
benchmark는 BF16 FlashInfer 프로세스마다 `mm_bf16_workspace`를 64 MiB로 먼저
할당한다. 이 보정은 GEMM timed region 밖에서 수행되며 latency에는 포함되지
않는다.

기본 job은 job별 새 AutoTuner cache를 사용하고 `FORCE_RETUNE=1`로 tuning을
실제로 다시 수행한다. 기존 cache를 재사용하려면:

```bash
sbatch \
  --export=ALL,FORCE_RETUNE=0,AUTOTUNE_CACHE_DIR=/lustre/hybyun0207/gptoss-gemm/cache/autotune \
  slurm/run_gemm_benchmark.sbatch
```

## 7. 축소 실험

긴 전체 실험 전에 config를 복사해 `m_values`, `warmup`, `repeat`를 줄이는 방법을
권장한다. 또는 실행 파일에 직접 override한다.

```bash
"$GEMM_ENV/bin/python" benchmark_gemm.py \
  --config configs/experiments.yaml \
  --mode default \
  --precision mxfp8 \
  --m-values 1 128 1024 \
  --warmup 2 \
  --repeat 3 \
  --output-dir "$GEMM_PROJECT_DIR/result/manual-smoke"
```

AutoTuner를 수동으로 실행하고 같은 cache로 측정하려면:

```bash
CACHE_DIR="$GEMM_ROOT/cache/autotune-manual"
RUN_DIR="$GEMM_PROJECT_DIR/result/manual-tuned"

"$GEMM_ENV/bin/python" benchmark_gemm.py \
  --config configs/experiments.yaml \
  --mode tune \
  --precision bf16 \
  --m-values 1 128 1024 \
  --force-retune \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$RUN_DIR"

"$GEMM_ENV/bin/python" benchmark_gemm.py \
  --config configs/experiments.yaml \
  --mode tuned \
  --precision bf16 \
  --m-values 1 128 1024 \
  --warmup 20 \
  --repeat 50 \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$RUN_DIR"
```

tune과 tuned에서 `m_values` 목록이 같아야 exact tuning bucket이 일치한다.

## 8. 결과 파일

```text
/home/hybyun0207/aso-internship/gemm/result/job-<job-id>/
├── environment.txt
├── tuning-bf16.json
├── tuning-mxfp8.json
├── tuning-mxfp4.json
├── raw_all.csv
└── summary.csv
```

raw CSV는 CUDA Event 반복 1회마다 한 행을 저장한다. 주요 열:

```text
trial, projection, experiment_id, precision, api, M, K, N,
iteration, latency_ms, effective_tflops,
autotune_enabled, autotune_cache_hit,
selected_backend, selected_tactic,
max_abs_error, mean_abs_error, all_finite, status, error
```

`raw_all.csv`는 스모크 테스트를 제외한 모든 측정 결과를 같은 schema로 합친
파일이다. `experiment_id`, `mode`, `precision`, `trial`, `projection` 열로 각
실험을 구분하며 `unsupported`와 `error` 행도 보존한다. 실행 중에는 프로세스별
임시 `raw-*.csv`를 사용하고, 집계가 모두 성공하면 스모크 CSV를 포함한 임시
CSV를 삭제한다. 따라서 정상 완료된 결과 디렉터리에는 `raw_all.csv`와
`summary.csv`만 남는다.

본 측정에서는 default와 tuned 모두 `tune_mode=False`이므로
`autotune_enabled=False`다. tuned 여부는 `experiment_id`와
`autotune_cache_hit=True`로 구분한다. tactic 탐색은 `tuning-*.json`에 별도로
기록된다.

유효 처리량:

```text
FLOPs = 2 * M * K * N
effective_tflops = FLOPs / latency_seconds / 1e12
```

`summary.csv`는 projection/experiment/M별 다음 통계를 제공한다.

```text
mean_ms, median_ms, std_ms, min_ms, p95_ms, mean_tflops
```

## 결과 해석 주의사항

- 작은 M에서 모든 구현의 TFLOP/s가 낮으면 launch overhead와 작은 GEMM의
  구조적 utilization 한계일 가능성이 크다.
- FlashInfer tuned만 빨라지면 default backend/tactic 선택 개선 여지가 있다.
- Router Gate는 N=32인 skinny GEMM이므로 QKV/O와 직접적인 utilization 비교에
  주의한다.
- quantization 시간은 제외했으므로 저정밀 end-to-end latency가 아니다.
- FlashInfer MXFP8 Router Gate는 미지원이며 누락 데이터가 아니다.
- `torch._scaled_mm`은 private API이므로 결과에 PyTorch 정확한 버전을 함께
  기록한다.
- 첫 실행의 JIT 및 AutoTuner 시간은 steady-state GEMM latency와 분리한다.

## 결과 그래프 생성

`raw_all.csv`가 있는 가장 최근 실행을 자동으로 선택해 PNG 그래프를 만든다.

```bash
cd /home/hybyun0207/aso-internship/gemm
source /lustre/hybyun0207/envs/gemm/bin/activate
python plot_results.py
```

실행을 명시하려면:

```bash
python plot_results.py \
  --run-dir result/srun-1957867-20260803-153505
```

그래프는 `<run-dir>/plots/`에 저장된다.

```text
latency_bf16_by_operation.png
latency_mxfp8_by_operation.png
latency_mxfp4_by_operation.png
flashinfer_autotune_speedup.png
speedup_vs_matching_torch_precision.png
```

latency 실선은 repeat 100회의 trial별 평균을 다시 3개 trial에 걸쳐 평균한 값이고,
음영은 세 trial 평균의 표준편차다.
`speedup_vs_matching_torch_precision.png`에서 BF16 FlashInfer의 기준은 PyTorch
BF16 `F.linear`, MXFP8 FlashInfer의 기준은 동일 quantized tensor와 block scale을
사용하는 PyTorch MXFP8 `aten._scaled_mm.out`이다. 각 값은
`PyTorch latency / FlashInfer latency`이므로 1보다 크면 FlashInfer가 빠르다.
PyTorch MXFP4 기준을 측정하지 않으므로 MXFP4는 이 그래프에서 제외한다.
quantization 시간은 양쪽 모두 제외된다.

`flashinfer_autotune_speedup.png`는 BF16 행의 y축을 `0~2`, MXFP8과 MXFP4
행의 y축을 `0~2.5`로 고정한다. 이 범위를 넘는 outlier는 plot 영역 밖에서
잘린다.
AutoTuner speedup은 같은 precision, projection, M에서
`FlashInfer default 평균 latency / FlashInfer tuned 평균 latency`로 계산한다.
따라서 1보다 크면 저장된 AutoTuner tactic을 적용한 실행이 더 빠르고, 1보다
작으면 오히려 tuned 실행이 더 느리다는 뜻이다. PyTorch 값은 이 그래프의
계산에 사용하지 않는다.
`speedup_vs_matching_torch_precision.png`는 BF16 행을 `0~2`로 고정하고, MXFP8
행에는 전체 speedup과 1.0 기준선을 포함하는 공통 y축 범위를 자동으로 적용한다.
따라서 같은 precision 그룹에 있는 QKV, O, Router, Expert W13, Expert W2
패널의 높이를 동일한 기준으로 비교할 수 있다.
