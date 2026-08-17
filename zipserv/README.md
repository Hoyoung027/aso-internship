# ZipServ ZipGEMM 논문 재현 실험 계획

이 디렉터리는 ZipServ 논문의 핵심 kernel 실험인 **BF16 cuBLAS Tensor Core
GEMM과 ZipGEMM의 성능 비교**를 재현하기 위한 계획을 정리한다. 첫 단계에서는
공개 artifact가 제공하는 확률분포 기반 합성 weight를 사용하고, 실험 파이프라인을
검증한 뒤 실제 모델 checkpoint weight로 확장한다.

논문 전체의 end-to-end vLLM 실험이나 DietGPU, nvCOMP, DFloat11 비교는 이번
범위에 포함하지 않는다.

## 바로 실행하기

실험 코드는 이 디렉터리에 구현되어 있다. 모든 모델 shape, batch, Split-K 후보,
warm-up/repeat/trial 수는 단일 파일인
[`configs/experiments.json`](configs/experiments.json)에서 관리한다. 원본 ZipServ
CUDA kernel은 수정하지 않고, 빌드 스크립트가 tuning용 `(20, 100)` binary와 최종
측정용 `(100, 1000)` binary를 별도로 만든다.

### 1. GPU allocation에서 최초 1회 빌드

CUDA가 정상인 RTX 4090 node에서 실행한다. `cs-gpu-01`은 CUDA error 999가
발생했던 node라 기본 Slurm script에서 제외했다.

```bash
cd /home/hybyun0207/aso-internship/zipserv
source scripts/setup_runtime.sh

"$CUDA_PATH/extras/demo_suite/deviceQuery"
scripts/build_benchmarks.sh
```

생성물은 `bin/test_mm_tune`, `bin/test_mm_final`, `bin/libL_API.so`다. 빌드 과정에서
원본 `utils.h`의 반복 횟수를 잠시 바꾸지만 종료 시 원래 내용으로 복구한다.

### 2. 현재 `srun --pty` allocation에서 실행

로그인 node에서 새 allocation을 받을 때는 다음처럼 알려진 문제 node를 제외한다.

```bash
srun \
  --partition=suma_rtx4090 \
  --qos=base_qos \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:RTX4090:1 \
  --cpus-per-task=20 \
  --time=05:00:00 \
  --exclude=cs-gpu-01 \
  --pty bash -l
```

allocation 안에서 `CUDA_VISIBLE_DEVICES`는 Slurm이 지정한 값(보통 `0`)을 그대로
사용한다. `scontrol`에 보이는 물리 GPU `IDX`로 다시 덮어쓰지 않는다.

먼저 LLaMA-3.1-8B 전체 5개 layer를 실행하는 것을 권장한다. 이는 단일 smoke
case가 아니라 5 layers × 3 batches의 Split-K tuning과 최종 측정을 모두 수행한다.

```bash
cd /home/hybyun0207/aso-internship/zipserv
scripts/run_on_allocation.sh all --models llama3.1-8b
```

11개 모델 전체를 실행하려면 filter를 제거한다.

```bash
scripts/run_on_allocation.sh all
```

긴 실험을 tuning과 final로 나누거나 중단 후 재개할 때는 결과 디렉터리를 고정한다.

```bash
export RUN_DIR=/home/hybyun0207/aso-internship/zipserv/results/run-manual-01

scripts/run_on_allocation.sh tune --output-dir "$RUN_DIR"
scripts/run_on_allocation.sh final --output-dir "$RUN_DIR"
```

같은 명령을 다시 실행하면 `status=ok`인 조합은 자동으로 건너뛴다. 다시 측정하려면
`--force`를 붙인다. 모델, layer, batch 범위를 줄이는 예시는 다음과 같다.

```bash
scripts/run_on_allocation.sh all \
  --models llama3.1-8b qwen2.5-7b \
  --layers qkv_proj gateup_proj \
  --batches 8 16 32
```

### 3. 로그인 node에서 `sbatch` 제출

```bash
cd /home/hybyun0207/aso-internship/zipserv

# 먼저 LLaMA-3.1-8B 전체 실험
sbatch slurm/run_zipserv.sbatch all --models llama3.1-8b

# 전체 모델 실험
sbatch slurm/run_zipserv.sbatch all
```

기본 wall time은 5시간이다. 전체 실험이 끝나지 않으면 Slurm log에 출력된 결과
디렉터리를 사용해 이어서 제출한다.

```bash
sbatch slurm/run_zipserv.sbatch all \
  --output-dir /home/hybyun0207/aso-internship/zipserv/results/run-기존경로
```

job별 시간을 바꾸려면 `sbatch --time=...` 옵션을 script 경로 앞에 둔다.

```bash
sbatch --time=10:00:00 slurm/run_zipserv.sbatch all
```

실행 전 조합만 확인하고 GPU 작업을 시작하지 않으려면 다음을 사용한다.

```bash
python3 scripts/run_experiments.py --mode dry-run
```

기본 전체 행렬은 55개 layer shape, 1,980회 tuning 실행, 165회 final 실행이다.
Tuning은 각 Split-K를 3회 독립 실행하고 final은 선택된 Split-K를 1회 실행하되,
각 binary 내부에서 논문 수준의 반복 측정을 수행한다.

### 결과 파일

각 실행은 `results/run-*/` 아래에 다음을 만든다.

```text
environment.txt       GPU, CUDA, git commit과 dirty 상태
manifest.json         실행 인자와 config SHA-256
experiments.json      실행 당시 조합 파일 사본
raw/                  조합별 run.log와 재개용 result.json
selected_splitk.csv   trial median으로 고른 Split-K
measurements.csv      모든 조합의 cuBLAS_TC/ZipGEMM 통합 측정값
summary.csv           ZipGEMM 대 cuBLAS_TC 최종 latency와 speedup
failures.csv          실패/timeout 조합
```

공개 `test_mm`는 cuBLAS non-TC도 함께 실행한다. 이번 비교의 runner는 `test_mm`이
만드는 케이스별 CSV를 읽은 직후 삭제하고 run 디렉터리의 단일
`measurements.csv`에 `cuBLAS_TC`와 `ZipGEMM` 행만 추가한다. non-TC 측정값은
재개용 `result.json`에만 보존된다. 원본 프로그램이 프로세스마다 합성 weight를
다시 생성하므로 각 Split-K 후보는 같은 seed와 분포를 사용하지만 동일한 메모리
인스턴스를 공유하지는 않는다.

### 모델별 Slurm job 제출

긴 전체 실험은 모델별로 결과 디렉터리를 분리해 제출할 수 있다. 각 모델 디렉터리의
`measurements.csv` 하나에 해당 모델의 모든 layer, batch, Split-K 결과가 실행 즉시
누적된다. 동일 모델의 job 두 개를 동시에 실행해 같은 디렉터리에 쓰면 안 된다.

11개 모델을 각각 독립 job으로 한 번에 제출:

```bash
cd /home/hybyun0207/aso-internship/zipserv
slurm/submit_by_model.sh
```

일부 모델만 제출:

```bash
slurm/submit_by_model.sh llama3.1-8b qwen2.5-7b gemma3-12b
```

한 모델만 직접 제출하는 형식:

```bash
sbatch slurm/run_zipserv.sbatch all \
  --models llama3.1-8b \
  --output-dir /home/hybyun0207/aso-internship/zipserv/results/by-model/llama3.1-8b
```

wall time으로 종료되면 같은 명령을 다시 제출하면 된다. 동일한 결과 디렉터리에서
성공한 케이스는 건너뛰고 나머지만 이어서 실행한다.

## 1. 목표와 측정 범위

비교하는 연산은 다음 BF16 GEMM이다.

```text
Y[M,N] = W[M,K] X[K,N]
```

- `cuBLAS_TC`: 압축되지 않은 BF16 weight를 `cublasGemmEx`로 계산
- `ZipGEMM`: TCA-TBE 압축 weight를 kernel 내부에서 복원하면서 Tensor Core로 계산
- `N`: 논문 kernel 실험에서 사용하는 batch 크기
- timed region: 각 GEMM kernel 호출만 포함
- 제외: weight 생성, TCA-TBE 압축, GPU 할당/복사, L2 flush, Split-K 탐색

주요 결과는 다음과 같이 계산한다.

```text
speedup = cuBLAS_TC_latency_ms / ZipGEMM_latency_ms
TFLOPS  = 2 * M * K * N / latency_seconds / 1e12
```

`speedup > 1`이면 ZipGEMM이 빠르다. 성능과 함께 cuBLAS_TC 대비 ZipGEMM의
수치 오차와 TCA-TBE 압축률도 기록한다.

## 2. 기준 구현

원본 구현은 다음 저장소를 사용한다.

```text
/home/hybyun0207/ZipServ_ASPLOS26
```

핵심 파일:

| 파일 | 역할 |
|---|---|
| `csrc/L_Kernel.cuh` | fused ZipGEMM과 standalone decompression kernel |
| `csrc/L_API.cu` | N별 kernel dispatch와 Split-K reduction 호출 |
| `kernel_benchmark/test_mm.cu` | cuBLAS/ZipGEMM 실행, 정확성 검사, CSV 저장 |
| `kernel_benchmark/utils.h` | 합성 weight 생성, 반복 횟수, 결과 출력 |

새 CUDA kernel은 작성하지 않는다. 본 실험에 필요한 추가 작업은 반복 횟수 정렬,
입력 검증, 모델 shape manifest, 자동 실행 runner, 결과 수집 코드다. 원본 구현의
동작을 바꾸는 수정은 별도 commit 또는 patch로 기록한다.

## 3. 논문과 맞출 고정 조건

| 항목 | 설정 |
|---|---|
| dtype | BF16 input/weight/output, FP32 accumulation |
| cuBLAS API | `cublasGemmEx`, `CUBLAS_DEFAULT_MATH`, algorithm 0 |
| warm-up | 100회 |
| timed repeat | 1,000회 |
| batch `N` | 8, 16, 32 |
| Split-K 후보 | 1, 2, 4, 8 |
| 입력 seed | 12345로 고정 |
| cache 조건 | 각 timed iteration 전 L2 크기의 2배 buffer를 write하여 cold-cache 유도 |
| bias | 없음 |
| 측정 장치 | CUDA Event |

`scripts/build_benchmarks.sh`가 공개 코드의 compile-time macro를 사용해 tuning용과
final용 binary를 각각 생성한다. 최종 결과에는 실제 적용된 warm-up/repeat 값이
함께 저장된다.

논문의 주 비교 환경은 다음과 같다.

| 환경 | GPU | CUDA |
|---|---|---|
| 우선 재현 | RTX4090 또는 L40S | CUDA/cuBLAS 12.4 계열 |
| forward-compatibility | RTX5090 | CUDA 12.8 |

다른 GPU에서 실행한 결과는 기능 및 경향 재현으로 분리하고 논문의 절대 성능과
직접 비교하지 않는다. 특히 RTX PRO 6000 Blackwell에서 실행하려면 Makefile의
`SMS`에 120을 지정하고 CUDA 12.8 이상을 사용해야 한다.

## 4. 실험 모델과 레이어

논문 Section 6.1과 동일한 모델군을 사용한다.

| 모델 계열 | 크기 |
|---|---|
| LLaMA-3.1 | 8B, 70B, 405B |
| Qwen2.5 | 7B, 14B, 32B, 72B |
| Gemma-3 | 12B, 27B |
| Mistral | 24B, 123B |

각 모델에서 다음 weight shape를 측정한다.

| 레이어 ID | 의미 |
|---|---|
| `qkv_proj` | 병합된 Query/Key/Value projection |
| `o_proj` | attention output projection |
| `gateup_proj` | 병합된 FFN gate/up projection |
| `down_proj` | FFN down projection |
| `lm_head` | vocabulary projection |

모델별 정확한 `M,K`는 추정값으로 입력하지 않고 해당 모델의 공식 config와
checkpoint tensor shape로 검증한 뒤 `configs/shapes.csv`에 고정한다. GQA 모델의
`qkv_proj`는 단순히 `3 * hidden_size`로 계산하면 안 되며 Q head와 KV head 수를
반영해야 한다. `GateUp`은 gate와 up 두 weight의 출력 차원을 합친 값이다.

예를 들어 LLaMA-3.1-8B의 대표 shape는 다음과 같다.

| 레이어 | M | K |
|---|---:|---:|
| `qkv_proj` | 6144 | 4096 |
| `o_proj` | 4096 | 4096 |
| `gateup_proj` | 28672 | 4096 |
| `down_proj` | 4096 | 14336 |
| `lm_head` | 128256 | 4096 |

모든 모델과 레이어 shape가 확정되면 기본 측정점 수는 다음과 같다.

```text
11 models * 5 layers * 3 batches = 165 final configurations
```

일부 모델이 layer를 병합하지 않은 checkpoint를 제공하더라도 논문과 동일하게
QKV와 Gate/Up의 출력 차원을 논리적으로 병합한 GEMM shape를 사용한다.

## 5. 1단계 입력: 확률분포 기반 합성 weight

첫 실험에서는 공개 `test_mm`의 입력 생성기를 그대로 사용한다.

- 고정 seed: 12345
- 약 95%의 weight를 7개 고빈도 exponent에서 생성
- 나머지는 fallback exponent 범위에서 생성
- activation은 `[-0.1, 0.1]`의 난수 BF16
- 같은 프로세스 안에서 cuBLAS_TC와 ZipGEMM이 동일한 weight와 activation 사용

이 단계의 목적은 다음과 같다.

1. 모든 모델/레이어 shape에서 kernel이 정상 실행되는지 확인
2. ZipGEMM 결과가 cuBLAS_TC 기준 허용 오차 안에 있는지 확인
3. batch별 Split-K 최적값 결정
4. 논문 Figure 11의 성능 경향과 비교
5. 실제 checkpoint를 도입하기 전에 실험 및 수집 파이프라인 고정

이 결과는 `synthetic-distribution benchmark`라고 명시한다. 모델 이름은 실제
weight를 사용했다는 뜻이 아니라 실제 모델의 GEMM shape를 사용했다는 뜻이다.

## 6. Split-K 선택 프로토콜

Split-K는 ZipGEMM에만 적용된다. 현재 `test_mm`에는 기본값이 없으므로 네 번째
위치 인자로 반드시 1 이상의 값을 전달해야 한다.

```bash
./test_mm M K N SplitK --model MODEL --layer LAYER
```

Split-K가 커지면 K 방향 병렬성과 block 수가 증가하지만 임시 workspace와 마지막
reduction 비용도 증가한다. 논문은 shape별 값을 공개하지 않았으므로 아래 절차를
재현 규칙으로 사용한다.

### 사전 튜닝

각 `(model, layer, M, K, N)`에 대해:

1. `SplitK={1,2,4,8}` 실행
2. 각 후보를 짧은 pilot 조건으로 3회 독립 실행
3. 실행 실패, timeout 또는 비정상 latency 후보 제외
4. 세 실행의 median ZipGEMM latency가 가장 작은 후보 선택
5. median이 정확히 같으면 더 작은 Split-K 선택
6. 선택 결과를 해당 run의 `selected_splitk.csv`에 저장

권장 pilot 조건은 warm-up 20회, repeat 100회다. pilot 결과는 최종 성능 집계에
포함하지 않는다. 반복 횟수는 compile-time macro이므로 빌드 스크립트가 tuning과
final binary를 따로 컴파일한다.

### 최종 측정

선택한 Split-K만 사용하여 논문 조건인 warm-up 100회와 timed repeat 1,000회를
수행한다. cuBLAS_TC는 Split-K와 무관하므로 `(M,K,N)`당 한 번만 측정한다.

공개 `test_mm`의 동일 프로세스 안에서 cuBLAS_TC와 ZipGEMM을 측정한다. 따라서
tuning 중에는 cuBLAS와 non-TC cuBLAS도 후보마다 중복 측정되지만 Split-K 선택에는
ZipGEMM latency만 사용한다. 최종 `summary.csv`도 cuBLAS_TC와 ZipGEMM만 비교한다.

## 7. 단계별 실행 계획

### Phase A: 환경 및 빌드 확인

GPU node에서 다음을 저장한다.

```bash
nvidia-smi
nvcc --version
```

추가로 GPU 이름, compute capability, driver, CUDA runtime, cuBLAS version,
power limit, application clock을 `environment.txt`에 기록한다.

합성 C++ benchmark에는 Python 가상환경이 필요하지 않다. CUDA Toolkit, cuBLAS,
지원 NVIDIA driver, `nvcc`, `g++`만 필요하다.

현재 시스템의 CUDA 12.8 경로를 사용할 때의 빌드 예시는 다음과 같다.

```bash
cd /home/hybyun0207/ZipServ_ASPLOS26
source Init.sh

make -C build \
  CUDA_PATH=/opt/ohpc/pub/apps/cuda/12.8

source kernel_benchmark/test_env

make -C kernel_benchmark \
  CUDA_PATH=/opt/ohpc/pub/apps/cuda/12.8
```

RTX4090/L40S는 기본 `SMS=80 86 89`의 `sm_89` binary를 사용한다. RTX5090이나
RTX PRO 6000은 다음처럼 빌드 대상을 명시한다.

```bash
make -C build CUDA_PATH=/opt/ohpc/pub/apps/cuda/12.8 SMS=120
make -C kernel_benchmark CUDA_PATH=/opt/ohpc/pub/apps/cuda/12.8 SMS=120
```

### Phase B: 단일 shape smoke test

LLaMA-3.1-8B GateUp, `N=32`, `SplitK=1`로 먼저 검증한다.

```bash
cd /home/hybyun0207/ZipServ_ASPLOS26/kernel_benchmark

./test_mm 28672 4096 32 1 \
  --model llama3.1-8b \
  --layer gateup_proj
```

확인 항목:

- cuBLAS_TC와 ZipGEMM 모두 CUDA error 없이 종료
- latency와 TFLOPS가 0보다 큼
- compression ratio가 합리적인 범위
- ZipGEMM과 cuBLAS_TC 결과 오차가 허용 범위 이내
- CSV row의 `M,K,N,SplitK`가 실행 인자와 일치

### Phase C: Split-K pilot tuning

논문 batch에 대해 네 후보를 실행한다.

```bash
for batch in 8 16 32; do
  for split_k in 1 2 4 8; do
    ./test_mm 28672 4096 "$batch" "$split_k" \
      --model llama3.1-8b \
      --layer gateup_proj
  done
done
```

실제 전체 튜닝은 runner가 `configs/shapes.csv`를 읽어 수행한다. 후보별 결과와
선택 사유를 모두 보존하며 가장 빠른 결과만 남기고 나머지를 삭제하지 않는다.

### Phase D: 합성 weight 최종 측정

선택된 Split-K manifest를 사용해 165개 설정을 실행한다. 각 설정에서:

```text
cuBLAS_TC: warm-up 100 + timed 1,000
ZipGEMM:   warm-up 100 + timed 1,000
```

실험 도중 GPU를 다른 작업과 공유하지 않는다. 가능하면 persistent mode와 clock을
고정하되 권한이 없으면 실제 clock, temperature, power 상태를 로그에 남긴다.

### Phase E: 집계 및 논문 비교

다음을 생성한다.

- 설정별 raw latency/TFLOPS/speedup 표
- RTX4090 또는 L40S 전체 평균 speedup
- 모델별 speedup
- 레이어별 speedup
- batch별 speedup
- ZipGEMM이 cuBLAS_TC보다 느린 shape 목록
- 압축률과 speedup의 관계
- Split-K 선택 분포

평균 방식의 불확실성을 피하기 위해 arithmetic mean과 geometric mean을 모두
계산하되 논문 수치와 비교하는 표에는 사용한 평균 방식을 명시한다.

## 8. 결과 파일 구조

현재 구현은 다음 구조를 사용한다.

```text
zipserv/
├── README.md
├── configs/
│   └── experiments.json
├── bin/
│   ├── test_mm_tune
│   ├── test_mm_final
│   └── libL_API.so
├── scripts/
│   ├── build_benchmarks.sh
│   ├── setup_runtime.sh
│   ├── run_on_allocation.sh
│   └── run_experiments.py
├── slurm/
│   ├── run_zipserv.sbatch
│   └── submit_by_model.sh
└── results/
    └── run-<timestamp>-<job>/
        ├── environment.txt
        ├── manifest.json
        ├── raw/
        ├── selected_splitk.csv
        ├── measurements.csv
        ├── summary.csv
        └── failures.csv
```

`measurements.csv`의 주요 schema:

```text
phase,status,error,model,layer,M,K,N,split_k,trial,
warmup,repeat,weight_source,seed,kernel,latency_ms,tflops,
compression_ratio,total_absolute_error,max_relative_error,
average_relative_error,significant_error_count,log
```

집계 단계에서 다음 파생 column을 추가한다.

```text
speedup_vs_cublas_tc = cublas_tc_latency_ms / zipgemm_latency_ms
```

## 9. 정확성 및 성공 기준

각 설정은 다음 조건을 만족해야 성공이다.

- 프로세스 exit code 0
- CUDA/cuBLAS error 없음
- NaN/Inf latency 또는 출력 없음
- cuBLAS_TC와 ZipGEMM 비교 오차가 누락 없이 기록됨
- warm-up 100, repeat 1,000이 manifest와 일치
- 동일 `M,K,N`에서 두 kernel이 같은 입력 사용
- 선택한 Split-K가 tuning manifest와 일치
- 결과에 GPU/CUDA/cuBLAS/source commit 기록

현재 공개 코드는 다음 비교값을 출력하며 runner가 함께 저장한다.

- total absolute error
- max relative error
- average relative error
- significant error element count와 비율

논문이 lossless라고 말하는 대상은 weight 압축과 복원이다. GEMM 결과는 Tensor
Core 연산 및 reduction 순서 때문에 서로 bit-exact하지 않을 수 있으므로 weight
복원의 bit-exact 검증과 GEMM 출력 tolerance 검증을 구분한다.

## 10. 2단계: 실제 checkpoint weight 실험

합성 실험을 완료한 후 실제 BF16 weight로 확장한다. 이 단계에는 별도의 Python
가상환경을 권장한다.

필요 패키지:

- PyTorch
- Transformers
- safetensors
- NumPy

처리 흐름:

```text
Hugging Face checkpoint
  -> 대상 layer BF16 tensor 추출
  -> QKV 또는 GateUp weight 병합
  -> metadata가 포함된 raw binary 저장
  -> test_mm에서 binary 로드
  -> 동일 weight를 cuBLAS_TC와 TCA-TBE/ZipGEMM에 사용
```

실제 checkpoint 실험에서는 모델 revision, tensor 이름, 원본 shape, 병합 순서,
파일 checksum을 manifest에 기록한다. 모델별 weight가 매우 크므로 checkpoint와
추출 binary는 home repository가 아닌 대용량 저장소에 둔다.

합성 결과와 실제 checkpoint 결과는 별도 디렉터리에 저장하며 직접 평균하지
않는다.

## 11. 해석상의 한계

- 논문은 shape별 Split-K 값을 공개하지 않아 동일한 tuning 결과를 보장할 수 없다.
- 공개 benchmark는 실제 checkpoint가 아닌 합성 exponent 분포를 사용한다.
- CUDA/cuBLAS 버전이 다르면 cuBLAS baseline과 kernel code generation이 달라진다.
- GPU 종류, clock, power limit, 온도, 다른 프로세스의 부하가 latency에 영향을 준다.
- 작은 `O_proj`와 같은 shape에서는 ZipGEMM이 cuBLAS_TC보다 느릴 수 있다.
- RTX PRO 6000 결과는 RTX4090/L40S 논문 수치의 직접 재현이 아니다.

따라서 1단계 결과는 다음과 같이 표현한다.

> 실제 LLM layer shape와 논문의 batch/반복 조건을 사용한 ZipServ 공개
> artifact의 합성-weight kernel-level reproduction.

실제 checkpoint까지 사용한 뒤에만 `real-weight reproduction`으로 구분한다.

## 12. 완료 조건

### 1단계 완료

- 11개 모델의 5개 layer shape 검증 완료
- `N={8,16,32}` 적용
- 모든 설정의 Split-K tuning 완료
- 선택된 설정으로 warm-up 100회, repeat 1,000회 완료
- cuBLAS_TC/ZipGEMM latency, TFLOPS, speedup, 정확성, 압축률 저장
- 실패 및 미지원 shape를 누락하지 않고 기록
- 환경과 source commit을 포함한 재현 manifest 생성
- 모델/레이어/batch별 표와 Figure 11 대응 그래프 생성

### 2단계 완료

- 실제 모델 revision과 weight checksum 고정
- 실제 BF16 weight 추출 및 병합 검증
- TCA-TBE weight 복원의 bit-exact 검증
- 동일 측정 matrix로 cuBLAS_TC와 ZipGEMM 재실행
- 합성 weight 결과와 실제 weight 결과 비교
