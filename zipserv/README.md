# ZipServ ZipGEMM 재현 실험

이 디렉터리는 `ZipServ_ASPLOS26`의 BF16 ZipGEMM kernel을 RTX 4090에서
재현하기 위한 실험 하네스다. 실제 model checkpoint가 아니라 ZipServ 공개
benchmark의 합성 exponent 분포 weight를 사용하며, cuBLAS Tensor Core GEMM과
ZipGEMM의 kernel latency를 비교한다.

## 실행 구조

Split-K 탐색은 한 번만 수행하고 이후 성능 실험에서 계속 재사용한다.

```text
최초 1회 tuning
  모든 (model, layer, N, Split-K) 후보 측정
    → results/zipserv-rtx4090-synthetic-<date>-tuning/result_all.csv
    → results/zipserv-rtx4090-synthetic-<date>-tuning/selected_splitk.csv

반복 가능한 성능 실험
  selected_splitk.csv 읽기
    → 선택된 Split-K로 final binary 실행
    → results/zipserv-rtx4090-synthetic-<date-time>-run/result_all.csv
```

각 모델은 기본적으로 독립 Slurm job 하나에서 실행된다. `--blocks`를 사용하면
모델×block마다 별도 job과 `raw/<model>/block-<index>/result.csv`가 생성되므로
동시 쓰기 충돌이 없다. 모든 job이 종료되면 dependency로 연결된 collector job이
최상위 `result_all.csv`를 만든다.

실험 디렉터리 이름에는 시스템, GPU, 가중치 종류, 날짜와 단계가 포함된다.

```text
zipserv-rtx4090-synthetic-20260822-tuning
zipserv-rtx4090-synthetic-20260822-153000-run
```

## 디렉터리

```text
zipserv/
├── benchmark/
│   ├── test_mm.cu
│   ├── utils.h
│   └── Makefile
├── configs/
│   └── experiments.json
├── scripts/
│   ├── setup_runtime.sh
│   ├── build_benchmarks.sh
│   ├── run_on_allocation.sh
│   ├── run_experiments.py
│   └── collect_results.py
├── slurm/
│   ├── run_zipserv.sh
│   ├── run_zipserv.sbatch
│   └── collect_results.sbatch
├── plots/
└── results/
    ├── zipserv-rtx4090-synthetic-<date>-tuning/
    └── zipserv-rtx4090-synthetic-<date-time>-run/
```

## 1. 최초 빌드

RTX 4090 GPU allocation에서 한 번 실행한다.

```bash
cd /home/hybyun0207/aso-internship/zipserv
source scripts/setup_runtime.sh
scripts/build_benchmarks.sh
```

생성물:

```text
bin/test_mm_tune   warm-up 20, repeat 200
bin/test_mm_final  warm-up 100, repeat 1000
bin/libL_API.so
bin/build_manifest.txt
```

현재 tune과 final의 warm-up/repeat 설정이 다르므로 빌드 스크립트는
`test_mm_tune`과 `test_mm_final`을 별도로 컴파일한다. 두 binary의 kernel과 GEMM
알고리즘은 같고 반복 횟수만 다르다.

benchmark frontend인 `test_mm.cu`, `utils.h`, Makefile은 이 저장소의
`benchmark/`에서 관리한다. `build_benchmarks.sh`는 이 로컬 소스를 컴파일하되,
ZipGEMM kernel과 `L_API.cuh`/`libL_API.so`는 `configs/experiments.json`의
`paths.zipserv_source`가 가리키는 외부 `ZipServ_ASPLOS26`에서 빌드한다. 따라서
실제 weight loader 같은 실험 전용 변경은 원본 artifact가 아니라 로컬
`benchmark/`에 적용한다.

## 2. Split-K tuning: 최초 한 번

활성화된 모든 모델과 모든 레이어를 tuning한다.

```bash
slurm/run_zipserv.sh --mode tune
```

일부 모델이나 레이어만 tuning할 수도 있다.

```bash
slurm/run_zipserv.sh \
  --mode tune \
  --models llama3.1-8b qwen2.5-7b \
  --layers qkv_proj o_proj
```

대형 모델에는 기본 5시간이 부족할 수 있다. 클러스터 QoS가 허용한다면 모델
job의 wall time을 늘린다.

```bash
slurm/run_zipserv.sh --mode tune --time 12:00:00
```

현재 각 `(model, layer, N)`에 대해 Split-K `1, 2, 4, 8`을 각각 한 번
실행한다. 각 실행 내부에서는 tune binary가 warm-up 20회와 측정 200회를
수행한다. 최종 성능은 warm-up 100회, 측정 1000회, trial 3회인 final run에서
별도로 측정한다.

결과:

```text
results/zipserv-rtx4090-synthetic-20260822-tuning/
├── experiments.json
├── logs/
│   ├── llama3.1-8b.log
│   ├── llama3.1-8b-slurm-<job>.out
│   └── collector-slurm-<job>.out
├── raw/
│   ├── llama3.1-8b/
│   │   ├── result.csv
│   │   ├── manifest.json
│   │   └── experiments.json
│   └── qwen2.5-7b/
│       └── result.csv
├── result_all.csv
├── selected_splitk.csv
└── failures.csv
```

동일한 tuning 명령을 다시 실행하면 `result.csv`에서 이미 성공한 조합을 찾아
건너뛰고 미완료 조합만 실행한다. GPU, CUDA, kernel, shape 또는 Split-K 후보가
변경되면 tuning 디렉터리를 별도로 지정해 다시 측정한다.

```bash
slurm/run_zipserv.sh \
  --mode tune \
  --tuning-dir results/tuning-cuda-new
```

### Tuning과 final run 자동 연결

`both` 모드는 tuning 모델 job, tuning collector, final 모델 job, final collector를
Slurm dependency로 연결해 한 번에 제출한다. 모든 tuning 모델 job과 collector가
성공해야 final 모델 job이 시작된다.

```bash
slurm/run_zipserv.sh \
  --mode both \
  --models llama3.1-8b llama3.1-70b \
  --layers qkv_proj o_proj gateup_proj down_proj lm_head \
  --time 12:00:00
```

대기 중인 final job은 `squeue`에서 dependency 상태로 보이며 사용자가 별도로
`run` 명령을 실행할 필요가 없다.

## 3. 성능 실험

`run`이 기본 모드다.

```bash
# 모든 활성 모델과 모든 레이어
slurm/run_zipserv.sh

# 선택 실행
slurm/run_zipserv.sh \
  --models llama3.1-8b qwen2.5-7b \
  --layers qkv_proj o_proj
```

기본적으로 같은 GPU/가중치 조건의 가장 최근 tuning 디렉터리에서
`selected_splitk.csv`를 읽고 새로운 run 디렉터리를 만든다.

```text
results/zipserv-rtx4090-synthetic-20260822-153000-run/
├── experiments.json
├── logs/
│   ├── llama3.1-8b.log
│   ├── llama3.1-8b-slurm-<job>.out
│   ├── qwen2.5-7b.log
│   ├── qwen2.5-7b-slurm-<job>.out
│   └── collector-slurm-<job>.out
├── raw/
│   ├── llama3.1-8b/
│   │   └── result.csv
│   └── qwen2.5-7b/
│       └── result.csv
├── result_all.csv
├── summary.csv
└── failures.csv
```

특정 tuning 결과를 사용하려면 다음과 같이 지정한다.

```bash
slurm/run_zipserv.sh \
  --tuning-dir results/tuning-cuda-new \
  --models llama3.1-8b
```

### 실제 가중치의 여러 transformer block 실행

`--blocks`는 `MODEL=IDX[,IDX...]` 형식이다. config의 `block_index`를 수정하지 않고
모델×block별 Slurm job을 제출하며, 기존 block 0 tuning의 Split-K를 재사용할 수
있다. `--models`를 생략하면 `--blocks`에 적은 모델만 자동으로 선택한다.

```bash
slurm/run_zipserv.sh \
  --mode run \
  --blocks llama3.1-8b=16,31 llama3.1-70b=40,79 \
  --layers qkv_proj o_proj gateup_proj down_proj \
  --tuning-dir results/zipserv-rtx4090-llama31realblock0-20260825-tuning \
  --time 12:00:00
```

위 명령은 네 개의 model/block job을 제출한다. `lm_head`는 transformer block과
무관한 공통 가중치이므로 block 비교에서는 제외한다.

```text
logs/
├── llama3.1-8b-block-16.log
├── llama3.1-8b-block-31.log
├── llama3.1-70b-block-40.log
└── llama3.1-70b-block-79.log
raw/
├── llama3.1-8b/
│   ├── block-16/result.csv
│   └── block-31/result.csv
└── llama3.1-70b/
    ├── block-40/result.csv
    └── block-79/result.csv
```

자동 생성되는 결과 디렉터리 이름에도 요청한 모델과 block 번호가 포함된다.
각 Slurm job과 collector는 제출 시 복사한 `experiments.json`을 사용하므로 작업이
대기 중일 때 원본 config를 수정해도 이미 제출된 실험에는 영향을 주지 않는다.

## 중단 후 재개

제출 시 출력된 run directory를 다시 넘긴다.

```bash
slurm/run_zipserv.sh \
  --run-dir results/zipserv-rtx4090-synthetic-20260822-153000-run \
  --models llama3.1-8b qwen2.5-7b \
  --layers qkv_proj o_proj
```

성공한 `(phase, model, block_index, layer, M, K, N, Split-K, trial)` 행은
건너뛰고 실패하거나 없는 행만 다시 실행한다. 동일한 run directory의 같은
model/block job을 동시에 두 개 제출하면 안 된다.

## Dry-run

실제 job을 제출하지 않고 모델 job과 collector dependency를 확인한다.

```bash
slurm/run_zipserv.sh --mode tune --models llama3.1-8b --dry-run
slurm/run_zipserv.sh --models llama3.1-8b --layers qkv_proj --dry-run
```

## 결과 CSV

모델별 `raw/<model>/result.csv`와 통합 `result_all.csv`는 동일한 schema를 사용한다.
한 benchmark 실행이 한 행이며 주요 열은 다음과 같다.

```text
phase, status, error
model, block_index, layer, M, K, N, split_k, trial
warmup, repeat, weight_source, weight_model_dir, seed
cublas_latency_ms, cublas_tflops
cublas_tc_latency_ms, cublas_tc_tflops
zipgemm_latency_ms, zipgemm_tflops
compression_ratio
tc_speedup_vs_non_tc
zipgemm_speedup_vs_non_tc
zipgemm_speedup_vs_tc
tc_vs_non_tc_*(오차 통계)
zip_vs_non_tc_*(오차 통계)
zip_vs_tc_*(오차 통계)
wall_seconds, log_file
```

`selected_splitk.csv`는 tuning 결과의 ZipGEMM latency가 가장 작은 후보를 담는다.
동률이면 작은 Split-K를 선택한다.

`result.csv`, `result_all.csv`, `summary.csv`에는 세 구현의 모든 두 개 조합에
대한 성능 비율이 저장된다.

```text
tc_speedup_vs_non_tc       = non_tc_latency / tc_latency
zipgemm_speedup_vs_non_tc  = non_tc_latency / zipgemm_latency
zipgemm_speedup_vs_tc      = tc_latency / zipgemm_latency
```

## 로그

각 모델의 모든 `test_mm` stdout/stderr는 하나의 파일에 case 구분자와 함께
누적된다.

```text
logs/llama3.1-8b.log
logs/qwen2.5-7b.log
```

Slurm 자체 출력은 모델명이 포함된 별도 파일이다.

```text
logs/llama3.1-8b-slurm-<job>.out
```

## 수동 집계

collector job이 실패했거나 실행 중간 결과를 확인하려면 직접 집계할 수 있다.

```bash
python3 scripts/collect_results.py \
  --mode run \
  --run-dir results/zipserv-rtx4090-synthetic-20260822-153000-run
```

tuning 결과를 다시 집계하려면 `--mode tune`을 사용한다.

## 시각화

성능 run directory의 `result_all.csv`를 입력으로 사용한다.

```bash
python3 plots/plot_results.py \
  --results-root results/zipserv-rtx4090-synthetic-20260822-153000-run
```

Split-K tuning latency를 모델별로 시각화하려면 다음을 실행한다. 결과 경로를
생략하면 가장 최근의 `zipserv-*-tuning` 디렉터리를 자동으로 사용한다.

```bash
python3 plots/plot_latency_by_splitk.py
```

각 모델마다 `plots/latency_by_splitk/<model>_latency_by_splitk.png`를 만든다.
상단은 레이어별 최적 Split-K를 batch별 막대로, 중단은 다섯 레이어 각각의
Split-K별 latency를, 하단은 Split-K=1 대비 최적 speedup을 보여준다.

LLaMA 3.1 8B/70B의 selected Split-K 결과에 대한 수치 오차 그래프는 다음과
같이 생성한다.

```bash
python3 plots/plot_llama_error.py
```

`plots/llama_error/`에 모델별 PNG를 만든다. cuBLAS non-TC를 기준으로
cuBLAS TC와 ZipServ의 평균 상대 오차, 원소당 평균 절대 오차, 상대 오차가
`1e-4`를 넘는 원소 비율을 비교한다. 두 모델은 동일한 y축 범위를 사용한다.

CUDA 배경 지식과 ZipServ 원본 코드 읽기 순서는
[`source/README.md`](source/README.md)에 정리되어 있다.
