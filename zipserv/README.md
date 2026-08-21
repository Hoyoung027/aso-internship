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

각 모델은 독립 Slurm job 하나에서 실행된다. 모델 job은 자기
`raw/<model>/result.csv`만 기록하므로 동시 쓰기 충돌이 없다. 모든 모델 job이
종료되면 dependency로 연결된 collector job이 최상위 `result_all.csv`를 만든다.

실험 디렉터리 이름에는 시스템, GPU, 가중치 종류, 날짜와 단계가 포함된다.

```text
zipserv-rtx4090-synthetic-20260822-tuning
zipserv-rtx4090-synthetic-20260822-153000-run
```

## 디렉터리

```text
zipserv/
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
bin/test_mm_tune   warm-up 100, repeat 1000
bin/test_mm_final  warm-up 100, repeat 1000
bin/libL_API.so
bin/build_manifest.txt
```

현재 tune과 final의 warm-up/repeat 설정이 같으므로 빌드 스크립트는 benchmark를
한 번만 컴파일하고 `test_mm_tune`, `test_mm_final` 두 이름으로 동일 binary를
배치한다. 두 이름은 runner가 실행 단계를 구분하기 위한 것이며 kernel이나 GEMM
알고리즘 차이는 없다.

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
실행한다. 각 실행 내부에서는 tune binary가 warm-up 100회와 측정 1000회를
수행한다. 따라서 tuning 결과도 최종 실험과 동일한 반복 조건의 성능 결과로
사용할 수 있다.

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

## 중단 후 재개

제출 시 출력된 run directory를 다시 넘긴다.

```bash
slurm/run_zipserv.sh \
  --run-dir results/zipserv-rtx4090-synthetic-20260822-153000-run \
  --models llama3.1-8b qwen2.5-7b \
  --layers qkv_proj o_proj
```

성공한 `(phase, model, layer, M, K, N, Split-K, trial)` 행은 건너뛰고 실패하거나
없는 행만 다시 실행한다. 동일한 run directory의 같은 모델 job을 동시에 두 개
제출하면 안 된다.

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
model, layer, M, K, N, split_k, trial
warmup, repeat, weight_source, seed
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

CUDA 배경 지식과 ZipServ 원본 코드 읽기 순서는
[`source/README.md`](source/README.md)에 정리되어 있다.
