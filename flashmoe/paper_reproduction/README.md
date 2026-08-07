# FlashMoE 논문 프로토콜 재현 실험

이 디렉터리는 기존 BF16 스모크 테스트와 분리된 **FlashMoE-only FP32 본 실험**을
담는다. RTX PRO 6000 Blackwell Server Edition(SM120)에서 논문의 Figure 8, 10,
11, 12에 대응하는 shape와 반복 조건을 실행한다.

GPU 8개를 한 작업에서 안정적으로 할당받기 어렵다는 현재 클러스터 상황을 반영해
2/4/8 GPU를 각각 독립적인 `sbatch` 작업으로 제출한다. 각 GPU 수 안의 조건들은
같은 allocation에서 연속 실행하므로 총 작업은 3개다. 조건 15개를 15개 작업으로
쪼개지 않는다.

## 1. 디렉터리 구조

```text
paper_reproduction/
  README.md
  prepare_source.sh                 # upstream commit의 깨끗한 복제본 생성
  patches/sm120_fp32.patch          # FP32/SM120/seed 전용 patch
  run_paper_benchmark.py            # GPU 수별 조건 실행 및 CSV 작성
  collect_results.py                # 세 작업 검증·병합·Figure별 CSV 생성
  slurm/
    run_job.sh                      # 공통 환경, build, 측정 로직
    run_2gpu.sbatch
    run_4gpu.sbatch
    run_8gpu.sbatch
    submit_all.sh
  generated/                        # 자동 생성된 격리 source; git 제외
  build-sm120-fp32/                 # 논문용 build; 스모크 build와 별개
```

기존 환경과 산출물은 그대로 보존한다.

- 재사용 환경: `/home/hybyun0207/miniconda3/envs/flashmoe`
- 재사용 dependency: CUDA 12.8, MathDx/cuBLASDx 25.12.1, NVSHMEM 3.7.2
- 기존 스모크 build: `../build-sm120-generic` — 본 실험이 수정하지 않음
- 본 실험 build: `build-sm120-fp32`

`prepare_source.sh`는 `/home/hybyun0207/FlashMoE`의 현재 checkout 파일을 그대로
복사하지 않는다. `git archive <현재 HEAD>`로 깨끗한 committed source를 만들고
그 복제본에 논문용 patch만 적용한다. 따라서 기존 스모크용 수정과 섞이지 않는다.
source 경로는 commit hash와 patch hash로 고정된다.

## 2. 고정 조건과 실제 측정 범위

| 항목 | 값 |
|---|---:|
| dtype | FP32 (`Element=float`) |
| hidden dimension `H` | 2048 |
| FFN dimension `I` | 2048 |
| routing | top-2 |
| capacity factor | 1.0 |
| MLP | upstream 기본값 Gated + SiLU |
| warmup | 32 forward |
| measurement | 32 forward |
| CUDA Graph | off (`graph_launches=0`) |
| seed | 42 |
| process | GPU당 MPI/NVSHMEM rank 1개 |
| CPU | rank당 2개, `OMP_NUM_THREADS=1` |
| host RAM | 작업당 64GB (`--mem=64G`) |
| node | 작업별 단일 노드 |

Attention head 16은 논문 모델 metadata지만 이 MoE-only 바이너리의 입력 shape에는
들어가지 않는다. `T`/`S`는 **GPU당 token 수**, `E`는 global expert 수다.
expert는 rank에 block partition되고 local expert 수는 `E / GPU 수`다.

### 중요한 측정 범위

현재 공개 `testFlashMoE`는 gate/router kernel을 한 번 실행해 `tokenIndices`와
`expertCounts`를 준비한 후, `flashmoe::moe::forwardHost`만 warmup 및 측정한다.
따라서 저장되는 latency는 다음 범위다.

```text
router 제외 fused distributed MoE kernel latency
```

Gate부터 Combine까지의 완전한 MoE layer latency라고 부르지 않는다. 바이너리는
32회 전체를 한 CUDA event 구간으로 측정하고 32로 나눈 **rank별 평균 한 개**를
출력한다. 분산 조건의 대표값은 rank별 평균 가운데 가장 느린 값인
`max_rank_mean_latency_ms`다. 개별 32회 latency가 없으므로 이 실행만으로
표준편차나 error bar를 만들 수 없다.

## 3. 실행 행렬

중복 조건을 재사용해 총 15개 unique 조건을 측정한다.

| 작업 | 조건 | 개수 | 대응 Figure |
|---|---|---:|---|
| 2 GPU | `T=8192, E=32` | 1 | 10, 11 기준점 |
| 4 GPU | `E=32, T=4096/8192/16384`; `T=16384, E=8/16/64/128` | 7 | 8, 10, 11, 12 |
| 8 GPU | `E=32, T=4096/8192/16384`; `T=16384, E=8/16/64/128` | 7 | 8, 10, 11, 12 |

Figure별 계산은 다음과 같다.

- Figure 8: 4/8 GPU, `E=32`, `T=4K/8K/16K` latency
- Figure 10: 2/4/8 GPU, `E=32`, `T=8K` throughput
- Figure 11: 같은 scaling 조건의 latency와 overlap efficiency
- Figure 12: 4/8 GPU, `T=16K`, `E=8/16/32/64/128` latency

```text
global_tokens = tokens_per_gpu * gpu_count
throughput_MTokens/s = global_tokens / (latency_ms * 1000)
OE(N) = latency(2 GPU) / latency(N GPU) * 100
```

Figure 9 GPU utilization, Table 1의 kernel launch 수, 다른 시스템과의 상대 성능은
Nsight와 baseline이 없으므로 이번 범위에서 제외한다.

## 4. SM120 전용 변경과 한계

논문 코드는 H100(SM90)의 arch-specific cuBLASDx 경로를 사용한다. 현재
MathDx/cuBLASDx 조합은 SM120 arch-specific 경로로 같은 코드를 컴파일할 수 없어
본 실험은 `sm_modifier::generic`을 사용한다. 이는 단순 호환성 변경이 아니라
GEMM 성능과 전체 latency에 영향을 줄 수 있다.

또한 FP32 Gated MLP의 SM120 tile은 pipeline stage `pS=2`에서 약 128 KiB의
shared memory가 필요하지만 RTX PRO 6000의 block당 opt-in 한도는 약 99 KiB다.
본 실험 patch는 실행 가능하게 만들기 위해 SM120에서 `pS=1`을 사용한다
(선택 tile 기준 약 80 KiB). 기존 BF16 스모크의 `pS=2`와 다르며, 성능 중립적인
변경이 아니다.

| 항목 | 논문 | 현재 실험 |
|---|---|---|
| GPU | H100 80 GB, SM90 | RTX PRO 6000 96 GB, SM120 |
| interconnect | NVLink | 실제 할당의 PCIe topology |
| cuBLASDx | H100 arch-specific | SM120 generic |
| pipeline stage | H100 최적 설정 | FP32 SM120 `pS=1` |
| precision | FP32 | FP32 |
| GPU 수 | 2/4/8 단일 노드 | 각기 별도 단일-node batch |

그러므로 결과는 `FlashMoE-only protocol replication on SM120/PCIe`로 표현하며,
H100 논문 절대 수치의 직접 재현이나 논문 speedup 재현으로 주장하지 않는다.

### 부분 완료 결과 시각화

8-GPU 작업을 기다리는 동안 동일 partition에서 완료된 2/4-GPU 결과만으로 논문
Figure 8, 10, 11, 12에 대응하는 PNG를 만들 수 있다. 누락된 8-GPU 결과는
보간하지 않는다. 공개 바이너리는 개별 32회 latency를 제공하지 않으므로 error
bar도 만들지 않는다.

```bash
/lustre/hybyun0207/envs/gemm/bin/python \
  paper_reproduction/plot_results.py \
  --job-dir results/paper-reproduction/job-1967079-2gpu \
  --job-dir results/paper-reproduction/job-1967080-4gpu \
  --output-dir results/paper-reproduction/asus-2gpu-4gpu/plots
```

생성 파일은 `figure8_forward_latency_asus.png`,
`figure10_throughput_asus.png`, `figure11_weak_scaling_asus.png`,
`figure12_expert_scalability_asus.png`, `plot_data.csv`다. 모든 figure의 latency는
`max_rank_mean_latency_ms`를 사용하며 Figure 11의 overlap efficiency는 논문과
같이 `T(2)/T(N) * 100`으로 계산한다.

## 5. 제출 방법

로그 디렉터리를 먼저 만들기 위해 제공된 제출 스크립트를 사용한다.

```bash
cd /home/hybyun0207/aso-internship/flashmoe
bash paper_reproduction/slurm/submit_all.sh
```

다른 RTX PRO 6000 파티션에 제출할 때는 다음처럼 명시한다.

```bash
PARTITION=gigabyte_pro6000 bash paper_reproduction/slurm/submit_all.sh
```

이 명령은 dependency 없이 서로 독립적인 작업 3개를 제출한다. 한 GPU 수가 오래
대기하거나 실패해도 나머지 작업은 실행될 수 있다. 로그 이름에는 제출 시각,
GPU 수, job ID가 들어간다.

```text
results/paper-reproduction/slurm/slurm-YYYYMMDD-HHMMSS-2gpu-JOBID.out
results/paper-reproduction/slurm/slurm-YYYYMMDD-HHMMSS-4gpu-JOBID.out
results/paper-reproduction/slurm/slurm-YYYYMMDD-HHMMSS-8gpu-JOBID.out
```

하나씩 제출하려면 다음처럼 출력 경로를 명시한다.

```bash
cd /home/hybyun0207/aso-internship/flashmoe
mkdir -p results/paper-reproduction/slurm

sbatch \
  --output="$PWD/results/paper-reproduction/slurm/slurm-$(date +%Y%m%d-%H%M%S)-2gpu-%j.out" \
  paper_reproduction/slurm/run_2gpu.sbatch
```

`run_4gpu.sbatch`, `run_8gpu.sbatch`도 같은 방법으로 제출한다. 모든 파일은
`asus_pro6000`, `pro6000_qos`, `--nodes=1`로 설정돼 있다. 클러스터에 한
노드당 요청 수만큼의 GPU가 없다면 해당 작업은 `(Resources)`로 계속 대기한다.
그 경우 여러 노드로 넓혀 실행하지 말고, 단일 노드 가용 GPU 구성을 먼저
관리자에게 확인해야 논문의 intra-node 조건을 유지할 수 있다.

## 6. 결과 구조와 성공 판정

각 작업은 다음 경로에 독립적으로 저장된다.

```text
results/paper-reproduction/job-<JOBID>-<N>gpu/
  job_status.txt
  environment.txt
  topology.txt
  manifest.json
  raw/<condition>.log
  rank_results.csv
  aggregates.csv
  failures.csv                 # 실패가 있을 때만 생성
```

성공한 로그 끝에는 다음이 보여야 한다.

```text
completed_conditions=1        # 2 GPU
completed_conditions=7        # 4 또는 8 GPU
failed_conditions=0
EXIT_CODE=0
```

`rank_results.csv`에는 rank별 32회 평균이, `aggregates.csv`에는 조건별
`max_rank_mean_latency_ms`와 throughput이 저장된다. `environment.txt`와
`topology.txt`는 별도 batch 간 node/topology 차이를 판단하는 근거다.

세 작업이 끝나면 합친다.

```bash
cd /home/hybyun0207/aso-internship/flashmoe

python paper_reproduction/collect_results.py \
  --job-dir results/paper-reproduction/job-<2GPU_JOBID>-2gpu \
  --job-dir results/paper-reproduction/job-<4GPU_JOBID>-4gpu \
  --job-dir results/paper-reproduction/job-<8GPU_JOBID>-8gpu \
  --output-dir results/paper-reproduction/combined-$(date +%Y%m%d-%H%M%S)
```

수집기는 2/4/8 GPU 작업이 각각 하나인지, 15개 조건이 전부 성공했는지 검사한 뒤
다음을 만든다.

```text
combined_aggregates.csv
figure8.csv
figure10_11.csv
figure12.csv
collection_manifest.json
```

별도 batch가 서로 다른 node에 배정되는 것은 허용하지만 성능 비교의 confound다.
세 작업의 GPU model/driver/partition이 같고 topology가 비교 가능한지 먼저 확인하고,
다르면 결과에 node와 topology를 함께 표시하거나 동일 node에서 재측정한다.

## 7. 완료 기준

- 모든 row가 FP32, `H=I=2048`, top-2, Graph off인지 검증됨
- 2 GPU 1개, 4 GPU 7개, 8 GPU 7개 조건 완료
- 모든 조건 correctness error가 허용 범위 이내
- `failures.csv` 없음, 각 작업 `EXIT_CODE=0`
- 세 작업 모두 단일 노드에서 GPU당 rank 1개로 실행
- software/source patch hash와 topology 저장
- router 제외 범위, generic cuBLASDx, `pS=1`, PCIe 한계를 결과에 명시
