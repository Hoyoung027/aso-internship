# FlashMoE SM120 generic performance experiment

FlashMoE만을 대상으로 하는 논문 Figure 8·10·11·12 본 실험 계획은
[paper_reproduction/README.md](paper_reproduction/README.md)에 정리되어 있다.
본 실험은 FP32이며 2/4/8 GPU를 각각 독립적인 batch로 실행한다. 기존
`run_benchmark.sbatch`는 BF16 SM120 포팅용이며 논문 본 실험 runner가 아니다.

RTX PRO 6000 단일 노드에서 SM120 generic으로 포팅한 FlashMoE의 정확성과
fused distributed MoE kernel 실행 시간을 측정한다. 실제 모델이나 데이터셋을
실행하지 않고 upstream C++ `testFlashMoE`에 Llama 4 Scout 형태의 BF16 shape를
입력한다.

## 기본 실험 조건

| 항목 | 값 |
|---|---:|
| GPU | RTX PRO 6000 Blackwell |
| 기본 GPU 수 | 2 |
| GPU당 토큰 | 1024 |
| hidden dimension | 5120 |
| FFN dimension | 8192 |
| 전체 expert | 16 |
| top-k | 1 |
| MLP | Gated MLP + SiLU |
| dtype | BF16 |

`testFlashMoE`는 먼저 fused 결과와 reference 결과를 비교해 rank별
`error(%)`를 검증하고 CUDA event로 `FlashMoE_Time(ms)`를 측정한다. router는
측정 전에 한 번 실행되므로 이 시간은 router를 제외한 FlashMoE fused
distributed kernel latency다.

## 논문 재현 범위와 RTX PRO 6000의 한계

이 실험은 FlashMoE 논문의 성능 수치를 직접 재현하는 실험이 아니다. 논문의
주요 성능 평가는 단일 노드 **8 x NVIDIA H100** 환경에서 수행되었지만, 현재
사용 가능한 환경은 단일 노드 **8 x RTX PRO 6000 Blackwell(SM120)** 이다.
RTX PRO 6000 Blackwell Server Edition은 NVLink를 지원하지 않으며, 실제 2 GPU
할당에서도 GPU 사이가 PCIe/CPU interconnect를 통과하는 `SYS`로 확인되었다.
할당되는 GPU 쌍에 따라 `PIX`, `PXB`, `PHB`, `NODE`, `SYS` 중 실제 경로는
달라질 수 있으므로 매 작업에서 `nvidia-smi topo -m`을 저장한다. 따라서 다음
차이가 동시에 존재한다.

- GPU 아키텍처: H100의 SM90과 RTX PRO 6000의 SM120
- GPU 수: 논문은 8 GPU, 기존 포팅 benchmark 기본값은 2 GPU이며 본 계획은
  단일 노드 1/2/4/8 GPU를 모두 측정
- GPU 간 연결: 논문의 NVLink와 현재 PCIe/NUMA 연결(최근 2 GPU에서 `SYS` 확인)
- 메모리 대역폭, Tensor Core 구현 및 통신/연산 중첩 조건
- cuBLASDx가 선택하는 커널 구현

특히 upstream FlashMoE v0.1.2는 `Arch >= 900`이면 cuBLASDx의
`arch_specific` 모드를 선택한다. 설치된 MathDx 25.12.1의 cuBLASDx 0.5.1은
SM90 등에 대한 `arch_specific` 모드는 제공하지만 SM120에 대해서는
`generic` 모드만 허용한다. 따라서 원본 조건 그대로 SM120을 컴파일하면
`SM<1200, arch_specific>` 관련 정적 검증에서 실패한다.

SM120에서 실행하려면 다음 중 하나가 필요하다.

1. SM120 `arch_specific`을 공식 지원하는 cuBLASDx/FlashMoE 조합을 사용한다.
2. FlashMoE의 SM120 경로를 `generic`으로 변경해 포팅한다.

현재 설치 조합에서는 두 번째 방법만 가능하다. `generic`은 CPU fallback은
아니며 GPU와 Tensor Core를 사용할 수 있지만, H100 논문 실험의
`arch_specific` 경로와 동일한 구현은 아니다. 따라서 generic 경로의 결과는
다음과 같이 해석한다.

- 허용: SM120에서의 컴파일/실행 가능성 및 reference 대비 정확성 검증
- 허용: 동일한 RTX PRO 6000 환경에서 여러 구현을 비교하는 탐색적 성능 평가
- 제한적 허용: GPU 수를 바꾼 동일 시스템 내 scaling 경향 확인
- 불가: 논문 H100 latency/throughput 수치의 직접 재현 주장
- 불가: 논문 대비 성능 차이를 FlashMoE 알고리즘 자체의 차이로만 해석

성능 결과를 기록할 때는 반드시 `GPU 모델과 수`, `nvidia-smi topo -m`, CUDA,
PyTorch, FlashMoE commit, MathDx/cuBLASDx, NVSHMEM 버전 및 cuBLASDx modifier
(`generic`/`arch_specific`)를 함께 남긴다. 보고서에서는 이 실험을
**논문 성능 재현**이 아니라 **RTX PRO 6000에서의 FlashMoE 포팅 및 동작
검증**으로 표기한다.

### SM120 shared-memory 제한과 pipeline stage 변경

SM120 generic 패치만 적용한 첫 1 GPU 스모크 작업(job `1962274`)은 빌드에는
성공했지만 fused kernel이 block당 `131072 bytes`(128 KiB)의 dynamic shared
memory를 요구해 실패했다. RTX PRO 6000에서 조회된 opt-in 한도는
`101376 bytes`(99 KiB)였다.

upstream 테스트의 gated BF16 tile은 `bM=128`, `bN=128`, `bK=64`이며 모든
`Arch >= 900`에 pipeline stage `pS=3`을 사용한다. 이때 첫 expert GEMM의
shared-memory 요구량은 다음과 같다.

```text
A/B pipeline = 2 bytes * 64 * 3 * (128 + 128) = 96 KiB
gated buffer  = 2 bytes * 128 * 128             = 32 KiB
total         = max(96 KiB, 32 KiB) + 32 KiB   = 128 KiB
```

따라서 현재 SM120 포팅은 `csrc/tests/flashmoe.cu`에서 SM120에 한해 `pS=2`를
사용한다. 그러면 A/B pipeline이 64 KiB로 줄어 총 요구량이 96 KiB가 되어
99 KiB 한도 안에 들어간다. `pS`는 K 방향 tile의 load와 GEMM을 겹치는
pipeline 깊이이므로 이 변경은 성능 중립적인 호환 패치가 아니다. load latency
hiding이 줄어 느려질 수도 있고, shared-memory pressure 감소로 유리할 수도
있다. 결과에는 반드시 다음 조건을 함께 기록한다.

```text
SM120 + cuBLASDx generic + pS=2
```

논문의 H100 `arch_specific + pS=3` 결과와 직접 비교할 수 없으며, 측정치는
RTX PRO 6000에서 실행 가능하도록 수정한 포팅 결과로만 해석한다.

참고 자료:

- [FlashMoE 프로젝트 페이지](https://flash-moe.github.io/)
- [FlashMoE NeurIPS 2025 논문](https://papers.nips.cc/paper_files/paper/2025/hash/918d938bd209e5b56072777366f8a211-Abstract-Conference.html)
- [RTX PRO 6000 Blackwell Server Edition 사양(NVLink 미지원)](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu)

## 구버전 논문 프로토콜 초안 (사용하지 않음)

> 아래 초안은 구현 전 기록으로 남아 있으며 현재 실행 사양이 아니다. FP32
> `pS=1`, router 제외 측정 범위, GPU 수별 독립 batch를 반영한 최신 사양과
> 명령은 [paper_reproduction/README.md](paper_reproduction/README.md)만 따른다.

최신 본 실험 행렬과 실행 절차는
[paper_reproduction/README.md](paper_reproduction/README.md)를 기준으로 한다.
이번 범위는 FlashMoE 단독 Figure 8·10·11·12이며 GPU utilization, kernel count와
baseline 상대 비교는 포함하지 않는다.

### 실험의 성격과 목표

이 계획의 목표는 논문의 H100 절대 성능 수치를 맞추는 것이 아니라, 논문에
기재된 shape, GPU 수, 반복 방법과 지표 정의를 가능한 한 동일하게 적용해
FlashMoE의 latency 추세와 분산 확장성을 RTX PRO 6000
단일 노드에서 검증하는 것이다. 결과는 다음처럼 표기한다.

> FlashMoE paper-protocol replication and SM120/PCIe portability evaluation

검증 질문은 다음과 같다.

1. GPU당 토큰 수가 증가할 때 논문 Figure 8과 유사한 latency 추세가 나타나는가?
2. GPU를 2개에서 4개와 8개로 늘릴 때 throughput과 overlap efficiency가 어떻게
   변하는가?
3. 전체 expert 수가 8개에서 128개로 증가해도 latency가 안정적인가?
4. NVLink가 없는 PCIe/NUMA 환경과 SM120 포팅 제약이 위 결과에 어떤 영향을
   주는가?

### 논문 공통 조건과 본 실험 조건

| 항목 | 논문 | 본 실험의 목표 조건 |
|---|---|---|
| GPU | 8 x H100 80GB, SM90 | 최대 8 x RTX PRO 6000 96GB, SM120 |
| 노드 | 단일 노드 | 단일 노드 강제 |
| GPU 연결 | NVLink | PCIe 5.0, NVLink 없음 |
| software | PyTorch 2.6, CUDA 12.8, Ubuntu 22.04 | CUDA 12.8과 현재 C++ 환경, 전체 버전 기록 |
| attention heads | 16 | 16, 단 MoE 단독 kernel 입력에는 직접 사용되지 않음 |
| embedding dimension | 2048 | 2048 |
| FFN intermediate dimension | 2048 | 2048 |
| routing | top-2, capacity factor 1.0 | 동일 |
| 병렬화 | DDP + Expert Parallelism | GPU당 MPI/NVSHMEM rank 1개, Expert Parallelism |
| precision | FlashMoE FP32 | FP32 목표; 별도 FP32 실행 경로 필요 |
| 측정 범위 | MoE layer forward 1회 | Gate부터 Combine까지 전체 forward를 목표로 함 |
| 반복 | warmup 32회 후 32회 평균 | 동일, 32개 개별 표본도 저장 |
| CUDA Graph | 논문의 single persistent launch 평가 | 끔 (`graph_launches=0`) |
| cuBLAS 구현 | 논문 H100 최적화 경로 | cuBLASDx SM120 `generic`, `pS=2` |

이번에는 FlashMoE FP32만 실행한다. 논문의 baseline 대비 speedup은 측정하거나
주장하지 않으며 필요할 때 별도 후속 실험으로 추가한다.

### 본 실험 전 필수 통과 조건

현재 `testFlashMoE` smoke 경로는 BF16, gated MLP + SiLU, router 제외 latency를
측정한다. 따라서 현재 스모크 성공을 논문 조건의 성공으로 간주하지 않는다.
본 측정 전에 다음 조건을 순서대로 해결한다.

1. **측정 범위 확인**: 현재 테스트처럼 router를 timed region 밖에서 실행하면
   결과 이름을 `fused_expert_dispatch_combine_latency`로 기록한다. 논문의 전체
   MoE layer latency와 맞추려면 Gate까지 포함하는 동일 측정 범위가 필요하다.
2. **FP32 경로 분리**: 기존 BF16 포팅 binary를 덮어쓰지 않고 논문용 FP32
   binary/build directory를 별도로 만든다.
3. **FFN 의미 확인**: 논문 artifact가 사용한 vanilla/gated FFN과 activation을
   확인한다. 현재 gated SiLU를 근거 없이 논문 조건으로 간주하지 않는다.
4. **SM120 빌드 검증**: FP32에서도 `generic + pS=2`가 shared-memory 한도 안에서
   컴파일 및 실행되는지 작은 shape로 검사한다.
5. **통신 경로 검증**: 2/4/8 GPU 각각에서 `nvidia-smi topo -m`, CUDA P2P 가능
   여부, NVSHMEM transport와 rank-to-GPU mapping을 저장한다. 로그에 나타난
   `IBRC Device enumeration failed`는 무시하지 않고 실제 사용 transport와 함께
   기록한다.
6. **메모리 사전 검사**: `E=128, T=16K`를 실행하기 전에 각 rank의 free/peak
   GPU memory와 workspace 예상량을 확인한다.
7. **정확성 검사**: 각 GPU 수에서 timed run 전에 reference 대비 오차가 허용
   범위 안인지 확인한다. 오류가 있는 조건의 latency는 성능 결과로 사용하지 않는다.

### 입력과 측정 규칙

- `T` 또는 `S`는 **GPU당 토큰 수**다. global token 수는 `T * world_size`다.
- `E`는 전체 GPU에 분산되는 **global expert 수**다. 구현에서는 원칙적으로
  `local_experts = E / world_size`로 설정하고 실제 partition 결과를 기록한다.
- `E`는 `world_size`로 나누어져야 한다.
- capacity factor 1.0에 대응하는 expert capacity와 정렬 후 실제 capacity(`EC`)를
  함께 저장한다.
- 입력 tensor, weight와 router 입력은 고정 seed로 조건마다 한 번 생성하고
  timed region 안에서 다시 생성하지 않는다.
- routing 결과가 매 iteration 동일한지와 expert별 token assignment를 저장한다.
  capacity 초과 및 dropped token이 있으면 함께 기록한다.
- 각 측정 전 모든 rank를 동기화한다. 분산 forward latency는 rank별 측정값 중
  최댓값을 해당 iteration의 latency로 사용한다.
- 본 결과는 warmup 32회를 버린 뒤 32회의 `mean`, `std`, `median`, `min`, `max`를
  저장한다. 논문 대응 대표값은 32회 평균이다.
- 첫 실행의 JIT/컴파일/초기화와 correctness/reference 시간은 측정에서 제외한다.

### 단계 A: 1-GPU 기능 및 순수 연산 기준선

| GPU | E | GPU당 T | 목적 |
|---:|---:|---:|---|
| 1 | 8, 16, 32 | 4096 | FP32 kernel 컴파일·실행·정확성, 통신 없는 기준선 |

이 단계는 논문 Figure의 직접 재현이 아니라 본 실험의 사전 검증이다. 세 조건이
모두 정확성 검사를 통과해야 다중 GPU 본 측정으로 진행한다.

### 단계 B: Forward Latency와 token scaling

| GPU | E | GPU당 T | 논문 대응 |
|---:|---:|---:|---|
| 2 | 32 | 4096, 8192, 16384 | 확장 실험 및 2-GPU 기준점 |
| 4 | 32 | 4096, 8192, 16384 | Figure 8(a) 프로토콜 |
| 8 | 32 | 4096, 8192, 16384 | Figure 8(b) 프로토콜 |

모든 작업은 `--nodes=1`로 제출한다. 2-GPU 결과는 Figure 8에는 없지만 이후
weak-scaling의 기준과 PCIe 환경 분석을 위해 같은 sweep으로 측정한다.

### 단계 C: Throughput과 Overlap Efficiency

단계 B의 `E=32, T=8192` 결과를 재사용하며 중복 실행하지 않는다.

```text
throughput(N) = (8192 * N) / latency(N)
overlap_efficiency(N) = latency(2) / latency(N) * 100
```

throughput은 MTokens/s로 변환한다. 논문 정의에 따라 2-GPU overlap efficiency는
100%이고, 4/8-GPU 값은 같은 node type과 같은 측정 방법으로 얻은 2-GPU
latency를 기준으로 계산한다.

### 단계 D: Expert Scalability

| GPU | global E | GPU당 T | 논문 대응 |
|---:|---:|---:|---|
| 4 | 8, 16, 32, 64, 128 | 16384 | Figure 12(a) 프로토콜 |
| 8 | 8, 16, 32, 64, 128 | 16384 | Figure 12(b) 프로토콜 |

논문 본문은 x축을 global expert 수라고 설명하지만 GPU당 expert 수 설명에는
4-GPU plot과 맞지 않는 모호한 표현이 있다. 실행 전 공개 artifact의 partition
방식을 확인하고, 본 실험에서는 실제 `global E`, `world_size`, `local E`를 모두
결과에 기록한다.

### 이번 범위에서 제외하는 항목

- Figure 9 GPU utilization: Nsight를 사용할 수 없어 제외
- Table 1 GPU operation/kernel count: Nsight를 사용할 수 없어 제외
- Comet/FasterMoE/Megatron baseline 비교: 필요할 때 후속 실험으로 추가

### 작업 단위와 우선순위

각 GPU 수는 별도 sbatch 작업으로 제출하되 모두 단일 노드를 요청한다.

1. FP32·2048/2048·top-2 전체 MoE forward 정합성 검증
2. 단계 B의 2/4/8-GPU token sweep
3. 단계 D의 4/8-GPU expert sweep

8-GPU 작업은 한 노드의 GPU 8개가 동시에 비어야 하므로 대기 시간이 길 수 있다.
성능 비교에는 `asus_pro6000`과 `gigabyte_pro6000` 결과를 섞지 않고, 가능하면
동일 partition과 동일 node에서 모든 GPU-count 조건을 측정한다.

### 저장할 결과와 그래프

최소한 다음 필드를 CSV/metadata에 저장한다.

```text
job_id, node, partition, gpu_model, gpu_count, topology,
flashmoe_commit, source_patch_hash, cuda_version, mathdx_version,
nvshmem_version, cublasdx_modifier, pipeline_stages,
dtype, mlp_type, activation, H, I, global_E, local_E, top_k,
capacity_factor, EC, tokens_per_gpu, global_tokens,
warmup, runs, iteration, rank, rank_latency_ms, max_rank_latency_ms,
error_pct, dropped_tokens, workspace_mib, peak_memory_mib
```

생성할 그래프는 다음과 같다.

1. GPU당 token 수 대 mean forward latency: 4/8 GPU, Figure 8 대응
2. GPU 수 대 throughput: 2/4/8 GPU, Figure 10 대응
3. GPU 수 대 latency 및 overlap efficiency: Figure 11 대응
4. global expert 수 대 mean forward latency: 4/8 GPU, Figure 12 대응

그래프에는 평균만 표시하지 말고 32개 표본의 표준편차 또는 분포를 함께 남긴다.

### 논문과의 차이 및 결과 해석 한계

| 차이 | 결과에 미칠 수 있는 영향 |
|---|---|
| H100 SM90 대신 RTX PRO 6000 SM120 | 연산 처리량, tile 선택, occupancy가 달라짐 |
| NVLink 대신 PCIe/NUMA | 통신 latency 증가, overlap efficiency 변화 |
| H100 최적화 대신 cuBLASDx generic | GEMM 절대 성능 및 kernel scheduling 차이 |
| `pS=3` 대신 SM120 `pS=2` | load/GEMM pipeline과 shared-memory pressure 변화 |
| 논문 코드와 현재 upstream commit 차이 | 구현·튜닝·dependency 차이 발생 가능 |

따라서 허용되는 결론은 “동일한 프로토콜을 RTX PRO 6000에 적용했을 때의
동작과 scaling 특성”이다. H100 수치와 차이가 나더라도 이를 FlashMoE 설계의
회귀로 단정하지 않으며, 반대로 RTX PRO 6000에서 빠르더라도 논문의 H100 성능을
재현했다고 표현하지 않는다.

## 파일

- `run_correctness.py`: 이전 Python JIT 정확성 실험 도우미(현재 sbatch에서는 사용하지 않음)
- `setup_runtime_env.sh`: CUDA 12.8, MathDx, NVSHMEM 및 캐시 경로 설정
- `bin/cmake`: upstream JIT의 SM120 CMake 인자를 실험 범위에서 보정
- `slurm/run_benchmark.sbatch`: 로컬 SM120 generic 소스를 빌드하고 C++ 성능 측정
- `results/`: Slurm 로그(커밋하지 않음)

## SM120 generic 빌드

성능 스크립트는 Python JIT를 사용하지 않고 `/home/hybyun0207/FlashMoE/csrc`를
직접 Release 빌드한다. 제출 전에 `tile.cuh`의 세 cuBLASDx 선택식에
`Arch != 1200` generic 조건이 적용되어 있어야 하며, sbatch가 이를 검사한다.

다음 내용은 이전 Python JIT 정확성 실험에만 해당한다. Python JIT는 RTX PRO
6000을 감지한 뒤 `CMAKE_CUDA_ARCHITECTURES=120`을 전달하지만, 같은 버전의
CMake 코드는 SM90 이상에서 `120a` 형식을 요구한다. `bin/cmake`는 이 인자를
다음처럼 바꾸는 호환 wrapper다.

```text
-DCMAKE_CUDA_ARCHITECTURES=120
    -> -DCMAKE_CUDA_ARCHITECTURES=120a
```

이 wrapper는 cuBLASDx modifier 문제를 해결하지 않으므로 현재 C++ 성능
실험에서는 사용하지 않는다.

## 실행

아래 명령은 현재 구현되어 있는 **BF16 SM120 포팅 benchmark**용이다. 위의 논문
프로토콜 계획에 필요한 FP32 전체-MoE 측정 스크립트는 아직 구현 전이며, 현재
`run_benchmark.sbatch` 결과를 논문 조건 결과로 사용하면 안 된다. 또한 현재
`submit_scaling_smoke.sh`의 4-GPU 항목은 과거 자원 가정에 따라 2개 노드를
요청한다. 논문 프로토콜용으로 사용할 때는 1/2/4/8 GPU 모두 `--nodes=1`이 되도록
별도 수정해야 한다.

### 1·2·4 GPU 스모크 검증

FlashMoE는 GPU당 MPI/NVSHMEM PE를 하나씩 실행하므로 GPU 수, Slurm task 수,
`mpirun -np` 값이 같아야 한다. 짧은 정확성/실행 검증 세 개를 제출하려면:

```bash
cd /home/hybyun0207/aso-internship/flashmoe
bash slurm/submit_scaling_smoke.sh
```

제출되는 자원 구성은 다음과 같다.

| 검증 | 노드 | 노드당 GPU | 전체 Slurm task/MPI rank |
|---|---:|---:|---:|
| 1 GPU | 1 | 1 | 1 |
| 2 GPU | 1 | 2 | 2 |
| 4 GPU | 2 | 2 | 4 |

세 작업은 공용 build directory의 동시 변경을 피하기 위해 1→2→4 GPU 순서로
실행된다. 앞 작업의 성공 여부와 관계없이 다음 검증이 시작되도록 `afterany`
dependency를 사용한다. 4 GPU 검증은 `gigabyte_pro6000` 파티션에서 GPU 2개인
노드 두 대를 동시에 할당받아야 하므로 대기 시간이 길 수 있다. 또한 이 검증은
다중 노드 NVSHMEM/MPI 통신까지 포함하므로, 1·2 GPU 성공과 별개로 네트워크나
NVSHMEM transport 설정에 의해 실패할 수 있다.

각 로그에서 다음 문구가 있으면 해당 크기의 스모크 테스트가 통과한 것이다.

```text
SMOKE_TEST=PASS
```

스모크 로그 이름에는 세 작업을 제출한 시각과 GPU 수가 포함된다.

```text
results/slurm-YYYYMMDD-HHMMSS-1gpu-<job-id>.out
results/slurm-YYYYMMDD-HHMMSS-2gpu-<job-id>.out
results/slurm-YYYYMMDD-HHMMSS-4gpu-<job-id>.out
```

실행 정보는 `results/job-<job-id>/`에 저장된다. 스모크 검증은
`benchmark.csv`를 만들지 않는다.

### 전체 벤치마크

2 GPU 기본 실행(CUDA Graph off, warmup 20회, 측정 50회, 독립 반복 5회):

```bash
cd /home/hybyun0207/aso-internship/flashmoe
sbatch slurm/run_benchmark.sbatch
```

기본값은 단일 노드에서 2 GPU가 확인된 `gigabyte_pro6000` 파티션을 사용한다.
Asus 파티션에서 1 GPU 동작만 확인하려면 다음처럼 자원과 GPU 수를 함께
덮어쓴다.

```bash
sbatch \
  --partition=asus_pro6000 \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gres=gpu:RTXPRO6000:1 \
  --export=ALL,NUM_GPUS=1 \
  slurm/run_benchmark.sbatch
```

GPU 수는 전체 expert 수 16의 약수여야 한다. 현재 기본 실험은 2 GPU다.

로그 확인:

```bash
squeue -u "$USER"
tail -f results/slurm-<job-id>.out
```

스크립트는 먼저 작은 shape로 정확성 스모크 테스트를 실행한 후 GPU당 토큰 수
`1024 2048 4096 8192`를 측정한다. 각 shape는 독립적으로 5번 실행하며, 기본
측정은 CUDA Graph를 끄고 warmup 20회와 본 측정 50회를 사용한다. 결과는
`results/job-<job-id>/benchmark.csv`와 `results/job-<job-id>/raw/`에 저장된다.
첫 실행은 C++ Release 빌드와 CPM 의존성 준비 때문에 이후 실행보다 오래 걸릴
수 있다.

CUDA Graph 결과는 별도 job으로 측정한다.

```bash
sbatch \
  --export=ALL,GRAPH_LAUNCHES=8 \
  slurm/run_benchmark.sbatch
```

`GRAPH_LAUNCHES>0`이면 upstream C++ 벤치마크는 `RUNS`개의 forward를 graph에
capture한 뒤 graph를 반복 실행한다. 이 경로에서는 `WARMUP` 인자가 일반
eager warmup 횟수로 사용되지 않으므로 graph-off 결과와 구분해서 해석한다.

## Shape 변경

제출할 때 환경변수로 shape와 반복 조건을 덮어쓴다.

```bash
sbatch \
  --export=ALL,TOKENS_PER_RANKS="1024 2048",TOKEN_DIM=5120,FFN_DIM=8192,NUM_EXPERTS=16,TOP_K=1,WARMUP=20,RUNS=50,OUTER_REPEATS=5 \
  slurm/run_benchmark.sbatch
```

`NUM_EXPERTS`는 실행하는 GPU 수로 나누어져야 한다. `S`는 GPU당 토큰 수이므로
전체 토큰 수는 `S * NUM_GPUS`다.
