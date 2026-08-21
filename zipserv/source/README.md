# ZipServ를 읽기 위한 CUDA 배경 지식

이 문서는 ZipServ의 BF16 Triple-Bitmap 압축과 ZipGEMM CUDA kernel을 직접 읽기
위해 필요한 배경 지식을 코드와 연결해서 정리한다. 목표는 CUDA 문법을 암기하는
것이 아니라 다음 실행 흐름을 설명할 수 있게 되는 것이다.

```text
BF16 weight 분석 및 압축 (CPU)
  → 압축 배열을 GPU global memory로 복사
  → cp.async로 compressed weight/bitmap/activation을 shared memory에 적재
  → warp가 bitmap을 해석해 BF16 weight를 register에 복원
  → ldmatrix로 activation fragment를 register에 적재
  → mma.sync로 Tensor Core GEMM 수행
  → FP32 accumulator를 BF16 output으로 저장
  → Split-K > 1이면 별도 reduction
```

ZipServ는 외부 CUTLASS GEMM을 호출하지 않고 CUDA C++로 빌드되지만, 핵심부에는
`cp.async`, `ldmatrix`, `mma.sync`를 직접 호출하는 inline PTX가 있다. 따라서
“일반 CUDA만 사용한다”는 말은 외부 kernel framework 의존성이 없다는 뜻으로는
맞지만, 코드 난이도가 기초 CUDA라는 뜻은 아니다.

## 1. 먼저 알아야 할 C++와 빌드 개념

### Host 코드와 device 코드

CUDA 소스 하나에는 CPU와 GPU 코드가 함께 존재한다.

| 선언 | 실행 위치 | ZipServ에서의 용도 |
|---|---|---|
| 일반 C++ 함수 | CPU | 입력 생성, weight 압축, 결과 검증 |
| `__host__` | CPU | host 압축 API |
| `__device__` | GPU | bitmap 복원, fragment load helper |
| `__global__` | GPU, CPU에서 launch | ZipGEMM과 reduction kernel |

`nvcc`는 host 부분을 host compiler에 넘기고 device 부분을 GPU machine code로
컴파일한다. 현재 RTX 4090 빌드는 다음 의미를 가진다.

```text
-gencode arch=compute_89,code=sm_89
```

- `compute_89`: CUDA virtual architecture
- `sm_89`: Ada RTX 4090용 실제 GPU instruction binary

### Template과 compile-time configuration

ZipServ는 tile 크기와 한 warp가 처리할 N 방향 tensor 수를 C++ template 상수로
정한다. 실행 중에 tile 크기를 바꾸는 것이 아니라, 여러 kernel specialization을
미리 컴파일하고 batch `N`에 맞는 함수를 선택한다.

관련 코드:

- [`TilingConfig.h`](../../../ZipServ_ASPLOS26/csrc/TilingConfig.h)
- [`L_API.cu`](../../../ZipServ_ASPLOS26/csrc/L_API.cu)

다음을 읽을 수 있어야 한다.

```cpp
template<int A, int B, int C>
struct Config {
    static constexpr int value = A * B * C;
};
```

`#pragma unroll`, `__forceinline__`, `constexpr`, template specialization도 자주 나온다.
이들은 기능뿐 아니라 compiler가 loop와 함수 호출을 제거해 register-level kernel을
만들도록 유도하는 성능 장치다.

## 2. CUDA 실행 모델: grid, block, warp, lane

CUDA kernel은 다음 계층으로 실행된다.

```text
Grid
└── Thread Block
    └── Warp (32 threads)
        └── Lane 0 ... 31
```

핵심 built-in 변수:

```cpp
blockIdx.x       // grid 안의 block 위치
threadIdx.x      // block 안의 thread 위치
blockDim.x       // block의 thread 수
threadIdx.x / 32 // warp ID
threadIdx.x % 32 // lane ID
```

ZipServ의 주요 계산 단위는 개별 thread가 아니라 warp다. Tensor Core의
`mma.sync`도 warp의 32개 thread가 협력해서 실행한다. 따라서 한 thread가 행렬
전체를 갖는 것이 아니라, matrix fragment의 일부가 각 lane의 register에 분산된다.

### SIMT와 divergence

같은 warp의 thread는 같은 instruction을 실행한다. lane마다 조건이 달라 분기하면
경로가 직렬화될 수 있다. ZipServ는 다음 방법으로 분기를 줄인다.

- compile-time template과 `#pragma unroll`
- predicate를 PTX instruction에 전달
- 경계가 없는 `Fast` kernel과 경계 검사하는 `Safe` kernel 분리

[`L_API.cu`](../../../ZipServ_ASPLOS26/csrc/L_API.cu)는 `N`이 tile 크기로 정확히
나누어지면 Fast kernel, 그렇지 않으면 Safe kernel을 launch한다.

## 3. CUDA memory hierarchy

GPU memory의 일반적인 관계는 다음과 같다.

| 메모리 | 범위 | 특징 | ZipServ에서의 용도 |
|---|---|---|---|
| Register | thread | 가장 빠르고 매우 제한적 | A/B fragment, FP32 accumulator |
| Shared memory | block | block thread가 공유 | 압축 tile, bitmap, B tile, output staging |
| L1/L2 cache | GPU hardware | 자동 관리 | global load cache |
| Global memory | GPU 전체 | 크지만 상대적으로 느림 | 압축 weight, activation, output |
| Host memory | CPU | GPU가 직접 계산하지 않음 | 합성 weight 생성과 사전 압축 |

핵심 kernel은 다음 포인터로 동적 shared memory를 요청한다.

```cpp
extern __shared__ __align__(128) __nv_bfloat16 smem[];
```

실제 byte 수는 kernel launch의 세 번째 인자로 전달한다.

```cpp
kernel<<<grid, block, shared_memory_bytes, stream>>>(...);
```

### Coalescing

한 warp의 global memory 접근이 연속되고 정렬돼 있으면 memory transaction 수가
줄어든다. ZipServ는 주로 16-byte 단위 `cp.async`를 사용하며, 압축 배열에 padding을
넣어 다음 조건을 맞춘다.

- sign+mantissa 배열: 16개 원소 단위
- fallback BF16 배열: 8개 원소, 즉 16 bytes 단위

### Shared-memory bank conflict와 swizzle

shared memory는 여러 bank로 나뉜다. 같은 warp의 lane들이 같은 bank의 서로 다른
주소를 동시에 읽으면 접근이 직렬화된다. ZipServ의 activation tile 복사에서 보이는
XOR 주소 변환은 bank conflict를 줄이기 위한 swizzle이다.

관련 코드:

- [`MatMulUtilities.cuh`](../../../ZipServ_ASPLOS26/csrc/MatMulUtilities.cuh)

## 4. 동기화와 CUDA stream

### `__syncthreads()`

block 전체 thread가 도달할 때까지 기다리고 shared-memory 접근 순서를 보장한다.
warp 일부만 도달하는 분기 안에서 잘못 사용하면 deadlock이 생길 수 있다.

### CUDA stream

같은 stream에 제출된 작업은 순서대로 실행된다. ZipServ API는 `cudaStream_t`를
받지만 benchmark는 stream `0`을 사용한다.

Kernel launch는 CPU 관점에서 비동기다. 오류와 결과를 확인하려면 필요에 따라
`cudaDeviceSynchronize`, event synchronization 또는 stream synchronization이
필요하다.

### `cp.async` 동기화

`cp.async`는 Ampere 이후 GPU에서 global→shared 복사를 비동기로 제출한다.

```text
cp.async ...
cp.async.commit_group
cp.async.wait_group N
__syncthreads
```

- `commit_group`: 지금까지 발행한 async copy를 하나의 group으로 확정
- `wait_group<N>`: 미완료 group 수가 N 이하가 될 때까지 대기
- `__syncthreads`: block의 다른 thread도 데이터를 안전하게 보도록 동기화

관련 코드:

- [`AsyncCopy_PTX.cuh`](../../../ZipServ_ASPLOS26/csrc/AsyncCopy_PTX.cuh)

## 5. GEMM과 행렬 layout

ZipServ가 계산하는 연산은 다음과 같다.

```text
C[M,N] = A[M,K] × B[K,N]
```

- `A`: LLM weight, BF16, ZipServ에서 압축되는 대상
- `B`: activation, BF16
- `C`: output, BF16
- 내부 누산: FP32

연산량은 multiply와 add를 각각 하나로 세어 다음과 같다.

```text
FLOPs = 2 × M × N × K
TFLOPS = FLOPs / latency_seconds / 10^12
```

### Row-major와 column-major

C/C++의 2차원 배열은 보통 row-major로 생각하지만 cuBLAS는 전통적으로
column-major interface를 사용한다. benchmark는 row-major weight buffer `A`를
cuBLAS에 넘기면서 `CUBLAS_OP_T`를 사용해 논리적 `A[M,K]`를 맞춘다. activation
`B`와 output `C`는 column-major indexing을 사용한다.

관련 코드:

- [`test_mm.cu`](../../../ZipServ_ASPLOS26/kernel_benchmark/test_mm.cu)

행렬 layout을 놓치면 계산값은 맞는 것처럼 보여도 실제로는 transpose된 다른
연산을 비교할 수 있다. 포인터 식을 볼 때 항상 다음 네 가지를 적어야 한다.

```text
logical shape / physical layout / leading dimension / transpose flag
```

## 6. GEMM tiling

큰 행렬 전체를 한 block이 처리하지 않는다. M, N, K를 작은 tile로 분할한다.

ZipServ의 기본 상수:

```text
Tensor Core K step = 16
Kernel K tile      = 16 × 4 = 64
Warp size          = 32
일반 M block tile  = 64
```

하나의 block은 `(M tile, N tile)`에 해당하는 output tile을 맡고 K 방향 tile을
순회하며 accumulator에 더한다.

```text
for each K tile:
    load compressed A tile
    load B tile
    decompress A fragment
    accumulator += A fragment × B fragment
```

tile을 이해할 때는 다음 세 수준을 구분해야 한다.

| 수준 | 질문 |
|---|---|
| Block tile | 한 thread block이 C의 어느 영역을 계산하는가? |
| Warp tile | block 안의 각 warp가 어느 영역을 계산하는가? |
| MMA tile | 한 `mma.sync`가 어느 fragment를 계산하는가? |

[`TilingConfig.h`](../../../ZipServ_ASPLOS26/csrc/TilingConfig.h)의 이름이 비슷한
상수들을 종이에 직접 전개해 `TILE_M`, `TILE_N`, `BLOCK_THREADS` 값을 계산해보는
것이 좋다.

## 7. BF16 표현과 정확도

BF16은 16 bit 부동소수점이다.

```text
bit 15      : sign 1 bit
bits 14..7  : exponent 8 bits
bits 6..0   : mantissa 7 bits
```

FP32와 exponent 폭은 같지만 mantissa가 짧다. ZipServ는 다음 bit 연산으로 BF16을
분해하고 다시 조립한다.

```text
sign     = bits >> 15
exponent = (bits >> 7) & 0xff
mantissa = bits & 0x7f
```

Tensor Core 명령은 BF16 A/B를 곱해 FP32 accumulator에 누적한다. 마지막 output은
BF16으로 반올림한다. Split-K 값이 달라지면 덧셈 순서가 달라지므로 최종 GEMM
결과가 cuBLAS와 bit-exact하지 않을 수 있다. 이것은 weight 압축/복원이 lossless인지와
별개의 문제다.

관련 타입과 intrinsic:

```cpp
__nv_bfloat16
__nv_bfloat162
__bfloat162float
__float2bfloat16_rn
__bfloat16_as_ushort
__ushort_as_bfloat16
```

## 8. Tensor Core, `ldmatrix`, `mma.sync`

ZipServ는 `nvcuda::wmma` API 대신 inline PTX를 직접 사용한다.

관련 코드:

- [`MMA_PTX.cuh`](../../../ZipServ_ASPLOS26/csrc/MMA_PTX.cuh)

### `ldmatrix`

```text
ldmatrix.sync.aligned.x4.m8n8.shared.b16
```

shared-memory matrix tile을 Tensor Core가 요구하는 lane별 register layout으로
읽는다. 각 lane은 fragment 전체가 아니라 일부 register만 가진다.

### `mma.sync`

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

이 suffix의 의미:

- output tile: `m16n8k16`
- A layout: row-major
- B layout: column-major
- accumulator/output: FP32
- A/B input: BF16

warp의 32개 lane 모두가 유효하게 참여해야 한다. 각 lane의 register가 행렬의 어느
좌표에 대응하는지는 일반 CUDA index 공식만으로 알 수 없고 PTX fragment layout
규칙을 알아야 한다. 이것이 ZipServ의 가장 어려운 부분이다.

ZipServ는 논리적 N=16 fragment를 만들기 위해 N=8 MMA를 두 번 호출하는 경로도
사용한다.

## 9. Warp-level primitive

### `__popcll`

64-bit integer의 set bit 수를 센다. Triple-Bitmap 복원에서 특정 위치 앞에 있는
high-frequency 원소 수, 즉 압축 배열 안의 rank를 계산한다.

```text
rank(pos) = popcount(high_frequency_bitmap & ((1 << pos) - 1))
```

이 rank를 이용해 해당 값이 `sign_mantissa` 배열의 몇 번째 원소인지 찾는다.

### `__shfl_sync`

warp lane 사이에서 register 값을 교환한다. ZipServ는 작은 8×8 tile의 마지막
위치까지 계산한 누적 high-frequency count를 lane 31에서 warp 전체로 broadcast해
다음 tile의 시작 offset을 갱신한다.

이 방식은 shared memory에 count를 쓰고 동기화하는 것보다 가볍다.

관련 코드:

- [`L_Kernel.cuh`](../../../ZipServ_ASPLOS26/csrc/L_Kernel.cuh)

## 10. Triple-Bitmap 압축 형식

ZipServ의 압축 단위를 먼저 이해해야 fused GEMM을 이해할 수 있다.

### BF16 exponent 선택

CPU가 weight의 exponent 빈도를 분석해 연속된 exponent 7개를 선택한다. 가장 작은
값보다 1 작은 값을 `start_exp`로 둔다.

```text
code 001 → start_exp + 1
code 010 → start_exp + 2
...
code 111 → start_exp + 7
code 000 → high-frequency가 아님
```

### 8×8 small tile과 bitmap

8×8 tile에는 64개 원소가 있으므로 한 bitmap을 `uint64_t` 하나로 표현할 수 있다.
각 원소의 exponent code는 3 bit이므로 bitmap 세 장에 bit-plane 방식으로 저장한다.

```text
bitmap1: code bit 0
bitmap2: code bit 1
bitmap3: code bit 2
```

세 bitmap의 OR에서 bit가 1이면 선택된 exponent 범위의 원소다.

### 두 payload 배열

| 원소 종류 | 저장 내용 | 원소당 payload |
|---|---|---:|
| High-frequency exponent | sign 1 bit + mantissa 7 bit | 1 byte |
| 그 외 exponent | 원래 BF16 전체 | 2 bytes |

High-frequency 원소의 exponent는 bitmap code로 복원하므로 payload에는 sign과
mantissa만 저장한다. 이론적인 크기에는 원소당 3-bit bitmap overhead와 tile offset,
정렬 padding도 추가된다.

### 계층적 tile과 offset

```text
small tile  : 8 × 8
medium tile : 16 × 64
global tile : 64 × 64
```

- `TileOffsets_Median`: warp가 담당할 compressed payload 시작 위치 계산
- `TileOffsets_Global`: global tile별 두 payload 배열의 누적 시작 위치
- padding: vectorized 16-byte async copy가 가능하도록 정렬

CPU 압축 구현:

- [`utils.h`](../../../ZipServ_ASPLOS26/kernel_benchmark/utils.h)
- [`L_API.cu`](../../../ZipServ_ASPLOS26/csrc/L_API.cu)의 별도 host 압축 구현

실제 benchmark는 `utils.h`의 `InitBF16MatrixTripleBitmap_Host` 경로를 사용한다.

## 11. Fused decompression과 GEMM pipeline

핵심 kernel은 [`L_Kernel.cuh`](../../../ZipServ_ASPLOS26/csrc/L_Kernel.cuh)에 있다.
중요한 점은 압축 weight 전체를 별도 global-memory buffer로 먼저 복원하지 않는다는
것이다.

```text
global compressed arrays
   │ cp.async
   ▼
shared bitmap/payload
   │ bitmap decode + rank
   ▼
register BF16 fragment
   │ mma.sync
   ▼
register FP32 accumulator
```

따라서 다음 두 효과가 동시에 존재한다.

- 장점: global-memory weight traffic 감소
- 비용: bitmap decode, popcount, bit 조립, offset 계산 증가

ZipGEMM의 성능은 “압축률이 높으면 무조건 빠르다”가 아니라, 줄어든 memory traffic이
복원 비용보다 큰지에 따라 결정된다.

### Double buffering

shared memory와 register에 buffer 두 세트를 만들어 현재 tile을 계산하는 동안 다음
tile을 불러온다.

```text
iteration t:     buffer 0 계산 + buffer 1 load
iteration t + 1: buffer 1 계산 + buffer 0 load
```

`cp.async`, `commit_group`, `wait_group`, buffer index의 홀짝 전환을 함께 추적해야 한다.
kernel을 읽을 때 각 포인터에 `read buffer`와 `write buffer`를 표시하면 이해가 쉽다.

## 12. Split-K

일반 GEMM은 하나의 output tile이 K 전체를 순회한다. Split-K는 K를 여러 구간으로
나눠 서로 다른 block이 부분합을 계산하게 한다.

```text
C_partial[split, M, N] = A[:, K_split] × B[K_split, :]
C[M,N] = sum(C_partial over split)
```

장점:

- 작은 batch에서도 block 수를 늘려 GPU 병렬성 확보
- 긴 K dimension을 여러 SM에 분산

비용:

- `M × N × SplitK` workspace
- 별도 reduction kernel
- global-memory traffic 증가
- 덧셈 순서 변화

관련 코드:

- Split-K 분배 및 launch: [`L_API.cu`](../../../ZipServ_ASPLOS26/csrc/L_API.cu)
- reduction: [`Reduction_Kernel.cuh`](../../../ZipServ_ASPLOS26/csrc/Reduction_Kernel.cuh)

ZipServ API에서 `Split_K == 1`이면 kernel이 C에 직접 쓰고, 1보다 크면 workspace에
쓴 뒤 reduction kernel이 최종 C를 만든다.

## 13. Register, shared memory, occupancy

한 SM에서 동시에 실행할 수 있는 block/warp 수는 다음 자원에 제한된다.

- block당 thread 수
- thread당 register 수
- block당 shared-memory 크기
- architecture의 최대 block/warp 수

ZipServ compile log에서는 specialization에 따라 thread당 register 사용량이 크게
달라진다. register가 많으면 spill 없이 fragment와 accumulator를 유지할 수 있지만,
동시에 resident할 warp 수가 감소할 수 있다.

중요한 관계:

```text
register 증가 → local-memory spill 감소 가능
register 증가 → occupancy 감소 가능
shared memory 증가 → block당 staging 용량 증가
shared memory 증가 → 동시에 resident 가능한 block 감소 가능
```

Occupancy가 높다고 항상 빠른 것도 아니고, 낮다고 항상 느린 것도 아니다. Tensor
Core pipeline을 충분히 채우고 memory latency를 숨길 수 있는지가 중요하다.

## 14. CUDA 오류 처리와 비동기 오류

Kernel launch 자체는 비동기라서 잘못된 memory access가 launch 줄이 아니라 다음
동기화 API에서 보고될 수 있다.

학습할 오류 확인 순서:

```cpp
kernel<<<...>>>(...);
cudaGetLastError();       // launch configuration 오류
cudaDeviceSynchronize(); // 실행 중 발생한 비동기 오류
```

개발 중에는 다음 환경 변수가 오류 위치 파악에 도움을 줄 수 있다.

```bash
export CUDA_LAUNCH_BLOCKING=1
```

성능 측정 시에는 비동기 실행을 막으므로 이 변수를 제거해야 한다.

## 15. CUDA Event와 올바른 성능 측정

CPU의 wall-clock으로 비동기 kernel 호출만 재면 실제 GPU 시간을 얻을 수 없다.
ZipServ benchmark는 CUDA Event를 사용한다.

```text
event record(start)
kernel launch
event record(stop)
event synchronize(stop)
elapsed time(start, stop)
```

현재 benchmark의 특징:

- 사전 warm-up
- 각 timed iteration 전에 L2 cache flush
- flush는 timed region 밖
- weight 생성과 압축도 timed region 밖
- ZipGEMM API 안의 Split-K reduction은 timed region 안

관련 코드:

- [`test_mm.cu`](../../../ZipServ_ASPLOS26/kernel_benchmark/test_mm.cu)

cuBLAS와 비교할 때 dtype, accumulator, transpose, stream, cache 조건, warm-up, 반복
횟수와 timed region이 동일한지 확인해야 한다.

## 16. ZipServ 파일 지도

| 읽는 순서 | 파일 | 역할 | 난이도 |
|---:|---|---|---|
| 1 | [`Reduction_Kernel.cuh`](../../../ZipServ_ASPLOS26/csrc/Reduction_Kernel.cuh) | 단순 Split-K 합산 kernel | 초급 |
| 2 | [`test_mm.cu`](../../../ZipServ_ASPLOS26/kernel_benchmark/test_mm.cu) | 할당, 복사, cuBLAS, 측정, 검증 | 초중급 |
| 3 | [`utils.h`](../../../ZipServ_ASPLOS26/kernel_benchmark/utils.h) | 입력 생성과 실제 host 압축 | 중급 |
| 4 | [`TilingConfig.h`](../../../ZipServ_ASPLOS26/csrc/TilingConfig.h) | tile과 warp 구성 상수 | 중급 |
| 5 | [`L_API.cu`](../../../ZipServ_ASPLOS26/csrc/L_API.cu) | N별 dispatch, launch, Split-K | 중급 |
| 6 | [`AsyncCopy_PTX.cuh`](../../../ZipServ_ASPLOS26/csrc/AsyncCopy_PTX.cuh) | `cp.async` wrapper | 고급 |
| 7 | [`MMA_PTX.cuh`](../../../ZipServ_ASPLOS26/csrc/MMA_PTX.cuh) | `ldmatrix`, `mma.sync` | 고급 |
| 8 | [`MatMulUtilities.cuh`](../../../ZipServ_ASPLOS26/csrc/MatMulUtilities.cuh) | tile load/store와 swizzle | 고급 |
| 9 | [`L_Kernel.cuh`](../../../ZipServ_ASPLOS26/csrc/L_Kernel.cuh) | fused decode + Tensor Core GEMM | 최상급 |

`L_Kernel.cuh`부터 읽기 시작하지 않는 것이 좋다. 앞 파일의 layout과 helper를 모르면
register 배열의 의미를 추론하기 어렵다.

## 17. 권장 실습 순서

### 단계 1: 기본 CUDA

1. vector addition
2. grid-stride loop
3. 오류 검사와 CUDA Event 측정
4. shared-memory reduction

완료 기준: thread가 담당하는 index와 memory access를 종이에 설명할 수 있다.

### 단계 2: GEMM

1. naive FP32 GEMM
2. shared-memory tiled GEMM
3. boundary-safe GEMM
4. row-major/column-major와 cuBLAS 결과 비교

완료 기준: block/warp/thread가 C의 어느 좌표를 계산하는지 설명할 수 있다.

### 단계 3: BF16과 Tensor Core

1. BF16 입력, FP32 accumulator의 naive kernel
2. WMMA API를 이용한 GEMM
3. lane별 fragment layout 확인
4. `ldmatrix`와 `mma.sync` inline PTX

완료 기준: `m16n8k16`의 M/N/K 의미와 accumulator register 역할을 설명할 수 있다.

### 단계 4: 비동기 pipeline

1. 동기식 global→shared copy
2. `cp.async`로 교체
3. single buffer와 double buffer 비교
4. Nsight Compute로 stall과 throughput 비교

완료 기준: 어느 iteration에서 어느 buffer를 읽고 쓰는지 표로 그릴 수 있다.

### 단계 5: ZipServ

1. 8×8 BF16 tile 하나를 CPU에서 압축/복원
2. `__popcll` rank 계산을 손으로 검증
3. standalone decompression kernel 분석
4. 복원 결과를 register fragment에 직접 배치
5. ZipGEMM main loop와 Split-K 분석

## 18. 코드를 읽을 때 작성할 추적표

Kernel을 읽으며 다음 표를 직접 채우면 포인터 계산을 놓치지 않는다.

| 항목 | 기록할 내용 |
|---|---|
| Logical matrix | A/B/C의 shape |
| Physical layout | row-major 또는 column-major |
| Block tile | 이 block이 담당하는 M/N 범위 |
| Warp tile | 각 warp의 M/N 범위 |
| K range | Split-K가 맡은 시작/끝 K block |
| Global pointer | tile 시작 주소 |
| Shared layout | buffer별 시작 주소와 크기 |
| Register fragment | lane별 A/B/C register 수 |
| Synchronization | 무엇을 기다리는 barrier인지 |
| Output | workspace 또는 최종 C인지 |

## 19. 사전 지식 점검 문제

다음 질문에 답할 수 있으면 ZipGEMM main kernel을 읽기 시작할 준비가 된 것이다.

1. block과 warp의 차이는 무엇인가?
2. `threadIdx.x % 32`가 lane ID인 이유는 무엇인가?
3. coalesced global-memory access가 왜 중요한가?
4. shared-memory bank conflict란 무엇인가?
5. `__syncthreads()`와 `cp.async.wait_group`의 역할은 어떻게 다른가?
6. BF16의 sign/exponent/mantissa bit 수는 각각 얼마인가?
7. `C[M,N] = A[M,K] × B[K,N]`의 FLOP 수는 왜 `2MNK`인가?
8. `m16n8k16`은 어떤 행렬 tile을 의미하는가?
9. `__popcll`로 bitmap rank를 어떻게 계산하는가?
10. Split-K가 병렬성을 늘리면서 reduction 비용을 만드는 이유는 무엇인가?
11. 압축 weight traffic 감소와 decode instruction 증가 사이의 trade-off는 무엇인가?
12. kernel launch 직후가 아니라 synchronize에서 오류가 나타날 수 있는 이유는 무엇인가?

## 20. 다음 학습 문서 제안

이 배경 문서 다음에는 아래 순서로 별도 실습 문서를 만드는 것이 적절하다.

```text
01_cuda_execution_and_memory.md
02_reduction_kernel_walkthrough.md
03_naive_to_tiled_gemm.md
04_bf16_and_tensor_cores.md
05_cp_async_pipeline.md
06_triple_bitmap_format.md
07_zipgemm_kernel_walkthrough.md
```

각 단계에서는 ZipServ 원본을 바로 수정하기보다 `source/examples/`에 작은 독립
프로그램을 만들고, 결과를 cuBLAS 또는 CPU reference와 비교하는 편이 안전하다.
