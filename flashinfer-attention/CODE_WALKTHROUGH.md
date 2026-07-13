# 코드 해설

이 문서는 `flashinfer-attention/`의 각 코드가 무엇을 담당하고, 실행 중 어떤
데이터가 어디로 이동하는지 설명합니다.

## 전체 흐름

```text
setup_variants.sh
  ├─ legacy PyTorch-binding FlashInfer worktree
  └─ TVM-FFI FlashInfer worktree

benchmark.py
  ├─ Q/K/V와 output/workspace 사전 할당
  ├─ single_decode 또는 single_prefill module.run 반복
  ├─ CUDA event GPU 시간 측정
  ├─ host enqueue 시간 측정
  └─ JSONL / output tensor 저장

compare_outputs.py
  └─ 두 구현의 저장 output 정확성 비교

prepare_cubin_sources.py
  └─ FlashInfer JIT generator로 SM86 전용 CUDA source/build.ninja 생성

build_cubin.py
  ├─ build.ninja에서 실제 NVCC 명령 추출
  ├─ .cuda.o compile을 --cubin compile로 변환
  └─ kernel symbol/resource 정보 저장

실제 CUBIN launcher
  └─ 생성된 symbol/launch metadata 확인 후 다음 단계에서 연결
```

## `setup_variants.sh`

### 왜 worktree가 두 개인가?

FlashInfer는 과거에 JIT `.so`를 `torch.ops`로 로드했고, 이후 TVM-FFI
`load_module()`로 전환했습니다. 두 커밋을 한 working directory에서 계속
checkout하면 build cache와 Python package가 섞일 수 있으므로 Git worktree로
분리합니다.

```text
ea569640  -> legacy torch.ops binding
86d3e136  -> TVM-FFI binding
```

이 방법은 1차 실험을 빠르게 만드는 용도입니다. 논문 수준으로 binding만
엄밀하게 비교하려면 최종적으로 하나의 kernel source revision에 두 binding을
함께 구현해야 합니다.

## `benchmark.py`

### GPU 검사 위치

`CUDA_VISIBLE_DEVICES` 검사는 `import torch`보다 먼저 실행됩니다. torch가 먼저
CUDA context를 만들면 뒤늦게 환경변수를 바꿔도 GPU 0 접근을 확실히 막을 수
없기 때문입니다.

```text
CUDA_VISIBLE_DEVICES=1
physical GPU 1 -> logical cuda:0
```

### `SingleAttentionRunner`

Runner는 한 프로세스에서 다음을 한 번만 준비합니다.

```text
FlashInfer module
temporary workspace
입력 Q/K/V
출력 tensor
```

반복 구간에서는 `module.run()`만 호출합니다. 이렇게 해야 public API의 tensor
allocation 비용이 PyTorch/TVM-FFI binding 차이를 가리지 않습니다.

### Decode 인자

```text
q: [num_qo_heads, head_dim]
k: [kv_len, num_kv_heads, head_dim]
v: [kv_len, num_kv_heads, head_dim]
```

`module.run()`을 직접 사용하므로 public API의 `use_tensor_cores` flag는 보이지
않습니다. 호출한 module 자체가 `get_single_decode_module()`에서 온 실제
single-decode specialization이므로 Prefill 우회 경로가 아닙니다.

### Prefill 인자

```text
q: [qo_len, num_qo_heads, head_dim]
k/v: [kv_len, num_kv_heads, head_dim]
mask_mode=1: causal
layout=0: NHD
backend: fa2
```

이번 실험은 FP16이므로 FP8 calibration scale 인자는 `None`입니다.

### 정확성 reference

`torch_reference()`는 GQA를 이해하기 쉬운 형태로 계산합니다.

```text
Q heads=32, KV heads=8
-> KV head를 reference 안에서 4번씩 반복
-> FP32 QK^T
-> softmax
-> FP32 probability x V
-> FP16 output
```

이는 검증용 구현이라 메모리 효율이 낮습니다. 큰 Prefill에서는 logits가
`[heads, qo_len, kv_len]` 크기가 되므로 `--check-max-len` 이하에서만 실행합니다.

### 시간의 의미

| 필드 | 포함되는 것 | 포함되지 않는 것 |
|---|---|---|
| `import_ms` | torch/flashinfer import | module compile |
| `module_load_ms` | JIT build 또는 cache load | 첫 kernel 실행 |
| `first_call_ms` | 첫 launch와 GPU 완료 | import/build |
| `gpu_us` | CUDA event 사이의 GPU stream 실행 | Python/JIT 시간 |
| `host_enqueue_median_us` | Python -> binding -> launch enqueue | 마지막 GPU 완료 대기 |

`gpu_us`가 비슷하고 host 시간만 다르면 binding 차이입니다. CUBIN에서
`module_load_ms`가 줄고 `gpu_us`가 같다면 예상한 결과입니다.

## `compare_outputs.py`

`benchmark.py --save-tensors`가 만든 다음 파일을 이름으로 짝지어 CPU에서
비교합니다.

```text
decode-128.pt
decode-512.pt
prefill-128.pt
...
```

이 비교 프로그램은 CUDA를 사용하지 않으므로 성능 측정과 독립적입니다.

## `prepare_cubin_sources.py`

직접 attention CUDA 코드를 새로 작성하는 프로그램이 아닙니다. FlashInfer의
기존 generator를 다음 고정 설정으로 호출합니다.

```text
SM86
FP16
head_dim_qk=head_dim_vo=128
position encoding=NONE
sliding window=False
logits soft cap=False
Prefill backend=FA2
```

generator 호출로 Jinja template이 실제 `.cu`와 `.inc` 파일로 렌더링됩니다.
그 다음 `JitSpec.write_ninja()`로 FlashInfer가 사용할 정확한 NVCC 명령을
기록합니다. 이 단계에서는 아직 NVCC compile을 실행하지 않습니다.

`manifest.json`은 source와 build 명령의 위치, compile flag, 실험 설정을 다음
단계로 전달합니다.

## `build_cubin.py`

### 명령 변환

기존 JIT object compile이 다음과 같다고 가정하면:

```text
nvcc <FlashInfer flags> -c single_decode_kernel.cu -o kernel.cuda.o
```

다음처럼 바꿉니다.

```text
nvcc <동일한 FlashInfer flags> single_decode_kernel.cu \
  --cubin -o single_decode_sm86_fp16_h128.cubin
```

include path, macro, optimization, SM86 gencode를 그대로 유지하는 것이
중요합니다. 기본 동작은 dry-run이며 `--execute`를 줘야 실제 NVCC를 실행합니다.

### 왜 바로 launcher를 만들지 않는가?

FlashInfer kernel은 C++ template이라 컴파일 후 symbol이 mangling됩니다. 또한
kernel마다 다음 정보가 다릅니다.

```text
argument ABI
grid/block
dynamic shared memory
register 사용량
Decode partition/merge 여부
Prefill CTA tile
```

따라서 실제 CUBIN에서 얻은 `*.symbols.txt`와 `*.resources.txt` 없이 이름과
launch 값을 추측하면 안 됩니다.

## CUBIN launcher에서 이어질 흐름

향후 launcher는 다음 역할을 수행합니다.

```text
torch.Tensor
 -> TVM-FFI TensorView
 -> FlashInfer Params 구성
 -> 현재 PyTorch CUDA stream 획득
 -> CUBIN module에서 kernel symbol 검색
 -> grid/block/shared-memory 설정
 -> kernel launch
```

첫 milestone은 Decode `kv_len=128`과 causal Prefill `qo_len=kv_len=128`만
지원합니다. 정확성 검증 후 긴 Decode의 partition/merge와 Prefill의 여러 CTA
tile을 추가합니다.

