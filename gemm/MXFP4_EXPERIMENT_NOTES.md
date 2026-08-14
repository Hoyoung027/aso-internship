# MXFP4 실험 기록: PyTorch, TorchAO, cuBLASLt, FlashInfer와 GPT-OSS-20B

작성일: 2026-08-11

이 문서는 RTX PRO 6000 Blackwell Server Edition(SM120)에서 GPT-OSS-20B의
MXFP4 GEMM 경로를 확인하면서 시도한 구현, 실패 원인, 실제 vLLM 실행 경로를
정리한다. 대상 환경은 PyTorch 2.11.0+cu130, TorchAO 0.17.0,
FlashInfer 0.6.12, cuDNN 9.19, CUDA 13.0이다.

## 먼저: MXFP4는 어떤 형식인가

MXFP4는 OCP Microscaling(MX) 계열의 block-scaled 4-bit floating-point
형식이다. 하나의 tensor 전체에 scale 하나를 적용하는 대신, 연속된 32개
값마다 scale 하나를 공유한다.

```text
32개의 원소로 이루어진 block

  FP4 values: [v0, v1, ..., v31]   각 원소는 E2M1, 4 bit
  shared scale: s                  E8M0, 8 bit

  복원값: x_i ≈ decode_e2m1(v_i) × decode_e8m0(s)
```

### E2M1 FP4 element

각 양자화 원소는 총 4 bit를 사용한다.

```text
+------+----------+----------+
| sign | exponent | mantissa |
+------+----------+----------+
| 1bit |  2bits   |  1bit    |
+------+----------+----------+
```

- `E2M1`은 exponent 2 bit, mantissa 1 bit라는 뜻이다.
- PyTorch는 두 FP4 값을 byte 하나에 pack한 shell dtype
  `torch.float4_e2m1fn_x2`로 표현한다.
- FP4 값만으로는 표현 범위가 매우 좁기 때문에 block scale과 함께 해석해야
  한다.

### E8M0 shared scale

32개 원소마다 8-bit `E8M0` scale 하나를 둔다.

```text
+----------+----------+
| exponent | mantissa |
+----------+----------+
|  8bits   |  0bits   |
+----------+----------+
```

- mantissa가 없으므로 정상적인 scale 값은 기본적으로 2의 거듭제곱이다.
- PyTorch dtype 이름은 `torch.float8_e8m0fnu`이다.
- zero-point를 쓰는 integer affine quantization과 달리, MXFP4는 FP4 element와
  공유 exponent scale의 조합으로 값을 표현한다.

### 왜 4.25 bits/parameter인가

원소 32개마다 FP4 data 128 bit와 scale 8 bit를 저장한다.

```text
(32 × 4 bit + 8 bit) / 32 = 4.25 bit/value
```

그래서 GPT-OSS model card는 MXFP4 MoE weight를 4.25 bits/parameter로
설명한다. 실제 kernel용 storage에는 정렬을 위한 padding과 scale swizzle이
추가될 수 있으므로 tensor의 물리적 byte 수가 항상 이 이론값과 정확히
일치하는 것은 아니다.

### logical format과 physical scale layout

MXFP4의 논리적 규격은 `E2M1 data + E8M0 scale + block size 32`이다. GPU
kernel은 scale을 읽기 좋게 만들기 위해 논리적인 2차원 scale matrix를
`128x4` 또는 `SWIZZLE_32_4_4` 형태로 재배열할 수 있다.

```text
logical scale:  각 row의 K축 32개마다 scale 하나
physical scale: kernel이 요구하는 padded/swizzled byte layout
```

swizzle은 양자화 수치 형식을 바꾸는 것이 아니라 scale의 메모리 배치를
바꾸는 것이다. 따라서 scale dtype과 block size가 같더라도 backend가 기대하는
swizzle이 다르면 그대로 호환되지 않는다. 형식 자체는
[OCP Microscaling Formats 사양](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)과
[TorchAO quantization overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html)에서
확인할 수 있다.

## 결론 요약

1. MXFP4와 NVFP4는 값 부분은 같은 E2M1 FP4를 사용하지만 scale 규격이 다르다.
2. `torch.nn.functional.linear`와 `torch.nn.Linear`는 scale metadata를 받는
   저수준 quantized GEMM API가 아니다.
3. private `aten._scaled_mm.out`에 FlashInfer MXFP4 storage를 전달하면 FP4에
   허용된 NVFP4 scaling 조건과 맞지 않아 입력 검증에서 실패한다.
4. public `torch.nn.functional.scaled_mm`은 MXFP4 recipe를 표현할 수 있지만,
   PyTorch 2.11 CUDA backend가 native MXFP4 실행을 B200/B300으로 제한하여
   SM120에서 실패한다.
5. TorchAO 0.17의 `MXDynamicActivationMXWeightConfig(..., AUTO)`도 내부적으로
   같은 `F.scaled_mm`을 호출하므로 동일하게 실패한다.
6. 문서화된 cuBLASLt FP4 block-scaling 경로는 E4M3 scale/block-16, 즉
   NVFP4이다. E8M0/block-32는 FP8용 MXFP8 scaling이며, public cuBLASLt API의
   FP4 MXFP4 경로가 아니다.
7. GPT-OSS 체크포인트의 MoE expert weight는 MXFP4이고 모델 activation은
   기본적으로 BF16이다. 따라서 Marlin과 같은 weight-only backend는 W4A16으로
   실행한다. 반면 확인한 FlashInfer CUTLASS 변형은 BF16 activation을 런타임에
   MXFP8로 바꾸어 MXFP4 weight와 곱하므로 W4A8이다.

## MXFP4와 NVFP4는 같은 FP4가 아니다

| 형식 | element | block scale | block 크기 | 추가 scale |
|---|---|---|---:|---|
| MXFP4 | E2M1 FP4 | E8M0 | 32 | 없음 |
| NVFP4 | E2M1 FP4 | E4M3 | 16 | 보통 tensor-wide FP32 scale |

두 형식 모두 packed element를 `torch.float4_e2m1fn_x2`로 표현할 수 있다.
그러나 scale dtype, block 크기, scale layout이 다르므로 MXFP4 storage를
NVFP4 kernel에 단순히 `view()`해서 사용할 수 없다. `view()`는 저장된 bit와
layout을 재해석할 뿐 block-32를 block-16으로 재양자화하지 않는다.

TorchAO도 MXFP4를 E2M1 data와 E8M0 scale의 조합으로 설명한다. 반면 NVIDIA의
NVFP4 설명은 E2M1 data, block-16 E4M3 scale, tensor-wide FP32 scale의 계층적
scaling을 사용한다. 참고: [TorchAO quantization overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html),
[NVIDIA Transformer Engine NVFP4 설명](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#beyond-fp8-training-with-nvfp4).

## PyTorch에서 시도한 경로

### API 계층

```text
torch.nn.Linear / torch.nn.functional.linear
    └─ 일반 Linear 계층 및 고수준 연산

torch.ops.aten._scaled_mm.out                 (private ATen overload)
    └─ 사전 양자화 A/B와 scale을 받고 preallocated out에 기록

torch.nn.functional.scaled_mm                (public Python API, PyTorch 2.11)
    └─ ScalingType/SwizzleType을 명시
       └─ aten._scaled_mm_v2                  (내부 ATen op)

TorchAO MXDynamicActivationMXWeightConfig(AUTO)
    └─ MXTensor로 weight/activation 양자화
       └─ FP4일 때 torch.nn.functional.scaled_mm 호출
```

`nn.Linear`는 parameter, bias, state dict와 일반적인 `y = xW^T + b` 의미를
제공한다. 기본 BF16 Linear는 BF16 input과 BF16 weight를 사용한다. MXFP4처럼
별도 block scale과 swizzle metadata가 필요한 저장 형식을 기본 `nn.Linear`에
직접 전달할 수는 없다. TorchAO는 tensor subclass와 quantized Linear dispatch를
통해 이 간극을 메운다.

### 1. 초기 benchmark import 버그

처음 추가한 `torch_mxfp4` 코드는 weight quantization에
`flashinfer.mxfp4_quantize()`를 사용하면서 Torch mode에서 local
`flashinfer = None`으로 설정했다. 따라서 실제 PyTorch op에 도달하기 전에
다음 오류로 중단되었다.

```text
AttributeError: 'NoneType' object has no attribute 'mxfp4_quantize'
```

이 오류는 PyTorch MXFP4 지원 여부와 무관한 benchmark harness 버그였다.
원본 로그는 [torch-mxfp4-smoke-1997424.out](./result/torch-mxfp4-smoke-1997424.out)에
남아 있다. import 조건을 수정한 뒤 아래의 실제 API 검증까지 진행했다.

### 2. private `aten._scaled_mm.out`

시도한 입력은 다음과 같다.

```text
A/B data  = torch.float4_e2m1fn_x2
scale     = torch.float8_e8m0fnu
block     = 32
layout    = FlashInfer 128x4 swizzled storage
output    = preallocated BF16 tensor
```

호출은 다음 형태였다.

```python
torch.ops.aten._scaled_mm.out(
    activation_q,
    weight_q_t,
    activation_scale,
    weight_scale_t,
    out_dtype=torch.bfloat16,
    out=output,
)
```

PyTorch는 FP4 input에 대해 block-16/E4M3 scale 조건을 요구했고, 전달된
block-32/E8M0 scale을 `Invalid scaling configuration`으로 거부했다. 즉 이
private overload가 인식한 FP4 경로는 NVFP4 조건이며 MXFP4 조건이 아니었다.

### 3. public `torch.nn.functional.scaled_mm`

PyTorch 2.11의 public API는 scaling recipe와 swizzle을 명시할 수 있다.
[benchmark_gemm.py](./benchmark_gemm.py)의 `torch_mxfp4` 경로에서 다음을
시도했다.

```python
F.scaled_mm(
    activation_q,
    weight_q_t,
    activation_scale,
    F.ScalingType.BlockWise1x32,
    weight_scale_t,
    F.ScalingType.BlockWise1x32,
    F.SwizzleType.SWIZZLE_32_4_4,
    F.SwizzleType.SWIZZLE_32_4_4,
    output_dtype=torch.bfloat16,
)
```

이 호출은 이전의 scaling-configuration 검증은 통과했다. 즉 MXFP4의
E8M0/block-32와 swizzled scale을 API 수준에서 올바르게 표현했다. 그러나 실제
CUDA dispatch에서 다음 오류가 발생했다.

```text
NotImplementedError: MXFP4 scaling only supported in CUDA for B200/B300
```

따라서 실패 원인은 format이나 shape가 아니라 PyTorch 2.11 native CUDA
backend의 제품 지원 범위이다. RTX PRO 6000은 Blackwell SM120이지만 이 경로가
허용한 B200/B300 계열이 아니다. `F.scaled_mm`은 내부적으로
`aten._scaled_mm_v2`를 호출하며, 이 내부 op를 직접 호출해도 backend 제한은
사라지지 않는다.

또한 `F.scaled_mm`에는 사용한 버전에서 `out=` overload가 없으므로, 실행이
가능한 GPU에서도 기존 `_scaled_mm.out` 및 FlashInfer의 preallocated-output
측정과 조건이 완전히 같지는 않다.

## TorchAO 0.17 실험

다음 공식 MXFP4 config로 `nn.Linear`를 양자화했다.

```python
MXDynamicActivationMXWeightConfig(
    block_size=32,
    activation_dtype=torch.float4_e2m1fn_x2,
    weight_dtype=torch.float4_e2m1fn_x2,
    kernel_preference=KernelPreference.AUTO,
)
```

TorchAO 공식 문서는 이 config를 Blackwell용 prototype MXFP4 inference 경로로
제시한다. 참고: [MXDynamicActivationMXWeightConfig API](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig.html).

설치된 TorchAO 0.17 구현을 확인하면 FP4 `AUTO` GEMM은 quantized A/W를 준비한
뒤 `F.scaled_mm`에 `BlockWise1x32`와 `SWIZZLE_32_4_4`를 전달한다. 따라서
PyTorch public API 실험과 같은 native CUDA backend에 도달한다.

실제 `torchao_mxfp4_auto` 결과는 5 projection × 4 M, 총 20개 case가 모두
다음 상태였다.

```text
status = unsupported
error  = NotImplementedError: MXFP4 scaling only supported in CUDA for B200/B300
```

결과는 [TorchAO raw CSV](./result/torchao-mxfp4-auto-20260811-161404/raw-torchao-mxfp4-trial1.csv)에
남아 있다.

TorchAO의 `KernelPreference.EMULATED`는 실행 가능성을 확인하는 성능 kernel이
아니다. MXFP4 tensor를 high precision으로 dequantize한 뒤 BF16/FP32 GEMM을
수행하는 correctness/debug 경로이므로 MXFP4 native 성능 비교에 사용하면 안
된다. TorchAO 0.17 MX format config는 이 경우 `AUTO` 또는 `EMULATED`만
허용하며, SM120용 별도 Triton MXFP4 GEMM 선택지는 제공하지 않았다.

## cuBLASLt가 제공하는 FP4 형식

“cuBLASLt가 FP4를 지원한다”와 “cuBLASLt가 MXFP4를 지원한다”는 같은 말이
아니다. CUDA 13의 공개 cuBLASLt 문서에서 block-scaled narrow precision은
다음과 같이 규정된다.

| data | public scale mode | 의미 |
|---|---|---|
| FP8 | `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` | MXFP8, E8M0/block-32 |
| FP4 | `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` | NVFP4, E4M3/block-16 |

문서는 FP8과 FP4 precision을 섞는 것도 지원하지 않는다고 명시한다. 따라서
공개 cuBLASLt API에는 `FP4 E2M1 + E8M0/block-32`, 즉 MXFP4 조합이 노출되어
있지 않다. cuBLASLt가 지원하는 documented FP4 block-scaling은 NVFP4이다.
참고: [cuBLAS 13.x, 16/32-element 1D block scaling](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling),
[cublasLtMatmulMatrixScale_t](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulmatrixscale-t).

이 사실은 private `_scaled_mm.out`이 FP4에 E4M3/block-16을 요구한 결과와도
일치한다. 한편 PyTorch의 새 `F.scaled_mm` MXFP4 경로가 B200/B300에서 어떤
내부 NVIDIA kernel/API를 사용하는지는 public cuBLASLt 지원표만으로 단정하지
않는다. 확실한 사실은 현재 PyTorch wheel이 그 경로를 SM120에 제공하지
않았다는 것이다.

## FlashInfer dense MXFP4 microbenchmark와 한계

현재 [benchmark_gemm.py](./benchmark_gemm.py)의 FlashInfer MXFP4 실험은 다음
호출을 사용한다.

```python
flashinfer.mm_fp4(
    activation_q,
    weight_q_t,
    activation_scale,
    weight_scale_t,
    block_size=32,
    use_nvfp4=False,
    backend="auto",
)
```

RTX PRO 6000 SM120과 FlashInfer 0.6.12에서는 이 dense MXFP4 경로가 cuDNN을
통해 실행되었다. PyTorch/cuBLASLt public FP4 경로가 실패한 것과
FlashInfer가 성공한 것이 모순은 아니다. 서로 다른 backend와 kernel을 사용한
결과이다.

중요하게도 이 microbenchmark는 activation과 weight를 모두 MXFP4로
양자화한다. 즉 A4W4 실험이며, GPT-OSS의 일반적인 BF16-activation/MXFP4-weight
W4A16 실행과 동일하지 않다. 이 결과는 FlashInfer `mm_fp4`의 default/tuned
tactic 및 동일 shape의 dense GEMM 특성을 보기 위한 실험으로 해석해야 한다.

## GPT-OSS-20B 체크포인트와 실제 실행 precision

### 체크포인트 자체

OpenAI는 GPT-OSS의 MoE weight를 MXFP4로 post-training했고, 공개 checkpoint의
MoE weight를 MXFP4로 배포했다. Attention, router, embedding, LM head 등은
MXFP4 변환 대상에서 제외된다. 참고: [OpenAI GPT-OSS model card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf),
[GPT-OSS-20B checkpoint 소개](https://huggingface.co/openai/gpt-oss-20b),
[Transformers MXFP4 config의 제외 module](https://huggingface.co/docs/transformers/main/en/quantization/mxfp4).

따라서 모델/checkpoint 관점의 자연스러운 계산은 다음과 같다.

```text
MoE input activation: BF16
MoE expert weight:    MXFP4
연산 분류:            W4A16
```

다만 “항상 W4A16이 기본”이라고 단정하면 안 된다. checkpoint가 activation을
MXFP8로 저장한다는 뜻은 아니지만, serving backend가 성능을 위해 BF16
activation을 런타임에 MXFP8로 바꿀 수 있기 때문이다.

### vLLM Marlin: 확인된 W4A16

현재 vLLM 0.23.0 실험의 `marlin_w4a16` manifest에는
`activation_dtype=bf16`, `backend_actual=MARLIN`, `status=completed`가 기록돼
있다.

```text
/lustre/hybyun0207/vllm023-moe/runs/job-1906585/manifests/marlin_w4a16.json
```

이 경로는 MXFP4 expert weight를 weight-only 방식으로 사용하고 activation은
BF16으로 유지하므로 W4A16이다.

### vLLM + FlashInfer CUTLASS: 확인된 MXFP4 × MXFP8

확인한 FlashInfer CUTLASS 실험은 다음 option을 명시했다.

```text
--moe-backend flashinfer_cutlass
--quantization-config.moe.activation mxfp8
```

manifest에는 다음이 기록되어 있다.

```text
backend_actual   = FLASHINFER_CUTLASS_MXFP4_MXFP8
activation_dtype = mxfp8
status           = completed
```

근거 파일:

```text
/lustre/hybyun0207/vllm023-moe/runs/pilot-1905565-20260728-193401/
  manifests/flashinfer_cutlass_w4a8_tune_on.json
```

같은 실행의 profiler에는 `vllm::mxfp8_quantize`가 각 측정 구간마다 1,200회
호출된 것이 기록되어 있다.

```text
/lustre/hybyun0207/vllm023-moe/runs/pilot-1905565-20260728-193401/
  flashinfer_cutlass_w4a8_tune_on/logs/server.log
```

설치된 vLLM source도 `FLASHINFER_CUTLASS_MXFP4_MXFP8` backend의 activation
key를 dynamic MXFP8로 지정하고, CUTLASS kernel에 swizzled MXFP8 activation
scale을 전달한다. 즉 입력은 MoE layer 진입 시 BF16이지만 expert GEMM 직전에
MXFP8로 동적 양자화된다.

```text
BF16 hidden state
    → dynamic MXFP8 quantization (E4M3 data + E8M0/block-32 scale)
    → FlashInfer CUTLASS fused MoE
       A = MXFP8, W = checkpoint MXFP4
    → BF16 output
```

따라서 이 kernel의 GEMM precision 분류는 W4A8이다. “BF16 × MXFP4를 그대로
계산한다”가 아니라 “BF16 source activation을 MXFP8로 변환한 뒤 MXFP4
weight와 계산한다”가 맞다. vLLM의 최신 backend 목록도 MXFP4-BF16과
MXFP4-MXFP8 변형을 별도로 구분한다. 참고: [vLLM 저장소](https://github.com/vllm-project/vllm),
[vLLM 릴리스의 MXFP4/FlashInfer 변경 내역](https://github.com/vllm-project/vllm/releases).

또한 이 FlashInfer CUTLASS fused MoE는 public cuBLASLt의 FP4 matmul과 동일한
경로가 아니다. CUTLASS/TensorRT-LLM 계열의 fused expert kernel이므로
cuBLASLt가 MXFP4를 공개 지원하지 않는다는 사실과 양립한다.

## 최종 해석

| 대상 | SM120 결과 | 올바른 해석 |
|---|---|---|
| BF16 `F.linear` | 지원 | 일반 BF16 baseline |
| private `aten._scaled_mm.out` + MXFP4 | 실패 | FP4 scaling contract가 NVFP4 조건 |
| public `F.scaled_mm` + MXFP4 | 실패 | API 표현은 가능하나 native CUDA가 B200/B300 한정 |
| TorchAO MXFP4 AUTO | 실패 | 내부적으로 같은 `F.scaled_mm` 경로 |
| public cuBLASLt FP4 | MXFP4 경로 없음 | documented FP4는 NVFP4 block-16/E4M3 |
| FlashInfer dense `mm_fp4` | 지원 | SM120 cuDNN 기반 A4W4 microbenchmark |
| vLLM Marlin GPT-OSS | 지원 | checkpoint MXFP4 weight + BF16 activation, W4A16 |
| vLLM FlashInfer CUTLASS 실험 | 지원 | BF16을 MXFP8로 동적 변환, MXFP4×MXFP8 W4A8 |

향후 PyTorch native MXFP4 비교를 다시 시도하려면 B200/B300에서 동일 코드를
실행하거나, PyTorch가 SM120 MXFP4 backend를 추가한 버전에서 재검증해야 한다.
현재 RTX PRO 6000 결과에서는 `torch_mxfp4`와 `torchao_mxfp4_auto`를 latency
그래프의 0 또는 실패값으로 취급하지 않고 `unsupported`로 유지해야 한다.
