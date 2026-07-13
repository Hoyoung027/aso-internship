# TVM-FFI + FlashInfer CUBIN launcher 단계

이 디렉터리의 launcher는 `build_cubin.py --execute`가 만든 실제 symbol과
resource 정보를 확인한 다음 구현합니다. 템플릿 kernel 이름이나 launch
configuration을 추측해서 넣지 않습니다.

## 1. 필요한 산출물

```text
artifacts/generated/cubin/single_decode_sm86_fp16_h128.cubin
artifacts/generated/cubin/single_decode_sm86_fp16_h128.symbols.txt
artifacts/generated/cubin/single_prefill_sm86_fp16_h128.cubin
artifacts/generated/cubin/single_prefill_sm86_fp16_h128.symbols.txt
```

추가 확인:

```bash
cuobjdump --dump-elf-symbols <file.cubin>
cuobjdump --dump-resource-usage <file.cubin>
cuobjdump --dump-sass <file.cubin>
```

## 2. launcher가 재현해야 하는 것

기존 FlashInfer host dispatcher와 정확히 동일하게 다음을 구성해야 합니다.

- FlashInfer `Params`의 binary layout과 모든 field
- CUBIN kernel argument 배열 (`Params`는 by-value argument)
- grid/block dimension
- dynamic shared-memory 크기
- `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
- `TVMFFIEnvGetStream`으로 얻은 현재 stream
- 현재 CUDA device guard

TVM-FFI CUBIN embedding의 최소 예시는 다음 파일에 있습니다.

```text
../../tvm-ffi/implementation/launcher.cpp
```

## 3. 첫 milestone

먼저 다음 두 고정점만 지원합니다.

```text
Decode:
  kv_len=128, q_heads=32, kv_heads=8, head_dim=128, FP16, NHD

Prefill:
  qo_len=kv_len=128, causal, q_heads=32, kv_heads=8,
  head_dim=128, FP16, NHD
```

지원하지 않는 값은 `TVM_FFI_ICHECK`로 즉시 거부합니다. 첫 milestone에서
sequence length를 일반화하지 않습니다.

## 4. Decode 확장

FlashInfer의 기본 single-decode dispatcher는 `kv_len > 256`이고 workspace가
있으면 KV partition을 사용합니다. 긴 Decode를 기존 경로와 동일하게 만들려면
다음이 모두 필요합니다.

1. partition Decode kernel
2. temporary output/LSE layout
3. occupancy에 따른 chunk 크기 계산
4. merge-state kernel

따라서 `kv_len=128` 단일 kernel 정확성을 통과한 다음 구현합니다.

## 5. Prefill 확장

Prefill은 `qo_len * GQA group size`에 따라 CTA tile을 선택합니다. 여러 길이를
지원하려면 CUBIN에 필요한 specialization이 실제로 들어 있는지 확인하고 기존
`FA2DetermineCtaTileQ`와 같은 선택을 launcher에서 재현해야 합니다.

## 6. 검증 순서

1. 같은 seed/input으로 기존 TVM-FFI 결과 저장
2. CUBIN launcher 결과 저장
3. `compare_outputs.py`로 FP16 tolerance 검증
4. Compute Sanitizer 실행
5. Nsight Systems에서 현재 stream과 kernel symbol 확인
6. 정확성 통과 후에만 benchmark adapter에 `cubin`을 추가

완료 기준은 "CUBIN이 launch됨"이 아니라 같은 입력에 대해 기존 FlashInfer와
같은 결과를 만들고, profiler에서 의도한 SM86 kernel이 확인되는 것입니다.
