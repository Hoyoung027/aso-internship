# flashinfer/hoyoung

FlashInfer에서 Python API가 C++/CUDA 쪽 함수로 연결되는 흐름을 발췌한 학습용 폴더입니다.

## JIT 경로

```text
Python API
  -> gen_gemm_module()
  -> JitSpec.build_and_load()
  -> tvm_ffi.load_module(.so)
  -> module.bmm_fp8(...)
  -> TVM_FFI_DLL_EXPORT_TYPED_FUNC
  -> C++ bmm_fp8(...)
  -> cuBLASLt/CUDA 실행
```

관련 파일:

- `1_python_api/bmm_fp8_api_excerpt.py`: Python API와 `module.bmm_fp8(...)` 호출 지점
- `4_jit_loader/gen_gemm_module_excerpt.py`: 어떤 `.cu` 파일을 하나의 모듈로 묶는지
- `4_jit_loader/jit_core_excerpt.py`: `.so` 빌드와 `tvm_ffi.load_module`
- `2_jit_binding/flashinfer_gemm_binding_excerpt.cu`: C++ 함수를 tvm-ffi 심볼로 export
- `2_jit_binding/tvm_ffi_utils_excerpt.h`: tvm-ffi 타입/유틸 include
- `3_launcher/bmm_fp8_launcher_excerpt.cu`: `TensorView`를 받아 실제 GEMM 호출

## CUBIN 경로

일부 커널은 `.cu` 소스 JIT 대신 사전 빌드된 `.cubin`을 CUDA Driver API로 로드합니다.
함수 호출 자체는 여전히 tvm-ffi로 들어오고, cubin 바이트 전달은 별도 ctypes 콜백을 사용합니다.

관련 파일:

- `5_cubin/`: cubin 다운로드, 캐시, callback 등록, driver API launch 흐름

## 원본에서 볼 곳

- `flashinfer/gemm/gemm_base.py`
- `flashinfer/jit/core.py`
- `flashinfer/jit/gemm/core.py`
- `csrc/flashinfer_gemm_binding.cu`
- `csrc/bmm_fp8.cu`
- `flashinfer/jit/cubin_loader.py`
- `include/flashinfer/cubin_loader.h`
