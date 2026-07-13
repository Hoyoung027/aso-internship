# TVM-FFI 정리: Python과 C++/CUDA는 어떻게 연결되는가

> 핵심: tvm-ffi는 Python 코드를 C++로 바꾸는 컴파일러가 아니다. 이미 컴파일된 C/C++ 함수와 Python 런타임 사이에서 **함수 호출 규격, 인자 전달, 함수 조회**를 맞춰주는 FFI 계층이다.

이 문서는 기본 예제(`add_one_cpu`)와 FlashInfer의 실전 구조(`bmm_fp8`, JIT `.so`, cubin)를 함께 정리한다.

## 한 줄 요약

```text
Python 값
  -> TVMFFIAny / PackedArgs로 포장
  -> safe_call 규격으로 C++ 함수 호출
  -> Tensor는 DLPack/TensorView로 전달
  -> C++ 런처가 실제 CPU/CUDA 계산 수행
```

tvm-ffi에서 중요한 것은 세 가지다.

- `TVMFFIAny`: 언어 경계를 넘는 값을 담는 공통 컨테이너
- `safe_call`: 모든 함수를 하나의 C ABI 호출 규격으로 맞춘 함수 포인터
- 함수 조회 방식: global registry 또는 loaded module symbol

## 폴더 구성

```text
1_example/       실제 add_one 예제
2_python_side/   Python 인자 포장과 C ABI 호출 진입
3_c_abi/         Python/C++이 공유하는 C ABI 정의
4_cpp_side/      safe_call, unpack, C++ 함수 호출
5_register/      global registry 등록/조회 흐름
```

앞의 `1_example`부터 `4_cpp_side`까지는 “이미 찾은 함수를 호출하는 흐름”이고, `5_register`는 “함수를 전역 이름표에 등록하는 흐름”이다.

## 호출 흐름: `mod.add_one_cpu(x, y)`

```text
Python: mod.add_one_cpu(x, y)
  -> Function.__call__
  -> Python 인자 x, y를 TVMFFIAny args[]로 포장
  -> TVMFFIFunctionCall(func, args, num_args, result)
  -> func->safe_call(...)
  -> __tvm_ffi_add_one_cpu(...)
  -> TVMFFIAny[]를 TensorView로 unpack
  -> C++ AddOne(TensorView x, TensorView y)
```

`numpy`나 `torch` 텐서는 값 전체를 복사해서 넘기기보다 DLPack/TensorView 형태로 포인터와 metadata를 전달한다.

## TVMFFIAny

`TVMFFIAny`는 경계를 넘는 값을 담는 공통 상자다.

```text
type_index  어떤 타입인지 표시
value       실제 값 또는 객체 포인터
```

Python 쪽은 인자를 `TVMFFIAny` 배열로 포장하고, C++ 쪽은 `unpack_call`을 통해 원하는 타입으로 꺼낸다.

예를 들어 Python에서 tensor를 넘기면 C++ 함수는 보통 다음처럼 받는다.

```cpp
void AddOne(TensorView x, TensorView y);
```

FlashInfer에서도 같은 방식으로 `torch.Tensor`가 C++에서는 `TensorView`로 보인다.

## safe_call

tvm-ffi는 다양한 C++ 함수 시그니처를 직접 Python에 노출하지 않는다. 대신 모든 함수를 다음 형태의 호출 규격으로 감싼다.

```text
int safe_call(void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result)
```

반환값 `int`는 에러 코드이고, 실제 결과는 `result`에 담긴다. 언어 경계 밖으로 C++ exception을 직접 던질 수 없기 때문에 이런 형태를 쓴다.

C++에서 다음 매크로를 쓰면:

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, AddOne);
```

대략 이런 외부 심볼이 생긴다.

```text
__tvm_ffi_add_one_cpu
```

이 심볼은 `safe_call` 규격을 따르고, 내부에서 `TVMFFIAny[]`를 `AddOne`의 실제 인자 타입으로 unpack한 뒤 호출한다.

## 함수 찾기 방식 2가지

tvm-ffi에서 헷갈리기 쉬운 지점은 “함수를 어디서 찾는가”이다. 크게 두 경로가 있다.

## 1. Global Registry 경로

이 경로는 프로세스 전역 해시맵에 이름과 함수를 저장한다.

```text
register_global_func("name", func)
  -> TVMFFIFunctionSetGlobal
  -> GlobalFunctionTable::Update
  -> table_["name"] = Function

get_global_func("name")
  -> TVMFFIFunctionGetGlobal
  -> table_["name"] 조회
```

관련 파일:

- `5_register/1_entry_python.py`
- `5_register/2_entry_cpp.cc`
- `5_register/3_setglobal.cc`
- `5_register/4_global_function_table.cc`

이 방식은 `name -> Function` 형태의 전역 registry를 쓴다.

## 2. Dynamic Module 경로

이 경로는 `.so` shared library를 열고, 그 안의 심볼을 조회한다.

```python
mod = tvm_ffi.load_module("path/to/library.so")
mod.add_one_cpu(x, y)
```

이때 `mod.add_one_cpu`는 전역 registry에서 `"add_one_cpu"`를 찾는 것이 아니다. 로드된 `.so` 안에서 다음 심볼을 찾는다.

```text
__tvm_ffi_add_one_cpu
```

구조는 이렇게 보는 것이 정확하다.

```text
loaded Module object
  -> library.so handle
  -> dynamic symbol table
  -> __tvm_ffi_<name>
```

즉 global registry와 dynamic module은 둘 다 tvm-ffi의 `Function` 호출 규격을 쓰지만, 함수 조회 위치가 다르다.

```text
Global registry:
  name -> process-wide hash map -> Function

Dynamic module:
  name -> loaded .so symbol table -> __tvm_ffi_name -> Function wrapper
```

`load_module(..., keep_module_alive=True)`는 기본적으로 로드된 module을 오래 살아 있게 붙잡아둔다. 하지만 이것은 `.so`가 unload되지 않게 하는 목적이지, `name -> Function` 전역 registry에 등록한다는 뜻은 아니다.

## `.so`란 무엇인가

`.so`는 Linux의 shared object, 즉 동적 라이브러리 파일이다. Windows의 `.dll`, macOS의 `.dylib`와 비슷하다.

FlashInfer 문맥에서는 C++/CUDA 코드를 컴파일하고 링크한 결과물이 `.so`가 된다.

```text
C++/CUDA source
  -> nvcc / compiler
  -> library.so
  -> tvm_ffi.load_module(library.so)
  -> module.<function>(...)
```

`.so` 안에는 실제 C++ 코드와 tvm-ffi export 심볼이 들어 있다.

## FlashInfer의 JIT `.so` 경로

FlashInfer의 `bmm_fp8`은 global registry 경로가 아니라 dynamic module 경로를 쓴다.

원본 흐름은 대략 다음과 같다.

```python
@functools.cache
def get_gemm_module():
    module = gen_gemm_module().build_and_load()
    ...
```

`gen_gemm_module()`은 GEMM 관련 `.cu` 파일을 하나의 native module로 만들기 위한 `JitSpec`를 생성한다.

```python
def gen_gemm_module() -> JitSpec:
    return gen_jit_spec(
        "gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "bmm_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_gemm_binding.cu",
        ],
        extra_ldflags=["-lcublas", "-lcublasLt"],
    )
```

각 파일의 역할은 다음과 같다.

- `bmm_fp8.cu`: `bmm_fp8` C++ 런처 구현
- `group_gemm.cu`: 같은 GEMM module에 들어가는 다른 GEMM 구현
- `flashinfer_gemm_binding.cu`: tvm-ffi export 심볼 생성
- `-lcublas`, `-lcublasLt`: cuBLAS/cuBLASLt 링크 옵션

`build_and_load()`를 호출하면 실제로는 다음 일이 일어난다.

```text
gen_gemm_module()
  -> JitSpec 생성
  -> build.ninja 작성
  -> nvcc로 .cu 컴파일
  -> gemm.so 링크
  -> tvm_ffi.load_module(gemm.so)
```

결과적으로 하나의 `gemm.so` 안에 여러 entry function이 들어간다.

```text
gemm.so
  -> __tvm_ffi_bmm_fp8
  -> __tvm_ffi_bmm_fp8_get_algos
  -> __tvm_ffi_bmm_fp8_run_with_algo
  -> __tvm_ffi_cutlass_segment_gemm
```

Python에서 다음 코드를 실행하면:

```python
module.bmm_fp8(a, b, out, scale_a, scale_b, workspace_buffer, cublas_handle)
```

tvm-ffi는 `gemm.so` 안의 `__tvm_ffi_bmm_fp8` 심볼을 찾아 호출한다.

이 함수는 GPU kernel 자체라기보다 C++ 런처에 가깝다. 런처 안에서 shape 검사, dtype dispatch, CUDA stream 설정, raw pointer 추출 등을 한 뒤 실제 cuBLASLt/GPU 연산을 호출한다.

## `bmm_fp8` 흐름

```text
Python bmm_fp8 API
  -> get_gemm_module()
  -> gen_gemm_module().build_and_load()
  -> tvm_ffi.load_module(gemm.so)
  -> module.bmm_fp8(...)
  -> __tvm_ffi_bmm_fp8
  -> C++ bmm_fp8(TensorView A, TensorView B, ...)
  -> cuBLASLt FP8 batched GEMM
```

따라서 `bmm_fp8`은 “전역 registry에서 커널 이름을 찾는 구조”가 아니라, “로드된 `gemm.so` 라이브러리 안에서 tvm-ffi export 심볼을 찾는 구조”이다.

## JIT `.so` 캐시

FlashInfer의 JIT 경로는 빌드 결과를 캐시한다.

```text
.cu / .cuh source
  -> nvcc/ninja
  -> .so
  -> JIT cache
```

대략 다음 위치에 저장된다.

```text
~/.cache/flashinfer/<version>/cached_ops/<hash-or-uri>/
```

여기에는 `build.ninja`, object file, `.so` 같은 빌드 산출물이 들어갈 수 있다.

cache miss이면 컴파일하고, cache hit이면 기존 `.so`를 다시 `load_module`한다.

## FlashInfer의 cubin 경로

FlashInfer에는 `.so` JIT 경로와 별도로 cubin 경로도 있다.

`.cubin`은 CUDA GPU가 실행할 수 있는 사전 빌드 kernel binary다. `.so`가 Python에서 부를 수 있는 C++ 런처 라이브러리라면, `.cubin`은 그 런처가 CUDA Driver API로 로드해서 실행하는 GPU용 바이너리다.

cubin 경로는 다음과 같다.

```text
Python module.<kernel>(...)
  -> tvm-ffi로 .so 안의 C++ 런처 호출
  -> C++ 런처가 getCubin(name, sha256) 호출
  -> Python callback이 cubin bytes를 가져옴
  -> C++에 cubin bytes 전달
  -> cuModuleLoadData
  -> cuModuleGetFunction
  -> cuLaunchKernelEx
```

중요한 점은 cubin 경로에서도 Python에서 C++ 런처를 부르는 첫 단계는 여전히 tvm-ffi라는 것이다. 다만 실제 GPU kernel binary는 `.so` 내부에 들어 있는 것이 아니라 외부 `.cubin`으로 확보된다.

## cubin은 어디서 오는가

FlashInfer가 첫 실행 때 `.cu`를 cubin으로 컴파일해서 저장하는 구조는 아니다.

cubin 전용 커널의 경우, 이미 빌드된 artifact가 원격 저장소 또는 별도 패키지에 있다.

기본 원격 저장소:

```text
https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/
```

FlashInfer repo에는 보통 다음 정보가 있다.

```text
artifact path
sha256 checksum
downloader / loader code
```

실제 `.cubin` 파일은 다음 중 하나에서 온다.

1. `flashinfer-cubin` 패키지 내부
2. `FLASHINFER_CUBIN_DIR` 환경변수가 가리키는 디렉터리
3. 기본 cache 디렉터리
4. 없으면 원격 저장소에서 다운로드

관련 흐름:

```text
get_artifact(file_name, sha256)
  -> FLASHINFER_CUBIN_DIR / file_name 확인
  -> sha256 검증
  -> 없거나 불일치하면 원격에서 다운로드
  -> 다시 읽고 검증
  -> bytes 반환
```

## `flashinfer-cubin` 패키지

`flashinfer-cubin`은 cubin 파일들을 미리 담은 별도 Python 패키지다.

패키지를 빌드할 때:

```text
download_artifacts()
  -> flashinfer-cubin/flashinfer_cubin/cubins/ 에 cubin 다운로드
  -> wheel package data로 포함
```

런타임에 `flashinfer-cubin`이 설치되어 있으면 FlashInfer는 그 패키지 내부의 cubins 폴더를 우선 사용한다. 따라서 많은 경우 런타임 다운로드를 피할 수 있다.

## cubin cache와 JIT cache의 차이

두 cache는 별도다.

```text
JIT .so cache:
  입력: .cu/.cuh source
  miss: nvcc로 컴파일
  결과: .so
  위치: ~/.cache/flashinfer/<version>/cached_ops/...

cubin cache:
  입력: 사전 빌드 .cubin artifact
  miss: 원격 저장소에서 다운로드
  결과: .cubin
  위치: flashinfer-cubin package, FLASHINFER_CUBIN_DIR, 또는 ~/.cache/flashinfer/cubins
```

즉 cubin은 JIT 컴파일 결과물이 아니라, 사전 빌드된 바이너리를 받아 저장한 것이다.

## cubin을 쓰는 이유와 한계

cubin의 장점:

- 첫 실행 때 nvcc 컴파일 비용을 피할 수 있다.
- 벤더/내부 도구로 사전 튜닝된 kernel binary를 그대로 쓸 수 있다.
- 사용자의 런타임 환경에 nvcc가 없어도 쓸 수 있는 경우가 많다.

cubin의 단점:

- 특정 GPU architecture에 묶인다.
- shape, dtype, layout, 옵션 조합이 많으면 artifact 수가 커진다.
- 새 GPU나 새 CUDA 환경에 맞춰 다시 빌드/배포해야 할 수 있다.
- 원격 저장소, checksum, 패키지 크기, 다운로드 실패 등 운영 비용이 있다.
- Python에서 바로 실행되는 것이 아니라 C++ 런처와 launch 설정이 여전히 필요하다.

그래서 FlashInfer는 둘을 섞어 쓴다.

```text
소스가 있고 옵션 유연성이 중요한 커널:
  -> JIT .so 경로

사전 튜닝 binary가 있고 source JIT가 어렵거나 부적합한 커널:
  -> cubin 경로
```

`bmm_fp8`은 JIT `.so` 경로이고, TRTLLM/cuDNN/DeepGEMM 계열 일부 커널은 cubin 경로를 쓴다.

## 정리

```text
tvm-ffi
  = Python/C++ 함수 호출 규격화 계층
  = TVMFFIAny, safe_call, TensorView, Function, Module 제공

global registry 경로
  = TVMFFIFunctionSetGlobal / GetGlobal
  = process-wide name -> Function map

dynamic module 경로
  = tvm_ffi.load_module(.so)
  = loaded .so 안의 __tvm_ffi_<name> 심볼 조회

FlashInfer JIT 경로
  = 여러 .cu를 .so로 컴파일
  = tvm_ffi.load_module(.so)
  = module.bmm_fp8(...) 같은 방식으로 C++ 런처 호출

FlashInfer cubin 경로
  = Python -> C++ 런처 호출은 tvm-ffi
  = 실제 GPU kernel binary는 .cubin으로 확보
  = C++ 런처가 CUDA Driver API로 cubin 로드/실행
```

## 원본에서 더 볼 곳

- `include/tvm/ffi/c_api.h`: C ABI 정의
- `include/tvm/ffi/function.h`: `Function`, `SetGlobal`, `TVM_FFI_DLL_EXPORT_TYPED_FUNC`
- `include/tvm/ffi/function_details.h`: unpack 관련 템플릿
- `python/tvm_ffi/cython/function.pxi`: Python `Function.__call__`
- `python/tvm_ffi/module.py`: `load_module`, `Module`
- `python/tvm_ffi/registry.py`: `register_global_func`, `get_global_func`
- `src/ffi/function.cc`: `TVMFFIFunctionSetGlobal`, `TVMFFIFunctionCall`
- `src/ffi/extra/library_module.cc`: `.so` module의 symbol lookup
- `flashinfer/jit/core.py`: `JitSpec.build_and_load`
- `flashinfer/jit/gemm/core.py`: `gen_gemm_module`
- `flashinfer/gemm/gemm_base.py`: `get_gemm_module`, `bmm_fp8`
- `flashinfer/jit/cubin_loader.py`: cubin download/cache/callback
- `flashinfer/artifacts.py`: cubin artifact path/checksum catalog
- `flashinfer-cubin/build_backend.py`: cubin bundled wheel 생성
