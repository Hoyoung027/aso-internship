# 5_cubin

사전 빌드된 `.cubin` 커널을 다운로드, 캐시, 로드, 실행하는 흐름을 모은 발췌입니다.

```text
Python module.<kernel>(...)
  -> tvm-ffi로 C++ 런처 호출
  -> C++ getCubin(...)
  -> Python callback이 cubin bytes 반환
  -> cuModuleLoadData
  -> cuModuleGetFunction
  -> cuLaunchKernelEx
```

파일:

- `1_artifacts_catalog.py`: cubin 원격 경로와 checksum
- `2_env_cubin_dir.py`: cubin을 찾을 로컬 디렉터리 결정
- `3_cubin_loader_py.py`: cubin 로드/다운로드와 ctypes callback 등록
- `4_cubin_loader_h.h`: C++ 쪽 callback 저장/호출
- `5_launch_from_cubin.cu`: CUDA Driver API로 cubin 실행
- `6_flashinfer_cubin_build.py`: cubin을 wheel에 미리 포함하는 빌드 훅
