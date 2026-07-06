"""
run.py - 커스텀 CUDA 커널(gemm_tiled.cubin)을 TVM FFI로 Python에서 호출

[1단계] launcher.cpp 를 즉석 컴파일 + cubin 임베드 → 모듈 로드
[2단계] Python에서 mod.gemm(A, B, C) 로 호출 → torch 결과와 비교

실행 전:  nvcc --cubin -arch=sm_86 gemm_tiled.cu -o gemm_tiled.cubin
실행:     ../.venv/bin/python run.py
"""

import os

# launcher 컴파일에 CUDA 12.x nvcc 가 필요
#    시스템 기본 /usr/bin/nvcc은 11.5 라 실패 → 12.4 를 PATH 앞에 강제로 세팅
CUDA_HOME = "/usr/local/cuda-12.4"
os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["PATH"] = f"{CUDA_HOME}/bin:" + os.environ.get("PATH", "")

import torch
from tvm_ffi import cpp

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    assert torch.cuda.is_available(), "CUDA GPU가 필요합니다"

    # launcher.cpp 읽기
    with open(os.path.join(HERE, "launcher.cpp")) as f:
        launcher_cpp = f.read()

    # 미리 컴파일해둔 cubin 읽기
    cubin_path = os.path.join(HERE, "gemm_tiled.cubin")

    assert os.path.exists(cubin_path), (
        "gemm_tiled.cubin 이 없습니다. 먼저: "
        "nvcc --cubin -arch=sm_86 gemm_tiled.cu -o gemm_tiled.cubin"
    )

    with open(cubin_path, "rb") as f:
        cubin_bytes = f.read()

    print(f"[load] gemm_tiled.cubin ({len(cubin_bytes)} bytes)")


    # [1단계] 런처 C++ 즉석 컴파일 + cubin 임베드 → 모듈 로드
    #   embed_cubin 의 키 "gemm_tiled_cubin" 은 launcher.cpp 의
    #   TVM_FFI_EMBED_CUBIN(gemm_tiled_cubin) 이름과 정확히 같아야 한다.
    #   ⚠️ cuLibraryLoadData 는 드라이버 API(libcuda) 심볼이라 -lcuda 가 필요.
    #      링크는 stubs/libcuda.so 를 보고, 런타임엔 실제 드라이버로 해석

    mod = cpp.load_inline(
        "gemm_mod",
        cuda_sources=launcher_cpp,
        embed_cubin={"gemm_tiled_cubin": cubin_bytes},
        extra_ldflags=["-lcudart", "-lcuda", "-L/usr/local/cuda-12.4/lib64/stubs"],
    )
    print("[build] load_inline 완료 → mod.gemm 사용 가능")

    # [2단계] Python에서 호출
    M, N, K = 512, 512, 512
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    mod.gemm(A, B, C)            # 내 CUDA 커널 실행
    torch.cuda.synchronize()

    # torch 정답과 비교 검증
    ref = A @ B
    max_err = (C - ref).abs().max().item()
    ok = torch.allclose(C, ref, atol=1e-2, rtol=1e-2)
    print(f"[verify] max abs error = {max_err:.3e}  -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
