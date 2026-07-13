# 원본: flashinfer/jit/gemm/core.py
# 역할: gemm native module에 포함할 소스 파일을 정한다.


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
