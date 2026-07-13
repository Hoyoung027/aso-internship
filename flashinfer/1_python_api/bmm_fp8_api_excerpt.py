# 원본: flashinfer/gemm/gemm_base.py
# 역할: Python API가 native module을 얻고 bmm_fp8 함수를 호출하는 지점.


@functools.cache
def get_gemm_module():
    module = gen_gemm_module().build_and_load()

    class CublasFp8GemmRunner:
        def forward(self, a, b, out, scale_a, scale_b, workspace_buffer, cublas_handle):
            module.bmm_fp8(
                a,
                b,
                out,
                scale_a,
                scale_b,
                workspace_buffer,
                cublas_handle,
            )

    ...


@backend_requirement(...)
@flashinfer_api(trace=bmm_fp8_trace)
def bmm_fp8(A, B, ...):
    out = ...
    fp8_gemm_sm100(...)
    return out
