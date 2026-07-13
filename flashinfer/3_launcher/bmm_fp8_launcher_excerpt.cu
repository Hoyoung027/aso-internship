// 원본: csrc/bmm_fp8.cu
// 역할: tvm-ffi에서 넘어온 TensorView를 받아 실제 FP8 BMM을 호출한다.

#include <flashinfer/gemm/bmm_fp8.cuh>

#include "tvm_ffi_utils.h"

void bmm_fp8(TensorView A, TensorView B, TensorView D, TensorView A_scale, TensorView B_scale,
             TensorView workspace_buffer, int64_t cublas_handle) {
  CHECK_CUDA(A);
  CHECK_CUDA(B);
  CHECK_CUDA(D);
  CHECK_DIM(3, A);
  CHECK_DIM(3, B);
  CHECK_DIM(3, D);
  TVM_FFI_ICHECK(A.size(0) == B.size(0) && A.size(0) == D.size(0)) << "Batch sizes must match";
  TVM_FFI_ICHECK(A.size(2) == B.size(1)) << "Incompatible matrix sizes";

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(B.dtype(), b_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(A.dtype(), a_type, [&] {
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(D.dtype(), d_type, [&] {
        auto batch_size = A.size(0);
        auto m = A.size(1);
        auto k = A.size(2);
        auto n = B.size(2);

        auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
        ffi::CUDADeviceGuard device_guard(A.device().device_id);
        auto stream = get_stream(A.device());

        auto status = flashinfer::bmm_fp8::bmm_fp8_internal_cublaslt(
            workspace_buffer.data_ptr(), workspace_buffer.numel() * get_element_size(workspace_buffer),
            static_cast<b_type*>(B.data_ptr()), static_cast<a_type*>(A.data_ptr()),
            static_cast<d_type*>(D.data_ptr()), batch_size, n, m, k,
            static_cast<float*>(B_scale.data_ptr()), static_cast<float*>(A_scale.data_ptr()),
            lt_handle, stream);

        TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS)
            << "bmm_fp8_internal_cublaslt failed: " << cublasGetStatusString(status);
        return true;
      });
    });
  });
}
