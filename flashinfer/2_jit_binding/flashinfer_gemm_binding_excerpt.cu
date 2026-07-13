// 원본: csrc/flashinfer_gemm_binding.cu
// 역할: C++ 함수를 tvm-ffi로 export한다.

#include "tvm_ffi_utils.h"

void bmm_fp8(TensorView A, TensorView B, TensorView D, TensorView A_scale, TensorView B_scale,
             TensorView workspace_buffer, int64_t cublas_handle);

int64_t bmm_fp8_get_algos(TensorView A, TensorView B, TensorView D, TensorView A_scale,
                          TensorView B_scale, TensorView workspace_buffer, int64_t cublas_handle,
                          TensorView algo_buffer);

void bmm_fp8_run_with_algo(TensorView A, TensorView B, TensorView D, TensorView A_scale,
                           TensorView B_scale, TensorView workspace_buffer, int64_t cublas_handle,
                           TensorView algo_buffer, int64_t algo_idx);

void CutlassSegmentGEMM(TensorView workspace_buffer, TensorView all_problems, TensorView x_ptr,
                        TensorView w_ptr, TensorView y_ptr, TensorView x_ld, TensorView w_ld,
                        TensorView y_ld, TensorView empty_x_data, bool weight_column_major);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cutlass_segment_gemm, CutlassSegmentGEMM);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_fp8, bmm_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_fp8_get_algos, bmm_fp8_get_algos);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_fp8_run_with_algo, bmm_fp8_run_with_algo);
