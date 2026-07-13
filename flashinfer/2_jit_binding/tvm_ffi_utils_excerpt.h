// 원본: csrc/tvm_ffi_utils.h
// 역할: FlashInfer binding 코드에서 쓰는 tvm-ffi 타입과 유틸을 모은 헤더.

#pragma once

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

#include "dlpack/dlpack.h"

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
  return (dtype.code << 16) | (dtype.bits << 8) | dtype.lanes;
}

constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType dl_float8_e4m3fn = DLDataType{kDLFloat8_e4m3fn, 8, 1};
constexpr DLDataType dl_float8_e5m2 = DLDataType{kDLFloat8_e5m2, 8, 1};
constexpr DLDataType dl_float4_e2m1fn = DLDataType{kDLFloat4_e2m1fn, 4, 1};

constexpr int64_t float8_e4m3fn_code = encode_dlpack_dtype(dl_float8_e4m3fn);

// 원본에는 dtype dispatch 매크로들이 이어진다.
