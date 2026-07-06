// ============================================================================
// 발췌 출처: python/tvm_ffi/cython/tvm_ffi_python_helpers.h
// __call__ 이 부른 TVMFFIPyFuncCall 의 실체. thread-local 매니저로 위임한다.
// 이 매니저가 (a) 파이썬 인자 -> TVMFFIAny 배열 변환, (b) TVMFFIFunctionCall 호출,
// (c) 스트림/디바이스 컨텍스트 준비 를 담당한다.
// ============================================================================

// ---- [원본 helpers.h:857] --------------------------------------------------
TVM_FFI_INLINE int TVMFFIPyFuncCall(void* func_handle, PyObject* py_arg_tuple, TVMFFIAny* result,
                                    int* c_api_ret_code, bool release_gil = true,
                                    const DLPackExchangeAPI** out_ctx_dlpack_api = nullptr) {
  return TVMFFIPyCallManager::ThreadLocal()->FuncCall(
      func_handle, py_arg_tuple, result, c_api_ret_code, release_gil, out_ctx_dlpack_api);
}

// ---- [원본 helpers.h:936] 파이썬 객체 하나를 FFI Any 로 변환하는 진입점 ------
//   FuncCall 내부에서 각 인자마다 이런 변환을 수행한다.
TVM_FFI_INLINE int TVMFFIPyPyObjectToFFIAny(PyObject* py_arg, TVMFFIAny* out, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->PyObjectToFFIAny(py_arg, out, c_api_ret_code);
}

// 흐름 요약:
//   FuncCall(...)  ->  각 py 인자를 PyObjectToFFIAny 로 TVMFFIAny[] 채움
//                  ->  TVMFFIFunctionCall(func_handle, args, num_args, result)  [C ABI 경계]
//                  ->  결과 TVMFFIAny 를 그대로 result 로 반환
