// ============================================================================
// 발췌 출처: src/ffi/function.cc
// C ABI 관문(3_)의 C++ 쪽 실제 구현. 파이썬이 넘어온 뒤 실제로 실행되는 코드.
// ============================================================================

// ---- [원본 function.cc:191-205] 경계를 넘어온 호출의 실제 처리 -------------
//   파이썬이 부른 TVMFFIFunctionCall 이 최종 도달하는 구현.
//   하는 일은 단 하나: func 객체 안에 저장돼 있는 safe_call 함수 포인터를 호출.
//   즉 "함수 핸들 -> 그 안의 safe_call -> 진짜 사용자 함수(AddOne)" 로 이어진다.
int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
                       TVMFFIAny* result) {
  using namespace tvm::ffi;
#ifdef _MSC_VER
  volatile int ret = reinterpret_cast<FunctionObj*>(func)->safe_call(func, args, num_args, result);
  return ret;
#else
  // NOTE: this is a tail call
  return reinterpret_cast<FunctionObj*>(func)->safe_call(func, args, num_args, result);
  //     └ func 를 FunctionObj 로 보고 그 안의 safe_call 을 그대로 호출 ┘
  //       safe_call 이 args(TVMFFIAny[]) 를 풀어서 실제 C++ 함수 AddOne 을 부른다.
#endif
}


// ---- [원본 function.cc:177-190] 이름으로 함수 찾기 (파이썬의 GetGlobal 구현) --
int TVMFFIFunctionGetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle* out) {
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();
  String name_str(name->data, name->size);
  const GlobalFunctionTable::Entry* fp = GlobalFunctionTable::Global()->Get(name_str);  // 전역표 조회
  if (fp != nullptr) {
    tvm::ffi::Function func(fp->func_data);
    *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(func));
  } else {
    *out = nullptr;
  }
  TVM_FFI_SAFE_CALL_END();
}


// ---- [원본 function.cc:161-168] 전역 레지스트리에 등록 (SetGlobal 구현) -----
int TVMFFIFunctionSetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle f, int override) {
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();
  String name_str(name->data, name->size);
  GlobalFunctionTable::Global()->Update(name_str, GetRef<Function>(static_cast<FunctionObj*>(f)),
                                        override != 0);   // 이름->함수 를 전역표에 넣음
  TVM_FFI_SAFE_CALL_END();
}


// ---- [원본 function.cc:146-153] C 콜백을 FFI Function 으로 감싸기 -----------
//   외부(파이썬/JIT/다른언어)에서 만든 safe_call 함수 포인터를 받아
//   FFI Function 객체로 포장한다. add_one_cpu.so 를 로드할 때 각 export 함수가
//   이 경로로 Function 객체가 된다.
int TVMFFIFunctionCreate(void* self, TVMFFISafeCallType safe_call, void (*deleter)(void* self),
                         TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::Function func = tvm::ffi::Function::FromExternC(self, safe_call, deleter);
  *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(func));
  TVM_FFI_SAFE_CALL_END();
}
