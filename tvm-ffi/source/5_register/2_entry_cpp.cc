// ============================================================================
// [입구 B, C] C++ 에서 전역표에 등록하기 (두 가지 방식)
// 발췌 출처: include/tvm/ffi/reflection/registry.h  +  include/tvm/ffi/function.h
//
// 파이썬(입구 A)과 마찬가지로, 결국 TVMFFIFunctionSetGlobal 로 수렴한다.
// ============================================================================


// ============================================================================
// [입구 B] 정적 등록: GlobalDef().def   — C++ 코어/확장에서 가장 흔함
// 발췌 출처: include/tvm/ffi/reflection/registry.h:518
// ============================================================================

// 사용 예 (라이브러리 로드 시 자동 실행):
//   TVM_FFI_STATIC_INIT_BLOCK() {
//     namespace refl = tvm::ffi::reflection;
//     refl::GlobalDef().def("my_ext.add_one", AddOne, "Add one to the input");
//   }
//
//   TVM_FFI_STATIC_INIT_BLOCK = "이 블록을 라이브러리 초기화 때 실행하라"

class GlobalDef {
 public:
  template <typename Func, typename... Extra>
  GlobalDef& def(const char* name, Func&& func, Extra&&... extra) {
    // (1) 임의의 C++ 함수/람다 func 를 FFI Function 으로 변환 (FromTyped)
    // (2) RegisterFunc 내부가 결국 TVMFFIFunctionSetGlobal 호출 → 전역표 등록
    RegisterFunc(name, ffi::Function::FromTyped(std::forward<Func>(func), std::string(name)),
                 std::forward<Extra>(extra)...);
    return *this;   // 체이닝 가능: .def(...).def(...)
  }
  // ... def_packed / def_method 등 변형도 있으나 모두 SetGlobal 로 수렴 ...
};


// ============================================================================
// [입구 C] export 매크로: TVM_FFI_DLL_EXPORT_TYPED_FUNC  — .so 배포용
// 발췌 출처: include/tvm/ffi/function.h:950
// ============================================================================

// 사용 예:
//   static int AddTwo(int x) { return x + 2; }
//   TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_two, AddTwo);
//
// ★ 주의: 이 매크로는 '즉시' SetGlobal 을 부르지 않는다. (지연 등록)
//   - 컴파일 시: __tvm_ffi_add_two 라는 C 심볼(=safe_call)만 생성해 .so 에 넣음
//   - 로드 시:   tvm_ffi.load_module(".so") 가 __tvm_ffi_ 접두사 심볼들을 스캔해서
//               그때 전역표(모듈 네임스페이스)에 등록함
//
//   즉 입구 A/B 는 "코드 실행 중 등록", 입구 C 는 "load_module 시점 등록".


// ============================================================================
// 세 입구의 공통 수렴점
//
//   [A] register_global_func (Python)  ─┐
//   [B] GlobalDef().def (C++ 정적)      ─┼─►  TVMFFIFunctionSetGlobal(...)  [3_setglobal.cc]
//   [C] EXPORT 매크로 (+ load_module)   ─┘         │
//                                                  ▼
//                                    GlobalFunctionTable::Update  [3_table 구현]
//                                                  │
//                                                  ▼
//                                    table_.Set("name", Entry(func))
// ============================================================================
