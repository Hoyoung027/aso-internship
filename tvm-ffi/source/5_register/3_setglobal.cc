// ============================================================================
// [수렴점] TVMFFIFunctionSetGlobal — 모든 등록 경로의 C ABI 관문
// 발췌 출처: src/ffi/function.cc:161
//
// 입구 A(파이썬) / B(GlobalDef) / C(load_module) 가 전부 여기로 들어온다.
// 한 일: 이름을 '복사'하고 함수를 '참조카운트+1' 한 뒤, 전역표 Update 호출.
//
// ※ 줄별 초상세 주석판은 ../4_cpp_side/setglobal_상세주석.cc 참고.
// ============================================================================

int TVMFFIFunctionSetGlobal(const TVMFFIByteArray* name,  // 이름 {data, size}. 호출 후 사라질 수 있음
                            TVMFFIObjectHandle f,          // 함수(불투명 핸들, 실제로는 FunctionObj*)
                            int override) {                // 이미 있으면 덮어쓸지 (0/1)
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();                  // try { ... 예외를 -1 로 변환

  // [1] 이름 복사: 원본이 사라져도 안전하도록 자체 String 생성
  String name_str(name->data, name->size);

  // [2]+[3] 함수 핸들 복원(+refcount) 후 전역표에 등록
  //   GetRef<Function>(static_cast<FunctionObj*>(f)) : 핸들 → Function, refcount +1
  //     (전역표가 이 함수를 붙잡는다는 표시. 안 올리면 파이썬이 손 뗄 때 삭제 사고)
  //   Global()->Update(...) : 프로세스 하나뿐인 전역표에 "이름→함수" 등록
  GlobalFunctionTable::Global()->Update(
      name_str,
      GetRef<Function>(static_cast<FunctionObj*>(f)),
      override != 0);                          // int → bool

  TVM_FFI_SAFE_CALL_END();                    // } catch { TLS에 에러 저장; return -1; } return 0;
}

// 다음 단계: GlobalFunctionTable::Update / table_  → 4_global_function_table.cc
