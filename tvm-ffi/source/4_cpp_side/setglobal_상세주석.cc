// ============================================================================
// TVMFFIFunctionSetGlobal — "전역 레지스트리 등록의 C ABI 관문" (상세 주석판)
// 발췌 출처: src/ffi/function.cc:161
//
// 임무: 이름 문자열 + 함수 핸들을 받아, 프로세스에 하나뿐인 전역 해시맵에
//       "이름 -> 함수" 를 안전하게 집어넣는다.
//
// 파이썬 register_global_func / C++ GlobalDef().def / .so 로드 등
// 모든 등록 경로가 최종적으로 이 함수로 수렴한다.
//
// "안전하게"의 두 축:
//   (1) 수명 안전 : 입력이 호출 후 사라져도 표에 남은 값은 유효해야 함
//                   -> 이름은 '복사', 함수는 '참조 카운트 +1'
//   (2) 에러 안전 : 도중 문제(중복 등록 등)가 나도 크래시 대신 에러코드 반환
//                   -> SAFE_CALL 매크로가 예외를 -1 로 변환
// ============================================================================

int TVMFFIFunctionSetGlobal(
    const TVMFFIByteArray* name,   // 등록할 '이름'  = {data 포인터, size}. \0 종료 보장 없음.
                                   //   ex) data="my.add", size=6
                                   //   ※ 이 메모리는 호출자(파이썬 바인딩) 소유 → 곧 사라질 수 있음
    TVMFFIObjectHandle f,          // 등록할 '함수'  = 타입 지워진 불투명 핸들
                                   //   ※ 실제로는 FunctionObj* (파이썬 함수를 감싼 FFI 함수 객체)
                                   //   ※ 지금 refcount 최소 1 (파이썬 쪽이 붙잡는 중)
    int override) {                // 같은 이름이 이미 있을 때 덮어쓸지 (0=금지, 1=허용)
                                   // 반환 int = 에러코드 (0=성공, -1=실패)

  using namespace tvm::ffi;        // 아래에서 tvm::ffi::String 등을 접두사 없이 쓰기 위함 (동작 무관)

  TVM_FFI_SAFE_CALL_BEGIN();       // ── 매크로 = "try {" ──
                                   //   여기부터 END 까지를 try 로 감쌈.
                                   //   아래에서 C++ 예외가 나면 언어 경계 밖으로 새지 않게 잡으려고.
                                   //   (C++ 예외는 C/파이썬으로 전파 불가 → 안 잡으면 크래시)

  // [1] 이름 '복사' : 원본(name->data)이 사라져도 안전하도록 자체 String 을 새로 만듦.
  //     전역표에 오래 보관할 값이므로 반드시 소유 복사본이 필요.
  String name_str(name->data, name->size);        // ex) "my.add" 를 담은 독립 문자열

  // [2] 함수 핸들 복원 + 참조 카운트 +1
  //     static_cast<FunctionObj*>(f) : 불투명 핸들 f 를 "사실 FunctionObj 다" 로 정체 복원
  //     GetRef<Function>(...)        : 참조카운트 관리 Function 래퍼로 감싸며 refcount +1
  //       -> "전역표도 이 함수를 쓴다"를 카운트로 표시.
  //          이래야 나중에 파이썬 참조가 사라져도 표가 붙잡고 있는 한 함수가 안 지워짐.
  //          (등록 전 refcount 1 → GetRef 후 2)
  //
  // [3] 전역표에 저장
  //     GlobalFunctionTable::Global() : 프로세스에 하나뿐인 전역표(싱글턴).
  //                                     SetGlobal/GetGlobal 이 공유하는 바로 그 표.
  //     Update(name, func, can_override) 내부(function.cc:85)에서:
  //       - 같은 이름이 이미 있고 override 아니면 → TVM_FFI_THROW (위 try 가 잡음)
  //       - 통과하면 → table_.Set(name, Entry(func))  // 해시맵에 "이름→함수(봉투)" 저장
  GlobalFunctionTable::Global()->Update(name_str,
                                        GetRef<Function>(static_cast<FunctionObj*>(f)),
                                        override != 0);   // int → bool 변환

  TVM_FFI_SAFE_CALL_END();         // ── 매크로 = "} catch(...) { 에러를 TLS 에 저장; return -1; } return 0; " ──
                                   //   예외 없었으면 → return 0 (성공)
                                   //   예외 났으면   → 잡아서 에러정보 TLS 저장 후 return -1 (실패)
                                   //   호출한 파이썬(CHECK_CALL)이 -1 보면 파이썬 예외로 변환해 던짐
}

// ============================================================================
// 실행 후 최종 상태 (register_global_func("my.add", add_fn) 예시)
//
//   전역표 table_:
//      "my.add" → Entry{ name:"my.add", func_data: add_fn }
//   add_fn refcount: 2   (파이썬 1 + 전역표 1)
//   반환값: 0 (성공)
//
//   이제 어디서든(C++ 포함) GetGlobal("my.add") 로 찾아 부를 수 있다.
//
// 짝 함수:
//   이 SetGlobal          = 등록 (표에 넣기)
//   TVMFFIFunctionGetGlobal = 조회 (표에서 이름으로 꺼내기)
//   둘이 같은 전역표를 공유 → C++이 넣은 걸 파이썬이 찾아 부를 수 있는 근거.
// ============================================================================
