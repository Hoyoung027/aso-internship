// ============================================================================
// "safe_call 의 정체" 편
// 발췌 출처: include/tvm/ffi/function.h  +  include/tvm/ffi/function_details.h
//
// function_cc_excerpt.cc 의 TVMFFIFunctionCall 은 "func->safe_call(...) 을 부른다"
// 까지만 보여줬다. 그 safe_call 이 실제로 뭔지가 여기 있다.
// safe_call = 등록 매크로가 자동 생성한 __tvm_ffi_<이름> 함수이고,
//             그 안에서 unpack_call 이 TVMFFIAny[] 를 실제 C++ 인자로 푼다.
//
// 이게 파이썬 쪽 마샬링(2_python_side 의 SetArgument: 파이썬값 -> TVMFFIAny)의
// 정반대 방향(TVMFFIAny -> C++ 타입값)이다. 포장과 개봉의 대칭.
// ============================================================================


// ============================================================================
// [1] 등록 매크로가 자동 생성하는 safe_call 껍데기
// 발췌 출처: include/tvm/ffi/function.h:950  (TVM_FFI_DLL_EXPORT_TYPED_FUNC_IMPL_)
//
//   1_example/add_one_cpu.cc 에서 쓴
//       TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, AddOne)
//   이 컴파일 타임에 아래 함수로 펼쳐진다. (bmm_fp8 도 동일)
// ============================================================================

extern "C" {
// 심볼 이름 = __tvm_ffi_<ExportName>.  load_module 이 이 접두사로 함수를 찾는다.
// 시그니처가 정확히 TVMFFISafeCallType 과 같다 → 이게 곧 safe_call.
TVM_FFI_DLL_EXPORT int __tvm_ffi_add_one_cpu(void* self, const TVMFFIAny* args,
                                             int32_t num_args, TVMFFIAny* result) {
  TVM_FFI_SAFE_CALL_BEGIN();                              // 예외 -> 에러코드 감싸개
  // decltype(AddOne) 로 컴파일러가 원래 시그니처(TensorView, TensorView)를 자동 분석
  using FuncInfo = ::tvm::ffi::details::FunctionInfo<decltype(AddOne)>;
  static std::string name = "add_one_cpu";
  // ★ 여기서 언패킹 → 진짜 커널(AddOne) 호출 ★
  ::tvm::ffi::details::unpack_call<typename FuncInfo::RetType>(
      std::make_index_sequence<FuncInfo::num_args>{}, &name, AddOne,
      reinterpret_cast<const ::tvm::ffi::AnyView*>(args), num_args,
      reinterpret_cast<::tvm::ffi::Any*>(result));
  TVM_FFI_SAFE_CALL_END();
}
}


// ============================================================================
// [2] 실제 언패킹: TVMFFIAny[] -> 타입 있는 C++ 인자
// 발췌 출처: include/tvm/ffi/function_details.h:206  (unpack_call)
// ============================================================================

// ┌─ 이 함수 한 줄 요약 ─────────────────────────────────────────────────┐
// │ TVMFFIAny 배열(args)을, 커널 f 가 원하는 "정확한 타입들"로 하나씩       │
// │ 변환해서 f 를 호출한다. 그것도 반복문 없이, 컴파일 타임에 코드를 펼쳐서.│
// └──────────────────────────────────────────────────────────────────────┘
//
// 왜 반복문(for)을 못 쓰나?
//   인자마다 타입이 다르다. bmm_fp8 은 (TensorView x6, int64_t x1).
//   "각 인자를 각자의 타입으로 꺼내기"는 타입이 컴파일 타임 정보라서
//   런타임 for 루프로는 불가능하다. 그래서 템플릿(파라미터 팩)으로 푼다.

template <typename R,        // R  = 커널의 반환 타입    (AddOne 은 void)
          typename F,        // F  = 커널의 타입         (AddOne 의 함수 타입)
          size_t... Is>      // Is = 인덱스 묶음 0,1,... (인자 개수만큼. AddOne 은 0,1)
TVM_FFI_INLINE void unpack_call(std::index_sequence<Is...>,  // 위 Is... 를 실어오는 통로
                                const std::string* optional_name,  // 에러메시지용 함수 이름
                                const F& f,               // ★ 진짜 커널 (AddOne / bmm_fp8) ★
                                const AnyView* args,      // ★ 넘어온 인자들 = TVMFFIAny 배열 ★
                                int32_t num_args,         // 실제로 넘어온 인자 개수
                                Any* rv) {                // 결과를 써넣을 곳 (return value)

  using FuncInfo = FunctionInfo<F>;                 // 컴파일러가 f 의 시그니처를 분석해주는 도구
  using PackedArgs = typename FuncInfo::ArgType;    // = 인자 타입들의 튜플.
  //   AddOne(TensorView, TensorView)  ->  PackedArgs = std::tuple<TensorView, TensorView>
  //   즉 std::tuple_element_t<0, PackedArgs> == TensorView  (0번 인자 타입)
  //      std::tuple_element_t<1, PackedArgs> == TensorView  (1번 인자 타입)
  //   => "몇 번째 인자가 무슨 타입인지"를 컴파일 타임에 알 수 있다.

  // (1) 인자 개수 검증 -------------------------------------------------------
  //     sizeof...(Is) = 묶음 Is... 의 원소 개수 = 커널이 원하는 인자 수 (AddOne 은 2).
  //     constexpr = 컴파일 타임 상수. 파이썬이 실제로 넘긴 num_args 와 다르면 에러.
  //     예: mod.add_one_cpu(x)  처럼 인자 1개만 주면 여기 걸린다.
  constexpr size_t nargs = sizeof...(Is);
  if (nargs != num_args) {
    TVM_FFI_THROW(TypeError) << "Mismatched number of arguments ... Expected " << nargs
                             << " but got " << num_args << " arguments";
  }

  // (2) ★핵심★ 인자를 타입 맞춰 꺼내서 커널 f 호출 ---------------------------
  //     if constexpr = 컴파일 타임 분기. 둘 중 한 쪽만 실제 코드로 남는다.
  //     (런타임 if 아님. R 이 void 인지 아닌지는 컴파일 시점에 정해지므로.)
  if constexpr (std::is_same_v<R, void>) {
    // 반환값 없는 커널 (AddOne, bmm_fp8) → 그냥 호출
    f(ArgValueWithContext<std::tuple_element_t<Is, PackedArgs>>{args, Is, optional_name, f_sig}...);
    //                    └── Is 번째 인자의 타입 ──┘         └ args, 인덱스 Is 를 기억한 래퍼 ┘ └↑팩 펼침
  } else {
    // 반환값 있는 커널 → 결과를 R 로 만들어 rv(Any) 에 담아 되돌림 (→ 나중에 파이썬으로)
    *rv = R(f(ArgValueWithContext<std::tuple_element_t<Is, PackedArgs>>{args, Is, ...}...));
  }
}

// ─────────────────────────────────────────────────────────────────────────
// (2) 의 그 한 줄이 실제로 어떻게 동작하는가 — 3단계로 쪼개서 보기
// ─────────────────────────────────────────────────────────────────────────
//
// 맨 끝의 ...  이 "Is... = 0,1 에 대해 안쪽 표현식을 복제"한다 (파라미터 팩 확장).
//
// [단계 A] 컴파일러가 Is=0, Is=1 을 넣어 코드를 펼친다:
//
//     f(
//        ArgValueWithContext<TensorView>{args, 0, name, sig},   // 0번 인자용 "래퍼"
//        ArgValueWithContext<TensorView>{args, 1, name, sig}    // 1번 인자용 "래퍼"
//     );
//
//   ※ 주의: 이 시점엔 아직 변환이 "안" 일어났다.
//     ArgValueWithContext 라는 래퍼 객체만 2개 만든 상태.
//     각 래퍼는 "나는 args[0] 을 TensorView 로 바꿔줄 애"라는 정보(인덱스+목표타입)만 들고 있다.
//     (bmm_fp8 이면 인자 7개 -> 래퍼 7개로 펼쳐진다.)
//
// [단계 B] f(...) 가 각 래퍼를 커널의 실제 파라미터(TensorView x, y)에 "끼워 넣는" 순간,
//   타입이 안 맞으므로(래퍼 != TensorView) 컴파일러가 ArgValueWithContext 에 정의된
//   "암묵적 형변환 연산자 operator TensorView()" 를 자동 호출한다. 그 안에서 진짜 변환:
//
//     // ArgValueWithContext (원본 function_details.h:176) 요약
//     operator TensorView() {                                  // 래퍼 -> TensorView 요청 시 실행
//         std::optional<TensorView> opt =
//             args[arg_index].try_cast<TensorView>();           // ★★ 진짜 변환 = 여기 ★★
//         if (!opt) TVM_FFI_THROW(TypeError)                    // 타입 안 맞으면 에러
//             << "Mismatched type on argument #" << arg_index;  //  예: 정수를 TensorView 로?
//         return *opt;
//     }
//
//   try_cast<T>() = TVMFFIAny 하나의 type_index 를 확인하고, 맞으면 그 타입 T 로 꺼낸다.
//   => 파이썬 쪽 SetArgument(파이썬값 -> TVMFFIAny 로 "포장")의 정반대("개봉")가 바로 이것.
//
// [단계 C] 변환이 끝나면 결국 이렇게 실행된 셈:
//
//     AddOne(
//        args[0].try_cast<TensorView>(),   // TVMFFIAny -> TensorView
//        args[1].try_cast<TensorView>()    // TVMFFIAny -> TensorView
//     );
//     // 드디어 타입 있는 진짜 커널 실행 → y = x + 1


// ============================================================================
// 전체 마샬링 대칭 (이걸로 파이썬<->C++ 전 경로 완성)
//
//   파이썬값  ──[SetArgument]──►  TVMFFIAny[]  ──[unpack_call]──►  C++ 타입 인자
//   (2_python_side)                (3_c_abi)                       (여기, 커널 직전)
//      포장 pack                     만능 상자                        개봉 unpack
//
//   호출 경로:
//     TVMFFIFunctionCall            [function_cc_excerpt.cc]
//        -> func->safe_call         = __tvm_ffi_add_one_cpu  [위 [1]]
//             -> unpack_call        [위 [2]]
//                  -> AddOne(TensorView x, y)   ← 진짜 커널, 드디어 타입 있는 인자로 실행
// ============================================================================
