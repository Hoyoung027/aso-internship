# tvm-ffi 흐름 학습 노트 — "파이썬 ↔ C 는 어떻게 연결되나"

> 핵심: tvm-ffi 는 파이썬을 C로 변환(컴파일)하는 게 아니라, 이미 컴파일된 C/C++ 함수와
> 파이썬 런타임을 **연결(통역)** 한다. 이 폴더는 그 연결의 실체를 코드로 따라가는 자료다.

원본 저장소에서 관련 부분만 발췌/복사했다. 각 파일 상단에 원본 경로와 줄 번호를 적어뒀다.
(발췌본은 핵심만 추린 학습용 뷰. 정확한 전체 구현은 각 파일에 적힌 원본 경로 참고.)

## 폴더 구성

```
1_example/       ← 실제로 실행하는 코드 (여기서 출발)           [호출]
2_python_side/   ← 파이썬이 인자를 포장(pack)해 C 경계로 넘김     [호출]
3_c_abi/         ← 두 세계가 공유하는 "통역 규약"(순수 C, 심장부) [호출]
4_cpp_side/      ← 경계를 넘어온 뒤 실행되는 C++ 구현 + 개봉(unpack) [호출]
5_register/      ← 함수가 전역표에 '등록'되는 과정              [등록]
cubin_발표대본.md ← 컴파일→등록→호출 3단계 발표 스크립트          [발표]
```

앞 4개(1~4)는 **"이미 등록된 함수를 호출하는 흐름"**, `5_register`는 그 앞단 **"함수를
전역표에 등록하는 흐름"** 이다. 등록(5) → 조회 → 호출(1~4) 순으로 이어진다.

---

## 흐름 A: 호출 (mod.add_one_cpu(x, y) 한 줄의 여정)

```
[1_example/load_numpy.py]  mod.add_one_cpu(x, y)
        │  파이썬 호출
        ▼
[2_python_side] Function.__call__               (function.pxi:941)
        │  파이썬 인자 x,y  ──►  TVMFFIAny args[] 로 마샬링(pack)
        │  (numpy/torch 텐서는 DLPack 으로 포인터만 전달, 복사 없음)
        ▼
[2_python_side] TVMFFIPyFuncCall → FuncCall      (helpers.h:857 / :476)
        │  인자 변환 + 스트림 컨텍스트 준비 후 C ABI 호출
        ▼
━━━━━━━━━━━━ 언어 경계 (파이썬 → C) ━━━━━━━━━━━━
        ▼
[3_c_abi] TVMFFIFunctionCall(func, args, n, result)   (c_api.h:632)
        │  args/result 는 전부 TVMFFIAny (c_api.h:289)  ← 통일 컨테이너
        ▼
[4_cpp_side] TVMFFIFunctionCall 구현            (function.cc:191)
        │  func 안의 safe_call 포인터로 점프 (tail call, 극도로 얇음)
        ▼
[4_cpp_side] safe_call = __tvm_ffi_add_one_cpu  (function.h:950)
        │  unpack_call 로 TVMFFIAny[] → TensorView 개봉 (function_details.h:206)
        ▼
[1_example/add_one_cpu.cc] AddOne(TensorView x, TensorView y)  ← 진짜 계산: y = x + 1
        │
        ▼  결과를 result(TVMFFIAny)에 담아 되돌림
[2_python_side] make_ret(result)  ──►  파이썬 값으로 역변환해 반환
```

## 흐름 B: 등록 (5_register)

```
[입구 A] Python  register_global_func("name")          ─┐
[입구 B] C++     GlobalDef().def("name", Func)          ─┼─► TVMFFIFunctionSetGlobal
[입구 C] C++     TVM_FFI_DLL_EXPORT_TYPED_FUNC (+load)   ─┘        │ (function.cc:161)
                                                                  ▼
                                          GlobalFunctionTable::Update
                                                                  ▼
                                          table_.Set("name", Entry(func))
                                          (Map<String,Any> : 프로세스에 하나뿐인 해시맵)
```

---

## 핵심 아이디어 3개 (이게 tvm-ffi 의 전부)

1. **TVMFFIAny — 만능 값 상자** (3_c_abi)
   정수/실수/텐서/함수 등 경계를 넘는 모든 값을 `type_index`(라벨) + `union`(값) 하나로
   표현. C 규약 하나로 모든 타입을 실어 나른다 (= 타입 소거).

2. **safe_call 단일 시그니처 = Packed Function** (3_c_abi)
   모든 함수를 `(handle, args[], num_args, result) → 에러코드` 하나로 통일.
   시그니처가 하나뿐이라 언어가 달라도 서로 부를 수 있다. 반환값=에러코드, 결과는
   포인터로 나감(예외가 언어 경계를 못 넘으므로).

3. **전역 레지스트리 — 이름으로 등록/조회** (5_register)
   `Map<String, Any> table_` : 프로세스에 하나뿐인 해시맵. C++이 등록(SetGlobal),
   파이썬이 조회(GetGlobal). 같은 표를 공유하는 게 언어 간 연결의 토대.

나머지(참조 카운트, DLPack 무복사, 모듈)는 이 셋을 **안전·고속·실용적**으로 만드는 장치.

---

## 함께 알아두면 좋은 개념

- **Any(소유) vs AnyView(차용)** — 메모리 레이아웃 동일, 소유권만 다름. `Any`는 참조
  카운트 +1로 붙잡음(오래 보관), `AnyView`는 안 올리고 잠깐 빌림(호출 인자, 더 빠름).
- **참조 카운트** — 객체마다 "몇 명이 쓰는 중"을 세어 0이 되면 자동 삭제(`IncRef`/`DecRef`).
  전역표 등록 시 +1 하는 이유: 파이썬이 손 떼도 표가 붙잡고 있으면 안 지워지게.
- **마샬링 대칭** — 파이썬 `SetArgument`(값→상자, 포장) ↔ C++ `unpack_call`(상자→값, 개봉).
- **메모리 위치** — 진짜 코드는 `code`영역(한 벌), 객체(Entry/FunctionObj/표)는 `heap`,
  전역표 포인터는 `data`, 호출 중 임시(args/result)는 `stack`. 표엔 코드가 아니라
  포인터 사슬이 담긴다: `table_ → Entry → FunctionObj → (safe_call) → 실제 코드`.

---

## 파일별 안내

### 4_cpp_side (3개)
- `function_cc_excerpt.cc` — 함수 생애주기 4함수(Create/SetGlobal/GetGlobal/Call).
  `Call`은 `func->safe_call(...)`을 "부르는" 데까지.
- `safe_call_unpack_excerpt.cc` — 그 `safe_call`의 정체. 등록 매크로가 생성한
  `__tvm_ffi_<이름>` + `unpack_call`(TVMFFIAny[] → C++ 타입 인자 개봉).
- `setglobal_상세주석.cc` — `TVMFFIFunctionSetGlobal` 줄별 초상세 주석판.

### 5_register (4개 + README)
- `1_entry_python.py` / `2_entry_cpp.cc` — 등록 입구 3개(Python/GlobalDef/EXPORT 매크로).
- `3_setglobal.cc` — 모든 경로의 수렴점(C ABI 관문).
- `4_global_function_table.cc` — 저장소 실체(해시맵 `table_`, 봉투 `Entry`, 싱글턴).

---

## 실전 응용: flashinfer

같은 메커니즘을 실전 GPU 라이브러리가 어떻게 쓰는지는 별도 폴더에 정리:
`~/flashinfer/hoyoung/` — gemm 커널(`bmm_fp8`)로 컴파일→등록→호출 추적.
`add_one_cpu` 예제와 `bmm_fp8` 실전 커널은 tvm-ffi 관점에서 **완전히 같은 메커니즘**이다.

---

## 원본에서 더 볼 곳

- `include/tvm/ffi/c_api.h`             — C ABI 전체 (3_ 확장판)
- `include/tvm/ffi/function.h`          — 등록 매크로(TVM_FFI_DLL_EXPORT_TYPED_FUNC) 정의
- `include/tvm/ffi/function_details.h`  — unpack_call 등 언패킹 템플릿 메타프로그래밍
- `python/tvm_ffi/cython/function.pxi`  — 파이썬 바인딩 전체
- `python/tvm_ffi/registry.py`          — register_global_func / get_global_func
- `src/ffi/function.cc`                 — C++ 함수 구현 + GlobalFunctionTable
- `examples/quickstart/`                — add_one 원본 예제
- `examples/cubin_launcher/`            — cubin 컴파일→등록→호출 예제 (발표대본 참고)
- `examples/abi_overview/example_code.c`— ABI 만 순수 C 로 보여주는 최소 예제
