# 5_register — 전역 레지스트리에 함수가 등록되는 과정

> 목표: "함수를 이름 붙여 전역표에 등록"이 실제 코드로 어떻게 되는지 따라가기.
> 핵심 구조: **입구 3개 → C ABI 관문 1개 → 전역표(해시맵) 1개**.

## 큰 그림

```
[입구 A] Python  register_global_func("name")          1_entry_python.py
[입구 B] C++     GlobalDef().def("name", Func)          2_entry_cpp.cc
[입구 C] C++     TVM_FFI_DLL_EXPORT_TYPED_FUNC (+load)   2_entry_cpp.cc
              │
              ▼  세 경로가 전부 수렴
     TVMFFIFunctionSetGlobal(name, func, override)       3_setglobal.cc   (C ABI 관문)
              │
              ▼
     GlobalFunctionTable::Update(name, func)             4_global_function_table.cc
              │
              ▼
     table_.Set("name", Entry(func))   ← ★ 실제 저장 ★
     (Map<String, Any> table_ : 프로세스에 하나뿐인 해시맵)
```

## 파일 구성 (읽는 순서)

| 파일 | 내용 |
|---|---|
| `1_entry_python.py` | 입구 A. `register_global_func` → cython `_register_global_func` → C 경계 |
| `2_entry_cpp.cc` | 입구 B(`GlobalDef().def`) + 입구 C(`EXPORT` 매크로, 지연 등록) |
| `3_setglobal.cc` | 모든 경로의 수렴점. 이름 복사 + 참조카운트 + Update 호출 |
| `4_global_function_table.cc` | 저장소 실체. 해시맵 `table_`, 봉투 `Entry`, 싱글턴 `Global()` |

## 핵심 개념 3가지

1. **입구는 여러 개, 종착지는 하나** — 파이썬/C++/매크로 어느 길로 와도 결국
   `TVMFFIFunctionSetGlobal` → `table_.Set` 하나로 수렴한다.

2. **저장소 = 프로세스에 하나뿐인 해시맵** — `Map<String, Any> table_`.
   싱글턴 `Global()`이 그 하나를 공유하므로 C++이 넣은 걸 파이썬이 찾아 부를 수 있다.

3. **담기는 건 코드가 아니라 포인터** — 표에는 `Entry` 봉투(포인터)가 담기고,
   `table_ → Entry → FunctionObj → (safe_call) → 실제 기계어 코드`로 이어진다.
   진짜 코드는 code 영역에 한 벌만 있고 복사되지 않는다.

## 등록 후: 조회·호출로 이어짐

- 조회: `TVMFFIFunctionGetGlobal` = `GlobalFunctionTable::Get` (같은 표에서 이름으로 꺼냄)
- 호출: 꺼낸 함수를 `TVMFFIFunctionCall` → `safe_call` → 실제 커널
  (자세한 호출 경로는 상위 폴더 `2_python_side` ~ `4_cpp_side` 참고)

## 관련 파일 (다른 폴더)

- `../4_cpp_side/setglobal_상세주석.cc` — `TVMFFIFunctionSetGlobal` 줄별 초상세 주석판
- `../4_cpp_side/function_cc_excerpt.cc` — 함수 생애주기 4함수(Create/SetGlobal/GetGlobal/Call)
- `../3_c_abi/c_api_h_excerpt.h` — SetGlobal/GetGlobal 의 C ABI 선언
