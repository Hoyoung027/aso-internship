# ============================================================================
# [입구 A] 파이썬에서 전역표에 등록하기
# 발췌 출처: python/tvm_ffi/registry.py  +  python/tvm_ffi/cython/function.pxi
#
# 파이썬 함수를 이름 붙여 전역 레지스트리에 올리는 경로.
# 최종적으로 C ABI 관문 TVMFFIFunctionSetGlobal 로 내려간다 (3_setglobal.cc).
# ============================================================================

# ---- [원본 registry.py:110] 사용자용 데코레이터 --------------------------
#   사용 예:
#     @tvm_ffi.register_global_func("mytest.echo")
#     def echo(x): return x
def register_global_func(func_name, f=None, override=False):
    # 데코레이터 형태와 직접 호출 형태를 모두 지원하기 위한 처리
    if not isinstance(func_name, str):
        f = func_name
        func_name = f.__name__

    def register(myf):
        # ↓ 실제 등록은 cython(core) 쪽 _register_global_func 에 위임
        return core._register_global_func(func_name, myf, override)

    if f is not None:
        return register(f)     # 직접 호출: register_global_func("name", fn)
    return register            # 데코레이터: @register_global_func("name")


# ---- [원본 function.pxi:1078] cython 브리지: 파이썬 -> C 경계 -------------
#   파이썬 함수를 FFI Function 으로 변환한 뒤, C ABI 함수를 호출한다.
def _register_global_func(name, pyfunc, override):
    cdef int ioverride = override
    cdef ByteArrayArg name_arg = ByteArrayArg(c_str(name))

    if not isinstance(pyfunc, Function):
        pyfunc = _convert_to_ffi_func(pyfunc)     # 순수 파이썬 함수 → FFI Function 객체로 래핑

    # ★ 여기서 C 경계를 넘어 등록 ★  (다음 단계: 3_setglobal.cc)
    CHECK_CALL(TVMFFIFunctionSetGlobal(
        name_arg.cptr(),                # 이름 (바이트배열)
        (<CObject>pyfunc).chandle,      # 함수 핸들
        ioverride))                     # 덮어쓰기 여부
    return pyfunc


# 흐름 요약:
#   register_global_func("name", fn)                       [registry.py:110]
#     → core._register_global_func("name", fn, override)
#     → _convert_to_ffi_func(fn)  (파이썬 함수를 FFI Function 으로)
#     → TVMFFIFunctionSetGlobal(...)                       [C 경계 → 3_setglobal.cc]
