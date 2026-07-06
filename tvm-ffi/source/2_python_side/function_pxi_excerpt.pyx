# ============================================================================
# 발췌 출처: python/tvm_ffi/cython/function.pxi
# 파이썬에서 func(a, b) 를 하면 여기로 들어온다. (파이썬 ↔ C 경계의 파이썬 쪽 끝단)
# ============================================================================

# ---- [원본 function.pxi:941] Function.__call__ ----------------------------
#   파이썬에서 mod.add_one_cpu(x, y) 호출 시 진입하는 지점.
    def __call__(self, *args: Any) -> Any:
        """Invoke the wrapped FFI function with ``args``."""
        cdef TVMFFIAny result                      # (1) 결과를 담을 통일 컨테이너 (c 구조체를 스택에 선언)
        cdef int c_api_ret_code
        cdef const DLPackExchangeAPI* c_ctx_dlpack_api = NULL
        # IMPORTANT: caller need to initialize result->type_index to kTVMFFINone
        result.type_index = kTVMFFINone      # 값 초기화
        result.v_int64 = 0
        TVMFFIPyFuncCall(                          # (2) C 헬퍼로 내려감 → 여기서 인자 마샬링 + 실제 C 호출
            (<CObject>self).chandle, <PyObject*>args,
            &result,
            &c_api_ret_code,
            self.release_gil,
            &c_ctx_dlpack_api
        )
        # NOTE: logic is same as check_call
        if c_api_ret_code == 0:
            return make_ret(result, c_ctx_dlpack_api)   # (5) 결과 TVMFFIAny → 파이썬 객체로 역변환
        if c_api_ret_code == -2:
            raise raise_existing_error()
        error = move_from_last_error()
        if error.kind == "EnvErrorAlreadySet":
            raise raise_existing_error()
        raise error.py_error()


# ---- [원본 function.pxi:172-181] 저수준 호출 예시 -------------------------
#   TVMFFIPyFuncCall 안에서 인자를 TVMFFIAny 배열로 만든 뒤, 최종적으로
#   아래처럼 C ABI 함수 TVMFFIFunctionCall 을 호출한다.
#   (이 코드는 컨테이너 스캔 경로의 실제 호출 예시)
            CHECK_CALL(TVMFFIFunctionCall(
                (<CObject>_FFI_CONTAINER_FIND_FIRST_NON_CPU_DEVICE).chandle,
                scan_args, 1, &scan_result))
#            └ func 핸들 ──────────────────────┘  └args┘ └n┘ └result┘
#   ↑ 이 한 줄이 "파이썬이 C 경계를 넘는" 물리적 지점. 다음은 C ABI(3_c_abi) 로 이어짐.


# ---- [원본 function.pxi:150-152] 인자 마샬링 예시 -------------------------
#   파이썬 객체(여기선 FFI Object) 하나를 TVMFFIAny 하나로 채워넣는 setter.
#   정수/실수/텐서/문자열마다 이런 setter 가 따로 있고, __call__ 은 인자 종류에
#   맞는 setter 를 골라 args 배열을 채운다.
    out.type_index = TVMFFIObjectGetTypeIndex((<CObject>arg).chandle)
    out.v_ptr = (<CObject>arg).chandle
    return 0
