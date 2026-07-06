/* ============================================================================
 * 발췌 출처: include/tvm/ffi/c_api.h
 * 여기가 "통역 규약(ABI)"의 심장. 파이썬 쪽(2_)과 C++ 쪽(4_)이 모두 이 규약만
 * 보고 대화한다. 이 파일에 의존성이 없기 때문에(순수 C) 언어 경계를 넘을 수 있다.
 * ============================================================================ */

/* ---- [원본 c_api.h:289-335] 값을 실어 나르는 통일 컨테이너 -----------------
 *   정수/실수/포인터/텐서/함수... 경계를 넘는 "모든 값"을 이 하나의 구조체로 표현.
 *   type_index 로 "무슨 타입인지" 구분하고, union 으로 "실제 값"을 담는다.
 *   지난 설명의 '타입 소거(type-erased) 컨테이너'가 바로 이것.                 */
typedef struct {
  int32_t type_index;          /* 이 Any 가 담은 값의 타입 (정수? 텐서? 함수?) */
  union {                      /* 4 bytes */
    uint32_t zero_padding;
    uint32_t small_str_len;
  };
  union {                      /* 8 bytes: 실제 페이로드 (아래 중 하나만 유효) */
    int64_t v_int64;           /*  - 정수                                       */
    double v_float64;          /*  - 실수                                       */
    void* v_ptr;               /*  - 일반 포인터                                */
    const char* v_c_str;       /*  - C 문자열                                   */
    TVMFFIObject* v_obj;       /*  - 참조카운트 객체 (텐서/함수/배열 등)         */
    DLDataType v_dtype;        /*  - 데이터 타입                                */
    DLDevice v_device;         /*  - 디바이스 (cpu/cuda...)                      */
    char v_bytes[8];           /*  - 작은 문자열                                */
    uint64_t v_uint64;
  };
} TVMFFIAny;


/* ---- [원본 c_api.h:493-494] 모든 함수의 통일된 시그니처 --------------------
 *   어떤 함수든 결국 이 한 가지 모양으로 통일된다:
 *      (핸들, 인자배열, 인자개수) -> 결과, 반환값은 에러코드(int)
 *   이 통일 덕분에 파이썬/C++/Rust 어디서 만든 함수든 서로 부를 수 있다.        */
typedef int (*TVMFFISafeCallType)(void* handle, const TVMFFIAny* args, int32_t num_args,
                                  TVMFFIAny* result);


/* ---- [원본 c_api.h:632] 파이썬이 실제로 넘어오는 C 관문 --------------------
 *   2_python_side 의 TVMFFIFunctionCall(...) 호출이 도달하는 곳(선언).
 *   내부적으로는 func 안에 저장된 safe_call 포인터를 그대로 호출한다(4_ 참고).  */
TVM_FFI_DLL int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
                                   TVMFFIAny* result);


/* ---- [원본 c_api.h:614] 이름으로 C++ 함수를 찾아오기 ----------------------
 *   load_module 후 mod.add_one_cpu 를 처음 접근할 때, 이름 문자열로 전역
 *   레지스트리에서 함수 핸들을 얻어온다.                                        */
TVM_FFI_DLL int TVMFFIFunctionGetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle* out);


/* ---- [원본 c_api.h:1276] C++/다른언어가 전역 레지스트리에 함수 등록 --------
 *   TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, ...) 매크로가 결국 이런 등록을
 *   수행한다. 등록해두면 위 GetGlobal 로 조회 가능해진다.                        */
TVM_FFI_DLL int TVMFFIFunctionSetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle f,
                                        int allow_override);
