// ============================================================================
// [저장소 실체] GlobalFunctionTable — 전역 레지스트리의 구현
// 발췌 출처: src/ffi/function.cc:51
//
// 한마디: "프로세스에 하나뿐인 파이썬 dict 의 C++ 버전".
// 핵심은 맨 아래 `Map<String, Any> table_` 한 줄. 나머지는 그걸 다루는 도구.
// ============================================================================

class GlobalFunctionTable {
 public:
  // ── ② 저장되는 '봉투' : 함수 + 부가정보를 한 데 묶음 ──────────────────
  //   함수만 넣지 않고 이름/문서/메타데이터까지 함께 보관 (목록·문서·타입검사용).
  class Entry : public Object, public TVMFFIMethodInfo {
   public:
    String name_data;         // 이름   ("my.add")
    String doc_data;          // 문서(설명)
    String metadata_data;     // 메타데이터(타입 정보 등)
    ffi::Function func_data;   // ★ 실제 함수(를 가리키는 참조카운트 포인터) ★

    // 생성자 2개 (등록 경로에 따라 다름) — 둘 다 결국 func_data 를 채움
    explicit Entry(String name, ffi::Function func)                  // SetGlobal 이 쓰는 것
        : name_data(std::move(name)), func_data(std::move(func)) {
      this->SyncMethodInfo(kTVMFFIFieldFlagBitMaskIsStaticMethod);
    }
    explicit Entry(const TVMFFIMethodInfo* method_info) { /* 리플렉션 등록용, 생략 */ }

   private:
    void SyncMethodInfo(int64_t flags) { /* TVMFFIMethodInfo 필드 채우기, 생략 */ }
  };

  // ── 등록: SetGlobal 이 부르는 것 ─────────────────────────────────────
  void Update(const String& name, Function func, bool can_override) {
    if (TVM_FFI_PREDICT_FALSE(table_.count(name) != 0)) {   // 이미 이 이름 있나?
      if (!can_override) {
        TVM_FFI_THROW(RuntimeError) << "Global Function `" << name << "` is already registered";
      }
    }
    // ★ 실제 저장 ★ : 함수를 Entry 봉투로 싸서 해시맵에 넣음
    //   파이썬으로 치면  table["my.add"] = Entry(name, func)
    table_.Set(name, ObjectRef(make_object<Entry>(name, std::move(func))));
  }

  // ── 조회: GetGlobal 이 부르는 것 ─────────────────────────────────────
  const Entry* Get(const String& name) {
    auto it = table_.find(name);                 // 이름으로 찾기
    if (it == table_.end()) return nullptr;      // 없으면 nullptr
    const Object* obj = (*it).second.cast<const Object*>();
    return static_cast<const Entry*>(obj);       // 찾은 봉투 반환 (호출자가 func_data 꺼내 실행)
  }

  // ── 삭제 / 목록 (파이썬 dict 의 del / keys 와 동일) ──────────────────
  bool Remove(const String& name) {
    auto it = table_.find(name);
    if (it == table_.end()) return false;
    table_.erase(name);
    return true;
  }
  Array<String> ListNames() const {
    Array<String> names;
    for (const auto& kv : table_) names.push_back(kv.first);
    return names;
  }

  // ── ① 프로세스에 하나뿐인 표 얻기 (싱글턴) ───────────────────────────
  static GlobalFunctionTable* Global() {
    // static 지역변수 → 최초 1회만 생성, 이후 항상 같은 인스턴스 반환.
    // 그래서 SetGlobal/GetGlobal 이 모두 '같은 표 하나'를 본다 (C++↔파이썬 공유의 근거).
    // 일부러 raw new 로 만들고 프로그램 종료까지 소멸시키지 않음:
    //   표 안에 파이썬 콜백이 있을 수 있어, 종료 순서가 꼬이면 위험하기 때문.
    static GlobalFunctionTable* inst = new GlobalFunctionTable();
    return inst;
  }

 private:
  // ── ③ ★ 진짜 저장소 ★ : 이름 → 값  해시맵 (파이썬 dict 와 동일한 개념) ──
  Map<String, Any> table_;
};


// ============================================================================
// 메모리 위치 정리 (프로세스 메모리 레이아웃 기준)
//
//   code  : AddOne/bmm_fp8 의 실제 기계어 (딱 한 벌). safe_call 이 여기를 가리킴.
//   data  : static inst 포인터 변수 자체 (전역표를 가리키는 고정 포인터).
//   heap  : GlobalFunctionTable 인스턴스, table_ 버킷, Entry, FunctionObj,
//           String 버퍼 등 '모든 객체'. 참조카운트로 수명 관리되는 대상.
//   stack : 호출 중 임시(name_str 헤더, TVMFFIAny result/args). 끝나면 소멸.
//
//   즉 표(heap)에는 '포인터'가 담기고, 그 사슬을 따라가면 code 영역의 진짜 코드에 도달:
//     table_ → Entry → FunctionObj → (safe_call) → 실제 기계어 코드
// ============================================================================
