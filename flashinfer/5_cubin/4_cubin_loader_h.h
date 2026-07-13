// 원본: include/flashinfer/cubin_loader.h
// 역할: C++에서 Python cubin callback을 호출한다.

// Python setup_cubin_loader()가 등록할 callback 포인터.
void (*callbackGetCubin)(const char* path, const char* sha256) = nullptr;

extern "C" void FlashInferSetCubinCallback(void (*callback)(const char* path, const char* sha256)) {
  callbackGetCubin = callback;
}

// Python callback이 받아온 cubin bytes를 임시로 넣어두는 thread-local 저장소.
thread_local std::string current_cubin;

extern "C" void FlashInferSetCurrentCubin(const char* binary, int size) {
  current_cubin = std::string(binary, size);
}

std::string getCubin(const std::string& name, const std::string& sha256) {
  if (!callbackGetCubin) {
    throw std::runtime_error("FlashInferSetCubinCallback not set");
  }
  // Python으로 왕복해서 current_cubin을 채운 뒤 반환한다.
  callbackGetCubin(name.c_str(), sha256.c_str());
  return current_cubin;
}
