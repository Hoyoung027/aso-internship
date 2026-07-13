# 원본: flashinfer/jit/core.py
# 역할: JIT 소스를 빌드하고 tvm-ffi module로 로드한다.


class JitSpec:
    def build(self, verbose: bool, need_lock: bool = True) -> None:
        with (FileLock(self.lock_path, thread_local=False) if need_lock else nullcontext()):
            self.write_ninja()
            run_ninja(self.build_dir, self.ninja_path, verbose)

    def load(self, so_path: Path):
        return tvm_ffi.load_module(str(so_path))

    def build_and_load(self):
        if self.is_aot:
            return self.load(self.aot_path)

        with FileLock(self.lock_path, thread_local=False):
            so_path = self.jit_library_path
            verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
            self.build(verbose, need_lock=False)
            result = self.load(so_path)
        return result
