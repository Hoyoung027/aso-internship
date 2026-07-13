# 원본: flashinfer/jit/cubin_loader.py
# 역할: cubin을 로드/다운로드하고 C++에 callback을 등록한다.

import ctypes

FLASHINFER_CUBINS_REPOSITORY = os.environ.get(
    "FLASHINFER_CUBINS_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",
)


def get_artifact(file_name: str, sha256: str, session=None) -> bytes:
    # 먼저 flashinfer-cubin package/env/cache가 가리키는 로컬 경로를 확인한다.
    local_path = str(FLASHINFER_CUBIN_DIR / file_name)
    data = load_cubin(local_path, sha256)
    if data:
        return data

    # 오프라인 모드에서는 로컬에 없을 때 즉시 실패한다.
    if os.getenv("FLASHINFER_NO_DOWNLOAD"):
        raise RuntimeError(f"Artifact not found locally: {file_name}")

    # 로컬에 없으면 원격 repository에서 받아 cache에 저장한 뒤 다시 읽는다.
    uri = safe_urljoin(FLASHINFER_CUBINS_REPOSITORY, file_name)
    download_file(uri, local_path, session=session)
    return load_cubin(local_path, sha256)


get_cubin = get_artifact

dll_cubin_handlers = {}


def setup_cubin_loader(dll_path: str) -> None:
    # 같은 .so에 callback을 중복 등록하지 않는다.
    if dll_path in dll_cubin_handlers:
        return

    # tvm_ffi.load_module으로 연 .so를 ctypes로도 열어 C ABI 함수를 호출한다.
    _LIB = ctypes.CDLL(dll_path)
    CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

    def get_cubin_callback(name: bytes, sha256: bytes):
        # C++ getCubin(name, sha256)이 호출되면 여기로 들어온다.
        cubin = get_artifact(name.decode("utf-8"), sha256.decode("utf-8"))
        # 받은 bytes를 다시 같은 .so의 C++ thread_local 저장소에 넣는다.
        _LIB.FlashInferSetCurrentCubin(convert_to_ctypes_char_p(cubin), ctypes.c_int(len(cubin)))

    cb = CALLBACK_TYPE(get_cubin_callback)
    # CFUNCTYPE callback이 GC되지 않게 Python 쪽에서 붙잡아 둔다.
    dll_cubin_handlers[dll_path] = cb
    _LIB.FlashInferSetCubinCallback(cb)
