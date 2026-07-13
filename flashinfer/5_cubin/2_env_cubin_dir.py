# 원본: flashinfer/jit/env.py
# 역할: cubin 파일을 찾을 디렉터리를 결정한다.


def _get_cubin_dir():
    # 1. 별도 flashinfer-cubin 패키지가 있으면 그 안의 bundled cubins를 우선 사용한다.
    if has_flashinfer_cubin():
        import flashinfer_cubin

        return pathlib.Path(flashinfer_cubin.get_cubin_dir())

    # 2. 사용자가 cubin 위치를 직접 지정한 경우.
    env_dir = os.getenv("FLASHINFER_CUBIN_DIR")
    if env_dir:
        return pathlib.Path(env_dir)

    # 3. 기본 런타임 다운로드/cache 위치.
    return FLASHINFER_CACHE_DIR / "cubins"


# 모듈 import 시 한 번 결정되고, cubin_loader.py가 이 값을 사용한다.
FLASHINFER_CUBIN_DIR: pathlib.Path = _get_cubin_dir()
