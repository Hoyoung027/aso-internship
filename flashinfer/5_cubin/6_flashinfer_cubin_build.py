# 원본: flashinfer-cubin/build_backend.py
# 역할: wheel 빌드 전에 cubin artifact를 패키지 폴더에 받아둔다.

from setuptools import build_meta as _orig


def _download_cubins():
    from flashinfer.artifacts import download_artifacts

    # wheel 안에 포함될 cubin 디렉터리.
    cubin_dir = Path(__file__).parent / "flashinfer_cubin" / "cubins"
    cubin_dir.mkdir(parents=True, exist_ok=True)

    # 일반 downloader의 목적지를 package 내부 폴더로 바꾼다.
    os.environ["FLASHINFER_CUBIN_DIR"] = str(cubin_dir)
    try:
        download_artifacts()
        cubin_files = list(cubin_dir.rglob("*.cubin"))
        print(f"Downloaded {len(cubin_files)} cubin files")
    finally:
        os.environ.pop("FLASHINFER_CUBIN_DIR", None)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    # wheel 생성 전에 cubin 파일을 먼저 받아 package data로 포함한다.
    _download_cubins()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _download_cubins()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)
