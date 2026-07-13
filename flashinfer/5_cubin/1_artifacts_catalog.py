# 원본: flashinfer/artifacts.py
# 역할: cubin artifact의 원격 경로와 checksum을 정의한다.


@dataclass(frozen=True)
class ArtifactPath:
    # 원격 repository 아래에서 artifact 종류별 하위 디렉터리를 가리킨다.
    TRTLLM_GEN_FMHA: str = "158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen/"
    TRTLLM_GEN_BMM: str = "481dce07c89a216cbfd18cf39de49a82d40739a8/batched_gemm-dd6d23e-721ae60/"
    TRTLLM_GEN_GEMM: str = "10f64528a1172dae8e29601a3b99ab9dc78d37be/gemm-91e0ba0-2710384/"
    CUDNN_SDPA: str = "a72d85b019dc125b9f711300cb989430f762f5a6/fmha/cudnn/"
    DEEPGEMM: str = "a72d85b019dc125b9f711300cb989430f762f5a6/deep-gemm/"
    DSL_FMHA: str = "801e770219613fbf088bc074c414732b26cc550d/fmha/cute-dsl/"
    DSL_FMHA_ARCHS: tuple = ("sm_100a", "sm_103a", "sm_110a")


class CheckSumHash:
    # 각 artifact 묶음의 checksum 매니페스트 검증에 쓰인다.
    TRTLLM_GEN_FMHA: str = "c2d9399b2537be785882354a4f9902ed6c03136c0ea341e201eac40c3923e1dc"
    TRTLLM_GEN_BMM: str = "aa19cf2a37eed029eee5b3f96b37e069e4ab40f419b25ed7a3fd9526d8833bfb"
    ...


def download_artifacts():
    # flashinfer-cubin 빌드나 CLI에서 전체 cubin artifact를 미리 받을 때 호출된다.
    ...
