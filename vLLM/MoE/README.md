# GPT-OSS 20B layer-0 MoE kernel 비교 실험 계획

이 문서는 NVIDIA RTX PRO 6000 Blackwell Server Edition(SM 12.0)에서
GPT-OSS 20B의 MoE backend를 비교하기 위한 실행 계획이다. 모든 서버와
측정 코드는 **vLLM 0.23.0**을 사용한다.

참고 문서:

- [연세대학교 HPC 클러스터 사용 가이드](../../../yonsei-hpc-cluster-guide.md)
- [기존 GPT-OSS 20B 서빙 및 프로파일링 기록](../../MoE/README.md)
- [기존 Torch trace 분석기](../../MoE/analyze_traces.py)
- [vLLM MoE kernel feature 표](../../../vllm/docs/design/moe_kernel_features.md)

## 1. 실험 목표

이번 실험의 질문은 다음 두 가지다.

1. 같은 GPT-OSS 20B layer 0 입력에서 사용 가능한 MoE backend 중
   `FusedMoE` 실행시간이 가장 긴 backend는 무엇인가?
2. RTX PRO 6000에서 FlashInfer CUTLASS backend의 AutoTuner를 켰을 때와
   껐을 때 latency가 얼마나 달라지는가?

전체 모델의 end-to-end latency, Attention latency, 모든 layer의 MoE 시간
합계는 이번 실험의 주 대상이 아니다. vLLM 서버는 실제 모델 적재와 backend
선택을 검증하기 위해 사용하지만, trace 집계는 layer 0의 router와 FusedMoE만
대상으로 한다.

## 2. 확정한 실험 조건

| 항목 | 값 |
|---|---|
| 모델 | GPT-OSS 20B MXFP4 |
| vLLM | `0.23.0` 고정 |
| GPU | RTX PRO 6000 Blackwell Server Edition, SM 12.0 |
| GPU 수 | 1 |
| 대상 layer | layer 0만 측정 |
| 독립변수 이름 | `num_tokens` |
| sequence 수 | 항상 1 |
| CUDA Graph | 끔 (`--enforce-eager`, `-O0`) |
| torch.compile | 끔 (`-O0`) |
| prefix caching | 끔 (`--no-enable-prefix-caching`) |
| warmup | 각 `num_tokens`에서 20회 |
| repeat | 각 `num_tokens`에서 50회 |
| 입력 seed | 42 |
| 시간 기준 | Torch Profiler의 GPU user annotation `dur` |
| Nsight Systems | 사용하지 않음 |

`batch_size`라는 이름은 새 실험 코드에서 사용하지 않는다. 이 실험에서
변하는 것은 요청 수가 아니라 한 model forward에서 layer 0에 들어오는 token
행의 수이므로 인자와 CSV 열 이름을 모두 `num_tokens`로 통일한다. 이전 코드와
호환하기 위한 alias도 두지 않는다.

실험할 `num_tokens`는 다음과 같다.

```text
1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
1024, 1536, 2048, 3072, 4096, 8192
```

결과에서는 Prefill과 Decode를 별도 workload로 구분하지 않는다. 서버 요청은
길이 `M=num_tokens`인 prompt 한 개와 `max_tokens=1`을 사용한다. 따라서 각
요청에는 layer 0에서 `M`개 token을 처리하는 model forward가 한 번 발생하며,
추가 Decode forward는 발생하지 않는다. CSV의 workload 이름도 `prefill`이
아니라 `num_tokens=M`으로 기록한다.

## 3. 비교할 backend

### 3.1 측정 대상

| experiment ID | vLLM 설정 | Expert activation | AutoTuner | 비교 용도 |
|---|---|---|---|---|
| `marlin_w4a16` | `--moe-backend marlin` | BF16 | OFF | 기본 MXFP4 backend |
| `humming_indexed_w4a16` | `--moe-backend humming` + `VLLM_HUMMING_MOE_GEMM_TYPE=indexed` | BF16 | OFF | Humming indexed GEMM |
| `humming_grouped_w4a16` | `--moe-backend humming` + `VLLM_HUMMING_MOE_GEMM_TYPE=grouped` | BF16 | OFF | Humming grouped GEMM |
| `flashinfer_cutlass_w4a8_tune_off` | `--moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8` | MXFP8 | OFF | FlashInfer heuristic tactic |
| `flashinfer_cutlass_w4a8_tune_on` | 위와 동일 | MXFP8 | ON | FlashInfer tuned tactic |
| `emulation_w4a16` | `--moe-backend emulation` | BF16/FP16 QDQ | OFF | 정확도·디버그 기준선 |

`auto`는 별도의 kernel이 아니라 선택 정책이다. latency 표에는 중복해서 넣지
않고, 사전 검증에서만 다음 두 선택을 확인한다.

- 기본 MXFP4 + BF16 activation: `auto`가 Marlin을 선택하는지 확인
- MXFP4 + MXFP8 activation: `auto`가 SM120에서 FlashInfer CUTLASS를 선택하는지
  확인

FlashInfer 실험에서 `flashinfer_cutlass_afp8`이라는 별도 backend 이름은 쓰지
않는다. vLLM 0.23.0의 공개 CLI 값인 `flashinfer_cutlass`에
`--quantization-config.moe.activation mxfp8`을 결합한다.

### 3.2 이번 실험에서 제외할 backend

| backend | 제외 이유 |
|---|---|
| `triton`, `triton_unfused` | 현재 vLLM 0.23.0의 CUDA Triton MoE 지원 판정에서 SM120 제외 |
| `deep_gemm`, `deep_gemm_mega_moe` | GPT-OSS MXFP4 및 SM120 조합과 맞지 않음 |
| `cutlass` | GPT-OSS MXFP4 weight scheme 미지원 |
| `flashinfer_trtllm` | 해당 경로는 SM100 family 대상이며 SM120 제외 |
| `flashinfer_cutedsl` | GPT-OSS MXFP4 scheme 미지원 |
| `flashinfer_b12x` | SM12x 대상이지만 NVFP4 scheme용이며 GPT-OSS MXFP4와 다름 |
| `aiter` | ROCm 전용 |

지원 판정은 코드상 가능 여부와 실제 서버 초기화 여부를 모두 통과해야 한다.
초기화가 실패한 backend는 억지로 fallback하지 않고 `unsupported`와 오류 메시지를
결과에 기록한다.

## 4. 공정한 비교 범위

### 4.1 동일 정밀도 비교

다음 네 조건은 MXFP4 weight와 BF16 activation을 사용하므로 한 그룹에서
직접 비교한다.

```text
Marlin
Humming indexed
Humming grouped
Emulation
```

Emulation은 테스트용 구현이므로 매우 느리거나 큰 `num_tokens`에서 OOM이 날 수
있다. 그런 경우 값을 임의로 줄이지 않고 해당 shape를 `oom`, `timeout` 또는
`unsupported`로 기록한다.

### 4.2 FlashInfer AutoTuner 비교

다음 두 조건은 backend와 precision이 모두 같고 AutoTuner만 다르므로 직접적인
A/B 비교다.

```text
flashinfer_cutlass_w4a8_tune_off
flashinfer_cutlass_w4a8_tune_on
```

Marlin/Humming의 W4A16 결과와 FlashInfer의 W4A8 결과는 실제 배포 구성 간
비교로는 볼 수 있지만, 순수 kernel 구현만의 차이라고 해석하면 안 된다.
activation 양자화 비용과 수치 형식 차이도 함께 포함되기 때문이다.

## 5. 측정 범위: router와 FusedMoE 분리

vLLM 0.23.0의 `MLPBlock.forward()`는 다음 순서로 실행된다.

```text
hidden states
  -> self.router(x)
  -> router logits
  -> self.experts(hidden_states=x, router_logits=g)
  -> MoE output
```

기존 [MoE 실험](../../MoE/README.md)은 `self.mlp(...)` 전체를
`GPTOSS_MOE_L*`로 감싸 router와 FusedMoE를 합쳐 측정했다. 새 실험은 layer 0에
한해 다음 두 annotation을 별도로 추가한다.

```python
if self.layer_idx == 0:
    with torch.profiler.record_function(
        f"GPTOSS_ROUTER_L0_M{x.shape[0]}"
    ):
        g = self.router(x)

    with torch.profiler.record_function(
        f"GPTOSS_FUSED_MOE_L0_M{x.shape[0]}"
    ):
        x = self.experts(hidden_states=x, router_logits=g)
else:
    g = self.router(x)
    x = self.experts(hidden_states=x, router_logits=g)
```

실제 구현에서는 ROCm 분기를 그대로 보존하고, 기존 결과 slicing과 sequence
parallel 처리도 변경하지 않는다. native CUDA extension이나 각 kernel 코드는
수정하지 않는다.

측정값의 의미는 다음과 같다.

| 값 | 포함 범위 |
|---|---|
| `router_gpu_ms` | layer 0 router linear |
| `fused_moe_gpu_ms` | top-k/routing 정보 처리, dispatch, expert GEMM, activation, combine을 포함한 `FusedMoE` 호출 전체 |
| `moe_total_gpu_ms` | 같은 반복의 `router_gpu_ms + fused_moe_gpu_ms` |

backend 순위는 우선 `fused_moe_gpu_ms`로 결정한다. Router는 backend에 따라
바뀌지 않아야 하므로 입력 동일성과 측정 안정성을 확인하는 control metric으로
사용한다.

## 6. 입력과 routing 고정

새 요청 스크립트는 seed 42로 길이 8192의 token ID 배열을 한 번 만들고,
`num_tokens=M`에서는 앞의 `M`개를 사용한다.

```text
master_tokens = random_token_ids(seed=42, length=8192)
request_tokens(M) = master_tokens[:M]
```

모든 backend에 같은 token ID를 보내고 다음 조건을 고정한다.

- sequence 수: 1
- `temperature=0`
- `max_tokens=1`
- `min_tokens=1`
- `ignore_eos=true`
- `add_special_tokens=false`
- prefix caching 비활성화
- 다른 client 요청 없음

요청은 `/v1/completions`에 token ID 배열을 직접 보내고, 응답의
`usage.prompt_tokens`가 정확히 `M`인지 매번 검증한다. tokenizer와 chat
template이 token 수를 바꾸는 요청은 측정하지 않는다.

layer 0 MoE 앞에는 아직 다른 MoE backend의 결과가 들어오지 않았으므로, 같은
모델 weight·Attention backend·token ID를 사용하면 layer 0 router 입력과 router
logit은 backend 간 동일해야 한다. 현재 구현은 timed forward에 GPU-to-CPU copy나
추가 synchronization을 넣지 않기 위해 router logit 자체를 저장하지 않고 다음을
검증한다.

- 모든 backend에서 같은 `prompt_sha256` 사용
- 응답의 `usage.prompt_tokens == num_tokens`
- router annotation event 수가 정확히 50개
- backend별 router latency가 비정상적으로 달라지지 않는지 확인

expert index와 expert별 token histogram 검증이 필요하면 별도의 correctness-only
실험으로 추가한다. 이 검증은 timed trace와 분리해야 한다.

## 7. warmup과 repeat

각 backend와 `num_tokens` 조합은 다음 순서로 실행한다.

```text
동일 요청 20회: profiler OFF, 결과 폐기
POST /start_profile
동일 요청 50회: profiler ON, latency 표본으로 사용
POST /stop_profile
trace flush 및 event count 검증
```

한 요청에서 router와 FusedMoE가 연속으로 실행되므로 20회의 warmup 요청이 두
연산을 모두 warmup한다. 50회의 측정 요청에서는 annotation별 event가 정확히
50개여야 한다. 개수가 다르면 해당 shape를 실패 처리한다.

각 shape 사이에는 profiler를 완전히 종료하고 trace를 별도 파일로 저장한다.
GPU clock과 온도 변화를 확인하기 위해 shape 시작 전후의 `nvidia-smi` 요약도
metadata에 남긴다. 측정 중에는 다른 요청을 서버에 보내지 않는다.

## 8. FlashInfer AutoTuner ON/OFF 규칙

vLLM 0.23.0에서는 서버의 kernel warmup 단계에서 FlashInfer autotuning을
수행할 수 있다. 두 조건은 반드시 **서로 다른 새 서버 프로세스**로 실행한다.

### OFF

```bash
--no-enable-flashinfer-autotune
```

### ON

```bash
--enable-flashinfer-autotune
```

공통으로 `--max-num-batched-tokens 8192`를 주면 ON 서버의 시작 단계에서 최대
token 수 8192로 dummy run을 수행해 그 이하 shape의 tactic을 조정한다.
AutoTuner 시간은 서버 startup 시간으로 별도 기록하며 20회 warmup과 50회
latency에 포함하지 않는다.

vLLM 0.23.0 소스의 현재 FlashInfer persistent autotune cache 경로는
`_FLASHINFER_USE_PERSISTENT_CACHE=False`이므로 선택 결과는 서버 프로세스 내부에
유지된다. 그래도 재현성을 위해 ON/OFF마다 프로세스를 완전히 종료하고, 로그에서
다음을 확인한다.

- OFF: FlashInfer autotune을 건너뛰었다는 로그
- ON: 8192-token dummy run을 이용한 autotune 완료
- 두 조건 모두 실제 expert backend가 FlashInfer CUTLASS인지 확인

`-O0`은 기본적으로 AutoTuner를 끄지만 사용자 지정 flag가 optimization level
기본값보다 우선한다. 따라서 ON 조건에는 `-O0`과
`--enable-flashinfer-autotune`을 함께 명시한다.

## 9. HPC 저장 및 실행 원칙

[클러스터 가이드](../../../yonsei-hpc-cluster-guide.md)에 따라 다음 원칙을
지킨다.

- 무거운 실행은 login node가 아니라 SLURM이 할당한 compute node에서 수행
- 정식 실험은 `srun`이 아니라 `sbatch` 사용
- GPU는 `--gres=gpu:1`로 요청
- `$CUDA_VISIBLE_DEVICES`는 SLURM 값을 사용하며 직접 설정하지 않음
- 패키지와 모델 다운로드는 login node에서 수행
- Python 환경, 모델, 대용량 trace는 `/lustre/hybyun0207`에 저장
- stdout/stderr와 작은 CSV만 저장소로 복사
- 서버 종료와 job 종료를 `trap`으로 보장

사용할 partition 후보는 RTX PRO 6000이 있는 다음 partition이다.

```text
asus_pro6000
gigabyte_pro6000
```

backend 간 GPU 편차를 피하기 위해 성능 비교 대상은 가능하면 **하나의 sbatch
allocation과 같은 물리 GPU에서 서버를 순차 재시작**하며 실행한다. Emulation은
실행시간이 지나치게 길 경우 별도 job으로 분리한다. 서로 다른 노드 결과를 한
표에서 비교해야 한다면 GPU 이름, node, driver, clock, power limit을 반드시
함께 기록한다.

## 10. 검증된 vLLM 0.23.0 환경

2026-07-28 RTX PRO 6000 smoke test를 통과한 기준 경로와 버전은 다음과 같다.

| 항목 | 확인값 |
|---|---|
| vLLM package | `0.23.0` |
| PyTorch | `2.11.0+cu130` |
| FlashInfer | `0.6.12` |
| vLLM source | `/lustre/hybyun0207/vllm023-moe/src/vllm`, tag `v0.23.0` |
| Python 환경 | `/lustre/hybyun0207/vllm023-moe/env` |
| 모델 | `/lustre/hybyun0207/vllm023-moe/models/gpt-oss-20b` |
| CUDA JIT toolkit | `/lustre/hybyun0207/vllm023-moe/cuda/13.0-full/nvidia/cu13` |
| FlashInfer workspace | `/lustre/hybyun0207/vllm023-moe/flashinfer-cuda130-full` |

실험 저장소는 home에 두고, 환경·모델·trace처럼 큰 파일만 하나의 `EXP_ROOT`
아래에 모은다.

```bash
export PROJECT_DIR=/home/hybyun0207/aso-internship/vLLM/MoE
export EXP_ROOT=/lustre/hybyun0207/vllm023-moe
export VLLM_ENV="$EXP_ROOT/env"
export VLLM_SRC="$EXP_ROOT/src/vllm"
export MODEL_PATH="$EXP_ROOT/models/gpt-oss-20b"
```

이 환경은 conda 환경이 아니라 uv로 만든 venv다. `conda activate`나
`source ~/.bashrc`가 필요하지 않으며, 항상 `$VLLM_ENV/bin/python`과
`$VLLM_ENV/bin/vllm`을 직접 사용한다. job 시작 직후 버전이 맞지 않으면
실행기가 즉시 종료한다.

```bash
"$VLLM_ENV/bin/python" -c "
from importlib.metadata import version
assert version('vllm') == '0.23.0', version('vllm')
print('vLLM:', version('vllm'))
print('PyTorch:', version('torch'))
print('FlashInfer:', version('flashinfer-python'))
"

git -C "$VLLM_SRC" describe --tags --exact-match
```

소스 checkout도 `v0.23.0`이어야 한다. annotation을 위해 수정한 Python 소스가
실제로 서버에서 import되는지 다음 정보도 로그에 남긴다.

```bash
"$VLLM_ENV/bin/python" -c "
import inspect
import vllm
import vllm.model_executor.models.gpt_oss as gpt_oss
print('vllm package:', vllm.__file__)
print('gpt_oss source:', inspect.getfile(gpt_oss))
"
```

## 11. 서버 공통 설정

기존 검증에서 FlashInfer sampler가 SM120 판정 문제를 일으켰으므로 sampler만
비활성화한다. 이 환경변수는 FlashInfer MoE backend를 비활성화하지 않는다.

RTX PRO 6000은 SM120이므로 FlashInfer 0.6.12의 JIT compiler가 CUDA 12.9
이상을 사용해야 한다. 클러스터 공용 CUDA 12.8을 그대로 사용하면
`SM 12.x requires CUDA >= 12.9` 이후 잘못된 `requires GPUs with sm75 or
higher` 오류가 발생한다. PyTorch가 cu130인 것만으로는 충분하지 않으며 실제
`nvcc`도 12.9 이상이어야 한다.

GPU node에서는 공통 runtime 환경 파일을 먼저 source한다. 이 파일은
`EXP_ROOT`를 `/lustre` 아래로 설정하고 별도로 설치한 CUDA 13.0 compiler와
header를 FlashInfer JIT에 연결한다.

부분적인 CUDA 패키지만 설치하면 compiler, NVVM, header가 서로 다른 minor
버전으로 섞일 수 있다. 실제로 `nvidia-cuda-nvcc 13.0`과 함께 NVVM 13.3이
설치되어 PTX 9.3을 만들었고, CUDA 13.0의 `ptxas`는 PTX 9.0까지만 받아
JIT compile이 실패했다. 따라서 vLLM 환경 내부의 혼합 CUDA 파일을 사용하지
않고 `cuda-toolkit[all]==13.0.2`를 별도 경로에 완전 설치한다.

```bash
# login node에서 최초 1회
bash /home/hybyun0207/aso-internship/vLLM/MoE/setup_flashinfer_cuda.sh
```

```bash
source /home/hybyun0207/aso-internship/vLLM/MoE/setup_runtime_env.sh
```

runtime script는 다음을 함께 설정하고 검증한다.

- `CUDA_HOME=$EXP_ROOT/cuda/13.0-full/nvidia/cu13`
- `nvcc`, header, NVVM이 모두 CUDA 13.0 계열인지 확인
- `cublasLt.h`, `curand_kernel.h`가 존재하는지 확인
- Python CUDA wheel에 없는 `lib64`와 unversioned linker 이름 확인
- `FLASHINFER_WORKSPACE_BASE=$EXP_ROOT/flashinfer-cuda130-full`
- `VLLM_USE_FLASHINFER_SAMPLER=0`, multiprocessing `spawn`
- Hugging Face, Torch, vLLM, Triton cache 경로
- `$VLLM_ENV/bin`을 `PATH`에 추가하고 `ninja` 실행 파일 확인

FlashInfer 0.6.12의 persistent JIT 위치를 지정하는 환경변수는
`FLASHINFER_WORKSPACE_BASE`다. `FLASHINFER_JIT_DIR`은 사용하지 않는다. AutoTuner
OFF도 MXFP4 weight 변환용 보조 커널을 JIT compile하므로 완전한 CUDA toolkit과
같은 workspace가 필요하다.

다음 preflight에서 FlashInfer CUDA version은 반드시 `12.9` 이상이어야 한다.

```bash
"$CUDA_HOME/bin/nvcc" --version

"$VLLM_ENV/bin/python" - <<'PY'
import torch
from flashinfer.jit.cpp_ext import get_cuda_path, get_cuda_version

print("Torch CUDA:", torch.version.cuda)
print("GPU capability:", torch.cuda.get_device_capability())
print("FlashInfer CUDA path:", get_cuda_path())
print("FlashInfer CUDA version:", get_cuda_version())
PY
```

`nvcc`, `ptxas`, NVVM과 `cuda_runtime_api.h`가 모두 `13.0`이어야 한다.
`setup_flashinfer_cuda.sh`는 SM120 compile probe와 CUDA link probe까지 통과해야
성공한다. `libcudart.so`, `libcublas.so`, `libcublasLt.so`, `libcurand.so` 같은
unversioned linker 이름도 이 단계에서 만든다.

### 11.1 실제 설치·smoke test에서 확인한 문제

| 증상 | 원인 | 현재 반영한 해결책 |
|---|---|---|
| `uv is required` | 별도 설치한 uv가 `PATH`에 없음 | setup script가 `$EXP_ROOT/tools/bin/uv`도 직접 탐색 |
| `No module named vllm._C` | editable 설치가 wheel의 native extension을 대체 | 공식 `vllm==0.23.0` cu130 wheel 유지 후 Python 파일만 patch |
| login node에서 NVML/Triton 경고 | login node에는 활성 GPU driver가 없음 | 설치 검증과 실제 GPU smoke test를 분리 |
| SM120/CUDA 12.8 오류 | 공용 CUDA 12.8은 FlashInfer SM12.x JIT에 부족 | 격리된 CUDA 13.0.2 toolkit 사용 |
| `cublasLt.h` 또는 `curand_kernel.h` 없음 | CUDA 일부 component만 설치 | `cuda-toolkit[all]==13.0.2` 설치 |
| PTX 9.3을 `ptxas`가 거부 | nvcc 13.0과 NVVM 13.3이 섞임 | NVVM `13.0.88` 고정 및 nvcc/ptxas/header 검사 |
| `-lcudart`, `-lcublas` link 실패 | Python CUDA wheel에는 versioned `.so`만 존재 | `lib64`와 unversioned `.so` compatibility link 생성 |
| FlashInfer JIT cache 재사용 실패 | 잘못된 cache 환경변수 사용 | `FLASHINFER_WORKSPACE_BASE`를 고정 |
| AutoTuner OFF인데도 JIT 발생 | weight 변환 보조 kernel은 tuner와 별개 | OFF/ON 모두 같은 완전한 CUDA와 workspace 사용 |
| `/smoke-...` permission denied | `EXP_ROOT`가 비어 있는 상태에서 경로 생성 | runtime script를 먼저 source하고 `$EXP_ROOT/runs/...` 사용 |
| sbatch에서 `ninja`를 찾지 못함 | venv를 직접 실행했지만 venv `bin`이 `PATH`에는 없음 | runtime에서 `$VLLM_ENV/bin`을 `PATH` 맨 앞에 추가 |
| Emulation 초기화 실패 | MXFP4 dequantization용 `amd-quark` 누락 | Quark 본체와 kernel import에 필요한 SciPy/ONNX 의존성을 고정 설치 |

첫 FlashInfer JIT가 오래 걸리는 것은 오류가 아니다. 서버 로그가 계속 갱신되고
compile process가 살아 있다면 startup timeout 7200초 안에서 기다린다.

sbatch script의 vLLM, CUDA, model, workspace 경로는 login shell에서 상속하지
않는다. 별도 실험 루트를 사용해야 할 때만 `MOE_EXP_ROOT`로 명시적으로
덮어쓴다. 이렇게 해야 과거의 `FLASHINFER_TOOLKIT_ROOT=.../cuda/13.0` 같은 값이
job에 전달되는 문제를 막을 수 있다.

각 backend별 trace와 log는 별도 디렉터리를 쓴다.

```bash
export RUN_ID=marlin_w4a16
export RUN_ROOT="$EXP_ROOT/runs/$RUN_ID"
export TRACE_DIR="$RUN_ROOT/traces"
export LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$TRACE_DIR" "$LOG_DIR" "$TRITON_CACHE_DIR"
```

공통 서버 명령은 다음 형태다.

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enforce-eager \
  -O0 \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir "$TRACE_DIR" \
  --profiler-config.torch_profiler_with_stack=false \
  --profiler-config.torch_profiler_record_shapes=false \
  --profiler-config.torch_profiler_with_memory=false \
  --profiler-config.ignore_frontend=true \
  "${BACKEND_ARGS[@]}" \
  > "$LOG_DIR/server.log" 2>&1 &
```

`max-num-batched-tokens=8192`와 client 1개를 사용해 모든 shape가 한 scheduler
step에서 처리되도록 한다. 서버 시작 후에는 `/health` 확인, 짧은 생성 요청,
실제 backend 확인을 통과한 뒤 측정을 시작한다.

## 12. backend별 실행 인자

각 조건은 새로운 서버 프로세스에서 실행한다.

### Marlin

```bash
unset VLLM_HUMMING_MOE_GEMM_TYPE
BACKEND_ARGS=(
  --moe-backend marlin
  --no-enable-flashinfer-autotune
)
```

### Humming indexed

```bash
export VLLM_HUMMING_MOE_GEMM_TYPE=indexed
BACKEND_ARGS=(
  --moe-backend humming
  --no-enable-flashinfer-autotune
)
```

### Humming grouped

```bash
export VLLM_HUMMING_MOE_GEMM_TYPE=grouped
BACKEND_ARGS=(
  --moe-backend humming
  --no-enable-flashinfer-autotune
)
```

### FlashInfer CUTLASS + AutoTuner OFF

```bash
unset VLLM_HUMMING_MOE_GEMM_TYPE
BACKEND_ARGS=(
  --moe-backend flashinfer_cutlass
  --quantization-config.moe.activation mxfp8
  --no-enable-flashinfer-autotune
)
```

### FlashInfer CUTLASS + AutoTuner ON

```bash
unset VLLM_HUMMING_MOE_GEMM_TYPE
BACKEND_ARGS=(
  --moe-backend flashinfer_cutlass
  --quantization-config.moe.activation mxfp8
  --enable-flashinfer-autotune
)
```

### Emulation

```bash
unset VLLM_HUMMING_MOE_GEMM_TYPE
BACKEND_ARGS=(
  --moe-backend emulation
  --no-enable-flashinfer-autotune
)
```

서버가 준비되면 다음을 확인한다.

```bash
curl -f http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

## 13. SLURM 실행 구성

정식 실험은 저장소의 sbatch script를 그대로 사용한다.

```bash
cd /home/hybyun0207/aso-internship/vLLM/MoE
sbatch slurm/run_moe_benchmark.sbatch
```

script 기본 partition은 `asus_pro6000`, QOS는 `pro6000_qos`, GPU는 한 장,
제한시간은 14시간이다.
`gigabyte_pro6000`을 사용할 때는 partition만 바꾼다. 정식 비교에서는 한 job
안에서 아래 순서를 번갈아 실행해 장시간 온도 변화가 한 backend에만 유리하게
작용하지 않는지 확인한다.

```text
pilot: marlin -> flashinfer OFF -> flashinfer ON
full:  humming indexed -> marlin -> flashinfer ON
       -> humming grouped -> flashinfer OFF
debug: emulation
```

각 서버는 `trap` 또는 `finally`에서 반드시 종료하고 `wait`한 뒤 다음 서버를
시작한다. 서버가 죽은 뒤 `nvidia-smi`에서 메모리가 반환됐는지도 확인한다.

## 14. trace 분석 방법

새 분석기는 기존 [`analyze_traces.py`](../../MoE/analyze_traces.py)와 같은
방식으로 gzip Torch trace를 event 단위로 읽는다.

latency에 사용할 event 조건은 다음과 같다.

```python
event["cat"] == "gpu_user_annotation"
event["ph"] == "X"
event["name"] matches "GPTOSS_ROUTER_L0_M*"
                       or "GPTOSS_FUSED_MOE_L0_M*"
latency_us = float(event["dur"])
```

CPU annotation 시간은 사용하지 않는다. Trace의 `ts`와 `dur` 단위는
microsecond이므로 CSV를 만들 때 millisecond로 변환한다.

각 backend/shape에서 다음 통계를 계산한다.

- 표본 수: 반드시 50
- mean
- median
- minimum
- maximum
- standard deviation
- p90
- token throughput: `num_tokens / fused_moe_seconds`
- FlashInfer ON 대비 OFF speedup
- 가장 빠른 backend 대비 slowdown

50개 표본만 사용하므로 p99는 보고하지 않는다. backend 순위의 대표값은
`fused_moe_gpu_ms_median`으로 하고, mean과 p90을 함께 제시한다.

결과 CSV의 최소 열은 다음과 같다.

```text
timestamp_utc
slurm_job_id
node
gpu_name
gpu_uuid
driver_version
torch_version
vllm_version
vllm_git_tag
flashinfer_version
experiment_id
backend_requested
backend_actual
activation_dtype
flashinfer_autotune
server_startup_seconds
seed
prompt_sha256
layer
num_tokens
warmup
repeat
router_event_count
fused_moe_event_count
router_gpu_ms_mean
router_gpu_ms_median
router_gpu_ms_p90
fused_moe_gpu_ms_mean
fused_moe_gpu_ms_median
fused_moe_gpu_ms_p90
fused_moe_tokens_per_second
moe_total_gpu_ms_mean
moe_total_gpu_ms_median
router_percent_mean
client_ms_mean
client_ms_median
status
error
trace_files
metadata_file
server_log
```

## 15. correctness 검증

현재 자동화한 검증은 다음과 같다.

- deterministic prompt hash
- prompt token 수
- router/FusedMoE event 수
- GPU annotation duration이 양수인지 여부
- 명시적으로 요청한 backend의 서버 초기화 성공 여부

출력 tensor의 NaN/Inf, Emulation 기준 오차, expert routing histogram은 timing
코드에 synchronization을 추가할 수 있으므로 후속 correctness-only script에서
분리해 측정한다. W4A16 backend끼리는 직접 비교하고, FlashInfer W4A8은 별도
tolerance를 사용해야 한다.

서빙 timing 경로에서는 layer 0 tensor를 결과 파일로 저장하지 않는다.

## 16. 실행 단계

### 단계 A: 환경·backend preflight

1. `sbatch`로 RTX PRO 6000 한 장 할당
2. vLLM package와 source tag가 모두 `0.23.0`/`v0.23.0`인지 확인
3. `auto`, Marlin, Humming, FlashInfer CUTLASS, Emulation 서버를 각각 한 번 기동
4. `/health`와 짧은 생성 요청 확인
5. requested backend와 actual expert implementation 기록
6. 실패 backend는 fallback 없이 제외 사유 기록

### 단계 B: instrumentation 검증

1. Marlin, `num_tokens=128`만 실행
2. warmup 2회, repeat 3회로 빠르게 trace 생성
3. router event 3개와 FusedMoE event 3개 확인
4. GPU annotation의 `dur`가 양수인지 확인
5. prompt token 수와 annotation의 `M`이 모두 128인지 확인

### 단계 C: 작은 pilot

다음 세 조건과 shape만 먼저 실행한다.

```text
backend: marlin, flashinfer tune OFF, flashinfer tune ON
num_tokens: 1, 32, 256, 1024, 8192
warmup: 20
repeat: 50
```

trace 크기, 전체 실행시간, FlashInfer autotune 시간, event 누락 여부를 확인한다.

### 단계 D: 전체 sweep

pilot을 통과하면 6개 experiment ID와 19개 `num_tokens` 전체를 실행한다.
Emulation이 과도하게 느리면 성능 backend job과 분리하되 입력과 측정 규칙은
바꾸지 않는다.

### 단계 E: 재측정

상위 두 backend의 median 차이가 5% 이내이면 새 서버 프로세스로 3회 독립
재실행한다. 실행 순서를 번갈아 GPU 온도와 시간 순서 효과를 줄인다.

## 17. 최종 결과 해석

최종 보고서는 다음 세 그래프를 기본으로 한다.

1. `num_tokens` 대비 FusedMoE median latency
2. `num_tokens` 대비 token throughput
3. FlashInfer AutoTuner ON/OFF speedup

판단 기준은 다음과 같다.

- W4A16에서 가장 느린 backend: Marlin/Humming/Emulation 내부 비교
- W4A16에서 최적화 우선순위: 큰 `num_tokens`와 작은 `num_tokens` 구간을 나눠 판단
- FlashInfer tuner 효과: 동일 W4A8 조건의 ON/OFF만 사용
- Router가 전체 MoE에서 차지하는 비율:
  `router / (router + fused_moe)`
- 개선 대상 선정: latency가 크고 실제 workload에서 자주 나타나는 token 구간을
  동시에 고려

단순히 가장 느린 한 점만 보고 개선 대상을 정하지 않는다. 작은 token 구간은
kernel launch 및 dispatch overhead, 큰 token 구간은 expert GEMM과 routing
불균형의 영향이 커질 수 있으므로 구간별 병목을 분리해 해석한다.

## 18. 구현 파일

실험 코드는 다음과 같이 구성되어 있다.

```text
aso-internship/vLLM/MoE/
├── README.md
├── setup_vllm023_editable.sh      # vLLM wheel 복구 및 Python source patch
├── setup_flashinfer_cuda.sh        # 일관된 CUDA 13.0 JIT toolkit 별도 설치
├── setup_runtime_env.sh            # EXP_ROOT와 SM120용 CUDA 13 JIT 환경
├── patches/
│   └── gpt_oss_layer0_moe_profile.patch
├── run_moe_requests.py            # seed 고정, 20 warmup, 50 repeat, profiler 제어
├── run_backend_experiments.py     # backend별 서버 재시작과 전체 shape sweep
├── analyze_moe_traces.py          # GPU user annotation 집계
├── visualize_moe_results.py       # 평균 latency 기반 결과 시각화
├── configs/experiments.yaml       # backend와 num_tokens matrix
├── slurm/run_moe_benchmark.sbatch
├── tests/test_benchmark_scripts.py
└── results/
    └── .gitignore
```

대용량 trace와 server log의 원본은 Git 저장소가 아니라 다음 경로에 둔다.

```text
/lustre/hybyun0207/vllm023-moe/runs/
```

## 19. 실험 시작 전 체크리스트

- [ ] SLURM으로 RTX PRO 6000 한 장을 할당받았다.
- [ ] `$CUDA_VISIBLE_DEVICES`를 직접 설정하지 않았다.
- [ ] package version이 `vllm==0.23.0`이다.
- [ ] source checkout이 `v0.23.0`이다.
- [ ] `EXP_ROOT`가 `/lustre/hybyun0207/vllm023-moe`이다.
- [ ] FlashInfer가 보는 `nvcc`가 CUDA 12.9 이상이다.
- [ ] `nvcc`, `ptxas`, NVVM과 CUDA runtime header가 CUDA 13.0으로 일치한다.
- [ ] `cublasLt.h`, `curand_kernel.h`, unversioned CUDA library 링크가 있다.
- [ ] `$VLLM_ENV/bin/ninja`가 실행 가능하고 `PATH`에서 검색된다.
- [ ] `amd-quark`의 `quark.torch.kernel.mx`를 import할 수 있다.
- [ ] `FLASHINFER_WORKSPACE_BASE`가 `flashinfer-cuda130-full`을 가리킨다.
- [ ] GPU capability가 `(12, 0)`이다.
- [ ] 수정한 `gpt_oss.py`가 실제 서버에서 import된다.
- [ ] CUDA Graph와 torch.compile이 꺼져 있다.
- [ ] prefix caching이 꺼져 있다.
- [ ] `max-num-batched-tokens=8192`이다.
- [ ] 요청 script는 `num_tokens`만 사용하고 `batch_size` alias가 없다.
- [ ] seed 42의 동일 token ID를 모든 backend에 사용한다.
- [ ] layer 0 router와 FusedMoE annotation이 분리되어 있다.
- [ ] 각 shape에서 warmup 20회와 측정 50회를 수행한다.
- [ ] 각 trace에 router/FusedMoE event가 각각 50개 있다.
- [ ] FlashInfer ON/OFF는 서로 다른 서버 프로세스다.
- [ ] requested backend와 actual expert class가 일치한다.
- [ ] trace와 모델은 `/lustre`에 저장한다.
- [ ] 실험 종료 후 서버와 GPU allocation을 정리한다.

## 20. 실행 방법

### 20.1 로그인 노드에서 최초 1회 설정

vLLM 0.23.0 wheel 복구와 설치된 GPT-OSS Python source patch는
다운로드·설치 작업이므로 login node에서 실행한다. 스크립트 이름에는 과거의
`editable` 표현이 남아 있지만 현재 구현은 native extension 보존을 위해
editable 설치를 사용하지 않는다.

```bash
cd /home/hybyun0207/aso-internship/vLLM/MoE
bash setup_vllm023_editable.sh
```

스크립트는 다음 조건을 검증한다.

- `/lustre/hybyun0207/vllm023-moe/src/vllm`이 정확히 `v0.23.0` tag인가?
- layer-0 annotation patch가 이미 적용됐는가?
- `/lustre/hybyun0207/vllm023-moe/env`에서 import되는 `gpt_oss.py`에 layer-0
  annotation이 적용됐는가?
- `vllm._C`와 `vllm._moe_C` native extension을 import할 수 있는가?
- package version이 `vllm==0.23.0`인가?
- FlashInfer JIT용 CUDA 13.0 compiler와 header가 별도 toolkit에 설치됐는가?

### 20.2 GPU 없이 명령 구성 확인

```bash
export EXP_ROOT=/lustre/hybyun0207/vllm023-moe

"$EXP_ROOT/env/bin/python" \
  run_backend_experiments.py \
  --model-path "$EXP_ROOT/models/gpt-oss-20b" \
  --run-root /tmp/gptoss-moe-dry-run \
  --vllm-source "$EXP_ROOT/src/vllm" \
  --vllm-bin "$EXP_ROOT/env/bin/vllm" \
  --backends marlin_w4a16,flashinfer_cutlass_w4a8_tune_on \
  --num-tokens 1,128 \
  --warmup 2 \
  --repeat 3 \
  --dry-run
```

### 20.3 할당한 GPU에서 smoke test

`srun`으로 받은 GPU node에서는 먼저 runtime 환경을 설정한다. `EXP_ROOT`가
설정되지 않은 상태에서 `RUN_ROOT="$EXP_ROOT/smoke-..."`를 사용하면 경로가
`/smoke-...`가 되어 permission 오류가 발생한다.

```bash
source /home/hybyun0207/aso-internship/vLLM/MoE/setup_runtime_env.sh

export RUN_ROOT="$EXP_ROOT/runs/smoke-${SLURM_JOB_ID:-interactive}-$(date +%Y%m%d-%H%M%S)"
echo "$RUN_ROOT"

"$VLLM_ENV/bin/python" \
  "$PROJECT_DIR/run_backend_experiments.py" \
  --config "$PROJECT_DIR/configs/experiments.yaml" \
  --model-path "$MODEL_PATH" \
  --run-root "$RUN_ROOT" \
  --vllm-source "$VLLM_SRC" \
  --vllm-bin "$VLLM_ENV/bin/vllm" \
  --backends flashinfer_cutlass_w4a8_tune_off \
  --num-tokens 128 \
  --warmup 2 \
  --repeat 3
```

첫 FlashInfer 실행은 SM120용 보조 커널을 JIT compile하므로 오래 걸릴 수 있다.
검증 당시 빈 workspace에서 AutoTuner OFF 서버 시작은 약 19분, 같은 workspace를
재사용한 AutoTuner ON 서버 시작은 약 68초였다. 이 시간은 요청별 MoE latency
측정에 포함하지 않고 `server_startup_seconds`로 별도 기록한다.

### 20.4 작은 pilot 제출

```bash
cd /home/hybyun0207/aso-internship/vLLM/MoE

MOE_BACKENDS=marlin_w4a16,flashinfer_cutlass_w4a8_tune_off,flashinfer_cutlass_w4a8_tune_on \
NUM_TOKENS=1,32,256,1024,8192 \
WARMUP=20 \
REPEAT=50 \
sbatch slurm/run_moe_benchmark.sbatch
```

다른 RTX PRO 6000 partition을 사용하려면 sbatch 명령행에서 덮어쓴다.

```bash
sbatch -p gigabyte_pro6000 slurm/run_moe_benchmark.sbatch
```

### 20.5 전체 sweep 제출

환경변수 override 없이 제출하면 YAML에 정의한 6개 backend와 19개
`num_tokens` 전체를 실행한다.

```bash
sbatch slurm/run_moe_benchmark.sbatch
```

작업 확인:

```bash
squeue -u "$USER"
tail -f results/slurm-JOB_ID.out
```

결과는 두 위치에 저장된다.

```text
/lustre/hybyun0207/vllm023-moe/runs/job-JOB_ID/
  ├── backend별 server log와 trace
  ├── metadata/
  ├── manifests/
  └── moe_kernel_results.csv

aso-internship/vLLM/MoE/results/
  ├── slurm-JOB_ID.out
  └── moe_kernel_results-JOB_ID.csv
```

## 21. 결과 시각화

`visualize_moe_results.py`는 `fused_moe_gpu_ms_mean`을 기준으로 다음
그래프를 PNG로 생성한다.

- FusedMoE 평균 GPU latency(50개 GPU event의 arithmetic mean)
- 평균 latency로 다시 계산한 FusedMoE throughput
- Marlin 대비 평균 latency speedup
- FlashInfer AutoTuner ON/OFF 평균 latency와 비율
- 대표 token 크기의 router/FusedMoE 평균 시간 구성

정규화 latency heatmap은 생성하지 않는다. Throughput, speedup,
AutoTuner 비율도 모두 median이 아닌 `fused_moe_gpu_ms_mean`에서
계산하여 그래프 간 기준을 통일한다.

```bash
RUN_ROOT=/lustre/hybyun0207/vllm023-moe/runs/job-1907179

/lustre/hybyun0207/vllm023-moe/env/bin/python \
  visualize_moe_results.py \
  --input "$RUN_ROOT/moe_kernel_results.csv"
```

`--output-dir`을 생략하면 입력 CSV의 run 디렉터리 이름을 사용해
다음 경로에 저장한다.

```text
/home/hybyun0207/aso-internship/vLLM/MoE/results/plots/job-1907179/
```

`run_moe_benchmark.sbatch`로 실행한 완전한 실험은 trace 분석이
성공하면 위 경로에 그래프를 자동 생성한다. 필요하면
`MOE_PLOT_DIR` 환경변수로 저장 경로를 덮어쓸 수 있다.

생성되는 파일은 다음과 같다.

```text
results/plots/job-1907179/
├── fused_moe_latency_mean.png
├── fused_moe_throughput_mean.png
├── speedup_vs_marlin_mean.png
├── flashinfer_autotune_mean.png
├── router_fused_breakdown_mean.png
└── kernel_winners_mean.csv
```
