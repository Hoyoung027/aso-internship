# FlashInfer Attention Binding/CUBIN Experiment

RTX 3090(SM86)의 FlashInfer single-decode/single-prefill을 대상으로 다음 세
실행 경로를 비교하기 위한 실험 디렉터리입니다.

1. legacy PyTorch `torch.ops` binding
2. TVM-FFI binding
3. TVM-FFI launcher + 사전 컴파일 SM86 CUBIN

이 실험의 주 비교 대상은 attention 알고리즘이 아니라 **호출 계층과 kernel
배포 방식**입니다. 세 경로에서 dtype, head 수, head dimension, mask와 kernel
specialization을 동일하게 유지해야 합니다.

## 디렉터리

```text
flashinfer-attention/
├── benchmark.py              # PyTorch/TVM-FFI 공통 benchmark worker
├── compare_outputs.py        # 서로 다른 구현의 저장 결과 비교
├── prepare_cubin_sources.py  # FlashInfer JIT source/build.ninja 추출
├── build_cubin.py            # NVCC object 명령을 CUBIN 명령으로 변환
├── setup_variants.sh         # binding 전환 전/후 FlashInfer worktree 생성
├── CODE_WALKTHROUGH.md       # 각 코드와 측정 흐름 상세 해설
├── cubin/
│   └── README.md             # CUBIN launcher 구현 체크리스트
├── artifacts/                # 생성 source/CUBIN (git 제외)
└── results/                  # JSONL 및 tensor 결과 (git 제외)
```

코드를 처음 읽는 경우 [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md)를 먼저
보는 것을 권장합니다.

## 0. GPU 안전 규칙

모든 Python worker는 `torch`를 import하기 전에 아래 환경을 검사합니다.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

이 상태에서 물리 GPU 1은 프로세스 내부의 `cuda:0`입니다. 코드에서
`cuda:1`을 사용하지 않습니다.

## 1. 비교용 FlashInfer worktree

FlashInfer의 PyTorch binding -> TVM-FFI 전환 직전/직후 커밋은 다음입니다.

```text
PyTorch binding: ea569640
TVM-FFI:        86d3e136
```

서버의 FlashInfer checkout을 기준으로 worktree를 만듭니다.

```bash
./setup_variants.sh /path/to/flashinfer /path/to/fi-variants
```

각 worktree는 별도 virtualenv를 사용하는 것이 안전합니다. 가능한 한 같은
Python, PyTorch, CUDA toolkit을 사용하되 각 커밋의 dependency 제약을
기록합니다.

각 worktree에서 다음 설치를 수행합니다.

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install --no-build-isolation -e . -v
```

benchmark 명령의 `python`은 각 worktree의 `.venv/bin/python`을 직접
지정하는 편이 명확합니다. `PYTHONPATH`만 바꾸는 방식은 FlashInfer data
symlink와 compiled dependency가 준비되지 않을 수 있습니다.

더 엄밀한 최종 실험에서는 과거 커밋 비교 대신 하나의 FlashInfer source에
legacy PyTorch binding을 port해야 합니다. 위 worktree 비교는 구현과 측정
pipeline을 먼저 검증하기 위한 1단계입니다.

## 2. PyTorch/TVM-FFI benchmark

공통 설정은 FP16, Q heads 32, KV heads 8, head dimension 128, NHD입니다.
`benchmark.py`는 public API의 tensor allocation을 제외하기 위해 미리 할당한
workspace/output으로 내부 `module.run()` wrapper를 반복 호출합니다.

PyTorch binding worktree:

```bash
CUDA_VISIBLE_DEVICES=1 \
/path/to/fi-variants/pybind/.venv/bin/python benchmark.py \
  --binding pybind --kernel decode \
  --lengths 128 512 2048 8192 \
  --output results/pybind-decode.jsonl \
  --save-tensors results/pybind-decode
```

TVM-FFI worktree:

```bash
CUDA_VISIBLE_DEVICES=1 \
/path/to/fi-variants/tvm-ffi/.venv/bin/python benchmark.py \
  --binding tvm-ffi --kernel decode \
  --lengths 128 512 2048 8192 \
  --output results/tvm-decode.jsonl \
  --save-tensors results/tvm-decode
```

Prefill:

```bash
CUDA_VISIBLE_DEVICES=1 \
/path/to/fi-variants/tvm-ffi/.venv/bin/python benchmark.py \
  --binding tvm-ffi --kernel prefill \
  --lengths 128 512 1024 2048 4096 \
  --output results/tvm-prefill.jsonl \
  --save-tensors results/tvm-prefill
```

저장 결과 비교:

```bash
python compare_outputs.py \
  results/pybind-decode \
  results/tvm-decode
```

각 kernel/구현은 가능하면 별도 프로세스로 5회 이상 실행합니다. 결과에는
다음 시간이 별도로 기록됩니다.

- import 시간
- module build/load 시간
- 첫 synchronized call 시간
- warm host enqueue 시간
- CUDA-event GPU 시간

## 3. SM86 CUBIN source 준비

최신 TVM-FFI worktree에서 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=1 \
FLASHINFER_CUDA_ARCH_LIST=8.6 \
/path/to/fi-variants/tvm-ffi/.venv/bin/python prepare_cubin_sources.py \
  --output artifacts/generated
```

이 명령은 FlashInfer의 generator를 그대로 사용해 다음을 준비합니다.

- FP16/head-dim 128 single-decode specialization
- FP16/head-dim 128 FA2 single-prefill specialization
- 생성된 config/source
- 원본 `build.ninja`
- JIT spec과 선택 source를 기록한 `manifest.json`

## 4. CUBIN 명령 확인 및 빌드

먼저 변환될 NVCC 명령을 출력만 합니다.

```bash
python build_cubin.py \
  --manifest artifacts/generated/manifest.json \
  --kind decode

python build_cubin.py \
  --manifest artifacts/generated/manifest.json \
  --kind prefill
```

명령과 `sm_86` target을 확인한 후 실행합니다.

```bash
python build_cubin.py \
  --manifest artifacts/generated/manifest.json \
  --kind decode --execute

python build_cubin.py \
  --manifest artifacts/generated/manifest.json \
  --kind prefill --execute
```

`build_cubin.py`는 생성된 `build.ninja`에서 FlashInfer가 실제 사용하는 NVCC
명령을 가져오므로 include/macro/optimization flag를 손으로 재작성하지
않습니다. Prefill은 causal mask인 `single_prefill_kernel_mask_1.cu`를
선택합니다.

FlashInfer translation unit을 `--cubin`으로 바꾸는 과정은 CUDA/FlashInfer
버전에 따라 device-link 또는 symbol export 문제가 드러날 수 있습니다. 이
단계의 실패를 우회해서 숨기지 말고, 출력된 명령과 오류를 기준으로 kernel
entry를 분리해야 합니다.

## 5. CUBIN launcher

CUBIN이 생성되면 `cuobjdump --list-elf` 결과에서 실제 kernel symbol을
확인하고 `cubin/README.md` 순서대로 TVM-FFI launcher를 구현합니다. Decode의
KV 길이가 256보다 크면 partition kernel과 merge kernel이 모두 필요하므로,
첫 성공 지점은 다음처럼 제한합니다.

```text
Decode:  kv_len=128
Prefill: qo_len=kv_len=128, causal
```

이 두 점의 정확성을 확인한 뒤 긴 Decode와 여러 Prefill CTA tile로 범위를
확장합니다.

## 결과 해석

동일 SASS/launch configuration이면 warm GPU 시간은 세 경로에서 거의 같아야
합니다. Decode에서는 짧은 kernel 때문에 binding overhead가 상대적으로 잘
보이고, Prefill에서는 GPU 계산 시간이 이를 덮을 가능성이 큽니다. CUBIN의
주 이점은 steady-state kernel 가속이 아니라 online NVCC JIT 제거입니다.
