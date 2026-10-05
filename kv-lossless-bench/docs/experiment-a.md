# 실험 A 실행 가이드

[전체 실험 계획으로 돌아가기](../README.md)

## 현재 구현·검증 상태

실험 A의 **KV 수집기·오프라인 분석기·그래프·RTX 4090용 Slurm 스크립트**를 구현했습니다. 실제 Llama GPU 실행 결과는 아직 수집하지 않았으며, GPU codec과 실험 B~D는 계획 단계입니다. 초기 대상은 로컬 Llama-3.1-8B-Instruct와 RTX 4090 1장입니다.

분석 로직과 실행 식별자 테스트를 제공하며, 작은 CPU Llama 모델의 BF16·SDPA 캐시 동작도 검증했습니다. RTX 4090에서 실제 모델을 사용한 수집은 아직 검증하지 않았습니다.

## 실행 방법

### 전용 가상환경

```bash
cd /home/hybyun0207/aso-internship/kv-lossless-bench
bash scripts/setup_env.sh
source .venv/bin/activate
```

`setup_env.sh`는 Python 3.11로 독립 `.venv`를 만들고 GPU용 Torch와 분석 의존성을 설치합니다. 기본 Python 실행 파일은 기존 로컬 Python 3.11 배포본을 사용하며, 필요하면 `KV_BASE_PYTHON`으로 바꿀 수 있습니다. 다른 실험의 site-packages는 상속하지 않습니다. 설치에는 패키지 서버 접속이 필요하고, GPU 수집은 로컬 모델·tokenizer만 사용합니다.

설치 후 `pip check`를 수행하고 실제 의존성 목록을 `requirements.lock.txt`에 기록합니다. Slurm 스크립트는 활성화 여부에 관계없이 이 프로젝트의 `.venv/bin/python`을 사용합니다. CUDA 13.0 Torch와 GPU 노드 driver의 호환성은 실제 GPU 실행 시 확인합니다.

### 준비

- 모델: `/lustre/hybyun0207/models/Llama-3.1-8B-Instruct`
- Python: 프로젝트 전용 `.venv/bin/python`. 다른 실험 환경의 패키지를 공유하지 않습니다. Torch 2.11.0(CUDA 13.0), Transformers 4.57.6, NumPy 2.3.5를 기준으로 설치합니다.
- **해당 모델과 일치하는 로컬 tokenizer가 필요합니다.** 현재 모델 폴더에는 tokenizer 파일이 없습니다. `KV_TOKENIZER`에 tokenizer 디렉터리를 지정합니다. 스크립트는 자동 다운로드하지 않습니다.
- 기본 입력은 직접 작성한 짧은 문장·코드·문서 예제 6개입니다. 512 tokens를 채우도록 반복하므로 **동작 확인용이며 대표성 있는 연구 결과가 아닙니다.** 본 실험에서는 별도 데이터로 교체합니다.

```bash
cd /home/hybyun0207/aso-internship/kv-lossless-bench
export KV_TOKENIZER=/path/to/matching/llama-3.1-tokenizer
sbatch slurm/run_experiment_a.sbatch
```

기본값은 batch 1, BF16, SDPA, prefill 512 tokens, decode 128회입니다. Greedy decoding으로 고정 횟수의 forward를 수행하며 EOS가 나와도 중단하지 않습니다. **마지막에 선택된 토큰이 아니라 실제 decode forward에 입력된 128개 토큰의 KV**를 저장합니다. 프롬프트는 BOS와 원문 token으로 구성하며 chat template은 적용하지 않습니다.

GPU에서 모든 forward를 마치고 `torch.cuda.synchronize()`한 뒤에만 최종 KV를 CPU로 복사합니다. 복사는 레이어별로 수행하고, BF16을 float로 변환하지 않고 `uint16` 비트 패턴으로 저장합니다. 수집 실행의 latency는 추론 성능으로 보고하지 않습니다.

실제 입력 JSONL은 각 줄에 `id`, `split` (`calibration` 또는 `evaluation`), `category`, `text`를 포함합니다. 두 split을 모두 준비하고 입력은 지정한 prefill 길이 이상이어야 합니다. 서로 다른 split에 같은 문서를 중복하지 않습니다.

```bash
export KV_PROMPTS=/path/to/prompts.jsonl
export KV_PREFILL_TOKENS=2048
export KV_DECODE_TOKENS=128
sbatch slurm/run_experiment_a.sbatch
```

현재 prefill 길이는 64의 배수로 제한하여 prefill/decode 경계와 압축 타일 경계를 맞춥니다. Context 확대는 첫 실행의 peak memory를 확인한 뒤 진행합니다.

### 결과와 분석 범위

결과는 **한국 시간(Asia/Seoul), 날짜·시분초, 실험 종류, Slurm job 번호**를 포함한 경로에 저장됩니다. 날짜 형식은 `YYMMDD_HHMMSS`입니다. 예시의 `250902_131205`는 2025년 9월 2일 13시 12분 5초를 의미합니다.

```text
results/250902_131205_exp-a_job-12345/
├── 250902_131205_exp-a_job-12345.log
├── run_metadata.json
├── kv/
└── analysis/
```

날짜는 Slurm 작업 시작 시 한 번 결정하며 같은 실행에서 모든 단계가 공유합니다. 작업 재시작 시에는 새 시작 시간이 적용됩니다. 동일 식별자 경로는 덮어쓰지 않습니다. Slurm 밖에서 직접 수집하면 job 자리에 `local-<PID>`를 명시합니다.

Slurm의 `#SBATCH`는 날짜 명령을 확장하지 않으므로 초기 스케줄러 로그는 `slurm-bootstrap-<job_id>.out`이고, 스크립트가 시작되면 위의 날짜·실험·job 포함 로그로 출력을 전환합니다. 시작 전 오류는 bootstrap 로그에서 확인합니다.

CSV에는 `run_id`, `timestamp`, `timezone`, `job_id`, `experiment` 열이 포함되고 보고서와 그래프에도 실행 식별자가 표시됩니다. JSON/NPZ/KV 등 부속 파일은 해당 실행 디렉터리 아래 보관합니다. 결과만 복사할 때도 식별자가 있는 상위 경로 또는 `run_metadata.json`을 함께 보존합니다.

| 파일 | 내용 |
|---|---|
| `run_metadata.json`, `analysis/run_metadata.json` | 날짜·timezone·job 번호·실험 종류·실행 식별자 |
| `kv/run.json` | 실행 식별자, 모델 설정·hash, GPU·버전, 수집 정책 |
| `kv/<sample>/sample.json` | 입력·decode token IDs, split, KV shape/stride, peak memory |
| `kv/<sample>/layer_XX_K.npy`, `layer_XX_V.npy` | `[kv_head, token, head_dim]`의 BF16 uint16 비트 |
| `kv/COMPLETE` | 전체 수집 완료 표시; 분석기는 불완전한 수집을 거부 |
| `analysis/calibration.json` | calibration에서만 구한 레이어별 K/V 고정 지수 구간 |
| `analysis/distributions.csv` | 레이어·head·phase별 p7, top7, entropy, 고정 구간 커버리지 |
| `analysis/histograms.npz` | 동일 단위의 256-bin histogram |
| `analysis/blocks.csv` | 64×64 타일별 크기·지수 구간·fallback 여부 |
| `analysis/storage.csv`, `report.md` | 원본 대비 저장량과 평가 데이터 요약 |
| `analysis/<run_id>_savings_*.png`, `<run_id>_tile_p7.png` | 평가 데이터 heatmap·p7 분포; matplotlib가 있을 때 생성 |

현재 저장량 분석은 **레이어별 K/V 고정 구간**과 **64×64 타일별 동적 구간**을 비교합니다. 기존 ZipServ 지수 선택 heuristic, 모델 전역 고정 구간, head별 고정 구간은 후속 확장입니다. 동적 선택의 계산 시간은 실험 A에서 추정하지 않습니다.

크기 모델은 head마다 `[token, head_dim]`을 64×64로 자르고 각 타일을 독립 레코드로 저장한다고 가정합니다. 기존 ZipServ 전체 행렬 포맷의 바이트 수와 동일하다는 주장은 하지 않습니다.

- 타일당 비트맵 1,536 B, median offsets 32 B, global start/end offsets 16 B, 제안 헤더 16 B.
- 부호·가수 payload와 원본값 payload는 각각 16 B 정렬.
- 압축이 불리하면 원본 BF16 + 16 B 헤더로 fallback.
- 미완성 토큰 행과 dimension tail은 원본 BF16로 유지하고 head/phase별 잔여 영역에 16 B 헤더를 계산.
- 동적 구간은 최솟값 `s`를 0~249에서 선택합니다. 향후 디코더는 지수 0 구간도 정확하게 처리해야 하며, 기존 `uint8 start_exp=s-1` 표현을 그대로 사용하는 것으로 가정하지 않습니다.
- GPU allocator, fragmentation, 임시 버퍼, 실제 compression throughput은 이 분석에 포함되지 않습니다.

아래 명령은 프로젝트 루트(`kv-lossless-bench/`)에서 실행합니다. 오프라인 분석·테스트는 GPU 없이 실행할 수 있습니다.

```bash
PYTHON=.venv/bin/python
RUN_ID=250902_131205_exp-a_job-12345  # 실제 수집 결과 식별자로 변경
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" src/analyze_kv.py --input "results/$RUN_ID/kv" --output results/reanalysis
"$PYTHON" src/plot_analysis.py --input "results/reanalysis/$RUN_ID/analysis"
```

수집기의 `--output`은 결과의 **상위 디렉터리**입니다. 내부에 자동으로 `<run_id>/kv/`를 만듭니다. 분석기도 `--output` 아래에 `<원본 run_id>/analysis/`를 만듭니다. 오프라인 재분석은 현재 시각이나 다른 job으로 원본 출처를 바꾸지 않습니다. 출처 정보가 없는 이전 형식의 `run.json`은 거부합니다.

## 지수 값별 proportion 막대그래프

저장된 `histograms.npz`와 `distributions.csv`로 그래프를 생성하므로 GPU 재실험이나 KV 재수집은 필요하지 않습니다.

```bash
cd /home/hybyun0207/aso-internship/kv-lossless-bench
RUN_ID=261005_022642_exp-a_job-2392257
.venv/bin/python src/plot_exponents.py --input "results/$RUN_ID/analysis"
# Prefill 또는 decode 신규 KV만 따로 볼 때:
.venv/bin/python src/plot_exponents.py --input "results/$RUN_ID/analysis" --phase decode
```

- 기본값은 **평가 데이터만**, prefill과 decode를 합산합니다. `--split calibration`, `--phase prefill`도 지원합니다.
- 가로축은 BF16에 저장된 **8bit 지수 필드 값(0~255)**입니다. 실제 텐서 값이나 bias 127을 뺀 지수가 아닙니다.
- 세로축은 `해당 지수 원소 수 / 선택한 데이터 전체 원소 수`입니다. K/V는 각각 전체 원소 수를 분모로 정규화합니다.
- `head=-1` histogram만 사용하여 개별 head 데이터와 중복 집계하지 않습니다. 레이어·샘플의 비율을 단순 평균하지 않고 원소 수를 먼저 합산합니다.
- 중심 확대본은 각 패널의 누적 비율 0.05%~99.95% 구간을 포괄하도록 표시합니다. 생략된 꼬리를 재정규화하지 않으며 실제 표시 비율(`Shown mass`)을 기재합니다. 전체 0~255 지수의 수치는 CSV에 보존합니다.
- 좌상단 `Entropy`는 지수 분포의 Shannon entropy, `Original: 8 bits`는 원본 지수 필드 크기입니다. `Compression`은 같은 데이터의 **고정 지수 구간 TCA-TBE 방식에 대한 예상 BF16 KV 저장량 절감률**이며, 메타데이터·패딩을 포함합니다. entropy/8이 아니라 `1 - 합산 저장 바이트 / 합산 원본 바이트`로 계산합니다.

그래프는 `analysis/exponent_plots/`에 **PNG 4개만** 생성됩니다. 각 그림은 K/V 두 패널을 포함하며 레이어 번호는 0부터 시작합니다. 파일 이름에는 원본 날짜·실험 종류·job 번호와 선택한 split·phase를 포함합니다.

| 파일 끝부분 | 내용 |
|---|---|
| `all_layers.png` | 전체 32개 레이어를 합산한 K/V 분포 |
| `layer_01.png` | 1번 레이어 K/V 분포 |
| `layer_20.png` | 20번 레이어 K/V 분포 |
| `layer_31.png` | 31번 레이어 K/V 분포 |

검증용 `*_counts.csv`와 `*_metadata.json`은 그림 폴더 밖인 `analysis/`에 저장합니다. PDF·SVG·다른 레이어의 그림은 생성하지 않습니다.

## 디렉터리 역할

디렉터리의 역할은 다음과 같습니다. 현재 실험 A용 `src/`, `tests/`, 예제 입력과 `slurm/`을 구현했으며, `csrc/`의 GPU codec은 후속 작업입니다.

| 경로 | 예정된 내용 |
|---|---|
| `src/` | KV 수집, 분포 분석, benchmark runner, 결과 집계 |
| `csrc/` | CUDA encoder/decoder, C++ binding, 후속 attention 통합 |
| `tests/` | 비트 단위 round-trip, 경계·패딩·스트림 검증 |
| `data/` | 입력 manifest, KV 샘플, calibration 정보 |
| `results/` | 실행 설정, 원시 측정값, 그래프, profiler 결과 |
| `scripts/` | 전용 가상환경 생성·설치 |
| `slurm/` | GPU 작업 제출·로그·결과 경로 설정 |
| `docs/` | 포맷·레이아웃 명세, 실험별 해석 |
