# KV Lossless Bench

ZipServ의 **TCA-TBE(Tensor-Core-Aware Triple Bitmap Encoding)**를 BF16 KV-cache에 적용할 수 있는지, 압축·압축 해제 비용과 GPU 자원 경합이 prefill/decode에 어떤 영향을 주는지 검증하는 실험 프로젝트입니다.

이 문서는 전체 실험 계획을 정리합니다. 실행 방법과 현재 구현 범위는 [실험 A 실행 가이드](docs/experiment-a.md)를 참고하세요.

## 1. 연구 질문과 실험 범위

핵심 질문은 **“어떤 조건에서 압축으로 줄어드는 메모리 비용보다 encoding/decoding 및 자원 경합 비용이 커지는가?”**입니다. 성능 저하를 전제하지 않고, 이득과 손해가 발생하는 조건을 모두 측정합니다.

1. 실제 KV-cache의 BF16 지수 분포가 TCA-TBE에 적합한가?
2. GPU encoding과 decoding의 지연시간·처리량은 얼마인가?
3. codec을 prefill/decode와 동시에 실행하면 양쪽 작업이 얼마나 느려지는가?
4. 실제 KV 읽기·쓰기 경로에 통합했을 때 메모리 절감이 추가 비용을 상쇄하는가?

초기 범위는 **BF16 KV-cache, 단일 GPU, 일반적인 full attention**입니다. FP16/FP8 KV, sliding-window attention, 다중 GPU 통신은 별도 확장으로 다룹니다.

실험은 다음 순서로 진행합니다.

| 단계 | 실험 | 답하려는 질문 |
|---|---|---|
| A | 실제 KV 데이터 분석 | 압축 가능성과 저장량 |
| B | GPU codec 단독 벤치마크 | encoding/decoding 자체 비용 |
| C | 추론과 codec 동시 실행 | 자원 경합과 스케줄링 영향 |
| D | 실제 KV 경로 통합 | 메모리·추론 성능의 최종 손익 |

## 2. KV-cache에 적용할 때의 전제

일반적인 autoregressive full attention에서 새 토큰의 K/V는 캐시에 추가되고, 과거 K/V는 이후 decode step에서 반복해서 읽힙니다.

- **Encoding:** 새로 생성된 KV를 저장할 때 수행합니다. 이미 압축된 과거 KV 전체를 매 step 재압축할 필요는 없습니다.
- **Decoding:** attention이 압축 KV를 읽을 때 수행합니다. 복원본을 계속 유지하지 않는다면 같은 KV를 여러 step에서 반복 복원합니다.
- **복원본 유지:** 반복 decoding은 줄일 수 있지만, 압축본과 복원본을 함께 유지하는 메모리 비용이 발생합니다.

따라서 신규 KV 쓰기량과 과거 KV 읽기량을 구분합니다. 다음 두 저장 정책을 비교합니다.

| 정책 | 동작 | 주요 측정 대상 |
|---|---|---|
| 즉시 압축 | 새 토큰 KV를 바로 압축 | 작은 작업의 launch·메타데이터 비용 |
| 블록 완성 후 압축 | 채워지는 블록은 BF16으로 유지하고 완성된 블록을 압축 | 비용 상각, 일시적인 비압축 메모리 |

ZipServ의 64×64 글로벌 타일과 decode의 토큰 하나 추가는 직접 대응하지 않습니다. 캐시 페이지, 압축 블록, CUDA thread block은 서로 다른 단위이며, 토큰·head dimension을 압축 타일에 매핑하는 방식을 명시해야 합니다. 즉시 압축에 작은 포맷을 사용한다면 기존 ZipServ 포맷과의 차이도 기록합니다.

## 3. 참고할 ZipServ 구현

로컬 참조 저장소: [`../../ZipServ_ASPLOS26/`](../../ZipServ_ASPLOS26/).

| 역할 | 파일·함수 | 활용 방법 |
|---|---|---|
| 벤치마크 CPU 인코더 | [`utils.h`](../../ZipServ_ASPLOS26/kernel_benchmark/utils.h), `CompressBF16MatrixTripleBitmap_Host` | 포맷 및 CPU 정답 기준 |
| 지수 분석·압축 진입점 | 같은 파일의 `analyzeExponentDistribution_BF16`, `InitBF16MatrixTripleBitmap_Host` | 기존 선택 정책과 비교 |
| 라이브러리 CPU 인코더 | [`L_API.cu`](../../ZipServ_ASPLOS26/csrc/L_API.cu), `InitBF16MatrixTripleBitmap` | 별도 구현이므로 벤치마크 경로와 구분 |
| GPU 복원 로직 | [`L_Kernel.cuh`](../../ZipServ_ASPLOS26/csrc/L_Kernel.cuh), `LoadBF16FragWithTripleBitmap_SingleRow` | 비트맵·popcount 기반 복원 참고 |
| 단독 GPU 디코더 | 같은 파일의 `BF16TripleBitmap_Decompress_Kernel` | 재사용·수정할 초기 디코더 |
| 타일 정의 | [`TilingConfig.h`](../../ZipServ_ASPLOS26/csrc/TilingConfig.h) | 8×8 → 16×64 → 64×64 배치 참고 |

현재 참조 코드의 압축기는 CPU 구현이므로 **GPU 인코더는 새로 작성**해야 합니다. `hoyoung/`의 설명용 발췌본보다 실제 실행 코드인 `csrc/`와 `kernel_benchmark/utils.h`를 기준으로 합니다. 로컬 수정 사항이 있으므로 실험에 사용한 commit과 diff를 보관합니다.

TCA-TBE는 원소마다 3bit 코드를 비트맵에 저장합니다.

| 원소 | 코드 | payload |
|---|---|---|
| 선택된 지수 7개에 해당 | 001~111 | 부호+가수 8bit |
| 나머지 | 000 | 원본 BF16 16bit |

현재 디코더는 `exponent = start_exp + code`로 복원하므로 **오름차순으로 연속된 지수 7개**를 전제로 합니다. 고빈도 비율을 `p`라고 하면 메타데이터와 패딩을 제외한 평균 저장량은 다음과 같습니다.

```text
bits_per_element = 3 + 8*p + 16*(1-p) = 19 - 8*p
```

`p=1`이면 11bit로 31.25% 절약하며, `p>0.375`여야 이 단순 계산에서 압축 이득이 있습니다. 실제 이득은 오프셋·헤더·패딩·할당 비용을 포함해 판단합니다.

소스의 고빈도/저빈도 처리에는 조건문이 있으므로 “branch-free”를 가정하지 않습니다. 실제 분기·predication은 필요한 경우 생성된 PTX/SASS로 확인합니다.

## 4. 실험 A — 실제 KV-cache의 압축 가능성

### 데이터 수집

Python/PyTorch로 **실제로 캐시에 저장되는 BF16 K/V**를 수집합니다. 일반적인 RoPE 모델에서는 K의 RoPE 적용 이후 값을 사용하고, GQA의 KV head를 query head 수만큼 복제하기 전 데이터를 수집합니다. 모델의 실제 cache-write 경로를 확인해 수집 위치를 결정합니다.

- K와 V를 분리합니다.
- 레이어, KV head, 토큰 위치, 캐시 블록을 식별할 수 있게 저장합니다.
- prefill에서 생성된 KV와 decode에서 추가된 KV를 구분합니다.
- 일반 문장, 코드, 긴 문서 등 입력 유형을 나눕니다.
- 수집 실행과 성능 실행을 분리해 hook·CPU 복사·파일 저장 비용을 성능 측정에서 제외합니다.

### 분석 지표

BF16을 숫자 변환하지 않고 16bit 비트 패턴으로 해석한 뒤 지수를 추출합니다.

```text
exponent = (bf16_bits >> 7) & 0xff
p7 = max over s in [0, 249] of P(s <= exponent <= s+6)
```

| 지표 | 목적 |
|---|---|
| 지수 histogram, entropy | 데이터 분포 설명 |
| 연속 지수 7개의 최대 커버리지 `p7` | 현재 디코딩 방식의 압축 가능성 |
| 연속 제약 없는 상위 7개 커버리지 | 연속 구간 제약으로 잃는 압축 가능성 |
| 실제 포맷으로 계산한 저장량 | 메타데이터·패딩을 포함한 이득 |
| 블록별 압축률 분포와 비압축 fallback 비율 | 평균에 가려지는 실패 구간 |

다음 지수 선택 정책을 비교합니다.

1. 기존 ZipServ 선택 함수: 현재 구현의 기준선. 항상 빈도 합 최대 구간을 선택하는 것은 아닙니다.
2. 빈도 합이 최대인 연속 지수 7개: 동일 디코딩 방식에서 가능한 개선.
3. 연속 제약 없는 상위 7개: 분석용 비교. 실제 사용하려면 지수 lookup table을 지원하는 디코더가 필요합니다.

선택 범위는 모델 전체 고정, 레이어별 K/V 고정, 레이어·head별 고정, 블록별 동적 선택을 비교합니다. 고정 구간은 **calibration 입력에서 결정하고 별도 평가 입력에 적용**합니다. 동적 선택은 온라인 histogram·구간 선택 비용을 이후 encoding 시간에 포함합니다.

실제 저장량에는 비트맵, payload, 오프셋, 지수 구간 정보, 헤더, 정렬 패딩, 미완성 블록, 원본 저장 fallback을 포함합니다. 압축률은 `원본 바이트 / 저장 바이트`, 절감률은 `1 - 저장 바이트 / 원본 바이트`로 구분합니다.

### 산출물

- 레이어 × K/V 압축률 heatmap
- 블록별 `p7` 및 압축률 분포
- 토큰 위치·prefill/decode에 따른 분포 변화
- 지수 선택 범위별 압축률과 메타데이터 비용
- calibration 구간의 평가 입력 일반화 결과

## 5. 실험 B — GPU encoding/decoding 단독 비용

### 채택한 구현 기준

실험 B는 **ZipServ의 기존 BF16 TCA-TBE 포맷과 구현 방식을 따릅니다.** CPU 정답 기준은 `kernel_benchmark/utils.h`의 `CompressBF16MatrixTripleBitmap_Host`, 초기 GPU 디코더는 `csrc/L_Kernel.cuh`의 `BF16TripleBitmap_Decompress_Kernel`입니다. GPU 인코더는 이 CPU 인코더와 같은 비트맵·payload·오프셋을 생성하도록 새로 작성합니다.

- 작은 타일 8×8, 중간 타일 16×64, 글로벌 타일 64×64를 사용합니다.
- 글로벌 타일을 행 우선으로 방문하고, 각 중간 타일 안에서는 CPU 인코더의 2×2 작은 타일 그룹 순서를 그대로 따릅니다. 작은 타일 내부는 행 우선입니다.
- 비트맵 3개, 부호·가수 payload, 원본 BF16 payload를 각각 배열로 저장합니다. 중간 오프셋은 글로벌 타일 내부의 payload 위치, 글로벌 오프셋은 전체 payload 배열의 누적 위치이며, 두 payload 각각의 **원소 단위**로 기록합니다.
- 글로벌 타일마다 부호·가수는 16개 원소(16 B), 원본 BF16은 8개 원소(16 B)로 정렬합니다. 글로벌 오프셋에는 이 패딩을 포함합니다.
- 초기 디코더의 지수 구간은 호출별로 하나이며, CPU 인코더의 선택된 지수 `s..s+6`에 대해 `start_exp=s-1`을 전달합니다. `s=0`은 기존 `uint8` 표현의 wraparound와 복원식 검증이 필요하므로 별도 정확성 항목으로 다룹니다.

실험 A의 독립 타일 헤더·raw fallback은 저장량 분석용 가정이며 기존 ZipServ 포맷의 구성 요소가 아닙니다. B에서는 실제 생성된 배열의 바이트 수를 다시 계산하고, 디코더가 사용하지 않는 작은 타일 개수 배열(`TileOffsets`)은 검증용 workspace와 저장에 필요한 배열을 구분해 보고합니다.

첫 성능 측정은 **완성된 64×64 글로벌 타일 하나부터 여러 타일을 포함한 행렬까지** 호출당 처리량을 늘려 수행합니다. KV head마다 `[token, head_dim]`을 행렬로 해석하고 K/V를 구분합니다. 1토큰 신규 KV는 64행을 채우지 못하므로, 이 결과는 완성 블록 압축 비용이며 즉시 압축 비용으로 해석하지 않습니다. 부분 블록과 dimension tail은 초기 측정 범위 밖으로 명시하고 후속 저장 정책에서 다룹니다.

### 구현 언어와 커널 범위

| 부분 | 언어·도구 | 계획 |
|---|---|---|
| 데이터 수집·분석·실험 제어 | Python/PyTorch | 신규 스크립트 |
| CPU reference | Python 또는 기존 C++ | 포맷·비트 단위 정답 |
| GPU encoder | CUDA C++ | 신규 작성 |
| GPU standalone decoder | CUDA C++ | ZipServ 재사용·수정 |
| PyTorch binding | C++ extension | 현재 CUDA stream을 사용하는 호출 경로 |
| 통계·그래프 | Python | 결과 집계 |

기존 CUDA 코드와 포맷을 공유하고 warp ballot/popcount를 제어하기 위해 CUDA C++를 우선 사용합니다. Triton은 선택적인 프로토타입 도구이며, 초기 구현에 직접 작성한 PTX는 필요하지 않습니다.

최소 GPU encoder는 다음 단계로 구성합니다.

1. **분류·비트맵·개수 계산:** 지수 코드를 결정하고 8×8 타일의 비트맵 3개와 고빈도/원본값 개수를 생성합니다.
2. **오프셋 계산:** 개수와 패딩으로 prefix sum을 수행합니다. 큰 배치는 CUB scan을 활용합니다.
3. **Packing:** 부호·가수와 원본 BF16을 각 위치에 쓰고 ZipServ의 중간·글로벌 타일 오프셋을 기록합니다.
4. **동적 지수 선택:** 해당 정책을 평가할 때만 histogram·구간 선택 단계를 추가합니다. 초기 디코더는 호출별 단일 구간을 사용하므로 블록별 서로 다른 구간을 지원하려면 호출 분리 또는 디코더 확장이 필요합니다.

먼저 다단계 구현으로 정확성을 확보한 뒤, 작은 압축 블록은 블록 내부 scan과 packing을 합친 버전으로 launch 비용을 줄입니다. 큰 prefill 배치와 작은 decode 배치에 같은 구현이 최적인지는 별도로 평가합니다.

초기 단독 측정에서는 출력과 workspace를 미리 할당합니다. 단, **최악 크기의 출력 버퍼를 계속 예약하는 구현은 실제 메모리 절감을 입증하지 못합니다.** 통합 단계에서는 압축 저장 공간의 할당·회수, fragmentation, 임시 버퍼를 측정해야 합니다.

### 정확성

- `decode(encode(KV))`를 원본과 **16bit 패턴으로 비교**합니다. 근사 오차 기준으로 무손실을 판정하지 않습니다.
- 실제 KV와 함께 전체 고빈도, 전체 원본값, 혼합, 지수 경계, signed zero, subnormal, Inf/NaN 비트 패턴을 검증합니다.
- 부분 블록, 패딩, 연속 호출에서 바뀌는 shape·분포·최대 payload 크기를 확인합니다.
- 포맷의 입력 제약과 fallback을 명시하고 Compute Sanitizer로 메모리 접근을 확인합니다.

### 측정

| 지표 | 정의 |
|---|---|
| encoding latency | 필요한 지수 분석 + 분류 + scan + packing 전체 |
| decoding latency | 압축 입력에서 BF16 출력이 준비될 때까지 |
| 단계별 latency | 각 커널 및 launch 비용 분리 |
| 유효 처리량 | 원본 BF16 바이트 수 / 시간; encode/decode 동일 기준 |
| 실제 메모리 트래픽 | profiler로 측정한 DRAM 바이트·처리량 |
| workspace·peak memory | 원본·압축본·임시 버퍼 동시 유지 포함 |

CUDA event로 GPU 구간을 측정하고, 호출·동기화를 포함한 wall-clock 시간도 별도로 기록합니다. 작은 작업은 CUDA Graph 사용 여부를 나누어 launch 영향을 확인합니다.

실제 KV 샘플을 주 입력으로 사용합니다. 합성 데이터에서는 고빈도 비율과 원소의 공간적 배치를 바꿔, 같은 압축률에서도 비용이 달라지는지 확인합니다. codec 처리량에 CPU↔GPU 전송은 포함하지 않으며, 전송을 평가한다면 별도 항목으로 보고합니다.

## 6. 실험 C — prefill/decode와 codec의 자원 경합

이 단계에서는 기존 추론의 KV 경로를 유지하고 **같은 프로세스·CUDA context의 별도 stream**에서 별도 버퍼의 codec 작업을 실행합니다. 먼저 자원 경합만 분리합니다.

| 추론 작업 | codec 부하 |
|---|---|
| Prefill | 없음 / encode / decode |
| Decode step | 없음 / encode / decode |

각 조합에서 추론 단독, codec 단독, 순차 실행, 동시 실행을 비교합니다. 비기본 stream과 event로 시작·종료를 조율하고 측정 중 불필요한 device-wide synchronization을 피합니다. 서로 다른 stream은 동시 실행을 허용할 뿐 보장하지 않으므로 실제 overlap을 확인합니다.

```text
inference_slowdown = inference_time_concurrent / inference_time_alone
codec_slowdown     = codec_time_concurrent / codec_time_alone
```

양쪽 지연뿐 아니라 두 작업이 모두 완료될 때까지의 시간, codec 대기열, 완료 처리량을 측정합니다. 추론이 빨라도 codec이 계속 밀리면 부하를 숨긴 것으로 판정하지 않습니다.

### 부하 주입 정책

- **지속 부하:** codec을 연속 실행해 경합 상한을 측정합니다.
- **실제 발생량에 맞춘 부하:** 레이어·step별 KV 생성량과 읽기량에 맞춰 작업 크기·빈도를 설정합니다.

균일한 head 구조를 가진 단일 GPU 모델에서 전체 레이어의 신규 BF16 KV 바이트는 다음과 같습니다.

```text
new_KV_bytes = 2 * L * B * H_kv * D * delta_tokens * 2
               K,V                              BF16 bytes
```

`L`은 레이어 수, `B`는 batch, `H_kv`는 KV head 수, `D`는 head dimension입니다. 일반적인 decode에서는 `delta_tokens=1`입니다. 과거 KV의 논리적 읽기량은 context 길이에 따라 증가하며, 실제 물리적 트래픽은 attention 구현과 재사용에 따라 달라집니다.

비슷한 바이트 수를 읽고 쓰는 단순 CUDA copy 커널을 대조군으로 추가해, 메모리 트래픽과 codec 추가 연산의 영향을 구분하는 근거로 사용합니다. copy와 codec의 접근 패턴이 완전히 같지는 않다는 한계도 기록합니다.

### 프로파일링과 해석

- **Nsight Systems:** 실제 커널 중첩, launch 간격, 대기, 실행 순서.
- **Nsight Compute:** DRAM 처리량, SM 사용, 레지스터·shared memory, occupancy 등.
- 정식 latency는 profiler 없이 측정합니다. Nsight Compute replay의 직렬화·캐시 변경 영향을 주의합니다.

이 실험은 **추가 codec 작업의 경합**을 검증합니다. 원래 KV 트래픽을 줄이는 효과는 반영하지 않으므로, 이 결과만으로 압축 KV-cache의 최종 손익을 결론 내리지 않습니다.

## 7. 실험 D — 실제 KV 경로 통합

다음 순서로 구현·비교합니다.

| 경로 | 목적 |
|---|---|
| BF16 KV → 기존 attention | 기준선 |
| 압축 KV → 별도 해제 → 기존 attention | 의존성을 포함한 codec 추가 비용 |
| 압축 KV → 해제 융합 attention | 복원본 global-memory 쓰기 제거 효과 |

먼저 별도 해제 경로로 정확성과 동작을 확인합니다. 전체 KV를 BF16 임시 버퍼로 복원하면 메모리 절감이 상쇄될 수 있으므로 임시 버퍼까지 peak memory에 포함합니다. 청크 단위 복원을 적용한다면 attention의 청크 처리·결과 결합 비용도 포함합니다.

융합 attention은 초기 A~C의 필수 구현은 아닙니다. 다만 별도 해제 방식의 성능 저하를 압축 방식 전체의 한계로 일반화하지 않으려면 후속 평가가 필요합니다.

융합 커널에는 다음 기능이 필요합니다.

- 압축 K 복원과 `QK^T` 계산
- softmax 처리
- 압축 V 복원과 `PV` 계산
- paged KV 주소, GQA, 마스크, 미완성 블록 처리

ZipGEMM을 그대로 호출하는 것으로 attention을 대체할 수는 없습니다. K와 V의 접근 패턴에 맞는 레이아웃을 검토하고, **decode attention 한 레이어**부터 구현합니다. Prefill attention 융합은 결과를 보고 확장합니다.

`KV 생성 완료 → 압축 → 필요한 복원 완료 → attention` 의존성을 보장합니다. 같은 KV를 소비하는 작업을 무조건 독립 stream에 배치하지 않습니다. 압축 전 원본을 당분간 사용하는 정책이라면 원본의 수명과 압축 완료 대기를 함께 기록합니다.

### 최종 지표

| 비교 조건 | 측정 항목 |
|---|---|
| 동일 batch·context | TTFT, TPOT, tokens/s, p50/p95 latency, peak memory |
| 동일 GPU 메모리·지연 제한 | 수용 가능한 batch·context, 요청 처리량 |

KV 복원은 비트 단위 일치를 요구합니다. attention 커널 자체를 바꿨을 때의 연산 순서에 따른 출력 차이는 별도로 검증해, 무손실 codec의 정확성과 구분합니다.

## 8. 초기 실험 매트릭스와 재현성

아래 값은 시작점이며, 모델의 지원 context와 GPU 메모리 범위 안에서 조정합니다. 처음부터 모든 조합의 전체 곱을 실행하지 않고 A → B → C → D 결과로 범위를 좁힙니다.

| 축 | 초기 설정 |
|---|---|
| 모델 | 로컬 Llama-3.1-8B-Instruct; 사용한 revision 기록 |
| GPU | RTX 4090 1장; 후속 하드웨어 비교는 별도 확장 |
| 입력 | 일반 문장·코드·긴 문서, calibration/평가 분리 |
| context 길이 | 512, 2K, 8K |
| batch | 1, 4, 16; 메모리 범위 내 |
| 생성 길이 | 128 tokens |
| 저장 정책 | 즉시 압축 / 완성 블록 압축 |
| 지수 정책 | 레이어별 K/V 고정 / 블록별 동적 |
| codec 입력 | 실제 KV + 고빈도 비율·배치를 조절한 합성 데이터 |

재현을 위해 다음을 기록합니다.

- 모델·tokenizer revision, 입력 목록·길이, seed, 데이터 분할.
- GPU 모델·메모리, driver/CUDA, PyTorch·추론 엔진 버전과 commit.
- attention backend, dtype, KV layout, page size, 압축 타일·블록 크기.
- CUDA Graph, stream priority, warm-up·반복 횟수, 클럭·전력 조건.
- 참조 ZipServ commit과 local diff, 빌드 옵션.

Warm-up 이후 여러 독립 실행을 수행하고 baseline/실험군 순서를 섞습니다. 평균뿐 아니라 분산·신뢰구간을 보고하고, p95는 충분한 표본으로 산출합니다. 데이터 수집·로그·할당 비용의 측정 포함 여부를 명시합니다. 실제 추론 측정에서 매번 L2를 강제 flush하지 않으며 cold-cache 벤치마크는 별도 조건으로 둡니다.

실험 B의 미리 할당된 버퍼는 커널 비용을 보기 위한 조건입니다. 실험 D에서는 압축 저장 공간 관리와 동적 메타데이터 갱신 비용까지 포함합니다. 논리적인 압축 바이트와 실제 예약·사용 GPU 메모리를 구분해 보고합니다.

## 9. 구현 순서

- A: KV 수집 위치와 데이터 schema 정의
- A: 지수 분포·지수 선택 정책·실제 포맷 저장량 분석
- B: CPU reference 및 압축 포맷 명세
- B: GPU encoder와 standalone decoder, 비트 단위 검증
- B: 작은 decode 작업·큰 prefill 배치의 단독 성능 측정
- C: 두 stream의 경합 benchmark와 실제 부하량 재현
- C: 타임라인·자원 사용 분석
- D: 별도 해제 방식의 KV 경로 통합
- D: 결과에 따라 decode attention 융합 및 저장 공간 관리 구현

**첫 구현 목표는 A~C입니다.** 이 범위에서 압축 가능성, codec 자체 비용, 추론 경합을 확인한 뒤 attention 융합의 구현 우선순위를 결정합니다.

## 참고 자료

- [ZipServ 로컬 README](../../ZipServ_ASPLOS26/README.md)
- [vLLM PagedAttention 소개](https://vllm.ai/blog/2023-06-20-vllm): KV 블록 관리와 반복 접근 구조.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf): stream과 커널 동시 실행.
- [CUB DeviceScan](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html): GPU prefix sum.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): 실행 타임라인 분석.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): 자원 분석과 replay 주의사항.
