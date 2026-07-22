# GPT-OSS 20B profiling plots

세 스크립트는 기본적으로 `../results/experiment.csv`를 읽고 `figures/`에 PNG를
저장한다.

`matplotlib`이 없다면 로그인 노드에서 profile 환경에 한 번 설치한다.

```bash
python -m pip install matplotlib
```

```bash
cd /home/hybyun0207/aso-internship/MoE

python plot/plot_prefill_latency.py
python plot/plot_decode_context_latency.py
python plot/plot_decode_batch_composition.py
```

생성 파일:

```text
plot/figures/prefill_composition_percent.png
plot/figures/prefill_latency_ms.png
plot/figures/decode_b1_composition_percent.png
plot/figures/decode_b1_latency_ms.png
plot/figures/decode_batch_composition_percent.png
plot/figures/decode_batch_latency_ms.png
```

세 스크립트는 각각 percentage와 millisecond 버전을 하나씩 만든다. 모든
구성요소가 동일한 막대 너비를 사용하며 막대 윤곽선은 그리지 않는다.

```text
보라색 배경 = End-to-end
파란색       = Attention
주황색       = MoE
회색         = Other

Attention + MoE + Other stack = Phase GPU total
```

E2E 막대를 먼저 그리고 GPU 구성 stack을 같은 위치에 덮으므로, E2E와 phase
GPU total의 차이는 stack 위에 남는 보라색 영역으로 나타난다. Attention과
MoE는 `bottom=attention` 방식으로 누적하므로 서로 겹치지 않는다.

Percentage 버전은 phase GPU total을 100%로 정규화하고, millisecond 버전은
CSV의 실제 GPU 시간과 E2E 시간을 사용한다. Decode batch percentage 그래프는
막대 위에 `MoE/Attention` 비율도 함께 출력한다.

다른 CSV나 출력 경로를 사용하려면 다음처럼 지정한다.

```bash
python plot/plot_prefill_latency.py \
  --csv results/experiment.csv \
  --percent-output plot/figures/custom-prefill-percent.png \
  --time-output plot/figures/custom-prefill-ms.png \
  --dpi 240
```
