#!/usr/bin/env python3
"""입력 길이별 Prefill 구성을 percentage와 millisecond로 그린다."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/{os.environ.get('USER', 'user')}/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PLOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PLOT_DIR.parent
DEFAULT_CSV = PROJECT_DIR / "results" / "experiment.csv"
DEFAULT_PERCENT_OUTPUT = PLOT_DIR / "figures" / "prefill_composition_percent.png"
DEFAULT_TIME_OUTPUT = PLOT_DIR / "figures" / "prefill_latency_ms.png"
BAR_WIDTH = 0.66
COLORS = {
    "e2e": "#cdb4db",
    "attention": "#3a86ff",
    "moe": "#ff9f1c",
    "other": "#b8b8b8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--percent-output", type=Path, default=DEFAULT_PERCENT_OUTPUT)
    parser.add_argument("--time-output", type=Path, default=DEFAULT_TIME_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["phase"] == "prefill"]
    rows.sort(key=lambda row: int(row["input_tokens_per_sequence"]))
    if not rows:
        raise ValueError(f"No Prefill rows found in {path}")
    return rows


def values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def draw_stacked(
    *,
    positions: list[int],
    labels: list[str],
    e2e: list[float],
    attention: list[float],
    moe: list[float],
    other: list[float],
    title: str,
    ylabel: str,
    output: Path,
    dpi: int,
    unit: str,
) -> None:
    attention_plus_moe = [a + m for a, m in zip(attention, moe, strict=True)]
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    # E2E를 먼저 채우고 GPU 구성 stack을 같은 너비로 덮는다. E2E와 GPU total의
    # 차이는 100% 또는 GPU stack 위쪽에 남는 보라색 영역으로 보인다.
    ax.bar(
        positions,
        e2e,
        width=BAR_WIDTH,
        color=COLORS["e2e"],
        label="End-to-end",
        zorder=1,
    )
    ax.bar(
        positions,
        attention,
        width=BAR_WIDTH,
        color=COLORS["attention"],
        label="Attention",
        zorder=2,
    )
    ax.bar(
        positions,
        moe,
        width=BAR_WIDTH,
        bottom=attention,
        color=COLORS["moe"],
        label="MoE",
        zorder=2,
    )
    ax.bar(
        positions,
        other,
        width=BAR_WIDTH,
        bottom=attention_plus_moe,
        color=COLORS["other"],
        label="Other (stack total = Phase GPU total)",
        zorder=2,
    )

    for x, e2e_value, attn, moe_value in zip(
        positions, e2e, attention, moe, strict=True
    ):
        value_label = (
            f"E2E {e2e_value:.1f}%"
            if unit == "%"
            else f"E2E {e2e_value:.1f} ms"
        )
        ax.annotate(
            value_label,
            (x, e2e_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        suffix = "%" if unit == "%" else " ms"
        ax.text(
            x,
            attn / 2,
            f"{attn:.1f}{suffix}",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.text(
            x,
            attn + moe_value / 2,
            f"{moe_value:.1f}{suffix}",
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("Input tokens per sequence")
    ax.set_ylabel(ylabel)
    if unit == "%":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    else:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g} ms")
        )
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(ncol=2, frameon=True)
    ax.set_ylim(0, max(e2e) * 1.13)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"figure={output.resolve()}")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv.resolve())
    positions = list(range(len(rows)))
    labels = [f"{int(row['input_tokens_per_sequence']):,}" for row in rows]

    e2e_ms = values(rows, "end_to_end_latency_ms")
    phase_gpu_ms = values(rows, "phase_gpu_total_ms")
    attention_ms = values(rows, "attention_gpu_total_ms")
    moe_ms = values(rows, "moe_gpu_total_ms")
    other_ms = values(rows, "other_gpu_total_ms")

    e2e_percent = [
        wall_time / gpu_time * 100
        for wall_time, gpu_time in zip(e2e_ms, phase_gpu_ms, strict=True)
    ]
    draw_stacked(
        positions=positions,
        labels=labels,
        e2e=e2e_percent,
        attention=values(rows, "attention_percent"),
        moe=values(rows, "moe_percent"),
        other=values(rows, "other_percent"),
        title="GPT-OSS 20B Prefill GPU Time Composition (Batch Size = 1)",
        ylabel="Normalized time (% of Phase GPU total)",
        output=args.percent_output,
        dpi=args.dpi,
        unit="%",
    )
    draw_stacked(
        positions=positions,
        labels=labels,
        e2e=e2e_ms,
        attention=attention_ms,
        moe=moe_ms,
        other=other_ms,
        title="GPT-OSS 20B Prefill Latency (Batch Size = 1)",
        ylabel="Latency (ms)",
        output=args.time_output,
        dpi=args.dpi,
        unit="ms",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
