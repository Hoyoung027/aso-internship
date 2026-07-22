#!/usr/bin/env python3
"""입력 1,024에서 Decode batch별 구성을 percentage와 millisecond로 그린다."""

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
DEFAULT_PERCENT_OUTPUT = PLOT_DIR / "figures" / "decode_batch_composition_percent.png"
DEFAULT_TIME_OUTPUT = PLOT_DIR / "figures" / "decode_batch_latency_ms.png"
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
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_rows(path: Path, input_len: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["phase"] == "decode"
            and int(row["input_tokens_per_sequence"]) == input_len
        ]
    rows.sort(key=lambda row: int(row["batch_size"]))
    if not rows:
        raise ValueError(f"No Decode rows for input_len={input_len} in {path}")
    return rows


def values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def add_mixed_note(fig: plt.Figure, rows: list[dict[str, str]]) -> None:
    mixed_rows = [row for row in rows if int(row["mixed_step_count"]) > 0]
    if not mixed_rows:
        return
    details = ", ".join(
        f"b{row['batch_size']}: {row['excluded_mixed_generation_tokens']} token"
        for row in mixed_rows
    )
    fig.text(
        0.5,
        0.025,
        f"Mixed scheduler steps excluded from pure Decode GPU time ({details}).",
        ha="center",
        fontsize=8,
        color="#555555",
    )


def configure_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    title: str,
    labels: list[str],
    ylabel: str,
    ymax: float,
    unit: str,
) -> None:
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.18)
    fig.suptitle(title, y=0.96)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    if unit == "%":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    else:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g} ms")
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        frameon=True,
    )


def draw_percent(
    rows: list[dict[str, str]], labels: list[str], output: Path, dpi: int, input_len: int
) -> None:
    positions = list(range(len(rows)))
    attention = values(rows, "attention_percent")
    moe = values(rows, "moe_percent")
    other = values(rows, "other_percent")
    attention_plus_moe = [a + m for a, m in zip(attention, moe, strict=True)]

    fig, ax = plt.subplots(figsize=(10, 6.5))
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
        label="Other",
        zorder=2,
    )

    for x, attn, moe_value in zip(positions, attention, moe, strict=True):
        ax.text(
            x,
            101.5,
            f"MoE/Attn\n{moe_value / attn:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(x, attn / 2, f"{attn:.1f}%", ha="center", va="center", fontsize=8)
        ax.text(
            x,
            attn + moe_value / 2,
            f"{moe_value:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
        )

    configure_figure(
        fig,
        ax,
        title=f"GPT-OSS 20B Decode GPU Time Composition (Input Length = {input_len:,})",
        labels=labels,
        ylabel="Share of pure Decode GPU time (%)",
        ymax=114,
        unit="%",
    )
    add_mixed_note(fig, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"figure={output.resolve()}")


def draw_time(
    rows: list[dict[str, str]], labels: list[str], output: Path, dpi: int, input_len: int
) -> None:
    positions = list(range(len(rows)))
    e2e = values(rows, "end_to_end_latency_ms")
    attention = values(rows, "attention_gpu_total_ms")
    moe = values(rows, "moe_gpu_total_ms")
    other = values(rows, "other_gpu_total_ms")
    attention_plus_moe = [a + m for a, m in zip(attention, moe, strict=True)]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.bar(
        positions,
        e2e,
        width=BAR_WIDTH,
        color=COLORS["e2e"],
        label="Request end-to-end",
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
        label="Other (stack total = Pure Decode GPU total)",
        zorder=2,
    )

    for x, e2e_value, attn, moe_value in zip(
        positions, e2e, attention, moe, strict=True
    ):
        ax.annotate(
            f"E2E {e2e_value:.0f} ms",
            (x, e2e_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            x,
            attn / 2,
            f"{attn:.0f} ms",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.text(
            x,
            attn + moe_value / 2,
            f"{moe_value:.0f} ms",
            ha="center",
            va="center",
            fontsize=8,
        )

    configure_figure(
        fig,
        ax,
        title=f"GPT-OSS 20B Decode Latency by Batch (Input Length = {input_len:,})",
        labels=labels,
        ylabel="Latency (ms)",
        ymax=max(e2e) * 1.12,
        unit="ms",
    )
    add_mixed_note(fig, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"figure={output.resolve()}")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv.resolve(), args.input_len)
    labels = [row["batch_size"] for row in rows]
    draw_percent(rows, labels, args.percent_output, args.dpi, args.input_len)
    draw_time(rows, labels, args.time_output, args.dpi, args.input_len)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
