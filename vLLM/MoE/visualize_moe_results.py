#!/usr/bin/env python3
"""Create publication-ready plots from the GPT-OSS MoE benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "results" / "moe_kernel_results.csv"

EXPERIMENT_ORDER = [
    "marlin_w4a16",
    "humming_indexed_w4a16",
    "humming_grouped_w4a16",
    "flashinfer_cutlass_w4a8_tune_off",
    "flashinfer_cutlass_w4a8_tune_on",
    "emulation_w4a16",
]

LABELS = {
    "marlin_w4a16": "Marlin W4A16",
    "humming_indexed_w4a16": "Humming indexed W4A16",
    "humming_grouped_w4a16": "Humming grouped W4A16",
    "flashinfer_cutlass_w4a8_tune_off": "FlashInfer CUTLASS W4A8 (tune off)",
    "flashinfer_cutlass_w4a8_tune_on": "FlashInfer CUTLASS W4A8 (tune on)",
    "emulation_w4a16": "Emulation W4A16",
}

SHORT_LABELS = {
    "marlin_w4a16": "Marlin",
    "humming_indexed_w4a16": "Humming\nindexed",
    "humming_grouped_w4a16": "Humming\ngrouped",
    "flashinfer_cutlass_w4a8_tune_off": "FlashInfer\noff",
    "flashinfer_cutlass_w4a8_tune_on": "FlashInfer\non",
    "emulation_w4a16": "Emulation",
}

COLORS = {
    "marlin_w4a16": "#4C78A8",
    "humming_indexed_w4a16": "#F58518",
    "humming_grouped_w4a16": "#E45756",
    "flashinfer_cutlass_w4a8_tune_off": "#72B7B2",
    "flashinfer_cutlass_w4a8_tune_on": "#54A24B",
    "emulation_w4a16": "#7F7F7F",
}

MARKERS = {
    "marlin_w4a16": "o",
    "humming_indexed_w4a16": "s",
    "humming_grouped_w4a16": "D",
    "flashinfer_cutlass_w4a8_tune_off": "^",
    "flashinfer_cutlass_w4a8_tune_on": "v",
    "emulation_w4a16": "X",
}

REQUIRED_FIELDS = {
    "experiment_id",
    "num_tokens",
    "router_gpu_ms_mean",
    "fused_moe_gpu_ms_mean",
    "fused_moe_gpu_ms_std",
    "router_percent_mean",
    "status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Defaults to "
            "<project>/results/plots/<input parent name>."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--representative-tokens",
        default="1,128,1024,8192",
        help="Token counts used in the router/FusedMoE breakdown.",
    )
    return parser.parse_args()


def finite_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field} value: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} value: {value!r}")
    return parsed


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Input CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing_fields = sorted(REQUIRED_FIELDS - fields)
        if missing_fields:
            raise ValueError(
                "Input CSV is missing fields: " + ", ".join(missing_fields)
            )
        raw_rows = list(reader)

    incomplete = [row for row in raw_rows if row["status"] != "ok"]
    if incomplete:
        raise ValueError(
            f"Input contains {len(incomplete)} non-ok rows; refusing to plot"
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in raw_rows:
        experiment_id = row["experiment_id"]
        num_tokens = int(row["num_tokens"])
        key = (experiment_id, num_tokens)
        if key in seen:
            raise ValueError(f"Duplicate backend/token row: {key}")
        seen.add(key)
        parsed = dict(row)
        parsed["num_tokens"] = num_tokens
        for field in (
            "router_gpu_ms_mean",
            "fused_moe_gpu_ms_mean",
            "fused_moe_gpu_ms_std",
            "router_percent_mean",
        ):
            parsed[field] = finite_float(row[field], field)
        if parsed["fused_moe_gpu_ms_mean"] <= 0:
            raise ValueError(f"Non-positive FusedMoE mean for {key}")
        rows.append(parsed)
    if not rows:
        raise ValueError(f"Input CSV contains no rows: {path}")
    return rows


def index_results(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[int], dict[str, dict[int, dict[str, Any]]]]:
    table: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        table[row["experiment_id"]][row["num_tokens"]] = row

    experiments = [value for value in EXPERIMENT_ORDER if value in table]
    experiments.extend(sorted(set(table) - set(experiments)))
    tokens = sorted({row["num_tokens"] for row in rows})
    for experiment_id in experiments:
        missing = sorted(set(tokens) - set(table[experiment_id]))
        if missing:
            raise ValueError(
                f"{experiment_id} is missing token counts: {missing}"
            )
    return experiments, tokens, table


def mean_throughput(row: dict[str, Any]) -> float:
    return row["num_tokens"] / (row["fused_moe_gpu_ms_mean"] / 1000.0)


def experiment_label(experiment_id: str) -> str:
    return LABELS.get(experiment_id, experiment_id)


def default_output_dir(input_path: Path) -> Path:
    run_name = input_path.parent.name or input_path.stem
    return PROJECT_DIR / "results" / "plots" / run_name


def plot_style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
        }
    )
    return plt


def set_token_axis(ax: Any, tokens: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(tokens)
    ax.set_xticklabels([str(value) for value in tokens], rotation=45)
    ax.set_xlim(min(tokens) / 1.12, max(tokens) * 1.12)


def line_style(experiment_id: str) -> dict[str, Any]:
    return {
        "label": experiment_label(experiment_id),
        "color": COLORS.get(experiment_id),
        "marker": MARKERS.get(experiment_id, "o"),
        "linestyle": "--" if experiment_id == "emulation_w4a16" else "-",
    }


def save_figure(
    fig: Any,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> Path:
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi)
    return path


def plot_latency(
    plt: Any,
    experiments: list[str],
    tokens: list[int],
    table: dict[str, dict[int, dict[str, Any]]],
) -> Any:
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    for experiment_id in experiments:
        means = [
            table[experiment_id][token]["fused_moe_gpu_ms_mean"]
            for token in tokens
        ]
        ax.plot(tokens, means, **line_style(experiment_id))
    set_token_axis(ax, tokens)
    ax.set_yscale("log")
    ax.set_xlabel("Number of tokens (M)")
    ax.set_ylabel("Mean FusedMoE GPU time (ms)")
    ax.set_title("GPT-OSS 20B layer-0 FusedMoE latency")
    ax.text(
        0.01,
        0.99,
        "Each point is the mean of 50 measured GPU events.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


def plot_throughput(
    plt: Any,
    experiments: list[str],
    tokens: list[int],
    table: dict[str, dict[int, dict[str, Any]]],
) -> Any:
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    for experiment_id in experiments:
        values = [mean_throughput(table[experiment_id][token]) for token in tokens]
        ax.plot(tokens, values, **line_style(experiment_id))
    set_token_axis(ax, tokens)
    ax.set_yscale("log")
    ax.set_xlabel("Number of tokens (M)")
    ax.set_ylabel("Mean-derived FusedMoE throughput (tokens/s)")
    ax.set_title("GPT-OSS 20B layer-0 FusedMoE throughput")
    ax.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


def plot_speedup(
    plt: Any,
    experiments: list[str],
    tokens: list[int],
    table: dict[str, dict[int, dict[str, Any]]],
) -> Any:
    baseline = "marlin_w4a16"
    if baseline not in table:
        raise ValueError("Marlin rows are required for the speedup plot")
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    for experiment_id in experiments:
        if experiment_id == baseline:
            continue
        values = [
            table[baseline][token]["fused_moe_gpu_ms_mean"]
            / table[experiment_id][token]["fused_moe_gpu_ms_mean"]
            for token in tokens
        ]
        ax.plot(tokens, values, **line_style(experiment_id))
    set_token_axis(ax, tokens)
    ax.axhline(1.0, color="#333333", linewidth=1.4, linestyle=":")
    ax.set_xlabel("Number of tokens (M)")
    ax.set_ylabel("Speedup over Marlin (Marlin mean / kernel mean)")
    ax.set_title("Mean FusedMoE latency speedup over Marlin")
    ax.text(
        0.01,
        0.99,
        "Above 1.0 is faster than Marlin.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    return fig


def plot_flashinfer_autotune(
    plt: Any,
    tokens: list[int],
    table: dict[str, dict[int, dict[str, Any]]],
) -> Any:
    tune_off = "flashinfer_cutlass_w4a8_tune_off"
    tune_on = "flashinfer_cutlass_w4a8_tune_on"
    if tune_off not in table or tune_on not in table:
        raise ValueError("Both FlashInfer autotune rows are required")

    off_values = [
        table[tune_off][token]["fused_moe_gpu_ms_mean"] for token in tokens
    ]
    on_values = [
        table[tune_on][token]["fused_moe_gpu_ms_mean"] for token in tokens
    ]
    ratios = [off / on for off, on in zip(off_values, on_values)]

    fig, (latency_ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(12.5, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )
    latency_ax.plot(tokens, off_values, **line_style(tune_off))
    latency_ax.plot(tokens, on_values, **line_style(tune_on))
    latency_ax.set_yscale("log")
    latency_ax.set_ylabel("Mean FusedMoE GPU time (ms)")
    latency_ax.set_title("FlashInfer CUTLASS autotune on/off comparison")
    latency_ax.legend()

    ratio_ax.plot(
        tokens,
        ratios,
        color="#6F4E7C",
        marker="o",
        label="tune-off mean / tune-on mean",
    )
    ratio_ax.axhline(1.0, color="#333333", linewidth=1.4, linestyle=":")
    ratio_ax.set_ylabel("Autotune speedup ratio")
    ratio_ax.set_xlabel("Number of tokens (M)")
    ratio_ax.text(
        0.01,
        0.97,
        "Above 1.0: tune-on is faster; below 1.0: tune-off is faster.",
        transform=ratio_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )
    ratio_ax.legend(loc="lower left")
    set_token_axis(ratio_ax, tokens)
    fig.tight_layout()
    return fig


def plot_router_breakdown(
    plt: Any,
    experiments: list[str],
    representative_tokens: list[int],
    table: dict[str, dict[int, dict[str, Any]]],
) -> Any:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    for ax, token in zip(axes.flat, representative_tokens):
        router = [
            table[experiment_id][token]["router_gpu_ms_mean"]
            for experiment_id in experiments
        ]
        fused = [
            table[experiment_id][token]["fused_moe_gpu_ms_mean"]
            for experiment_id in experiments
        ]
        percentages = [
            table[experiment_id][token]["router_percent_mean"]
            for experiment_id in experiments
        ]
        positions = list(range(len(experiments)))
        colors = [
            COLORS.get(experiment_id, "#777777")
            for experiment_id in experiments
        ]
        ax.bar(positions, router, color="#222222", label="Router")
        ax.bar(
            positions,
            fused,
            bottom=router,
            color=colors,
            alpha=0.88,
            label="FusedMoE",
        )
        totals = [r + f for r, f in zip(router, fused)]
        for position, total, percentage in zip(
            positions, totals, percentages
        ):
            ax.annotate(
                f"{percentage:.1f}%",
                (position, total),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
        ax.set_title(f"M = {token}")
        ax.set_ylabel("Mean GPU time (ms)")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [SHORT_LABELS.get(value, value) for value in experiments],
            fontsize=8,
        )
        ax.margins(y=0.16)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
    )
    fig.suptitle(
        "Layer-0 router and FusedMoE mean GPU-time breakdown\n"
        "Labels above bars show router percentage",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return fig


def write_winner_summary(
    output_path: Path,
    experiments: list[str],
    tokens: list[int],
    table: dict[str, dict[int, dict[str, Any]]],
) -> None:
    baseline = "marlin_w4a16"
    tune_off = "flashinfer_cutlass_w4a8_tune_off"
    tune_on = "flashinfer_cutlass_w4a8_tune_on"
    fields = [
        "num_tokens",
        "best_experiment_id",
        "best_label",
        "best_fused_moe_gpu_ms_mean",
        "second_experiment_id",
        "second_fused_moe_gpu_ms_mean",
        "speedup_vs_marlin",
        "flashinfer_tune_off_over_on_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for token in tokens:
            ranked = sorted(
                experiments,
                key=lambda value: table[value][token]["fused_moe_gpu_ms_mean"],
            )
            best, second = ranked[:2]
            best_time = table[best][token]["fused_moe_gpu_ms_mean"]
            marlin_time = table[baseline][token]["fused_moe_gpu_ms_mean"]
            off_time = table[tune_off][token]["fused_moe_gpu_ms_mean"]
            on_time = table[tune_on][token]["fused_moe_gpu_ms_mean"]
            writer.writerow(
                {
                    "num_tokens": token,
                    "best_experiment_id": best,
                    "best_label": experiment_label(best),
                    "best_fused_moe_gpu_ms_mean": f"{best_time:.6f}",
                    "second_experiment_id": second,
                    "second_fused_moe_gpu_ms_mean": (
                        f"{table[second][token]['fused_moe_gpu_ms_mean']:.6f}"
                    ),
                    "speedup_vs_marlin": f"{marlin_time / best_time:.6f}",
                    "flashinfer_tune_off_over_on_ratio": (
                        f"{off_time / on_time:.6f}"
                    ),
                }
            )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.input)
    representative_tokens = [
        int(value.strip())
        for value in args.representative_tokens.split(",")
        if value.strip()
    ]
    if len(representative_tokens) != 4:
        raise ValueError("--representative-tokens must contain exactly 4 values")

    rows = load_results(args.input)
    experiments, tokens, table = index_results(rows)
    missing_representatives = sorted(set(representative_tokens) - set(tokens))
    if missing_representatives:
        raise ValueError(
            "Representative token counts are absent: "
            + ", ".join(map(str, missing_representatives))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = plot_style()
    plotters = [
        (
            "fused_moe_latency_mean",
            lambda: plot_latency(plt, experiments, tokens, table),
        ),
        (
            "fused_moe_throughput_mean",
            lambda: plot_throughput(plt, experiments, tokens, table),
        ),
        (
            "speedup_vs_marlin_mean",
            lambda: plot_speedup(plt, experiments, tokens, table),
        ),
        (
            "flashinfer_autotune_mean",
            lambda: plot_flashinfer_autotune(plt, tokens, table),
        ),
        (
            "router_fused_breakdown_mean",
            lambda: plot_router_breakdown(
                plt, experiments, representative_tokens, table
            ),
        ),
    ]

    output_paths: list[Path] = []
    for stem, make_figure in plotters:
        figure = make_figure()
        output_paths.append(save_figure(figure, output_dir, stem, args.dpi))
        plt.close(figure)

    summary_path = output_dir / "kernel_winners_mean.csv"
    write_winner_summary(summary_path, experiments, tokens, table)
    output_paths.append(summary_path)
    for path in output_paths:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
