#!/usr/bin/env python3
"""Plot paper-style FlashMoE figures from completed partial GPU jobs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FLASHMOE_COLOR = "#1874CD"
EDGE_COLOR = "#174A6E"
GRID_COLOR = "#D9DEE3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-dir",
        type=Path,
        action="append",
        required=True,
        help="Completed paper-reproduction job directory; repeat for each GPU count.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_status(path: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            status[key] = value
    return status


def load_jobs(job_dirs: Iterable[Path]) -> tuple[list[dict[str, object]], str, list[str]]:
    rows: list[dict[str, object]] = []
    partitions: set[str] = set()
    nodes: list[str] = []
    gpu_counts: set[int] = set()

    for job_dir_arg in job_dirs:
        job_dir = job_dir_arg.resolve()
        manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
        status = read_status(job_dir / "job_status.txt")
        gpu_count = int(manifest["world_size"])
        if gpu_count in gpu_counts:
            raise ValueError(f"duplicate {gpu_count}-GPU result: {job_dir}")
        if status.get("EXIT_CODE") != "0":
            raise ValueError(f"job did not complete successfully: {job_dir}")

        partition = status.get("PARTITION", "unknown")
        node = status.get("NODE", "unknown")
        partitions.add(partition)
        nodes.append(f"{gpu_count} GPU: {node}")
        gpu_counts.add(gpu_count)

        job_rows = read_csv(job_dir / "aggregates.csv")
        completed = [row for row in job_rows if row["status"] == "completed"]
        if len(completed) != len(job_rows):
            raise ValueError(f"job has incomplete conditions: {job_dir}")
        if any(float(row["max_error_pct"]) != 0.0 for row in completed):
            raise ValueError(f"job has a nonzero correctness error: {job_dir}")
        for row in completed:
            rows.append(
                {
                    **row,
                    "job_dir": str(job_dir),
                    "partition": partition,
                    "node": node,
                }
            )

    if len(partitions) != 1:
        raise ValueError(f"mixed partitions are not comparable: {sorted(partitions)}")
    return rows, next(iter(partitions)), sorted(nodes)


def select(
    rows: list[dict[str, object]],
    *,
    gpu_count: int | None = None,
    tokens_per_gpu: int | None = None,
    global_experts: int | None = None,
) -> list[dict[str, object]]:
    selected = rows
    if gpu_count is not None:
        selected = [row for row in selected if int(row["gpu_count"]) == gpu_count]
    if tokens_per_gpu is not None:
        selected = [
            row for row in selected if int(row["tokens_per_gpu"]) == tokens_per_gpu
        ]
    if global_experts is not None:
        selected = [
            row for row in selected if int(row["global_experts"]) == global_experts
        ]
    return selected


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def add_bar_labels(axis: plt.Axes, bars, *, decimals: int = 1) -> None:
    labels = [f"{bar.get_height():.{decimals}f}" for bar in bars]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=9)


def footer(figure: plt.Figure, partition: str, nodes: list[str]) -> None:
    figure.text(
        0.5,
        0.012,
        (
            f"{partition} | {', '.join(nodes)} | FP32 | SM120 generic | pS=1 | "
            "router excluded | warmup 32 + mean of 32 runs"
        ),
        ha="center",
        fontsize=8,
        color="#444444",
    )


def save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"plot={path}")


def plot_figure8(
    rows: list[dict[str, object]], output_dir: Path, partition: str, nodes: list[str]
) -> None:
    data = sorted(
        select(rows, gpu_count=4, global_experts=32),
        key=lambda row: int(row["tokens_per_gpu"]),
    )
    if not data:
        return
    tokens = [int(row["tokens_per_gpu"]) for row in data]
    latency = [float(row["max_rank_mean_latency_ms"]) for row in data]
    labels = [f"{value // 1024}K" for value in tokens]

    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    bars = axis.bar(
        labels,
        latency,
        width=0.58,
        color=FLASHMOE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        label="FlashMoE",
    )
    add_bar_labels(axis, bars)
    axis.set_xlabel("Number of tokens per GPU")
    axis.set_ylabel("Runtime (ms)")
    axis.set_title("Forward Latency | E=32 | k=2 | 4 RTX PRO 6000s | ↓ is better")
    axis.legend(frameon=False)
    axis.set_ylim(0, max(latency) * 1.18)
    style_axis(axis)
    footer(figure, partition, nodes)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save(figure, output_dir / "figure8_forward_latency_asus.png")


def scaling_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        select(rows, tokens_per_gpu=8192, global_experts=32),
        key=lambda row: int(row["gpu_count"]),
    )


def plot_figure10(
    rows: list[dict[str, object]], output_dir: Path, partition: str, nodes: list[str]
) -> None:
    data = scaling_rows(rows)
    if not data:
        return
    gpu_counts = [int(row["gpu_count"]) for row in data]
    throughput = [float(row["throughput_mtokens_s"]) for row in data]

    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    bars = axis.bar(
        [str(value) for value in gpu_counts],
        throughput,
        width=0.52,
        color=FLASHMOE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        label="FlashMoE",
    )
    add_bar_labels(axis, bars, decimals=3)
    axis.set_xlabel("Number of GPUs")
    axis.set_ylabel("Throughput (MTokens/s)")
    axis.set_title("Throughput | T=8K/GPU | E=32 | k=2 | ↑ is better")
    axis.legend(frameon=False)
    axis.set_ylim(0, max(throughput) * 1.2)
    style_axis(axis)
    footer(figure, partition, nodes)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save(figure, output_dir / "figure10_throughput_asus.png")


def plot_figure11(
    rows: list[dict[str, object]], output_dir: Path, partition: str, nodes: list[str]
) -> None:
    data = scaling_rows(rows)
    if not data:
        return
    gpu_counts = [int(row["gpu_count"]) for row in data]
    latency = [float(row["max_rank_mean_latency_ms"]) for row in data]
    latency_2gpu = next(
        value for gpu, value in zip(gpu_counts, latency, strict=True) if gpu == 2
    )
    efficiency = [100.0 * latency_2gpu / value for value in latency]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    bars_latency = axes[0].bar(
        [str(value) for value in gpu_counts],
        latency,
        width=0.52,
        color=FLASHMOE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
    )
    add_bar_labels(axes[0], bars_latency)
    axes[0].set_xlabel("Number of GPUs")
    axes[0].set_ylabel("Runtime (ms)")
    axes[0].set_title("(a) Forward latency | ↓ is better")
    axes[0].set_ylim(0, max(latency) * 1.18)
    style_axis(axes[0])

    bars_efficiency = axes[1].bar(
        [str(value) for value in gpu_counts],
        efficiency,
        width=0.52,
        color=FLASHMOE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
    )
    add_bar_labels(axes[1], bars_efficiency)
    axes[1].axhline(100.0, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of GPUs")
    axes[1].set_ylabel("Overlap efficiency (%)")
    axes[1].set_title("(b) Weak-scaling efficiency | ↑ is better")
    axes[1].set_ylim(0, 112)
    style_axis(axes[1])

    figure.suptitle("T=8K/GPU | E=32 | k=2", y=0.99, fontsize=12)
    footer(figure, partition, nodes)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(figure, output_dir / "figure11_weak_scaling_asus.png")


def plot_figure12(
    rows: list[dict[str, object]], output_dir: Path, partition: str, nodes: list[str]
) -> None:
    data = sorted(
        select(rows, gpu_count=4, tokens_per_gpu=16384),
        key=lambda row: int(row["global_experts"]),
    )
    if not data:
        return
    experts = [int(row["global_experts"]) for row in data]
    latency = [float(row["max_rank_mean_latency_ms"]) for row in data]

    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(experts))
    bars = axis.bar(
        x,
        latency,
        width=0.62,
        color=FLASHMOE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        label="FlashMoE",
    )
    add_bar_labels(axis, bars)
    axis.set_xticks(x, [str(value) for value in experts])
    axis.set_xlabel("Number of global experts")
    axis.set_ylabel("Runtime (ms)")
    axis.set_title("Expert Scalability | T=16K/GPU | k=2 | 4 RTX PRO 6000s | ↓ is better")
    axis.legend(frameon=False)
    axis.set_ylim(0, max(latency) * 1.17)
    style_axis(axis)
    footer(figure, partition, nodes)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save(figure, output_dir / "figure12_expert_scalability_asus.png")


def write_plot_data(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "partition",
        "node",
        "gpu_count",
        "tokens_per_gpu",
        "global_tokens",
        "global_experts",
        "local_experts",
        "max_rank_mean_latency_ms",
        "throughput_mtokens_s",
        "max_error_pct",
        "job_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                int(item["gpu_count"]),
                int(item["tokens_per_gpu"]),
                int(item["global_experts"]),
            ),
        ):
            writer.writerow({field: row[field] for field in fieldnames})


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, partition, nodes = load_jobs(args.job_dir)
    configure_style()
    write_plot_data(output_dir / "plot_data.csv", rows)
    plot_figure8(rows, output_dir, partition, nodes)
    plot_figure10(rows, output_dir, partition, nodes)
    plot_figure11(rows, output_dir, partition, nodes)
    plot_figure12(rows, output_dir, partition, nodes)
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
