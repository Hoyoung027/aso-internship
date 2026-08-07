#!/usr/bin/env python3
"""Plot latency and speedup figures from a GEMM raw_all.csv file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT_DIR / "result"
OPERATIONS = ["qkv", "o", "router_gate", "expert_w13", "expert_w2"]
OPERATION_TITLES = {
    "qkv": "QKV (K=2880, N=5120)",
    "o": "O (K=4096, N=2880)",
    "router_gate": "Router (K=2880, N=32)",
    "expert_w13": "Expert W13 (K=2880, N=5760)",
    "expert_w2": "Expert W2 (K=2880, N=2880)",
}
EXPERIMENT_LABELS = {
    "torch_bf16": "PyTorch BF16 F.linear",
    "torch_mxfp8": "PyTorch MXFP8 _scaled_mm",
    "flashinfer_bf16_default": "FlashInfer BF16 default",
    "flashinfer_bf16_tuned": "FlashInfer BF16 tuned",
    "flashinfer_mxfp8_default": "FlashInfer MXFP8 default",
    "flashinfer_mxfp8_tuned": "FlashInfer MXFP8 tuned",
    "flashinfer_mxfp4_default": "FlashInfer MXFP4 default",
    "flashinfer_mxfp4_tuned": "FlashInfer MXFP4 tuned",
}
COLORS = {
    "torch_bf16": "#202124",
    "torch_mxfp8": "#6a1b9a",
    "flashinfer_bf16_default": "#90caf9",
    "flashinfer_bf16_tuned": "#1565c0",
    "flashinfer_mxfp8_default": "#ffcc80",
    "flashinfer_mxfp8_tuned": "#ef6c00",
    "flashinfer_mxfp4_default": "#a5d6a7",
    "flashinfer_mxfp4_tuned": "#2e7d32",
}
MARKERS = {
    "torch_bf16": "o",
    "torch_mxfp8": "o",
    "flashinfer_bf16_default": "s",
    "flashinfer_bf16_tuned": "^",
    "flashinfer_mxfp8_default": "s",
    "flashinfer_mxfp8_tuned": "^",
    "flashinfer_mxfp4_default": "s",
    "flashinfer_mxfp4_tuned": "^",
}
PRECISION_GROUPS = {
    "bf16": {
        "title": "BF16 latency by GPT-OSS-20B GEMM case",
        "experiments": [
            "torch_bf16",
            "flashinfer_bf16_default",
            "flashinfer_bf16_tuned",
        ],
    },
    "mxfp8": {
        "title": "MXFP8 latency by GPT-OSS-20B GEMM case",
        "experiments": [
            "torch_mxfp8",
            "flashinfer_mxfp8_default",
            "flashinfer_mxfp8_tuned",
        ],
    },
    "mxfp4": {
        "title": "MXFP4 latency by GPT-OSS-20B GEMM case",
        "experiments": [
            "flashinfer_mxfp4_default",
            "flashinfer_mxfp4_tuned",
        ],
    },
}
AUTOTUNE_PAIRS = {
    "BF16": ("flashinfer_bf16_default", "flashinfer_bf16_tuned"),
    "MXFP8": ("flashinfer_mxfp8_default", "flashinfer_mxfp8_tuned"),
    "MXFP4": ("flashinfer_mxfp4_default", "flashinfer_mxfp4_tuned"),
}
X_TICKS = [1, 4, 16, 64, 256, 1024, 4096, 8192, 16384, 32768]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Result directory containing raw_all.csv; defaults to the latest run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PNG output directory; defaults to <run-dir>/plots.",
    )
    return parser.parse_args()


def find_latest_run() -> Path:
    candidates = [path.parent for path in RESULT_ROOT.glob("*/raw_all.csv")]
    if not candidates:
        raise FileNotFoundError(f"No raw_all.csv found below {RESULT_ROOT}")
    return max(candidates, key=lambda path: (path / "raw_all.csv").stat().st_mtime)


def load_trial_statistics(run_dir: Path) -> pd.DataFrame:
    input_path = run_dir / "raw_all.csv"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    frame = pd.read_csv(input_path, low_memory=False)
    frame = frame.loc[frame["status"].eq("ok")].copy()
    for column in ("trial", "M", "K", "N", "latency_ms"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    keys = ["projection", "experiment_id", "M", "K", "N", "trial"]
    trial_means = (
        frame.groupby(keys, as_index=False, observed=True)["latency_ms"]
        .mean()
        .rename(columns={"latency_ms": "trial_mean_ms"})
    )
    aggregate_keys = ["projection", "experiment_id", "M", "K", "N"]
    statistics = (
        trial_means.groupby(aggregate_keys, as_index=False, observed=True)
        .agg(
            mean_ms=("trial_mean_ms", "mean"),
            trial_std_ms=("trial_mean_ms", "std"),
            trial_count=("trial", "nunique"),
        )
        .sort_values(aggregate_keys)
    )
    statistics["trial_std_ms"] = statistics["trial_std_ms"].fillna(0.0)
    return statistics


def configure_axis(axis: plt.Axes, *, max_m: int, log_y: bool = False) -> None:
    axis.set_xscale("log", base=2)
    ticks = [value for value in X_TICKS if value <= max_m]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(value) for value in ticks], rotation=45)
    axis.set_xlim(1, max_m)
    if log_y:
        axis.set_yscale("log")
    axis.grid(True, which="both", linewidth=0.5, alpha=0.28)
    axis.set_xlabel("GEMM rows (M)")


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"plot={output_path}")


def plot_latency_by_precision(statistics: pd.DataFrame, output_dir: Path) -> None:
    max_m = int(statistics["M"].max())
    for precision, group in PRECISION_GROUPS.items():
        experiments = group["experiments"]
        figure, axes = plt.subplots(1, len(OPERATIONS), figsize=(24, 4.8))
        for axis, operation in zip(axes, OPERATIONS, strict=True):
            operation_data = statistics.loc[statistics["projection"].eq(operation)]
            for experiment in experiments:
                data = operation_data.loc[
                    operation_data["experiment_id"].eq(experiment)
                ].sort_values("M")
                if data.empty:
                    continue
                x = data["M"].to_numpy(dtype=float)
                mean = data["mean_ms"].to_numpy(dtype=float)
                std = data["trial_std_ms"].to_numpy(dtype=float)
                color = COLORS[experiment]
                axis.plot(
                    x,
                    mean,
                    color=color,
                    marker=MARKERS[experiment],
                    markersize=3.5,
                    linewidth=1.7,
                    label=EXPERIMENT_LABELS[experiment],
                )
                axis.fill_between(
                    x,
                    np.maximum(mean - std, np.finfo(float).tiny),
                    mean + std,
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                )
            if not axis.lines:
                axis.text(0.5, 0.5, "Unsupported", ha="center", va="center")
            axis.set_title(OPERATION_TITLES[operation], fontsize=10)
            configure_axis(axis, max_m=max_m, log_y=True)
        axes[0].set_ylabel("Mean latency (ms, log scale)")
        handles = [
            Line2D(
                [0],
                [0],
                color=COLORS[experiment],
                marker=MARKERS[experiment],
                linewidth=1.8,
                label=EXPERIMENT_LABELS[experiment],
            )
            for experiment in experiments
        ]
        figure.suptitle(group["title"], y=0.985, fontsize=14)
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=len(handles),
        )
        figure.text(
            0.5,
            0.015,
            "Line: mean of 3 trial means; band: ±1 standard deviation across trials.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.86))
        save_figure(figure, output_dir / f"latency_{precision}_by_operation.png")


def merge_speedup(
    statistics: pd.DataFrame,
    operation: str,
    baseline_experiment: str,
    target_experiment: str,
) -> pd.DataFrame:
    columns = ["M", "mean_ms"]
    operation_data = statistics.loc[statistics["projection"].eq(operation)]
    baseline = operation_data.loc[
        operation_data["experiment_id"].eq(baseline_experiment), columns
    ].rename(columns={"mean_ms": "baseline_ms"})
    target = operation_data.loc[
        operation_data["experiment_id"].eq(target_experiment), columns
    ].rename(columns={"mean_ms": "target_ms"})
    merged = baseline.merge(target, on="M", how="inner").sort_values("M")
    merged["speedup"] = merged["baseline_ms"] / merged["target_ms"]
    return merged


def apply_shared_speedup_limits(
    axes: np.ndarray, speedup_values: list[np.ndarray]
) -> tuple[float, float]:
    """Apply one y-axis range to every panel in a speedup figure."""
    finite_values = [
        values[np.isfinite(values)] for values in speedup_values if values.size
    ]
    finite_values = [values for values in finite_values if values.size]
    if finite_values:
        combined = np.concatenate(finite_values)
        lower_data = min(1.0, float(combined.min()))
        upper_data = max(1.0, float(combined.max()))
    else:
        lower_data = upper_data = 1.0

    data_span = upper_data - lower_data
    padding = max(
        data_span * 0.05,
        max(abs(lower_data), abs(upper_data), 1.0) * 0.02,
    )
    lower_limit = max(0.0, lower_data - padding)
    upper_limit = upper_data + padding
    for axis in axes.flat:
        axis.set_ylim(lower_limit, upper_limit)
    return lower_limit, upper_limit


def plot_autotune_speedup(statistics: pd.DataFrame, output_dir: Path) -> None:
    max_m = int(statistics["M"].max())
    figure, axes = plt.subplots(
        len(AUTOTUNE_PAIRS), len(OPERATIONS), figsize=(24, 12), squeeze=False
    )
    for row, (precision, (default_id, tuned_id)) in enumerate(
        AUTOTUNE_PAIRS.items()
    ):
        for column, operation in enumerate(OPERATIONS):
            axis = axes[row, column]
            data = merge_speedup(statistics, operation, default_id, tuned_id)
            if data.empty:
                axis.text(0.5, 0.5, "Unsupported", ha="center", va="center")
            else:
                axis.plot(
                    data["M"],
                    data["speedup"],
                    color=COLORS[tuned_id],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.7,
                )
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
            axis.set_title(OPERATION_TITLES[operation], fontsize=10)
            configure_axis(axis, max_m=max_m)
            if column == 0:
                axis.set_ylabel(f"{precision}\nDefault / tuned")
    for axis in axes[0, :]:
        axis.set_ylim(0.0, 2.0)
    for axis in axes[1:, :].flat:
        axis.set_ylim(0.0, 2.5)
    figure.suptitle(
        "FlashInfer AutoTuner speedup (>1 means tuned is faster)",
        y=1.01,
        fontsize=14,
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "flashinfer_autotune_speedup.png")


def plot_speedup_vs_matching_torch_precision(
    statistics: pd.DataFrame, output_dir: Path
) -> None:
    max_m = int(statistics["M"].max())
    rows = {
        "BF16": {
            "baseline": "torch_bf16",
            "experiments": [
                "flashinfer_bf16_default",
                "flashinfer_bf16_tuned",
            ],
        },
        "MXFP8": {
            "baseline": "torch_mxfp8",
            "experiments": [
                "flashinfer_mxfp8_default",
                "flashinfer_mxfp8_tuned",
            ],
        },
    }
    figure, axes = plt.subplots(
        len(rows), len(OPERATIONS), figsize=(24, 10), squeeze=False
    )
    low_precision_speedup_values: list[np.ndarray] = []
    for row, (precision, group) in enumerate(rows.items()):
        baseline = group["baseline"]
        experiments = group["experiments"]
        for column, operation in enumerate(OPERATIONS):
            axis = axes[row, column]
            for experiment in experiments:
                data = merge_speedup(
                    statistics, operation, baseline, experiment
                )
                if data.empty:
                    continue
                if row > 0:
                    low_precision_speedup_values.append(
                        data["speedup"].to_numpy(dtype=float)
                    )
                axis.plot(
                    data["M"],
                    data["speedup"],
                    color=COLORS[experiment],
                    marker=MARKERS[experiment],
                    markersize=3.5,
                    linewidth=1.7,
                    label=EXPERIMENT_LABELS[experiment],
                )
            if not axis.lines:
                axis.text(0.5, 0.5, "Unsupported", ha="center", va="center")
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
            axis.set_title(OPERATION_TITLES[operation], fontsize=10)
            configure_axis(axis, max_m=max_m)
            if column == 0:
                baseline_precision = "BF16" if baseline == "torch_bf16" else "MXFP8"
                axis.set_ylabel(
                    f"{precision}\nTorch {baseline_precision} / FlashInfer"
                )
    for axis in axes[0, :]:
        axis.set_ylim(0.0, 2.0)
    apply_shared_speedup_limits(axes[1:, :], low_precision_speedup_values)
    legend_experiments = [
        experiment
        for group in rows.values()
        for experiment in group["experiments"]
    ]
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[experiment],
            marker=MARKERS[experiment],
            linewidth=1.8,
            label=EXPERIMENT_LABELS[experiment],
        )
        for experiment in legend_experiments
    ]
    figure.suptitle(
        "Kernel-only speedup vs matching PyTorch precision (>1 means FlashInfer is faster)",
        y=0.995,
        fontsize=14,
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
    )
    figure.text(
        0.5,
        0.008,
        "Both MXFP8 paths use identical E4M3 values and E8M0 block-32 scales; quantization time is excluded. "
        "MXFP4 is omitted because no PyTorch MXFP4 baseline was measured.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.12,
        top=0.86,
        wspace=0.28,
        hspace=0.42,
    )
    save_figure(figure, output_dir / "speedup_vs_matching_torch_precision.png")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir else find_latest_run().resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else run_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    statistics = load_trial_statistics(run_dir)
    plot_latency_by_precision(statistics, output_dir)
    plot_autotune_speedup(statistics, output_dir)
    plot_speedup_vs_matching_torch_precision(statistics, output_dir)
    print(f"run_dir={run_dir}")
    print(f"plot_dir={output_dir}")


if __name__ == "__main__":
    main()
