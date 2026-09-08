#!/usr/bin/env python3
"""Compare ZipServ numerical error under bf16 vs fp32 partial-sum accumulation.

Both variants are measured against the same reference, cuBLAS TC (which is
shown as an explicit zero-error baseline bar). Reuses the panel layout and
error metrics from plot_llama_error.py, generalized from one bar per layer
to a 3-bar cluster (cuBLAS TC / ZipServ bf16 / ZipServ fp32).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from plot_results import save_svg, svg_document, svg_to_png
from plot_llama_error import (
    DEFAULT_MODELS,
    LAYER_LABELS,
    METRICS,
    ErrorPoint,
    Metric,
    add_style,
    format_tick,
    linear_bounds,
    load_config,
    load_selected_points,
    log_bounds,
    model_display_name,
)

METHODS = ("tc", "zip_bf16", "zip_fp32")
METHOD_LABELS = {
    "tc": "cuBLAS TC (reference, error = 0)",
    "zip_bf16": "ZipServ · bf16 partial sums",
    "zip_fp32": "ZipServ · fp32 partial sums",
}
METHOD_COLORS = {
    "tc": "#9aa5b1",
    "zip_bf16": "#e8743b",
    "zip_fp32": "#2b83ba",
}


def tc_reference_points(points: list[ErrorPoint]) -> list[ErrorPoint]:
    # cuBLAS TC compared against itself: every error metric is exactly zero,
    # by construction rather than measurement.
    return [
        ErrorPoint(
            model=p.model, layer=p.layer, m=p.m, k=p.k, n=p.n, split_k=p.split_k,
            zip_relative_percent=0.0, zip_mae=0.0, zip_significant_percent=0.0,
            zip_absolute_significant_percent=0.0,
        )
        for p in points
    ]


def metric_scales(points_by_method: dict[str, list[ErrorPoint]]) -> list[tuple[float, float, list[float]]]:
    scales = []
    for metric in METRICS:
        values = [
            value
            for points in points_by_method.values()
            for point in points
            for value in (metric.zip_value(point),)
            if value is not None
        ]
        scales.append(log_bounds(values) if metric.scale.startswith("log") else
                      linear_bounds(values) if values else (0.0, 100.0, [0, 20, 40, 60, 80, 100]))
    return scales


def draw_panel(
    svg: list[str], points_by_method: dict[str, list[ErrorPoint]], methods: list[str],
    layers: list[str], metric: Metric, bounds: tuple[float, float, list[float]],
    left: float, right: float, top: float, bottom: float,
) -> None:
    minimum, maximum, ticks = bounds
    group_width = (right - left) / len(layers)
    for index in range(len(layers)):
        if index % 2 == 0:
            group_left = left + index * group_width
            svg.append(f'<rect x="{group_left:.2f}" y="{top}" width="{group_width:.2f}" height="{bottom - top:.2f}" fill="#f6f8fb"/>')
    if metric.scale.startswith("log"):
        log_minimum = math.log10(minimum)
        log_span = math.log10(maximum) - log_minimum

        def y(value: float) -> float:
            clipped = max(minimum, min(maximum, value))
            fraction = (math.log10(clipped) - log_minimum) / log_span
            return bottom - fraction * (bottom - top)
    else:
        def y(value: float) -> float:
            return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    for tick in ticks:
        yy = y(tick)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{left - 7}" y="{yy + 4:.2f}" text-anchor="end" class="axis">{format_tick(tick, metric.scale)}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis-line"/>')
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>')

    lookups = {method: {p.layer: p for p in pts} for method, pts in points_by_method.items()}
    n_methods = len(methods)
    cluster_width = group_width * 0.72
    bar_gap = 5.0
    bar_width = min(24.0, (cluster_width - bar_gap * (n_methods - 1)) / n_methods)
    cluster_span = bar_width * n_methods + bar_gap * (n_methods - 1)

    for index, layer in enumerate(layers):
        group_left = left + index * group_width
        center = group_left + group_width / 2
        cluster_left = center - cluster_span / 2
        any_point = next((lookups[m][layer] for m in methods if layer in lookups[m]), None)
        if any_point is None:
            continue
        bar_tops: list[float] = []
        for m_index, method in enumerate(methods):
            point = lookups[method].get(layer)
            bx = cluster_left + m_index * (bar_width + bar_gap)
            bar_center = bx + bar_width / 2
            # Only the zero/missing status is called out per bar (rotated to fit
            # the narrow per-method bar); the measured value itself is not
            # printed here -- Split-K is what matters for this comparison, and
            # it is shown once above the whole cluster below.
            if point is None or metric.zip_value(point) is None:
                label_y = bottom - 4
                svg.append(f'<text x="{bar_center:.2f}" y="{label_y:.2f}" text-anchor="start" class="val-label" transform="rotate(-90 {bar_center:.2f} {label_y:.2f})">N/A</text>')
                continue
            value = metric.zip_value(point)
            if value > 0:
                yy = y(value)
                svg.append(f'<rect x="{bx:.2f}" y="{yy:.2f}" width="{bar_width:.2f}" height="{bottom - yy:.2f}" rx="2" fill="{METHOD_COLORS[method]}" class="bar"/>')
                bar_tops.append(yy)
            else:
                # A zero error (the cuBLAS TC reference) cannot sit on a log axis: label it explicitly.
                label_y = bottom - 4
                svg.append(f'<text x="{bar_center:.2f}" y="{label_y:.2f}" text-anchor="start" class="val-label" transform="rotate(-90 {bar_center:.2f} {label_y:.2f})">0</text>')
        cluster_top = min(bar_tops) if bar_tops else bottom - 14
        klabel_y = max(top + 12, cluster_top - 6)
        svg.append(f'<text x="{center:.2f}" y="{klabel_y:.2f}" text-anchor="middle" class="k-label">K={any_point.split_k}</text>')
        svg.append(f'<text x="{center:.2f}" y="{bottom + 18}" text-anchor="middle" class="layer">{LAYER_LABELS.get(layer, layer)}</text>')
        svg.append(f'<text x="{center:.2f}" y="{bottom + 33}" text-anchor="middle" class="shape">{any_point.m:,}×{any_point.k:,}</text>')

    if all(layer not in lookups[method] for method in methods for layer in layers):
        svg.append(f'<text x="{(left + right) / 2}" y="{(top + bottom) / 2}" text-anchor="middle" class="note">Not measured in this run</text>')


def plot_model(
    path: Path, model: str, points_by_method: dict[str, list[ErrorPoint]], methods: list[str],
    layers: list[str], batches: list[int], scales: list[tuple[float, float, list[float]]],
    block_index: int,
) -> None:
    width, height = 1800, 1950
    svg = svg_document(width, height)
    add_style(svg)
    svg.append("<style>.val-label{font-size:9px;font-weight:600}</style>")
    svg.append(f'<text x="{width / 2}" y="38" text-anchor="middle" class="title">{model_display_name(model)} — partial-sum dtype error comparison</text>')
    scope = f'block {block_index}' + (' + shared LM Head' if 'lm_head' in layers else '')
    svg.append(f'<text x="{width / 2}" y="65" text-anchor="middle" class="subtitle">Reference: cuBLAS TC · real weights ({scope}) · synthetic BF16 activations · final run, median of 3 trials</text>')

    legend_gap, swatch_w = 260, 28
    legend_x = width / 2 - (legend_gap * (len(methods) - 1)) / 2 - 90
    for method in methods:
        svg.append(f'<rect x="{legend_x:.2f}" y="87" width="{swatch_w}" height="16" rx="2" fill="{METHOD_COLORS[method]}" class="bar"/>')
        svg.append(f'<text x="{legend_x + swatch_w + 10:.2f}" y="100" class="legend">{METHOD_LABELS[method]}</text>')
        legend_x += legend_gap

    outer_left, outer_right, column_gap = 92.0, width - 28.0, 28.0
    column_width = (outer_right - outer_left - column_gap * (len(batches) - 1)) / len(batches)
    row_bounds = ((180.0, 470.0), (640.0, 930.0), (1100.0, 1390.0), (1560.0, 1850.0))
    for batch_index, batch in enumerate(batches):
        panel_left = outer_left + batch_index * (column_width + column_gap)
        svg.append(f'<text x="{panel_left + column_width / 2:.2f}" y="126" text-anchor="middle" class="batch-title">Batch size N={batch}</text>')

    for metric_index, (metric, bounds, (top, bottom)) in enumerate(zip(METRICS, scales, row_bounds)):
        svg.append(f'<text x="{width / 2}" y="{top - 27:.2f}" text-anchor="middle" class="row-title">{chr(65 + metric_index)}. {metric.title}</text>')
        axis_center = (top + bottom) / 2
        svg.append(f'<text x="18" y="{axis_center:.2f}" text-anchor="middle" transform="rotate(-90 18 {axis_center:.2f})" class="axis-label">{metric.axis_label}</text>')
        for batch_index, batch in enumerate(batches):
            panel_left = outer_left + batch_index * (column_width + column_gap)
            plot_left, plot_right = panel_left + 66, panel_left + column_width - 10
            batch_points_by_method = {
                method: [point for point in points if point.n == batch]
                for method, points in points_by_method.items()
            }
            draw_panel(svg, batch_points_by_method, methods, layers, metric, bounds, plot_left, plot_right, top, bottom)

    svg.append(f'<text x="{outer_left}" y="{height - 22}" class="note">K = tuned Split-K, shared by both runs; M×K = weight shape. N/A = unmeasured, not zero. cuBLAS TC is 0 by construction (compared against itself).</text>')
    save_svg(path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-root", type=Path, required=True, help="Real-weight final-run directory using bf16 partial sums")
    parser.add_argument("--fp32-root", type=Path, required=True, help="Real-weight final-run directory using fp32 partial sums")
    parser.add_argument("--block-index", type=int, default=0, help="Plot this block only; default: 0")
    parser.add_argument("--config", type=Path, help="Default: experiment snapshot in --fp32-root")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "image" / "real" / "llama")
    parser.add_argument("--png-scale", type=float, default=2.0)
    parser.add_argument("--keep-svg", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.png_scale <= 0:
        raise ValueError("--png-scale must be positive")
    bf16_root = args.bf16_root.expanduser().resolve()
    fp32_root = args.fp32_root.expanduser().resolve()
    config = load_config((args.config or fp32_root / "experiments.json").expanduser().resolve())
    specs = {model["id"]: model for model in config["models"]}
    unknown = [model for model in args.models if model not in specs]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if args.block_index < 0:
        raise ValueError("--block-index must be non-negative")

    bf16_points = load_selected_points(bf16_root, args.models, args.block_index)
    fp32_points = load_selected_points(fp32_root, args.models, args.block_index)
    for label, points in (("bf16", bf16_points), ("fp32", fp32_points)):
        missing_absolute = sum(p.zip_absolute_significant_percent is None for p in points)
        if missing_absolute:
            print(f'WARNING: {missing_absolute} {label} cases have no absolute-error exceedance data; panel D will show N/A.', file=sys.stderr)

    batches = [int(batch) for batch in config["matrix"]["batches"]]
    expected = {
        (model, layer["id"], batch)
        for model in args.models
        for layer in specs[model]["layers"]
        if layer.get("block_scoped", True) or args.block_index == 0
        for batch in batches
    }
    for label, points in (("bf16", bf16_points), ("fp32", fp32_points)):
        present = {(point.model, point.layer, point.n) for point in points}
        missing = sorted(expected - present)
        if missing:
            raise ValueError(f"Missing selected error cases in {label} run ({len(missing)}): {missing}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for model in args.models:
        model_bf16 = [point for point in bf16_points if point.model == model]
        model_fp32 = [point for point in fp32_points if point.model == model]
        points_by_method = {
            "tc": tc_reference_points(model_fp32),
            "zip_bf16": model_bf16,
            "zip_fp32": model_fp32,
        }
        scales = metric_scales(points_by_method)
        layers = [layer["id"] for layer in specs[model]["layers"]
                  if layer.get("block_scoped", True) or args.block_index == 0]
        suffix = "" if args.block_index == 0 else f"_block-{args.block_index}"
        svg_path = output_dir / f"{model}{suffix}_error_bf16_vs_fp32.svg"
        png_path = output_dir / f"{model}{suffix}_error_bf16_vs_fp32.png"
        plot_model(svg_path, model, points_by_method, list(METHODS), layers, batches, scales, args.block_index)
        svg_to_png(svg_path, png_path, args.png_scale)
        if not args.keep_svg:
            svg_path.unlink()
        created.append(png_path)
        print(f"Created: {png_path}")
    print(f"Created {len(created)} error comparison plot(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
