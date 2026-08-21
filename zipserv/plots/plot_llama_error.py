#!/usr/bin/env python3
"""Plot selected Split-K and ZipGEMM/cuBLAS_TC errors for one LLaMA model."""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path

from plot_results import save_svg, svg_document, svg_to_png


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "zipserv-rtx4090-synthetic-20260818"
LAYERS = ("qkv_proj", "o_proj", "gateup_proj", "down_proj", "lm_head")
LAYER_LABELS = {
    "qkv_proj": "QKV",
    "o_proj": "O",
    "gateup_proj": "GateUp",
    "down_proj": "Down",
    "lm_head": "LM head",
}
BATCHES = (8, 16, 32)
BATCH_COLORS = {8: "#2b83ba", 16: "#fdae61", 32: "#4d7c0f"}

# Diagnostic guides, not universal pass/fail criteria. BF16 epsilon is a useful
# scale-free reference for relative differences. Absolute tolerances necessarily
# depend on output scale, so both a strict watch line and a loose line are shown.
BF16_EPSILON_PERCENT = 100.0 / 128.0
RELATIVE_WATCH_PERCENT = 1.0
ABSOLUTE_STRICT_GUIDE = 1e-4
ABSOLUTE_LOOSE_GUIDE = 1e-3


@dataclass(frozen=True)
class Point:
    layer: str
    n: int
    split_k: int
    average_relative_error_percent: float
    mean_absolute_error: float


def read_points(results_root: Path, model: str) -> list[Point]:
    path = results_root / "result_all.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Consolidated result not found: {path}")
    points: list[Point] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != "run" or row.get("status") != "ok":
                continue
            if row.get("model") != model:
                continue
            m = int(row["M"])
            n = int(row["N"])
            total_absolute_error = float(row["zip_vs_tc_total_absolute_error"])
            points.append(
                Point(
                    layer=row["layer"],
                    n=n,
                    split_k=int(row["split_k"]),
                    average_relative_error_percent=float(row["zip_vs_tc_average_relative_error"]) * 100.0,
                    mean_absolute_error=total_absolute_error / (m * n),
                )
            )
    lookup = {(point.layer, point.n): point for point in points}
    missing = [(layer, n) for layer in LAYERS for n in BATCHES if (layer, n) not in lookup]
    if missing:
        raise ValueError(f"Missing final points for {model}: {missing}")
    return [lookup[(layer, n)] for layer in LAYERS for n in BATCHES]


def add_style(svg: list[str]) -> None:
    svg.append(
        """<style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#172033}
        .main-title{font-size:27px;font-weight:700}
        .subtitle{font-size:15px;fill:#526078}
        .panel-title{font-size:20px;font-weight:700}
        .axis{font-size:13px}
        .value{font-size:11px;font-weight:700}
        .layer{font-size:14px;font-weight:700}
        .legend{font-size:14px}
        .note{font-size:13px;fill:#5b6474}
        .grid{stroke:#dce2ea;stroke-width:1}
        .axis-line{stroke:#283548;stroke-width:1.4}
        .guide{stroke-width:2;stroke-dasharray:8 5}
        .bar{stroke:#263342;stroke-width:.8}
        </style>"""
    )


def linear_axis(
    svg: list[str], left: float, right: float, top: float, bottom: float,
    ymax: float, ticks: list[float], formatter,
):
    def y(value: float) -> float:
        return bottom - value / ymax * (bottom - top)

    for tick in ticks:
        yy = y(tick)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" class="axis">{formatter(tick)}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis-line"/>')
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>')
    return y


def log_axis(svg: list[str], left: float, right: float, top: float, bottom: float):
    minimum, maximum = 1e-6, 1e-3

    def y(value: float) -> float:
        clipped = max(minimum, min(maximum, value))
        fraction = (math.log10(clipped) - math.log10(minimum)) / 3.0
        return bottom - fraction * (bottom - top)

    for tick in (1e-6, 1e-5, 1e-4, 1e-3):
        yy = y(tick)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" class="axis">{tick:.0e}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis-line"/>')
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>')
    return y


def draw_groups(svg: list[str], points: list[Point], left: float, right: float, top: float, bottom: float, y, value, label) -> None:
    group_width = (right - left) / len(LAYERS)
    bar_step = group_width / 4.2
    bar_width = bar_step * 0.72
    for layer_index, layer in enumerate(LAYERS):
        group_left = left + layer_index * group_width
        if layer_index % 2 == 0:
            svg.append(f'<rect x="{group_left:.2f}" y="{top}" width="{group_width:.2f}" height="{bottom-top:.2f}" fill="#f5f7fa"/>')
        center = group_left + group_width / 2
        layer_points = [point for point in points if point.layer == layer]
        for batch_index, point in enumerate(layer_points):
            x = center + (batch_index - 1) * bar_step - bar_width / 2
            raw = value(point)
            yy = y(raw)
            svg.append(f'<rect x="{x:.2f}" y="{yy:.2f}" width="{bar_width:.2f}" height="{bottom-yy:.2f}" fill="{BATCH_COLORS[point.n]}" class="bar"/>')
            svg.append(f'<text x="{x + bar_width/2:.2f}" y="{max(top + 12, yy - 5):.2f}" text-anchor="middle" class="value">{html.escape(label(point))}</text>')
        svg.append(f'<text x="{center:.2f}" y="{bottom + 21}" text-anchor="middle" class="layer">{LAYER_LABELS[layer]}</text>')


def plot(path: Path, points: list[Point], model: str) -> None:
    width, height = 1500, 1120
    left, right = 105, width - 35
    panels = ((145, 365), (465, 705), (805, 1030))
    svg = svg_document(width, height)
    add_style(svg)
    svg.append(f'<text x="{width/2}" y="38" text-anchor="middle" class="main-title">{html.escape(model)} — selected Split-K and numerical difference</text>')
    svg.append(f'<text x="{width/2}" y="65" text-anchor="middle" class="subtitle">Synthetic BF16 weights, RTX 4090 final runs; error = ZipGEMM BF16 output vs cuBLAS_TC BF16 output</text>')
    legend_x = width / 2 - 185
    for batch in BATCHES:
        svg.append(f'<rect x="{legend_x}" y="91" width="18" height="14" fill="{BATCH_COLORS[batch]}" class="bar"/>')
        svg.append(f'<text x="{legend_x + 25}" y="103" class="legend">N={batch}</text>')
        legend_x += 125

    top, bottom = panels[0]
    y = linear_axis(svg, left, right, top, bottom, 8.8, [0, 2, 4, 6, 8], lambda x: str(int(x)))
    svg.append(f'<text x="{left}" y="{top - 16}" class="panel-title">A. Selected Split-K from tuning</text>')
    draw_groups(svg, points, left, right, top, bottom, y, lambda p: p.split_k, lambda p: str(p.split_k))

    top, bottom = panels[1]
    ymax = max(2.5, math.ceil(max(p.average_relative_error_percent for p in points) * 5) / 5 + 0.2)
    ticks = [value / 2 for value in range(0, int(ymax * 2) + 1)]
    y = linear_axis(svg, left, right, top, bottom, ymax, ticks, lambda x: f"{x:.1f}%")
    svg.append(f'<text x="{left}" y="{top - 16}" class="panel-title">B. Mean element-wise relative error</text>')
    draw_groups(svg, points, left, right, top, bottom, y, lambda p: p.average_relative_error_percent, lambda p: f"{p.average_relative_error_percent:.2f}%")
    for value, color, text_value in (
        (BF16_EPSILON_PERCENT, "#7b61a8", "BF16 epsilon ≈ 0.781%"),
        (RELATIVE_WATCH_PERCENT, "#c23b33", "1% diagnostic watch line"),
    ):
        yy = y(value)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="{color}" class="guide"/>')
        svg.append(f'<text x="{right - 5}" y="{yy - 6:.2f}" text-anchor="end" fill="{color}" class="legend">{text_value}</text>')

    top, bottom = panels[2]
    y = log_axis(svg, left, right, top, bottom)
    svg.append(f'<text x="{left}" y="{top - 16}" class="panel-title">C. Mean absolute error per output element (log scale)</text>')
    draw_groups(svg, points, left, right, top, bottom, y, lambda p: p.mean_absolute_error, lambda p: "0 exact" if p.mean_absolute_error == 0 else f"{p.mean_absolute_error:.1e}")
    for value, color, text_value in (
        (ABSOLUTE_STRICT_GUIDE, "#c23b33", "1e-4 strict watch line"),
        (ABSOLUTE_LOOSE_GUIDE, "#7b61a8", "1e-3 loose guide"),
    ):
        yy = y(value)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="{color}" class="guide"/>')
        svg.append(f'<text x="{right - 5}" y="{yy - 6:.2f}" text-anchor="end" fill="{color}" class="legend">{text_value}</text>')
    svg.append(f'<text x="{left}" y="{height - 18}" class="note">Guides are diagnostic references, not universal pass/fail thresholds. Absolute tolerances must be interpreted relative to output scale.</text>')
    save_svg(path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model", default="llama3.1-8b")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "llama31_8b_splitk_error.png")
    parser.add_argument("--png-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    points = read_points(results_root, args.model)
    svg_path = output.with_suffix(".svg")
    plot(svg_path, points, args.model)
    svg_to_png(svg_path, output, args.png_scale)
    svg_path.unlink()
    print(f"Created: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
