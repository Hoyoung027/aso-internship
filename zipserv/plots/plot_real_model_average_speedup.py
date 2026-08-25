#!/usr/bin/env python3
"""Compare real-weight model-average ZipServ speedup across model families."""

from __future__ import annotations

import argparse
import html
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from plot_real_layer_speedup import Point, display_model, read_points
from plot_results import save_svg, svg_document, svg_to_png


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "image" / "real"
BATCH_COLORS = {8: "#2b83ba", 16: "#f28e2b", 32: "#4d9221"}
FAMILY_BACKGROUNDS = {"LLaMA 3.1": "#edf4fa", "Qwen 2.5": "#edf7ef"}


@dataclass(frozen=True)
class AveragePoint:
    model: str
    batch: int
    mean: float
    minimum: float
    maximum: float
    stddev: float
    cases: int


def family(model: str) -> str:
    if model.startswith("llama3.1-"):
        return "LLaMA 3.1"
    if model.startswith("qwen2.5-"):
        return "Qwen 2.5"
    return model.split("-", 1)[0]


def summarize(points: list[Point], models: list[str], batches: list[int]) -> list[AveragePoint]:
    summarized: list[AveragePoint] = []
    for model in models:
        for batch in batches:
            # LM Head is shared across transformer blocks. Some LLaMA result
            # directories contain convenience copies, so only block 0 counts.
            values = [
                point.speedup
                for point in points
                if point.model == model
                and point.n == batch
                and (point.layer != "lm_head" or point.block == 0)
            ]
            if len(values) != 13:
                raise ValueError(
                    f"Expected 13 unique operations for {model}/N={batch}, got {len(values)}"
                )
            summarized.append(
                AveragePoint(
                    model=model,
                    batch=batch,
                    mean=statistics.mean(values),
                    minimum=min(values),
                    maximum=max(values),
                    stddev=statistics.pstdev(values),
                    cases=len(values),
                )
            )
    return summarized


def plot(
    svg_path: Path,
    averages: list[AveragePoint],
    models: list[str],
    batches: list[int],
) -> None:
    width, height = 2400, 1120
    left, right, top, bottom = 135.0, width - 55.0, 175.0, 925.0
    lower = 1.0
    upper = max(1.5, math.ceil((max(point.maximum for point in averages) + 0.02) / 0.1) * 0.1)
    group_width = (right - left) / len(models)
    lookup = {(point.model, point.batch): point for point in averages}

    def y(value: float) -> float:
        return bottom - (value - lower) / (upper - lower) * (bottom - top)

    svg = svg_document(width, height)
    svg.append(
        """<style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#172033}
        .title{font-size:34px;font-weight:700}.subtitle{font-size:18px;fill:#59677c}
        .axis{font-size:15px}.axis-label{font-size:19px;font-weight:700}
        .model{font-size:18px;font-weight:700}.family{font-size:18px;font-weight:700}
        .value{font-size:14px;font-weight:700}.legend{font-size:17px}
        .note{font-size:14px;fill:#64748b}.grid{stroke:#dce3ec;stroke-width:1}
        .axis-line{stroke:#283548;stroke-width:1.5}.baseline{stroke:#364152;stroke-width:2;stroke-dasharray:8 5}
        .whisker{stroke:#263342;stroke-width:2}.cap{stroke:#263342;stroke-width:2}
        </style>"""
    )
    svg.append(
        f'<text x="{width / 2}" y="44" text-anchor="middle" class="title">'
        'RTX 4090 — Real-weight model-average ZipServ speedup</text>'
    )
    svg.append(
        f'<text x="{width / 2}" y="75" text-anchor="middle" class="subtitle">'
        'Arithmetic mean of 13 independently normalized operations: 3 blocks × 4 projections + shared LM Head</text>'
    )

    legend_width = 185.0
    legend_x = (width - legend_width * len(batches) - 240) / 2
    for batch in batches:
        color = BATCH_COLORS.get(batch, "#555")
        svg.append(f'<rect x="{legend_x:.2f}" y="107" width="32" height="18" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 43:.2f}" y="123" class="legend">N={batch}</text>')
        legend_x += legend_width
    svg.append(f'<line x1="{legend_x}" y1="116" x2="{legend_x + 44}" y2="116" class="whisker"/>')
    svg.append(f'<line x1="{legend_x + 22}" y1="106" x2="{legend_x + 22}" y2="126" class="whisker"/>')
    svg.append(f'<text x="{legend_x + 57}" y="123" class="legend">min–max across operations</text>')

    # Family background bands.
    family_spans: dict[str, list[int]] = {}
    for index, model in enumerate(models):
        family_spans.setdefault(family(model), []).append(index)
    for name, indices in family_spans.items():
        x = left + min(indices) * group_width
        band_width = (max(indices) - min(indices) + 1) * group_width
        svg.append(
            f'<rect x="{x:.2f}" y="{top}" width="{band_width:.2f}" height="{bottom - top:.2f}" '
            f'fill="{FAMILY_BACKGROUNDS.get(name, "#f5f7fa")}"/>'
        )
        svg.append(
            f'<text x="{x + band_width / 2:.2f}" y="{top + 28:.2f}" '
            f'text-anchor="middle" class="family">{html.escape(name)}</text>'
        )

    tick = lower
    while tick <= upper + 1e-9:
        yy = y(tick)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{left - 15}" y="{yy + 5:.2f}" text-anchor="end" class="axis">{tick:.1f}×</text>')
        tick += 0.1
    baseline_y = y(1.0)
    svg.append(f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{right}" y2="{baseline_y:.2f}" class="baseline"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis-line"/>')
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>')
    center_y = (top + bottom) / 2
    svg.append(
        f'<text x="31" y="{center_y:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 31 {center_y:.2f})" class="axis-label">Mean cuBLAS TC / ZipServ</text>'
    )

    bar_step = min(105.0, group_width / (len(batches) + 1.0))
    bar_width = bar_step * 0.70
    cap_width = bar_width * 0.75
    for model_index, model in enumerate(models):
        center = left + (model_index + 0.5) * group_width
        for batch_index, batch in enumerate(batches):
            point = lookup[(model, batch)]
            x_center = center + (batch_index - (len(batches) - 1) / 2) * bar_step
            yy = y(point.mean)
            svg.append(
                f'<rect x="{x_center - bar_width / 2:.2f}" y="{yy:.2f}" '
                f'width="{bar_width:.2f}" height="{baseline_y - yy:.2f}" rx="3" '
                f'fill="{BATCH_COLORS.get(batch, "#555")}"/>'
            )
            y_min, y_max = y(point.minimum), y(point.maximum)
            svg.append(f'<line x1="{x_center:.2f}" y1="{y_max:.2f}" x2="{x_center:.2f}" y2="{y_min:.2f}" class="whisker"/>')
            svg.append(f'<line x1="{x_center - cap_width / 2:.2f}" y1="{y_max:.2f}" x2="{x_center + cap_width / 2:.2f}" y2="{y_max:.2f}" class="cap"/>')
            svg.append(f'<line x1="{x_center - cap_width / 2:.2f}" y1="{y_min:.2f}" x2="{x_center + cap_width / 2:.2f}" y2="{y_min:.2f}" class="cap"/>')
            svg.append(f'<text x="{x_center:.2f}" y="{y_max - 10:.2f}" text-anchor="middle" class="value">{point.mean:.3f}×</text>')
        svg.append(
            f'<text x="{center:.2f}" y="{bottom + 34:.2f}" text-anchor="middle" '
            f'class="model">{html.escape(display_model(model))}</text>'
        )

    svg.append(
        f'<text x="{left}" y="{height - 34}" class="note">'
        'Each operation is normalized before averaging; shared LM Head is counted once. Whiskers show the minimum and maximum layer/block speedup, not measurement uncertainty.</text>'
    )
    save_svg(svg_path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="real_model_average_speedup.png")
    parser.add_argument("--png-scale", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.png_scale <= 0:
        raise ValueError("--png-scale must be positive")
    result_files = [root.expanduser().resolve() / "result_all.csv" for root in args.results_root]
    missing = [path for path in result_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Results not found: {missing}")
    points = read_points(result_files, args.models)
    batches = sorted({point.n for point in points if point.model in args.models})
    averages = summarize(points, args.models, batches)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / args.output_name
    svg_path = png_path.with_suffix(".svg")
    plot(svg_path, averages, args.models, batches)
    svg_to_png(svg_path, png_path, args.png_scale)
    svg_path.unlink()
    print(f"Created: {png_path}")
    for point in averages:
        print(
            f"  {point.model} N={point.batch}: mean={point.mean:.4f}x, "
            f"std={point.stddev:.4f}, range={point.minimum:.4f}-{point.maximum:.4f}x"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
