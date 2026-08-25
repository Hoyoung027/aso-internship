#!/usr/bin/env python3
"""Plot block- and layer-wise ZipServ speedup for real-weight final runs."""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from plot_results import save_svg, svg_document, svg_to_png


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "image" / "real"
LAYER_ORDER = ("qkv_proj", "o_proj", "gateup_proj", "down_proj")
LAYER_LABELS = {
    "qkv_proj": "QKV",
    "o_proj": "O",
    "gateup_proj": "GateUp",
    "down_proj": "Down",
    "lm_head": "LM Head",
}
BATCH_COLORS = {8: "#2b83ba", 16: "#f28e2b", 32: "#4d9221"}
MODEL_LABELS = {
    "llama3.1-8b": "LLaMA 3.1 8B",
    "llama3.1-70b": "LLaMA 3.1 70B",
    "qwen2.5-7b": "Qwen 2.5 7B",
    "qwen2.5-14b": "Qwen 2.5 14B",
}


@dataclass(frozen=True)
class Point:
    model: str
    block: int
    layer: str
    m: int
    k: int
    n: int
    split_k: int
    speedup: float
    trials: int


def read_points(result_files: list[Path], models: list[str] | None) -> list[Point]:
    # Later files win for an identical trial. This lets callers combine the
    # block-0 and mid/last run directories without duplicating a copied LM Head.
    unique_rows: dict[tuple[str, int, str, int, int, int, int], dict[str, str]] = {}
    for result_file in sorted(result_files, key=lambda path: path.stat().st_mtime):
        with result_file.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("phase") != "run" or row.get("status") != "ok":
                    continue
                if models and row.get("model") not in models:
                    continue
                trial_key = (
                    row["model"], int(row.get("block_index") or 0), row["layer"],
                    int(row["M"]), int(row["K"]), int(row["N"]), int(row["trial"]),
                )
                unique_rows[trial_key] = row

    grouped: dict[tuple[str, int, str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in unique_rows.values():
        key = (
            row["model"], int(row.get("block_index") or 0), row["layer"],
            int(row["M"]), int(row["K"]), int(row["N"]),
        )
        grouped[key].append(row)

    points: list[Point] = []
    for (model, block, layer, m, k, n), rows in grouped.items():
        split_counts = Counter(int(row["split_k"]) for row in rows)
        split_k = min(split_counts, key=lambda value: (-split_counts[value], value))
        if len(split_counts) != 1:
            raise ValueError(f"Final trials use inconsistent Split-K: {(model, block, layer, n)}")
        tc = statistics.median(float(row["cublas_tc_latency_ms"]) for row in rows)
        zipserv = statistics.median(float(row["zipgemm_latency_ms"]) for row in rows)
        points.append(Point(model, block, layer, m, k, n, split_k, tc / zipserv, len(rows)))
    if not points:
        raise ValueError(f"No successful final-run rows found in {result_files}")
    return points


def display_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def model_categories(points: list[Point]) -> list[tuple[int | None, str]]:
    blocks = sorted({point.block for point in points if point.layer in LAYER_ORDER})
    categories = [(block, layer) for block in blocks for layer in LAYER_ORDER]
    if any(point.layer == "lm_head" for point in points):
        categories.append((None, "lm_head"))
    return categories


def validate(points: list[Point], models: list[str], batches: list[int]) -> None:
    lookup = {(p.model, p.block, p.layer, p.n): p for p in points}
    missing: list[tuple[str, int | None, str, int]] = []
    for model in models:
        model_points = [point for point in points if point.model == model]
        for block, layer in model_categories(model_points):
            actual_block = 0 if block is None else block
            for batch in batches:
                if (model, actual_block, layer, batch) not in lookup:
                    missing.append((model, block, layer, batch))
    if missing:
        raise ValueError(f"Missing layer-wise final cases ({len(missing)}): {missing}")
    short = [p for p in points if p.model in models and p.trials != 3]
    if short:
        raise ValueError(f"Expected 3 final trials per point; mismatches: {short}")


def nice_upper(values: list[float]) -> float:
    return max(1.2, math.ceil((max(values) + 0.03) / 0.1) * 0.1)


def figure_family(models: list[str]) -> str:
    if all(model.startswith("llama3.1-") for model in models):
        return "LLaMA 3.1"
    if all(model.startswith("qwen2.5-") for model in models):
        return "Qwen 2.5"
    return "Real-weight models"


def plot(
    svg_path: Path,
    points: list[Point],
    models: list[str],
    batches: list[int],
) -> None:
    width = 2400
    left, right = 125.0, width - 45.0
    panel_height, panel_gap, first_top = 430.0, 105.0, 170.0
    height = int(first_top + len(models) * panel_height + (len(models) - 1) * panel_gap + 115)
    lower = 1.0
    upper = nice_upper([point.speedup for point in points if point.model in models])
    lookup = {(p.model, p.block, p.layer, p.n): p for p in points}

    svg = svg_document(width, height)
    svg.append(
        """<style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#172033}
        .title{font-size:34px;font-weight:700}.subtitle{font-size:18px;fill:#59677c}
        .panel-title{font-size:23px;font-weight:700}.block-title{font-size:17px;font-weight:700}
        .axis{font-size:14px}.axis-label{font-size:18px;font-weight:700}
        .layer{font-size:15px;font-weight:700}.shape{font-size:11px;fill:#68758a}
        .value{font-size:12px;font-weight:700}.split{font-size:10px;fill:#526078}
        .legend{font-size:16px}.note{font-size:13px;fill:#64748b}
        .grid{stroke:#dce3ec;stroke-width:1}.axis-line{stroke:#283548;stroke-width:1.5}
        .baseline{stroke:#364152;stroke-width:2;stroke-dasharray:8 5}
        </style>"""
    )
    svg.append(
        f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">'
        f'RTX 4090 — {html.escape(figure_family(models))} real-weight layer-wise speedup</text>'
    )
    svg.append(
        f'<text x="{width / 2}" y="73" text-anchor="middle" class="subtitle">'
        'Speedup = median(cuBLAS TC latency) / median(ZipServ latency), using 3 final trials</text>'
    )
    legend_width = 150.0
    legend_x = (width - legend_width * len(batches)) / 2
    for batch in batches:
        color = BATCH_COLORS.get(batch, "#555")
        svg.append(f'<rect x="{legend_x:.2f}" y="101" width="30" height="17" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 40:.2f}" y="116" class="legend">N={batch}</text>')
        legend_x += legend_width

    def y(value: float, top: float, bottom: float) -> float:
        return bottom - (value - lower) / (upper - lower) * (bottom - top)

    for model_index, model in enumerate(models):
        top = first_top + model_index * (panel_height + panel_gap)
        bottom = top + panel_height
        model_points = [point for point in points if point.model == model]
        categories = model_categories(model_points)
        category_width = (right - left) / len(categories)

        # Alternating block backgrounds and separators.
        previous_block: int | None | object = object()
        for index, (block, _) in enumerate(categories):
            fill = "#f5f7fa" if (0 if block is None else block) % 2 == 0 else "#ffffff"
            x = left + index * category_width
            svg.append(f'<rect x="{x:.2f}" y="{top}" width="{category_width:.2f}" height="{panel_height}" fill="{fill}"/>')
            if index and block != previous_block:
                svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#aeb8c6" stroke-width="2"/>')
            previous_block = block

        tick = lower
        while tick <= upper + 1e-9:
            yy = y(tick, top + 42, bottom)
            svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
            svg.append(f'<text x="{left - 14}" y="{yy + 5:.2f}" text-anchor="end" class="axis">{tick:.1f}×</text>')
            tick += 0.1
        baseline_y = y(1.0, top + 42, bottom)
        svg.append(f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{right}" y2="{baseline_y:.2f}" class="baseline"/>')
        svg.append(f'<line x1="{left}" y1="{top + 42}" x2="{left}" y2="{bottom}" class="axis-line"/>')
        svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>')
        svg.append(f'<text x="{(left + right) / 2}" y="{top + 28:.2f}" text-anchor="middle" class="panel-title">{html.escape(display_model(model))}</text>')
        center_y = (top + 42 + bottom) / 2
        svg.append(f'<text x="30" y="{center_y:.2f}" text-anchor="middle" transform="rotate(-90 30 {center_y:.2f})" class="axis-label">cuBLAS TC / ZipServ</text>')

        # Block headers.
        spans: dict[int | None, list[int]] = defaultdict(list)
        for index, (block, _) in enumerate(categories):
            spans[block].append(index)
        for block, indices in spans.items():
            center = left + (statistics.mean(indices) + 0.5) * category_width
            label = "Shared output" if block is None else f"Block {block}"
            svg.append(f'<text x="{center:.2f}" y="{top + 54:.2f}" text-anchor="middle" class="block-title">{label}</text>')

        bar_step = min(42.0, category_width / (len(batches) + 0.7))
        bar_width = bar_step * 0.76
        for category_index, (block, layer) in enumerate(categories):
            actual_block = 0 if block is None else block
            center = left + (category_index + 0.5) * category_width
            sample = lookup[(model, actual_block, layer, batches[0])]
            for batch_index, batch in enumerate(batches):
                point = lookup[(model, actual_block, layer, batch)]
                x = center + (batch_index - (len(batches) - 1) / 2) * bar_step - bar_width / 2
                yy = y(point.speedup, top + 42, bottom)
                svg.append(f'<rect x="{x:.2f}" y="{yy:.2f}" width="{bar_width:.2f}" height="{baseline_y - yy:.2f}" rx="2" fill="{BATCH_COLORS.get(batch, "#555")}"/>')
                svg.append(f'<text x="{x + bar_width / 2:.2f}" y="{yy - 16:.2f}" text-anchor="middle" class="value">{point.speedup:.2f}×</text>')
                svg.append(f'<text x="{x + bar_width / 2:.2f}" y="{yy - 4:.2f}" text-anchor="middle" class="split">K={point.split_k}</text>')
            svg.append(f'<text x="{center:.2f}" y="{bottom + 24:.2f}" text-anchor="middle" class="layer">{html.escape(LAYER_LABELS.get(layer, layer))}</text>')
            svg.append(f'<text x="{center:.2f}" y="{bottom + 42:.2f}" text-anchor="middle" class="shape">{sample.m:,}×{sample.k:,}</text>')

    svg.append(
        f'<text x="{left}" y="{height - 22}" class="note">'
        'Bars use block-specific selected Split-K values; LM Head is model-wide and shown once. M×K is printed below each layer.</text>'
    )
    save_svg(svg_path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, nargs="+", required=True,
        help="One or more final-run directories containing result_all.csv",
    )
    parser.add_argument("--models", nargs="+", help="Default: every model present in the result")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="real_layerwise_speedup.png")
    parser.add_argument("--png-scale", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.png_scale <= 0:
        raise ValueError("--png-scale must be positive")
    result_files = [root.expanduser().resolve() / "result_all.csv" for root in args.results_root]
    missing_files = [path for path in result_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Results not found: {missing_files}")
    points = read_points(result_files, args.models)
    available = sorted({point.model for point in points})
    models = args.models or available
    unknown = sorted(set(models) - set(available))
    if unknown:
        raise ValueError(f"Models not found in result: {unknown}")
    batches = sorted({point.n for point in points if point.model in models})
    validate(points, models, batches)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / args.output_name
    svg_path = png_path.with_suffix(".svg")
    plot(svg_path, points, models, batches)
    svg_to_png(svg_path, png_path, args.png_scale)
    svg_path.unlink()
    print(f"Created: {png_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
