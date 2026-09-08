#!/usr/bin/env python3
"""Plot selected Split-K, latency curves, and tuning speedup for each model."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from plot_results import save_svg, svg_document, svg_to_png


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments.json"
BATCH_COLORS = {8: "#2b83ba", 16: "#f28e2b", 32: "#4d9221"}
LAYER_LABELS = {
    "qkv_proj": "QKV",
    "o_proj": "O",
    "gateup_proj": "GateUp",
    "down_proj": "Down",
    "lm_head": "LM Head",
}


def latest_tuning_dir() -> Path:
    candidates = sorted(
        path
        for path in (PROJECT_ROOT / "results").glob("zipserv-*-tuning")
        if (path / "result_all.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError("No completed results/zipserv-*-tuning directory found")
    return candidates[-1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_latencies(path: Path, block_index: int = 0) -> dict[tuple[str, str, int, int], float]:
    grouped: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("block_index") or 0) != block_index:
                continue
            if row.get("phase") != "tune" or row.get("status") != "ok":
                continue
            value = row.get("zipgemm_latency_ms", "")
            if not value:
                continue
            key = (row["model"], row["layer"], int(row["N"]), int(row["split_k"]))
            grouped[key].append(float(value))
    return {key: statistics.median(values) for key, values in grouped.items()}


def nice_ceiling(maximum: float) -> tuple[float, float]:
    if maximum <= 0:
        return 1.0, 0.2
    rough = maximum / 4
    magnitude = 10 ** math.floor(math.log10(rough))
    residual = rough / magnitude
    step = (1 if residual <= 1 else 2 if residual <= 2 else 5 if residual <= 5 else 10) * magnitude
    return math.ceil(maximum / step) * step, step


def nice_bounds(minimum: float, maximum: float) -> tuple[float, float, float]:
    """Return padded, rounded bounds that emphasize the observed value range."""
    span = maximum - minimum
    if span <= 0:
        span = max(abs(maximum) * 0.1, 1e-6)
    padded_min = max(0.0, minimum - span * 0.12)
    padded_max = maximum + span * 0.12
    rough = (padded_max - padded_min) / 4
    magnitude = 10 ** math.floor(math.log10(rough))
    residual = rough / magnitude
    step = (1 if residual <= 1 else 2 if residual <= 2 else 5 if residual <= 5 else 10) * magnitude
    lower = math.floor(padded_min / step) * step
    upper = math.ceil(padded_max / step) * step
    if upper <= lower:
        upper = lower + step
    return lower, upper, step


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def add_legend(svg: list[str], width: int, batches: list[int], y: float) -> None:
    item_width = 135
    x = (width - item_width * len(batches)) / 2
    for batch in batches:
        color = BATCH_COLORS.get(batch, "#555")
        svg.append(f'<rect x="{x:.2f}" y="{y - 8:.2f}" width="34" height="16" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{x + 44:.2f}" y="{y + 6:.2f}" class="legend">N={batch}</text>')
        x += item_width


def plot_model(
    path: Path,
    model: str,
    layers: list[str],
    layer_shapes: dict[str, tuple[int, int]],
    batches: list[int],
    splits: list[int],
    latency: dict[tuple[str, str, int, int], float],
    title: str | None = None,
) -> None:
    width, height = 1800, 1320
    svg = svg_document(width, height)
    svg.append(
        """<style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#172033}
        .title{font-size:29px;font-weight:700}.subtitle{font-size:16px;fill:#526078}
        .panel-title{font-size:20px;font-weight:700}.axis{font-size:13px}
        .axis-label{font-size:16px;font-weight:700}.layer-label{font-size:15px;font-weight:700}
        .shape-label{font-size:12px;fill:#68758a}
        .legend{font-size:16px}.grid{stroke:#dce2ea;stroke-width:1}
        .axis-line{stroke:#283548;stroke-width:1.5}.best-label{font-size:12px;font-weight:700}
        </style>"""
    )
    svg.append(f'<text x="{width / 2}" y="38" text-anchor="middle" class="title">{html.escape(title or model)} — ZipGEMM Split-K tuning</text>')
    svg.append(f'<text x="{width / 2}" y="66" text-anchor="middle" class="subtitle">Selected Split-K · absolute latency across candidates · best tuning speedup over Split-K=1</text>')
    add_legend(svg, width, batches, 96)

    # Top panel: grouped bars for the selected Split-K of every layer and batch.
    left, right, top, bottom = 105.0, width - 45.0, 145.0, 405.0
    svg.append(f'<text x="{(left + right) / 2}" y="{top - 18}" text-anchor="middle" class="panel-title">Best-performing Split-K</text>')
    selected_ymax = max(splits) * 1.18
    selected_y = lambda value: bottom - value / selected_ymax * (bottom - top)
    selected_ticks = sorted(set([0, *splits]))
    for split in selected_ticks:
        yy = selected_y(split)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{left - 14}" y="{yy + 5:.2f}" text-anchor="end" class="axis">{split}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis-line"/>')
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>')
    svg.append(f'<text x="28" y="{(top + bottom) / 2}" text-anchor="middle" transform="rotate(-90 28 {(top + bottom) / 2})" class="axis-label">Selected Split-K</text>')
    group_width = (right - left) / len(layers)
    layer_x = {layer: left + (index + 0.5) * group_width for index, layer in enumerate(layers)}
    for index, layer in enumerate(layers):
        x = layer_x[layer]
        if index:
            boundary = left + index * (right - left) / len(layers)
            svg.append(f'<line x1="{boundary:.2f}" y1="{top}" x2="{boundary:.2f}" y2="{bottom}" stroke="#eef1f5"/>')
        svg.append(f'<text x="{x:.2f}" y="{bottom + 24}" text-anchor="middle" class="layer-label">{html.escape(LAYER_LABELS.get(layer, layer))}</text>')
        if layer in layer_shapes:
            m_size, k_size = layer_shapes[layer]
            svg.append(f'<text x="{x:.2f}" y="{bottom + 43}" text-anchor="middle" class="shape-label">M×K: {m_size:,}×{k_size:,}</text>')

    bar_gap = 7.0
    bar_width = min(54.0, (group_width * 0.72 - bar_gap * (len(batches) - 1)) / len(batches))
    for layer in layers:
        group_span = len(batches) * bar_width + (len(batches) - 1) * bar_gap
        group_left = layer_x[layer] - group_span / 2
        for batch_index, batch in enumerate(batches):
            color = BATCH_COLORS.get(batch, "#555")
            candidates = [
                (latency[(model, layer, batch, split)], split)
                for split in splits if (model, layer, batch, split) in latency
            ]
            if not candidates:
                continue
            _, best_split = min(candidates, key=lambda item: (item[0], item[1]))
            x = group_left + batch_index * (bar_width + bar_gap)
            y = selected_y(best_split)
            svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bottom - y:.2f}" rx="3" fill="{color}"/>')
            svg.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 7:.2f}" text-anchor="middle" class="best-label">{best_split}</text>')

    # Middle row: one latency-vs-Split-K panel per layer.
    panel_top, panel_bottom = 535.0, 810.0
    gap, outer_left, outer_right = 24.0, 75.0, width - 30.0
    panel_width = (outer_right - outer_left - gap * (len(layers) - 1)) / len(layers)
    for layer_index, layer in enumerate(layers):
        panel_left = outer_left + layer_index * (panel_width + gap)
        plot_left, plot_right = panel_left + 57, panel_left + panel_width - 18
        values = [
            latency[(model, layer, batch, split)]
            for batch in batches for split in splits
            if (model, layer, batch, split) in latency
        ]
        ymin, ymax, tick = nice_bounds(min(values), max(values)) if values else (0.0, 1.0, 0.2)
        y = lambda value: panel_bottom - (value - ymin) / (ymax - ymin) * (panel_bottom - panel_top)
        x = lambda split: plot_left + splits.index(split) * (plot_right - plot_left) / max(1, len(splits) - 1)
        svg.append(f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{panel_top - 22}" text-anchor="middle" class="panel-title">{html.escape(LAYER_LABELS.get(layer, layer))}</text>')
        value = ymin
        while value <= ymax + tick / 10:
            yy = y(value)
            svg.append(f'<line x1="{plot_left}" y1="{yy:.2f}" x2="{plot_right}" y2="{yy:.2f}" class="grid"/>')
            svg.append(f'<text x="{plot_left - 8}" y="{yy + 4:.2f}" text-anchor="end" class="axis">{value:.3g}</text>')
            value += tick
        svg.append(f'<line x1="{plot_left}" y1="{panel_top}" x2="{plot_left}" y2="{panel_bottom}" class="axis-line"/>')
        svg.append(f'<line x1="{plot_left}" y1="{panel_bottom}" x2="{plot_right}" y2="{panel_bottom}" class="axis-line"/>')
        for split in splits:
            xx = x(split)
            svg.append(f'<text x="{xx:.2f}" y="{panel_bottom + 22}" text-anchor="middle" class="axis">{split}</text>')
        for batch in batches:
            color = BATCH_COLORS.get(batch, "#555")
            points = [
                (x(split), y(latency[(model, layer, batch, split)]))
                for split in splits if (model, layer, batch, split) in latency
            ]
            if len(points) > 1:
                svg.append(f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
            for xx, yy in points:
                svg.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5" fill="{color}" stroke="white" stroke-width="1.5"/>')
        if layer_index == 0:
            svg.append(f'<text x="18" y="{(panel_top + panel_bottom) / 2}" text-anchor="middle" transform="rotate(-90 18 {(panel_top + panel_bottom) / 2})" class="axis-label">Latency (ms)</text>')
    svg.append(f'<text x="{width / 2}" y="860" text-anchor="middle" class="axis-label">Split-K</text>')

    # Bottom panel: best Split-K speedup relative to Split-K=1 for each layer/batch.
    speed_top, speed_bottom = 960.0, 1225.0
    speedups: dict[tuple[str, int], tuple[float, int]] = {}
    for layer in layers:
        for batch in batches:
            baseline = latency.get((model, layer, batch, 1))
            candidates = [
                (latency[(model, layer, batch, split)], split)
                for split in splits if (model, layer, batch, split) in latency
            ]
            if baseline is not None and candidates:
                best_latency, best_split = min(candidates, key=lambda item: (item[0], item[1]))
                speedups[(layer, batch)] = (baseline / best_latency, best_split)
    speed_values = [speedup for speedup, _ in speedups.values()]
    _, speed_ymax, speed_tick = nice_bounds(1.0, max(speed_values, default=1.05))
    speed_ymin = 1.0
    speed_y = lambda value: speed_bottom - (value - speed_ymin) / (speed_ymax - speed_ymin) * (speed_bottom - speed_top)
    svg.append(f'<text x="{(left + right) / 2}" y="{speed_top - 24}" text-anchor="middle" class="panel-title">Best Split-K speedup over Split-K=1</text>')
    value = speed_ymin
    while value <= speed_ymax + speed_tick / 10:
        yy = speed_y(value)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{left - 14}" y="{yy + 5:.2f}" text-anchor="end" class="axis">{value:.2g}×</text>')
        value += speed_tick
    svg.append(f'<line x1="{left}" y1="{speed_top}" x2="{left}" y2="{speed_bottom}" class="axis-line"/>')
    svg.append(f'<line x1="{left}" y1="{speed_bottom}" x2="{right}" y2="{speed_bottom}" class="axis-line"/>')
    svg.append(f'<text x="28" y="{(speed_top + speed_bottom) / 2}" text-anchor="middle" transform="rotate(-90 28 {(speed_top + speed_bottom) / 2})" class="axis-label">Speedup (×)</text>')
    for index, layer in enumerate(layers):
        center = layer_x[layer]
        if index:
            boundary = left + index * group_width
            svg.append(f'<line x1="{boundary:.2f}" y1="{speed_top}" x2="{boundary:.2f}" y2="{speed_bottom}" stroke="#eef1f5"/>')
        svg.append(f'<text x="{center:.2f}" y="{speed_bottom + 24}" text-anchor="middle" class="layer-label">{html.escape(LAYER_LABELS.get(layer, layer))}</text>')
        if layer in layer_shapes:
            m_size, k_size = layer_shapes[layer]
            svg.append(f'<text x="{center:.2f}" y="{speed_bottom + 43}" text-anchor="middle" class="shape-label">M×K: {m_size:,}×{k_size:,}</text>')
    for layer in layers:
        group_span = len(batches) * bar_width + (len(batches) - 1) * bar_gap
        group_left = layer_x[layer] - group_span / 2
        for batch_index, batch in enumerate(batches):
            result = speedups.get((layer, batch))
            if result is None:
                continue
            speedup, best_split = result
            x = group_left + batch_index * (bar_width + bar_gap)
            y = speed_y(speedup)
            bar_height = max(4.0, speed_bottom - y)
            bar_y = speed_bottom - bar_height
            color = BATCH_COLORS.get(batch, "#555")
            svg.append(f'<rect x="{x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="3" fill="{color}"/>')
            label_x = x + bar_width / 2
            svg.append(f'<text x="{label_x:.2f}" y="{bar_y - 22:.2f}" text-anchor="middle" class="best-label">{speedup:.2f}×</text>')
            svg.append(f'<text x="{label_x:.2f}" y="{bar_y - 7:.2f}" text-anchor="middle" class="best-label">K={best_split}</text>')
    save_svg(path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, help="Tuning directory containing result_all.csv; default: latest")
    parser.add_argument("--config", type=Path, help="Default: experiment snapshot, or current config if absent")
    parser.add_argument("--block-index", type=int, default=0, help="Never pool blocks; default: 0")
    parser.add_argument("--models", nargs="+", help="Default: every enabled model present in the tuning result")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "latency_by_splitk")
    parser.add_argument(
        "--png-scale", type=float, default=2.0,
        help="PNG raster scale relative to the SVG canvas (default: 2.0, 3600 px wide)",
    )
    parser.add_argument(
        "--keep-svg", action="store_true",
        help="Keep the intermediate vector SVG (default: PNG only)",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.png_scale <= 0:
        raise ValueError("--png-scale must be positive")
    results_root = (args.results_root or latest_tuning_dir()).expanduser().resolve()
    result_file = results_root / "result_all.csv"
    if not result_file.is_file():
        raise FileNotFoundError(f"Tuning result not found: {result_file}")
    if args.block_index < 0:
        raise ValueError("--block-index must be non-negative")
    config_path = args.config or results_root / "experiments.json"
    if not config_path.is_file() and args.config is None:
        config_path = DEFAULT_CONFIG
    config = load_config(config_path.expanduser().resolve())
    enabled = {model["id"]: model for model in config["models"] if model.get("enabled", True)}
    available = {key[0] for key in load_latencies(result_file, args.block_index)}
    models = args.models or [model for model in enabled if model in available]
    unknown = [model for model in models if model not in enabled]
    if unknown:
        raise ValueError(f"Unknown or disabled models: {unknown}")
    latency = load_latencies(result_file, args.block_index)
    batches = config["matrix"]["batches"]
    splits = config["matrix"]["split_k_candidates"]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for model in models:
        layers = [layer["id"] for layer in enabled[model]["layers"]
                  if layer.get("block_scoped", True) or args.block_index == 0]
        layer_shapes = {
            layer["id"]: (int(layer["M"]), int(layer["K"]))
            for layer in enabled[model]["layers"]
        }
        expected = {(model, layer, batch, split) for layer in layers for batch in batches for split in splits}
        missing = sorted(expected - latency.keys())
        if missing and not args.allow_incomplete:
            print(f"SKIP {model}: missing {len(missing)}/{len(expected)} tuning points", file=sys.stderr)
            continue
        suffix = "" if args.block_index == 0 else f"_block-{args.block_index}"
        svg_path = output_dir / f"{model}{suffix}_latency_by_splitk.svg"
        png_path = output_dir / f"{model}{suffix}_latency_by_splitk.png"
        plot_model(svg_path, model, layers, layer_shapes, batches, splits, latency,
                   title=f"{model} · Block {args.block_index}")
        svg_to_png(svg_path, png_path, args.png_scale)
        if not args.keep_svg:
            svg_path.unlink()
        created.append(png_path)
        print(f"Created: {png_path}")
    if not created:
        raise RuntimeError("No model plots were created")
    print(f"Created {len(created)} model plot(s) from {result_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
