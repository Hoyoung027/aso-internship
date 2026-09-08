#!/usr/bin/env python3
"""Plot every Split-K candidate's speedup over Split-K=1 for synthetic weights."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "image" / "synthetic" / "splitk_speedup"
SPLIT_COLORS = {
    1: "#9ca3af",
    2: "#59a14f",
    4: "#f28e2b",
    8: "#4e79a7",
}
LAYER_LABELS = {
    "qkv_proj": "QKV",
    "o_proj": "O",
    "gateup_proj": "GateUp",
    "down_proj": "Down",
    "lm_head": "LM Head",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_synthetic_tuning_dir() -> Path:
    candidates: list[Path] = []
    for path in (PROJECT_ROOT / "results").glob("zipserv-*-tuning"):
        result_file = path / "result_all.csv"
        if not result_file.is_file():
            continue
        rows = read_csv(result_file)
        if any("synthetic" in row.get("weight_source", "").lower() for row in rows):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No completed synthetic tuning result was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_latencies(
    result_file: Path,
) -> tuple[dict[tuple[str, str, int, int], float], dict[tuple[str, str], tuple[int, int]]]:
    grouped: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    shapes: dict[tuple[str, str], tuple[int, int]] = {}
    rows = read_csv(result_file)
    sources = {
        row.get("weight_source", "")
        for row in rows
        if row.get("phase") == "tune" and row.get("status") == "ok"
    }
    if not sources or any("synthetic" not in source.lower() for source in sources):
        raise ValueError(
            "plot_splitk_speedup.py accepts synthetic-weight tuning results only; "
            f"found weight_source={sorted(sources)}"
        )

    for row in rows:
        if row.get("phase") != "tune" or row.get("status") != "ok":
            continue
        latency = row.get("zipgemm_latency_ms", "")
        if not latency:
            continue
        model = row["model"]
        layer = row["layer"]
        key = (model, layer, int(row["N"]), int(row["split_k"]))
        grouped[key].append(float(latency))
        shapes[(model, layer)] = (int(row["M"]), int(row["K"]))

    return (
        {key: statistics.median(values) for key, values in grouped.items()},
        shapes,
    )


def calculate_speedups(
    latencies: dict[tuple[str, str, int, int], float],
) -> dict[tuple[str, str, int, int], float]:
    speedups: dict[tuple[str, str, int, int], float] = {}
    for (model, layer, batch, split_k), latency in latencies.items():
        baseline = latencies.get((model, layer, batch, 1))
        if baseline is None or baseline <= 0 or latency <= 0:
            continue
        speedups[(model, layer, batch, split_k)] = baseline / latency
    return speedups


def ordered_models(config: dict[str, Any], available: set[str]) -> list[str]:
    configured = [
        model["id"]
        for model in config["models"]
        if model.get("enabled", True) and model["id"] in available
    ]
    return configured + sorted(available - set(configured))


def model_layers(config: dict[str, Any], model: str, available: set[str]) -> list[str]:
    spec = next((item for item in config["models"] if item["id"] == model), None)
    if spec is None:
        return sorted(available)
    configured = [item["id"] for item in spec["layers"] if item["id"] in available]
    return configured + sorted(available - set(configured))


def add_legend(svg: list[str], width: int, splits: list[int]) -> None:
    item_width = 142
    total_width = item_width * len(splits) + 220
    x = (width - total_width) / 2
    y = 102
    for split_k in splits:
        color = SPLIT_COLORS.get(split_k, "#6b7280")
        svg.append(
            f'<rect x="{x:.2f}" y="{y - 12}" width="32" height="18" rx="2" '
            f'fill="{color}" stroke="#263342" stroke-width=".7"/>'
        )
        svg.append(
            f'<text x="{x + 42:.2f}" y="{y + 3}" class="legend">K={split_k}</text>'
        )
        x += item_width
    svg.append(
        f'<line x1="{x + 6:.2f}" y1="{y - 3}" x2="{x + 52:.2f}" y2="{y - 3}" '
        'class="baseline"/>'
    )
    svg.append(
        f'<text x="{x + 62:.2f}" y="{y + 3}" class="legend">K=1 baseline</text>'
    )


def y_axis(maximum: float) -> tuple[float, float]:
    step = 0.25 if maximum <= 2.5 else 0.5
    upper = max(1.25, math.ceil((maximum + 0.08) / step) * step)
    return upper, step


def plot_model(
    output_path: Path,
    model: str,
    layers: list[str],
    batches: list[int],
    splits: list[int],
    speedups: dict[tuple[str, str, int, int], float],
    shapes: dict[tuple[str, str], tuple[int, int]],
    *,
    title: str | None = None,
    measurement_note: str = "",
) -> None:
    width = 1800
    panel_height = 345
    panel_gap = 38
    first_top = 175
    height = first_top + len(batches) * panel_height + (len(batches) - 1) * panel_gap + 70
    svg = svg_document(width, height)
    svg.append(
        """<style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#172033}
        .title{font-size:29px;font-weight:700}.subtitle{font-size:16px;fill:#526078}
        .panel-title{font-size:20px;font-weight:700}.axis{font-size:13px}
        .axis-label{font-size:16px;font-weight:700}.layer-label{font-size:15px;font-weight:700}
        .shape-label{font-size:12px;fill:#68758a}.legend{font-size:16px}
        .grid{stroke:#dce2ea;stroke-width:1}.axis-line{stroke:#283548;stroke-width:1.5}
        .baseline{stroke:#172033;stroke-width:2;stroke-dasharray:8 5}
        .bar-value{font-size:12px;font-weight:700}.best-label{font-size:10px;font-weight:700;fill:#9a3412}
        .note{font-size:13px;fill:#596579}
        </style>"""
    )
    title = title or f"{model} — Synthetic-weight Split-K speedup"
    svg.append(
        f'<text x="{width / 2}" y="40" text-anchor="middle" '
        f'class="title">{html.escape(title)}</text>'
    )
    svg.append(
        f'<text x="{width / 2}" y="69" text-anchor="middle" class="subtitle">'
        'Speedup = latency(K=1) / latency(candidate K) · higher is better'
        f'{html.escape(measurement_note)}</text>'
    )
    add_legend(svg, width, splits)

    model_values = [
        value
        for (item_model, layer, batch, split_k), value in speedups.items()
        if item_model == model and layer in layers and batch in batches and split_k in splits
    ]
    ymax, tick = y_axis(max(model_values, default=1.0))
    left, right = 105.0, width - 40.0
    plot_width = right - left
    group_width = plot_width / len(layers)
    bar_gap = 8.0
    bar_width = min(54.0, (group_width * 0.78 - bar_gap * (len(splits) - 1)) / len(splits))

    for batch_index, batch in enumerate(batches):
        top = first_top + batch_index * (panel_height + panel_gap)
        bottom = top + 265
        value_y = lambda value: bottom - value / ymax * (bottom - top)
        svg.append(
            f'<text x="{(left + right) / 2}" y="{top - 20}" text-anchor="middle" '
            f'class="panel-title">Batch / token count N={batch}</text>'
        )

        value = 0.0
        while value <= ymax + tick / 10:
            yy = value_y(value)
            svg.append(
                f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>'
            )
            svg.append(
                f'<text x="{left - 14}" y="{yy + 5:.2f}" text-anchor="end" '
                f'class="axis">{value:.2f}×</text>'
            )
            value += tick

        baseline_y = value_y(1.0)
        svg.append(
            f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{right}" '
            f'y2="{baseline_y:.2f}" class="baseline"/>'
        )
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis-line"/>')
        svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>')
        svg.append(
            f'<text x="28" y="{(top + bottom) / 2}" text-anchor="middle" '
            f'transform="rotate(-90 28 {(top + bottom) / 2})" class="axis-label">'
            'Speedup over K=1</text>'
        )

        for layer_index, layer in enumerate(layers):
            center = left + (layer_index + 0.5) * group_width
            if layer_index:
                boundary = left + layer_index * group_width
                svg.append(
                    f'<line x1="{boundary:.2f}" y1="{top}" x2="{boundary:.2f}" '
                    f'y2="{bottom}" stroke="#eef1f5"/>'
                )
            values = {
                split_k: speedups.get((model, layer, batch, split_k))
                for split_k in splits
            }
            present = {split_k: value for split_k, value in values.items() if value is not None}
            best_split = max(present, key=lambda split_k: (present[split_k], -split_k)) if present else None
            group_span = len(splits) * bar_width + (len(splits) - 1) * bar_gap
            group_left = center - group_span / 2

            for split_index, split_k in enumerate(splits):
                speedup = values[split_k]
                if speedup is None:
                    continue
                x = group_left + split_index * (bar_width + bar_gap)
                y = value_y(speedup)
                is_best = split_k == best_split
                stroke_width = 2.5 if is_best else 0.8
                stroke = "#9a3412" if is_best else "#263342"
                svg.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                    f'height="{bottom - y:.2f}" rx="3" fill="{SPLIT_COLORS.get(split_k, "#6b7280")}" '
                    f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
                )
                svg.append(
                    f'<text x="{x + bar_width / 2:.2f}" y="{y - 7:.2f}" '
                    f'text-anchor="middle" class="bar-value">{speedup:.2f}×</text>'
                )
                if is_best:
                    svg.append(
                        f'<text x="{x + bar_width / 2:.2f}" y="{y - 22:.2f}" '
                        f'text-anchor="middle" class="best-label">best</text>'
                    )

            svg.append(
                f'<text x="{center:.2f}" y="{bottom + 24}" text-anchor="middle" '
                f'class="layer-label">{html.escape(LAYER_LABELS.get(layer, layer))}</text>'
            )
            shape = shapes.get((model, layer))
            if shape:
                svg.append(
                    f'<text x="{center:.2f}" y="{bottom + 43}" text-anchor="middle" '
                    f'class="shape-label">M×K: {shape[0]:,}×{shape[1]:,}</text>'
                )

    svg.append(
        f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" class="note">'
        'Bars below 1.00× are slower than unsplit ZipGEMM (K=1). '
        'The best bar includes the Split-K reduction cost.</text>'
    )
    save_svg(output_path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Synthetic tuning directory containing result_all.csv; default: latest synthetic tuning run",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", help="Default: every model present in the result")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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

    results_root = (args.results_root or latest_synthetic_tuning_dir()).expanduser().resolve()
    result_file = results_root / "result_all.csv"
    config = load_config(args.config.expanduser().resolve())
    latencies, shapes = load_latencies(result_file)
    speedups = calculate_speedups(latencies)
    available_models = {key[0] for key in latencies}
    models = args.models or ordered_models(config, available_models)
    unknown = sorted(set(models) - available_models)
    if unknown:
        raise ValueError(f"Models absent from tuning result: {unknown}")

    batches = [int(value) for value in config["matrix"]["batches"]]
    splits = [int(value) for value in config["matrix"]["split_k_candidates"]]
    if 1 not in splits:
        raise ValueError("Split-K candidates must contain K=1 as the speedup baseline")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for model in models:
        available_layers = {key[1] for key in latencies if key[0] == model}
        layers = model_layers(config, model, available_layers)
        expected = {
            (model, layer, batch, split_k)
            for layer in layers
            for batch in batches
            for split_k in splits
        }
        missing = sorted(expected - latencies.keys())
        if missing and not args.allow_incomplete:
            print(
                f"SKIP {model}: missing {len(missing)}/{len(expected)} tuning points",
                file=sys.stderr,
            )
            continue

        svg_path = output_dir / f"{model}_splitk_speedup.svg"
        png_path = output_dir / f"{model}_splitk_speedup.png"
        plot_model(svg_path, model, layers, batches, splits, speedups, shapes)
        svg_to_png(svg_path, png_path, args.png_scale)
        if not args.keep_svg:
            svg_path.unlink()
        created.append(png_path)
        print(f"Created: {png_path}")

    if not created:
        raise RuntimeError("No Split-K speedup plots were created")
    print(f"Created {len(created)} model plot(s) from {result_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
