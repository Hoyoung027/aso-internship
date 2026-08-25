#!/usr/bin/env python3
"""Plot selected-case numerical errors for LLaMA 3.1 8B and 70B."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from plot_results import save_svg, svg_document, svg_to_png


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments.json"
DEFAULT_MODELS = ("llama3.1-8b", "llama3.1-70b")
LAYER_LABELS = {
    "qkv_proj": "QKV",
    "o_proj": "O",
    "gateup_proj": "GateUp",
    "down_proj": "Down",
    "lm_head": "LM Head",
}
METHODS = ("tc", "zip")
METHOD_LABELS = {"tc": "cuBLAS TC", "zip": "ZipServ"}
METHOD_COLORS = {"tc": "#f28e2b", "zip": "#2b83ba"}


@dataclass(frozen=True)
class ErrorPoint:
    model: str
    layer: str
    m: int
    k: int
    n: int
    split_k: int
    tc_relative_percent: float
    zip_relative_percent: float
    tc_mae: float
    zip_mae: float
    tc_significant_percent: float
    zip_significant_percent: float


@dataclass(frozen=True)
class Metric:
    title: str
    axis_label: str
    scale: str
    tc_value: Callable[[ErrorPoint], float]
    zip_value: Callable[[ErrorPoint], float]


METRICS = (
    Metric(
        title="Mean element-wise relative error",
        axis_label="Mean relative error (%)",
        scale="log-percent",
        tc_value=lambda point: point.tc_relative_percent,
        zip_value=lambda point: point.zip_relative_percent,
    ),
    Metric(
        title="Mean absolute error per output element",
        axis_label="Mean absolute error / element",
        scale="log-scientific",
        tc_value=lambda point: point.tc_mae,
        zip_value=lambda point: point.zip_mae,
    ),
    Metric(
        title="Elements with relative error > 1e-4 (0.01%)",
        axis_label="Significant elements (%)",
        scale="linear-percent",
        tc_value=lambda point: point.tc_significant_percent,
        zip_value=lambda point: point.zip_significant_percent,
    ),
)


def latest_tuning_dir() -> Path:
    candidates = sorted(
        path
        for path in (PROJECT_ROOT / "results").glob("zipserv-*-tuning")
        if (path / "result_all.csv").is_file()
        and (path / "selected_splitk.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError("No completed results/zipserv-*-tuning directory found")
    return candidates[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_selected_points(results_root: Path, models: list[str]) -> list[ErrorPoint]:
    selected = {
        (row["model"], row["layer"], int(row["N"])): int(row["split_k"])
        for row in read_csv(results_root / "selected_splitk.csv")
        if row["model"] in models
    }
    grouped: dict[tuple[str, str, int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(results_root / "result_all.csv"):
        if row.get("status") != "ok" or row.get("model") not in models:
            continue
        key = (row["model"], row["layer"], int(row["N"]))
        if int(row["split_k"]) != selected.get(key):
            continue
        group_key = (
            row["model"], row["layer"], int(row["M"]), int(row["K"]),
            int(row["N"]), int(row["split_k"]),
        )
        grouped[group_key].append(row)

    def median(rows: list[dict[str, str]], field: str) -> float:
        values = [float(row[field]) for row in rows if row.get(field, "") != ""]
        if not values:
            raise ValueError(f"Missing error metric {field}")
        return statistics.median(values)

    points: list[ErrorPoint] = []
    for (model, layer, m, k, n, split_k), rows in grouped.items():
        output_elements = m * n
        points.append(
            ErrorPoint(
                model=model,
                layer=layer,
                m=m,
                k=k,
                n=n,
                split_k=split_k,
                tc_relative_percent=median(rows, "tc_vs_non_tc_average_relative_error") * 100.0,
                zip_relative_percent=median(rows, "zip_vs_non_tc_average_relative_error") * 100.0,
                tc_mae=median(rows, "tc_vs_non_tc_total_absolute_error") / output_elements,
                zip_mae=median(rows, "zip_vs_non_tc_total_absolute_error") / output_elements,
                tc_significant_percent=median(rows, "tc_vs_non_tc_significant_error_percent"),
                zip_significant_percent=median(rows, "zip_vs_non_tc_significant_error_percent"),
            )
        )
    return points


def log_bounds(values: list[float]) -> tuple[float, float, list[float]]:
    positive = [value for value in values if value > 0]
    if not positive:
        raise ValueError("Log-scale metric has no positive values")
    minimum = 10 ** math.floor(math.log10(min(positive)))
    maximum = 10 ** math.ceil(math.log10(max(positive)))
    if maximum <= minimum:
        maximum = minimum * 10
    ticks: list[float] = []
    value = minimum
    while value <= maximum * (1 + 1e-12):
        ticks.append(value)
        value *= 10
    return minimum, maximum, ticks


def linear_bounds(values: list[float]) -> tuple[float, float, list[float]]:
    maximum = max(values, default=1.0)
    rough = max(maximum / 5, 1e-9)
    magnitude = 10 ** math.floor(math.log10(rough))
    residual = rough / magnitude
    step = (1 if residual <= 1 else 2 if residual <= 2 else 5 if residual <= 5 else 10) * magnitude
    upper = math.ceil(maximum / step) * step
    ticks = [index * step for index in range(int(round(upper / step)) + 1)]
    return 0.0, upper, ticks


def metric_scales(points: list[ErrorPoint]) -> list[tuple[float, float, list[float]]]:
    scales = []
    for metric in METRICS:
        values = [
            value
            for point in points
            for value in (metric.tc_value(point), metric.zip_value(point))
        ]
        scales.append(log_bounds(values) if metric.scale.startswith("log") else linear_bounds(values))
    return scales


def model_display_name(model: str) -> str:
    size = model.rsplit("-", 1)[-1].upper()
    return f"LLaMA 3.1 {size}"


def add_style(svg: list[str]) -> None:
    svg.append(
        """<style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#172033}
        .title{font-size:29px;font-weight:700}.subtitle{font-size:15px;fill:#526078}
        .row-title{font-size:19px;font-weight:700}.batch-title{font-size:18px;font-weight:700}
        .axis{font-size:11px}.axis-label{font-size:14px;font-weight:700}
        .layer{font-size:12px;font-weight:700}.shape{font-size:9px;fill:#68758a}
        .legend{font-size:15px}.k-label{font-size:10px;font-weight:700}
        .note{font-size:13px;fill:#596579}.grid{stroke:#dce2ea;stroke-width:1}
        .axis-line{stroke:#283548;stroke-width:1.4}.bar{stroke:#263342;stroke-width:.7}
        </style>"""
    )


def format_tick(value: float, scale: str) -> str:
    if scale == "log-percent":
        return f"{value:g}%"
    if scale == "log-scientific":
        return f"{value:.0e}"
    return f"{value:g}%"


def draw_panel(
    svg: list[str], panel_points: list[ErrorPoint], layers: list[str], metric: Metric,
    bounds: tuple[float, float, list[float]], left: float, right: float,
    top: float, bottom: float,
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

    lookup = {point.layer: point for point in panel_points}
    bar_width = min(27.0, group_width * 0.27)
    bar_gap = 7.0
    for index, layer in enumerate(layers):
        group_left = left + index * group_width
        center = group_left + group_width / 2
        point = lookup.get(layer)
        if point:
            values = {"tc": metric.tc_value(point), "zip": metric.zip_value(point)}
            pair_width = 2 * bar_width + bar_gap
            for method_index, method in enumerate(METHODS):
                x = center - pair_width / 2 + method_index * (bar_width + bar_gap)
                yy = y(values[method])
                svg.append(f'<rect x="{x:.2f}" y="{yy:.2f}" width="{bar_width:.2f}" height="{bottom - yy:.2f}" rx="2" fill="{METHOD_COLORS[method]}" class="bar"/>')
                if method == "zip":
                    svg.append(f'<text x="{x + bar_width / 2:.2f}" y="{max(top + 11, yy - 5):.2f}" text-anchor="middle" class="k-label">K={point.split_k}</text>')
            svg.append(f'<text x="{center:.2f}" y="{bottom + 18}" text-anchor="middle" class="layer">{html.escape(LAYER_LABELS.get(layer, layer))}</text>')
            svg.append(f'<text x="{center:.2f}" y="{bottom + 33}" text-anchor="middle" class="shape">{point.m:,}×{point.k:,}</text>')


def plot_model(
    path: Path, model: str, points: list[ErrorPoint], layers: list[str],
    batches: list[int], scales: list[tuple[float, float, list[float]]],
) -> None:
    width, height = 1800, 1490
    svg = svg_document(width, height)
    add_style(svg)
    svg.append(f'<text x="{width / 2}" y="38" text-anchor="middle" class="title">{model_display_name(model)} — numerical error comparison</text>')
    svg.append(f'<text x="{width / 2}" y="65" text-anchor="middle" class="subtitle">Reference: cuBLAS non-TC · selected Split-K only · synthetic BF16 inputs</text>')

    legend_x = width / 2 - 230
    svg.append(f'<line x1="{legend_x}" y1="95" x2="{legend_x + 32}" y2="95" stroke="#263342" stroke-width="3"/>')
    svg.append(f'<text x="{legend_x + 42}" y="100" class="legend">cuBLAS non-TC (reference)</text>')
    legend_x += 265
    for method in METHODS:
        svg.append(f'<rect x="{legend_x}" y="87" width="28" height="16" rx="2" fill="{METHOD_COLORS[method]}" class="bar"/>')
        svg.append(f'<text x="{legend_x + 38}" y="100" class="legend">{METHOD_LABELS[method]}</text>')
        legend_x += 145

    outer_left, outer_right, column_gap = 92.0, width - 28.0, 28.0
    column_width = (outer_right - outer_left - column_gap * (len(batches) - 1)) / len(batches)
    row_bounds = ((180.0, 470.0), (640.0, 930.0), (1100.0, 1390.0))
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
            batch_points = [point for point in points if point.n == batch]
            draw_panel(svg, batch_points, layers, metric, bounds, plot_left, plot_right, top, bottom)

    svg.append(f'<text x="{outer_left}" y="{height - 22}" class="note">ZipServ bar labels show the selected Split-K. M×K is printed below each layer. Both model figures use identical y-axis ranges.</text>')
    save_svg(path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, help="Tuning directory; default: latest completed tuning run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "llama_error")
    parser.add_argument("--png-scale", type=float, default=1.0)
    parser.add_argument("--keep-svg", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.png_scale <= 0:
        raise ValueError("--png-scale must be positive")
    results_root = (args.results_root or latest_tuning_dir()).expanduser().resolve()
    config = load_config(args.config.expanduser().resolve())
    specs = {model["id"]: model for model in config["models"]}
    unknown = [model for model in args.models if model not in specs]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    points = load_selected_points(results_root, args.models)
    batches = [int(batch) for batch in config["matrix"]["batches"]]
    expected = {
        (model, layer["id"], batch)
        for model in args.models
        for layer in specs[model]["layers"]
        for batch in batches
    }
    present = {(point.model, point.layer, point.n) for point in points}
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"Missing selected error cases ({len(missing)}): {missing}")

    scales = metric_scales(points)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for model in args.models:
        model_points = [point for point in points if point.model == model]
        layers = [layer["id"] for layer in specs[model]["layers"]]
        svg_path = output_dir / f"{model}_error.svg"
        png_path = output_dir / f"{model}_error.png"
        plot_model(svg_path, model, model_points, layers, batches, scales)
        svg_to_png(svg_path, png_path, args.png_scale)
        if not args.keep_svg:
            svg_path.unlink()
        created.append(png_path)
        print(f"Created: {png_path}")
    print(f"Created {len(created)} error plot(s) from {results_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
