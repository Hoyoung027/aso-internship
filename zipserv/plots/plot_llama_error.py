#!/usr/bin/env python3
"""Plot ZipServ errors against cuBLAS TC from one real-weight LLaMA block."""

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
METHOD_COLORS = {"zip": "#2b83ba"}


@dataclass(frozen=True)
class ErrorPoint:
    model: str
    layer: str
    m: int
    k: int
    n: int
    split_k: int
    zip_relative_percent: float
    zip_mae: float
    zip_significant_percent: float
    zip_absolute_significant_percent: float | None = None


@dataclass(frozen=True)
class Metric:
    title: str
    axis_label: str
    scale: str
    zip_value: Callable[[ErrorPoint], float | None]


METRICS = (
    Metric(
        title="Mean element-wise relative error",
        axis_label="Mean relative error (%)",
        scale="log-percent",
        zip_value=lambda point: point.zip_relative_percent,
    ),
    Metric(
        title="Mean absolute error per output element",
        axis_label="Mean absolute error / element",
        scale="log-scientific",
        zip_value=lambda point: point.zip_mae,
    ),
    Metric(
        title="Elements with relative error > 1e-4 (0.01%)",
        axis_label="Significant elements (%)",
        scale="linear-percent",
        zip_value=lambda point: point.zip_significant_percent,
    ),
    Metric(
        title="Elements with absolute error > 1e-4 (output units)",
        axis_label="Absolute-error exceedance (%)",
        scale="linear-percent",
        zip_value=lambda point: point.zip_absolute_significant_percent,
    ),
)


def absolute_exceedance_percent(rows: list[dict[str, str]], output_elements: int) -> float | None:
    fields = ('zip_vs_tc_absolute_error_threshold',
              'zip_vs_tc_absolute_error_exceedance_count',
              'zip_vs_tc_absolute_error_exceedance_percent')
    # All trials must have measured data; the historical MAE is insufficient.
    if any(row.get(field, '') in ('', None) for row in rows for field in fields):
        return None
    values = []
    for row in rows:
        threshold = float(row[fields[0]])
        count = int(row[fields[1]])
        percent = float(row[fields[2]])
        if not math.isclose(threshold, 1e-4, rel_tol=1e-12, abs_tol=0):
            raise ValueError(f'Unexpected absolute-error threshold: {threshold}')
        if not 0 <= count <= output_elements or not math.isfinite(percent):
            raise ValueError('Invalid absolute-error exceedance measurement')
        value = 100.0 * count / output_elements
        if not math.isclose(value, percent, abs_tol=1e-6):
            raise ValueError('Absolute-error count and percent disagree')
        values.append(value)
    return statistics.median(values)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_selected_points(results_root: Path, models: list[str], block_index: int = 0) -> list[ErrorPoint]:
    # Final run rows already contain the Split-K used for the measurement.
    # Do not reselect from a newer tuning CSV or mix transformer blocks.
    selected = {}
    grouped: dict[tuple[str, str, int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(results_root / "result_all.csv"):
        if row.get("model") not in models:
            continue
        if int(row.get("block_index") or 0) != block_index:
            continue
        if (row.get("phase") != "run" or row.get("status") != "ok"
                or not row.get("weight_source")
                or "synthetic" in row["weight_source"].lower()):
            raise ValueError("Expected successful real-weight final-run rows only")
        key = (row["model"], row["layer"], int(row["N"]))
        split_k = int(row["split_k"])
        if selected.setdefault(key, split_k) != split_k:
            raise ValueError(f"Inconsistent final-run Split-K: {key}")
        group_key = (
            row["model"], row["layer"], int(row["M"]), int(row["K"]),
            int(row["N"]), int(row["split_k"]),
        )
        grouped[group_key].append(row)

    def median(rows: list[dict[str, str]], field: str) -> float:
        values = [float(row[field]) for row in rows]
        if not values or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Missing or invalid error metric {field}")
        return statistics.median(values)

    points: list[ErrorPoint] = []
    for (model, layer, m, k, n, split_k), rows in grouped.items():
        if len(rows) != 3 or {int(row['trial']) for row in rows} != {1, 2, 3}:
            raise ValueError(f"Expected three unique trials: {model}/{layer}/N={n}")
        output_elements = m * n
        points.append(
            ErrorPoint(
                model=model,
                layer=layer,
                m=m,
                k=k,
                n=n,
                split_k=split_k,
                zip_relative_percent=median(rows, "zip_vs_tc_average_relative_error") * 100.0,
                zip_mae=median(rows, "zip_vs_tc_total_absolute_error") / output_elements,
                zip_significant_percent=median(rows, "zip_vs_tc_significant_error_percent"),
                zip_absolute_significant_percent=absolute_exceedance_percent(rows, output_elements),
            )
        )
    return points


def log_bounds(values: list[float]) -> tuple[float, float, list[float]]:
    positive = [value for value in values if value > 0]
    if not positive:
        return 1e-6, 1.0, [1e-6, 1e-4, 1e-2, 1.0]
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
    maximum = max(1.0, max(values, default=1.0))
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
            for value in (metric.zip_value(point),)
            if value is not None
        ]
        scales.append(log_bounds(values) if metric.scale.startswith("log") else
                      linear_bounds(values) if values else (0.0, 100.0, [0, 20, 40, 60, 80, 100]))
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
    bar_width = min(40.0, group_width * 0.45)
    for index, layer in enumerate(layers):
        group_left = left + index * group_width
        center = group_left + group_width / 2
        point = lookup.get(layer)
        if point:
            value = metric.zip_value(point)
            x = center - bar_width / 2
            yy = y(value) if value is not None else bottom
            if value is not None and value > 0:
                svg.append(f'<rect x="{x:.2f}" y="{yy:.2f}" width="{bar_width:.2f}" height="{bottom - yy:.2f}" rx="2" fill="{METHOD_COLORS["zip"]}" class="bar"/>')
            # A zero error cannot be plotted on a logarithmic axis: label it explicitly.
            label = ('N/A' if value is None else
                     f'K={point.split_k}' if value > 0 else f'0 (K={point.split_k})')
            svg.append(f'<text x="{center:.2f}" y="{max(top + 11, yy - 5):.2f}" text-anchor="middle" class="k-label">{label}</text>')
            svg.append(f'<text x="{center:.2f}" y="{bottom + 18}" text-anchor="middle" class="layer">{html.escape(LAYER_LABELS.get(layer, layer))}</text>')
            svg.append(f'<text x="{center:.2f}" y="{bottom + 33}" text-anchor="middle" class="shape">{point.m:,}×{point.k:,}</text>')
    if panel_points and all(metric.zip_value(point) is None for point in panel_points):
        svg.append(f'<text x="{(left + right) / 2}" y="{(top + bottom) / 2}" text-anchor="middle" class="note">Not measured in this run</text>')


def plot_model(
    path: Path, model: str, points: list[ErrorPoint], layers: list[str],
    batches: list[int], scales: list[tuple[float, float, list[float]]],
    block_index: int = 0,
) -> None:
    width, height = 1800, 1950
    svg = svg_document(width, height)
    add_style(svg)
    svg.append(f'<text x="{width / 2}" y="38" text-anchor="middle" class="title">{model_display_name(model)} — numerical error comparison</text>')
    scope = f'block {block_index}' + (' + shared LM Head' if 'lm_head' in layers else '')
    svg.append(f'<text x="{width / 2}" y="65" text-anchor="middle" class="subtitle">Reference: cuBLAS TC · real weights ({scope}) · synthetic BF16 activations · final run, median of 3 trials</text>')
    legend_x = width / 2 - 120
    svg.append(f'<rect x="{legend_x}" y="87" width="28" height="16" rx="2" fill="{METHOD_COLORS["zip"]}" class="bar"/>')
    svg.append(f'<text x="{legend_x + 38}" y="100" class="legend">ZipServ vs cuBLAS TC</text>')

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
            batch_points = [point for point in points if point.n == batch]
            draw_panel(svg, batch_points, layers, metric, bounds, plot_left, plot_right, top, bottom)

    svg.append(f'<text x="{outer_left}" y="{height - 22}" class="note">K = measured Split-K; M×K = weight shape. N/A = unmeasured, not zero. Absolute threshold is illustrative, not an accuracy acceptance criterion.</text>')
    save_svg(path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="Real-weight final-run directory")
    parser.add_argument("--block-index", type=int, default=0, help="Plot this block only; default: 0")
    parser.add_argument("--config", type=Path, help="Default: experiment snapshot in results-root")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "image" / "real" / "llama")
    parser.add_argument("--png-scale", type=float, default=2.0)
    parser.add_argument("--keep-svg", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.png_scale <= 0:
        raise ValueError("--png-scale must be positive")
    results_root = args.results_root.expanduser().resolve()
    config = load_config((args.config or results_root / "experiments.json").expanduser().resolve())
    specs = {model["id"]: model for model in config["models"]}
    unknown = [model for model in args.models if model not in specs]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if args.block_index < 0:
        raise ValueError("--block-index must be non-negative")
    points = load_selected_points(results_root, args.models, args.block_index)
    missing_absolute = sum(p.zip_absolute_significant_percent is None for p in points)
    if missing_absolute:
        print(f'WARNING: {missing_absolute} cases have no absolute-error exceedance data; panel D will show N/A. Rebuild and remeasure to populate it.', file=sys.stderr)
    batches = [int(batch) for batch in config["matrix"]["batches"]]
    expected = {
        (model, layer["id"], batch)
        for model in args.models
        for layer in specs[model]["layers"]
        if layer.get("block_scoped", True) or args.block_index == 0
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
        layers = [layer["id"] for layer in specs[model]["layers"]
                  if layer.get("block_scoped", True) or args.block_index == 0]
        suffix = "" if args.block_index == 0 else f"_block-{args.block_index}"
        svg_path = output_dir / f"{model}{suffix}_error.svg"
        png_path = output_dir / f"{model}{suffix}_error.png"
        plot_model(svg_path, model, model_points, layers, batches, scales, args.block_index)
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
