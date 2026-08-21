#!/usr/bin/env python3
"""Draw the two paper-style ZipGEMM/cuBLAS_TC comparison figures."""

from __future__ import annotations

import argparse
import ctypes
import csv
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments.json"
KERNELS = ("cublas_tc", "zipgemm")
BATCH_COLORS = {8: "#2b83ba", 16: "#fdae61", 32: "#4d7c0f"}
LLAMA_MODELS = ("llama3.1-8b", "llama3.1-70b", "llama3.1-405b")
FAMILY_LABELS = {
    "llama3.1": "LLaMA3.1",
    "qwen2.5": "Qwen2.5",
    "gemma3": "Gemma3",
    "mistral": "Mistral",
}
FAMILY_BACKGROUNDS = {
    "llama3.1": "#f2f7fb",
    "qwen2.5": "#f1f8f3",
    "gemma3": "#fff7ed",
    "mistral": "#f7f3fa",
}
MODEL_COLORS = {
    "llama3.1-8b": "#9ecae1",
    "llama3.1-70b": "#4292c6",
    "llama3.1-405b": "#08519c",
    "qwen2.5-7b": "#c7e9c0",
    "qwen2.5-14b": "#74c476",
    "qwen2.5-32b": "#31a354",
    "qwen2.5-72b": "#006d2c",
    "gemma3-12b": "#fdae6b",
    "gemma3-27b": "#e6550d",
    "mistral-24b": "#bcbddc",
    "mistral-123b": "#756bb1",
}
LAYER_LABELS = {
    "qkv_proj": "QKV_proj",
    "o_proj": "O_proj",
    "gateup_proj": "GateUp_proj",
    "down_proj": "Down_proj",
    "lm_head": "lm_head",
}


@dataclass(frozen=True)
class FinalPoint:
    model: str
    layer: str
    m: int
    k: int
    n: int
    split_k: int
    trials: int
    cublas_tc_ms: float
    zipgemm_ms: float

    @property
    def speedup(self) -> float:
        """Normalized speedup used by Figure 11: cuBLAS_TC / ZipGEMM."""
        return self.cublas_tc_ms / self.zipgemm_ms


class RsvgDimensionData(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("em", ctypes.c_double),
        ("ex", ctypes.c_double),
    ]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def model_order(config: dict[str, Any]) -> list[str]:
    return [model["id"] for model in config["models"] if model.get("enabled", True)]


def model_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        model["id"]: model
        for model in config["models"]
        if model.get("enabled", True)
    }


def display_model(model: str) -> str:
    family, size = model.rsplit("-", 1)
    return f"{FAMILY_LABELS.get(family, family)}-{size.upper()}"


def model_family(model: str) -> str:
    return model.rsplit("-", 1)[0]


def model_size(model: str) -> str:
    return model.rsplit("-", 1)[1].upper()


def analyze_results(
    results_root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[FinalPoint]]:
    """Read consolidated performance rows and report experiment completion."""
    batches = config["matrix"]["batches"]
    final_trials = config["phases"]["final"]["trials"]
    specs = model_specs(config)
    completion: list[dict[str, Any]] = []
    points: list[FinalPoint] = []
    all_rows = read_csv(results_root / "result_all.csv")
    if not all_rows:
        raise FileNotFoundError(f"Consolidated result not found or empty: {results_root / 'result_all.csv'}")

    for model_id in model_order(config):
        rows = [row for row in all_rows if row.get("model") == model_id]
        failure_keys: set[tuple[str, ...]] = set()
        final_groups: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)

        for row in rows:
            key = (
                row.get("phase", ""), row.get("model", ""), row.get("layer", ""),
                row.get("M", ""), row.get("K", ""), row.get("N", ""),
                row.get("split_k", ""), row.get("trial", ""),
            )
            if row.get("status") != "ok":
                failure_keys.add(key)
                continue
            if row.get("phase") == "run":
                final_groups[key].append({
                    "cublas_tc": float(row["cublas_tc_latency_ms"]),
                    "zipgemm": float(row["zipgemm_latency_ms"]),
                })

        aggregated: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
        for key, latencies in final_groups.items():
            aggregate_key = (
                key[1], key[2], int(key[3]), int(key[4]), int(key[5]), int(key[6])
            )
            aggregated[aggregate_key].extend(latencies)

        for key, trials in aggregated.items():
            points.append(
                FinalPoint(
                    model=key[0], layer=key[1], m=key[2], k=key[3],
                    n=key[4], split_k=key[5], trials=len(trials),
                    cublas_tc_ms=statistics.median(
                        trial["cublas_tc"] for trial in trials
                    ),
                    zipgemm_ms=statistics.median(
                        trial["zipgemm"] for trial in trials
                    ),
                )
            )

        layer_count = len(specs[model_id]["layers"])
        expected_final = layer_count * len(batches) * final_trials
        final_ok = sum(len(values) for values in final_groups.values())
        completion.append(
            {
                "model": model_id,
                "final_ok": final_ok,
                "final_expected": expected_final,
                "failed": len(failure_keys),
                "status": "complete" if final_ok == expected_final else "incomplete",
            }
        )

    return completion, points


def svg_document(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        """<style>
        text{font-family:'Liberation Serif','Times New Roman',serif;fill:#111}
        .main-title{font-size:27px;font-weight:700}
        .subtitle{font-size:16px}
        .panel-title{font-size:22px;font-weight:700}
        .axis{font-size:15px}
        .bar-value{font-size:12px}
        .x-label{font-size:17px}
        .legend{font-size:17px}
        .family-label{font-size:16px;font-weight:700}
        .missing{font-size:16px;font-style:italic;fill:#666}
        .grid{stroke:#d8d8d8;stroke-width:1}
        .axis-line{stroke:#111;stroke-width:1.5}
        .baseline{stroke:#555;stroke-width:2;stroke-dasharray:8 5}
        .bar{stroke:#111;stroke-width:1}
        </style>""",
    ]


def save_svg(path: Path, parts: list[str]) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_to_png(svg_path: Path, png_path: Path, scale: float) -> None:
    """Rasterize an SVG through system librsvg/cairo libraries."""
    try:
        rsvg = ctypes.CDLL("librsvg-2.so.2")
        cairo = ctypes.CDLL("libcairo.so.2")
        gobject = ctypes.CDLL("libgobject-2.0.so.0")
    except OSError as error:
        raise RuntimeError("PNG output requires system librsvg and cairo") from error

    rsvg.rsvg_handle_new_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    rsvg.rsvg_handle_new_from_file.restype = ctypes.c_void_p
    rsvg.rsvg_handle_get_dimensions.argtypes = [ctypes.c_void_p, ctypes.POINTER(RsvgDimensionData)]
    rsvg.rsvg_handle_render_cairo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    rsvg.rsvg_handle_render_cairo.restype = ctypes.c_int
    cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cairo.cairo_image_surface_create.restype = ctypes.c_void_p
    cairo.cairo_create.argtypes = [ctypes.c_void_p]
    cairo.cairo_create.restype = ctypes.c_void_p
    cairo.cairo_scale.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    cairo.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    cairo.cairo_surface_write_to_png.restype = ctypes.c_int
    cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
    gobject.g_object_unref.argtypes = [ctypes.c_void_p]

    error_pointer = ctypes.c_void_p()
    handle = rsvg.rsvg_handle_new_from_file(str(svg_path).encode(), ctypes.byref(error_pointer))
    if not handle:
        raise RuntimeError(f"librsvg could not open {svg_path}")
    surface = context = None
    try:
        dimensions = RsvgDimensionData()
        rsvg.rsvg_handle_get_dimensions(handle, ctypes.byref(dimensions))
        surface = cairo.cairo_image_surface_create(
            0,
            max(1, round(dimensions.width * scale)),
            max(1, round(dimensions.height * scale)),
        )
        context = cairo.cairo_create(surface)
        cairo.cairo_scale(context, scale, scale)
        if not rsvg.rsvg_handle_render_cairo(handle, context):
            raise RuntimeError(f"librsvg failed while rendering {svg_path}")
        status = cairo.cairo_surface_write_to_png(surface, str(png_path).encode())
        if status != 0:
            raise RuntimeError(f"cairo PNG write failed with status {status}")
    finally:
        if context:
            cairo.cairo_destroy(context)
        if surface:
            cairo.cairo_surface_destroy(surface)
        gobject.g_object_unref(handle)


def nice_ymax(values: list[float]) -> float:
    maximum = max([1.25, *values])
    return max(1.5, math.ceil(maximum * 1.08 / 0.25) * 0.25)


def draw_y_axis(
    svg: list[str], left: float, right: float, top: float, bottom: float, ymax: float
) -> Any:
    plot_height = bottom - top

    def y(value: float) -> float:
        return bottom - value / ymax * plot_height

    value = 0.0
    while value <= ymax + 1e-9:
        yy = y(value)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{left - 10}" y="{yy + 5:.2f}" text-anchor="end" class="axis">{value:.2f}</text>')
        value += 0.25
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis-line"/>')
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>')
    svg.append(f'<line x1="{left}" y1="{y(1.0):.2f}" x2="{right}" y2="{y(1.0):.2f}" class="baseline"/>')
    return y


def draw_batch_legend(svg: list[str], width: int, batches: list[int], y: int) -> None:
    item_width, baseline_width = 105, 190
    x = (width - (item_width * len(batches) + baseline_width)) / 2
    for batch in batches:
        svg.append(f'<rect x="{x}" y="{y - 14}" width="20" height="15" fill="{BATCH_COLORS[batch]}" class="bar"/>')
        svg.append(f'<text x="{x + 28}" y="{y}" class="legend">N={batch}</text>')
        x += item_width
    svg.append(f'<line x1="{x}" y1="{y - 7}" x2="{x + 50}" y2="{y - 7}" class="baseline"/>')
    svg.append(f'<text x="{x + 60}" y="{y}" class="legend">cuBLAS_TC = 1.0×</text>')


def plot_llama_layer_speedup(
    path: Path,
    points: list[FinalPoint],
    specs: dict[str, dict[str, Any]],
    batches: list[int],
) -> None:
    """One panel per enabled LLaMA model; each layer contains batch bars."""
    llama_models = [model for model in LLAMA_MODELS if model in specs]
    if not llama_models:
        raise ValueError("No enabled LLaMA models are available for the layer figure")

    width = 1320
    left, right = 105, width - 35
    panel_height, panel_gap, first_top = 270, 55, 155
    height = first_top + len(llama_models) * panel_height + (len(llama_models) - 1) * panel_gap + 70
    lookup = {(point.model, point.layer, point.n): point.speedup for point in points}
    ymax = nice_ymax([point.speedup for point in points if point.model in llama_models])
    svg = svg_document(width, height)
    svg.append(f'<text x="{width / 2}" y="38" text-anchor="middle" class="main-title">RTX 4090 — LLaMA3.1 layer-wise speedup</text>')
    svg.append(f'<text x="{width / 2}" y="65" text-anchor="middle" class="subtitle">ZipGEMM normalized to cuBLAS_TC: speedup = cuBLAS_TC latency / ZipGEMM latency</text>')
    draw_batch_legend(svg, width, batches, 104)

    for panel_index, model in enumerate(llama_models):
        top = first_top + panel_index * (panel_height + panel_gap)
        bottom = top + panel_height
        layers = [layer["id"] for layer in specs[model]["layers"]]
        group_width = (right - left) / len(layers)
        for index in range(len(layers)):
            if index % 2 == 0:
                svg.append(f'<rect x="{left + index * group_width:.2f}" y="{top}" width="{group_width:.2f}" height="{panel_height}" fill="#f1f1f1"/>')
        y = draw_y_axis(svg, left, right, top, bottom, ymax)
        svg.append(f'<text x="{(left + right) / 2}" y="{top + 25}" text-anchor="middle" class="panel-title">{html.escape(display_model(model))}</text>')
        svg.append(f'<text x="25" y="{(top + bottom) / 2}" text-anchor="middle" transform="rotate(-90 25 {(top + bottom) / 2})" class="x-label">Speedup</text>')
        bar_step = min(52.0, group_width / (len(batches) + 1))
        bar_width = bar_step * 0.78
        available = 0
        for layer_index, layer in enumerate(layers):
            center = left + group_width * (layer_index + 0.5)
            for batch_index, batch in enumerate(batches):
                value = lookup.get((model, layer, batch))
                if value is None:
                    continue
                available += 1
                x = center + (batch_index - 1) * bar_step - bar_width / 2
                yy = y(value)
                svg.append(f'<rect x="{x:.2f}" y="{yy:.2f}" width="{bar_width:.2f}" height="{bottom - yy:.2f}" fill="{BATCH_COLORS[batch]}" class="bar"/>')
                svg.append(f'<text x="{x + bar_width / 2:.2f}" y="{yy - 6:.2f}" text-anchor="middle" class="bar-value">{value:.2f}×</text>')
            label = LAYER_LABELS.get(layer, layer)
            svg.append(f'<text x="{center:.2f}" y="{bottom + 25}" text-anchor="middle" class="x-label">{html.escape(label)}</text>')
        if available == 0:
            svg.append(f'<text x="{(left + right) / 2}" y="{(top + bottom) / 2 + 15}" text-anchor="middle" class="missing">N/A — final measurements are not complete</text>')
    save_svg(path, svg)


def complete_model_averages(
    points: list[FinalPoint],
    specs: dict[str, dict[str, Any]],
    batches: list[int],
) -> dict[tuple[str, int], float]:
    """Arithmetic mean after normalizing every expected layer separately."""
    lookup = {(point.model, point.layer, point.n): point.speedup for point in points}
    averages: dict[tuple[str, int], float] = {}
    for model, spec in specs.items():
        layers = [layer["id"] for layer in spec["layers"]]
        for batch in batches:
            values = [lookup.get((model, layer, batch)) for layer in layers]
            if all(value is not None for value in values):
                averages[(model, batch)] = statistics.mean(value for value in values if value is not None)
    return averages


def plot_model_average_speedup(
    path: Path,
    points: list[FinalPoint],
    specs: dict[str, dict[str, Any]],
    models: list[str],
    batches: list[int],
) -> None:
    """Three N panels; one five-layer normalized mean per model."""
    # Keep the multi-model figure compact. Models in the same family use fixed,
    # narrow slots while the larger family gap preserves group boundaries.
    width, height = 920, 1160
    left, right = 90, width - 25
    panel_height, panel_gap, first_top = 270, 55, 155
    averages = complete_model_averages(points, specs, batches)
    ymax = nice_ymax(list(averages.values()))
    families: list[str] = []
    family_models: dict[str, list[str]] = defaultdict(list)
    for model in models:
        family = model_family(model)
        if family not in families:
            families.append(family)
        family_models[family].append(model)
    family_gap = 40.0
    slot_width = 64.0
    content_width = slot_width * len(models) + family_gap * (len(families) - 1)
    if content_width > right - left:
        raise ValueError("Model groups do not fit in the model-average figure")
    centers: dict[str, float] = {}
    family_spans: dict[str, tuple[float, float]] = {}
    cursor = left + ((right - left) - content_width) / 2
    for family in families:
        start = cursor
        for model in family_models[family]:
            centers[model] = cursor + slot_width / 2
            cursor += slot_width
        family_spans[family] = (start, cursor)
        cursor += family_gap
    svg = svg_document(width, height)
    svg.append(f'<text x="{width / 2}" y="38" text-anchor="middle" class="main-title">RTX 4090 — model-wise mean ZipGEMM speedup</text>')
    svg.append(f'<text x="{width / 2}" y="65" text-anchor="middle" class="subtitle">Arithmetic mean of five independently normalized layer speedups; incomplete models are not averaged</text>')
    legend_x = 90.0
    svg.append(f'<line x1="{legend_x}" y1="99" x2="{legend_x + 50}" y2="99" class="baseline"/>')
    svg.append(f'<text x="{legend_x + 62}" y="106" class="legend">cuBLAS_TC = 1.0×</text>')
    legend_x += 220
    for family in families:
        palette = [MODEL_COLORS[model] for model in family_models[family]]
        for color in palette:
            svg.append(f'<rect x="{legend_x}" y="91" width="13" height="15" fill="{color}" class="bar"/>')
            legend_x += 14
        svg.append(f'<text x="{legend_x + 6}" y="106" class="legend">{html.escape(FAMILY_LABELS[family])}</text>')
        legend_x += 105

    for panel_index, batch in enumerate(batches):
        top = first_top + panel_index * (panel_height + panel_gap)
        bottom = top + panel_height
        for family in families:
            family_left, family_right = family_spans[family]
            svg.append(f'<rect x="{family_left:.2f}" y="{top}" width="{family_right - family_left:.2f}" height="{panel_height}" fill="{FAMILY_BACKGROUNDS[family]}"/>')
        y = draw_y_axis(svg, left, right, top, bottom, ymax)
        svg.append(f'<text x="{(left + right) / 2}" y="{top + 25}" text-anchor="middle" class="panel-title">N = {batch}</text>')
        svg.append(f'<text x="25" y="{(top + bottom) / 2}" text-anchor="middle" transform="rotate(-90 25 {(top + bottom) / 2})" class="x-label">Mean speedup</text>')
        # Keep sizes visually grouped without making adjacent bars touch.
        # The larger family_gap still makes family boundaries unambiguous.
        bar_width = min(42.0, slot_width * 0.78)
        for model in models:
            center = centers[model]
            value = averages.get((model, batch))
            if value is None:
                svg.append(f'<text x="{center:.2f}" y="{bottom - 12}" text-anchor="middle" class="missing">N/A</text>')
            else:
                yy = y(value)
                svg.append(f'<rect x="{center - bar_width / 2:.2f}" y="{yy:.2f}" width="{bar_width:.2f}" height="{bottom - yy:.2f}" fill="{MODEL_COLORS[model]}" class="bar"/>')
                svg.append(f'<text x="{center:.2f}" y="{yy - 7:.2f}" text-anchor="middle" class="axis">{value:.2f}×</text>')
            svg.append(f'<text x="{center:.2f}" y="{bottom + 20}" text-anchor="middle" class="axis">{html.escape(model_size(model))}</text>')
        for family in families:
            family_left, family_right = family_spans[family]
            svg.append(f'<text x="{(family_left + family_right) / 2:.2f}" y="{bottom + 43}" text-anchor="middle" class="family-label">{html.escape(FAMILY_LABELS[family])}</text>')
    save_svg(path, svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the two paper-style ZipServ PNG figures")
    parser.add_argument("--results-root", type=Path, required=True, help="Run directory containing result_all.csv")
    parser.add_argument("--output-dir", type=Path, help="Default: directory containing this script")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--png-scale", type=float, default=1.0, help="PNG resolution multiplier (default: 1.0)")
    parser.add_argument("--allow-incomplete", action="store_true", help="Generate figures with N/A panels/bars for incomplete models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.png_scale <= 0:
        raise ValueError("--png-scale must be greater than zero")
    results_root = args.results_root.expanduser().resolve()
    output_dir = (args.output_dir or Path(__file__).resolve().parent).expanduser().resolve()
    config = load_config(args.config.expanduser().resolve())
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    completion, points = analyze_results(results_root, config)
    specs = model_specs(config)
    models = model_order(config)
    batches = config["matrix"]["batches"]
    complete = all(row["status"] == "complete" for row in completion)
    figures = (
        ("llama31_layer_speedup", lambda path: plot_llama_layer_speedup(path, points, specs, batches)),
        ("model_average_speedup", lambda path: plot_model_average_speedup(path, points, specs, models, batches)),
    )
    for name, plotter in figures:
        svg_path = output_dir / f"{name}.svg"
        png_path = output_dir / f"{name}.png"
        plotter(svg_path)
        svg_to_png(svg_path, png_path, args.png_scale)
        svg_path.unlink()

    incomplete = [row for row in completion if row["status"] != "complete"]
    print(f"Figures: {output_dir}")
    print("Created: llama31_layer_speedup.png, model_average_speedup.png")
    print(f"Completion: {len(completion) - len(incomplete)}/{len(completion)} models; performance points={len(points)}")
    for row in incomplete:
        print(f"Incomplete: {row['model']} (run {row['final_ok']}/{row['final_expected']})")
    if not complete and not args.allow_incomplete:
        print("ERROR: incomplete models are marked N/A; use --allow-incomplete to accept partial figures", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
