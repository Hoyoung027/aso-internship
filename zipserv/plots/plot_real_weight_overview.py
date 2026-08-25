#!/usr/bin/env python3
"""Plot real-weight exponent coverage, compression, and ZipServ speedup by block."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from plot_results import save_svg, svg_document, svg_to_png


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "image" / "real" / "llama"
DEFAULT_MODELS = ("llama3.1-8b", "llama3.1-70b")
MODELS = DEFAULT_MODELS + ("qwen2.5-7b", "qwen2.5-14b")
EXPECTED_BLOCKS = {
    "llama3.1-8b": (0, 16, 31),
    "llama3.1-70b": (0, 40, 79),
    "qwen2.5-7b": (0, 14, 27),
    "qwen2.5-14b": (0, 24, 47),
}
MODEL_LABELS = {
    "llama3.1-8b": "LLaMA 3.1 8B",
    "llama3.1-70b": "LLaMA 3.1 70B",
    "qwen2.5-7b": "Qwen 2.5 7B",
    "qwen2.5-14b": "Qwen 2.5 14B",
}
LAYERS = ("qkv_proj", "o_proj", "gateup_proj", "down_proj")
LAYER_LABELS = {
    "qkv_proj": "QKV",
    "o_proj": "O",
    "gateup_proj": "GateUp",
    "down_proj": "Down",
}
LAYER_COLORS = {
    "qkv_proj": "#4e79a7",
    "o_proj": "#59a14f",
    "gateup_proj": "#f28e2b",
    "down_proj": "#8b6bb1",
}
BATCH_COLORS = {8: "#2b83ba", 16: "#f28e2b", 32: "#4d9221"}
SYNTHETIC_COMPRESSION_RATIO = 1.42

CASE_PATTERN = re.compile(r"CASE START .*?layer=(\S+)")
EXPONENT_PATTERN = re.compile(r"High-freq exponent list \(\d+ values\):\s*(.*)")
HIGH_FREQUENCY_PATTERN = re.compile(
    r"High-frequency exponent elements:\s*\d+\s*\(([0-9.]+)%\)"
)
ORIGINAL_PATTERN = re.compile(r"Original size:\s*(\d+) bytes")
COMPRESSED_PATTERN = re.compile(r"Compressed size:\s*(\d+) bytes")
RATIO_PATTERN = re.compile(r"Compression ratio:\s*([0-9.]+):1")


@dataclass(frozen=True)
class CompressionPoint:
    model: str
    block: int
    layer: str
    high_frequency_percent: float
    exponents: tuple[int, ...]
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float

    @property
    def original_mib(self) -> float:
        return self.original_bytes / (1024 * 1024)

    @property
    def compressed_mib(self) -> float:
        return self.compressed_bytes / (1024 * 1024)

    @property
    def saving_percent(self) -> float:
        return (1.0 - self.compressed_bytes / self.original_bytes) * 100.0

    @property
    def exponent_label(self) -> str:
        if not self.exponents:
            return "E?"
        if len(self.exponents) == 1:
            return f"E{self.exponents[0]}"
        return f"E{self.exponents[0]}–{self.exponents[-1]}"


@dataclass(frozen=True)
class SpeedPoint:
    model: str
    block: int
    layer: str
    batch: int
    split_k: int
    speedup: float


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_result_roots() -> list[Path]:
    candidates = []
    for path in (PROJECT_ROOT / "results").glob("*-run"):
        result_file = path / "result_all.csv"
        if not result_file.is_file():
            continue
        rows = read_csv(result_file)
        if any(
            row.get("model") in MODELS
            and row.get("phase") == "run"
            and row.get("status") == "ok"
            and "synthetic" not in row.get("weight_source", "").lower()
            for row in rows
        ):
            candidates.append(path.resolve())
    if not candidates:
        raise FileNotFoundError("No completed supported real-weight run directory was found")
    return sorted(candidates)


def choose_model_block_sources(
    roots: Iterable[Path],
) -> dict[tuple[str, int], tuple[Path, list[dict[str, str]]]]:
    choices: dict[tuple[str, int], tuple[float, Path, list[dict[str, str]]]] = {}
    for root in roots:
        root = root.expanduser().resolve()
        result_file = root / "result_all.csv"
        rows = [
            row for row in read_csv(result_file)
            if row.get("model") in MODELS
            and row.get("phase") == "run"
            and row.get("status") == "ok"
            and row.get("layer") in LAYERS
            and "synthetic" not in row.get("weight_source", "").lower()
        ]
        by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_key[(row["model"], int(row.get("block_index") or 0))].append(row)
        for key, group in by_key.items():
            score = result_file.stat().st_mtime
            if key not in choices or score > choices[key][0]:
                choices[key] = (score, root, group)
    return {key: (value[1], value[2]) for key, value in choices.items()}


def parse_compression_log(
    log_path: Path,
    model: str,
    block: int,
) -> dict[tuple[str, int, str], CompressionPoint]:
    if not log_path.is_file():
        raise FileNotFoundError(f"Benchmark log not found: {log_path}")

    parsed: dict[tuple[str, int, str], CompressionPoint] = {}
    current_layer: str | None = None
    values: dict[str, Any] = {}

    def finish() -> None:
        nonlocal values
        if current_layer not in LAYERS:
            values = {}
            return
        required = {
            "high_frequency_percent", "exponents", "original_bytes",
            "compressed_bytes", "compression_ratio",
        }
        if required.issubset(values):
            key = (model, block, current_layer)
            point = CompressionPoint(
                model=model,
                block=block,
                layer=current_layer,
                high_frequency_percent=values["high_frequency_percent"],
                exponents=values["exponents"],
                original_bytes=values["original_bytes"],
                compressed_bytes=values["compressed_bytes"],
                compression_ratio=values["compression_ratio"],
            )
            previous = parsed.get(key)
            if previous is not None and previous != point:
                raise ValueError(f"Inconsistent compression data in {log_path}: {key}")
            parsed[key] = point
        values = {}

    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            case_match = CASE_PATTERN.search(line)
            if case_match:
                finish()
                current_layer = case_match.group(1)
                continue
            if current_layer not in LAYERS:
                continue
            exponent_match = EXPONENT_PATTERN.search(line)
            if exponent_match:
                values["exponents"] = tuple(
                    int(value) for value in exponent_match.group(1).split()
                )
                continue
            frequency_match = HIGH_FREQUENCY_PATTERN.search(line)
            if frequency_match:
                values["high_frequency_percent"] = float(frequency_match.group(1))
                continue
            original_match = ORIGINAL_PATTERN.search(line)
            if original_match and "original_bytes" not in values:
                values["original_bytes"] = int(original_match.group(1))
                continue
            compressed_match = COMPRESSED_PATTERN.search(line)
            if compressed_match and "compressed_bytes" not in values:
                values["compressed_bytes"] = int(compressed_match.group(1))
                continue
            ratio_match = RATIO_PATTERN.search(line)
            if ratio_match and "compression_ratio" not in values:
                values["compression_ratio"] = float(ratio_match.group(1))
    finish()
    return parsed


def compression_points(
    sources: dict[tuple[str, int], tuple[Path, list[dict[str, str]]]],
) -> dict[tuple[str, int, str], CompressionPoint]:
    points: dict[tuple[str, int, str], CompressionPoint] = {}
    for (model, block), (_, rows) in sources.items():
        qkv_rows = [row for row in rows if row["layer"] == "qkv_proj"]
        if not qkv_rows:
            continue
        log_value = qkv_rows[0].get("log_file", "")
        if not log_value:
            raise ValueError(f"log_file is missing for {model}/block-{block}")
        log_path = Path(log_value).expanduser()
        points.update(parse_compression_log(log_path, model, block))
    return points


def speed_points(
    sources: dict[tuple[str, int], tuple[Path, list[dict[str, str]]]],
) -> dict[tuple[str, int, str, int], SpeedPoint]:
    points: dict[tuple[str, int, str, int], SpeedPoint] = {}
    for (model, block), (_, rows) in sources.items():
        grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["layer"], int(row["N"]))].append(row)
        for (layer, batch), group in grouped.items():
            tc_values = [float(row["cublas_tc_latency_ms"]) for row in group]
            zip_values = [float(row["zipgemm_latency_ms"]) for row in group]
            split_counts = Counter(int(row["split_k"]) for row in group)
            split_k = min(
                split_counts,
                key=lambda value: (-split_counts[value], value),
            )
            key = (model, block, layer, batch)
            points[key] = SpeedPoint(
                model=model,
                block=block,
                layer=layer,
                batch=batch,
                split_k=split_k,
                speedup=statistics.median(tc_values) / statistics.median(zip_values),
            )
    return points


def tick_values(lower: float, upper: float, step: float) -> list[float]:
    count = int(math.floor((upper - lower) / step + 0.5))
    return [lower + index * step for index in range(count + 1)]


def add_panel_backgrounds(
    svg: list[str],
    left: float,
    category_width: float,
    top: float,
    bottom: float,
    blocks: tuple[int, ...],
) -> None:
    for block_index, _ in enumerate(blocks):
        x = left + block_index * len(LAYERS) * category_width
        width = len(LAYERS) * category_width
        fill = "#f6f8fb" if block_index % 2 == 0 else "#ffffff"
        svg.append(
            f'<rect x="{x:.2f}" y="{top:.2f}" width="{width:.2f}" '
            f'height="{bottom - top:.2f}" fill="{fill}"/>'
        )
        if block_index:
            svg.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
                'stroke="#aeb8c6" stroke-width="2"/>'
            )


def add_grid(
    svg: list[str],
    left: float,
    right: float,
    top: float,
    bottom: float,
    lower: float,
    upper: float,
    ticks: list[float],
    formatter,
) -> None:
    y = lambda value: bottom - (value - lower) / (upper - lower) * (bottom - top)
    for value in ticks:
        yy = y(value)
        svg.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" '
            'class="grid"/>'
        )
        svg.append(
            f'<text x="{left - 16}" y="{yy + 5:.2f}" text-anchor="end" '
            f'class="axis">{formatter(value)}</text>'
        )
    svg.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis-line"/>'
    )
    svg.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis-line"/>'
    )


def add_block_headers(
    svg: list[str],
    left: float,
    category_width: float,
    top: float,
    blocks: tuple[int, ...],
) -> None:
    for block_index, block in enumerate(blocks):
        center = left + (block_index * len(LAYERS) + len(LAYERS) / 2) * category_width
        svg.append(
            f'<text x="{center:.2f}" y="{top + 25:.2f}" text-anchor="middle" '
            f'class="block-title">Block {block}</text>'
        )


def add_y_label(
    svg: list[str], label: str, x: float, top: float, bottom: float,
) -> None:
    center = (top + bottom) / 2
    svg.append(
        f'<text x="{x}" y="{center:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 {x} {center:.2f})" class="axis-label">'
        f'{html.escape(label)}</text>'
    )


def plot_model(
    svg_path: Path,
    model: str,
    blocks: tuple[int, ...],
    compression: dict[tuple[str, int, str], CompressionPoint],
    speeds: dict[tuple[str, int, str, int], SpeedPoint],
) -> None:
    width, height = 2400, 1660
    left, right = 125.0, width - 55.0
    category_count = len(blocks) * len(LAYERS)
    category_width = (right - left) / category_count
    centers = {
        (block, layer): left + (index + 0.5) * category_width
        for index, (block, layer) in enumerate(
            (block, layer) for block in blocks for layer in LAYERS
        )
    }
    svg = svg_document(width, height)
    svg.append(
        """<style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#172033}
        .title{font-size:32px;font-weight:700}.subtitle{font-size:17px;fill:#59677c}
        .panel-title{font-size:22px;font-weight:700}.block-title{font-size:17px;font-weight:700;fill:#334155}
        .axis{font-size:13px}.axis-label{font-size:17px;font-weight:700}
        .grid{stroke:#dce3ec;stroke-width:1}.axis-line{stroke:#283548;stroke-width:1.5}
        .value{font-size:13px;font-weight:700}.detail{font-size:11px;fill:#526078}
        .inside-value{font-size:12px;font-weight:700;fill:white}
        .speed-value{font-size:10.5px;font-weight:700}.speed-detail{font-size:9.5px;fill:#526078}
        .category{font-size:14px;font-weight:700}.legend{font-size:15px}
        .note{font-size:13px;fill:#64748b}
        </style>"""
    )
    display_model = MODEL_LABELS[model]
    svg.append(
        f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">'
        f'{html.escape(display_model)} — Real-weight compression and performance</text>'
    )
    svg.append(
        f'<text x="{width / 2}" y="72" text-anchor="middle" class="subtitle">'
        'The same block × projection axis links exponent concentration, storage, and speedup</text>'
    )

    # Panel 1: high-frequency exponent coverage.
    p1_top, p1_bottom = 140.0, 500.0
    add_panel_backgrounds(svg, left, category_width, p1_top, p1_bottom, blocks)
    add_grid(
        svg, left, right, p1_top + 45, p1_bottom, 70.0, 100.0,
        tick_values(70.0, 100.0, 5.0), lambda value: f"{value:.0f}%",
    )
    add_block_headers(svg, left, category_width, p1_top, blocks)
    add_y_label(svg, "High-frequency elements", 28, p1_top + 45, p1_bottom)
    svg.append(
        f'<text x="{left}" y="{p1_top - 15}" class="panel-title">'
        '1. Exponent concentration and selected seven-exponent window</text>'
    )
    p1_y = lambda value: p1_bottom - (value - 70.0) / 30.0 * (p1_bottom - (p1_top + 45))
    bar_width = category_width * 0.58
    for block in blocks:
        for layer in LAYERS:
            point = compression[(model, block, layer)]
            center = centers[(block, layer)]
            yy = p1_y(point.high_frequency_percent)
            svg.append(
                f'<rect x="{center - bar_width / 2:.2f}" y="{yy:.2f}" '
                f'width="{bar_width:.2f}" height="{p1_bottom - yy:.2f}" rx="3" '
                f'fill="{LAYER_COLORS[layer]}" opacity=".9"/>'
            )
            svg.append(
                f'<text x="{center:.2f}" y="{yy - 22:.2f}" text-anchor="middle" '
                f'class="value">{point.high_frequency_percent:.1f}%</text>'
            )
            svg.append(
                f'<text x="{center:.2f}" y="{yy - 7:.2f}" text-anchor="middle" '
                f'class="detail">{point.exponent_label}</text>'
            )

    # Panel 2: compression ratio and storage footprint.
    p2_top, p2_bottom = 610.0, 985.0
    cmin, cmax = 1.20, 1.48
    add_panel_backgrounds(svg, left, category_width, p2_top, p2_bottom, blocks)
    add_grid(
        svg, left, right, p2_top + 45, p2_bottom, cmin, cmax,
        tick_values(1.20, 1.45, 0.05), lambda value: f"{value:.2f}×",
    )
    add_block_headers(svg, left, category_width, p2_top, blocks)
    add_y_label(svg, "Compression ratio", 28, p2_top + 45, p2_bottom)
    svg.append(
        f'<text x="{left}" y="{p2_top - 15}" class="panel-title">'
        '2. Compression ratio, BF16 → compressed MiB, and storage saving</text>'
    )
    p2_y = lambda value: p2_bottom - (value - cmin) / (cmax - cmin) * (p2_bottom - (p2_top + 45))
    synthetic_y = p2_y(SYNTHETIC_COMPRESSION_RATIO)
    svg.append(
        f'<line x1="{left}" y1="{synthetic_y:.2f}" x2="{right}" y2="{synthetic_y:.2f}" '
        'stroke="#c2410c" stroke-width="2.5" stroke-dasharray="10 6"/>'
    )
    svg.append(
        f'<rect x="{left + 7:.2f}" y="{synthetic_y - 25:.2f}" width="235" height="20" '
        'fill="white" opacity=".92"/>'
    )
    svg.append(
        f'<text x="{left + 15:.2f}" y="{synthetic_y - 10:.2f}" '
        'font-size="14" font-weight="700" fill="#c2410c">Synthetic baseline 1.42×</text>'
    )
    for block in blocks:
        for layer in LAYERS:
            point = compression[(model, block, layer)]
            center = centers[(block, layer)]
            yy = p2_y(point.compression_ratio)
            svg.append(
                f'<rect x="{center - bar_width / 2:.2f}" y="{yy:.2f}" '
                f'width="{bar_width:.2f}" height="{p2_bottom - yy:.2f}" rx="3" '
                f'fill="{LAYER_COLORS[layer]}" opacity=".9"/>'
            )
            if point.compression_ratio >= 1.36:
                svg.append(
                    f'<text x="{center:.2f}" y="{yy + 18:.2f}" text-anchor="middle" '
                    f'class="inside-value">{point.compression_ratio:.3f}×</text>'
                )
            else:
                svg.append(
                    f'<text x="{center:.2f}" y="{yy - 8:.2f}" text-anchor="middle" '
                    f'class="value">{point.compression_ratio:.3f}×</text>'
                )
            svg.append(
                f'<text x="{center:.2f}" y="{p2_bottom + 19:.2f}" text-anchor="middle" '
                f'class="detail">{point.original_mib:.1f}→{point.compressed_mib:.1f} MiB</text>'
            )
            svg.append(
                f'<text x="{center:.2f}" y="{p2_bottom + 36:.2f}" text-anchor="middle" '
                f'class="detail">save {point.saving_percent:.1f}%</text>'
            )

    # Panel 3: final-run performance relative to cuBLAS TC.
    p3_top, p3_bottom = 1100.0, 1505.0
    speed_values = [
        point.speedup for key, point in speeds.items()
        if key[0] == model and key[1] in blocks and key[2] in LAYERS
    ]
    smin = 1.0
    smax = max(1.55, math.ceil((max(speed_values, default=1.5) + 0.04) / 0.1) * 0.1)
    add_panel_backgrounds(svg, left, category_width, p3_top, p3_bottom, blocks)
    add_grid(
        svg, left, right, p3_top + 45, p3_bottom, smin, smax,
        tick_values(smin, smax, 0.1), lambda value: f"{value:.1f}×",
    )
    add_block_headers(svg, left, category_width, p3_top, blocks)
    add_y_label(svg, "cuBLAS TC / ZipServ", 28, p3_top + 45, p3_bottom)
    svg.append(
        f'<text x="{left}" y="{p3_top - 15}" class="panel-title">'
        '3. ZipServ speedup over cuBLAS TC (median of 3 final trials)</text>'
    )
    legend_x = right - 360
    for batch in (8, 16, 32):
        svg.append(
            f'<rect x="{legend_x}" y="{p3_top - 32}" width="27" height="16" rx="2" '
            f'fill="{BATCH_COLORS[batch]}"/>'
        )
        svg.append(
            f'<text x="{legend_x + 35}" y="{p3_top - 18}" class="legend">N={batch}</text>'
        )
        legend_x += 115
    p3_y = lambda value: p3_bottom - (value - smin) / (smax - smin) * (p3_bottom - (p3_top + 45))
    baseline_y = p3_y(1.0)
    svg.append(
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{right}" y2="{baseline_y:.2f}" '
        'stroke="#334155" stroke-width="2" stroke-dasharray="8 5"/>'
    )
    speed_bar_width = category_width * 0.20
    speed_gap = category_width * 0.035
    group_span = 3 * speed_bar_width + 2 * speed_gap
    for block in blocks:
        for layer in LAYERS:
            center = centers[(block, layer)]
            group_left = center - group_span / 2
            for batch_index, batch in enumerate((8, 16, 32)):
                point = speeds[(model, block, layer, batch)]
                xx = group_left + batch_index * (speed_bar_width + speed_gap)
                yy = p3_y(point.speedup)
                svg.append(
                    f'<rect x="{xx:.2f}" y="{yy:.2f}" width="{speed_bar_width:.2f}" '
                    f'height="{p3_bottom - yy:.2f}" rx="2" fill="{BATCH_COLORS[batch]}"/>'
                )
                svg.append(
                    f'<text x="{xx + speed_bar_width / 2:.2f}" y="{yy - 20:.2f}" '
                    f'text-anchor="middle" class="speed-value">{point.speedup:.2f}×</text>'
                )
                svg.append(
                    f'<text x="{xx + speed_bar_width / 2:.2f}" y="{yy - 6:.2f}" '
                    f'text-anchor="middle" class="speed-detail">K={point.split_k}</text>'
                )

    # Shared x-axis labels appear only once, beneath the last panel.
    for block in blocks:
        for layer in LAYERS:
            center = centers[(block, layer)]
            svg.append(
                f'<text x="{center:.2f}" y="{p3_bottom + 24:.2f}" text-anchor="middle" '
                f'class="category">{LAYER_LABELS[layer]}</text>'
            )
    for block_index, block in enumerate(blocks):
        center = left + (block_index * len(LAYERS) + len(LAYERS) / 2) * category_width
        svg.append(
            f'<text x="{center:.2f}" y="{p3_bottom + 50:.2f}" text-anchor="middle" '
            f'class="block-title">Block {block}</text>'
        )
    svg.append(
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" class="note">'
        'High-frequency and byte counts come from the benchmark compression log; '
        'speedup uses median(cuBLAS TC latency) / median(ZipServ latency).</text>'
    )
    save_svg(svg_path, svg)


def validate_points(
    models: list[str],
    compression: dict[tuple[str, int, str], CompressionPoint],
    speeds: dict[tuple[str, int, str, int], SpeedPoint],
    allow_incomplete: bool,
) -> list[str]:
    valid = []
    for model in models:
        blocks = EXPECTED_BLOCKS[model]
        missing_compression = [
            (block, layer) for block in blocks for layer in LAYERS
            if (model, block, layer) not in compression
        ]
        missing_speeds = [
            (block, layer, batch)
            for block in blocks for layer in LAYERS for batch in (8, 16, 32)
            if (model, block, layer, batch) not in speeds
        ]
        if missing_compression or missing_speeds:
            message = (
                f"{model}: missing compression={missing_compression}, "
                f"speed={missing_speeds}"
            )
            if not allow_incomplete:
                raise ValueError(message)
            print(f"SKIP {message}", file=sys.stderr)
            continue
        valid.append(model)
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, nargs="+",
        help="Real-weight run directories; default: auto-discover completed supported-model runs",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--png-scale", type=float, default=2.0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.png_scale <= 0:
        raise ValueError("--png-scale must be positive")
    roots = args.results_root or discover_result_roots()
    sources = choose_model_block_sources(roots)
    compression = compression_points(sources)
    speeds = speed_points(sources)
    models = validate_points(args.models, compression, speeds, args.allow_incomplete)
    if not models:
        raise RuntimeError("No complete model data was found")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for model in models:
        svg_path = output_dir / f"{model}_real_weight_overview.svg"
        png_path = output_dir / f"{model}_real_weight_overview.png"
        plot_model(
            svg_path, model, EXPECTED_BLOCKS[model], compression, speeds,
        )
        svg_to_png(svg_path, png_path, args.png_scale)
        svg_path.unlink()
        created.append(png_path)
        print(f"Created: {png_path}")
    print(f"Created {len(created)} real-weight overview plot(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
