#!/usr/bin/env python3
"""Torch Profiler trace를 분석해 Attention/MoE 실험 CSV를 생성한다."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import ijson


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REQUEST_DIR = PROJECT_DIR / "results" / "requests"
DEFAULT_TRACE_DIR = Path("/lustre/hybyun0207/gptoss-profile/torch-traces")
DEFAULT_CSV = PROJECT_DIR / "results" / "experiment.csv"

ATTENTION_RE = re.compile(r"^GPTOSS_ATTENTION_L(?P<layer>\d+)$")
MOE_RE = re.compile(r"^GPTOSS_MOE_L(?P<layer>\d+)$")
EXECUTE_RE = re.compile(
    r"^execute_context_(?P<context_requests>\d+)"
    r"\((?P<context_tokens>\d+)\)_generation_"
    r"(?P<generation_requests>\d+)\((?P<generation_tokens>\d+)\)$"
)

CSV_FIELDS = [
    "experiment_id",
    "timestamp_utc",
    "phase",
    "batch_size",
    "input_tokens_per_sequence",
    "output_tokens_per_sequence",
    "prompt_tokens_total",
    "completion_tokens_total",
    "layer_count",
    "model_forward_count",
    "step_count",
    "profiled_context_tokens",
    "profiled_generation_tokens",
    "analyzed_context_tokens",
    "analyzed_generation_tokens",
    "mixed_step_count",
    "excluded_mixed_context_tokens",
    "excluded_mixed_generation_tokens",
    "end_to_end_latency_ms",
    "phase_gpu_total_ms",
    "gpu_ms_per_step",
    "attention_gpu_total_ms",
    "attention_ms_per_step",
    "attention_percent",
    "moe_gpu_total_ms",
    "moe_ms_per_step",
    "moe_percent",
    "other_gpu_total_ms",
    "other_ms_per_step",
    "other_percent",
    "trace_file",
    "request_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-dir", type=Path, default=DEFAULT_REQUEST_DIR)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--match-window-seconds",
        type=float,
        default=600.0,
        help="기존 metadata와 trace timestamp 매칭 허용 범위 (default: 600).",
    )
    return parser.parse_args()


def trace_events(path: Path) -> Iterator[dict[str, Any]]:
    """대용량 gzip trace를 메모리에 올리지 않고 event 단위로 읽는다."""

    with gzip.open(path, "rb") as file:
        yield from ijson.items(file, "traceEvents.item")


def timestamp_seconds(value: str) -> float:
    """파일명의 UTC timestamp를 Unix time으로 변환한다."""

    parsed = datetime.strptime(value, "%Y%m%dT%H%M%S%fZ")
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def load_requests(request_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    requests = []
    for path in sorted(request_dir.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if data.get("profile_enabled"):
            requests.append((path.resolve(), data))
    return requests


def choose_trace_files(
    requests: list[tuple[Path, dict[str, Any]]],
    trace_dir: Path,
    match_window_seconds: float,
) -> dict[Path, Path]:
    """Metadata의 명시적 경로를 우선하고 과거 결과만 mtime으로 매칭한다."""

    all_traces = sorted(trace_dir.rglob("*.pt.trace.json.gz"))
    unused = {path.resolve() for path in all_traces}
    matches: dict[Path, Path] = {}

    # 새 스크립트로 생성한 결과에는 trace path가 직접 기록된다.
    for request_path, data in requests:
        candidates = [Path(value).resolve() for value in data.get("trace_files", [])]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) == 1:
            matches[request_path] = existing[0]
            unused.discard(existing[0])

    # trace_files가 없던 기존 결과는 요청 저장시각과 trace mtime의 최근접값을 쓴다.
    for request_path, data in requests:
        if request_path in matches:
            continue
        request_time = timestamp_seconds(data["timestamp_utc"])
        candidates = [
            (abs(path.stat().st_mtime - request_time), path) for path in unused
        ]
        if not candidates:
            continue
        distance, trace_path = min(candidates)
        if distance <= match_window_seconds:
            matches[request_path] = trace_path
            unused.remove(trace_path)
    return matches


def event_is_inside_ranges(
    event: tuple[float, float],
    ranges: list[tuple[float, float, int, int]],
) -> bool:
    """GPU annotation이 선택한 scheduler range 하나에 포함되는지 확인한다."""

    event_start, event_duration = event
    event_end = event_start + event_duration
    tolerance_us = 0.01
    return any(
        event_start >= range_start - tolerance_us
        and event_end <= range_start + range_duration + tolerance_us
        for range_start, range_duration, _, _ in ranges
    )


def analyze_trace(path: Path, phase: str) -> dict[str, float | int]:
    """GPU annotation만 사용해 선택 phase의 latency를 집계한다."""

    attention_by_layer: dict[int, list[tuple[float, float]]] = defaultdict(list)
    moe_by_layer: dict[int, list[tuple[float, float]]] = defaultdict(list)
    # (timestamp_us, duration_us, context_tokens, generation_tokens)
    prefill_totals: list[tuple[float, float, int, int]] = []
    decode_totals: list[tuple[float, float, int, int]] = []
    mixed_totals: list[tuple[float, float, int, int]] = []
    profiled_context_tokens = 0
    profiled_generation_tokens = 0

    for event in trace_events(path):
        if event.get("cat") != "gpu_user_annotation" or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        timestamp = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))

        if match := ATTENTION_RE.match(name):
            attention_by_layer[int(match.group("layer"))].append(
                (timestamp, duration)
            )
        elif match := MOE_RE.match(name):
            moe_by_layer[int(match.group("layer"))].append((timestamp, duration))
        elif match := EXECUTE_RE.match(name):
            context_requests = int(match.group("context_requests"))
            context_tokens = int(match.group("context_tokens"))
            generation_requests = int(match.group("generation_requests"))
            generation_tokens = int(match.group("generation_tokens"))
            profiled_context_tokens += context_tokens
            profiled_generation_tokens += generation_tokens

            if context_requests > 0 and generation_requests == 0:
                prefill_totals.append(
                    (timestamp, duration, context_tokens, generation_tokens)
                )
            elif context_requests == 0 and generation_requests > 0:
                decode_totals.append(
                    (timestamp, duration, context_tokens, generation_tokens)
                )
            elif context_requests > 0 and generation_requests > 0:
                mixed_totals.append(
                    (timestamp, duration, context_tokens, generation_tokens)
                )

    layers = sorted(set(attention_by_layer) & set(moe_by_layer))
    if not layers:
        raise ValueError(f"No GPTOSS Attention/MoE GPU annotations in {path}")
    for events in (*attention_by_layer.values(), *moe_by_layer.values()):
        events.sort()

    call_counts = {
        len(events)
        for by_layer in (attention_by_layer, moe_by_layer)
        for layer, events in by_layer.items()
        if layer in layers
    }
    if len(call_counts) != 1:
        raise ValueError(f"Uneven Attention/MoE calls across layers in {path}")
    forward_count = call_counts.pop()

    # batch API 요청은 각 sequence가 scheduler에 도착하는 시점 차이로 한 번의
    # mixed Prefill/Decode forward를 만들 수 있다. block annotation만으로 mixed
    # forward의 두 phase를 분리할 수 없으므로 선택 phase 집계에서는 제외한다.
    if phase == "prefill":
        selected_totals = prefill_totals
        step_count = len(prefill_totals)
    else:
        selected_totals = decode_totals
        step_count = len(decode_totals)

    selected_attention = [
        event
        for layer in layers
        for event in attention_by_layer[layer]
        if event_is_inside_ranges(event, selected_totals)
    ]
    selected_moe = [
        event
        for layer in layers
        for event in moe_by_layer[layer]
        if event_is_inside_ranges(event, selected_totals)
    ]

    if step_count < 1:
        raise ValueError(f"No {phase} steps found in {path}")
    expected_forward_count = len(prefill_totals) + len(decode_totals) + len(
        mixed_totals
    )
    if forward_count != expected_forward_count:
        raise ValueError(
            f"Model-forward mismatch in {path}: annotations={forward_count}, "
            f"scheduler_steps={expected_forward_count}"
        )
    expected_selected_calls = len(layers) * step_count
    if len(selected_attention) != expected_selected_calls:
        raise ValueError(
            f"Selected Attention-call mismatch in {path}: "
            f"annotations={len(selected_attention)}, "
            f"expected={expected_selected_calls}"
        )
    if len(selected_moe) != expected_selected_calls:
        raise ValueError(
            f"Selected MoE-call mismatch in {path}: "
            f"annotations={len(selected_moe)}, expected={expected_selected_calls}"
        )

    # Trace의 ts/dur 단위는 microsecond다.
    attention_us = sum(duration for _, duration in selected_attention)
    moe_us = sum(duration for _, duration in selected_moe)
    total_us = sum(duration for _, duration, _, _ in selected_totals)
    if total_us <= 0:
        raise ValueError(f"Non-positive {phase} GPU duration in {path}")
    other_us = max(total_us - attention_us - moe_us, 0.0)
    return {
        "layer_count": len(layers),
        "model_forward_count": forward_count,
        "step_count": step_count,
        "profiled_context_tokens": profiled_context_tokens,
        "profiled_generation_tokens": profiled_generation_tokens,
        "analyzed_context_tokens": sum(item[2] for item in prefill_totals)
        if phase == "prefill"
        else 0,
        "analyzed_generation_tokens": sum(item[3] for item in decode_totals)
        if phase == "decode"
        else 0,
        "mixed_step_count": len(mixed_totals),
        "excluded_mixed_context_tokens": sum(item[2] for item in mixed_totals),
        "excluded_mixed_generation_tokens": sum(item[3] for item in mixed_totals),
        "phase_gpu_total_ms": total_us / 1000.0,
        "gpu_ms_per_step": total_us / step_count / 1000.0,
        "attention_gpu_total_ms": attention_us / 1000.0,
        "attention_ms_per_step": attention_us / step_count / 1000.0,
        "attention_percent": attention_us / total_us * 100.0,
        "moe_gpu_total_ms": moe_us / 1000.0,
        "moe_ms_per_step": moe_us / step_count / 1000.0,
        "moe_percent": moe_us / total_us * 100.0,
        "other_gpu_total_ms": other_us / 1000.0,
        "other_ms_per_step": other_us / step_count / 1000.0,
        "other_percent": other_us / total_us * 100.0,
    }


def make_row(
    request_path: Path,
    request: dict[str, Any],
    trace_path: Path,
) -> dict[str, Any]:
    metrics = analyze_trace(trace_path, request["phase"])
    usage = request.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if metrics["profiled_context_tokens"] != prompt_tokens:
        raise ValueError(
            f"Profiled context-token mismatch in {trace_path}: "
            f"trace={metrics['profiled_context_tokens']}, request={prompt_tokens}. "
            "Prefix caching is likely enabled; rerun with "
            "--no-enable-prefix-caching."
        )

    expected_generation_tokens = 0
    if request["phase"] == "decode":
        expected_generation_tokens = completion_tokens - request["batch_size"]
    if metrics["profiled_generation_tokens"] != expected_generation_tokens:
        raise ValueError(
            f"Profiled generation-token mismatch in {trace_path}: "
            f"trace={metrics['profiled_generation_tokens']}, "
            f"expected={expected_generation_tokens}"
        )

    row: dict[str, Any] = {
        "experiment_id": request["experiment_id"],
        "timestamp_utc": request["timestamp_utc"],
        "phase": request["phase"],
        "batch_size": request["batch_size"],
        "input_tokens_per_sequence": request["input_len"],
        "output_tokens_per_sequence": request["output_len"],
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "end_to_end_latency_ms": request["client_elapsed_seconds"] * 1000.0,
        "trace_file": str(trace_path),
        "request_file": str(request_path),
    }
    row.update(metrics)
    return row


def format_row(row: dict[str, Any]) -> dict[str, Any]:
    """CSV의 floating-point 값을 읽기 좋은 고정 정밀도로 변환한다."""

    formatted = {}
    for key in CSV_FIELDS:
        value = row.get(key, "")
        if isinstance(value, float):
            value = "" if not math.isfinite(value) else f"{value:.6f}"
        formatted[key] = value
    return formatted


def main() -> int:
    args = parse_args()
    requests = load_requests(args.request_dir.resolve())
    matches = choose_trace_files(
        requests,
        args.trace_dir.resolve(),
        args.match_window_seconds,
    )

    rows = []
    for request_path, request in requests:
        trace_path = matches.get(request_path)
        if trace_path is None:
            print(f"warning: no trace matched {request_path}", file=sys.stderr)
            continue
        rows.append(make_row(request_path, request, trace_path))

    rows.sort(key=lambda row: row["timestamp_utc"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(format_row(row) for row in rows)

    print(f"rows={len(rows)}")
    print(f"csv={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
