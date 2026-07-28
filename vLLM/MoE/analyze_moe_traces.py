#!/usr/bin/env python3
"""Aggregate layer-0 GPT-OSS router and FusedMoE GPU annotations."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

import ijson


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_DIR = PROJECT_DIR / "results" / "metadata"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "moe_kernel_results.csv"

ROUTER_RE = re.compile(r"^GPTOSS_ROUTER_L0_M(?P<num_tokens>\d+)$")
FUSED_MOE_RE = re.compile(r"^GPTOSS_FUSED_MOE_L0_M(?P<num_tokens>\d+)$")

CSV_FIELDS = [
    "timestamp_utc",
    "slurm_job_id",
    "node",
    "gpu_name",
    "gpu_uuid",
    "driver_version",
    "torch_version",
    "vllm_version",
    "vllm_git_tag",
    "flashinfer_version",
    "experiment_id",
    "backend_requested",
    "backend_actual",
    "activation_dtype",
    "flashinfer_autotune",
    "server_startup_seconds",
    "seed",
    "prompt_sha256",
    "layer",
    "num_tokens",
    "warmup",
    "repeat",
    "router_event_count",
    "fused_moe_event_count",
    "router_gpu_ms_mean",
    "router_gpu_ms_median",
    "router_gpu_ms_min",
    "router_gpu_ms_max",
    "router_gpu_ms_std",
    "router_gpu_ms_p90",
    "fused_moe_gpu_ms_mean",
    "fused_moe_gpu_ms_median",
    "fused_moe_gpu_ms_min",
    "fused_moe_gpu_ms_max",
    "fused_moe_gpu_ms_std",
    "fused_moe_gpu_ms_p90",
    "fused_moe_tokens_per_second",
    "moe_total_gpu_ms_mean",
    "moe_total_gpu_ms_median",
    "router_percent_mean",
    "client_ms_mean",
    "client_ms_median",
    "status",
    "error",
    "trace_files",
    "metadata_file",
    "server_log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write incomplete rows and exit successfully.",
    )
    return parser.parse_args()


def trace_events(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        yield from ijson.items(file, "traceEvents.item")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "std": math.nan,
            "p90": math.nan,
        }
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values),
        "p90": percentile(values, 0.90),
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def scan_traces(
    paths: list[Path],
    expected_num_tokens: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    router: list[tuple[float, float]] = []
    fused_moe: list[tuple[float, float]] = []
    for path in paths:
        for event in trace_events(path):
            if event.get("cat") != "gpu_user_annotation":
                continue
            if event.get("ph") != "X":
                continue
            name = str(event.get("name", ""))
            timestamp = float(event.get("ts", 0.0))
            duration_us = float(event.get("dur", 0.0))
            if match := ROUTER_RE.match(name):
                if int(match.group("num_tokens")) == expected_num_tokens:
                    router.append((timestamp, duration_us / 1000.0))
            elif match := FUSED_MOE_RE.match(name):
                if int(match.group("num_tokens")) == expected_num_tokens:
                    fused_moe.append((timestamp, duration_us / 1000.0))
    router.sort()
    fused_moe.sort()
    return router, fused_moe


def manifest_values(metadata: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(metadata.get("run_manifest", "")))
    if not manifest_path.is_file():
        return {}
    manifest = load_json(manifest_path)
    versions = manifest.get("versions", {})
    gpu = manifest.get("gpu", {})
    return {
        "gpu_name": gpu.get("name", ""),
        "gpu_uuid": gpu.get("uuid", ""),
        "driver_version": gpu.get("driver_version", ""),
        "torch_version": versions.get("torch", ""),
        "vllm_version": versions.get("vllm", ""),
        "vllm_git_tag": versions.get("vllm_git_tag", ""),
        "flashinfer_version": versions.get("flashinfer", ""),
        "server_startup_seconds": manifest.get("server_startup_seconds", ""),
    }


def analyze_metadata(path: Path) -> dict[str, Any]:
    metadata = load_json(path)
    num_tokens = int(metadata["num_tokens"])
    repeat = int(metadata["repeat"])
    trace_paths = [Path(value) for value in metadata.get("trace_files", [])]
    missing = [trace for trace in trace_paths if not trace.is_file()]

    errors: list[str] = []
    if missing:
        errors.append("missing trace files: " + ", ".join(map(str, missing)))
    existing = [trace for trace in trace_paths if trace.is_file()]
    router_events, fused_events = scan_traces(existing, num_tokens)

    if len(router_events) != repeat:
        errors.append(
            f"router events={len(router_events)}, expected={repeat}"
        )
    if len(fused_events) != repeat:
        errors.append(
            f"fused MoE events={len(fused_events)}, expected={repeat}"
        )

    router_ms = [duration for _, duration in router_events]
    fused_ms = [duration for _, duration in fused_events]
    if any(duration <= 0 for duration in router_ms):
        errors.append("router contains non-positive durations")
    if any(duration <= 0 for duration in fused_ms):
        errors.append("FusedMoE contains non-positive durations")
    router_stats = describe(router_ms)
    fused_stats = describe(fused_ms)

    paired_count = min(len(router_ms), len(fused_ms))
    moe_total_ms = [
        router_ms[index] + fused_ms[index] for index in range(paired_count)
    ]
    router_percent = [
        router_ms[index] / moe_total_ms[index] * 100.0
        for index in range(paired_count)
        if moe_total_ms[index] > 0
    ]
    total_stats = describe(moe_total_ms)

    client_ms = [
        float(value) * 1000.0 for value in metadata.get("client_seconds", [])
    ]
    client_stats = describe(client_ms)
    throughput = math.nan
    if fused_stats["median"] > 0:
        throughput = num_tokens / (fused_stats["median"] / 1000.0)

    row: dict[str, Any] = {
        "timestamp_utc": metadata.get("timestamp_utc", ""),
        "slurm_job_id": metadata.get("slurm_job_id", ""),
        "node": metadata.get("host", ""),
        "experiment_id": metadata.get("experiment_id", ""),
        "backend_requested": metadata.get("backend_requested", ""),
        "backend_actual": metadata.get("backend_actual", ""),
        "activation_dtype": metadata.get("activation_dtype", ""),
        "flashinfer_autotune": metadata.get("flashinfer_autotune", ""),
        "seed": metadata.get("seed", ""),
        "prompt_sha256": metadata.get("prompt_sha256", ""),
        "layer": metadata.get("layer", ""),
        "num_tokens": num_tokens,
        "warmup": metadata.get("warmup", ""),
        "repeat": repeat,
        "router_event_count": len(router_events),
        "fused_moe_event_count": len(fused_events),
        "router_gpu_ms_mean": router_stats["mean"],
        "router_gpu_ms_median": router_stats["median"],
        "router_gpu_ms_min": router_stats["min"],
        "router_gpu_ms_max": router_stats["max"],
        "router_gpu_ms_std": router_stats["std"],
        "router_gpu_ms_p90": router_stats["p90"],
        "fused_moe_gpu_ms_mean": fused_stats["mean"],
        "fused_moe_gpu_ms_median": fused_stats["median"],
        "fused_moe_gpu_ms_min": fused_stats["min"],
        "fused_moe_gpu_ms_max": fused_stats["max"],
        "fused_moe_gpu_ms_std": fused_stats["std"],
        "fused_moe_gpu_ms_p90": fused_stats["p90"],
        "fused_moe_tokens_per_second": throughput,
        "moe_total_gpu_ms_mean": total_stats["mean"],
        "moe_total_gpu_ms_median": total_stats["median"],
        "router_percent_mean": (
            statistics.fmean(router_percent) if router_percent else math.nan
        ),
        "client_ms_mean": client_stats["mean"],
        "client_ms_median": client_stats["median"],
        "status": "ok" if not errors else "incomplete",
        "error": "; ".join(errors),
        "trace_files": ";".join(str(trace) for trace in trace_paths),
        "metadata_file": str(path.resolve()),
        "server_log": metadata.get("server_log", ""),
    }
    row.update(manifest_values(metadata))
    return row


def format_row(row: dict[str, Any]) -> dict[str, Any]:
    formatted: dict[str, Any] = {}
    for field in CSV_FIELDS:
        value = row.get(field, "")
        if isinstance(value, float):
            value = "" if not math.isfinite(value) else f"{value:.6f}"
        formatted[field] = value
    return formatted


def add_missing_expected_rows(
    rows: list[dict[str, Any]],
    run_manifest: dict[str, Any],
) -> int:
    experiment_ids = [
        str(value) for value in run_manifest.get("expected_experiment_ids", [])
    ]
    num_tokens_values = [
        int(value) for value in run_manifest.get("expected_num_tokens", [])
    ]
    expected_pairs = {
        (experiment_id, num_tokens)
        for experiment_id in experiment_ids
        for num_tokens in num_tokens_values
    }
    actual_pairs = {
        (str(row["experiment_id"]), int(row["num_tokens"])) for row in rows
    }
    missing_pairs = sorted(expected_pairs - actual_pairs)
    for experiment_id, num_tokens in missing_pairs:
        rows.append(
            {
                "experiment_id": experiment_id,
                "num_tokens": num_tokens,
                "warmup": run_manifest.get("expected_warmup", ""),
                "repeat": run_manifest.get("expected_repeat", ""),
                "status": "missing",
                "error": "missing metadata for expected backend/token combination",
            }
        )
    return len(missing_pairs)


def main() -> int:
    args = parse_args()
    metadata_paths = sorted(args.metadata_dir.rglob("*.json"))
    if not metadata_paths:
        raise ValueError(f"No metadata JSON files in {args.metadata_dir}")

    rows = [analyze_metadata(path) for path in metadata_paths]
    hashes_by_tokens: dict[int, set[str]] = {}
    for row in rows:
        hashes_by_tokens.setdefault(int(row["num_tokens"]), set()).add(
            str(row["prompt_sha256"])
        )
    mismatched_tokens = {
        num_tokens
        for num_tokens, hashes in hashes_by_tokens.items()
        if len(hashes) > 1
    }
    for row in rows:
        if int(row["num_tokens"]) in mismatched_tokens:
            row["status"] = "incomplete"
            mismatch = "prompt hash differs across backends"
            row["error"] = "; ".join(
                value for value in (row["error"], mismatch) if value
            )

    run_manifest_path = args.metadata_dir.parent / "run.json"
    if run_manifest_path.is_file():
        run_manifest = load_json(run_manifest_path)
        add_missing_expected_rows(rows, run_manifest)
    rows.sort(key=lambda row: (row["experiment_id"], row["num_tokens"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(format_row(row) for row in rows)

    incomplete = [row for row in rows if row["status"] != "ok"]
    print(f"rows={len(rows)}")
    print(f"incomplete={len(incomplete)}")
    print(f"csv={args.output.resolve()}")
    if incomplete and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
