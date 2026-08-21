#!/usr/bin/env python3
"""Merge per-model result.csv files and derive tuning/performance summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def collect_rows(run_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    paths = sorted((run_dir / "raw").glob("*/result.csv"))
    fields: list[str] = []
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not fields:
                fields = list(reader.fieldnames or [])
            rows.extend(reader)
    rows.sort(key=lambda row: (
        row.get("model", ""), row.get("layer", ""), int(row.get("N", 0)),
        int(row.get("split_k", 0)), int(row.get("trial", 0)),
    ))
    return fields, rows


def select_splitk(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    expected_trials = config["phases"]["tune"]["trials"]
    grouped: dict[tuple[str, str, int, int, int], dict[int, list[float]]] = {}
    for row in rows:
        if row.get("phase") != "tune" or row.get("status") != "ok":
            continue
        key = (row["model"], row["layer"], int(row["M"]), int(row["K"]), int(row["N"]))
        grouped.setdefault(key, {}).setdefault(int(row["split_k"]), []).append(
            float(row["zipgemm_latency_ms"])
        )
    selections = []
    for key, by_split in sorted(grouped.items()):
        candidates = []
        for split_k, values in by_split.items():
            if len(values) == expected_trials and all(math.isfinite(value) and value > 0 for value in values):
                candidates.append((statistics.median(values), split_k, values))
        if not candidates:
            continue
        median_ms, split_k, values = min(candidates, key=lambda item: (item[0], item[1]))
        selections.append({
            "model": key[0], "layer": key[1], "M": key[2], "K": key[3], "N": key[4],
            "split_k": split_k, "median_zipgemm_ms": median_ms,
            "trial_count": len(values), "candidate_count": len(candidates),
        })
    return selections


def performance_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int, int, int], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("phase") != "run" or row.get("status") != "ok":
            continue
        key = (
            row["model"], row["layer"], int(row["M"]), int(row["K"]),
            int(row["N"]), int(row["split_k"]),
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        zip_ms = statistics.median(float(row["zipgemm_latency_ms"]) for row in group)
        tc_ms = statistics.median(float(row["cublas_tc_latency_ms"]) for row in group)
        non_tc_ms = statistics.median(float(row["cublas_latency_ms"]) for row in group)
        ratios = [float(row["compression_ratio"]) for row in group if row.get("compression_ratio")]
        summary.append({
            "model": key[0], "layer": key[1], "M": key[2], "K": key[3],
            "N": key[4], "split_k": key[5], "trials": len(group),
            "cublas_non_tc_median_ms": non_tc_ms, "cublas_tc_median_ms": tc_ms,
            "zipgemm_median_ms": zip_ms,
            "tc_speedup_vs_non_tc": non_tc_ms / tc_ms,
            "zipgemm_speedup_vs_non_tc": non_tc_ms / zip_ms,
            "zipgemm_speedup_vs_tc": tc_ms / zip_ms,
            "compression_ratio": statistics.median(ratios) if ratios else "",
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tune", "run"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    fields, rows = collect_rows(run_dir)
    if not fields:
        raise SystemExit(f"No per-model result.csv files found under {run_dir / 'raw'}")
    write_csv(run_dir / "result_all.csv", fields, rows)

    failures = [row for row in rows if row.get("status") != "ok"]
    write_csv(run_dir / "failures.csv", fields, failures)
    if args.mode == "tune":
        selection_fields = [
            "model", "layer", "M", "K", "N", "split_k", "median_zipgemm_ms",
            "trial_count", "candidate_count",
        ]
        selections = select_splitk(rows, config)
        write_csv(run_dir / "selected_splitk.csv", selection_fields, selections)
        print(f"Collected {len(rows)} tuning rows; selected {len(selections)} Split-K values")
    else:
        summary_fields = [
            "model", "layer", "M", "K", "N", "split_k", "trials",
            "cublas_non_tc_median_ms", "cublas_tc_median_ms", "zipgemm_median_ms",
            "tc_speedup_vs_non_tc", "zipgemm_speedup_vs_non_tc", "zipgemm_speedup_vs_tc",
            "compression_ratio",
        ]
        summary = performance_summary(rows)
        write_csv(run_dir / "summary.csv", summary_fields, summary)
        print(f"Collected {len(rows)} performance rows; summarized {len(summary)} points")
    print(f"Combined result: {run_dir / 'result_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
