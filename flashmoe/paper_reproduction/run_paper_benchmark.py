#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


BINARY_COLUMNS = [
    "rank",
    "dtype",
    "S",
    "H",
    "I",
    "E",
    "k",
    "FlashMoE_Time(ms)",
    "error(%)",
    "EC",
    "GPU",
    "SM",
    "Topology",
    "MLPType",
    "bM",
    "bN0",
    "bK0",
    "bN1",
    "bK1",
    "threads",
    "blocks/SM",
    "SMs",
    "blocks",
    "rtol",
    "atol",
    "graph_launches",
    "warmup",
    "runs",
    "workspace(MiB)",
]


@dataclass(frozen=True)
class Condition:
    experiment_id: str
    tokens_per_gpu: int
    global_experts: int


def conditions_for(world_size: int) -> list[Condition]:
    if world_size == 2:
        return [Condition("figure10_11", 8192, 32)]
    if world_size not in (4, 8):
        raise ValueError("world size must be 2, 4, or 8")

    return [
        Condition("figure8", 4096, 32),
        Condition("figure8_10_11", 8192, 32),
        Condition("figure8_12", 16384, 32),
        Condition("figure12", 16384, 8),
        Condition("figure12", 16384, 16),
        Condition("figure12", 16384, 64),
        Condition("figure12", 16384, 128),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FlashMoE-only paper-protocol FP32 matrix."
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--launcher", default="mpirun")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-error-pct", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stream_command(command: list[str], raw_path: Path) -> tuple[int, list[str]]:
    lines: list[str] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        raw_file.write(f"COMMAND={shlex.join(command)}\n")
        raw_file.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            raw_file.write(line)
            lines.append(line.rstrip("\n"))
        return process.wait(), lines


def parse_rank_rows(lines: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        values = [value.strip() for value in next(csv.reader([stripped]))]
        if len(values) != len(BINARY_COLUMNS):
            continue
        rows.append(dict(zip(BINARY_COLUMNS, values, strict=True)))
    return rows


def validate_rows(
    rows: list[dict[str, str]],
    condition: Condition,
    args: argparse.Namespace,
) -> None:
    if len(rows) != args.world_size:
        raise RuntimeError(
            f"expected {args.world_size} rank rows, found {len(rows)}"
        )

    ranks = {int(row["rank"]) for row in rows}
    if ranks != set(range(args.world_size)):
        raise RuntimeError(f"unexpected ranks: {sorted(ranks)}")

    for row in rows:
        expected = {
            "dtype": "fp32",
            "S": str(condition.tokens_per_gpu),
            "H": str(args.hidden_size),
            "I": str(args.ffn_size),
            "E": str(condition.global_experts),
            "k": str(args.top_k),
            "graph_launches": "0",
            "warmup": str(args.warmup),
            "runs": str(args.runs),
        }
        for key, value in expected.items():
            if row[key] != value:
                raise RuntimeError(
                    f"rank {row['rank']} has {key}={row[key]}, expected {value}"
                )
        if float(row["error(%)"]) > args.max_error_pct:
            raise RuntimeError(
                f"rank {row['rank']} error {row['error(%)']}% exceeds "
                f"{args.max_error_pct}%"
            )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.binary.is_file() and not args.dry_run:
        raise FileNotFoundError(f"benchmark binary not found: {args.binary}")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be nonnegative and runs must be positive")

    result_dir = args.result_dir.resolve()
    if not args.dry_run and (result_dir / "manifest.json").exists():
        raise FileExistsError(
            f"result directory already contains a run: {result_dir}"
        )
    raw_dir = result_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    conditions = conditions_for(args.world_size)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": "router_excluded_fused_distributed_moe_kernel",
        "paper_figures": [8, 10, 11, 12],
        "excluded": [
            "figure9_gpu_utilization",
            "table1_kernel_count",
            "baseline_comparisons",
        ],
        "binary": str(args.binary.resolve()),
        "prepared_source": os.environ.get("FLASHMOE_PREPARED_SOURCE"),
        "flashmoe_commit": os.environ.get("FLASHMOE_COMMIT"),
        "patch_sha256": os.environ.get("FLASHMOE_PATCH_SHA256"),
        "world_size": args.world_size,
        "hidden_size": args.hidden_size,
        "attention_heads_metadata_only": 16,
        "ffn_size": args.ffn_size,
        "top_k": args.top_k,
        "dtype": "fp32",
        "mlp_type": "gated",
        "activation": "silu",
        "capacity_factor": 1.0,
        "seed": args.seed,
        "warmup": args.warmup,
        "runs": args.runs,
        "cuda_graph": False,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_error_pct": args.max_error_pct,
        "conditions": [asdict(condition) for condition in conditions],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rank_results: list[dict[str, object]] = []
    aggregates: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, condition in enumerate(conditions, start=1):
        if condition.global_experts % args.world_size != 0:
            raise RuntimeError(
                f"E={condition.global_experts} is not divisible by "
                f"world={args.world_size}"
            )

        capacity = math.ceil(
            condition.tokens_per_gpu / condition.global_experts
        ) * args.top_k
        condition_name = (
            f"{condition.experiment_id}-g{args.world_size}"
            f"-t{condition.tokens_per_gpu}-e{condition.global_experts}"
        )
        raw_path = raw_dir / f"{condition_name}.log"
        command = [
            args.launcher,
            "-np",
            str(args.world_size),
            "--bind-to",
            "none",
            "--map-by",
            "slot",
            str(args.binary),
            str(condition.tokens_per_gpu),
            str(args.hidden_size),
            str(args.ffn_size),
            str(condition.global_experts),
            str(args.top_k),
            "0",
            str(args.warmup),
            str(args.runs),
            str(args.rtol),
            str(args.atol),
            str(capacity),
            str(args.seed),
        ]

        print(
            f"condition={index}/{len(conditions)} id={condition_name}",
            flush=True,
        )
        print(f"command={shlex.join(command)}", flush=True)
        if args.dry_run:
            continue

        returncode, lines = stream_command(command, raw_path)
        if returncode != 0:
            failures.append(
                {
                    "condition": condition_name,
                    "returncode": returncode,
                    "reason": "benchmark process failed",
                    "raw_log": str(raw_path),
                }
            )
            continue

        try:
            parsed_rows = parse_rank_rows(lines)
            validate_rows(parsed_rows, condition, args)
        except Exception as exc:
            failures.append(
                {
                    "condition": condition_name,
                    "returncode": returncode,
                    "reason": str(exc),
                    "raw_log": str(raw_path),
                }
            )
            continue

        rank_times = [float(row["FlashMoE_Time(ms)"]) for row in parsed_rows]
        max_rank_latency_ms = max(rank_times)
        global_tokens = condition.tokens_per_gpu * args.world_size
        throughput_mtokens_s = global_tokens / (max_rank_latency_ms * 1000.0)
        for row in parsed_rows:
            rank_results.append(
                {
                    "experiment_id": condition.experiment_id,
                    "condition": condition_name,
                    "gpu_count": args.world_size,
                    "rank": int(row["rank"]),
                    "dtype": row["dtype"],
                    "tokens_per_gpu": int(row["S"]),
                    "global_tokens": global_tokens,
                    "hidden_size": int(row["H"]),
                    "ffn_size": int(row["I"]),
                    "global_experts": int(row["E"]),
                    "local_experts": int(row["E"]) // args.world_size,
                    "top_k": int(row["k"]),
                    "capacity_factor": 1.0,
                    "expert_capacity": int(row["EC"]),
                    "rank_mean_latency_ms": float(row["FlashMoE_Time(ms)"]),
                    "max_rank_mean_latency_ms": max_rank_latency_ms,
                    "error_pct": float(row["error(%)"]),
                    "gpu": row["GPU"],
                    "sm": row["SM"],
                    "topology": row["Topology"],
                    "mlp_type": row["MLPType"],
                    "pipeline_stages": 1,
                    "cublasdx_modifier": "generic",
                    "warmup": int(row["warmup"]),
                    "runs": int(row["runs"]),
                    "seed": args.seed,
                    "workspace_mib": float(row["workspace(MiB)"]),
                    "raw_log": str(raw_path),
                }
            )

        aggregates.append(
            {
                "experiment_id": condition.experiment_id,
                "condition": condition_name,
                "gpu_count": args.world_size,
                "tokens_per_gpu": condition.tokens_per_gpu,
                "global_tokens": global_tokens,
                "global_experts": condition.global_experts,
                "local_experts": condition.global_experts // args.world_size,
                "top_k": args.top_k,
                "dtype": "fp32",
                "rank_mean_latency_min_ms": min(rank_times),
                "rank_mean_latency_avg_ms": sum(rank_times) / len(rank_times),
                "max_rank_mean_latency_ms": max_rank_latency_ms,
                "throughput_mtokens_s": throughput_mtokens_s,
                "max_error_pct": max(float(row["error(%)"]) for row in parsed_rows),
                "warmup": args.warmup,
                "runs": args.runs,
                "seed": args.seed,
                "status": "completed",
            }
        )

    rank_fields = [
        "experiment_id",
        "condition",
        "gpu_count",
        "rank",
        "dtype",
        "tokens_per_gpu",
        "global_tokens",
        "hidden_size",
        "ffn_size",
        "global_experts",
        "local_experts",
        "top_k",
        "capacity_factor",
        "expert_capacity",
        "rank_mean_latency_ms",
        "max_rank_mean_latency_ms",
        "error_pct",
        "gpu",
        "sm",
        "topology",
        "mlp_type",
        "pipeline_stages",
        "cublasdx_modifier",
        "warmup",
        "runs",
        "seed",
        "workspace_mib",
        "raw_log",
    ]
    aggregate_fields = [
        "experiment_id",
        "condition",
        "gpu_count",
        "tokens_per_gpu",
        "global_tokens",
        "global_experts",
        "local_experts",
        "top_k",
        "dtype",
        "rank_mean_latency_min_ms",
        "rank_mean_latency_avg_ms",
        "max_rank_mean_latency_ms",
        "throughput_mtokens_s",
        "max_error_pct",
        "warmup",
        "runs",
        "seed",
        "status",
    ]
    write_csv(result_dir / "rank_results.csv", rank_fields, rank_results)
    write_csv(result_dir / "aggregates.csv", aggregate_fields, aggregates)
    if failures:
        write_csv(
            result_dir / "failures.csv",
            ["condition", "returncode", "reason", "raw_log"],
            failures,
        )

    print(f"rank_results={result_dir / 'rank_results.csv'}")
    print(f"aggregates={result_dir / 'aggregates.csv'}")
    print(f"completed_conditions={len(aggregates)}")
    print(f"failed_conditions={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        raise
