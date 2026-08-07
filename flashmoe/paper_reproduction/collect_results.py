#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    2: {(8192, 32)},
    4: {
        (4096, 32),
        (8192, 32),
        (16384, 8),
        (16384, 16),
        (16384, 32),
        (16384, 64),
        (16384, 128),
    },
    8: {
        (4096, 32),
        (8192, 32),
        (16384, 8),
        (16384, 16),
        (16384, 32),
        (16384, 64),
        (16384, 128),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and combine the independent 2/4/8-GPU jobs."
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        action="append",
        required=True,
        help="Completed job directory; pass once for each of 2, 4 and 8 GPUs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result: {path}")
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    seen_world_sizes: set[int] = set()
    job_metadata: list[dict[str, object]] = []

    for job_dir_arg in args.job_dir:
        job_dir = job_dir_arg.resolve()
        manifest_path = job_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        world_size = int(manifest["world_size"])
        if world_size in seen_world_sizes:
            raise ValueError(f"duplicate {world_size}-GPU job: {job_dir}")
        if world_size not in EXPECTED:
            raise ValueError(f"unexpected world size {world_size}: {job_dir}")

        rows = read_csv(job_dir / "aggregates.csv")
        actual = {
            (int(row["tokens_per_gpu"]), int(row["global_experts"]))
            for row in rows
            if row["status"] == "completed"
        }
        if actual != EXPECTED[world_size]:
            missing = sorted(EXPECTED[world_size] - actual)
            extra = sorted(actual - EXPECTED[world_size])
            raise ValueError(
                f"incomplete {world_size}-GPU job {job_dir}; "
                f"missing={missing}, extra={extra}"
            )
        if (job_dir / "failures.csv").exists():
            raise ValueError(f"job contains failures.csv: {job_dir}")

        for row in rows:
            all_rows.append({"job_dir": str(job_dir), **row})
        seen_world_sizes.add(world_size)
        job_metadata.append(
            {
                "gpu_count": world_size,
                "job_dir": str(job_dir),
                "slurm_job_id": manifest.get("slurm_job_id"),
                "slurm_node_list": manifest.get("slurm_node_list"),
                "cuda_visible_devices": manifest.get("cuda_visible_devices"),
            }
        )

    if seen_world_sizes != {2, 4, 8}:
        raise ValueError(
            f"expected completed 2/4/8-GPU jobs, found {sorted(seen_world_sizes)}"
        )

    all_rows.sort(
        key=lambda row: (
            int(row["gpu_count"]),
            int(row["tokens_per_gpu"]),
            int(row["global_experts"]),
        )
    )
    write_csv(output_dir / "combined_aggregates.csv", all_rows)

    figure8 = [
        row
        for row in all_rows
        if int(row["gpu_count"]) in (4, 8)
        and int(row["global_experts"]) == 32
        and int(row["tokens_per_gpu"]) in (4096, 8192, 16384)
    ]
    figure12 = [
        row
        for row in all_rows
        if int(row["gpu_count"]) in (4, 8)
        and int(row["tokens_per_gpu"]) == 16384
        and int(row["global_experts"]) in (8, 16, 32, 64, 128)
    ]
    scaling = [
        row
        for row in all_rows
        if int(row["tokens_per_gpu"]) == 8192
        and int(row["global_experts"]) == 32
    ]
    latency_2gpu = next(
        float(row["max_rank_mean_latency_ms"])
        for row in scaling
        if int(row["gpu_count"]) == 2
    )
    figure10_11: list[dict[str, object]] = []
    for row in sorted(scaling, key=lambda item: int(item["gpu_count"])):
        latency = float(row["max_rank_mean_latency_ms"])
        figure10_11.append(
            {
                **row,
                "overlap_efficiency_pct": 100.0 * latency_2gpu / latency,
            }
        )

    write_csv(output_dir / "figure8.csv", figure8)
    write_csv(output_dir / "figure10_11.csv", figure10_11)
    write_csv(output_dir / "figure12.csv", figure12)
    (output_dir / "collection_manifest.json").write_text(
        json.dumps(
            {
                "jobs": sorted(job_metadata, key=lambda item: int(item["gpu_count"])),
                "latency_definition": "max_of_rank_mean_latency_ms",
                "measurement_scope": "router_excluded_fused_distributed_moe_kernel",
                "conditions": 15,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"combined={output_dir / 'combined_aggregates.csv'}")
    print(f"figure8={output_dir / 'figure8.csv'}")
    print(f"figure10_11={output_dir / 'figure10_11.csv'}")
    print(f"figure12={output_dir / 'figure12.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

