#!/usr/bin/env python3
"""GPT-OSS-20B representative-shape GEMM microbenchmark.

The timed region contains only one GEMM call. Tensor generation, quantization,
FlashInfer autotuning, correctness checks, and CSV writes are outside it.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "experiments.yaml"
# FlashInfer includes the BF16 workspace tensor shape in its AutoTuner cache
# key.  A cuDNN tactic can grow the default 32 MiB buffer slightly during
# tuning, causing the persisted key and the later inference key to differ.
# Preallocating a larger, stable buffer in every process avoids that mismatch.
FLASHINFER_BF16_WORKSPACE_BYTES = 64 * 1024 * 1024
RAW_FIELDS = [
    "trial",
    "timestamp_utc",
    "mode",
    "projection",
    "experiment_id",
    "precision",
    "api",
    "M",
    "K",
    "N",
    "iteration",
    "latency_ms",
    "effective_tflops",
    "autotune_enabled",
    "autotune_cache_hit",
    "selected_backend",
    "selected_tactic",
    "max_abs_error",
    "mean_abs_error",
    "all_finite",
    "status",
    "error",
]


@dataclass(frozen=True)
class Projection:
    name: str
    k: int
    n: int


@dataclass
class PreparedCase:
    call: Callable[[], torch.Tensor]
    reference: torch.Tensor
    api: str
    selected_backend: str
    selected_tactic: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("smoke", "torch", "default", "tune", "tuned", "aggregate"),
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=("bf16", "mxfp8", "mxfp4"),
    )
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--m-values", type=int, nargs="+")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--force-retune", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def projections_from_config(config: dict[str, Any]) -> list[Projection]:
    return [
        Projection(name=name, k=int(values["k"]), n=int(values["n"]))
        for name, values in config["projections"].items()
    ]


def stable_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, (base_seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def random_bf16(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)


def quantize_mxfp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create E4M3 values and 128x4-swizzled E8M0 block-32 scales."""
    import flashinfer

    return flashinfer.mxfp8_quantize(
        tensor,
        sf_swizzle_layout=flashinfer.SfLayout.layout_128x4,
    )


def cuda_event_times(
    call: Callable[[], torch.Tensor], warmup: int, repeat: int
) -> tuple[list[float], torch.Tensor]:
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for index in range(repeat):
        starts[index].record()
        output = call()
        ends[index].record()
    ends[-1].synchronize()

    assert output is not None
    return [start.elapsed_time(end) for start, end in zip(starts, ends)], output


def error_metrics(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float, bool]:
    output_f32 = output.float()
    reference_f32 = reference.float()
    difference = (output_f32 - reference_f32).abs()
    all_finite = bool(torch.isfinite(output_f32).all().item())
    return (
        float(difference.max().item()),
        float(difference.mean().item()),
        all_finite,
    )


def cache_path(cache_dir: Path, precision: str, projection: str) -> Path:
    return cache_dir / f"{precision}-{projection}.json"


def load_selected_configs(path: Path) -> dict[int, tuple[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    selected: dict[int, tuple[str, str]] = {}
    for raw_key, value in payload.items():
        if raw_key == "_metadata":
            continue
        try:
            parsed = ast.literal_eval(raw_key)
            profile = parsed[2]
            m_value = int(profile[0][0])
            selected[m_value] = (str(value[0]), json.dumps(value[1], sort_keys=True))
        except (ValueError, TypeError, SyntaxError, IndexError):
            continue
    return selected


def flashinfer_fallback_backend(precision: str) -> str:
    import flashinfer

    function_name = {
        "bf16": "mm_bf16",
        "mxfp8": "mm_mxfp8",
        "mxfp4": "mm_fp4",
    }[precision]
    function = getattr(flashinfer, function_name)
    candidates = getattr(function, "suitable_auto_backends", None)
    if candidates:
        return f"{candidates[0]}(fallback)"
    return "auto(fallback)"


def stabilize_flashinfer_bf16_workspace() -> int:
    """Preallocate the shared BF16 GEMM workspace with a stable cache-key shape."""
    from flashinfer.utils import _get_cache_buf

    device = torch.device("cuda", torch.cuda.current_device())
    workspace = _get_cache_buf(
        "mm_bf16_workspace",
        FLASHINFER_BF16_WORKSPACE_BYTES,
        device,
    )
    if workspace.numel() < FLASHINFER_BF16_WORKSPACE_BYTES:
        raise RuntimeError(
            "Failed to preallocate the FlashInfer BF16 workspace: "
            f"expected at least {FLASHINFER_BF16_WORKSPACE_BYTES} bytes, "
            f"got {workspace.numel()}"
        )
    return int(workspace.numel())


def prepare_torch_case(
    precision: str,
    activation: torch.Tensor,
    weight: torch.Tensor,
    prepared_weight: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> PreparedCase:
    reference = F.linear(activation, weight)
    if precision == "bf16":
        return PreparedCase(
            call=lambda: F.linear(activation, weight),
            reference=reference,
            api="torch.nn.functional.linear",
            selected_backend="pytorch",
            selected_tactic="",
        )
    if precision == "mxfp8":
        activation_q, activation_scale_u8 = quantize_mxfp8(activation)
        if prepared_weight is None:
            weight_q, weight_scale_u8 = quantize_mxfp8(weight)
        else:
            weight_q, weight_scale_u8 = prepared_weight
        weight_q_t = weight_q.t()
        # FlashInfer exposes the E8M0 scale storage as uint8. PyTorch expects
        # the same bits to be viewed as float8_e8m0fnu.
        activation_scale = activation_scale_u8.view(torch.float8_e8m0fnu)
        weight_scale = weight_scale_u8.view(torch.float8_e8m0fnu)
        output = torch.empty(
            (activation.shape[0], weight.shape[0]),
            device="cuda",
            dtype=torch.bfloat16,
        )

        def call() -> torch.Tensor:
            return torch.ops.aten._scaled_mm.out(
                activation_q,
                weight_q_t,
                activation_scale,
                weight_scale,
                out_dtype=torch.bfloat16,
                out=output,
            )

        return PreparedCase(
            call=call,
            reference=reference,
            api="torch.ops.aten._scaled_mm.out",
            selected_backend="pytorch",
            selected_tactic="mxfp8_block32_e8m0_128x4",
        )
    raise ValueError(f"Unsupported PyTorch precision: {precision}")


def prepare_flashinfer_case(
    precision: str,
    activation: torch.Tensor,
    weight: torch.Tensor,
    mode: str,
    selected: tuple[str, str] | None,
    prepared_weight: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> PreparedCase:
    import flashinfer

    reference = F.linear(activation, weight)
    m, k = activation.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    if precision == "bf16":
        weight_t = weight.t()

        def call() -> torch.Tensor:
            return flashinfer.mm_bf16(
                activation,
                weight_t,
                out=output,
                out_dtype=torch.bfloat16,
                backend="auto",
            )

        api = "flashinfer.mm_bf16"
    elif precision == "mxfp8":
        if n < 128:
            raise NotImplementedError(
                f"FlashInfer CUTLASS MXFP8 requires N>=128 on SM120; got N={n}"
            )
        activation_q, activation_scale = quantize_mxfp8(activation)
        if prepared_weight is None:
            weight_q, weight_scale = quantize_mxfp8(weight)
        else:
            weight_q, weight_scale = prepared_weight
        weight_q_t = weight_q.t()

        def call() -> torch.Tensor:
            return flashinfer.mm_mxfp8(
                activation_q,
                weight_q_t,
                activation_scale,
                weight_scale,
                out=output,
                out_dtype=torch.bfloat16,
                use_8x4_sf_layout=False,
                backend="auto",
            )

        api = "flashinfer.mm_mxfp8"
    elif precision == "mxfp4":
        activation_q, activation_scale = flashinfer.mxfp4_quantize(activation)
        if prepared_weight is None:
            weight_q, weight_scale = flashinfer.mxfp4_quantize(weight)
        else:
            weight_q, weight_scale = prepared_weight
        weight_q_t = weight_q.t()
        weight_scale_t = weight_scale.t()
        alpha = torch.ones(1, device="cuda", dtype=torch.float32)

        def call() -> torch.Tensor:
            return flashinfer.mm_fp4(
                activation_q,
                weight_q_t,
                activation_scale,
                weight_scale_t,
                alpha=alpha,
                out_dtype=torch.bfloat16,
                out=output,
                block_size=32,
                use_8x4_sf_layout=False,
                backend="auto",
                use_nvfp4=False,
            )

        api = "flashinfer.mm_fp4(use_nvfp4=False)"
    else:
        raise ValueError(f"Unsupported FlashInfer precision: {precision}")

    if selected is not None:
        backend, tactic = selected
    elif mode == "default":
        backend, tactic = "auto(fallback)", "-1"
    else:
        backend, tactic = "auto", ""

    return PreparedCase(
        call=call,
        reference=reference,
        api=api,
        selected_backend=backend,
        selected_tactic=tactic,
    )


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def status_row(
    *,
    trial: int,
    mode: str,
    projection: Projection,
    experiment_id: str,
    precision: str,
    m_value: int,
    status: str,
    error: str,
) -> dict[str, Any]:
    row = {field: "" for field in RAW_FIELDS}
    row.update(
        {
            "trial": trial,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "projection": projection.name,
            "experiment_id": experiment_id,
            "precision": precision,
            "M": m_value,
            "K": projection.k,
            "N": projection.n,
            "iteration": -1,
            "status": status,
            "error": error,
        }
    )
    return row


def run_measurements(
    *,
    config: dict[str, Any],
    mode: str,
    precision: str,
    trial: int,
    output_path: Path,
    cache_dir: Path,
    m_values: list[int],
    warmup: int,
    repeat: int,
    fail_fast: bool,
) -> None:
    base_seed = int(config["seed"])
    rows: list[dict[str, Any]] = []
    is_torch = mode == "torch"
    experiment_id = (
        f"torch_{precision}"
        if is_torch
        else f"flashinfer_{precision}_{'tuned' if mode == 'tuned' else 'default'}"
    )

    if not is_torch:
        import flashinfer
        from flashinfer.autotuner import AutoTuner, autotune
        if precision == "bf16":
            workspace_bytes = stabilize_flashinfer_bf16_workspace()
            print(
                f"flashinfer_bf16_workspace_bytes={workspace_bytes}",
                flush=True,
            )
    else:
        flashinfer = None
        AutoTuner = None
        autotune = None

    for projection in projections_from_config(config):
        weight = random_bf16(
            (projection.n, projection.k),
            stable_seed(base_seed, projection.name, "weight"),
        )
        selected_configs = (
            load_selected_configs(cache_path(cache_dir, precision, projection.name))
            if mode == "tuned"
            else {}
        )

        if not is_torch and precision == "mxfp8" and projection.n < 128:
            for m_value in m_values:
                rows.append(
                    status_row(
                        trial=trial,
                        mode=mode,
                        projection=projection,
                        experiment_id=experiment_id,
                        precision=precision,
                        m_value=m_value,
                        status="unsupported",
                        error=(
                            "FlashInfer CUTLASS MXFP8 requires N>=128 on SM120; "
                            f"got N={projection.n}"
                        ),
                    )
                )
            continue

        if mode == "tuned" and not cache_path(
            cache_dir, precision, projection.name
        ).is_file():
            for m_value in m_values:
                rows.append(
                    status_row(
                        trial=trial,
                        mode=mode,
                        projection=projection,
                        experiment_id=experiment_id,
                        precision=precision,
                        m_value=m_value,
                        status="unsupported",
                        error="autotune cache is missing",
                    )
                )
            continue

        prepared_weight: tuple[torch.Tensor, torch.Tensor] | None = None
        try:
            if precision == "mxfp8":
                prepared_weight = quantize_mxfp8(weight)
            elif not is_torch and precision == "mxfp4":
                prepared_weight = flashinfer.mxfp4_quantize(weight)
        except (RuntimeError, ValueError, AssertionError, NotImplementedError) as exc:
            for m_value in m_values:
                rows.append(
                    status_row(
                        trial=trial,
                        mode=mode,
                        projection=projection,
                        experiment_id=experiment_id,
                        precision=precision,
                        m_value=m_value,
                        status="error",
                        error=f"weight preparation failed: {type(exc).__name__}: {exc}",
                    )
                )
            if fail_fast:
                raise
            continue

        if is_torch:
            context = nullcontext()
        elif mode == "tuned":
            context = autotune(
                False,
                cache=str(cache_path(cache_dir, precision, projection.name)),
                tuning_buckets=tuple(m_values),
            )
        else:
            AutoTuner.get().clear_cache()
            context = autotune(False, tuning_buckets=tuple(m_values))

        with context:
            for m_value in m_values:
                activation = random_bf16(
                    (m_value, projection.k),
                    stable_seed(base_seed, projection.name, m_value, "activation"),
                )
                try:
                    if is_torch:
                        prepared = prepare_torch_case(
                            precision, activation, weight, prepared_weight
                        )
                    else:
                        prepared = prepare_flashinfer_case(
                            precision,
                            activation,
                            weight,
                            mode,
                            selected_configs.get(m_value),
                            prepared_weight,
                        )
                    latencies, output = cuda_event_times(prepared.call, warmup, repeat)
                    if mode == "default":
                        prepared.selected_backend = flashinfer_fallback_backend(
                            precision
                        )
                    max_error, mean_error, all_finite = error_metrics(
                        output, prepared.reference
                    )
                    for iteration, latency_ms in enumerate(latencies):
                        tflops = (
                            2.0
                            * m_value
                            * projection.k
                            * projection.n
                            / (latency_ms / 1000.0)
                            / 1.0e12
                        )
                        rows.append(
                            {
                                "trial": trial,
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "mode": mode,
                                "projection": projection.name,
                                "experiment_id": experiment_id,
                                "precision": precision,
                                "api": prepared.api,
                                "M": m_value,
                                "K": projection.k,
                                "N": projection.n,
                                "iteration": iteration,
                                "latency_ms": f"{latency_ms:.9f}",
                                "effective_tflops": f"{tflops:.9f}",
                                # Both measurement modes run with tune_mode=False.
                                # "tuned" means a persisted tactic cache is loaded.
                                "autotune_enabled": False,
                                "autotune_cache_hit": bool(
                                    mode == "tuned" and m_value in selected_configs
                                ),
                                "selected_backend": prepared.selected_backend,
                                "selected_tactic": prepared.selected_tactic,
                                "max_abs_error": f"{max_error:.9g}",
                                "mean_abs_error": f"{mean_error:.9g}",
                                "all_finite": all_finite,
                                "status": "ok",
                                "error": "",
                            }
                        )
                except (RuntimeError, ValueError, AssertionError, NotImplementedError) as exc:
                    status = "unsupported" if isinstance(exc, NotImplementedError) else "error"
                    rows.append(
                        status_row(
                            trial=trial,
                            mode=mode,
                            projection=projection,
                            experiment_id=experiment_id,
                            precision=precision,
                            m_value=m_value,
                            status=status,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    print(
                        f"[{status}] {experiment_id} {projection.name} M={m_value}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if fail_fast:
                        raise
                    with torch.cuda.device(0):
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                        torch.cuda.empty_cache()

    write_rows(output_path, rows)
    ok_rows = sum(row.get("status") == "ok" for row in rows)
    print(f"raw_csv={output_path} ok_rows={ok_rows} total_rows={len(rows)}")


def tune_flashinfer(
    *,
    config: dict[str, Any],
    precision: str,
    output_dir: Path,
    cache_dir: Path,
    m_values: list[int],
    force_retune: bool,
    fail_fast: bool,
) -> None:
    import flashinfer
    from flashinfer.autotuner import AutoTuner, autotune

    if precision == "bf16":
        workspace_bytes = stabilize_flashinfer_bf16_workspace()
        print(
            f"flashinfer_bf16_workspace_bytes={workspace_bytes}",
            flush=True,
        )

    base_seed = int(config["seed"])
    summaries: list[dict[str, Any]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    for projection in projections_from_config(config):
        path = cache_path(cache_dir, precision, projection.name)
        if force_retune and path.exists():
            path.unlink()

        if precision == "mxfp8" and projection.n < 128:
            summaries.append(
                {
                    "projection": projection.name,
                    "precision": precision,
                    "status": "unsupported",
                    "error": f"FlashInfer CUTLASS MXFP8 requires N>=128; got N={projection.n}",
                }
            )
            continue

        weight = random_bf16(
            (projection.n, projection.k),
            stable_seed(base_seed, projection.name, "weight"),
        )
        activation = random_bf16(
            (max(m_values), projection.k),
            stable_seed(base_seed, projection.name, max(m_values), "activation"),
        )
        tuner = AutoTuner.get()
        tuner.clear_cache()
        tuner.reset_statistics()
        started: float | None = None
        try:
            prepared = prepare_flashinfer_case(
                precision, activation, weight, "tune", selected=None
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            with autotune(
                True,
                cache=str(path),
                tuning_buckets=tuple(m_values),
            ):
                prepared.call()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            selected = load_selected_configs(path)
            summaries.append(
                {
                    "projection": projection.name,
                    "precision": precision,
                    "status": "ok",
                    "cache": str(path),
                    "autotune_seconds": elapsed,
                    "selected_config_count": len(selected),
                    "selected_configs": {
                        str(m): {"runner": value[0], "tactic": value[1]}
                        for m, value in sorted(selected.items())
                    },
                    "statistics": str(tuner.stats),
                }
            )
        except (RuntimeError, ValueError, AssertionError, NotImplementedError) as exc:
            summaries.append(
                {
                    "projection": projection.name,
                    "precision": precision,
                    "status": "error",
                    "cache": str(path),
                    "autotune_seconds": (
                        time.perf_counter() - started if started is not None else 0.0
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                    "statistics": str(tuner.stats),
                }
            )
            print(
                f"[tune error] {precision} {projection.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if fail_fast:
                raise

    output_path = output_dir / f"tuning-{precision}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    print(f"tuning_summary={output_path}")


def aggregate_results(output_dir: Path) -> None:
    raw_all_path = output_dir / "raw_all.csv"
    temporary_raw_paths = sorted(output_dir.glob("raw-*.csv"))
    measurement_paths = [
        path
        for path in temporary_raw_paths
        if not path.name.startswith("raw-smoke-")
    ]

    all_rows: list[dict[str, str]] = []
    if measurement_paths:
        for path in measurement_paths:
            with path.open(newline="", encoding="utf-8") as handle:
                all_rows.extend(csv.DictReader(handle))
        write_rows(raw_all_path, all_rows)
        print(f"raw_all_csv={raw_all_path} rows={len(all_rows)}")
    elif raw_all_path.is_file():
        with raw_all_path.open(newline="", encoding="utf-8") as handle:
            all_rows.extend(csv.DictReader(handle))
        print(f"raw_all_csv={raw_all_path} rows={len(all_rows)} reused=true")
    else:
        raise FileNotFoundError(
            f"No measurement CSV files found in {output_dir}"
        )

    rows = [row for row in all_rows if row["status"] == "ok"]

    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["projection"],
            row["experiment_id"],
            row["precision"],
            row["M"],
            row["K"],
            row["N"],
        )
        groups.setdefault(key, []).append(row)

    summary_fields = [
        "projection",
        "experiment_id",
        "precision",
        "M",
        "K",
        "N",
        "samples",
        "mean_ms",
        "median_ms",
        "std_ms",
        "min_ms",
        "p95_ms",
        "mean_tflops",
        "max_abs_error",
        "mean_abs_error",
        "selected_backends",
        "selected_tactics",
    ]
    output_path = output_dir / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            latencies = [float(row["latency_ms"]) for row in group]
            tflops = [float(row["effective_tflops"]) for row in group]
            writer.writerow(
                {
                    "projection": key[0],
                    "experiment_id": key[1],
                    "precision": key[2],
                    "M": key[3],
                    "K": key[4],
                    "N": key[5],
                    "samples": len(latencies),
                    "mean_ms": f"{statistics.fmean(latencies):.9f}",
                    "median_ms": f"{statistics.median(latencies):.9f}",
                    "std_ms": f"{statistics.pstdev(latencies):.9f}",
                    "min_ms": f"{min(latencies):.9f}",
                    "p95_ms": f"{float(np.percentile(latencies, 95)):.9f}",
                    "mean_tflops": f"{statistics.fmean(tflops):.9f}",
                    "max_abs_error": max(float(row["max_abs_error"]) for row in group),
                    "mean_abs_error": statistics.fmean(
                        float(row["mean_abs_error"]) for row in group
                    ),
                    "selected_backends": "|".join(
                        sorted({row["selected_backend"] for row in group})
                    ),
                    "selected_tactics": "|".join(
                        sorted({row["selected_tactic"] for row in group})
                    ),
                }
            )
    print(f"summary_csv={output_path} groups={len(groups)}")

    for path in temporary_raw_paths:
        path.unlink()
    print(f"temporary_raw_csv_removed={len(temporary_raw_paths)}")


def validate_gpu_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f"Expected RTX PRO 6000 SM120, got capability={capability}")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected PyTorch CUDA 13.0, got {torch.version.cuda}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "autotune-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "aggregate":
        aggregate_results(args.output_dir)
        return

    validate_gpu_environment()
    m_values = args.m_values or [int(value) for value in config["m_values"]]
    warmup = int(args.warmup if args.warmup is not None else config["warmup"])
    repeat = int(args.repeat if args.repeat is not None else config["repeat"])
    if warmup < 0 or repeat <= 0 or any(value <= 0 for value in m_values):
        raise ValueError("warmup must be >=0 and repeat/M values must be positive")

    print(
        json.dumps(
            {
                "mode": args.mode,
                "precision": args.precision,
                "trial": args.trial,
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "m_values": m_values,
                "warmup": warmup,
                "repeat": repeat,
                "cache_dir": str(cache_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.mode == "smoke":
        output_path = args.output_dir / "raw-smoke-bf16-trial0.csv"
        run_measurements(
            config=config,
            mode="default",
            precision="bf16",
            trial=0,
            output_path=output_path,
            cache_dir=cache_dir,
            m_values=m_values[:1],
            warmup=min(warmup, 2),
            repeat=min(repeat, 3),
            fail_fast=True,
        )
        return

    if args.mode == "tune":
        if args.precision not in {"bf16", "mxfp8", "mxfp4"}:
            raise ValueError("FlashInfer tuning supports bf16, mxfp8, or mxfp4")
        tune_flashinfer(
            config=config,
            precision=args.precision,
            output_dir=args.output_dir,
            cache_dir=cache_dir,
            m_values=m_values,
            force_retune=args.force_retune,
            fail_fast=args.fail_fast,
        )
        return

    if args.mode == "torch" and args.precision not in {"bf16", "mxfp8"}:
        raise ValueError("PyTorch mode supports bf16 or mxfp8")
    if args.mode in {"default", "tuned"} and args.precision not in {
        "bf16",
        "mxfp8",
        "mxfp4",
    }:
        raise ValueError("FlashInfer mode supports bf16, mxfp8, or mxfp4")

    output_path = (
        args.output_dir
        / f"raw-{args.mode}-{args.precision}-trial{args.trial}.csv"
    )
    run_measurements(
        config=config,
        mode=args.mode,
        precision=args.precision,
        trial=args.trial,
        output_path=output_path,
        cache_dir=cache_dir,
        m_values=m_values,
        warmup=warmup,
        repeat=repeat,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
