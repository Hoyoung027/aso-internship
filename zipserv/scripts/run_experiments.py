#!/usr/bin/env python3
"""Run one model's ZipServ tuning or performance experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments.json"


@dataclass(frozen=True)
class Case:
    model: str
    layer: str
    m: int
    k: int
    n: int
    split_k: int
    trial: int
    block_index: int | None


RESULT_FIELDS = [
    "phase", "status", "error", "started_at", "finished_at", "wall_seconds",
    "returncode", "model", "layer", "M", "K", "N", "split_k", "trial",
    "warmup", "repeat", "weight_source", "weight_model_dir", "block_index",
    "seed", "cublas_latency_ms",
    "cublas_tflops", "cublas_tc_latency_ms", "cublas_tc_tflops",
    "zipgemm_latency_ms", "zipgemm_tflops", "compression_ratio",
    "tc_speedup_vs_non_tc", "zipgemm_speedup_vs_non_tc", "zipgemm_speedup_vs_tc",
    "tc_vs_non_tc_total_absolute_error", "tc_vs_non_tc_max_relative_error",
    "tc_vs_non_tc_average_relative_error", "tc_vs_non_tc_significant_error_count",
    "tc_vs_non_tc_significant_error_percent",
    "zip_vs_non_tc_total_absolute_error", "zip_vs_non_tc_max_relative_error",
    "zip_vs_non_tc_average_relative_error", "zip_vs_non_tc_significant_error_count",
    "zip_vs_non_tc_significant_error_percent",
    "zip_vs_tc_total_absolute_error", "zip_vs_tc_max_relative_error",
    "zip_vs_tc_average_relative_error", "zip_vs_tc_significant_error_count",
    "zip_vs_tc_significant_error_percent", "log_file",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {path}")
    batches = config["matrix"]["batches"]
    splits = config["matrix"]["split_k_candidates"]
    if not batches or not splits or any(value <= 0 for value in batches + splits):
        raise ValueError("Batch sizes and Split-K candidates must be positive")
    seen_models: set[str] = set()
    for model in config["models"]:
        if model["id"] in seen_models:
            raise ValueError(f"Duplicate model id: {model['id']}")
        seen_models.add(model["id"])
        seen_layers: set[str] = set()
        for layer in model["layers"]:
            if layer["id"] in seen_layers:
                raise ValueError(f"Duplicate layer id: {model['id']}/{layer['id']}")
            seen_layers.add(layer["id"])
            if layer["M"] <= 0 or layer["K"] <= 0:
                raise ValueError(f"Invalid shape: {model['id']}/{layer['id']}")
            if layer["M"] % 64 or layer["K"] % 64:
                raise ValueError(
                    f"ZipGEMM requires M and K divisible by 64: "
                    f"{model['id']}/{layer['id']}={layer['M']}x{layer['K']}"
                )
    return config


def select_values(values: Iterable[Any], requested: list[Any] | None, label: str) -> list[Any]:
    available = list(values)
    if requested is None:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; available={available}")
    return [value for value in available if value in requested]


def selected_shapes(
    config: dict[str, Any], models: list[str] | None, layers: list[str] | None
) -> list[tuple[str, str, int, int]]:
    enabled = [model for model in config["models"] if model.get("enabled", True)]
    model_ids = select_values((model["id"] for model in enabled), models, "models")
    requested_layers = set(layers) if layers else None
    shapes = []
    for model in enabled:
        if model["id"] not in model_ids:
            continue
        available_layers = {layer["id"] for layer in model["layers"]}
        if requested_layers:
            unknown = sorted(requested_layers - available_layers)
            if unknown:
                raise ValueError(f"Unknown layers for {model['id']}: {unknown}")
        for layer in model["layers"]:
            if requested_layers is None or layer["id"] in requested_layers:
                shapes.append((model["id"], layer["id"], layer["M"], layer["K"]))
    return shapes


def model_weight_config(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model in config["models"]:
        if model["id"] == model_id:
            weight = model.get("weight")
            if not isinstance(weight, dict):
                raise ValueError(
                    f"Real-weight configuration is missing for model {model_id!r}"
                )
            if not weight.get("model_dir"):
                raise ValueError(f"weight.model_dir is missing for model {model_id!r}")
            return weight
    raise ValueError(f"Model configuration not found: {model_id!r}")


def block_index_for_model(
    config: dict[str, Any], model_id: str, override: int | None,
) -> int | None:
    if "synthetic" in config["input"]["weight_source"].lower():
        if override is not None:
            raise ValueError("--block-index is only valid for real-weight experiments")
        return None

    weight = model_weight_config(config, model_id)
    block_index = int(weight.get("block_index", 0) if override is None else override)
    if block_index < 0:
        raise ValueError(f"Block index must be non-negative: {model_id}={block_index}")

    model_config_path = Path(weight["model_dir"]).expanduser().resolve() / "config.json"
    if model_config_path.is_file():
        with model_config_path.open(encoding="utf-8") as handle:
            num_hidden_layers = json.load(handle).get("num_hidden_layers")
        if num_hidden_layers is not None and block_index >= int(num_hidden_layers):
            raise ValueError(
                f"Block index out of range for {model_id}: {block_index}; "
                f"available=0..{int(num_hidden_layers) - 1}"
            )
    return block_index


def make_tune_cases(
    config: dict[str, Any], shapes: list[tuple[str, str, int, int]], args: argparse.Namespace
) -> list[Case]:
    batches = select_values(config["matrix"]["batches"], args.batches, "batches")
    splits = select_values(config["matrix"]["split_k_candidates"], args.splits, "splits")
    trials = config["phases"]["tune"]["trials"]
    cases = [
        Case(
            model, layer, m, k, n, split_k, trial,
            block_index_for_model(config, model, args.block_index),
        )
        for model, layer, m, k in shapes
        for n in batches
        for split_k in splits
        for trial in range(1, trials + 1)
    ]
    return cases[: args.max_cases] if args.max_cases else cases


def read_selected_splitk(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Split-K tuning result not found: {path}. "
            "Run slurm/run_zipserv.sh --mode tune first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_run_cases(
    config: dict[str, Any], shapes: list[tuple[str, str, int, int]],
    selection_file: Path, args: argparse.Namespace,
) -> list[Case]:
    batches = set(select_values(config["matrix"]["batches"], args.batches, "batches"))
    trials = config["phases"]["final"]["trials"]
    selected: dict[tuple[str, str, int, int, int, int], int] = {}
    unscoped: dict[tuple[str, str, int, int, int], int] = {}
    candidates_by_shape: dict[tuple[str, str, int, int, int], set[int]] = {}
    for row in read_selected_splitk(selection_file):
        shape_key = (
            row["model"], row["layer"], int(row["M"]), int(row["K"]), int(row["N"]),
        )
        split_k = int(row["split_k"])
        candidates_by_shape.setdefault(shape_key, set()).add(split_k)
        block_value = row.get("block_index", "").strip()
        if block_value:
            selected[(*shape_key, int(block_value))] = split_k
        else:
            unscoped[shape_key] = split_k
    cases, missing = [], []
    for model, layer, m, k in shapes:
        block_index = block_index_for_model(config, model, args.block_index)
        for n in sorted(batches):
            shape_key = (model, layer, m, k, n)
            split_k = selected.get((*shape_key, block_index)) if block_index is not None else None
            if split_k is None:
                split_k = unscoped.get(shape_key)
            if split_k is None and len(candidates_by_shape.get(shape_key, set())) == 1:
                split_k = next(iter(candidates_by_shape[shape_key]))
            if split_k is None:
                block_label = "" if block_index is None else f" block={block_index}"
                missing.append(f"{model}/{layer}{block_label} M={m} K={k} N={n}")
                continue
            cases.extend(
                Case(model, layer, m, k, n, split_k, trial, block_index)
                for trial in range(1, trials + 1)
            )
    if missing:
        raise RuntimeError(
            f"{selection_file} is missing {len(missing)} requested tuning selection(s):\n"
            + "\n".join(missing[:20])
        )
    return cases[: args.max_cases] if args.max_cases else cases


def runtime_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    cuda_path = Path(os.environ.get("CUDA_PATH", config["paths"]["cuda_path"]))
    env["CUDA_PATH"] = str(cuda_path)
    env["CUDA_HOME"] = str(cuda_path)
    env["PATH"] = f"{cuda_path / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{PROJECT_ROOT / 'bin'}:{cuda_path / 'lib64'}"
    env.pop("LD_PRELOAD", None)
    return env


def run_checked(command: list[str], env: dict[str, str], timeout: int = 30) -> str:
    completed = subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout, check=False,
    )
    output = completed.stdout or ""
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed ({completed.returncode}):\n{output}")
    return output


def preflight(config: dict[str, Any], env: dict[str, str], args: argparse.Namespace) -> str:
    host = socket.gethostname().split(".")[0]
    if host in set(config["gpu"].get("known_bad_nodes", [])) and not args.allow_known_bad_node:
        raise RuntimeError(f"{host} is a known-bad CUDA node")
    device_query = Path(env["CUDA_PATH"]) / "extras" / "demo_suite" / "deviceQuery"
    if not device_query.is_file():
        raise FileNotFoundError(f"deviceQuery not found: {device_query}")
    output = run_checked([str(device_query)], env, timeout=60)
    if "Result = PASS" not in output:
        raise RuntimeError(f"CUDA deviceQuery did not pass:\n{output}")
    smi = run_checked(
        ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
         "--format=csv,noheader"], env,
    ).strip()
    expected = config["gpu"].get("expected_name_substring")
    if expected and expected not in smi:
        raise RuntimeError(f"Expected GPU containing {expected!r}, got: {smi}")
    print(f"CUDA preflight passed on {host}: {smi}", flush=True)
    return output


def ensure_binaries(config: dict[str, Any]) -> None:
    paths = [
        PROJECT_ROOT / "bin" / "test_mm_tune", PROJECT_ROOT / "bin" / "test_mm_final",
        PROJECT_ROOT / "bin" / "libL_API.so", PROJECT_ROOT / "bin" / "build_manifest.txt",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark binaries:\n" + "\n".join(missing))
    manifest = {}
    for line in paths[-1].read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            manifest[key] = value
    for phase in ("tune", "final"):
        for metric in ("warmup", "repeat"):
            key = f"{phase}_{metric}"
            expected = str(config["phases"][phase][metric])
            if manifest.get(key) != expected:
                raise RuntimeError(f"Binary manifest has {key}={manifest.get(key)!r}, expected {expected}")


def command_output(command: list[str], env: dict[str, str]) -> str:
    try:
        return run_checked(command, env).strip()
    except Exception as error:
        return f"ERROR: {error}"


def write_manifest(
    output_dir: Path, config_path: Path, config: dict[str, Any], args: argparse.Namespace,
    env: dict[str, str], device_query: str,
) -> None:
    zip_root = Path(os.environ.get("ZIP_ROOT", config["paths"]["zipserv_source"]))
    manifest = {
        "created_at": utc_now(), "command": sys.argv, "mode": args.mode,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "host": socket.gethostname(), "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "UNSET"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "zipserv_commit": command_output(["git", "-C", str(zip_root), "rev-parse", "HEAD"], env),
        "device_query": device_query, "input": config["input"],
        "matrix": config["matrix"], "phases": config["phases"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, output_dir / "experiments.json")


def parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_metrics(csv_path: Path, log_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    names = {"cuBLAS": "cublas", "cuBLAS_TC": "cublas_tc", "CompGEMM": "zipgemm"}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = names.get(row["Kernel"])
                if key:
                    metrics[key] = {
                        "latency_ms": float(row["Duration(ms)"]),
                        "tflops": float(row["TFLOPS"]),
                    }
    metrics["compression_ratio"] = parse_float(
        r"^\s*Compression ratio:\s*([0-9.eE+-]+)x\s*$", log_text
    )
    comparisons = (
        (
            "tc_vs_non_tc",
            r"CuBLAS TC vs non-TC comparison results:\s*(.*?)\n\s*(?:==========|Running BF16)",
        ),
        (
            "zip_vs_non_tc",
            r"Triple bitmap vs non-TC CuBLAS:\s*(.*?)\n\s*Triple bitmap vs TC CuBLAS:",
        ),
        (
            "zip_vs_tc",
            r"Triple bitmap vs TC CuBLAS:\s*(.*?)\n\s*(?:Error samples|========== Performance Results)",
        ),
    )
    for prefix, pattern in comparisons:
        match = re.search(pattern, log_text, flags=re.DOTALL)
        if not match:
            continue
        block = match.group(1)
        metrics[f"{prefix}_total_absolute_error"] = parse_float(
            r"Total absolute error:\s*([0-9.eE+-]+)", block
        )
        metrics[f"{prefix}_max_relative_error"] = parse_float(
            r"Max relative error:\s*([0-9.eE+-]+)", block
        )
        metrics[f"{prefix}_average_relative_error"] = parse_float(
            r"Average relative error:\s*([0-9.eE+-]+)", block
        )
        metrics[f"{prefix}_significant_error_count"] = int(
            parse_float(r"Significant error element count:\s*([0-9]+)", block) or 0
        )
        metrics[f"{prefix}_significant_error_percent"] = parse_float(
            r"Significant error element count:.*?\(([0-9.eE+-]+)%\)", block
        )
    return metrics


KEY_FIELDS = (
    "phase", "model", "block_index", "layer", "M", "K", "N", "split_k", "trial",
)


def row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def case_key(phase: str, case: Case) -> tuple[str, ...]:
    block_index = "" if case.block_index is None else case.block_index
    return tuple(map(str, (
        phase, case.model, block_index, case.layer, case.m, case.k,
        case.n, case.split_k, case.trial,
    )))


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_log(path: Path, phase: str, case: Case, status: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n===== CASE START phase={phase} model={case.model} "
            f"block={'' if case.block_index is None else case.block_index} layer={case.layer} "
            f"M={case.m} K={case.k} N={case.n} split_k={case.split_k} trial={case.trial} =====\n"
        )
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.write(f"===== CASE END status={status} =====\n")
        handle.flush()
        os.fsync(handle.fileno())


def execute_case(
    phase: str, case: Case, config: dict[str, Any], env: dict[str, str], log_file: Path,
) -> dict[str, Any]:
    binary_phase = "tune" if phase == "tune" else "final"
    command = [
        str(PROJECT_ROOT / "bin" / f"test_mm_{binary_phase}"),
        str(case.m), str(case.k), str(case.n), str(case.split_k),
        "--model", case.model, "--layer", case.layer,
    ]
    input_config = config["input"]
    command.extend(["--seed", str(input_config["seed"])])
    weight_model_dir = ""
    block_index: Any = "" if case.block_index is None else case.block_index
    if "synthetic" not in input_config["weight_source"].lower():
        weight_config = model_weight_config(config, case.model)
        model_dir = Path(weight_config["model_dir"]).expanduser().resolve()
        index_file = model_dir / "model.safetensors.index.json"
        if not index_file.is_file():
            raise FileNotFoundError(f"Safetensors index not found: {index_file}")
        if case.block_index is None:
            raise ValueError(f"Real-weight case has no block index: {case.model}")
        block_index = case.block_index
        weight_model_dir = str(model_dir)
        command.extend([
            "--model-dir", weight_model_dir,
            "--block-index", str(block_index),
        ])
    started_at, start = utc_now(), time.monotonic()
    status, error, returncode, log_text = "failed", "", None, ""
    metrics: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=f"zipserv-{case.model}-") as work:
        work_dir = Path(work)
        try:
            completed = subprocess.run(
                command, cwd=work_dir, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=config["timeout_seconds"], check=False,
            )
            returncode, log_text = completed.returncode, completed.stdout or ""
            if returncode != 0:
                error = f"benchmark exited with status {returncode}"
            elif "========== Test Complete ==========" not in log_text:
                error = "completion marker missing from benchmark output"
            else:
                status = "ok"
        except subprocess.TimeoutExpired as exc:
            status, error = "timeout", f"timed out after {config['timeout_seconds']} seconds"
            stdout = exc.stdout or ""
            log_text = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        metrics = parse_metrics(work_dir / "bf16_triplebm_res.csv", log_text)
    required = {"cublas", "cublas_tc", "zipgemm"}
    if status == "ok" and not required.issubset(metrics):
        status, error = "failed", f"missing performance rows: {sorted(required - set(metrics))}"
    append_log(log_file, phase, case, status, log_text)

    settings = config["phases"][binary_phase]
    row: dict[str, Any] = {field: "" for field in RESULT_FIELDS}
    row.update({
        "phase": phase, "status": status, "error": error, "started_at": started_at,
        "finished_at": utc_now(), "wall_seconds": time.monotonic() - start,
        "returncode": "" if returncode is None else returncode, "model": case.model,
        "layer": case.layer, "M": case.m, "K": case.k, "N": case.n,
        "split_k": case.split_k, "trial": case.trial, "warmup": settings["warmup"],
        "repeat": settings["repeat"], "weight_source": input_config["weight_source"],
        "weight_model_dir": weight_model_dir, "block_index": block_index,
        "seed": input_config["seed"], "compression_ratio": metrics.get("compression_ratio", ""),
        "log_file": str(log_file),
    })
    for kernel in ("cublas", "cublas_tc", "zipgemm"):
        values = metrics.get(kernel, {})
        row[f"{kernel}_latency_ms"] = values.get("latency_ms", "")
        row[f"{kernel}_tflops"] = values.get("tflops", "")
    if required.issubset(metrics):
        non_tc_ms = metrics["cublas"]["latency_ms"]
        tc_ms = metrics["cublas_tc"]["latency_ms"]
        zip_ms = metrics["zipgemm"]["latency_ms"]
        row["tc_speedup_vs_non_tc"] = non_tc_ms / tc_ms
        row["zipgemm_speedup_vs_non_tc"] = non_tc_ms / zip_ms
        row["zipgemm_speedup_vs_tc"] = tc_ms / zip_ms
    for field in RESULT_FIELDS:
        if field in metrics:
            row[field] = metrics[field]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one model's ZipServ experiment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("dry-run", "tune", "run"), default="run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--layers", nargs="+")
    parser.add_argument("--block-index", type=int, help="Override the configured real-weight block")
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument("--splits", nargs="+", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-device-check", action="store_true")
    parser.add_argument("--allow-known-bad-node", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    shapes = selected_shapes(config, args.models, args.layers)
    if args.mode == "tune":
        cases, phase = make_tune_cases(config, shapes, args), "tune"
    elif args.mode == "run":
        if args.selection_file is None:
            raise ValueError("--selection-file is required for --mode run")
        cases = make_run_cases(config, shapes, args.selection_file.expanduser().resolve(), args)
        phase = "run"
    else:
        tune_count = len(make_tune_cases(config, shapes, args))
        run_points = len(shapes) * len(select_values(config["matrix"]["batches"], args.batches, "batches"))
        print(f"Configuration valid: shapes={len(shapes)} tune_cases={tune_count} run_points={run_points}")
        return 0

    if args.output_dir is None or args.log_file is None:
        raise ValueError("--output-dir and --log-file are required")
    output_dir, log_file = args.output_dir.expanduser().resolve(), args.log_file.expanduser().resolve()
    result_file = output_dir / "result.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_binaries(config)
    env = runtime_env(config)
    device_query = "SKIPPED" if args.skip_device_check else preflight(config, env, args)
    if not (output_dir / "manifest.json").is_file():
        write_manifest(output_dir, config_path, config, args, env, device_query)

    indexed = {row_key(row): row for row in load_rows(result_file)}
    failures = 0
    print(f"Mode={phase}; cases={len(cases)}; result={result_file}; log={log_file}", flush=True)
    for index, case in enumerate(cases, start=1):
        key = case_key(phase, case)
        existing = indexed.get(key)
        if existing and existing.get("status") == "ok" and not args.force:
            block_label = "" if case.block_index is None else f"/block-{case.block_index}"
            print(
                f"[{index}/{len(cases)}] SKIP {case.model}{block_label}/{case.layer} "
                f"N={case.n} split={case.split_k}", flush=True,
            )
            continue
        row = execute_case(phase, case, config, env, log_file)
        indexed[key] = row
        write_rows(result_file, list(indexed.values()))
        if row["status"] == "ok":
            speedup = float(row["cublas_tc_latency_ms"]) / float(row["zipgemm_latency_ms"])
            block_label = "" if case.block_index is None else f"/block-{case.block_index}"
            print(
                f"[{index}/{len(cases)}] OK {case.model}{block_label}/{case.layer} N={case.n} "
                f"split={case.split_k} speedup={speedup:.3f}x", flush=True,
            )
        else:
            failures += 1
            block_label = "" if case.block_index is None else f"/block-{case.block_index}"
            print(
                f"[{index}/{len(cases)}] FAIL {case.model}{block_label}/{case.layer}: {row['error']}",
                file=sys.stderr, flush=True,
            )
            if args.fail_fast:
                break
    print(f"Completed with {failures} failed invocation(s): {result_file}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
