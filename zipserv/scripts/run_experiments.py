#!/usr/bin/env python3
"""Run and collect the ZipServ synthetic-weight kernel experiment matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
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

    model_ids: set[str] = set()
    for model in config["models"]:
        model_id = model["id"]
        if model_id in model_ids:
            raise ValueError(f"Duplicate model id: {model_id}")
        model_ids.add(model_id)
        layer_ids: set[str] = set()
        for layer in model["layers"]:
            if layer["id"] in layer_ids:
                raise ValueError(f"Duplicate layer id: {model_id}/{layer['id']}")
            layer_ids.add(layer["id"])
            if layer["M"] <= 0 or layer["K"] <= 0:
                raise ValueError(f"Invalid shape: {model_id}/{layer['id']}")
            if layer["M"] % 64 or layer["K"] % 64:
                raise ValueError(
                    f"ZipGEMM requires M and K divisible by 64: "
                    f"{model_id}/{layer['id']}={layer['M']}x{layer['K']}"
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
    known_layers = {layer["id"] for model in enabled for layer in model["layers"]}
    if requested_layers:
        unknown = sorted(requested_layers - known_layers)
        if unknown:
            raise ValueError(f"Unknown layers: {unknown}; available={sorted(known_layers)}")

    shapes = []
    for model in enabled:
        if model["id"] not in model_ids:
            continue
        for layer in model["layers"]:
            if requested_layers is None or layer["id"] in requested_layers:
                shapes.append((model["id"], layer["id"], layer["M"], layer["K"]))
    return shapes


def make_tune_cases(
    config: dict[str, Any], shapes: list[tuple[str, str, int, int]], args: argparse.Namespace
) -> list[Case]:
    batches = select_values(config["matrix"]["batches"], args.batches, "batches")
    splits = select_values(config["matrix"]["split_k_candidates"], args.splits, "splits")
    trials = config["phases"]["tune"]["trials"]
    cases = [
        Case(model, layer, m, k, n, split_k, trial)
        for model, layer, m, k in shapes
        for n in batches
        for split_k in splits
        for trial in range(1, trials + 1)
    ]
    return cases[: args.max_cases] if args.max_cases else cases


def run_checked(command: list[str], env: dict[str, str], timeout: int = 30) -> str:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout or ""
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed ({completed.returncode}):\n{output}")
    return output


def runtime_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    cuda_path = Path(os.environ.get("CUDA_PATH", config["paths"]["cuda_path"]))
    bin_dir = PROJECT_ROOT / "bin"
    env["CUDA_PATH"] = str(cuda_path)
    env["CUDA_HOME"] = str(cuda_path)
    env["PATH"] = f"{cuda_path / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{cuda_path / 'lib64'}"
    env.pop("LD_PRELOAD", None)
    return env


def preflight(config: dict[str, Any], env: dict[str, str], args: argparse.Namespace) -> str:
    host = socket.gethostname().split(".")[0]
    bad_nodes = set(config["gpu"].get("known_bad_nodes", []))
    if host in bad_nodes and not args.allow_known_bad_node:
        raise RuntimeError(
            f"{host} is listed as a known-bad CUDA node. Reallocate with "
            f"--exclude={host}, or pass --allow-known-bad-node only after it is repaired."
        )

    cuda_path = Path(env["CUDA_PATH"])
    device_query = cuda_path / "extras" / "demo_suite" / "deviceQuery"
    if not device_query.is_file():
        raise FileNotFoundError(f"deviceQuery not found: {device_query}")
    output = run_checked([str(device_query)], env, timeout=60)
    if "Result = PASS" not in output:
        raise RuntimeError(f"CUDA deviceQuery did not pass:\n{output}")

    smi = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        env,
        timeout=30,
    ).strip()
    expected = config["gpu"].get("expected_name_substring")
    if expected and expected not in smi:
        raise RuntimeError(f"Expected GPU containing {expected!r}, got: {smi}")
    print(f"CUDA preflight passed on {host}: {smi}", flush=True)
    return output


def ensure_binaries(config: dict[str, Any]) -> None:
    expected = {
        "tune": PROJECT_ROOT / "bin" / "test_mm_tune",
        "final": PROJECT_ROOT / "bin" / "test_mm_final",
        "library": PROJECT_ROOT / "bin" / "libL_API.so",
        "manifest": PROJECT_ROOT / "bin" / "build_manifest.txt",
    }
    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Benchmark binaries are missing. Run scripts/build_benchmarks.sh first:\n"
            + "\n".join(missing)
        )

    manifest: dict[str, str] = {}
    for line in expected["manifest"].read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            manifest[key] = value
    for phase in ("tune", "final"):
        for metric in ("warmup", "repeat"):
            expected_value = str(config["phases"][phase][metric])
            key = f"{phase}_{metric}"
            if manifest.get(key) != expected_value:
                raise RuntimeError(
                    f"Binary manifest has {key}={manifest.get(key)!r}, expected "
                    f"{expected_value}. Re-run scripts/build_benchmarks.sh."
                )


def command_output(command: list[str], env: dict[str, str]) -> str:
    try:
        completed = subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        return (completed.stdout or "").strip()
    except Exception as error:  # environment capture must not abort a run
        return f"ERROR: {error}"


def write_run_manifest(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    env: dict[str, str],
    device_query_output: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    zip_root = Path(os.environ.get("ZIP_ROOT", config["paths"]["zipserv_source"]))
    environment_sections = [
        ("timestamp", utc_now()),
        ("host", socket.gethostname()),
        ("platform", platform.platform()),
        ("slurm_job_id", os.environ.get("SLURM_JOB_ID", "UNSET")),
        ("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")),
        ("nvidia-smi", command_output(["nvidia-smi"], env)),
        ("nvcc --version", command_output([str(Path(env["CUDA_PATH"]) / "bin" / "nvcc"), "--version"], env)),
        ("deviceQuery", device_query_output),
        ("ZipServ commit", command_output(["git", "-C", str(zip_root), "rev-parse", "HEAD"], env)),
        ("ZipServ status", command_output(["git", "-C", str(zip_root), "status", "--short"], env)),
        ("experiment commit", command_output(["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"], env)),
        ("build manifest", (PROJECT_ROOT / "bin" / "build_manifest.txt").read_text(encoding="utf-8")),
    ]
    environment_text = "\n\n".join(f"## {title}\n{body}" for title, body in environment_sections)
    (output_dir / "environment.txt").write_text(environment_text + "\n", encoding="utf-8")

    manifest = {
        "created_at": utc_now(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "command": sys.argv,
        "mode": args.mode,
        "filters": {
            "models": args.models,
            "layers": args.layers,
            "batches": args.batches,
            "splits": args.splits,
            "max_cases": args.max_cases,
        },
        "input": config["input"],
        "matrix": config["matrix"],
        "phases": config["phases"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, output_dir / "experiments.json")


def case_dir(output_dir: Path, phase: str, case: Case) -> Path:
    return (
        output_dir
        / "raw"
        / phase
        / case.model
        / case.layer
        / f"n{case.n}"
        / f"split{case.split_k}"
        / f"trial{case.trial}"
    )


def parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_metrics(csv_path: Path, log_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    kernel_names = {
        "cuBLAS": "cublas",
        "cuBLAS_TC": "cublas_tc",
        "CompGEMM": "zipgemm",
    }
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = kernel_names.get(row["Kernel"])
                if key:
                    metrics[key] = {
                        "latency_ms": float(row["Duration(ms)"]),
                        "tflops": float(row["TFLOPS"]),
                    }

    metrics["compression_ratio"] = parse_float(
        r"^\s*Compression ratio:\s*([0-9.eE+-]+)x\s*$", log_text
    )
    tc_block = re.search(
        r"Triple bitmap vs TC CuBLAS:\s*(.*?)\n\s*========== Performance Results",
        log_text,
        flags=re.DOTALL,
    )
    if tc_block:
        block = tc_block.group(1)
        metrics["zip_vs_cublas_tc"] = {
            "total_absolute_error": parse_float(r"Total absolute error:\s*([0-9.eE+-]+)", block),
            "max_relative_error": parse_float(r"Max relative error:\s*([0-9.eE+-]+)", block),
            "average_relative_error": parse_float(r"Average relative error:\s*([0-9.eE+-]+)", block),
            "significant_error_count": int(
                parse_float(r"Significant error element count:\s*([0-9]+)", block) or 0
            ),
            "significant_error_percent": parse_float(
                r"Significant error element count:.*?\(([0-9.eE+-]+)%\)", block
            ),
        }
    return metrics


def execute_case(
    output_dir: Path,
    phase: str,
    case: Case,
    config: dict[str, Any],
    env: dict[str, str],
    force: bool,
) -> tuple[dict[str, Any], bool, bool]:
    work_dir = case_dir(output_dir, phase, case)
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "result.json"
    replaces_existing = result_path.is_file()
    if result_path.is_file() and not force:
        with result_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("status") == "ok":
            print(f"SKIP {phase} {case.model}/{case.layer} N={case.n} split={case.split_k} trial={case.trial}")
            return existing, False, False

    binary = PROJECT_ROOT / "bin" / f"test_mm_{phase}"
    command = [
        str(binary),
        str(case.m),
        str(case.k),
        str(case.n),
        str(case.split_k),
        "--model",
        case.model,
        "--layer",
        case.layer,
    ]
    # Never parse a CSV left by an earlier failed or interrupted attempt.
    (work_dir / "bf16_triplebm_res.csv").unlink(missing_ok=True)
    started_at = utc_now()
    start = time.monotonic()
    status = "failed"
    error = ""
    returncode: int | None = None
    log_text = ""
    try:
        completed = subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=config["timeout_seconds"],
            check=False,
        )
        returncode = completed.returncode
        log_text = completed.stdout or ""
        if returncode != 0:
            error = f"benchmark exited with status {returncode}"
        elif "========== Test Complete ==========" not in log_text:
            error = "completion marker missing from benchmark output"
        else:
            status = "ok"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        error = f"timed out after {config['timeout_seconds']} seconds"
        stdout = exc.stdout or ""
        log_text = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    (work_dir / "run.log").write_text(log_text, encoding="utf-8")
    temporary_csv = work_dir / "bf16_triplebm_res.csv"
    metrics = parse_metrics(temporary_csv, log_text)
    # test_mm always writes this file. Its rows are copied into the run-level
    # measurements.csv below, so do not retain one CSV for every case.
    temporary_csv.unlink(missing_ok=True)
    required = {"cublas", "cublas_tc", "zipgemm"}
    if status == "ok" and not required.issubset(metrics):
        status = "failed"
        error = f"missing performance rows: {sorted(required - set(metrics))}"

    phase_settings = config["phases"][phase]
    result = {
        "status": status,
        "error": error,
        "phase": phase,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": time.monotonic() - start,
        "returncode": returncode,
        "command": command,
        "model": case.model,
        "layer": case.layer,
        "M": case.m,
        "K": case.k,
        "N": case.n,
        "split_k": case.split_k,
        "trial": case.trial,
        "warmup": phase_settings["warmup"],
        "repeat": phase_settings["repeat"],
        "weight_source": config["input"]["weight_source"],
        "seed": config["input"]["seed"],
        "metrics": metrics,
        "log": str((work_dir / "run.log").relative_to(output_dir)),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if status == "ok":
        tc_ms = metrics["cublas_tc"]["latency_ms"]
        zip_ms = metrics["zipgemm"]["latency_ms"]
        print(
            f"OK   {phase} {case.model}/{case.layer} N={case.n} split={case.split_k} "
            f"trial={case.trial}: Zip={zip_ms:.6f} ms TC={tc_ms:.6f} ms "
            f"speedup={tc_ms / zip_ms:.3f}x",
            flush=True,
        )
    else:
        print(
            f"FAIL {phase} {case.model}/{case.layer} N={case.n} split={case.split_k} "
            f"trial={case.trial}: {error}",
            file=sys.stderr,
            flush=True,
        )
    return result, True, replaces_existing


def load_results(output_dir: Path, phase: str | None = None) -> list[dict[str, Any]]:
    root = output_dir / "raw"
    if phase:
        root = root / phase
    results = []
    if not root.exists():
        return results
    for path in sorted(root.rglob("result.json")):
        with path.open(encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results


def remove_case_csvs(output_dir: Path) -> int:
    """Remove test_mm CSVs after their data has been captured in result.json."""
    raw_root = output_dir / "raw"
    if not raw_root.exists():
        return 0
    paths = list(raw_root.rglob("bf16_triplebm_res.csv"))
    for path in paths:
        path.unlink(missing_ok=True)
    return len(paths)


MEASUREMENT_FIELDS = [
    "phase", "status", "error", "model", "layer", "M", "K", "N", "split_k",
    "trial", "warmup", "repeat", "weight_source", "seed", "kernel", "latency_ms",
    "tflops", "compression_ratio", "total_absolute_error", "max_relative_error",
    "average_relative_error", "significant_error_count", "significant_error_percent",
    "wall_seconds", "log",
]


def measurement_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    base = {key: result.get(key, "") for key in MEASUREMENT_FIELDS}
    error_metrics = result.get("metrics", {}).get("zip_vs_cublas_tc", {})
    base.update(error_metrics)
    base["compression_ratio"] = result.get("metrics", {}).get("compression_ratio", "")
    if result["status"] != "ok":
        return [base]

    rows = []
    # cuBLAS non-TC remains available in result.json, but is outside the
    # experiment's main cuBLAS Tensor Core vs ZipGEMM comparison.
    for kernel in ("cublas_tc", "zipgemm"):
        row = base.copy()
        row["kernel"] = kernel
        row.update(result["metrics"][kernel])
        rows.append(row)
    return rows


def append_measurement(output_dir: Path, result: dict[str, Any]) -> None:
    """Durably append a completed case to the single run-level CSV."""
    path = output_dir / "measurements.csv"
    needs_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(measurement_rows(result))
        handle.flush()
        os.fsync(handle.fileno())


def write_measurements(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in load_results(output_dir):
        rows.extend(measurement_rows(result))

    measurements_path = output_dir / "measurements.csv"
    measurements_tmp = output_dir / ".measurements.csv.tmp"
    with measurements_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(measurements_tmp, measurements_path)

    failure_fields = [
        "phase", "model", "layer", "M", "K", "N", "split_k", "trial", "status", "error", "log"
    ]
    failures = [result for result in load_results(output_dir) if result["status"] != "ok"]
    with (output_dir / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failures)


def choose_split_k(output_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    expected_trials = config["phases"]["tune"]["trials"]
    grouped: dict[tuple[str, str, int, int, int], list[dict[str, Any]]] = {}
    for result in load_results(output_dir, "tune"):
        if result["status"] != "ok":
            continue
        key = (result["model"], result["layer"], result["M"], result["K"], result["N"])
        grouped.setdefault(key, []).append(result)

    selections: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        by_split: dict[int, list[float]] = {}
        for result in group:
            by_split.setdefault(result["split_k"], []).append(
                result["metrics"]["zipgemm"]["latency_ms"]
            )
        candidates = []
        for split_k, values in by_split.items():
            if len(values) == expected_trials and all(math.isfinite(value) and value > 0 for value in values):
                candidates.append((statistics.median(values), split_k, values))
        if not candidates:
            continue
        median_ms, split_k, values = min(candidates, key=lambda item: (item[0], item[1]))
        selections.append(
            {
                "model": key[0],
                "layer": key[1],
                "M": key[2],
                "K": key[3],
                "N": key[4],
                "split_k": split_k,
                "median_zipgemm_ms": median_ms,
                "trial_count": len(values),
                "candidate_count": len(candidates),
            }
        )

    fields = [
        "model", "layer", "M", "K", "N", "split_k", "median_zipgemm_ms",
        "trial_count", "candidate_count",
    ]
    with (output_dir / "selected_splitk.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selections)
    return selections


def read_selections(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "selected_splitk.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run tune first in the same --output-dir, or use --mode all."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_final_cases(
    output_dir: Path,
    config: dict[str, Any],
    shapes: list[tuple[str, str, int, int]],
    args: argparse.Namespace,
) -> list[Case]:
    allowed_shapes = {(model, layer, m, k) for model, layer, m, k in shapes}
    batches = set(select_values(config["matrix"]["batches"], args.batches, "batches"))
    trials = config["phases"]["final"]["trials"]
    cases = []
    for row in read_selections(output_dir):
        shape = (row["model"], row["layer"], int(row["M"]), int(row["K"]))
        if shape not in allowed_shapes or int(row["N"]) not in batches:
            continue
        cases.extend(
            Case(*shape, int(row["N"]), int(row["split_k"]), trial)
            for trial in range(1, trials + 1)
        )
    return cases[: args.max_cases] if args.max_cases else cases


def write_summary(output_dir: Path) -> None:
    grouped: dict[tuple[str, str, int, int, int, int], list[dict[str, Any]]] = {}
    for result in load_results(output_dir, "final"):
        if result["status"] != "ok":
            continue
        key = (
            result["model"], result["layer"], result["M"], result["K"],
            result["N"], result["split_k"],
        )
        grouped.setdefault(key, []).append(result)

    rows = []
    for key, results in sorted(grouped.items()):
        zip_values = [result["metrics"]["zipgemm"]["latency_ms"] for result in results]
        tc_values = [result["metrics"]["cublas_tc"]["latency_ms"] for result in results]
        zip_ms = statistics.median(zip_values)
        tc_ms = statistics.median(tc_values)
        rows.append(
            {
                "model": key[0],
                "layer": key[1],
                "M": key[2],
                "K": key[3],
                "N": key[4],
                "split_k": key[5],
                "trials": len(results),
                "zipgemm_median_ms": zip_ms,
                "cublas_tc_median_ms": tc_ms,
                "speedup_vs_cublas_tc": tc_ms / zip_ms,
                "compression_ratio": statistics.median(
                    result["metrics"]["compression_ratio"] for result in results
                ),
            }
        )
    fields = [
        "model", "layer", "M", "K", "N", "split_k", "trials",
        "zipgemm_median_ms", "cublas_tc_median_ms", "speedup_vs_cublas_tc",
        "compression_ratio",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = os.environ.get("SLURM_JOB_ID", f"interactive-{os.getpid()}")
    return PROJECT_ROOT / "results" / f"run-{timestamp}-{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune Split-K and run the ZipServ ZipGEMM vs cuBLAS experiment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("dry-run", "tune", "final", "all", "collect"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--layers", nargs="+")
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument("--splits", nargs="+", type=int)
    parser.add_argument("--max-cases", type=int, help="Debug aid: run at most this many cases per phase")
    parser.add_argument("--force", action="store_true", help="Re-run successful cases")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-device-check", action="store_true", help="Only for harness debugging")
    parser.add_argument("--allow-known-bad-node", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    shapes = selected_shapes(config, args.models, args.layers)
    tune_cases = make_tune_cases(config, shapes, args)
    final_points = len(shapes) * len(
        select_values(config["matrix"]["batches"], args.batches, "batches")
    )

    print(
        f"Selected {len(shapes)} layer shapes; tune invocations={len(tune_cases)}; "
        f"expected final points={final_points}",
        flush=True,
    )
    if args.mode == "dry-run":
        print("Configuration is valid. No GPU work was started.")
        return 0

    if args.mode in {"final", "collect"} and args.output_dir is None:
        raise ValueError(f"--output-dir is required for --mode {args.mode}")
    output_dir = (args.output_dir or default_output_dir()).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "collect":
        write_measurements(output_dir)
        removed = remove_case_csvs(output_dir)
        if (output_dir / "selected_splitk.csv").is_file():
            write_summary(output_dir)
        print(f"Collected results in {output_dir}; removed {removed} per-case CSV file(s)")
        return 0

    ensure_binaries(config)
    env = runtime_env(config)
    device_query_output = "SKIPPED"
    if not args.skip_device_check:
        device_query_output = preflight(config, env, args)
    if not (output_dir / "manifest.json").exists():
        write_run_manifest(output_dir, config_path, config, args, env, device_query_output)
    # Reconstruct once when starting/resuming. Thereafter each completed case is
    # appended immediately, so a Slurm timeout does not lose completed rows.
    write_measurements(output_dir)
    removed = remove_case_csvs(output_dir)
    if removed:
        print(f"Removed {removed} legacy per-case CSV file(s).", flush=True)
    print(f"Results: {output_dir}", flush=True)

    failures = 0
    if args.mode in {"tune", "all"}:
        for index, case in enumerate(tune_cases, start=1):
            print(f"[{index}/{len(tune_cases)}]", end=" ", flush=True)
            result, executed, replaced = execute_case(
                output_dir, "tune", case, config, env, args.force
            )
            if executed:
                if replaced:
                    write_measurements(output_dir)
                else:
                    append_measurement(output_dir, result)
            if result["status"] != "ok":
                failures += 1
                if args.fail_fast:
                    break
        write_measurements(output_dir)
        selections = choose_split_k(output_dir, config)
        print(f"Selected Split-K for {len(selections)} shape/batch points.", flush=True)

    if args.mode in {"final", "all"} and not (args.fail_fast and failures):
        final_cases = make_final_cases(output_dir, config, shapes, args)
        if not final_cases:
            raise RuntimeError("No complete tuning selections match the requested final filters")
        print(f"Final invocations={len(final_cases)}", flush=True)
        for index, case in enumerate(final_cases, start=1):
            print(f"[{index}/{len(final_cases)}]", end=" ", flush=True)
            result, executed, replaced = execute_case(
                output_dir, "final", case, config, env, args.force
            )
            if executed:
                if replaced:
                    write_measurements(output_dir)
                else:
                    append_measurement(output_dir, result)
            if result["status"] != "ok":
                failures += 1
                if args.fail_fast:
                    break
        write_measurements(output_dir)
        write_summary(output_dir)

    print(f"Completed with {failures} failed invocation(s). Results: {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
