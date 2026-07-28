#!/usr/bin/env python3
"""Restart vLLM per MoE backend and run the configured token sweep."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "experiments.yaml"
EXPECTED_VLLM_VERSION = "0.23.0"
INSTRUMENTATION_MARKER = "GPTOSS_FUSED_MOE_L0_M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=os.environ.get("MODEL_PATH"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=os.environ.get("RUN_ROOT"),
    )
    parser.add_argument(
        "--vllm-source",
        type=Path,
        default=Path("/lustre/hybyun0207/vllm023-moe/src/vllm"),
    )
    parser.add_argument(
        "--vllm-bin",
        type=Path,
        default=shutil.which("vllm"),
    )
    parser.add_argument(
        "--backends",
        help="Comma-separated experiment IDs. Default: all configured backends.",
    )
    parser.add_argument(
        "--num-tokens",
        help="Comma-separated token counts overriding the YAML sweep.",
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML object in {path}")
    return data


def parse_num_tokens(value: str) -> list[int]:
    tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not tokens or any(token < 1 for token in tokens):
        raise ValueError("num_tokens must contain positive integers")
    return tokens


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def verify_environment(vllm_source: Path) -> dict[str, str]:
    vllm_version = package_version("vllm")
    if vllm_version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"Expected vLLM {EXPECTED_VLLM_VERSION}, got {vllm_version}"
        )

    import vllm.model_executor.models.gpt_oss as gpt_oss

    source_path = Path(inspect.getfile(gpt_oss)).resolve()
    source_text = source_path.read_text(encoding="utf-8")
    if INSTRUMENTATION_MARKER not in source_text:
        raise RuntimeError(
            "The imported GPT-OSS source is not instrumented: "
            f"{source_path}. Run setup_vllm023_editable.sh on the login node."
        )

    import vllm._C  # noqa: F401
    import vllm._moe_C  # noqa: F401

    git_tag = command_output(
        ["git", "-C", str(vllm_source), "describe", "--tags", "--always"]
    )
    if not git_tag.startswith("v0.23.0"):
        raise RuntimeError(f"Expected v0.23.0 source, got {git_tag}")
    return {
        "vllm": vllm_version,
        "torch": package_version("torch"),
        "flashinfer": package_version("flashinfer-python"),
        "vllm_git_tag": git_tag,
        "gpt_oss_source": str(source_path),
    }


def gpu_metadata() -> dict[str, str]:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    first = output.splitlines()[0]
    values = [value.strip() for value in first.split(",")]
    keys = ["name", "uuid", "driver_version", "memory_total_mib", "power_limit_w"]
    return dict(zip(keys, values, strict=False))


def health_ready(url: str, timeout: float) -> bool:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError):
        return False


def wait_for_server(
    process: subprocess.Popen[Any],
    health_url: str,
    timeout_seconds: float,
) -> float:
    start = time.monotonic()
    deadline = start + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"vLLM exited before health check with code {return_code}"
            )
        if health_ready(health_url, timeout=2.0):
            return time.monotonic() - start
        time.sleep(1.0)
    raise TimeoutError(f"vLLM health check timed out after {timeout_seconds}s")


def stop_server(
    process: subprocess.Popen[Any],
    timeout_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10.0)


def server_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    backend: dict[str, Any],
    trace_dir: Path,
) -> list[str]:
    server = config["server"]
    command = [
        str(args.vllm_bin),
        "serve",
        str(args.model_path),
        "--served-model-name",
        str(server["served_model_name"]),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--dtype",
        "auto",
        "--max-model-len",
        str(server["max_model_len"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(server["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--attention-backend",
        str(server["attention_backend"]),
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "-O0",
        "--profiler-config.profiler",
        "torch",
        "--profiler-config.torch_profiler_dir",
        str(trace_dir),
        "--profiler-config.torch_profiler_with_stack=false",
        "--profiler-config.torch_profiler_record_shapes=false",
        "--profiler-config.torch_profiler_with_memory=false",
        "--profiler-config.ignore_frontend=true",
    ]
    command.extend(str(value) for value in backend["server_args"])
    return command


def request_command(
    config: dict[str, Any],
    backend: dict[str, Any],
    backend_actual: str,
    num_tokens: int,
    trace_dir: Path,
    metadata_dir: Path,
    server_log: Path,
    manifest_path: Path,
) -> list[str]:
    common = config["common"]
    server = config["server"]
    command = [
        sys.executable,
        str(PROJECT_DIR / "run_moe_requests.py"),
        "--num-tokens",
        str(num_tokens),
        "--warmup",
        str(common["warmup"]),
        "--repeat",
        str(common["repeat"]),
        "--seed",
        str(common["seed"]),
        "--master-num-tokens",
        str(common["master_num_tokens"]),
        "--prompt-token-min",
        str(common["prompt_token_min"]),
        "--prompt-token-max",
        str(common["prompt_token_max"]),
        "--base-url",
        f"http://{server['host']}:{server['port']}",
        "--model",
        str(server["served_model_name"]),
        "--timeout",
        str(common["request_timeout_seconds"]),
        "--trace-dir",
        str(trace_dir),
        "--metadata-dir",
        str(metadata_dir),
        "--experiment-id",
        str(backend["id"]),
        "--backend-requested",
        str(backend["requested_backend"]),
        "--backend-actual",
        backend_actual,
        "--activation-dtype",
        str(backend["activation_dtype"]),
        "--server-log",
        str(server_log),
        "--run-manifest",
        str(manifest_path),
    ]
    if backend["flashinfer_autotune"]:
        command.append("--flashinfer-autotune")
    else:
        command.append("--no-flashinfer-autotune")
    return command


def expected_backend(backend: dict[str, Any]) -> str:
    value = str(backend.get("expected_backend", "")).strip()
    if not value:
        raise ValueError(
            f"Backend {backend.get('id', '<unknown>')} has no expected_backend"
        )
    return value


def validate_backend_selection(
    backend: dict[str, Any],
    server_log: Path,
) -> str:
    """Verify the implementation selected by vLLM before collecting traces."""
    actual = expected_backend(backend)
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    backend_marker = f"Using '{actual}' Mxfp4 MoE backend."
    if backend_marker not in log_text:
        raise RuntimeError(
            f"Expected backend marker not found in {server_log}: "
            f"{backend_marker}"
        )

    if backend["requested_backend"] == "humming":
        configured = str(
            backend.get("environment", {}).get(
                "VLLM_HUMMING_MOE_GEMM_TYPE", "indexed"
            )
        ).lower()
        gemm_type = (
            "grouped_contiguous"
            if configured in {"grouped", "grouped_contiguous"}
            else "indexed"
        )
        gemm_marker = f"Using {gemm_type} gemm for humming moe"
        if gemm_marker not in log_text:
            raise RuntimeError(
                f"Expected Humming GEMM marker not found in {server_log}: "
                f"{gemm_marker}"
            )

    if backend["requested_backend"] == "flashinfer_cutlass":
        if backend["flashinfer_autotune"]:
            required_markers = (
                "[Autotuner]: Autotuning process starts",
                "[Autotuner]: Autotuning process ends",
            )
        else:
            required_markers = (
                "Skipping FlashInfer autotune because it is disabled.",
            )
        missing = [marker for marker in required_markers if marker not in log_text]
        if missing:
            raise RuntimeError(
                "FlashInfer AutoTuner state could not be verified in "
                f"{server_log}; missing: {missing}"
            )

    return actual


def select_backends(
    configured: list[dict[str, Any]],
    requested: str | None,
) -> list[dict[str, Any]]:
    if not requested:
        return configured
    wanted = {value.strip() for value in requested.split(",") if value.strip()}
    selected = [backend for backend in configured if backend["id"] in wanted]
    found = {backend["id"] for backend in selected}
    missing = wanted - found
    if missing:
        raise ValueError(f"Unknown backend experiment IDs: {sorted(missing)}")
    return selected


def verify_backend_dependencies(backends: list[dict[str, Any]]) -> None:
    requested = {str(backend["requested_backend"]) for backend in backends}
    if requested & {"humming", "flashinfer_cutlass"}:
        ninja = shutil.which("ninja")
        if ninja is None:
            raise RuntimeError(
                "Humming and FlashInfer require the ninja executable. "
                "Add $VLLM_ENV/bin to PATH or run setup_vllm023_editable.sh."
            )
    if "emulation" in requested:
        try:
            importlib.import_module("quark.torch.kernel.mx")
        except ImportError as exc:
            raise RuntimeError(
                "The emulation backend requires a working amd-quark install. "
                "Run setup_vllm023_editable.sh on the login node."
            ) from exc


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_backend(
    args: argparse.Namespace,
    config: dict[str, Any],
    backend: dict[str, Any],
    base_manifest: dict[str, Any],
) -> bool:
    backend_id = str(backend["id"])
    backend_root = args.run_root / backend_id
    trace_dir = backend_root / "traces"
    metadata_dir = args.run_root / "metadata" / backend_id
    log_dir = backend_root / "logs"
    manifest_path = args.run_root / "manifests" / f"{backend_id}.json"
    server_log = log_dir / "server.log"
    for directory in (trace_dir, metadata_dir, log_dir, manifest_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    command = server_command(args, config, backend, trace_dir)
    print(f"server[{backend_id}]={shlex.join(command)}", flush=True)
    if args.dry_run:
        backend_actual = expected_backend(backend)
        sample = request_command(
            config,
            backend,
            backend_actual,
            int(config["common"]["num_tokens"][0]),
            trace_dir,
            metadata_dir,
            server_log,
            manifest_path,
        )
        print(f"request[{backend_id}]={shlex.join(sample)}", flush=True)
        return True

    environment = os.environ.copy()
    environment["GPTOSS_MOE_PROFILE_LAYER0"] = "1"
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    environment["PYTHONUNBUFFERED"] = "1"
    environment.pop("VLLM_HUMMING_MOE_GEMM_TYPE", None)
    for key, value in backend.get("environment", {}).items():
        environment[str(key)] = str(value)

    manifest = dict(base_manifest)
    manifest.update(
        {
            "experiment_id": backend_id,
            "backend_requested": backend["requested_backend"],
            "backend_expected": expected_backend(backend),
            "activation_dtype": backend["activation_dtype"],
            "flashinfer_autotune": backend["flashinfer_autotune"],
            "server_command": command,
            "server_log": str(server_log.resolve()),
            "status": "starting",
            "error": "",
        }
    )
    write_json(manifest_path, manifest)

    process: subprocess.Popen[Any] | None = None
    success = True
    try:
        with server_log.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            startup_seconds = wait_for_server(
                process,
                f"http://{config['server']['host']}:{config['server']['port']}/health",
                float(config["server"]["startup_timeout_seconds"]),
            )
            backend_actual = validate_backend_selection(backend, server_log)
            manifest["server_startup_seconds"] = startup_seconds
            manifest["backend_actual"] = backend_actual
            manifest["status"] = "running"
            write_json(manifest_path, manifest)
            print(
                f"server[{backend_id}]=ready "
                f"startup_seconds={startup_seconds:.3f}",
                flush=True,
            )

            for num_tokens in config["common"]["num_tokens"]:
                if process.poll() is not None:
                    raise RuntimeError(
                        "vLLM exited during the token sweep with code "
                        f"{process.returncode}"
                    )
                request = request_command(
                    config,
                    backend,
                    backend_actual,
                    int(num_tokens),
                    trace_dir,
                    metadata_dir,
                    server_log,
                    manifest_path,
                )
                print(
                    f"run[{backend_id}].num_tokens={num_tokens}",
                    flush=True,
                )
                completed = subprocess.run(request, env=environment, check=False)
                if completed.returncode != 0:
                    success = False
                    manifest["status"] = "request_failed"
                    manifest["error"] = (
                        f"num_tokens={num_tokens} returned "
                        f"{completed.returncode}"
                    )
                    write_json(manifest_path, manifest)
                    if args.fail_fast:
                        break
    except (OSError, RuntimeError, TimeoutError) as exc:
        success = False
        manifest["status"] = "server_failed"
        manifest["error"] = str(exc)
        write_json(manifest_path, manifest)
        print(f"error[{backend_id}]={exc}", file=sys.stderr, flush=True)
    finally:
        if process is not None:
            stop_server(
                process,
                float(config["server"]["shutdown_timeout_seconds"]),
            )

    if success:
        manifest["status"] = "completed"
        write_json(manifest_path, manifest)
    print(f"server[{backend_id}]=stopped", flush=True)
    return success


def main() -> int:
    args = parse_args()
    if args.model_path is None:
        raise ValueError("--model-path or MODEL_PATH is required")
    if args.run_root is None:
        raise ValueError("--run-root or RUN_ROOT is required")
    if args.vllm_bin is None:
        raise ValueError("Could not find the vllm executable")

    config = load_config(args.config)
    if args.num_tokens:
        config["common"]["num_tokens"] = parse_num_tokens(args.num_tokens)
    if args.warmup is not None:
        if args.warmup < 0:
            raise ValueError("warmup must be non-negative")
        config["common"]["warmup"] = args.warmup
    if args.repeat is not None:
        if args.repeat < 1:
            raise ValueError("repeat must be at least 1")
        config["common"]["repeat"] = args.repeat
    backends = select_backends(config["backends"], args.backends)
    args.run_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        versions = {"dry_run": "version check skipped"}
        gpu = {"dry_run": "GPU check skipped"}
    else:
        verify_backend_dependencies(backends)
        versions = verify_environment(args.vllm_source)
        gpu = gpu_metadata()

    base_manifest = {
        "timestamp_utc": utc_timestamp(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "node": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "model_path": str(args.model_path.resolve()),
        "config": str(args.config.resolve()),
        "versions": versions,
        "gpu": gpu,
        "expected_experiment_ids": [str(backend["id"]) for backend in backends],
        "expected_num_tokens": [int(value) for value in config["common"]["num_tokens"]],
        "expected_warmup": int(config["common"]["warmup"]),
        "expected_repeat": int(config["common"]["repeat"]),
    }
    write_json(args.run_root / "run.json", base_manifest)

    failed: list[str] = []
    for backend in backends:
        if not run_backend(args, config, backend, base_manifest):
            failed.append(str(backend["id"]))
            if args.fail_fast:
                break

    print(f"completed_backends={len(backends) - len(failed)}", flush=True)
    print(f"failed_backends={','.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
