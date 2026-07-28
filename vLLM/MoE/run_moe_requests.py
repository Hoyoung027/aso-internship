#!/usr/bin/env python3
"""Send fixed-token GPT-OSS requests and collect one Torch trace per shape."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "gpt-oss-20b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master-num-tokens", type=int, default=8192)
    parser.add_argument("--prompt-token-min", type=int, default=1000)
    parser.add_argument("--prompt-token-max", type=int, default=100000)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--backend-requested", required=True)
    parser.add_argument("--backend-actual", required=True)
    parser.add_argument("--activation-dtype", required=True)
    parser.add_argument(
        "--flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    return parser.parse_args()


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data else {}
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            parsed = json.loads(body) if body else None
            return response.status, parsed
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc.reason}") from exc


def check_health(base_url: str, timeout: float) -> None:
    request = Request(f"{base_url}/health", method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Health check returned HTTP {response.status}"
                )
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"vLLM health check failed: {exc}") from exc


def build_prompt(args: argparse.Namespace) -> list[int]:
    if args.num_tokens < 1:
        raise ValueError("num_tokens must be at least 1")
    if args.master_num_tokens < args.num_tokens:
        raise ValueError("master_num_tokens must cover num_tokens")
    if args.prompt_token_min < 0:
        raise ValueError("prompt_token_min must be non-negative")
    if args.prompt_token_max <= args.prompt_token_min:
        raise ValueError("prompt_token_max must exceed prompt_token_min")

    generator = random.Random(args.seed)
    master = [
        generator.randrange(args.prompt_token_min, args.prompt_token_max)
        for _ in range(args.master_num_tokens)
    ]
    return master[: args.num_tokens]


def build_payload(model: str, prompt: list[int]) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": [prompt],
        "add_special_tokens": False,
        "max_tokens": 1,
        "min_tokens": 1,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
    }


def send_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    expected_prompt_tokens: int,
) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter()
    status, response = request_json(
        "POST",
        f"{base_url}/v1/completions",
        payload=payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    if status != 200 or not isinstance(response, dict):
        raise RuntimeError(f"Unexpected completion response: HTTP {status}")

    usage = response.get("usage", {})
    actual_prompt_tokens = usage.get("prompt_tokens")
    if actual_prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            "Prompt-token count mismatch: "
            f"expected {expected_prompt_tokens}, got {actual_prompt_tokens}"
        )
    return elapsed, usage


def trace_files(root: Path) -> set[Path]:
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.rglob("*.pt.trace.json*")}


def wait_for_new_traces(
    root: Path,
    before: set[Path],
    timeout_seconds: float = 30.0,
) -> list[Path]:
    deadline = time.monotonic() + timeout_seconds
    newest: set[Path] = set()
    previous: set[Path] = set()
    stable_since: float | None = None
    while time.monotonic() < deadline:
        newest = trace_files(root) - before
        if newest != previous:
            previous = newest
            stable_since = time.monotonic() if newest else None
        elif newest and stable_since is not None:
            if time.monotonic() - stable_since >= 2.0:
                return sorted(newest)
        if newest and stable_since is None:
            stable_since = time.monotonic()
        if newest and time.monotonic() + 0.5 >= deadline:
            return sorted(newest)
        time.sleep(0.5)
    return sorted(newest)


def prompt_sha256(prompt: list[int]) -> str:
    encoded = ",".join(str(token) for token in prompt).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeat < 1:
        raise ValueError("repeat must be at least 1")

    base_url = args.base_url.rstrip("/")
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    check_health(base_url, min(args.timeout, 30.0))

    prompt = build_prompt(args)
    payload = build_payload(args.model, prompt)
    print(
        f"experiment={args.experiment_id} num_tokens={args.num_tokens} "
        f"warmup={args.warmup} repeat={args.repeat}",
        flush=True,
    )

    for index in range(args.warmup):
        elapsed, _ = send_completion(
            base_url,
            payload,
            args.timeout,
            args.num_tokens,
        )
        if index == 0 or (index + 1) % 5 == 0 or index + 1 == args.warmup:
            print(
                f"warmup={index + 1}/{args.warmup} "
                f"client_seconds={elapsed:.6f}",
                flush=True,
            )

    traces_before = trace_files(args.trace_dir)
    profile_started = False
    client_seconds: list[float] = []
    usages: list[dict[str, Any]] = []
    try:
        request_json(
            "POST",
            f"{base_url}/start_profile",
            payload=None,
            timeout=args.timeout,
        )
        profile_started = True
        print("profiler=start", flush=True)

        for index in range(args.repeat):
            elapsed, usage = send_completion(
                base_url,
                payload,
                args.timeout,
                args.num_tokens,
            )
            client_seconds.append(elapsed)
            usages.append(usage)
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == args.repeat:
                print(
                    f"repeat={index + 1}/{args.repeat} "
                    f"client_seconds={elapsed:.6f}",
                    flush=True,
                )
    finally:
        if profile_started:
            request_json(
                "POST",
                f"{base_url}/stop_profile",
                payload=None,
                timeout=args.timeout,
            )
            print("profiler=stop", flush=True)

    new_traces = wait_for_new_traces(args.trace_dir, traces_before)
    if not new_traces:
        raise RuntimeError(f"No new Torch trace appeared in {args.trace_dir}")

    timestamp = utc_timestamp()
    metadata_path = args.metadata_dir / (
        f"{args.experiment_id}-m{args.num_tokens}-{timestamp}.json"
    )
    metadata = {
        "timestamp_utc": timestamp,
        "experiment_id": args.experiment_id,
        "backend_requested": args.backend_requested,
        "backend_actual": args.backend_actual,
        "activation_dtype": args.activation_dtype,
        "flashinfer_autotune": args.flashinfer_autotune,
        "layer": 0,
        "num_tokens": args.num_tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "master_num_tokens": args.master_num_tokens,
        "prompt_token_min": args.prompt_token_min,
        "prompt_token_max": args.prompt_token_max,
        "prompt_sha256": prompt_sha256(prompt),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "client_seconds": client_seconds,
        "usages": usages,
        "trace_files": [str(path) for path in new_traces],
        "server_log": str(args.server_log.resolve()),
        "run_manifest": str(args.run_manifest.resolve()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"metadata={metadata_path.resolve()}", flush=True)
    print(f"trace_files={len(new_traces)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
