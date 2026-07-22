#!/usr/bin/env python3
"""정확한 길이의 Prefill/Decode 요청을 vLLM 서버에 전송한다.

텍스트를 tokenizer에 통과시키지 않고 token ID를 직접 전송한다. 따라서
chat template이나 special token 때문에 입력 길이가 달라지는 일을 피할 수
있다. ``--profile``을 사용하면 측정 요청 하나만 Torch Profiler에 포함한다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "gpt-oss-20b"
PROJECT_DIR = Path(__file__).resolve().parent
# 작은 요청 metadata는 저장소 안에 두고, 대용량 trace는 Lustre에 분리한다.
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "requests"


def trace_files_from_environment() -> set[Path]:
    """현재 profiler 출력 디렉터리에 있는 trace 파일 집합을 반환한다."""

    trace_dir = os.environ.get("TORCH_PROFILE_DIR")
    if not trace_dir:
        return set()
    root = Path(trace_dir).expanduser()
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.rglob("*.pt.trace.json.gz")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send exact-length Prefill/Decode requests to vLLM."
    )
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-len", type=int, help="Prompt tokens (default: 1024).")
    parser.add_argument(
        "--output-len",
        type=int,
        help="Generated tokens (default: prefill=1, decode=128).",
    )
    parser.add_argument("--batch-size", type=int, help="Prompts (default: 1).")
    parser.add_argument(
        "--prompt-pattern",
        choices=("random", "repeated"),
        default="random",
        help="Prompt token pattern (default: random).",
    )
    parser.add_argument(
        "--prompt-token-id",
        type=int,
        default=1000,
        help="Token ID for the repeated pattern (default: 1000).",
    )
    parser.add_argument(
        "--prompt-token-min",
        type=int,
        default=1000,
        help="Inclusive minimum token ID for random prompts (default: 1000).",
    )
    parser.add_argument(
        "--prompt-token-max",
        type=int,
        default=100000,
        help="Exclusive maximum token ID for random prompts (default: 100000).",
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=42,
        help="Seed used to generate deterministic random prompts (default: 42).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unprofiled warmup requests (default: 1).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Call /start_profile and /stop_profile around measurement.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, Any]:
    """JSON HTTP 요청을 보내고 status code와 역직렬화한 body를 반환한다."""

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, json.loads(body) if body else None
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc.reason}") from exc


def check_health(base_url: str, timeout: float) -> None:
    """측정 전에 API 서버와 model engine이 준비됐는지 빠르게 확인한다."""

    request = Request(f"{base_url}/health", method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Health check returned HTTP {response.status}")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"vLLM health check failed: {exc}") from exc


def make_payload(
    *,
    model: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    prompt_pattern: str,
    prompt_token_id: int,
    prompt_token_min: int,
    prompt_token_max: int,
    prompt_seed: int,
) -> dict[str, Any]:
    """정확히 ``batch_size * input_len``개의 prompt token을 만든다."""

    if prompt_pattern == "repeated":
        # 디버깅에는 단순하지만 동일 expert로 routing이 편향될 수 있다.
        prompt = [prompt_token_id] * input_len
        prompts = [prompt.copy() for _ in range(batch_size)]
    else:
        # 하나의 generator를 순서대로 사용해 batch마다 서로 다른 prompt를
        # 만들면서도 seed가 같으면 실험 전체를 재현할 수 있게 한다.
        generator = random.Random(prompt_seed)
        prompts = [
            [
                generator.randrange(prompt_token_min, prompt_token_max)
                for _ in range(input_len)
            ]
            for _ in range(batch_size)
        ]
    return {
        "model": model,
        # list[list[int]]는 여러 prompt를 하나의 API request로 묶는다.
        "prompt": prompts,
        # token ID를 정확히 input_len개 유지하기 위해 BOS 등을 추가하지 않는다.
        "add_special_tokens": False,
        # EOS가 발생하더라도 조건별 Decode step 수를 동일하게 유지한다.
        "max_tokens": output_len,
        "min_tokens": output_len,
        "ignore_eos": True,
        # Sampling 변동을 제거해 같은 입력에서 같은 실행 경로를 사용한다.
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
    }


def send_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[float, dict[str, Any]]:
    """Completion 요청의 client-side wall time과 JSON 응답을 반환한다."""

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
    return elapsed, response


def main() -> int:
    args = parse_args()
    # CLI에서 생략한 경우 YAML 실험 설계와 같은 기본값을 적용한다.
    input_len = args.input_len if args.input_len is not None else 1024
    output_len = args.output_len
    if output_len is None:
        output_len = 1 if args.phase == "prefill" else 128
    batch_size = args.batch_size if args.batch_size is not None else 1

    for name, value in (
        ("input_len", input_len),
        ("output_len", output_len),
        ("batch_size", batch_size),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if args.prompt_token_id < 0:
        raise ValueError("prompt_token_id must be non-negative")
    if args.prompt_token_min < 0:
        raise ValueError("prompt_token_min must be non-negative")
    if args.prompt_token_max <= args.prompt_token_min:
        raise ValueError("prompt_token_max must be greater than prompt_token_min")

    base_url = args.base_url.rstrip("/")
    check_health(base_url, min(args.timeout, 30.0))
    payload = make_payload(
        model=args.model,
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
        prompt_pattern=args.prompt_pattern,
        prompt_token_id=args.prompt_token_id,
        prompt_token_min=args.prompt_token_min,
        prompt_token_max=args.prompt_token_max,
        prompt_seed=args.prompt_seed,
    )

    print(
        f"phase={args.phase} batch={batch_size} "
        f"input={input_len} output={output_len}",
        flush=True,
    )
    # Warmup은 profiler를 켜기 전에 실행한다. 커널 초기화와 cache warming을
    # 측정 trace에서 제외하기 위한 것이다.
    for index in range(args.warmup):
        elapsed, _ = send_completion(base_url, payload, args.timeout)
        print(f"warmup[{index + 1}]={elapsed:.6f}s", flush=True)

    profile_started = False
    traces_before_profile = trace_files_from_environment()
    if args.profile:
        # 이 endpoint는 profiler 설정으로 시작한 vLLM 서버에만 등록된다.
        request_json(
            "POST",
            f"{base_url}/start_profile",
            payload=None,
            timeout=args.timeout,
        )
        profile_started = True
        print("profiler=start", flush=True)

    try:
        # Profiler 범위에는 이 측정 요청 하나만 포함한다.
        elapsed, response = send_completion(base_url, payload, args.timeout)
    finally:
        # 요청이 실패해도 profiler를 정지해 trace flush를 시도한다.
        if profile_started:
            request_json(
                "POST",
                f"{base_url}/stop_profile",
                payload=None,
                timeout=args.timeout,
            )
            print("profiler=stop", flush=True)

    # /stop_profile 응답은 trace flush가 끝난 뒤 돌아온다. 시작 전후 파일
    # 차이를 metadata에 기록하면 분석기가 timestamp 추측 없이 매칭할 수 있다.
    traces_after_profile = trace_files_from_environment()
    new_trace_files = sorted(traces_after_profile - traces_before_profile)

    usage = response.get("usage", {})
    expected_prompt_tokens = batch_size * input_len
    actual_prompt_tokens = usage.get("prompt_tokens")
    # 길이 sweep의 독립변수가 실제 서버에서도 정확히 유지됐는지 검증한다.
    if actual_prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            "Prompt-token count mismatch: "
            f"expected {expected_prompt_tokens}, got {actual_prompt_tokens}"
        )

    # 파일명만으로 실험 조건을 식별할 수 있게 한다.
    experiment_id = f"{args.phase}-b{batch_size}-i{input_len}-o{output_len}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{experiment_id}-{timestamp}.json"
    result = {
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "phase": args.phase,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "prompt_pattern": args.prompt_pattern,
        "prompt_token_id": args.prompt_token_id,
        "prompt_token_min": args.prompt_token_min,
        "prompt_token_max": args.prompt_token_max,
        "prompt_seed": args.prompt_seed,
        "profile_enabled": args.profile,
        "warmup_requests": args.warmup,
        "trace_files": [str(path) for path in new_trace_files],
        "client_elapsed_seconds": elapsed,
        "expected_prompt_tokens": expected_prompt_tokens,
        "usage": usage,
        "response": response,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"measured={elapsed:.6f}s", flush=True)
    print(f"usage={json.dumps(usage, ensure_ascii=False)}", flush=True)
    print(f"result={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
