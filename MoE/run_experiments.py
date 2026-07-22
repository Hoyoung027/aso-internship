#!/usr/bin/env python3
"""YAML에 선언한 Prefill/Decode 실험 매트릭스를 순서대로 실행한다."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "experiments.yaml"
# 단일 조건 실행과 HTTP/profiler 제어는 별도 스크립트에 위임한다.
REQUEST_SCRIPT = PROJECT_DIR / "run_profile_request.py"
ANALYZE_SCRIPT = PROJECT_DIR / "analyze_traces.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("all", "prefill", "decode"),
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without contacting the vLLM server.",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable profiler endpoints even if enabled in YAML.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """YAML을 읽고 최상위 구조가 mapping인지 확인한다."""

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return config


def resolve_project_path(value: str) -> Path:
    """상대 결과 경로는 MoE 프로젝트 디렉터리를 기준으로 해석한다."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def build_command(
    *,
    phase: str,
    condition: dict[str, Any],
    config: dict[str, Any],
    profile: bool,
) -> list[str]:
    """YAML의 공통값과 한 조건을 단일 요청 CLI 명령으로 변환한다."""

    server = config.get("server", {})
    common = config.get("common", {})
    paths = config.get("paths", {})
    output_dir = resolve_project_path(paths.get("request_results", "results/requests"))

    command = [
        # 현재 활성화한 profile Conda 환경과 동일한 interpreter를 사용한다.
        sys.executable,
        str(REQUEST_SCRIPT),
        "--phase",
        phase,
        "--base-url",
        str(server.get("base_url", "http://127.0.0.1:8000")),
        "--model",
        str(server.get("model", "gpt-oss-20b")),
        "--input-len",
        str(condition["input_len"]),
        "--output-len",
        str(condition["output_len"]),
        "--batch-size",
        str(condition["batch_size"]),
        "--prompt-token-id",
        str(common.get("prompt_token_id", 1000)),
        "--prompt-pattern",
        str(common.get("prompt_pattern", "random")),
        "--prompt-token-min",
        str(common.get("prompt_token_min", 1000)),
        "--prompt-token-max",
        str(common.get("prompt_token_max", 100000)),
        "--prompt-seed",
        str(common.get("prompt_seed", 42)),
        "--warmup",
        str(common.get("warmup", 1)),
        "--timeout",
        str(common.get("timeout", 1800)),
        "--output-dir",
        str(output_dir),
    ]
    if profile:
        command.append("--profile")
    return command


def selected_phases(requested: str) -> tuple[str, ...]:
    """all을 실제 실행 순서인 prefill, decode로 확장한다."""

    if requested == "all":
        return ("prefill", "decode")
    return (requested,)


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    profile = bool(config.get("common", {}).get("profile", True))
    # 최초 HTTP 검증에서는 CLI의 --no-profile로 YAML 설정을 덮어쓸 수 있다.
    profile = profile and not args.no_profile
    repetitions = int(config.get("common", {}).get("repetitions", 1))
    if repetitions < 1:
        raise ValueError("common.repetitions must be at least 1")

    for phase in selected_phases(args.phase):
        conditions = config.get("experiments", {}).get(phase, [])
        if not isinstance(conditions, list) or not conditions:
            raise ValueError(f"No experiment conditions configured for {phase}")
        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError(f"Invalid {phase} condition: {condition!r}")
            command = build_command(
                phase=phase,
                condition=condition,
                config=config,
                profile=profile,
            )
            # 같은 조건을 독립된 trace로 반복해 평균과 분산을 계산할 수 있다.
            for repeat in range(1, repetitions + 1):
                print(
                    f"[{phase}] repeat={repeat}/{repetitions}: "
                    f"{shlex.join(command)}",
                    flush=True,
                )
                if not args.dry_run:
                    # 한 조건이 실패하면 다음 조건을 실행하지 않아 결과가
                    # 부분적으로 정상인 것처럼 보이는 일을 막는다.
                    subprocess.run(command, check=True)

    # 모든 요청이 성공한 뒤 CSV를 다시 생성한다. 기존 행과 새 행을 request
    # metadata에서 재구성하므로 append 중복이나 부분 행이 생기지 않는다.
    if not args.dry_run:
        paths = config.get("paths", {})
        request_dir = resolve_project_path(
            paths.get("request_results", "results/requests")
        )
        csv_path = resolve_project_path(
            paths.get("experiment_csv", "results/experiment.csv")
        )
        trace_dir = Path(
            paths.get(
                "torch_traces",
                "/lustre/hybyun0207/gptoss-profile/torch-traces",
            )
        ).expanduser()
        subprocess.run(
            [
                sys.executable,
                str(ANALYZE_SCRIPT),
                "--request-dir",
                str(request_dir),
                "--trace-dir",
                str(trace_dir),
                "--output",
                str(csv_path),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
