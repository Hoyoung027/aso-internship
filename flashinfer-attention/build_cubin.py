#!/usr/bin/env python3
"""Derive a CUBIN build command from FlashInfer's generated build.ninja.

Dry-run is the default. Pass --execute only after checking the printed command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def ninja_commands(ninja_path: Path) -> list[str]:
    """Ninja가 실제 실행할 compile/link 명령을 문자열로 얻는다.

    build.ninja를 직접 파싱하는 대신 ninja의 공식 -t commands 출력을 사용한다.
    """

    result = subprocess.run(
        ["ninja", "-f", str(ninja_path), "-t", "commands"],
        cwd=ninja_path.parent,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def select_nvcc_command(commands: list[str], source_name: str) -> list[str]:
    """여러 compile/link 명령 중 원하는 kernel translation unit만 선택한다."""

    matches = []
    for command in commands:
        tokens = shlex.split(command)
        if any(Path(token).name == source_name for token in tokens):
            matches.append(tokens)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one NVCC command for {source_name}, found {len(matches)}"
        )
    return matches[0]


def to_cubin_command(command: list[str], output: Path) -> list[str]:
    """`.cuda.o` compile 명령을 standalone CUBIN compile 명령으로 바꾼다.

    include, macro, optimization, gencode flag는 전부 보존한다. 단지 object용
    `-c/-o old.o`를 제거하고 `--cubin -o new.cubin`을 추가한다.
    """

    converted = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token == "-c":
            continue
        if token == "-o":
            skip_next = True
            continue
        if token.startswith("-o") and token != "-o":
            continue
        converted.append(token)

    converted.extend(["--cubin", "-o", str(output.resolve())])
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=("decode", "prefill"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entry = manifest[args.kind]
    ninja_path = Path(entry["original_ninja_path"])
    if not ninja_path.exists():
        raise RuntimeError(
            f"Original ninja file no longer exists: {ninja_path}. "
            "Re-run prepare_cubin_sources.py in the same environment."
        )

    source_name = (
        "single_decode_kernel.cu"
        if args.kind == "decode"
        else manifest["configuration"]["prefill_mask_source"]
    )
    output_dir = args.manifest.parent / "cubin"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"single_{args.kind}_sm86_fp16_h128.cubin"

    # 손으로 NVCC flag를 재작성하지 않고 FlashInfer JIT와 동일한 명령에서
    # 출발하는 것이 비교의 핵심이다.
    original = select_nvcc_command(ninja_commands(ninja_path), source_name)
    command = to_cubin_command(original, output)
    command_text = shlex.join(command)
    print(command_text)

    if not any("sm_86" in token or "compute_86" in token for token in command):
        raise RuntimeError("Refusing to build: transformed command does not target SM86")
    # 기본이 dry-run인 이유는 FlashInfer/CUDA 버전에 따라 --cubin과 함께 쓸 수
    # 없는 object/dependency flag가 있을 수 있기 때문이다.
    if not args.execute:
        print("Dry-run only. Re-run with --execute after reviewing the command.")
        return

    subprocess.run(command, cwd=ninja_path.parent, check=True)
    print(f"Wrote {output.resolve()} ({output.stat().st_size} bytes)")

    # launcher 구현에 필요한 실제 mangled symbol과 register/shared-memory
    # 사용량을 보존한다. 이 정보 없이 kernel 이름이나 launch 값을 추측하지 않는다.
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump:
        inspections = {
            "symbols": "--dump-elf-symbols",
            "resources": "--dump-resource-usage",
        }
        for suffix, option in inspections.items():
            result = subprocess.run(
                [cuobjdump, option, str(output)],
                check=True,
                text=True,
                capture_output=True,
            )
            inspection_path = output.with_suffix(f".{suffix}.txt")
            inspection_path.write_text(result.stdout, encoding="utf-8")
            print(f"Wrote {inspection_path.resolve()}")
    else:
        print("cuobjdump not found; kernel symbol inspection was skipped")


if __name__ == "__main__":
    main()
