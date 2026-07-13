#!/usr/bin/env python3
"""Render FlashInfer SM86 single-attention JIT sources for CUBIN work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time


def require_physical_gpu_1() -> None:
    """SM86 이외 architecture가 build 설정에 섞이는 것을 막는다."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1 before running this script")
    arch_list = os.environ.get("FLASHINFER_CUDA_ARCH_LIST")
    if arch_list not in ("8.6", "8.6+PTX"):
        raise RuntimeError(
            "Set FLASHINFER_CUDA_ARCH_LIST=8.6 to prevent non-SM86 artifacts"
        )


require_physical_gpu_1()

import torch  # noqa: E402
from flashinfer.jit.attention import (  # noqa: E402
    gen_single_decode_module,
    gen_single_prefill_module,
)


def copy_generated_tree(spec, destination: Path) -> dict:
    """JitSpec의 생성 source와 build 정보를 실험 artifacts로 복사한다.

    gen_single_* 호출 시 Jinja template은 이미 source로 렌더링된다. write_ninja는
    compile하지 않고 해당 specialization의 정확한 NVCC 명령만 기록한다.
    """

    source_parents = {source.parent.resolve() for source in spec.sources}
    if len(source_parents) != 1:
        raise RuntimeError(f"Expected one generated source directory: {source_parents}")
    source_dir = source_parents.pop()
    destination.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    # 이 시점에는 NVCC를 실행하지 않는다. build.ninja만 생성한다.
    spec.write_ninja()
    shutil.copy2(spec.ninja_path, destination / "build.ninja")
    return {
        "name": spec.name,
        "original_source_dir": str(source_dir),
        "original_build_dir": str(spec.build_dir.resolve()),
        "original_ninja_path": str(spec.ninja_path.resolve()),
        "copied_source_dir": str(destination.resolve()),
        "sources": [str(source.resolve()) for source in spec.sources],
        "source_basenames": [source.name for source in spec.sources],
        "extra_cuda_cflags": spec.extra_cuda_cflags or [],
        "extra_cflags": spec.extra_cflags or [],
        "extra_ldflags": spec.extra_ldflags or [],
        "extra_include_dirs": [str(p) for p in (spec.extra_include_dirs or [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated"))
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (8, 6):
        raise RuntimeError(f"Expected SM86, got {props.name} sm_{props.major}{props.minor}")

    # FlashInfer가 평소 JIT에 사용하는 generator를 그대로 호출한다. 별도의
    # hand-written kernel을 만들지 않으므로 TVM-FFI baseline과 specialization
    # 조건을 공유할 수 있다.
    decode = gen_single_decode_module(
        torch.float16,
        torch.float16,
        torch.float16,
        128,
        128,
        0,  # PosEncodingMode.NONE
        False,
        False,
    )
    prefill = gen_single_prefill_module(
        "fa2",
        torch.float16,
        torch.float16,
        torch.float16,
        128,
        128,
        0,  # PosEncodingMode.NONE
        False,
        False,
        False,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_unix": time.time(),
        "gpu": props.name,
        "compute_capability": "8.6",
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "configuration": {
            "dtype": "float16",
            "head_dim_qk": 128,
            "head_dim_vo": 128,
            "pos_encoding": "NONE",
            "sliding_window": False,
            "logits_soft_cap": False,
            "prefill_backend": "fa2",
            # FlashInfer MaskMode enum에서 1이 causal이다. 생성기는 mask별로
            # translation unit을 나누므로 CUBIN 실험에서는 이 파일을 선택한다.
            "prefill_mask_source": "single_prefill_kernel_mask_1.cu",
        },
        "decode": copy_generated_tree(decode, args.output / "decode"),
        "prefill": copy_generated_tree(prefill, args.output / "prefill"),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path.resolve()}")
    print(f"Decode ninja: {manifest['decode']['original_ninja_path']}")
    print(f"Prefill ninja: {manifest['prefill']['original_ninja_path']}")


if __name__ == "__main__":
    main()
