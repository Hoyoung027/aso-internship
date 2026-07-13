#!/usr/bin/env python3
"""Compare tensors saved by two benchmark implementations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    # benchmark.py는 동일한 이름(decode-128.pt 등)으로 저장하므로 파일명을
    # key로 사용해 서로 다른 binding의 같은 workload를 짝짓는다.
    left_files = {p.name: p for p in args.left.glob("*.pt")}
    right_files = {p.name: p for p in args.right.glob("*.pt")}
    common = sorted(left_files.keys() & right_files.keys())
    if not common:
        raise RuntimeError("No matching .pt files were found")

    missing_left = sorted(right_files.keys() - left_files.keys())
    missing_right = sorted(left_files.keys() - right_files.keys())
    if missing_left or missing_right:
        raise RuntimeError(
            f"Tensor sets differ: missing_left={missing_left}, "
            f"missing_right={missing_right}"
        )

    for name in common:
        # 비교는 CPU에서 수행한다. 이 도구 자체는 GPU 0/1 어느 것도 사용하지
        # 않으며 성능 측정에도 포함되지 않는다.
        left = torch.load(left_files[name], map_location="cpu", weights_only=True)
        right = torch.load(right_files[name], map_location="cpu", weights_only=True)
        diff = (left.float() - right.float()).abs()
        torch.testing.assert_close(left, right, rtol=args.rtol, atol=args.atol)
        print(f"PASS {name}: max_abs_error={diff.max().item():.6e}")


if __name__ == "__main__":
    main()
