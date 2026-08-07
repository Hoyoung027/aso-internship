#!/usr/bin/env python3
"""Run FlashMoE's distributed correctness check with explicit experiment args."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


DEFAULT_FLASHMOE_SOURCE = Path("/home/hybyun0207/FlashMoE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FlashMoE's fused MoE output with its reference implementation."
    )
    parser.add_argument(
        "--flashmoe-source",
        type=Path,
        default=Path(os.environ.get("FLASHMOE_SOURCE", DEFAULT_FLASHMOE_SOURCE)),
    )
    parser.add_argument("--tokens-per-rank", type=int, default=1024)
    parser.add_argument("--token-dim", type=int, default=5120)
    parser.add_argument("--ffn-size", type=int, default=8192)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args()


def load_upstream_quickstart(source: Path):
    source = source.resolve()
    quickstart_path = source / "quickstart.py"
    if not quickstart_path.is_file():
        raise FileNotFoundError(f"Missing upstream quickstart: {quickstart_path}")

    # This experiment directory is also named "flashmoe". Put the upstream
    # source first so Python cannot mistake this directory for the package.
    source_string = str(source)
    if sys.path[0] != source_string:
        sys.path.insert(0, source_string)

    spec = importlib.util.spec_from_file_location(
        "flashmoe_upstream_quickstart", quickstart_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load upstream quickstart: {quickstart_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive")
    if args.num_experts % world_size != 0:
        raise ValueError(
            f"num_experts={args.num_experts} must be divisible by "
            f"world_size={world_size}"
        )
    if not 1 <= args.top_k <= args.num_experts:
        raise ValueError("--top-k must be between 1 and --num-experts")

    quickstart = load_upstream_quickstart(args.flashmoe_source)
    device_id = quickstart.flashmoe.get_local_rank()

    if rank == 0:
        metadata = {
            "experiment": "flashmoe_correctness",
            "flashmoe_source": str(args.flashmoe_source.resolve()),
            "world_size": world_size,
            "tokens_per_rank": args.tokens_per_rank,
            "total_tokens": args.tokens_per_rank * world_size,
            "token_dim": args.token_dim,
            "ffn_size": args.ffn_size,
            "num_experts": args.num_experts,
            "experts_per_rank": args.num_experts // world_size,
            "top_k": args.top_k,
            "dtype": "BF16",
            "mlp_type": "GATED",
            "activation": "SILU",
        }
        print("EXPERIMENT_CONFIG=" + json.dumps(metadata, sort_keys=True), flush=True)

    quickstart.run_fused_moe_forward_w_correctness_check(
        tokens_per_rank=args.tokens_per_rank,
        token_dim=args.token_dim,
        ffn_size=args.ffn_size,
        num_experts=args.num_experts,
        k=args.top_k,
        device_id=device_id,
        use_torch_init=True,
    )


if __name__ == "__main__":
    main()
