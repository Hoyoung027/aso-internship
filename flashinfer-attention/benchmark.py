#!/usr/bin/env python3
"""Benchmark FlashInfer single-decode/single-prefill binding paths.

Run this file in a fresh process for each binding/kernel pair.  Physical GPU 1
must be isolated with CUDA_VISIBLE_DEVICES=1 before Python starts.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


# import 시간도 cold-start 결과에 포함하기 위해 torch/flashinfer를 import하기 전에 프로세스 시작 시각을 기록
PROCESS_START_NS = time.perf_counter_ns()

def require_physical_gpu_1() -> None:
    """물리 GPU 0이 실수로 보이는 상태에서는 실행을 즉시 중단한다.

    CUDA_VISIBLE_DEVICES=1이면 물리 GPU 1만 노출되고, 프로세스 안에서는 그
    장치가 logical cuda:0이 된다. 이 검사는 CUDA context 생성 전이어야 하므로
    torch import보다 먼저 실행한다.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1":
        raise RuntimeError(
            "Refusing to import torch: set CUDA_VISIBLE_DEVICES=1 so only "
            "physical GPU 1 is visible. Inside Python, use logical cuda:0."
        )


require_physical_gpu_1()

import torch  # noqa: E402
import flashinfer  # noqa: E402

from flashinfer.decode import (  # noqa: E402
    SINGLE_KERNEL_TMP_SIZE,
    get_single_decode_module,
)
from flashinfer.prefill import get_single_prefill_module  # noqa: E402


# torch와 flashinfer import가 끝난 시각. PROCESS_START_NS와의 차이가
# 결과 JSON의 import_ms가 된다.
IMPORT_DONE_NS = time.perf_counter_ns()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", choices=("pybind", "tvm-ffi"), required=True)
    parser.add_argument("--kernel", choices=("decode", "prefill"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--host-repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-max-len", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-tensors", type=Path)
    return parser.parse_args()


def verify_environment(binding: str) -> dict[str, str | int]:
    """GPU 격리와 SM86 여부를 검사하고 재현성 metadata를 만든다."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got {torch.cuda.device_count()}"
        )

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (8, 6):
        raise RuntimeError(
            f"Expected RTX 3090 class SM86, got sm_{props.major}{props.minor} "
            f"({props.name})"
        )

    return {
        "binding": binding,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "logical_device": 0,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
    }


class SingleAttentionRunner:
    """
    커널 모듈 준비, workspace 준비, 입력 생성, 커널 호출 담당

    입출력 allocation을 제외하고 같은 module.run을 반복하기 위한 runner.

    public single_* API는 호출마다 tmp/output을 만들 수 있다. 여기서는 tmp와
    output을 재사용해서 binding + C++ launcher + CUDA kernel에 집중한다.
    """

    def __init__(self, args: argparse.Namespace):

        # Flashinfer 연산 중간 결과 저장을 위한 GPU 작업공간(workspace) 할당
        self.args = args
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.tmp = torch.empty(
            SINGLE_KERNEL_TMP_SIZE, dtype=torch.uint8, device=self.device
        )

        # get_single_*의 첫 호출은 JIT cache miss라면 NVCC compile까지 수행한다.
        # 따라서 이 구간은 warm kernel 시간이 아니라 module build/load 시간이다.
        build_begin = time.perf_counter_ns()

        # decode, prefill에 해당하는 module(컴파일된 상태의 .so) 불러오기
        # 제공한 인자에 맞는 CUDA kernel이 JIT compile되어 module에 포함된다. 이후 run() 호출은
        # 이미 compile된 kernel을 바로 호출한다.
        if args.kernel == "decode":
            self.module = get_single_decode_module(
                self.dtype,
                self.dtype,
                self.dtype,
                args.head_dim,
                args.head_dim,
                0,  # PosEncodingMode.NONE
                False,  # sliding window
                False,  # logits soft cap
            )
        else:
            self.module = get_single_prefill_module(
                "fa2",
                self.dtype,
                self.dtype,
                self.dtype,
                args.head_dim,
                args.head_dim,
                0,  # PosEncodingMode.NONE
                False,  # sliding window
                False,  # logits soft cap
                False,  # fp16 qk reduction
            )
        torch.cuda.synchronize()
        self.module_load_ms = (time.perf_counter_ns() - build_begin) / 1e6

        from flashinfer.jit.core import JitSpec

        # flashinfer가 .so를 pybind로 load하는지 tvm_ffi로 load하는지 확인하여 binding label을 결정
        loader_source = inspect.getsource(JitSpec.load)

        if "tvm_ffi.load_module" in loader_source:
            self.detected_binding = "tvm-ffi"
        elif "torch.ops" in loader_source or "load_library" in loader_source:
            self.detected_binding = "pybind"
        else:
            self.detected_binding = "unknown"

    def make_inputs(self, length: int) -> tuple[torch.Tensor, ...]:
        """두 binding에서 동일하게 재현되는 FP16 입력과 output을 생성한다."""

        # length별 seed를 고정하면 서로 다른 프로세스/worktree도 같은 입력을
        # 생성하므로 저장된 output끼리 직접 비교할 수 있다.
        torch.manual_seed(self.args.seed + length)
        hq = self.args.num_qo_heads
        hkv = self.args.num_kv_heads
        d = self.args.head_dim
        if hq % hkv:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads")

        # 연산 대상 tensor를 미리 할당
        if self.args.kernel == "decode":
            q = torch.randn(hq, d, device=self.device, dtype=self.dtype)
        else:
            q = torch.randn(length, hq, d, device=self.device, dtype=self.dtype)
        k = torch.randn(length, hkv, d, device=self.device, dtype=self.dtype)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        return q, k, v, out

    def call(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """미리 할당된 tensor로 FlashInfer 내부 module.run만 호출한다."""

        scale = 1.0 / math.sqrt(self.args.head_dim)
        if self.args.kernel == "decode":
            # use_tensor_cores=False인 실제 single-decode 경로와 같은 인자다.
            self.module.run(
                q,
                k,
                v,
                self.tmp, # 임시 workspace
                out, # 출력
                None,  # lse
                None,  # alibi slopes
                0,  # NHD
                -1,  # full attention window
                0.0,  # logits soft cap
                scale, # attention scale
                1.0,  # RoPE scale
                1e4,  # RoPE theta
            )
        else:
            # FA2 single-prefill의 causal/NHD 경로다. q/k/v scale은 FP8용이므로
            # 이번 FP16 실험에서는 None을 전달한다.
            self.module.run(
                q,
                k,
                v,
                self.tmp,
                out,
                None,  # lse
                1,  # MaskMode.CAUSAL
                0,  # NHD
                -1,  # full attention window
                None,  # packed custom mask
                None,  # alibi slopes
                0.0,  # logits soft cap
                scale,
                None,  # q scale
                None,  # k scale
                None,  # v scale
                1.0,  # RoPE scale
                1e4,  # RoPE theta
            )


def torch_reference(
    kernel: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """작은 입력의 정확성을 확인하는 단순 FP32 attention reference.

    성능 측정용이 아니다. 큰 Prefill은 [heads, qo_len, kv_len] logits가 매우
    커지므로 --check-max-len 이하에서만 호출한다.
    """

    hq = q.shape[-2] if kernel == "prefill" else q.shape[0]
    hkv = k.shape[1]
    group = hq // hkv
    # GQA에서 KV head 하나를 group개의 Q head가 공유한다. reference 계산은
    # 이해하기 쉽게 K/V head를 물리적으로 반복하지만 FlashInfer kernel은
    # 실제 KV cache를 복제하지 않는다.
    k_expanded = k.float().repeat_interleave(group, dim=1)
    v_expanded = v.float().repeat_interleave(group, dim=1)
    scale = 1.0 / math.sqrt(q.shape[-1])

    if kernel == "decode":
        logits = torch.einsum("hd,lhd->hl", q.float(), k_expanded) * scale
        probs = torch.softmax(logits, dim=-1)
        return torch.einsum("hl,lhd->hd", probs, v_expanded).to(q.dtype)

    qo_len = q.shape[0]
    kv_len = k.shape[0]
    logits = torch.einsum("qhd,khd->hqk", q.float(), k_expanded) * scale
    q_pos = torch.arange(qo_len, device=q.device)[:, None]
    k_pos = torch.arange(kv_len, device=q.device)[None, :]
    causal = k_pos <= q_pos + (kv_len - qo_len)
    logits.masked_fill_(~causal.unsqueeze(0), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v_expanded).to(q.dtype)


def measure_gpu_us(call, iterations: int) -> float:
    """CUDA event 사이의 GPU stream 시간을 측정한다.

    Python import/JIT 시간은 포함되지 않는다. 같은 GPU kernel이면 세 binding의
    이 값은 거의 같아야 한다.
    """

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def measure_host_enqueue_us(call, iterations: int, repeats: int) -> list[float]:
    """Python에서 비동기 CUDA launch를 enqueue하는 평균 host 시간을 잰다.

    시작 전 synchronize로 이전 작업을 비우고, finish 시각은 마지막 synchronize
    전에 기록한다. 따라서 뒤의 synchronize 대기 시간은 sample에 포함하지 않는다.
    """

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        finish = time.perf_counter_ns()
        torch.cuda.synchronize()
        samples.append((finish - begin) / 1000.0 / iterations)
    return samples


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:

    # 옵션, 환경, FlashInfer module 준비
    args = parse_args()
    metadata = verify_environment(args.binding)
    runner = SingleAttentionRunner(args)

    print(json.dumps(metadata, indent=2))
    print(
        f"requested_binding={args.binding} detected_binding={runner.detected_binding} "
        f"module_load_ms={runner.module_load_ms:.3f}"
    )
    if runner.detected_binding != args.binding:
        print(
            "WARNING: binding detection did not match the requested label. "
            "Confirm the FlashInfer commit/worktree before using the result.",
            file=sys.stderr,
        )

    if args.save_tensors:
        args.save_tensors.mkdir(parents=True, exist_ok=True)

    # 각 길이별로 입력 만들고 커널 호출 후 성능 측정
    for index, length in enumerate(args.lengths):
        q, k, v, out = runner.make_inputs(length)
        call = lambda: runner.call(q, k, v, out)

        # 첫 실행 시간 측정
        # 이 shape의 첫 실행에는 CUDA module/kernel lazy initialization이 포함될 수 있으므로 warm timing과 분리한다.
        first_begin = time.perf_counter_ns() 
        call()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter_ns() - first_begin) / 1e6

        # warmup 실행
        for _ in range(args.warmup):  
            call()
        torch.cuda.synchronize()

        max_abs_error = None
        max_rel_error = None
        if length <= args.check_max_len:
            ref = torch_reference(args.kernel, q, k, v)
            diff = (out.float() - ref.float()).abs()
            max_abs_error = diff.max().item()
            denom = ref.float().abs().clamp_min(1e-5)
            max_rel_error = (diff / denom).max().item()
            torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

        gpu_us = measure_gpu_us(call, args.iterations)
        host_samples = measure_host_enqueue_us(
            call, args.iterations, args.host_repeats
        )

        # JSONL 한 줄이 구현/kernel/length 한 조합의 결과다.
        record = {
            **metadata,
            "kernel": args.kernel,
            "length": length,
            "dtype": "float16",
            "num_qo_heads": args.num_qo_heads,
            "num_kv_heads": args.num_kv_heads,
            "gqa_group_size": args.num_qo_heads // args.num_kv_heads,
            "head_dim": args.head_dim,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "import_ms": (IMPORT_DONE_NS - PROCESS_START_NS) / 1e6,
            # 한 프로세스에서 module은 한 번만 load되므로 첫 row에만 기록한다.
            "module_load_ms": runner.module_load_ms if index == 0 else 0.0,
            "first_call_ms": first_call_ms,
            "gpu_us": gpu_us,
            "host_enqueue_median_us": statistics.median(host_samples),
            "host_enqueue_min_us": min(host_samples),
            "host_enqueue_max_us": max(host_samples),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "output_sum": out.float().sum().item(),
        }
        print(json.dumps(record, sort_keys=True))
        if args.output:
            append_jsonl(args.output, record)
        if args.save_tensors:
            torch.save(out.detach().cpu(), args.save_tensors / f"{args.kernel}-{length}.pt")


if __name__ == "__main__":
    main()
