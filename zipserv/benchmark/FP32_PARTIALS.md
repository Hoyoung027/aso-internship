# FP32 Split-K partial storage

`--partial-dtype bf16|fp32` is supported by `slurm/run_zipserv.sh`,
`scripts/run_experiments.py`, and `bin/test_mm_{tune,final}`. The default is BF16.
There is deliberately no simultaneous `both` precision mode. (`--mode both`
still means tuning followed by a final run, at one partial dtype.)

## Numerical paths

- BF16: BF16 inputs, Tensor Core FP32 accumulators, BF16 partial stores,
  FP32 reduction, BF16 final output.
- FP32: same inputs, MMA and K partitioning, FP32 partial stores,
  same reduction mapping/order in FP32, BF16 final output.
- Split-K=1 uses the same old GEMM path in both modes; no workspace allocation.
- This is **not** full-FP32 input/multiply computation and is not an exact oracle.

Only one partial buffer is allocated. Its size is `M*N*SplitK*sizeof(dtype)`
when Split-K>1 and zero otherwise. This doubles workspace bytes, not total
GPU memory. The selected 20260907-204057 LLaMA cases need at most 31.3125 MiB
of FP32 workspace (70B shared LM Head, N=16, Split-K=4).

## Rebuild and check

From the zipserv project directory:

```bash
bash scripts/build_benchmarks.sh
python3 scripts/test_partial_dtype.py -v
```

The build script compiles the shared library and both experiment binaries.
It records supported partial dtypes in `bin/build_manifest.txt`. The runner
also checks the binary's reported dtype and allocated byte count, rejecting
stale binaries that ignore the new option.

`scripts/smoke_partial_dtype.py --baseline-bin DIR --output-dir DIR` runs on an
allocated GPU, not the login node. `--baseline-bin` must contain the previous
`test_mm_final` and its matching `libL_API.so`. The test checks old/new BF16
error summaries, Split-K=1 identity, Fast/Safe kernels, all split candidates,
workspace sizes and real 8B block-0 Down weights. These are summary-based
regression checks, not a bitwise dump comparison or a full correctness proof.

## Reuse previous tuning, run FP32 only

```bash
bash slurm/run_zipserv.sh \
  --mode run --partial-dtype fp32 \
  --models llama3.1-8b llama3.1-70b \
  --blocks llama3.1-8b=0,16,31 llama3.1-70b=0,40,79 \
  --layers all --time 12:00:00 \
  --tuning-dir "$PWD/results/zipserv-rtx4090-llama31-realweights-3blocks-abserror-20260907-204057-tuning"
```

This submits six model/block jobs plus one collector, using the existing 78
Split-K selections for 234 final invocations (3 trials). Shared LM Head is
measured only in each model's block-0 job. Current iteration settings remain
100 warmup / 1000 timed repetitions. No new tuning is submitted.

New result directories contain `partial-fp32` in their name. The selected
precision is stored in the configuration snapshot, invocation manifest and
`partial_dtype` CSV column. `partial_workspace_bytes` records allocation size.
The selection file path/hash is in the invocation manifest. All existing
performance and error metrics remain, including absolute-error exceedance.

Reusing a result directory containing a different dtype is rejected, even
with `--force`. Keep BF16 and FP32 results in different directories; do not
pool their trials when plotting. Use the same selected Split-K for causal
comparison first; separately retune FP32 only if optimizing its performance.

The original GEMM API/ABI is retained. An additional
`BF16TripleBitmap_MM_FP32Workspace_API` accepts a `float*` workspace. Both
Fast and Safe GEMM kernels share output-type templates. The FP32 reduction
uses the original thread mapping and final BF16 conversion.
