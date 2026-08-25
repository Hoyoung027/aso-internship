# Local ZipServ benchmark frontend

This directory contains the experiment-owned copy of the ZipServ matrix
multiplication benchmark frontend. It was copied from
`ZipServ_ASPLOS26/kernel_benchmark` at ZipServ commit
`6f8a209bfe46b0905f90d61b0796e352147ff041`.

- `test_mm.cu` owns input preparation, cuBLAS/ZipGEMM timing, and validation.
- `utils.h` owns synthetic BF16 generation and host-side compression helpers.
- `Makefile` compiles `test_mm` and links the external ZipServ `libL_API.so`.

Build through `../scripts/build_benchmarks.sh`. The script temporarily applies
the configured warm-up/repeat constants to this local `utils.h`, restores the
source afterward, and places runtime artifacts under `../bin/`.

Experiment-specific changes, including real-weight file loading, belong here.
The CUDA ZipGEMM implementation remains in the external `ZipServ_ASPLOS26`
source tree referenced by `configs/experiments.json`.
