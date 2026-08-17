# aso-internship
ASO Lab summer internship

## Experiments

- [`flashinfer-attention/`](flashinfer-attention/README.md): RTX 3090에서
  FlashInfer single Decode/Prefill의 PyTorch binding, TVM-FFI,
  TVM-FFI+CUBIN 경로를 비교하는 단계별 실험
- [`MoE/`](MoE/README.md): RTX PRO 6000 Blackwell에서 GPT-OSS 20B를
  vLLM 0.23.0으로 서빙하고 Attention/MoE 프로파일링을 준비하는 과정
- [`flashmoe/`](flashmoe/README.md): RTX PRO 6000에서 FlashMoE의 분산 JIT,
  NVSHMEM 통신 및 reference 대비 정확성을 검증하는 실험
- [`gemm/`](gemm/README.md): GPT-OSS-20B의 QKV/O/Router Gate projection
  shape에서 PyTorch와 FlashInfer BF16·FP8·FP4 GEMM 및 AutoTuner를 비교하는 실험
- [`zipserv/`](zipserv/README.md): ZipServ 공개 artifact의 합성 weight를 사용해
  논문 모델 shape에서 ZipGEMM과 cuBLAS Tensor Core 성능을 비교하는 재현 실험
