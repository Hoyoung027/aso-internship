"""Collect final, post-RoPE Llama KV. No tensor D2H inside inference loop."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import re
import sys
from pathlib import Path
from run_metadata import new_metadata


def main():
    # --- 1. 실행 인자: 모델·입력 경로와 수집할 토큰 수를 검증 ---
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/lustre/hybyun0207/models/Llama-3.1-8B-Instruct")
    p.add_argument("--tokenizer", required=True, help="Matching local tokenizer directory")
    p.add_argument("--prompts", type=Path, required=True, help="JSONL: id, split, category, text")
    p.add_argument("--output", type=Path, required=True, help="Result root; timestamp/experiment/job directory is added automatically")
    p.add_argument("--prefill-tokens", type=int, default=512)
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--repeat-short-prompts", action="store_true", help="Smoke tests only")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    identity = new_metadata()
    run_directory = args.output / identity["run_id"]
    args.output = run_directory / "kv"
    if args.prefill_tokens < 1 or args.decode_tokens < 0:
        p.error("Invalid token lengths")
    if args.prefill_tokens % 64:
        p.error("Use a multiple of 64 to separate prefill/decode tile accounting")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- 2. 입력 준비: 샘플 ID와 calibration/evaluation 구분을 확인 ---
    samples = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    ids = set()
    for sample in samples:
        sid = sample["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in ids:
            raise ValueError("Sample IDs must be unique safe path components")
        ids.add(sid)
        if sample["split"] not in ("calibration", "evaluation") or not sample["text"].strip():
            raise ValueError("Each sample requires split=calibration/evaluation and nonempty text")
    if not samples:
        raise ValueError("No prompts")
    # 텍스트를 지정 길이로 토큰화한다. 짧은 입력 반복은 smoke test에서만 허용한다.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prepared = []
    for sample in samples:
        tokens = tokenizer.encode(sample["text"], add_special_tokens=False)
        if not tokens:
            raise ValueError("Empty tokenized input")
        bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
        needed = args.prefill_tokens - len(bos)
        repeated = len(tokens) < needed
        if repeated and not args.repeat_short_prompts:
            raise ValueError(f"{sample['id']}: input too short; supply real longer text")
        if repeated:
            tokens = tokens * ((needed + len(tokens) - 1) // len(tokens))
        prepared.append((sample, bos + tokens[:needed], repeated))

    # --- 3. 모델 준비: BF16 모델을 GPU에 올리고 캐시 수집 API를 확인 ---
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    args.output.mkdir(parents=True, exist_ok=False)
    (run_directory / "run_metadata.json").write_text(json.dumps(identity, indent=2) + "\n")
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").eval().to("cuda")
    if model.config.model_type != "llama":
        raise ValueError("This collector is validated for Llama only")
    if args.prefill_tokens + args.decode_tokens > model.config.max_position_embeddings:
        raise ValueError("Requested sequence exceeds model context limit")
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise RuntimeError("Use a Transformers version supporting Llama logits_to_keep")
    # --- 4. 재현 정보: 모델 설정·라이브러리·GPU·수집 정책을 기록 ---
    config_bytes = (Path(args.model) / "config.json").read_bytes()
    run = dict(**identity, schema=1, model=str(Path(args.model).resolve()),
               config_sha256=hashlib.sha256(config_bytes).hexdigest(),
               tokenizer=args.tokenizer, tokenizer_vocab_size=len(tokenizer),
               gpu=torch.cuda.get_device_name(), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
               python=platform.python_version(), python_executable=sys.executable, cuda=torch.version.cuda,
               versions={n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")},
               seed=args.seed, backend="sdpa", batch=1, dtype="bfloat16",
               prefill_tokens=args.prefill_tokens, decode_tokens=args.decode_tokens,
               layout="[kv_head, token, head_dim], uint16 BF16 bits",
               capture="final cache, after RoPE and before GQA head repetition",
               generation="greedy; exactly decode_tokens forward calls; EOS does not stop collection",
               timing="No inference latency claim; all tensor D2H occurs after final CUDA synchronize",
               model_config=model.config.to_dict())
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    # --- 5. 샘플별 추론: KV는 종료 시점까지 GPU에 유지 ---
    for sample, prompt_ids, repeated in prepared:
        print(f"Collecting {sample['id']}", flush=True)
        destination = args.output / sample["id"]
        destination.mkdir()
        input_ids = torch.tensor([prompt_ids], device="cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            # Prefill: 전체 입력의 KV를 만들고 다음 토큰을 GPU에서 선택한다.
            out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)
            generated = []
            # Decode: 선택된 토큰을 하나씩 입력해 KV를 누적한다. EOS에도 정해진 횟수를 채운다.
            for _ in range(args.decode_tokens):
                generated.append(next_id)
                out = model(input_ids=next_id, past_key_values=cache,
                            use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                next_id = out.logits[:, -1].argmax(-1, keepdim=True)

            # --- 6. 추론 종료 후 저장: GPU 작업 완료를 기다린 뒤 CPU 전송 시작 ---
            torch.cuda.synchronize()
            # 수집용 전송 비용을 추론 성능으로 해석하지 않는다.
            generated_ids = (torch.cat(generated, dim=1).cpu().tolist()[0] if generated else [])
            metadata = dict(**identity, id=sample["id"], split=sample["split"], category=sample.get("category", "unspecified"),
                            text=sample["text"], repeated_smoke_input=repeated,
                            prompt_token_ids=prompt_ids, decode_token_ids=generated_ids,
                            prefill_tokens=len(prompt_ids), decode_tokens=len(generated_ids),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(), layers=[])
            # 레이어별 K/V를 검증하고 저장한다. 토큰 위치로 prefill/decode를 나중에 구분한다.
            for layer, (key, value) in enumerate(cache):
                entry = dict(layer=layer, shape=list(key.shape), k_stride=list(key.stride()), v_stride=list(value.stride()))
                for kind, tensor in (("K", key), ("V", value)):
                    expected = (1, model.config.num_key_value_heads,
                                args.prefill_tokens + args.decode_tokens,
                                model.config.hidden_size // model.config.num_attention_heads)
                    if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected:
                        raise ValueError(f"Unexpected cache {tensor.dtype}, {tensor.shape}; expected {expected}")
                    # 수치 변환 없이 BF16의 원본 16bit 패턴을 uint16으로 재해석한다.
                    bits = tensor[0].detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)
                    np.save(destination / f"layer_{layer:02d}_{kind}.npy", bits, allow_pickle=False)
                metadata["layers"].append(entry)
            if len(metadata["layers"]) != model.config.num_hidden_layers:
                raise ValueError("Incomplete cache")
            (destination / "sample.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # 다음 샘플을 처리하기 전에 이전 GPU 캐시를 가리키는 참조를 해제한다.
        del out, cache, key, value, tensor, generated, next_id, input_ids
    # 모든 샘플 저장이 끝난 경우에만 완료 표시를 남긴다.
    (args.output / "COMPLETE").write_text("collection complete\n")
    print(f"Saved KV to {args.output}", flush=True)


if __name__ == "__main__":
    main()
