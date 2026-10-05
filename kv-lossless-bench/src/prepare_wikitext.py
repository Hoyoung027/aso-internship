"""Build the Experiment A prompt JSONL from WikiText-103 (raw) articles."""
import argparse
import json
import random
import re
from pathlib import Path


def clean(text):
    # WikiText의 토큰화 흔적(@-@ 등과 구두점 앞 공백)만 되돌려 자연스러운 문장으로 만든다.
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r" ([,.;:!?%)\]'])", r"\1", text)
    text = re.sub(r"([(\[$]) ", r"\1", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def articles(rows):
    # 최상위 제목(" = Title = ")마다 새 문서를 시작한다. 하위 제목(" = = ")은 본문에 포함한다.
    current = []
    for line in rows:
        if re.fullmatch(r" = [^=].* = \n?", line):
            if current:
                yield "".join(current)
            current = []
        current.append(line)
    if current:
        yield "".join(current)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True, help="Local Llama-3.1 tokenizer directory")
    p.add_argument("--output", type=Path, default=Path("data/prompts_wikitext.jsonl"))
    p.add_argument("--min-tokens", type=int, default=2048, help="Minimum article length; use >= planned prefill tokens")
    p.add_argument("--per-split", type=int, default=16, help="Articles for each of calibration/evaluation")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"{args.output} already exists; refusing to overwrite")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
    docs = [clean(a) for a in articles(rows)]
    # 서로 다른 문서만 사용하므로 split 간 중복이 없다. 길이는 토큰 수로 확인한다.
    random.Random(args.seed).shuffle(docs)
    need = 2 * args.per_split
    chosen = []
    for doc in docs:
        if len(tokenizer.encode(doc, add_special_tokens=False)) >= args.min_tokens:
            chosen.append(doc)
            if len(chosen) == need:
                break
    if len(chosen) < need:
        raise RuntimeError(f"Only {len(chosen)} articles have >= {args.min_tokens} tokens")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for i, doc in enumerate(chosen):
            split = "calibration" if i < args.per_split else "evaluation"
            sid = f"{split[:4]}_wiki_{i % args.per_split:03d}"
            f.write(json.dumps(dict(id=sid, split=split, category="wikitext_prose", text=doc),
                               ensure_ascii=False) + "\n")
    print(f"Wrote {need} articles to {args.output}")


if __name__ == "__main__":
    main()
