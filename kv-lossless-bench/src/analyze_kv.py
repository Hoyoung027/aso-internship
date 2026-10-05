"""Analyze saved KV offline; fit fixed exponent intervals on calibration only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analysis_core import histogram, best_start, describe, tile_storage, residual_storage
from run_metadata import source_metadata


# --- 공통 출력: 같은 필드를 가진 분석 결과를 CSV로 저장 ---
def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, output):
    # --- 1. 입력 검증: 완료된 수집과 두 데이터 split이 모두 있는지 확인 ---
    if not (source / "COMPLETE").exists():
        raise ValueError("Collection is incomplete (missing COMPLETE)")
    provenance = source_metadata(source)
    samples = [(p.parent, json.loads(p.read_text())) for p in sorted(source.glob("*/sample.json"))]
    if {m["split"] for _, m in samples} != {"calibration", "evaluation"}:
        raise ValueError("Both calibration and evaluation samples are required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # --- 2. Calibration: 레이어별 K/V histogram을 합쳐 고정 지수 구간 선택 ---
    # Evaluation 데이터는 구간 선택에 사용하지 않아 평가 데이터 누수를 막는다.
    calibrated = {}
    for directory, meta in samples:
        if meta["split"] != "calibration":
            continue
        for path in sorted(directory.glob("layer_*.npy")):
            h = histogram(np.load(path, mmap_mode="r"))
            calibrated[path.stem] = calibrated.get(path.stem, np.zeros(256, dtype=np.int64)) + h
    starts = {name: best_start(h) for name, h in calibrated.items()}
    (output / "calibration.json").write_text(json.dumps(starts, indent=2) + "\n")

    # --- 3. 분포 분석: 샘플·레이어·K/V별 최종 캐시를 메모리 매핑으로 읽기 ---
    distributions, blocks, summaries = [], [], []
    histograms = {}
    for directory, meta in samples:
        for path in sorted(directory.glob("layer_*.npy")):
            bits = np.load(path, mmap_mode="r")
            if bits.ndim != 3 or bits.shape[1] != meta["prefill_tokens"] + meta["decode_tokens"]:
                raise ValueError(f"Unexpected shape: {path}")
            fixed = starts[path.stem]
            _, layer, kind = path.stem.split("_")
            # 최종 캐시를 토큰 위치 기준으로 나눠 prefill/decode 신규 KV를 중복 없이 분석한다.
            boundary = meta["prefill_tokens"]
            for phase, begin, end in (("prefill", 0, boundary), ("decode", boundary, bits.shape[1])):
                if begin == end:
                    continue
                if begin % 64:
                    raise ValueError("Phase boundary must align with 64-token tiles")
                identity = dict(**provenance, sample=meta["id"], split=meta["split"], category=meta["category"],
                                layer=int(layer), kind=kind, phase=phase)
                # head=-1은 모든 KV head를 합친 분포이며, 나머지는 개별 head 분포이다.
                for head in [-1] + list(range(bits.shape[0])):
                    part = bits[:, begin:end] if head == -1 else bits[head, begin:end]
                    h = histogram(part)
                    histograms[f"{meta['id']}_{path.stem}_{phase}_head{head}"] = h
                    distributions.append(dict(**identity, head=head, **describe(h), fixed_start=fixed,
                                              fixed_p7=float(h[fixed:fixed+7].sum() / h.sum())))

                # --- 4. 저장량 추정: 고정 구간과 타일별 동적 구간을 같은 포맷으로 비교 ---
                for policy in ("layer_fixed", "tile_dynamic"):
                    total, raw, fallback, count, residual = 0, 0, 0, 0, 0
                    rows = ((end - begin) // 64) * 64
                    cols = (bits.shape[2] // 64) * 64
                    for head in range(bits.shape[0]):
                        for token in range(begin, begin + rows, 64):
                            for dim in range(0, cols, 64):
                                tile = bits[head, token:token+64, dim:dim+64]
                                start = fixed if policy == "layer_fixed" else best_start(histogram(tile))
                                size = tile_storage(tile, start)
                                blocks.append(dict(**identity, policy=policy, head=head, token_start=token,
                                                   dim_start=dim, start=start, **size))
                                total += size["stored_bytes"]
                                raw += size["raw_bytes"]
                                fallback += int(size["fallback"])
                                count += 1
                        # 완성된 64×64 타일 밖의 토큰·차원은 BF16으로 유지한다.
                        tail_elements = (end - begin) * bits.shape[2] - rows * cols
                        residual += residual_storage(tail_elements)
                        raw += tail_elements * 2
                    total += residual
                    summaries.append(dict(**identity, policy=policy, raw_bytes=raw, stored_bytes=total,
                                          ratio=raw/total, saving_fraction=1-total/raw, tiles=count,
                                          fallback_tiles=fallback, residual_bytes=residual))

    # --- 5. 상세 결과 저장: 분포·타일·저장량 표와 원본 histogram 출력 ---
    write_csv(output / "distributions.csv", distributions)
    if blocks:
        write_csv(output / "blocks.csv", blocks)
    write_csv(output / "storage.csv", summaries)
    np.savez_compressed(output / "histograms.npz", **histograms)

    # --- 6. 요약 보고서: evaluation 바이트를 합산해 전체 절감률 계산 ---
    # 샘플별 절감률의 단순 평균 대신 원본 바이트 수에 따른 가중 결과를 보고한다.
    report = ["# Experiment A report", "", f"Run: {provenance['run_id']} ({provenance['timezone']})", "",
              "Logical storage estimate, not GPU allocator usage.",
              "Fixed intervals are fit on calibration samples only. Dynamic intervals are per 64x64 tile.",
              "Record: 16 B header, 32 B median offsets, 16 B global offsets, 1536 B bitmaps;",
              "payload aligned to 16 B. Raw fallback: original bytes + 16 B header.",
              "Residual region: BF16 + 16 B per nonempty head/phase residual.", "",
              "| Evaluation policy | Original bytes | Stored bytes | Saving |", "|---|---:|---:|---:|"]
    for policy in ("layer_fixed", "tile_dynamic"):
        subset = [r for r in summaries if r["split"] == "evaluation" and r["policy"] == policy]
        original = sum(r["raw_bytes"] for r in subset)
        stored = sum(r["stored_bytes"] for r in subset)
        report.append(f"| {policy} | {original} | {stored} | {1-stored/original:.2%} |")
    if any(m.get("repeated_smoke_input") for _, m in samples):
        report += ["", "WARNING: repeated short smoke inputs are present; these results are not representative."]
    (output / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Result root; source run ID and analysis/ are added automatically")
    args = p.parse_args()
    analyze(args.input, args.output / source_metadata(args.input)["run_id"] / "analysis")
