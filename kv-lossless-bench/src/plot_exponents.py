"""저장된 histogram으로 지수 값별 비율을 그린다. GPU 재실행은 필요 없다."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def aggregate_histograms(directory, split="evaluation", phase="both"):
    # --- 1. 집계: head=-1만 사용해 개별 head histogram과의 중복 합산을 방지 ---
    with (directory / "distributions.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split and r["head"] == "-1"
                and (phase == "both" or r["phase"] == phase)]
    if not rows:
        raise ValueError("No distributions match the requested split/phase")
    groups, seen = {}, set()
    with np.load(directory / "histograms.npz", allow_pickle=False) as archive:
        for row in rows:
            layer, kind = int(row["layer"]), row["kind"]
            key = f"{row['sample']}_layer_{layer:02d}_{kind}_{row['phase']}_head-1"
            if key in seen:
                raise ValueError(f"Duplicate histogram: {key}")
            seen.add(key)
            hist = archive[key]
            if hist.shape != (256,) or (hist < 0).any() or int(hist.sum()) != int(row["elements"]):
                raise ValueError(f"Histogram/count mismatch: {key}")
            groups.setdefault(layer, {}).setdefault(kind, np.zeros(256, dtype=np.int64))
            groups[layer][kind] += hist
    if any(set(group) != {"K", "V"} for group in groups.values()):
        raise ValueError("Each layer must have both K and V histograms")
    return groups, len({r["sample"] for r in rows})


def proportions(hist):
    # 각 지수 count를 전체 count로 나눈다. 확대 그림에서도 분모는 바꾸지 않는다.
    total = int(hist.sum())
    if total <= 0:
        raise ValueError("Empty histogram")
    return hist / total


def display_limits(histograms, full=False):
    # --- 2. 표시 범위: 희귀한 꼬리 때문에 중심 분포가 눌리지 않도록 확대본도 제공 ---
    # 전체 범위 그림은 관측된 모든 지수를 표시한다. 원본 CSV는 항상 0~255 전체이다.
    lows, highs = [], []
    for hist in histograms:
        if full:
            nz = np.flatnonzero(hist)
            lows.append(int(nz[0]))
            highs.append(int(nz[-1]))
        else:
            cdf = np.cumsum(proportions(hist))
            lows.append(int(np.searchsorted(cdf, 0.0005)))
            highs.append(int(np.searchsorted(cdf, 0.9995)))
    return max(0, min(lows)-1), min(255, max(highs)+1)


def compression_savings(directory, split, phase):
    # 그림과 같은 입력·구간의 고정 방식 저장량을 합산한다. 비율을 단순 평균하지 않는다.
    totals = {}
    with (directory / "storage.csv").open() as f:
        for row in csv.DictReader(f):
            if row["split"] != split or row["policy"] != "layer_fixed":
                continue
            if phase != "both" and row["phase"] != phase:
                continue
            for layer in ("all", int(row["layer"])):
                pair = totals.setdefault(layer, {}).setdefault(row["kind"], [0, 0])
                pair[0] += int(row["raw_bytes"])
                pair[1] += int(row["stored_bytes"])
    return {layer: {kind: 1-stored/raw for kind, (raw, stored) in kinds.items()}
            for layer, kinds in totals.items()}


def make_figure(histograms, title, savings, full=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    limits = display_limits(list(histograms.values()), full)
    fig, axes = plt.subplots(1, len(histograms), figsize=(6*len(histograms), 5.7),
                             sharey=True, squeeze=False)
    colors = {"K": "#1e567d", "V": "#bf6a35", "K+V": "#497b68"}
    for ax, (kind, hist) in zip(axes[0], histograms.items()):
        prob = proportions(hist)
        nonzero = prob[prob > 0]
        entropy = float(-(nonzero*np.log2(nonzero)).sum())
        ax.bar(np.arange(256), prob, width=0.82, color=colors[kind], zorder=3)
        ax.set_xlim(limits[0]-0.6, limits[1]+0.6)
        ax.set_ylim(0, 0.30)
        ax.set_title({"K": "Key (K)", "V": "Value (V)", "K+V": "Key + Value"}[kind], fontsize=17)
        ax.set_xlabel("Exponent Value (BF16 field, 0–255)", fontsize=13)
        if limits[1]-limits[0] <= 30:
            ax.set_xticks(np.arange(limits[0], limits[1]+1))
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
        ax.grid(alpha=0.25, linestyle="--", zorder=0)
        shown = prob[limits[0]:limits[1]+1].sum()
        ax.text(0.035, 0.965, f"Entropy: {entropy:.2f} bits\n"
                f"Original: 8 bits\nCompression: {savings[kind]:.2%}\nShown mass: {shown:.3%}",
                transform=ax.transAxes, va="top", fontsize=11,
                bbox=dict(facecolor="white", edgecolor="#aaaaaa", alpha=0.9, boxstyle="round,pad=0.4"))
    axes[0, 0].set_ylabel("Proportion", fontsize=15)
    fig.suptitle(title, fontsize=18, y=0.985)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.085, right=0.975, wspace=0.13)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="analysis/ directory")
    p.add_argument("--split", choices=("calibration", "evaluation"), default="evaluation")
    p.add_argument("--phase", choices=("both", "prefill", "decode"), default="both")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    provenance = json.loads((args.input / "run_metadata.json").read_text())
    run_id = provenance["run_id"]
    groups, samples = aggregate_histograms(args.input, args.split, args.phase)
    savings = compression_savings(args.input, args.split, args.phase)
    totals = {kind: sum((g[kind] for g in groups.values()), np.zeros(256, dtype=np.int64)) for kind in ("K", "V")}
    source_run = args.input.parent / "kv" / "run.json"
    model = "Model"
    if source_run.exists():
        model = Path(json.loads(source_run.read_text())["model"]).name
    prefix = f"{run_id}_{args.split}_{args.phase}_exponents"
    destination = args.input / "exponent_plots"
    destination.mkdir(exist_ok=True)

    # --- 3. 전체 레이어와 지정 레이어: K/V를 나란히 배치한 PNG 4개만 저장 ---
    selected_layers = [1, 20, 31]
    missing = set(selected_layers) - set(groups)
    if missing:
        raise ValueError(f"Requested layers are missing: {sorted(missing)}")
    fig = make_figure(totals, f"{model}\n(All {len(groups)} layers)", savings["all"])
    fig.savefig(destination / f"{prefix}_all_layers.png", dpi=180)
    plt.close(fig)
    for layer in selected_layers:
        fig = make_figure(groups[layer], f"{model}\n(Layer {layer})", savings[layer])
        fig.savefig(destination / f"{prefix}_layer_{layer:02d}.png", dpi=180)
        plt.close(fig)

    # --- 4. 검증 가능한 원시 표: 모든 지수의 count와 전체 원소 대비 proportion ---
    with (args.input / f"{prefix}_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "split", "phase", "layer", "kind", "exponent", "count", "total", "proportion"])
        for layer, histograms in [("all", {**totals, "K+V": totals["K"]+totals["V"]})] + list(sorted(groups.items())):
            for kind, hist in histograms.items():
                prob = proportions(hist)
                for exponent in range(256):
                    writer.writerow([run_id, args.split, args.phase, layer, kind, exponent,
                                     int(hist[exponent]), int(hist.sum()), float(prob[exponent])])
    (args.input / f"{prefix}_metadata.json").write_text(json.dumps({**provenance,
        "split": args.split, "phase": args.phase, "samples": samples, "layers": sorted(groups), "plotted_layers": selected_layers,
        "normalization": "sum counts / total elements; head=-1 only; K and V separately unless K+V",
        "axis": "raw 8-bit BF16 exponent field, not exponent-minus-127 or tensor value",
        "compression": "1 - sum(stored_bytes)/sum(raw_bytes), layer_fixed, matching split/phase; includes metadata",
        "zoom": "union of per-panel 0.0005..0.9995 quantiles plus 1 bin; no renormalization",
        "counts": {kind: int(hist.sum()) for kind, hist in totals.items()}}, indent=2)+"\n")
    print(f"Saved exponent plots: {destination}")


if __name__ == "__main__":
    main()
