"""Plot evaluation-only layer/KV savings and tile p7 distributions."""
import argparse
import csv
import json
from pathlib import Path


def main():
    # --- 1. 입력·출력 준비: GUI 없는 서버에서 그림 파일을 생성 ---
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    provenance = json.loads((args.input / "run_metadata.json").read_text())
    run_id = provenance["run_id"]
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Calibration 결과를 제외하고 평가 데이터만 시각화한다.
    with (args.input / "storage.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "evaluation"]
    layers = sorted({int(r["layer"]) for r in rows})

    # --- 2. Heatmap: phase별 레이어×K/V 절감률을 두 선택 정책에서 비교 ---
    for phase in sorted({r["phase"] for r in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(10, max(4, len(layers)*0.2)), constrained_layout=True)
        for ax, policy in zip(axes, ("layer_fixed", "tile_dynamic")):
            values = np.full((len(layers), 2), np.nan)
            for i, layer in enumerate(layers):
                for j, kind in enumerate(("K", "V")):
                    group = [r for r in rows if int(r["layer"]) == layer and r["kind"] == kind
                             and r["policy"] == policy and r["phase"] == phase]
                    if group:
                        # 샘플 크기가 달라도 올바르게 합산되도록 바이트 총량으로 계산한다.
                        values[i, j] = 100*(1-sum(int(r["stored_bytes"]) for r in group)/sum(int(r["raw_bytes"]) for r in group))
            im = ax.imshow(values, aspect="auto", vmin=-2, vmax=32, cmap="viridis")
            ax.set(xticks=[0, 1], xticklabels=["K", "V"], yticks=range(len(layers)),
                   yticklabels=layers, ylabel="Layer", title=f"{phase}: {policy}")
            fig.colorbar(im, ax=ax, label="Estimated storage saving (%)")
        fig.suptitle(run_id, fontsize=9)
        fig.savefig(args.input / f"{run_id}_savings_{phase}.png", dpi=160)
        plt.close(fig)

    # --- 3. 분포 그래프: 동적 지수 구간의 타일별 p7을 K/V별로 표시 ---
    if (args.input / "blocks.csv").exists():
        with (args.input / "blocks.csv").open() as f:
            blocks = [r for r in csv.DictReader(f) if r["split"] == "evaluation" and r["policy"] == "tile_dynamic"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for kind in ("K", "V"):
            values = [float(r["p_selected"]) for r in blocks if r["kind"] == kind]
            if values:
                ax.hist(values, bins=np.linspace(0, 1, 41), alpha=0.5, label=kind)
        ax.set(xlabel="Best contiguous 7-exponent coverage per tile", ylabel="Tile count")
        ax.legend()
        ax.set_title(run_id, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.input / f"{run_id}_tile_p7.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
