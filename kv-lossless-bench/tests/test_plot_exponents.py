import csv
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plot_exponents import aggregate_histograms, proportions


class HistogramAggregationTests(unittest.TestCase):
    def test_no_head_double_count_or_calibration_leak(self):
        # Prefill/decode의 크기가 달라도 count를 합친 뒤 정규화해야 한다.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, histograms = [], {}
            for split, sample in (("evaluation", "eval_with_underscores"), ("calibration", "cal")):
                for phase, count, exponent in (("prefill", 90, 120), ("decode", 10, 121)):
                    for kind in ("K", "V"):
                        for head in (-1, 0):
                            hist = np.zeros(256, dtype=np.int64)
                            hist[exponent] = count
                            histograms[f"{sample}_layer_00_{kind}_{phase}_head{head}"] = hist
                            rows.append(dict(sample=sample, split=split, phase=phase, layer=0,
                                             kind=kind, head=head, elements=count))
            with (root / "distributions.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.savez(root / "histograms.npz", **histograms)
            groups, samples = aggregate_histograms(root)
            self.assertEqual(samples, 1)
            self.assertEqual(int(groups[0]["K"].sum()), 100)
            self.assertEqual(proportions(groups[0]["K"])[120], 0.9)
            self.assertEqual(proportions(groups[0]["V"])[121], 0.1)
            only_decode, _ = aggregate_histograms(root, phase="decode")
            self.assertEqual(int(only_decode[0]["K"].sum()), 10)


if __name__ == "__main__":
    unittest.main()
