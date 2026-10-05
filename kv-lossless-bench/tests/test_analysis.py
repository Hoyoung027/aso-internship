import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_core import exponents, histogram, best_start, tile_storage, residual_storage
from analyze_kv import analyze
from run_metadata import new_metadata


class AnalysisTests(unittest.TestCase):
    # 전체 65,536개 비트 패턴에서 지수 추출과 histogram을 검증한다.
    def test_all_bf16_patterns(self):
        bits = np.arange(65536, dtype=np.uint16)
        np.testing.assert_array_equal(exponents(bits), (bits.astype(np.uint32)//128) % 256)
        np.testing.assert_array_equal(histogram(bits), np.full(256, 256))
        self.assertEqual(best_start(histogram(bits)), 0)

    # 지수 범위의 마지막 유효 구간(249~255)도 선택 가능한지 확인한다.
    def test_last_exponent_window(self):
        h = np.zeros(256, dtype=np.int64)
        h[249:] = 10
        self.assertEqual(best_start(h), 249)

    # 손으로 계산 가능한 입력으로 패딩·압축 크기·원본 fallback을 검증한다.
    def test_known_storage_and_fallback(self):
        all_high = np.full((64, 64), 120 << 7, dtype=np.uint16)
        size = tile_storage(all_high, 116)
        self.assertEqual(size["stored_bytes"], 4096 + 1536 + 64)
        self.assertFalse(size["fallback"])
        size = tile_storage(all_high, 0)
        self.assertEqual(size["candidate_bytes"], 8192 + 1536 + 64)
        self.assertEqual(size["stored_bytes"], 8192 + 16)
        self.assertTrue(size["fallback"])
        all_high[0, 0] = 0
        size = tile_storage(all_high, 116)
        self.assertEqual(size["sign_bytes"], 4096)
        self.assertEqual(size["full_bytes"], 16)
        self.assertEqual(residual_storage(0), 0)
        self.assertEqual(residual_storage(128), 272)

    # 두 split의 지수를 다르게 만들어 평가 데이터가 calibration에 섞이지 않는지 검증한다.
    # 미완성 토큰·차원까지 원본 바이트 합계에 빠짐없이 포함되는지도 확인한다.
    def test_pipeline_and_no_calibration_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "capture"
            source.mkdir()
            (source / "COMPLETE").touch()
            provenance = new_metadata()
            (source / "run.json").write_text(json.dumps(provenance))
            for split, exponent in (("calibration", 100), ("evaluation", 200)):
                dest = source / split
                dest.mkdir()
                # 64 prefill + 65 decode, dimension tail, two heads.
                bits = np.full((2, 129, 65), exponent << 7, dtype=np.uint16)
                for kind in ("K", "V"):
                    np.save(dest / f"layer_00_{kind}.npy", bits)
                (dest / "sample.json").write_text(json.dumps(dict(
                    id=split, split=split, category="test", prefill_tokens=64, decode_tokens=65)))
            output = root / "analysis"
            analyze(source, output)
            starts = json.loads((output / "calibration.json").read_text())
            self.assertEqual(starts["layer_00_K"], 94)
            import csv
            with (output / "storage.csv").open() as f:
                rows = list(csv.DictReader(f))
            selected = [r for r in rows if r["sample"] == "evaluation" and r["policy"] == "layer_fixed"]
            self.assertTrue(all(r["run_id"] == provenance["run_id"] for r in rows))
            self.assertEqual(json.loads((output / "run_metadata.json").read_text()), provenance)
            self.assertEqual(sum(int(r["raw_bytes"]) for r in selected), 2*2*129*65*2)
            self.assertTrue(all(int(r["fallback_tiles"]) == 2 for r in selected))
            self.assertTrue((output / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
