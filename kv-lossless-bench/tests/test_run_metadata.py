import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_metadata import new_metadata, source_metadata


class RunMetadataTests(unittest.TestCase):
    # Slurm 식별자는 KST 초 단위 날짜·실험 종류·실제 job 번호를 모두 포함한다.
    def test_slurm_identity(self):
        with patch.dict(os.environ, {"KV_RUN_TIMESTAMP": "250902_131205", "SLURM_JOB_ID": "12345", "KV_EXPERIMENT": "exp-a"}, clear=True):
            metadata = new_metadata()
        self.assertEqual(metadata["run_id"], "250902_131205_exp-a_job-12345")
        self.assertEqual(metadata["timezone"], "Asia/Seoul")

    # Slurm 밖에서는 가짜 job 번호를 만들지 않고 local-PID를 명시한다.
    def test_local_identity(self):
        with patch.dict(os.environ, {}, clear=True):
            metadata = new_metadata()
        self.assertEqual(metadata["job_id"], f"local-{os.getpid()}")
        self.assertRegex(metadata["timestamp"], r"^\d{6}_\d{6}$")

    def test_invalid_timestamp(self):
        with patch.dict(os.environ, {"KV_RUN_TIMESTAMP": "250902_1312"}, clear=True):
            with self.assertRaises(ValueError):
                new_metadata()

    # 다른 프로세스에서 재분석해도 원본 수집의 출처를 보존한다.
    def test_source_identity_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            with self.assertRaises(ValueError):
                source_metadata(source)
            metadata = new_metadata()
            (source / "run.json").write_text(json.dumps({**metadata, "model": "test"}))
            self.assertEqual(source_metadata(source), metadata)


if __name__ == "__main__":
    unittest.main()
