from __future__ import annotations

import argparse
import gzip
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import analyze_moe_traces  # noqa: E402
import run_backend_experiments  # noqa: E402
import run_moe_requests  # noqa: E402
import visualize_moe_results  # noqa: E402


class PromptTests(unittest.TestCase):
    def make_args(self, num_tokens: int) -> argparse.Namespace:
        return argparse.Namespace(
            num_tokens=num_tokens,
            master_num_tokens=8192,
            prompt_token_min=1000,
            prompt_token_max=100000,
            seed=42,
        )

    def test_short_prompt_is_prefix_of_long_prompt(self) -> None:
        short = run_moe_requests.build_prompt(self.make_args(128))
        long = run_moe_requests.build_prompt(self.make_args(8192))
        self.assertEqual(short, long[:128])
        self.assertEqual(len(short), 128)
        self.assertEqual(len(long), 8192)


class TraceAnalysisTests(unittest.TestCase):
    def test_analyze_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trace_path = root / "worker.pt.trace.json.gz"
            events = []
            for index, (router_us, fused_us) in enumerate(
                [(1000, 4000), (2000, 5000), (3000, 6000)]
            ):
                base = index * 10000
                events.extend(
                    [
                        {
                            "cat": "gpu_user_annotation",
                            "ph": "X",
                            "name": "GPTOSS_ROUTER_L0_M128",
                            "ts": base,
                            "dur": router_us,
                        },
                        {
                            "cat": "gpu_user_annotation",
                            "ph": "X",
                            "name": "GPTOSS_FUSED_MOE_L0_M128",
                            "ts": base + router_us,
                            "dur": fused_us,
                        },
                    ]
                )
            with gzip.open(trace_path, "wt", encoding="utf-8") as file:
                json.dump({"traceEvents": events}, file)

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "versions": {
                            "vllm": "0.23.0",
                            "torch": "2.11.0+cu130",
                            "flashinfer": "0.6.12",
                            "vllm_git_tag": "v0.23.0",
                        },
                        "gpu": {
                            "name": "RTX PRO 6000",
                            "uuid": "GPU-test",
                            "driver_version": "580.0",
                        },
                        "server_startup_seconds": 10.0,
                    }
                ),
                encoding="utf-8",
            )

            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "timestamp_utc": "20260727T000000000000Z",
                        "experiment_id": "marlin_w4a16",
                        "backend_requested": "marlin",
                        "backend_actual": "marlin",
                        "activation_dtype": "bf16",
                        "flashinfer_autotune": False,
                        "layer": 0,
                        "num_tokens": 128,
                        "warmup": 2,
                        "repeat": 3,
                        "seed": 42,
                        "prompt_sha256": "test",
                        "host": "node-test",
                        "slurm_job_id": "1",
                        "client_seconds": [0.01, 0.02, 0.03],
                        "trace_files": [str(trace_path)],
                        "server_log": str(root / "server.log"),
                        "run_manifest": str(manifest_path),
                    }
                ),
                encoding="utf-8",
            )

            row = analyze_moe_traces.analyze_metadata(metadata_path)
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["router_event_count"], 3)
            self.assertEqual(row["fused_moe_event_count"], 3)
            self.assertAlmostEqual(row["router_gpu_ms_median"], 2.0)
            self.assertAlmostEqual(row["fused_moe_gpu_ms_median"], 5.0)
            self.assertAlmostEqual(row["moe_total_gpu_ms_mean"], 7.0)
            self.assertAlmostEqual(row["fused_moe_tokens_per_second"], 25600.0)

    def test_main_reports_missing_expected_combination(self) -> None:
        rows = [
            {
                "experiment_id": "marlin_w4a16",
                "num_tokens": 1,
                "status": "ok",
                "error": "",
            }
        ]
        missing = analyze_moe_traces.add_missing_expected_rows(
            rows,
            {
                "expected_experiment_ids": [
                    "marlin_w4a16",
                    "humming_indexed_w4a16",
                ],
                "expected_num_tokens": [1, 2],
                "expected_warmup": 20,
                "expected_repeat": 50,
            },
        )
        self.assertEqual(missing, 3)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            sum(row["status"] == "missing" for row in rows),
            3,
        )


class BackendSelectionTests(unittest.TestCase):
    def test_validate_flashinfer_autotune_off(self) -> None:
        backend = {
            "id": "flashinfer_cutlass_w4a8_tune_off",
            "requested_backend": "flashinfer_cutlass",
            "expected_backend": "FLASHINFER_CUTLASS_MXFP4_MXFP8",
            "flashinfer_autotune": False,
            "environment": {},
        }
        with TemporaryDirectory() as temporary_directory:
            server_log = Path(temporary_directory) / "server.log"
            server_log.write_text(
                "Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8' Mxfp4 MoE backend.\n"
                "Skipping FlashInfer autotune because it is disabled.\n",
                encoding="utf-8",
            )
            actual = run_backend_experiments.validate_backend_selection(
                backend, server_log
            )
        self.assertEqual(actual, "FLASHINFER_CUTLASS_MXFP4_MXFP8")

    def test_validate_humming_grouped(self) -> None:
        backend = {
            "id": "humming_grouped_w4a16",
            "requested_backend": "humming",
            "expected_backend": "HUMMING",
            "flashinfer_autotune": False,
            "environment": {"VLLM_HUMMING_MOE_GEMM_TYPE": "grouped"},
        }
        with TemporaryDirectory() as temporary_directory:
            server_log = Path(temporary_directory) / "server.log"
            server_log.write_text(
                "Using 'HUMMING' Mxfp4 MoE backend.\n"
                "Using grouped_contiguous gemm for humming moe\n",
                encoding="utf-8",
            )
            actual = run_backend_experiments.validate_backend_selection(
                backend, server_log
            )
        self.assertEqual(actual, "HUMMING")

    def test_reject_backend_mismatch(self) -> None:
        backend = {
            "id": "marlin_w4a16",
            "requested_backend": "marlin",
            "expected_backend": "MARLIN",
            "flashinfer_autotune": False,
            "environment": {},
        }
        with TemporaryDirectory() as temporary_directory:
            server_log = Path(temporary_directory) / "server.log"
            server_log.write_text(
                "Using 'EMULATION' Mxfp4 MoE backend.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Expected backend marker"):
                run_backend_experiments.validate_backend_selection(
                    backend, server_log
                )


class VisualizationTests(unittest.TestCase):
    def test_default_output_dir_uses_input_run_name(self) -> None:
        output = visualize_moe_results.default_output_dir(
            Path("/lustre/example/runs/job-123/moe_kernel_results.csv")
        )
        self.assertEqual(
            output,
            PROJECT_DIR / "results" / "plots" / "job-123",
        )

    def test_mean_throughput_uses_mean_latency(self) -> None:
        throughput = visualize_moe_results.mean_throughput(
            {"num_tokens": 128, "fused_moe_gpu_ms_mean": 2.0}
        )
        self.assertEqual(throughput, 64000.0)

    def test_index_results_rejects_missing_shape(self) -> None:
        rows = [
            {"experiment_id": "marlin_w4a16", "num_tokens": 1},
            {"experiment_id": "marlin_w4a16", "num_tokens": 2},
            {"experiment_id": "humming_indexed_w4a16", "num_tokens": 1},
        ]
        with self.assertRaisesRegex(ValueError, "missing token counts"):
            visualize_moe_results.index_results(rows)


if __name__ == "__main__":
    unittest.main()
