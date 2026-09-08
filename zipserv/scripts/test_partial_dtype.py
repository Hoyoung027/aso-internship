"""CPU-only regression checks for partial-dtype plumbing and result isolation."""
import argparse
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collect_results as collector
import run_experiments as runner


class PartialDtypeTests(unittest.TestCase):
    def setUp(self):
        self.config = runner.load_config(runner.DEFAULT_CONFIG)
        self.config['input']['weight_source'] = 'synthetic'
        self.case = runner.Case('llama3.1-8b', 'o_proj', 64, 512, 8, 2, 1, None)

    def execute(self, dtype, reported=None, workspace=None, split=2):
        self.config['partial_dtype'] = dtype
        case = runner.Case('llama3.1-8b', 'o_proj', 64, 512, 8, split, 1, None)
        expected = 64 * 8 * split * (4 if dtype == 'fp32' else 2) if split > 1 else 0
        log = (f'Partial dtype: {reported or dtype}\n'
               f'Partial workspace bytes: {expected if workspace is None else workspace}\n'
               '========== Test Complete ==========\n')
        metrics = {name: {'latency_ms': 1, 'tflops': 1} for name in ('cublas', 'cublas_tc', 'zipgemm')}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, log)) as launch:
                with patch.object(runner, 'parse_metrics', return_value=metrics):
                    row = runner.execute_case('run', case, self.config, {}, Path(directory) / 'test.log')
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--partial-dtype') + 1], dtype)
            return row

    def test_workspace_size_and_option(self):
        for dtype, size in [('bf16', 2048), ('fp32', 4096)]:
            row = self.execute(dtype)
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['partial_dtype'], dtype)
            self.assertEqual(row['partial_workspace_bytes'], size)

    def test_unsplit_has_no_workspace(self):
        for dtype in ('bf16', 'fp32'):
            row = self.execute(dtype, split=1)
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['partial_workspace_bytes'], 0)

    def test_stale_or_wrong_binary_fails(self):
        self.assertEqual(self.execute('fp32', reported='bf16')['status'], 'failed')
        self.assertEqual(self.execute('fp32', workspace=2048)['status'], 'failed')

    def test_collector_rejects_mixed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dtype in ('bf16', 'fp32'):
                path = root / 'raw' / dtype / 'result.csv'
                path.parent.mkdir(parents=True)
                with path.open('w') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['partial_dtype'])
                    writer.writeheader()
                    writer.writerow({'partial_dtype': dtype})
            with self.assertRaisesRegex(ValueError, 'Mixed partial dtypes'):
                collector.collect_rows(root)

    def test_resume_cannot_silently_skip_other_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'result.csv').open('w') as handle:
                # Old CSV without the new column means BF16.
                handle.write('status\nok\n')
            args = argparse.Namespace(config=runner.DEFAULT_CONFIG, partial_dtype='fp32', models=['llama3.1-8b'],
                                      layers=['o_proj'], mode='tune', block_index=0,
                                      output_dir=root, log_file=root / 'test.log')
            with patch.object(runner, 'parse_args', return_value=args), patch.object(runner, 'make_tune_cases', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'Cannot mix partial dtypes'):
                    runner.main()


if __name__ == '__main__':
    unittest.main()
