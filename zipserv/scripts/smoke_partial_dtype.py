"""GPU smoke test: old/new BF16 regression, FP32 workspace, unsplit identity.

Run inside a GPU allocation; no Slurm jobs are submitted by this script.
"""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

from run_experiments import PROJECT_ROOT, load_config, parse_metrics, runtime_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-bin', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT_ROOT / 'configs/experiments.json')
    env = runtime_env(config)
    records = []

    def measure(label, n, split, dtype=None, real=False):
        binary_dir = args.baseline_bin.resolve() if label == 'old_bf16' else PROJECT_ROOT / 'bin'
        current_env = dict(env, LD_LIBRARY_PATH=f"{binary_dir}:{config['paths']['cuda_path']}/lib64")
        m, k = (4096, 14336) if real else (256, 512)
        command = [str(binary_dir / 'test_mm_final'), str(m), str(k), str(n), str(split), '--seed', '12345']
        if dtype:
            command.extend(['--partial-dtype', dtype])
        if real:
            spec = next(model for model in config['models'] if model['id'] == 'llama3.1-8b')
            command.extend(['--model-dir', spec['weight']['model_dir'], '--block-index', '0',
                            '--weight-tensor', 'model.layers.0.mlp.down_proj.weight'])
        with tempfile.TemporaryDirectory(prefix='zipserv-partial-smoke-') as directory:
            completed = subprocess.run(command, cwd=directory, env=current_env, capture_output=True, text=True, timeout=180)
            log = completed.stdout + completed.stderr
            name = f'{"real" if real else "synthetic"}-{label}-n{n}-s{split}'
            (args.output_dir / f'{name}.log').write_text(log)
            assert completed.returncode == 0 and '========== Test Complete ==========' in log, name
            metrics = parse_metrics(Path(directory) / 'bf16_triplebm_res.csv', log)
        assert all(math.isfinite(metrics[method]['latency_ms']) and metrics[method]['latency_ms'] > 0
                   for method in ('cublas', 'cublas_tc', 'zipgemm'))
        errors = {key: value for key, value in metrics.items() if 'error' in key}
        assert errors and all(math.isfinite(v) and v >= 0 for v in errors.values())
        if dtype:
            expected = m * n * split * (4 if dtype == 'fp32' else 2) if split > 1 else 0
            assert f'Partial dtype: {dtype}\n' in log
            assert f'Partial workspace bytes: {expected}\n' in log
        records.append(dict(case=name, errors=errors))
        print(f'{name}: MAE vs TC={errors["zip_vs_tc_total_absolute_error"]/(m*n):.9g}', flush=True)
        return errors

    # Fast kernels (N=8/16/32), Safe kernel (N=9), all Split-K candidates.
    cases = [(n, s) for n in (8, 16, 32, 9) for s in (1, 4)] + [(8, 2), (8, 8)]
    for n, split in cases:
        old = measure('old_bf16', n, split)
        bf16 = measure('new_bf16', n, split, 'bf16')
        fp32 = measure('new_fp32', n, split, 'fp32')
        assert old == bf16, f'BF16 numerical regression: N={n}, split={split}'
        if split == 1:
            assert bf16 == fp32, f'Unsplit paths differ: N={n}'
    for dtype in ('bf16', 'fp32'):
        measure(f'real_{dtype}', 8, 8, dtype, real=True)
    (args.output_dir / 'checks.json').write_text(json.dumps(records, indent=2) + '\n')
    print('PASS: BF16 regression, Split-K=1 identity, FP32 Fast/Safe paths, real-weight load, workspace sizes.', flush=True)


if __name__ == '__main__':
    main()
