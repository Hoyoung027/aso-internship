#!/usr/bin/env python3
"""Reuse the synthetic Split-K figure layout for each real-weight model/block."""

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from plot_splitk_speedup import (
    calculate_speedups, load_config, model_layers, plot_model, read_csv,
    svg_to_png,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--png-scale', type=float, default=2.0)
    args = parser.parse_args()
    if not math.isfinite(args.png_scale) or args.png_scale <= 0:
        raise ValueError('PNG scale must be positive and finite')
    root = args.results_root.resolve()
    config = load_config(root / 'experiments.json')
    batches = config['matrix']['batches']
    splits = config['matrix']['split_k_candidates']
    trials = config['phases']['tune']['trials']
    assert 1 in splits, 'Split-K=1 is required as the baseline'
    groups = defaultdict(list)
    for row in read_csv(root / 'result_all.csv'):
        if row['phase'] != 'tune' or row['model'] not in args.models:
            continue
        if row['status'] != 'ok' or 'synthetic' in row['weight_source'].lower():
            raise ValueError('Expected successful real-weight tuning rows')
        groups[(row['model'], int(row['block_index']))].append(row)
    assert set(args.models) == {m for m, _ in groups}, 'Requested model is missing'

    # Validate every figure before writing any output. Never pool different blocks.
    figures = []
    for (model, block), rows in sorted(groups.items()):
        values = defaultdict(list)
        trial_ids = defaultdict(set)
        shapes = {}
        for row in rows:
            key = (model, row['layer'], int(row['N']), int(row['split_k']))
            value = float(row['zipgemm_latency_ms'])
            assert math.isfinite(value) and value > 0, f'Invalid latency: {key}'
            values[key].append(value)
            trial_ids[key].add(int(row['trial']))
            shape = (int(row['M']), int(row['K']))
            shape_key = (model, row['layer'])
            assert shapes.get(shape_key, shape) == shape, f'Shape mismatch: {key}'
            shapes[shape_key] = shape
        spec = next(m for m in config['models'] if m['id'] == model)
        expected_layers = {l['id'] for l in spec['layers']
                           if l.get('block_scoped', True) or block == 0}
        layers = model_layers(config, model, expected_layers)
        expected = {(model, l, n, s) for l in layers for n in batches for s in splits}
        assert set(values) == expected, f'Missing/extra candidates: {model} block {block}'
        for key in expected:
            assert len(values[key]) == trials, f'Trial count mismatch: {key}'
            assert trial_ids[key] == set(range(1, trials + 1)), f'Duplicate/missing trial: {key}'
        latencies = {key: statistics.median(v) for key, v in values.items()}
        figures.append((model, block, layers, calculate_speedups(latencies), shapes))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model, block, layers, speedups, shapes in figures:
        png = args.output_dir / f'{model}_block-{block}_splitk_speedup.png'
        svg = png.with_suffix('.svg')
        plot_model(
            svg, model, layers, batches, splits, speedups, shapes,
            title=f'{model} · Block {block} — Real-weight Split-K speedup',
            measurement_note=f' · median of {trials} trials',
        )
        try:
            svg_to_png(svg, png, args.png_scale)
        finally:
            svg.unlink(missing_ok=True)
        print(f'Created: {png.resolve()}')
    print(f'Source: {root / "result_all.csv"}')


if __name__ == '__main__':
    main()
