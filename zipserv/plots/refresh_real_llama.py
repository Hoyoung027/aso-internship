#!/usr/bin/env python3
"""Regenerate all real LLaMA figures from one explicit tuning/final experiment pair.

Use a fresh --output-dir to inspect the complete set before replacing older images.
Block 0 retains the historical filenames; other blocks have explicit suffixes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import plot_latency_by_splitk as latency
import plot_llama_error as error
import plot_results as legacy

BLOCKS = {"llama3.1-8b": (0, 16, 31), "llama3.1-70b": (0, 40, 79)}
PLOTS = Path(__file__).resolve().parent


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def case_key(row):
    return row['model'], int(row['block_index']), row['layer'], int(row['N'])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    tune, run, out = (p.resolve() for p in (args.tuning_root, args.run_root, args.output_dir))
    config = json.loads((run / 'experiments.json').read_text())
    tune_config = json.loads((tune / 'experiments.json').read_text())
    if config != tune_config:
        raise ValueError('Tuning/final configuration snapshots differ')
    specs = {m['id']: m for m in config['models'] if m['id'] in BLOCKS}
    batches = config['matrix']['batches']
    splits = config['matrix']['split_k_candidates']
    assert batches == [8, 16, 32] and splits == [1, 2, 4, 8]
    for phase in ('tune', 'final'):
        assert config['phases'][phase] == dict(warmup=100, repeat=1000, trials=3)
    expected = {(m, b, l['id'], n) for m, blocks in BLOCKS.items() for b in blocks
                for l in specs[m]['layers'] if b == 0 or l.get('block_scoped', True)
                for n in batches}
    selection_rows = read(tune / 'selected_splitk.csv')
    selected = {case_key(r): int(r['split_k']) for r in selection_rows}
    assert len(selection_rows) == len(selected) == 78 and set(selected) == expected
    for phase, root in [('tune', tune), ('run', run)]:
        rows = read(root / 'result_all.csv')
        assert all(r['status'] == 'ok' and r['phase'] == phase and r['returncode'] == '0'
                   and r['weight_source'] == 'real_safetensors'
                   and (r['warmup'], r['repeat']) == ('100', '1000') for r in rows)
        keys = Counter((*case_key(r), int(r['split_k']), int(r['trial'])) for r in rows)
        wanted = {(*k, s, t) for k in expected
                  for s in (splits if phase == 'tune' else [selected[k]]) for t in (1, 2, 3)}
        assert set(keys) == wanted and all(v == 1 for v in keys.values())
        for r in rows:
            assert error.absolute_exceedance_percent([r], int(r['M']) * int(r['N'])) is not None
    print('Validated: 936 tuning rows, 234 final rows, 78 selections; absolute metrics complete.', flush=True)
    out.mkdir(parents=True, exist_ok=True)

    def render(svg, draw):
        draw(svg)
        legacy.svg_to_png(svg, svg.with_suffix('.png'), 2.0)
        svg.unlink()
        print(f'Created: {svg.with_suffix(".png")}', flush=True)

    # Keep identical error axes across both models AND all three blocks.
    points = {(m, b): error.load_selected_points(run, [m], b)
              for m, blocks in BLOCKS.items() for b in blocks}
    scales = error.metric_scales([p for group in points.values() for p in group])
    assert sum(map(len, points.values())) == 78
    for (model, block), group in points.items():
        layers = [l['id'] for l in specs[model]['layers']
                  if block == 0 or l.get('block_scoped', True)]
        assert {(p.layer, p.n) for p in group} == {(l, n) for l in layers for n in batches}
        suffix = '' if block == 0 else f'_block-{block}'
        render(out / f'{model}{suffix}_error.svg', lambda svg: error.plot_model(
            svg, model, group, layers, batches, scales, block))
        values = latency.load_latencies(tune / 'result_all.csv', block)
        shapes = {l['id']: (l['M'], l['K']) for l in specs[model]['layers']}
        for layer in layers:
            for n in batches:
                chosen = selected[model, block, layer, n]
                assert values[model, layer, n, chosen] == min(values[model, layer, n, s] for s in splits)
        render(out / f'{model}{suffix}_latency_by_splitk.svg', lambda svg: latency.plot_model(
            svg, model, layers, shapes, batches, splits, values,
            title=f'{model} · Block {block} · real weights'))

    def invoke(script, root, destination, *extra):
        subprocess.run([sys.executable, str(PLOTS / script), '--results-root', str(root),
                        '--models', *BLOCKS, '--output-dir', str(destination), *extra], check=True)

    invoke('plot_real_weight_overview.py', run, out)
    invoke('plot_real_layer_speedup.py', run, out, '--output-name', 'llama3.1_layerwise_speedup.png')
    invoke('plot_real_model_average_speedup.py', run, out, '--output-name', 'model_average_speedup.png')
    invoke('plot_real_splitk_speedup.py', tune, out / 'splitk_speedup')

    # Historical compact layer figure remains block 0 only, now explicitly labeled.
    rows = read(run / 'result_all.csv')
    final_points = []
    for m in BLOCKS:
        for layer in specs[m]['layers']:
            for n in batches:
                group = [r for r in rows if case_key(r) == (m, 0, layer['id'], n)]
                final_points.append(legacy.FinalPoint(
                    model=m, layer=layer['id'], m=layer['M'], k=layer['K'], n=n,
                    split_k=selected[m, 0, layer['id'], n], trials=len(group),
                    cublas_tc_ms=statistics.median(float(r['cublas_tc_latency_ms']) for r in group),
                    zipgemm_ms=statistics.median(float(r['zipgemm_latency_ms']) for r in group)))
    render(out / 'llama31_layer_speedup.svg', lambda svg: legacy.plot_llama_layer_speedup(
        svg, final_points, specs, batches, scope_note=' · Block 0 + LM Head'))

    images = sorted(out.rglob('*.png'))
    assert len(images) == 23 and not list(out.rglob('*.svg'))
    inputs = [tune / 'result_all.csv', tune / 'selected_splitk.csv', run / 'result_all.csv',
              tune / 'experiments.json', run / 'experiments.json']
    inputs.extend(sorted({Path(r['log_file']) for r in rows}))
    manifest = dict(generated_at=datetime.now(timezone.utc).isoformat(),
                    tuning_root=str(tune), run_root=str(run), blocks=BLOCKS,
                    note='Block-0 legacy filenames; other blocks explicit. LM Head counted once. '
                         'Model average is the arithmetic mean of 13 operation speedups, not end-to-end speedup.',
                    inputs={str(p): digest(p) for p in inputs},
                    scripts={p.name: digest(p) for p in PLOTS.glob('*.py')},
                    images={str(p.relative_to(out)): digest(p) for p in images})
    (out / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Complete: {len(images)} PNGs and sources.json in {out}', flush=True)


if __name__ == '__main__':
    main()
