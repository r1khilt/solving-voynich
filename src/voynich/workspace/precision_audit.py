"""Prespecified sensitivity to head-row reconstruction, without replacing the lens."""

import json
from pathlib import Path
import time

import numpy as np

from .campaign import digest, load_config, write_json
from .lens_analysis import row_cosine
from .mlx_backend import QwenWorkspace


def main():
    config = load_config('configs/jspace0001.json')
    output, result = Path(config['output']), Path(config['results'])
    if (result / 'head-transport-sensitivity.json').exists():
        raise FileExistsError('Precision sensitivity already measured')
    fit = json.loads((result / 'fit.json').read_text())
    if not fit['complete'] or digest(output / 'derivatives.npy') != fit['derivatives_sha256']:
        raise ValueError('Need the complete unchanged derivative archive')
    inputs = json.loads((result / 'inputs.json').read_text())
    model = QwenWorkspace(config['model_snapshot'], dense_transport=True)
    mx = model.mx
    mx.set_memory_limit(config['memory_bytes_cap'])
    token_ids = [r['token_id'] for r in inputs['dictionary'][:10]]
    old = model.output_rows(token_ids)
    head, ids = model.model.lm_head, mx.array(token_ids)
    new = mx.dequantize(head.weight[ids], head.scales[ids].astype(mx.float32), head.biases[ids].astype(mx.float32),
                        group_size=head.group_size, bits=head.bits, mode=head.mode)
    new = new * model.model.model.norm.weight.astype(mx.float32)
    mx.eval(new)
    correction = np.array(new)-old
    windows = json.loads((output / 'calibration-windows.json').read_text())
    if digest(output / 'calibration-windows.json') != inputs['calibration_sha256']:
        raise ValueError('Calibration windows changed')
    original = np.load(output / 'derivatives.npy', mmap_mode='r')
    indices = list(range(0, 512, 16))
    started, rows, corrections = time.monotonic(), [], []
    for index in indices:
        parts = []
        for start in range(0, 10, 4):
            if time.monotonic()-started >= 300:
                break
            parts.append(model.jacobian_rows(windows[index]['token_ids'], correction[start:start+4], tuple(config['layers']), skip_first=config['skip_first']))
        if len(parts) != 3:
            break
        delta = np.concatenate(parts, axis=1)
        baseline = np.asarray(original[index, :, :10], dtype=np.float64)
        relative = np.linalg.norm(delta, axis=-1)/np.maximum(np.linalg.norm(baseline, axis=-1), 1e-20)
        cosine = row_cosine(baseline, baseline+delta)
        rows.append({'calibration_index': index, 'relative_transport_change': relative.tolist(), 'transport_cosine': cosine.tolist()})
        corrections.append(delta)
        if len(rows) % 8 == 0:
            print(json.dumps({'phase': 'head_precision_audit', 'completed': len(rows), 'seconds': time.monotonic()-started}), flush=True)
    np.save(output / 'head-row-transport-corrections.npy', np.stack(corrections) if corrections else np.empty((0,)))
    report = {'complete': len(rows) == len(indices), 'completed': len(rows), 'planned_indices': indices,
              'seconds': time.monotonic()-started, 'peak_memory_bytes': mx.get_peak_memory(),
              'input_row_relative_change': (np.linalg.norm(correction, axis=-1)/np.linalg.norm(old, axis=-1)).tolist(),
              'rows': rows, 'corrections_sha256': digest(output / 'head-row-transport-corrections.npy'),
              'fit_sha256': digest(result / 'fit.json'), 'source_sha256': digest(__file__),
              'interpretation': 'Sensitivity audit only. Primary calibrated directions and causal selection are unchanged. No guarantee of identical generated answers follows from small angular changes.'}
    write_json(result / 'head-transport-sensitivity.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'planned_indices')}), flush=True)


if __name__ == '__main__':
    main()
