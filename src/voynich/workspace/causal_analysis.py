"""Compact plots and clustered descriptive checks for completed causal runs."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .campaign import digest, load_config, write_json
from .causal_campaign import rates


def leave_one_pair_out(rows):
    pairs = sorted({r['pair_id'] for r in rows})
    results = {}
    for pair in pairs:
        subset = [r for r in rows if r['pair_id'] != pair]
        groups = defaultdict(list)
        for row in subset:
            groups[row['condition']].append(row)
        results[pair] = {condition: rates(members) for condition, members in groups.items()}
    return results


def audit_observations(rows):
    identity = [r for r in rows if r['condition'] == 'identity' and r['valid']]
    swaps = [r for r in rows if r['condition'] == 'swap' and r['valid']]
    errors = [max(abs(a-b) for a, b in zip(r['geometry']['after_applied_coordinates'], r['geometry']['before'][::-1])) for r in swaps]
    grouped = defaultdict(list)
    for row in swaps:
        if row['expected_effect'] == 'change':
            grouped[row['layer']].append(row)
    layer_readouts = {}
    for layer, members in grouped.items():
        positive = [r for r in members if r['geometry']['before'][0] > r['geometry']['before'][1]]
        negative = [r for r in members if r['geometry']['before'][0] <= r['geometry']['before'][1]]
        layer_readouts[str(layer)] = {'source_coordinate_exceeds_donor': len(positive)/len(members),
                                      'positive_contrast': rates(positive), 'nonpositive_contrast': rates(negative),
                                      'note': 'Post-hoc descriptive strata; the primary score includes both.'}
    return {'identity_text_mismatches': sum(r['answer'] != r['baseline'] for r in identity),
            'max_swap_coordinate_error': max(errors, default=0.),
            'invalid_interventions': sum(not r['valid'] for r in rows),
            'truncated_generations': sum(r.get('truncated', False) for r in rows),
            'layer_readouts': layer_readouts}


def plot(config, development, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'swap': '#2563eb', 'random': '#64748b', 'orthogonal': '#94a3b8', 'raw_swap': '#db2777', 'donor': '#059669'}
    labels = {'swap': 'Jacobian country swap', 'random': 'Matched random', 'orthogonal': 'Orthogonal random',
              'raw_swap': 'Raw output-direction swap', 'donor': 'Full donor state'}
    groups = defaultdict(list)
    for r in development:
        groups[(r['layer'], r['condition'])].append(r)
    layers = config['layers']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for condition, color in colors.items():
        summaries = [rates(groups[(layer, condition)]) for layer in layers]
        y = [s['counterfactual']['rate'] for s in summaries]
        axes[0, 0].plot(np.array(layers)+1, y, marker='o', label=labels[condition], color=color)
        axes[0, 1].plot(np.array(layers)+1, [s['copy_preserved']['rate'] for s in summaries], marker='o', color=color)
        norms = []
        for layer in layers:
            members = [r for r in groups[(layer, condition)] if r['valid'] and r['expected_effect'] == 'change']
            norms.append(np.median([r['geometry']['applied_norm']/max(r['geometry']['hidden_norm'], 1e-12) for r in members]))
        axes[1, 1].plot(np.array(layers)+1, norms, marker='o', color=color)
    readouts = audit_observations(development)['layer_readouts']
    axes[1, 0].plot(np.array(layers)+1, [readouts[str(layer)]['source_coordinate_exceeds_donor'] for layer in layers], marker='o', color='#7c3aed')
    axes[0, 0].set(title='Does the answer follow the donor country?', ylabel='Success among initially correct semantic records', ylim=(-.03, 1.03))
    axes[0, 1].set(title='Does unrelated copying survive?', ylabel='Correct on all copy controls', ylim=(-.03, 1.03))
    axes[1, 0].set(title='Is the source country readable before editing?', ylabel='Source coordinate > paired donor coordinate', ylim=(-.03, 1.03))
    axes[1, 1].set(title='How large is the actual edit?', ylabel='Median displacement / recipient residual norm')
    for ax in axes.flat:
        ax.set_xlabel('Source block (one based)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('JSPACE-0001 development | one model, three country pairs\nReadability, intervention success, and nuisance preservation are different tests', fontsize=13)
    fig.savefig(result / 'development-causal-map.png', dpi=180)
    plt.close(fig)


def main():
    config = load_config('configs/jspace0001.json')
    output, result = Path(config['output']), Path(config['results'])
    development_path = output / 'development-observations.jsonl'
    report = json.loads((result / 'development.json').read_text())
    if digest(development_path) != report['observations_sha256']:
        raise ValueError('Development observations changed')
    development = [json.loads(line) for line in development_path.read_text().splitlines()]
    audit = {'development': audit_observations(development), 'development_sha256': digest(result / 'development.json')}
    final_path = output / 'final-observations.jsonl'
    if (result / 'final.json').exists():
        final_report = json.loads((result / 'final.json').read_text())
        if digest(final_path) != final_report['observations_sha256']:
            raise ValueError('Final observations changed')
        final = [json.loads(line) for line in final_path.read_text().splitlines()]
        audit['final'] = audit_observations(final)
        audit['final_leave_one_pair_out'] = leave_one_pair_out(final)
        audit['final_sha256'] = digest(result / 'final.json')
    audit['limitations'] = 'Only three unordered country pairs per scored split; these descriptive checks do not support broad population inference. Readout-sign strata do not change the primary denominator.'
    write_json(result / 'causal-audit.json', audit)
    plot(config, development, result)


if __name__ == '__main__':
    main()
