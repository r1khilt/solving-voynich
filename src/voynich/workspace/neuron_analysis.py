"""Descriptive natural-input response maps, separated from intervention verdicts."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .campaign import canonical_digest, digest, write_json
from .lens_analysis import row_cosine
from .tasks import build_tasks


def swiglu_change(gate_before, up_before, gate_after, up_after):
    """Symmetric exact decomposition; gate inputs are already SiLU activated.

    This algebraically splits the interaction equally between gate and value.
    It is a descriptive decomposition, not separately verified causal paths.
    """
    g0, u0, g1, u1 = [np.asarray(x, dtype=np.float64) for x in (gate_before, up_before, gate_after, up_after)]
    if len({x.shape for x in (g0, u0, g1, u1)}) != 1:
        raise ValueError('SwiGLU trace shapes differ')
    gate_effect = (g1-g0)*(u1+u0)/2
    value_effect = (u1-u0)*(g1+g0)/2
    return gate_effect, value_effect


def relative_interaction(changes):
    """Average pairwise nonadditivity, bounded [0,2] for nonzero vectors."""
    x = np.asarray(changes, dtype=np.float64)
    values = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            denominator = np.sum(x[i]**2, axis=-1) + np.sum(x[j]**2, axis=-1)
            numerator = np.sum((x[i]-x[j])**2, axis=-1)
            values.append(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0))
    if not values:
        raise ValueError('At least two query changes needed')
    return np.mean(values, axis=0)


def analyze():
    output, result = Path('outputs/NEURON-0001'), Path('results/NEURON-0001')
    if not (result / 'results.json').exists():
        raise ValueError('Finish the registered neuron study before this descriptive summary')
    manifest_hash = digest(result / 'inputs.json')
    cache = {}

    def trace(prompt):
        key = canonical_digest(prompt)
        if key not in cache:
            meta = json.loads((output / 'traces' / f'{key}.json').read_text())
            path = output / 'traces' / f'{key}.npz'
            if meta['inputs_sha256'] != manifest_hash or digest(path) != meta['arrays_sha256']:
                raise ValueError('Trace provenance changed')
            with np.load(path) as data:
                cache[key] = {k: data[k] for k in data.files}
        return cache[key]

    study = json.loads((result / 'results.json').read_text())
    norm_path = output / 'down-column-norms.npy'
    if digest(norm_path) != study['down_column_norms_sha256']:
        raise ValueError('Down-projection column norms changed')
    norms = np.load(norm_path)
    all_tasks = build_tasks(split='development')
    lookup = {(r['family'], r['pair_id'], r['paraphrase_id'], r['relation'], r['latent_concept']): r for r in all_tasks}
    tasks = [r for r in all_tasks if r['family'] != 'surface_copy' and r['paraphrase_id'] == 1 and r['id'].endswith('/d0')]
    changes = defaultdict(list)
    gate_records = []
    for task in tasks:
        before, after = trace(task['prompt']), trace(task['donor_prompt'])
        gate, value = swiglu_change(before['gate_activation'], before['up_value'], after['gate_activation'], after['up_value'])
        measured = after['neurons'].astype(np.float64)-before['neurons']
        residual = gate+value-measured
        relative = np.linalg.norm(residual, axis=-1) / np.maximum(np.linalg.norm(measured, axis=-1), 1e-12)
        energy = (measured*norms)**2
        participation = energy.sum(axis=-1)**2 / np.maximum((energy**2).sum(axis=-1), 1e-30)
        other_task = lookup[(task['family'], task['pair_id'], 0, task['relation'], task['latent_concept'])]
        other = trace(other_task['prompt'])
        paraphrase_norm = np.linalg.norm(before['residual']-other['residual'], axis=-1)
        country_norm = np.linalg.norm(after['residual']-before['residual'], axis=-1)
        gate_records.append({'id': task['id'], 'reconstruction_relative_error': relative.tolist(),
                             'weighted_gate_value_cosine': row_cosine(gate*norms, value*norms).tolist(),
                             'weighted_gate_norm': np.linalg.norm(gate*norms, axis=-1).tolist(),
                             'weighted_value_norm': np.linalg.norm(value*norms, axis=-1).tolist(),
                             'contribution_energy_participation': participation.tolist(),
                             'country_residual_norm': country_norm.tolist(), 'paraphrase_residual_norm': paraphrase_norm.tolist(),
                             'note': 'Column-norm weighted neuron coordinates remove reciprocal up/down rescaling. Participation is descriptive, not a count of causal mechanisms.'})
        key = (task['family'], task['pair_id'])
        changes[key].append({'relation': task['relation'], 'residual': after['residual']-before['residual'],
                             'mlp_write': after['mlp_write']-before['mlp_write'],
                             'attention': after['attention']-before['attention']})
    groups = []
    for (family, pair), rows in changes.items():
        if len(rows) != 4:
            raise ValueError('Incomplete query-function panel')
        groups.append({'family': family, 'pair_id': pair,
                       **{f'{field}_query_interaction': relative_interaction([r[field] for r in rows]).tolist()
                          for field in ('residual', 'mlp_write', 'attention')}})
    report = {'status': 'descriptive, no new causal pass criterion', 'groups': groups, 'gate_value_changes': gate_records,
              'inputs_sha256': manifest_hash, 'results_sha256': digest(result / 'results.json'),
              'interpretation': 'Low interaction supports similar natural country displacements across these four queries; it does not prove a causal latent variable. Large raw neuron changes are not unit importance.'}
    write_json(result / 'natural-response-analysis.json', report)
    plot(groups, result)


def plot(groups, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, field, title in zip(axes, ('residual', 'attention', 'mlp_write'), ('Residual state', 'Attention write', 'MLP write')):
        for family, color in [('indirect_fact', '#2563eb'), ('anchored_alias', '#db2777')]:
            data = np.array([g[f'{field}_query_interaction'] for g in groups if g['family'] == family])
            ax.plot(np.arange(data.shape[1])+1, data.mean(axis=0), label=family.replace('_', ' '), color=color)
            ax.fill_between(np.arange(data.shape[1])+1, data.min(axis=0), data.max(axis=0), color=color, alpha=.15)
        ax.set(title=title, xlabel='Block (one based)', ylim=(0, 2), ylabel='Cross-query interaction (0 = same displacement)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('How much does a natural country change depend on the question?\nMean and range over three pairs; descriptive geometry, not causal proof', fontsize=12)
    fig.savefig(result / 'query-dependence.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    analyze()
