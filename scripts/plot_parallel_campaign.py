"""Plot the three registered campaign outcomes from compact archived reports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path.cwd()
    def read(path):
        return json.loads((root/path).read_text())
    blind = read('results/EXP-0008/compact_report.json')
    mapping = read('results/EXP-0009/completion.json')
    context = read('results/EXP-0010/summary.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.titlepad': 15,
                         'figure.facecolor': '#fafaf8', 'axes.facecolor': '#fafaf8',
                         'text.color': '#182c35', 'axes.labelcolor': '#182c35',
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    fig.subplots_adjust(left=.08, right=.97, top=.82, bottom=.19, hspace=.68, wspace=.30)
    fig.suptitle('Three experiments: structure, causality and longer context',
                 x=.08, y=.97, ha='left', fontsize=21, weight='bold')
    fig.text(.08, .925, '25 new models  ·  41,200 training updates  ·  Local Mac GPU  ·  Final manuscript test unscored', fontsize=11)
    fig.text(.08, .885, 'Artificial-generator results and manuscript findings are separate. None is a decipherment.', fontsize=11, color='#53666e')
    teal, orange, grey = '#087f8c', '#ad6e23', '#647780'

    ax = axes[0, 0]
    cases = [('cycle_null', 'unseen_key', 'Cycle\nnew keys'),
             ('branch_null', 'unseen_key', 'Branch\nnew keys'),
             ('rrxor', 'novel_family', 'XOR\nnew family'),
             ('copy_lag', 'fresh_stream_control', 'Copy\ncontrol'),
             ('iid', 'fresh_stream_control', 'Random\ncontrol')]
    for method, offset, color, label in [('residual', -.13, teal, 'Hidden-vector clusters'),
                                          ('forecast', .13, orange, 'Prediction-only clusters')]:
        for i, (family, transfer, _) in enumerate(cases):
            values = np.array([r['methods'][method]['mean_bits_gain_over_last_symbol']
                               for r in blind['reports'] if r['family'] == family and r['transfer'] == transfer])
            if not len(values):
                raise ValueError(f'Missing blind family: {family}/{transfer}')
            jitter = np.linspace(-.055, .055, len(values))
            ax.scatter(i+offset+jitter, values, color=color, alpha=.28, s=14)
            ax.plot([i+offset-.075, i+offset+.075], [values.mean()]*2, color=color, lw=3,
                    label=label if i == 0 else None)
    ax.axhline(0, color=grey, lw=.8)
    ax.set_xticks(range(len(cases)), [c[2] for c in cases])
    ax.set(title='A. Copying is detectable; unfamiliar-key recovery fails',
           ylabel='Improvement in future-symbol loss (bits) ↑')
    ax.legend(fontsize=8, frameon=False)
    ax.text(0, -.31, 'All four models and applicable keys; dots are model/key cases.\nXOR/random choose one cluster; small gains reflect a simpler estimator.\nCopy gains mainly concern the next symbol; later predictions remain poor.',
            transform=ax.transAxes, fontsize=8, linespacing=1.6)

    ax = axes[0, 1]
    trained = [r for r in mapping['results'] if r['kind'] == 'synthetic']
    reports = [read(r['report_path'].replace('outputs/', 'results/', 1)) for r in trained]
    for name, color, label in [('selected', teal, 'Earlier component'),
                               ('random', orange, 'Matched random'),
                               ('late_readout', grey, 'Late output change')]:
        values = []
        for r in reports:
            base = np.array(r['confirmation']['conditions']['recipient']['equal_group_mean'])[:, 0]
            site = r['decision']['selected_site']
            suffix = '' if site['head'] is None else str(site['head'])
            key = f"L{site['layer']}-{site['kind']}{suffix}-{site['region']}"
            if name == 'selected':
                loss = np.array(r['confirmation']['conditions'][key+'/donor']['equal_group_mean'])[:, 0]
            elif name == 'random':
                loss = np.mean([np.array(r['confirmation']['conditions'][key+f'/random{i}']['equal_group_mean'])[:, 0]
                                for i in range(4)], axis=0)
            else:
                loss = np.array(r['confirmation']['conditions'][name]['equal_group_mean'])[:, 0]
            values.append(base-loss)
        ax.plot(range(4), np.mean(values, axis=0), marker='o', color=color, label=label)
    ax.axhline(0, color=grey, lw=.8)
    ax.set_xticks(range(4), ['Next', 'After 1', 'After 3', 'After 7'])
    ax.set(title='B. Toy cipher: earlier changes reach later predictions',
           ylabel='Reduction in oracle KL (bits) ↑', xlabel='Shared additional symbols observed')
    ax.legend(fontsize=8, frameon=False)
    ax.text(0, -.38, 'Means across three trained seeds; all pass the fixed future-effect criterion.\nBroad component replacements; no isolated state or transition algorithm.',
            transform=ax.transAxes, fontsize=8, linespacing=1.6)

    ax = axes[1, 0]
    for i, r in enumerate(r for r in mapping['results'] if r['kind'] == 'manuscript'):
        ax.plot(range(4), r['confirmation_gain_by_horizon'], marker='o', alpha=.8,
                color=[teal, orange, grey][i], label=f"Seed {r['seed']}")
    ax.axhline(0, color=grey, lw=.8)
    ax.axhline(.02, color=grey, lw=.8, ls='--', label='Required after 1 and 3 new symbols')
    ax.set_xticks(range(4), ['Next', 'After 1', 'After 3', 'After 7'])
    ax.set(title='C. Voynich: the causal confirmation criterion fails',
           ylabel='Improvement in actual-target loss (bits) ↑', xlabel='Shared additional symbols observed')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.text(0, -.38, 'Five confirmation leaves; all three models fail the registered criterion.\nFailure does not establish absence of structure. Metrics differ from panel B.',
            transform=ax.transAxes, fontsize=8, linespacing=1.6)

    ax = axes[1, 1]
    labels = [('main-c512-original', 'Main\n512'), ('main-c1024-original', 'Main\n1,024'),
              ('main-c2048-original', 'Main\n2,048'), ('compact-c2048-original', 'Compact\n2,048')]
    for i, (key, _) in enumerate(labels):
        comparison = context['comparisons'][key]
        values = np.array([p['gain_bits_reference_minus_treatment']['primary']['sample_mean']
                           for p in comparison['per_seed']])
        if len(values) != 3:
            raise ValueError('Plot requires three complete matched seed pairs')
        color = teal if comparison['registered_context_signal'] else grey
        ax.scatter(i+np.array([-.07, 0, .07]), values, color=color, s=25, alpha=.7)
        ax.plot([i-.18, i+.18], [values.mean()]*2, color=color, lw=3)
    ax.axhline(0, color=grey, lw=.8)
    ax.axhline(.02, color=orange, lw=.8, ls='--')
    ax.set_xticks(range(len(labels)), [x[1] for x in labels])
    ax.set(title='D. Voynich: longer context with equal training exposure',
           ylabel='Gain over matching 256-unit model (bits) ↑')
    ax.text(0, -.31, '384 targets / 10 leaves; many have shorter available prefixes.\nDots = seeds; bars = means. Teal requires all registered conditions.',
            transform=ax.transAxes, fontsize=8, linespacing=1.6)
    for ax in axes.flat:
        ax.grid(axis='y', color='#d6dddf', alpha=.65)
        ax.set_axisbelow(True)
    fig.text(.08, .065, 'No translated words, manuscript filler assignments or historical encoding rules have been established.', fontsize=11, weight='bold')
    fig.text(.08, .033, 'EXP-0008 / 0009 / 0010  ·  Full model/key tables, boundary ablation, controls and provenance accompany the notebook.', fontsize=9, color=grey)
    out = root/'results/CAMPAIGN-0001'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'overview.png', dpi=180)
    fig.savefig(out/'overview.svg')
    svg = out/'overview.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (out/'plotting-environment.json').write_text(json.dumps({'matplotlib': matplotlib.__version__, 'numpy': np.__version__}, indent=2)+'\n')


if __name__ == '__main__':
    main()
