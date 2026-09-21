"""Scientific overview of EXP-0005/0006/0007 from archived measurements."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]

    def read(path):
        return json.loads((root/path).read_text())

    context = read('results/EXP-0005/report.json')
    order = read('results/EXP-0007/report.json')
    causal = {site: [read(f'results/EXP-0006/trained-seed{seed}-site{site}.json')
                     for seed in [42, 43, 44]] for site in [0, 1]}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.titlepad': 16,
                         'figure.facecolor': '#fafaf8', 'axes.facecolor': '#fafaf8',
                         'text.color': '#182c35', 'axes.labelcolor': '#182c35',
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.6))
    fig.subplots_adjust(left=.08, right=.97, bottom=.21, top=.83, hspace=.70, wspace=.26)
    fig.suptitle('Voynich: stronger controls change the interpretation', x=.08, y=.97,
                 ha='left', fontsize=21, weight='bold')
    fig.text(.08, .922, 'Three new experiments  ·  Frozen language models  ·  Local MPS compute  ·  Manuscript final test unscored', fontsize=11)
    fig.text(.08, .89, 'Dots show the three model seeds; horizontal bars show their mean. The panels use different metrics.', fontsize=10, color='#53666e')

    def dots(ax, values, labels, colors):
        for i, (scores, color) in enumerate(zip(values, colors, strict=True)):
            scores = np.asarray(scores)
            ax.scatter(i+np.array([-.08, 0, .08]), scores, color=color, s=28, zorder=3, alpha=.65)
            ax.plot([i-.21, i+.21], [scores.mean()]*2, color=color, lw=3, zorder=4)
        ax.set_xticks(range(len(labels)), labels)
        ax.tick_params(axis='x', labelsize=8)

    ax = axes[0, 0]
    conditions = ['shuffle_original', 'same_random_ordered', 'same_matched_ordered',
                  'different_random_ordered', 'different_matched_ordered']
    values = [[m['summaries'][c]['sample_mean']-m['summaries']['full128']['sample_mean']
               for m in context['models'].values()] for c in conditions]
    dots(ax, values, ['Shuffle\noriginal', 'Same category\nrandom', 'Same category\nhistogram match',
                      'Other category\nrandom', 'Other category\nhistogram match'],
         ['#53666e', '#087f8c', '#087f8c', '#ad6e23', '#ad6e23'])
    ax.axhline(0, color='#53666e', lw=.8)
    ax.set(title='A. Distant content and order both affect prediction', ylabel='Added prediction loss (bits / unit) ↑', ylim=(-.02, .40))
    ax.text(0, -.33, '768 manuscript targets; local 16 units fixed. Histogram matching is approximate.\nCategory = illustration metadata, not a known language or cipher key.',
            transform=ax.transAxes, fontsize=8, linespacing=1.6)

    ax = axes[0, 1]
    conditions = ['group_order', 'matched_character_scramble', 'block1']
    values = [[m['summaries'][c]['sample_mean']-m['summaries']['full128']['sample_mean']
               for m in order['models'].values()] for c in conditions]
    dots(ax, values, ['Move equal-length\ncomplete groups', 'Scramble characters\nat matched positions',
                      'Unrestricted distant\nunit shuffle'], ['#087f8c', '#ad6e23', '#53666e'])
    ax.axhline(0, color='#53666e', lw=.8)
    ax.set(title='B. The matched group-form contrast stays small', ylabel='Added prediction loss (bits / unit) ↑', ylim=(-.01, .066))
    ax.text(0, -.33, 'First two conditions preserve separators and change identical positions.\nTheir mean difference is 0.0051 bits; all leaf-bootstrap intervals include zero.',
            transform=ax.transAxes, fontsize=8, linespacing=1.6)

    methods = ['clean', 'learned', 'shuffled_supervision', 'readout_span', 'random_norm_mean', 'full']
    labels = ['Unchanged', 'Learned\nbasis', 'Shuffled\nsupervision', 'Output-weight\nspan', 'Matched\nrandom', 'Whole\nresidual']
    colors = ['#7d8e94', '#087f8c', '#8e6c88', '#ad6e23', '#7d8e94', '#7d8e94']
    for site, ax in zip([0, 1], axes[1], strict=True):
        values = []
        for method in methods:
            if method == 'random_norm_mean':
                values.append([np.mean([r['scores']['test128'][f'random_norm_{i}']['mean']['oracle_kl']
                                        for i in range(8)]) for r in causal[site]])
            else:
                values.append([r['scores']['test128'][method]['mean']['oracle_kl'] for r in causal[site]])
        dots(ax, values, labels, colors)
        ax.set(ylabel='Distance to desired prediction (KL bits) ↓', ylim=(-.03, 1.57))
    axes[1, 0].set_title('C. Early intervention fails the supervision control')
    axes[1, 0].text(0, -.34, 'Toy cipher, 512 fresh pairs per seed. Real and shuffled supervision perform\nsimilarly. Lower distance alone does not identify the intended computation.',
                    transform=axes[1, 0].transAxes, fontsize=8, linespacing=1.6)
    axes[1, 1].set_title('D. Late steering works; the output control matches it')
    axes[1, 1].text(0, -.34, 'Learned basis: 97.07% agreement on desired next category. The simple\noutput-weight span is nearly as effective. This is not circuit identification.',
                    transform=axes[1, 1].transAxes, fontsize=8, linespacing=1.6)
    for ax in axes.flat:
        ax.grid(axis='y', color='#d6dddf', alpha=.65, zorder=0)
        ax.set_axisbelow(True)
    fig.text(.08, .066, 'Synthetic intervention fitting uses the known rules. No Voynich decoding or translation has been demonstrated.',
             fontsize=10, weight='bold')
    fig.text(.08, .035, 'EXP-0005 / 0006 / 0007  ·  Exact sources, per-target results, controls and limitations are in the research notebook.',
             fontsize=9, color='#53666e')
    out = root/'results/research-round-2-2026-09-20'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'overview.png', dpi=180)
    fig.savefig(out/'overview.svg')
    svg = out/'overview.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (out/'plotting-environment.json').write_text(json.dumps({'matplotlib': matplotlib.__version__,
                                                          'numpy': np.__version__}, indent=2)+'\n')


if __name__ == '__main__':
    main()
