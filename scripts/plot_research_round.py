"""Static scientific overview; run with matplotlib==3.10.0, no training dependencies needed."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    def read(name):
        return json.loads((root/name).read_text())
    architectures = read('results/EXP-0002/comparison.json')
    contexts = read('results/EXP-0003/summary.json')
    synthetic = read('results/EXP-0004/summary.json')
    causal = read('results/EXP-0004/cycle_null-seed42.json')['causal_state_patch']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlepad': 15,
                         'figure.facecolor': '#fafaf8', 'axes.facecolor': '#fafaf8',
                         'text.color': '#182c35', 'axes.labelcolor': '#182c35', 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
    fig.subplots_adjust(left=.08, right=.965, bottom=.20, top=.84, hspace=.62, wspace=.29)
    fig.suptitle('Voynich research: what we learned, and what we did not', x=.08, y=.966,
                 ha='left', fontsize=21, weight='bold')
    fig.text(.08, .916, '24 local training runs  ·  Three seeds per model family  ·  Manuscript test set still unscored', fontsize=11)
    colors = {'small': '#087f8c', 'reference': '#8e6c88', 'mtp': '#dd8b32'}
    ax = axes[0, 0]
    for family in colors:
        runs = [r for r in architectures['runs'] if r['config'] == family]
        steps = sorted(set.intersection(*[{p['step'] for p in r['learning_curve']} for r in runs]))
        steps = [s for s in steps if s > 0]
        means = [np.mean([next(p['validation_bits'] for p in r['learning_curve'] if p['step'] == s) for r in runs]) for s in steps]
        ax.plot(steps, means, label=family, color=colors[family], lw=2.2)
    ax.axhline(2.087802, color='#778186', ls='--', lw=1, label='five-gram baseline')
    ax.set(title='A. More updates help, then flatten out', xlabel='Training updates', ylabel='Validation bits / unit ↓')
    ax.set_ylim(1.79, 2.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.text(0, -.32, 'Whole validation set. Curves end at the last step shared by all 3 seeds.', transform=ax.transAxes, fontsize=8)
    ax = axes[0, 1]
    names = ['full128', 'last64', 'last16', 'last4', 'remote_shuffle_keep16']
    labels = ['128 units', '64 units', '16 units', '4 units', '128, distant\norder shuffled']
    for i, name in enumerate(names):
        scores = contexts['conditions'][name]['seed_scores']
        ax.scatter([i-.09, i, i+.09], scores, color='#779ea4', s=24, zorder=3)
        ax.plot([i-.23, i+.23], [np.mean(scores)]*2, color='#087f8c', lw=3)
    ax.set(title='B. Context helps; distant order is less clear', ylabel='Prediction bits / unit ↓', xticks=range(5), xticklabels=labels)
    ax.tick_params(axis='x', labelsize=8)
    ax.text(0, -.32, 'Same 192 sampled targets; dots = seeds, bars = means. Different sample from A.', transform=ax.transAxes, fontsize=8)
    ax = axes[1, 0]
    group = synthetic['groups']['cycle_null']['probes']
    x = np.arange(2)
    for offset, feature, label, color in [(-.24, 'embedding', 'Visible symbol only', '#afb9bc'),
                                         (0, 'final_residual', 'Readout of model activity', '#087f8c'),
                                         (.24, 'oracle_full', 'Knows true rules', '#dd8b32')]:
        values = [100*group[t][feature]['mean_accuracy'] for t in ['state', 'signal']]
        bars = ax.bar(x+offset, values, width=.22, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)
    ax.set(title='C. Hidden information is readable in the toy system', ylabel='Synthetic test accuracy (%) ↑',
           xticks=x, xticklabels=['Hidden state', 'Signal vs filler'], ylim=(0, 104))
    ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3)
    ax.text(0, -.38, 'Readouts use labeled toy examples after text-only LM training.\nRandom-text control: state 25.4% (chance 25%); roles match majority guessing.', transform=ax.transAxes, fontsize=8, linespacing=1.6)
    ax = axes[1, 1]
    names = ['clean', 'state_subspace', 'random_norm_matched', 'full_residual']
    vals = [causal['kl_bits_to_donor_oracle'][n] for n in names]
    bars = ax.bar(range(4), vals, color=['#afb9bc', '#087f8c', '#8e6c88', '#dd8b32'], width=.65)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set(title='D. Reading a state did not give us causal control', ylabel='Distance to donor prediction (KL bits) ↓',
           xticks=range(4), xticklabels=['Unchanged', 'State-direction\npatch', 'Matched random\npatch', 'Whole residual\nreplacement'], ylim=(0, 1.7))
    ax.tick_params(axis='x', labelsize=8)
    ax.text(0, -.29, 'Toy model, seed 42, 64 fixed pairs. State-direction patch barely helps.\nWhole-residual replacement is a plumbing control, not a discovered circuit.', transform=ax.transAxes, fontsize=8, linespacing=1.6)
    for ax in axes.flat:
        ax.grid(axis='y', color='#d6dddf', alpha=.55, zorder=0)
        ax.set_axisbelow(True)
    fig.text(.08, .035, 'These results support further mechanism research. They do not identify Voynich meanings, filler, or a historical cipher.', fontsize=10, weight='bold')
    out = root/'results/research-round-2026-09-20'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'overview.png', dpi=180)
    fig.savefig(out/'overview.svg')
    svg = out/'overview.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (out/'plotting-environment.json').write_text(json.dumps({'matplotlib': matplotlib.__version__, 'numpy': np.__version__}, indent=2)+'\n')


if __name__ == '__main__':
    main()
