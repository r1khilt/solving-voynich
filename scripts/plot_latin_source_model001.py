"""Plot already audited source-development results; no fitting or source selection."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/voynich-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/LATIN-SOURCE-MODEL-001'
result = json.loads((OUT / 'comparison.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.4, 1]})
colors = {'small': '#929bad', 'large': '#168678'}
for row in result['neural_runs']:
    seed = 'A' if row['seed'] == 31103 else 'B'
    left.plot([p['step'] for p in row['curve']], [p['bpc'] for p in row['curve']],
              label=f"{row['dataset'].title()} data · neural {seed}", color=colors[row['dataset']],
              linestyle='-' if seed == 'A' else '--', marker='o', markersize=4)
    left.scatter([row['selected_step']], [row['selected_bpc']], color=colors[row['dataset']], s=70, zorder=5)
left.set(xlabel='Training updates', ylabel='Pliny prediction loss (bits per letter)',
         title='Small-data memorization versus transfer')
left.set_xscale('symlog', linthresh=100)
left.set_xticks([0, 100, 500, 1000, 2000, 4000, 6000], ['0', '100', '500', '1k', '2k', '4k', '6k'])
left.grid(axis='y', alpha=.16)
left.legend(frameon=False, fontsize=9)
labels, values, palette = [], [], []
for dataset in ('small', 'large'):
    labels.append(f'{dataset.title()} · statistical')
    values.append(result['statistical'][dataset]['selected_bpc'])
    palette.append(colors[dataset])
    for row in result['neural_runs']:
        if row['dataset'] == dataset:
            labels.append(f"{dataset.title()} · neural {'A' if row['seed'] == 31103 else 'B'}")
            values.append(row['selected_bpc'])
            palette.append(colors[dataset])
right.barh(labels, values, color=palette)
right.invert_yaxis()
for i, value in enumerate(values):
    right.text(value + .04, i, f'{value:.3f}', va='center', fontsize=10)
right.set(xlim=(0, max(values) + .5), xlabel='Selected bits per letter · lower is better',
          title='Same evaluation letters, matched training text')
fig.suptitle('Broader Latin data helps; reading accuracy is a separate test', fontsize=14, fontweight='bold')
fig.text(.01, .012, 'Pliny was excluded from training but used for checkpoint selection. These are development scores, not cipher results.', fontsize=9)
fig.tight_layout(rect=(0, .04, 1, .94))
fig.savefig(OUT / 'learning-curves.png', dpi=160)
plt.close(fig)
