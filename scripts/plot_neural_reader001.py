"""Render the completed supplied-key reader comparison; no model fitting."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/voynich-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/NEURAL-READER-001'
result = json.loads((OUT / 'summary.json').read_text())
arms = ['statistical-small', 'statistical-large', 'neural-31103-b128', 'neural-31109-b128']
labels = ['Small-data\nstatistical', 'Large-data\nstatistical', 'Neural A\nprimary', 'Neural B\nprimary']
colors = ['#adb5c3', '#4d5d78', '#168678', '#92b948']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [1, 1.4]})
values = [result['arms'][a]['edits'] for a in arms]
left.bar(labels, values, color=colors, width=.65)
for i, value in enumerate(values):
    row = result['arms'][arms[i]]
    left.text(i, value + 3, f'{value} edits\n{100 * row["cer"]:.2f}%', ha='center', va='bottom', fontsize=10)
left.set(ylabel='Edits across 7,168 true letters', ylim=(0, max(values) * 1.28),
         title='More data helps; the neural models improve further')
left.grid(axis='y', alpha=.15)
left.set_axisbelow(True)
for arm, label, color, marker in zip(arms[1:], ['Large statistical', 'Neural A', 'Neural B'], colors[1:], ['s', 'o', '^']):
    right.plot(range(1, 17), result['arms'][arm]['per_key_edits'], label=label, color=color,
               marker=marker, linewidth=1.2, markersize=5, alpha=.85)
right.set(xlabel='Fresh key · two passages per key', ylabel='Edits out of 448 true letters',
          title='Every key is retained in the comparison', ylim=(-.5, 12))
right.set_xticks(range(1, 17))
right.grid(axis='y', alpha=.15)
right.legend(frameon=False, fontsize=9)
fig.suptitle('Both neural readers pass the registered supplied-key test', fontsize=15, fontweight='bold')
fig.text(.015, .035, 'Correct keys, Latin alphabet and record boundaries were given. Two reserved authors; 32 positive passages.', fontsize=10)
fig.text(.015, .005, 'Primary beam128 results shown. Shuffled controls and beam32 diagnostics remain in the complete report. No unknown-key or Voynich claim.', fontsize=9)
fig.tight_layout(rect=(0, .07, 1, .94))
fig.savefig(OUT / 'reader-comparison.png', dpi=160)
plt.close(fig)
