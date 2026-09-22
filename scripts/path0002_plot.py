"""Render the completed PATH-0002 position/layer outcome map."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT/'results/PATH-0002'
LAYERS = (15, 19, 23, 27, 31)
CONDITIONS = ('identity', 'last_donor', 'value_donor', 'other_earlier_donor',
              'earlier_donor', 'all_donor', 'value_random')
LABELS = ('Identity', 'Final token', 'Value tokens', 'Other earlier',
          'All earlier', 'Whole prompt', 'Random values')


def main():
    summary = json.loads((RESULT/'summary.json').read_text())
    matrices = []
    count_matrices = []
    for category, numerator in (('semantic', 'switches'), ('copy', 'preserved')):
        values = np.zeros((len(LAYERS), len(CONDITIONS)))
        counts = []
        for i, layer in enumerate(LAYERS):
            row = []
            for j, condition in enumerate(CONDITIONS):
                cell = summary[str(layer)][condition][category]
                values[i, j] = cell[numerator]/cell['count'] if cell['count'] else np.nan
                row.append(f"{cell[numerator]}/{cell['count']}")
            counts.append(row)
        matrices.append(values)
        count_matrices.append(counts)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    titles = ('Donor answer after intervention', 'Unrelated copying preserved')
    for axis, values, counts, title in zip(axes, matrices, count_matrices, titles):
        image = axis.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        axis.set_xticks(range(len(CONDITIONS)), LABELS, rotation=45, ha='right')
        axis.set_yticks(range(len(LAYERS)), [f'Block {i+1}' for i in LAYERS])
        axis.set_title(title, fontsize=14, pad=15)
        for i in range(len(LAYERS)):
            for j in range(len(CONDITIONS)):
                color = 'white' if values[i, j] < .55 else 'black'
                axis.text(j, i, counts[i][j], ha='center', va='center', color=color, fontsize=9, weight='bold')
        axis.add_patch(Rectangle((-.5, 1.5), len(CONDITIONS), 1, fill=False,
                                 edgecolor='#f97316', linewidth=2.5))
        axis.tick_params(length=0)
    fig.colorbar(image, ax=axes, fraction=.023, pad=.02, label='Fraction of eligible records')
    fig.suptitle('PATH-0002: where a swapped record value affects the answer', fontsize=17, weight='bold')
    target = RESULT/'position-map.png'
    fig.savefig(target, dpi=220, facecolor='white')
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
