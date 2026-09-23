"""Render the stopped PATH-0004 discovery panel without implying confirmation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT/'results/PATH-0004'
LAYERS = (7, 15, 23)
CONDITIONS = ('identity', 'value', 'suffix', 'value_suffix', 'all_earlier', 'whole', 'suffix_random')
LABELS = ('Identity', 'First-table keys', 'Downstream suffix', 'Keys + suffix',
          'All earlier', 'Whole prompt', 'Random suffix')


def main():
    rows = json.loads((RESULT/'discovery-rows.json').read_text())
    matrices, counts = [], []
    for family, eligible_field, outcome in (
        ('composed', 'eligible', 'desired_answer'),
        ('copy', 'baseline_copy', 'copy_preserved'),
    ):
        matrix = np.zeros((len(LAYERS), len(CONDITIONS)))
        labels = []
        for i, layer in enumerate(LAYERS):
            current = []
            for j, condition in enumerate(CONDITIONS):
                subset = [r for r in rows if r['layer'] == layer and r['condition'] == condition
                          and r['family'] == family and r[eligible_field]]
                n, k = len(subset), sum(r[outcome] for r in subset)
                matrix[i, j] = k/n if n else np.nan
                current.append(f'{k}/{n}')
            labels.append(current)
        matrices.append(matrix)
        counts.append(labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for axis, matrix, labels, title in zip(
        axes, matrices, counts,
        ('Donor object after swap', 'Unrelated copying preserved'),
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        axis.set_xticks(range(len(CONDITIONS)), LABELS, rotation=42, ha='right')
        axis.set_yticks(range(len(LAYERS)), [f'Block {layer+1}' for layer in LAYERS])
        axis.set_title(title, fontsize=13, pad=14)
        axis.tick_params(length=0)
        for i in range(len(LAYERS)):
            for j in range(len(CONDITIONS)):
                axis.text(j, i, labels[i][j], ha='center', va='center', fontsize=9,
                          color='white' if matrix[i, j] < .55 else 'black', weight='bold')
        axis.add_patch(Rectangle((-.5, .5), len(CONDITIONS), 1, fill=False,
                                 edgecolor='#f97316', linewidth=2.4))
    fig.colorbar(image, ax=axes, fraction=.025, pad=.02, label='Fraction of eligible records')
    fig.suptitle('PATH-0004 discovery only: confirmation competence gate failed', fontsize=15, weight='bold')
    target = RESULT/'discovery-position-map.png'
    fig.savefig(target, dpi=220, facecolor='white')
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
