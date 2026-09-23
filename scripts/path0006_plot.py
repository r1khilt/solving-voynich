"""A labeled, descriptive figure for the frozen PATH-0006 result."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path('results/PATH-0006')


def read(name):
    return json.loads((ROOT/name).read_text())


def run():
    ranking = read('selection.json')['ranking']
    summary = read('summary.json')
    n_discovery = ranking[0]['upstream_success']
    n_confirmation = summary['upstream_success']
    names = ['24–27', '28–31', '32–35', '28–31\nnew words', 'random\nsame size']
    counts = [row['donor_removed'] for row in ranking]
    counts += [summary['reductions']['28-31'], summary['reductions']['random_selected']]
    colors = ['#9eb4c6', '#367eaa', '#9eb4c6', '#15664d', '#b5aca0']
    fig, ax = plt.subplots(figsize=(9.2, 5.3), dpi=180)
    x = list(range(5))
    ax.bar(x, [counts[i]/(n_discovery if i < 3 else n_confirmation) for i in x],
           color=colors, width=.64)
    for i, count in enumerate(counts):
        denominator = n_discovery if i < 3 else n_confirmation
        ax.text(i, count/denominator+.025, f'{count}/{denominator}', ha='center', va='bottom',
                fontsize=12, fontweight='semibold', color='#173143')
    ax.axvline(2.5, color='#c7d0d3', linewidth=1)
    ax.text(1, .93, 'DISCOVERY · selected 28–31', ha='center', fontsize=10,
            color='#355668', fontweight='semibold')
    ax.text(3.5, .93, 'FRESH CONFIRMATION', ha='center', fontsize=10,
            color='#355668', fontweight='semibold')
    ax.set_xticks(x, names, fontsize=11)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel('Edited answer first token removed', fontsize=11)
    ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e0e6e8', linewidth=.8)
    ax.set_axisbelow(True)
    fig.suptitle('A later attention window has causal leverage in direct lookup',
                 x=.07, y=.98, ha='left', fontsize=15, fontweight='bold', color='#173143')
    fig.text(.07, .015, 'Pretrained model on supplied English tables. Dependent examples; not Voynich decoding.',
             fontsize=9.5, color='#536873')
    fig.subplots_adjust(left=.10, right=.98, top=.87, bottom=.20)
    fig.savefig(ROOT/'attention-window-confirmation.png', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    run()
