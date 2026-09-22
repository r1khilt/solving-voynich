"""Calibration-only stability diagnostics, including country contrast instability."""

import argparse
import json
from pathlib import Path

import numpy as np

from .campaign import digest, load_config, write_json
from .geometry import normalize_rows


def row_cosine(a, b):
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def contrast_diagnostics(first, second, mean, names):
    """Normalization then contrast matches the actual swap's reflection axis."""
    records = []
    for i in range(5):
        for j in range(i + 1, 5):
            f = normalize_rows(first[[i, j]])
            s = normalize_rows(second[[i, j]])
            m = normalize_rows(mean[[i, j]])
            cosine = float(m[0] @ m[1])
            records.append({"countries": [names[i], names[j]],
                            "half_contrast_cosine": float(row_cosine(f[0]-f[1], s[0]-s[1])),
                            "country_cosine": cosine,
                            "gram_condition": float(np.linalg.cond(m @ m.T))})
    return records


def analyze(config, *, prefix=None):
    output, result = Path(config["output"]), Path(config["results"])
    state = json.loads((output / "fit-progress.json").read_text())
    count = state["completed"] if prefix is None else prefix
    if count < 4 or count > state["completed"]:
        raise ValueError("Need at least four completed prompts")
    complete = count == config["calibration_prompts"]
    if prefix is None and not complete:
        raise ValueError("Full analysis requires the complete registered fit; use explicit --prefix for engineering progress")
    values = np.load(output / "derivatives.npy", mmap_mode="r")[:count]
    half = count // 2
    mean = values.mean(axis=0, dtype=np.float64)
    first = values[:half].mean(axis=0, dtype=np.float64)
    second = values[half:].mean(axis=0, dtype=np.float64)
    # The mean is a signed vector; high per-prompt sensitivity can cancel.
    squared = np.einsum('nlrd,nlrd->nlr', values, values, dtype=np.float64)
    mean_energy = np.sum(mean * mean, axis=-1)
    total_energy = squared.mean(axis=0)
    ratio = np.divide(mean_energy, total_energy, out=np.zeros_like(mean_energy), where=total_energy > 0)
    mean_country = mean.reshape(len(config['layers']), len(config['words']), 2, -1).mean(axis=2)
    first_country = first.reshape(len(config['layers']), len(config['words']), 2, -1).mean(axis=2)
    second_country = second.reshape(len(config['layers']), len(config['words']), 2, -1).mean(axis=2)
    rows = []
    for i, layer in enumerate(config['layers']):
        rows.append({'layer': layer, 'half_cosine': row_cosine(first[i], second[i]).tolist(),
                     'signed_energy_fraction': ratio[i].tolist(),
                     'prompt_norm_median': np.median(np.sqrt(squared[:, i]), axis=0).tolist(),
                     'prompt_norm_max_over_median': (np.max(np.sqrt(squared[:, i]), axis=0) / np.maximum(np.median(np.sqrt(squared[:, i]), axis=0), 1e-20)).tolist(),
                     'token_variant_cosine': row_cosine(mean[i, 0::2], mean[i, 1::2]).tolist(),
                     'country_contrasts': contrast_diagnostics(first_country[i], second_country[i], mean_country[i], config['words'])})
    report = {'complete_registered_sample': complete, 'prompts': count, 'article_halves': [half, count-half],
              'layers': rows, 'inputs_sha256': digest(result / 'inputs.json'),
              'interpretation': 'Calibration geometry only; no behavioral or decipherment claim. Zero cosine also denotes a zero-norm undefined direction.'}
    destination = result if complete else output
    write_json(destination / f'lens-stability-{count}.json', report)
    return report


def plot(config, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = report['layers']
    layers = [r['layer'] + 1 for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for index, word in enumerate(config['words'][:5]):
        axes[0].plot(layers, [np.mean(r['half_cosine'][2*index:2*index+2]) for r in rows], marker='o', label=word)
        axes[1].plot(layers, [np.mean(r['signed_energy_fraction'][2*index:2*index+2]) for r in rows], marker='o', label=word)
    axes[0].set(title='Do independent articles yield similar directions?', ylabel='Mean cosine of bare/space token forms', ylim=(-.05, 1.05))
    axes[1].set(title='How much derivative energy survives averaging?', ylabel='Squared mean / mean squared norm', ylim=(-.05, 1.05))
    for pair_index in range(10):
        pairs = [r['country_contrasts'][pair_index] for r in rows]
        axes[2].plot(layers, [p['half_contrast_cosine'] for p in pairs], alpha=.65,
                     label='/'.join(pairs[0]['countries']))
    axes[2].set(title='Is the actual swap axis stable?', ylabel='Independent-half contrast cosine', ylim=(-1.05, 1.05))
    for ax in axes:
        ax.set_xlabel('Source block (one based)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"JSPACE-0001 | {report['prompts']} independent calibration articles | geometry, not semantic proof", fontsize=13)
    path = Path(config['results']) / 'lens-stability.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=int)
    args = parser.parse_args()
    config = load_config('configs/jspace0001.json')
    report = analyze(config, prefix=args.prefix)
    if report['complete_registered_sample']:
        plot(config, report)
    print(json.dumps({'prompts': report['prompts'], 'complete': report['complete_registered_sample']}))


if __name__ == '__main__':
    main()
