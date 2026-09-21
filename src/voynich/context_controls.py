"""EXP-0005: length-controlled distant-context replacement and frequency baselines."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .context_analysis import bits, paired_summary, sample_contexts
from .evaluate import NGram
from .runtime import PageWindows, corpus_identity, digest, environment, resolve_device, write_json
from .train import load_checkpoint


def section(page):
    return page['metadata']['page_variables']['I']


def histograms(rows, vocab):
    return np.stack([np.bincount(row, minlength=vocab) for row in rows]).astype(np.float64)/rows.shape[1]


def donor_pool(train, width=112):
    rows, locations, sections = [], [], []
    for page in train.pages:
        seq = train.sequences[page['page_id']]
        for start in range(1, len(seq)-width, 32):
            rows.append(seq[start:start+width])
            locations.append({'page_id': page['page_id'], 'start': start})
            sections.append(section(page))
    return np.array(rows), locations, np.array(sections)


def make_conditions(examples, train, sections, repeats=4):
    prefixes = np.array([e['prefix'] for e in examples])
    pool, coordinates, groups = donor_pool(train)
    if not len(pool):
        raise ValueError('No training donor windows')
    vocabulary = train.tokenizer.vocab_size
    pool_hist = histograms(pool, vocabulary)
    original_hist = histograms(prefixes[:, :112], vocabulary)
    conditions = {'full128': prefixes[None], 'last16': prefixes[None, :, -16:]}
    conditions['shuffle_original'] = np.repeat(prefixes[None], repeats, axis=0)
    records = []
    for group_name in ['same', 'different']:
        for choice in ['random', 'matched']:
            for order in ['ordered', 'shuffled']:
                conditions[f'{group_name}_{choice}_{order}'] = np.repeat(prefixes[None], repeats, axis=0)
    for repeat in range(repeats):
        rng = np.random.default_rng(5202+repeat)
        for i, example in enumerate(examples):
            conditions['shuffle_original'][repeat, i, :112] = rng.permutation(prefixes[i, :112])
            for group_name, allowed in [('same', groups == sections[example['page_id']]),
                                        ('different', groups != sections[example['page_id']])]:
                eligible = np.flatnonzero(allowed)
                if not len(eligible):
                    raise ValueError('Missing a donor section group')
                candidates = rng.choice(eligible, min(128, len(eligible)), replace=False)
                distance = .5*np.abs(pool_hist[candidates]-original_hist[i]).sum(-1)
                for choice, offset in [('random', 0), ('matched', int(distance.argmin()))]:
                    index = int(candidates[offset])
                    label = f'{group_name}_{choice}'
                    conditions[label+'_ordered'][repeat, i, :112] = pool[index]
                    conditions[label+'_shuffled'][repeat, i, :112] = rng.permutation(pool[index])
                    records.append({'example': i, 'repeat': repeat, 'condition': label,
                                    'donor': coordinates[index], 'donor_section': str(groups[index]),
                                    'histogram_total_variation': float(distance[offset])})
    for rows in conditions.values():
        if rows.shape[-1] == 128 and not np.array_equal(rows[:, :, -16:], np.broadcast_to(prefixes[:, -16:], rows[:, :, -16:].shape)):
            raise ValueError('Local context altered')
    return conditions, records


def frequency_adapt(probabilities, prefix, prior, alpha, strength=32.):
    counts = np.bincount(prefix, minlength=len(prior))
    local = (counts+strength*prior)/(len(prefix)+strength)
    adapted = probabilities*np.power(local/prior, alpha)
    return adapted/adapted.sum()


def score_baselines(examples, train, sections):
    vocab = train.tokenizer.vocab_size
    global_model = NGram(vocab, order=5).fit(train)
    prior = np.array([global_model.probability([], t) for t in range(vocab)])
    by_section = {}
    for label in set(sections.values()):
        selected = {p['page_id']: train.sequences[p['page_id']] for p in train.pages if section(p) == label}
        by_section[label] = NGram(vocab, order=5).fit(SimpleNamespace(sequences=selected, ignored_ids=train.ignored_ids))
    losses = {key: [] for key in ['fivegram', 'fivegram_hist_alpha0.5', 'fivegram_hist_alpha1', 'fivegram_section_mix0.5']}
    for e in examples:
        prefix, target = e['prefix'], e['target']
        probs = np.array([global_model.probability(prefix, t) for t in range(vocab)])
        local_probs = np.array([by_section[sections[e['page_id']]].probability(prefix, t) for t in range(vocab)])
        losses['fivegram'].append(float(-np.log2(probs[target])))
        losses['fivegram_section_mix0.5'].append(float(-np.log2(.5*probs[target]+.5*local_probs[target])))
        for alpha in [.5, 1]:
            adapted = frequency_adapt(probs, prefix[:112], prior, alpha)
            losses[f'fivegram_hist_alpha{alpha}'].append(float(-np.log2(adapted[target])))
    return {'losses_bits': losses, 'summaries': {k: paired_summary(v, examples) for k, v in losses.items()},
            'damage_vs_fivegram': {k: paired_summary(np.array(v)-losses['fivegram'], examples) for k, v in losses.items()},
            'note': 'All counts fit training only. Section mixture uses illustration metadata. Fixed strengths, no test or validation tuning.'}


@torch.no_grad()
def score_conditions(model, conditions, targets, device, batch_size=64):
    losses = {}
    for name, rows in conditions.items():
        flattened = rows.reshape(-1, rows.shape[-1])
        repeated_targets = np.tile(targets, rows.shape[0])
        scores = []
        for start in range(0, len(flattened), batch_size):
            x = torch.tensor(flattened[start:start+batch_size], device=device)
            y = torch.tensor(repeated_targets[start:start+batch_size], device=device)
            scores.extend(bits(model(x).logits, y).cpu().tolist())
        losses[name] = np.array(scores).reshape(rows.shape[:2]).mean(0).tolist()
    return losses


def run(output_root, device='mps'):
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh experiment output directory')
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    identity = corpus_identity('data/processed/zl3b')
    train = PageWindows('data/processed/zl3b', 'train', 256)
    validation = PageWindows('data/processed/zl3b', 'validation', 256)
    examples = sample_contexts(validation, per_page=32, seed=5201)
    sections = {p['page_id']: section(p) for p in validation.pages}
    conditions, donors = make_conditions(examples, train, sections)
    report = {'experiment': 'EXP-0005', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'environment': environment(), 'device': device, 'corpus_identity': identity,
              'test_evaluated': False, 'sample_seed': 5201, 'corruption_seeds': [5202, 5203, 5204, 5205],
              'examples': [{k: v for k, v in e.items() if k != 'prefix'} | {'section': sections[e['page_id']]} for e in examples],
              'donors': donors, 'baselines': score_baselines(examples, train, sections), 'models': {}}
    targets = np.array([e['target'] for e in examples])
    for seed in [42, 43, 44]:
        checkpoint = Path(f'outputs/EXP-0002/small-seed{seed}/best.pt')
        model, payload = load_checkpoint(checkpoint, device)
        model.eval()
        if payload['corpus_identity'] != identity:
            raise ValueError('Checkpoint and data mismatch')
        losses = score_conditions(model, conditions, targets, device)
        contrasts = {'last16_minus_full': np.array(losses['last16'])-losses['full128'],
                     'shuffle_minus_full': np.array(losses['shuffle_original'])-losses['full128']}
        for choice in ['random', 'matched']:
            contrasts[f'different_minus_same_{choice}'] = np.array(losses[f'different_{choice}_ordered'])-losses[f'same_{choice}_ordered']
            for group in ['same', 'different']:
                contrasts[f'shuffle_damage_{group}_{choice}'] = np.array(losses[f'{group}_{choice}_shuffled'])-losses[f'{group}_{choice}_ordered']
        report['models'][str(seed)] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                                      'losses_bits': losses,
                                      'summaries': {k: paired_summary(v, examples, seed=5206) for k, v in losses.items()},
                                      'contrasts': {k: paired_summary(v, examples, seed=5206) for k, v in contrasts.items()}}
        print(json.dumps({'seed': seed, 'scores': {k: float(np.mean(v)) for k, v in losses.items()}}), flush=True)
        del model
    report['elapsed_seconds'] = time.monotonic()-started
    report['limitations'] = ['Exploratory repeatedly-consulted manuscript validation; no final test.',
                             'Histogram matching is nearest among 128 candidates, not exact; report distances.',
                             'Cross-section differences can reflect script/style/lexicon/layout rather than meaning.',
                             'Replacement contexts and concatenation are out of distribution.',
                             'Four corruption repetitions averaged per target before descriptive leaf bootstrap.']
    write_json(root/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='outputs/EXP-0005')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output_root, resolve_device(args.device))


if __name__ == '__main__':
    main()
