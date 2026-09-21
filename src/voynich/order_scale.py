"""EXP-0007: order scale and matched corruption of transcription groups."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .context_analysis import paired_summary, sample_contexts
from .context_controls import score_conditions
from .runtime import PageWindows, corpus_identity, digest, environment, resolve_device, write_json
from .train import load_checkpoint


def nonidentity_permutation(count, rng):
    order = rng.permutation(count)
    if count > 1 and np.array_equal(order, np.arange(count)):
        order = np.roll(order, 1)
    return order


def matched_group_corruptions(row, separators, rng):
    """Swap complete equal-length groups; scramble the exact changed positions."""
    original = np.asarray(row)
    groups = defaultdict(list)
    start = 0
    for end in range(len(original)+1):
        if end == len(original) or original[end] in separators:
            if start > 0 and end < len(original) and end-start >= 2:
                groups[end-start].append(np.arange(start, end))
            start = end+1
    swapped = original.copy()
    for spans in groups.values():
        if len(spans) < 2:
            continue
        permutation = nonidentity_permutation(len(spans), rng)
        for recipient, donor in enumerate(permutation):
            swapped[spans[recipient]] = original[spans[donor]]
    mask = swapped != original
    positions = np.flatnonzero(mask)
    scrambled = swapped.copy()
    # Constraint-preserving random walk; no claim of a uniform draw over all derangements.
    if len(positions) >= 2:
        for _ in range(min(1024, 10*len(positions))):
            i, j = rng.choice(positions, 2, replace=False)
            if scrambled[j] != original[i] and scrambled[i] != original[j]:
                scrambled[i], scrambled[j] = scrambled[j], scrambled[i]
    if not np.array_equal(scrambled != original, mask):
        raise ValueError('Changed-position matching failed')
    if not np.array_equal(np.sort(scrambled), np.sort(original)):
        raise ValueError('Histogram changed')
    separator_mask = np.isin(original, list(separators))
    if not np.array_equal(swapped[separator_mask], original[separator_mask]):
        raise ValueError('Separator positions changed')
    return swapped, scrambled, {'changed_fraction': float(mask.mean()),
                                'control_difference_fraction': float((scrambled != swapped).mean())}


def make_conditions(examples, separators, repeats=4):
    prefixes = np.array([e['prefix'] for e in examples])
    conditions = {'full128': prefixes[None]}
    for block in [1, 2, 4, 8, 16, 28, 56]:
        conditions[f'block{block}'] = np.repeat(prefixes[None], repeats, axis=0)
    for name in ['group_order', 'matched_character_scramble']:
        conditions[name] = np.repeat(prefixes[None], repeats, axis=0)
    exposure = []
    for repeat in range(repeats):
        rng = np.random.default_rng(7201+repeat)
        for i, prefix in enumerate(prefixes):
            for block in [1, 2, 4, 8, 16, 28, 56]:
                remote = prefix[:112].reshape(-1, block)
                conditions[f'block{block}'][repeat, i, :112] = remote[nonidentity_permutation(len(remote), rng)].reshape(-1)
            swapped, scrambled, stats = matched_group_corruptions(prefix[:112], separators, rng)
            conditions['group_order'][repeat, i, :112] = swapped
            conditions['matched_character_scramble'][repeat, i, :112] = scrambled
            exposure.append({'repeat': repeat, 'example': i, **stats})
    for name, rows in conditions.items():
        if not np.array_equal(np.sort(rows, axis=-1), np.broadcast_to(np.sort(prefixes, axis=-1), rows.shape)):
            raise ValueError(f'Histogram differs for {name}')
        if not np.array_equal(rows[:, :, -16:], np.broadcast_to(prefixes[:, -16:], rows[:, :, -16:].shape)):
            raise ValueError('Last16 context changed')
    return conditions, exposure


def run(output_root, device):
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use fresh output directory')
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    identity = corpus_identity('data/processed/zl3b')
    data = PageWindows('data/processed/zl3b', 'validation', 256)
    examples = sample_contexts(data, per_page=32, seed=5201)
    conditions, exposure = make_conditions(examples, {data.tokenizer.space_id, data.tokenizer.line_id})
    previous = json.loads(Path('results/EXP-0005/report.json').read_text())
    coordinates = [{k: v for k, v in e.items() if k != 'prefix'} for e in examples]
    if coordinates != [{k: v for k, v in e.items() if k != 'section'} for e in previous['examples']]:
        raise ValueError('Fixed target coordinates differ from EXP-0005')
    targets = np.array([e['target'] for e in examples])
    report = {'experiment': 'EXP-0007', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'environment': environment(), 'device': device, 'corpus_identity': identity,
              'examples': coordinates, 'corruption_exposure': exposure,
              'test_evaluated': False, 'models': {}, 'corruption_seeds': [7201, 7202, 7203, 7204]}
    for seed in [42, 43, 44]:
        checkpoint = Path(f'outputs/EXP-0002/small-seed{seed}/best.pt')
        model, payload = load_checkpoint(checkpoint, device)
        if payload['corpus_identity'] != identity:
            raise ValueError('Checkpoint corpus mismatch')
        model.eval()
        losses = score_conditions(model, conditions, targets, device)
        parity = float(np.max(np.abs(np.array(losses['full128'])-previous['models'][str(seed)]['losses_bits']['full128'])))
        if parity > 1e-5:
            raise ValueError('Original-context score changed from EXP-0005')
        contrasts = {key+'_minus_full': np.array(value)-losses['full128'] for key, value in losses.items()}
        contrasts['matched_character_minus_group'] = np.array(losses['matched_character_scramble'])-losses['group_order']
        report['models'][str(seed)] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                                       'original_score_max_error': parity, 'losses_bits': losses,
                                       'summaries': {k: paired_summary(v, examples, seed=7205) for k, v in losses.items()},
                                       'contrasts': {k: paired_summary(v, examples, seed=7205) for k, v in contrasts.items()}}
        print(json.dumps({'seed': seed, 'scores': {k: float(np.mean(v)) for k, v in losses.items()}}), flush=True)
    report['elapsed_seconds'] = time.monotonic()-start
    report['limitations'] = ['Same previously explored validation targets; no independent confirmation.',
                             'Transcription groups need not be plaintext words.',
                             'Equal-length group swaps restrict the tested order changes; all targets retained including no-op corruptions.',
                             'Matched character control is a constrained random walk, not guaranteed uniform mixing.',
                             'Block sizes have different disruption amounts; no exact additive information decomposition.']
    write_json(root/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='outputs/EXP-0007')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output_root, resolve_device(args.device))


if __name__ == '__main__':
    main()
