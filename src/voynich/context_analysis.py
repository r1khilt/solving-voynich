"""EXP-0003: paired context damage and final-position head interventions."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from voynich.interpret import patch_activation
from voynich.runtime import PageWindows, corpus_identity, digest, environment, resolve_device, write_json
from voynich.train import load_checkpoint


def sample_contexts(data, width=128, per_page=8, seed=1201):
    """Fixed eligible next-target positions, sampled without examining model scores."""
    rng = np.random.default_rng(seed)
    examples = []
    for page in data.pages:
        seq = data.sequences[page['page_id']]
        eligible = [p for p in range(width + 1, len(seq) - 1) if seq[p] not in data.ignored_ids]
        for p in sorted(rng.choice(eligible, min(per_page, len(eligible)), replace=False).tolist()):
            examples.append({'page_id': page['page_id'], 'leaf_id': page['leaf_id'],
                             'position': p, 'prefix': seq[p-width:p], 'target': seq[p]})
    if not examples:
        raise ValueError('No eligible contexts')
    return examples


def remote_shuffle(prefixes, keep=16, seed=1202):
    """Preserve the local suffix exactly and shuffle only the distant prefix."""
    if prefixes.ndim != 2 or not 0 < keep < prefixes.shape[1]:
        raise ValueError('Require a matrix and a proper nonempty suffix')
    rng = np.random.default_rng(seed)
    result = prefixes.copy()
    for row in result:
        row[:-keep] = rng.permutation(row[:-keep])
    return result


def paired_summary(values, examples, seed=1203):
    """Report sample mean and equal-leaf mean with descriptive cluster bootstrap."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(examples) or not np.isfinite(values).all():
        raise ValueError('Invalid measurements')
    by_leaf = defaultdict(list)
    for value, example in zip(values, examples, strict=True):
        by_leaf[example['leaf_id']].append(float(value))
    means = np.array([np.mean(v) for v in by_leaf.values()])
    rng = np.random.default_rng(seed)
    boot = rng.choice(means, size=(2000, len(means)), replace=True).mean(1)
    return {'sample_mean': float(values.mean()), 'equal_leaf_mean': float(means.mean()),
            'equal_leaf_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'leaf_means': {k: float(np.mean(v)) for k, v in by_leaf.items()},
            'leaf_count': len(means), 'target_count': len(values)}


def bits(logits, target):
    return F.cross_entropy(logits[:, -1], target, reduction='none') / math.log(2)


@torch.no_grad()
def run(checkpoint, data_dir, device='cpu', heads=False):
    identity = corpus_identity(data_dir)
    model, payload = load_checkpoint(checkpoint, device)
    model.eval()
    if payload['corpus_identity'] != identity or model.config.context_length < 128:
        raise ValueError('Checkpoint/corpus or context mismatch')
    windows = PageWindows(data_dir, 'validation', model.config.context_length)
    examples = sample_contexts(windows)
    prefixes = np.array([e['prefix'] for e in examples])
    corrupted = remote_shuffle(prefixes)
    conditions = {'full128': prefixes, 'last64': prefixes[:, -64:],
                  'last16': prefixes[:, -16:], 'last4': prefixes[:, -4:],
                  'remote_shuffle_keep16': corrupted}
    losses = defaultdict(list)
    effects = defaultdict(list)
    identity_max = restore_max = 0.0
    sites = [f'blocks.{i}.attn.result' for i in range(model.config.n_layers)]
    final_site = f'blocks.{model.config.n_layers-1}.resid_post'
    for start in range(0, len(examples), 8):
        end = min(start + 8, len(examples))
        target = torch.tensor([e['target'] for e in examples[start:end]], device=device)
        tensors = {k: torch.tensor(v[start:end].copy(), device=device) for k, v in conditions.items()}
        for name, x in tensors.items():
            losses[name].extend(bits(model(x).logits, target).cpu().tolist())
        if not heads:
            continue
        clean = model(tensors['full128'], cache_names=[*sites, final_site])
        corrupt = model(tensors['remote_shuffle_keep16'], cache_names=sites)
        clean_bits = bits(clean.logits, target)
        corrupt_bits = bits(corrupt.logits, target)
        for layer, site in enumerate(sites):
            for head in range(model.config.n_heads):
                label = f'L{layer}H{head}'
                donors = {'clean_patch_gain': clean.cache[site],
                          'mismatched_patch_gain': clean.cache[site].roll(1, dims=0)}
                for name, donor in donors.items():
                    out = model(tensors['remote_shuffle_keep16'], interventions={
                        site: patch_activation(donor, positions=[-1], head=head)})
                    effects[f'{label}/{name}'].extend((corrupt_bits-bits(out.logits, target)).cpu().tolist())
                out = model(tensors['full128'], interventions={
                    site: patch_activation(torch.zeros_like(clean.cache[site]), positions=[-1], head=head)})
                effects[f'{label}/zero_ablation_damage'].extend((bits(out.logits, target)-clean_bits).cpu().tolist())
        identity_out = model(tensors['remote_shuffle_keep16'], interventions={
            sites[0]: patch_activation(corrupt.cache[sites[0]], positions=[-1])})
        restore = model(tensors['remote_shuffle_keep16'], interventions={
            final_site: patch_activation(clean.cache[final_site], positions=[-1])})
        identity_max = max(identity_max, float((identity_out.logits[:, -1]-corrupt.logits[:, -1]).abs().max().cpu()))
        restore_max = max(restore_max, float((restore.logits[:, -1]-clean.logits[:, -1]).abs().max().cpu()))
    clean_loss = np.array(losses['full128'])
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
              'checkpoint_step': payload['step'], 'corpus_identity': identity,
              'environment': environment(), 'device': device, 'test_evaluated': False,
              'sampling': {'width': 128, 'per_page': 8, 'seed': 1201, 'shuffle_seed': 1202},
              'examples': [{k: v for k, v in e.items() if k != 'prefix'} for e in examples],
              'losses_bits': dict(losses),
              'conditions': {k: {'bits': paired_summary(v, examples),
                                'damage_vs_full128': paired_summary(np.array(v)-clean_loss, examples)}
                             for k, v in losses.items()},
              'head_effects': {k: paired_summary(v, examples) for k, v in effects.items()},
              'head_effects_per_target': dict(effects),
              'controls': {'identity_max_logit_error': identity_max if heads else None,
                           'full_residual_restore_max_logit_error': restore_max if heads else None},
              'limitations': ['Validation-selected model and exploratory localization; no semantics.',
                              'Context truncation/shuffling is a distribution shift.',
                              'Final-position interventions are not complete circuit tests.',
                              'Leaf-bootstrap intervals are descriptive; few clusters and no multiplicity correction.',
                              'Sampling denominator differs from whole-validation training scores.']}
    if heads and (identity_max > 1e-5 or restore_max > 1e-5):
        raise ValueError('Intervention identity/restore control failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/processed/zl3b')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--heads', action='store_true')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Use a new output file')
    torch.set_num_threads(4)
    report = run(args.checkpoint, args.data, resolve_device(args.device), args.heads)
    write_json(args.output, report)
    print({k: v['bits']['sample_mean'] for k, v in report['conditions'].items()})


if __name__ == '__main__':
    main()
