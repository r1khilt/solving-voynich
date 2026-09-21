"""Validate and summarize the registered EXP-0002 sweep without manuscript test access."""

import json
from pathlib import Path
import statistics

import torch

from voynich.runtime import PageWindows, corpus_identity, digest, environment, evaluate_model, resolve_device, write_json
from voynich.train import load_checkpoint


def main():
    torch.set_num_threads(4)
    root = Path('outputs/EXP-0002')
    destination = Path('results/EXP-0002/comparison.json')
    if destination.exists():
        raise ValueError('Do not overwrite completed comparison')
    identity = corpus_identity('data/processed/zl3b')
    configs = ['small', 'reference', 'mtp', 'qk_norm', 'gated_attention', 'attention_only']
    runs, groups = [], {}
    source_identity = None
    for cfg in configs:
        members = []
        for seed in [42, 43, 44]:
            path = root/f'{cfg}-seed{seed}'
            summary = json.loads((path/'summary.json').read_text())
            manifest = json.loads((path/'manifest.json').read_text())
            history = json.loads((path/'history.json').read_text())
            if manifest['corpus_identity'] != identity or summary['test_evaluated']:
                raise ValueError('Corpus or test isolation mismatch')
            if manifest['environment']['git_dirty'] or manifest['training_config']['seed'] != seed:
                raise ValueError('Source dirty or wrong seed')
            current = (manifest['environment']['git_commit'], manifest['environment']['source_sha256'])
            source_identity = source_identity or current
            if current != source_identity:
                raise ValueError('Inconsistent training source across runs')
            best = next(item for item in history if item['step'] == summary['best_step'])
            row = {'config': cfg, 'seed': seed, **summary,
                   'best_validation': best['validation'],
                   'learning_curve': [{'step': item['step'], 'validation_bits': item['validation']['bits_per_token'],
                                       'elapsed_seconds': item['elapsed_seconds'],
                                       'sampled_targets': item.get('scored_training_tokens_this_invocation', 0)}
                                      for item in history],
                   'checkpoints': {f.name: {'sha256': digest(f), 'bytes': f.stat().st_size}
                                   for f in sorted(path.glob('*.pt'))}}
            runs.append(row)
            members.append(row)
        scores = [m['best_validation_bits_per_token'] for m in members]
        groups[cfg] = {'mean_validation_bits': statistics.mean(scores),
                       'sample_sd': statistics.stdev(scores), 'scores': scores,
                       'best_steps': [m['best_step'] for m in members],
                       'parameters': members[0]['parameter_count']}
    best_mean = min(g['mean_validation_bits'] for g in groups.values())
    eligible = [k for k, g in groups.items() if g['mean_validation_bits']-best_mean < .02]
    selected = min(eligible, key=lambda k: (groups[k]['parameters'], groups[k]['mean_validation_bits']))
    device = resolve_device('mps')
    training = PageWindows('data/processed/zl3b', 'train', 256)
    train_diagnostics = {}
    for seed in [42, 43, 44]:
        model, payload = load_checkpoint(root/f'{selected}-seed{seed}'/'best.pt', device)
        if payload['corpus_identity'] != identity:
            raise ValueError('Checkpoint provenance mismatch')
        train_diagnostics[str(seed)] = evaluate_model(model, training, device)
    report = {'experiment': 'EXP-0002', 'corpus_identity': identity, 'training_source_commit': source_identity[0],
              'training_source_sha256': source_identity[1], 'analysis_environment': environment(),
              'groups': groups, 'selected_for_interpretation': selected,
              'selection_rule': 'Smallest parameter count within 0.02 bits of minimum mean; preregistered engineering threshold',
              'selected_train_diagnostics': train_diagnostics, 'runs': runs,
              'test_evaluated': False, 'summed_training_seconds': sum(r['elapsed_seconds'] for r in runs),
              'total_steps': sum(r['completed_steps'] for r in runs),
              'total_scored_training_targets': sum(r['scored_training_tokens_this_invocation'] for r in runs),
              'limitations': 'Validation-selected scores; repeated validation tuning; three optimization seeds, one manuscript; no decipherment claim'}
    write_json(destination, report)
    print(json.dumps({'groups': groups, 'selected': selected, 'total_steps': report['total_steps']}, indent=2))


if __name__ == '__main__':
    main()
