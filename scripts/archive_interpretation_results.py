"""Audit and retain compact results of the fixed EXP-0003/0004 execution."""

import json
from pathlib import Path
import shutil
import statistics

from voynich.runtime import corpus_identity, digest, write_json


def main():
    contexts = []
    selected = json.loads(Path('results/EXP-0002/comparison.json').read_text())['selected_for_interpretation']
    source = None
    def check_source(report):
        nonlocal source
        env = report['environment']
        current = (env['git_commit'], env['source_sha256'])
        source = source or current
        if current != source or env['git_dirty']:
            raise ValueError('Analysis/training source changed within the registered execution')
    for seed in [42, 43, 44]:
        path = Path(f'outputs/EXP-0003/{selected}-seed{seed}.json')
        report = json.loads(path.read_text())
        check_source(report)
        if report['corpus_identity'] != corpus_identity('data/processed/zl3b') or report['test_evaluated']:
            raise ValueError('Manuscript provenance/test isolation failed')
        if contexts and report['examples'] != contexts[0]['examples']:
            raise ValueError('Context targets differ across seeds')
        if report['checkpoint_sha256'] != digest(report['checkpoint']):
            raise ValueError('Weights changed')
        contexts.append(report)
        destination = Path('results/EXP-0003')/path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError('Do not overwrite archived experiment output')
        shutil.copyfile(path, destination)
    context_summary = {'experiment': 'EXP-0003', 'seeds': [42, 43, 44], 'family': selected,
                       'sample_count': len(contexts[0]['examples']), 'test_evaluated': False,
                       'conditions': {key: {'mean_bits': statistics.mean(r['conditions'][key]['bits']['sample_mean'] for r in contexts),
                                            'seed_scores': [r['conditions'][key]['bits']['sample_mean'] for r in contexts],
                                            'mean_damage': statistics.mean(r['conditions'][key]['damage_vs_full128']['sample_mean'] for r in contexts)}
                                      for key in contexts[0]['conditions']}}
    write_json('results/EXP-0003/summary.json', context_summary)
    groups = {}
    runs = []
    for family in ['cycle_null', 'iid']:
        members = []
        identity = corpus_identity(f'data/processed/synthetic_{family}')
        for seed in [42, 43, 44]:
            path = Path(f'outputs/EXP-0004/{family}-seed{seed}')
            report = json.loads((path/'calibration.json').read_text())
            manifest = json.loads((path/'manifest.json').read_text())
            summary = json.loads((path/'summary.json').read_text())
            history = json.loads((path/'history.json').read_text())
            check_source(report)
            check_source(manifest)
            if report['corpus_identity'] != identity or manifest['corpus_identity'] != identity:
                raise ValueError('Synthetic identity mismatch')
            if report['manuscript_test_evaluated'] or report['checkpoint_sha256'] != digest(path/'best.pt'):
                raise ValueError('Manuscript exposure or checkpoint drift')
            if report['probes']['state']['final_residual']['count'] != 4608:
                raise ValueError('Registered test positions differ')
            destination = Path(f'results/EXP-0004/{family}-seed{seed}.json')
            if destination.exists():
                raise ValueError('Do not overwrite archived experiment output')
            shutil.copyfile(path/'calibration.json', destination)
            members.append(report)
            runs.append({'family': family, 'seed': seed, 'training_summary': summary,
                         'manifest': manifest,
                         'learning_curve': [{'step': h['step'], 'validation_bits': h['validation']['bits_per_token']}
                                            for h in history],
                         'checkpoints': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                                         for p in sorted(path.glob('*.pt'))}})
        groups[family] = {'mean_test_character_bits': statistics.mean(r['test_character_bits'] for r in members),
                          'oracle_character_bits': members[0]['oracle_character_bits'],
                          'probes': {task: {feature: {'mean_accuracy': statistics.mean(r['probes'][task][feature]['accuracy'] for r in members),
                                                      'seed_accuracy_sd': statistics.stdev(r['probes'][task][feature]['accuracy'] for r in members)}
                                           for feature in members[0]['probes'][task]}
                                     for task in ['state', 'signal']}}
    result = {'experiment': 'EXP-0004', 'groups': groups, 'runs': runs,
              'source_commit': source[0], 'source_sha256': source[1],
              'summed_training_seconds': sum(r['training_summary']['elapsed_seconds'] for r in runs),
              'total_steps': sum(r['training_summary']['completed_steps'] for r in runs),
              'manuscript_test_evaluated': False, 'synthetic_test_evaluated': True}
    write_json('results/EXP-0004/summary.json', result)
    print(json.dumps({'contexts': context_summary, 'synthetic_groups': groups}, indent=2))


if __name__ == '__main__':
    main()
