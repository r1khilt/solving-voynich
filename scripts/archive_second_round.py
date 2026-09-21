"""Audit and archive the fixed EXP-0006/0007 results and decision criteria."""

import json
from pathlib import Path
import shutil
import statistics

import numpy as np
import torch

from voynich.runtime import digest, write_json


def main():
    root = Path('outputs/EXP-0006')
    dest = Path('results/EXP-0006')
    if (dest/'summary.json').exists():
        raise ValueError('Do not overwrite completed archival summary')
    manifest = json.loads((root/'manifest.json').read_text())
    completed = json.loads((root/'completion.json').read_text())
    if manifest['environment']['git_dirty'] or manifest['manuscript_test_evaluated'] or not completed['frozen_weights_verified']:
        raise ValueError('Source/holdout/frozen-weight check failed')
    for name, spec in manifest['dataset_manifest']['pools'].items():
        if digest(f'data/processed/alignment_v1/{name}.jsonl') != spec['sha256']:
            raise ValueError('Fresh synthetic data drift')
    reports = []
    for path in sorted(root.glob('*-site*.json')):
        report = json.loads(path.read_text())
        if digest(report['checkpoint']) != report['checkpoint_sha256'] or digest(report['basis_file']) != report['basis_sha256']:
            raise ValueError('Weight/basis artifact drift')
        bases = torch.load(report['basis_file'], map_location='cpu', weights_only=True)['bases']
        for basis in bases.values():
            if float((basis.T@basis-torch.eye(3)).abs().max()) > 1e-5:
                raise ValueError('Saved basis is not orthonormal')
        reports.append(report)
        shutil.copyfile(path, dest/path.name)
    if len(reports) != 8:
        raise ValueError('Missing registered model/site results')
    for filename in ['manifest.json', 'completion.json']:
        shutil.copyfile(root/filename, dest/filename)
    groups = {}
    for site in [0, 1]:
        selected = [r for r in reports if r['model_kind'] == 'trained' and r['site'] == f'blocks.{site}.resid_post']
        summaries = {}
        for split in ['test128', 'test96', 'test192', 'preservation']:
            methods = {}
            for method in selected[0]['scores'][split]:
                methods[method] = {metric: statistics.mean(r['scores'][split][method]['mean'][metric] for r in selected)
                                   for metric in selected[0]['scores'][split][method]['mean']}
            methods['random_norm_mean'] = {metric: statistics.mean(methods[f'random_norm_{i}'][metric] for i in range(8))
                                           for metric in methods['learned']}
            summaries[split] = methods
        decisions = []
        for r in selected:
            scores = r['scores']['test128']
            learned = scores['learned']['mean']['oracle_kl']
            gain = scores['clean']['mean']['oracle_kl']-learned
            random_margin = statistics.mean(scores[f'random_norm_{i}']['mean']['oracle_kl'] for i in range(8))-learned
            shuffle_margin = scores['shuffled_supervision']['mean']['oracle_kl']-learned
            decisions.append({'seed': r['model_seed'], 'gain': gain, 'margin_over_random': random_margin,
                              'margin_over_shuffled': shuffle_margin,
                              'meets_registered_threshold': gain >= .10 and random_margin >= .05 and shuffle_margin >= .05})
        groups[str(site)] = {'scores_mean_across_seeds': summaries, 'per_seed_decisions': decisions,
                             'all_seeds_meet_threshold': all(d['meets_registered_threshold'] for d in decisions)}
    summary = {'experiment': 'EXP-0006', 'source_commit': manifest['environment']['git_commit'],
               'completion': completed, 'trained_sites': groups,
               'interpretation': 'Late practical steering criterion can pass while a readout-span control matches it; this is not unique circuit recovery. Early true-vs-shuffled distinction is required.'}
    write_json(dest/'summary.json', summary)
    source7 = Path('outputs/EXP-0007/report.json')
    report7 = json.loads(source7.read_text())
    if report7['environment']['git_dirty'] or report7['test_evaluated']:
        raise ValueError('EXP-0007 source/holdout check failed')
    for model in report7['models'].values():
        if digest(model['checkpoint']) != model['checkpoint_sha256'] or model['original_score_max_error'] > 1e-5:
            raise ValueError('EXP-0007 original-context integrity failed')
    exposure = report7['corruption_exposure']
    report7['exposure_summary'] = {key: float(np.mean([item[key] for item in exposure]))
                                    for key in ['changed_fraction', 'control_difference_fraction']}
    report7['across_seed_mean_bits'] = {key: statistics.mean(m['summaries'][key]['sample_mean'] for m in report7['models'].values())
                                       for key in report7['models']['42']['summaries']}
    report7['raw_report_sha256'] = digest(source7)
    if Path('results/EXP-0007/report.json').exists():
        raise ValueError('Do not overwrite EXP-0007 archive')
    write_json('results/EXP-0007/report.json', report7)
    print(json.dumps({'alignment_thresholds': {s: g['all_seeds_meet_threshold'] for s, g in groups.items()},
                      'order_scale_means': report7['across_seed_mean_bits']}, indent=2))


if __name__ == '__main__':
    main()
