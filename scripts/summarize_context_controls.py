"""Archive EXP-0005 compact measurements while keeping full donor ledger local."""

import json
from pathlib import Path
import statistics

import numpy as np

from voynich.runtime import corpus_identity, digest, write_json


def main():
    source = Path('outputs/EXP-0005/report.json')
    output = Path('results/EXP-0005/report.json')
    if output.exists():
        raise ValueError('Do not overwrite completed summary')
    report = json.loads(source.read_text())
    if report['environment']['git_dirty'] or report['test_evaluated']:
        raise ValueError('Experiment source or holdout integrity failed')
    if report['corpus_identity'] != corpus_identity('data/processed/zl3b'):
        raise ValueError('Corpus drift')
    for model in report['models'].values():
        if digest(model['checkpoint']) != model['checkpoint_sha256']:
            raise ValueError('Checkpoint drift')
    donors = report.pop('donors')
    report['donor_distance_summary'] = {}
    for name in ['same_random', 'same_matched', 'different_random', 'different_matched']:
        values = [d['histogram_total_variation'] for d in donors if d['condition'] == name]
        report['donor_distance_summary'][name] = {'mean': float(np.mean(values)), 'count': len(values),
                                                  'quartiles': np.quantile(values, [.25, .5, .75]).tolist()}
    report['across_seed_mean_bits'] = {name: statistics.mean(m['summaries'][name]['sample_mean'] for m in report['models'].values())
                                      for name in report['models']['42']['summaries']}
    report['full_report'] = {'path': str(source), 'sha256': digest(source), 'bytes': source.stat().st_size,
                             'note': 'Full donor coordinates remain in ignored outputs; regenerable from fixed seeds/source. Per-target scores and coordinates retained here.'}
    write_json(output, report)
    print(json.dumps(report['across_seed_mean_bits'], indent=2))


if __name__ == '__main__':
    main()
