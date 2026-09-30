"""Read-only comparisons after complete frozen fits and independent algorithmic audits."""
import hashlib
import json

import torch

from scripts.run_blind_channel_dev004 import save_new
from scripts.run_latin_source_model001 import OUT, ROOT, SEEDS, artifact


def checked(spec):
    path = ROOT / spec['path']
    if artifact(path) != spec:
        raise ValueError('Artifact identity mismatch')
    return path


def summarize():
    neural, statistical, identities, initializations = [], {}, {}, {}
    for dataset in ('small', 'large'):
        path = ROOT / f'results/LATIN-SOURCE-COMPACT-001/{dataset}.json'
        result = json.loads(path.read_text())
        audit = json.loads(path.with_name(path.stem + '-audit.json').read_text())
        if audit['status'] != 'PASS' or audit['result'] != artifact(path):
            raise ValueError('Statistical audit missing or changed')
        identities[dataset] = json.loads(checked(result['inputs']).read_text())
        statistical[dataset] = {'selected_bpc': result['selected']['selection_bpc'],
                                'tau': result['selected']['tau'], 'source_result': artifact(path)}
        for seed in SEEDS:
            path = OUT / f'{dataset}-seed{seed}.json'
            run = json.loads(path.read_text())
            audit_path = OUT / f'{dataset}-seed{seed}-audit.json'
            audit = json.loads(audit_path.read_text())
            if audit['status'] != 'PASS' or audit['result'] != artifact(path):
                raise ValueError('Neural audit missing or changed')
            if json.loads(checked(run['inputs']).read_text()) != identities[dataset]:
                raise ValueError('Training/selection inputs differ between source families')
            initial = torch.load(checked(run['checkpoints'][0]['checkpoint']), weights_only=True, map_location='cpu')
            identity = hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial['state_dict'].values())).hexdigest()
            if identity != run['initialization_sha256']:
                raise ValueError('Recorded initialization hash disagrees with step0 weights')
            initializations[dataset, seed] = identity
            if run['updates'] != 6000 or run['target_characters'] != 49_152_000:
                raise ValueError('Incomplete paired exposure')
            neural.append({'dataset': dataset, 'seed': seed, 'parameters': run['parameters'],
                           'selected_step': run['selected']['step'],
                           'selected_bpc': run['selected']['selection']['bits_per_character'],
                           'final_bpc': run['checkpoints'][-1]['selection']['bits_per_character'],
                           'curve': [{'step': c['step'], 'bpc': c['selection']['bits_per_character']} for c in run['checkpoints']],
                           'resources': run['resources'], 'source_result': artifact(path), 'audit': artifact(audit_path)})
    if any(initializations['small', s] != initializations['large', s] for s in SEEDS):
        raise ValueError('Paired initial weights differ')
    pairs = []
    for seed in SEEDS:
        small = next(r for r in neural if r['dataset'] == 'small' and r['seed'] == seed)
        large = next(r for r in neural if r['dataset'] == 'large' and r['seed'] == seed)
        data_gain = small['selected_bpc'] - large['selected_bpc']
        architecture_gain = statistical['large']['selected_bpc'] - large['selected_bpc']
        pairs.append({'seed': seed, 'data_gain_bpc': data_gain, 'large_neural_gain_over_large_statistical_bpc': architecture_gain,
                      'supplementary_fixed_thresholds_met': data_gain >= .10 and architecture_gain >= .02})
    original_failure = json.loads((OUT / 'statistical-large-failure.json').read_text())
    result = {'experiment': 'LATIN-SOURCE-MODEL-001', 'status': 'completed_four_neural_fits_with_recorded_statistical_cap_failure',
              'original_full_architecture_screen': 'UNAVAILABLE_failed_large_statistical_arm',
              'statistical_revision': 'LATIN-SOURCE-COMPACT-001', 'neural_runs': neural, 'statistical': statistical,
              'paired_comparisons': pairs,
              'supplementary_priority_screen_pass': all(p['supplementary_fixed_thresholds_met'] for p in pairs),
              'all_paired_initial_weights_verified': True, 'all_same_dataset_inputs_identical_between_families': True,
              'total_presented_target_characters': 196_608_000,
              'neural_fit_wall_seconds_sum': sum(r['resources']['wall_seconds'] for r in neural),
              'neural_fit_cpu_seconds_sum': sum(r['resources']['cpu_seconds'] for r in neural),
              'training_characters_by_dataset': {d: value['training_characters'] for d, value in identities.items()},
              'original_failure': original_failure, 'paid_experiment_api_cloud_spend_usd': 0,
              'cipher_reading_accuracy_not_measured_here': True}
    save_new(OUT / 'comparison.json', result)
    lines = ['# LATIN-SOURCE-MODEL-001 results', '',
             'All four fixed 7,405,079-parameter recurrent fits completed and passed saved-model audits. '
             'Both seeds used the same initial weights across data sizes. Each fit presented 49,152,000 '
             'target letters, reusing its training corpus; these are not unique training letters.', '',
             '| Training data | Seed | Selected update | Pliny bits/letter | Final-update bits/letter |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in neural:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['selected_step']} | {row['selected_bpc']:.9f} | {row['final_bpc']:.9f} |")
    lines += ['', 'The matched statistical controls in the separately named compact storage revision score '
              f"{statistical['small']['selected_bpc']:.9f} (small) and {statistical['large']['selected_bpc']:.9f} (large). "
              'Both select interpolation mass64. Small uses97,850eligible letters; large6,497,939. '
              'Every family receives identical source segments and the same359,276Plinyselection letters.', '']
    for row in pairs:
        lines.append(f"Seed{row['seed']}: large data improves the selected neural source by{row['data_gain_bpc']:.9f}bits/letter; "
                     f"large neural improves over large statistical by{row['large_neural_gain_over_large_statistical_bpc']:.9f}bits/letter.")
    lines += ['', 'The original large statistical arm failed its1.2million-context cap; its full architecture '
              'screen remains UNAVAILABLE. The separate compact implementation preserves that failure and '
              'exactly reproduces all old small-data counts and scores. Applying the original effect-size '
              f"thresholds to that supplementary comparison gives {result['supplementary_priority_screen_pass']}. "
              'This is an adaptive source-development screen, not a reader qualification.', '',
              'The small neural fits overfit strongly. The coarse common checkpoint schedule may miss a '
              'better small-data checkpoint between registered observations. These results compare the fixed '
              'recipe and budgets, not each model family at its best achievable tuning. Pliny is used for '
              'selection, so these are development losses with selection optimism, not final test estimates. '
              'Only two initialization seeds and one selection author are represented.', '',
              'Audits verify every checkpoint hash and all6,000trace rows per fit, independently replay '
              'the selected full validation from disk with changed batch size, and compare128input '
              'steps with explicit double-precision LSTM gate equations. This summary independently '
              'recomputes all four step0weight hashes and checks exact data identities across families. '
              'Root-authored algorithmic checks are not independent researcher replication.', '',
              f"Summed neural fit wall time:{result['neural_fit_wall_seconds_sum']:.3f}s; "
              f"host CPU:{result['neural_fit_cpu_seconds_sum']:.3f}s. GPU compute is reflected in wall time, "
              'not that host-CPU total. No paid experiment API/cloud calls; local energy unmetered. '
              'All checkpoints and raw data stay local and ignored; tracked manifests preserve full hashes.', '',
              'No fresh cipher or Voynich reading is established here. The separately prepared '
              '[NEURAL-READER-001](NEURAL-READER-001.md) requires a new source/panel/prediction freeze '
              'and true-path/search diagnostics before an accuracy claim.', '',
              '![All source-learning curves and selected development losses](../../results/LATIN-SOURCE-MODEL-001/learning-curves.png)', '']
    target = ROOT / 'docs/experiments/LATIN-SOURCE-MODEL-001-results.md'
    with target.open('x') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps({'pairs': pairs, 'total_wall_seconds': result['neural_fit_wall_seconds_sum']}))


if __name__ == '__main__':
    summarize()
