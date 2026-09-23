"""Independent compact PATH-0010 audit; never runs the research model."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from voynich.workspace.campaign import answer_correct, canonical_digest, digest, write_json
from voynich.workspace.path10_tasks import path10_tasks


ROOT = Path('results/PATH-0010')
RAW = Path('outputs/PATH-0010')
WINDOW = (28, 29, 30, 31)
MASKS = tuple(range(16))
SINGLES = (1, 2, 4, 8)
SOURCE_PATHS = {
    'src/voynich/workspace/path10_tasks.py',
    'src/voynich/workspace/path10_campaign.py',
    'src/voynich/workspace/path3_backend.py',
    'src/voynich/workspace/path6_tasks.py',
    'src/voynich/workspace/path6_campaign.py',
    'src/voynich/workspace/path8_tasks.py',
    'src/voynich/workspace/route_backend.py',
    'src/voynich/workspace/campaign.py',
    'src/voynich/workspace/causal_campaign.py',
    'src/voynich/workspace/geometry.py',
    'scripts/path0010_analyze.py',
    'configs/jspace0001.json',
    'docs/experiments/PATH-0010.md',
}


def read(path):
    return json.loads(path.read_text())


def layers(mask):
    return [block for index, block in enumerate(WINDOW) if mask & (1 << index)]


def interaction_terms(margins):
    """Independent anchored inclusion-exclusion check of the saved margins."""
    effects = {mask: margins[0]-margins[mask] for mask in MASKS}
    terms = {}
    for mask in MASKS[1:]:
        terms[mask] = sum(((-1)**((mask.bit_count()-part.bit_count())))*effects[part]
                          for part in MASKS if part & mask == part)
    assert np.isclose(sum(terms.values()), effects[15], atol=1e-8, rtol=0)
    return effects, terms


def _bootstrap(lookup, success):
    members = {bundle: sorted(key for key in success if int(key.split('/')[1]) == bundle)
               for bundle in range(8)}
    d = {bundle: float(np.mean([
        min(lookup[key, 'subset', bit]['donor_margin']
            - lookup[key, 'subset', 15]['donor_margin'] for bit in SINGLES)
        for key in keys])) for bundle, keys in members.items()}
    rng = np.random.default_rng(510111)
    means, gaps = [], []
    for _ in range(10_000):
        picks = rng.integers(0, 8, size=8)
        means.append(float(np.mean([d[int(bundle)] for bundle in picks])))
        keys = [key for bundle in picks for key in members[int(bundle)]]
        full = np.mean([not lookup[key, 'subset', 15]['donor_first_token'] for key in keys])
        single = max(np.mean([not lookup[key, 'subset', bit]['donor_first_token']
                              for key in keys]) for bit in SINGLES)
        gaps.append(float(full-single))
    return {
        'resamples': 10_000,
        'equal_bundle_mean_d_95': [float(np.quantile(means, .025)),
                                   float(np.quantile(means, .975))],
        'full_minus_best_single_rate_95': [float(np.quantile(gaps, .025)),
                                            float(np.quantile(gaps, .975))],
    }


def audit():
    inputs = read(ROOT/'inputs.json')
    tasks = path10_tasks()
    assert canonical_digest(tasks) == inputs['tasks_sha256']
    assert inputs['seed'] == 510111 and inputs['upstream'] == 23
    assert inputs['window'] == list(WINDOW) and inputs['masks'] == list(MASKS)
    assert inputs['all_cut'] == list(range(24, 36))
    assert inputs['cap_seconds'] == 1800 and inputs['memory_bytes_cap'] == 45_000_000_000
    assert inputs['source_worktree_status'] == []
    assert set(inputs['source_sha256']) == SOURCE_PATHS
    assert digest(RAW/'rendered-inputs.json') == inputs['rendered_inputs_sha256']
    assert digest('results/JSPACE-0001/inputs.json') == inputs['model_input_sha256']
    for path, expected in inputs['source_sha256'].items():
        blob = subprocess.check_output(['git', 'show', f"{inputs['source_revision']}:{path}"])
        assert hashlib.sha256(blob).hexdigest() == expected, path
    rendered = read(RAW/'rendered-inputs.json')
    changed = {}
    for task in tasks:
        source = rendered[canonical_digest(task['prompt'])]
        donor = rendered[canonical_digest(task['donor_prompt'])]
        assert len(source) == len(donor)
        positions = np.flatnonzero(np.asarray(source) != np.asarray(donor)).tolist()
        assert len(positions) >= 2 and positions[-1] < len(source)-8
        changed[task['id']] = positions
    assert canonical_digest(changed) == inputs['changed_positions_sha256']

    baselines = read(ROOT/'baselines.json')
    assert [r['id'] for r in baselines] == [r['id'] for r in tasks]
    task_lookup = {r['id']: r for r in tasks}
    eligible, copies = set(), set()
    donor_first = {}
    for baseline in baselines:
        task = task_lookup[baseline['id']]
        assert baseline['bundle'] == task['bundle'] and baseline['family'] == task['family']
        assert baseline['source_correct'] == answer_correct(
            baseline['source_text'], [task['answer']])
        assert baseline['donor_correct'] == answer_correct(
            baseline['donor_text'], [task['donor_answer']])
        first = {}
        for side, prompt in (('source', task['prompt']), ('donor', task['donor_prompt'])):
            local = RAW/'baselines'/(canonical_digest(prompt)+'.npz')
            assert digest(local) == baseline[f'{side}_arrays_sha256']
            with np.load(local) as arrays:
                assert np.isfinite(arrays['logits']).all() and np.isfinite(arrays['field']).all()
                assert all(f'write_{layer}' in arrays for layer in range(24, 36))
                first[side] = int(arrays['logits'].argmax())
        donor_first[task['id']] = first['donor']
        if baseline['source_correct'] and baseline['donor_correct']:
            assert baseline['first_token_differs'] == (first['source'] != first['donor'])
        if task['family'] == 'binding' and baseline['source_correct'] and baseline['donor_correct'] \
                and baseline['first_token_differs']:
            eligible.add(task['id'])
        if task['family'] == 'copy' and baseline['source_correct']:
            copies.add(task['id'])
    assert read(ROOT/'capability.json') == {'eligible': len(eligible), 'source_copy': len(copies)}
    decision = read(ROOT/'decision.json')
    if len(eligible) < 24 or len(copies) < 14:
        assert decision['status'] == 'inconclusive'
        assert not (ROOT/'rows.json').exists()
        report = {'passed': True, 'status': decision['status'], 'rows': 0,
                  'limitation': 'Baseline gate stopped before interventions.'}
        write_json(ROOT/'audit.json', report)
        return report

    rows = read(ROOT/'rows.json')
    expected = {(task['id'], 'subset', mask) for task in tasks for mask in MASKS}
    expected.update((task['id'], name, None) for task in tasks
                    for name in ('random_full', 'all_cut'))
    keys = [(r['task_id'], r['condition'], r['mask']) for r in rows]
    assert len(keys) == len(expected) == 864 and set(keys) == expected
    assert [json.loads(line) for line in (RAW/'observations.jsonl').read_text().splitlines()] == rows
    lookup = {(r['task_id'], r['condition'], r['mask']): r for r in rows}
    for row in rows:
        task = task_lookup[row['task_id']]
        assert row['bundle'] == task['bundle'] and row['family'] == task['family']
        assert row['layers'] == (layers(row['mask']) if row['mask'] is not None else None)
        assert row['eligible'] == (task['id'] in eligible)
        assert row['baseline_copy'] == (task['id'] in copies)
        assert row['desired_answer'] == (answer_correct(row['text'], [task['donor_answer']])
                                         if task['family'] == 'binding' else False)
        assert row['source_answer'] == answer_correct(row['text'], [task['answer']])
        assert row['copy_preserved'] == (answer_correct(row['text'], [task['answer']])
                                        if task['family'] == 'copy' else False)
        assert row['donor_first_token'] == (bool(row['tokens'])
                                            and row['tokens'][0] == donor_first[task['id']])
        assert row['donor_margin'] is None or np.isfinite(row['donor_margin'])

    success = {key for key in eligible if lookup[key, 'subset', 0]['desired_answer']}
    summary = read(ROOT/'summary.json')
    assert summary['eligible'] == len(eligible)
    assert summary['source_copy'] == len(copies)
    assert summary['upstream_success'] == len(success)
    if success:
        by_bundle = {}
        effect_by_bundle = {mask: [] for mask in MASKS[1:]}
        term_by_bundle = {mask: [] for mask in MASKS[1:]}
        for bundle in range(8):
            members = sorted(key for key in success if int(key.split('/')[1]) == bundle)
            by_bundle[str(bundle)] = (float(np.mean([
                min(lookup[key, 'subset', bit]['donor_margin']
                    - lookup[key, 'subset', 15]['donor_margin'] for bit in SINGLES)
                for key in members])) if members else None)
            for key in members:
                margins = {mask: lookup[key, 'subset', mask]['donor_margin'] for mask in MASKS}
                effects, terms = interaction_terms(margins)
                for mask in MASKS[1:]:
                    effect_by_bundle[mask].append((bundle, effects[mask]))
                    term_by_bundle[mask].append((bundle, terms[mask]))
        covered = sum(value is not None for value in by_bundle.values())
        assert summary['bundle_coverage'] == covered
        assert all(np.isclose(summary['d_by_bundle'][key], value)
                   for key, value in by_bundle.items() if value is not None)
        assert all(summary['d_by_bundle'][key] is None for key, value in by_bundle.items()
                   if value is None)
        assert summary['positive_d_bundles'] == sum(value is not None and value > 0
                                                     for value in by_bundle.values())
        if covered == 8:
            assert np.isclose(summary['equal_bundle_mean_d'],
                              np.mean(list(by_bundle.values())))
            assert summary['bootstrap'] == _bootstrap(lookup, success)
        else:
            assert summary['equal_bundle_mean_d'] is None and summary['bootstrap'] is None
        for mask in MASKS[1:]:
            for name, saved in (('equal_bundle_mean_effects', effect_by_bundle),
                                ('equal_bundle_mean_mobius', term_by_bundle)):
                bundles = sorted({bundle for bundle, _ in saved[mask]})
                means = [np.mean([value for group, value in saved[mask] if group == bundle])
                         for bundle in bundles]
                assert np.isclose(summary[name][str(mask)], np.mean(means))
        full_effect = summary['equal_bundle_mean_effects']['15']
        singles = sum(summary['equal_bundle_mean_effects'][str(bit)] for bit in SINGLES)
        assert np.isclose(summary['full_minus_sum_singles'], full_effect-singles)
        for mask in MASKS:
            group = [lookup[key, 'subset', mask] for key in success]
            cell = summary['subset'][str(mask)]
            assert cell['layers'] == layers(mask) and cell['denominator'] == len(success)
            assert cell['donor_first_removed'] == sum(not r['donor_first_token'] for r in group)
            assert cell['source_full'] == sum(r['source_answer'] for r in group)
            assert cell['donor_full'] == sum(r['desired_answer'] for r in group)
            assert cell['copy_preserved'] == sum(lookup[key, 'subset', mask]['copy_preserved']
                                                 for key in copies)
            assert np.isclose(cell['mean_donor_margin'], np.mean([r['donor_margin'] for r in group]))
        for name in ('random_full', 'all_cut'):
            group = [lookup[key, name, None] for key in success]
            cell = summary['other'][name]
            assert cell['denominator'] == len(success)
            assert cell['donor_first_removed'] == sum(not r['donor_first_token'] for r in group)
            assert cell['source_full'] == sum(r['source_answer'] for r in group)
            assert cell['donor_full'] == sum(r['desired_answer'] for r in group)
            assert np.isclose(cell['mean_donor_margin'],
                              np.mean([r['donor_margin'] for r in group]))
            assert cell['copy_preserved'] == sum(lookup[key, name, None]['copy_preserved']
                                                for key in copies)
    assert all(value < .002 for value in decision['controls'].values())
    assert 'whole_donor' in decision['controls'] and 'all_cut_source' in decision['controls']
    upstream_rate = len(success)/len(eligible)
    copy_upstream = summary['subset']['0']['copy_preserved']/len(copies) if success else 0
    if upstream_rate < .75 or summary['bundle_coverage'] < 8 or copy_upstream < .9:
        assert decision['status'] == 'uninformative_upstream'
    else:
        full = summary['subset']['15']['donor_first_removed']/len(success)
        random = summary['other']['random_full']['donor_first_removed']/len(success)
        single_rates = {str(bit): summary['subset'][str(bit)]['donor_first_removed']/len(success)
                        for bit in SINGLES}
        single = max(single_rates.values())
        full_copy = summary['subset']['15']['copy_preserved']/len(copies)
        supported = (full >= .35 and full-random >= .2 and full-single >= .2
                     and full_copy >= .9 and summary['equal_bundle_mean_d'] >= 2.0
                     and summary['positive_d_bundles'] >= 6)
        expected_status = ('exploratory_supported' if supported else
                           'uninformative_positive_control' if full < .35 else
                           'exploratory_failed')
        assert decision['status'] == expected_status and decision['supported'] == supported
        assert np.isclose(decision['upstream_rate'], upstream_rate)
        assert np.isclose(decision['full_rate'], full)
        assert np.isclose(decision['random_rate'], random)
        assert decision['single_rates'] == single_rates
        assert np.isclose(decision['best_single_rate'], single)
        assert np.isclose(decision['full_minus_best_single_rate'], full-single)
        assert np.isclose(decision['equal_bundle_mean_d'], summary['equal_bundle_mean_d'])
        assert decision['positive_d_bundles'] == summary['positive_d_bundles']
    report = {'passed': True, 'status': decision['status'], 'rows': len(rows),
              'eligible': len(eligible), 'upstream_success': len(success),
              'limitation': 'Compact audit cannot recompute unsaved intervention logits.'}
    write_json(ROOT/'audit.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
