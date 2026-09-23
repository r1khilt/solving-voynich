"""Independent compact-result audit for PATH-0006; never reruns the model."""

import json
from pathlib import Path

import numpy as np

from voynich.workspace.campaign import answer_correct, canonical_digest, digest, write_json
from voynich.workspace.path6_campaign import SEED, WINDOWS, bootstrap_difference
from voynich.workspace.path6_tasks import path6_tasks

ROOT = Path('results/PATH-0006')
RAW = Path('outputs/PATH-0006')


def read(path):
    return json.loads(path.read_text())


def audit():
    inputs = read(ROOT/'inputs.json')
    tasks = {split: path6_tasks(split) for split in ('discovery', 'confirmation')}
    assert canonical_digest(tasks) == inputs['tasks_sha256']
    assert digest(RAW/'rendered-inputs.json') == inputs['rendered_inputs_sha256']
    for name, expected in inputs['source_sha256'].items():
        assert digest(name) == expected, name
    assert digest('results/JSPACE-0001/inputs.json') == inputs['model_input_sha256']
    assert inputs['seed'] == SEED
    rendered = read(RAW/'rendered-inputs.json')
    positions = {}
    for split in tasks:
        for task in tasks[split]:
            source = rendered[canonical_digest(task['prompt'])]
            donor = rendered[canonical_digest(task['donor_prompt'])]
            assert len(source) == len(donor)
            indices = np.flatnonzero(np.asarray(source) != np.asarray(donor)).tolist()
            assert len(indices) >= 2 and indices[-1] < len(source)-8
            positions[task['id']] = indices
    assert canonical_digest(positions) == inputs['changed_positions_sha256']

    decision = read(ROOT/'decision.json')
    observed_splits = []
    all_rows = []
    diagnostics = {}
    for split in tasks:
        baseline_path = ROOT/f'{split}-baselines.json'
        if not baseline_path.exists():
            break
        baselines = read(baseline_path)
        assert [r['id'] for r in baselines] == [r['id'] for r in tasks[split]]
        indexed = {r['id']: r for r in tasks[split]}
        eligible, copies = set(), set()
        for row in baselines:
            task = indexed[row['id']]
            assert row['source_correct'] == answer_correct(row['source_text'], [task['answer']])
            assert row['donor_correct'] == answer_correct(row['donor_text'], [task['donor_answer']])
            for label, prompt in (('source', task['prompt']), ('donor', task['donor_prompt'])):
                local = RAW/'baselines'/(canonical_digest(prompt)+'.npz')
                assert digest(local) == row[f'{label}_arrays_sha256']
                with np.load(local) as arrays:
                    assert np.isfinite(arrays['logits']).all()
                    assert np.isfinite(arrays['field']).all()
                    assert all(f'write_{layer}' in arrays for layer in range(24, 36))
            with np.load(RAW/'baselines'/(canonical_digest(task['prompt'])+'.npz')) as source_array, \
                    np.load(RAW/'baselines'/(canonical_digest(task['donor_prompt'])+'.npz')) as donor_array:
                first_differs = int(source_array['logits'].argmax()) != int(donor_array['logits'].argmax())
            assert row['first_token_differs'] == first_differs or not (row['source_correct'] and row['donor_correct'])
            if row['family'] == 'binding' and row['source_correct'] and row['donor_correct'] and row['first_token_differs']:
                eligible.add(row['id'])
            if row['family'] == 'copy' and row['source_correct']:
                copies.add(row['id'])
        capability = read(ROOT/f'{split}-capability.json')
        assert capability == {'eligible': len(eligible), 'source_copy': len(copies)}
        diagnostics[split] = capability
        observed_splits.append(split)
        rows_path = ROOT/f'{split}-rows.json'
        if not rows_path.exists():
            assert decision['status'] == 'inconclusive'
            assert len(eligible) < 24 or len(copies) < 14
            break
        rows = read(rows_path)
        expected_conditions = ('upstream', *WINDOWS, 'all_cut') if split == 'discovery' else (
            'upstream', read(ROOT/'selection.json')['selected_window'], 'all_cut', 'random_selected')
        expected = {(task['id'], condition) for task in tasks[split] for condition in expected_conditions}
        actual = [(r['task_id'], r['condition']) for r in rows]
        assert len(actual) == len(expected) and len(set(actual)) == len(actual) and set(actual) == expected
        for row in rows:
            task = indexed[row['task_id']]
            assert row['split'] == split and row['family'] == task['family']
            assert row['eligible'] == (task['id'] in eligible)
            assert row['baseline_copy'] == (task['id'] in copies)
            assert row['desired_answer'] == (answer_correct(row['text'], [task['donor_answer']])
                                             if task['family'] == 'binding' else False)
            assert row['copy_preserved'] == (answer_correct(row['text'], [task['answer']])
                                            if task['family'] == 'copy' else False)
            assert isinstance(row['tokens'], list) and isinstance(row['donor_first_token'], bool)
            assert row['donor_margin'] is None or np.isfinite(row['donor_margin'])
        all_rows.extend(rows)

    if decision['status'] != 'inconclusive':
        assert observed_splits == ['discovery', 'confirmation']
        discovery = read(ROOT/'discovery-rows.json')
        disc_eligible = {r['id'] for r in read(ROOT/'discovery-baselines.json') if
                         r['family'] == 'binding' and r['source_correct'] and r['donor_correct']
                         and r['first_token_differs']}
        success = {key for key in disc_eligible if next(r for r in discovery if r['task_id'] == key
                                                       and r['condition'] == 'upstream')['desired_answer']}
        ranking = [{
            'window': name, 'upstream_success': len(success),
            'donor_removed': sum(not r['donor_first_token'] for r in discovery
                                 if r['condition'] == name and r['task_id'] in success),
        } for name in WINDOWS]
        selected = sorted(ranking, key=lambda r: (-r['donor_removed'], list(WINDOWS).index(r['window'])))[0]['window']
        assert read(ROOT/'selection.json')['selected_window'] == selected
        assert read(ROOT/'selection.json')['ranking'] == ranking
        confirmation = read(ROOT/'confirmation-rows.json')
        eligible = {r['id'] for r in read(ROOT/'confirmation-baselines.json') if
                    r['family'] == 'binding' and r['source_correct'] and r['donor_correct']
                    and r['first_token_differs']}
        copies = {r['id'] for r in read(ROOT/'confirmation-baselines.json') if
                  r['family'] == 'copy' and r['source_correct']}
        lookup = {(r['task_id'], r['condition']): r for r in confirmation}
        success = {key for key in eligible if lookup[key, 'upstream']['desired_answer']}
        reductions = {condition: sum(not lookup[key, condition]['donor_first_token'] for key in success)
                      for condition in ('upstream', selected, 'all_cut', 'random_selected')}
        preserved = {condition: sum(lookup[key, condition]['copy_preserved'] for key in copies)
                     for condition in reductions}
        summary = read(ROOT/'summary.json')
        assert summary['eligible'] == len(eligible) and summary['upstream_success'] == len(success)
        assert summary['source_copy'] == len(copies) and summary['reductions'] == reductions
        assert summary['copy_preserve'] == preserved and summary['selected_window'] == selected
        assert summary['selected_minus_random_bundle_bootstrap'] == bootstrap_difference(confirmation, success, selected)
        rate = len(success)/len(eligible)
        selected_rate = reductions[selected]/len(success) if success else 0.0
        random_rate = reductions['random_selected']/len(success) if success else 0.0
        informative = rate >= .75 and preserved['upstream']/len(copies) >= .9
        supported = (informative and selected_rate >= .35 and selected_rate-random_rate >= .25
                     and preserved[selected]/len(copies) >= .9)
        assert decision['supported'] == supported and decision['upstream_gate'] == informative
        assert decision['selected_window'] == selected
        assert decision['status'] == ('exploratory_supported' if supported else
                                     ('uninformative_upstream' if not informative else 'exploratory_failed'))
        assert decision['all_cut_max_logit_error'] < .002
        assert decision['whole_donor_max_logit_error'] < .002
        assert decision['random_max_norm_error'] < .002
        assert len(all_rows) == 432
    else:
        assert decision['reason'] in ('discovery_baseline_competence', 'confirmation_baseline_competence')
    stream = [json.loads(line) for line in (RAW/'observations.jsonl').read_text().splitlines()]
    assert stream == all_rows
    if (ROOT/'rows.json').exists():
        assert read(ROOT/'rows.json') == all_rows
    report = {'passed': True, 'status': decision['status'], 'observed_splits': observed_splits,
              'rows': len(all_rows), 'capability': diagnostics,
              'limitation': 'Compact rows cannot independently recompute unsaved intervention logits; numerical maxima come from run.'}
    write_json(ROOT/'audit.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
