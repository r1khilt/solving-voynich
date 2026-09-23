"""Independent provenance, grid, label and decision replay for PATH-0004."""

import json
from pathlib import Path

import numpy as np

from voynich.workspace.campaign import answer_correct, canonical_digest, digest
from voynich.workspace.path4_tasks import path4_tasks

ROOT = Path('results/PATH-0004')
LAYERS = (7, 15, 23)
CONDITIONS = ('identity', 'value', 'suffix', 'value_suffix', 'all_earlier', 'whole', 'suffix_random')


def load(path):
    return json.loads(Path(path).read_text())


def audit():
    manifest = load(ROOT/'inputs.json')
    assert {name: digest(name) for name in manifest['source_sha256']} == manifest['source_sha256']
    assert manifest['seed'] == 51071 and tuple(manifest['layers']) == LAYERS
    assert tuple(manifest['conditions']) == CONDITIONS
    assert manifest['model_input_sha256'] == digest('results/JSPACE-0001/inputs.json')
    tasks = {split: path4_tasks(split) for split in ('discovery', 'confirmation')}
    assert manifest['tasks_sha256'] == canonical_digest(tasks)
    rendered = load('outputs/PATH-0004/rendered-inputs.json')
    assert manifest['rendered_inputs_sha256'] == digest('outputs/PATH-0004/rendered-inputs.json')
    changed = {}
    for group in tasks.values():
        for task in group:
            source = rendered[canonical_digest(task['prompt'])]
            donor = rendered[canonical_digest(task['donor_prompt'])]
            assert len(source) == len(donor)
            positions = np.flatnonzero(np.asarray(source) != np.asarray(donor)).tolist()
            assert len(positions) >= 2 and positions[-1] < len(source)-20
            changed[task['id']] = positions
    assert manifest['changed_positions_sha256'] == canonical_digest(changed)
    qualification = load(ROOT/'qualification.json')
    assert qualification['passed'] and {r['layer'] for r in qualification['rows']} == set(LAYERS)
    assert all(r['max_error'] < .002 and r['argmax_same'] for r in qualification['rows'])

    baselines = {split: load(ROOT/f'{split}-baselines.json') for split in tasks}
    by_baseline = {split: {r['id']: r for r in group} for split, group in baselines.items()}
    all_tasks = {r['id']: r for group in tasks.values() for r in group}
    arrays = {}
    for split, group in tasks.items():
        assert {r['id'] for r in group} == set(by_baseline[split])
        for task in group:
            row = by_baseline[split][task['id']]
            assert row['source_correct'] == answer_correct(row['source_text'], [task['answer']])
            assert row['donor_correct'] == answer_correct(row['donor_text'], [task['donor_answer']])
            for role, prompt in (('source', task['prompt']), ('donor', task['donor_prompt'])):
                file = Path('outputs/PATH-0004/baselines')/(canonical_digest(prompt)+'.npz')
                assert row[role+'_arrays_sha256'] == digest(file)
                arrays[prompt] = np.load(file)
                assert all(np.isfinite(arrays[prompt][str(layer)]).all() for layer in LAYERS)
        eligible = sum(r['family'] == 'composed' and r['source_correct'] and r['donor_correct']
                       and r['first_token_differs'] for r in baselines[split])
        copy = sum(r['family'] == 'copy' and r['source_correct'] for r in baselines[split])
        assert load(ROOT/f'{split}-capability.json') == {'eligible': eligible, 'source_copy': copy}

    decision = load(ROOT/'decision.json')
    stopped = decision['status'] == 'inconclusive'
    if stopped:
        assert decision['reason'] == 'confirmation_baseline_competence'
        assert load(ROOT/'confirmation-capability.json')['eligible'] < 12
        rows = load(ROOT/'discovery-rows.json')
        scoring_tasks = tasks['discovery']
    else:
        rows = load(ROOT/'rows.json')
        assert rows == load(ROOT/'discovery-rows.json')+load(ROOT/'confirmation-rows.json')
        scoring_tasks = [task for group in tasks.values() for task in group]
    assert rows == [json.loads(line) for line in Path('outputs/PATH-0004/observations.jsonl').read_text().splitlines()]
    expected = {(task['id'], layer, condition) for task in scoring_tasks
                for layer in LAYERS for condition in CONDITIONS}
    assert len(rows) == len(expected) == (756 if stopped else 1512)
    assert {(r['task_id'], r['layer'], r['condition']) for r in rows} == expected
    for row in rows:
        task = all_tasks[row['task_id']]
        baseline = by_baseline[task['split']][task['id']]
        assert row['split'] == task['split'] and row['family'] == task['family']
        assert row['bundle'] == task['bundle'] and row['template'] == task['template']
        assert row['eligible'] == (task['family'] == 'composed' and baseline['source_correct']
                                   and baseline['donor_correct'] and baseline['first_token_differs'])
        assert row['baseline_copy'] == (task['family'] == 'copy' and baseline['source_correct'])
        assert row['desired_answer'] == (answer_correct(row['text'], [task['donor_answer']])
                                         if task['family'] == 'composed' else False)
        assert row['copy_preserved'] == (answer_correct(row['text'], [task['answer']])
                                        if task['family'] == 'copy' else False)
        donor_first = int(arrays[task['donor_prompt']]['logits'].argmax())
        assert row['donor_first_token'] == (bool(row['tokens']) and row['tokens'][0] == donor_first)
        source_first = int(arrays[task['prompt']]['logits'].argmax())
        if row['condition'] == 'identity':
            assert row['text'] == baseline['source_text']
            assert not row['tokens'] or row['tokens'][0] == source_first
        if row['condition'] == 'whole':
            assert not row['tokens'] or row['tokens'][0] == donor_first
        delta = arrays[task['donor_prompt']][str(row['layer'])]-arrays[task['prompt']][str(row['layer'])]
        positions = changed[task['id']]
        value = np.zeros_like(delta)
        value[positions] = delta[positions]
        suffix = np.zeros_like(delta)
        suffix[max(positions)+1:-1] = delta[max(positions)+1:-1]
        if row['condition'] == 'identity':
            expected_norm = 0.0
        elif row['condition'] == 'value':
            expected_norm = float(np.linalg.norm(value))
        elif row['condition'] == 'suffix':
            expected_norm = float(np.linalg.norm(suffix))
        elif row['condition'] == 'value_suffix':
            expected_norm = float(np.linalg.norm(value+suffix))
        elif row['condition'] == 'all_earlier':
            expected_norm = float(np.linalg.norm(delta[:-1]))
        elif row['condition'] == 'whole':
            expected_norm = float(np.linalg.norm(delta))
        else:
            # The control norm equals suffix norm by construction, even though direction differs.
            expected_norm = float(np.linalg.norm(suffix))
        assert abs(row['field_norm']-expected_norm) < 1e-2

    if stopped:
        baseline = baselines['discovery']
        eligible = {r['id'] for r in baseline if r['family'] == 'composed' and r['source_correct']
                    and r['donor_correct'] and r['first_token_differs']}
        copies = {r['id'] for r in baseline if r['family'] == 'copy' and r['source_correct']}
        layer_table = {}
        for layer in LAYERS:
            layer_table[str(layer)] = {}
            for name in CONDITIONS:
                chosen = [r for r in rows if r['layer'] == layer and r['condition'] == name]
                layer_table[str(layer)][name] = {
                    'switches': sum(r['desired_answer'] for r in chosen if r['task_id'] in eligible),
                    'eligible': len(eligible),
                    'copies_preserved': sum(r['copy_preserved'] for r in chosen if r['task_id'] in copies),
                    'copies': len(copies),
                }
        return {'status': 'inconclusive', 'reason': decision['reason'], 'rows': len(rows),
                'discovery_eligible': len(eligible), 'confirmation_eligible':
                load(ROOT/'confirmation-capability.json')['eligible'],
                'discovery_table': layer_table}

    confirm_base = baselines['confirmation']
    eligible = {r['id'] for r in confirm_base if r['family'] == 'composed' and r['source_correct']
                and r['donor_correct'] and r['first_token_differs']}
    copies = {r['id'] for r in confirm_base if r['family'] == 'copy' and r['source_correct']}
    confirm = [r for r in rows if r['split'] == 'confirmation' and r['layer'] == 15]
    summary = {}
    for name in CONDITIONS:
        chosen = [r for r in confirm if r['condition'] == name]
        summary[name] = {'eligible': len(eligible),
                         'switches': sum(r['desired_answer'] for r in chosen if r['task_id'] in eligible),
                         'copies': len(copies),
                         'copy_preserved': sum(r['copy_preserved'] for r in chosen if r['task_id'] in copies)}
    assert load(ROOT/'summary.json') == summary
    rates = {name: summary[name]['switches']/len(eligible) for name in CONDITIONS}
    copy_rates = {name: summary[name]['copy_preserved']/len(copies) for name in CONDITIONS}
    positive = rates['whole'] >= .9 and copy_rates['whole'] >= .9
    suffix = positive and rates['suffix'] >= .5 and rates['suffix']-rates['suffix_random'] >= .2 and copy_rates['suffix'] >= .9
    combined = positive and rates['value_suffix'] >= .5 and rates['value_suffix']-rates['value'] >= .2 and copy_rates['value_suffix'] >= .9
    assert decision['status'] == ('exploratory_scored' if positive else 'uninformative_positive_control')
    assert decision['suffix_bypass_supported'] == suffix
    assert decision['combined_rescue_supported'] == combined
    assert decision['rates'] == rates and decision['copy_rates'] == copy_rates
    assert decision['whole_max_logit_error'] < .002 and decision['random_max_norm_error'] < 1e-2
    lookup = {(r['task_id'], r['condition']): r for r in confirm}
    task_bundle = {r['task_id']: r['bundle'] for r in confirm}
    groups = {bundle: sorted(key for key in eligible if task_bundle[key] == bundle) for bundle in range(6)}
    for key, first, second in (
        ('suffix_minus_random_bundle_bootstrap', 'suffix', 'suffix_random'),
        ('combined_minus_value_bundle_bootstrap', 'value_suffix', 'value'),
    ):
        rng = np.random.default_rng(51071)
        values = []
        for _ in range(10000):
            picks = rng.integers(0, 6, size=6)
            differences = [int(lookup[(task, first)]['desired_answer'])-int(lookup[(task, second)]['desired_answer'])
                           for bundle in picks for task in groups[int(bundle)]]
            if differences:
                values.append(float(np.mean(differences)))
        assert decision[key] == {'resamples': len(values),
                                 'lower_95': float(np.quantile(values, .025)),
                                 'upper_95': float(np.quantile(values, .975))}
    return {'rows': len(rows), 'eligible_confirmation': len(eligible), 'copy_confirmation': len(copies),
            'rates': rates, 'copy_rates': copy_rates, 'suffix_supported': suffix,
            'combined_supported': combined, 'max_control_error': decision['whole_max_logit_error']}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
