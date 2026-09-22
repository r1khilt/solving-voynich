"""Independent result and provenance audit for PATH-0003."""

import json
from pathlib import Path

import numpy as np

from voynich.workspace.campaign import answer_correct, canonical_digest, digest
from voynich.workspace.path3_campaign import ALL, SEED, UPSTREAM, WINDOWS, changed_positions
from voynich.workspace.path3_tasks import path3_tasks


def load(path):
    return json.loads(Path(path).read_text())


def audit():
    root = Path('results/PATH-0003')
    inputs = load(root/'inputs.json')
    tasks = {split: path3_tasks(split) for split in ('discovery', 'confirmation')}
    assert inputs['source_sha256'] == {path: digest(path) for path in inputs['source_sha256']}
    assert inputs['tasks_sha256'] == canonical_digest(tasks)
    assert inputs['rendered_inputs_sha256'] == digest('outputs/PATH-0003/rendered-inputs.json')
    rendered = load('outputs/PATH-0003/rendered-inputs.json')
    differences = {r['id']: changed_positions(rendered[canonical_digest(r['prompt'])],
                                               rendered[canonical_digest(r['donor_prompt'])])
                   for group in tasks.values() for r in group}
    assert inputs['changed_positions_sha256'] == canonical_digest(differences)
    assert inputs['seed'] == SEED and inputs['upstream_layer'] == UPSTREAM
    assert inputs['windows'] == {k: list(v) for k, v in WINDOWS.items()}
    assert inputs['all_cut'] == list(ALL)

    baselines = {split: load(root/f'{split}-baselines.json') for split in tasks}
    by_baseline = {split: {r['id']: r for r in group} for split, group in baselines.items()}
    for split, group in tasks.items():
        assert {r['id'] for r in baselines[split]} == {r['id'] for r in group}
        for task in group:
            row = by_baseline[split][task['id']]
            assert row['source_correct'] == answer_correct(row['source_text'], [task['answer']])
            assert row['donor_correct'] == answer_correct(row['donor_text'], [task['donor_answer']])
            for kind, prompt in (('source', task['prompt']), ('donor', task['donor_prompt'])):
                file = Path('outputs/PATH-0003/baselines')/(canonical_digest(prompt)+'.npz')
                assert row[kind+'_arrays_sha256'] == digest(file)
                arrays = np.load(file)
                assert len(arrays['field']) == len(rendered[canonical_digest(prompt)])
                assert np.isfinite(arrays['logits']).all()
        eligible = sum(r['family'] == 'composed' and r['source_correct'] and r['donor_correct']
                       and r['first_token_differs'] for r in baselines[split])
        copies = sum(r['family'] == 'copy' and r['source_correct'] for r in baselines[split])
        assert load(root/f'{split}-capability.json') == {'eligible': eligible, 'source_copy': copies}

    decision = load(root/'decision.json')
    if decision['status'] == 'inconclusive':
        assert decision['reason'] == 'confirmation_baseline_competence'
        assert load(root/'confirmation-capability.json')['source_copy'] < 6
        rows = load(root/'discovery-rows.json')
        assert len(rows) == 24*7
        assert {(r['task_id'], r['condition']) for r in rows} == {
            (task['id'], name) for task in tasks['discovery']
            for name in ('upstream', *WINDOWS, 'all_cut')}
        assert rows == [json.loads(line) for line in Path('outputs/PATH-0003/observations.jsonl').read_text().splitlines()]
        for row in rows:
            task = next(t for t in tasks['discovery'] if t['id'] == row['task_id'])
            baseline = by_baseline['discovery'][task['id']]
            assert row['eligible'] == (task['family'] == 'composed' and baseline['source_correct']
                                       and baseline['donor_correct'] and baseline['first_token_differs'])
            assert row['desired_answer'] == (answer_correct(row['text'], [task['donor_answer']])
                                             if task['family'] == 'composed' else False)
            assert row['copy_preserved'] == (answer_correct(row['text'], [task['answer']])
                                            if task['family'] == 'copy' else False)
            if row['condition'] == 'all_cut':
                source_file = Path('outputs/PATH-0003/baselines')/(canonical_digest(task['prompt'])+'.npz')
                source_first = int(np.load(source_file)['logits'].argmax())
                assert row['tokens'] and row['tokens'][0] == source_first
        selection = load(root/'selection.json')
        discovery_eligible = {r['id'] for r in baselines['discovery'] if r['family'] == 'composed'
                              and r['source_correct'] and r['donor_correct'] and r['first_token_differs']}
        upstream = {r['task_id']: r for r in rows if r['condition'] == 'upstream'}
        success = {key for key in discovery_eligible if upstream[key]['desired_answer']}
        ranking = [{'window': name, 'upstream_success': len(success),
                    'donor_removed': sum(not r['donor_first_token'] for r in rows
                                         if r['condition'] == name and r['task_id'] in success)}
                   for name in WINDOWS]
        assert ranking == selection['ranking']
        assert selection['selected_window'] == sorted(
            ranking, key=lambda r: (-r['donor_removed'], list(WINDOWS).index(r['window'])))[0]['window']
        return {'status': 'inconclusive', 'reason': decision['reason'], 'discovery_rows': len(rows),
                'discovery_eligible': len(discovery_eligible), 'discovery_upstream_success': len(success),
                'selected_window_unconfirmed': selection['selected_window'],
                'confirmation_copy': load(root/'confirmation-capability.json')['source_copy']}

    rows = load(root/'rows.json')
    assert rows == load(root/'discovery-rows.json')+load(root/'confirmation-rows.json')
    assert rows == [json.loads(line) for line in Path('outputs/PATH-0003/observations.jsonl').read_text().splitlines()]
    selection = load(root/'selection.json')
    selected = selection['selected_window']
    expected = {(task['id'], condition) for group in tasks.values() for task in group
                for condition in ('upstream', *WINDOWS, 'all_cut')}
    expected.update((task['id'], 'random_selected') for task in tasks['confirmation'])
    assert len(rows) == len(expected) == len({(r['task_id'], r['condition']) for r in rows})
    assert expected == {(r['task_id'], r['condition']) for r in rows}
    task_by_id = {r['id']: r for group in tasks.values() for r in group}
    for row in rows:
        task = task_by_id[row['task_id']]
        baseline = by_baseline[task['split']][task['id']]
        assert row['split'] == task['split'] and row['bundle'] == task['bundle']
        assert row['template'] == task['template'] and row['family'] == task['family']
        assert row['eligible'] == (task['family'] == 'composed' and baseline['source_correct']
                                   and baseline['donor_correct'] and baseline['first_token_differs'])
        assert row['baseline_copy'] == (task['family'] == 'copy' and baseline['source_correct'])
        assert row['desired_answer'] == (answer_correct(row['text'], [task['donor_answer']])
                                         if task['family'] == 'composed' else False)
        assert row['copy_preserved'] == (answer_correct(row['text'], [task['answer']])
                                        if task['family'] == 'copy' else False)
    discovery = load(root/'discovery-rows.json')
    discovery_eligible = {r['id'] for r in baselines['discovery'] if r['family'] == 'composed'
                          and r['source_correct'] and r['donor_correct'] and r['first_token_differs']}
    upstream = {r['task_id']: r for r in discovery if r['condition'] == 'upstream'}
    success = {key for key in discovery_eligible if upstream[key]['desired_answer']}
    ranking = [{'window': name, 'upstream_success': len(success),
                'donor_removed': sum(not r['donor_first_token'] for r in discovery
                                     if r['condition'] == name and r['task_id'] in success)}
               for name in WINDOWS]
    assert ranking == selection['ranking']
    assert selected == sorted(ranking, key=lambda r: (-r['donor_removed'], list(WINDOWS).index(r['window'])))[0]['window']

    confirmation = load(root/'confirmation-rows.json')
    eligible = {r['id'] for r in baselines['confirmation'] if r['family'] == 'composed'
                and r['source_correct'] and r['donor_correct'] and r['first_token_differs']}
    copies = {r['id'] for r in baselines['confirmation'] if r['family'] == 'copy' and r['source_correct']}
    by_condition = {name: {r['task_id']: r for r in confirmation if r['condition'] == name}
                    for name in ('upstream', *WINDOWS, 'all_cut', 'random_selected')}
    success = {key for key in eligible if by_condition['upstream'][key]['desired_answer']}
    reductions = {name: sum(not by_condition[name][key]['donor_first_token'] for key in success)
                  for name in by_condition}
    copy_preserve = {name: sum(by_condition[name][key]['copy_preserved'] for key in copies)
                     for name in by_condition}
    assert load(root/'summary.json') == {'eligible': len(eligible), 'upstream_success': len(success),
                                         'baseline_copy': len(copies), 'reductions': reductions,
                                         'copy_preserve': copy_preserve, 'selected_window': selected}
    gate = len(success)/len(eligible) >= .5 and copy_preserve['upstream']/len(copies) >= .9
    selected_rate = reductions[selected]/len(success) if success else 0.0
    random_rate = reductions['random_selected']/len(success) if success else 0.0
    supported = gate and selected_rate >= .5 and selected_rate-random_rate >= .2
    supported &= copy_preserve[selected]/len(copies) >= .9
    status = ('exploratory_supported' if supported else
              ('uninformative_upstream' if not gate else 'exploratory_failed'))
    assert decision['status'] == status and decision['supported'] == supported
    assert decision['selected_window'] == selected and decision['upstream_gate'] == gate
    assert decision['selected_reduction_rate'] == selected_rate
    assert decision['random_reduction_rate'] == random_rate
    assert decision['all_cut_max_logit_error'] < .002
    assert decision['full_donor_max_logit_error'] < .002
    assert decision['random_max_norm_error'] < 1e-3
    return {'status': status, 'rows': len(rows), 'selected_window': selected,
            'eligible_confirmation': len(eligible), 'upstream_success': len(success),
            'selected_reduction': reductions[selected], 'random_reduction': reductions['random_selected']}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
