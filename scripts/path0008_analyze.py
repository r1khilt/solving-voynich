"""Independent compact audit of PATH-0008; never reruns the 8B model."""

import json
from pathlib import Path

import numpy as np

from voynich.workspace.campaign import answer_correct, canonical_digest, digest, write_json
from voynich.workspace.path8_tasks import path8_tasks


ROOT = Path('results/PATH-0008')
RAW = Path('outputs/PATH-0008')
LAYERS = (28, 29, 30, 31)
CONDITIONS = ('upstream', 'selected_one', 'selected_four', 'random_four_norm',
              'mismatch_four_norm', 'all_heads', 'full_window', 'all_cut', 'reverse_four')


def read(path):
    return json.loads(path.read_text())


def _selection(screen, success):
    lookup = {(r['task_id'], r['layer'], r['head']): r for r in screen}
    ranking = []
    for layer in LAYERS:
        heads = []
        for head in range(32):
            values = [lookup[key, layer, head]['margin_drop'] for key in sorted(success)]
            heads.append({'head': head, 'mean_margin_drop': float(np.mean(values)),
                          'min_margin_drop': float(min(values)),
                          'max_margin_drop': float(max(values))})
        heads.sort(key=lambda row: (-row['mean_margin_drop'], row['head']))
        ranking.append({'layer': layer, 'top_four': [r['head'] for r in heads[:4]],
                        'top_four_sum': float(sum(r['mean_margin_drop'] for r in heads[:4])),
                        'heads': heads})
    selected = sorted(ranking, key=lambda row: (-row['top_four_sum'], row['layer']))[0]
    return selected, ranking


def _bootstrap(rows, success, control):
    lookup = {(r['task_id'], r['condition']): r for r in rows}
    bundles = sorted({int(key.split('/')[2]) for key in success})
    if not bundles:
        return {'resamples': 0, 'lower_95': None, 'upper_95': None}
    members = {bundle: sorted(key for key in success if int(key.split('/')[2]) == bundle)
               for bundle in bundles}
    rng = np.random.default_rng(510101)
    values = []
    for _ in range(10_000):
        picks = rng.choice(bundles, size=len(bundles), replace=True)
        effects = [int(not lookup[key, 'selected_four']['donor_first_token'])
                   - int(not lookup[key, control]['donor_first_token'])
                   for bundle in picks for key in members[bundle]]
        values.append(float(np.mean(effects)))
    return {'resamples': len(values), 'lower_95': float(np.quantile(values, .025)),
            'upper_95': float(np.quantile(values, .975))}


def audit():
    inputs = read(ROOT/'inputs.json')
    tasks = {split: path8_tasks(split) for split in ('discovery', 'confirmation')}
    assert inputs['tasks_sha256'] == canonical_digest(tasks)
    assert inputs['rendered_inputs_sha256'] == digest(RAW/'rendered-inputs.json')
    assert inputs['model_input_sha256'] == digest('results/JSPACE-0001/inputs.json')
    assert inputs['seed'] == 510101 and inputs['head_layers'] == list(LAYERS)
    assert inputs['cap_seconds'] == 5400 and inputs['memory_bytes_cap'] == 45_000_000_000
    for path, expected in inputs['source_sha256'].items():
        assert digest(path) == expected, path
    rendered = read(RAW/'rendered-inputs.json')
    positions = {}
    for split in tasks:
        for task in tasks[split]:
            source = rendered[canonical_digest(task['prompt'])]
            donor = rendered[canonical_digest(task['donor_prompt'])]
            assert len(source) == len(donor)
            changed = np.flatnonzero(np.asarray(source) != np.asarray(donor)).tolist()
            assert len(changed) >= 2 and changed[-1] < len(source)-8
            positions[task['id']] = changed
    assert inputs['changed_positions_sha256'] == canonical_digest(positions)

    decision = read(ROOT/'decision.json')
    capabilities = {}
    eligible_by_split, copy_by_split = {}, {}
    for split in tasks:
        path = ROOT/f'{split}-baselines.json'
        if not path.exists():
            break
        observed = read(path)
        assert [r['id'] for r in observed] == [r['id'] for r in tasks[split]]
        task_lookup = {r['id']: r for r in tasks[split]}
        eligible, copies = set(), set()
        for baseline in observed:
            task = task_lookup[baseline['id']]
            assert baseline['family'] == task['family']
            assert baseline['source_correct'] == answer_correct(
                baseline['source_text'], [task['answer']])
            assert baseline['donor_correct'] == answer_correct(
                baseline['donor_text'], [task['donor_answer']])
            for side, prompt in (('source', task['prompt']), ('donor', task['donor_prompt'])):
                artifact = RAW/'baselines'/(canonical_digest(prompt)+'.npz')
                assert digest(artifact) == baseline[f'{side}_arrays_sha256']
                with np.load(artifact) as arrays:
                    assert np.isfinite(arrays['logits']).all() and np.isfinite(arrays['field']).all()
                    assert all(f'write_{layer}' in arrays for layer in range(24, 36))
                    assert all(f'heads_{layer}' in arrays for layer in LAYERS)
            if (task['family'] == 'binding' and baseline['source_correct']
                    and baseline['donor_correct'] and baseline['first_token_differs']):
                eligible.add(task['id'])
            if task['family'] == 'copy' and baseline['source_correct']:
                copies.add(task['id'])
        assert read(ROOT/f'{split}-capability.json') == {
            'eligible': len(eligible), 'source_copy': len(copies)}
        eligible_by_split[split], copy_by_split[split] = eligible, copies
        capabilities[split] = {'eligible': len(eligible), 'source_copy': len(copies)}
        if len(eligible) < 24 or len(copies) < 14:
            assert decision['status'] == 'inconclusive'
            assert decision['reason'] == f'{split}_baseline_competence'
            break

    screen_path = ROOT/'discovery-screen.json'
    selection_path = ROOT/'selection.json'
    upstream_path = ROOT/'discovery-upstream.json'
    if upstream_path.exists():
        upstream = read(upstream_path)
        observed = [r['task_id'] for r in upstream]
        assert len(observed) == len(set(observed))
        assert set(observed) == eligible_by_split['discovery']
        lookup = {r['id']: r for r in tasks['discovery']}
        for row in upstream:
            assert row['desired_answer'] == answer_correct(
                row['text'], [lookup[row['task_id']]['donor_answer']])
        upstream_success = {r['task_id'] for r in upstream if r['desired_answer']}
    else:
        upstream_success = set()
    if screen_path.exists() and selection_path.exists():
        screen = read(screen_path)
        assert [json.loads(line) for line in (RAW/'scan.jsonl').read_text().splitlines()] == screen
        success = {r['task_id'] for r in screen}
        assert success == upstream_success
        expected = {(key, layer, head) for key in success for layer in LAYERS for head in range(32)}
        observed = [(r['task_id'], r['layer'], r['head']) for r in screen]
        assert len(observed) == len(expected) and set(observed) == expected
        assert success.issubset(eligible_by_split['discovery'])
        assert all(np.isfinite(r['margin_drop']) and np.isfinite(r['delta_norm']) for r in screen)
        selected, ranking = _selection(screen, success)
        selection = read(selection_path)
        assert selection['layer'] == selected['layer']
        assert selection['heads'] == selected['top_four']
        assert selection['ranking'] == ranking
        assert selection['upstream_success'] == len(success)
    else:
        assert decision['status'] in ('inconclusive', 'uninformative_upstream')
        if decision['status'] == 'uninformative_upstream':
            assert decision['upstream_success'] == len(upstream_success) < 18

    rows_path = ROOT/'confirmation-rows.json'
    if rows_path.exists():
        rows = read(rows_path)
        assert 'confirmation' in eligible_by_split
        expected = {(task['id'], condition) for task in tasks['confirmation']
                    for condition in CONDITIONS}
        actual = [(r['task_id'], r['condition']) for r in rows]
        assert len(actual) == len(expected) and set(actual) == expected
        assert [json.loads(line) for line in (RAW/'observations.jsonl').read_text().splitlines()] == rows
        task_lookup = {r['id']: r for r in tasks['confirmation']}
        donor_first = {}
        for task in tasks['confirmation']:
            artifact = RAW/'baselines'/(canonical_digest(task['donor_prompt'])+'.npz')
            with np.load(artifact) as arrays:
                donor_first[task['id']] = int(arrays['logits'].argmax())
        for row in rows:
            task = task_lookup[row['task_id']]
            key = task['id']
            assert row['family'] == task['family'] and row['bundle'] == task['bundle']
            assert row['eligible'] == (key in eligible_by_split['confirmation'])
            assert row['baseline_copy'] == (key in copy_by_split['confirmation'])
            assert row['desired_answer'] == (answer_correct(row['text'], [task['donor_answer']])
                                             if task['family'] == 'binding' else False)
            assert row['source_answer'] == answer_correct(row['text'], [task['answer']])
            assert row['copy_preserved'] == (answer_correct(row['text'], [task['answer']])
                                            if task['family'] == 'copy' else False)
            assert row['donor_first_token'] == (bool(row['tokens'])
                                                and row['tokens'][0] == donor_first[key])
            assert row['donor_margin'] is None or np.isfinite(row['donor_margin'])
        lookup = {(r['task_id'], r['condition']): r for r in rows}
        eligible = eligible_by_split['confirmation']
        copies = copy_by_split['confirmation']
        success = {key for key in eligible if lookup[key, 'upstream']['desired_answer']}
        reductions = {condition: sum(not lookup[key, condition]['donor_first_token']
                                     for key in success) for condition in CONDITIONS}
        preserved = {condition: sum(lookup[key, condition]['copy_preserved']
                                    for key in copies) for condition in CONDITIONS}
        full_source = {condition: sum(lookup[key, condition]['source_answer']
                                      for key in success) for condition in CONDITIONS}
        summary = read(ROOT/'summary.json')
        assert summary['eligible'] == len(eligible)
        assert summary['upstream_success'] == len(success)
        assert summary['source_copy'] == len(copies)
        assert summary['reductions'] == reductions
        assert summary['copy_preserved'] == preserved
        assert summary['full_source_answers'] == full_source
        assert summary['selected_layer'] == selection['layer']
        assert summary['selected_heads'] == selection['heads']
        assert summary['selected_minus_random_bundle_bootstrap'] == _bootstrap(
            rows, success, 'random_four_norm')
        assert summary['selected_minus_mismatch_bundle_bootstrap'] == _bootstrap(
            rows, success, 'mismatch_four_norm')
        n = len(success)
        upstream_rate = n/len(eligible)
        selected_rate = reductions['selected_four']/n if n else 0.0
        random_rate = reductions['random_four_norm']/n if n else 0.0
        mismatch_rate = reductions['mismatch_four_norm']/n if n else 0.0
        window_rate = reductions['full_window']/n if n else 0.0
        upstream_gate = upstream_rate >= .75 and preserved['upstream']/len(copies) >= .9
        window_gate = window_rate >= .35
        supported = (upstream_gate and window_gate and selected_rate >= .35
                     and selected_rate-random_rate >= .25
                     and selected_rate-mismatch_rate >= .25
                     and preserved['selected_four']/len(copies) >= .9)
        assert decision['supported'] == supported
        assert decision['upstream_gate'] == upstream_gate
        assert decision['full_window_gate'] == window_gate
        expected_status = ('exploratory_supported' if supported else
                           'uninformative_upstream' if not upstream_gate else
                           'uninformative_positive_control' if not window_gate else
                           'exploratory_failed')
        assert decision['status'] == expected_status
        assert all(value < .002 for value in decision['controls'].values())
    report = {'passed': True, 'status': decision['status'], 'capability': capabilities,
              'confirmation_rows': len(read(rows_path)) if rows_path.exists() else 0,
              'limitation': 'Saved intervention logits cannot be recomputed without the model.'}
    write_json(ROOT/'audit.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
