"""Independent compact audit of PATH-0008; never reruns the 8B model."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from voynich.workspace.campaign import answer_correct, canonical_digest, digest, write_json
from voynich.workspace.path8_tasks import path8_tasks


ROOT = Path('results/PATH-0008')
RAW = Path('outputs/PATH-0008')
LAYERS = (28, 29, 30, 31)
CONDITIONS = ('upstream', 'selected_one', 'selected_four', 'random_four_norm',
              'mismatch_four_norm', 'all_heads', 'full_window', 'all_cut', 'reverse_four')
LAUNCH_SNAPSHOTS = {
    'docs/experiments/PATH-0008.md': ROOT/'launch-source/PATH-0008.md',
    'src/voynich/workspace/path8_campaign.py': ROOT/'launch-source/path8_campaign.py',
}
ALLOWED_LAUNCH_AMENDMENTS = {
    'docs/experiments/PATH-0008.md': (
        ("The direct lookup and answer rules are exactly PATH-0006's renderer; source/donor",
         "The direct lookup and answer rules are exactly PATH-0006's renderer with greedy generation capped at12 tokens; source/donor"),
        ('and first expected answer tokens must differ.',
         'and expected source/donor answer words must differ.'),
        ('Seed510101. One local run, hard cap',
         "Seed510101. The discovery screen has at most32 upstream-success records ×128 single-head prefills =4,096 screen interventions, followed by at most48 confirmation records ×9 generation conditions plus numerical checks. PATH-0001's 2,048 head interventions took ~17 minutes and PATH-0006's 432 multi-block conditions ~4 minutes, so roughly30–60 minutes is a planning estimate, not a measured PATH-0008 runtime. One local run, hard cap"),
    ),
    'src/voynich/workspace/path8_campaign.py': (
        ("    def finish(status, reason=None, **extra):\n        decision =",
         "    def finish(status, reason=None, **extra):\n"
         "        write_json(result/'qualification.json', {\n"
         "            'observed_control_maxima': controls,\n"
         "            'all_confirmation_controls_executed': 'all_cut_source' in controls,\n"
         "            'tolerance': TOLERANCE,\n"
         "        })\n"
         "        decision ="),
        ("if __name__ == '__main__':\n    run()\n",
         "if __name__ == '__main__':\n"
         "    try:\n"
         "        run()\n"
         "    except Exception as error:\n"
         "        failure = Path('results/PATH-0008/failure.json')\n"
         "        failure.parent.mkdir(parents=True, exist_ok=True)\n"
         "        if not failure.exists():\n"
         "            write_json(failure, {'status': 'incomplete_or_uninformative',\n"
         "                                 'error_type': type(error).__name__, 'message': str(error)})\n"
         "        raise\n"),
    ),
}


def read(path):
    return json.loads(path.read_text())


def committed_blob(revision, path):
    return subprocess.check_output(['git', 'show', f'{revision}:{path}'])


def committed_digest(revision, path):
    """Hash a source blob in the recorded base commit, not today's worktree."""
    return hashlib.sha256(committed_blob(revision, path)).hexdigest()


def verified_launch_source(revision, path, expected_digest):
    """Allow only the two exact pre-launch amendments recorded as snapshots."""
    if path not in LAUNCH_SNAPSHOTS:
        if committed_digest(revision, path) != expected_digest:
            raise AssertionError(f'Base-commit source hash mismatch: {path}')
        return 'base_commit'
    snapshot = LAUNCH_SNAPSHOTS[path].read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != expected_digest:
        raise AssertionError(f'Launch-snapshot hash mismatch: {path}')
    expected_text = committed_blob(revision, path).decode('utf-8')
    for before, after in ALLOWED_LAUNCH_AMENDMENTS[path]:
        if expected_text.count(before) != 1:
            raise AssertionError(f'Base-commit amendment anchor mismatch: {path}')
        expected_text = expected_text.replace(before, after, 1)
    if snapshot != expected_text.encode('utf-8'):
        raise AssertionError(f'Launch snapshot has unregistered changes: {path}')
    return 'hash_matched_launch_amendment'


def frozen_tasks(expected_digest):
    tasks = {split: path8_tasks(split) for split in ('discovery', 'confirmation')}
    if canonical_digest(tasks) != expected_digest:
        raise AssertionError('Current task generator differs from the frozen task manifest')
    return tasks


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
    tasks = frozen_tasks(inputs['tasks_sha256'])
    assert inputs['rendered_inputs_sha256'] == digest(RAW/'rendered-inputs.json')
    assert inputs['model_input_sha256'] == digest('results/JSPACE-0001/inputs.json')
    assert inputs['seed'] == 510101 and inputs['head_layers'] == list(LAYERS)
    assert inputs['cap_seconds'] == 5400 and inputs['memory_bytes_cap'] == 45_000_000_000
    assert set(LAUNCH_SNAPSHOTS).issubset(inputs['source_sha256'])
    source_provenance = {
        path: verified_launch_source(inputs['source_revision'], path, expected)
        for path, expected in inputs['source_sha256'].items()
    }
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
              'source_provenance': {
                  'source_revision': inputs['source_revision'],
                  'source_revision_role': 'base commit before two pre-launch amendments',
                  'checks': source_provenance,
              },
              'limitation': 'Saved intervention logits cannot be recomputed without the model.'}
    write_json(ROOT/'audit.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
