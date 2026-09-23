"""Registered, bounded original-model attention mediation on fresh bindings."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_campaign import condition_seed
from .geometry import matched_random_delta
from .path3_backend import Path3Workspace
from .path6_tasks import path6_tasks

UPSTREAM = 23
WINDOWS = {'24-27': tuple(range(24, 28)), '28-31': tuple(range(28, 32)),
           '32-35': tuple(range(32, 36))}
ALL_CUT = tuple(range(24, 36))
SEED = 51081
CAP_SECONDS = 3000


def changed_positions(source, donor):
    if len(source) != len(donor) or len(source) < 12:
        raise ValueError('Source/donor rendered lengths differ')
    changed = np.flatnonzero(np.asarray(source) != np.asarray(donor)).tolist()
    if len(changed) < 2 or changed[-1] >= len(source)-8:
        raise ValueError('Value edits overlap the unchanged query suffix')
    return changed


def select_window(rows, eligible):
    upstream = {r['task_id']: r for r in rows if r['condition'] == 'upstream'}
    success = {key for key in eligible if upstream[key]['desired_answer']}
    ranking = [{
        'window': name, 'upstream_success': len(success),
        'donor_removed': sum(not r['donor_first_token'] for r in rows
                             if r['condition'] == name and r['task_id'] in success),
    } for name in WINDOWS]
    selected = sorted(ranking, key=lambda r: (-r['donor_removed'], list(WINDOWS).index(r['window'])))[0]['window']
    return selected, ranking


def bootstrap_difference(rows, success, selected):
    by_condition = {(r['task_id'], r['condition']): r for r in rows}
    bundles = sorted({int(key.split('/')[2]) for key in success})
    if not bundles:
        return {'resamples': 0, 'lower_95': None, 'upper_95': None}
    members = {bundle: [key for key in success if int(key.split('/')[2]) == bundle] for bundle in bundles}
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(10000):
        picked = rng.choice(bundles, size=len(bundles), replace=True)
        differences = [int(not by_condition[key, selected]['donor_first_token'])
                       - int(not by_condition[key, 'random_selected']['donor_first_token'])
                       for bundle in picked for key in members[bundle]]
        samples.append(float(np.mean(differences)))
    return {'resamples': len(samples), 'lower_95': float(np.quantile(samples, .025)),
            'upper_95': float(np.quantile(samples, .975))}


def run():
    output, result = Path('outputs/PATH-0006'), Path('results/PATH-0006')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result/'inputs.json').exists() or (output/'observations.jsonl').exists():
        raise FileExistsError('No automatic repeat of PATH-0006')
    started = time.monotonic()
    config = load_config('configs/jspace0001.json')
    model = Path3Workspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(min(config['memory_bytes_cap'], 45_000_000_000))

    def budget(count):
        elapsed = time.monotonic()-started
        if elapsed >= CAP_SECONDS or model.mx.get_peak_memory() > 45_000_000_000:
            write_json(result/'incomplete.json', {'elapsed_seconds': elapsed, 'completed_rows': count,
                                                   'reason': 'resource_cap'})
            raise RuntimeError('PATH-0006 resource cap')

    tasks = {split: path6_tasks(split) for split in ('discovery', 'confirmation')}
    prompts = {r[key] for group in tasks.values() for r in group for key in ('prompt', 'donor_prompt')}
    ids = {prompt: model.encode(prompt, chat=True) for prompt in prompts}
    changed = {r['id']: changed_positions(ids[r['prompt']], ids[r['donor_prompt']])
               for group in tasks.values() for r in group}
    write_json(output/'rendered-inputs.json', {canonical_digest(prompt): token_ids for prompt, token_ids in ids.items()})
    sources = [Path('src/voynich/workspace')/name for name in
               ('path6_tasks.py', 'path6_campaign.py', 'path3_backend.py', 'route_backend.py')]
    sources.append(Path('docs/experiments/PATH-0006.md'))
    write_json(result/'inputs.json', {
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(path): digest(path) for path in sources},
        'tasks_sha256': canonical_digest(tasks), 'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
        'changed_positions_sha256': canonical_digest(changed), 'model_input_sha256': digest('results/JSPACE-0001/inputs.json'),
        'seed': SEED, 'upstream_layer': UPSTREAM, 'windows': WINDOWS, 'all_cut': ALL_CUT,
        'cap_seconds': CAP_SECONDS, 'memory_bytes_cap': 45_000_000_000,
    })

    baselines = {}
    baseline_dir = output/'baselines'
    baseline_dir.mkdir(exist_ok=True)

    def baseline(prompt):
        if prompt not in baselines:
            budget(0)
            phrase, tokens, fields, logits = model.generate_field(ids[prompt], capture_layers=(UPSTREAM,))
            clean_logits, writes = model.clean_final_writes(ids[prompt], start=24)
            error = float(np.max(np.abs(clean_logits-logits)))
            if error >= .002 or int(clean_logits.argmax()) != int(logits.argmax()):
                raise ValueError('Clean attention capture disagrees with baseline')
            local = baseline_dir/(canonical_digest(prompt)+'.npz')
            np.savez_compressed(local, logits=logits, field=fields[UPSTREAM],
                                **{f'write_{layer}': value for layer, value in writes.items()})
            baselines[prompt] = {'text': phrase, 'tokens': tokens, 'logits': logits,
                                 'field': fields[UPSTREAM], 'writes': writes, 'arrays_sha256': digest(local)}
        return baselines[prompt]

    all_rows = []
    public_baselines = {}
    selected = None
    max_donor_error = max_cut_error = max_random_norm_error = 0.0
    with (output/'observations.jsonl').open('w') as log:
        for split in ('discovery', 'confirmation'):
            public = []
            for task in tasks[split]:
                source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
                public.append({
                    'id': task['id'], 'split': split, 'bundle': task['bundle'], 'template': task['template'],
                    'family': task['family'], 'source_text': source['text'], 'donor_text': donor['text'],
                    'source_correct': answer_correct(source['text'], [task['answer']]),
                    'donor_correct': answer_correct(donor['text'], [task['donor_answer']]),
                    'first_token_differs': bool(source['tokens']) and bool(donor['tokens'])
                    and source['tokens'][0] != donor['tokens'][0],
                    'source_arrays_sha256': source['arrays_sha256'], 'donor_arrays_sha256': donor['arrays_sha256'],
                })
            public_baselines[split] = public
            write_json(result/f'{split}-baselines.json', public)
            eligible = {r['id'] for r in public if r['family'] == 'binding' and r['source_correct']
                        and r['donor_correct'] and r['first_token_differs']}
            copy_ids = {r['id'] for r in public if r['family'] == 'copy' and r['source_correct']}
            write_json(result/f'{split}-capability.json', {'eligible': len(eligible), 'source_copy': len(copy_ids)})
            if len(eligible) < 24 or len(copy_ids) < 14:
                write_json(result/'decision.json', {
                    'status': 'inconclusive', 'reason': f'{split}_baseline_competence',
                    'eligible': len(eligible), 'source_copy': len(copy_ids),
                    'elapsed_seconds': time.monotonic()-started, 'peak_mlx_bytes': model.mx.get_peak_memory(),
                })
                return

            split_rows = []
            for task in tasks[split]:
                budget(len(all_rows))
                source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
                delta = donor['field']-source['field']
                edit = np.zeros_like(delta)
                edit[changed[task['id']]] = delta[changed[task['id']]]
                whole_logits, _, _ = model.prefill_field(ids[task['prompt']], fields={UPSTREAM: delta})
                error = float(np.max(np.abs(whole_logits-donor['logits'])))
                max_donor_error = max(max_donor_error, error)
                if error >= .002 or int(whole_logits.argmax()) != int(donor['logits'].argmax()):
                    raise ValueError('Whole donor transport control failed')
                conditions = {'upstream': (), **(WINDOWS if split == 'discovery' else {selected: WINDOWS[selected]}),
                              'all_cut': ALL_CUT}
                upstream_writes = None
                for condition, cut_layers in conditions.items():
                    budget(len(all_rows))
                    phrase, tokens, logits, writes = model.generate_cut(
                        ids[task['prompt']], upstream_layer=UPSTREAM, upstream_field=edit,
                        clean_writes=source['writes'], cut_layers=cut_layers,
                        capture_writes=condition == 'upstream',
                    )
                    if condition == 'upstream':
                        upstream_writes = writes
                    if condition == 'all_cut':
                        error = float(np.max(np.abs(logits-source['logits'])))
                        max_cut_error = max(max_cut_error, error)
                        if error >= .002 or int(logits.argmax()) != int(source['logits'].argmax()):
                            raise ValueError('All-cut source restoration control failed')
                    row = _row(task, condition, phrase, tokens, logits, source, donor, eligible, copy_ids)
                    split_rows.append(row)
                    all_rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    log.flush()
                if split == 'confirmation':
                    random_deltas = {}
                    for layer in WINDOWS[selected]:
                        actual = source['writes'][layer]-upstream_writes[layer]
                        random_deltas[layer] = matched_random_delta(
                            actual, seed=condition_seed(SEED, task['id'], layer, 'random-cut'))
                        max_random_norm_error = max(max_random_norm_error,
                                                    abs(float(np.linalg.norm(random_deltas[layer]))
                                                        - float(np.linalg.norm(actual))))
                    phrase, tokens, logits, _ = model.generate_cut(
                        ids[task['prompt']], upstream_layer=UPSTREAM, upstream_field=edit,
                        clean_writes=source['writes'], cut_layers=WINDOWS[selected],
                        random_deltas=random_deltas,
                    )
                    row = _row(task, 'random_selected', phrase, tokens, logits,
                               source, donor, eligible, copy_ids)
                    split_rows.append(row)
                    all_rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    log.flush()
            write_json(result/f'{split}-rows.json', split_rows)
            if split == 'discovery':
                selected, ranking = select_window(split_rows, eligible)
                write_json(result/'selection.json', {'selected_window': selected, 'ranking': ranking, 'seed': SEED})

    write_json(result/'rows.json', all_rows)
    confirmation = [r for r in all_rows if r['split'] == 'confirmation']
    eligible = {r['id'] for r in public_baselines['confirmation'] if r['family'] == 'binding'
                and r['source_correct'] and r['donor_correct'] and r['first_token_differs']}
    copy_ids = {r['id'] for r in public_baselines['confirmation'] if r['family'] == 'copy' and r['source_correct']}
    by_condition = {condition: {r['task_id']: r for r in confirmation if r['condition'] == condition}
                    for condition in ('upstream', selected, 'all_cut', 'random_selected')}
    success = {key for key in eligible if by_condition['upstream'][key]['desired_answer']}
    reductions = {condition: sum(not by_condition[condition][key]['donor_first_token'] for key in success)
                  for condition in by_condition}
    copy_preserve = {condition: sum(by_condition[condition][key]['copy_preserved'] for key in copy_ids)
                     for condition in by_condition}
    upstream_rate = len(success)/len(eligible)
    selected_rate = reductions[selected]/len(success) if success else 0.0
    random_rate = reductions['random_selected']/len(success) if success else 0.0
    informative = upstream_rate >= .75 and copy_preserve['upstream']/len(copy_ids) >= .9
    supported = (informative and selected_rate >= .35 and selected_rate-random_rate >= .25
                 and copy_preserve[selected]/len(copy_ids) >= .9)
    write_json(result/'summary.json', {
        'eligible': len(eligible), 'upstream_success': len(success), 'source_copy': len(copy_ids),
        'selected_window': selected, 'reductions': reductions, 'copy_preserve': copy_preserve,
        'selected_minus_random_bundle_bootstrap': bootstrap_difference(confirmation, success, selected),
    })
    write_json(result/'decision.json', {
        'status': 'exploratory_supported' if supported else ('uninformative_upstream' if not informative else 'exploratory_failed'),
        'supported': supported, 'upstream_gate': informative, 'selected_window': selected,
        'upstream_success_rate': upstream_rate, 'selected_reduction_rate': selected_rate,
        'random_reduction_rate': random_rate, 'all_cut_max_logit_error': max_cut_error,
        'whole_donor_max_logit_error': max_donor_error, 'random_max_norm_error': max_random_norm_error,
        'elapsed_seconds': time.monotonic()-started, 'peak_mlx_bytes': model.mx.get_peak_memory(),
        'scope': 'Supplied English lookup only; no historical decipherment',
    })


def _row(task, condition, phrase, tokens, logits, source, donor, eligible, copy_ids):
    return {
        'task_id': task['id'], 'split': task['split'], 'bundle': task['bundle'],
        'template': task['template'], 'family': task['family'], 'condition': condition,
        'text': phrase, 'tokens': tokens, 'eligible': task['id'] in eligible,
        'baseline_copy': task['id'] in copy_ids,
        'desired_answer': answer_correct(phrase, [task['donor_answer']]) if task['family'] == 'binding' else False,
        'donor_first_token': bool(tokens) and bool(donor['tokens']) and tokens[0] == donor['tokens'][0],
        'copy_preserved': answer_correct(phrase, [task['answer']]) if task['family'] == 'copy' else False,
        'donor_margin': (float(logits[donor['tokens'][0]]-logits[source['tokens'][0]])
                         if source['tokens'] and donor['tokens'] and source['tokens'][0] != donor['tokens'][0] else None),
    }


if __name__ == '__main__':
    run()
