"""Registered composed-lookup attention-window mediation campaign."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_campaign import condition_seed
from .geometry import matched_random_delta
from .path3_backend import Path3Workspace
from .path3_tasks import path3_tasks

UPSTREAM = 15
WINDOWS = {
    '16-19': tuple(range(16, 20)), '20-23': tuple(range(20, 24)),
    '24-27': tuple(range(24, 28)), '28-31': tuple(range(28, 32)),
    '32-35': tuple(range(32, 36)),
}
ALL = tuple(range(16, 36))
CAP_SECONDS = 3000
SEED = 51061


def changed_positions(source, donor):
    if len(source) != len(donor) or len(source) < 30:
        raise ValueError('Source/donor rendered lengths differ')
    changed = np.flatnonzero(np.asarray(source) != np.asarray(donor)).tolist()
    if len(changed) < 2 or changed[-1] >= len(source)-20:
        raise ValueError('Key changes overlap the common suffix')
    return changed


def choose_window(rows, eligible_ids):
    upstream = {r['task_id']: r for r in rows if r['condition'] == 'upstream'}
    success = {key for key in eligible_ids if upstream[key]['desired_answer']}
    ranking = []
    for name in WINDOWS:
        members = [r for r in rows if r['condition'] == name and r['task_id'] in success]
        reduction = sum(not r['donor_first_token'] for r in members)
        ranking.append({'window': name, 'upstream_success': len(success), 'donor_removed': reduction})
    selected = sorted(ranking, key=lambda r: (-r['donor_removed'], list(WINDOWS).index(r['window'])))[0]['window']
    return selected, ranking


def run():
    output, result = Path('outputs/PATH-0003'), Path('results/PATH-0003')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result/'inputs.json').exists() or (output/'observations.jsonl').exists():
        raise FileExistsError('No automatic repeat of PATH-0003')
    started = time.monotonic()
    config = load_config('configs/jspace0001.json')
    model = Path3Workspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(config['memory_bytes_cap'])

    def budget(count):
        elapsed = time.monotonic()-started
        if elapsed >= CAP_SECONDS or model.mx.get_peak_memory() > config['memory_bytes_cap']:
            write_json(result/'incomplete.json', {'elapsed_seconds': elapsed, 'completed_rows': count,
                                                   'reason': 'resource_cap'})
            raise RuntimeError('PATH-0003 resource cap')

    tasks = {split: path3_tasks(split) for split in ('discovery', 'confirmation')}
    prompts = {r['prompt'] for group in tasks.values() for r in group}
    prompts.update(r['donor_prompt'] for group in tasks.values() for r in group)
    ids = {p: model.encode(p, chat=True) for p in prompts}
    differences = {r['id']: changed_positions(ids[r['prompt']], ids[r['donor_prompt']])
                   for group in tasks.values() for r in group}
    write_json(output/'rendered-inputs.json', {canonical_digest(p): token_ids for p, token_ids in ids.items()})
    sources = [Path('src/voynich/workspace')/name for name in
               ('path3_tasks.py', 'path3_backend.py', 'path3_campaign.py', 'route_backend.py')]
    sources += [Path('docs/experiments/PATH-0003.md'),
                Path('docs/research/latent-mechanisms-2026-09-21/SIMILARITY_THEORY.md')]
    write_json(result/'inputs.json', {
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): digest(p) for p in sources},
        'tasks_sha256': canonical_digest(tasks), 'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
        'changed_positions_sha256': canonical_digest(differences), 'model_input_sha256': digest('results/JSPACE-0001/inputs.json'),
        'seed': SEED, 'upstream_layer': UPSTREAM, 'windows': WINDOWS, 'all_cut': ALL,
        'cap_seconds': CAP_SECONDS, 'memory_bytes_cap': config['memory_bytes_cap'],
    })

    baselines = {}
    baseline_dir = output/'baselines'
    baseline_dir.mkdir(exist_ok=True)

    def baseline(prompt):
        if prompt not in baselines:
            budget(0)
            phrase, tokens, fields, logits = model.generate_field(ids[prompt], capture_layers=(UPSTREAM,))
            clean_logits, writes = model.clean_final_writes(ids[prompt], 16)
            error = float(np.max(np.abs(clean_logits-logits)))
            if error >= .002 or int(clean_logits.argmax()) != int(logits.argmax()):
                raise ValueError('Clean attention capture disagrees with cached baseline')
            local = baseline_dir/(canonical_digest(prompt)+'.npz')
            np.savez_compressed(local, logits=logits, field=fields[UPSTREAM],
                                **{f'write_{k}': v for k, v in writes.items()})
            baselines[prompt] = {'text': phrase, 'tokens': tokens, 'logits': logits,
                                 'field': fields[UPSTREAM], 'writes': writes, 'arrays_sha256': digest(local)}
        return baselines[prompt]

    all_rows = []
    baselines_public = {}
    selected = None
    max_full_error = 0.0
    max_donor_error = 0.0
    max_random_norm_error = 0.0
    with (output/'observations.jsonl').open('w') as log:
        for split in ('discovery', 'confirmation'):
            if split == 'confirmation' and selected is None:
                raise AssertionError('Confirmation opened before discovery selection')
            group = tasks[split]
            public = []
            for task in group:
                source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
                public.append({
                    'id': task['id'], 'split': split, 'bundle': task['bundle'], 'family': task['family'],
                    'source_text': source['text'], 'donor_text': donor['text'],
                    'source_correct': answer_correct(source['text'], [task['answer']]),
                    'donor_correct': answer_correct(donor['text'], [task['donor_answer']]),
                    'first_token_differs': bool(source['tokens']) and bool(donor['tokens'])
                    and source['tokens'][0] != donor['tokens'][0],
                    'source_arrays_sha256': source['arrays_sha256'], 'donor_arrays_sha256': donor['arrays_sha256'],
                })
            baselines_public[split] = public
            write_json(result/f'{split}-baselines.json', public)
            eligible = {r['id'] for r in public if r['family'] == 'composed' and r['source_correct']
                        and r['donor_correct'] and r['first_token_differs']}
            copy_ids = {r['id'] for r in public if r['family'] == 'copy' and r['source_correct']}
            write_json(result/f'{split}-capability.json', {'eligible': len(eligible), 'source_copy': len(copy_ids)})
            if len(eligible) < 8 or len(copy_ids) < 6:
                write_json(result/'decision.json', {'status': 'inconclusive', 'reason': f'{split}_baseline_competence'})
                return

            split_rows = []
            for task in group:
                budget(len(all_rows))
                source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
                delta = donor['field']-source['field']
                edit = np.zeros_like(delta)
                edit[differences[task['id']]] = delta[differences[task['id']]]
                donor_logits, _, _ = model.prefill_field(ids[task['prompt']], fields={UPSTREAM: delta})
                error = float(np.max(np.abs(donor_logits-donor['logits'])))
                max_donor_error = max(max_donor_error, error)
                if error >= .002 or int(donor_logits.argmax()) != int(donor['logits'].argmax()):
                    raise ValueError('Full-field donor first-token control failed')
                record_conditions = {'upstream': ()} | WINDOWS | {'all_cut': ALL}
                upstream_writes = None
                for condition, layers in record_conditions.items():
                    budget(len(all_rows))
                    phrase, tokens, logits, writes = model.generate_cut(
                        ids[task['prompt']], upstream_layer=UPSTREAM, upstream_field=edit,
                        clean_writes=source['writes'], cut_layers=layers,
                        capture_writes=condition == 'upstream',
                    )
                    if condition == 'upstream':
                        upstream_writes = writes
                    if condition == 'all_cut':
                        error = float(np.max(np.abs(logits-source['logits'])))
                        max_full_error = max(max_full_error, error)
                        if error >= .002 or int(logits.argmax()) != int(source['logits'].argmax()):
                            raise ValueError('All downstream final-attention cut failed to restore source logits')
                    row = {
                        'task_id': task['id'], 'split': split, 'bundle': task['bundle'],
                        'template': task['template'], 'family': task['family'],
                        'condition': condition, 'text': phrase, 'tokens': tokens,
                        'eligible': task['id'] in eligible, 'baseline_copy': task['id'] in copy_ids,
                        'desired_answer': answer_correct(phrase, [task['donor_answer']]) if task['family'] == 'composed' else False,
                        'donor_first_token': bool(tokens) and bool(donor['tokens']) and tokens[0] == donor['tokens'][0],
                        'copy_preserved': answer_correct(phrase, [task['answer']]) if task['family'] == 'copy' else False,
                        'donor_margin': (float(logits[donor['tokens'][0]]-logits[source['tokens'][0]])
                                         if source['tokens'] and donor['tokens'] and source['tokens'][0] != donor['tokens'][0] else None),
                    }
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
                                                        -float(np.linalg.norm(actual))))
                    phrase, tokens, logits, _ = model.generate_cut(
                        ids[task['prompt']], upstream_layer=UPSTREAM, upstream_field=edit,
                        clean_writes=source['writes'], cut_layers=WINDOWS[selected],
                        random_deltas=random_deltas,
                    )
                    row = {
                        'task_id': task['id'], 'split': split, 'bundle': task['bundle'],
                        'template': task['template'], 'family': task['family'],
                        'condition': 'random_selected', 'text': phrase, 'tokens': tokens,
                        'eligible': task['id'] in eligible, 'baseline_copy': task['id'] in copy_ids,
                        'desired_answer': answer_correct(phrase, [task['donor_answer']]) if task['family'] == 'composed' else False,
                        'donor_first_token': bool(tokens) and bool(donor['tokens']) and tokens[0] == donor['tokens'][0],
                        'copy_preserved': answer_correct(phrase, [task['answer']]) if task['family'] == 'copy' else False,
                        'donor_margin': (float(logits[donor['tokens'][0]]-logits[source['tokens'][0]])
                                         if source['tokens'] and donor['tokens'] and source['tokens'][0] != donor['tokens'][0] else None),
                    }
                    split_rows.append(row)
                    all_rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    log.flush()
            write_json(result/f'{split}-rows.json', split_rows)
            if split == 'discovery':
                selected, ranking = choose_window(split_rows, eligible)
                write_json(result/'selection.json', {'selected_window': selected, 'ranking': ranking,
                                                      'eligible': len(eligible), 'seed': SEED})

    write_json(result/'rows.json', all_rows)
    confirmation = [r for r in all_rows if r['split'] == 'confirmation']
    eligible = {r['id'] for r in baselines_public['confirmation'] if r['family'] == 'composed'
                and r['source_correct'] and r['donor_correct'] and r['first_token_differs']}
    copy_ids = {r['id'] for r in baselines_public['confirmation'] if r['family'] == 'copy' and r['source_correct']}
    upstream = {r['task_id']: r for r in confirmation if r['condition'] == 'upstream'}
    success = {key for key in eligible if upstream[key]['desired_answer']}
    by_condition = {name: {r['task_id']: r for r in confirmation if r['condition'] == name}
                    for name in ('upstream', *WINDOWS, 'all_cut', 'random_selected')}
    reductions = {name: sum(not by_condition[name][key]['donor_first_token'] for key in success)
                  for name in by_condition}
    copy_preserve = {name: sum(by_condition[name][key]['copy_preserved'] for key in copy_ids)
                     for name in by_condition}
    random_rate = reductions['random_selected']/len(success) if success else 0.0
    selected_rate = reductions[selected]/len(success) if success else 0.0
    gate = len(success)/len(eligible) >= .5 and copy_preserve['upstream']/len(copy_ids) >= .9
    supported = (gate and selected_rate >= .5 and selected_rate-random_rate >= .2
                 and copy_preserve[selected]/len(copy_ids) >= .9)
    write_json(result/'summary.json', {'eligible': len(eligible), 'upstream_success': len(success),
                                       'baseline_copy': len(copy_ids), 'reductions': reductions,
                                       'copy_preserve': copy_preserve, 'selected_window': selected})
    write_json(result/'decision.json', {
        'status': ('exploratory_supported' if supported else
                   ('uninformative_upstream' if not gate else 'exploratory_failed')),
        'upstream_gate': gate, 'selected_window': selected, 'supported': supported,
        'selected_reduction_rate': selected_rate, 'random_reduction_rate': random_rate,
        'all_cut_max_logit_error': max_full_error, 'full_donor_max_logit_error': max_donor_error,
        'random_max_norm_error': max_random_norm_error,
        'elapsed_seconds': time.monotonic()-started, 'peak_mlx_bytes': model.mx.get_peak_memory(),
        'scope': 'Fresh composed English lookups only; no historical decipherment',
    })


if __name__ == '__main__':
    run()
