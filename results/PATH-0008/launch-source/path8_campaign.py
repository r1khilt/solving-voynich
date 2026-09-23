"""Registered head-level mediation after a direct-lookup residual transplant."""

from pathlib import Path
import json
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_campaign import condition_seed
from .geometry import matched_random_delta
from .path6_campaign import changed_positions
from .path8_backend import Path8Workspace
from .path8_tasks import path8_tasks


UPSTREAM = 23
HEAD_LAYERS = (28, 29, 30, 31)
WINDOW = tuple(range(28, 32))
ALL_CUT = tuple(range(24, 36))
SEED = 510101
CAP_SECONDS = 5400
CAP_BYTES = 45_000_000_000
TOLERANCE = .002
CONDITIONS = ('upstream', 'selected_one', 'selected_four', 'random_four_norm',
              'mismatch_four_norm', 'all_heads', 'full_window', 'all_cut', 'reverse_four')


def select_heads(rows, success):
    """Mean margin reduction, top four per layer, maximum top-four sum."""
    if not success:
        raise ValueError('No discovery upstream successes')
    by_key = {(r['task_id'], r['layer'], r['head']): r for r in rows}
    ranking = []
    for layer in HEAD_LAYERS:
        heads = []
        for head in range(32):
            values = [by_key[(task_id, layer, head)]['margin_drop'] for task_id in sorted(success)]
            heads.append({'head': head, 'mean_margin_drop': float(np.mean(values)),
                          'min_margin_drop': float(min(values)), 'max_margin_drop': float(max(values))})
        heads.sort(key=lambda r: (-r['mean_margin_drop'], r['head']))
        ranking.append({'layer': layer, 'top_four': [r['head'] for r in heads[:4]],
                        'top_four_sum': float(sum(r['mean_margin_drop'] for r in heads[:4])),
                        'heads': heads})
    selected = sorted(ranking, key=lambda r: (-r['top_four_sum'], r['layer']))[0]
    return selected['layer'], tuple(selected['top_four']), ranking


def bundle_bootstrap(rows, successes, control):
    by_key = {(r['task_id'], r['condition']): r for r in rows}
    bundles = sorted({int(key.split('/')[2]) for key in successes})
    if not bundles:
        return {'resamples': 0, 'lower_95': None, 'upper_95': None}
    members = {bundle: sorted(key for key in successes if int(key.split('/')[2]) == bundle)
               for bundle in bundles}
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(10_000):
        chosen = rng.choice(bundles, size=len(bundles), replace=True)
        effects = [int(not by_key[(key, 'selected_four')]['donor_first_token'])
                   - int(not by_key[(key, control)]['donor_first_token'])
                   for bundle in chosen for key in members[bundle]]
        estimates.append(float(np.mean(effects)))
    return {'resamples': len(estimates), 'lower_95': float(np.quantile(estimates, .025)),
            'upper_95': float(np.quantile(estimates, .975))}


def _row(task, condition, phrase, tokens, logits, source, donor, eligible, copies):
    source_first = source['tokens'][0] if source['tokens'] else None
    donor_first = donor['tokens'][0] if donor['tokens'] else None
    return {
        'task_id': task['id'], 'split': task['split'], 'bundle': task['bundle'],
        'template': task['template'], 'family': task['family'], 'condition': condition,
        'text': phrase, 'tokens': tokens, 'eligible': task['id'] in eligible,
        'baseline_copy': task['id'] in copies,
        'desired_answer': answer_correct(phrase, [task['donor_answer']])
        if task['family'] == 'binding' else False,
        'source_answer': answer_correct(phrase, [task['answer']]),
        'donor_first_token': bool(tokens) and donor_first is not None and tokens[0] == donor_first,
        'copy_preserved': answer_correct(phrase, [task['answer']])
        if task['family'] == 'copy' else False,
        'donor_margin': (float(logits[donor_first]-logits[source_first])
                         if source_first is not None and donor_first is not None
                         and source_first != donor_first else None),
    }


def _check_logits(actual, expected, label, controls):
    error = float(np.max(np.abs(actual-expected)))
    controls[label] = max(controls.get(label, 0.0), error)
    if error >= TOLERANCE or int(actual.argmax()) != int(expected.argmax()):
        raise ValueError(f'{label} numerical control failed: {error}')


def _save_row(row, path, rows):
    rows.append(row)
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def run():
    output, result = Path('outputs/PATH-0008'), Path('results/PATH-0008')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result/'inputs.json').exists() or (output/'observations.jsonl').exists():
        raise FileExistsError('No automatic PATH-0008 rerun')
    started = time.monotonic()
    config = load_config('configs/jspace0001.json')
    model = Path8Workspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(min(config['memory_bytes_cap'], CAP_BYTES))
    controls = {}
    completed = 0

    def budget():
        if time.monotonic()-started >= CAP_SECONDS or model.mx.get_peak_memory() > CAP_BYTES:
            write_json(result/'incomplete.json', {'elapsed_seconds': time.monotonic()-started,
                                                   'completed_interventions': completed,
                                                   'reason': 'resource_cap'})
            raise RuntimeError('PATH-0008 resource cap')

    def finish(status, reason=None, **extra):
        write_json(result/'qualification.json', {
            'observed_control_maxima': controls,
            'all_confirmation_controls_executed': 'all_cut_source' in controls,
            'tolerance': TOLERANCE,
        })
        decision = {'status': status, 'reason': reason, 'controls': controls,
                    'elapsed_seconds': time.monotonic()-started,
                    'peak_mlx_bytes': model.mx.get_peak_memory(), **extra}
        write_json(result/'decision.json', decision)
        return decision

    tasks = {split: path8_tasks(split) for split in ('discovery', 'confirmation')}
    prompts = {r[key] for group in tasks.values() for r in group for key in ('prompt', 'donor_prompt')}
    ids = {prompt: model.encode(prompt, chat=True) for prompt in prompts}
    changed = {r['id']: changed_positions(ids[r['prompt']], ids[r['donor_prompt']])
               for group in tasks.values() for r in group}
    write_json(output/'rendered-inputs.json', {canonical_digest(prompt): token_ids
                                               for prompt, token_ids in ids.items()})
    sources = [Path('src/voynich/workspace')/name for name in
               ('path8_tasks.py', 'path8_backend.py', 'path8_campaign.py', 'path6_tasks.py',
                'path6_campaign.py', 'path3_backend.py', 'attention_route_backend.py',
                'route_backend.py')]
    sources.append(Path('docs/experiments/PATH-0008.md'))
    write_json(result/'inputs.json', {
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(path): digest(path) for path in sources},
        'tasks_sha256': canonical_digest(tasks),
        'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
        'changed_positions_sha256': canonical_digest(changed),
        'model_input_sha256': digest('results/JSPACE-0001/inputs.json'),
        'seed': SEED, 'upstream': UPSTREAM, 'head_layers': HEAD_LAYERS,
        'window': WINDOW, 'all_cut': ALL_CUT, 'cap_seconds': CAP_SECONDS,
        'memory_bytes_cap': CAP_BYTES, 'tolerance': TOLERANCE,
    })

    baseline_dir = output/'baselines'
    baseline_dir.mkdir(exist_ok=True)
    baselines = {}

    def baseline(prompt):
        if prompt not in baselines:
            budget()
            token_ids = ids[prompt]
            phrase, tokens, fields, logits = model.generate_field(token_ids, capture_layers=(UPSTREAM,))
            clean_logits, writes = model.clean_final_writes(token_ids, start=24)
            _check_logits(clean_logits, logits, 'clean_write_capture', controls)
            zeros = np.zeros_like(fields[UPSTREAM])
            head_logits, heads = model.final_heads_after_field(
                token_ids, upstream_layer=UPSTREAM, field=zeros, capture_layers=HEAD_LAYERS)
            _check_logits(head_logits, logits, 'clean_head_capture', controls)
            for layer in HEAD_LAYERS:
                controls['clean_head_projection'] = max(
                    controls.get('clean_head_projection', 0.0), heads[layer]['native_error'])
                if heads[layer]['native_error'] >= TOLERANCE:
                    raise ValueError('Clean head reconstruction failed')
            local = baseline_dir/(canonical_digest(prompt)+'.npz')
            np.savez_compressed(local, logits=logits, field=fields[UPSTREAM],
                                **{f'write_{layer}': value for layer, value in writes.items()},
                                **{f'heads_{layer}': heads[layer]['heads'] for layer in HEAD_LAYERS})
            baselines[prompt] = {'text': phrase, 'tokens': tokens, 'logits': logits,
                                 'field': fields[UPSTREAM], 'writes': writes,
                                 'heads': heads, 'arrays_sha256': digest(local)}
        return baselines[prompt]

    public_baselines = {}
    scan = []
    confirm_rows = []
    selected_layer = selected_heads = None
    with (output/'observations.jsonl').open('w'):
        pass
    for split in ('discovery', 'confirmation'):
        public = []
        for task in tasks[split]:
            source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
            public.append({
                'id': task['id'], 'split': split, 'bundle': task['bundle'],
                'template': task['template'], 'family': task['family'],
                'source_text': source['text'], 'donor_text': donor['text'],
                'source_correct': answer_correct(source['text'], [task['answer']]),
                'donor_correct': answer_correct(donor['text'], [task['donor_answer']]),
                'first_token_differs': bool(source['tokens']) and bool(donor['tokens'])
                and source['tokens'][0] != donor['tokens'][0],
                'source_arrays_sha256': source['arrays_sha256'],
                'donor_arrays_sha256': donor['arrays_sha256'],
            })
        public_baselines[split] = public
        write_json(result/f'{split}-baselines.json', public)
        eligible = {r['id'] for r in public if r['family'] == 'binding' and r['source_correct']
                    and r['donor_correct'] and r['first_token_differs']}
        copies = {r['id'] for r in public if r['family'] == 'copy' and r['source_correct']}
        write_json(result/f'{split}-capability.json', {'eligible': len(eligible),
                                                       'source_copy': len(copies)})
        if len(eligible) < 24 or len(copies) < 14:
            finish('inconclusive', f'{split}_baseline_competence',
                   eligible=len(eligible), source_copy=len(copies))
            return

        if split == 'discovery':
            success = set()
            discovery_state = {}
            discovery_upstream = []
            for task in tasks[split]:
                if task['id'] not in eligible:
                    continue
                budget()
                source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
                field = np.zeros_like(source['field'])
                field[changed[task['id']]] = (donor['field']-source['field'])[changed[task['id']]]
                full_logits, _, _ = model.prefill_field(
                    ids[task['prompt']], fields={UPSTREAM: donor['field']-source['field']})
                _check_logits(full_logits, donor['logits'], 'whole_donor', controls)
                phrase, tokens, upstream_logits, _ = model.generate_cut(
                    ids[task['prompt']], upstream_layer=UPSTREAM, upstream_field=field,
                    clean_writes=source['writes'])
                completed += 1
                desired = answer_correct(phrase, [task['donor_answer']])
                discovery_upstream.append({'task_id': task['id'], 'bundle': task['bundle'],
                                           'text': phrase, 'tokens': tokens,
                                           'desired_answer': desired})
                if desired:
                    success.add(task['id'])
                    discovery_state[task['id']] = {'field': field, 'upstream_logits': upstream_logits}
            write_json(result/'discovery-upstream.json', discovery_upstream)
            if len(success) < 18:
                finish('uninformative_upstream', 'discovery_upstream_below_18',
                       upstream_success=len(success), eligible=len(eligible))
                return
            for task in tasks[split]:
                if task['id'] not in success:
                    continue
                budget()
                source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
                field = discovery_state[task['id']]['field']
                upstream_logits = discovery_state[task['id']]['upstream_logits']
                edited_logits, edited_heads = model.final_heads_after_field(
                    ids[task['prompt']], upstream_layer=UPSTREAM, field=field,
                    capture_layers=HEAD_LAYERS)
                _check_logits(edited_logits, upstream_logits, 'edited_head_capture', controls)
                for layer in HEAD_LAYERS:
                    controls['edited_head_projection'] = max(
                        controls.get('edited_head_projection', 0.0), edited_heads[layer]['native_error'])
                    if edited_heads[layer]['native_error'] >= TOLERANCE:
                        raise ValueError('Edited head reconstruction failed')
                donor_first, source_first = donor['tokens'][0], source['tokens'][0]
                original_margin = float(upstream_logits[donor_first]-upstream_logits[source_first])
                for layer in HEAD_LAYERS:
                    for head in range(32):
                        budget()
                        delta = model.head_delta(layer, edited_heads[layer]['heads'],
                                                 source['heads'][layer]['heads'], (head,))
                        cut_logits, _, _ = model.prefill_cut(
                            ids[task['prompt']], upstream_layer=UPSTREAM, upstream_field=field,
                            clean_writes=source['writes'], cut_layers=(layer,),
                            random_deltas={layer: delta})
                        margin = float(cut_logits[donor_first]-cut_logits[source_first])
                        row = {'task_id': task['id'], 'bundle': task['bundle'],
                               'layer': layer, 'head': head,
                               'margin_drop': original_margin-margin,
                               'donor_first_token': int(cut_logits.argmax()) == donor_first,
                               'delta_norm': float(np.linalg.norm(delta))}
                        _save_row(row, output/'scan.jsonl', scan)
                        completed += 1
            write_json(result/'discovery-screen.json', scan)
            selected_layer, selected_heads, ranking = select_heads(scan, success)
            write_json(result/'selection.json', {'layer': selected_layer, 'heads': selected_heads,
                                                 'ranking': ranking, 'upstream_success': len(success),
                                                 'seed': SEED})
            continue

        # Confirmation: selected head group is frozen from exposed discovery only.
        max_norm_error = 0.0
        prepared = {}
        for task in tasks[split]:
            budget()
            source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
            token_ids = ids[task['prompt']]
            field = np.zeros_like(source['field'])
            field[changed[task['id']]] = (donor['field']-source['field'])[changed[task['id']]]
            full_logits, _, _ = model.prefill_field(
                token_ids, fields={UPSTREAM: donor['field']-source['field']})
            _check_logits(full_logits, donor['logits'], 'whole_donor', controls)
            identity_text, identity_tokens, identity_logits, _ = model.generate_cut(
                token_ids, upstream_layer=UPSTREAM, upstream_field=np.zeros_like(field),
                clean_writes=source['writes'])
            _check_logits(identity_logits, source['logits'], 'zero_edit_identity', controls)
            if identity_text != source['text'] or identity_tokens != source['tokens']:
                raise ValueError('Zero-edit full-generation identity failed')
            edited_logits, edited_heads = model.final_heads_after_field(
                token_ids, upstream_layer=UPSTREAM, field=field,
                capture_layers=HEAD_LAYERS)
            for layer in HEAD_LAYERS:
                controls['edited_head_projection'] = max(
                    controls.get('edited_head_projection', 0.0), edited_heads[layer]['native_error'])
                if edited_heads[layer]['native_error'] >= TOLERANCE:
                    raise ValueError('Edited head reconstruction failed')
            group_delta = model.head_delta(selected_layer, edited_heads[selected_layer]['heads'],
                                           source['heads'][selected_layer]['heads'], selected_heads)
            one_delta = model.head_delta(selected_layer, edited_heads[selected_layer]['heads'],
                                         source['heads'][selected_layer]['heads'], selected_heads[:1])
            all_delta = model.head_delta(selected_layer, edited_heads[selected_layer]['heads'],
                                         source['heads'][selected_layer]['heads'], tuple(range(32)))
            write_delta = source['heads'][selected_layer]['write']-edited_heads[selected_layer]['write']
            controls['all_head_write_projection'] = max(
                controls.get('all_head_write_projection', 0.0),
                float(np.max(np.abs(all_delta-write_delta))))
            if controls['all_head_write_projection'] >= TOLERANCE:
                raise ValueError('All-head projected delta differs from full attention write')
            random_delta = matched_random_delta(
                group_delta, seed=condition_seed(SEED, task['id'], selected_layer, 'random-four'))
            max_norm_error = max(max_norm_error,
                                 abs(float(np.linalg.norm(group_delta))
                                     - float(np.linalg.norm(random_delta))))
            prepared[task['id']] = {
                'field': field, 'edited_logits': edited_logits, 'group_delta': group_delta,
                'one_delta': one_delta, 'all_delta': all_delta, 'random_delta': random_delta,
            }

        for task in tasks[split]:
            budget()
            source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
            token_ids = ids[task['prompt']]
            item = prepared[task['id']]
            field = item['field']
            group_delta = item['group_delta']
            parts = task['id'].split('/')
            mismatched_id = '/'.join(parts[:2]+[str((task['bundle']+1) % 8)]+parts[3:])
            unrelated_delta = prepared[mismatched_id]['group_delta']
            target_norm = float(np.linalg.norm(group_delta))
            unrelated_norm = float(np.linalg.norm(unrelated_delta))
            if unrelated_norm == 0.0 and target_norm > 0.0:
                raise ValueError('Zero-norm mismatched-bundle delta cannot be matched')
            mismatch_delta = (unrelated_delta * (target_norm/unrelated_norm)
                              if unrelated_norm else np.zeros_like(unrelated_delta))
            max_norm_error = max(max_norm_error,
                                 abs(float(np.linalg.norm(mismatch_delta))-target_norm))
            conditions = {
                'upstream': (field, (), None),
                'selected_one': (field, (selected_layer,), {selected_layer: item['one_delta']}),
                'selected_four': (field, (selected_layer,), {selected_layer: group_delta}),
                'random_four_norm': (field, (selected_layer,), {selected_layer: item['random_delta']}),
                'mismatch_four_norm': (field, (selected_layer,), {selected_layer: mismatch_delta}),
                'all_heads': (field, (selected_layer,), None),
                'full_window': (field, WINDOW, None),
                'all_cut': (field, ALL_CUT, None),
                'reverse_four': (np.zeros_like(field), (selected_layer,),
                                 {selected_layer: -group_delta}),
            }
            first_logits = {}
            for condition in CONDITIONS:
                budget()
                intervention_field, cut_layers, additive = conditions[condition]
                phrase, tokens, logits, _ = model.generate_cut(
                    token_ids, upstream_layer=UPSTREAM, upstream_field=intervention_field,
                    clean_writes=source['writes'], cut_layers=cut_layers,
                    random_deltas=additive)
                first_logits[condition] = logits
                row = _row(task, condition, phrase, tokens, logits,
                           source, donor, eligible, copies)
                _save_row(row, output/'observations.jsonl', confirm_rows)
                completed += 1
                if condition == 'all_cut':
                    _check_logits(logits, source['logits'], 'all_cut_source', controls)
            _check_logits(item['edited_logits'], first_logits['upstream'], 'edited_head_capture', controls)
            additive_logits, _, _ = model.prefill_cut(
                token_ids, upstream_layer=UPSTREAM, upstream_field=field,
                clean_writes=source['writes'], cut_layers=(selected_layer,),
                random_deltas={selected_layer: item['all_delta']})
            _check_logits(additive_logits, first_logits['all_heads'],
                          'all_head_vs_full_block', controls)
        controls['random_norm_error'] = max_norm_error
        if max_norm_error >= .002:
            raise ValueError('Norm-matched random control failed')
        write_json(result/'confirmation-rows.json', confirm_rows)

    eligible = {r['id'] for r in public_baselines['confirmation'] if r['family'] == 'binding'
                and r['source_correct'] and r['donor_correct'] and r['first_token_differs']}
    copies = {r['id'] for r in public_baselines['confirmation'] if r['family'] == 'copy'
              and r['source_correct']}
    by_condition = {name: {r['task_id']: r for r in confirm_rows if r['condition'] == name}
                    for name in CONDITIONS}
    success = {key for key in eligible if by_condition['upstream'][key]['desired_answer']}
    reductions = {name: sum(not by_condition[name][key]['donor_first_token'] for key in success)
                  for name in CONDITIONS}
    copies_preserved = {name: sum(by_condition[name][key]['copy_preserved'] for key in copies)
                        for name in CONDITIONS}
    full_source = {name: sum(by_condition[name][key]['source_answer'] for key in success)
                   for name in CONDITIONS}
    denominator = len(success)
    upstream_rate = denominator/len(eligible)
    selected_rate = reductions['selected_four']/denominator if denominator else 0.0
    random_rate = reductions['random_four_norm']/denominator if denominator else 0.0
    mismatch_rate = reductions['mismatch_four_norm']/denominator if denominator else 0.0
    window_rate = reductions['full_window']/denominator if denominator else 0.0
    behavioral_gate = (upstream_rate >= .75 and copies_preserved['upstream']/len(copies) >= .9)
    window_gate = window_rate >= .35
    supported = (behavioral_gate and window_gate and selected_rate >= .35
                 and selected_rate-random_rate >= .25
                 and selected_rate-mismatch_rate >= .25
                 and copies_preserved['selected_four']/len(copies) >= .9)
    summary = {
        'eligible': len(eligible), 'upstream_success': denominator, 'source_copy': len(copies),
        'selected_layer': selected_layer, 'selected_heads': selected_heads,
        'reductions': reductions, 'full_source_answers': full_source,
        'copy_preserved': copies_preserved,
        'selected_minus_random_bundle_bootstrap': bundle_bootstrap(
            confirm_rows, success, 'random_four_norm'),
        'selected_minus_mismatch_bundle_bootstrap': bundle_bootstrap(
            confirm_rows, success, 'mismatch_four_norm'),
    }
    write_json(result/'summary.json', summary)
    status = ('exploratory_supported' if supported else
              'uninformative_upstream' if not behavioral_gate else
              'uninformative_positive_control' if not window_gate else 'exploratory_failed')
    finish(status, None, supported=supported, upstream_gate=behavioral_gate,
           full_window_gate=window_gate, upstream_success_rate=upstream_rate,
           selected_reduction_rate=selected_rate, random_reduction_rate=random_rate,
           mismatch_reduction_rate=mismatch_rate,
           full_window_reduction_rate=window_rate, selected_layer=selected_layer,
           selected_heads=selected_heads,
           scope='Supplied English direct lookup only; no historical decipherment')


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        failure = Path('results/PATH-0008/failure.json')
        failure.parent.mkdir(parents=True, exist_ok=True)
        if not failure.exists():
            write_json(failure, {'status': 'incomplete_or_uninformative',
                                 'error_type': type(error).__name__, 'message': str(error)})
        raise
