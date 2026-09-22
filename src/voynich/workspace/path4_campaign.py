"""Frozen layer-by-position transport grid on a new two-step lookup panel."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_campaign import condition_seed
from .geometry import matched_random_delta
from .path4_tasks import path4_tasks
from .route_backend import RouteWorkspace

LAYERS = (7, 15, 23)
CONDITIONS = ('identity', 'value', 'suffix', 'value_suffix', 'all_earlier', 'whole', 'suffix_random')
SEED = 51071
CAP_SECONDS = 3000


def changed_positions(source, donor):
    if len(source) != len(donor) or len(source) < 30:
        raise ValueError('Unequal or short rendered prompts')
    positions = np.flatnonzero(np.asarray(source) != np.asarray(donor)).tolist()
    if len(positions) < 2 or positions[-1] >= len(source)-20:
        raise ValueError('First-table changes overlap the common suffix')
    return positions


def make_fields(delta, changed, *, seed):
    value = np.zeros_like(delta)
    value[changed] = delta[changed]
    suffix = np.zeros_like(delta)
    suffix[max(changed)+1:-1] = delta[max(changed)+1:-1]
    random = np.zeros_like(delta)
    for position in range(max(changed)+1, len(delta)-1):
        random[position] = matched_random_delta(delta[position],
                                                seed=condition_seed(seed, str(position), len(delta), 'suffix'))
    earlier = delta.copy()
    earlier[-1] = 0
    return {'identity': np.zeros_like(delta), 'value': value, 'suffix': suffix,
            'value_suffix': value+suffix, 'all_earlier': earlier,
            'whole': delta.copy(), 'suffix_random': random}


def bundle_bootstrap(rows, eligible, a, b):
    lookup = {(r['task_id'], r['condition']): r for r in rows}
    task_bundle = {r['task_id']: r['bundle'] for r in rows}
    grouped = {bundle: sorted(key for key in eligible if task_bundle[key] == bundle) for bundle in range(6)}
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(10000):
        picks = rng.integers(0, 6, size=6)
        values = [int(lookup[(key, a)]['desired_answer'])-int(lookup[(key, b)]['desired_answer'])
                  for bundle in picks for key in grouped[int(bundle)]]
        if values:
            estimates.append(float(np.mean(values)))
    return {'resamples': len(estimates), 'lower_95': float(np.quantile(estimates, .025)),
            'upper_95': float(np.quantile(estimates, .975))}


def run():
    output, result = Path('outputs/PATH-0004'), Path('results/PATH-0004')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result/'inputs.json').exists() or (output/'observations.jsonl').exists():
        raise FileExistsError('No automatic repeat of PATH-0004')
    started = time.monotonic()
    config = load_config('configs/jspace0001.json')
    model = RouteWorkspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(config['memory_bytes_cap'])

    def budget(count):
        elapsed = time.monotonic()-started
        if elapsed >= CAP_SECONDS or model.mx.get_peak_memory() > config['memory_bytes_cap']:
            write_json(result/'incomplete.json', {'elapsed_seconds': elapsed, 'completed_rows': count,
                                                   'reason': 'resource_cap'})
            raise RuntimeError('PATH-0004 resource cap')

    probe = model.encode('Follow two tables, then return the final word.', chat=True)
    rng = np.random.default_rng(SEED)
    patch = rng.normal(size=(len(probe), model.width)).astype(np.float32)*.01
    qualification = []
    for layer in LAYERS:
        candidate, _, _ = model.prefill_field(probe, fields={layer: patch})
        manual = [{'layer': layer, 'position': p, 'delta': value} for p, value in enumerate(patch)]
        reference, _ = model.forward(probe, patches=manual)
        qualification.append({'layer': layer, 'max_error': float(np.max(np.abs(candidate-reference[0]))),
                              'argmax_same': int(candidate.argmax()) == int(reference[0].argmax())})
    write_json(result/'qualification.json', {'passed': all(r['max_error'] < .002 and r['argmax_same']
                                                          for r in qualification), 'rows': qualification})
    if not load_qualification(result/'qualification.json'):
        raise ValueError('PATH-0004 numerical qualification failed')

    tasks = {split: path4_tasks(split) for split in ('discovery', 'confirmation')}
    if len({word for split in tasks for task in tasks[split] for word in
            (task['answer'], task['donor_answer'])}) < 24:
        raise ValueError('Output vocabulary unexpectedly repeated')
    prompts = {r['prompt'] for group in tasks.values() for r in group}
    prompts.update(r['donor_prompt'] for group in tasks.values() for r in group)
    ids = {p: model.encode(p, chat=True) for p in prompts}
    changed = {r['id']: changed_positions(ids[r['prompt']], ids[r['donor_prompt']])
               for group in tasks.values() for r in group}
    write_json(output/'rendered-inputs.json', {canonical_digest(p): value for p, value in ids.items()})
    sources = [Path('src/voynich/workspace')/name for name in
               ('path4_tasks.py', 'path4_campaign.py', 'route_backend.py')]
    sources.append(Path('docs/experiments/PATH-0004.md'))
    write_json(result/'inputs.json', {
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): digest(p) for p in sources},
        'tasks_sha256': canonical_digest(tasks), 'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
        'changed_positions_sha256': canonical_digest(changed),
        'model_input_sha256': digest('results/JSPACE-0001/inputs.json'),
        'seed': SEED, 'layers': LAYERS, 'conditions': CONDITIONS,
        'cap_seconds': CAP_SECONDS, 'memory_bytes_cap': config['memory_bytes_cap'],
    })

    baselines = {}
    baseline_dir = output/'baselines'
    baseline_dir.mkdir(exist_ok=True)

    def baseline(prompt):
        if prompt not in baselines:
            budget(0)
            phrase, tokens, captures, logits = model.generate_field(ids[prompt], capture_layers=LAYERS)
            local = baseline_dir/(canonical_digest(prompt)+'.npz')
            np.savez_compressed(local, logits=logits, **{str(layer): field for layer, field in captures.items()})
            baselines[prompt] = {'text': phrase, 'tokens': tokens, 'logits': logits,
                                 'captures': captures, 'arrays_sha256': digest(local)}
        return baselines[prompt]

    all_rows = []
    baselines_public = {}
    max_control_error = 0.0
    max_random_norm_error = 0.0
    with (output/'observations.jsonl').open('w') as log:
        for split in ('discovery', 'confirmation'):
            public = []
            for task in tasks[split]:
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
            copies = {r['id'] for r in public if r['family'] == 'copy' and r['source_correct']}
            write_json(result/f'{split}-capability.json', {'eligible': len(eligible), 'source_copy': len(copies)})
            if len(eligible) < 12 or len(copies) < 10:
                write_json(result/'decision.json', {'status': 'inconclusive', 'reason': f'{split}_baseline_competence'})
                return
            for task in tasks[split]:
                source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
                for layer in LAYERS:
                    delta = donor['captures'][layer]-source['captures'][layer]
                    edits = make_fields(delta, changed[task['id']],
                                        seed=condition_seed(SEED, task['id'], layer, 'field'))
                    if not np.array_equal(edits['value_suffix'], edits['value']+edits['suffix']):
                        raise AssertionError('Value/suffix decomposition failed')
                    for position in range(max(changed[task['id']])+1, len(delta)-1):
                        max_random_norm_error = max(max_random_norm_error,
                            abs(float(np.linalg.norm(edits['suffix_random'][position]))
                                -float(np.linalg.norm(edits['suffix'][position]))))
                    for condition in CONDITIONS:
                        budget(len(all_rows))
                        phrase, tokens, _, logits = model.generate_field(
                            ids[task['prompt']], fields={layer: edits[condition]})
                        if condition == 'identity' and (phrase != source['text'] or tokens != source['tokens']):
                            raise ValueError('Identity generation changed')
                        if condition == 'whole':
                            error = float(np.max(np.abs(logits-donor['logits'])))
                            max_control_error = max(max_control_error, error)
                            if error >= .002 or int(logits.argmax()) != int(donor['logits'].argmax()):
                                raise ValueError('Whole-prefix donor first-token control failed')
                        row = {
                            'task_id': task['id'], 'split': split, 'bundle': task['bundle'],
                            'template': task['template'], 'family': task['family'],
                            'layer': layer, 'condition': condition, 'text': phrase, 'tokens': tokens,
                            'eligible': task['id'] in eligible, 'baseline_copy': task['id'] in copies,
                            'desired_answer': answer_correct(phrase, [task['donor_answer']]) if task['family'] == 'composed' else False,
                            'donor_first_token': bool(tokens) and bool(donor['tokens']) and tokens[0] == donor['tokens'][0],
                            'copy_preserved': answer_correct(phrase, [task['answer']]) if task['family'] == 'copy' else False,
                            'field_norm': float(np.linalg.norm(edits[condition])),
                        }
                        all_rows.append(row)
                        log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        log.flush()
            write_json(result/f'{split}-rows.json', [r for r in all_rows if r['split'] == split])

    write_json(result/'rows.json', all_rows)
    confirm = [r for r in all_rows if r['split'] == 'confirmation' and r['layer'] == 15]
    base = baselines_public['confirmation']
    eligible = {r['id'] for r in base if r['family'] == 'composed' and r['source_correct']
                and r['donor_correct'] and r['first_token_differs']}
    copies = {r['id'] for r in base if r['family'] == 'copy' and r['source_correct']}
    summary = {}
    for name in CONDITIONS:
        selected = [r for r in confirm if r['condition'] == name]
        summary[name] = {'eligible': len(eligible),
                         'switches': sum(r['desired_answer'] for r in selected if r['task_id'] in eligible),
                         'copies': len(copies),
                         'copy_preserved': sum(r['copy_preserved'] for r in selected if r['task_id'] in copies)}
    write_json(result/'summary.json', summary)
    rates = {name: summary[name]['switches']/len(eligible) for name in CONDITIONS}
    copy_rates = {name: summary[name]['copy_preserved']/len(copies) for name in CONDITIONS}
    control = rates['whole'] >= .9 and copy_rates['whole'] >= .9
    suffix = control and rates['suffix'] >= .5 and rates['suffix']-rates['suffix_random'] >= .2 and copy_rates['suffix'] >= .9
    combined = control and rates['value_suffix'] >= .5 and rates['value_suffix']-rates['value'] >= .2 and copy_rates['value_suffix'] >= .9
    write_json(result/'decision.json', {
        'status': 'exploratory_scored' if control else 'uninformative_positive_control',
        'suffix_bypass_supported': suffix, 'combined_rescue_supported': combined,
        'eligible_confirmation': len(eligible), 'copy_confirmation': len(copies),
        'rates': rates, 'copy_rates': copy_rates,
        'suffix_minus_random_bundle_bootstrap': bundle_bootstrap(confirm, eligible, 'suffix', 'suffix_random'),
        'combined_minus_value_bundle_bootstrap': bundle_bootstrap(confirm, eligible, 'value_suffix', 'value'),
        'whole_max_logit_error': max_control_error, 'random_max_norm_error': max_random_norm_error,
        'elapsed_seconds': time.monotonic()-started, 'peak_mlx_bytes': model.mx.get_peak_memory(),
        'scope': 'Known two-step English lookup only; no historical decipherment',
    })


def load_qualification(path):
    return json.loads(Path(path).read_text())['passed']


if __name__ == '__main__':
    run()
