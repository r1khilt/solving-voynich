"""Fresh, bounded positional routing test for explicit two-slot bindings."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_campaign import condition_seed
from .geometry import matched_random_delta
from .path2_tasks import path2_tasks
from .route_backend import RouteWorkspace

LAYERS = (15, 19, 23, 27, 31)
CONDITIONS = ('identity', 'last_donor', 'value_donor', 'other_earlier_donor',
              'earlier_donor', 'all_donor', 'value_random')
CAP_SECONDS = 3000


def differences(source, donor):
    if len(source) != len(donor) or len(source) < 12:
        raise ValueError('Paired rendered prompts differ in length')
    positions = np.flatnonzero(np.asarray(source) != np.asarray(donor)).tolist()
    if len(positions) < 2 or positions[-1] >= len(source)-8:
        raise ValueError('Value-token differences overlap unchanged query suffix')
    return positions


def field_conditions(delta, positions, *, seed):
    value = np.zeros_like(delta)
    value[positions] = delta[positions]
    other = delta.copy()
    other[positions] = 0
    other[-1] = 0
    earlier = delta.copy()
    earlier[-1] = 0
    last = np.zeros_like(delta)
    last[-1] = delta[-1]
    random = np.zeros_like(delta)
    for position in positions:
        random[position] = matched_random_delta(delta[position], seed=condition_seed(seed, str(position), len(delta), 'value'))
    return {
        'identity': np.zeros_like(delta), 'last_donor': last,
        'value_donor': value, 'other_earlier_donor': other,
        'earlier_donor': earlier, 'all_donor': delta.copy(),
        'value_random': random,
    }


def summary(rows, eligible_ids, baseline_copy_ids):
    result = {}
    for layer in LAYERS:
        by_layer = {}
        for condition in CONDITIONS:
            chosen = [r for r in rows if r['layer'] == layer and r['condition'] == condition]
            semantic = [r for r in chosen if r['task_id'] in eligible_ids]
            copies = [r for r in chosen if r['task_id'] in baseline_copy_ids]
            by_template = {}
            by_bundle = {}
            for template in range(3):
                members = [r for r in semantic if r['template'] == template]
                by_template[str(template)] = {'count': len(members), 'switches': sum(r['desired_answer'] for r in members)}
            for bundle in range(12):
                members = [r for r in semantic if r['bundle'] == bundle]
                by_bundle[str(bundle)] = {'count': len(members), 'switches': sum(r['desired_answer'] for r in members)}
            by_layer[condition] = {
                'semantic': {'count': len(semantic), 'switches': sum(r['desired_answer'] for r in semantic)},
                'copy': {'count': len(copies), 'preserved': sum(r['copy_preserved'] for r in copies)},
                'all_semantic': {'count': sum(r['family'] == 'binding' for r in chosen),
                                 'switches': sum(r['desired_answer'] for r in chosen if r['family'] == 'binding')},
                'by_template': by_template, 'by_bundle': by_bundle,
            }
        result[str(layer)] = by_layer
    return result


def bootstrap_difference(rows, eligible_ids, a, b, *, seed=51052):
    # Equal-weight bundle resampling, descriptive only; reverse rows share a bundle.
    lookup = {(r['task_id'], r['condition']): r for r in rows if r['layer'] == 23 and r['task_id'] in eligible_ids}
    task_ids = sorted(eligible_ids)
    bundle = {i: [task_id for task_id in task_ids if int(task_id.split('/')[1]) == i] for i in range(12)}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(10000):
        picked = rng.integers(0, 12, size=12)
        values = [int(lookup[(task_id, a)]['desired_answer'])-int(lookup[(task_id, b)]['desired_answer'])
                  for index in picked for task_id in bundle[int(index)]]
        if values:
            estimates.append(float(np.mean(values)))
    return {'resamples': len(estimates), 'lower_95': float(np.quantile(estimates, .025)),
            'upper_95': float(np.quantile(estimates, .975))}


def run():
    config = load_config('configs/jspace0001.json')
    output, result = Path('outputs/PATH-0002'), Path('results/PATH-0002')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result/'inputs.json').exists() or (output/'observations.jsonl').exists():
        raise FileExistsError('No automatic repeat of PATH-0002')
    started = time.monotonic()
    model = RouteWorkspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(config['memory_bytes_cap'])

    def budget(count):
        elapsed = time.monotonic()-started
        if elapsed >= CAP_SECONDS or model.mx.get_peak_memory() > config['memory_bytes_cap']:
            write_json(result/'incomplete.json', {'elapsed_seconds': elapsed, 'completed_rows': count, 'reason': 'resource_cap'})
            raise RuntimeError('PATH-0002 resource cap')

    # The full-recompute oracle is independent of cached field execution.
    probe = model.encode('Read these two records, then return one short word.', chat=True)
    rng = np.random.default_rng(51051)
    patch = rng.normal(size=(len(probe), model.width)).astype(np.float32)*.01
    qualification = []
    for layer in (15, 23, 31):
        candidate, _, _ = model.prefill_field(probe, fields={layer: patch})
        patches = [{'layer': layer, 'position': position, 'delta': value} for position, value in enumerate(patch)]
        reference, _ = model.forward(probe, patches=patches)
        qualification.append({'layer': layer, 'max_error': float(np.max(np.abs(candidate-reference[0]))),
                              'argmax_same': int(candidate.argmax()) == int(reference[0].argmax())})
    write_json(result/'qualification.json', {'passed': all(r['max_error'] < .002 and r['argmax_same'] for r in qualification),
                                             'rows': qualification})
    if not all(r['max_error'] < .002 and r['argmax_same'] for r in qualification):
        raise ValueError('Residual-field numerical qualification failed')

    tasks = path2_tasks()
    prompts = {r['prompt'] for r in tasks} | {r['donor_prompt'] for r in tasks}
    ids = {prompt: model.encode(prompt, chat=True) for prompt in prompts}
    changed = {r['id']: differences(ids[r['prompt']], ids[r['donor_prompt']]) for r in tasks}
    write_json(output/'rendered-inputs.json', {canonical_digest(prompt): token_ids for prompt, token_ids in ids.items()})
    sources = sorted(Path('src/voynich/workspace').glob('*.py'))+[Path('docs/experiments/PATH-0002.md')]
    write_json(result/'inputs.json', {
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): digest(p) for p in sources},
        'tasks_sha256': canonical_digest(tasks), 'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
        'changed_positions_sha256': canonical_digest(changed),
        'jspace_inputs_sha256': digest('results/JSPACE-0001/inputs.json'),
        'config_sha256': digest('configs/jspace0001.json'),
        'layers': LAYERS, 'conditions': CONDITIONS, 'seed': 51051,
        'cap_seconds': CAP_SECONDS, 'memory_bytes_cap': config['memory_bytes_cap'],
    })

    baselines = {}
    baseline_dir = output/'baselines'
    baseline_dir.mkdir(exist_ok=True)

    def baseline(prompt):
        if prompt not in baselines:
            budget(0)
            text, tokens, captures, logits = model.generate_field(ids[prompt], capture_layers=LAYERS)
            key = canonical_digest(prompt)
            filename = baseline_dir/(key+'.npz')
            np.savez(filename, logits=logits, **{str(layer): value for layer, value in captures.items()})
            baselines[prompt] = {'text': text, 'tokens': tokens, 'captures': captures, 'logits': logits,
                                 'arrays_sha256': digest(filename)}
        return baselines[prompt]

    baseline_rows = []
    for row in tasks:
        source, donor = baseline(row['prompt']), baseline(row['donor_prompt'])
        baseline_rows.append({
            'id': row['id'], 'bundle': row['bundle'], 'template': row['template'], 'family': row['family'],
            'source_text': source['text'], 'donor_text': donor['text'],
            'source_correct': answer_correct(source['text'], [row['answer']]),
            'donor_correct': answer_correct(donor['text'], [row['donor_answer']]),
            'first_token_differs': bool(source['tokens']) and bool(donor['tokens']) and source['tokens'][0] != donor['tokens'][0],
            'source_arrays_sha256': source['arrays_sha256'], 'donor_arrays_sha256': donor['arrays_sha256'],
        })
    write_json(result/'baselines.json', baseline_rows)
    eligible_ids = {r['id'] for r in baseline_rows if r['family'] == 'binding' and r['source_correct']
                    and r['donor_correct'] and r['first_token_differs']}
    baseline_copy_ids = {r['id'] for r in baseline_rows if r['family'] == 'copy' and r['source_correct']}
    if len(eligible_ids) < 24 or len(baseline_copy_ids) < 20:
        write_json(result/'decision.json', {'status': 'inconclusive', 'reason': 'baseline_competence',
                                            'eligible_semantic': len(eligible_ids), 'baseline_copy': len(baseline_copy_ids)})
        return

    rows = []
    max_control_error = 0.0
    all_control_argmax = True
    with (output/'observations.jsonl').open('w') as log:
        for task in tasks:
            source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
            positions = changed[task['id']]
            for layer in LAYERS:
                delta = donor['captures'][layer]-source['captures'][layer]
                edits = field_conditions(delta, positions, seed=condition_seed(51051, task['id'], layer, 'field'))
                for condition in CONDITIONS:
                    budget(len(rows))
                    text, tokens, _, first_logits = model.generate_field(ids[task['prompt']], fields={layer: edits[condition]})
                    if condition == 'identity' and (text != source['text'] or tokens != source['tokens']):
                        raise ValueError('Identity generation changed')
                    if condition == 'all_donor':
                        error = float(np.max(np.abs(first_logits-donor['logits'])))
                        max_control_error = max(max_control_error, error)
                        all_control_argmax &= int(first_logits.argmax()) == int(donor['logits'].argmax())
                        if error >= .002 or not all_control_argmax:
                            raise ValueError('Whole-prefix donor first-token positive control failed')
                    observation = {
                        'task_id': task['id'], 'bundle': task['bundle'], 'template': task['template'],
                        'family': task['family'], 'layer': layer, 'condition': condition,
                        'text': text, 'tokens': tokens,
                        'baseline_correct': answer_correct(source['text'], [task['answer']]),
                        'donor_correct': answer_correct(donor['text'], [task['donor_answer']]),
                        'eligible': task['id'] in eligible_ids,
                        'desired_answer': answer_correct(text, [task['donor_answer']]) if task['family'] == 'binding' else False,
                        'copy_preserved': answer_correct(text, [task['answer']]) if task['family'] == 'copy' else False,
                        'field_norm': float(np.linalg.norm(edits[condition])),
                    }
                    rows.append(observation)
                    log.write(json.dumps(observation, ensure_ascii=False, allow_nan=False)+'\n')
                    log.flush()
    write_json(result/'rows.json', rows)
    summary_data = summary(rows, eligible_ids, baseline_copy_ids)
    write_json(result/'summary.json', summary_data)
    primary = summary_data['23']
    n = len(eligible_ids)
    rates = {condition: primary[condition]['semantic']['switches']/n for condition in CONDITIONS}
    copy_rates = {condition: primary[condition]['copy']['preserved']/len(baseline_copy_ids) for condition in CONDITIONS}
    control_adequate = rates['all_donor'] >= .9 and copy_rates['all_donor'] >= .95
    value_supported = (control_adequate and rates['value_donor'] >= .5
                       and rates['value_donor']-rates['value_random'] >= .2 and copy_rates['value_donor'] >= .95)
    bypass_supported = (control_adequate and rates['earlier_donor'] >= .5
                        and rates['earlier_donor']-rates['last_donor'] >= .25
                        and copy_rates['earlier_donor'] >= .95)
    write_json(result/'decision.json', {
        'status': 'exploratory_scored' if control_adequate else 'uninformative_positive_control',
        'value_route_supported': value_supported, 'earlier_bypass_supported': bypass_supported,
        'eligible_semantic': n, 'eligible_bundles': len({r['bundle'] for r in tasks if r['id'] in eligible_ids}),
        'baseline_copy': len(baseline_copy_ids), 'rates': rates, 'copy_rates': copy_rates,
        'all_donor_max_first_logit_error': max_control_error, 'all_donor_argmax_all': all_control_argmax,
        'value_minus_random_bundle_bootstrap': bootstrap_difference(rows, eligible_ids, 'value_donor', 'value_random'),
        'earlier_minus_last_bundle_bootstrap': bootstrap_difference(rows, eligible_ids, 'earlier_donor', 'last_donor'),
        'elapsed_seconds': time.monotonic()-started, 'peak_mlx_bytes': model.mx.get_peak_memory(),
        'scope': 'Explicit new two-slot bindings only; no historical decipherment',
    })


if __name__ == '__main__':
    run()
