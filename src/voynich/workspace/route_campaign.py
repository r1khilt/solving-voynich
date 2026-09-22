"""Exploratory position-coverage diagnosis; completed verdicts stay unchanged."""

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_campaign import condition_seed, meets, rates
from .geometry import coordinate_swap, matched_random_delta
from .route_backend import RouteWorkspace, positional_field
from .tasks import build_tasks

LAYERS = (23, 27)
CONDITIONS = ('identity', 'last_donor', 'earlier_donor', 'all_donor', 'last_swap', 'earlier_swap', 'all_swap', 'all_random', 'all_raw')


def route_tasks():
    tasks = [r for r in build_tasks(split='development') if r['family'] == 'anchored_alias']
    contexts = {(r['pair_id'], r['latent_concept']): r for r in tasks}
    copies = []
    for (pair, country), task in sorted(contexts.items()):
        for word in ('amber', 'velvet'):
            row = dict(task)
            row.update({'id': f'route-copy/{pair}/{country}/{word}', 'family': 'codebook_copy', 'relation': 'copy',
                        'paired_id': f"route-copy/{pair}/{task['donor_concept']}/{word}",
                        'prompt': task['prompt'].split('\n\n')[0]+'\n\nCopy only the literal word '+word+'.',
                        'donor_prompt': task['donor_prompt'].split('\n\n')[0]+'\n\nCopy only the literal word '+word+'.',
                        'answers': [word], 'counterfactual_answers': [word], 'answer': word, 'counterfactual_answer': word,
                        'answer_absent_from_prompt': False, 'expected_effect': 'preserve'})
            row['prompt_group'] = canonical_digest(row['prompt'])
            copies.append(row)
    result = tasks+copies
    if len(tasks) != 48 or len(copies) != 12 or len({r['id'] for r in result}) != 60:
        raise ValueError('Unexpected route input panel')
    lookup = {r['id']: r for r in result}
    if any(lookup[r['paired_id']]['prompt'] != r['donor_prompt'] for r in result):
        raise ValueError('Route donor pairing changed')
    return result


def summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['layer'], row['condition'])].append(row)
    result = {}
    for (layer, condition), members in groups.items():
        item = rates(members)
        for field in ('relation', 'pair_id', 'family'):
            item['by_'+field] = {key: rates([r for r in members if r[field] == key]) for key in sorted({r[field] for r in members})}
        eligible = [r for r in members if r['expected_effect'] == 'change' and r['baseline_correct'] and r['donor_correct']]
        item['first_token_donor_agreement'] = {'count': len(eligible), 'successes': sum(r['first_token_matches_donor'] for r in eligible)}
        result.setdefault(str(layer), {})[condition] = item
    return result


def diagnostic(summary):
    group = summary['23']
    metric = 'counterfactual_both_clean_correct'
    value = {name: group[name][metric]['rate'] for name in CONDITIONS}
    def difference(a, b):
        return value[a]-value[b] if value[a] is not None and value[b] is not None else None
    bypass = difference('earlier_donor', 'last_donor')
    coverage = difference('all_swap', 'last_swap')
    random = difference('all_swap', 'all_random')
    return {'exploratory': True, 'primary_layer': 23,
            'earlier_context_bypass_supported': meets(bypass, .2) and meets(group['earlier_donor']['copy_preserved']['rate'], .95),
            'position_coverage_improves_lens': meets(coverage, .2) and meets(random, .2) and meets(group['all_swap']['copy_preserved']['rate'], .95),
            'earlier_minus_last_donor': bypass, 'all_minus_last_swap': coverage, 'all_swap_minus_random': random,
            'scope': 'Exposed development facts only; not fresh confirmation or a replacement for JSPACE/NEURON decisions'}


def qualify(model):
    ids = model.encode('Read this carefully. Give a short word for a clear daytime sky.', chat=True)
    rng = np.random.default_rng(51031)
    delta = rng.normal(size=(len(ids), model.width)).astype(np.float32)*.01
    rows = []
    for scope in ('last', 'earlier', 'all'):
        field = positional_field(delta, scope)
        actual, _, _ = model.prefill_field(ids, fields={23: field})
        patches = [{'layer': 23, 'position': i, 'delta': value} for i, value in enumerate(field) if np.any(value)]
        expected, _ = model.forward(ids, patches=patches)
        rows.append({'scope': scope, 'max_error': float(np.max(np.abs(actual-expected[0]))),
                     'argmax_same': int(actual.argmax()) == int(expected[0].argmax())})
    return {'passed': all(r['max_error'] < .002 and r['argmax_same'] for r in rows), 'rows': rows}


def run():
    config = load_config('configs/jspace0001.json')
    output, result = Path('outputs/ROUTE-0001'), Path('results/ROUTE-0001')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result/'inputs.json').exists() or (output/'observations.jsonl').exists():
        raise FileExistsError('No automatic repeat of this diagnostic')
    tasks = route_tasks()
    fit = json.loads(Path('results/JSPACE-0001/fit.json').read_text())
    if not fit['complete'] or digest('outputs/JSPACE-0001/lens.npz') != fit['lens_sha256']:
        raise ValueError('Completed lens changed')
    started = time.monotonic()
    model = RouteWorkspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(config['memory_bytes_cap'])
    qualification = qualify(model)
    write_json(result/'qualification.json', qualification)
    if not qualification['passed']:
        raise ValueError('Field numerical qualification failed')
    ids = {prompt: model.encode(prompt, chat=True) for row in tasks for prompt in (row['prompt'], row['donor_prompt'])}
    if any(len(ids[r['prompt']]) != len(ids[r['donor_prompt']]) for r in tasks):
        raise ValueError('Route donor/source sequence lengths must match')
    write_json(output/'rendered-inputs.json', ids)
    source = sorted(Path('src/voynich/workspace').glob('*.py'))+[Path('docs/experiments/ROUTE-0001.md')]
    manifest = {'source_sha256': {str(p): digest(p) for p in source}, 'source_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'tasks_sha256': canonical_digest(tasks), 'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
                'jspace_inputs_sha256': digest('results/JSPACE-0001/inputs.json'), 'lens_sha256': digest('outputs/JSPACE-0001/lens.npz'),
                'config_sha256': canonical_digest(config),
                'layers': LAYERS, 'conditions': CONDITIONS, 'cap_seconds': 1200, 'memory_cap': config['memory_bytes_cap'], 'seed': 51031}
    write_json(result/'inputs.json', manifest)
    baselines = {}
    baseline_dir = output/'baselines'
    baseline_dir.mkdir(exist_ok=True)

    def budget():
        elapsed = time.monotonic()-started
        if elapsed >= 1200 or model.mx.get_peak_memory() > config['memory_bytes_cap']:
            write_json(result/'incomplete.json', {'seconds': elapsed, 'reason': 'resource_cap'})
            raise RuntimeError('Route resource cap')

    def baseline(prompt):
        if prompt not in baselines:
            budget()
            text, tokens, captures, logits = model.generate_field(ids[prompt], capture_layers=LAYERS)
            key = canonical_digest(prompt)
            prior = Path('outputs/JSPACE-0001/development-baselines')/(key+'.json')
            if prior.exists() and json.loads(prior.read_text())['generated_tokens'] != tokens:
                raise ValueError('Existing development baseline did not reproduce')
            path = baseline_dir/(key+'.npz')
            np.savez(path, logits=logits, **{str(k): v for k,v in captures.items()})
            meta = {'text': text, 'tokens': tokens, 'arrays_sha256': digest(path)}
            write_json(baseline_dir/(key+'.json'), meta)
            baselines[prompt] = (meta, captures, logits)
        return baselines[prompt]

    with np.load('outputs/JSPACE-0001/lens.npz') as lens:
        directions = lens['mean'].reshape(len(config['layers']), len(config['words']), 2, model.width).mean(axis=2)
        raw = lens['output_rows'].reshape(len(config['words']), 2, model.width).mean(axis=1)
    rows = []
    with (output/'observations.jsonl').open('w') as log:
        for task in tasks:
            own, own_h, own_logits = baseline(task['prompt'])
            donor, donor_h, donor_logits = baseline(task['donor_prompt'])
            countries = [config['words'].index(task[k]) for k in ('latent_concept','donor_concept')]
            for layer in LAYERS:
                country_rows = directions[config['layers'].index(layer),countries]
                h = own_h[layer]
                swap = np.stack([coordinate_swap(value, country_rows)[0] for value in h])
                raw_swap = np.stack([coordinate_swap(value, raw[countries])[0] for value in h])
                random = np.stack([matched_random_delta(value, seed=condition_seed(51031,task['id'],layer,str(i))) for i,value in enumerate(swap)])
                edits = {'identity': np.zeros_like(h), 'all_raw': raw_swap, 'all_random': random}
                for scope in ('last','earlier','all'):
                    edits[scope+'_donor'] = positional_field(donor_h[layer]-h,scope)
                    edits[scope+'_swap'] = positional_field(swap,scope)
                for condition in CONDITIONS:
                    budget()
                    field = edits[condition]
                    text, tokens, _, logits = model.generate_field(ids[task['prompt']], fields={layer: field})
                    error = float(np.max(np.abs(logits-donor_logits))) if condition == 'all_donor' else None
                    if condition == 'all_donor' and (error >= .002 or logits.argmax() != donor_logits.argmax()):
                        raise ValueError('Full-prefix donor first-token control failed')
                    if condition == 'identity' and tokens != own['tokens']:
                        raise ValueError('Identity generation changed')
                    row = {k: task[k] for k in ('id','family','relation','pair_id','expected_effect','latent_concept','donor_concept')}
                    row.update({'layer':layer,'condition':condition,'valid':True,'answer':text,'generated_tokens':tokens,
                                'baseline':own['text'],'donor':donor['text'],
                                'baseline_correct':answer_correct(own['text'],task['answers']),
                                'donor_correct':answer_correct(donor['text'],task['counterfactual_answers']),
                                'original_correct':answer_correct(text,task['answers']),
                                'counterfactual_correct':answer_correct(text,task['counterfactual_answers']),
                                'first_token_matches_donor':int(logits.argmax())==int(donor_logits.argmax()),
                                'first_token_matches_original':int(logits.argmax())==int(own_logits.argmax()),
                                'full_donor_max_logit_error':error,'field_norm':float(np.linalg.norm(field)),
                                'recipient_field_norm':float(np.linalg.norm(h)), 'positions_edited':int(np.any(field!=0,axis=1).sum()),
                                'truncated':len(tokens)==12})
                    log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    log.flush()
                    rows.append(row)
                    write_json(output/'progress.json',{'completed':len(rows),'seconds':time.monotonic()-started})
                    if len(rows)%120==0:
                        print(json.dumps({'completed':len(rows),'seconds':time.monotonic()-started}),flush=True)
    if len(rows)!=1080:
        raise ValueError('Incomplete route grid')
    budget()
    summary=summaries(rows)
    write_json(result/'results.json',{'summary':summary,'rows':len(rows),'seconds':time.monotonic()-started,
                                     'observations_sha256':digest(output/'observations.jsonl'),'inputs_sha256':digest(result/'inputs.json'),
                                     'peak_memory_bytes':model.mx.get_peak_memory()})
    verdict=diagnostic(summary)
    verdict['results_sha256']=digest(result/'results.json')
    write_json(result/'decision.json',verdict)
    print(json.dumps(verdict),flush=True)


if __name__=='__main__':
    run()
