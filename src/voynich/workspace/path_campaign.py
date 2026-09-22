"""Finite attention-route study on disjoint two-slot binding bundles."""

from collections import defaultdict
from pathlib import Path
import subprocess
import time

import numpy as np

from .attention_route_backend import AttentionRouteWorkspace
from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .path_tasks import path_tasks

LAYERS = (23, 24, 27, 28)
CONDITIONS = ('identity', 'head_one', 'head_four', 'value_token_four',
              'qk_four', 'value_all_four', 'random_four', 'all_heads')
CAP_SECONDS = 2700


def eligible(row, own, donor):
    return (row['family'] == 'binding' and answer_correct(own['text'], [row['answer']])
            and answer_correct(donor['text'], [row['donor_answer']])
            and bool(own['tokens']) and bool(donor['tokens'])
            and own['tokens'][0] != donor['tokens'][0])


def token_differences(source, donor):
    if len(source) != len(donor) or len(source) < 12:
        raise ValueError('Paired rendered prompt lengths differ')
    diff = np.flatnonzero(np.asarray(source) != np.asarray(donor))
    if len(diff) < 2 or diff[-1] >= len(source)-8:
        raise ValueError('Changed tokens overlap answer query or fail to cover two records')
    return diff.tolist()


def summarize(rows, eligible_ids):
    result = {}
    for condition in CONDITIONS:
        members = [r for r in rows if r['condition'] == condition]
        semantic = [r for r in members if r['task_id'] in eligible_ids]
        copy = [r for r in members if r['family'] == 'copy']
        bundles = defaultdict(list)
        for r in semantic:
            bundles[r['bundle']].append(r)
        result[condition] = {
            'semantic': {'count': len(semantic), 'switches': sum(r['desired_answer'] for r in semantic)},
            'copy': {'count': len(copy), 'preserved': sum(r['copy_preserved'] for r in copy)},
            'mean_margin_gain': float(np.mean([r['margin_gain'] for r in semantic])) if semantic else None,
            'bundles': {str(k): {'count': len(v), 'switches': sum(r['desired_answer'] for r in v)} for k, v in sorted(bundles.items())},
        }
    return result


def run():
    config = load_config('configs/jspace0001.json')
    output, result = Path('outputs/PATH-0001'), Path('results/PATH-0001')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result/'inputs.json').exists():
        raise FileExistsError('No automatic repeat of PATH-0001')
    started = time.monotonic()
    model = AttentionRouteWorkspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(config['memory_bytes_cap'])

    def budget():
        elapsed = time.monotonic()-started
        if elapsed >= CAP_SECONDS or model.mx.get_peak_memory() > config['memory_bytes_cap']:
            write_json(result/'incomplete.json', {'elapsed_seconds': elapsed, 'reason': 'resource_cap'})
            raise RuntimeError('PATH-0001 resource cap')

    # Generic numerical gates precede any task score.
    probe = model.encode('Records: Daro: plum. Leni: stone. Which value belongs to Daro?', chat=True)
    reference, native_cache, _ = model.prefill(probe)
    zero, _ = model.prefill_attention_delta(probe, layer=23, delta=np.zeros(model.width, dtype=np.float32))
    captured, heads = model.capture_heads(probe, LAYERS)
    qualification = {'clean_vs_zero_max_error': float(np.max(np.abs(reference-zero))),
                     'clean_vs_capture_max_error': float(np.max(np.abs(reference-captured))),
                     'head_reconstruction_max_error': max(heads[i]['native_error'] for i in LAYERS),
                     'argmax_agreement': int(reference.argmax()) == int(zero.argmax()) == int(captured.argmax())}
    del native_cache
    qualification['passed'] = (qualification['argmax_agreement']
                               and qualification['clean_vs_zero_max_error'] < .002
                               and qualification['clean_vs_capture_max_error'] < .002
                               and qualification['head_reconstruction_max_error'] < .002)
    write_json(result/'qualification.json', qualification)
    if not qualification['passed']:
        raise ValueError('Attention numerical gate failed')

    tasks = {split: path_tasks(split) for split in ('discovery', 'confirmation')}
    prompts = {row['prompt'] for split in tasks for row in tasks[split]}
    prompts.update(row['donor_prompt'] for split in tasks for row in tasks[split])
    ids = {prompt: model.encode(prompt, chat=True) for prompt in prompts}
    differences = {row['id']: token_differences(ids[row['prompt']], ids[row['donor_prompt']])
                   for split in tasks for row in tasks[split]}
    write_json(output/'rendered-inputs.json', {canonical_digest(k): v for k, v in ids.items()})
    source = [Path('src/voynich/workspace/attention_route_backend.py'),
              Path('src/voynich/workspace/path_campaign.py'),
              Path('src/voynich/workspace/path_tasks.py'), Path('docs/experiments/PATH-0001.md')]
    write_json(result/'inputs.json', {
        'source_sha256': {str(p): digest(p) for p in source},
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'jspace_inputs_sha256': digest('results/JSPACE-0001/inputs.json'),
        'tasks_sha256': canonical_digest(tasks),
        'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
        'difference_index_sha256': canonical_digest(differences),
        'layers': LAYERS, 'conditions': CONDITIONS, 'seconds_cap': CAP_SECONDS,
        'memory_bytes_cap': config['memory_bytes_cap'], 'selection_seed': 51041,
    })

    baselines = {}

    def baseline(prompt):
        if prompt not in baselines:
            budget()
            text, tokens, _, logits = model.generate_field(ids[prompt])
            baselines[prompt] = {'text': text, 'tokens': tokens, 'first_token': int(logits.argmax())}
        return baselines[prompt]

    def public_baselines(split):
        return [{
            'id': row['id'], 'family': row['family'], 'bundle': row['bundle'],
            'source_text': baseline(row['prompt'])['text'],
            'donor_text': baseline(row['donor_prompt'])['text'],
            'source_correct': answer_correct(baseline(row['prompt'])['text'], [row['answer']]),
            'donor_correct': answer_correct(baseline(row['donor_prompt'])['text'], [row['donor_answer']]),
            'eligible': eligible(row, baseline(row['prompt']), baseline(row['donor_prompt'])),
        } for row in tasks[split]]

    discovery_baselines = public_baselines('discovery')
    write_json(result/'discovery-baselines.json', discovery_baselines)
    discovery_eligible = [row for row, status in zip(tasks['discovery'], discovery_baselines) if status['eligible']]
    write_json(result/'discovery-capability.json', {
        'eligible_count': len(discovery_eligible), 'eligible_bundles': len({r['bundle'] for r in discovery_eligible}),
        'semantic_count': 16, 'copy_count': 8})
    if len(discovery_eligible) < 8:
        write_json(result/'decision.json', {'status': 'inconclusive', 'reason': 'discovery_baseline_competence'})
        return

    captures = {}

    def capture(prompt):
        if prompt not in captures:
            budget()
            logits, data = model.capture_heads(ids[prompt], LAYERS)
            if int(logits.argmax()) != baseline(prompt)['first_token'] or any(
                data[layer]['native_error'] >= .002 for layer in LAYERS
            ):
                raise ValueError('Task-specific attention reconstruction failed')
            captures[prompt] = (logits, data)
        return captures[prompt]

    scan = []
    for row in discovery_eligible:
        source = baseline(row['prompt'])
        donor = baseline(row['donor_prompt'])
        clean_logits, source_heads = capture(row['prompt'])
        _, donor_heads = capture(row['donor_prompt'])
        s, d = source['first_token'], donor['first_token']
        original_margin = float(clean_logits[d]-clean_logits[s])
        for layer in LAYERS:
            for head in range(model.layers[layer].self_attn.n_heads):
                budget()
                delta = model.head_delta(layer, source_heads[layer]['heads'], donor_heads[layer]['heads'], (head,))
                changed, _ = model.prefill_attention_delta(ids[row['prompt']], layer=layer, delta=delta)
                scan.append({'task_id': row['id'], 'bundle': row['bundle'], 'layer': layer, 'head': head,
                             'margin_gain': float(changed[d]-changed[s])-original_margin,
                             'first_token_switch': int(changed.argmax()) == d})
    write_json(result/'discovery-head-scan.json', scan)
    groups = defaultdict(list)
    for row in scan:
        groups[(row['layer'], row['head'])].append(row['margin_gain'])
    ranked = sorted(groups, key=lambda key: (-float(np.mean(groups[key])), key[0], key[1]))
    selected_layer = ranked[0][0]
    selected_heads = [head for layer, head in ranked if layer == selected_layer][:4]
    others = [h for h in range(model.layers[selected_layer].self_attn.n_heads) if h not in selected_heads]
    random_heads = sorted(int(x) for x in np.random.default_rng(51041).choice(others, 4, replace=False))
    write_json(result/'selection.json', {
        'selected_layer': selected_layer, 'selected_heads': selected_heads,
        'random_heads': random_heads, 'best_mean_margin_gain': float(np.mean(groups[ranked[0]])),
        'ranking': [{'layer': layer, 'head': head, 'mean_margin_gain': float(np.mean(groups[(layer, head)]))}
                    for layer, head in ranked],
        'discovery_eligible': len(discovery_eligible),
    })

    # Confirmation outputs are now opened once, after selection has been saved.
    confirmation_baselines = public_baselines('confirmation')
    write_json(result/'confirmation-baselines.json', confirmation_baselines)
    confirmation_eligible = {row['id'] for row, status in zip(tasks['confirmation'], confirmation_baselines) if status['eligible']}
    if len(confirmation_eligible) < 8:
        write_json(result/'decision.json', {'status': 'inconclusive', 'reason': 'confirmation_baseline_competence',
                                            'eligible_count': len(confirmation_eligible)})
        return

    rows = []
    for row in tasks['confirmation']:
        own, donor = baseline(row['prompt']), baseline(row['donor_prompt'])
        own_logits, source = capture(row['prompt'])
        _, other = capture(row['donor_prompt'])
        layer = selected_layer
        source_heads, donor_heads = source[layer]['heads'], other[layer]['heads']
        value_positions = differences[row['id']]
        value_v = source[layer]['v'].copy()
        value_v[:, value_positions] = other[layer]['v'][:, value_positions]
        value_heads = model.heads_from_qkv(source[layer]['q'], source[layer]['k'], value_v)
        value_all_heads = model.heads_from_qkv(source[layer]['q'], source[layer]['k'], other[layer]['v'])
        qk_heads = model.heads_from_qkv(other[layer]['q'], other[layer]['k'], source[layer]['v'])
        if np.max(np.abs(model.heads_from_qkv(source[layer]['q'], source[layer]['k'], source[layer]['v'])-source_heads)) > .002:
            raise ValueError('Source head reconstruction changed')
        four_delta = model.head_delta(layer, source_heads, donor_heads, selected_heads)
        random_delta = model.head_delta(layer, source_heads, donor_heads, random_heads)
        norm = float(np.linalg.norm(random_delta))
        if norm > 0:
            random_delta *= float(np.linalg.norm(four_delta))/norm
        edits = {
            'identity': np.zeros(model.width, dtype=np.float32),
            'head_one': model.head_delta(layer, source_heads, donor_heads, selected_heads[:1]),
            'head_four': four_delta,
            'value_token_four': model.head_delta(layer, source_heads, value_heads, selected_heads),
            'qk_four': model.head_delta(layer, source_heads, qk_heads, selected_heads),
            'value_all_four': model.head_delta(layer, source_heads, value_all_heads, selected_heads),
            'random_four': random_delta,
            'all_heads': model.head_delta(layer, source_heads, donor_heads, range(len(source_heads))),
        }
        s, d = own['first_token'], donor['first_token']
        for condition in CONDITIONS:
            budget()
            text, tokens, logits = model.generate_attention_delta(ids[row['prompt']], layer=layer, delta=edits[condition])
            rows.append({
                'task_id': row['id'], 'bundle': row['bundle'], 'family': row['family'],
                'condition': condition, 'text': text, 'tokens': tokens,
                'desired_answer': answer_correct(text, [row['donor_answer']]) if row['family'] == 'binding' else False,
                'copy_preserved': answer_correct(text, [row['answer']]) if row['family'] == 'copy' else False,
                'baseline_correct': answer_correct(own['text'], [row['answer']]),
                'donor_correct': answer_correct(donor['text'], [row['donor_answer']]),
                'eligible': row['id'] in confirmation_eligible,
                'margin_gain': float(logits[d]-logits[s]-(own_logits[d]-own_logits[s])) if s != d else None,
                'edit_norm': float(np.linalg.norm(edits[condition])),
            })
    write_json(result/'confirmation-rows.json', rows)
    lookup = {r['id']: r for r in tasks['confirmation']}
    if any(r['text'] != baseline(lookup[r['task_id']]['prompt'])['text']
           or r['tokens'] != baseline(lookup[r['task_id']]['prompt'])['tokens']
           for r in rows if r['condition'] == 'identity'):
        raise ValueError('Identity generation did not match baseline')
    summary = summarize(rows, confirmation_eligible)
    write_json(result/'summary.json', summary)
    n = len(confirmation_eligible)
    rates = {condition: summary[condition]['semantic']['switches']/n for condition in CONDITIONS}
    copy_rate = summary['value_token_four']['copy']['preserved']/summary['value_token_four']['copy']['count']
    positive = rates['all_heads'] >= .5
    supported = (positive and rates['value_token_four'] >= .5
                 and rates['value_token_four']-rates['random_four'] >= .2
                 and rates['value_token_four']-rates['qk_four'] >= .2 and copy_rate >= .95)
    write_json(result/'decision.json', {
        'status': 'exploratory_supported' if supported else ('uninformative' if not positive else 'exploratory_failed'),
        'confirmation_eligible': n, 'confirmation_eligible_bundles': len({r['bundle'] for r in tasks['confirmation'] if r['id'] in confirmation_eligible}),
        'rates': rates, 'copy_rate': copy_rate, 'positive_control_passed': positive,
        'elapsed_seconds': time.monotonic()-started, 'peak_mlx_bytes': model.mx.get_peak_memory(),
        'scope': 'Explicit two-slot binding in one model; no Voynich decoding claim',
    })


if __name__ == '__main__':
    run()
